"""
Main keyword extraction orchestration.

LLMKeywordExtractor class coordinates:
- LLM client initialization
- Keyword extraction pipeline
- Entity extraction integration
- Topic detection
- Result building
"""

import re
import json
import logging
import os
from typing import List, Dict

from .models import KeywordResult, PrioritizedKeyword
from .prompts import KEYWORD_EXTRACTION_PROMPT, KEYWORD_EXPANSION_PROMPT
from .validator import validate_visual_keywords
from .entity_extractor import extract_entities
from .topic_detector import detect_topic
from .prioritizer import build_prioritized_keywords
from .segment_processor import extract_keyword_per_segment

logger = logging.getLogger(__name__)


class LLMKeywordExtractor:
    """
    Extracts stock footage search keywords from voiceover segments.
    Uses LLM for intelligent keyword generation optimized for YouTube search.
    """

    # Suffixes that improve YouTube footage search results
    FOOTAGE_SUFFIXES = [
        "4K footage",
        "news footage",
        "drone footage",
        "aerial footage",
        "stock footage",
        "documentary footage"
    ]

    def __init__(self, config):
        """Initialize with config"""
        self.config = config
        self.llm_client = None
        self.llm_provider = None
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client based on config"""
        llm_config = self.config.llm

        # Try to get API key from config first, then environment
        api_key = llm_config.api_key

        try:
            from src.llm_client import create_client

            if llm_config.provider == 'anthropic':
                if not api_key:
                    api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    logger.warning("No Anthropic API key found - set ANTHROPIC_API_KEY environment variable")
                    return
                self.llm_client = create_client("anthropic", api_key=api_key, model=llm_config.model)
                self.llm_provider = 'anthropic'
                logger.info("Using Anthropic Claude for keyword extraction")

            elif llm_config.provider == 'google':
                if not api_key:
                    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    logger.warning("No Gemini API key found - set GEMINI_API_KEY environment variable")
                    return
                self.llm_client = create_client("gemini", api_key=api_key, model=llm_config.model)
                self.llm_provider = 'google'
                logger.info("Using Google Gemini for keyword extraction")

            if not self.llm_client:
                logger.warning("No LLM client available - falling back to TF-IDF extraction")

        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")

    def _call_llm(self, prompt: str) -> str:
        """Call LLM and return response text (expects JSON array)"""
        from src.llm_client import LLMRequest, ResponseFormat

        max_tokens = getattr(self.config.llm, 'max_tokens', 2000)
        try:
            request = LLMRequest(
                prompt=prompt,
                response_format=ResponseFormat.JSON_ARRAY,
                max_tokens=max_tokens,
                cache_key_prefix="keyword_extraction"
            )
            response = self.llm_client.generate(request)
            return response.text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "[]"

    def _call_llm_text(self, prompt: str) -> str:
        """Call LLM and return plain text response (no JSON parsing)"""
        from src.llm_client import LLMRequest, ResponseFormat

        max_tokens = getattr(self.config.llm, 'max_tokens', 2000)
        try:
            request = LLMRequest(
                prompt=prompt,
                response_format=ResponseFormat.TEXT,
                max_tokens=max_tokens,
                cache_key_prefix="keyword_extraction"
            )
            response = self.llm_client.generate(request)
            return response.text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _parse_keywords_json(self, response: str) -> List[str]:
        """Parse JSON array from LLM response"""
        # Clean up response
        response = response.strip()

        # Try to find JSON array in response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            try:
                keywords = json.loads(match.group())
                if isinstance(keywords, list):
                    return [str(k).strip() for k in keywords if k][:50]  # Safety limit
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse line by line
        lines = response.split('\n')
        keywords = []
        for line in lines:
            line = line.strip().strip('-').strip('•').strip('"').strip("'").strip(',')
            if line and not line.startswith('[') and not line.startswith(']'):
                keywords.append(line)

        return keywords[:50]  # Safety limit

    def extract_keywords(
        self,
        segments: List[Dict],
        expand: bool = True
    ) -> KeywordResult:
        """
        Extract search keywords from voiceover segments.

        Args:
            segments: List of voiceover segments with 'text' field
            expand: Whether to expand keywords with LLM refinement

        Returns:
            KeywordResult with extracted keywords
        """
        if not segments:
            return KeywordResult(keywords=[], segments_analyzed=0, extraction_method="none")

        # Combine all text
        full_text = self._combine_voiceover_text(segments)

        if not full_text:
            return KeywordResult(keywords=[], segments_analyzed=len(segments), extraction_method="none")

        # Detect topic for context
        topic = detect_topic(full_text, self.llm_client)

        # Use LLM if available
        if self.llm_client:
            return self._extract_with_llm(full_text, topic, expand, len(segments))
        else:
            return self._extract_with_tfidf(segments)

    def _extract_with_llm(
        self,
        text: str,
        topic: str,
        expand: bool,
        num_segments: int
    ) -> KeywordResult:
        """Extract keywords using LLM with entity-aware extraction"""

        # Step 1: Extract named entities first (people, places, dates, orgs)
        logger.info("Extracting named entities with LLM...")
        entity_keywords, raw_entities = extract_entities(text[:8000], topic, self._call_llm)
        logger.info(f"Entity extraction: {len(entity_keywords)} entity-based keywords, {len(raw_entities)} entities")

        # Step 2: General keyword extraction (request 50 keywords, no hard limit)
        prompt = KEYWORD_EXTRACTION_PROMPT.format(
            voiceover_text=text[:8000],  # Limit text length
            max_keywords=50  # Request a reasonable number from LLM
        )

        logger.info("Extracting general keywords with LLM...")
        response = self._call_llm(prompt)
        initial_keywords = self._parse_keywords_json(response)

        logger.info(f"General extraction: {len(initial_keywords)} keywords")

        # Step 3: Merge entity keywords FIRST (they're high priority)
        # Entity keywords come first to ensure they're included
        merged_keywords = entity_keywords + [k for k in initial_keywords if k not in entity_keywords]

        # Step 4: Expand if requested and we have keywords
        if expand and merged_keywords:
            expand_prompt = KEYWORD_EXPANSION_PROMPT.format(
                initial_keywords=merged_keywords[:30],  # Limit for expansion prompt
                topic=topic,
                max_keywords=50  # Request a reasonable number from LLM
            )

            logger.info("Expanding keywords with LLM...")
            expand_response = self._call_llm(expand_prompt)
            expanded_keywords = self._parse_keywords_json(expand_response)

            if expanded_keywords:
                # Keep entity keywords at the front, add expanded ones
                all_keywords = entity_keywords + [k for k in expanded_keywords if k not in entity_keywords]
                keywords = list(dict.fromkeys(all_keywords))
            else:
                keywords = list(dict.fromkeys(merged_keywords))
        else:
            keywords = list(dict.fromkeys(merged_keywords))

        # Validate keywords - filter out abstract/narrative phrases
        pre_validation_count = len(keywords)
        max_keyword_words = getattr(self.config.keyword, 'max_keyword_words', 8)
        keywords = validate_visual_keywords(keywords, max_words=max_keyword_words)

        logger.info(f"Final keywords: {len(keywords)} (validated from {pre_validation_count}, {len(entity_keywords)} entity-based)")

        # Build prioritized keywords list
        prioritized = build_prioritized_keywords(
            keywords, entity_keywords, raw_entities, text, topic
        )

        return KeywordResult(
            keywords=keywords,
            segments_analyzed=num_segments,
            extraction_method="llm_entity_aware",
            entities=raw_entities,
            topic=topic,
            prioritized_keywords=prioritized
        )

    def _extract_with_tfidf(
        self,
        segments: List[Dict]
    ) -> KeywordResult:
        """Fallback: extract keywords using TF-IDF"""
        from keyword_extractor import KeywordWeightExtractor

        logger.info("Falling back to TF-IDF keyword extraction")

        # Convert segments to text strings
        texts = []
        for seg in segments:
            text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
            if text:
                texts.append(text.strip())

        if not texts:
            logger.warning("No text found in segments for TF-IDF extraction")
            return KeywordResult(
                keywords=[],
                segments_analyzed=len(segments),
                extraction_method="tfidf"
            )

        extractor = KeywordWeightExtractor(self.config)

        # Get weighted terms (returns tuple of boost_terms, penalty_terms, tfidf_scores)
        boost_terms, _, tfidf_scores = extractor.extract_weighted_terms(texts)

        # Convert to search keywords (use boost terms which are top TF-IDF terms)
        keywords = list(boost_terms)

        return KeywordResult(
            keywords=keywords,
            segments_analyzed=len(segments),
            extraction_method="tfidf"
        )

    def add_footage_suffixes(
        self,
        keywords: List[str],
        suffixes: List[str] = None
    ) -> List[str]:
        """
        Optionally add footage-related suffixes to keywords.
        This can improve search results but increases keyword count.

        Args:
            keywords: Base keywords
            suffixes: List of suffixes to add (uses defaults if None)

        Returns:
            Expanded keyword list with suffixes
        """
        if suffixes is None:
            # Use a subset of default suffixes
            suffixes = ["4K footage", "news footage"]

        expanded = []
        for kw in keywords:
            expanded.append(kw)
            for suffix in suffixes:
                expanded.append(f"{kw} {suffix}")

        return expanded

    def _combine_voiceover_text(self, segments: List[Dict]) -> str:
        """Combine all voiceover segments into single text"""
        texts = []
        for seg in segments:
            text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
            if text:
                texts.append(text.strip())
        return " ".join(texts)

    # Wrapper method for backward compatibility
    def extract_keyword_per_segment(
        self,
        segments: List[Dict],
        topic: str = ""
    ) -> List[str]:
        """
        Extract ONE keyword per segment for precise B-roll matching.

        Args:
            segments: List of segment dicts with 'text' key
            topic: Documentary topic for context

        Returns:
            List of keywords (one per segment, in order)
        """
        return extract_keyword_per_segment(
            segments=segments,
            llm_client=self.llm_client,
            llm_call_function=self._call_llm,
            topic=topic
        )

    def extract_keywords_grouped(
        self,
        segments: List[Dict],
        topic: str = "",
        segments_per_query: int = None
    ) -> List[str]:
        """
        Extract ONE search query per group of N segments.

        Groups segments into batches and generates one unified search query
        per batch, reducing total queries while maintaining context.

        Example: 90 segments with segments_per_query=3 → 30 search queries

        Args:
            segments: List of segment dicts with 'text' key
            topic: Documentary topic for context
            segments_per_query: Number of segments per search query
                              (default: config.keyword.segments_per_query or 3)

        Returns:
            List of search queries (one per segment group)
        """
        from .segment_processor import extract_keywords_grouped as _extract_grouped

        # Use config default if not specified
        if segments_per_query is None:
            segments_per_query = getattr(self.config.keyword, 'segments_per_query', 3)

        # Use plain text LLM call since grouped prompts return plain text, not JSON
        return _extract_grouped(
            segments=segments,
            llm_client=self.llm_client,
            llm_call_function=self._call_llm_text,
            topic=topic,
            segments_per_query=segments_per_query
        )
