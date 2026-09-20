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
        """Initialize LLM client based on config.

        Delegates to ``create_client_from_config`` so every provider supported
        by the LLM client factory (gemini, anthropic, minimax, ollama) gets
        initialized correctly. The historical implementation only matched
        ``anthropic`` and ``google`` literally, so providers like ``ollama``
        silently fell through to the TF-IDF fallback.
        """
        llm_config = self.config.llm

        try:
            from src.llm_client import create_client_from_config

            self.llm_client = create_client_from_config(self.config)
            self.llm_provider = getattr(llm_config, 'provider', None)
            logger.info(
                "Using %s (%s) for keyword extraction",
                self.llm_provider,
                getattr(llm_config, 'model', 'unknown'),
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize LLM client (%s) - falling back to TF-IDF extraction",
                e,
            )
            self.llm_client = None

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
            logger.info("[KEYWORD_EXTRACT] No segments provided for keyword extraction")
            return KeywordResult(keywords=[], segments_analyzed=0, extraction_method="none")

        # Combine all text
        full_text = self._combine_voiceover_text(segments)
        text_length = len(full_text)

        if not full_text:
            logger.info("[KEYWORD_EXTRACT] No text content after combining segments")
            return KeywordResult(keywords=[], segments_analyzed=len(segments), extraction_method="none")

        logger.info(f"[KEYWORD_EXTRACT] Starting keyword extraction: {len(segments)} segments, {text_length} chars")

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

        # Log extracted keywords by category
        entity_count = len(entity_keywords)
        general_count = len(keywords) - entity_count
        logger.info(f"[KEYWORD_EXTRACT] Keywords extracted: {len(keywords)} total, {entity_count} entity-based, {general_count} general")

        # Build prioritized keywords list
        prioritized = build_prioritized_keywords(
            keywords, entity_keywords, raw_entities, text, topic
        )

        logger.info(f"[KEYWORD_EXTRACT] Keyword extraction complete: method=llm_entity_aware, keywords={len(keywords)}, topic={topic}")

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
        """Fallback: extract keywords using TF-IDF (sklearn-backed, inline).

        Replaces a previous reference to ``keyword_extractor.KeywordWeightExtractor``
        which never existed in this repo. Preserves the historical
        ``(boost_terms, penalty_terms, tfidf_scores)`` contract by computing
        the same triple in-place.
        """
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

        # Pull TF-IDF knobs from config if present, else use sensible defaults.
        max_features = getattr(
            getattr(self.config, 'keyword', None), 'tfidf_max_features', 100
        ) or 100

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np

            # French stop words cover the common case for this repo (FRENCH SRT
            # projects); for other languages the default English list still
            # removes most noise without breaking the fallback.
            try:
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='french',
                    ngram_range=(1, 2),
                    min_df=1,
                )
            except ValueError:
                # sklearn may raise if the 'french' stop-word list is unavailable
                # in this build — fall back to English.
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                )

            tfidf_matrix = vectorizer.fit_transform(texts)
            terms = vectorizer.get_feature_names_out()
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

            # Sort by score descending, pick top 30 boost terms
            top_idx = mean_scores.argsort()[::-1][:30]
            boost_terms = [terms[i] for i in top_idx if mean_scores[i] > 0]
            penalty_terms = []
            tfidf_scores = dict(zip(terms, mean_scores.tolist()))

            keywords = list(boost_terms)
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed ({e}); using simple word frequency")
            # Absolute last-resort fallback: most-common non-trivial tokens.
            from collections import Counter
            import re
            counter = Counter()
            for t in texts:
                counter.update(
                    w for w in re.findall(r"\b\w{3,}\b", t.lower())
                )
            keywords = [w for w, _ in counter.most_common(30)]
            penalty_terms = []
            tfidf_scores = dict(counter)

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
