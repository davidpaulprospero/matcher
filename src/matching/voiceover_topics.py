"""
Voiceover Topic Extraction Module

Extracts topics from voiceover segments using LLM for better context-aware matching.
Part of US-111-002: Voiceover Segment Topic Extraction for Context-Aware Matching

This module provides functionality to:
- Extract 3-5 key topics from voiceover segment text using LLM
- Store extracted topics in segment metadata for use in matching
- Provide fallback behavior when topic extraction fails
- Retry with exponential backoff on LLM failures
- Cache results for improved performance
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from functools import lru_cache

if TYPE_CHECKING:
    from ..config import Config
    from ..utils import SRTSegment

logger = logging.getLogger(__name__)


# Default prompt for voiceover topic extraction
DEFAULT_TOPIC_EXTRACTION_PROMPT = """You are a topic extraction system. Given a voiceover segment, extract 3-5 key topics that best represent what this segment is about.

Requirements:
1. Extract ONLY the most important keywords/phrases (nouns, noun phrases, or specific concepts)
2. Topics should be specific enough to help find relevant video content
3. Return ONLY a JSON array of strings, nothing else
4. Do not include articles (a, an, the) or common verbs
5. Focus on: locations, people, events, concepts, activities, objects

Voiceover segment:
{text}

Return a JSON array of 3-5 topic strings."""


def _compute_text_hash(text: str) -> str:
    """Compute a hash of the text for caching purposes."""
    return hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()


def _extract_keywords_fallback(text: str, max_topics: int = 5) -> List[str]:
    """
    Fallback keyword extraction using simple pattern matching.

    This is used when LLM extraction fails and fallback_to_keywords is enabled.

    Args:
        text: Voiceover segment text
        max_topics: Maximum number of keywords to return

    Returns:
        List of keyword strings
    """
    import re

    # Extract potential keywords using common patterns
    # Look for capitalized phrases, quoted terms, and common noun patterns
    keywords = []

    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', text)
    keywords.extend(quoted)

    # Extract capitalized phrases (potential proper nouns)
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    keywords.extend(capitalized)

    # Extract words after common topic indicators
    indicators = ['about', 'regarding', 'concerning', 'involving', 'featuring']
    for indicator in indicators:
        pattern = rf'\b{indicator}\s+(\w+(?:\s+\w+){{0,3}})\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        keywords.extend(matches)

    # Dedupe and filter
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_clean = kw.strip().lower()
        if kw_clean and len(kw_clean) > 2 and kw_clean not in seen:
            seen.add(kw_clean)
            unique_keywords.append(kw.strip())

    # Limit to max_topics
    return unique_keywords[:max_topics]


def extract_voiceover_topics(
    text: str,
    config: 'Config',
    min_topics: int = 3,
    max_topics: int = 5,
) -> Optional[List[str]]:
    """
    Extract topics from voiceover segment text using LLM with retry logic.

    Implements:
    - Retry with exponential backoff on LLM failures
    - Fallback to keyword extraction when LLM fails
    - Caching of results for performance

    Args:
        text: Voiceover segment text to extract topics from
        config: Config object with LLM settings
        min_topics: Minimum number of topics to extract
        max_topics: Maximum number of topics to extract

    Returns:
        List of 3-5 topic strings, or None if extraction fails
    """
    if not text or len(text.strip()) < 20:
        logger.debug(f"Text too short for topic extraction: '{text[:50]}...'")
        return None

    # Get config settings for retry and fallback
    vt_config = getattr(config.matching, 'voiceover_topic', None)
    cache_enabled = getattr(vt_config, 'cache_enabled', True) if vt_config else True
    retry_max_attempts = getattr(vt_config, 'retry_max_attempts', 3) if vt_config else 3
    retry_base_delay = getattr(vt_config, 'retry_base_delay', 1.0) if vt_config else 1.0
    retry_max_delay = getattr(vt_config, 'retry_max_delay', 10.0) if vt_config else 10.0
    fallback_to_keywords = getattr(vt_config, 'fallback_to_keywords', True) if vt_config else True

    # Check cache first (US-126-006)
    if cache_enabled:
        text_hash = _compute_text_hash(text)
        cached = _get_cached_topics(text_hash)
        if cached is not None:
            logger.debug(f"Returning cached topics for hash: {text_hash[:8]}...")
            return cached

    # Try LLM extraction with retry logic (US-126-006)
    last_error = None
    for attempt in range(retry_max_attempts):
        try:
            topics = _llm_extract_topics(text, config, min_topics, max_topics)
            if topics is not None:
                # Cache the result
                if cache_enabled:
                    _cache_topics(text_hash, topics)
                return topics
        except Exception as e:
            last_error = e
            logger.warning(f"Voiceover topic extraction attempt {attempt + 1} failed: {e}")

        # Exponential backoff if more attempts remaining
        if attempt < retry_max_attempts - 1:
            delay = min(retry_base_delay * (2 ** attempt), retry_max_delay)
            logger.debug(f"Retrying after {delay:.1f}s (attempt {attempt + 2}/{retry_max_attempts})")
            time.sleep(delay)

    # All LLM attempts failed - try fallback to keyword extraction (US-126-006)
    if fallback_to_keywords:
        logger.info("LLM extraction failed, falling back to keyword extraction")
        fallback_topics = _extract_keywords_fallback(text, max_topics)
        if fallback_topics and len(fallback_topics) >= min_topics:
            # Cache the fallback result too
            if cache_enabled:
                _cache_topics(text_hash, fallback_topics)
            logger.debug(f"Fallback extracted {len(fallback_topics)} keywords: {fallback_topics}")
            return fallback_topics

    logger.warning(f"Topic extraction failed after {retry_max_attempts} attempts. Last error: {last_error}")
    return None


# Simple in-memory cache for topic extraction results
_topic_cache: dict = {}


def _get_cached_topics(text_hash: str) -> Optional[List[str]]:
    """Get cached topics for a text hash."""
    return _topic_cache.get(text_hash)


def _cache_topics(text_hash: str, topics: List[str]) -> None:
    """Cache topics for a text hash."""
    _topic_cache[text_hash] = topics
    # Limit cache size to prevent memory bloat
    if len(_topic_cache) > 1000:
        # Remove oldest entries (simple FIFO - could be improved)
        keys_to_remove = list(_topic_cache.keys())[:100]
        for key in keys_to_remove:
            del _topic_cache[key]


def _llm_extract_topics(
    text: str,
    config: 'Config',
    min_topics: int,
    max_topics: int,
) -> Optional[List[str]]:
    """
    Internal LLM extraction function (called with retry logic).

    Args:
        text: Voiceover segment text to extract topics from
        config: Config object with LLM settings
        min_topics: Minimum number of topics to extract
        max_topics: Maximum number of topics to extract

    Returns:
        List of topic strings, or None if extraction fails
    """
    from src.llm_client import create_client, LLMRequest, ResponseFormat

    # Create LLM client using config settings
    mc = config.matching
    client = create_client(
        "gemini",
        api_key=config.gemini_api_key,
        model=getattr(mc, 'gemini_model', 'gemini-2.0-flash')
    )

    # Build prompt with text
    prompt = DEFAULT_TOPIC_EXTRACTION_PROMPT.format(text=text[:2000])  # Limit text length

    # Make LLM request
    request = LLMRequest(
        prompt=prompt,
        response_format=ResponseFormat.JSON_ARRAY,
    )

    response = client.generate(request)

    if response and response.content:
        import json
        import re

        # Try to parse JSON array from response
        # Look for array pattern in response
        content = response.content

        # Handle if response is already a list
        if isinstance(content, list):
            topics = [str(t).strip() for t in content if t]
        else:
            # Try to extract JSON array from string
            # Find array pattern [...]
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                try:
                    topics = json.loads(match.group(0))
                    topics = [str(t).strip() for t in topics if t]
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Failed to parse JSON from LLM response: {content[:100]}")
                    return None
            else:
                # Fallback: split by common delimiters
                topics = [t.strip() for t in re.split(r'[,;\n]', content) if t.strip()]
                topics = topics[:max_topics]

        # Filter and limit topics
        topics = [t for t in topics if t and len(t) > 1]  # Remove empty/1-char topics

        if len(topics) >= min_topics:
            logger.debug(f"Extracted {len(topics)} topics: {topics}")
            return topics[:max_topics]
        else:
            logger.warning(f"Only extracted {len(topics)} topics, expected >= {min_topics}")
            return None

    return None


def enrich_voiceover_segments_with_topics(
    segments: List['SRTSegment'],
    config: 'Config',
) -> List['SRTSegment']:
    """
    Enrich voiceover segments with extracted topics.

    For each segment, extracts 3-5 key topics and stores them in the segment's
    topics field for use in context-aware matching.

    Args:
        segments: List of voiceover SRTSegment objects
        config: Config object with voiceover_topic settings

    Returns:
        The same segments list (modified in-place), now with topics populated
    """
    # Get config settings
    vt_config = getattr(config.matching, 'voiceover_topic', None)
    if vt_config is None:
        logger.debug("voiceover_topic config not found, skipping topic extraction")
        return segments

    enabled = getattr(vt_config, 'enabled', True)
    if not enabled:
        logger.debug("Voiceover topic extraction disabled in config")
        return segments

    min_topics = getattr(vt_config, 'min_topics', 3)
    max_topics = getattr(vt_config, 'max_topics', 5)
    min_segment_length = getattr(vt_config, 'min_segment_length', 50)

    # Track stats
    extracted_count = 0
    skipped_short = 0

    for segment in segments:
        # Skip if already has topics
        if getattr(segment, 'topics', None):
            continue

        # Skip short segments
        text = getattr(segment, 'text', '') or ''
        if len(text.strip()) < min_segment_length:
            skipped_short += 1
            continue

        # Extract topics
        topics = extract_voiceover_topics(
            text=text,
            config=config,
            min_topics=min_topics,
            max_topics=max_topics,
        )

        if topics:
            segment.topics = topics
            extracted_count += 1
        else:
            skipped_short += 1

    logger.info(
        f"Voiceover topic extraction: {extracted_count} segments enriched, "
        f"{skipped_short} skipped (short or failed)"
    )

    return segments


@dataclass
class VoiceoverTopicResult:
    """Result of voiceover topic extraction for a segment"""
    segment_index: int
    topics: List[str]
    success: bool
    error: Optional[str] = None


def extract_topics_for_segments_batch(
    segments: List['SRTSegment'],
    config: 'Config',
    batch_size: int = 10,
) -> List[VoiceoverTopicResult]:
    """
    Extract topics for multiple segments in batch.

    This is a convenience function that processes segments in batches
    for more efficient LLM usage.

    Args:
        segments: List of voiceover SRTSegment objects
        config: Config object
        batch_size: Number of segments to process in each batch

    Returns:
        List of VoiceoverTopicResult objects
    """
    results: List[VoiceoverTopicResult] = []

    # Get config settings
    vt_config = getattr(config.matching, 'voiceover_topic', None)
    if vt_config is None or not getattr(vt_config, 'enabled', True):
        # Return empty results if disabled
        return [
            VoiceoverTopicResult(
                segment_index=i,
                topics=[],
                success=False,
                error="disabled"
            )
            for i in range(len(segments))
        ]

    min_topics = getattr(vt_config, 'min_topics', 3)
    max_topics = getattr(vt_config, 'max_topics', 5)

    for i, segment in enumerate(segments):
        text = getattr(segment, 'text', '') or ''

        if len(text.strip()) < getattr(vt_config, 'min_segment_length', 50):
            results.append(VoiceoverTopicResult(
                segment_index=i,
                topics=[],
                success=False,
                error="too_short"
            ))
            continue

        topics = extract_voiceover_topics(
            text=text,
            config=config,
            min_topics=min_topics,
            max_topics=max_topics,
        )

        if topics:
            results.append(VoiceoverTopicResult(
                segment_index=i,
                topics=topics,
                success=True,
            ))
        else:
            results.append(VoiceoverTopicResult(
                segment_index=i,
                topics=[],
                success=False,
                error="extraction_failed"
            ))

    return results


def calculate_topic_similarity(
    topics1: List[str],
    topics2: List[str],
) -> float:
    """
    Calculate similarity between two sets of topics.

    Uses Jaccard similarity with word-level tokenization for fuzzy matching.

    Args:
        topics1: First list of topic strings
        topics2: Second list of topic strings

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not topics1 or not topics2:
        return 0.0

    # Tokenize topics into words for fuzzy matching
    def tokenize(topics: List[str]) -> set:
        tokens = set()
        for topic in topics:
            # Split on spaces, underscores, hyphens
            words = topic.lower().replace('_', ' ').replace('-', ' ').split()
            tokens.update(words)
        return tokens

    tokens1 = tokenize(topics1)
    tokens2 = tokenize(topics2)

    if not tokens1 or not tokens2:
        return 0.0

    # Jaccard similarity
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union) if union else 0.0


def calculate_segment_coherence(
    segments: List['SRTSegment'],
    idx: int,
    similarity_threshold: float = 0.3,
) -> float:
    """
    Calculate topic coherence score for a segment relative to its neighbors.

    Measures how well the segment's topics align with surrounding segments.
    Higher coherence = smoother topic flow between adjacent segments.

    Args:
        segments: List of voiceover segments with topics
        idx: Index of the current segment
        similarity_threshold: Minimum similarity to count as "coherent"

    Returns:
        Coherence score between 0.0 and 1.0
    """
    if idx < 0 or idx >= len(segments):
        return 0.0

    current = segments[idx]
    current_topics = getattr(current, 'topics', None)

    if not current_topics:
        return 0.0

    # Get adjacent segments (before and after)
    adjacent = []
    if idx > 0:
        adjacent.append(segments[idx - 1])
    if idx < len(segments) - 1:
        adjacent.append(segments[idx + 1])

    if not adjacent:
        return 1.0  # No neighbors = can't compute, assume coherent

    # Calculate similarity to each neighbor
    similarities = []
    for seg in adjacent:
        seg_topics = getattr(seg, 'topics', None)
        if seg_topics:
            sim = calculate_topic_similarity(current_topics, seg_topics)
            similarities.append(sim)

    if not similarities:
        return 0.0  # No topics in neighbors

    # Coherence is average similarity above threshold
    coherent_count = sum(1 for s in similarities if s >= similarity_threshold)
    return coherent_count / len(similarities)


def build_topic_aware_context(
    vo_segments: List['SRTSegment'],
    current_idx: int,
    base_window: int = 2,
    max_window: int = 4,
    min_similarity: float = 0.3,
    config: Optional['Config'] = None,
) -> tuple:
    """
    Build topic-aware context for a voiceover segment.

    Instead of fixed window, expands context when adjacent segments
    have similar topics (high coherence) for richer context.

    Args:
        vo_segments: List of all voiceover segments
        current_idx: Index of current segment
        base_window: Base number of segments before/after to include
        max_window: Maximum segments to include
        min_similarity: Minimum topic similarity for expansion
        config: Optional config for topic coherence settings

    Returns:
        Tuple of (context_before, context_after, coherence_score)
    """
    if current_idx < 0 or current_idx >= len(vo_segments):
        return [], [], 0.0

    # Get config settings
    if config:
        vt_config = getattr(config.matching, 'voiceover_topic', None)
        if vt_config:
            max_window = getattr(vt_config, 'max_context_segments', max_window)
            min_similarity = getattr(vt_config, 'min_topic_similarity', min_similarity)

    # Calculate coherence with neighbors
    coherence = calculate_segment_coherence(
        vo_segments, current_idx, similarity_threshold=min_similarity
    )

    # Adjust window size based on coherence
    # Higher coherence = can safely expand context
    adjusted_window = base_window
    if config and getattr(getattr(config.matching, 'voiceover_topic', None), 'context_window_adjustment', True):
        if coherence >= 0.7:  # High coherence
            adjusted_window = min(base_window + 1, max_window)
        elif coherence >= 0.4:  # Medium coherence
            adjusted_window = base_window

    # Get context segments
    context_before = []
    context_after = []

    # Build before context
    start_idx = max(0, current_idx - adjusted_window)
    for i in range(start_idx, current_idx):
        context_before.append(vo_segments[i])

    # Build after context
    end_idx = min(len(vo_segments), current_idx + adjusted_window + 1)
    for i in range(current_idx + 1, end_idx):
        context_after.append(vo_segments[i])

    return context_before, context_after, coherence


def get_topic_enriched_text(
    segment: 'SRTSegment',
    max_text_length: int = 50,
) -> str:
    """
    Get text from a segment, enriched with its topics for context.

    Includes segment text with topic keywords appended in brackets.

    Args:
        segment: SRTSegment with optional topics
        max_text_length: Maximum length of text to include

    Returns:
        Topic-enriched text string
    """
    text = getattr(segment, 'text', '') or ''
    text = text[:max_text_length]

    topics = getattr(segment, 'topics', None)
    if topics:
        topic_str = ' | '.join(topics[:3])  # Include up to 3 topics
        return f"{text} [{topic_str}]"

    return text
