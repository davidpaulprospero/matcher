"""Recovery keyword generation for high matches mode.

Generates targeted keywords for segments with low confidence matches
to improve coverage in subsequent download iterations.

Strategies:
- weak_segments: Simple word extraction from segment text
- diversify: Find different angles not yet covered
- llm: Use LLM to generate search-optimized keywords (best quality)
- both: Combine weak_segments and diversify
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .coverage_analyzer import WeakSegment

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

# Common stop words to filter out
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'this', 'that',
    'these', 'those', 'what', 'which', 'who', 'whom', 'its', 'it', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'would', 'could', 'might',
    'must', 'shall', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they',
    'them', 'their', 'theirs', 'themselves', 'as', 'if', 'because',
}

# Suffixes to add for better search results
SEARCH_SUFFIXES = ['footage', '4K', 'video', 'documentary']


def extract_keywords_from_text(
    text: str,
    max_keywords: int = 5,
    min_word_length: int = 3,
) -> List[str]:
    """Extract keywords from text using simple word frequency.

    Args:
        text: Text to extract keywords from
        max_keywords: Maximum keywords to return
        min_word_length: Minimum word length to consider

    Returns:
        List of extracted keywords
    """
    if not text:
        return []

    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    # Filter by length and stop words
    filtered = [
        w for w in words
        if len(w) >= min_word_length and w not in STOP_WORDS
    ]

    # Count frequencies
    counter = Counter(filtered)

    # Get top keywords
    keywords = [word for word, _ in counter.most_common(max_keywords)]
    return keywords


def generate_recovery_keywords(
    weak_segments: List[WeakSegment],
    existing_keywords: List[str],
    existing_videos: Optional[List[Any]] = None,
    strategy: str = "weak_segments",
    max_keywords: int = 10,
    keywords_per_segment: int = 2,
    config: Optional['Config'] = None,
) -> List[str]:
    """Generate targeted keywords for segments with low confidence.

    Args:
        weak_segments: List of WeakSegment objects to improve
        existing_keywords: Keywords already used for downloads
        existing_videos: List of already downloaded videos (for diversification)
        strategy: Keyword generation strategy
            - weak_segments: Extract from weak segment text (simple)
            - diversify: Find different angles not yet covered
            - llm: Use LLM to generate search-optimized keywords (best)
            - both: Combine weak_segments and diversify
        max_keywords: Maximum total keywords to return
        keywords_per_segment: Keywords to extract per weak segment
        config: Optional Config object for LLM strategy

    Returns:
        List of new keywords for video search
    """
    logger.info(f"[recovery_keywords] === GENERATING RECOVERY KEYWORDS ===")
    logger.info(f"[recovery_keywords] Input: {len(weak_segments)} weak segments")
    logger.info(f"[recovery_keywords] Existing keywords: {len(existing_keywords)}")
    logger.info(f"[recovery_keywords] Existing videos: {len(existing_videos) if existing_videos else 0}")
    logger.info(f"[recovery_keywords] Strategy: {strategy}")
    logger.info(f"[recovery_keywords] Max keywords: {max_keywords}, per segment: {keywords_per_segment}")

    if not weak_segments:
        logger.info("[recovery_keywords] No weak segments - no recovery keywords needed")
        return []

    existing_set = set(k.lower() for k in existing_keywords)
    new_keywords: List[str] = []

    # Log weakest segments for context
    logger.info(f"[recovery_keywords] Weakest 3 segments to target:")
    for ws in weak_segments[:3]:
        logger.info(f"[recovery_keywords]   {ws.segment_id} ({ws.current_confidence:.2f}): '{ws.text[:60]}...'")

    # LLM strategy - use AI to generate search-optimized keywords
    if strategy == "llm":
        logger.info(f"[recovery_keywords] Using LLM to generate search keywords...")
        llm_keywords = _generate_llm_keywords(
            weak_segments,
            existing_set,
            max_keywords,
            config,
        )
        logger.info(f"[recovery_keywords] LLM generated {len(llm_keywords)} keywords")
        new_keywords.extend(llm_keywords)

    # Simple extraction strategies
    elif strategy in ("weak_segments", "both"):
        logger.info(f"[recovery_keywords] Extracting keywords from weak segment text...")
        extracted = _extract_from_weak_segments(
            weak_segments,
            existing_set,
            keywords_per_segment,
        )
        logger.info(f"[recovery_keywords] Extracted {len(extracted)} keywords from segments")
        new_keywords.extend(extracted)

    if strategy in ("diversify", "both"):
        logger.info(f"[recovery_keywords] Generating diversity keywords...")
        diversity = _generate_diversity_keywords(
            weak_segments,
            existing_set,
            existing_videos,
        )
        logger.info(f"[recovery_keywords] Generated {len(diversity)} diversity keywords")
        new_keywords.extend(diversity)

    logger.info(f"[recovery_keywords] Total raw keywords: {len(new_keywords)}")

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_keywords = []
    duplicates = 0
    filtered_existing = 0

    for kw in new_keywords:
        kw_lower = kw.lower()
        if kw_lower in existing_set:
            filtered_existing += 1
        elif kw_lower in seen:
            duplicates += 1
        else:
            seen.add(kw_lower)
            unique_keywords.append(kw)

    logger.info(f"[recovery_keywords] Filtered: {filtered_existing} existing, {duplicates} duplicates")

    # Limit to max_keywords
    result = unique_keywords[:max_keywords]

    logger.info(f"[recovery_keywords] === RECOVERY KEYWORDS COMPLETE ===")
    logger.info(f"[recovery_keywords] Final keywords ({len(result)}):")
    for i, kw in enumerate(result, 1):
        logger.info(f"[recovery_keywords]   {i}. {kw}")

    return result


def _extract_from_weak_segments(
    weak_segments: List[WeakSegment],
    existing_set: Set[str],
    keywords_per_segment: int,
) -> List[str]:
    """Extract keywords from weak segment text."""
    keywords = []

    # Prioritize segments with lowest confidence
    sorted_segments = sorted(weak_segments, key=lambda s: s.current_confidence)

    for seg in sorted_segments:
        seg_keywords = extract_keywords_from_text(
            seg.text,
            max_keywords=keywords_per_segment * 2,  # Extra for filtering
        )

        # Filter out already used keywords
        new_kws = [k for k in seg_keywords if k.lower() not in existing_set]

        if new_kws:
            logger.debug(f"[recovery_keywords] {seg.segment_id}: extracted {new_kws[:keywords_per_segment]}")

        # Add suffixes for better search
        for kw in new_kws[:keywords_per_segment]:
            # Try with "footage" suffix first
            kw_with_suffix = f"{kw} footage"
            keywords.append(kw_with_suffix)

    return keywords


def _generate_diversity_keywords(
    weak_segments: List[WeakSegment],
    existing_set: Set[str],
    existing_videos: Optional[List[Any]],
) -> List[str]:
    """Generate diverse keywords covering different topics."""
    keywords = []

    # Analyze what topics are covered by existing videos
    covered_topics: Set[str] = set()
    if existing_videos:
        for vid in existing_videos:
            if isinstance(vid, dict):
                title = vid.get('title', '')
                keyword = vid.get('keyword', '')
            else:
                title = getattr(vid, 'title', '')
                keyword = getattr(vid, 'keyword', '')

            # Extract topics from title/keyword
            for word in extract_keywords_from_text(f"{title} {keyword}"):
                covered_topics.add(word.lower())

    # Find topics in weak segments not covered
    for seg in weak_segments:
        seg_topics = extract_keywords_from_text(seg.text)
        for topic in seg_topics:
            topic_lower = topic.lower()
            if topic_lower not in covered_topics and topic_lower not in existing_set:
                # Add with different suffix for variety
                keywords.append(f"{topic} documentary")
                covered_topics.add(topic_lower)

    return keywords


def format_keywords_for_search(
    keywords: List[str],
    add_suffix: bool = True,
    suffix: str = "footage",
) -> List[str]:
    """Format keywords for YouTube search.

    Args:
        keywords: Raw keywords
        add_suffix: Whether to add search suffix
        suffix: Suffix to add (e.g., "footage", "4K", "video")

    Returns:
        Formatted search queries
    """
    formatted = []
    for kw in keywords:
        # Check if keyword already has a suffix
        has_suffix = any(s in kw.lower() for s in SEARCH_SUFFIXES)

        if add_suffix and not has_suffix:
            formatted.append(f"{kw} {suffix}")
        else:
            formatted.append(kw)

    return formatted


def _generate_llm_keywords(
    weak_segments: List[WeakSegment],
    existing_set: Set[str],
    max_keywords: int,
    config: Optional['Config'] = None,
) -> List[str]:
    """Use LLM to generate search-optimized keywords for weak segments.

    This produces much better keywords than simple word extraction because
    the LLM understands context and can generate visually-oriented search
    queries that will find relevant stock footage.

    Args:
        weak_segments: Segments needing better matches
        existing_set: Keywords already used (to avoid)
        max_keywords: Maximum keywords to generate
        config: Config object for LLM settings

    Returns:
        List of search-optimized keywords
    """
    logger.info(f"[recovery_keywords] " + "=" * 50)
    logger.info(f"[recovery_keywords] LLM KEYWORD GENERATION STARTING")
    logger.info(f"[recovery_keywords] " + "=" * 50)
    logger.info(f"[recovery_keywords] Input:")
    logger.info(f"[recovery_keywords]   weak_segments: {len(weak_segments)}")
    logger.info(f"[recovery_keywords]   existing_keywords: {len(existing_set)}")
    logger.info(f"[recovery_keywords]   max_keywords: {max_keywords}")
    logger.info(f"[recovery_keywords]   config provided: {config is not None}")

    if not weak_segments:
        logger.info("[recovery_keywords] No weak segments - skipping LLM generation")
        return []

    try:
        from ..llm_client import create_client_from_config, LLMRequest, ResponseFormat
        logger.info("[recovery_keywords] LLM client imported successfully")
    except ImportError as e:
        logger.warning(f"[recovery_keywords] LLM client import failed: {e}")
        logger.warning("[recovery_keywords] Falling back to simple extraction")
        return []

    # Prepare segment texts for the prompt (batch for efficiency)
    # Take the weakest segments first, limited to avoid token limits
    sorted_segments = sorted(weak_segments, key=lambda s: s.current_confidence)
    segments_to_process = sorted_segments[:min(20, len(sorted_segments))]

    logger.info(f"[recovery_keywords] Processing {len(segments_to_process)} weakest segments:")
    for i, seg in enumerate(segments_to_process[:5], 1):
        logger.info(f"[recovery_keywords]   {i}. {seg.segment_id} (conf={seg.current_confidence:.2f}): '{seg.text[:50]}...'")
    if len(segments_to_process) > 5:
        logger.info(f"[recovery_keywords]   ... and {len(segments_to_process) - 5} more")

    segment_texts = []
    for i, seg in enumerate(segments_to_process, 1):
        segment_texts.append(f"{i}. [{seg.segment_id}] \"{seg.text}\"")

    segments_block = "\n".join(segment_texts)
    existing_block = ", ".join(list(existing_set)[:30]) if existing_set else "none"

    prompt = f"""You are helping find stock footage for a video project. I have voiceover segments that need matching B-roll footage, but current search results don't match well.

Generate {max_keywords} YouTube search queries to find stock footage that would visually match these voiceover segments.

SEGMENTS NEEDING FOOTAGE:
{segments_block}

KEYWORDS ALREADY TRIED (avoid these):
{existing_block}

REQUIREMENTS:
1. Focus on VISUAL elements that can be filmed (actions, scenes, objects, emotions)
2. Include terms like "footage", "stock", "4K", "cinematic" for better YouTube results
3. Be specific but searchable (e.g., "dog tilting head confused 4K" not just "confused")
4. Each keyword should target 1-3 related segments
5. Prioritize the segments with lowest confidence (listed first)

Return a JSON array of search queries:
["query 1", "query 2", ...]

Generate exactly {max_keywords} unique, search-optimized queries:"""

    logger.info(f"[recovery_keywords] Prompt prepared ({len(prompt)} chars)")
    logger.info(f"[recovery_keywords] Sending to LLM...")

    try:
        # Create LLM client
        if config:
            logger.info("[recovery_keywords] Creating LLM client from config...")
            client = create_client_from_config(config)
            logger.info(f"[recovery_keywords] LLM client created: {type(client).__name__}")
        else:
            # Fallback to default Gemini
            logger.info("[recovery_keywords] No config, using default Gemini client...")
            from ..llm_client import create_client
            import os
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                logger.warning("[recovery_keywords] No GEMINI_API_KEY, falling back to extraction")
                return []
            client = create_client("gemini", api_key=api_key)
            logger.info("[recovery_keywords] Default Gemini client created")

        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON_ARRAY,
            temperature=0.7,  # Some creativity for diverse keywords
            max_tokens=1000,
            cache_key_prefix="recovery_keywords",
        )

        logger.info("[recovery_keywords] Calling LLM.generate()...")
        response = client.generate(request)
        logger.info(f"[recovery_keywords] LLM response received")
        logger.info(f"[recovery_keywords]   response.text length: {len(response.text) if response.text else 0}")
        logger.info(f"[recovery_keywords]   response.parsed_data type: {type(response.parsed_data)}")

        if response.parsed_data and isinstance(response.parsed_data, list):
            keywords = [str(kw).strip() for kw in response.parsed_data if kw]
            logger.info(f"[recovery_keywords] " + "=" * 50)
            logger.info(f"[recovery_keywords] LLM KEYWORD GENERATION COMPLETE")
            logger.info(f"[recovery_keywords] " + "=" * 50)
            logger.info(f"[recovery_keywords] Generated {len(keywords)} keywords:")

            # Log all generated keywords
            for i, kw in enumerate(keywords, 1):
                logger.info(f"[recovery_keywords]   {i:2d}. {kw}")

            result = keywords[:max_keywords]
            logger.info(f"[recovery_keywords] Returning {len(result)} keywords (max={max_keywords})")
            return result
        else:
            logger.warning(f"[recovery_keywords] LLM returned unexpected format: {type(response.parsed_data)}")
            logger.info(f"[recovery_keywords] Raw response text: {response.text[:500] if response.text else 'None'}...")
            # Try to parse raw text
            if response.text:
                try:
                    parsed = json.loads(response.text)
                    if isinstance(parsed, list):
                        keywords = [str(kw).strip() for kw in parsed[:max_keywords]]
                        logger.info(f"[recovery_keywords] Parsed {len(keywords)} keywords from raw text")
                        return keywords
                except json.JSONDecodeError as e:
                    logger.warning(f"[recovery_keywords] JSON parse failed: {e}")
            return []

    except Exception as e:
        logger.error(f"[recovery_keywords] LLM generation failed: {e}", exc_info=True)
        logger.info("[recovery_keywords] Falling back to simple extraction...")
        # Fallback to simple extraction
        fallback = _extract_from_weak_segments(
            weak_segments,
            existing_set,
            keywords_per_segment=2,
        )[:max_keywords]
        logger.info(f"[recovery_keywords] Fallback generated {len(fallback)} keywords")
        return fallback
