"""
Initial Detection Pass

Pass 1: Detect chapters using topic-based (and optionally location/narrative) strategies.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional

from ..models import ChapterCandidate
from ..prompts import (
    format_topic_prompt,
    format_location_prompt,
    format_content_type_prompt,
)
from ..chunking import create_chunks, create_indexed_text, merge_chunk_results

logger = logging.getLogger(__name__)


def run_initial_detection(
    segments: List[Dict[str, Any]],
    llm_client: Any,
    config: Any,
    strategy: str = 'topic',
    overall_topic: str = None,
) -> List[ChapterCandidate]:
    """
    Run initial chapter detection using specified strategy.

    Args:
        segments: List of segment dicts with 'index' and 'text'
        llm_client: LLM client for API calls
        config: Configuration with chapter detection settings
        strategy: Detection strategy ('topic', 'location', 'auto')
        overall_topic: Optional topic context

    Returns:
        List of ChapterCandidate objects
    """
    if not segments:
        return []

    # Get config values
    chapter_config = _get_chapter_config(config)
    max_chunk_chars = chapter_config.get('max_chunk_chars', 6000)
    min_segments = chapter_config.get('min_chapter_segments', 3)
    max_chapters = chapter_config.get('max_chapters', 20)

    # Auto-detect content type if requested
    if strategy == 'auto' and chapter_config.get('auto_detect_content_type', True):
        detected_type = detect_content_type(segments, llm_client)
        strategy = _map_content_type_to_strategy(detected_type)
        logger.info(f"Auto-detected content type: {detected_type} -> strategy: {strategy}")

    # Create chunks for processing
    chunks = create_chunks(segments, max_chars=max_chunk_chars)

    # Process each chunk
    chunk_results = []
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)} (segments {chunk.start_segment_idx}-{chunk.end_segment_idx})")

        # Run detection on this chunk
        if strategy == 'location':
            chapters = _detect_location_chapters(
                chunk.text,
                chunk.segment_count,
                llm_client,
                min_segments=min_segments,
            )
        else:
            chapters = _detect_topic_chapters(
                chunk.text,
                chunk.segment_count,
                llm_client,
                min_segments=min_segments,
                max_chapters=max_chapters,
            )

        # Adjust indices for chunk offset
        for ch in chapters:
            ch['start_segment_idx'] += chunk.start_segment_idx
            ch['end_segment_idx'] += chunk.start_segment_idx

        chunk_results.append(chapters)

    # Merge results from all chunks
    merged = merge_chunk_results(chunk_results, chunks, len(segments))

    # Convert to ChapterCandidate objects
    candidates = []
    for ch in merged:
        candidate = _dict_to_candidate(ch, strategy)
        candidates.append(candidate)

    logger.info(f"Initial detection found {len(candidates)} chapters")
    return candidates


def detect_content_type(
    segments: List[Dict[str, Any]],
    llm_client: Any
) -> str:
    """
    Detect the content type of the transcript.

    Returns: 'travel', 'educational', 'documentary', 'narrative', or 'general'
    """
    # Get text sample from first ~2000 chars
    text_parts = []
    total_chars = 0
    for seg in segments:
        text = seg.get('text', '')
        if total_chars + len(text) > 2000:
            text_parts.append(text[:2000 - total_chars])
            break
        text_parts.append(text)
        total_chars += len(text)

    text_sample = " ".join(text_parts)

    if not text_sample.strip():
        return 'general'

    prompt = format_content_type_prompt(text_sample)

    try:
        from src.llm_client import LLMRequest, ResponseFormat

        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON,
            max_tokens=200,
            cache_key_prefix="content_type_detection"
        )
        response = llm_client.generate(request)

        if response.parsed_data and isinstance(response.parsed_data, dict):
            return response.parsed_data.get('content_type', 'general')

    except Exception as e:
        logger.warning(f"Content type detection failed: {e}")

    return 'general'


def _detect_topic_chapters(
    indexed_text: str,
    total_segments: int,
    llm_client: Any,
    min_segments: int = 3,
    max_chapters: int = 20,
) -> List[Dict[str, Any]]:
    """Detect chapters using topic-based strategy."""
    prompt = format_topic_prompt(
        indexed_text=indexed_text,
        total_segments=total_segments,
        min_segments=min_segments,
        max_chapters=max_chapters,
    )

    try:
        from src.llm_client import LLMRequest, ResponseFormat

        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON_ARRAY,
            max_tokens=2000,
            cache_key_prefix="chapter_detection_topic"
        )
        response = llm_client.generate(request)

        if response.parsed_data and isinstance(response.parsed_data, list):
            chapters = _validate_chapters(response.parsed_data, total_segments, min_segments)
            return chapters

    except Exception as e:
        logger.warning(f"Topic chapter detection failed: {e}")

    return []


def _detect_location_chapters(
    indexed_text: str,
    total_segments: int,
    llm_client: Any,
    min_segments: int = 3,
) -> List[Dict[str, Any]]:
    """Detect chapters using location-based strategy."""
    prompt = format_location_prompt(
        indexed_text=indexed_text,
        total_segments=total_segments,
        min_segments=min_segments,
    )

    try:
        from src.llm_client import LLMRequest, ResponseFormat

        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON_ARRAY,
            max_tokens=2000,
            cache_key_prefix="chapter_detection_location"
        )
        response = llm_client.generate(request)

        if response.parsed_data and isinstance(response.parsed_data, list):
            chapters = _validate_chapters(response.parsed_data, total_segments, min_segments)
            return chapters

    except Exception as e:
        logger.warning(f"Location chapter detection failed: {e}")

    return []


def _validate_chapters(
    chapters: List[Dict[str, Any]],
    total_segments: int,
    min_segments: int = 3,
) -> List[Dict[str, Any]]:
    """Validate and clean chapter data from LLM response."""
    valid = []

    for idx, ch in enumerate(chapters):
        if not isinstance(ch, dict):
            continue

        # Get and validate segment indices
        start = ch.get('start_segment_idx', 0)
        end = ch.get('end_segment_idx', total_segments - 1)

        # Clamp to valid range
        start = max(0, min(start, total_segments - 1))
        end = max(start, min(end, total_segments - 1))

        # Check minimum size
        if end - start + 1 < min_segments:
            logger.debug(f"Skipping chapter {idx}: too small ({end - start + 1} < {min_segments} segments)")
            continue

        valid_ch = {
            'chapter_id': len(valid),
            'start_segment_idx': start,
            'end_segment_idx': end,
            'title': ch.get('title', f'Chapter {len(valid) + 1}'),
            'topics': ch.get('topics', []),
            'location_name': ch.get('location_name', ''),
            'location_type': ch.get('location_type', 'city'),
            'visual_keywords': ch.get('visual_keywords', []),
            'context_keywords': ch.get('context_keywords', []),
            'boundary_reasoning': ch.get('boundary_reasoning', ''),
            'confidence': ch.get('confidence', 'medium'),
        }
        valid.append(valid_ch)

    return valid


def _dict_to_candidate(ch: Dict[str, Any], strategy: str) -> ChapterCandidate:
    """Convert chapter dict to ChapterCandidate object."""
    # Map confidence string to float
    conf_str = ch.get('confidence', 'medium')
    conf_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
    confidence = conf_map.get(conf_str, 0.7)

    return ChapterCandidate(
        chapter_id=ch.get('chapter_id', 0),
        start_segment_idx=ch.get('start_segment_idx', 0),
        end_segment_idx=ch.get('end_segment_idx', 0),
        title=ch.get('title', ''),
        topics=ch.get('topics', []),
        location_name=ch.get('location_name', ''),
        location_type=ch.get('location_type', 'city'),
        visual_keywords=ch.get('visual_keywords', []),
        context_keywords=ch.get('context_keywords', []),
        confidence=confidence,
        detection_strategy=strategy,
        boundary_reasoning=ch.get('boundary_reasoning', ''),
    )


def _get_chapter_config(config: Any) -> Dict[str, Any]:
    """Extract chapter detection config with safe defaults."""
    defaults = {
        'max_chunk_chars': 6000,
        'min_chapter_segments': 3,
        'max_chapters': 20,
        'auto_detect_content_type': True,
    }

    if config is None:
        return defaults

    # Try to get chapter_detection config
    chapter_config = None
    if hasattr(config, 'matching'):
        matching = config.matching
        if hasattr(matching, 'chapter_detection'):
            chapter_config = matching.chapter_detection

    if chapter_config is None:
        return defaults

    # Extract values
    result = {}
    for key, default in defaults.items():
        if isinstance(chapter_config, dict):
            result[key] = chapter_config.get(key, default)
        else:
            result[key] = getattr(chapter_config, key, default)

    return result


def _map_content_type_to_strategy(content_type: str) -> str:
    """Map detected content type to detection strategy."""
    mapping = {
        'travel': 'location',
        'educational': 'topic',
        'documentary': 'topic',
        'narrative': 'topic',
        'general': 'topic',
    }
    return mapping.get(content_type, 'topic')
