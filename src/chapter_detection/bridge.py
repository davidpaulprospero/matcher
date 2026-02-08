"""
Bridge between listicle detection and chapter detection systems.

Converts ListicleGroup objects into ChapterCandidate objects so that
downstream scoring adjustments (chapter_topic_match, chapter_source_consistency,
chapter_coherence_penalty, cross_chapter_relevance) work uniformly regardless
of whether structure came from YouTube chapters or listicle detection.

US-71-010
"""

import logging
from typing import List, Dict

from .models import ChapterCandidate, ListicleGroup

logger = logging.getLogger(__name__)


def listicle_groups_to_chapters(groups: List[ListicleGroup]) -> List[ChapterCandidate]:
    """
    Convert ListicleGroup objects into ChapterCandidate objects.

    Each listicle group becomes a chapter with:
    - title derived from item_label and topic_keywords
    - topics populated from topic_keywords
    - detection_strategy set to 'listicle'
    - confidence based on presence of expected_count validation

    Args:
        groups: List of ListicleGroup objects from listicle detection

    Returns:
        List of ChapterCandidate objects
    """
    chapters = []
    for group in groups:
        # Build a descriptive title from the label and keywords
        title_parts = [group.item_label] if group.item_label else []
        if group.topic_keywords:
            title_parts.append(' '.join(group.topic_keywords[:3]))
        title = ' - '.join(title_parts) if title_parts else f"Item {group.group_id + 1}"

        # Confidence: higher if expected_count matches actual detection
        confidence = 0.7  # Base confidence for listicle-derived chapters
        if group.expected_count is not None:
            confidence = 0.75  # Slightly higher when header count was detected

        chapter = ChapterCandidate(
            chapter_id=group.group_id,
            start_segment_idx=group.start_segment_idx,
            end_segment_idx=group.end_segment_idx,
            title=title,
            topics=list(group.topic_keywords),
            confidence=confidence,
            detection_strategy='listicle',
        )
        chapters.append(chapter)

    return chapters


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Check if two segment index ranges overlap."""
    return a_start <= b_end and b_start <= a_end


def merge_chapters(
    youtube_chapters: List[ChapterCandidate],
    listicle_chapters: List[ChapterCandidate],
) -> List[ChapterCandidate]:
    """
    Merge YouTube-derived chapters with listicle-derived chapters.

    YouTube chapters take precedence for overlapping segment ranges.
    Non-overlapping listicle chapters are included to fill gaps.

    Args:
        youtube_chapters: Chapters from YouTube chapter detection (higher priority)
        listicle_chapters: Chapters converted from listicle groups (lower priority)

    Returns:
        Merged list of ChapterCandidate objects sorted by start_segment_idx,
        with chapter_id reassigned sequentially.
    """
    if not youtube_chapters and not listicle_chapters:
        return []
    if not listicle_chapters:
        return list(youtube_chapters)
    if not youtube_chapters:
        return list(listicle_chapters)

    # Start with all YouTube chapters (they take precedence)
    merged = list(youtube_chapters)

    # Add listicle chapters that don't overlap with any YouTube chapter
    for lc in listicle_chapters:
        overlaps = any(
            _ranges_overlap(
                lc.start_segment_idx, lc.end_segment_idx,
                yc.start_segment_idx, yc.end_segment_idx,
            )
            for yc in youtube_chapters
        )
        if not overlaps:
            merged.append(lc)

    # Sort by start index and reassign chapter IDs
    merged.sort(key=lambda c: c.start_segment_idx)
    for i, chapter in enumerate(merged):
        chapter.chapter_id = i

    return merged


def build_unified_chapters(
    location_chapters: List[ChapterCandidate],
    listicle_groups: List[ListicleGroup],
) -> List[ChapterCandidate]:
    """
    Build a unified chapter list from both detection sources.

    This is the main entry point for the bridge. It:
    1. Converts listicle groups to chapter format
    2. Merges with YouTube/location chapters (YouTube takes precedence)
    3. Returns a unified list usable by all chapter-aware scoring

    Args:
        location_chapters: Existing chapters from YouTube/location detection
        listicle_groups: ListicleGroup objects from listicle detection

    Returns:
        Unified list of ChapterCandidate objects
    """
    listicle_chapters = listicle_groups_to_chapters(listicle_groups)

    if not location_chapters and not listicle_chapters:
        return []

    merged = merge_chapters(location_chapters, listicle_chapters)

    yt_count = len(location_chapters) if location_chapters else 0
    listicle_count = len(listicle_chapters) if listicle_chapters else 0
    logger.info(
        "US-71-010 unified chapters: %d YouTube + %d listicle -> %d merged",
        yt_count, listicle_count, len(merged),
    )

    return merged


def build_segment_chapter_map(chapters: List[ChapterCandidate]) -> Dict[int, int]:
    """
    Build a mapping from segment index to chapter index.

    Args:
        chapters: List of ChapterCandidate objects (should be sorted by start_segment_idx)

    Returns:
        Dict mapping segment_index -> chapter_index (chapter_id)
    """
    mapping: Dict[int, int] = {}
    for chapter in chapters:
        for seg_idx in range(chapter.start_segment_idx, chapter.end_segment_idx + 1):
            mapping[seg_idx] = chapter.chapter_id
    return mapping
