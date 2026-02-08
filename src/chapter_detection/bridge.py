"""
Bridge between listicle detection and chapter detection systems.

Converts ListicleGroup objects into ChapterCandidate objects so that
downstream scoring adjustments (chapter_topic_match, chapter_source_consistency,
chapter_coherence_penalty, cross_chapter_relevance) work uniformly regardless
of whether structure came from YouTube chapters or listicle detection.

US-71-010, US-72-003
"""

import logging
from typing import List, Dict, Optional, Callable

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

        # Graduated confidence based on marker type
        marker_confidence = {
            'transition': 0.6,
            'ordinal': 0.7,
            'numbered': 0.8,
        }
        confidence = marker_confidence.get(group.marker_type, 0.7)

        # Boost to 0.85 when expected_count matches detected group count
        if (group.expected_count is not None
                and group.expected_count == len(groups)):
            confidence = 0.85

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


def compute_relevance_matrix(
    voiceover_chapters: List[ChapterCandidate],
    video_chapters: List[ChapterCandidate],
    embedding_fn: Optional[Callable[[str, str], float]] = None,
) -> List[List[float]]:
    """
    Compute a cross-chapter relevance matrix for candidate boosting (US-72-009).

    Each cell [i][j] is a similarity score between voiceover chapter i and
    video chapter j. When embedding_fn is provided, the score is a weighted
    blend: 0.6 * Jaccard keyword similarity + 0.4 * embedding cosine similarity.
    When embedding_fn is None, pure Jaccard similarity is used.

    Args:
        voiceover_chapters: Voiceover ChapterCandidate objects with topics lists
        video_chapters: Video ChapterCandidate objects with topics lists
        embedding_fn: Optional callable(text_a, text_b) -> float cosine similarity
                      in [0.0, 1.0]. When provided, blends with Jaccard.

    Returns:
        2D list of floats (vo_chapters x video_chapters), each in [0.0, 1.0].
        Empty list if either input is empty.
    """
    if not voiceover_chapters or not video_chapters:
        return []

    # Extract keyword lists from ChapterCandidate.topics
    vo_keywords = [ch.topics if ch.topics else [] for ch in voiceover_chapters]
    vid_keywords = [ch.topics if ch.topics else [] for ch in video_chapters]

    # Build chapter text representations for embedding similarity
    if embedding_fn is not None:
        vo_texts = [' '.join(kw) for kw in vo_keywords]
        vid_texts = [' '.join(kw) for kw in vid_keywords]

    # Compute matrix with optional embedding blend
    matrix = []
    for i, vo_kw in enumerate(vo_keywords):
        vo_set = {k.lower() for k in vo_kw} if vo_kw else set()
        row = []
        for j, vid_kw in enumerate(vid_keywords):
            vid_set = {k.lower() for k in vid_kw} if vid_kw else set()
            union = vo_set | vid_set
            jaccard = (len(vo_set & vid_set) / len(union)) if union else 0.0

            if embedding_fn is not None:
                cosine_sim = embedding_fn(vo_texts[i], vid_texts[j])
                score = 0.6 * jaccard + 0.4 * cosine_sim
            else:
                score = jaccard

            row.append(score)
        matrix.append(row)

    return matrix


def _compute_overlap(seg_start: float, seg_end: float, ch_start: float, ch_end: float) -> float:
    """Compute the overlap duration between a segment and a chapter time range."""
    overlap_start = max(seg_start, ch_start)
    overlap_end = min(seg_end, ch_end)
    return max(0.0, overlap_end - overlap_start)


def assign_chapter_indices(
    segments: List,
    chapters: List[Dict],
) -> None:
    """
    Assign chapter_index and chapter_title to each TranscriptSegment by timestamp overlap.

    Each segment is assigned to the chapter with the greatest overlap duration.
    Segments spanning two chapters are assigned to the one with >50% overlap.
    Segments outside all chapter ranges get chapter_index=None, chapter_title=''.

    Mutates segments in-place.

    Args:
        segments: List of TranscriptSegment objects (must have start_time, end_time)
        chapters: List of chapter dicts with keys: title, start_time, end_time.
                  Chapters are assumed to be sorted by start_time.

    US-72-003
    """
    if not chapters:
        return

    for seg in segments:
        best_idx: Optional[int] = None
        best_overlap: float = 0.0
        seg_duration = seg.end_time - seg.start_time

        for i, ch in enumerate(chapters):
            ch_start = ch.get('start_time', 0.0)
            ch_end = ch.get('end_time', 0.0)
            overlap = _compute_overlap(seg.start_time, seg.end_time, ch_start, ch_end)

            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        # Only assign if there's meaningful overlap (>0)
        if best_idx is not None and best_overlap > 0.0:
            seg.chapter_index = best_idx
            seg.chapter_title = chapters[best_idx].get('title', '')
        else:
            seg.chapter_index = None
            seg.chapter_title = ''
