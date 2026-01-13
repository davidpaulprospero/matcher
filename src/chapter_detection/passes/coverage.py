"""
Coverage Resolution Pass

Pass 4: Ensure complete, non-overlapping chapter coverage.
"""

import logging
from typing import List, Dict, Any, Tuple

from ..models import ChapterCandidate

logger = logging.getLogger(__name__)


def run_coverage_resolution(
    chapters: List[ChapterCandidate],
    total_segments: int,
    config: Any,
) -> List[ChapterCandidate]:
    """
    Ensure chapters cover all segments without gaps or overlaps.

    Args:
        chapters: Detected chapters
        total_segments: Total number of segments in transcript
        config: Configuration

    Returns:
        Chapters with complete, non-overlapping coverage
    """
    if not chapters:
        return chapters

    if total_segments == 0:
        return chapters

    # Sort by start index
    chapters = sorted(chapters, key=lambda c: c.start_segment_idx)

    # Get config values
    min_segments = _get_min_segments(config)

    # Step 1: Resolve overlaps
    chapters = _resolve_overlaps(chapters)

    # Step 2: Fill gaps
    chapters = _fill_gaps(chapters, total_segments)

    # Step 3: Merge tiny chapters
    chapters = _merge_tiny_chapters(chapters, min_segments)

    # Step 4: Renumber
    for i, ch in enumerate(chapters):
        ch.chapter_id = i

    logger.info(f"Coverage resolution complete: {len(chapters)} chapters covering {total_segments} segments")
    return chapters


def _resolve_overlaps(chapters: List[ChapterCandidate]) -> List[ChapterCandidate]:
    """Resolve overlapping chapters by adjusting boundaries."""
    if len(chapters) < 2:
        return chapters

    result = [chapters[0]]

    for i in range(1, len(chapters)):
        current = chapters[i]
        prev = result[-1]

        if current.start_segment_idx <= prev.end_segment_idx:
            # Overlap detected
            overlap_start = current.start_segment_idx
            overlap_end = min(prev.end_segment_idx, current.end_segment_idx)
            overlap_size = overlap_end - overlap_start + 1

            # Choose split point (middle of overlap)
            split_point = overlap_start + overlap_size // 2

            # Adjust boundaries
            if prev.confidence >= current.confidence:
                # Previous chapter wins overlap
                current.start_segment_idx = split_point + 1
                current.boundary_reasoning += " [Start adjusted for overlap]"
            else:
                # Current chapter wins overlap
                prev.end_segment_idx = split_point
                prev.boundary_reasoning += " [End adjusted for overlap]"

            # Only add if still valid
            if current.start_segment_idx <= current.end_segment_idx:
                result.append(current)
        else:
            result.append(current)

    return result


def _fill_gaps(
    chapters: List[ChapterCandidate],
    total_segments: int,
) -> List[ChapterCandidate]:
    """Fill gaps between chapters."""
    if not chapters:
        # No chapters, create one covering all segments
        return [ChapterCandidate(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=total_segments - 1,
            title="Main Content",
            confidence=0.5,
            detection_strategy='gap_fill',
            boundary_reasoning="Created to cover entire transcript",
        )]

    result = []

    # Check for gap at start
    if chapters[0].start_segment_idx > 0:
        gap_chapter = ChapterCandidate(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=chapters[0].start_segment_idx - 1,
            title="Introduction",
            confidence=0.5,
            detection_strategy='gap_fill',
            boundary_reasoning="Created to fill gap at start",
        )
        result.append(gap_chapter)

    # Process chapters and gaps between them
    for i, chapter in enumerate(chapters):
        result.append(chapter)

        if i < len(chapters) - 1:
            next_chapter = chapters[i + 1]
            gap_start = chapter.end_segment_idx + 1
            gap_end = next_chapter.start_segment_idx - 1

            if gap_start <= gap_end:
                # There's a gap
                gap_size = gap_end - gap_start + 1

                if gap_size <= 3:
                    # Small gap - extend adjacent chapter
                    if chapter.confidence >= next_chapter.confidence:
                        chapter.end_segment_idx = gap_end
                        chapter.boundary_reasoning += f" [Extended to fill {gap_size} segment gap]"
                    else:
                        next_chapter.start_segment_idx = gap_start
                        next_chapter.boundary_reasoning += f" [Extended to fill {gap_size} segment gap]"
                else:
                    # Large gap - create new chapter
                    gap_chapter = ChapterCandidate(
                        chapter_id=len(result),
                        start_segment_idx=gap_start,
                        end_segment_idx=gap_end,
                        title=f"Section {len(result) + 1}",
                        confidence=0.4,
                        detection_strategy='gap_fill',
                        boundary_reasoning=f"Created to fill {gap_size} segment gap",
                    )
                    result.append(gap_chapter)

    # Check for gap at end
    last_chapter = result[-1]
    if last_chapter.end_segment_idx < total_segments - 1:
        gap_start = last_chapter.end_segment_idx + 1
        gap_end = total_segments - 1
        gap_size = gap_end - gap_start + 1

        if gap_size <= 3:
            # Extend last chapter
            last_chapter.end_segment_idx = gap_end
            last_chapter.boundary_reasoning += f" [Extended to fill {gap_size} segment gap at end]"
        else:
            # Create conclusion chapter
            gap_chapter = ChapterCandidate(
                chapter_id=len(result),
                start_segment_idx=gap_start,
                end_segment_idx=gap_end,
                title="Conclusion",
                confidence=0.5,
                detection_strategy='gap_fill',
                boundary_reasoning="Created to fill gap at end",
            )
            result.append(gap_chapter)

    return result


def _merge_tiny_chapters(
    chapters: List[ChapterCandidate],
    min_segments: int,
) -> List[ChapterCandidate]:
    """Merge chapters that are smaller than minimum size."""
    if not chapters or min_segments <= 1:
        return chapters

    result = []

    for chapter in chapters:
        chapter_size = chapter.end_segment_idx - chapter.start_segment_idx + 1

        if chapter_size < min_segments and result:
            # Merge with previous chapter
            prev = result[-1]

            # Combine properties
            prev.end_segment_idx = chapter.end_segment_idx
            prev.topics = list(dict.fromkeys(prev.topics + chapter.topics))

            if chapter.title and chapter.title != prev.title:
                prev.title = f"{prev.title} & {chapter.title}"

            prev.confidence = (prev.confidence + chapter.confidence) / 2
            prev.boundary_reasoning += f" [Merged with tiny chapter: {chapter.title}]"

        elif chapter_size < min_segments and not result:
            # First chapter is tiny, keep it but mark low confidence
            chapter.confidence = max(0.3, chapter.confidence - 0.2)
            result.append(chapter)

        else:
            result.append(chapter)

    return result


def _get_min_segments(config: Any) -> int:
    """Get minimum chapter size from config."""
    default = 3

    if config is None:
        return default

    if hasattr(config, 'matching'):
        matching = config.matching
        if hasattr(matching, 'chapter_detection'):
            ch_config = matching.chapter_detection
            if isinstance(ch_config, dict):
                return ch_config.get('min_chapter_segments', default)
            return getattr(ch_config, 'min_chapter_segments', default)

    return default
