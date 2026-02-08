"""
Maps caption/voiceover segments to YouTube chapters based on temporal overlap.

US-70-009: Chapter-aware segment mapping for chapter topic scoring.
"""

from typing import Dict, List, Optional, Tuple
from .detector import VideoChapter


def map_segments_to_chapters(
    segments: List,
    chapters: List[VideoChapter],
) -> Dict[int, Tuple[int, str]]:
    """
    Map segments to chapters based on temporal overlap.

    Each segment is assigned to the chapter with the most temporal overlap.
    Segments outside all chapter ranges get chapter_index=-1.

    Args:
        segments: List of segments with start_time and end_time attributes.
        chapters: List of VideoChapter objects (sorted by start_time).

    Returns:
        Dict mapping segment index to (chapter_index, chapter_title).
        chapter_index=-1 and title="" for segments outside all chapters.
    """
    if not segments or not chapters:
        return {i: (-1, "") for i in range(len(segments))}

    result: Dict[int, Tuple[int, str]] = {}

    for seg_idx, segment in enumerate(segments):
        seg_start = getattr(segment, 'start_time', 0.0)
        seg_end = getattr(segment, 'end_time', 0.0)

        best_chapter_idx = -1
        best_chapter_title = ""
        best_overlap = 0.0

        for ch_idx, chapter in enumerate(chapters):
            ch_end = chapter.end_time if chapter.end_time is not None else float('inf')

            # Calculate temporal overlap
            overlap_start = max(seg_start, chapter.start_time)
            overlap_end = min(seg_end, ch_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_chapter_idx = ch_idx
                best_chapter_title = chapter.title

        result[seg_idx] = (best_chapter_idx, best_chapter_title)

    return result
