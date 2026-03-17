"""
Transcript Chunking

Handles long transcripts by splitting into overlapping chunks
for processing, then merging results.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text for processing."""
    text: str
    start_segment_idx: int
    end_segment_idx: int
    segment_indices: List[int] = field(default_factory=list)
    overlap_start: int = 0  # Segments shared with previous chunk
    overlap_end: int = 0    # Segments shared with next chunk

    @property
    def segment_count(self) -> int:
        return len(self.segment_indices)


def create_indexed_text(segments: List[Dict[str, Any]]) -> str:
    """
    Create indexed text from segments for LLM processing.

    Format: [0] First segment text\n[1] Second segment text...
    """
    lines = []
    for seg in segments:
        idx = seg.get('index', len(lines))
        text = seg.get('text', '').strip()
        lines.append(f"[{idx}] {text}")
    return "\n".join(lines)


def create_chunks(
    segments: List[Dict[str, Any]],
    max_chars: int = 6000,
    overlap_segments: int = 5
) -> List[TextChunk]:
    """
    Create overlapping chunks that fit within token limits.

    Args:
        segments: List of segment dicts with 'index' and 'text' fields
        max_chars: Maximum characters per chunk (estimate ~4 chars per token)
        overlap_segments: Number of segments to overlap between chunks

    Returns:
        List of TextChunk objects
    """
    if not segments:
        return []

    # Build indexed text for all segments first
    all_indexed = create_indexed_text(segments)

    # If it fits in one chunk, return single chunk
    if len(all_indexed) <= max_chars:
        return [TextChunk(
            text=all_indexed,
            start_segment_idx=0,
            end_segment_idx=len(segments) - 1,
            segment_indices=list(range(len(segments))),
            overlap_start=0,
            overlap_end=0,
        )]

    # Need to split into multiple chunks
    chunks = []
    current_start = 0

    while current_start < len(segments):
        # Find how many segments fit in max_chars
        chunk_segments = []
        chunk_text_parts = []
        current_chars = 0

        for i in range(current_start, len(segments)):
            seg = segments[i]
            idx = seg.get('index', i)
            text = seg.get('text', '').strip()
            line = f"[{idx}] {text}\n"

            if current_chars + len(line) > max_chars and chunk_segments:
                # This segment would exceed limit, stop here
                break

            chunk_segments.append(i)
            chunk_text_parts.append(line.rstrip('\n'))
            current_chars += len(line)

        if not chunk_segments:
            # Single segment exceeds max_chars, include it anyway
            seg = segments[current_start]
            idx = seg.get('index', current_start)
            text = seg.get('text', '')[:max_chars - 20]  # Truncate
            chunk_segments = [current_start]
            chunk_text_parts = [f"[{idx}] {text}"]

        # Determine overlap with previous chunk
        overlap_start = 0
        if chunks and overlap_segments > 0:
            prev_chunk = chunks[-1]
            overlap_start = min(overlap_segments, len(chunk_segments))
            prev_chunk.overlap_end = overlap_start

        chunk = TextChunk(
            text="\n".join(chunk_text_parts),
            start_segment_idx=chunk_segments[0],
            end_segment_idx=chunk_segments[-1],
            segment_indices=chunk_segments,
            overlap_start=overlap_start,
            overlap_end=0,  # Will be set by next chunk
        )
        chunks.append(chunk)

        # Move start past the non-overlapping portion
        if len(chunk_segments) <= overlap_segments:
            # Chunk is too small, just move to next
            current_start = chunk_segments[-1] + 1
        else:
            # Move past this chunk minus overlap
            current_start = chunk_segments[-1] - overlap_segments + 1

        # Ensure progress
        if current_start <= chunk_segments[0]:
            current_start = chunk_segments[-1] + 1

    logger.info(f"Created {len(chunks)} chunks from {len(segments)} segments")
    return chunks


def _compute_overlap_range(ch: Dict[str, Any]) -> set:
    """Return the set of segment indices covered by a chapter."""
    start = ch.get('start_segment_idx', 0)
    end = ch.get('end_segment_idx', start)
    return set(range(start, end + 1))


def _segment_overlap_ratio(ch_a: Dict[str, Any], ch_b: Dict[str, Any]) -> float:
    """Return the fraction of overlap between two chapters' segment ranges.

    Overlap ratio = intersection / min(len_a, len_b).
    Returns 0.0 if either chapter has zero range.
    """
    range_a = _compute_overlap_range(ch_a)
    range_b = _compute_overlap_range(ch_b)
    if not range_a or not range_b:
        return 0.0
    intersection = len(range_a & range_b)
    return intersection / min(len(range_a), len(range_b))


def _score_chapter(ch: Dict[str, Any]) -> int:
    """Score a chapter by confidence level (higher is better)."""
    conf_str = ch.get('confidence', 'medium')
    conf_scores = {'high': 3, 'medium': 2, 'low': 1}
    return conf_scores.get(conf_str, 2)


def _deduplicate_overlap_zone(
    chunk_results: List[List[Dict[str, Any]]],
    chunks: List[TextChunk],
) -> List[List[Dict[str, Any]]]:
    """Deduplicate chapters that fall in the overlap zone between adjacent chunks.

    For each pair of adjacent chunks, identifies the overlap region (defined by
    overlap_segments on the chunk boundaries). Chapters from *different* chunks
    whose segment ranges overlap by more than 50% within that zone are
    considered duplicates — the one with higher confidence wins.

    Returns a new list of chapter lists with duplicates removed.
    """
    if len(chunk_results) < 2:
        return [list(chs) for chs in chunk_results]

    # Track which chapters to drop (by chunk_idx, chapter list index)
    to_drop: set = set()

    for ci in range(len(chunks) - 1):
        chunk_a = chunks[ci]
        chunk_b = chunks[ci + 1]

        # The overlap zone is the range of segment indices shared between
        # the end of chunk_a and the start of chunk_b.
        overlap_size = chunk_a.overlap_end
        if overlap_size <= 0:
            continue

        # Overlap zone segment range
        overlap_zone_start = chunk_b.start_segment_idx
        overlap_zone_end = overlap_zone_start + overlap_size - 1

        def _in_overlap_zone(ch: Dict[str, Any]) -> bool:
            """Check if a chapter's segment range intersects the overlap zone."""
            ch_start = ch.get('start_segment_idx', 0)
            ch_end = ch.get('end_segment_idx', ch_start)
            return ch_start <= overlap_zone_end and ch_end >= overlap_zone_start

        # Gather chapters from each adjacent chunk that touch the overlap zone
        chapters_a = [
            (idx, ch) for idx, ch in enumerate(chunk_results[ci])
            if _in_overlap_zone(ch)
        ]
        chapters_b = [
            (idx, ch) for idx, ch in enumerate(chunk_results[ci + 1])
            if _in_overlap_zone(ch)
        ]

        # Compare each pair across the two chunks
        for idx_a, ch_a in chapters_a:
            for idx_b, ch_b in chapters_b:
                if (ci, idx_a) in to_drop or (ci + 1, idx_b) in to_drop:
                    continue
                ratio = _segment_overlap_ratio(ch_a, ch_b)
                if ratio > 0.5:
                    # Keep higher confidence, drop the other
                    if _score_chapter(ch_a) >= _score_chapter(ch_b):
                        to_drop.add((ci + 1, idx_b))
                        logger.debug(
                            f"Dedup overlap: drop chunk {ci+1} ch {idx_b} "
                            f"(conf={ch_b.get('confidence')}) in favour of "
                            f"chunk {ci} ch {idx_a} (conf={ch_a.get('confidence')})"
                        )
                    else:
                        to_drop.add((ci, idx_a))
                        logger.debug(
                            f"Dedup overlap: drop chunk {ci} ch {idx_a} "
                            f"(conf={ch_a.get('confidence')}) in favour of "
                            f"chunk {ci+1} ch {idx_b} (conf={ch_b.get('confidence')})"
                        )

    # Rebuild chapter lists without dropped entries
    deduped = []
    for ci, chapters in enumerate(chunk_results):
        deduped.append([
            ch for idx, ch in enumerate(chapters)
            if (ci, idx) not in to_drop
        ])

    dropped_count = len(to_drop)
    if dropped_count:
        logger.info(f"Deduplicated {dropped_count} chapters in overlap zones")

    return deduped


def merge_chunk_results(
    chunk_results: List[List[Dict[str, Any]]],
    chunks: List[TextChunk],
    total_segments: int
) -> List[Dict[str, Any]]:
    """
    Merge chapter detection results from multiple chunks.

    First deduplicates near-duplicate chapters in chunk overlap zones
    (>50% segment overlap, keeping higher confidence), then merges
    remaining chapters across all chunks.

    Args:
        chunk_results: List of chapter lists, one per chunk
        chunks: List of TextChunk objects
        total_segments: Total number of segments in transcript

    Returns:
        Merged list of chapter dicts
    """
    if not chunk_results:
        return []

    if len(chunk_results) == 1:
        return chunk_results[0]

    # Phase 1: Deduplicate chapters in overlap zones between adjacent chunks
    deduped_results = _deduplicate_overlap_zone(chunk_results, chunks)

    # Phase 2: Collect all remaining chapters with their source chunk
    all_chapters = []
    for chunk_idx, chapters in enumerate(deduped_results):
        for ch in chapters:
            ch_copy = dict(ch)
            ch_copy['_source_chunk'] = chunk_idx
            all_chapters.append(ch_copy)

    # Sort by start segment
    all_chapters.sort(key=lambda c: c.get('start_segment_idx', 0))

    # Merge overlapping chapters
    merged = []
    i = 0

    while i < len(all_chapters):
        current = all_chapters[i]
        current_start = current.get('start_segment_idx', 0)
        current_end = current.get('end_segment_idx', 0)

        # Look for overlapping chapters
        overlapping = [current]
        j = i + 1
        while j < len(all_chapters):
            next_ch = all_chapters[j]
            next_start = next_ch.get('start_segment_idx', 0)

            if next_start <= current_end:
                # Overlaps with current
                overlapping.append(next_ch)
                current_end = max(current_end, next_ch.get('end_segment_idx', 0))
                j += 1
            else:
                break

        # Pick best from overlapping chapters
        if len(overlapping) == 1:
            best = overlapping[0]
        else:
            overlapping.sort(key=_score_chapter, reverse=True)
            best = overlapping[0]

            # Merge boundaries from all overlapping
            all_starts = [ch.get('start_segment_idx', 0) for ch in overlapping]
            all_ends = [ch.get('end_segment_idx', 0) for ch in overlapping]
            best['start_segment_idx'] = min(all_starts)
            best['end_segment_idx'] = max(all_ends)

        # Remove internal metadata
        if '_source_chunk' in best:
            del best['_source_chunk']

        merged.append(best)
        i = j

    # Renumber chapter IDs
    for idx, ch in enumerate(merged):
        ch['chapter_id'] = idx

    logger.info(f"Merged {sum(len(r) for r in deduped_results)} chapters into {len(merged)}")
    return merged


def estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough: ~4 chars per token)."""
    return len(text) // 4
