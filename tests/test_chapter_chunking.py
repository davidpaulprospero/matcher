"""Tests for chapter detection chunking, specifically overlap deduplication."""

import pytest
from src.chapter_detection.chunking import (
    TextChunk,
    merge_chunk_results,
    _deduplicate_overlap_zone,
    _segment_overlap_ratio,
)


def _make_chapter(start: int, end: int, confidence: str = 'medium', title: str = '') -> dict:
    """Helper to create a chapter dict."""
    return {
        'start_segment_idx': start,
        'end_segment_idx': end,
        'confidence': confidence,
        'title': title or f'Chapter {start}-{end}',
        'chapter_id': 0,
    }


def _make_chunks_pair(
    chunk_a_start: int, chunk_a_end: int,
    chunk_b_start: int, chunk_b_end: int,
    overlap_segments: int,
) -> list:
    """Helper to create a pair of TextChunks with specified overlap."""
    chunk_a = TextChunk(
        text='',
        start_segment_idx=chunk_a_start,
        end_segment_idx=chunk_a_end,
        segment_indices=list(range(chunk_a_start, chunk_a_end + 1)),
        overlap_start=0,
        overlap_end=overlap_segments,
    )
    chunk_b = TextChunk(
        text='',
        start_segment_idx=chunk_b_start,
        end_segment_idx=chunk_b_end,
        segment_indices=list(range(chunk_b_start, chunk_b_end + 1)),
        overlap_start=overlap_segments,
        overlap_end=0,
    )
    return [chunk_a, chunk_b]


class TestOverlapDeduplication:
    """Tests for overlap-zone deduplication in merge_chunk_results."""

    def test_duplicate_chapters_in_overlap_zone_produce_single_chapter(self):
        """Two chunks with 3-segment overlap produce single chapter for
        chapters in the overlap zone (AC #4)."""
        # Chunk A: segments 0-9, Chunk B: segments 7-19, overlap=3 (segs 7,8,9)
        chunks = _make_chunks_pair(0, 9, 7, 19, overlap_segments=3)

        # Both chunks detect a chapter spanning segs 7-9 (the overlap zone)
        chunk_a_chapters = [
            _make_chapter(0, 4, 'high', 'Intro'),
            _make_chapter(7, 9, 'medium', 'Overlap Chapter A'),
        ]
        chunk_b_chapters = [
            _make_chapter(7, 9, 'high', 'Overlap Chapter B'),
            _make_chapter(12, 17, 'medium', 'Middle'),
        ]

        result = merge_chunk_results(
            [chunk_a_chapters, chunk_b_chapters],
            chunks,
            total_segments=20,
        )

        # Should have 3 chapters: Intro, one overlap chapter, Middle
        titles = [ch['title'] for ch in result]
        assert len(result) == 3, f"Expected 3 chapters, got {len(result)}: {titles}"

        # The overlap chapter should be the high-confidence one from chunk B
        overlap_chapters = [ch for ch in result if 7 <= ch['start_segment_idx'] <= 9]
        assert len(overlap_chapters) == 1
        assert overlap_chapters[0]['confidence'] == 'high'

    def test_chapters_outside_overlap_zone_preserved(self):
        """Chapters outside the overlap zone are preserved from both chunks (AC #5)."""
        chunks = _make_chunks_pair(0, 9, 7, 19, overlap_segments=3)

        chunk_a_chapters = [
            _make_chapter(0, 3, 'medium', 'Before Overlap'),
            _make_chapter(5, 6, 'low', 'Just Before'),
        ]
        chunk_b_chapters = [
            _make_chapter(12, 15, 'high', 'After Overlap'),
            _make_chapter(17, 19, 'medium', 'End'),
        ]

        result = merge_chunk_results(
            [chunk_a_chapters, chunk_b_chapters],
            chunks,
            total_segments=20,
        )

        # All 4 chapters should be preserved (none in overlap zone)
        assert len(result) == 4
        result_titles = {ch['title'] for ch in result}
        assert 'Before Overlap' in result_titles
        assert 'Just Before' in result_titles
        assert 'After Overlap' in result_titles
        assert 'End' in result_titles

    def test_higher_confidence_wins_deduplication(self):
        """When deduplicating, the higher-confidence chapter wins (AC #6)."""
        chunks = _make_chunks_pair(0, 9, 7, 19, overlap_segments=3)

        # Chunk A has HIGH confidence chapter in overlap zone
        chunk_a_chapters = [
            _make_chapter(7, 9, 'high', 'High Conf'),
        ]
        # Chunk B has LOW confidence chapter in overlap zone
        chunk_b_chapters = [
            _make_chapter(7, 9, 'low', 'Low Conf'),
        ]

        result = merge_chunk_results(
            [chunk_a_chapters, chunk_b_chapters],
            chunks,
            total_segments=20,
        )

        assert len(result) == 1
        assert result[0]['confidence'] == 'high'
        assert result[0]['title'] == 'High Conf'

    def test_50_percent_overlap_threshold(self):
        """Chapters must overlap by MORE than 50% to be deduplicated (AC #1)."""
        chunks = _make_chunks_pair(0, 9, 7, 19, overlap_segments=3)
        # Overlap zone: segments 7, 8, 9

        # Chapter A: segs 6-9 (4 segments), Chapter B: segs 7-10 (4 segments)
        # Intersection in overlap zone: {7,8,9} from A, {7,8,9} from B = {7,8,9}
        # But full chapter ranges: A={6,7,8,9}, B={7,8,9,10}
        # Overlap ratio = |{7,8,9}| / min(4,4) = 3/4 = 75% > 50% → deduplicate
        chunk_a_chapters = [_make_chapter(6, 9, 'medium', 'ChA')]
        chunk_b_chapters = [_make_chapter(7, 10, 'high', 'ChB')]

        result = merge_chunk_results(
            [chunk_a_chapters, chunk_b_chapters],
            chunks,
            total_segments=20,
        )

        # Should deduplicate to 1 chapter (75% overlap > 50%)
        assert len(result) == 1
        assert result[0]['confidence'] == 'high'

    def test_low_overlap_not_deduplicated(self):
        """Chapters with less than 50% segment overlap are NOT deduplicated."""
        chunks = _make_chunks_pair(0, 9, 7, 19, overlap_segments=3)
        # Overlap zone: segments 7, 8, 9

        # Chapter A: segs 3-8 (6 segments), Chapter B: segs 8-14 (7 segments)
        # Only segment 8 overlaps in both. Ratio = 1/min(6,7) = 1/6 ≈ 17% < 50%
        chunk_a_chapters = [_make_chapter(3, 8, 'low', 'ChA')]
        chunk_b_chapters = [_make_chapter(8, 14, 'high', 'ChB')]

        result = merge_chunk_results(
            [chunk_a_chapters, chunk_b_chapters],
            chunks,
            total_segments=20,
        )

        # These should NOT be deduplicated (17% overlap < 50%)
        # But the generic merge phase may still combine if ranges overlap
        # The key is the dedup phase doesn't remove either
        deduped = _deduplicate_overlap_zone(
            [chunk_a_chapters, chunk_b_chapters], chunks
        )
        # Both chapters should survive the dedup phase
        assert len(deduped[0]) == 1
        assert len(deduped[1]) == 1

    def test_deduplication_only_in_overlap_region(self):
        """Deduplication only applies to chapters in the overlap region (AC #3)."""
        chunks = _make_chunks_pair(0, 9, 7, 19, overlap_segments=3)
        # Overlap zone: segments 7, 8, 9

        # Two identical chapters at segments 0-2 — NOT in overlap zone
        chunk_a_chapters = [_make_chapter(0, 2, 'low', 'Start')]
        # Chunk B also claims segs 0-2 (shouldn't happen normally, but test the guard)
        chunk_b_chapters = [_make_chapter(0, 2, 'high', 'Start Dup')]

        deduped = _deduplicate_overlap_zone(
            [chunk_a_chapters, chunk_b_chapters], chunks
        )

        # Both should survive dedup since they're NOT in the overlap zone
        assert len(deduped[0]) == 1
        assert len(deduped[1]) == 1

    def test_no_overlap_no_deduplication(self):
        """When overlap_segments is 0, no deduplication happens."""
        chunks = _make_chunks_pair(0, 9, 10, 19, overlap_segments=0)

        chunk_a_chapters = [_make_chapter(8, 9, 'low')]
        chunk_b_chapters = [_make_chapter(10, 11, 'high')]

        deduped = _deduplicate_overlap_zone(
            [chunk_a_chapters, chunk_b_chapters], chunks
        )

        assert len(deduped[0]) == 1
        assert len(deduped[1]) == 1

    def test_three_chunks_deduplication(self):
        """Deduplication works across three chunks (two overlap boundaries)."""
        chunk_a = TextChunk(
            text='', start_segment_idx=0, end_segment_idx=9,
            segment_indices=list(range(10)), overlap_start=0, overlap_end=3,
        )
        chunk_b = TextChunk(
            text='', start_segment_idx=7, end_segment_idx=19,
            segment_indices=list(range(7, 20)), overlap_start=3, overlap_end=3,
        )
        chunk_c = TextChunk(
            text='', start_segment_idx=17, end_segment_idx=29,
            segment_indices=list(range(17, 30)), overlap_start=3, overlap_end=0,
        )
        chunks = [chunk_a, chunk_b, chunk_c]

        results_a = [_make_chapter(7, 9, 'medium', 'Overlap AB')]
        results_b = [
            _make_chapter(7, 9, 'high', 'Overlap AB dup'),
            _make_chapter(17, 19, 'low', 'Overlap BC'),
        ]
        results_c = [_make_chapter(17, 19, 'high', 'Overlap BC dup')]

        result = merge_chunk_results(
            [results_a, results_b, results_c],
            chunks,
            total_segments=30,
        )

        # Should have 2 chapters: one from AB boundary, one from BC boundary
        assert len(result) == 2
        # AB boundary: high wins (from chunk B)
        assert result[0]['confidence'] == 'high'
        # BC boundary: high wins (from chunk C)
        assert result[1]['confidence'] == 'high'


class TestSegmentOverlapRatio:
    """Tests for _segment_overlap_ratio helper."""

    def test_identical_chapters(self):
        a = _make_chapter(5, 10)
        b = _make_chapter(5, 10)
        assert _segment_overlap_ratio(a, b) == 1.0

    def test_no_overlap(self):
        a = _make_chapter(0, 4)
        b = _make_chapter(5, 9)
        assert _segment_overlap_ratio(a, b) == 0.0

    def test_partial_overlap(self):
        a = _make_chapter(0, 5)  # 6 segments
        b = _make_chapter(3, 8)  # 6 segments
        # Intersection: {3,4,5} = 3 segments
        # Ratio: 3 / min(6,6) = 0.5
        assert _segment_overlap_ratio(a, b) == 0.5

    def test_subset(self):
        a = _make_chapter(2, 8)  # 7 segments
        b = _make_chapter(3, 5)  # 3 segments
        # Intersection: {3,4,5} = 3 segments
        # Ratio: 3 / min(7,3) = 1.0
        assert _segment_overlap_ratio(a, b) == 1.0
