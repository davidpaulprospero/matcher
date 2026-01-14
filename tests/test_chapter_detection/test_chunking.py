"""Tests for transcript chunking."""

import pytest
from src.chapter_detection.chunking import (
    create_chunks,
    create_indexed_text,
    merge_chunk_results,
)


class TestCreateIndexedText:
    """Test indexed text creation."""

    def test_basic(self):
        """Test basic indexed text creation."""
        segments = [
            {"index": 0, "text": "First segment"},
            {"index": 1, "text": "Second segment"},
        ]
        result = create_indexed_text(segments)
        assert "[0] First segment" in result
        assert "[1] Second segment" in result

    def test_empty(self):
        """Test with empty segments."""
        result = create_indexed_text([])
        assert result == ""

    def test_strips_whitespace(self):
        """Test that text is stripped."""
        segments = [{"index": 0, "text": "  Text with spaces  "}]
        result = create_indexed_text(segments)
        assert "[0] Text with spaces" in result


class TestCreateChunks:
    """Test chunk creation."""

    def test_single_chunk_small_input(self):
        """Test that small input creates single chunk."""
        segments = [
            {"index": 0, "text": "Short segment one"},
            {"index": 1, "text": "Short segment two"},
        ]
        chunks = create_chunks(segments, max_chars=6000)
        assert len(chunks) == 1
        assert chunks[0].start_segment_idx == 0
        assert chunks[0].end_segment_idx == 1

    def test_empty_input(self):
        """Test with empty input."""
        chunks = create_chunks([])
        assert len(chunks) == 0

    def test_multiple_chunks_large_input(self):
        """Test that large input creates multiple chunks."""
        # Create segments that exceed max_chars
        segments = [
            {"index": i, "text": f"This is segment number {i} with some text " * 10}
            for i in range(50)
        ]
        chunks = create_chunks(segments, max_chars=1000, overlap_segments=2)
        assert len(chunks) > 1

        # Check that chunks cover all segments
        all_indices = set()
        for chunk in chunks:
            all_indices.update(chunk.segment_indices)
        assert len(all_indices) == 50

    def test_overlap_between_chunks(self):
        """Test that chunks have proper overlap."""
        segments = [
            {"index": i, "text": f"Segment {i} " * 50}
            for i in range(20)
        ]
        chunks = create_chunks(segments, max_chars=500, overlap_segments=3)

        if len(chunks) > 1:
            # Check overlap exists
            for i in range(len(chunks) - 1):
                current_end = chunks[i].end_segment_idx
                next_start = chunks[i + 1].start_segment_idx
                # There should be some overlap
                assert next_start <= current_end + 1


class TestMergeChunkResults:
    """Test chunk result merging."""

    def test_single_chunk(self):
        """Test merging single chunk result."""
        results = [[
            {"chapter_id": 0, "start_segment_idx": 0, "end_segment_idx": 5, "title": "Ch1", "confidence": "high"},
        ]]
        chunks = [type('Chunk', (), {'start_segment_idx': 0, 'end_segment_idx': 5})()]

        merged = merge_chunk_results(results, chunks, 6)
        assert len(merged) == 1
        assert merged[0]['title'] == "Ch1"

    def test_non_overlapping_chunks(self):
        """Test merging non-overlapping chunk results."""
        results = [
            [{"chapter_id": 0, "start_segment_idx": 0, "end_segment_idx": 4, "title": "Ch1", "confidence": "high"}],
            [{"chapter_id": 0, "start_segment_idx": 5, "end_segment_idx": 9, "title": "Ch2", "confidence": "high"}],
        ]
        chunks = [
            type('Chunk', (), {'start_segment_idx': 0, 'end_segment_idx': 4})(),
            type('Chunk', (), {'start_segment_idx': 5, 'end_segment_idx': 9})(),
        ]

        merged = merge_chunk_results(results, chunks, 10)
        assert len(merged) == 2
        assert merged[0]['chapter_id'] == 0
        assert merged[1]['chapter_id'] == 1  # Renumbered

    def test_overlapping_chapters_prefer_high_confidence(self):
        """Test that overlapping chapters prefer high confidence."""
        results = [
            [{"chapter_id": 0, "start_segment_idx": 0, "end_segment_idx": 5, "title": "LowConf", "confidence": "low"}],
            [{"chapter_id": 0, "start_segment_idx": 3, "end_segment_idx": 8, "title": "HighConf", "confidence": "high"}],
        ]
        chunks = [
            type('Chunk', (), {'start_segment_idx': 0, 'end_segment_idx': 5})(),
            type('Chunk', (), {'start_segment_idx': 3, 'end_segment_idx': 8})(),
        ]

        merged = merge_chunk_results(results, chunks, 9)
        # Should merge overlapping chapters and prefer high confidence
        assert len(merged) == 1
        assert merged[0]['title'] == "HighConf"

    def test_empty_results(self):
        """Test merging empty results."""
        merged = merge_chunk_results([], [], 0)
        assert merged == []
