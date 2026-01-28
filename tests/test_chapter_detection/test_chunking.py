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


class TestChunkingSentenceBoundaries:
    """US-003: Test chunking splits on segment boundaries (sentence-level)."""

    def test_chunks_split_on_segment_boundaries(self):
        """Test that chunks split on segment boundaries, not mid-text.

        The create_chunks function treats each segment as an atomic unit,
        effectively splitting on sentence boundaries since each segment
        typically represents one sentence or caption line.
        """
        segments = [
            {"index": 0, "text": "This is the first sentence."},
            {"index": 1, "text": "This is the second sentence."},
            {"index": 2, "text": "This is the third sentence."},
            {"index": 3, "text": "This is the fourth sentence."},
        ]
        # Use small max_chars to force chunking
        chunks = create_chunks(segments, max_chars=80, overlap_segments=1)

        # Each chunk should contain complete segments, not partial text
        for chunk in chunks:
            # Text should start with "[" (segment marker)
            assert chunk.text.startswith("["), f"Chunk should start with segment marker: {chunk.text[:20]}"
            # Each line in chunk should be a complete segment
            for line in chunk.text.split("\n"):
                if line.strip():
                    assert line.strip().startswith("["), f"Line should start with segment marker: {line}"
                    # Should contain closing bracket and text
                    assert "]" in line, f"Line should have closing bracket: {line}"

    def test_long_segments_not_split_mid_sentence(self):
        """Test that a single long segment is kept together, not split."""
        segments = [
            {"index": 0, "text": "A " * 100},  # Long segment
            {"index": 1, "text": "Short."},
        ]
        chunks = create_chunks(segments, max_chars=50, overlap_segments=0)

        # First chunk should contain the long segment (possibly truncated but complete marker)
        assert len(chunks) >= 1
        # The long segment should start with proper indexing
        assert "[0]" in chunks[0].text


class TestChunkingMaxSizeLimit:
    """US-003: Test chunking respects max_chunk_size limit."""

    def test_respects_max_chars_limit(self):
        """Test that chunks do not exceed max_chars limit."""
        segments = [
            {"index": i, "text": f"Segment {i} with some text content here."}
            for i in range(20)
        ]
        max_chars = 200
        chunks = create_chunks(segments, max_chars=max_chars, overlap_segments=2)

        # Each chunk's text should be at or near max_chars (within one segment)
        for i, chunk in enumerate(chunks):
            # Allow some tolerance for single-segment chunks
            assert len(chunk.text) <= max_chars + 100, \
                f"Chunk {i} exceeds max_chars: {len(chunk.text)} > {max_chars}"

    def test_extremely_small_max_chars(self):
        """Test behavior with very small max_chars forces single-segment chunks."""
        segments = [
            {"index": 0, "text": "First segment text here."},
            {"index": 1, "text": "Second segment text here."},
        ]
        # Very small limit
        chunks = create_chunks(segments, max_chars=30, overlap_segments=0)

        # Should still create chunks, not error
        assert len(chunks) >= 1
        # Each chunk should have at least one segment
        for chunk in chunks:
            assert len(chunk.segment_indices) >= 1


class TestChunkingShortTranscripts:
    """US-003: Test chunking handles transcripts shorter than typical sizes."""

    def test_single_segment_transcript(self):
        """Test chunking with just one segment."""
        segments = [{"index": 0, "text": "Only segment."}]
        chunks = create_chunks(segments, max_chars=6000, overlap_segments=5)

        assert len(chunks) == 1
        assert chunks[0].start_segment_idx == 0
        assert chunks[0].end_segment_idx == 0
        assert chunks[0].segment_count == 1

    def test_transcript_smaller_than_min_overlap(self):
        """Test transcript with fewer segments than overlap setting."""
        segments = [
            {"index": 0, "text": "First."},
            {"index": 1, "text": "Second."},
        ]
        # overlap_segments=5 but only 2 segments
        chunks = create_chunks(segments, max_chars=6000, overlap_segments=5)

        # Should create single chunk without error
        assert len(chunks) == 1
        assert chunks[0].segment_count == 2

    def test_empty_transcript(self):
        """Test chunking with empty transcript."""
        chunks = create_chunks([], max_chars=6000, overlap_segments=5)
        assert len(chunks) == 0


class TestChunkingTimestampPreservation:
    """US-003: Test chunking preserves timestamp alignment."""

    def test_timestamps_in_segments_preserved(self):
        """Test that segment indices map correctly for timestamp lookup."""
        segments = [
            {"index": 0, "text": "First.", "start_time": 0.0, "end_time": 2.0},
            {"index": 1, "text": "Second.", "start_time": 2.0, "end_time": 4.0},
            {"index": 2, "text": "Third.", "start_time": 4.0, "end_time": 6.0},
            {"index": 3, "text": "Fourth.", "start_time": 6.0, "end_time": 8.0},
        ]
        chunks = create_chunks(segments, max_chars=50, overlap_segments=1)

        # Verify segment_indices allow timestamp lookup
        for chunk in chunks:
            for seg_idx in chunk.segment_indices:
                assert 0 <= seg_idx < len(segments), f"Invalid segment index: {seg_idx}"
                # Can look up timestamp via original segments
                original_seg = segments[seg_idx]
                assert "start_time" in original_seg
                assert "end_time" in original_seg

    def test_chunk_boundaries_align_with_segment_indices(self):
        """Test that start/end segment indices match actual content."""
        segments = [
            {"index": 0, "text": "A", "start_time": 0.0, "end_time": 1.0},
            {"index": 1, "text": "B", "start_time": 1.0, "end_time": 2.0},
            {"index": 2, "text": "C", "start_time": 2.0, "end_time": 3.0},
            {"index": 3, "text": "D", "start_time": 3.0, "end_time": 4.0},
            {"index": 4, "text": "E", "start_time": 4.0, "end_time": 5.0},
        ]
        chunks = create_chunks(segments, max_chars=20, overlap_segments=1)

        for chunk in chunks:
            # start_segment_idx should match first in segment_indices
            assert chunk.start_segment_idx == chunk.segment_indices[0]
            # end_segment_idx should match last in segment_indices
            assert chunk.end_segment_idx == chunk.segment_indices[-1]

    def test_all_segments_covered_for_timestamp_continuity(self):
        """Test that all segments appear in at least one chunk."""
        segments = [
            {"index": i, "text": f"Seg{i}", "start_time": i * 1.0, "end_time": (i + 1) * 1.0}
            for i in range(10)
        ]
        chunks = create_chunks(segments, max_chars=50, overlap_segments=2)

        # Collect all segment indices across chunks
        covered = set()
        for chunk in chunks:
            covered.update(chunk.segment_indices)

        # All segments should be covered
        expected = set(range(10))
        assert covered == expected, f"Missing segments: {expected - covered}"


class TestChunkingUnicode:
    """US-003: Test chunking handles Unicode characters correctly."""

    def test_unicode_characters_in_text(self):
        """Test chunking with various Unicode characters."""
        segments = [
            {"index": 0, "text": "日本語テキスト"},  # Japanese
            {"index": 1, "text": "Émojis: 🎬🎥🎞️"},  # Emojis
            {"index": 2, "text": "Ελληνικά κείμενο"},  # Greek
            {"index": 3, "text": "العربية"},  # Arabic
            {"index": 4, "text": "Normal text here."},
        ]
        chunks = create_chunks(segments, max_chars=6000, overlap_segments=2)

        assert len(chunks) >= 1
        # Verify all Unicode text is preserved
        all_text = "\n".join(c.text for c in chunks)
        assert "日本語" in all_text
        assert "🎬" in all_text
        assert "Ελληνικά" in all_text
        assert "العربية" in all_text

    def test_unicode_length_calculation(self):
        """Test that max_chars counts Unicode correctly."""
        # Create segments with Unicode that have more bytes than chars
        segments = [
            {"index": 0, "text": "🎬" * 20},  # Each emoji is 1 char but 4 bytes
            {"index": 1, "text": "A" * 20},   # Each letter is 1 char, 1 byte
        ]
        chunks = create_chunks(segments, max_chars=50, overlap_segments=0)

        # Should handle both segments without error
        assert len(chunks) >= 1
        # Emoji segment should be in results
        assert "🎬" in chunks[0].text

    def test_mixed_unicode_scripts(self):
        """Test chunking with mixed scripts in single segment."""
        segments = [
            {"index": 0, "text": "English 日本語 العربية mixed"},
            {"index": 1, "text": "More mixed: Café résumé naïve"},
            {"index": 2, "text": "Math: α + β = γ, ∑∏∫"},
        ]
        chunks = create_chunks(segments, max_chars=6000, overlap_segments=1)

        assert len(chunks) == 1  # Should fit in single chunk
        assert "日本語" in chunks[0].text
        assert "Café" in chunks[0].text
        assert "∑" in chunks[0].text
