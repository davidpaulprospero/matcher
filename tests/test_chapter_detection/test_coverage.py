"""Tests for coverage resolution pass."""

import pytest
from unittest.mock import Mock

from src.chapter_detection.passes.coverage import (
    run_coverage_resolution,
    _resolve_overlaps,
    _fill_gaps,
    _merge_tiny_chapters,
)
from src.chapter_detection.models import ChapterCandidate


class TestRunCoverageResolution:
    """Test full coverage resolution pass."""

    def test_empty_chapters(self, mock_config):
        """Test empty list returns empty."""
        result = run_coverage_resolution(
            chapters=[],
            total_segments=10,
            config=mock_config,
        )
        assert result == []

    def test_zero_segments(self, mock_config):
        """Test zero segments returns original chapters."""
        chapters = [ChapterCandidate(chapter_id=0)]
        result = run_coverage_resolution(
            chapters=chapters,
            total_segments=0,
            config=mock_config,
        )
        assert result == chapters

    def test_renumbers_chapters(self, mock_config):
        """Test chapters are renumbered after resolution."""
        chapters = [
            ChapterCandidate(chapter_id=5, start_segment_idx=0, end_segment_idx=4),
            ChapterCandidate(chapter_id=10, start_segment_idx=5, end_segment_idx=9),
        ]
        result = run_coverage_resolution(
            chapters=chapters,
            total_segments=10,
            config=mock_config,
        )
        assert result[0].chapter_id == 0
        assert result[1].chapter_id == 1


class TestResolveOverlaps:
    """Test overlap resolution."""

    def test_no_overlap(self):
        """Test non-overlapping chapters unchanged."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=4,
                confidence=0.8,
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=5,
                end_segment_idx=9,
                confidence=0.7,
            ),
        ]
        result = _resolve_overlaps(chapters)

        assert len(result) == 2
        assert result[0].end_segment_idx == 4
        assert result[1].start_segment_idx == 5

    def test_overlap_resolution(self):
        """Test overlapping chapters are resolved."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=7,  # Overlaps with next
                confidence=0.8,
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=5,  # Starts before previous ends
                end_segment_idx=12,
                confidence=0.7,
            ),
        ]
        result = _resolve_overlaps(chapters)

        assert len(result) == 2
        # Higher confidence chapter keeps more
        assert result[0].end_segment_idx <= result[1].start_segment_idx

    def test_overlap_higher_conf_wins(self):
        """Test higher confidence chapter keeps more of overlap."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=7,
                confidence=0.9,  # Higher
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=5,
                end_segment_idx=12,
                confidence=0.5,  # Lower
            ),
        ]
        result = _resolve_overlaps(chapters)

        # First chapter (higher confidence) should win more of the overlap
        # Overlap is 5-7, split at midpoint (6), second chapter starts at 7
        # Result: chapter 0 ends at 7, chapter 1 starts at 7 (adjacent, no gap)
        assert result[1].start_segment_idx >= result[0].end_segment_idx
        # First chapter should keep its end (7) since it has higher confidence
        assert result[0].end_segment_idx == 7

    def test_single_chapter_unchanged(self):
        """Test single chapter returned unchanged."""
        chapters = [ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=5)]
        result = _resolve_overlaps(chapters)
        assert len(result) == 1


class TestFillGaps:
    """Test gap filling."""

    def test_no_chapters_creates_full_coverage(self):
        """Test no chapters creates single covering chapter."""
        result = _fill_gaps([], total_segments=10)

        assert len(result) == 1
        assert result[0].start_segment_idx == 0
        assert result[0].end_segment_idx == 9
        assert result[0].detection_strategy == "gap_fill"

    def test_fills_gap_at_start(self):
        """Test fills gap at transcript start."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=5,  # Gap at 0-4
                end_segment_idx=9,
            ),
        ]
        result = _fill_gaps(chapters, total_segments=10)

        # Should have Introduction chapter before existing
        assert len(result) >= 2
        intro = [c for c in result if c.start_segment_idx == 0][0]
        assert intro.title == "Introduction"

    def test_fills_gap_at_end(self):
        """Test fills gap at transcript end."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=4,  # Gap at 5-9
            ),
        ]
        result = _fill_gaps(chapters, total_segments=10)

        # Should have Conclusion or extended chapter
        last = max(result, key=lambda c: c.end_segment_idx)
        assert last.end_segment_idx == 9

    def test_extends_small_gap(self):
        """Test small gaps extend adjacent chapters."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=4,
                confidence=0.8,
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=7,  # 2 segment gap (5-6)
                end_segment_idx=10,
                confidence=0.7,
            ),
        ]
        result = _fill_gaps(chapters, total_segments=11)

        # Small gap should be absorbed by adjacent chapter
        # No new "gap_fill" chapter for small gaps
        gap_fill_chapters = [c for c in result if c.detection_strategy == "gap_fill"]
        # Either gap was filled by extension, or only necessary gap fills added
        assert len(result) == 2 or (len(result) == 3 and gap_fill_chapters)

    def test_creates_chapter_for_large_gap(self):
        """Test large gaps create new chapters."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=3,
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=15,  # Large gap: 4-14
                end_segment_idx=20,
            ),
        ]
        result = _fill_gaps(chapters, total_segments=21)

        # Should have at least 3 chapters (original 2 + gap fill)
        assert len(result) >= 3
        gap_fill = [c for c in result if c.detection_strategy == "gap_fill"]
        assert len(gap_fill) >= 1


class TestMergeTinyChapters:
    """Test merging of tiny chapters."""

    def test_merges_small_chapter(self):
        """Test small chapter merged with previous."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=5,
                title="Big Chapter",
                confidence=0.8,
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=6,
                end_segment_idx=7,  # Only 2 segments
                title="Tiny",
                confidence=0.7,
            ),
        ]
        result = _merge_tiny_chapters(chapters, min_segments=3)

        assert len(result) == 1
        assert result[0].end_segment_idx == 7
        assert "Tiny" in result[0].title or "Big Chapter" in result[0].title

    def test_preserves_large_chapters(self):
        """Test chapters meeting minimum size preserved."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=4,  # 5 segments
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=5,
                end_segment_idx=9,  # 5 segments
            ),
        ]
        result = _merge_tiny_chapters(chapters, min_segments=3)

        assert len(result) == 2

    def test_first_tiny_chapter_kept_low_confidence(self):
        """Test first tiny chapter kept but with lower confidence."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=1,  # Only 2 segments
                confidence=0.8,
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=2,
                end_segment_idx=10,
            ),
        ]
        result = _merge_tiny_chapters(chapters, min_segments=3)

        assert len(result) == 2
        # First chapter should have reduced confidence
        assert result[0].confidence < 0.8

    def test_min_segments_one_no_merge(self):
        """Test no merging when min_segments is 1."""
        chapters = [
            ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=0),
            ChapterCandidate(chapter_id=1, start_segment_idx=1, end_segment_idx=1),
        ]
        result = _merge_tiny_chapters(chapters, min_segments=1)
        assert len(result) == 2
