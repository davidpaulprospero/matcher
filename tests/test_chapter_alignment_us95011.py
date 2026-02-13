"""
Tests for compute_chapter_alignment_boost (US-95-011).

Verifies that:
- Video segments aligned with chapter timestamps receive a confidence boost
- Segment boundaries (start_time, end_time) matching chapter boundaries are preferred
- Config option 'prefer_chapter_aligned_segments' controls the feature
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import compute_chapter_alignment_boost, CHAPTER_ALIGNMENT_TOLERANCE
from src.utils import SRTSegment


def _make_segment(start_time: float, end_time: float, source_file: str = "video123") -> SRTSegment:
    """Create an SRTSegment with given timing."""
    seg = SRTSegment(
        index=1,
        start_time=start_time,
        end_time=end_time,
        text="Sample video segment",
        source_file=source_file,
    )
    return seg


def _make_mock_config(prefer_chapter_aligned: bool = True, chapter_alignment_boost: float = 0.05) -> MagicMock:
    """Create a mock config object with the chapter alignment settings."""
    config = MagicMock()
    config.prefer_chapter_aligned_segments = prefer_chapter_aligned
    config.chapter_alignment_boost = chapter_alignment_boost
    return config


class TestComputeChapterAlignmentBoost:
    """Tests for the compute_chapter_alignment_boost function."""

    def test_no_chapters_returns_zero(self):
        """No boost when video has no chapters."""
        segment = _make_segment(10.0, 15.0)
        config = _make_mock_config()
        boost, reason = compute_chapter_alignment_boost(segment, [], config)
        assert boost == 0.0
        assert reason == "no_chapters"

    def test_disabled_config_returns_zero(self):
        """No boost when feature is disabled in config."""
        segment = _make_segment(10.0, 15.0)
        chapters = [
            {'title': 'Intro', 'start_time': 0.0, 'end_time': 30.0},
            {'title': 'Main', 'start_time': 30.0, 'end_time': 120.0},
        ]
        config = _make_mock_config(prefer_chapter_aligned=False)
        boost, reason = compute_chapter_alignment_boost(segment, chapters, config)
        assert boost == 0.0
        assert reason == "chapter_alignment_disabled"

    def test_segment_start_aligned_with_chapter(self):
        """Boost when segment start aligns with chapter start."""
        # Chapter starts at 30.0, segment starts at 32.0 (within tolerance of 3s)
        segment = _make_segment(32.0, 45.0)
        chapters = [
            {'title': 'Introduction', 'start_time': 30.0, 'end_time': 60.0},
            {'title': 'Part 1', 'start_time': 60.0, 'end_time': 120.0},
        ]
        config = _make_mock_config()
        boost, reason = compute_chapter_alignment_boost(segment, chapters, config)
        assert boost == pytest.approx(0.05, abs=0.001)
        assert "boundary_aligned" in reason

    def test_segment_end_aligned_with_chapter(self):
        """Boost when segment end aligns with chapter start."""
        # Chapter starts at 60.0, segment ends at 58.0 (within tolerance of 3s)
        segment = _make_segment(45.0, 58.0)
        chapters = [
            {'title': 'Introduction', 'start_time': 30.0, 'end_time': 60.0},
            {'title': 'Part 1', 'start_time': 60.0, 'end_time': 120.0},
        ]
        config = _make_mock_config()
        boost, reason = compute_chapter_alignment_boost(segment, chapters, config)
        assert boost == pytest.approx(0.05, abs=0.001)
        assert "boundary_aligned" in reason

    def test_segment_within_chapter(self):
        """Boost when segment starts and ends at chapter boundaries."""
        # Segment exactly matches chapter boundaries
        segment = _make_segment(30.0, 60.0)
        chapters = [
            {'title': 'Introduction', 'start_time': 30.0, 'end_time': 60.0},
        ]
        config = _make_mock_config()
        boost, reason = compute_chapter_alignment_boost(segment, chapters, config)
        # Both start and end align with chapter boundaries
        assert boost == pytest.approx(0.05, abs=0.001)
        assert "segment_within_chapter" in reason

    def test_no_alignment_returns_zero(self):
        """No boost when segment doesn't align with any chapter."""
        # Segment at 15-25, chapters at 0-30 and 60-120 - no alignment
        segment = _make_segment(15.0, 25.0)
        chapters = [
            {'title': 'Intro', 'start_time': 0.0, 'end_time': 30.0},
            {'title': 'Main', 'start_time': 60.0, 'end_time': 120.0},
        ]
        config = _make_mock_config()
        boost, reason = compute_chapter_alignment_boost(segment, chapters, config)
        assert boost == 0.0
        assert reason == "not_aligned"

    def test_custom_boost_amount(self):
        """Custom boost amount from config is applied."""
        segment = _make_segment(32.0, 45.0)
        chapters = [
            {'title': 'Intro', 'start_time': 30.0, 'end_time': 60.0},
        ]
        config = _make_mock_config(chapter_alignment_boost=0.10)
        boost, reason = compute_chapter_alignment_boost(segment, chapters, config)
        assert boost == pytest.approx(0.10, abs=0.001)

    def test_chapter_end_alignment(self):
        """Boost when segment aligns with chapter end time."""
        # Chapter ends at 60.0, segment ends at 58.0 (within tolerance)
        segment = _make_segment(45.0, 58.0)
        chapters = [
            {'title': 'Intro', 'start_time': 0.0, 'end_time': 60.0},
        ]
        config = _make_mock_config()
        boost, reason = compute_chapter_alignment_boost(segment, chapters, config)
        assert boost == pytest.approx(0.05, abs=0.001)

    def test_no_segment_times_returns_zero(self):
        """No boost when segment has no timing information."""
        segment = MagicMock()
        segment.start_time = None
        segment.end_time = None
        chapters = [{'title': 'Intro', 'start_time': 0.0, 'end_time': 30.0}]
        config = _make_mock_config()
        boost, reason = compute_chapter_alignment_boost(segment, chapters, config)
        assert boost == 0.0
        assert reason == "no_segment_times"

    def test_tolerance_threshold(self):
        """Verify the tolerance threshold is correctly applied."""
        # Exactly at tolerance boundary
        segment = _make_segment(33.0, 45.0)  # 33.0 - 30.0 = 3.0 (exactly at tolerance)
        chapters = [{'title': 'Intro', 'start_time': 30.0, 'end_time': 60.0}]
        config = _make_mock_config()
        boost, reason = compute_chapter_alignment_boost(segment, chapters, config)
        assert boost == pytest.approx(0.05, abs=0.001)

        # Just outside tolerance
        segment = _make_segment(33.1, 45.0)  # 33.1 - 30.0 = 3.1 (outside tolerance)
        boost, reason = compute_chapter_alignment_boost(segment, chapters, config)
        assert boost == 0.0
        assert reason == "not_aligned"
