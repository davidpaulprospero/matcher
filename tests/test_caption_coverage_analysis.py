"""Tests for caption coverage analysis (US-73-010).

Verifies analyze_caption_coverage() and CaptionResult.coverage_analysis property
with full coverage, partial coverage, and large gaps scenarios.
"""

import logging
import pytest

from src.caption.models import (
    CaptionSegment,
    CaptionResult,
    CoverageAnalysis,
    analyze_caption_coverage,
)


def _make_segments(timings):
    """Create CaptionSegment list from (start, end) tuples."""
    return [
        CaptionSegment(index=i, start_time=s, end_time=e, text=f"text {i}")
        for i, (s, e) in enumerate(timings)
    ]


class TestAnalyzeCaptionCoverage:
    """Tests for the analyze_caption_coverage() function."""

    def test_full_coverage_no_gaps(self):
        """Continuous captions covering entire video have 0 gaps."""
        segments = _make_segments([
            (0.0, 10.0),
            (10.0, 20.0),
            (20.0, 30.0),
        ])
        result = analyze_caption_coverage(segments, video_duration=30.0)

        assert result is not None
        assert result.total_video_duration == 30.0
        assert result.total_captioned_duration == 30.0
        assert result.coverage_ratio == pytest.approx(1.0)
        assert result.gap_count == 0
        assert result.largest_gap_seconds == 0.0

    def test_partial_coverage_with_gaps(self):
        """Captions with gaps >2s are detected and counted."""
        # 0-5s captioned, 5-10s gap (5s), 10-15s captioned, 15-20s gap (5s)
        segments = _make_segments([
            (0.0, 5.0),
            (10.0, 15.0),
        ])
        result = analyze_caption_coverage(segments, video_duration=20.0)

        assert result is not None
        assert result.total_video_duration == 20.0
        assert result.total_captioned_duration == 10.0
        assert result.coverage_ratio == pytest.approx(0.5)
        assert result.gap_count == 2  # gap between segments + trailing gap
        assert result.largest_gap_seconds == pytest.approx(5.0)

    def test_large_gap_at_beginning(self):
        """Gap before first caption segment >2s is detected."""
        segments = _make_segments([
            (30.0, 40.0),
            (40.0, 50.0),
        ])
        result = analyze_caption_coverage(segments, video_duration=50.0)

        assert result is not None
        assert result.gap_count >= 1  # At least the leading gap
        assert result.largest_gap_seconds == pytest.approx(30.0)

    def test_large_trailing_gap(self):
        """Gap after last caption segment >2s is detected."""
        segments = _make_segments([
            (0.0, 10.0),
        ])
        result = analyze_caption_coverage(segments, video_duration=60.0)

        assert result is not None
        assert result.gap_count >= 1  # trailing gap
        assert result.largest_gap_seconds == pytest.approx(50.0)

    def test_small_gaps_not_counted(self):
        """Gaps <=2s between segments are not counted."""
        segments = _make_segments([
            (0.0, 10.0),
            (11.5, 20.0),  # 1.5s gap - below threshold
            (21.0, 30.0),  # 1.0s gap - below threshold
        ])
        result = analyze_caption_coverage(segments, video_duration=30.0)

        assert result is not None
        assert result.gap_count == 0

    def test_no_segments_returns_full_gap(self):
        """Empty segments list means entire video is one gap."""
        result = analyze_caption_coverage([], video_duration=100.0)

        assert result is not None
        assert result.total_captioned_duration == 0.0
        assert result.coverage_ratio == 0.0
        assert result.gap_count == 1
        assert result.largest_gap_seconds == 100.0

    def test_no_video_duration_returns_none(self):
        """Returns None when video_duration is not available."""
        segments = _make_segments([(0.0, 10.0)])
        assert analyze_caption_coverage(segments, video_duration=None) is None
        assert analyze_caption_coverage(segments, video_duration=0.0) is None

    def test_coverage_ratio_capped_at_one(self):
        """Coverage ratio doesn't exceed 1.0 even with overlapping segments."""
        # Overlapping segments sum to more than video duration
        segments = _make_segments([
            (0.0, 10.0),
            (5.0, 15.0),  # overlaps by 5s
        ])
        result = analyze_caption_coverage(segments, video_duration=10.0)

        assert result is not None
        assert result.coverage_ratio == 1.0  # capped

    def test_to_dict(self):
        """CoverageAnalysis.to_dict() returns all fields."""
        analysis = CoverageAnalysis(
            total_video_duration=100.0,
            total_captioned_duration=70.0,
            coverage_ratio=0.7,
            gap_count=3,
            largest_gap_seconds=15.0,
        )
        d = analysis.to_dict()
        assert d == {
            'total_video_duration': 100.0,
            'total_captioned_duration': 70.0,
            'coverage_ratio': 0.7,
            'gap_count': 3,
            'largest_gap_seconds': 15.0,
        }


class TestCaptionResultCoverageAnalysis:
    """Tests for CaptionResult.coverage_analysis property."""

    def test_coverage_analysis_property_returns_analysis(self):
        """coverage_analysis property computes and returns CoverageAnalysis."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments([(0.0, 50.0)]),
            video_duration=100.0,
        )
        analysis = result.coverage_analysis

        assert analysis is not None
        assert isinstance(analysis, CoverageAnalysis)
        assert analysis.total_video_duration == 100.0

    def test_coverage_analysis_cached(self):
        """coverage_analysis is computed once and cached."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments([(0.0, 50.0)]),
            video_duration=100.0,
        )
        first = result.coverage_analysis
        second = result.coverage_analysis
        assert first is second

    def test_coverage_analysis_none_without_duration(self):
        """coverage_analysis returns None when video_duration not set."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments([(0.0, 50.0)]),
        )
        assert result.coverage_analysis is None

    def test_low_coverage_emits_warning(self, caplog):
        """WARNING logged when coverage_ratio < 0.7."""
        with caplog.at_level(logging.WARNING, logger="src.caption.models"):
            result = CaptionResult(
                video_id="low_cov_vid",
                segments=_make_segments([(0.0, 10.0)]),
                video_duration=100.0,
            )
            _ = result.coverage_analysis

        assert any(
            "Low caption coverage" in r.message and "low_cov_vid" in r.message
            for r in caplog.records
        )

    def test_good_coverage_no_warning(self, caplog):
        """No WARNING when coverage_ratio >= 0.7."""
        with caplog.at_level(logging.WARNING, logger="src.caption.models"):
            result = CaptionResult(
                video_id="good_cov_vid",
                segments=_make_segments([(0.0, 80.0)]),
                video_duration=100.0,
            )
            _ = result.coverage_analysis

        assert not any(
            "Low caption coverage" in r.message
            for r in caplog.records
        )
