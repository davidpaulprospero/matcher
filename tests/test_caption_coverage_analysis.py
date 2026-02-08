"""Tests for caption coverage analysis (US-73-010)."""

import logging
import pytest

from src.caption.models import (
    CaptionSegment,
    CaptionResult,
    CoverageAnalysis,
    analyze_caption_coverage,
)


def _seg(index: int, start: float, end: float, text: str = "hello") -> CaptionSegment:
    """Helper to build a CaptionSegment quickly."""
    return CaptionSegment(index=index, start_time=start, end_time=end, text=text, source_file="vid1")


class TestAnalyzeCaptionCoverage:
    """Tests for the analyze_caption_coverage() function."""

    def test_returns_none_without_video_duration(self):
        segs = [_seg(0, 0, 5)]
        assert analyze_caption_coverage(segs, None) is None
        assert analyze_caption_coverage(segs, 0.0) is None
        assert analyze_caption_coverage(segs, -1.0) is None

    def test_full_coverage_no_gaps(self):
        """Segments span entire video with no gaps > 2s."""
        segs = [
            _seg(0, 0.0, 10.0),
            _seg(1, 10.0, 20.0),
            _seg(2, 20.0, 30.0),
        ]
        result = analyze_caption_coverage(segs, 30.0)
        assert result is not None
        assert result.total_video_duration == 30.0
        assert result.total_captioned_duration == 30.0
        assert result.coverage_ratio == pytest.approx(1.0)
        assert result.gap_count == 0
        assert result.largest_gap_seconds == 0.0

    def test_partial_coverage_with_gaps(self):
        """Segments with a large gap in the middle."""
        segs = [
            _seg(0, 0.0, 10.0),
            # 20-second gap here (10.0 to 30.0)
            _seg(1, 30.0, 40.0),
        ]
        result = analyze_caption_coverage(segs, 40.0)
        assert result is not None
        assert result.total_video_duration == 40.0
        assert result.total_captioned_duration == pytest.approx(20.0)
        assert result.coverage_ratio == pytest.approx(0.5)
        assert result.gap_count == 1  # The 20s gap in the middle
        assert result.largest_gap_seconds == pytest.approx(20.0)

    def test_large_gap_at_start(self):
        """Gap before first segment > 2s counts."""
        segs = [_seg(0, 10.0, 20.0)]
        result = analyze_caption_coverage(segs, 20.0)
        assert result is not None
        assert result.gap_count == 1  # 10s gap at start
        assert result.largest_gap_seconds == pytest.approx(10.0)
        assert result.coverage_ratio == pytest.approx(0.5)

    def test_large_gap_at_end(self):
        """Gap after last segment > 2s counts."""
        segs = [_seg(0, 0.0, 10.0)]
        result = analyze_caption_coverage(segs, 30.0)
        assert result is not None
        assert result.gap_count == 1  # 20s gap at end
        assert result.largest_gap_seconds == pytest.approx(20.0)
        assert result.coverage_ratio == pytest.approx(10.0 / 30.0)

    def test_small_gaps_below_threshold_ignored(self):
        """Gaps <= 2 seconds are not counted."""
        segs = [
            _seg(0, 0.0, 10.0),
            _seg(1, 11.5, 20.0),  # 1.5s gap — below threshold
        ]
        result = analyze_caption_coverage(segs, 20.0)
        assert result is not None
        assert result.gap_count == 0
        assert result.largest_gap_seconds == 0.0

    def test_empty_segments_whole_video_is_gap(self):
        """No segments means entire video is one gap."""
        result = analyze_caption_coverage([], 60.0)
        assert result is not None
        assert result.total_captioned_duration == 0.0
        assert result.coverage_ratio == 0.0
        assert result.gap_count == 1
        assert result.largest_gap_seconds == 60.0

    def test_multiple_gaps(self):
        """Multiple gaps detected correctly."""
        segs = [
            _seg(0, 5.0, 10.0),   # 5s gap at start
            # 10s gap (10.0 to 20.0)
            _seg(1, 20.0, 25.0),
            # 15s gap (25.0 to 40.0)
            _seg(2, 40.0, 50.0),
        ]
        result = analyze_caption_coverage(segs, 60.0)
        assert result is not None
        assert result.gap_count == 4  # start(5s) + mid(10s) + mid(15s) + end(10s)
        assert result.largest_gap_seconds == pytest.approx(15.0)

    def test_to_dict(self):
        """CoverageAnalysis serializes correctly."""
        analysis = CoverageAnalysis(
            total_video_duration=100.0,
            total_captioned_duration=70.0,
            coverage_ratio=0.7,
            gap_count=2,
            largest_gap_seconds=15.0,
        )
        d = analysis.to_dict()
        assert d['total_video_duration'] == 100.0
        assert d['coverage_ratio'] == 0.7
        assert d['gap_count'] == 2


class TestCaptionResultCoverageAnalysis:
    """Tests for CaptionResult.coverage_analysis property."""

    def test_coverage_analysis_property_returns_analysis(self):
        """coverage_analysis property returns CoverageAnalysis when video_duration set."""
        result = CaptionResult(
            video_id="abc123",
            segments=[_seg(0, 0.0, 50.0, "some text here for this segment")],
            video_duration=100.0,
        )
        analysis = result.coverage_analysis
        assert analysis is not None
        assert isinstance(analysis, CoverageAnalysis)
        assert analysis.coverage_ratio == pytest.approx(0.5)

    def test_coverage_analysis_cached(self):
        """coverage_analysis is computed once and cached."""
        result = CaptionResult(
            video_id="abc123",
            segments=[_seg(0, 0.0, 80.0, "some longer text for this segment")],
            video_duration=100.0,
        )
        first = result.coverage_analysis
        second = result.coverage_analysis
        assert first is second  # Same object (cached)

    def test_coverage_analysis_none_without_duration(self):
        """coverage_analysis returns None when video_duration not set."""
        result = CaptionResult(
            video_id="abc123",
            segments=[_seg(0, 0.0, 10.0)],
        )
        assert result.coverage_analysis is None

    def test_low_coverage_emits_warning(self, caplog):
        """WARNING logged when coverage_ratio < 0.7."""
        with caplog.at_level(logging.WARNING, logger="src.caption.models"):
            result = CaptionResult(
                video_id="low_cov_vid",
                segments=[_seg(0, 0.0, 20.0, "some text here for the segment")],
                video_duration=100.0,
            )
            _ = result.coverage_analysis
        assert any("Low caption coverage" in msg for msg in caplog.messages)
        assert any("low_cov_vid" in msg for msg in caplog.messages)

    def test_high_coverage_no_warning(self, caplog):
        """No WARNING when coverage_ratio >= 0.7."""
        with caplog.at_level(logging.WARNING, logger="src.caption.models"):
            result = CaptionResult(
                video_id="high_cov_vid",
                segments=[_seg(0, 0.0, 80.0, "some text here for this segment")],
                video_duration=100.0,
            )
            _ = result.coverage_analysis
        assert not any("Low caption coverage" in msg for msg in caplog.messages)
