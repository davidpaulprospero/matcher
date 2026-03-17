"""
Tests for apply_duration_ratio_calibration (US-77-010).

Verifies confidence calibration based on the ratio of video segment
duration to voiceover segment duration.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_duration_ratio_calibration
from src.utils import SRTSegment


def _make_segment(start: float, end: float, text: str = "test") -> SRTSegment:
    """Create an SRTSegment with the given start/end times."""
    return SRTSegment(index=0, start_time=start, end_time=end, text=text)


class TestDurationRatioCalibration:
    """Tests for apply_duration_ratio_calibration."""

    def test_ratio_1_0_no_penalty(self):
        """Ratio 1.0 (equal durations) gives no penalty."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 10.0)  # 10s video
        adjusted, reason = apply_duration_ratio_calibration(0.80, vo, vid)
        assert adjusted == 0.80
        assert reason == ""

    def test_ratio_in_acceptable_range_no_penalty(self):
        """Ratio between 0.5 and 2.0 gives no penalty."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 15.0)  # 15s video -> ratio 1.5
        adjusted, reason = apply_duration_ratio_calibration(0.80, vo, vid)
        assert adjusted == 0.80
        assert reason == ""

    def test_ratio_at_boundary_3_0_no_penalty(self):
        """Ratio exactly 3.0 gives no penalty (boundary is >3.0)."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 30.0)  # 30s video -> ratio 3.0
        adjusted, reason = apply_duration_ratio_calibration(0.80, vo, vid)
        assert adjusted == 0.80
        assert reason == ""

    def test_ratio_5_0_excessive_penalty(self):
        """Ratio 5.0 (video much longer) gives -0.03 penalty."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 50.0)  # 50s video -> ratio 5.0
        adjusted, reason = apply_duration_ratio_calibration(0.80, vo, vid)
        assert adjusted == pytest.approx(0.77, abs=1e-6)
        assert "duration ratio calibration" in reason
        assert "excessive" in reason

    def test_ratio_0_1_too_short_penalty(self):
        """Ratio 0.1 (video much shorter) gives -0.05 penalty."""
        vo = _make_segment(0.0, 30.0)   # 30s voiceover
        vid = _make_segment(0.0, 3.0)   # 3s video -> ratio 0.1
        adjusted, reason = apply_duration_ratio_calibration(0.80, vo, vid)
        assert adjusted == pytest.approx(0.75, abs=1e-6)
        assert "duration ratio calibration" in reason
        assert "too short" in reason

    def test_ratio_at_boundary_0_3_no_penalty(self):
        """Ratio exactly 0.3 gives no penalty (boundary is <0.3)."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 3.0)   # 3s video -> ratio 0.3
        adjusted, reason = apply_duration_ratio_calibration(0.80, vo, vid)
        assert adjusted == 0.80
        assert reason == ""

    def test_missing_vo_duration_no_penalty(self):
        """Missing voiceover duration (0) gives no penalty."""
        vo = Mock()
        vo.duration = 0.0
        vid = _make_segment(0.0, 10.0)
        adjusted, reason = apply_duration_ratio_calibration(0.80, vo, vid)
        assert adjusted == 0.80
        assert reason == ""

    def test_missing_vid_duration_no_penalty(self):
        """Missing video duration (0) gives no penalty."""
        vo = _make_segment(0.0, 10.0)
        vid = Mock()
        vid.duration = 0.0
        adjusted, reason = apply_duration_ratio_calibration(0.80, vo, vid)
        assert adjusted == 0.80
        assert reason == ""

    def test_no_duration_attr_no_penalty(self):
        """Segment without duration attribute gives no penalty."""
        vo = Mock(spec=[])  # No attributes
        vid = _make_segment(0.0, 10.0)
        adjusted, reason = apply_duration_ratio_calibration(0.80, vo, vid)
        assert adjusted == 0.80
        assert reason == ""

    def test_penalty_does_not_go_below_zero(self):
        """Penalty should not reduce confidence below 0."""
        vo = _make_segment(0.0, 30.0)
        vid = _make_segment(0.0, 1.0)   # ratio 0.033 -> too short -> -0.05
        adjusted, reason = apply_duration_ratio_calibration(0.02, vo, vid)
        assert adjusted == 0.0
        assert "too short" in reason

    def test_high_ratio_penalty_value(self):
        """Excessive ratio penalty is exactly 0.03."""
        vo = _make_segment(0.0, 2.0)    # 2s voiceover
        vid = _make_segment(0.0, 30.0)  # 30s video -> ratio 15.0
        adjusted, reason = apply_duration_ratio_calibration(1.0, vo, vid)
        assert adjusted == pytest.approx(0.97, abs=1e-6)

    def test_low_ratio_penalty_value(self):
        """Too-short ratio penalty is exactly 0.05."""
        vo = _make_segment(0.0, 30.0)   # 30s voiceover
        vid = _make_segment(0.0, 2.0)   # 2s video -> ratio 0.067
        adjusted, reason = apply_duration_ratio_calibration(1.0, vo, vid)
        assert adjusted == pytest.approx(0.95, abs=1e-6)

    def test_breakdown_component_name(self):
        """Reason includes component name for breakdown tracking."""
        vo = _make_segment(0.0, 10.0)
        vid = _make_segment(0.0, 50.0)  # ratio 5.0
        _, reason = apply_duration_ratio_calibration(0.80, vo, vid)
        assert "duration ratio calibration" in reason
