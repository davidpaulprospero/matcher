"""
Tests for apply_duration_context_boost (US-134-006).

Verifies confidence boost/penalty based on duration ratio similarity:
- Boost when ratio is within optimal range (0.8-1.2 by default)
- Penalty when ratio is outside optimal but within acceptable range (0.3-3.0)
- No adjustment when disabled or when ratio is outside acceptable range.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_duration_context_boost
from src.utils import SRTSegment


def _make_segment(start: float, end: float, text: str = "test") -> SRTSegment:
    """Create an SRTSegment with the given start/end times."""
    return SRTSegment(index=0, start_time=start, end_time=end, text=text)


def _make_scoring_config(
    enabled: bool = True,
    opt_range: List[float] = None,
    boost_max: float = 0.05,
    penalty_max: float = 0.10,
) -> Mock:
    """Create a mock MatchingScoringConfig."""
    config = Mock()
    config.duration_context_boost_enabled = enabled
    config.duration_optimal_ratio_range = opt_range if opt_range else [0.8, 1.2]
    config.duration_boost_max = boost_max
    config.duration_mismatch_penalty_max = penalty_max
    return config


class TestDurationContextBoost:
    """Tests for apply_duration_context_boost."""

    def test_disabled_returns_original(self):
        """When disabled, returns original confidence without adjustment."""
        vo = _make_segment(0.0, 10.0)
        vid = _make_segment(0.0, 10.0)
        config = _make_scoring_config(enabled=False)

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        assert adjusted == 0.80
        assert reason == ""

    def test_none_config_returns_original(self):
        """When config is None, returns original confidence."""
        vo = _make_segment(0.0, 10.0)
        vid = _make_segment(0.0, 10.0)

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, None)

        assert adjusted == 0.80
        assert reason == ""

    def test_optimal_ratio_1_0_max_boost(self):
        """Ratio 1.0 (equal durations) gets maximum boost."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 10.0)  # 10s video -> ratio 1.0
        config = _make_scoring_config(boost_max=0.05)

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        assert adjusted == pytest.approx(0.85, abs=1e-6)  # 0.80 + 0.05
        assert "duration_context_boost" in reason
        assert "optimal" in reason

    def test_optimal_ratio_0_8_partial_boost(self):
        """Ratio 0.8 gets partial boost (at edge of optimal range)."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 8.0)   # 8s video -> ratio 0.8
        config = _make_scoring_config(boost_max=0.05)

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        # At ratio 0.8 (edge of 0.8-1.0 range), normalized is ~0, so boost is ~0
        assert adjusted == 0.80
        assert "duration_context_boost" in reason

    def test_optimal_ratio_1_2_partial_boost(self):
        """Ratio 1.2 gets partial boost (at edge of optimal range)."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 12.0)  # 12s video -> ratio 1.2
        config = _make_scoring_config(boost_max=0.05)

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        # At ratio 1.2 (edge of 1.0-1.2 range), normalized is ~0, so boost is ~0
        assert adjusted == 0.80
        assert "duration_context_boost" in reason

    def test_below_optimal_ratio_penalty(self):
        """Ratio below optimal (0.5) gets penalty."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 5.0)   # 5s video -> ratio 0.5
        config = _make_scoring_config(opt_range=[0.8, 1.2], penalty_max=0.10)

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        # Ratio 0.5 is outside optimal (0.8-1.2) but within acceptable (0.3-3.0)
        # distance_from_optimal = (0.8 - 0.5) / 0.8 = 0.375
        # penalty = 0.10 * 0.375 = 0.0375
        assert adjusted < 0.80
        assert "duration_context_boost" in reason
        assert "mismatch" in reason

    def test_above_optimal_ratio_penalty(self):
        """Ratio above optimal (1.5) gets penalty."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 15.0)  # 15s video -> ratio 1.5
        config = _make_scoring_config(opt_range=[0.8, 1.2], penalty_max=0.10)

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        # Ratio 1.5 is outside optimal (0.8-1.2) but within acceptable (0.3-3.0)
        # distance_from_optimal = (1.5 - 1.2) / 1.2 = 0.25
        # penalty = 0.10 * 0.25 = 0.025
        assert adjusted < 0.80
        assert "duration_context_boost" in reason
        assert "mismatch" in reason

    def test_ratio_outside_acceptable_no_adjustment(self):
        """Ratio outside acceptable range (e.g., 5.0) returns original confidence."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 50.0)  # 50s video -> ratio 5.0
        config = _make_scoring_config()

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        assert adjusted == 0.80
        assert reason == ""

    def test_ratio_too_short_no_adjustment(self):
        """Ratio too short (<0.3) returns original confidence."""
        vo = _make_segment(0.0, 10.0)   # 10s voiceover
        vid = _make_segment(0.0, 2.0)   # 2s video -> ratio 0.2
        config = _make_scoring_config()

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        assert adjusted == 0.80
        assert reason == ""

    def test_missing_vo_duration_no_adjustment(self):
        """Missing voiceover duration returns original confidence."""
        vo = Mock()
        vo.start_time = 0
        vo.end_time = 0  # Zero duration
        vo.duration = 0

        vid = _make_segment(0.0, 10.0)
        config = _make_scoring_config()

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        assert adjusted == 0.80
        assert reason == ""

    def test_missing_video_duration_no_adjustment(self):
        """Missing video duration returns original confidence."""
        vo = _make_segment(0.0, 10.0)

        vid = Mock()
        vid.start_time = 0
        vid.end_time = 0  # Zero duration
        vid.duration = 0

        config = _make_scoring_config()

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        assert adjusted == 0.80
        assert reason == ""

    def test_boost_respects_max(self):
        """Boost is capped at duration_boost_max."""
        vo = _make_segment(0.0, 10.0)
        vid = _make_segment(0.0, 10.0)  # ratio 1.0
        config = _make_scoring_config(boost_max=0.03)

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        assert adjusted == pytest.approx(0.83, abs=1e-6)  # 0.80 + 0.03

    def test_penalty_respects_max(self):
        """Penalty is capped at duration_mismatch_penalty_max."""
        vo = _make_segment(0.0, 10.0)
        vid = _make_segment(0.0, 3.0)   # ratio 0.3 (edge of acceptable)
        config = _make_scoring_config(penalty_max=0.05)

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        # At ratio 0.3, distance is (0.8 - 0.3) / 0.8 = 0.625
        # But cap at 1.0, penalty = 0.05 * 0.625 = 0.03125
        assert adjusted == pytest.approx(0.80 - 0.03125, abs=1e-4)

    def test_confidence_capped_at_1_0(self):
        """Boost cannot exceed 1.0 confidence."""
        vo = _make_segment(0.0, 10.0)
        vid = _make_segment(0.0, 10.0)  # ratio 1.0
        config = _make_scoring_config(boost_max=0.50)  # Large boost

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        assert adjusted == 1.0  # Capped at 1.0

    def test_custom_optimal_range(self):
        """Custom optimal range is respected."""
        vo = _make_segment(0.0, 10.0)
        vid = _make_segment(0.0, 10.0)  # ratio 1.0
        config = _make_scoring_config(opt_range=[0.9, 1.1], boost_max=0.05)

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        # Ratio 1.0 is within 0.9-1.1, so it gets boost
        assert adjusted > 0.80

    def test_default_config_values(self):
        """Default config values work correctly."""
        vo = _make_segment(0.0, 10.0)
        vid = _make_segment(0.0, 10.0)  # ratio 1.0
        config = _make_scoring_config(
            enabled=True,
            opt_range=[0.8, 1.2],
            boost_max=0.05,
            penalty_max=0.10,
        )

        adjusted, reason = apply_duration_context_boost(0.80, vo, vid, config)

        assert adjusted == pytest.approx(0.85, abs=1e-6)
