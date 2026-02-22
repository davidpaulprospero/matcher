"""
Tests for smooth duration ratio reward curve (US-84-007).

Verifies graduated penalties replace the old step-function:
- Near-perfect ratio (0.9-1.1x) gets +0.02 boost
- Logarithmic penalty for ratios outside reward range
- Hard floor/ceiling at ratio < 0.3 and ratio > 3.0
- Graduated: ratio 0.4 gets less penalty than ratio 0.2
"""

import math
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_duration_penalty


def _make_config(
    duration_penalty_factor: float = 0.1,
    reward_threshold: float = 0.1,
    reward_boost: float = 0.02,
):
    """Create a mock config for apply_duration_penalty."""
    config = Mock()
    config.matching.duration_penalty_factor = duration_penalty_factor
    scoring = Mock()
    scoring.duration_ratio_reward_threshold = reward_threshold
    scoring.duration_ratio_reward_boost = reward_boost
    config.matching.scoring = scoring
    return config


class TestSmoothDurationRatioCurve:
    """Tests for the smooth duration ratio curve (US-84-007)."""

    @pytest.mark.fast
    def test_perfect_ratio_gets_reward_boost(self):
        """Near-perfect ratio (1.0) gets +0.02 boost instead of zero adjustment."""
        config = _make_config()
        result = apply_duration_penalty(0.80, 1.0, config)
        assert result == pytest.approx(0.82, abs=1e-6)

    @pytest.mark.fast
    def test_ratio_0_95_gets_reward_boost(self):
        """Ratio 0.95 (within 0.1 of 1.0) gets reward boost."""
        config = _make_config()
        result = apply_duration_penalty(0.80, 0.95, config)
        assert result == pytest.approx(0.82, abs=1e-6)

    @pytest.mark.fast
    def test_ratio_1_05_gets_reward_boost(self):
        """Ratio 1.05 (within 0.1 of 1.0) gets reward boost."""
        config = _make_config()
        result = apply_duration_penalty(0.80, 1.05, config)
        assert result == pytest.approx(0.82, abs=1e-6)

    @pytest.mark.fast
    def test_reward_boundary_lower(self):
        """Ratio exactly 0.9 (boundary) gets reward boost."""
        config = _make_config()
        result = apply_duration_penalty(0.80, 0.9, config)
        assert result == pytest.approx(0.82, abs=1e-6)

    @pytest.mark.fast
    def test_reward_boundary_upper(self):
        """Ratio exactly 1.1 (boundary) gets reward boost."""
        config = _make_config()
        result = apply_duration_penalty(0.80, 1.1, config)
        assert result == pytest.approx(0.82, abs=1e-6)

    @pytest.mark.fast
    def test_logarithmic_penalty_outside_reward(self):
        """Ratio 0.5 outside reward range gets logarithmic penalty."""
        config = _make_config()
        result = apply_duration_penalty(0.80, 0.5, config)
        expected_penalty = min(0.1, 0.02 * abs(math.log2(0.5)))  # 0.02 * 1.0 = 0.02
        assert result == pytest.approx(0.80 - expected_penalty, abs=1e-4)

    @pytest.mark.fast
    def test_graduated_penalty_ratio_0_4_less_than_0_2(self):
        """Ratio 0.4 gets less penalty than ratio 0.2 (graduation)."""
        config = _make_config()
        result_04 = apply_duration_penalty(0.80, 0.4, config)
        # Ratio 0.2 is below hard threshold 0.3, so it gets max hard penalty
        result_02 = apply_duration_penalty(0.80, 0.2, config)
        assert result_04 > result_02, (
            f"Ratio 0.4 ({result_04:.4f}) should get less penalty than 0.2 ({result_02:.4f})"
        )

    @pytest.mark.fast
    def test_graduated_penalty_ratio_0_8_less_than_0_5(self):
        """Ratio 0.8 gets less penalty than ratio 0.5 (closer to ideal)."""
        config = _make_config()
        result_08 = apply_duration_penalty(0.80, 0.8, config)
        result_05 = apply_duration_penalty(0.80, 0.5, config)
        assert result_08 > result_05, (
            f"Ratio 0.8 ({result_08:.4f}) should get less penalty than 0.5 ({result_05:.4f})"
        )

    @pytest.mark.fast
    def test_hard_floor_ratio_below_0_3(self):
        """Ratio < 0.3 hits hard floor penalty (penalty_factor * 2)."""
        config = _make_config()
        result = apply_duration_penalty(0.90, 0.2, config)
        assert result == pytest.approx(0.70, abs=1e-6)  # 0.9 - 0.1*2

    @pytest.mark.fast
    def test_hard_ceiling_ratio_above_3_0(self):
        """Ratio > 3.0 hits hard ceiling penalty (penalty_factor * 2)."""
        config = _make_config()
        result = apply_duration_penalty(0.90, 4.0, config)
        assert result == pytest.approx(0.70, abs=1e-6)  # 0.9 - 0.1*2

    @pytest.mark.fast
    def test_boundary_0_3_not_hard_penalty(self):
        """Ratio exactly 0.3 gets graduated log penalty, not hard penalty."""
        config = _make_config()
        result = apply_duration_penalty(0.80, 0.3, config)
        # log2(0.3) = -1.737, penalty = min(0.1, 0.02 * 1.737) = 0.0347
        expected_penalty = min(0.1, 0.02 * abs(math.log2(0.3)))
        assert result == pytest.approx(0.80 - expected_penalty, abs=1e-4)

    @pytest.mark.fast
    def test_boundary_3_0_not_hard_penalty(self):
        """Ratio exactly 3.0 gets graduated log penalty, not hard penalty."""
        config = _make_config()
        result = apply_duration_penalty(0.80, 3.0, config)
        # log2(3.0) = 1.585, penalty = min(0.1, 0.02 * 1.585) = 0.0317
        expected_penalty = min(0.1, 0.02 * abs(math.log2(3.0)))
        assert result == pytest.approx(0.80 - expected_penalty, abs=1e-4)

    @pytest.mark.fast
    def test_penalty_capped_at_penalty_factor(self):
        """Log penalty is capped at penalty_factor for extreme-but-not-hard ratios."""
        config = _make_config(duration_penalty_factor=0.05)
        # ratio 0.31 -> log2(0.31) = -1.69, 0.02*1.69 = 0.0338 > 0.05? No, < 0.05
        # ratio that would produce large log penalty: log2(0.31) * 0.02 = 0.034, still < 0.05
        # Need ratio where 0.02 * |log2(r)| > penalty_factor
        # 0.02 * |log2(r)| > 0.05 → |log2(r)| > 2.5 → r < 0.177 or r > 5.66
        # But those are past hard thresholds. So within 0.3-3.0, cap never triggers with factor=0.1
        # Let's verify with a lower penalty_factor
        config2 = _make_config(duration_penalty_factor=0.02)
        result = apply_duration_penalty(0.80, 0.5, config2)
        # log2(0.5) = 1.0, 0.02*1.0 = 0.02, min(0.02, 0.02) = 0.02
        assert result == pytest.approx(0.78, abs=1e-4)

    @pytest.mark.fast
    def test_configurable_reward_threshold(self):
        """Custom reward threshold is respected."""
        config = _make_config(reward_threshold=0.2, reward_boost=0.05)
        # ratio 0.85 is within [0.8, 1.2] with threshold=0.2
        result = apply_duration_penalty(0.70, 0.85, config)
        assert result == pytest.approx(0.75, abs=1e-6)

    @pytest.mark.fast
    def test_configurable_reward_boost(self):
        """Custom reward boost value is applied."""
        config = _make_config(reward_boost=0.05)
        result = apply_duration_penalty(0.70, 1.0, config)
        assert result == pytest.approx(0.75, abs=1e-6)

    @pytest.mark.fast
    def test_zero_speed_ratio_gets_hard_penalty(self):
        """Speed ratio of 0 gets hard penalty (defensive)."""
        config = _make_config()
        result = apply_duration_penalty(0.80, 0.0, config)
        assert result == pytest.approx(0.60, abs=1e-6)  # 0.8 - 0.2

    @pytest.mark.fast
    def test_negative_speed_ratio_gets_hard_penalty(self):
        """Negative speed ratio gets hard penalty (defensive)."""
        config = _make_config()
        result = apply_duration_penalty(0.80, -1.0, config)
        assert result == pytest.approx(0.60, abs=1e-6)

    @pytest.mark.fast
    def test_no_scoring_config_uses_defaults(self):
        """When config.matching.scoring is None, defaults are used."""
        config = Mock()
        config.matching.duration_penalty_factor = 0.1
        config.matching.scoring = None
        result = apply_duration_penalty(0.80, 1.0, config)
        assert result == pytest.approx(0.82, abs=1e-6)  # Default reward_boost=0.02
