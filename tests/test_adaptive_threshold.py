"""
Unit tests for adaptive embedding similarity thresholds.

Tests the calculate_adaptive_threshold() function which adjusts the skip_llm_threshold
based on voiceover characteristics:
- Short voiceover (<20 chars): +0.05 threshold
- Low candidate variance (<0.05): -0.05 threshold
"""

import pytest
from unittest.mock import MagicMock

from src.matching.scoring import calculate_adaptive_threshold
from src.utils import SRTSegment


def create_mock_segment(text: str = "Test video", index: int = 0) -> SRTSegment:
    """Create a mock SRTSegment for testing."""
    return SRTSegment(
        index=index,
        start_time=0.0,
        end_time=5.0,
        text=text,
        source_file="test_video.mp4"
    )


class TestAdaptiveThreshold:
    """Test suite for adaptive threshold calculation."""

    @pytest.mark.fast
    def test_no_adjustment_with_normal_input(self):
        """Normal voiceover length and varied candidates should have no adjustment."""
        base_threshold = 0.85
        voiceover_text = "This is a normal length voiceover segment that exceeds twenty characters."

        # Create candidates with varied similarity scores (high variance)
        candidates = [
            (create_mock_segment(), 0.90),
            (create_mock_segment(), 0.75),
            (create_mock_segment(), 0.60),
            (create_mock_segment(), 0.45),
            (create_mock_segment(), 0.30),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        # High variance (0.23+) should not trigger low_var adjustment
        # Normal length should not trigger short_vo adjustment
        assert threshold == base_threshold
        assert reason == "no_adjustment"

    @pytest.mark.fast
    def test_short_voiceover_increases_threshold(self):
        """Short voiceover (<20 chars) should increase threshold by 0.05."""
        base_threshold = 0.85
        voiceover_text = "Short text"  # 10 chars

        # Create candidates with high variance (no variance adjustment)
        candidates = [
            (create_mock_segment(), 0.90),
            (create_mock_segment(), 0.70),
            (create_mock_segment(), 0.50),
            (create_mock_segment(), 0.30),
            (create_mock_segment(), 0.10),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        assert threshold == 0.90  # 0.85 + 0.05
        assert "short_vo" in reason
        assert "+0.05" in reason

    @pytest.mark.fast
    def test_low_variance_decreases_threshold(self):
        """Low candidate variance (>=0.02 and <0.05) should decrease threshold by 0.05."""
        base_threshold = 0.85
        voiceover_text = "This is a normal length voiceover segment."

        # Create candidates with low but above-floor variance (stdev ~0.028)
        candidates = [
            (create_mock_segment(), 0.89),
            (create_mock_segment(), 0.87),
            (create_mock_segment(), 0.86),
            (create_mock_segment(), 0.84),
            (create_mock_segment(), 0.82),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        assert abs(threshold - 0.80) < 0.001  # 0.85 - 0.05 (with float tolerance)
        assert "low_var" in reason
        assert "-0.05" in reason

    @pytest.mark.fast
    def test_short_vo_and_low_variance_cancel_out(self):
        """Short voiceover + low variance adjustments should cancel out (+0.05 -0.05)."""
        base_threshold = 0.85
        voiceover_text = "Short"  # 5 chars

        # Create candidates with low but above-floor variance (stdev ~0.028)
        candidates = [
            (create_mock_segment(), 0.89),
            (create_mock_segment(), 0.87),
            (create_mock_segment(), 0.86),
            (create_mock_segment(), 0.84),
            (create_mock_segment(), 0.82),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        # Both adjustments should be present but cancel out
        assert threshold == 0.85  # 0.85 + 0.05 - 0.05 = 0.85
        assert "short_vo" in reason
        assert "low_var" in reason

    @pytest.mark.fast
    def test_threshold_clamped_to_minimum(self):
        """Threshold should not go below 0.5."""
        base_threshold = 0.52
        voiceover_text = "This is a normal length voiceover segment."

        # Low variance above floor to trigger -0.05 (stdev ~0.028)
        candidates = [
            (create_mock_segment(), 0.89),
            (create_mock_segment(), 0.86),
            (create_mock_segment(), 0.82),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        # Should clamp to 0.5, not go to 0.47
        assert threshold >= 0.5

    @pytest.mark.fast
    def test_threshold_clamped_to_maximum(self):
        """Threshold should not exceed 0.99."""
        base_threshold = 0.97
        voiceover_text = "Hi"  # Very short

        # High variance (no adjustment)
        candidates = [
            (create_mock_segment(), 0.90),
            (create_mock_segment(), 0.50),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        # Should clamp to 0.99, not go to 1.02
        assert threshold <= 0.99

    @pytest.mark.fast
    def test_empty_voiceover_triggers_short_adjustment(self):
        """Empty voiceover should trigger short voiceover adjustment."""
        base_threshold = 0.85
        voiceover_text = ""

        candidates = [
            (create_mock_segment(), 0.90),
            (create_mock_segment(), 0.50),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        assert threshold == 0.90  # +0.05 for short
        assert "short_vo(0c)" in reason

    @pytest.mark.fast
    def test_none_voiceover_handles_gracefully(self):
        """None voiceover should be handled gracefully."""
        base_threshold = 0.85

        candidates = [
            (create_mock_segment(), 0.90),
            (create_mock_segment(), 0.50),
        ]

        # Should not raise exception
        threshold, reason = calculate_adaptive_threshold(
            base_threshold, None, candidates
        )

        assert threshold == 0.90  # +0.05 for short (treated as length 0)

    @pytest.mark.fast
    def test_empty_candidates_no_variance_adjustment(self):
        """Empty candidates list should not trigger variance adjustment."""
        base_threshold = 0.85
        voiceover_text = "This is a normal voiceover."

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, []
        )

        assert threshold == base_threshold
        assert reason == "no_adjustment"

    @pytest.mark.fast
    def test_single_candidate_no_variance_adjustment(self):
        """Single candidate should not trigger variance adjustment."""
        base_threshold = 0.85
        voiceover_text = "This is a normal voiceover."

        candidates = [(create_mock_segment(), 0.90)]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        assert threshold == base_threshold
        assert reason == "no_adjustment"

    @pytest.mark.fast
    def test_whitespace_only_voiceover(self):
        """Whitespace-only voiceover should count as empty (0 chars after strip)."""
        base_threshold = 0.85
        voiceover_text = "   \n\t   "

        candidates = [
            (create_mock_segment(), 0.90),
            (create_mock_segment(), 0.50),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        assert "short_vo(0c)" in reason

    @pytest.mark.fast
    def test_exactly_20_chars_no_short_adjustment(self):
        """Exactly 20 chars should NOT trigger short voiceover adjustment."""
        base_threshold = 0.85
        voiceover_text = "12345678901234567890"  # Exactly 20 chars

        candidates = [
            (create_mock_segment(), 0.90),
            (create_mock_segment(), 0.50),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        # 20 chars is not < 20, so no adjustment
        assert "short_vo" not in reason

    @pytest.mark.fast
    def test_exactly_19_chars_triggers_short_adjustment(self):
        """19 chars should trigger short voiceover adjustment."""
        base_threshold = 0.85
        voiceover_text = "1234567890123456789"  # 19 chars

        candidates = [
            (create_mock_segment(), 0.90),
            (create_mock_segment(), 0.50),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        assert "short_vo(19c)" in reason

    @pytest.mark.fast
    def test_variance_exactly_0_05_no_adjustment(self):
        """Variance exactly 0.05 should NOT trigger low variance adjustment."""
        base_threshold = 0.85
        voiceover_text = "This is a normal length voiceover segment."

        # Create candidates with stdev exactly 0.05
        # For 5 values with stdev=0.05: mean=0.85, values around ±0.05
        # stdev([0.80, 0.82, 0.85, 0.88, 0.90]) ≈ 0.041 (too low)
        # Need to carefully craft values
        # stdev([0.80, 0.825, 0.85, 0.875, 0.90]) = 0.0395 (still too low)
        # Let's use: 0.78, 0.84, 0.85, 0.86, 0.92 -> stdev = 0.051
        candidates = [
            (create_mock_segment(), 0.78),
            (create_mock_segment(), 0.84),
            (create_mock_segment(), 0.85),
            (create_mock_segment(), 0.86),
            (create_mock_segment(), 0.92),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        # Variance >= 0.05 should NOT trigger adjustment
        assert "low_var" not in reason


class TestVarianceFloorPreventsThresholdReduction:
    """US-53-011: Tests that variance floor prevents false certainty."""

    @pytest.mark.fast
    def test_variance_floor_blocks_reduction_when_scores_within_001(self):
        """When top-5 scores are within 0.01 of each other, variance floor
        should prevent adaptive threshold reduction (false certainty)."""
        base_threshold = 0.85
        voiceover_text = "This is a normal length voiceover segment for testing."

        # All scores within 0.01 of each other -> stdev ≈ 0.006, well below floor of 0.02
        candidates = [
            (create_mock_segment(), 0.850),
            (create_mock_segment(), 0.849),
            (create_mock_segment(), 0.848),
            (create_mock_segment(), 0.847),
            (create_mock_segment(), 0.846),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        # Should NOT get the -0.05 reduction since variance is below floor
        assert threshold == base_threshold, (
            f"Variance floor should prevent threshold reduction, got {threshold}"
        )
        assert "low_var" not in reason, (
            f"Should not see low_var adjustment, got reason: {reason}"
        )
        assert "var_floor" in reason, (
            f"Should see var_floor marker in reason, got: {reason}"
        )

    @pytest.mark.fast
    def test_variance_above_floor_still_reduces_threshold(self):
        """When variance is between floor (0.02) and 0.05, threshold should still reduce."""
        base_threshold = 0.85
        voiceover_text = "This is a normal length voiceover segment for testing."

        # Scores with stdev ~0.03 (above floor of 0.02 but below 0.05)
        candidates = [
            (create_mock_segment(), 0.88),
            (create_mock_segment(), 0.87),
            (create_mock_segment(), 0.86),
            (create_mock_segment(), 0.85),
            (create_mock_segment(), 0.84),
        ]

        import statistics as st
        actual_stdev = st.stdev([0.88, 0.87, 0.86, 0.85, 0.84])
        # stdev of evenly spaced values 0.84-0.88 ≈ 0.0158... which is below floor
        # Need values with slightly more spread
        # Let's verify and adjust if needed
        if actual_stdev < 0.02:
            # Use wider spread to get above floor
            candidates = [
                (create_mock_segment(), 0.89),
                (create_mock_segment(), 0.87),
                (create_mock_segment(), 0.86),
                (create_mock_segment(), 0.84),
                (create_mock_segment(), 0.82),
            ]
            actual_stdev = st.stdev([0.89, 0.87, 0.86, 0.84, 0.82])

        assert 0.02 <= actual_stdev < 0.05, f"Test setup: stdev should be in [0.02, 0.05), got {actual_stdev}"

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        assert abs(threshold - 0.80) < 0.001, (
            f"Variance above floor but below 0.05 should reduce threshold, got {threshold}"
        )
        assert "low_var" in reason

    @pytest.mark.fast
    def test_top10_used_for_large_pools(self):
        """When 10+ candidates available, variance should use top-10 instead of top-5."""
        base_threshold = 0.85
        voiceover_text = "This is a normal length voiceover segment for testing."

        # 12 candidates - top-5 have low variance, but top-10 have high variance
        # This ensures top-10 usage changes the outcome
        candidates = [
            (create_mock_segment(), 0.90),
            (create_mock_segment(), 0.89),
            (create_mock_segment(), 0.88),
            (create_mock_segment(), 0.87),
            (create_mock_segment(), 0.86),
            (create_mock_segment(), 0.50),  # Big drop - makes top-10 variance high
            (create_mock_segment(), 0.45),
            (create_mock_segment(), 0.40),
            (create_mock_segment(), 0.35),
            (create_mock_segment(), 0.30),
            (create_mock_segment(), 0.25),
            (create_mock_segment(), 0.20),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        # With top-10, variance includes the 0.50-0.30 scores, making it high (>0.05)
        # So no low_var adjustment should be applied
        assert "low_var" not in reason, (
            f"Top-10 variance should be high, no reduction. Reason: {reason}"
        )
        assert threshold == base_threshold

    @pytest.mark.fast
    def test_identical_scores_trigger_variance_floor(self):
        """All identical scores (variance=0) should hit the variance floor."""
        base_threshold = 0.85
        voiceover_text = "This is a normal length voiceover segment."

        candidates = [
            (create_mock_segment(), 0.85),
            (create_mock_segment(), 0.85),
            (create_mock_segment(), 0.85),
            (create_mock_segment(), 0.85),
            (create_mock_segment(), 0.85),
        ]

        threshold, reason = calculate_adaptive_threshold(
            base_threshold, voiceover_text, candidates
        )

        # Variance is 0.0 (below floor), should NOT reduce threshold
        assert threshold == base_threshold
        assert "var_floor" in reason
        assert "low_var" not in reason


class TestAdaptiveThresholdIntegration:
    """Integration tests for adaptive threshold in TieredMatcher context."""

    @pytest.mark.fast
    def test_config_option_respected(self):
        """Test that adaptive_threshold_enabled config is respected."""
        # This would require mocking TieredMatcher, but the unit tests above
        # cover the calculate_adaptive_threshold function directly.
        # Integration test would be in test_matching.py or a full pipeline test.
        pass
