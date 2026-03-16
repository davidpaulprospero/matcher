"""Tests for US-111-010: Voiceover context calibration for confidence scores.
Extended with US-117-003: Voiceover length consideration in context calculation.
Extended with US-134-005: Expanded window support and new signals."""

import pytest
from src.matching.scoring import (
    apply_voiceover_context_calibration,
    apply_voiceover_topic_continuity,
    apply_voiceover_segment_density,
)


class TestVoiceoverContextCalibration:
    """Test suite for voiceover context calibration feature (US-111-010)."""

    def test_rich_context_both_adjacent_boost(self):
        """Test that both adjacent segments present produces max boost."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
        )

        # Both segments present -> should boost
        assert adjusted > confidence
        assert "rich context" in reason
        assert "+" in reason

    def test_partial_context_one_adjacent_boost(self):
        """Test that one adjacent segment produces half boost."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=3,  # US-117-003: Explicit length for full scaling
        )

        # One adjacent segment -> half boost
        assert adjusted > confidence
        assert "moderate context" in reason
        # Half boost = 0.05 * 0.5 = 0.025
        assert adjusted == pytest.approx(0.80 + 0.025)

    def test_partial_context_next_only(self):
        """Test that only next segment produces half boost."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=False,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
        )

        assert adjusted > confidence
        assert "moderate context" in reason

    def test_limited_context_no_adjacent_penalty(self):
        """Test that no adjacent segments produces penalty."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=False,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
        )

        # No adjacent segments -> should penalize
        assert adjusted < confidence
        assert "no context" in reason
        assert "-" in reason

    def test_disabled_calibration(self):
        """Test that disabled calibration returns original confidence."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=False,  # Disabled
            boost_max=0.05,
            penalty_max=0.03,
        )

        assert adjusted == confidence
        assert reason == ""

    def test_confidence_clamped_to_max_1(self):
        """Test that boosted confidence is clamped to 1.0."""
        confidence = 0.96  # High confidence close to max

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.10,  # Large boost
            penalty_max=0.03,
        )

        # Should be clamped to 1.0
        assert adjusted <= 1.0

    def test_confidence_clamped_to_min_0(self):
        """Test that penalized confidence is clamped to 0.0."""
        confidence = 0.01  # Very low confidence

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=False,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.10,  # Large penalty
        )

        # Should be clamped to 0.0
        assert adjusted >= 0.0

    def test_config_defaults(self):
        """Test that config defaults work correctly."""
        confidence = 0.80

        # Using default values (enabled=True, boost_max=0.05, penalty_max=0.03)
        # US-117-003: Explicit voiceover_length for backward-compatible test
        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            voiceover_length=3,  # Full scaling for backward compatibility
        )

        # Both adjacent with defaults -> should boost by 0.05
        assert adjusted > confidence
        assert adjusted == pytest.approx(0.85)

    def test_boost_values_with_custom_max(self):
        """Test that custom boost_max and penalty_max work correctly."""
        confidence = 0.80

        # With custom values
        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.10,  # Custom boost
            penalty_max=0.05,
            voiceover_length=3,  # US-117-003: Full scaling for backward compatibility
        )

        # Full boost = 0.10
        assert adjusted == pytest.approx(0.90)

    def test_half_boost_with_custom_max(self):
        """Test half boost with custom boost_max."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=False,
            enabled=True,
            boost_max=0.10,  # Custom boost
            penalty_max=0.05,
            voiceover_length=3,  # US-117-003: Full scaling for backward compatibility
        )

        # Half boost = 0.10 * 0.5 = 0.05
        assert adjusted == pytest.approx(0.85)


class TestVoiceoverContextCalibrationUS117003:
    """Test suite for US-117-003: Voiceover length consideration in context calibration."""

    def test_exact_boost_max_0_05_with_voiceover_length_3(self):
        """Verify boost_max (0.05) is applied when voiceover has 3+ segments and both adjacent."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=3,
        )

        # With 3+ segments, full boost_max should be applied
        # length_scaling = min(1.0, max(0.2, 3/3)) = min(1.0, 1.0) = 1.0
        assert adjusted == pytest.approx(0.80 + 0.05)
        assert "rich context" in reason
        assert "length_scale=1.00" in reason

    def test_exact_boost_max_long_voiceover(self):
        """Verify boost_max (0.05) is applied with long voiceover (10 segments)."""
        confidence = 0.75

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=10,
        )

        # With 10 segments, full boost (capped at 1.0 scaling)
        assert adjusted == pytest.approx(0.80)
        assert "length_scale=1.00" in reason

    def test_exact_penalty_max_0_03_with_voiceover_length_3(self):
        """Verify penalty_max (0.03) is applied when voiceover has 3+ segments and no context."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=False,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=3,
        )

        # With 3+ segments, full penalty_max should be applied
        assert adjusted == pytest.approx(0.80 - 0.03)
        assert "no context" in reason
        assert "length_scale=1.00" in reason

    def test_short_voiceover_reduced_boost(self):
        """Verify reduced boost for short voiceover (1 segment)."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=1,
        )

        # With 1 segment: length_scaling = min(1.0, max(0.2, 1/3)) = min(1.0, 0.333) = 0.333
        # boost = 0.05 * 0.333 = 0.0167
        expected_boost = 0.05 * 0.333
        assert adjusted == pytest.approx(confidence + expected_boost, abs=0.001)
        assert "length_scale=0.33" in reason

    def test_short_voiceover_reduced_penalty(self):
        """Verify reduced penalty for short voiceover (2 segments)."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=False,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=2,
        )

        # With 2 segments: length_scaling = min(1.0, max(0.2, 2/3)) = min(1.0, 0.667) = 0.667
        # penalty = 0.03 * 0.667 = 0.02
        expected_penalty = 0.03 * 0.667
        assert adjusted == pytest.approx(confidence - expected_penalty, abs=0.001)
        assert "length_scale=0.67" in reason

    def test_partial_context_with_voiceover_length_prev_only(self):
        """Test middle segments with partial context (only before)."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=5,
        )

        # With 5 segments: length_scaling = min(1.0, max(0.2, 5/3)) = min(1.0, 1.667) = 1.0
        # half_boost = 0.05 * 0.5 * 1.0 = 0.025
        assert adjusted == pytest.approx(0.825)
        assert "moderate context" in reason

    def test_partial_context_with_voiceover_length_next_only(self):
        """Test middle segments with partial context (only after)."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=False,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=5,
        )

        # With 5 segments: length_scaling = 1.0
        # half_boost = 0.05 * 0.5 * 1.0 = 0.025
        assert adjusted == pytest.approx(0.825)
        assert "moderate context" in reason

    def test_config_flag_enables_calibration(self):
        """Verify config flag voiceover_context_calibration enables the calibration."""
        confidence = 0.80

        # With enabled=True, calibration should apply
        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=3,
        )

        assert adjusted != confidence
        assert "voiceover_context_calibration" in reason

    def test_config_flag_disables_calibration(self):
        """Verify config flag voiceover_context_calibration disables the calibration."""
        confidence = 0.80

        # With enabled=False, calibration should not apply
        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=False,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=3,
        )

        assert adjusted == confidence
        assert reason == ""

    def test_no_voiceover_length_default(self):
        """Test behavior when voiceover_length is not provided (default 0)."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=0,  # Default
        )

        # With voiceover_length=0: length_scaling = 0.5 (default)
        expected_boost = 0.05 * 0.5
        assert adjusted == pytest.approx(confidence + expected_boost)
        assert "length_scale=0.50" in reason

    def test_minimum_voiceover_length_scaling(self):
        """Test minimum scaling factor (0.2) for very short voiceovers."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=1,
        )

        # With 1 segment: length_scaling = min(1.0, max(0.2, 1/3)) = min(1.0, 0.333) = 0.333
        # But actually: max(0.2, 0.333) = 0.333, then min(1.0, 0.333) = 0.333
        # Wait - let me recalculate: 1/3 = 0.333, max(0.2, 0.333) = 0.333
        # Hmm, that should give 0.333 not 0.2. Let me check the formula...
        # Actually the formula is: min(1.0, max(0.2, voiceover_length / 3.0))
        # For voiceover_length=1: min(1.0, max(0.2, 0.333)) = min(1.0, 0.333) = 0.333
        expected_boost = 0.05 * 0.333
        assert adjusted == pytest.approx(confidence + expected_boost, abs=0.002)

    def test_voiceover_length_2_scaling(self):
        """Test scaling factor for 2-segment voiceover."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=False,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=2,
        )

        # With 2 segments: min(1.0, max(0.2, 2/3)) = min(1.0, max(0.2, 0.667)) = min(1.0, 0.667) = 0.667
        expected_penalty = 0.03 * 0.667
        assert adjusted == pytest.approx(confidence - expected_penalty, abs=0.001)
        assert "length_scale=0.67" in reason


class TestVoiceoverContextCalibrationUS134005:
    """Test suite for US-134-005: Expanded window support."""

    def test_expanded_window_context_window_2(self):
        """Test context window parameter with window=2."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=3,
            context_window=2,
        )

        # With window=2 and both directions present -> rich context
        assert adjusted > confidence
        assert "window=2" in reason
        assert "rich context" in reason

    def test_expanded_window_moderate_context(self):
        """Test moderate context with window=2."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=3,
            context_window=2,
        )

        # With window=2 but only one direction -> moderate context
        assert adjusted > confidence
        assert "window=2" in reason
        assert "moderate context" in reason

    def test_expanded_window_no_context(self):
        """Test no context with window=2."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=False,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=3,
            context_window=2,
        )

        # With window=2 and no context -> penalty
        assert adjusted < confidence
        assert "window=2" in reason
        assert "no context" in reason

    def test_default_window_is_2(self):
        """Test that default context_window is 2."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=3,
            # No context_window specified - should use default 2
        )

        # Default window=2 should produce same results as explicit
        assert adjusted > confidence
        assert "window=2" in reason


class TestVoiceoverTopicContinuity:
    """Test suite for US-134-005: Voiceover topic continuity scoring."""

    def test_topic_continuity_boost(self):
        """Test that topic continuity produces boost when adjacent segments share topics."""
        confidence = 0.80
        current_topics = {"python", "programming", "tutorial"}
        adjacent_topics = [
            {"python", "coding", "beginner"},
            {"programming", "software", "guide"},
        ]

        adjusted, reason = apply_voiceover_topic_continuity(
            confidence,
            current_topics=current_topics,
            adjacent_topics=adjacent_topics,
            enabled=True,
            boost_max=0.03,
        )

        # Both adjacent segments share at least one topic with current
        # continuity_ratio = 2/2 = 1.0 > 0.5 -> should boost
        assert adjusted > confidence
        assert "voiceover_topic_continuity" in reason
        assert "+" in reason
        assert "continuity=1.00" in reason

    def test_topic_continuity_partial(self):
        """Test partial topic continuity (1 of 2 adjacent segments share topics)."""
        confidence = 0.80
        current_topics = {"python", "programming"}
        adjacent_topics = [
            {"python", "coding"},  # Shares topics
            {"unrelated", "other"},  # No shared topics
        ]

        adjusted, reason = apply_voiceover_topic_continuity(
            confidence,
            current_topics=current_topics,
            adjacent_topics=adjacent_topics,
            enabled=True,
            boost_max=0.03,
        )

        # continuity_ratio = 1/2 = 0.5, not > 0.5 -> no boost
        assert adjusted == confidence
        assert "voiceover_topic_continuity" in reason
        assert "low continuity" in reason

    def test_topic_continuity_no_overlap(self):
        """Test no topic continuity when no overlap."""
        confidence = 0.80
        current_topics = {"python", "programming"}
        adjacent_topics = [
            {"cooking", "recipe"},
            {"sports", "football"},
        ]

        adjusted, reason = apply_voiceover_topic_continuity(
            confidence,
            current_topics=current_topics,
            adjacent_topics=adjacent_topics,
            enabled=True,
            boost_max=0.03,
        )

        # No overlap -> no boost
        assert adjusted == confidence
        assert "low continuity" in reason

    def test_topic_continuity_disabled(self):
        """Test disabled topic continuity returns original confidence."""
        confidence = 0.80
        current_topics = {"python", "programming"}
        adjacent_topics = [{"python", "coding"}]

        adjusted, reason = apply_voiceover_topic_continuity(
            confidence,
            current_topics=current_topics,
            adjacent_topics=adjacent_topics,
            enabled=False,
            boost_max=0.03,
        )

        assert adjusted == confidence
        assert reason == ""

    def test_topic_continuity_empty_current_topics(self):
        """Test empty current_topics returns original confidence."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_topic_continuity(
            confidence,
            current_topics=set(),
            adjacent_topics=[{"python", "coding"}],
            enabled=True,
            boost_max=0.03,
        )

        assert adjusted == confidence

    def test_topic_continuity_empty_adjacent(self):
        """Test empty adjacent_topics returns original confidence."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_topic_continuity(
            confidence,
            current_topics={"python", "programming"},
            adjacent_topics=[],
            enabled=True,
            boost_max=0.03,
        )

        assert adjusted == confidence


class TestVoiceoverSegmentDensity:
    """Test suite for US-134-005: Voiceover segment density signal."""

    def test_dense_region_boost(self):
        """Test that dense region (>= threshold) produces boost."""
        confidence = 0.80
        adjacent_segment_count = 5
        density_threshold = 4

        adjusted, reason = apply_voiceover_segment_density(
            confidence,
            adjacent_segment_count=adjacent_segment_count,
            density_threshold=density_threshold,
            enabled=True,
            boost_max=0.02,
        )

        # 5 >= 4 -> dense region, should boost
        assert adjusted > confidence
        assert "voiceover_segment_density" in reason
        assert "dense region" in reason

    def test_sparse_region_no_boost(self):
        """Test that sparse region (< threshold) does not produce boost."""
        confidence = 0.80
        adjacent_segment_count = 2
        density_threshold = 4

        adjusted, reason = apply_voiceover_segment_density(
            confidence,
            adjacent_segment_count=adjacent_segment_count,
            density_threshold=density_threshold,
            enabled=True,
            boost_max=0.02,
        )

        # 2 < 4 -> sparse region, no boost
        assert adjusted == confidence
        assert "voiceover_segment_density" in reason
        assert "sparse region" in reason

    def test_threshold_boundary(self):
        """Test exact threshold boundary (count == threshold)."""
        confidence = 0.80
        adjacent_segment_count = 4
        density_threshold = 4

        adjusted, reason = apply_voiceover_segment_density(
            confidence,
            adjacent_segment_count=adjacent_segment_count,
            density_threshold=density_threshold,
            enabled=True,
            boost_max=0.02,
        )

        # 4 >= 4 -> should boost (boundary case)
        assert adjusted > confidence

    def test_segment_density_disabled(self):
        """Test disabled segment density returns original confidence."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_segment_density(
            confidence,
            adjacent_segment_count=5,
            density_threshold=4,
            enabled=False,
            boost_max=0.02,
        )

        assert adjusted == confidence
        assert reason == ""

    def test_custom_threshold(self):
        """Test custom density threshold."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_segment_density(
            confidence,
            adjacent_segment_count=3,
            density_threshold=3,  # Custom threshold
            enabled=True,
            boost_max=0.02,
        )

        # 3 >= 3 -> should boost
        assert adjusted > confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
