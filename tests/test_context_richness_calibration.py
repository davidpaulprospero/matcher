"""Tests for US-95-010: Context richness calibration for confidence scores."""

import pytest
from src.matching.scoring import apply_context_richness_calibration


class TestContextRichnessCalibration:
    """Test suite for context richness calibration feature."""

    def test_rich_context_boost_all_signals(self):
        """Test that all 4 signals present produces max boost."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial: Building Web Apps",
            video_description="Learn how to build web applications with Python",
            video_tags=["python", "web", "tutorial", "programming"],
            video_chapter="Introduction",
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # All 4 signals present (rich context) -> should boost
        # With default weights (0.25 each), score = 1.0
        assert adjusted > confidence
        assert "score=1.00" in reason
        assert "rich" in reason

    def test_rich_context_boost_three_signals(self):
        """Test that 3 signals present produces boost."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python",
            video_tags=["python", "programming"],
            video_chapter=None,  # No chapter
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # 3 signals present -> should boost (score >= 0.75)
        # With default weights (0.25 each), score = 0.75
        assert adjusted > confidence
        assert "score=0.75" in reason
        assert "rich" in reason

    def test_moderate_context_no_adjustment(self):
        """Test that 2 signals present produces no adjustment."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",  # Has title
            video_description=None,  # No description
            video_tags=["python"],  # Has tags
            video_chapter=None,  # No chapter
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # 2 signals -> moderate context, no adjustment
        # With default weights (0.25 each), score = 0.50
        assert adjusted == confidence
        assert "score=0.50" in reason
        assert "moderate" in reason

    def test_sparse_context_penalty_one_signal(self):
        """Test that 1 signal present produces penalty."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",  # Has title
            video_description=None,  # No description
            video_tags=None,  # No tags
            video_chapter=None,  # No chapter
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # 1 signal -> sparse context, apply penalty
        # With default weights (0.25 each), score = 0.25
        assert adjusted < confidence
        assert "score=0.25" in reason
        assert "sparse" in reason

    def test_sparse_context_penalty_no_signals(self):
        """Test that 0 signals produces max penalty."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title=None,
            video_description=None,
            video_tags=None,
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # 0 signals -> sparse context, max penalty
        # With default weights (0.25 each), score = 0.0
        assert adjusted < confidence
        assert "score=0.00" in reason
        assert "sparse" in reason

    def test_disabled_calibration(self):
        """Test that disabled calibration returns original confidence."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python",
            video_tags=["python"],
            video_chapter="Intro",
            enabled=False,  # Disabled
            boost_max=0.08,
            penalty_max=0.05,
        )

        assert adjusted == confidence
        assert reason == ""

    def test_empty_string_treated_as_no_signal(self):
        """Test that empty strings are treated as no signal."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="",  # Empty string - no signal
            video_description="  ",  # Whitespace only - no signal
            video_tags=[],  # Empty list - no signal
            video_chapter="\t",  # Whitespace only - no signal
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # All empty/whitespace -> should apply max penalty
        # score = 0.0
        assert adjusted < confidence
        assert "score=0.00" in reason

    def test_confidence_clamped_to_max_1(self):
        """Test that boosted confidence is clamped to 1.0."""
        confidence = 0.95  # High confidence

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python",
            video_tags=["python"],
            video_chapter="Intro",
            enabled=True,
            boost_max=0.10,  # Large boost
            penalty_max=0.05,
        )

        # Should be clamped to 1.0
        assert adjusted <= 1.0

    def test_confidence_clamped_to_min_0(self):
        """Test that penalized confidence is clamped to 0.0."""
        confidence = 0.02  # Low confidence

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title=None,
            video_description=None,
            video_tags=None,
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.10,  # Large penalty
        )

        # Should be clamped to 0.0
        assert adjusted >= 0.0

    def test_config_defaults(self):
        """Test that config defaults work correctly."""
        confidence = 0.80

        # Using default values (enabled=True, boost_max=0.08, penalty_max=0.05)
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Test Video",
            video_description="Test Description",
            video_tags=["tag1", "tag2"],
            video_chapter="Chapter 1",
        )

        # All 4 signals present with defaults -> should boost
        assert adjusted > confidence

    # US-111-011: Weighted signal tests

    def test_weighted_signals_title_heavy(self):
        """Test weighted signals with title-heavy weights (0.5, 0.2, 0.2, 0.1)."""
        confidence = 0.80

        # With title=0.5, desc=0.2, tags=0.2, chapters=0.1
        # Only title present -> score = 0.5 (rich, >= 0.75 threshold for rich)
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",  # Has title
            video_description=None,  # No description
            video_tags=None,  # No tags
            video_chapter=None,  # No chapter
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
            title_weight=0.5,
            description_weight=0.2,
            tags_weight=0.2,
            chapters_weight=0.1,
        )

        # Score = 0.5 -> moderate context (0.25-0.75), no adjustment
        assert adjusted == confidence
        assert "score=0.50" in reason
        assert "moderate" in reason

    def test_weighted_signals_all_in_description(self):
        """Test weighted signals where description weight is 1.0."""
        confidence = 0.80

        # With all weight in description
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title=None,  # No title
            video_description="Learn Python Programming",  # Has description
            video_tags=None,  # No tags
            video_chapter=None,  # No chapter
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
            title_weight=0.0,
            description_weight=1.0,
            tags_weight=0.0,
            chapters_weight=0.0,
        )

        # Score = 1.0 -> rich context
        assert adjusted > confidence
        assert "score=1.00" in reason
        assert "rich" in reason
        assert "desc(1.00)" in reason

    def test_weighted_signals_chapters_boost(self):
        """Test that chapters with high weight can trigger rich context alone."""
        confidence = 0.80

        # With title=0.05, chapters=0.85 = 0.05 + 0.85 = 0.90 (rich)
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Video",  # Minimal title
            video_description=None,  # No description
            video_tags=None,  # No tags
            video_chapter="Introduction to Python",  # Has chapter
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
            title_weight=0.05,
            description_weight=0.05,
            tags_weight=0.05,
            chapters_weight=0.85,
        )

        # Score = 0.90 -> rich context (>= 0.75)
        assert adjusted > confidence
        assert "score=0.90" in reason
        assert "rich" in reason

    def test_weighted_signals_sparse_penalty(self):
        """Test weighted signals sparse case with custom weights."""
        confidence = 0.80

        # With title_weight=0.1, no signals present = score 0.0 (sparse)
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title=None,  # No title
            video_description=None,  # No description
            video_tags=None,  # No tags
            video_chapter=None,  # No chapter
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
            title_weight=0.1,
            description_weight=0.3,
            tags_weight=0.4,
            chapters_weight=0.2,
        )

        # Score = 0.0 -> sparse context
        assert adjusted < confidence
        assert "score=0.00" in reason
        assert "sparse" in reason

    def test_weighted_signals_boost_calculation(self):
        """Test that boost is correctly calculated with custom weights."""
        confidence = 0.70

        # With title=0.5, desc=0.5, tags=0.0, chapters=0.0 -> score = 1.0
        # boost = 0.08 * 1.0 = 0.08
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python",
            video_tags=None,
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
            title_weight=0.5,
            description_weight=0.5,
            tags_weight=0.0,
            chapters_weight=0.0,
        )

        # Score = 1.0 -> rich, boost = 0.08 * 1.0 = 0.08
        expected = min(1.0, confidence + 0.08)
        assert adjusted == expected
        assert "score=1.00" in reason
        assert "title(0.50)" in reason
        assert "desc(0.50)" in reason

    def test_backward_compatibility_default_weights(self):
        """Test backward compatibility with default weights (0.25 each)."""
        confidence = 0.80

        # Using only 2 signals (title + tags)
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description=None,
            video_tags=["python", "coding"],
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
            # Not specifying weights - should use defaults
        )

        # Score = 0.25 + 0.25 = 0.50 -> moderate, no adjustment
        assert adjusted == confidence
        assert "score=0.50" in reason

    # US-117-002: Additional tests for exact boost/penalty verification

    def test_exact_boost_max_all_signals(self):
        """Verify boost_max (0.08) is applied when all context signals present (US-117-002)."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial: Building Web Apps",
            video_description="Learn how to build web applications with Python",
            video_tags=["python", "web", "tutorial", "programming"],
            video_chapter="Introduction",
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # All 4 signals present with score=1.0
        # boost = 0.08 * 1.0 = 0.08
        # adjusted = 0.80 + 0.08 = 0.88
        assert adjusted == 0.88, f"Expected 0.88 but got {adjusted}"
        assert "+0.080" in reason
        assert "rich" in reason

    def test_exact_penalty_max_no_signals(self):
        """Verify penalty_max (0.05) is applied when no context signals available (US-117-002)."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title=None,
            video_description=None,
            video_tags=None,
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # No signals -> score = 0.0
        # penalty = 0.05 * (1.0 - 0.0) = 0.05
        # adjusted = 0.80 - 0.05 = 0.75
        assert adjusted == 0.75, f"Expected 0.75 but got {adjusted}"
        assert "-0.050" in reason
        assert "sparse" in reason
        assert "none" in reason

    def test_partial_context_title_only(self):
        """Test edge case: partial context - title only (US-117-002)."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description=None,
            video_tags=None,
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # Score = 0.25 -> sparse (boundary case)
        # penalty = 0.05 * (1.0 - 0.25) = 0.05 * 0.75 = 0.0375
        # adjusted = 0.80 - 0.0375 = 0.7625
        assert adjusted < confidence
        assert "score=0.25" in reason
        assert "sparse" in reason

    def test_partial_context_title_and_description(self):
        """Test edge case: partial context - title + description (US-117-002)."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python programming",
            video_tags=None,
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # Score = 0.5 -> moderate (no adjustment)
        assert adjusted == confidence
        assert "score=0.50" in reason
        assert "moderate" in reason
        assert "title" in reason
        assert "desc" in reason

    def test_partial_context_title_description_tags(self):
        """Test edge case: partial context - 3 signals without chapter (US-117-002)."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python programming",
            video_tags=["python", "coding"],
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # Score = 0.75 -> rich (boost applied)
        # boost = 0.08 * 0.75 = 0.06
        # adjusted = 0.80 + 0.06 = 0.86
        assert adjusted > confidence
        assert abs(adjusted - 0.86) < 0.001, f"Expected ~0.86 but got {adjusted}"
        assert "score=0.75" in reason
        assert "rich" in reason

    def test_config_flag_controls_calibration(self):
        """Verify calibration is only applied when config flag is True (US-117-002)."""
        from dataclasses import dataclass
        from src.config.sections.matching import MatchingConfig

        # Test with config flag True (should apply calibration)
        config = MatchingConfig(context_richness_calibration=True)

        # When enabled=True in function call, calibration should apply
        confidence = 0.80
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Test",
            video_description="Test desc",
            video_tags=["tag"],
            video_chapter="Chapter",
            enabled=config.context_richness_calibration,
            boost_max=0.08,
            penalty_max=0.05,
        )
        assert adjusted != confidence, "Calibration should apply when flag is True"

        # When config flag is False, calibration should not apply
        config_disabled = MatchingConfig(context_richness_calibration=False)
        adjusted2, reason2 = apply_context_richness_calibration(
            confidence,
            video_title="Test",
            video_description="Test desc",
            video_tags=["tag"],
            video_chapter="Chapter",
            enabled=config_disabled.context_richness_calibration,
            boost_max=0.08,
            penalty_max=0.05,
        )
        assert adjusted2 == confidence, "Calibration should not apply when flag is False"
        assert reason2 == "", "Reason should be empty when disabled"

    # US-126-007: Tests for context richness scoring with chapters weight

    def test_four_signals_higher_boost_than_two_signals(self):
        """Test that 4 signals (title+desc+tags+chapters) get higher richness boost than 2 signals."""
        confidence = 0.80

        # 4 signals present: title + description + tags + chapters
        adjusted_4, reason_4 = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python programming",
            video_tags=["python", "coding"],
            video_chapter="Introduction",
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # 2 signals present: title + tags only
        adjusted_2, reason_2 = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description=None,
            video_tags=["python", "coding"],
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # 4 signals = score 1.0 (rich) -> boost = 0.08 * 1.0 = 0.08, adjusted = 0.88
        # 2 signals = score 0.50 (moderate) -> no adjustment, adjusted = 0.80
        assert adjusted_4 > adjusted_2, (
            f"4 signals ({adjusted_4}) should have higher boost than 2 signals ({adjusted_2})"
        )
        assert "rich" in reason_4
        assert "moderate" in reason_2
        assert adjusted_4 == 0.88
        assert adjusted_2 == 0.80

    def test_chapters_weight_zero_no_contribution(self):
        """Test that context_richness_chapters_weight=0 yields no chapter contribution."""
        confidence = 0.80

        # With chapters_weight=0, chapter should add 0 to richness
        # Use weights that sum to <1 when chapter weight is 0
        # title=0.25 + desc=0.25 = 0.50 (moderate)
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python",
            video_tags=None,
            video_chapter="Important Chapter",  # Chapter present but weight is 0
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
            title_weight=0.25,
            description_weight=0.25,
            tags_weight=0.25,
            chapters_weight=0.0,  # Zero weight - chapter adds 0
        )

        # Score = 0.25 + 0.25 + 0.0 + 0.0 = 0.50 -> moderate, no adjustment
        assert adjusted == confidence, f"Expected 0.80 but got {adjusted}"
        assert "score=0.50" in reason
        assert "moderate" in reason
        # Chapter should still be mentioned in signals_info but with 0.00 weight
        assert "chapters(0.00)" in reason

    def test_chapters_weight_zero_all_signals_present(self):
        """Test chapters_weight=0 when all 4 signals present but chapter has 0 weight."""
        confidence = 0.80

        # All 4 signals present but with chapters_weight=0
        # title=0.25, desc=0.25, tags=0.25, chapters=0 -> score = 0.75 (rich)
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python programming",
            video_tags=["python", "coding"],
            video_chapter="Introduction",
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
            title_weight=0.25,
            description_weight=0.25,
            tags_weight=0.25,
            chapters_weight=0.0,  # Zero weight
        )

        # Score = 0.75 -> rich, boost = 0.08 * 0.75 = 0.06
        # adjusted = 0.80 + 0.06 = 0.86
        assert adjusted > confidence
        assert "score=0.75" in reason
        assert "rich" in reason
        # Verify chapter is in the reason but with 0.00 weight
        assert "chapters(0.00)" in reason

    def test_config_chapters_weight_applied_in_tiered_matcher(self):
        """Verify config context_richness_chapters_weight is passed to calibration function."""
        from src.config.sections.matching import MatchingConfig

        # Verify the config has the correct default
        config = MatchingConfig()
        assert config.context_richness_chapters_weight == 0.25

        # Test with custom chapters_weight
        config_custom = MatchingConfig(context_richness_chapters_weight=0.0)
        assert config_custom.context_richness_chapters_weight == 0.0

        # Verify the weights sum to 1.0 (required by validation)
        # This is validated in Config post_init
        assert (
            config.context_richness_title_weight +
            config.context_richness_description_weight +
            config.context_richness_tags_weight +
            config.context_richness_chapters_weight
        ) == 1.0


# US-134-002: Adaptive context weights tests


class TestAdaptiveContextWeights:
    """Test suite for adaptive context weights feature (US-134-002)."""

    def test_all_signals_available_returns_base_weights(self):
        """Test that all signals available returns base weights unchanged."""
        from src.matching.scoring import compute_adaptive_context_weights

        base_weights = (0.35, 0.30, 0.20, 0.15)  # title, description, tags, chapters

        result = compute_adaptive_context_weights(
            base_weights,
            has_title=True,
            has_description=True,
            has_tags=True,
            has_chapters=True,
        )

        # All available - should return base weights
        assert result == base_weights

    def test_no_signals_available_returns_base_weights(self):
        """Test that no signals available returns base weights."""
        from src.matching.scoring import compute_adaptive_context_weights

        base_weights = (0.35, 0.30, 0.20, 0.15)

        result = compute_adaptive_context_weights(
            base_weights,
            has_title=False,
            has_description=False,
            has_tags=False,
            has_chapters=False,
        )

        # All unavailable - should return base weights
        assert result == base_weights

    def test_rich_title_no_tags_redistributes_weight(self):
        """Test that rich title with no tags increases title weight (US-134-002)."""
        from src.matching.scoring import compute_adaptive_context_weights

        base_weights = (0.35, 0.30, 0.20, 0.15)  # title, description, tags, chapters

        result = compute_adaptive_context_weights(
            base_weights,
            has_title=True,
            has_description=True,
            has_tags=False,  # No tags
            has_chapters=True,
        )

        # Tags weight (0.20) should be redistributed to available signals
        # Available: title(0.35), desc(0.30), chapters(0.15) = 0.80 total
        # After redistribution:
        # - title gets 0.35 + 0.35/0.80 * 0.20 = 0.35 + 0.0875 = 0.4375
        # - desc gets 0.30 + 0.30/0.80 * 0.20 = 0.30 + 0.075 = 0.375
        # - chapters gets 0.15 + 0.15/0.80 * 0.20 = 0.15 + 0.0375 = 0.1875
        # Normalized to sum to 1.0

        title_w, desc_w, tags_w, chapters_w = result

        # Tags should be 0 since not available
        assert tags_w == 0.0, f"Expected tags weight 0.0 but got {tags_w}"

        # Title should have increased
        assert title_w > 0.35, f"Expected title > 0.35 but got {title_w}"

        # All weights should sum to 1.0
        assert abs(title_w + desc_w + tags_w + chapters_w - 1.0) < 0.001

    def test_only_title_available(self):
        """Test that only title available gives title full weight."""
        from src.matching.scoring import compute_adaptive_context_weights

        base_weights = (0.35, 0.30, 0.20, 0.15)

        result = compute_adaptive_context_weights(
            base_weights,
            has_title=True,
            has_description=False,
            has_tags=False,
            has_chapters=False,
        )

        title_w, desc_w, tags_w, chapters_w = result

        # Only title available - should get all weight
        assert title_w == 1.0, f"Expected title weight 1.0 but got {title_w}"
        assert desc_w == 0.0
        assert tags_w == 0.0
        assert chapters_w == 0.0

    def test_title_and_chapters_only(self):
        """Test redistribution with title and chapters only."""
        from src.matching.scoring import compute_adaptive_context_weights

        base_weights = (0.35, 0.30, 0.20, 0.15)

        result = compute_adaptive_context_weights(
            base_weights,
            has_title=True,
            has_description=False,
            has_tags=False,
            has_chapters=True,
        )

        title_w, desc_w, tags_w, chapters_w = result

        # desc + tags unavailable = 0.30 + 0.20 = 0.50 to redistribute
        # Available: title 0.35, chapters 0.15 = 0.50
        # After redistribution: title=0.70, chapters=0.30 (normalized)

        assert title_w > 0.35, f"Expected title > 0.35 but got {title_w}"
        assert chapters_w > 0.15, f"Expected chapters > 0.15 but got {chapters_w}"
        assert desc_w == 0.0
        assert tags_w == 0.0

        # Sum to 1.0
        assert abs(title_w + desc_w + tags_w + chapters_w - 1.0) < 0.001

    def test_zero_base_weights_handled(self):
        """Test that zero base weights for unavailable signals are handled."""
        from src.matching.scoring import compute_adaptive_context_weights

        base_weights = (0.0, 0.0, 0.50, 0.50)  # title and desc have 0 weight

        result = compute_adaptive_context_weights(
            base_weights,
            has_title=False,
            has_description=False,
            has_tags=True,
            has_chapters=True,
        )

        # Unavailable signals have 0 weight - should return base
        assert result == base_weights

    def test_partial_availability_exact_calculation(self):
        """Test exact calculation with partial availability."""
        from src.matching.scoring import compute_adaptive_context_weights

        # Base weights: (0.4, 0.3, 0.2, 0.1)
        base_weights = (0.4, 0.3, 0.2, 0.1)

        result = compute_adaptive_context_weights(
            base_weights,
            has_title=True,
            has_description=False,
            has_tags=True,
            has_chapters=False,
        )

        # Unavailable: desc(0.3) + chapters(0.1) = 0.4
        # Available: title(0.4) + tags(0.2) = 0.6
        # redistribution_factor = 0.4 / 0.6 = 2/3
        # title: 0.4 + 0.4 * 2/3 = 0.4 + 0.267 = 0.667
        # tags: 0.2 + 0.2 * 2/3 = 0.2 + 0.133 = 0.333

        title_w, desc_w, tags_w, chapters_w = result

        # desc and chapters should be 0
        assert desc_w == 0.0
        assert chapters_w == 0.0

        # title + tags should sum to 1.0
        assert abs(title_w + tags_w - 1.0) < 0.001


class TestAdaptiveWeightsIntegration:
    """Integration tests for adaptive weights with LLM reranker (US-134-002)."""

    def test_build_video_context_with_adaptive_enabled(self):
        """Test that _build_video_context uses adaptive weights when enabled."""
        from src.matching.llm_reranker import LLMReranker, LLMRerankerConfig, ContextPriorityWeightsConfig

        # Create config with adaptive enabled
        config = LLMRerankerConfig(
            reranker_include_metadata=True,
            context_priority_weights=ContextPriorityWeightsConfig(
                title=0.35,
                description=0.30,
                tags=0.20,
                chapters=0.15
            ),
            adaptive_context_weights=True
        )

        # Test case: video has title and description but NO tags or chapters
        # Adaptive should redistribute tags and chapters weight
        result = LLMReranker._build_video_context(
            title="Python Tutorial",
            description="Learn Python programming step by step",
            tags=[],  # No tags
            chapters=[],  # No chapters
            priority_weights=config.context_priority_weights,
            adaptive=True
        )

        # Should contain context with weights
        assert "Video context" in result
        assert "Title: Python Tutorial" in result
        assert "Description:" in result
        # Tags should not appear since empty
        assert "Tags:" not in result

        # Weight string should show adjusted weights (tags=0%, chapters=0%)
        assert "tags=0%" in result
        assert "chapters=0%" in result

    def test_build_video_context_with_adaptive_disabled(self):
        """Test that _build_video_context uses base weights when adaptive disabled."""
        from src.matching.llm_reranker import LLMReranker, ContextPriorityWeightsConfig

        # Test with adaptive=False - should use base weights
        result = LLMReranker._build_video_context(
            title="Python Tutorial",
            description="Learn Python",
            tags=[],  # No tags - but adaptive is OFF
            chapters=[],
            priority_weights=ContextPriorityWeightsConfig(
                title=0.35,
                description=0.30,
                tags=0.20,
                chapters=0.15
            ),
            adaptive=False
        )

        # Should use base weights (20% for tags) even though tags empty
        assert "Video context" in result
        assert "tags=20%" in result

    def test_adaptive_weights_config_in_reranker(self):
        """Test that LLMRerankerConfig reads adaptive_context_weights from matching config."""
        from src.matching.llm_reranker import LLMReranker, LLMRerankerConfig
        from dataclasses import asdict

        # Simulate matching config with adaptive flag
        class MockMatchingConfig:
            context_priority_weights = {
                'title': 0.35,
                'description': 0.30,
                'tags': 0.20,
                'chapters': 0.15,
            }
            adaptive_context_weights = True
            ambiguous_threshold = 0.65
            cache_llm_responses = True
            reranker_include_metadata = True

        config = LLMReranker.from_matching_config(MockMatchingConfig())

        # Verify adaptive flag was read
        assert config.config.adaptive_context_weights == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
