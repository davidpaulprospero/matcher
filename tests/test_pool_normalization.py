"""
Tests for confidence normalization by candidate pool size.

Tests the normalize_confidence_by_pool() function in src/matching/scoring.py.
"""

import pytest
from unittest.mock import MagicMock

from src.matching.scoring import (
    normalize_confidence_by_pool,
    POOL_NORMALIZATION_REFERENCE_SIZE,
    POOL_NORMALIZATION_MIN_FACTOR,
    POOL_NORMALIZATION_MAX_FACTOR,
    POOL_SMALL_THRESHOLD,
    POOL_LARGE_THRESHOLD,
    POOL_TIGHT_MARGIN_THRESHOLD,
)


class TestPoolNormalizationBasic:
    """Basic tests for normalize_confidence_by_pool function."""

    @pytest.mark.fast
    def test_disabled_returns_original_confidence(self):
        """When disabled, confidence is returned unchanged."""
        confidence = 0.75
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=50,
            pool_normalization_enabled=False
        )
        assert result == confidence
        assert reason == "pool_normalization_disabled"

    @pytest.mark.fast
    def test_empty_pool_returns_original_confidence(self):
        """Empty pool returns original confidence."""
        confidence = 0.8
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=0,
            pool_normalization_enabled=True
        )
        assert result == confidence
        assert reason == "empty_pool"

    @pytest.mark.fast
    def test_negative_pool_returns_original_confidence(self):
        """Negative pool size is treated as empty."""
        confidence = 0.8
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=-5,
            pool_normalization_enabled=True
        )
        assert result == confidence
        assert reason == "empty_pool"

    @pytest.mark.fast
    def test_reference_pool_size_no_change(self):
        """At reference pool size (50), normalization factor is 1.0."""
        confidence = 0.8
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=POOL_NORMALIZATION_REFERENCE_SIZE,
            pool_normalization_enabled=True
        )
        # Factor = sqrt(50/50) = 1.0, inverse = 1.0
        # Should be very close to original
        assert abs(result - confidence) < 0.001
        assert "medium_pool" in reason


class TestSmallPoolNormalization:
    """Tests for small pool (<10 candidates) normalization."""

    @pytest.mark.fast
    def test_small_pool_boosts_confidence(self):
        """Small pools get confidence boost due to factor < 1.0."""
        confidence = 0.7
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=5,
            pool_normalization_enabled=True
        )
        # Factor = sqrt(5/50) = 0.316, clamped to 0.8
        # Inverse factor = 1/0.8 = 1.25
        # Result = 0.7 * 1.25 = 0.875
        assert result > confidence
        assert "small_pool(5)" in reason

    @pytest.mark.fast
    def test_small_pool_with_clear_winner_extra_boost(self):
        """Small pool with clear winner gets additional +0.05 boost."""
        # Create mock candidates with clear winner (top - 2nd >= 0.1)
        mock_seg = MagicMock()
        candidates = [
            (mock_seg, 0.85),  # Top
            (mock_seg, 0.70),  # 2nd - margin = 0.15
            (mock_seg, 0.60),
        ]

        confidence = 0.75
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=5,
            candidates=candidates,
            pool_normalization_enabled=True
        )

        # Should get both factor boost and +0.05 adjustment
        assert result > confidence
        assert "small_pool" in reason
        assert "clear_winner" in reason
        assert "+0.05" in reason

    @pytest.mark.fast
    def test_small_pool_without_clear_winner(self):
        """Small pool without clear winner (tight margins) gets factor boost only."""
        mock_seg = MagicMock()
        candidates = [
            (mock_seg, 0.80),  # Top
            (mock_seg, 0.78),  # 2nd - margin = 0.02 (not clear)
            (mock_seg, 0.75),
        ]

        confidence = 0.75
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=5,
            candidates=candidates,
            pool_normalization_enabled=True
        )

        # Factor boost, but no extra +0.05
        assert result > confidence
        assert "small_pool" in reason
        assert "clear_winner" not in reason

    @pytest.mark.fast
    def test_small_pool_single_candidate(self):
        """Single candidate pool still gets boost."""
        mock_seg = MagicMock()
        candidates = [(mock_seg, 0.90)]

        confidence = 0.8
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=1,
            candidates=candidates,
            pool_normalization_enabled=True
        )

        assert result > confidence
        assert "small_pool(1)" in reason


class TestLargePoolNormalization:
    """Tests for large pool (>100 candidates) normalization."""

    @pytest.mark.fast
    def test_large_pool_reduces_confidence(self):
        """Large pools get confidence reduction due to factor > 1.0."""
        confidence = 0.85
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=150,
            pool_normalization_enabled=True
        )
        # Factor = sqrt(150/50) = 1.73, clamped to 1.2
        # Inverse factor = 1/1.2 = 0.833
        # Result = 0.85 * 0.833 = 0.708
        assert result < confidence
        assert "large_pool(150)" in reason

    @pytest.mark.fast
    def test_large_pool_with_tight_margin_extra_penalty(self):
        """Large pool with tight margin gets additional -0.05 penalty."""
        mock_seg = MagicMock()
        candidates = [
            (mock_seg, 0.82),  # Top
            (mock_seg, 0.80),  # 2nd - margin = 0.02 (tight)
            (mock_seg, 0.78),
        ]

        confidence = 0.85
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=150,
            candidates=candidates,
            pool_normalization_enabled=True
        )

        # Should get both factor reduction and -0.05 adjustment
        assert result < confidence
        assert "large_pool" in reason
        assert "tight_margin" in reason
        assert "-0.05" in reason

    @pytest.mark.fast
    def test_large_pool_without_tight_margin(self):
        """Large pool with clear winner gets factor reduction only."""
        mock_seg = MagicMock()
        candidates = [
            (mock_seg, 0.90),  # Top
            (mock_seg, 0.75),  # 2nd - margin = 0.15 (clear)
            (mock_seg, 0.70),
        ]

        confidence = 0.85
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=150,
            candidates=candidates,
            pool_normalization_enabled=True
        )

        # Factor reduction, but no extra -0.05
        assert result < confidence
        assert "large_pool" in reason
        assert "tight_margin" not in reason

    @pytest.mark.fast
    def test_very_large_pool_capped_at_max_factor(self):
        """Very large pools are capped at max factor (1.2)."""
        # Pool of 500: raw factor = sqrt(500/50) = 3.16, clamped to 1.2
        confidence = 0.9
        result1, _ = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=500,
            pool_normalization_enabled=True
        )

        # Pool of 1000: raw factor = sqrt(1000/50) = 4.47, still clamped to 1.2
        result2, _ = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=1000,
            pool_normalization_enabled=True
        )

        # Both should have same result due to capping
        assert abs(result1 - result2) < 0.001


class TestMediumPoolNormalization:
    """Tests for medium pool (10-100 candidates) normalization."""

    @pytest.mark.fast
    def test_medium_pool_moderate_adjustment(self):
        """Medium pools get moderate factor-based adjustment."""
        confidence = 0.8
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=50,  # Reference size
            pool_normalization_enabled=True
        )

        # At reference size, factor = 1.0
        assert abs(result - confidence) < 0.01
        assert "medium_pool(50)" in reason

    @pytest.mark.fast
    def test_medium_pool_lower_range(self):
        """Medium pool at lower range (10-25) gets slight boost."""
        confidence = 0.75
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=20,
            pool_normalization_enabled=True
        )

        # Factor = sqrt(20/50) = 0.632, inverse = 1.58
        # But capped at 0.8, so inverse = 1.25
        assert result > confidence
        assert "medium_pool(20)" in reason

    @pytest.mark.fast
    def test_medium_pool_upper_range(self):
        """Medium pool at upper range (75-100) gets slight reduction."""
        confidence = 0.85
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=75,
            pool_normalization_enabled=True
        )

        # Factor = sqrt(75/50) = 1.22, capped to 1.2, inverse = 0.833
        assert result < confidence
        assert "medium_pool(75)" in reason


class TestFactorCapping:
    """Tests for factor capping at [0.8, 1.2]."""

    @pytest.mark.fast
    def test_factor_capped_at_minimum(self):
        """Factor should be capped at 0.8 for very small pools."""
        # Pool of 2: raw factor = sqrt(2/50) = 0.2, clamped to 0.8
        confidence = 0.6
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=2,
            pool_normalization_enabled=True
        )

        # Inverse of 0.8 = 1.25, so max boost is 25%
        # 0.6 * 1.25 = 0.75 (plus potential +0.05 adjustment)
        assert result <= 1.0
        assert "small_pool" in reason

    @pytest.mark.fast
    def test_factor_capped_at_maximum(self):
        """Factor should be capped at 1.2 for very large pools."""
        # Pool of 200: raw factor = sqrt(200/50) = 2.0, clamped to 1.2
        confidence = 0.9
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=200,
            pool_normalization_enabled=True
        )

        # Inverse of 1.2 = 0.833, so max reduction is ~17%
        # 0.9 * 0.833 = 0.75 (minus potential -0.05 adjustment)
        assert result >= 0.0
        assert "large_pool" in reason


class TestEdgeCases:
    """Edge case tests for pool normalization."""

    @pytest.mark.fast
    def test_confidence_capped_at_one(self):
        """Result should never exceed 1.0."""
        confidence = 0.95
        mock_seg = MagicMock()
        candidates = [
            (mock_seg, 0.95),
            (mock_seg, 0.80),  # Clear winner
        ]

        result, _ = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=3,  # Small pool with clear winner = double boost
            candidates=candidates,
            pool_normalization_enabled=True
        )

        assert result <= 1.0

    @pytest.mark.fast
    def test_confidence_capped_at_zero(self):
        """Result should never go below 0.0."""
        confidence = 0.1
        mock_seg = MagicMock()
        candidates = [
            (mock_seg, 0.10),
            (mock_seg, 0.08),  # Tight margin
        ]

        result, _ = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=500,  # Large pool with tight margin = double penalty
            candidates=candidates,
            pool_normalization_enabled=True
        )

        assert result >= 0.0

    @pytest.mark.fast
    def test_zero_confidence_stays_zero(self):
        """Zero confidence remains zero after normalization."""
        result, _ = normalize_confidence_by_pool(
            confidence=0.0,
            pool_size=50,
            pool_normalization_enabled=True
        )

        assert result == 0.0

    @pytest.mark.fast
    def test_one_confidence_may_decrease(self):
        """Perfect confidence can decrease for large pools."""
        result, _ = normalize_confidence_by_pool(
            confidence=1.0,
            pool_size=150,  # Large pool
            pool_normalization_enabled=True
        )

        # Factor = 1.2 (capped), inverse = 0.833
        # 1.0 * 0.833 = 0.833
        assert result < 1.0

    @pytest.mark.fast
    def test_no_candidates_provided(self):
        """Works without candidates list (no margin-based adjustment)."""
        confidence = 0.75
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=5,
            candidates=None,
            pool_normalization_enabled=True
        )

        # Should still apply factor-based normalization
        assert result > confidence
        assert "small_pool" in reason
        assert "clear_winner" not in reason  # No margin analysis without candidates


class TestConfigIntegration:
    """Tests for config integration with pool normalization."""

    @pytest.mark.fast
    def test_config_option_default_enabled(self):
        """pool_normalization_enabled defaults to True in config."""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig()
        assert config.pool_normalization_enabled is True

    @pytest.mark.fast
    def test_config_option_can_be_disabled(self):
        """pool_normalization_enabled can be set to False."""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig(pool_normalization_enabled=False)
        assert config.pool_normalization_enabled is False


class TestConstants:
    """Tests for module constants."""

    @pytest.mark.fast
    def test_reference_size_is_50(self):
        """Reference pool size should be 50."""
        assert POOL_NORMALIZATION_REFERENCE_SIZE == 50

    @pytest.mark.fast
    def test_min_factor_is_point_eight(self):
        """Minimum factor should be 0.8."""
        assert POOL_NORMALIZATION_MIN_FACTOR == 0.8

    @pytest.mark.fast
    def test_max_factor_is_one_point_two(self):
        """Maximum factor should be 1.2."""
        assert POOL_NORMALIZATION_MAX_FACTOR == 1.2

    @pytest.mark.fast
    def test_small_threshold_is_ten(self):
        """Small pool threshold should be 10."""
        assert POOL_SMALL_THRESHOLD == 10

    @pytest.mark.fast
    def test_large_threshold_is_hundred(self):
        """Large pool threshold should be 100."""
        assert POOL_LARGE_THRESHOLD == 100

    @pytest.mark.fast
    def test_tight_margin_threshold_is_point_zero_five(self):
        """Tight margin threshold should be 0.05."""
        assert POOL_TIGHT_MARGIN_THRESHOLD == 0.05


class TestNormalizationFormula:
    """Tests verifying the sqrt(pool_size/50) formula."""

    @pytest.mark.fast
    def test_formula_at_reference_size(self):
        """sqrt(50/50) = 1.0"""
        raw_factor = (50 / POOL_NORMALIZATION_REFERENCE_SIZE) ** 0.5
        assert abs(raw_factor - 1.0) < 0.001

    @pytest.mark.fast
    def test_formula_at_small_pool(self):
        """sqrt(10/50) = 0.447, clamped to 0.8"""
        raw_factor = (10 / POOL_NORMALIZATION_REFERENCE_SIZE) ** 0.5
        assert abs(raw_factor - 0.447) < 0.01
        clamped = max(POOL_NORMALIZATION_MIN_FACTOR, min(POOL_NORMALIZATION_MAX_FACTOR, raw_factor))
        assert clamped == POOL_NORMALIZATION_MIN_FACTOR

    @pytest.mark.fast
    def test_formula_at_large_pool(self):
        """sqrt(200/50) = 2.0, clamped to 1.2"""
        raw_factor = (200 / POOL_NORMALIZATION_REFERENCE_SIZE) ** 0.5
        assert abs(raw_factor - 2.0) < 0.001
        clamped = max(POOL_NORMALIZATION_MIN_FACTOR, min(POOL_NORMALIZATION_MAX_FACTOR, raw_factor))
        assert clamped == POOL_NORMALIZATION_MAX_FACTOR

    @pytest.mark.fast
    def test_formula_at_boundary_pools(self):
        """Test formula at boundary pool sizes."""
        # Pool of 32: sqrt(32/50) = 0.8 (exactly at min factor)
        raw_factor = (32 / POOL_NORMALIZATION_REFERENCE_SIZE) ** 0.5
        assert abs(raw_factor - 0.8) < 0.01

        # Pool of 72: sqrt(72/50) = 1.2 (exactly at max factor)
        raw_factor = (72 / POOL_NORMALIZATION_REFERENCE_SIZE) ** 0.5
        assert abs(raw_factor - 1.2) < 0.01
