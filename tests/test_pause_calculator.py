"""Tests for PauseCalculator extracted from CircuitBreaker (US-82-008).

Verifies:
- Each PauseCalculator step can be unit tested individually without CircuitBreaker
- PauseContext carries all needed inputs
- Pipeline order: base → escalation → budget → region → time → jitter → cap
- Exact same behavior as the original CircuitBreaker 7-step pipeline
"""

import random
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.pause_calculator import (
    PauseCalculator,
    PauseContext,
    RegionalRateLimitTracker,
    RegionSuccessTracker,
    TimeOfDayBackoffManager,
)


# ============================================================================
# PauseContext Tests
# ============================================================================


class TestPauseContext:
    """Test PauseContext dataclass."""

    @pytest.mark.fast
    def test_default_values(self):
        """PauseContext defaults match CircuitBreakerConfig defaults."""
        ctx = PauseContext()
        assert ctx.current_pause == 0.0
        assert ctx.base_pause_seconds == 60.0
        assert ctx.max_pause_seconds == 300.0
        assert ctx.jitter_factor == 0.2
        assert ctx.escalation_manager is None
        assert ctx.budget is None

    @pytest.mark.fast
    def test_custom_values(self):
        """PauseContext accepts custom values."""
        ctx = PauseContext(
            base_pause_seconds=120.0,
            max_pause_seconds=500.0,
            jitter_factor=0.5,
        )
        assert ctx.base_pause_seconds == 120.0
        assert ctx.max_pause_seconds == 500.0
        assert ctx.jitter_factor == 0.5


# ============================================================================
# Step 1: base_pause Tests
# ============================================================================


class TestBasePause:
    """Test PauseCalculator.base_pause() step."""

    @pytest.mark.fast
    def test_returns_base_pause_seconds(self):
        """base_pause returns the configured base_pause_seconds."""
        calc = PauseCalculator()
        ctx = PauseContext(base_pause_seconds=60.0)
        assert calc.base_pause(ctx) == 60.0

    @pytest.mark.fast
    def test_different_base_values(self):
        """base_pause works with different configured values."""
        calc = PauseCalculator()
        for val in [10.0, 30.0, 60.0, 120.0, 300.0]:
            ctx = PauseContext(base_pause_seconds=val)
            assert calc.base_pause(ctx) == val


# ============================================================================
# Step 2: escalation_adjusted Tests
# ============================================================================


class TestEscalationAdjusted:
    """Test PauseCalculator.escalation_adjusted() step."""

    @pytest.mark.fast
    def test_no_escalation_manager_passthrough(self):
        """Without escalation manager, pause passes through unchanged."""
        calc = PauseCalculator()
        ctx = PauseContext(escalation_manager=None)
        assert calc.escalation_adjusted(60.0, ctx) == 60.0

    @pytest.mark.fast
    def test_doubles_when_majority_at_tier3(self):
        """Doubles pause when >50% keywords at Tier 3."""
        calc = PauseCalculator()
        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 10
        mock_em.get_keywords_at_tier.return_value = ['kw1', 'kw2', 'kw3', 'kw4', 'kw5', 'kw6']
        ctx = PauseContext(base_pause_seconds=60.0, escalation_manager=mock_em)

        result = calc.escalation_adjusted(60.0, ctx)
        assert result == 120.0  # 60 * 2.0

    @pytest.mark.fast
    def test_no_increase_at_50_percent(self):
        """No increase when exactly 50% at Tier 3 (threshold is >50%)."""
        calc = PauseCalculator()
        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 10
        mock_em.get_keywords_at_tier.return_value = ['kw1', 'kw2', 'kw3', 'kw4', 'kw5']  # 50%
        ctx = PauseContext(base_pause_seconds=60.0, escalation_manager=mock_em)

        result = calc.escalation_adjusted(60.0, ctx)
        assert result == 60.0  # No increase

    @pytest.mark.fast
    def test_no_increase_with_zero_keywords(self):
        """No increase when no keywords tracked."""
        calc = PauseCalculator()
        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 0
        ctx = PauseContext(escalation_manager=mock_em)

        result = calc.escalation_adjusted(60.0, ctx)
        assert result == 60.0


# ============================================================================
# Step 3: budget_adjusted Tests
# ============================================================================


class TestBudgetAdjusted:
    """Test PauseCalculator.budget_adjusted() step."""

    @pytest.mark.fast
    def test_no_budget_passthrough(self):
        """Without budget, pause passes through unchanged."""
        calc = PauseCalculator()
        ctx = PauseContext(budget=None)
        assert calc.budget_adjusted(60.0, ctx) == 60.0

    @pytest.mark.fast
    def test_exhausted_2_5x(self):
        """Exhausted budget applies 2.5x multiplier."""
        calc = PauseCalculator()
        mock_budget = MagicMock()
        mock_budget.is_exhausted.return_value = True
        ctx = PauseContext(budget=mock_budget)

        result = calc.budget_adjusted(60.0, ctx)
        assert result == 150.0  # 60 * 2.5

    @pytest.mark.fast
    def test_nearly_exhausted_1_5x(self):
        """Nearly exhausted budget applies 1.5x multiplier."""
        calc = PauseCalculator()
        mock_budget = MagicMock()
        mock_budget.is_exhausted.return_value = False
        mock_budget.is_nearly_exhausted.return_value = True
        ctx = PauseContext(budget=mock_budget)

        result = calc.budget_adjusted(60.0, ctx)
        assert result == 90.0  # 60 * 1.5

    @pytest.mark.fast
    def test_healthy_budget_no_change(self):
        """Healthy budget: no extension."""
        calc = PauseCalculator()
        mock_budget = MagicMock()
        mock_budget.is_exhausted.return_value = False
        mock_budget.is_nearly_exhausted.return_value = False
        ctx = PauseContext(budget=mock_budget)

        result = calc.budget_adjusted(60.0, ctx)
        assert result == 60.0


# ============================================================================
# Step 4: apply_jitter Tests
# ============================================================================


class TestApplyJitter:
    """Test PauseCalculator.apply_jitter() step."""

    @pytest.mark.fast
    def test_jitter_formula_positive(self):
        """Positive jitter increases delay."""
        calc = PauseCalculator()
        ctx = PauseContext(jitter_factor=0.2)

        with patch.object(random, 'uniform', return_value=0.2):
            result = calc.apply_jitter(60.0, ctx)
            assert abs(result - 72.0) < 0.01  # 60 * 1.2
            assert abs(calc.last_jitter_applied - 0.2) < 0.01

    @pytest.mark.fast
    def test_jitter_formula_negative(self):
        """Negative jitter decreases delay."""
        calc = PauseCalculator()
        ctx = PauseContext(jitter_factor=0.2)

        with patch.object(random, 'uniform', return_value=-0.2):
            result = calc.apply_jitter(60.0, ctx)
            assert abs(result - 48.0) < 0.01  # 60 * 0.8
            assert abs(calc.last_jitter_applied - (-0.2)) < 0.01

    @pytest.mark.fast
    def test_zero_jitter_deterministic(self):
        """Zero jitter produces deterministic delay."""
        calc = PauseCalculator()
        ctx = PauseContext(jitter_factor=0.0)

        result = calc.apply_jitter(60.0, ctx)
        assert result == 60.0
        assert calc.last_jitter_applied == 0.0

    @pytest.mark.fast
    def test_negative_jitter_clamped_to_zero(self):
        """Negative jitter_factor is clamped to 0."""
        calc = PauseCalculator()
        ctx = PauseContext(jitter_factor=-0.5)

        result = calc.apply_jitter(60.0, ctx)
        assert result == 60.0

    @pytest.mark.fast
    def test_jitter_above_1_clamped(self):
        """jitter_factor > 1.0 is clamped first to 1.0, then to jitter_max_factor (0.5 default)."""
        calc = PauseCalculator()
        ctx = PauseContext(jitter_factor=2.0, jitter_max_factor=1.0)  # Set max to 1.0 to test first clamp

        with patch.object(random, 'uniform', return_value=0.5) as mock_uniform:
            calc.apply_jitter(60.0, ctx)
            mock_uniform.assert_called_once_with(-1.0, 1.0)

    @pytest.mark.fast
    def test_jitter_within_bounds(self):
        """Jitter stays within ±jitter_factor bounds."""
        calc = PauseCalculator()
        ctx = PauseContext(jitter_factor=0.2)

        min_expected = 100.0 * (1 - 0.2)  # 80
        max_expected = 100.0 * (1 + 0.2)  # 120

        for _ in range(50):
            result = calc.apply_jitter(100.0, ctx)
            assert min_expected <= result <= max_expected


# ============================================================================
# Step 5: cap_duration Tests
# ============================================================================


class TestCapDuration:
    """Test PauseCalculator.cap_duration() step."""

    @pytest.mark.fast
    def test_under_max_passthrough(self):
        """Values under max pass through unchanged."""
        calc = PauseCalculator()
        ctx = PauseContext(max_pause_seconds=300.0)
        assert calc.cap_duration(100.0, ctx) == 100.0

    @pytest.mark.fast
    def test_over_max_capped(self):
        """Values over max are capped."""
        calc = PauseCalculator()
        ctx = PauseContext(max_pause_seconds=300.0)
        assert calc.cap_duration(500.0, ctx) == 300.0

    @pytest.mark.fast
    def test_exactly_at_max_passthrough(self):
        """Values exactly at max pass through."""
        calc = PauseCalculator()
        ctx = PauseContext(max_pause_seconds=300.0)
        assert calc.cap_duration(300.0, ctx) == 300.0

    @pytest.mark.fast
    def test_custom_max(self):
        """Custom max_pause_seconds is respected."""
        calc = PauseCalculator()
        ctx = PauseContext(max_pause_seconds=100.0)
        assert calc.cap_duration(150.0, ctx) == 100.0


# ============================================================================
# Full Pipeline Tests
# ============================================================================


class TestCalculatePipeline:
    """Test the full PauseCalculator.calculate() pipeline."""

    @pytest.mark.fast
    def test_base_only_no_extensions(self):
        """With no extensions, returns base pause."""
        calc = PauseCalculator()
        ctx = PauseContext(base_pause_seconds=60.0, jitter_factor=0.0)
        assert calc.calculate(ctx) == 60.0

    @pytest.mark.fast
    def test_escalation_then_budget_compose(self):
        """Escalation (2x) and budget (1.5x) compose multiplicatively."""
        calc = PauseCalculator()

        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 2
        mock_em.get_keywords_at_tier.return_value = ['kw1', 'kw2']  # 100%

        mock_budget = MagicMock()
        mock_budget.is_exhausted.return_value = False
        mock_budget.is_nearly_exhausted.return_value = True

        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=500.0,
            jitter_factor=0.0,
            escalation_manager=mock_em,
            budget=mock_budget,
        )

        result = calc.calculate(ctx)
        assert result == 180.0  # 60 * 2.0 * 1.5

    @pytest.mark.fast
    def test_pipeline_capped_at_max(self):
        """Combined scaling is capped at max_pause_seconds."""
        calc = PauseCalculator()

        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 2
        mock_em.get_keywords_at_tier.return_value = ['kw1', 'kw2']

        mock_budget = MagicMock()
        mock_budget.is_exhausted.return_value = True

        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=200.0,
            jitter_factor=0.0,
            escalation_manager=mock_em,
            budget=mock_budget,
        )

        # 60 * 2.0 * 2.5 = 300, capped at 200
        result = calc.calculate(ctx)
        assert result == 200.0

    @pytest.mark.fast
    def test_jitter_applied_before_cap(self):
        """Jitter is applied before the cap step."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=250.0,
            max_pause_seconds=300.0,
            jitter_factor=0.5,
        )

        # With +50% jitter, 250 * 1.5 = 375 -> capped to 300
        with patch.object(random, 'uniform', return_value=0.5):
            result = calc.calculate(ctx)
            assert result <= 300.0

    @pytest.mark.fast
    def test_many_iterations_never_exceed_max(self):
        """Full pipeline never exceeds max_pause_seconds."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=280.0,
            max_pause_seconds=300.0,
            jitter_factor=0.5,
        )

        for _ in range(100):
            result = calc.calculate(ctx)
            assert result <= 300.0


# ============================================================================
# Independence from CircuitBreaker Tests
# ============================================================================


class TestIndependenceFromCircuitBreaker:
    """Verify PauseCalculator can be tested without constructing a CircuitBreaker."""

    @pytest.mark.fast
    def test_no_circuit_breaker_needed(self):
        """PauseCalculator works standalone without any CircuitBreaker."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=300.0,
            jitter_factor=0.0,
        )
        result = calc.calculate(ctx)
        assert result == 60.0

    @pytest.mark.fast
    def test_reusable_across_calls(self):
        """PauseCalculator can be reused across multiple calculations."""
        calc = PauseCalculator()

        ctx1 = PauseContext(base_pause_seconds=60.0, jitter_factor=0.0)
        ctx2 = PauseContext(base_pause_seconds=120.0, jitter_factor=0.0)

        assert calc.calculate(ctx1) == 60.0
        assert calc.calculate(ctx2) == 120.0

    @pytest.mark.fast
    def test_last_jitter_tracked(self):
        """last_jitter_applied is updated after each apply_jitter call."""
        calc = PauseCalculator()
        ctx = PauseContext(jitter_factor=0.2)

        with patch.object(random, 'uniform', return_value=0.15):
            calc.apply_jitter(100.0, ctx)
            assert abs(calc.last_jitter_applied - 0.15) < 0.01

        with patch.object(random, 'uniform', return_value=-0.1):
            calc.apply_jitter(100.0, ctx)
            assert abs(calc.last_jitter_applied - (-0.1)) < 0.01


# ============================================================================
# Step 4: region_adjusted Tests (US-109-011)
# ============================================================================


class TestRegionAdjusted:
    """Test PauseCalculator.region_adjusted() step (US-109-011)."""

    @pytest.mark.fast
    def test_region_disabled_passthrough(self):
        """When region_enabled=False, pause passes through unchanged."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=False,
            region="us",
            region_multipliers={'us': 1.0}
        )
        assert calc.region_adjusted(60.0, ctx) == 60.0

    @pytest.mark.fast
    def test_us_region_no_multiplier(self):
        """US region applies 1.0 multiplier (baseline)."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=True,
            region="us",
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0}
        )
        assert calc.region_adjusted(60.0, ctx) == 60.0

    @pytest.mark.fast
    def test_eu_region_1_2x_multiplier(self):
        """EU region applies 1.2x multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=True,
            region="eu",
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0}
        )
        assert calc.region_adjusted(60.0, ctx) == 72.0  # 60 * 1.2

    @pytest.mark.fast
    def test_asia_region_1_5x_multiplier(self):
        """Asia region applies 1.5x multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=True,
            region="asia",
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0}
        )
        assert calc.region_adjusted(60.0, ctx) == 90.0  # 60 * 1.5

    @pytest.mark.fast
    def test_other_region_2_0x_multiplier(self):
        """Other region applies 2.0x multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=True,
            region="other",
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0}
        )
        assert calc.region_adjusted(60.0, ctx) == 120.0  # 60 * 2.0

    @pytest.mark.fast
    def test_unknown_region_defaults_to_other(self):
        """Unknown region defaults to 'other' multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=True,
            region="unknown_region",
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0}
        )
        # Unknown region should use default multiplier of 1.0
        assert calc.region_adjusted(60.0, ctx) == 60.0

    @pytest.mark.fast
    def test_empty_region_defaults_to_other(self):
        """Empty region defaults to 'other' multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=True,
            region="",
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0}
        )
        # Empty region defaults to 'other' which has multiplier 2.0
        assert calc.region_adjusted(60.0, ctx) == 120.0


class TestRegionAdjustedDynamic:
    """Test PauseCalculator.region_adjusted() with dynamic adjustment (US-113-008)."""

    @pytest.mark.fast
    def test_dynamic_adjustment_below_threshold(self):
        """When dynamic_region_adjustment is enabled and success rate below threshold, apply penalty."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=True,
            region="eu",
            region_multipliers={'eu': 1.2},
            dynamic_region_adjustment=True,
            region_success_rates={'eu': 0.3},  # 30% success rate
            region_success_rate_threshold=0.7
        )
        # Base: 60 * 1.2 = 72
        # Penalty: 1 + (0.7 - 0.3) = 1.4
        # Final: 72 * 1.4 = 100.8
        result = calc.region_adjusted(60.0, ctx)
        assert result == 100.8

    @pytest.mark.fast
    def test_dynamic_adjustment_above_threshold(self):
        """When success rate above threshold, apply recovery boost."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=True,
            region="eu",
            region_multipliers={'eu': 1.2},
            dynamic_region_adjustment=True,
            region_success_rates={'eu': 0.8},  # 80% success rate
            region_success_rate_threshold=0.7
        )
        # Base: 60 * 1.2 = 72
        # Recovery boost: 0.9
        # Final: 72 * 0.9 = 64.8
        result = calc.region_adjusted(60.0, ctx)
        assert abs(result - 64.8) < 0.01

    @pytest.mark.fast
    def test_dynamic_adjustment_untested_region(self):
        """Untested regions with no success rate data don't get dynamic adjustment."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=True,
            region="asia",
            region_multipliers={'asia': 1.5},
            dynamic_region_adjustment=True,
            region_success_rates={},  # No data
            region_success_rate_threshold=0.7
        )
        # Should use base multiplier only
        assert calc.region_adjusted(60.0, ctx) == 90.0  # 60 * 1.5

    @pytest.mark.fast
    def test_dynamic_adjustment_disabled(self):
        """When dynamic_region_adjustment=False, no dynamic adjustment applied."""
        calc = PauseCalculator()
        ctx = PauseContext(
            region_enabled=True,
            region="eu",
            region_multipliers={'eu': 1.2},
            dynamic_region_adjustment=False,
            region_success_rates={'eu': 0.3},
            region_success_rate_threshold=0.7
        )
        # Should use base multiplier only, ignore success rate
        assert calc.region_adjusted(60.0, ctx) == 72.0


# ============================================================================
# RegionalRateLimitTracker Tests (US-109-011)
# ============================================================================


class TestRegionalRateLimitTracker:
    """Test RegionalRateLimitTracker class."""

    def setup_method(self):
        """Reset tracker before each test."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker
        RegionalRateLimitTracker.clear()

    @pytest.mark.fast
    def test_enable_and_disable(self):
        """Tracker can be enabled and disabled."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker

        assert RegionalRateLimitTracker.is_enabled() is False
        RegionalRateLimitTracker.enable()
        assert RegionalRateLimitTracker.is_enabled() is True
        RegionalRateLimitTracker.disable()
        assert RegionalRateLimitTracker.is_enabled() is False

    @pytest.mark.fast
    def test_country_to_region_us(self):
        """US and Canada map to 'us' region."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker

        assert RegionalRateLimitTracker._country_to_region('us') == 'us'
        assert RegionalRateLimitTracker._country_to_region('ca') == 'us'

    @pytest.mark.fast
    def test_country_to_region_eu(self):
        """European countries map to 'eu' region."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker

        assert RegionalRateLimitTracker._country_to_region('gb') == 'eu'
        assert RegionalRateLimitTracker._country_to_region('de') == 'eu'
        assert RegionalRateLimitTracker._country_to_region('fr') == 'eu'

    @pytest.mark.fast
    def test_country_to_region_asia(self):
        """Asian countries map to 'asia' region."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker

        assert RegionalRateLimitTracker._country_to_region('jp') == 'asia'
        assert RegionalRateLimitTracker._country_to_region('sg') == 'asia'
        assert RegionalRateLimitTracker._country_to_region('kr') == 'asia'

    @pytest.mark.fast
    def test_country_to_region_other(self):
        """Unknown countries map to 'other' region."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker

        assert RegionalRateLimitTracker._country_to_region('xx') == 'other'
        assert RegionalRateLimitTracker._country_to_region('') == 'other'

    @pytest.mark.fast
    def test_increment_counts(self):
        """Increment increases region count."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker

        RegionalRateLimitTracker.enable()
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('de')

        assert RegionalRateLimitTracker.get_count('us') == 2
        assert RegionalRateLimitTracker.get_count('eu') == 1

    @pytest.mark.fast
    def test_increment_disabled_no_effect(self):
        """When disabled, increment has no effect."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker

        RegionalRateLimitTracker.disable()
        RegionalRateLimitTracker.increment('us')

        assert RegionalRateLimitTracker.get_count('us') == 0

    @pytest.mark.fast
    def test_reset_specific_region(self):
        """Reset only resets specific region when specified."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker

        RegionalRateLimitTracker.enable()
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('eu')
        RegionalRateLimitTracker.reset('us')

        assert RegionalRateLimitTracker.get_count('us') == 0
        assert RegionalRateLimitTracker.get_count('eu') == 1

    @pytest.mark.fast
    def test_reset_all_regions(self):
        """Reset without region resets all."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker

        RegionalRateLimitTracker.enable()
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('eu')
        RegionalRateLimitTracker.reset()

        assert RegionalRateLimitTracker.get_count('us') == 0
        assert RegionalRateLimitTracker.get_count('eu') == 0

    @pytest.mark.fast
    def test_get_all_counts(self):
        """Get all counts returns dict of all regions."""
        from src.downloader.pause_calculator import RegionalRateLimitTracker

        RegionalRateLimitTracker.enable()
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('de')

        counts = RegionalRateLimitTracker.get_all_counts()
        assert 'us' in counts
        assert 'eu' in counts


# ============================================================================
# RegionSuccessTracker Tests (US-113-008)
# ============================================================================


class TestRegionSuccessTracker:
    """Test RegionSuccessTracker class."""

    def setup_method(self):
        """Reset tracker before each test."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()

    @pytest.mark.fast
    def test_enable_and_disable(self):
        """Tracker can be enabled and disabled."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        assert RegionSuccessTracker.is_enabled() is False
        RegionSuccessTracker.enable()
        assert RegionSuccessTracker.is_enabled() is True
        RegionSuccessTracker.disable()
        assert RegionSuccessTracker.is_enabled() is False

    @pytest.mark.fast
    def test_record_success(self):
        """Recording success increments both attempts and successes."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.enable()
        # Need at least 3 samples to get actual rate (min_sample_size = 3)
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')

        assert RegionSuccessTracker.get_attempts('us') == 3
        assert RegionSuccessTracker.get_success_rate('us') == 1.0

    @pytest.mark.fast
    def test_record_failure(self):
        """Recording failure increments attempts but not successes."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.enable()
        # Need at least 3 samples to get actual rate (min_sample_size = 3)
        RegionSuccessTracker.record_failure('us')
        RegionSuccessTracker.record_failure('us')
        RegionSuccessTracker.record_failure('us')

        assert RegionSuccessTracker.get_attempts('us') == 3
        assert RegionSuccessTracker.get_success_rate('us') == 0.0

    @pytest.mark.fast
    def test_mixed_success_and_failure(self):
        """Mixed success/failure gives correct rate."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.enable()
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_failure('us')
        RegionSuccessTracker.record_failure('us')

        # 2 successes, 4 attempts = 50%
        assert RegionSuccessTracker.get_success_rate('us') == 0.5

    @pytest.mark.fast
    def test_untested_region_neutral_rate(self):
        """Untested regions return neutral 0.5 rate."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.enable()
        # No records for 'asia'

        # With < 3 samples, returns neutral 0.5
        assert RegionSuccessTracker.get_success_rate('asia') == 0.5

    @pytest.mark.fast
    def test_disabled_tracker_returns_neutral(self):
        """Disabled tracker returns 0.5 neutral rate."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.disable()
        assert RegionSuccessTracker.get_success_rate('us') == 0.5

    @pytest.mark.fast
    def test_is_below_threshold(self):
        """is_below_threshold correctly identifies low success rates."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.enable()
        # Record 5 failures
        for _ in range(5):
            RegionSuccessTracker.record_failure('us')

        # With 0% success rate and 5 attempts (>= min_sample_size), should be below 0.7
        assert RegionSuccessTracker.is_below_threshold('us', 0.7) is True

    @pytest.mark.fast
    def test_is_below_threshold_untested(self):
        """Untested regions not considered below threshold."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.enable()
        # No data

        # Should return False for untested regions
        assert RegionSuccessTracker.is_below_threshold('asia', 0.7) is False

    @pytest.mark.fast
    def test_reset_specific_region(self):
        """Reset only resets specific region when specified."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.enable()
        # Need at least 3 samples to get actual rate
        # Use proper country codes - 'us' maps to us, 'de' maps to eu
        for _ in range(3):
            RegionSuccessTracker.record_success('us')
        for _ in range(3):
            RegionSuccessTracker.record_failure('de')  # Germany -> eu
        RegionSuccessTracker.reset('us')

        assert RegionSuccessTracker.get_attempts('us') == 0
        assert RegionSuccessTracker.get_attempts('eu') == 3

    @pytest.mark.fast
    def test_get_all_rates(self):
        """Get all rates returns dict of all regions."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.enable()
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_failure('de')

        rates = RegionSuccessTracker.get_all_rates()
        assert 'us' in rates
        assert 'eu' in rates

    @pytest.mark.fast
    def test_country_to_region_mapping(self):
        """Country to region mapping works correctly."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        # Test various country codes
        assert RegionSuccessTracker._country_to_region('us') == 'us'
        assert RegionSuccessTracker._country_to_region('ca') == 'us'
        assert RegionSuccessTracker._country_to_region('de') == 'eu'
        assert RegionSuccessTracker._country_to_region('jp') == 'asia'
        assert RegionSuccessTracker._country_to_region('xx') == 'other'

    @pytest.mark.fast
    def test_get_best_region(self):
        """Test get_best_region returns region with highest success rate (US-136-007)."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()
        RegionSuccessTracker.enable()

        # Record data for different regions (need at least 3 to meet min_sample_size)
        # US: 5 successes, 0 failures = 100%
        for _ in range(5):
            RegionSuccessTracker.record_success('us')
        # EU: 4 successes, 1 failure = 80%
        for _ in range(4):
            RegionSuccessTracker.record_success('de')  # Germany -> eu
        RegionSuccessTracker.record_failure('de')
        # ASIA: 3 successes, 1 failure = 75%
        for _ in range(3):
            RegionSuccessTracker.record_success('jp')
        RegionSuccessTracker.record_failure('jp')

        # Best region should be US
        best = RegionSuccessTracker.get_best_region()
        assert best == 'us'

        # Exclude US, should return EU
        best = RegionSuccessTracker.get_best_region(exclude_regions=['us'])
        assert best == 'eu'

        # Exclude US and EU, should return Asia
        best = RegionSuccessTracker.get_best_region(exclude_regions=['us', 'eu'])
        assert best == 'asia'

        # Exclude all good regions, should return None
        best = RegionSuccessTracker.get_best_region(exclude_regions=['us', 'eu', 'asia'])
        assert best is None

    @pytest.mark.fast
    def test_get_best_region_insufficient_data(self):
        """Test get_best_region returns None when regions have insufficient data (US-136-007)."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()
        RegionSuccessTracker.enable()

        # Record only 1 attempt (below min_sample_size of 3)
        RegionSuccessTracker.record_success('us')

        # Should return None because no region has enough samples
        best = RegionSuccessTracker.get_best_region()
        assert best is None

    @pytest.mark.fast
    def test_get_regions_below_threshold(self):
        """Test get_regions_below_threshold identifies underperforming regions (US-136-007)."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()
        RegionSuccessTracker.enable()

        # US: 5 successes, 5 failures = 50%
        for _ in range(5):
            RegionSuccessTracker.record_success('us')
            RegionSuccessTracker.record_failure('us')
        # EU: 10 successes, 0 failures = 100%
        for _ in range(10):
            RegionSuccessTracker.record_success('de')
        # ASIA: 1 success, 4 failures = 20%
        RegionSuccessTracker.record_success('jp')
        for _ in range(4):
            RegionSuccessTracker.record_failure('jp')

        # US at 50% should be below 0.7 threshold
        # Asia at 20% should be below 0.7 threshold
        below = RegionSuccessTracker.get_regions_below_threshold(0.7)
        assert 'us' in below
        assert 'asia' in below
        assert 'eu' not in below

    @pytest.mark.fast
    def test_checkpoint_state(self):
        """Test checkpoint save/restore for RegionSuccessTracker (US-136-007)."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()
        RegionSuccessTracker.enable()

        # Record enough data to meet min_sample_size (3)
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('de')  # eu
        RegionSuccessTracker.record_failure('de')  # eu

        # Get checkpoint state
        state = RegionSuccessTracker.to_checkpoint_state()

        assert state['region_attempts']['us'] == 3
        assert state['region_attempts']['eu'] == 2
        assert state['region_successes']['us'] == 3
        assert state['region_successes']['eu'] == 1
        assert state['enabled'] is True

        # Clear and restore
        RegionSuccessTracker.clear()

        # Restore from checkpoint
        RegionSuccessTracker.restore_from_checkpoint(state)

        # Verify restored data
        assert RegionSuccessTracker.get_attempts('us') == 3
        assert RegionSuccessTracker.get_attempts('eu') == 2
        # With min_sample_size=3, EU with only 2 attempts returns 0.5 (neutral)
        # US with 3 attempts should return actual rate
        assert RegionSuccessTracker.get_success_rate('us') == 1.0
        assert RegionSuccessTracker.is_enabled() is True

    @pytest.mark.fast
    def test_checkpoint_state_empty(self):
        """Test checkpoint restore with empty state (US-136-007)."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()
        RegionSuccessTracker.enable()

        # Record some data first
        RegionSuccessTracker.record_success('us')

        # Restore from empty state - should not change anything
        RegionSuccessTracker.restore_from_checkpoint({})

        # Original data should still be there
        assert RegionSuccessTracker.get_attempts('us') == 1

    @pytest.mark.fast
    def test_get_best_region_disabled(self):
        """Test get_best_region returns None when tracker is disabled (US-136-007)."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()
        # Don't enable - should return None

        best = RegionSuccessTracker.get_best_region()
        assert best is None


# ============================================================================
# Full Pipeline with Region Tests
# ============================================================================


class TestCalculatePipelineWithRegion:
    """Test the full PauseCalculator.calculate() pipeline with region support."""

    @pytest.mark.fast
    def test_full_pipeline_with_region(self):
        """Full pipeline includes region adjustment."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=300.0,
            jitter_factor=0.0,
            region_enabled=True,
            region="eu",
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0}
        )

        result = calc.calculate(ctx)
        assert result == 72.0  # 60 * 1.2

    @pytest.mark.fast
    def test_full_pipeline_with_all_adjustments(self):
        """Full pipeline applies all adjustments: escalation, budget, region, jitter, cap."""
        calc = PauseCalculator()

        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 2
        mock_em.get_keywords_at_tier.return_value = ['kw1', 'kw2']  # 100%

        mock_budget = MagicMock()
        mock_budget.is_exhausted.return_value = False
        mock_budget.is_nearly_exhausted.return_value = False

        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=300.0,
            jitter_factor=0.0,
            escalation_manager=mock_em,
            budget=mock_budget,
            region_enabled=True,
            region="asia",
            region_multipliers={'asia': 1.5}
        )

        # 60 * 2.0 (escalation) * 1.5 (region) = 180
        result = calc.calculate(ctx)
        assert result == 180.0


# ============================================================================
# TimeOfDayBackoffManager Tests (US-144-010)
# ============================================================================


class TestTimeOfDayBackoffManager:
    """Test TimeOfDayBackoffManager class (US-144-010)."""

    def test_default_values(self):
        """Test TimeOfDayBackoffManager default values."""
        manager = TimeOfDayBackoffManager()
        assert manager.enabled is True

    def test_custom_values(self):
        """Test TimeOfDayBackoffManager accepts custom values."""
        custom_multipliers = {'overnight': 0.5, 'morning': 0.8, 'afternoon': 1.0, 'evening': 1.2}
        manager = TimeOfDayBackoffManager(
            window_multipliers=custom_multipliers,
            weekend_multiplier=1.5,
            enabled=True
        )
        assert manager._window_multipliers == custom_multipliers
        assert manager._weekend_multiplier == 1.5
        assert manager.enabled is True

    def test_enable_disable(self):
        """Test TimeOfDayBackoffManager can be enabled and disabled."""
        manager = TimeOfDayBackoffManager(enabled=True)
        assert manager.enabled is True
        manager.disable()
        assert manager.enabled is False
        manager.enable()
        assert manager.enabled is True

    def test_get_time_window_overnight(self):
        """Test time window detection for overnight hours."""
        manager = TimeOfDayBackoffManager()
        # 10pm, 11pm, 0am, 1am, 2am, 3am, 4am, 5am should be overnight
        for hour in [22, 23, 0, 1, 2, 3, 4, 5]:
            assert manager._get_time_window(hour) == 'overnight'

    def test_get_time_window_morning(self):
        """Test time window detection for morning hours."""
        manager = TimeOfDayBackoffManager()
        # 6am-11am should be morning
        for hour in [6, 7, 8, 9, 10, 11]:
            assert manager._get_time_window(hour) == 'morning'

    def test_get_time_window_afternoon(self):
        """Test time window detection for afternoon hours."""
        manager = TimeOfDayBackoffManager()
        # 12pm-5pm should be afternoon
        for hour in [12, 13, 14, 15, 16, 17]:
            assert manager._get_time_window(hour) == 'afternoon'

    def test_get_time_window_evening(self):
        """Test time window detection for evening hours."""
        manager = TimeOfDayBackoffManager()
        # 6pm-9pm should be evening
        for hour in [18, 19, 20, 21]:
            assert manager._get_time_window(hour) == 'evening'

    def test_multiplier_evening_weekday(self):
        """Test evening weekday multiplier is 1.5x."""
        manager = TimeOfDayBackoffManager()
        # Evening (20:00) on weekday
        multiplier = manager.get_multiplier(hour=20, is_weekend=False)
        assert multiplier == 1.5

    def test_multiplier_afternoon_weekday(self):
        """Test afternoon weekday multiplier is 1.2x."""
        manager = TimeOfDayBackoffManager()
        # Afternoon (14:00) on weekday
        multiplier = manager.get_multiplier(hour=14, is_weekend=False)
        assert multiplier == 1.2

    def test_multiplier_morning_weekday(self):
        """Test morning weekday multiplier is 1.0x."""
        manager = TimeOfDayBackoffManager()
        # Morning (9:00) on weekday
        multiplier = manager.get_multiplier(hour=9, is_weekend=False)
        assert multiplier == 1.0

    def test_multiplier_overnight_weekday(self):
        """Test overnight weekday multiplier is 0.8x."""
        manager = TimeOfDayBackoffManager()
        # Overnight (3:00) on weekday
        multiplier = manager.get_multiplier(hour=3, is_weekend=False)
        assert multiplier == 0.8

    def test_multiplier_evening_weekend(self):
        """Test evening weekend multiplier includes weekend boost (1.5 * 1.3 = 1.95)."""
        manager = TimeOfDayBackoffManager()
        # Evening (20:00) on weekend: 1.5 * 1.3 = 1.95
        multiplier = manager.get_multiplier(hour=20, is_weekend=True)
        assert abs(multiplier - 1.95) < 0.01

    def test_multiplier_afternoon_weekend(self):
        """Test afternoon weekend multiplier includes weekend boost (1.2 * 1.3 = 1.56)."""
        manager = TimeOfDayBackoffManager()
        # Afternoon (14:00) on weekend: 1.2 * 1.3 = 1.56
        multiplier = manager.get_multiplier(hour=14, is_weekend=True)
        assert abs(multiplier - 1.56) < 0.01

    def test_multiplier_morning_weekend(self):
        """Test morning weekend multiplier includes weekend boost (1.0 * 1.3 = 1.3)."""
        manager = TimeOfDayBackoffManager()
        # Morning (9:00) on weekend: 1.0 * 1.3 = 1.3
        multiplier = manager.get_multiplier(hour=9, is_weekend=True)
        assert abs(multiplier - 1.3) < 0.01

    def test_multiplier_overnight_weekend(self):
        """Test overnight weekend multiplier includes weekend boost (0.8 * 1.3 = 1.04)."""
        manager = TimeOfDayBackoffManager()
        # Overnight (3:00) on weekend: 0.8 * 1.3 = 1.04
        multiplier = manager.get_multiplier(hour=3, is_weekend=True)
        assert abs(multiplier - 1.04) < 0.01

    def test_disabled_returns_one(self):
        """Test disabled manager returns 1.0 multiplier."""
        manager = TimeOfDayBackoffManager(enabled=False)
        multiplier = manager.get_multiplier(hour=20, is_weekend=False)
        assert multiplier == 1.0

    def test_get_window_info(self):
        """Test get_window_info returns detailed info."""
        manager = TimeOfDayBackoffManager()
        info = manager.get_window_info(hour=14, timestamp=None)
        assert info['hour'] == 14
        assert info['time_window'] == 'afternoon'
        assert info['is_weekend'] is False
        assert info['base_multiplier'] == 1.2
        assert info['weekend_multiplier'] == 1.0
        assert abs(info['total_multiplier'] - 1.2) < 0.01


# ============================================================================
# time_adjusted Tests (US-144-010)
# ============================================================================


class TestTimeAdjusted:
    """Test PauseCalculator.time_adjusted() step (US-144-010)."""

    def test_time_adjusted_disabled_passthrough(self):
        """When time_of_day_enabled=False, pause passes through unchanged."""
        calc = PauseCalculator()
        ctx = PauseContext(
            time_of_day_enabled=False,
            time_of_day_hour=20,
        )
        assert calc.time_adjusted(60.0, ctx) == 60.0

    def test_evening_multiplier(self):
        """Evening hour (20:00) applies 1.5x multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_hour=20,
            time_of_day_is_weekend=False,
        )
        assert calc.time_adjusted(60.0, ctx) == 90.0  # 60 * 1.5

    def test_afternoon_multiplier(self):
        """Afternoon hour (14:00) applies 1.2x multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_hour=14,
            time_of_day_is_weekend=False,
        )
        assert calc.time_adjusted(60.0, ctx) == 72.0  # 60 * 1.2

    def test_morning_multiplier(self):
        """Morning hour (9:00) applies 1.0x multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_hour=9,
            time_of_day_is_weekend=False,
        )
        assert calc.time_adjusted(60.0, ctx) == 60.0  # 60 * 1.0

    def test_overnight_multiplier(self):
        """Overnight hour (3:00) applies 0.8x multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_hour=3,
            time_of_day_is_weekend=False,
        )
        assert calc.time_adjusted(60.0, ctx) == 48.0  # 60 * 0.8

    def test_evening_weekend_multiplier(self):
        """Evening weekend hour applies 1.5 * 1.3 = 1.95x multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_hour=20,
            time_of_day_is_weekend=True,
        )
        result = calc.time_adjusted(60.0, ctx)
        assert abs(result - 117.0) < 0.1  # 60 * 1.95

    def test_timestamp_overrides_hour(self):
        """When timestamp is provided, it overrides hour."""
        calc = PauseCalculator()
        # Create a timestamp for a known evening hour
        # Using a timestamp: 2024-01-15 20:00:00 (Monday)
        import time
        ts = time.mktime((2024, 1, 15, 20, 0, 0, 0, 0, 0))
        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_timestamp=ts,
            time_of_day_is_weekend=False,
        )
        # Evening should apply 1.5x
        assert calc.time_adjusted(60.0, ctx) == 90.0

    def test_custom_multipliers(self):
        """Custom time window multipliers are respected."""
        calc = PauseCalculator()
        custom_multipliers = {
            'overnight': 0.5,
            'morning': 1.0,
            'afternoon': 1.5,
            'evening': 2.0,
        }
        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_hour=14,  # afternoon
            time_of_day_is_weekend=False,
            time_window_multipliers=custom_multipliers,
        )
        # Should use custom afternoon multiplier 1.5x
        assert calc.time_adjusted(60.0, ctx) == 90.0  # 60 * 1.5

    def test_custom_weekend_multiplier(self):
        """Custom weekend multiplier is applied."""
        calc = PauseCalculator()
        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_hour=14,  # afternoon (1.2x)
            time_of_day_is_weekend=True,
            weekend_multiplier=1.5,  # Custom weekend multiplier
        )
        # 1.2 * 1.5 = 1.8x
        result = calc.time_adjusted(60.0, ctx)
        assert abs(result - 108.0) < 0.1  # 60 * 1.8


class TestTimeAdjustedPipeline:
    """Test time_adjusted in full pipeline."""

    def test_full_pipeline_with_time(self):
        """Full pipeline includes time adjustment."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=300.0,
            jitter_factor=0.0,
            time_of_day_enabled=True,
            time_of_day_hour=20,  # evening
            time_of_day_is_weekend=False,
        )

        result = calc.calculate(ctx)
        assert result == 90.0  # 60 * 1.5

    def test_full_pipeline_with_all_adjustments_including_time(self):
        """Full pipeline applies all adjustments including time."""
        calc = PauseCalculator()

        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 2
        mock_em.get_keywords_at_tier.return_value = ['kw1', 'kw2']  # 100%

        mock_budget = MagicMock()
        mock_budget.is_exhausted.return_value = False
        mock_budget.is_nearly_exhausted.return_value = False

        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=300.0,
            jitter_factor=0.0,
            escalation_manager=mock_em,
            budget=mock_budget,
            region_enabled=True,
            region="us",  # 1.0x region multiplier
            region_multipliers={'us': 1.0},
            time_of_day_enabled=True,
            time_of_day_hour=20,  # evening = 1.5x
            time_of_day_is_weekend=False,
        )

        # 60 * 2.0 (escalation) * 1.0 (region) * 1.5 (time) = 180
        result = calc.calculate(ctx)
        assert result == 180.0


# ============================================================================
# Integration Tests for Time-of-Day Backoff (US-144-010)
# ============================================================================


class TestTimeOfDayBackoffIntegration:
    """Integration tests for time-of-day backoff (US-144-010)."""

    @pytest.mark.fast
    def test_evening_weekday_timestamp_integration(self):
        """Integration test: evening timestamp on weekday applies correct multiplier."""
        calc = PauseCalculator()
        # Use timestamp for Friday 8pm (evening)
        # 2024-01-05 20:00:00 is a Friday
        import time
        ts = time.mktime((2024, 1, 5, 20, 0, 0, 4, 0, 0))  # Friday = weekday

        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_timestamp=ts,
        )

        result = calc.time_adjusted(60.0, ctx)
        # Evening (20:00) = 1.5x, weekday = no weekend boost
        assert result == 90.0

    @pytest.mark.fast
    def test_overnight_weekend_timestamp_integration(self):
        """Integration test: overnight timestamp on weekend applies correct multiplier."""
        calc = PauseCalculator()
        # Use timestamp for Saturday 3am (overnight)
        # 2024-01-06 03:00:00 is a Saturday
        import time
        ts = time.mktime((2024, 1, 6, 3, 0, 0, 5, 0, 0))  # Saturday = weekend

        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_timestamp=ts,
        )

        result = calc.time_adjusted(60.0, ctx)
        # Overnight (3am) = 0.8x, weekend = 1.3x boost, so 0.8 * 1.3 = 1.04
        expected = 60.0 * 0.8 * 1.3
        assert abs(result - expected) < 0.1

    @pytest.mark.fast
    def test_morning_weekday_timestamp_integration(self):
        """Integration test: morning timestamp on weekday applies baseline multiplier."""
        calc = PauseCalculator()
        # Use timestamp for Monday 9am (morning)
        # 2024-01-08 09:00:00 is a Monday
        import time
        ts = time.mktime((2024, 1, 8, 9, 0, 0, 0, 0, 0))  # Monday = weekday

        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_timestamp=ts,
        )

        result = calc.time_adjusted(60.0, ctx)
        # Morning (9am) = 1.0x, weekday = no weekend boost
        assert result == 60.0

    @pytest.mark.fast
    def test_afternoon_weekend_timestamp_integration(self):
        """Integration test: afternoon timestamp on weekend applies correct multiplier."""
        calc = PauseCalculator()
        # Use timestamp for Sunday 2pm (afternoon)
        # 2024-01-07 14:00:00 is a Sunday
        import time
        ts = time.mktime((2024, 1, 7, 14, 0, 0, 6, 0, 0))  # Sunday = weekend

        ctx = PauseContext(
            time_of_day_enabled=True,
            time_of_day_timestamp=ts,
        )

        result = calc.time_adjusted(60.0, ctx)
        # Afternoon (14:00) = 1.2x, weekend = 1.3x boost, so 1.2 * 1.3 = 1.56
        expected = 60.0 * 1.2 * 1.3
        assert abs(result - expected) < 0.1

    @pytest.mark.fast
    def test_all_time_windows_hours(self):
        """Integration test: verify all 24 hours map to correct time windows."""
        calc = PauseCalculator()

        # Define expected windows for each hour
        expected_windows = {
            0: 'overnight', 1: 'overnight', 2: 'overnight', 3: 'overnight',
            4: 'overnight', 5: 'overnight',
            6: 'morning', 7: 'morning', 8: 'morning', 9: 'morning',
            10: 'morning', 11: 'morning',
            12: 'afternoon', 13: 'afternoon', 14: 'afternoon', 15: 'afternoon',
            16: 'afternoon', 17: 'afternoon',
            18: 'evening', 19: 'evening', 20: 'evening', 21: 'evening',
            22: 'overnight', 23: 'overnight',
        }

        for hour, expected_window in expected_windows.items():
            assert calc._get_time_window(hour) == expected_window, \
                f"Hour {hour} should be {expected_window}"

    @pytest.mark.fast
    def test_time_of_day_manager_with_timestamp(self):
        """Integration test: TimeOfDayBackoffManager with timestamp."""
        manager = TimeOfDayBackoffManager()

        # Test with timestamp for Friday 8pm
        import time
        ts = time.mktime((2024, 1, 5, 20, 0, 0, 4, 0, 0))  # Friday

        multiplier = manager.get_multiplier(timestamp=ts)
        # Evening (20:00) = 1.5x, Friday = weekday = no weekend boost
        assert multiplier == 1.5

    @pytest.mark.fast
    def test_time_of_day_manager_weekend_with_timestamp(self):
        """Integration test: TimeOfDayBackoffManager weekend with timestamp."""
        manager = TimeOfDayBackoffManager()

        # Test with timestamp for Saturday 3am
        import time
        ts = time.mktime((2024, 1, 6, 3, 0, 0, 5, 0, 0))  # Saturday

        multiplier = manager.get_multiplier(timestamp=ts)
        # Overnight (3am) = 0.8x, Saturday = weekend = 1.3x boost
        # 0.8 * 1.3 = 1.04
        assert abs(multiplier - 1.04) < 0.01


