"""Tests for PauseCalculator extracted from CircuitBreaker (US-82-008).

Verifies:
- Each PauseCalculator step can be unit tested individually without CircuitBreaker
- PauseContext carries all needed inputs
- Pipeline order: base → escalation → budget → jitter → cap
- Exact same behavior as the original CircuitBreaker 5-step pipeline
"""

import random
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.pause_calculator import PauseCalculator, PauseContext


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
        """jitter_factor > 1.0 is clamped to 1.0."""
        calc = PauseCalculator()
        ctx = PauseContext(jitter_factor=2.0)

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
