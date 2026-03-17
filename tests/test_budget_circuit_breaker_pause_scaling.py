"""Tests for budget-aware circuit breaker pause scaling (Sprint 12 US-004).

Tests cover:
1. is_nearly_exhausted() on RateLimitBudget (>80% of any resource)
2. Circuit breaker pause scaling: 1.5x nearly exhausted, 2.5x exhausted, 1.0x healthy
3. max_pause_seconds cap on CircuitBreakerConfig
4. Logging of pause extension reasons
5. Budget + escalation combined scaling
6. Edge cases (no budget, unlimited resources)

Created: January 2026
User Story: US-004 - Add budget-aware circuit breaker pause scaling
"""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from src.downloader.rate_limit_budget import RateLimitBudget


# ============================================================================
# Test is_nearly_exhausted() on RateLimitBudget
# ============================================================================


class TestIsNearlyExhausted:
    """Test is_nearly_exhausted() method on RateLimitBudget."""

    @pytest.mark.fast
    def test_healthy_budget_not_nearly_exhausted(self):
        """Budget with low usage is not nearly exhausted."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 2
        budget.vpn_switches_used = 1
        budget.backoff_time_spent = 50.0

        assert budget.is_nearly_exhausted() is False

    @pytest.mark.fast
    def test_rotations_above_80_percent(self):
        """Nearly exhausted when rotations exceed 80%."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 9  # 90% > 80%

        assert budget.is_nearly_exhausted() is True

    @pytest.mark.fast
    def test_vpn_switches_above_80_percent(self):
        """Nearly exhausted when VPN switches exceed 80%."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.vpn_switches_used = 5  # 100% > 80%

        assert budget.is_nearly_exhausted() is True

    @pytest.mark.fast
    def test_backoff_time_above_80_percent(self):
        """Nearly exhausted when backoff time exceeds 80%."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.backoff_time_spent = 250.0  # 83% > 80%

        assert budget.is_nearly_exhausted() is True

    @pytest.mark.fast
    def test_exactly_at_80_percent_not_nearly_exhausted(self):
        """At exactly 80%, not nearly exhausted (threshold is >80%)."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.rotations_used = 8  # Exactly 80%

        assert budget.is_nearly_exhausted() is False

    @pytest.mark.fast
    def test_just_above_80_percent(self):
        """Just above 80% is nearly exhausted."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.rotations_used = 9  # 90% > 80%

        assert budget.is_nearly_exhausted() is True

    @pytest.mark.fast
    def test_unlimited_resources_not_nearly_exhausted(self):
        """Unlimited resources (0 max) are never nearly exhausted."""
        budget = RateLimitBudget()
        budget.max_rotations = 0  # unlimited
        budget.max_vpn_switches = 0  # unlimited
        budget.max_backoff_time = 0  # unlimited
        budget.rotations_used = 100
        budget.vpn_switches_used = 100
        budget.backoff_time_spent = 10000.0

        assert budget.is_nearly_exhausted() is False

    @pytest.mark.fast
    def test_mixed_some_unlimited_some_high(self):
        """Some unlimited, some high-usage: nearly exhausted if any limited resource >80%."""
        budget = RateLimitBudget()
        budget.max_rotations = 0  # unlimited
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.vpn_switches_used = 5  # 100%

        assert budget.is_nearly_exhausted() is True

    @pytest.mark.fast
    def test_fresh_budget_not_nearly_exhausted(self):
        """Fresh budget with no usage is not nearly exhausted."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0

        assert budget.is_nearly_exhausted() is False


# ============================================================================
# Test Budget-Aware Pause Scaling
# ============================================================================


class TestBudgetAwarePauseScaling:
    """Test circuit breaker pause scaling based on budget state."""

    def _make_breaker(self, pause_seconds=60.0, max_pause_seconds=300.0):
        """Create a circuit breaker with budget support.

        Uses jitter_factor=0.0 so tests can assert exact pause values
        without jitter randomness (these tests focus on budget/escalation scaling).
        """
        config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3,
            pause_seconds=pause_seconds,
            max_pause_seconds=max_pause_seconds,
            jitter_factor=0.0,
        )
        return CircuitBreaker(config)

    def _make_healthy_budget(self):
        """Create a budget with low usage."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 2
        budget.vpn_switches_used = 1
        budget.backoff_time_spent = 50.0
        return budget

    def _make_nearly_exhausted_budget(self):
        """Create a budget with >80% usage on rotations."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 9  # 90% > 80%
        budget.vpn_switches_used = 1
        budget.backoff_time_spent = 50.0
        return budget

    def _make_exhausted_budget(self):
        """Create a fully exhausted budget."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 10
        budget.vpn_switches_used = 5
        budget.backoff_time_spent = 300.0
        return budget

    @pytest.mark.fast
    def test_healthy_budget_no_extension(self):
        """Healthy budget: 1.0x pause (no extension)."""
        breaker = self._make_breaker(pause_seconds=60.0)
        breaker._budget = self._make_healthy_budget()

        effective = breaker._get_effective_pause_seconds()
        assert effective == 60.0

    @pytest.mark.fast
    def test_nearly_exhausted_1_5x_extension(self):
        """Nearly exhausted budget: 1.5x pause extension."""
        breaker = self._make_breaker(pause_seconds=60.0)
        breaker._budget = self._make_nearly_exhausted_budget()

        effective = breaker._get_effective_pause_seconds()
        assert effective == 90.0  # 60 * 1.5

    @pytest.mark.fast
    def test_exhausted_2_5x_extension(self):
        """Fully exhausted budget: 2.5x pause extension."""
        breaker = self._make_breaker(pause_seconds=60.0)
        breaker._budget = self._make_exhausted_budget()

        effective = breaker._get_effective_pause_seconds()
        assert effective == 150.0  # 60 * 2.5

    @pytest.mark.fast
    def test_no_budget_no_extension(self):
        """Without budget linked, no extension applied."""
        breaker = self._make_breaker(pause_seconds=60.0)
        # No _budget set

        effective = breaker._get_effective_pause_seconds()
        assert effective == 60.0


# ============================================================================
# Test max_pause_seconds Cap
# ============================================================================


class TestMaxPauseSecondsCap:
    """Test that pause extension is capped at max_pause_seconds."""

    @pytest.mark.fast
    def test_cap_applied_with_exhausted_budget(self):
        """Exhausted budget extension is capped at max_pause_seconds."""
        config = CircuitBreakerConfig(
            pause_seconds=200.0,
            max_pause_seconds=300.0,
            jitter_factor=0.0,
        )
        breaker = CircuitBreaker(config)

        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 10
        budget.vpn_switches_used = 5
        budget.backoff_time_spent = 300.0
        breaker._budget = budget

        # 200 * 2.5 = 500, but capped at 300
        effective = breaker._get_effective_pause_seconds()
        assert effective == 300.0

    @pytest.mark.fast
    def test_cap_not_applied_when_under(self):
        """No capping when extension is under max_pause_seconds."""
        config = CircuitBreakerConfig(
            pause_seconds=60.0,
            max_pause_seconds=300.0,
            jitter_factor=0.0,
        )
        breaker = CircuitBreaker(config)

        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 10
        budget.vpn_switches_used = 5
        budget.backoff_time_spent = 300.0
        breaker._budget = budget

        # 60 * 2.5 = 150, under 300 cap
        effective = breaker._get_effective_pause_seconds()
        assert effective == 150.0

    @pytest.mark.fast
    def test_default_max_pause_seconds_is_300(self):
        """Default max_pause_seconds is 300."""
        config = CircuitBreakerConfig()
        assert config.max_pause_seconds == 300.0

    @pytest.mark.fast
    def test_custom_max_pause_seconds(self):
        """Custom max_pause_seconds is respected."""
        config = CircuitBreakerConfig(
            pause_seconds=100.0,
            max_pause_seconds=120.0,
        )
        breaker = CircuitBreaker(config)

        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.rotations_used = 9  # nearly exhausted
        breaker._budget = budget

        # 100 * 1.5 = 150, capped at 120
        effective = breaker._get_effective_pause_seconds()
        assert effective == 120.0


# ============================================================================
# Test Logging of Pause Extension
# ============================================================================


class TestPauseExtensionLogging:
    """Test that pause extension is logged with reason."""

    @pytest.mark.fast
    def test_nearly_exhausted_logged(self, caplog):
        """Nearly exhausted extension logs the reason."""
        config = CircuitBreakerConfig(pause_seconds=60.0)
        breaker = CircuitBreaker(config)

        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.rotations_used = 9  # 90%
        breaker._budget = budget

        with caplog.at_level(logging.INFO, logger='src.downloader.circuit_breaker'):
            breaker._get_effective_pause_seconds()

        assert any(
            "Circuit breaker pause extended" in r.message
            and "nearly exhausted" in r.message
            for r in caplog.records
        )

    @pytest.mark.fast
    def test_exhausted_logged(self, caplog):
        """Exhausted extension logs the reason."""
        config = CircuitBreakerConfig(pause_seconds=60.0)
        breaker = CircuitBreaker(config)

        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 10
        budget.vpn_switches_used = 5
        budget.backoff_time_spent = 300.0
        breaker._budget = budget

        with caplog.at_level(logging.INFO, logger='src.downloader.circuit_breaker'):
            breaker._get_effective_pause_seconds()

        assert any(
            "Circuit breaker pause extended" in r.message
            and "exhausted" in r.message
            for r in caplog.records
        )

    @pytest.mark.fast
    def test_healthy_budget_no_log(self, caplog):
        """Healthy budget does not log extension."""
        config = CircuitBreakerConfig(pause_seconds=60.0)
        breaker = CircuitBreaker(config)

        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.rotations_used = 2
        breaker._budget = budget

        with caplog.at_level(logging.INFO, logger='src.downloader.circuit_breaker'):
            breaker._get_effective_pause_seconds()

        budget_logs = [
            r for r in caplog.records
            if "budget" in r.message.lower() and "extended" in r.message.lower()
        ]
        assert len(budget_logs) == 0

    @pytest.mark.fast
    def test_log_format_includes_original_and_extended(self, caplog):
        """Log message includes original and extended pause values."""
        config = CircuitBreakerConfig(pause_seconds=60.0)
        breaker = CircuitBreaker(config)

        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.rotations_used = 9
        breaker._budget = budget

        with caplog.at_level(logging.INFO, logger='src.downloader.circuit_breaker'):
            breaker._get_effective_pause_seconds()

        extension_logs = [
            r for r in caplog.records
            if "Circuit breaker pause extended" in r.message
        ]
        assert len(extension_logs) == 1
        msg = extension_logs[0].message
        assert "60s" in msg
        assert "90s" in msg


# ============================================================================
# Test Combined Escalation + Budget Scaling
# ============================================================================


class TestCombinedScaling:
    """Test escalation + budget scaling compose correctly."""

    @pytest.mark.fast
    def test_escalation_and_budget_compose(self):
        """Both escalation (2x) and budget (1.5x) apply multiplicatively."""
        config = CircuitBreakerConfig(
            pause_seconds=60.0,
            max_pause_seconds=500.0,  # High cap to allow full composition
            jitter_factor=0.0,
        )
        breaker = CircuitBreaker(config)

        # Mock escalation manager with >50% at Tier 3
        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 2
        mock_em.get_keywords_at_tier.return_value = ["kw1", "kw2"]  # 100% at Tier 3
        breaker._escalation_manager = mock_em

        # Nearly exhausted budget
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.rotations_used = 9  # 90%
        breaker._budget = budget

        # 60 * 2.0 (escalation) * 1.5 (budget) = 180
        effective = breaker._get_effective_pause_seconds()
        assert effective == 180.0

    @pytest.mark.fast
    def test_escalation_and_exhausted_budget_compose(self):
        """Escalation (2x) and exhausted budget (2.5x) compose."""
        config = CircuitBreakerConfig(
            pause_seconds=60.0,
            max_pause_seconds=500.0,
            jitter_factor=0.0,
        )
        breaker = CircuitBreaker(config)

        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 2
        mock_em.get_keywords_at_tier.return_value = ["kw1", "kw2"]
        breaker._escalation_manager = mock_em

        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 10
        budget.vpn_switches_used = 5
        budget.backoff_time_spent = 300.0
        breaker._budget = budget

        # 60 * 2.0 * 2.5 = 300
        effective = breaker._get_effective_pause_seconds()
        assert effective == 300.0

    @pytest.mark.fast
    def test_combined_scaling_capped(self):
        """Combined scaling is capped at max_pause_seconds."""
        config = CircuitBreakerConfig(
            pause_seconds=60.0,
            max_pause_seconds=200.0,  # Low cap
            jitter_factor=0.0,
        )
        breaker = CircuitBreaker(config)

        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 2
        mock_em.get_keywords_at_tier.return_value = ["kw1", "kw2"]
        breaker._escalation_manager = mock_em

        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 10
        budget.vpn_switches_used = 5
        budget.backoff_time_spent = 300.0
        breaker._budget = budget

        # 60 * 2.0 * 2.5 = 300, capped at 200
        effective = breaker._get_effective_pause_seconds()
        assert effective == 200.0


# ============================================================================
# Test Config Section Integration
# ============================================================================


class TestConfigSectionMaxPauseSeconds:
    """Test max_pause_seconds in config/sections/download.py."""

    @pytest.mark.fast
    def test_config_section_has_max_pause_seconds(self):
        """CircuitBreakerConfig in config/sections/download.py has max_pause_seconds."""
        from src.config.sections.download import CircuitBreakerConfig as ConfigCB
        config = ConfigCB()
        assert hasattr(config, 'max_pause_seconds')
        assert config.max_pause_seconds == 300.0

    @pytest.mark.fast
    def test_config_section_custom_max_pause(self):
        """Can set custom max_pause_seconds."""
        from src.config.sections.download import CircuitBreakerConfig as ConfigCB
        config = ConfigCB(max_pause_seconds=500.0)
        assert config.max_pause_seconds == 500.0

    @pytest.mark.fast
    def test_config_yaml_has_max_pause_seconds(self):
        """config.yaml circuit_breaker section has max_pause_seconds."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert 'download' in config
        assert 'circuit_breaker' in config['download']
        assert 'max_pause_seconds' in config['download']['circuit_breaker']
        assert config['download']['circuit_breaker']['max_pause_seconds'] == 300.0


# ============================================================================
# Test _budget Assignment
# ============================================================================


class TestSetBudgetMethod:
    """Test CircuitBreaker._budget assignment."""

    @pytest.mark.fast
    def test_set_budget_stores_reference(self):
        """Assigning _budget stores the budget reference."""
        breaker = CircuitBreaker()
        budget = RateLimitBudget()
        breaker._budget = budget
        assert breaker._budget is budget

    @pytest.mark.fast
    def test_no_budget_by_default(self):
        """Budget is None by default."""
        breaker = CircuitBreaker()
        assert breaker._budget is None

    @pytest.mark.fast
    def test_budget_can_be_replaced(self):
        """Budget can be replaced with a new one."""
        breaker = CircuitBreaker()
        budget1 = RateLimitBudget()
        budget2 = RateLimitBudget()
        breaker._budget = budget1
        assert breaker._budget is budget1
        breaker._budget = budget2
        assert breaker._budget is budget2


# ============================================================================
# Test Composable Pause Strategy (US-57-009)
# ============================================================================


class TestCapPauseDurationConsistency:
    """Test that _cap_pause_duration is consistent between effective pause and jitter paths."""

    @pytest.mark.fast
    def test_same_max_cap_in_effective_pause_and_jitter(self):
        """Both _get_effective_pause_seconds and _apply_jitter use the same max_pause cap."""
        config = CircuitBreakerConfig(
            pause_seconds=200.0,
            max_pause_seconds=250.0,
            jitter_factor=0.0,  # Disable jitter randomness for deterministic test
        )
        breaker = CircuitBreaker(config)

        # Set exhausted budget to push effective pause above cap: 200 * 2.5 = 500
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0
        budget.rotations_used = 10
        budget.vpn_switches_used = 5
        budget.backoff_time_spent = 300.0
        breaker._budget = budget

        effective = breaker._get_effective_pause_seconds()
        assert effective == 250.0  # Capped

        # Jitter path with a value above cap should also cap at same max
        jittered = breaker._apply_jitter(400.0)
        assert jittered == 250.0  # Same cap applied

        # Both use the same _cap_pause_duration method
        assert breaker._cap_pause_duration(999.0) == 250.0

    @pytest.mark.fast
    def test_cap_pause_duration_shared_by_both_paths(self):
        """_cap_pause_duration is the single source of truth for capping."""
        config = CircuitBreakerConfig(
            pause_seconds=100.0,
            max_pause_seconds=150.0,
        )
        breaker = CircuitBreaker(config)

        # Values under cap pass through
        assert breaker._cap_pause_duration(100.0) == 100.0
        assert breaker._cap_pause_duration(150.0) == 150.0

        # Values over cap are capped
        assert breaker._cap_pause_duration(151.0) == 150.0
        assert breaker._cap_pause_duration(1000.0) == 150.0

    @pytest.mark.fast
    def test_cap_applied_identically_with_different_max_values(self):
        """Custom max_pause_seconds is respected by both paths."""
        for max_pause in [50.0, 100.0, 200.0, 500.0]:
            config = CircuitBreakerConfig(
                pause_seconds=10.0,
                max_pause_seconds=max_pause,
                jitter_factor=0.0,
            )
            breaker = CircuitBreaker(config)

            # Both effective pause cap and jitter cap use the same max
            over_max = max_pause + 100.0
            assert breaker._cap_pause_duration(over_max) == max_pause
            assert breaker._apply_jitter(over_max) == max_pause


class TestEscalationProportionalPauseIncrease:
    """Test that escalation tier increases pause duration proportionally."""

    @pytest.mark.fast
    def test_escalation_doubles_base_pause(self):
        """Escalation at >50% Tier 3 doubles the pause from base."""
        config = CircuitBreakerConfig(
            pause_seconds=60.0,
            max_pause_seconds=500.0,
            jitter_factor=0.0,
        )
        breaker = CircuitBreaker(config)

        # No escalation: base pause
        assert breaker._get_effective_pause_seconds() == 60.0

        # Mock escalation manager with >50% at Tier 3
        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 10
        mock_em.get_keywords_at_tier.return_value = ['kw1', 'kw2', 'kw3', 'kw4', 'kw5', 'kw6']
        breaker._escalation_manager = mock_em

        effective = breaker._get_effective_pause_seconds()
        assert effective == 120.0  # 60 * 2.0

        # Proportional: double the base
        base = breaker._base_pause()
        escalated = breaker._escalation_adjusted_pause(base)
        assert escalated == base * 2.0

    @pytest.mark.fast
    def test_escalation_below_threshold_no_increase(self):
        """Escalation at <=50% Tier 3 does not increase pause."""
        config = CircuitBreakerConfig(
            pause_seconds=60.0,
            max_pause_seconds=500.0,
            jitter_factor=0.0,
        )
        breaker = CircuitBreaker(config)

        # Mock escalation manager with exactly 50% at Tier 3 (not >50%)
        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 10
        mock_em.get_keywords_at_tier.return_value = ['kw1', 'kw2', 'kw3', 'kw4', 'kw5']  # 50%
        breaker._escalation_manager = mock_em

        base = breaker._base_pause()
        escalated = breaker._escalation_adjusted_pause(base)
        assert escalated == base  # No increase at exactly 50%

    @pytest.mark.fast
    def test_escalation_proportional_with_different_base_pauses(self):
        """Escalation 2x multiplier is proportional regardless of base pause."""
        mock_em = MagicMock()
        mock_em.get_active_keyword_count.return_value = 4
        mock_em.get_keywords_at_tier.return_value = ['kw1', 'kw2', 'kw3']  # 75% > 50%

        for base_seconds in [30.0, 60.0, 120.0, 200.0]:
            config = CircuitBreakerConfig(
                pause_seconds=base_seconds,
                max_pause_seconds=1000.0,  # High cap to avoid interference
                jitter_factor=0.0,
            )
            breaker = CircuitBreaker(config)
            breaker._escalation_manager = mock_em

            effective = breaker._get_effective_pause_seconds()
            assert effective == base_seconds * 2.0, (
                f"Expected {base_seconds * 2.0} for base={base_seconds}, got {effective}"
            )
