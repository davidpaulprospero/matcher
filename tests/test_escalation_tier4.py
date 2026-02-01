"""Integration tests for Tier 4 (VPN rotation) escalation flow.

These tests verify that escalation properly reaches Tier 4 after cookie
exhaustion and that VPN rotation integrates correctly with the escalation system.
"""

import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.escalation_manager import EscalationManager, EscalationResult
from src.downloader.types import EscalationTier
from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.downloader.rate_limit_budget import RateLimitBudget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeExtractorArgsConfig:
    """Minimal stand-in for ExtractorArgsConfig with Tier 4 support."""
    enabled: bool = True
    player_clients: List[str] = field(
        default_factory=lambda: ["web_safari", "tv_downgraded", "web"]
    )
    escalation_threshold: int = 2
    cooldown_seconds: float = 0.0  # No cooldown for tests
    max_tier: int = 4  # Allow Tier 4 (VPN_ROTATION)


def _make_impersonation_manager():
    """Create a mock ImpersonationManager that returns deterministic args."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", "Chrome-136:Macos-15"
    ]
    return mgr


@pytest.fixture
def ext_config_tier4():
    """Config that allows escalation to Tier 4."""
    return FakeExtractorArgsConfig()


@pytest.fixture
def imp_manager():
    return _make_impersonation_manager()


@pytest.fixture
def manager_tier4(imp_manager, ext_config_tier4):
    """EscalationManager that can reach Tier 4."""
    mgr = EscalationManager(imp_manager, ext_config_tier4)
    # Add mock Mullvad VPN
    mock_mullvad = MagicMock()
    mock_mullvad.can_switch.return_value = True
    mgr.set_mullvad_vpn(mock_mullvad)
    return mgr


# ---------------------------------------------------------------------------
# Tier 4 Escalation Tests
# ---------------------------------------------------------------------------

class TestEscalationReachesTier4:
    """Tests for escalation reaching Tier 4 after cookie exhaustion."""

    def test_escalation_reaches_tier4_after_cookie_exhaustion(self, manager_tier4):
        """Test that escalation advances to Tier 4 when Tier 3 cookies are exhausted."""
        keyword = "test_keyword"

        # Record failures to escalate through tiers
        # With threshold=2, cooldown=0:
        # - 2 failures -> Tier 2
        # - 2 more failures -> Tier 3
        # - 2 more failures -> Tier 4
        for _ in range(6):
            manager_tier4.record_failure(keyword, "HTTP Error 403: Forbidden")

        # Get escalation result
        result = manager_tier4.get_escalation_args(keyword)

        # Should be at Tier 4 (VPN_ROTATION)
        assert result.tier == EscalationTier.VPN_ROTATION, f"Expected Tier 4, got {result.tier}"
        assert result.rotate_vpn is True, "Expected rotate_vpn=True at Tier 4"

    def test_tier4_has_rotate_vpn_flag(self, manager_tier4):
        """Test that Tier 4 sets rotate_vpn=True in EscalationResult."""
        keyword = "vpn_flag_test"

        # Escalate to Tier 4
        for _ in range(6):
            manager_tier4.record_failure(keyword, "HTTP Error 403: Forbidden")

        result = manager_tier4.get_escalation_args(keyword)

        # rotate_vpn should be True at Tier 4
        assert result.rotate_vpn is True


class TestVPNRotationResetsCircuitBreaker:
    """Tests for circuit breaker reset after VPN rotation."""

    def test_vpn_rotation_resets_circuit_breaker(self):
        """Test that circuit breaker resets after successful VPN rotation."""
        # Create circuit breaker with correct parameters
        config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3,
            pause_seconds=60,
        )
        cb = CircuitBreaker(config)

        # Trip the circuit breaker
        for _ in range(5):
            cb.record_failure()

        # Circuit breaker should be open
        assert cb.is_open, "Circuit breaker should be open after failures"

        # Reset (simulates what happens after VPN rotation)
        cb.reset()

        # Circuit breaker should be closed
        assert not cb.is_open, "Circuit breaker should be closed after reset"
        assert cb.state.consecutive_failures == 0, "Failure count should be 0 after reset"


class TestVPNRotationRecordedInBudget:
    """Tests for VPN rotation budget tracking."""

    def test_vpn_rotation_recorded_in_budget(self):
        """Test that VPN rotation is tracked in rate limit budget."""
        # Create budget with VPN limits
        budget = RateLimitBudget(max_vpn_switches=5)

        # Initial state
        assert budget.vpn_switches_used == 0

        # Record VPN rotation
        budget.record_vpn_rotation()

        # Should be tracked
        assert budget.vpn_switches_used == 1

        # Record another
        budget.record_vpn_rotation()
        assert budget.vpn_switches_used == 2

    def test_can_rotate_vpn_respects_budget(self):
        """Test that can_rotate_vpn() respects the budget limit."""
        # Create budget with limit of 2 VPN switches
        budget = RateLimitBudget(max_vpn_switches=2)

        # Should be able to rotate initially
        assert budget.can_rotate_vpn() is True

        # Use one rotation
        budget.record_vpn_rotation()
        assert budget.can_rotate_vpn() is True

        # Use second rotation
        budget.record_vpn_rotation()
        assert budget.can_rotate_vpn() is False  # Budget exhausted


class TestEscalationMetricsIncludeTier4:
    """Tests for Tier 4 in escalation metrics."""

    def test_escalation_metrics_include_tier4_keywords(self, manager_tier4):
        """Test that get_metrics() includes Tier 4 keywords."""
        keyword = "metrics_test"

        # Escalate to Tier 4
        for _ in range(6):
            manager_tier4.record_failure(keyword, "HTTP Error 403: Forbidden")

        # Get metrics
        metrics = manager_tier4.get_metrics()

        # Should have keywords_at_each_tier
        assert "keywords_at_each_tier" in metrics
        # Keyword should be at Tier 4
        tier4_name = EscalationTier.VPN_ROTATION.name
        assert tier4_name in metrics["keywords_at_each_tier"]
        assert keyword in metrics["keywords_at_each_tier"][tier4_name]


class TestTier4WithBudgetIntegration:
    """Tests for Tier 4 with budget integration."""

    def test_tier4_with_budget_tracks_vpn_rotations(self, imp_manager, ext_config_tier4):
        """Test that Tier 4 escalation with budget tracks VPN rotations."""
        # Create budget
        budget = RateLimitBudget(max_vpn_switches=5)

        # Create escalation manager with budget
        esc_mgr = EscalationManager(
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config_tier4,
            budget=budget,
        )

        # Mock Mullvad VPN
        mock_mullvad = MagicMock()
        mock_mullvad.can_switch.return_value = True
        esc_mgr.set_mullvad_vpn(mock_mullvad)

        keyword = "budget_test"

        # Escalate to Tier 4
        for _ in range(6):
            esc_mgr.record_failure(keyword, "HTTP Error 403: Forbidden")

        # Get escalation args (this should be at Tier 4)
        result = esc_mgr.get_escalation_args(keyword)
        assert result.tier == EscalationTier.VPN_ROTATION

        # When core.py processes rotate_vpn=True, it calls:
        # budget.record_vpn_rotation()
        # Simulate that
        budget.record_vpn_rotation()
        assert budget.vpn_switches_used == 1


class TestResetOnIPChange:
    """Tests for budget reset after VPN rotation (IP change)."""

    def test_reset_on_ip_change_resets_rotation_budgets(self):
        """Test that reset_on_ip_change() resets cookie rotation and backoff budgets."""
        budget = RateLimitBudget(
            max_rotations=10,
            max_backoff_time=300.0,
            max_vpn_switches=5,
        )

        # Use some of the cookie rotation budget
        budget.rotations_used = 5
        budget.backoff_time_spent = 100.0

        # Use one VPN switch
        budget.record_vpn_rotation()
        assert budget.vpn_switches_used == 1

        # Reset on IP change (VPN rotation successful)
        budget.reset_on_ip_change()

        # Cookie rotations and backoff should be reset
        assert budget.rotations_used == 0
        assert budget.backoff_time_spent == 0.0

        # VPN switches should be preserved (we just used one)
        assert budget.vpn_switches_used == 1
