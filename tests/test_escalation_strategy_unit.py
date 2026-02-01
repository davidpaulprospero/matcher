"""
Unit tests for EscalationStrategy module.

Tests the decision logic for tier progression, cooldown checks,
and escalation triggers. [US-36-007]
"""

import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from src.downloader.escalation_strategy import EscalationDecision, EscalationStrategy
from src.downloader.types import EscalationState, EscalationTier


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockExtractorArgsConfig:
    """Mock config for testing EscalationStrategy."""
    escalation_threshold: int = 2
    cooldown_seconds: float = 300.0
    max_tier: int = 3


def make_state(
    tier: EscalationTier = EscalationTier.IMPERSONATE_ONLY,
    consecutive_403s: int = 0,
    last_escalation_time: Optional[float] = None,
) -> EscalationState:
    """Factory for creating EscalationState test fixtures."""
    return EscalationState(
        current_tier=tier,
        consecutive_403s=consecutive_403s,
        last_escalation_time=last_escalation_time,
    )


# ============================================================================
# Strategy Selection Based on Error Type
# ============================================================================


class TestStrategySelectionByErrorType:
    """Test: Strategy selection based on error type is correct."""

    @pytest.mark.fast
    def test_failure_escalation_triggers_on_403_threshold(self):
        """403 errors at threshold trigger escalation."""
        strategy = EscalationStrategy(MockExtractorArgsConfig(escalation_threshold=2))
        state = make_state(consecutive_403s=2)

        decision = strategy.should_escalate_on_failure(state)

        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.EXTRACTOR_ARGS

    @pytest.mark.fast
    def test_slow_speed_escalation_triggers_on_threshold(self):
        """Slow speed signals at threshold trigger escalation."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())
        state = make_state()

        decision = strategy.should_escalate_on_slow_speed(
            state, consecutive_slow_count=3, slow_threshold=3
        )

        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.EXTRACTOR_ARGS

    @pytest.mark.fast
    def test_circuit_breaker_triggers_full_bypass(self):
        """Open circuit breaker triggers shortcut to full bypass."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())

        decision = strategy.should_shortcut_to_max(
            EscalationTier.IMPERSONATE_ONLY, circuit_breaker_open=True
        )

        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.FULL_BYPASS
        assert decision.skip_to_max is True

    @pytest.mark.fast
    def test_closed_circuit_breaker_no_escalation(self):
        """Closed circuit breaker does not trigger escalation."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())

        decision = strategy.should_shortcut_to_max(
            EscalationTier.IMPERSONATE_ONLY, circuit_breaker_open=False
        )

        assert decision.should_escalate is False
        assert "not open" in decision.reason


# ============================================================================
# Escalation Level Progression
# ============================================================================


class TestEscalationLevelProgression:
    """Test: Escalation levels progress correctly (tier1 -> tier2 -> tier3)."""

    @pytest.mark.fast
    def test_tier1_to_tier2_progression(self):
        """Tier 1 escalates to Tier 2."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())

        next_tier = strategy.get_next_tier(EscalationTier.IMPERSONATE_ONLY)

        assert next_tier == EscalationTier.EXTRACTOR_ARGS

    @pytest.mark.fast
    def test_tier2_to_tier3_progression(self):
        """Tier 2 escalates to Tier 3."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())

        next_tier = strategy.get_next_tier(EscalationTier.EXTRACTOR_ARGS)

        assert next_tier == EscalationTier.FULL_BYPASS

    @pytest.mark.fast
    def test_tier3_has_no_next(self):
        """Tier 3 has no next tier."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())

        next_tier = strategy.get_next_tier(EscalationTier.FULL_BYPASS)

        assert next_tier is None

    @pytest.mark.fast
    def test_failure_decision_advances_one_tier(self):
        """Failure decision advances exactly one tier at a time."""
        strategy = EscalationStrategy(MockExtractorArgsConfig(escalation_threshold=2))
        state = make_state(
            tier=EscalationTier.IMPERSONATE_ONLY,
            consecutive_403s=2,
        )

        decision = strategy.should_escalate_on_failure(state)

        assert decision.target_tier == EscalationTier.EXTRACTOR_ARGS

        # Now check from tier 2
        state2 = make_state(
            tier=EscalationTier.EXTRACTOR_ARGS,
            consecutive_403s=2,
        )
        decision2 = strategy.should_escalate_on_failure(state2)

        assert decision2.target_tier == EscalationTier.FULL_BYPASS


# ============================================================================
# Max Escalation Level Prevents Infinite Loops
# ============================================================================


class TestMaxEscalationLevel:
    """Test: Max escalation level prevents infinite loops."""

    @pytest.mark.fast
    def test_at_max_tier_no_escalation(self):
        """At max tier, no further escalation is possible."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())
        state = make_state(
            tier=EscalationTier.FULL_BYPASS,
            consecutive_403s=10,  # Even with high count
        )

        decision = strategy.should_escalate_on_failure(state)

        assert decision.should_escalate is False
        assert "max tier" in decision.reason.lower()

    @pytest.mark.fast
    def test_is_at_max_tier_check(self):
        """is_at_max_tier correctly identifies max tier."""
        strategy = EscalationStrategy(MockExtractorArgsConfig(max_tier=3))

        assert strategy.is_at_max_tier(EscalationTier.FULL_BYPASS) is True
        assert strategy.is_at_max_tier(EscalationTier.EXTRACTOR_ARGS) is False
        assert strategy.is_at_max_tier(EscalationTier.IMPERSONATE_ONLY) is False

    @pytest.mark.fast
    def test_custom_max_tier_limits_progression(self):
        """Custom max_tier=2 limits progression to Tier 2."""
        strategy = EscalationStrategy(MockExtractorArgsConfig(max_tier=2))
        state = make_state(
            tier=EscalationTier.EXTRACTOR_ARGS,
            consecutive_403s=5,
        )

        decision = strategy.should_escalate_on_failure(state)

        assert decision.should_escalate is False
        assert strategy.is_at_max_tier(EscalationTier.EXTRACTOR_ARGS) is True

    @pytest.mark.fast
    def test_slow_speed_also_respects_max_tier(self):
        """Slow speed escalation respects max tier."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())
        state = make_state(tier=EscalationTier.FULL_BYPASS)

        decision = strategy.should_escalate_on_slow_speed(
            state, consecutive_slow_count=10, slow_threshold=3
        )

        assert decision.should_escalate is False


# ============================================================================
# De-escalation After Success
# ============================================================================


class TestDeescalationAfterSuccess:
    """Test: De-escalation after success works correctly.

    Note: EscalationStrategy is stateless. De-escalation happens in EscalationState.
    We test that the strategy correctly identifies when state allows re-escalation.
    """

    @pytest.mark.fast
    def test_success_resets_consecutive_403s(self):
        """EscalationState.record_success resets 403 counter."""
        state = make_state(consecutive_403s=5)
        state.record_success()

        assert state.consecutive_403s == 0

    @pytest.mark.fast
    def test_success_maintains_tier(self):
        """Success resets counter but maintains tier (sticky escalation)."""
        state = make_state(
            tier=EscalationTier.EXTRACTOR_ARGS,
            consecutive_403s=5,
        )
        state.record_success()

        # Tier unchanged
        assert state.current_tier == EscalationTier.EXTRACTOR_ARGS
        # Counter reset
        assert state.consecutive_403s == 0

    @pytest.mark.fast
    def test_after_success_new_errors_start_fresh(self):
        """After success, new 403s must accumulate from zero."""
        strategy = EscalationStrategy(MockExtractorArgsConfig(escalation_threshold=2))
        state = make_state(
            tier=EscalationTier.IMPERSONATE_ONLY,
            consecutive_403s=5,
        )

        # Record success
        state.record_success()

        # One new 403 - not at threshold
        state.consecutive_403s = 1
        decision = strategy.should_escalate_on_failure(state)

        assert decision.should_escalate is False
        assert "Below threshold" in decision.reason


# ============================================================================
# Strategy Reset After Configurable Timeout
# ============================================================================


class TestStrategyResetAfterTimeout:
    """Test: Strategy reset after configurable timeout."""

    @pytest.mark.fast
    def test_cooldown_blocks_escalation(self):
        """Cooldown period blocks escalation."""
        strategy = EscalationStrategy(MockExtractorArgsConfig(cooldown_seconds=300.0))
        state = make_state(
            consecutive_403s=5,
            last_escalation_time=time.time() - 100,  # 100s ago
        )

        decision = strategy.should_escalate_on_failure(state)

        assert decision.should_escalate is False
        assert "cooldown" in decision.reason.lower()

    @pytest.mark.fast
    def test_past_cooldown_allows_escalation(self):
        """Past cooldown allows escalation."""
        strategy = EscalationStrategy(MockExtractorArgsConfig(cooldown_seconds=300.0))
        state = make_state(
            consecutive_403s=5,
            last_escalation_time=time.time() - 400,  # 400s ago
        )

        decision = strategy.should_escalate_on_failure(state)

        assert decision.should_escalate is True

    @pytest.mark.fast
    def test_is_past_cooldown_with_no_prior_escalation(self):
        """No prior escalation means cooldown is passed."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())
        state = make_state(last_escalation_time=None)

        assert strategy.is_past_cooldown(state) is True

    @pytest.mark.fast
    def test_get_cooldown_remaining(self):
        """get_cooldown_remaining returns correct value."""
        strategy = EscalationStrategy(MockExtractorArgsConfig(cooldown_seconds=300.0))

        with patch('src.downloader.escalation_strategy.time.time', return_value=1000.0):
            state = make_state(last_escalation_time=850.0)  # 150s ago
            remaining = strategy.get_cooldown_remaining(state)

            assert remaining == pytest.approx(150.0, abs=0.1)

    @pytest.mark.fast
    def test_cooldown_remaining_zero_when_expired(self):
        """Cooldown remaining is 0 when expired."""
        strategy = EscalationStrategy(MockExtractorArgsConfig(cooldown_seconds=300.0))

        with patch('src.downloader.escalation_strategy.time.time', return_value=1000.0):
            state = make_state(last_escalation_time=500.0)  # 500s ago
            remaining = strategy.get_cooldown_remaining(state)

            assert remaining == 0.0


# ============================================================================
# Additional Edge Cases
# ============================================================================


class TestEdgeCases:
    """Additional edge case tests for comprehensive coverage."""

    @pytest.mark.fast
    def test_default_config_values_no_config(self):
        """Strategy works with None config using defaults."""
        strategy = EscalationStrategy(None)

        assert strategy.threshold == 2
        assert strategy.cooldown_seconds == 300.0
        assert strategy.max_tier == EscalationTier.VPN_ROTATION

    @pytest.mark.fast
    def test_budget_exhausted_skips_to_max(self):
        """Budget exhausted skips intermediate tiers."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())
        state = make_state(
            tier=EscalationTier.IMPERSONATE_ONLY,
            consecutive_403s=5,
        )

        decision = strategy.should_escalate_on_failure(state, budget_exhausted=True)

        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.FULL_BYPASS
        assert decision.skip_to_max is True

    @pytest.mark.fast
    def test_below_threshold_no_escalation(self):
        """Below threshold does not escalate."""
        strategy = EscalationStrategy(MockExtractorArgsConfig(escalation_threshold=5))
        state = make_state(consecutive_403s=3)

        decision = strategy.should_escalate_on_failure(state)

        assert decision.should_escalate is False
        assert "3/5" in decision.reason

    @pytest.mark.fast
    def test_slow_speed_below_threshold(self):
        """Slow speed below threshold does not escalate."""
        strategy = EscalationStrategy(MockExtractorArgsConfig())
        state = make_state()

        decision = strategy.should_escalate_on_slow_speed(
            state, consecutive_slow_count=2, slow_threshold=3
        )

        assert decision.should_escalate is False
        assert "2/3" in decision.reason

    @pytest.mark.fast
    def test_escalation_decision_dataclass_defaults(self):
        """EscalationDecision has correct defaults."""
        decision = EscalationDecision(should_escalate=False)

        assert decision.target_tier is None
        assert decision.reason == ""
        assert decision.skip_to_max is False

    @pytest.mark.fast
    def test_max_tier_clamps_to_valid_range(self):
        """max_tier is clamped to valid EscalationTier range."""
        # max_tier=5 should clamp to 4 (VPN_ROTATION)
        strategy = EscalationStrategy(MockExtractorArgsConfig(max_tier=5))
        assert strategy.max_tier == EscalationTier.VPN_ROTATION

        # max_tier=0 should clamp to 1
        strategy = EscalationStrategy(MockExtractorArgsConfig(max_tier=0))
        assert strategy.max_tier == EscalationTier.IMPERSONATE_ONLY
