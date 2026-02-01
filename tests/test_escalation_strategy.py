"""Unit tests for EscalationStrategy (US-35-009).

Tests the strategy class that encapsulates escalation decision logic:
- Level progression through tiers
- Cooldown checks
- Budget-aware skip-to-max tier decisions
- Slow speed escalation decisions
- Circuit breaker shortcut decisions
"""

import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import patch

import pytest

from src.downloader.escalation_strategy import EscalationStrategy, EscalationDecision
from src.downloader.types import EscalationState, EscalationTier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeExtractorArgsConfig:
    """Minimal stand-in for ExtractorArgsConfig."""
    enabled: bool = True
    player_clients: List[str] = field(
        default_factory=lambda: ["web_safari", "tv_downgraded", "web"]
    )
    escalation_threshold: int = 2
    cooldown_seconds: float = 300.0
    max_tier: int = 3


@pytest.fixture
def config():
    """Default config for testing."""
    return FakeExtractorArgsConfig()


@pytest.fixture
def strategy(config):
    """Strategy instance with default config."""
    return EscalationStrategy(config)


@pytest.fixture
def strategy_no_config():
    """Strategy instance without config (uses defaults)."""
    return EscalationStrategy(None)


# ---------------------------------------------------------------------------
# Test: Level progression
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestLevelProgression:
    """Tests for tier progression logic."""

    def test_get_next_tier_from_tier_1(self, strategy):
        """Next tier from IMPERSONATE_ONLY is EXTRACTOR_ARGS."""
        next_tier = strategy.get_next_tier(EscalationTier.IMPERSONATE_ONLY)
        assert next_tier == EscalationTier.EXTRACTOR_ARGS

    def test_get_next_tier_from_tier_2(self, strategy):
        """Next tier from EXTRACTOR_ARGS is FULL_BYPASS."""
        next_tier = strategy.get_next_tier(EscalationTier.EXTRACTOR_ARGS)
        assert next_tier == EscalationTier.FULL_BYPASS

    def test_get_next_tier_from_tier_3_is_none(self, strategy):
        """At FULL_BYPASS, there is no next tier."""
        next_tier = strategy.get_next_tier(EscalationTier.FULL_BYPASS)
        assert next_tier is None

    def test_is_at_max_tier_false_for_tier_1(self, strategy):
        """Tier 1 is not at max."""
        assert strategy.is_at_max_tier(EscalationTier.IMPERSONATE_ONLY) is False

    def test_is_at_max_tier_false_for_tier_2(self, strategy):
        """Tier 2 is not at max."""
        assert strategy.is_at_max_tier(EscalationTier.EXTRACTOR_ARGS) is False

    def test_is_at_max_tier_true_for_tier_3(self, strategy):
        """Tier 3 is at max."""
        assert strategy.is_at_max_tier(EscalationTier.FULL_BYPASS) is True

    def test_max_tier_respects_config(self):
        """max_tier from config is respected."""
        config = FakeExtractorArgsConfig(max_tier=2)
        strat = EscalationStrategy(config)
        assert strat.max_tier == EscalationTier.EXTRACTOR_ARGS
        assert strat.is_at_max_tier(EscalationTier.EXTRACTOR_ARGS) is True

    def test_max_tier_default_without_config(self, strategy_no_config):
        """Without config, max_tier defaults to FULL_BYPASS."""
        assert strategy_no_config.max_tier == EscalationTier.FULL_BYPASS


# ---------------------------------------------------------------------------
# Test: Cooldown checks
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestCooldownChecks:
    """Tests for cooldown period logic."""

    def test_is_past_cooldown_never_escalated(self, strategy):
        """No previous escalation means past cooldown."""
        state = EscalationState()
        assert state.last_escalation_time is None
        assert strategy.is_past_cooldown(state) is True

    def test_is_past_cooldown_within_window(self, strategy):
        """Within cooldown window returns False."""
        state = EscalationState(last_escalation_time=time.time())
        assert strategy.is_past_cooldown(state) is False

    def test_is_past_cooldown_after_expiry(self, strategy):
        """After cooldown expiry returns True."""
        state = EscalationState(last_escalation_time=time.time() - 400)
        assert strategy.is_past_cooldown(state) is True

    def test_get_cooldown_remaining_never_escalated(self, strategy):
        """No previous escalation means 0.0 remaining."""
        state = EscalationState()
        assert strategy.get_cooldown_remaining(state) == 0.0

    def test_get_cooldown_remaining_positive(self, strategy):
        """Within cooldown window returns positive remaining."""
        state = EscalationState(last_escalation_time=time.time())
        remaining = strategy.get_cooldown_remaining(state)
        assert remaining > 290  # Should be close to 300s

    def test_get_cooldown_remaining_zero_after_expiry(self, strategy):
        """After cooldown expiry returns 0.0."""
        state = EscalationState(last_escalation_time=time.time() - 400)
        assert strategy.get_cooldown_remaining(state) == 0.0


# ---------------------------------------------------------------------------
# Test: Failure escalation decisions
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestFailureEscalationDecisions:
    """Tests for should_escalate_on_failure() decision logic."""

    def test_below_threshold_no_escalation(self, strategy):
        """Below threshold returns should_escalate=False."""
        state = EscalationState(consecutive_403s=1)  # threshold is 2
        decision = strategy.should_escalate_on_failure(state)
        assert decision.should_escalate is False
        assert "threshold" in decision.reason.lower()

    def test_at_threshold_escalates(self, strategy):
        """At threshold returns should_escalate=True."""
        state = EscalationState(consecutive_403s=2)
        decision = strategy.should_escalate_on_failure(state)
        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.EXTRACTOR_ARGS

    def test_above_threshold_escalates(self, strategy):
        """Above threshold returns should_escalate=True."""
        state = EscalationState(consecutive_403s=5)
        decision = strategy.should_escalate_on_failure(state)
        assert decision.should_escalate is True

    def test_at_max_tier_no_escalation(self, strategy):
        """At max tier returns should_escalate=False."""
        state = EscalationState(
            current_tier=EscalationTier.FULL_BYPASS,
            consecutive_403s=5
        )
        decision = strategy.should_escalate_on_failure(state)
        assert decision.should_escalate is False
        assert "max tier" in decision.reason.lower()

    def test_in_cooldown_no_escalation(self, strategy):
        """In cooldown returns should_escalate=False."""
        state = EscalationState(
            consecutive_403s=5,
            last_escalation_time=time.time()  # Just escalated
        )
        decision = strategy.should_escalate_on_failure(state)
        assert decision.should_escalate is False
        assert "cooldown" in decision.reason.lower()

    def test_budget_exhausted_skips_to_max(self, strategy):
        """Budget exhausted causes skip_to_max=True."""
        state = EscalationState(consecutive_403s=2)
        decision = strategy.should_escalate_on_failure(state, budget_exhausted=True)
        assert decision.should_escalate is True
        assert decision.skip_to_max is True
        assert decision.target_tier == EscalationTier.FULL_BYPASS
        assert "budget" in decision.reason.lower()

    def test_normal_escalation_not_skip_to_max(self, strategy):
        """Normal escalation does not skip to max."""
        state = EscalationState(consecutive_403s=2)
        decision = strategy.should_escalate_on_failure(state, budget_exhausted=False)
        assert decision.should_escalate is True
        assert decision.skip_to_max is False
        assert decision.target_tier == EscalationTier.EXTRACTOR_ARGS


# ---------------------------------------------------------------------------
# Test: Slow speed escalation decisions
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestSlowSpeedDecisions:
    """Tests for should_escalate_on_slow_speed() decision logic."""

    def test_below_slow_threshold_no_escalation(self, strategy):
        """Below slow threshold returns should_escalate=False."""
        state = EscalationState()
        decision = strategy.should_escalate_on_slow_speed(state, consecutive_slow_count=2)
        assert decision.should_escalate is False
        assert "threshold" in decision.reason.lower()

    def test_at_slow_threshold_escalates(self, strategy):
        """At slow threshold returns should_escalate=True."""
        state = EscalationState()
        decision = strategy.should_escalate_on_slow_speed(state, consecutive_slow_count=3)
        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.EXTRACTOR_ARGS

    def test_slow_at_max_tier_no_escalation(self, strategy):
        """At max tier, slow speed does not escalate."""
        state = EscalationState(current_tier=EscalationTier.FULL_BYPASS)
        decision = strategy.should_escalate_on_slow_speed(state, consecutive_slow_count=5)
        assert decision.should_escalate is False
        assert "max tier" in decision.reason.lower()

    def test_slow_in_cooldown_no_escalation(self, strategy):
        """In cooldown, slow speed does not escalate."""
        state = EscalationState(last_escalation_time=time.time())
        decision = strategy.should_escalate_on_slow_speed(state, consecutive_slow_count=3)
        assert decision.should_escalate is False
        assert "cooldown" in decision.reason.lower()

    def test_custom_slow_threshold(self, strategy):
        """Custom slow_threshold is respected."""
        state = EscalationState()
        decision = strategy.should_escalate_on_slow_speed(
            state, consecutive_slow_count=4, slow_threshold=5
        )
        assert decision.should_escalate is False

        decision = strategy.should_escalate_on_slow_speed(
            state, consecutive_slow_count=5, slow_threshold=5
        )
        assert decision.should_escalate is True


# ---------------------------------------------------------------------------
# Test: Circuit breaker shortcut decisions
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestCircuitBreakerShortcut:
    """Tests for should_shortcut_to_max() decision logic."""

    def test_circuit_breaker_not_open_no_shortcut(self, strategy):
        """Circuit breaker not open means no shortcut."""
        decision = strategy.should_shortcut_to_max(
            EscalationTier.IMPERSONATE_ONLY, circuit_breaker_open=False
        )
        assert decision.should_escalate is False

    def test_circuit_breaker_open_shortcuts(self, strategy):
        """Circuit breaker open shortcuts to max."""
        decision = strategy.should_shortcut_to_max(
            EscalationTier.IMPERSONATE_ONLY, circuit_breaker_open=True
        )
        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.FULL_BYPASS
        assert decision.skip_to_max is True

    def test_circuit_breaker_open_from_tier_2(self, strategy):
        """Circuit breaker open shortcuts from Tier 2."""
        decision = strategy.should_shortcut_to_max(
            EscalationTier.EXTRACTOR_ARGS, circuit_breaker_open=True
        )
        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.FULL_BYPASS

    def test_circuit_breaker_open_already_at_max(self, strategy):
        """Already at max tier, circuit breaker has no effect."""
        decision = strategy.should_shortcut_to_max(
            EscalationTier.FULL_BYPASS, circuit_breaker_open=True
        )
        assert decision.should_escalate is False
        assert "already at full bypass" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Test: Default configuration values
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestDefaultConfiguration:
    """Tests for default values when config is None."""

    def test_default_threshold(self, strategy_no_config):
        """Default threshold is 2."""
        assert strategy_no_config.threshold == 2

    def test_default_cooldown_seconds(self, strategy_no_config):
        """Default cooldown is 300.0 seconds."""
        assert strategy_no_config.cooldown_seconds == 300.0

    def test_default_max_tier(self, strategy_no_config):
        """Default max tier is FULL_BYPASS."""
        assert strategy_no_config.max_tier == EscalationTier.FULL_BYPASS


# ---------------------------------------------------------------------------
# Test: Strategy correctly progresses through levels
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestLevelProgressionFlow:
    """Test that strategy correctly progresses through all escalation levels."""

    def test_full_tier_1_to_3_progression(self):
        """Strategy correctly identifies progression from Tier 1 to 3."""
        config = FakeExtractorArgsConfig(cooldown_seconds=0.0)  # No cooldown
        strategy = EscalationStrategy(config)

        # Start at Tier 1
        state = EscalationState(
            current_tier=EscalationTier.IMPERSONATE_ONLY,
            consecutive_403s=2
        )

        # First escalation: Tier 1 -> 2
        decision = strategy.should_escalate_on_failure(state)
        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.EXTRACTOR_ARGS
        assert decision.skip_to_max is False

        # Simulate escalation
        state.current_tier = EscalationTier.EXTRACTOR_ARGS
        state.consecutive_403s = 2  # Reset and reach threshold again

        # Second escalation: Tier 2 -> 3
        decision = strategy.should_escalate_on_failure(state)
        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.FULL_BYPASS
        assert decision.skip_to_max is False

        # Simulate escalation
        state.current_tier = EscalationTier.FULL_BYPASS
        state.consecutive_403s = 2

        # At max tier: no more escalation
        decision = strategy.should_escalate_on_failure(state)
        assert decision.should_escalate is False

    def test_budget_exhaustion_skips_tiers(self):
        """Budget exhaustion causes immediate skip to max tier."""
        strategy = EscalationStrategy(FakeExtractorArgsConfig())

        # Start at Tier 1 with budget exhausted
        state = EscalationState(
            current_tier=EscalationTier.IMPERSONATE_ONLY,
            consecutive_403s=2
        )

        decision = strategy.should_escalate_on_failure(state, budget_exhausted=True)
        assert decision.should_escalate is True
        assert decision.target_tier == EscalationTier.FULL_BYPASS
        assert decision.skip_to_max is True
        # Skipped Tier 2 entirely!
