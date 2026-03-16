"""
Integration tests for RateLimitBudget cross-keyword and session-wide behavior.

Verifies end-to-end budget tracking works correctly:
- Budget shared across multiple keywords in same session
- Budget exhaustion triggers escalation (cookie -> VPN)
- Budget reset on new download session
- Budget tracks successes and failures accurately
- should_skip_backoff returns True when budget exceeded

Sprint 35, US-35-007: RateLimitBudget integration tests
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.rate_limit_budget import RateLimitBudget


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def session_budget():
    """Create a budget simulating a typical download session."""
    budget = RateLimitBudget()
    budget.max_rotations = 5
    budget.max_vpn_switches = 3
    budget.max_backoff_time = 60.0
    return budget


@pytest.fixture
def tight_budget():
    """Create a budget with very limited resources for testing exhaustion."""
    budget = RateLimitBudget()
    budget.max_rotations = 2
    budget.max_vpn_switches = 1
    budget.max_backoff_time = 10.0
    return budget


# ============================================================================
# AC1: Budget shared across multiple keywords in same session
# ============================================================================

@pytest.mark.fast
class TestBudgetSharedAcrossKeywords:
    """Verify budget is shared correctly when processing multiple keywords."""

    def test_rotations_shared_across_keywords(self, session_budget):
        """Cookie rotations from keyword A reduce available rotations for keyword B."""
        # Keyword A uses 3 rotations
        for _ in range(3):
            session_budget.record_rotation(keyword="sunset_timelapse")

        # Keyword B should see only 2 rotations remaining
        assert session_budget.rotations_remaining() == 2
        assert session_budget.can_rotate() is True

        # Keyword B uses 2 more
        session_budget.record_rotation(keyword="ocean_waves")
        session_budget.record_rotation(keyword="ocean_waves")

        # Now both keywords have exhausted the shared budget
        assert session_budget.rotations_remaining() == 0
        assert session_budget.can_rotate() is False

    def test_backoff_time_shared_across_keywords(self, session_budget):
        """Backoff time from keyword A reduces available backoff for keyword B."""
        # Keyword A uses 40 seconds of backoff
        session_budget.record_backoff(40.0, keyword="mountain_scenery")

        # Keyword B should have 20 seconds remaining
        assert session_budget.backoff_time_remaining() == 20.0
        assert session_budget.can_backoff(15.0) is True
        assert session_budget.can_backoff(25.0) is False  # Would exceed budget

    def test_vpn_switches_shared_across_keywords(self, session_budget):
        """VPN switches from keyword A reduce available switches for keyword B."""
        # Keyword A uses 2 VPN switches
        session_budget.record_vpn_switch(keyword="forest_aerial")
        session_budget.record_vpn_switch(keyword="forest_aerial")

        # Keyword B should see only 1 switch remaining
        assert session_budget.vpn_switches_remaining() == 1
        assert session_budget.can_switch_vpn() is True

        # Keyword B uses the last switch
        session_budget.record_vpn_switch(keyword="city_skyline")
        assert session_budget.vpn_switches_remaining() == 0
        assert session_budget.can_switch_vpn() is False

    def test_multi_keyword_session_tracks_all_keywords(self, session_budget):
        """All keywords that trigger rate limit events are tracked."""
        keywords = ["sunset", "ocean", "mountain", "forest", "city"]

        for kw in keywords:
            session_budget.record_rotation(keyword=kw)

        # All 5 keywords should be tracked
        assert len(session_budget.keywords_rate_limited) == 5
        assert set(session_budget.keywords_rate_limited) == set(keywords)

    def test_same_keyword_multiple_events_tracked_once(self, session_budget):
        """Same keyword appearing multiple times is only tracked once."""
        # Same keyword triggers multiple events
        session_budget.record_rotation(keyword="sunset")
        session_budget.record_backoff(5.0, keyword="sunset")
        session_budget.record_rotation(keyword="sunset")
        session_budget.record_vpn_switch(keyword="sunset")

        # Keyword appears only once in tracking list
        assert session_budget.keywords_rate_limited.count("sunset") == 1

    def test_budget_state_accurate_after_multiple_keywords(self, session_budget):
        """Budget state accurately reflects cumulative usage across keywords."""
        # Simulate realistic multi-keyword session
        session_budget.record_backoff(10.0, keyword="kw1")
        session_budget.record_rotation(keyword="kw1")
        session_budget.record_success(keyword="kw1")

        session_budget.record_backoff(5.0, keyword="kw2")
        session_budget.record_rotation(keyword="kw2")
        session_budget.record_rotation(keyword="kw2")
        session_budget.record_failure(keyword="kw2")

        session_budget.record_vpn_switch(keyword="kw3")
        session_budget.record_success(keyword="kw3")

        # Verify cumulative state
        assert session_budget.backoff_time_spent == 15.0
        assert session_budget.rotations_used == 3
        assert session_budget.vpn_switches_used == 1
        assert session_budget.successes == 2
        assert session_budget.failures == 1
        assert len(session_budget.keywords_rate_limited) == 3


# ============================================================================
# AC2: Budget exhaustion triggers escalation (cookie -> VPN)
# ============================================================================

@pytest.mark.fast
class TestBudgetExhaustionTriggersEscalation:
    """Verify budget exhaustion triggers appropriate escalation advice."""

    def test_exhausted_rotations_triggers_skip_to_vpn(self, tight_budget):
        """When rotations exhausted, get_budget_advice returns skip_to_vpn."""
        # Exhaust all rotations
        tight_budget.record_rotation(keyword="kw1")
        tight_budget.record_rotation(keyword="kw2")

        assert tight_budget.can_rotate() is False
        assert tight_budget.get_budget_advice() == "skip_to_vpn"

    def test_exhausted_rotations_and_vpn_triggers_abort(self, tight_budget):
        """When both rotations and VPN exhausted, get_budget_advice returns abort_keyword."""
        # Exhaust rotations
        tight_budget.record_rotation(keyword="kw1")
        tight_budget.record_rotation(keyword="kw1")

        # Exhaust VPN
        tight_budget.record_vpn_switch(keyword="kw1")

        assert tight_budget.can_rotate() is False
        assert tight_budget.can_switch_vpn() is False
        assert tight_budget.get_budget_advice() == "abort_keyword"

    def test_escalation_progression_continue_to_vpn_to_abort(self, tight_budget):
        """Advice progresses through escalation levels correctly."""
        # Initial state: continue
        assert tight_budget.get_budget_advice() == "continue"

        # Exhaust rotations: skip_to_vpn
        tight_budget.record_rotation()
        tight_budget.record_rotation()
        assert tight_budget.get_budget_advice() == "skip_to_vpn"

        # Exhaust VPN: abort_keyword
        tight_budget.record_vpn_switch()
        assert tight_budget.get_budget_advice() == "abort_keyword"

    def test_keyword_a_exhaustion_affects_keyword_b_escalation(self, tight_budget):
        """Keyword A's exhaustion affects advice for keyword B processing."""
        # Keyword A exhausts rotations
        tight_budget.record_rotation(keyword="keyword_a")
        tight_budget.record_rotation(keyword="keyword_a")

        # When processing keyword B, should already see skip_to_vpn
        assert tight_budget.get_budget_advice() == "skip_to_vpn"

        # Keyword B can use VPN
        tight_budget.record_vpn_switch(keyword="keyword_b")

        # Now fully exhausted
        assert tight_budget.get_budget_advice() == "abort_keyword"

    def test_is_exhausted_reflects_total_exhaustion(self, tight_budget):
        """is_exhausted() returns True only when all recovery resources gone."""
        assert tight_budget.is_exhausted() is False

        # Exhaust rotations only
        tight_budget.record_rotation()
        tight_budget.record_rotation()
        assert tight_budget.is_exhausted() is False  # VPN still available

        # Exhaust VPN
        tight_budget.record_vpn_switch()

        # Backoff still available but can't help - fully exhausted
        # is_exhausted checks get_recommended_escalation
        # With backoff exhausted or at limit, and rotations/VPN gone
        tight_budget.record_backoff(10.0)  # Exhaust backoff
        assert tight_budget.is_exhausted() is True


# ============================================================================
# AC3: Budget reset on new download session
# ============================================================================

@pytest.mark.fast
class TestBudgetResetOnNewSession:
    """Verify budget can be reset for new download sessions."""

    def test_clear_resets_all_usage_counters(self, session_budget):
        """clear() resets all usage counters to zero."""
        # Use up resources
        session_budget.record_rotation(keyword="kw1")
        session_budget.record_rotation(keyword="kw2")
        session_budget.record_vpn_switch(keyword="kw1")
        session_budget.record_backoff(30.0, keyword="kw2")
        session_budget.record_success(keyword="kw1")
        session_budget.record_failure(keyword="kw2")

        # Clear for new session
        session_budget.clear()

        # All counters reset
        assert session_budget.rotations_used == 0
        assert session_budget.vpn_switches_used == 0
        assert session_budget.backoff_time_spent == 0.0
        assert session_budget.successes == 0
        assert session_budget.failures == 0
        assert session_budget.keywords_rate_limited == []
        assert session_budget.last_escalation_level == "none"

    def test_clear_preserves_budget_limits(self, session_budget):
        """clear() preserves configured budget limits."""
        original_max_rotations = session_budget.max_rotations
        original_max_vpn = session_budget.max_vpn_switches
        original_max_backoff = session_budget.max_backoff_time

        # Use resources and clear
        session_budget.record_rotation()
        session_budget.clear()

        # Limits unchanged
        assert session_budget.max_rotations == original_max_rotations
        assert session_budget.max_vpn_switches == original_max_vpn
        assert session_budget.max_backoff_time == original_max_backoff

    def test_clear_restores_full_budget_availability(self, tight_budget):
        """After clear(), all budget resources are available again."""
        # Exhaust all resources
        tight_budget.record_rotation()
        tight_budget.record_rotation()
        tight_budget.record_vpn_switch()
        tight_budget.record_backoff(10.0)

        assert tight_budget.is_exhausted() is True

        # Clear for new session
        tight_budget.clear()

        # Full budget available
        assert tight_budget.can_rotate() is True
        assert tight_budget.can_switch_vpn() is True
        assert tight_budget.can_backoff(10.0) is True
        assert tight_budget.get_budget_advice() == "continue"
        assert tight_budget.is_exhausted() is False

    def test_new_budget_from_config_starts_clean(self):
        """Creating new budget from config starts with clean state."""
        from src.config.sections.download import RateLimitBudgetConfig

        config = RateLimitBudgetConfig(
            max_rotations=10,
            max_backoff_time=300.0,
            max_vpn_switches=5,
        )

        budget = RateLimitBudget.from_config(config)

        # Fresh budget has no usage
        assert budget.rotations_used == 0
        assert budget.vpn_switches_used == 0
        assert budget.backoff_time_spent == 0.0
        assert budget.successes == 0
        assert budget.failures == 0
        assert budget.keywords_rate_limited == []

    def test_session_isolation_via_clear(self):
        """Multiple sessions can be isolated via clear() calls."""
        budget = RateLimitBudget()
        budget.max_rotations = 3
        budget.max_vpn_switches = 2

        # Session 1
        budget.record_rotation(keyword="session1_kw")
        budget.record_success()
        assert budget.rotations_used == 1
        assert budget.successes == 1

        # Clear for session 2
        budget.clear()

        # Session 2 starts fresh
        assert budget.rotations_used == 0
        assert budget.successes == 0
        assert "session1_kw" not in budget.keywords_rate_limited

        # Session 2 operations
        budget.record_rotation(keyword="session2_kw")
        budget.record_failure()
        assert budget.rotations_used == 1
        assert budget.failures == 1
        assert "session2_kw" in budget.keywords_rate_limited


# ============================================================================
# AC4: Budget tracks successes and failures accurately
# ============================================================================

@pytest.mark.fast
class TestBudgetTracksSuccessesAndFailures:
    """Verify success and failure tracking works correctly."""

    def test_record_success_increments_counter(self, session_budget):
        """record_success() increments success counter."""
        assert session_budget.successes == 0

        session_budget.record_success()
        assert session_budget.successes == 1

        session_budget.record_success(keyword="test_kw")
        assert session_budget.successes == 2

    def test_record_failure_increments_counter(self, session_budget):
        """record_failure() increments failure counter."""
        assert session_budget.failures == 0

        session_budget.record_failure()
        assert session_budget.failures == 1

        session_budget.record_failure(keyword="test_kw")
        assert session_budget.failures == 2

    def test_failure_with_keyword_adds_to_tracking(self, session_budget):
        """record_failure() with keyword adds keyword to rate_limited list."""
        session_budget.record_failure(keyword="problematic_keyword")

        assert "problematic_keyword" in session_budget.keywords_rate_limited

    def test_success_does_not_add_to_tracking(self, session_budget):
        """record_success() does not add keyword to rate_limited list."""
        session_budget.record_success(keyword="successful_keyword")

        # Success shouldn't add to rate limited list
        assert "successful_keyword" not in session_budget.keywords_rate_limited

    def test_mixed_success_failure_tracking(self, session_budget):
        """Mixed successes and failures are tracked correctly."""
        # Simulate realistic download session
        session_budget.record_success(keyword="kw1")
        session_budget.record_success(keyword="kw1")
        session_budget.record_failure(keyword="kw2")
        session_budget.record_success(keyword="kw2")
        session_budget.record_failure(keyword="kw3")
        session_budget.record_failure(keyword="kw3")

        assert session_budget.successes == 3
        assert session_budget.failures == 3

    def test_success_failure_in_summary(self, session_budget):
        """get_summary() includes success and failure counts."""
        session_budget.record_success()
        session_budget.record_success()
        session_budget.record_failure()

        summary = session_budget.get_summary()

        assert summary["successes"] == 2
        assert summary["failures"] == 1

    def test_success_failure_roundtrip_serialization(self, session_budget):
        """Successes and failures survive serialization roundtrip."""
        session_budget.record_success()
        session_budget.record_success()
        session_budget.record_success()
        session_budget.record_failure()
        session_budget.record_failure()

        data = session_budget.to_dict()
        restored = RateLimitBudget.from_dict(data)

        assert restored.successes == 3
        assert restored.failures == 2

    def test_success_failure_cleared_on_reset(self, session_budget):
        """clear() resets success and failure counters."""
        session_budget.record_success()
        session_budget.record_failure()

        session_budget.clear()

        assert session_budget.successes == 0
        assert session_budget.failures == 0


# ============================================================================
# AC5: should_skip_backoff returns True when budget exceeded
# ============================================================================

@pytest.mark.fast
class TestShouldSkipBackoff:
    """Verify should_skip_backoff() behavior when budget exceeded."""

    def test_should_skip_backoff_false_when_budget_available(self, session_budget):
        """should_skip_backoff returns False when backoff budget available."""
        assert session_budget.should_skip_backoff(5.0) is False
        assert session_budget.should_skip_backoff(60.0) is False  # Exactly at limit

    def test_should_skip_backoff_true_when_budget_exceeded(self, session_budget):
        """should_skip_backoff returns True when proposed backoff exceeds budget."""
        # Would exceed budget
        assert session_budget.should_skip_backoff(61.0) is True

        # Use up some budget
        session_budget.record_backoff(50.0)

        # Now even small backoff would exceed
        assert session_budget.should_skip_backoff(11.0) is True

    def test_should_skip_backoff_at_exact_boundary(self, session_budget):
        """should_skip_backoff returns False at exact boundary."""
        session_budget.record_backoff(55.0)

        # Exactly 5.0 remaining
        assert session_budget.should_skip_backoff(5.0) is False
        assert session_budget.should_skip_backoff(5.1) is True

    def test_should_skip_backoff_when_budget_exhausted(self, tight_budget):
        """should_skip_backoff returns True when budget already exhausted."""
        # Exhaust backoff budget
        tight_budget.record_backoff(10.0)

        # Any backoff should be skipped
        assert tight_budget.should_skip_backoff(0.1) is True
        assert tight_budget.should_skip_backoff(1.0) is True

    def test_should_skip_backoff_zero_duration(self, session_budget):
        """should_skip_backoff with zero duration returns False."""
        assert session_budget.should_skip_backoff(0.0) is False

        # Even when budget exhausted, zero backoff is ok
        session_budget.record_backoff(60.0)
        assert session_budget.should_skip_backoff(0.0) is False

    def test_should_skip_backoff_unlimited_budget(self):
        """should_skip_backoff always returns False with unlimited budget."""
        budget = RateLimitBudget()
        budget.max_backoff_time = 0  # Unlimited

        # Even huge backoff is ok
        budget.record_backoff(1000000.0)
        assert budget.should_skip_backoff(1000000.0) is False

    def test_should_skip_backoff_integration_with_escalation(self, tight_budget):
        """should_skip_backoff integrates with escalation decisions."""
        # Initial state: backoff available
        assert tight_budget.should_skip_backoff(5.0) is False
        assert tight_budget.get_recommended_escalation() == "backoff"

        # Exhaust backoff
        tight_budget.record_backoff(10.0)

        # Now should skip backoff and escalation recommends cookie
        assert tight_budget.should_skip_backoff(1.0) is True
        assert tight_budget.get_recommended_escalation() == "cookie"


# ============================================================================
# Integration: End-to-End Download Session Simulation
# ============================================================================

@pytest.mark.fast
class TestEndToEndSessionSimulation:
    """Simulate realistic download session to verify integration."""

    def test_multi_keyword_download_session(self):
        """Simulate a complete multi-keyword download session."""
        from src.config.sections.download import RateLimitBudgetConfig

        # Create budget from config (realistic setup)
        config = RateLimitBudgetConfig(
            enabled=True,
            max_rotations=5,
            max_backoff_time=30.0,
            max_vpn_switches=2,
        )
        budget = RateLimitBudget.from_config(config)

        # --- Keyword 1: "sunset" ---
        budget.record_attempt(keyword="sunset")
        budget.record_backoff(5.0, keyword="sunset")  # First rate limit
        budget.record_rotation(keyword="sunset")
        budget.record_success(keyword="sunset")

        # --- Keyword 2: "ocean" ---
        budget.record_attempt(keyword="ocean")
        budget.record_backoff(10.0, keyword="ocean")
        budget.record_rotation(keyword="ocean")
        budget.record_rotation(keyword="ocean")
        budget.record_failure(keyword="ocean")  # Still failed

        # Check budget state mid-session
        assert budget.rotations_used == 3
        assert budget.backoff_time_spent == 15.0
        assert budget.successes == 1
        assert budget.failures == 1
        assert len(budget.keywords_rate_limited) == 2

        # --- Keyword 3: "mountain" ---
        # Should see reduced budget
        assert budget.rotations_remaining() == 2
        assert budget.backoff_time_remaining() == 15.0

        budget.record_attempt(keyword="mountain")
        budget.record_backoff(10.0, keyword="mountain")
        budget.record_rotation(keyword="mountain")
        budget.record_rotation(keyword="mountain")  # Exhausts rotations

        assert budget.can_rotate() is False
        assert budget.get_budget_advice() == "skip_to_vpn"

        # Use VPN
        budget.record_vpn_switch(keyword="mountain")
        budget.record_success(keyword="mountain")

        # --- Keyword 4: "forest" ---
        # Very limited resources remaining
        assert budget.vpn_switches_remaining() == 1
        assert budget.backoff_time_remaining() == 5.0

        budget.record_attempt(keyword="forest")

        # Backoff would exceed budget - should skip
        assert budget.should_skip_backoff(10.0) is True

        # Final session state
        summary = budget.get_summary()
        assert summary["rotations_used"] == 5
        assert summary["vpn_switches_used"] == 1
        assert summary["successes"] == 2
        assert summary["failures"] == 1
        # Only keywords that triggered rate limit events (rotations, backoff, vpn, failures)
        # are tracked - "forest" only had attempt (not a rate limit event)
        assert summary["keywords_affected"] == 3

    def test_session_checkpoint_and_resume(self):
        """Verify budget can be checkpointed and resumed."""
        # Session 1: Start work
        budget1 = RateLimitBudget()
        budget1.max_rotations = 10
        budget1.max_vpn_switches = 5
        budget1.max_backoff_time = 120.0

        for i in range(3):
            budget1.record_rotation(keyword=f"session1_kw{i}")
        budget1.record_backoff(45.0)
        budget1.record_success()
        budget1.record_failure()

        # Checkpoint
        checkpoint_data = budget1.to_dict()

        # Session 2: Resume from checkpoint
        budget2 = RateLimitBudget.from_dict(checkpoint_data)

        # Verify state restored
        assert budget2.rotations_used == 3
        assert budget2.backoff_time_spent == 45.0
        assert budget2.successes == 1
        assert budget2.failures == 1
        assert len(budget2.keywords_rate_limited) == 3

        # Can continue where we left off
        assert budget2.rotations_remaining() == 7
        assert budget2.backoff_time_remaining() == 75.0

        # Continue work
        budget2.record_rotation(keyword="resumed_kw")
        assert budget2.rotations_used == 4

    def test_auto_scaling_for_large_keyword_count(self):
        """Budget auto-scales for sessions with many keywords."""
        budget = RateLimitBudget()
        budget.max_rotations = 5
        budget.max_vpn_switches = 2
        budget.max_backoff_time = 60.0

        # Simulate session with 15 keywords
        budget.scale_for_keywords(15, auto_scale=True)

        # Budget should scale up (15/5 = 3x multiplier)
        assert budget.max_rotations == 15  # 5 * 3
        assert budget.max_vpn_switches == 6  # 2 * 3
        assert budget.max_backoff_time == 180.0  # 60 * 3

    def test_nearly_exhausted_detection(self):
        """is_nearly_exhausted() detects >80% threshold correctly."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 100.0

        # Use up 80% of rotations - not nearly exhausted (threshold is >80%)
        for _ in range(8):
            budget.record_rotation()
        assert budget.is_nearly_exhausted() is False

        # Use one more (90%) - now nearly exhausted (9/10 > 0.8)
        budget.record_rotation()
        assert budget.is_nearly_exhausted() is True
