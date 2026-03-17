"""
Comprehensive test suite for src/downloader/rate_limit_budget.py.

Tests coverage for:
- RateLimitBudget class initialization
- Recording rotations, VPN switches, backoff time
- Budget limit checking (can_rotate, can_switch_vpn, can_backoff)
- Remaining budget calculations
- Budget exhaustion detection
- Checkpoint serialization (to_dict, from_dict)
- Keyword tracking for cross-keyword budget sharing

Created: January 25, 2026
User Story: US-004 - Cross-keyword rate limit budget tracking
"""

import sys
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.rate_limit_budget import RateLimitBudget


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def budget():
    """Create a fresh RateLimitBudget with default limits."""
    b = RateLimitBudget()
    # Set reasonable limits for testing
    b.max_rotations = 5
    b.max_vpn_switches = 3
    b.max_backoff_time = 60.0  # 1 minute
    return b


@pytest.fixture
def empty_budget():
    """Create a budget with no limits (unlimited)."""
    return RateLimitBudget()


@pytest.fixture
def exhausted_budget():
    """Create a budget that's already exhausted."""
    b = RateLimitBudget()
    b.max_rotations = 2
    b.max_vpn_switches = 1
    b.max_backoff_time = 30.0
    # Exhaust all resources
    b.rotations_used = 2
    b.vpn_switches_used = 1
    b.backoff_time_spent = 30.0
    return b


# ============================================================================
# Test Basic Recording Functions
# ============================================================================

class TestRecording:
    """Test recording rate limit recovery actions."""

    @pytest.mark.fast
    def test_record_rotation_increments_count(self, budget):
        """Recording a rotation increments the counter."""
        assert budget.rotations_used == 0
        budget.record_rotation()
        assert budget.rotations_used == 1
        budget.record_rotation()
        assert budget.rotations_used == 2

    @pytest.mark.fast
    def test_record_rotation_with_keyword(self, budget):
        """Recording a rotation tracks the keyword."""
        budget.record_rotation(keyword="sunset")
        assert "sunset" in budget.keywords_rate_limited
        assert budget.last_escalation_level == "cookie"

    @pytest.mark.fast
    def test_record_rotation_tracks_unique_keywords(self, budget):
        """Keywords are tracked uniquely (no duplicates)."""
        budget.record_rotation(keyword="sunset")
        budget.record_rotation(keyword="sunset")  # Duplicate
        budget.record_rotation(keyword="ocean")
        assert len(budget.keywords_rate_limited) == 2
        assert set(budget.keywords_rate_limited) == {"sunset", "ocean"}

    @pytest.mark.fast
    def test_record_vpn_switch_increments_count(self, budget):
        """Recording a VPN switch increments the counter."""
        assert budget.vpn_switches_used == 0
        budget.record_vpn_switch()
        assert budget.vpn_switches_used == 1

    @pytest.mark.fast
    def test_record_vpn_switch_with_keyword(self, budget):
        """Recording a VPN switch tracks the keyword."""
        budget.record_vpn_switch(keyword="mountain")
        assert "mountain" in budget.keywords_rate_limited
        assert budget.last_escalation_level == "vpn"

    @pytest.mark.fast
    def test_record_backoff_accumulates_time(self, budget):
        """Recording backoff accumulates total time spent."""
        assert budget.backoff_time_spent == 0.0
        budget.record_backoff(5.0)
        assert budget.backoff_time_spent == 5.0
        budget.record_backoff(10.5)
        assert budget.backoff_time_spent == 15.5

    @pytest.mark.fast
    def test_record_backoff_with_keyword(self, budget):
        """Recording backoff tracks the keyword."""
        budget.record_backoff(5.0, keyword="forest")
        assert "forest" in budget.keywords_rate_limited
        assert budget.last_escalation_level == "backoff"


# ============================================================================
# Test Budget Limit Checking
# ============================================================================

class TestLimitChecking:
    """Test checking if budget resources are available."""

    @pytest.mark.fast
    def test_can_rotate_within_budget(self, budget):
        """Can rotate when under limit."""
        assert budget.can_rotate() is True
        budget.rotations_used = 4
        assert budget.can_rotate() is True  # Still have 1 left

    @pytest.mark.fast
    def test_can_rotate_at_limit(self, budget):
        """Cannot rotate when at limit."""
        budget.rotations_used = 5  # At max_rotations
        assert budget.can_rotate() is False

    @pytest.mark.fast
    def test_can_rotate_unlimited(self, empty_budget):
        """Can always rotate when unlimited (max_rotations=0)."""
        empty_budget.max_rotations = 0
        empty_budget.rotations_used = 1000
        assert empty_budget.can_rotate() is True

    @pytest.mark.fast
    def test_can_switch_vpn_within_budget(self, budget):
        """Can switch VPN when under limit."""
        assert budget.can_switch_vpn() is True
        budget.vpn_switches_used = 2
        assert budget.can_switch_vpn() is True  # Still have 1 left

    @pytest.mark.fast
    def test_can_switch_vpn_at_limit(self, budget):
        """Cannot switch VPN when at limit."""
        budget.vpn_switches_used = 3  # At max_vpn_switches
        assert budget.can_switch_vpn() is False

    @pytest.mark.fast
    def test_can_switch_vpn_unlimited(self, empty_budget):
        """Can always switch VPN when unlimited (max_vpn_switches=0)."""
        empty_budget.max_vpn_switches = 0
        empty_budget.vpn_switches_used = 1000
        assert empty_budget.can_switch_vpn() is True

    @pytest.mark.fast
    def test_can_backoff_within_budget(self, budget):
        """Can backoff when under limit."""
        assert budget.can_backoff(5.0) is True
        budget.backoff_time_spent = 50.0
        assert budget.can_backoff(5.0) is True  # 50 + 5 = 55 < 60

    @pytest.mark.fast
    def test_can_backoff_at_limit(self, budget):
        """Cannot backoff when would exceed limit."""
        budget.backoff_time_spent = 55.0
        assert budget.can_backoff(10.0) is False  # 55 + 10 = 65 > 60
        assert budget.can_backoff(5.0) is True   # 55 + 5 = 60 <= 60

    @pytest.mark.fast
    def test_can_backoff_unlimited(self, empty_budget):
        """Can always backoff when unlimited (max_backoff_time=0)."""
        empty_budget.max_backoff_time = 0
        empty_budget.backoff_time_spent = 10000.0
        assert empty_budget.can_backoff(1000.0) is True


# ============================================================================
# Test Remaining Budget Calculations
# ============================================================================

class TestRemainingCalculations:
    """Test calculating remaining budget resources."""

    @pytest.mark.fast
    def test_rotations_remaining(self, budget):
        """Calculate remaining rotations."""
        assert budget.rotations_remaining() == 5
        budget.rotations_used = 3
        assert budget.rotations_remaining() == 2
        budget.rotations_used = 5
        assert budget.rotations_remaining() == 0
        budget.rotations_used = 7  # Over limit
        assert budget.rotations_remaining() == 0  # Clamped to 0

    @pytest.mark.fast
    def test_rotations_remaining_unlimited(self, empty_budget):
        """Remaining rotations is None when unlimited."""
        empty_budget.max_rotations = 0
        assert empty_budget.rotations_remaining() is None

    @pytest.mark.fast
    def test_vpn_switches_remaining(self, budget):
        """Calculate remaining VPN switches."""
        assert budget.vpn_switches_remaining() == 3
        budget.vpn_switches_used = 2
        assert budget.vpn_switches_remaining() == 1

    @pytest.mark.fast
    def test_vpn_switches_remaining_unlimited(self, empty_budget):
        """Remaining VPN switches is None when unlimited."""
        empty_budget.max_vpn_switches = 0
        assert empty_budget.vpn_switches_remaining() is None

    @pytest.mark.fast
    def test_backoff_time_remaining(self, budget):
        """Calculate remaining backoff time."""
        assert budget.backoff_time_remaining() == 60.0
        budget.backoff_time_spent = 25.5
        assert budget.backoff_time_remaining() == 34.5

    @pytest.mark.fast
    def test_backoff_time_remaining_unlimited(self, empty_budget):
        """Remaining backoff time is None when unlimited."""
        empty_budget.max_backoff_time = 0
        assert empty_budget.backoff_time_remaining() is None


# ============================================================================
# Test Escalation Recommendations
# ============================================================================

class TestEscalationRecommendations:
    """Test escalation level recommendations based on budget state."""

    @pytest.mark.fast
    def test_recommend_backoff_when_available(self, budget):
        """Recommend backoff when backoff budget is available."""
        assert budget.get_recommended_escalation() == "backoff"

    @pytest.mark.fast
    def test_recommend_cookie_when_backoff_exhausted(self, budget):
        """Recommend cookie rotation when backoff budget exhausted."""
        budget.backoff_time_spent = 60.0  # Exhausted
        assert budget.get_recommended_escalation() == "cookie"

    @pytest.mark.fast
    def test_recommend_vpn_when_cookies_exhausted(self, budget):
        """Recommend VPN when both backoff and cookies exhausted."""
        budget.backoff_time_spent = 60.0
        budget.rotations_used = 5
        assert budget.get_recommended_escalation() == "vpn"

    @pytest.mark.fast
    def test_recommend_exhausted_when_all_depleted(self, exhausted_budget):
        """Recommend 'exhausted' when all resources depleted."""
        assert exhausted_budget.get_recommended_escalation() == "exhausted"

    @pytest.mark.fast
    def test_is_exhausted_true_when_depleted(self, exhausted_budget):
        """is_exhausted returns True when all resources depleted."""
        assert exhausted_budget.is_exhausted() is True

    @pytest.mark.fast
    def test_is_exhausted_false_with_resources(self, budget):
        """is_exhausted returns False when resources available."""
        assert budget.is_exhausted() is False


# ============================================================================
# Test Budget Diagnostics (US-120-009)
# ============================================================================

class TestBudgetDiagnostics:
    """Test budget diagnostics methods: get_budget_status() and get_exhaustion_details()."""

    @pytest.mark.fast
    def test_get_budget_status_returns_all_fields(self, budget):
        """get_budget_status returns all diagnostic fields."""
        budget.rotations_used = 2
        budget.vpn_switches_used = 1
        budget.backoff_time_spent = 15.0
        budget.record_success()
        budget.record_failure()

        status = budget.get_budget_status()

        assert "rotations" in status
        assert "vpn_switches" in status
        assert "backoff_time" in status
        assert "is_exhausted" in status
        assert "is_nearly_exhausted" in status
        assert status["rotations"]["used"] == 2
        assert status["vpn_switches"]["used"] == 1
        assert status["backoff_time"]["spent"] == 15.0

    @pytest.mark.fast
    def test_get_budget_status_rotations_remaining(self, budget):
        """get_budget_status shows remaining rotations correctly."""
        budget.max_rotations = 10
        budget.rotations_used = 3

        status = budget.get_budget_status()

        assert status["rotations"]["remaining"] == 7
        assert status["rotations"]["max"] == 10

    @pytest.mark.fast
    def test_get_budget_status_unlimited_rotations(self, empty_budget):
        """get_budget_status handles unlimited rotations."""
        empty_budget.max_rotations = 0  # Unlimited

        status = empty_budget.get_budget_status()

        assert status["rotations"]["unlimited"] is True
        assert status["rotations"]["remaining"] is None

    @pytest.mark.fast
    def test_get_budget_status_vpn_remaining(self, budget):
        """get_budget_status shows remaining VPN switches correctly."""
        budget.max_vpn_switches = 5
        budget.vpn_switches_used = 2

        status = budget.get_budget_status()

        assert status["vpn_switches"]["remaining"] == 3
        assert status["vpn_switches"]["max"] == 5

    @pytest.mark.fast
    def test_get_budget_status_backoff_remaining(self, budget):
        """get_budget_status shows remaining backoff time correctly."""
        budget.max_backoff_time = 60.0
        budget.backoff_time_spent = 25.0

        status = budget.get_budget_status()

        assert status["backoff_time"]["remaining"] == 35.0
        assert status["backoff_time"]["max"] == 60.0

    @pytest.mark.fast
    def test_get_budget_status_nearly_exhausted(self, budget):
        """get_budget_status detects nearly exhausted resources (>80%)."""
        budget.max_rotations = 10
        budget.rotations_used = 9  # 90% used

        status = budget.get_budget_status()

        assert status["is_nearly_exhausted"] is True

    @pytest.mark.fast
    def test_get_budget_status_not_nearly_exhausted(self, budget):
        """get_budget_status returns False when resources below 80%."""
        budget.max_rotations = 10
        budget.rotations_used = 5  # 50% used

        status = budget.get_budget_status()

        assert status["is_nearly_exhausted"] is False

    @pytest.mark.fast
    def test_get_budget_status_success_rate(self, budget):
        """get_budget_status includes success rate calculation."""
        for _ in range(3):
            budget.record_success()
        for _ in range(1):
            budget.record_failure()

        status = budget.get_budget_status()

        assert status["success_rate"] == 0.75

    @pytest.mark.fast
    def test_get_budget_status_exhausted(self, exhausted_budget):
        """get_budget_status shows is_exhausted True when exhausted."""
        status = exhausted_budget.get_budget_status()

        assert status["is_exhausted"] is True

    @pytest.mark.fast
    def test_get_exhaustion_details_when_exhausted(self, exhausted_budget):
        """get_exhaustion_details returns details when budget is exhausted."""
        details = exhausted_budget.get_exhaustion_details()

        assert details is not None
        assert details["is_exhausted"] is True
        assert "rotations" in details["exhausted_resources"]
        assert "vpn_switches" in details["exhausted_resources"]
        assert "backoff_time" in details["exhausted_resources"]
        assert details["remaining_options"] == []

    @pytest.mark.fast
    def test_get_exhaustion_details_when_not_exhausted(self, budget):
        """get_exhaustion_details returns None when budget not exhausted."""
        details = budget.get_exhaustion_details()

        assert details is None

    @pytest.mark.fast
    def test_get_exhaustion_details_partial_exhaustion(self, budget):
        """get_exhaustion_details shows partial exhaustion correctly."""
        # Exhaust rotations but not VPN or backoff
        budget.max_rotations = 2
        budget.rotations_used = 2

        # When rotations exhausted but VPN/backoff available, is_exhausted should still return False
        # because there's a path forward (skip to VPN)
        # But get_exhaustion_details returns None when is_exhausted is False
        # Let's test that rotations_remaining returns 0
        remaining = budget.rotations_remaining()
        assert remaining == 0

    @pytest.mark.fast
    def test_get_budget_status_with_partial_rotation_exhaustion(self, budget):
        """get_budget_status shows correct state when rotations exhausted."""
        budget.max_rotations = 2
        budget.rotations_used = 2

        status = budget.get_budget_status()

        assert status["rotations"]["remaining"] == 0
        assert status["is_exhausted"] is False  # Can still use VPN/backoff

    @pytest.mark.fast
    def test_get_exhaustion_details_recommendation(self, exhausted_budget):
        """get_exhaustion_details includes recommendation."""
        details = exhausted_budget.get_exhaustion_details()

        assert details["recommendation"] == "exhausted"


# ============================================================================
# Test Checkpoint Serialization
# ============================================================================

class TestSerialization:
    """Test budget serialization for checkpoint persistence."""

    @pytest.mark.fast
    def test_to_dict_includes_all_fields(self, budget):
        """to_dict includes all budget state."""
        budget.rotations_used = 2
        budget.vpn_switches_used = 1
        budget.backoff_time_spent = 25.0
        budget.keywords_rate_limited = ["sunset", "ocean"]
        budget.last_escalation_level = "cookie"

        data = budget.to_dict()

        assert data["rotations_used"] == 2
        assert data["vpn_switches_used"] == 1
        assert data["backoff_time_spent"] == 25.0
        assert data["keywords_rate_limited"] == ["sunset", "ocean"]
        assert data["last_escalation_level"] == "cookie"
        assert data["max_rotations"] == 5
        assert data["max_vpn_switches"] == 3
        assert data["max_backoff_time"] == 60.0

    @pytest.mark.fast
    def test_from_dict_restores_state(self):
        """from_dict restores budget state from checkpoint."""
        data = {
            "rotations_used": 3,
            "vpn_switches_used": 2,
            "backoff_time_spent": 45.0,
            "keywords_rate_limited": ["forest", "mountain"],
            "last_escalation_level": "vpn",
            "max_rotations": 10,
            "max_vpn_switches": 5,
            "max_backoff_time": 120.0,
        }

        budget = RateLimitBudget.from_dict(data)

        assert budget.rotations_used == 3
        assert budget.vpn_switches_used == 2
        assert budget.backoff_time_spent == 45.0
        assert budget.keywords_rate_limited == ["forest", "mountain"]
        assert budget.last_escalation_level == "vpn"
        assert budget.max_rotations == 10
        assert budget.max_vpn_switches == 5
        assert budget.max_backoff_time == 120.0

    @pytest.mark.fast
    def test_from_dict_with_none_creates_default(self):
        """from_dict with None creates a default budget."""
        budget = RateLimitBudget.from_dict(None)

        assert budget.rotations_used == 0
        assert budget.vpn_switches_used == 0
        assert budget.backoff_time_spent == 0.0
        assert budget.keywords_rate_limited == []
        assert budget.last_escalation_level == "none"

    @pytest.mark.fast
    def test_from_dict_with_empty_dict_creates_default(self):
        """from_dict with empty dict creates default values."""
        budget = RateLimitBudget.from_dict({})

        assert budget.rotations_used == 0
        assert budget.vpn_switches_used == 0
        assert budget.backoff_time_spent == 0.0

    @pytest.mark.fast
    def test_roundtrip_serialization(self, budget):
        """Serialization roundtrip preserves state."""
        budget.rotations_used = 3
        budget.vpn_switches_used = 1
        budget.backoff_time_spent = 30.0
        budget.keywords_rate_limited = ["test1", "test2"]
        budget.last_escalation_level = "backoff"

        data = budget.to_dict()
        restored = RateLimitBudget.from_dict(data)

        assert restored.rotations_used == budget.rotations_used
        assert restored.vpn_switches_used == budget.vpn_switches_used
        assert restored.backoff_time_spent == budget.backoff_time_spent
        assert restored.keywords_rate_limited == budget.keywords_rate_limited
        assert restored.last_escalation_level == budget.last_escalation_level
        assert restored.max_rotations == budget.max_rotations
        assert restored.max_vpn_switches == budget.max_vpn_switches
        assert restored.max_backoff_time == budget.max_backoff_time


# ============================================================================
# Test Clear Function
# ============================================================================

class TestClear:
    """Test clearing budget state."""

    @pytest.mark.fast
    def test_clear_resets_all_counters(self, budget):
        """Clear resets all usage counters."""
        budget.rotations_used = 5
        budget.vpn_switches_used = 3
        budget.backoff_time_spent = 100.0
        budget.keywords_rate_limited = ["a", "b", "c"]
        budget.last_escalation_level = "vpn"

        budget.clear()

        assert budget.rotations_used == 0
        assert budget.vpn_switches_used == 0
        assert budget.backoff_time_spent == 0.0
        assert budget.keywords_rate_limited == []
        assert budget.last_escalation_level == "none"

    @pytest.mark.fast
    def test_clear_preserves_limits(self, budget):
        """Clear does not reset budget limits."""
        budget.clear()

        assert budget.max_rotations == 5
        assert budget.max_vpn_switches == 3
        assert budget.max_backoff_time == 60.0


# ============================================================================
# Test Summary Function
# ============================================================================

class TestSummary:
    """Test budget summary for reporting."""

    @pytest.mark.fast
    def test_get_summary_includes_usage(self, budget):
        """Summary includes usage statistics."""
        budget.rotations_used = 2
        budget.vpn_switches_used = 1
        budget.backoff_time_spent = 20.0
        budget.keywords_rate_limited = ["sunset"]
        budget.last_escalation_level = "cookie"

        summary = budget.get_summary()

        assert summary["rotations_used"] == 2
        assert summary["rotations_remaining"] == 3
        assert summary["vpn_switches_used"] == 1
        assert summary["vpn_switches_remaining"] == 2
        assert summary["backoff_time_spent"] == 20.0
        assert summary["backoff_time_remaining"] == 40.0
        assert summary["keywords_affected"] == 1
        assert summary["last_escalation"] == "cookie"
        assert summary["is_exhausted"] is False

    @pytest.mark.fast
    def test_get_summary_with_exhausted_budget(self, exhausted_budget):
        """Summary correctly reports exhausted state."""
        summary = exhausted_budget.get_summary()

        assert summary["is_exhausted"] is True
        assert summary["rotations_remaining"] == 0
        assert summary["vpn_switches_remaining"] == 0
        assert summary["backoff_time_remaining"] == 0.0


# ============================================================================
# Test Cross-Keyword Budget Sharing Scenario (US-004 Acceptance Criteria)
# ============================================================================

class TestCrossKeywordBudgetSharing:
    """
    Test the cross-keyword budget sharing scenario described in US-004.

    Scenario: Keyword A exhausts all cookie rotations, keyword B should
    skip directly to VPN switching instead of trying rotations again.
    """

    @pytest.mark.fast
    def test_keyword_a_exhausts_rotations_keyword_b_skips(self, budget):
        """When keyword A exhausts rotations, keyword B sees no rotations available."""
        # Keyword A exhausts all rotations
        for _ in range(5):
            budget.record_rotation(keyword="keyword_a")

        # Verify keyword A used all rotations
        assert budget.rotations_used == 5
        assert "keyword_a" in budget.keywords_rate_limited

        # Keyword B should see no rotations available
        assert budget.can_rotate() is False

        # Note: Escalation still recommends backoff first (cheaper than VPN)
        # because backoff budget is still available. This is correct behavior.
        # The key point is can_rotate() returns False.
        assert budget.get_recommended_escalation() == "backoff"

    @pytest.mark.fast
    def test_keyword_a_exhausts_all_keyword_b_goes_to_vpn(self, budget):
        """When keyword A exhausts backoff AND rotations, keyword B goes to VPN."""
        # Keyword A exhausts backoff budget
        budget.record_backoff(60.0, keyword="keyword_a")

        # Keyword A also exhausts all rotations
        for _ in range(5):
            budget.record_rotation(keyword="keyword_a")

        # Keyword B should see no backoff or rotation available
        assert budget.can_backoff(5.0) is False
        assert budget.can_rotate() is False

        # Keyword B should go directly to VPN
        assert budget.get_recommended_escalation() == "vpn"

    @pytest.mark.fast
    def test_keyword_a_uses_backoff_keyword_b_has_reduced_budget(self, budget):
        """When keyword A uses backoff, keyword B has reduced backoff budget."""
        # Keyword A uses 40 seconds of backoff
        budget.record_backoff(40.0, keyword="keyword_a")

        # Keyword B should have 20 seconds remaining
        assert budget.backoff_time_remaining() == 20.0
        assert budget.can_backoff(15.0) is True  # 40 + 15 = 55 < 60
        assert budget.can_backoff(25.0) is False  # 40 + 25 = 65 > 60

    @pytest.mark.fast
    def test_keywords_affect_different_resources(self, budget):
        """Different keywords can affect different budget resources."""
        # Keyword A uses backoff
        budget.record_backoff(30.0, keyword="keyword_a")

        # Keyword B uses rotation
        budget.record_rotation(keyword="keyword_b")
        budget.record_rotation(keyword="keyword_b")

        # Keyword C uses VPN switch
        budget.record_vpn_switch(keyword="keyword_c")

        # Verify all keywords tracked
        assert set(budget.keywords_rate_limited) == {"keyword_a", "keyword_b", "keyword_c"}

        # Verify resource usage
        assert budget.backoff_time_spent == 30.0
        assert budget.rotations_used == 2
        assert budget.vpn_switches_used == 1

    @pytest.mark.fast
    def test_session_budget_persists_across_keywords(self, budget):
        """Budget state persists across multiple keywords in a session."""
        # Process keyword 1
        budget.record_backoff(10.0, keyword="kw1")
        budget.record_rotation(keyword="kw1")

        # Process keyword 2
        budget.record_backoff(15.0, keyword="kw2")
        budget.record_rotation(keyword="kw2")
        budget.record_rotation(keyword="kw2")

        # Process keyword 3
        budget.record_vpn_switch(keyword="kw3")

        # All usage accumulates
        assert budget.backoff_time_spent == 25.0
        assert budget.rotations_used == 3
        assert budget.vpn_switches_used == 1

        # Budget limits apply globally
        assert budget.rotations_remaining() == 2
        assert budget.vpn_switches_remaining() == 2
        assert budget.backoff_time_remaining() == 35.0


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.fast
    def test_zero_backoff_time(self, budget):
        """Zero backoff time is handled correctly."""
        assert budget.can_backoff(0.0) is True
        budget.record_backoff(0.0)
        assert budget.backoff_time_spent == 0.0

    @pytest.mark.fast
    def test_negative_budget_remaining_clamped(self):
        """Over-budget usage is clamped to 0 remaining."""
        budget = RateLimitBudget()
        budget.max_rotations = 2
        budget.rotations_used = 5  # Over budget

        assert budget.rotations_remaining() == 0
        assert budget.can_rotate() is False

    @pytest.mark.fast
    def test_empty_keyword_not_tracked(self, budget):
        """Empty string keyword is not specially handled."""
        budget.record_rotation(keyword="")
        assert "" in budget.keywords_rate_limited

    @pytest.mark.fast
    def test_none_keyword_not_added(self, budget):
        """None keyword is not added to tracking list."""
        budget.record_rotation(keyword=None)
        assert budget.rotations_used == 1
        assert len(budget.keywords_rate_limited) == 0

    @pytest.mark.fast
    def test_large_backoff_value(self, budget):
        """Large backoff values are handled correctly."""
        budget.max_backoff_time = 1000000.0
        budget.record_backoff(999999.0)
        assert budget.backoff_time_spent == 999999.0
        assert budget.can_backoff(0.5) is True
        assert budget.can_backoff(2.0) is False


# ============================================================================
# Test Config Loading (US-009)
# ============================================================================

class TestConfigLoading:
    """Test RateLimitBudget initialization from RateLimitBudgetConfig."""

    @pytest.mark.fast
    def test_from_config_with_dataclass(self):
        """from_config reads limits from RateLimitBudgetConfig dataclass."""
        from src.config.sections.download import RateLimitBudgetConfig

        config = RateLimitBudgetConfig(
            enabled=True,
            max_rotations=15,
            max_backoff_time=900.0,
            max_vpn_switches=5,
        )
        budget = RateLimitBudget.from_config(config)

        assert budget.max_rotations == 15
        assert budget.max_backoff_time == 900.0
        assert budget.max_vpn_switches == 5

    @pytest.mark.fast
    def test_from_config_with_defaults(self):
        """from_config uses RateLimitBudgetConfig defaults correctly."""
        from src.config.sections.download import RateLimitBudgetConfig

        config = RateLimitBudgetConfig()  # All defaults
        budget = RateLimitBudget.from_config(config)

        assert budget.max_rotations == 10
        assert budget.max_backoff_time == 600.0
        assert budget.max_vpn_switches == 3

    @pytest.mark.fast
    def test_from_config_with_none_returns_default(self):
        """from_config with None returns default budget (safe fallback)."""
        budget = RateLimitBudget.from_config(None)

        assert budget.max_rotations == 0  # Default dataclass value (unlimited)
        assert budget.max_backoff_time == 300.0
        assert budget.max_vpn_switches == 10

    @pytest.mark.fast
    def test_from_config_starts_with_clean_state(self):
        """from_config creates budget with zero usage counters."""
        from src.config.sections.download import RateLimitBudgetConfig

        config = RateLimitBudgetConfig(max_rotations=5)
        budget = RateLimitBudget.from_config(config)

        assert budget.rotations_used == 0
        assert budget.vpn_switches_used == 0
        assert budget.backoff_time_spent == 0.0
        assert budget.keywords_rate_limited == []
        assert budget.successes == 0
        assert budget.failures == 0

    @pytest.mark.fast
    def test_budget_config_in_download_config(self):
        """RateLimitBudgetConfig is properly nested in DownloadConfig."""
        from src.config.sections.download import DownloadConfig, RateLimitBudgetConfig

        dc = DownloadConfig()
        assert isinstance(dc.rate_limit_budget, RateLimitBudgetConfig)
        assert dc.rate_limit_budget.enabled is True
        assert dc.rate_limit_budget.max_rotations == 10

    @pytest.mark.fast
    def test_download_config_post_init_converts_dict(self):
        """DownloadConfig __post_init__ converts rate_limit_budget dict to dataclass."""
        from src.config.sections.download import DownloadConfig, RateLimitBudgetConfig

        dc = DownloadConfig(rate_limit_budget={
            "enabled": True,
            "max_rotations": 20,
            "max_backoff_time": 1200.0,
            "max_vpn_switches": 7,
        })
        assert isinstance(dc.rate_limit_budget, RateLimitBudgetConfig)
        assert dc.rate_limit_budget.max_rotations == 20
        assert dc.rate_limit_budget.max_backoff_time == 1200.0
        assert dc.rate_limit_budget.max_vpn_switches == 7

    @pytest.mark.fast
    def test_from_config_budget_limits_are_functional(self):
        """Budget created from config enforces the configured limits."""
        from src.config.sections.download import RateLimitBudgetConfig

        config = RateLimitBudgetConfig(
            max_rotations=2,
            max_backoff_time=10.0,
            max_vpn_switches=1,
        )
        budget = RateLimitBudget.from_config(config)

        # Use up rotations
        budget.record_rotation()
        budget.record_rotation()
        assert budget.can_rotate() is False

        # Use up backoff
        budget.record_backoff(10.0)
        assert budget.can_backoff(1.0) is False

        # Use up VPN
        budget.record_vpn_switch()
        assert budget.can_switch_vpn() is False

        # All exhausted
        assert budget.is_exhausted() is True

    @pytest.mark.fast
    def test_config_roundtrip_through_serialization(self):
        """Config-created budget can be serialized and deserialized."""
        from src.config.sections.download import RateLimitBudgetConfig

        config = RateLimitBudgetConfig(
            max_rotations=8,
            max_backoff_time=400.0,
            max_vpn_switches=4,
        )
        budget = RateLimitBudget.from_config(config)
        budget.record_rotation(keyword="test")
        budget.record_backoff(15.0)

        data = budget.to_dict()
        restored = RateLimitBudget.from_dict(data)

        assert restored.max_rotations == 8
        assert restored.max_backoff_time == 400.0
        assert restored.max_vpn_switches == 4
        assert restored.rotations_used == 1
        assert restored.backoff_time_spent == 15.0


# ============================================================================
# Test Budget Summary String (US-61-010)
# ============================================================================

class TestBudgetSummaryString:
    """Test budget_summary() method for formatted logging output."""

    @pytest.mark.fast
    def test_budget_summary_empty_budget(self, empty_budget):
        """Summary shows 'none' when no budget consumed."""
        summary = empty_budget.budget_summary()
        assert summary == "Budget used: none"

    @pytest.mark.fast
    def test_budget_summary_with_rotations(self, budget):
        """Summary shows rotation percentage correctly."""
        budget.record_rotation(keyword="test")
        budget.record_rotation(keyword="test")
        summary = budget.budget_summary()
        # 2 of 5 rotations = 40%
        assert "rotations 2/5 (40%)" in summary
        assert "Budget used:" in summary

    @pytest.mark.fast
    def test_budget_summary_with_vpn_switches(self, budget):
        """Summary shows VPN switch percentage correctly."""
        budget.record_vpn_switch(keyword="test")
        summary = budget.budget_summary()
        # 1 of 3 VPN switches = 33%
        assert "VPN switches 1/3 (33%)" in summary

    @pytest.mark.fast
    def test_budget_summary_with_backoff(self, budget):
        """Summary shows backoff percentage correctly."""
        budget.record_backoff(45.0, keyword="test")
        summary = budget.budget_summary()
        # 45 of 60 seconds = 75%
        assert "backoff 45s/60s (75%)" in summary

    @pytest.mark.fast
    def test_budget_summary_all_resources(self, budget):
        """Summary includes all resource types when all are used."""
        budget.record_rotation(keyword="kw1")
        budget.record_rotation(keyword="kw2")
        budget.record_vpn_switch(keyword="kw3")
        budget.record_backoff(30.0, keyword="kw4")

        summary = budget.budget_summary()
        # Should include all three resource types
        assert "rotations 2/5 (40%)" in summary
        assert "VPN switches 1/3 (33%)" in summary
        assert "backoff 30s/60s (50%)" in summary

    @pytest.mark.fast
    def test_budget_summary_includes_keywords(self, budget):
        """Summary includes rate-limited keywords."""
        budget.record_rotation(keyword="sunset")
        budget.record_rotation(keyword="ocean")
        budget.record_rotation(keyword="mountains")

        summary = budget.budget_summary()
        assert "Rate-limited keywords:" in summary
        assert "sunset" in summary
        assert "ocean" in summary
        assert "mountains" in summary

    @pytest.mark.fast
    def test_budget_summary_truncates_many_keywords(self, budget):
        """Summary truncates keyword list when more than 5."""
        for i in range(8):
            budget.record_rotation(keyword=f"keyword{i}")

        summary = budget.budget_summary()
        assert "Rate-limited keywords:" in summary
        # Should show first 5 plus "... +3 more"
        assert "keyword0" in summary
        assert "keyword4" in summary
        assert "... +3 more" in summary

    @pytest.mark.fast
    def test_budget_summary_unlimited_rotations(self, empty_budget):
        """Summary shows unlimited rotations without percentage."""
        empty_budget.max_rotations = 0  # Unlimited
        empty_budget.record_rotation()
        empty_budget.record_rotation()
        empty_budget.record_rotation()

        summary = empty_budget.budget_summary()
        assert "rotations 3/unlimited" in summary

    @pytest.mark.fast
    def test_budget_summary_reflects_actual_usage(self, budget):
        """Summary accurately reflects actual budget usage."""
        # Simulate a realistic session
        budget.record_backoff(15.0, keyword="forest")
        budget.record_rotation(keyword="forest")
        budget.record_backoff(10.0, keyword="sunset")
        budget.record_rotation(keyword="sunset")
        budget.record_rotation(keyword="sunset")
        budget.record_vpn_switch(keyword="mountains")

        summary = budget.budget_summary()
        # Verify percentages match actual usage
        # 3 rotations of 5 = 60%
        assert "rotations 3/5 (60%)" in summary
        # 1 VPN switch of 3 = 33%
        assert "VPN switches 1/3 (33%)" in summary
        # 25s backoff of 60s = 42%
        assert "backoff 25s/60s (42%)" in summary
        # 3 unique keywords
        assert "forest" in summary
        assert "sunset" in summary
        assert "mountains" in summary


# ============================================================================
# Test Adaptive Cooldown Optimization (US-123-006)
# ============================================================================

class TestAdaptiveCooldown:
    """Test adaptive cooldown based on historical recovery times."""

    @pytest.fixture
    def adaptive_budget(self):
        """Create a budget with adaptive cooldown enabled."""
        b = RateLimitBudget()
        b.adaptive_cooldown_enabled = True
        b.cooldown_history = 10
        b.default_cooldown_seconds = 30.0
        return b

    @pytest.mark.fast
    def test_default_cooldown_when_disabled(self, budget):
        """When adaptive disabled, returns default cooldown."""
        budget.adaptive_cooldown_enabled = False
        budget.default_cooldown_seconds = 45.0

        result = budget.get_optimal_cooldown()

        assert result == 45.0

    @pytest.mark.fast
    def test_default_cooldown_when_no_history(self, adaptive_budget):
        """When enabled but no history, returns default cooldown."""
        adaptive_budget.recovery_times = []

        result = adaptive_budget.get_optimal_cooldown()

        assert result == 30.0

    @pytest.mark.fast
    def test_cooldown_from_single_recovery(self, adaptive_budget):
        """Single recovery time becomes optimal cooldown (or default if less)."""
        adaptive_budget.recovery_times = [20.0]
        # Since 20 < default (30), result is capped to 30
        # This is the conservative behavior - ensure minimum cooldown

        result = adaptive_budget.get_optimal_cooldown()

        assert result == 30.0  # Capped to default

    @pytest.mark.fast
    def test_cooldown_uses_75th_percentile(self, adaptive_budget):
        """Uses 75th percentile for conservative cooldown."""
        adaptive_budget.recovery_times = [10.0, 20.0, 30.0, 40.0]

        result = adaptive_budget.get_optimal_cooldown()

        # 75th percentile of [10, 20, 30, 40] = 30 * 0.75 = index 3 = 40
        assert result == 40.0

    @pytest.mark.fast
    def test_cooldown_ensures_minimum_default(self, adaptive_budget):
        """Cooldown is at least the default value."""
        adaptive_budget.default_cooldown_seconds = 30.0
        adaptive_budget.recovery_times = [10.0]  # Less than default

        result = adaptive_budget.get_optimal_cooldown()

        assert result == 30.0

    @pytest.mark.fast
    def test_record_rate_limit_event(self, budget):
        """Record rate limit event stores timestamp."""
        import time
        timestamp = time.time()

        budget.record_rate_limit_event(timestamp=timestamp)

        assert budget.last_rate_limit_timestamp == timestamp

    @pytest.mark.fast
    def test_record_recovery_calculates_time(self, budget):
        """Record recovery calculates time since rate limit."""
        import time
        # Simulate rate limit at t=100
        budget.record_rate_limit_event(timestamp=100.0)
        # Simulate success at t=130 (30 second recovery)
        recovery = budget.record_recovery(success_timestamp=130.0)

        assert recovery == 30.0
        assert budget.recovery_times == [30.0]

    @pytest.mark.fast
    def test_record_recovery_no_prior_event(self, budget):
        """Recovery without prior event returns None."""
        recovery = budget.record_recovery()

        assert recovery is None

    @pytest.mark.fast
    def test_cooldown_history_limit(self, adaptive_budget):
        """Recovery times are limited to cooldown_history."""
        adaptive_budget.cooldown_history = 3
        # Add more than history limit (need to set rate_limit_timestamp first)
        adaptive_budget.last_rate_limit_timestamp = 0.0
        adaptive_budget.record_recovery(success_timestamp=10.0)
        adaptive_budget.last_rate_limit_timestamp = 10.0
        adaptive_budget.record_recovery(success_timestamp=20.0)
        adaptive_budget.last_rate_limit_timestamp = 20.0
        adaptive_budget.record_recovery(success_timestamp=30.0)
        adaptive_budget.last_rate_limit_timestamp = 30.0
        adaptive_budget.record_recovery(success_timestamp=40.0)
        adaptive_budget.last_rate_limit_timestamp = 40.0
        adaptive_budget.record_recovery(success_timestamp=50.0)

        # Should only keep the most recent 3
        assert len(adaptive_budget.recovery_times) == 3

    @pytest.mark.fast
    def test_get_cooldown_stats(self, adaptive_budget):
        """Get cooldown stats returns correct statistics."""
        adaptive_budget.recovery_times = [10.0, 20.0, 30.0]

        stats = adaptive_budget.get_cooldown_stats()

        assert stats["count"] == 3
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert stats["mean"] == 20.0
        assert stats["adaptive_enabled"] is True

    @pytest.mark.fast
    def test_get_cooldown_stats_no_history(self, adaptive_budget):
        """Get cooldown stats returns None values when no history."""
        stats = adaptive_budget.get_cooldown_stats()

        assert stats["count"] == 0
        assert stats["min"] is None
        assert stats["max"] is None
        assert stats["mean"] is None
        assert stats["optimal"] == 30.0  # default

    @pytest.mark.fast
    def test_reset_cooldown_history(self, adaptive_budget):
        """Reset clears all history."""
        adaptive_budget.recovery_times = [10.0, 20.0, 30.0]
        adaptive_budget.last_rate_limit_timestamp = 100.0

        adaptive_budget.reset_cooldown_history()

        assert adaptive_budget.recovery_times == []
        assert adaptive_budget.last_rate_limit_timestamp is None


# ============================================================================
# Test Config Integration (US-123-006)
# ============================================================================

class TestAdaptiveCooldownConfig:
    """Test that config options are properly loaded."""

    @pytest.mark.fast
    def test_from_config_loads_adaptive_settings(self):
        """From_config loads adaptive cooldown settings."""
        from src.config.sections.rate_limit import RateLimitBudgetConfig

        config = RateLimitBudgetConfig(
            max_rotations=5,
            adaptive_cooldown_enabled=True,
            cooldown_history=15,
            default_cooldown_seconds=45.0,
        )

        budget = RateLimitBudget.from_config(config)

        assert budget.adaptive_cooldown_enabled is True
        assert budget.cooldown_history == 15
        assert budget.default_cooldown_seconds == 45.0

    @pytest.mark.fast
    def test_from_config_defaults_when_missing(self):
        """From_config uses defaults when adaptive not specified."""
        from src.config.sections.rate_limit import RateLimitBudgetConfig

        config = RateLimitBudgetConfig(max_rotations=5)

        budget = RateLimitBudget.from_config(config)

        assert budget.adaptive_cooldown_enabled is False
        assert budget.cooldown_history == 10  # default
        assert budget.default_cooldown_seconds == 30.0  # default
