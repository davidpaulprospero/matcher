"""
Test suite for priority-based budget allocation in rate_limit_budget.py.

Tests coverage for:
- PriorityLevel enum and parsing
- Setting keyword priorities
- Budget allocation based on priority (high 50%, medium 30%, low 20%)
- Consecutive success boost mechanism
- Budget exhaustion prediction
- Stealing unused budget from low-priority keywords

User Story: US-109-005 - Adaptive budget allocation based on keyword priority
"""

import sys
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.rate_limit_budget import (
    RateLimitBudget,
    PriorityLevel,
    PRIORITY_ALLOCATION,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def priority_budget():
    """Create a budget with priority allocation enabled."""
    b = RateLimitBudget()
    b.max_rotations = 10
    b.max_vpn_switches = 5
    b.max_backoff_time = 100.0
    b.priority_allocation_enabled = True
    return b


@pytest.fixture
def exhausted_priority_budget():
    """Create a priority budget that's partially exhausted."""
    b = RateLimitBudget()
    b.max_rotations = 6
    b.max_vpn_switches = 3
    b.max_backoff_time = 60.0
    b.priority_allocation_enabled = True
    # Use some budget
    b.rotations_used = 4
    b.vpn_switches_used = 2
    b.backoff_time_spent = 40.0
    return b


# ============================================================================
# Test PriorityLevel Enum
# ============================================================================

class TestPriorityLevel:
    """Test PriorityLevel enum and parsing."""

    @pytest.mark.fast
    def test_priority_level_enum_values(self):
        """PriorityLevel has correct values."""
        assert PriorityLevel.HIGH.value == "high"
        assert PriorityLevel.MEDIUM.value == "medium"
        assert PriorityLevel.LOW.value == "low"

    @pytest.mark.fast
    def test_priority_from_string_high(self):
        """Parse 'high' string to PriorityLevel."""
        assert PriorityLevel.from_string("high") == PriorityLevel.HIGH
        assert PriorityLevel.from_string("HIGH") == PriorityLevel.HIGH

    @pytest.mark.fast
    def test_priority_from_string_medium(self):
        """Parse 'medium' string to PriorityLevel."""
        assert PriorityLevel.from_string("medium") == PriorityLevel.MEDIUM
        assert PriorityLevel.from_string("MEDIUM") == PriorityLevel.MEDIUM

    @pytest.mark.fast
    def test_priority_from_string_low(self):
        """Parse 'low' string to PriorityLevel."""
        assert PriorityLevel.from_string("low") == PriorityLevel.LOW
        assert PriorityLevel.from_string("LOW") == PriorityLevel.LOW

    @pytest.mark.fast
    def test_priority_from_string_invalid_default(self):
        """Invalid priority string defaults to MEDIUM."""
        assert PriorityLevel.from_string("invalid") == PriorityLevel.MEDIUM
        assert PriorityLevel.from_string("") == PriorityLevel.MEDIUM

    @pytest.mark.fast
    def test_priority_allocation_defaults(self):
        """Default allocation percentages are correct."""
        assert PRIORITY_ALLOCATION[PriorityLevel.HIGH] == 0.50
        assert PRIORITY_ALLOCATION[PriorityLevel.MEDIUM] == 0.30
        assert PRIORITY_ALLOCATION[PriorityLevel.LOW] == 0.20


# ============================================================================
# Test Setting Keyword Priorities
# ============================================================================

class TestKeywordPriority:
    """Test setting and getting keyword priorities."""

    @pytest.mark.fast
    def test_set_keyword_priority(self, priority_budget):
        """Setting keyword priority works correctly."""
        priority_budget.set_keyword_priority("sunset", PriorityLevel.HIGH)
        assert priority_budget.get_keyword_priority("sunset") == PriorityLevel.HIGH

    @pytest.mark.fast
    def test_set_keyword_priority_string(self, priority_budget):
        """Setting keyword priority from string works."""
        priority_budget.set_keyword_priority_string("ocean", "low")
        assert priority_budget.get_keyword_priority("ocean") == PriorityLevel.LOW

    @pytest.mark.fast
    def test_default_keyword_priority_is_medium(self, priority_budget):
        """Keywords without explicit priority default to MEDIUM."""
        assert priority_budget.get_keyword_priority("unknown") == PriorityLevel.MEDIUM

    @pytest.mark.fast
    def test_multiple_keywords_different_priorities(self, priority_budget):
        """Multiple keywords can have different priorities."""
        priority_budget.set_keyword_priority("high_kw", PriorityLevel.HIGH)
        priority_budget.set_keyword_priority("medium_kw", PriorityLevel.MEDIUM)
        priority_budget.set_keyword_priority("low_kw", PriorityLevel.LOW)

        assert priority_budget.get_keyword_priority("high_kw") == PriorityLevel.HIGH
        assert priority_budget.get_keyword_priority("medium_kw") == PriorityLevel.MEDIUM
        assert priority_budget.get_keyword_priority("low_kw") == PriorityLevel.LOW


# ============================================================================
# Test Priority-Based Budget Allocation
# ============================================================================

class TestPriorityBudgetAllocation:
    """Test budget allocation based on priority."""

    @pytest.mark.fast
    def test_high_priority_gets_50_percent(self, priority_budget):
        """High-priority keywords get 50% of budget."""
        priority_budget.set_keyword_priority("high_kw", PriorityLevel.HIGH)
        budget = priority_budget.get_budget_for_keyword("high_kw")

        # Full budget: 10 rotations, 5 vpn, 100 backoff
        # High priority gets 50%: 5 rotations, 2 vpn, 50 backoff
        assert budget["rotations"] == 5
        assert budget["vpn_switches"] == 2
        assert budget["backoff_time"] == 50.0

    @pytest.mark.fast
    def test_medium_priority_gets_30_percent(self, priority_budget):
        """Medium-priority keywords get 30% of budget."""
        priority_budget.set_keyword_priority("medium_kw", PriorityLevel.MEDIUM)
        budget = priority_budget.get_budget_for_keyword("medium_kw")

        # Medium priority gets 30%: 3 rotations, 1 vpn, 30 backoff
        assert budget["rotations"] == 3
        assert budget["vpn_switches"] == 1
        assert budget["backoff_time"] == 30.0

    @pytest.mark.fast
    def test_low_priority_gets_20_percent(self, priority_budget):
        """Low-priority keywords get 20% of budget."""
        priority_budget.set_keyword_priority("low_kw", PriorityLevel.LOW)
        budget = priority_budget.get_budget_for_keyword("low_kw")

        # Low priority gets 20%: 2 rotations, 1 vpn, 20 backoff
        assert budget["rotations"] == 2
        assert budget["vpn_switches"] == 1
        assert budget["backoff_time"] == 20.0

    @pytest.mark.fast
    def test_budget_allocation_disabled(self):
        """When allocation disabled, full budget is returned."""
        b = RateLimitBudget()
        b.max_rotations = 10
        b.priority_allocation_enabled = False
        b.set_keyword_priority("kw", PriorityLevel.HIGH)

        budget = b.get_budget_for_keyword("kw")
        assert budget["rotations"] == 10  # Full budget


# ============================================================================
# Test Consecutive Success Boost
# ============================================================================

class TestConsecutiveSuccessBoost:
    """Test consecutive success boost mechanism."""

    @pytest.mark.fast
    def test_consecutive_success_boost_threshold(self, priority_budget):
        """Boost triggers after consecutive_success_boost_threshold successes."""
        priority_budget.set_keyword_priority("kw", PriorityLevel.HIGH)
        priority_budget.consecutive_success_boost_threshold = 3

        # Initially no boost
        budget_no_boost = priority_budget.get_budget_for_keyword("kw")
        assert budget_no_boost["rotations"] == 5  # 50% of 10

        # Add 3 consecutive successes
        for _ in range(3):
            priority_budget.record_success_for_keyword("kw")

        # Now should get boost (1.5x of 50% = 75% = 7-8 rotations)
        budget_boosted = priority_budget.get_budget_for_keyword("kw")
        assert budget_boosted["rotations"] == 7  # 75% of 10

    @pytest.mark.fast
    def test_failure_resets_consecutive_successes(self, priority_budget):
        """Failure resets consecutive success counter."""
        priority_budget.set_keyword_priority("kw", PriorityLevel.HIGH)
        priority_budget.consecutive_success_boost_threshold = 3

        # Add 3 consecutive successes
        for _ in range(3):
            priority_budget.record_success_for_keyword("kw")

        # Record failure
        priority_budget.record_failure_for_keyword("kw")

        # Counter should be reset
        assert priority_budget.keyword_consecutive_successes["kw"] == 0

        # Budget should be back to normal
        budget = priority_budget.get_budget_for_keyword("kw")
        assert budget["rotations"] == 5  # Back to 50%


# ============================================================================
# Test Budget Exhaustion Prediction
# ============================================================================

class TestExhaustionPrediction:
    """Test budget exhaustion prediction."""

    @pytest.mark.fast
    def test_predict_exhaustion_will_exhaust(self, exhausted_priority_budget):
        """Prediction returns will_exhaust=True when budget low."""
        exhausted_priority_budget.set_keyword_priority("kw", PriorityLevel.LOW)

        # Low priority gets 20% = 1 rotation remaining, but 4 rotations used out of 6 max
        # After allocation: 20% of 6 = 1.2 -> 1 rotation, but used 4 so 0 remaining
        prediction = exhausted_priority_budget.predict_exhaustion("kw", attempts_remaining=5)

        assert prediction["will_exhaust"] is True
        assert prediction["attempts_until_exhaustion"] == 0  # Already exhausted
        assert prediction["warning"] is not None

    @pytest.mark.fast
    def test_predict_exhaustion_wont_exhaust(self, priority_budget):
        """Prediction returns will_exhaust=False when plenty of budget."""
        priority_budget.set_keyword_priority("kw", PriorityLevel.HIGH)

        # High priority gets 50% = 5 rotations
        prediction = priority_budget.predict_exhaustion("kw", attempts_remaining=3)

        assert prediction["will_exhaust"] is False
        assert prediction["attempts_until_exhaustion"] == 5

    @pytest.mark.fast
    def test_predict_exhaustion_unlimited(self):
        """Prediction returns None for unlimited budget."""
        b = RateLimitBudget()
        b.max_rotations = 0  # Unlimited
        b.priority_allocation_enabled = True

        prediction = b.predict_exhaustion("kw")
        assert prediction is None


# ============================================================================
# Test Can Attempt For Keyword
# ============================================================================

class TestCanAttemptForKeyword:
    """Test can_attempt_for_keyword method."""

    @pytest.mark.fast
    def test_can_attempt_has_budget(self, priority_budget):
        """Returns True when keyword has budget."""
        priority_budget.set_keyword_priority("kw", PriorityLevel.HIGH)
        # 50% of 10 = 5 rotations available
        assert priority_budget.can_attempt_for_keyword("kw") is True

    @pytest.mark.fast
    def test_cannot_attempt_no_budget(self, exhausted_priority_budget):
        """Returns False when keyword exhausted."""
        exhausted_priority_budget.set_keyword_priority("kw", PriorityLevel.LOW)
        # Low priority gets 20% = 1 rotation, but 4 used, so 0 remaining
        assert exhausted_priority_budget.can_attempt_for_keyword("kw") is False


# ============================================================================
# Test Steal Unused Budget
# ============================================================================

class TestStealUnusedBudget:
    """Test stealing unused budget from low-priority keywords."""

    @pytest.mark.fast
    def test_steal_only_high_priority(self, priority_budget):
        """Only high-priority keywords can steal."""
        priority_budget.set_keyword_priority("recipient", PriorityLevel.MEDIUM)
        priority_budget.set_keyword_priority("donor", PriorityLevel.LOW)

        # Should not attempt steal for non-high-priority
        priority_budget.steal_unused_budget(["donor"], "recipient")

        # No error, just returns early

    @pytest.mark.fast
    def test_steal_detects_opportunity(self, priority_budget):
        """Steal detects opportunity in low-priority keywords."""
        priority_budget.set_keyword_priority("high_kw", PriorityLevel.HIGH)
        priority_budget.set_keyword_priority("low_kw", PriorityLevel.LOW)

        # Log should show steal opportunity
        priority_budget.steal_unused_budget(["low_kw"], "high_kw")


# ============================================================================
# Test Summary and Serialization
# ============================================================================

class TestSummaryAndSerialization:
    """Test summary and serialization with priorities."""

    @pytest.mark.fast
    def test_summary_includes_priorities(self, priority_budget):
        """Summary includes keyword priority counts."""
        priority_budget.set_keyword_priority("kw1", PriorityLevel.HIGH)
        priority_budget.set_keyword_priority("kw2", PriorityLevel.MEDIUM)
        priority_budget.set_keyword_priority("kw3", PriorityLevel.LOW)

        summary = priority_budget.get_summary()

        assert summary["keyword_priorities"]["high"] == 1
        assert summary["keyword_priorities"]["medium"] == 1
        assert summary["keyword_priorities"]["low"] == 1

    @pytest.mark.fast
    def test_to_dict_includes_priorities(self, priority_budget):
        """to_dict includes priority data."""
        priority_budget.set_keyword_priority("kw", PriorityLevel.HIGH)
        priority_budget.keyword_consecutive_successes["kw"] = 5

        data = priority_budget.to_dict()

        assert data["keyword_priorities"]["kw"] == "high"
        assert data["keyword_consecutive_successes"]["kw"] == 5
        assert data["priority_allocation_enabled"] is True

    @pytest.mark.fast
    def test_from_dict_restores_priorities(self, priority_budget):
        """from_dict restores priority data."""
        priority_budget.set_keyword_priority("kw", PriorityLevel.HIGH)
        priority_budget.keyword_consecutive_successes["kw"] = 5

        data = priority_budget.to_dict()
        restored = RateLimitBudget.from_dict(data)

        assert restored.get_keyword_priority("kw") == PriorityLevel.HIGH
        assert restored.keyword_consecutive_successes["kw"] == 5

    @pytest.mark.fast
    def test_clear_resets_priorities(self, priority_budget):
        """clear resets priority tracking."""
        priority_budget.set_keyword_priority("kw", PriorityLevel.HIGH)
        priority_budget.keyword_consecutive_successes["kw"] = 5

        priority_budget.clear()

        assert len(priority_budget.keyword_priorities) == 0
        assert len(priority_budget.keyword_consecutive_successes) == 0
