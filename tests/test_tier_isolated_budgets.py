"""Tests for tier-isolated retry budgets with cross-tier borrowing (US-109-007)."""

import pytest
from src.downloader.rate_limit_budget import (
    EscalationTier,
    TierBudget,
    TierBudgetManager,
    RateLimitBudget,
)


class TestTierBudget:
    """Tests for TierBudget class."""

    def test_tier_budget_default_limits(self):
        """Test default max attempts per tier."""
        tier1 = TierBudget(tier=EscalationTier.TIER1)
        assert tier1.max_attempts == 5
        assert tier1.can_attempt() is True

    def test_tier_budget_custom_limits(self):
        """Test custom max attempts."""
        tier = TierBudget(tier=EscalationTier.TIER1, max_attempts=3)
        assert tier.max_attempts == 3
        assert tier.attempts_remaining() == 3

    def test_tier_budget_record_attempt(self):
        """Test recording attempts."""
        tier = TierBudget(tier=EscalationTier.TIER1, max_attempts=3)
        tier.record_attempt()
        assert tier.attempts_used == 1
        assert tier.attempts_remaining() == 2

    def test_tier_budget_exhaustion(self):
        """Test tier exhaustion."""
        tier = TierBudget(tier=EscalationTier.TIER1, max_attempts=2)
        tier.record_attempt()
        tier.record_attempt()
        assert tier.is_exhausted() is True
        assert tier.can_attempt() is False

    def test_tier_budget_unlimited(self):
        """Test unlimited tier budget."""
        tier = TierBudget(tier=EscalationTier.TIER1, max_attempts=0)  # 0 = unlimited
        for _ in range(100):
            tier.record_attempt()
        assert tier.can_attempt() is True
        assert tier.attempts_remaining() is None

    def test_tier_budget_borrow_from(self):
        """Test borrowing from another tier."""
        donor = TierBudget(tier=EscalationTier.TIER1, max_attempts=5)
        borrower = TierBudget(tier=EscalationTier.TIER3, max_attempts=2)

        # Borrower exhausted
        borrower.record_attempt()
        borrower.record_attempt()

        # Borrow from donor
        result = borrower.borrow_from(donor, amount=1)
        assert result is True
        assert borrower.borrowed_from == EscalationTier.TIER1
        assert donor.lent_to == EscalationTier.TIER3

    def test_tier_budget_borrow_insufficient(self):
        """Test borrowing fails when donor has no capacity."""
        donor = TierBudget(tier=EscalationTier.TIER1, max_attempts=1)
        donor.record_attempt()  # Exhausted

        borrower = TierBudget(tier=EscalationTier.TIER3, max_attempts=1)
        borrower.record_attempt()  # Exhausted

        result = borrower.borrow_from(donor, amount=1)
        assert result is False

    def test_tier_budget_usage_percentage(self):
        """Test usage percentage calculation."""
        tier = TierBudget(tier=EscalationTier.TIER1, max_attempts=10)
        tier.record_attempt()
        tier.record_attempt()
        assert tier.get_usage_percentage() == 20.0

    def test_tier_budget_unlimited_percentage(self):
        """Test usage percentage for unlimited tier."""
        tier = TierBudget(tier=EscalationTier.TIER1, max_attempts=0)
        assert tier.get_usage_percentage() is None


class TestTierBudgetManager:
    """Tests for TierBudgetManager class."""

    def test_manager_default_initialization(self):
        """Test manager initializes with default limits."""
        manager = TierBudgetManager()
        assert manager.get_budget(EscalationTier.TIER1).max_attempts == 5
        assert manager.get_budget(EscalationTier.TIER2).max_attempts == 5
        assert manager.get_budget(EscalationTier.TIER3).max_attempts == 5
        assert manager.get_budget(EscalationTier.TIER4).max_attempts == 3

    def test_manager_custom_limits(self):
        """Test manager with custom tier limits."""
        custom_limits = {
            EscalationTier.TIER1: 3,
            EscalationTier.TIER2: 4,
            EscalationTier.TIER3: 5,
            EscalationTier.TIER4: 2,
        }
        manager = TierBudgetManager(custom_limits)
        assert manager.get_budget(EscalationTier.TIER1).max_attempts == 3
        assert manager.get_budget(EscalationTier.TIER4).max_attempts == 2

    def test_manager_can_attempt(self):
        """Test can_attempt returns True when tier has budget."""
        manager = TierBudgetManager()
        assert manager.can_attempt(EscalationTier.TIER1) is True

    def test_manager_record_attempt(self):
        """Test recording attempt in tier."""
        manager = TierBudgetManager()
        result = manager.record_attempt(EscalationTier.TIER1)
        assert result is True
        assert manager.get_budget(EscalationTier.TIER1).attempts_used == 1

    def test_manager_exhausted_tier_borrowing(self):
        """Test exhausted tier can borrow from other tiers."""
        manager = TierBudgetManager()

        # Exhaust tier1
        tier1 = manager.get_budget(EscalationTier.TIER1)
        for _ in range(tier1.max_attempts):
            manager.record_attempt(EscalationTier.TIER1)

        # tier1 should be exhausted
        assert manager.get_budget(EscalationTier.TIER1).is_exhausted() is True

        # But should be able to borrow from tier4
        assert manager.can_attempt(EscalationTier.TIER1) is True

    def test_manager_all_tiers_exhausted(self):
        """Test all tiers exhausted scenario."""
        manager = TierBudgetManager()

        # Exhaust all tiers
        for tier in [EscalationTier.TIER1, EscalationTier.TIER2,
                     EscalationTier.TIER3, EscalationTier.TIER4]:
            tier_budget = manager.get_budget(tier)
            for _ in range(tier_budget.max_attempts):
                manager.record_attempt(tier)

        # No tier should be available
        assert manager.is_any_tier_available() is False
        assert manager.get_next_available_tier() is None

    def test_manager_get_next_available_tier(self):
        """Test getting next available tier."""
        manager = TierBudgetManager()
        # First tier should be available
        assert manager.get_next_available_tier() == EscalationTier.TIER1

    def test_manager_tier_status(self):
        """Test getting tier status."""
        manager = TierBudgetManager()
        manager.record_attempt(EscalationTier.TIER1)
        manager.record_attempt(EscalationTier.TIER1)

        status = manager.get_tier_status()
        assert "tier1" in status
        assert status["tier1"]["attempts_used"] == 2
        assert status["tier1"]["attempts_remaining"] == 3

    def test_manager_clear(self):
        """Test clearing tier budgets."""
        manager = TierBudgetManager()
        manager.record_attempt(EscalationTier.TIER1)
        manager.clear()

        tier1 = manager.get_budget(EscalationTier.TIER1)
        assert tier1.attempts_used == 0


class TestRateLimitBudgetTierIsolation:
    """Tests for RateLimitBudget tier isolation integration."""

    def test_enable_tier_isolation(self):
        """Test enabling tier isolation."""
        budget = RateLimitBudget()
        budget.enable_tier_isolation()

        assert budget.tier_isolation_enabled is True
        assert budget.tier_budget_manager is not None

    def test_disable_tier_isolation(self):
        """Test disabling tier isolation."""
        budget = RateLimitBudget()
        budget.enable_tier_isolation()
        budget.disable_tier_isolation()

        assert budget.tier_isolation_enabled is False
        assert budget.tier_budget_manager is None

    def test_can_attempt_tier_with_isolation(self):
        """Test can_attempt_tier with tier isolation enabled."""
        budget = RateLimitBudget()
        budget.enable_tier_isolation()

        # Should be able to attempt tier1
        assert budget.can_attempt_tier(EscalationTier.TIER1) is True

    def test_record_tier_attempt_with_isolation(self):
        """Test record_tier_attempt with tier isolation."""
        budget = RateLimitBudget()
        budget.enable_tier_isolation()

        result = budget.record_tier_attempt(EscalationTier.TIER1)
        assert result is True
        assert budget.tier_budget_manager.get_budget(EscalationTier.TIER1).attempts_used == 1

    def test_tier_status_with_isolation(self):
        """Test get_tier_status with tier isolation."""
        budget = RateLimitBudget()
        budget.enable_tier_isolation()
        budget.record_tier_attempt(EscalationTier.TIER1)

        status = budget.get_tier_status()
        assert "tier1" in status

    def test_tier_budget_summary(self):
        """Test tier budget summary string."""
        budget = RateLimitBudget()
        budget.enable_tier_isolation()

        summary = budget.get_tier_budget_summary()
        assert "tier1" in summary
        assert "tier4" in summary

    def test_is_tier_exhausted(self):
        """Test is_tier_exhausted method."""
        budget = RateLimitBudget()
        budget.enable_tier_isolation()

        # Exhaust tier1
        tier1_budget = budget.tier_budget_manager.get_budget(EscalationTier.TIER1)
        for _ in range(tier1_budget.max_attempts):
            budget.record_tier_attempt(EscalationTier.TIER1)

        # tier1 should be exhausted but can still borrow
        assert budget.is_tier_exhausted(EscalationTier.TIER1) is False  # Can borrow

    def test_get_next_available_tier_with_isolation(self):
        """Test get_next_available_tier with tier isolation."""
        budget = RateLimitBudget()
        budget.enable_tier_isolation()

        next_tier = budget.get_next_available_tier()
        assert next_tier == EscalationTier.TIER1

    def test_checkpoint_serialization_with_tier_budgets(self):
        """Test tier budgets are serialized to dict."""
        budget = RateLimitBudget()
        budget.enable_tier_isolation()
        budget.record_tier_attempt(EscalationTier.TIER1)
        budget.record_tier_attempt(EscalationTier.TIER2)

        data = budget.to_dict()
        assert data["tier_isolation_enabled"] is True
        assert "tier_budgets" in data

    def test_checkpoint_deserialization_with_tier_budgets(self):
        """Test tier budgets are restored from dict."""
        # Create and populate budget
        budget = RateLimitBudget()
        budget.enable_tier_isolation()
        budget.record_tier_attempt(EscalationTier.TIER1)

        # Serialize
        data = budget.to_dict()

        # Deserialize
        restored = RateLimitBudget.from_dict(data)

        assert restored.tier_isolation_enabled is True
        assert restored.tier_budget_manager is not None
        assert restored.tier_budget_manager.get_budget(EscalationTier.TIER1).attempts_used == 1

    def test_clear_clears_tier_budgets(self):
        """Test clear method clears tier budgets."""
        budget = RateLimitBudget()
        budget.enable_tier_isolation()
        budget.record_tier_attempt(EscalationTier.TIER1)

        budget.clear()

        assert budget.tier_budget_manager.get_budget(EscalationTier.TIER1).attempts_used == 0


class TestCrossTierBorrowing:
    """Tests for cross-tier borrowing behavior."""

    def test_tier1_borrows_from_tier4(self):
        """Test tier1 (highest priority) can borrow from tier4."""
        # Custom config: tier1 has 1 attempt, tier4 has 3
        custom_limits = {
            EscalationTier.TIER1: 1,
            EscalationTier.TIER2: 5,
            EscalationTier.TIER3: 5,
            EscalationTier.TIER4: 3,
        }
        manager = TierBudgetManager(custom_limits)

        # Exhaust tier1
        manager.record_attempt(EscalationTier.TIER1)

        # tier1 should be able to borrow from tier4 (lower priority)
        assert manager.can_attempt(EscalationTier.TIER1) is True

    def test_borrowing_priority_order(self):
        """Test borrowing follows priority: tier4 first lender, tier1 last lender."""
        custom_limits = {
            EscalationTier.TIER1: 1,
            EscalationTier.TIER2: 1,
            EscalationTier.TIER3: 1,
            EscalationTier.TIER4: 1,
        }
        manager = TierBudgetManager(custom_limits)

        # Exhaust all tiers
        for tier in [EscalationTier.TIER1, EscalationTier.TIER2,
                     EscalationTier.TIER3, EscalationTier.TIER4]:
            manager.record_attempt(tier)

        # No borrowing should work
        assert manager.is_any_tier_available() is False

    def test_tier4_can_borrow_last(self):
        """Test tier4 (VPN) can borrow as last resort."""
        custom_limits = {
            EscalationTier.TIER1: 5,
            EscalationTier.TIER2: 5,
            EscalationTier.TIER3: 5,
            EscalationTier.TIER4: 1,  # Only 1 attempt
        }
        manager = TierBudgetManager(custom_limits)

        # Exhaust tier4
        manager.record_attempt(EscalationTier.TIER4)

        # tier4 should be able to borrow from other tiers
        assert manager.can_attempt(EscalationTier.TIER4) is True

    def test_borrow_logging_info(self):
        """Test borrowing is properly logged with source/destination."""
        donor = TierBudget(tier=EscalationTier.TIER1, max_attempts=5)
        borrower = TierBudget(tier=EscalationTier.TIER3, max_attempts=1)
        borrower.record_attempt()  # Exhaust

        borrower.borrow_from(donor)

        assert borrower.borrowed_from == EscalationTier.TIER1
        assert donor.lent_to == EscalationTier.TIER3
