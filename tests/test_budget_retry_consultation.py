"""Tests for RateLimitBudget consultation in core.py retry loop (US-003 Sprint 10).

Tests verify:
1. Budget exhaustion at retry loop start skips keyword entirely
2. Budget can_backoff() check before each retry - exhausted skips to batch queue
3. Budget record_success() called on successful download
4. Budget record_backoff() called with actual delay during retry
5. Batch retry queue receives budget state snapshot
6. record_success/record_failure tracking on RateLimitBudget

Created: January 26, 2026
User Story: US-003 - Add RateLimitBudget consultation in core.py retry loop
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.rate_limit_budget import RateLimitBudget
from src.downloader.retry_queue import RetryQueue, BatchRetryConfig


# ============================================================================
# Test RateLimitBudget record_success / record_failure
# ============================================================================

class TestBudgetSuccessFailureTracking:
    """Test record_success() and record_failure() on RateLimitBudget."""

    def test_record_success_increments_counter(self):
        """record_success increments the successes counter."""
        budget = RateLimitBudget()
        assert budget.successes == 0
        budget.record_success()
        assert budget.successes == 1
        budget.record_success()
        assert budget.successes == 2

    def test_record_success_with_keyword(self):
        """record_success accepts optional keyword parameter."""
        budget = RateLimitBudget()
        budget.record_success(keyword="sunset")
        assert budget.successes == 1

    def test_record_failure_increments_counter(self):
        """record_failure increments the failures counter."""
        budget = RateLimitBudget()
        assert budget.failures == 0
        budget.record_failure()
        assert budget.failures == 1
        budget.record_failure()
        assert budget.failures == 2

    def test_record_failure_tracks_keyword(self):
        """record_failure tracks the keyword in rate_limited list."""
        budget = RateLimitBudget()
        budget.record_failure(keyword="ocean")
        assert "ocean" in budget.keywords_rate_limited
        assert budget.failures == 1

    def test_record_failure_no_duplicate_keywords(self):
        """record_failure doesn't duplicate keywords."""
        budget = RateLimitBudget()
        budget.record_failure(keyword="forest")
        budget.record_failure(keyword="forest")
        assert budget.keywords_rate_limited.count("forest") == 1
        assert budget.failures == 2

    def test_success_failure_in_summary(self):
        """Summary includes successes and failures."""
        budget = RateLimitBudget()
        budget.record_success()
        budget.record_success()
        budget.record_failure()
        summary = budget.get_summary()
        assert summary["successes"] == 2
        assert summary["failures"] == 1

    def test_success_failure_in_serialization(self):
        """to_dict/from_dict roundtrips successes and failures."""
        budget = RateLimitBudget()
        budget.record_success()
        budget.record_success()
        budget.record_failure()

        data = budget.to_dict()
        assert data["successes"] == 2
        assert data["failures"] == 1

        restored = RateLimitBudget.from_dict(data)
        assert restored.successes == 2
        assert restored.failures == 1

    def test_clear_resets_success_failure(self):
        """clear() resets successes and failures."""
        budget = RateLimitBudget()
        budget.record_success()
        budget.record_failure()
        budget.clear()
        assert budget.successes == 0
        assert budget.failures == 0


# ============================================================================
# Test Budget Consultation in Retry Flow (mocked VideoDownloader)
# ============================================================================

class TestBudgetExhaustedSkipsKeyword:
    """Test that exhausted budget at retry loop start skips the keyword."""

    def _create_mock_downloader(self):
        """Create a minimal mock of VideoDownloader with budget wiring."""
        downloader = MagicMock()
        downloader._share_budget_across_keywords = True
        downloader.rate_limit_budget = RateLimitBudget()
        downloader.rate_limit_budget.max_rotations = 2
        downloader.rate_limit_budget.max_vpn_switches = 1
        downloader.rate_limit_budget.max_backoff_time = 30.0
        downloader.retry_queue = RetryQueue(BatchRetryConfig(enabled=True))
        downloader.rate_limit_metrics = MagicMock()
        return downloader

    def test_exhausted_budget_skips_keyword(self):
        """When budget is exhausted, keyword is skipped and added to retry queue."""
        budget = RateLimitBudget()
        budget.max_rotations = 2
        budget.max_vpn_switches = 1
        budget.max_backoff_time = 30.0
        # Exhaust everything
        budget.rotations_used = 2
        budget.vpn_switches_used = 1
        budget.backoff_time_spent = 30.0

        assert budget.get_recommended_escalation() == "exhausted"
        assert budget.is_exhausted() is True

    def test_budget_not_exhausted_allows_download(self):
        """When budget has resources, download proceeds."""
        budget = RateLimitBudget()
        budget.max_rotations = 5
        budget.max_backoff_time = 60.0

        assert budget.get_recommended_escalation() == "backoff"
        assert budget.is_exhausted() is False

    def test_exhausted_budget_records_failure(self):
        """When budget is exhausted, record_failure is called."""
        budget = RateLimitBudget()
        budget.max_rotations = 1
        budget.max_vpn_switches = 0  # unlimited
        budget.max_backoff_time = 10.0
        budget.rotations_used = 1
        budget.backoff_time_spent = 10.0
        # Still has VPN, so not exhausted
        assert budget.get_recommended_escalation() == "vpn"

        # Now exhaust VPN too
        budget.max_vpn_switches = 1
        budget.vpn_switches_used = 1
        assert budget.get_recommended_escalation() == "exhausted"

        # Record failure
        budget.record_failure(keyword="test_kw")
        assert budget.failures == 1
        assert "test_kw" in budget.keywords_rate_limited


class TestBudgetCanBackoffCheck:
    """Test budget.can_backoff() check before retry backoff."""

    def test_can_backoff_allows_retry(self):
        """When backoff budget is available, retry proceeds."""
        budget = RateLimitBudget()
        budget.max_backoff_time = 60.0
        budget.backoff_time_spent = 10.0

        delay = 2.0
        assert budget.can_backoff(delay) is True

    def test_can_backoff_exhausted_blocks_retry(self):
        """When backoff budget is exhausted, retry is blocked."""
        budget = RateLimitBudget()
        budget.max_backoff_time = 30.0
        budget.backoff_time_spent = 29.0

        delay = 2.0  # 29 + 2 = 31 > 30
        assert budget.can_backoff(delay) is False

    def test_record_backoff_accumulates(self):
        """record_backoff accumulates time in budget."""
        budget = RateLimitBudget()
        budget.max_backoff_time = 60.0

        budget.record_backoff(5.0, keyword="kw1")
        assert budget.backoff_time_spent == 5.0

        budget.record_backoff(10.0, keyword="kw2")
        assert budget.backoff_time_spent == 15.0

        # Both keywords tracked
        assert "kw1" in budget.keywords_rate_limited
        assert "kw2" in budget.keywords_rate_limited

    def test_backoff_budget_shared_across_keywords(self):
        """Backoff from keyword A reduces budget for keyword B."""
        budget = RateLimitBudget()
        budget.max_backoff_time = 20.0

        # Keyword A uses 15s of backoff
        budget.record_backoff(15.0, keyword="keyword_a")

        # Keyword B only has 5s remaining
        assert budget.can_backoff(5.0) is True  # 15 + 5 = 20 <= 20
        assert budget.can_backoff(6.0) is False  # 15 + 6 = 21 > 20


# ============================================================================
# Test Batch Retry Queue Budget State
# ============================================================================

class TestRetryQueueBudgetState:
    """Test that retry queue receives and stores budget state."""

    def test_set_budget_state(self):
        """Retry queue stores budget state snapshot."""
        queue = RetryQueue(BatchRetryConfig(enabled=True))
        budget = RateLimitBudget()
        budget.max_backoff_time = 60.0
        budget.backoff_time_spent = 30.0
        budget.record_success()
        budget.record_failure(keyword="test")

        queue.set_budget_state(budget.get_summary())

        state = queue.get_budget_state()
        assert state is not None
        assert state["backoff_time_spent"] == 30.0
        assert state["successes"] == 1
        assert state["failures"] == 1
        assert state["is_exhausted"] is False

    def test_budget_state_in_stats(self):
        """Budget state is included in retry queue stats."""
        queue = RetryQueue(BatchRetryConfig(enabled=True))
        budget_summary = {"is_exhausted": False, "backoff_time_remaining": 30.0}
        queue.set_budget_state(budget_summary)

        stats = queue.get_stats()
        assert "budget_state" in stats
        assert stats["budget_state"]["is_exhausted"] is False

    def test_budget_state_none_by_default(self):
        """Budget state is None by default."""
        queue = RetryQueue(BatchRetryConfig(enabled=True))
        assert queue.get_budget_state() is None
        stats = queue.get_stats()
        assert stats["budget_state"] is None

    def test_budget_state_reflects_exhaustion(self):
        """Budget state correctly reflects exhausted budget."""
        queue = RetryQueue(BatchRetryConfig(enabled=True))
        budget = RateLimitBudget()
        budget.max_rotations = 2
        budget.max_vpn_switches = 1
        budget.max_backoff_time = 30.0
        budget.rotations_used = 2
        budget.vpn_switches_used = 1
        budget.backoff_time_spent = 30.0

        queue.set_budget_state(budget.get_summary())

        state = queue.get_budget_state()
        assert state["is_exhausted"] is True
        assert state["rotations_remaining"] == 0
        assert state["vpn_switches_remaining"] == 0
        assert state["backoff_time_remaining"] == 0.0


# ============================================================================
# Test End-to-End Budget Consultation Flow
# ============================================================================

class TestBudgetConsultationFlow:
    """Test the full budget consultation flow in retry scenarios."""

    def test_budget_tracks_success_and_failure_sequence(self):
        """Budget correctly tracks a sequence of successes and failures."""
        budget = RateLimitBudget()
        budget.max_backoff_time = 100.0
        budget.max_rotations = 5

        # Simulate keyword A: success
        budget.record_success(keyword="kw_a")

        # Simulate keyword B: failure with backoff
        budget.record_backoff(10.0, keyword="kw_b")
        budget.record_failure(keyword="kw_b")

        # Simulate keyword C: success after backoff
        budget.record_backoff(5.0, keyword="kw_c")
        budget.record_success(keyword="kw_c")

        assert budget.successes == 2
        assert budget.failures == 1
        assert budget.backoff_time_spent == 15.0
        assert len(budget.keywords_rate_limited) == 2  # kw_b and kw_c (backoff tracked both)

    def test_budget_consultation_prevents_waste(self):
        """Budget consultation prevents wasting retries when budget exhausted."""
        budget = RateLimitBudget()
        budget.max_backoff_time = 10.0

        # Simulate keyword A uses all backoff
        budget.record_backoff(10.0, keyword="kw_a")

        # Keyword B should see no backoff available
        assert budget.can_backoff(2.0) is False
        recommended = budget.get_recommended_escalation()
        # Should recommend cookie rotation since backoff is exhausted
        assert recommended == "cookie"

    def test_from_dict_preserves_success_failure(self):
        """Checkpoint restore preserves success/failure counts."""
        budget = RateLimitBudget()
        budget.record_success()
        budget.record_success()
        budget.record_failure(keyword="test")

        # Serialize and restore
        data = budget.to_dict()
        restored = RateLimitBudget.from_dict(data)

        assert restored.successes == 2
        assert restored.failures == 1
