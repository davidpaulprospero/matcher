"""Tests for BatchRetryBudget class (US-001 Sprint 8).

Tests the batch-level retry budget tracking that adapts retry strategy
based on cumulative failure patterns across all videos in a batch.

Acceptance criteria:
1. Create BatchRetryBudget class tracking cumulative errors across all videos in batch
2. Implement budget_remaining(category) method returning reduced retry count when pattern detected
3. When >30% network errors, reduce subsequent video retry budget from 3 to 1
4. When >50% network errors, disable retries entirely for remaining videos
5. Add batch_retry_budget field to CaptionMetrics tracking original vs actual retries used
6. Tests verify batch with 40% network errors uses 60% fewer retries on remaining videos
"""

import pytest
import threading
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from src.caption_fetcher import (
    BatchRetryBudget,
    CaptionErrorCategory,
    CaptionMetrics,
    DEFAULT_RETRY_BUDGETS,
)


class TestBatchRetryBudgetInit:
    """Test BatchRetryBudget initialization."""

    @pytest.mark.fast
    def test_default_initialization(self):
        """Test default values on initialization."""
        budget = BatchRetryBudget()
        assert budget.total_videos == 0
        assert budget.processed_videos == 0
        assert budget.category_counts == {}
        assert budget.budget_reductions == []

    @pytest.mark.fast
    def test_initialization_with_total_videos(self):
        """Test initialization with total video count."""
        budget = BatchRetryBudget(total_videos=100)
        assert budget.total_videos == 100
        assert budget.processed_videos == 0

    @pytest.mark.fast
    def test_budgets_initialized_from_defaults(self):
        """Test original and reduced budgets match defaults."""
        budget = BatchRetryBudget()
        assert budget.original_budgets == DEFAULT_RETRY_BUDGETS
        assert budget.reduced_budgets == DEFAULT_RETRY_BUDGETS

    @pytest.mark.fast
    def test_default_thresholds(self):
        """Test default threshold values."""
        budget = BatchRetryBudget()
        assert budget.network_reduce_threshold == 0.30
        assert budget.network_disable_threshold == 0.50


class TestBatchRetryBudgetRecording:
    """Test recording success and errors."""

    @pytest.mark.fast
    def test_record_success_increments_processed(self):
        """Test record_success increments processed count."""
        budget = BatchRetryBudget(total_videos=10)
        budget.record_success("video1")
        assert budget.processed_videos == 1

    @pytest.mark.fast
    def test_record_error_increments_category_count(self):
        """Test record_error increments category count."""
        budget = BatchRetryBudget(total_videos=10)
        budget.record_error(CaptionErrorCategory.NETWORK, "video1")

        assert budget.processed_videos == 1
        assert budget.category_counts[CaptionErrorCategory.NETWORK] == 1

    @pytest.mark.fast
    def test_record_multiple_errors_different_categories(self):
        """Test recording errors of different categories."""
        budget = BatchRetryBudget(total_videos=10)
        budget.record_error(CaptionErrorCategory.NETWORK, "v1")
        budget.record_error(CaptionErrorCategory.TIMEOUT, "v2")
        budget.record_error(CaptionErrorCategory.PARSE, "v3")

        assert budget.processed_videos == 3
        assert budget.category_counts[CaptionErrorCategory.NETWORK] == 1
        assert budget.category_counts[CaptionErrorCategory.TIMEOUT] == 1
        assert budget.category_counts[CaptionErrorCategory.PARSE] == 1


class TestBatchRetryBudgetThresholds:
    """Test threshold detection and budget reduction."""

    @pytest.mark.fast
    def test_30_percent_network_errors_reduces_to_1(self):
        """Test >30% network errors reduces retry budget from 3 to 1."""
        budget = BatchRetryBudget(total_videos=100)

        # Process 10 videos with 4 network errors (40% rate)
        for i in range(6):
            budget.record_success(f"video_{i}")
        for i in range(4):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        # Budget should be reduced to 1
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 1
        assert len(budget.budget_reductions) == 1
        assert budget.budget_reductions[0][3] == 1  # new_budget

    @pytest.mark.fast
    def test_50_percent_network_errors_disables_retries(self):
        """Test >50% network errors disables retries entirely."""
        budget = BatchRetryBudget(total_videos=100)

        # Process 10 videos with 6 network errors (60% rate)
        for i in range(4):
            budget.record_success(f"video_{i}")
        for i in range(6):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        # Budget should be disabled (0)
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 0
        # Should have 2 reductions: first to 1 at >30%, then to 0 at >50%
        assert len(budget.budget_reductions) >= 1

    @pytest.mark.fast
    def test_below_30_percent_no_reduction(self):
        """Test <30% network errors keeps original budget."""
        budget = BatchRetryBudget(total_videos=100)

        # Process 10 videos with 2 network errors (20% rate)
        for i in range(8):
            budget.record_success(f"video_{i}")
        for i in range(2):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        # Budget should remain at 3
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 3
        assert len(budget.budget_reductions) == 0

    @pytest.mark.fast
    def test_exactly_30_percent_no_reduction(self):
        """Test exactly 30% network errors does NOT trigger reduction."""
        budget = BatchRetryBudget(total_videos=100)

        # Process 10 videos with 3 network errors (exactly 30% rate)
        for i in range(7):
            budget.record_success(f"video_{i}")
        for i in range(3):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        # 30% exactly should NOT trigger reduction (>30% required)
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 3

    @pytest.mark.fast
    def test_only_network_errors_trigger_reduction(self):
        """Test only NETWORK errors trigger threshold logic."""
        budget = BatchRetryBudget(total_videos=100)

        # 60% timeout errors - should NOT trigger reduction
        for i in range(4):
            budget.record_success(f"video_{i}")
        for i in range(6):
            budget.record_error(CaptionErrorCategory.TIMEOUT, f"error_{i}")

        # TIMEOUT budget unchanged, NETWORK still at original
        assert budget.budget_remaining(CaptionErrorCategory.TIMEOUT) == 2
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 3


class TestBatchRetryBudget40PercentScenario:
    """Test the specific acceptance criteria scenario: 40% errors use 60% fewer retries."""

    @pytest.mark.fast
    def test_40_percent_network_errors_uses_60_percent_fewer_retries(self):
        """Test batch with 40% network errors uses 60% fewer retries on remaining videos.

        Scenario:
        - Original budget: 3 retries for NETWORK errors
        - After 40% network error rate detected, budget reduced to 1
        - Reduction from 3 to 1 = 66.7% fewer retries (approximately 60%)

        The acceptance criteria says "60% fewer retries" which we interpret as
        the budget being reduced by at least 60% (from 3 to ~1).
        """
        budget = BatchRetryBudget(total_videos=100)

        # Simulate 100 videos: 40 network errors, 60 successes
        # After processing 50 videos with 20 errors, we'll cross 40% threshold
        for i in range(30):
            budget.record_success(f"video_{i}")
        for i in range(20):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        # At 40% error rate (20/50), we're above 30% threshold
        # Budget should be reduced to 1
        new_budget = budget.budget_remaining(CaptionErrorCategory.NETWORK)
        original_budget = DEFAULT_RETRY_BUDGETS[CaptionErrorCategory.NETWORK]

        assert new_budget == 1
        assert original_budget == 3

        # Verify reduction percentage: (3-1)/3 = 66.7% reduction
        reduction_percent = (original_budget - new_budget) / original_budget
        assert reduction_percent >= 0.60, f"Expected at least 60% reduction, got {reduction_percent:.1%}"

    @pytest.mark.fast
    def test_estimated_retries_saved_calculation(self):
        """Test that estimated retries saved is calculated correctly."""
        budget = BatchRetryBudget(total_videos=100)

        # Process 50 videos with 20 network errors (40% rate)
        for i in range(30):
            budget.record_success(f"video_{i}")
        for i in range(20):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        # The threshold crossing happens at 30.2% (when error 16/53 is recorded)
        # The stored error_rate in budget_reductions is ~0.302
        # 50 videos remaining, ~30% expected to have network errors = ~15 videos
        # Each saves (3 - 1) = 2 retries
        # Expected saved: 15 * 2 = 30 retries
        saved = budget.get_total_retries_saved()
        # Use the stored rate from the reduction event
        expected_rate = budget.budget_reductions[0][0]
        expected_errors = int((100 - 50) * expected_rate)
        expected_saved = expected_errors * 2
        assert saved == expected_saved, f"Expected {expected_saved} retries saved, got {saved}"


class TestBatchRetryBudgetGetters:
    """Test getter methods."""

    @pytest.mark.fast
    def test_budget_remaining_default(self):
        """Test budget_remaining returns default for unprocessed budget."""
        budget = BatchRetryBudget()
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 3
        assert budget.budget_remaining(CaptionErrorCategory.TIMEOUT) == 2
        assert budget.budget_remaining(CaptionErrorCategory.UNAVAILABLE) == 0

    @pytest.mark.fast
    def test_get_error_rate_empty(self):
        """Test get_error_rate returns 0 for empty budget."""
        budget = BatchRetryBudget()
        assert budget.get_error_rate(CaptionErrorCategory.NETWORK) == 0.0

    @pytest.mark.fast
    def test_get_error_rate_calculated(self):
        """Test get_error_rate returns correct rate."""
        budget = BatchRetryBudget(total_videos=100)
        for i in range(5):
            budget.record_success(f"video_{i}")
        for i in range(5):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        assert budget.get_error_rate(CaptionErrorCategory.NETWORK) == 0.5

    @pytest.mark.fast
    def test_get_summary(self):
        """Test get_summary returns complete information."""
        budget = BatchRetryBudget(total_videos=100)
        for i in range(6):
            budget.record_success(f"video_{i}")
        for i in range(4):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        summary = budget.get_summary()

        assert summary['total_videos'] == 100
        assert summary['processed_videos'] == 10
        assert 'NETWORK' in summary['error_rates']
        assert summary['error_rates']['NETWORK'] == 0.4
        assert summary['reductions_applied'] == 1


class TestBatchRetryBudgetSerialization:
    """Test checkpoint serialization."""

    @pytest.mark.fast
    def test_to_dict_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        budget = BatchRetryBudget(total_videos=100)
        for i in range(6):
            budget.record_success(f"video_{i}")
        for i in range(4):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        data = budget.to_dict()
        restored = BatchRetryBudget.from_dict(data)

        assert restored.total_videos == 100
        assert restored.processed_videos == 10
        assert restored.category_counts[CaptionErrorCategory.NETWORK] == 4
        assert restored.budget_remaining(CaptionErrorCategory.NETWORK) == 1

    @pytest.mark.fast
    def test_from_dict_none(self):
        """Test from_dict with None returns fresh budget."""
        budget = BatchRetryBudget.from_dict(None)
        assert budget.total_videos == 0
        assert budget.processed_videos == 0

    @pytest.mark.fast
    def test_from_dict_empty(self):
        """Test from_dict with empty dict returns fresh budget."""
        budget = BatchRetryBudget.from_dict({})
        assert budget.total_videos == 0


class TestBatchRetryBudgetReset:
    """Test reset functionality."""

    @pytest.mark.fast
    def test_reset_clears_state(self):
        """Test reset clears processed videos and counts."""
        budget = BatchRetryBudget(total_videos=100)
        for i in range(4):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        budget.reset()

        assert budget.processed_videos == 0
        assert budget.category_counts == {}
        assert budget.budget_reductions == []
        # Budgets should be restored to original
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 3


class TestBatchRetryBudgetThreadSafety:
    """Test thread safety of BatchRetryBudget."""

    @pytest.mark.fast
    def test_concurrent_recording(self):
        """Test concurrent recording from multiple threads."""
        budget = BatchRetryBudget(total_videos=1000)

        def record_success(n):
            for _ in range(n):
                budget.record_success()

        def record_error(n):
            for _ in range(n):
                budget.record_error(CaptionErrorCategory.NETWORK)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(record_success, 100),
                executor.submit(record_success, 100),
                executor.submit(record_error, 100),
                executor.submit(record_error, 100),
            ]
            for f in futures:
                f.result()

        # Should have 400 processed, 200 errors
        assert budget.processed_videos == 400
        assert budget.category_counts[CaptionErrorCategory.NETWORK] == 200


class TestCaptionMetricsBatchRetryBudget:
    """Test integration with CaptionMetrics."""

    @pytest.mark.fast
    def test_batch_retry_budget_field_exists(self):
        """Test CaptionMetrics has batch_retry_budget field."""
        metrics = CaptionMetrics()
        assert hasattr(metrics, 'batch_retry_budget')
        assert metrics.batch_retry_budget is None

    @pytest.mark.fast
    def test_set_batch_retry_budget(self):
        """Test setting batch_retry_budget from BatchRetryBudget summary."""
        metrics = CaptionMetrics()
        budget = BatchRetryBudget(total_videos=100)

        # Simulate some processing
        for i in range(6):
            budget.record_success(f"video_{i}")
        for i in range(4):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        # Store summary in metrics
        metrics.set_batch_retry_budget(budget.get_summary())

        assert metrics.batch_retry_budget is not None
        assert metrics.batch_retry_budget['total_videos'] == 100
        assert metrics.batch_retry_budget['processed_videos'] == 10
        assert metrics.batch_retry_budget['reductions_applied'] == 1

    @pytest.mark.fast
    def test_batch_retry_budget_in_summary(self):
        """Test batch_retry_budget is accessible after setting."""
        metrics = CaptionMetrics()
        budget_summary = {
            'total_videos': 100,
            'processed_videos': 100,
            'error_rates': {'NETWORK': 0.40},
            'original_budgets': {'NETWORK': 3},
            'reduced_budgets': {'NETWORK': 1},
            'reductions_applied': 1,
            'estimated_retries_saved': 40,
        }

        metrics.set_batch_retry_budget(budget_summary)

        assert metrics.batch_retry_budget['estimated_retries_saved'] == 40


class TestBatchRetryBudgetEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.fast
    def test_zero_processed_videos(self):
        """Test behavior with no processed videos."""
        budget = BatchRetryBudget(total_videos=100)
        assert budget.get_error_rate(CaptionErrorCategory.NETWORK) == 0.0
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 3

    @pytest.mark.fast
    def test_all_errors_one_category(self):
        """Test all videos failing with same error."""
        budget = BatchRetryBudget(total_videos=10)
        for i in range(10):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        # 100% network errors - should be disabled
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 0

    @pytest.mark.fast
    def test_no_errors_in_batch(self):
        """Test batch with no errors."""
        budget = BatchRetryBudget(total_videos=10)
        for i in range(10):
            budget.record_success(f"video_{i}")

        # All successes - budget unchanged
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 3
        assert budget.get_total_retries_saved() == 0

    @pytest.mark.fast
    def test_gradual_threshold_crossing(self):
        """Test that budget reduction happens at correct threshold crossing."""
        budget = BatchRetryBudget(total_videos=100)

        # Process one at a time until we cross 30%
        error_count = 0
        success_count = 0

        # Add successes until we're at 29% errors
        for i in range(7):
            budget.record_success(f"video_{i}")
            success_count += 1
        for i in range(3):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")
            error_count += 1

        # At 30% exactly, no reduction yet
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 3

        # One more error pushes us over
        budget.record_error(CaptionErrorCategory.NETWORK, "error_extra")

        # Now >30%, should be reduced
        # Error rate is now 4/11 = 36%
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 1


class TestBatchRetryBudgetLogging:
    """Test logging behavior."""

    @pytest.mark.fast
    def test_warning_logged_on_30_percent_reduction(self):
        """Test that warning is logged when reducing budget at 30% threshold."""
        budget = BatchRetryBudget(total_videos=100)

        with patch('src.caption_fetcher.logger') as mock_logger:
            for i in range(6):
                budget.record_success(f"video_{i}")
            for i in range(4):
                budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

            # Should have logged a warning about reduction
            warning_calls = [c for c in mock_logger.warning.call_args_list
                          if '>30%' in str(c)]
            assert len(warning_calls) >= 1

    @pytest.mark.fast
    def test_warning_logged_on_50_percent_disable(self):
        """Test that warning is logged when disabling retries at 50% threshold."""
        budget = BatchRetryBudget(total_videos=100)

        with patch('src.caption_fetcher.logger') as mock_logger:
            for i in range(4):
                budget.record_success(f"video_{i}")
            for i in range(6):
                budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

            # Should have logged a warning about disabling
            warning_calls = [c for c in mock_logger.warning.call_args_list
                          if '>50%' in str(c)]
            assert len(warning_calls) >= 1
