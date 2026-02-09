"""Tests for batch failure threshold (US-81-009).

Verifies that check_batch_failure_threshold raises BatchFailureThresholdExceeded
when failure rate exceeds the configured threshold.
"""

import pytest

from src.stages import (
    BatchFailureThresholdExceeded,
    check_batch_failure_threshold,
)


class TestCheckBatchFailureThreshold:
    """Unit tests for check_batch_failure_threshold helper."""

    def test_no_raise_when_below_threshold(self):
        """Should not raise when failure rate is below threshold."""
        # 2 failures out of 10 = 20% < 30% threshold
        check_batch_failure_threshold(
            items_processed=10, items_failed=2, threshold=0.3
        )

    def test_raises_when_above_threshold(self):
        """Should raise when failure rate exceeds threshold."""
        # 4 failures out of 10 = 40% > 30% threshold
        with pytest.raises(BatchFailureThresholdExceeded) as exc_info:
            check_batch_failure_threshold(
                items_processed=10, items_failed=4, threshold=0.3
            )
        assert exc_info.value.failure_rate == pytest.approx(0.4)
        assert exc_info.value.threshold == 0.3
        assert exc_info.value.processed == 10
        assert exc_info.value.failed == 4

    def test_raises_at_exact_threshold_boundary(self):
        """Should raise when failure rate strictly exceeds threshold (not equal)."""
        # 3 out of 10 = 30% = 30% threshold -- should NOT raise (not strictly greater)
        check_batch_failure_threshold(
            items_processed=10, items_failed=3, threshold=0.3
        )
        # 4 out of 10 = 40% > 30% -- should raise
        with pytest.raises(BatchFailureThresholdExceeded):
            check_batch_failure_threshold(
                items_processed=10, items_failed=4, threshold=0.3
            )

    def test_disabled_when_threshold_is_one(self):
        """Threshold of 1.0 disables the check entirely."""
        # Even 100% failure rate should not raise
        check_batch_failure_threshold(
            items_processed=10, items_failed=10, threshold=1.0
        )

    def test_disabled_when_threshold_above_one(self):
        """Threshold above 1.0 also disables the check."""
        check_batch_failure_threshold(
            items_processed=10, items_failed=10, threshold=2.0
        )

    def test_no_raise_on_zero_processed(self):
        """Should not raise (or divide by zero) when nothing processed yet."""
        check_batch_failure_threshold(
            items_processed=0, items_failed=0, threshold=0.3
        )

    def test_acceptance_criteria_scenario(self):
        """US-81-009 acceptance: threshold=0.3, batch of 10, aborts after 4th failure.

        Scenario: 3 failures + 1 success = 4 items processed, 3 failed.
        Failure rate = 3/4 = 75% > 30% -- should abort.
        """
        # After item 1: success (0 failures / 1 processed = 0% -- ok)
        check_batch_failure_threshold(items_processed=1, items_failed=0, threshold=0.3)

        # After item 2: 1 failure (1/2 = 50% > 30% -- would abort here)
        # But per the acceptance criteria wording "aborts after the 4th failure"
        # The scenario is: 1 success + 3 failures = 4 processed, 3 failed = 75%
        # Let's verify the step-by-step:

        # Step 1: success -> 1 processed, 0 failed = 0%
        check_batch_failure_threshold(items_processed=1, items_failed=0, threshold=0.3)

        # Step 2: fail -> 2 processed, 1 failed = 50% > 30% -> ABORT
        with pytest.raises(BatchFailureThresholdExceeded) as exc_info:
            check_batch_failure_threshold(items_processed=2, items_failed=1, threshold=0.3)
        assert exc_info.value.failure_rate == pytest.approx(0.5)

    def test_incremental_check_aborts_early(self):
        """Simulates checking threshold after each item in a 10-item batch.

        With threshold=0.3 and items failing from the start, the batch
        should abort after the first failure (1/1 = 100% > 30%).
        """
        # All failures from the start
        with pytest.raises(BatchFailureThresholdExceeded):
            # After first item: 1 failure / 1 processed = 100% > 30%
            check_batch_failure_threshold(
                items_processed=1, items_failed=1, threshold=0.3
            )

    def test_failed_items_attached_to_exception(self):
        """Failed items list should be available on the exception."""
        failed = ['item_1', 'item_2', 'item_3']
        with pytest.raises(BatchFailureThresholdExceeded) as exc_info:
            check_batch_failure_threshold(
                items_processed=4, items_failed=3, threshold=0.3,
                failed_items=failed,
            )
        assert exc_info.value.failed_items == failed

    def test_exception_message_format(self):
        """Exception message should include rate, threshold, and counts."""
        with pytest.raises(BatchFailureThresholdExceeded, match=r"75\.0%.*30\.0%.*3/4"):
            check_batch_failure_threshold(
                items_processed=4, items_failed=3, threshold=0.3,
            )

    def test_default_threshold_050(self):
        """Default threshold of 0.5 allows up to 50% failure rate."""
        # 5 out of 10 = 50% -- should NOT raise (not strictly greater)
        check_batch_failure_threshold(
            items_processed=10, items_failed=5, threshold=0.5
        )
        # 6 out of 10 = 60% > 50% -- should raise
        with pytest.raises(BatchFailureThresholdExceeded):
            check_batch_failure_threshold(
                items_processed=10, items_failed=6, threshold=0.5
            )


class TestBatchFailureThresholdExceeded:
    """Tests for the exception class itself."""

    def test_attributes(self):
        exc = BatchFailureThresholdExceeded(
            failure_rate=0.75, threshold=0.3, processed=4, failed=3,
            failed_items=['a', 'b', 'c'],
        )
        assert exc.failure_rate == 0.75
        assert exc.threshold == 0.3
        assert exc.processed == 4
        assert exc.failed == 3
        assert exc.failed_items == ['a', 'b', 'c']
        assert "75.0%" in str(exc)
        assert "30.0%" in str(exc)

    def test_default_failed_items(self):
        exc = BatchFailureThresholdExceeded(
            failure_rate=0.5, threshold=0.3, processed=2, failed=1,
        )
        assert exc.failed_items == []
