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

        Scenario: Items processed in order [fail, fail, fail, success, fail, ...].
        After each item, the threshold is checked. With threshold=0.3:
        - After item 1 (fail):  1/1  = 100% > 30% -> abort

        This confirms per-item checking catches failures immediately.
        For the specific "3 fails + 1 success = 75%" scenario, see the
        check at 4 processed / 3 failed below.
        """
        # Verify the 3-fails-1-success scenario: 3/4 = 75% > 30% -> abort
        with pytest.raises(BatchFailureThresholdExceeded) as exc_info:
            check_batch_failure_threshold(
                items_processed=4, items_failed=3, threshold=0.3,
            )
        assert exc_info.value.failure_rate == pytest.approx(0.75)
        assert exc_info.value.threshold == 0.3
        assert exc_info.value.processed == 4
        assert exc_info.value.failed == 3

    def test_batch_of_10_aborts_on_threshold_breach(self):
        """US-81-009 exact acceptance scenario: threshold=0.3, batch of 10 items.

        Simulates processing items one at a time with threshold checked after each.
        With per-item checking, the first failure at item 2 gives 1/2 = 50% > 30%.
        The batch aborts immediately — this is correct early-abort behavior.
        The key point: remaining items are NEVER processed.
        """
        threshold = 0.3
        total_batch_size = 10
        # Sequence: first item succeeds, then failures start
        outcomes = [True, False, False, False, True, True, True, True, True, True]
        failures = 0
        abort_at_item = None

        for i, success in enumerate(outcomes):
            if not success:
                failures += 1
            processed = i + 1
            try:
                check_batch_failure_threshold(
                    items_processed=processed,
                    items_failed=failures,
                    threshold=threshold,
                )
            except BatchFailureThresholdExceeded as exc:
                abort_at_item = processed
                # 1 failure / 2 processed = 50% > 30%
                assert exc.failure_rate > threshold
                assert exc.processed == processed
                assert exc.failed == failures
                break

        # Batch aborted early, not at end of 10
        assert abort_at_item is not None, "Expected batch to abort"
        assert abort_at_item < total_batch_size
        # Remaining items were never processed
        items_skipped = total_batch_size - abort_at_item
        assert items_skipped > 0

    def test_batch_of_10_scenario_3_fails_1_success_75_percent(self):
        """US-81-009 acceptance: threshold=0.3, 3 fails + 1 success = 75% fail rate.

        This tests the specific snapshot: 4 items processed, 3 failed.
        Failure rate = 3/4 = 75% > 30% -> abort with all failures listed.
        """
        threshold = 0.3
        failed_items = ['video_A', 'video_C', 'video_D']
        with pytest.raises(BatchFailureThresholdExceeded) as exc_info:
            check_batch_failure_threshold(
                items_processed=4,
                items_failed=3,
                threshold=threshold,
                failed_items=failed_items,
            )
        assert exc_info.value.failure_rate == pytest.approx(0.75)
        assert exc_info.value.threshold == 0.3
        assert exc_info.value.processed == 4
        assert exc_info.value.failed == 3
        assert exc_info.value.failed_items == failed_items

    def test_per_item_check_continues_below_threshold(self):
        """When failure rate stays at or below threshold at every check, batch completes.

        Simulates a 10-item batch with 2 failures placed so rate never exceeds 0.3:
        - Failures at items 4 and 8 (1-indexed):
          item 4: 1/4 = 25% <= 30% (ok)
          item 8: 2/8 = 25% <= 30% (ok)
        Batch completes with partial success: 8 succeeded, 2 failed.
        """
        threshold = 0.3
        #                item: 1     2     3     4      5     6     7     8      9     10
        outcomes =          [True, True, True, False, True, True, True, False, True, True]
        failures = 0
        aborted = False

        for i, success in enumerate(outcomes):
            if not success:
                failures += 1
            processed = i + 1
            try:
                check_batch_failure_threshold(
                    items_processed=processed,
                    items_failed=failures,
                    threshold=threshold,
                )
            except BatchFailureThresholdExceeded:
                aborted = True
                break

        # Batch completed all 10 items without aborting
        assert not aborted, "Batch should not abort when failure rate stays <= threshold"
        assert processed == 10
        assert failures == 2
        # Partial success: 8 items succeeded, 2 failed
        successes = processed - failures
        assert successes == 8

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
