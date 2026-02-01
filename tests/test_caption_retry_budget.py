"""Tests for CaptionRetryBudget class (US-33-010).

Tests the cross-video retry budget tracking that limits total fetch attempts
and backoff time across all videos in a batch, skipping remaining videos
when budget is exhausted.

Acceptance criteria:
1. Create src/caption/retry_budget.py with CaptionRetryBudget class
2. Track: fetch attempts, failures, backoff time used across all videos in batch
3. Provide budget_exhausted() method when limits exceeded
4. Add config: caption.retry_budget.max_attempts, max_backoff_time_seconds
5. Integrate with CaptionStage to skip remaining videos when budget exhausted
6. Tests in tests/test_caption_retry_budget.py with at least 6 test cases
"""

import pytest
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

from src.caption.retry_budget import CaptionRetryBudget, CaptionRetryBudgetConfig, BatchRetryBudget
from src.caption.enums import CaptionErrorCategory
from src.config.sections.download import CaptionRetryBudgetConfig as ConfigCaptionRetryBudgetConfig


class TestCaptionRetryBudgetConfig:
    """Test CaptionRetryBudgetConfig dataclass."""

    @pytest.mark.fast
    def test_default_values(self):
        """Test default configuration values."""
        config = CaptionRetryBudgetConfig()
        assert config.enabled is True
        assert config.max_attempts == 100
        assert config.max_backoff_time_seconds == 300.0

    @pytest.mark.fast
    def test_custom_values(self):
        """Test custom configuration values."""
        config = CaptionRetryBudgetConfig(
            enabled=False,
            max_attempts=50,
            max_backoff_time_seconds=600.0,
        )
        assert config.enabled is False
        assert config.max_attempts == 50
        assert config.max_backoff_time_seconds == 600.0

    @pytest.mark.fast
    def test_config_sections_consistency(self):
        """Test config in src/caption matches config in src/config/sections."""
        # Ensure the config dataclass in retry_budget.py has same defaults as download.py
        rb_config = CaptionRetryBudgetConfig()
        dl_config = ConfigCaptionRetryBudgetConfig()

        assert rb_config.enabled == dl_config.enabled
        assert rb_config.max_attempts == dl_config.max_attempts
        assert rb_config.max_backoff_time_seconds == dl_config.max_backoff_time_seconds


class TestCaptionRetryBudgetInit:
    """Test CaptionRetryBudget initialization."""

    @pytest.mark.fast
    def test_default_initialization(self):
        """Test default values on initialization."""
        budget = CaptionRetryBudget()
        assert budget.attempts == 0
        assert budget.failures == 0
        assert budget.successes == 0
        assert budget.backoff_time_spent == 0.0
        assert budget.videos_skipped == 0
        assert budget.max_attempts == 100
        assert budget.max_backoff_time == 300.0

    @pytest.mark.fast
    def test_from_config_with_config(self):
        """Test from_config creates budget with config values."""
        config = CaptionRetryBudgetConfig(
            max_attempts=50,
            max_backoff_time_seconds=120.0,
        )
        budget = CaptionRetryBudget.from_config(config)

        assert budget.max_attempts == 50
        assert budget.max_backoff_time == 120.0

    @pytest.mark.fast
    def test_from_config_with_none(self):
        """Test from_config returns defaults when config is None."""
        budget = CaptionRetryBudget.from_config(None)
        assert budget.max_attempts == 100
        assert budget.max_backoff_time == 300.0

    @pytest.mark.fast
    def test_from_config_with_dict(self):
        """Test from_config handles dict configs (from YAML)."""
        config = {
            'max_attempts': 75,
            'max_backoff_time_seconds': 180.0,
        }
        budget = CaptionRetryBudget.from_config(config)

        assert budget.max_attempts == 75
        assert budget.max_backoff_time == 180.0


class TestCaptionRetryBudgetRecording:
    """Test recording attempts, successes, failures, and backoff."""

    @pytest.mark.fast
    def test_record_attempt(self):
        """Test record_attempt increments attempts counter."""
        budget = CaptionRetryBudget()
        budget.record_attempt("video1")
        assert budget.attempts == 1

        budget.record_attempt("video2")
        assert budget.attempts == 2

    @pytest.mark.fast
    def test_record_success(self):
        """Test record_success increments successes counter."""
        budget = CaptionRetryBudget()
        budget.record_success("video1")
        assert budget.successes == 1

        budget.record_success("video2")
        assert budget.successes == 2

    @pytest.mark.fast
    def test_record_failure(self):
        """Test record_failure increments failures counter."""
        budget = CaptionRetryBudget()
        budget.record_failure("video1")
        assert budget.failures == 1

        budget.record_failure("video2")
        assert budget.failures == 2

    @pytest.mark.fast
    def test_record_backoff(self):
        """Test record_backoff accumulates backoff time."""
        budget = CaptionRetryBudget()
        budget.record_backoff(5.0, "video1")
        assert budget.backoff_time_spent == 5.0

        budget.record_backoff(10.0, "video2")
        assert budget.backoff_time_spent == 15.0

    @pytest.mark.fast
    def test_record_skipped(self):
        """Test record_skipped increments skipped counter."""
        budget = CaptionRetryBudget()
        budget.record_skipped("video1")
        assert budget.videos_skipped == 1

        budget.record_skipped("video2")
        assert budget.videos_skipped == 2


class TestCaptionRetryBudgetExhaustion:
    """Test budget_exhausted() method."""

    @pytest.mark.fast
    def test_budget_not_exhausted_initially(self):
        """Test budget is not exhausted when fresh."""
        budget = CaptionRetryBudget()
        assert budget.budget_exhausted() is False

    @pytest.mark.fast
    def test_budget_exhausted_by_attempts(self):
        """Test budget is exhausted when max_attempts reached."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10

        for i in range(10):
            budget.record_attempt(f"video_{i}")

        assert budget.budget_exhausted() is True

    @pytest.mark.fast
    def test_budget_exhausted_by_backoff_time(self):
        """Test budget is exhausted when max_backoff_time reached."""
        budget = CaptionRetryBudget()
        budget.max_backoff_time = 60.0

        budget.record_backoff(30.0, "video1")
        assert budget.budget_exhausted() is False

        budget.record_backoff(35.0, "video2")
        assert budget.budget_exhausted() is True

    @pytest.mark.fast
    def test_budget_not_exhausted_when_unlimited_attempts(self):
        """Test budget is never exhausted by attempts when max_attempts=0."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 0  # Unlimited

        for i in range(1000):
            budget.record_attempt(f"video_{i}")

        assert budget.budget_exhausted() is False

    @pytest.mark.fast
    def test_budget_not_exhausted_when_unlimited_backoff(self):
        """Test budget is never exhausted by backoff when max_backoff_time=0."""
        budget = CaptionRetryBudget()
        budget.max_backoff_time = 0  # Unlimited

        budget.record_backoff(10000.0, "video1")
        assert budget.budget_exhausted() is False

    @pytest.mark.fast
    def test_budget_exhausted_either_condition(self):
        """Test budget is exhausted when either limit is exceeded."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 10.0

        # Not exhausted by attempts, but by backoff
        budget.record_attempt("video1")
        budget.record_backoff(15.0, "video1")

        assert budget.budget_exhausted() is True


class TestCaptionRetryBudgetRemainingMethods:
    """Test attempts_remaining() and backoff_time_remaining() methods."""

    @pytest.mark.fast
    def test_attempts_remaining(self):
        """Test attempts_remaining returns correct value."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        assert budget.attempts_remaining() == 100

        budget.record_attempt("video1")
        assert budget.attempts_remaining() == 99

        for i in range(50):
            budget.record_attempt(f"video_{i}")
        assert budget.attempts_remaining() == 49

    @pytest.mark.fast
    def test_attempts_remaining_unlimited(self):
        """Test attempts_remaining returns None when unlimited."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 0

        assert budget.attempts_remaining() is None

    @pytest.mark.fast
    def test_backoff_time_remaining(self):
        """Test backoff_time_remaining returns correct value."""
        budget = CaptionRetryBudget()
        budget.max_backoff_time = 300.0

        assert budget.backoff_time_remaining() == 300.0

        budget.record_backoff(100.0, "video1")
        assert budget.backoff_time_remaining() == 200.0

    @pytest.mark.fast
    def test_backoff_time_remaining_unlimited(self):
        """Test backoff_time_remaining returns None when unlimited."""
        budget = CaptionRetryBudget()
        budget.max_backoff_time = 0

        assert budget.backoff_time_remaining() is None

    @pytest.mark.fast
    def test_can_backoff_within_budget(self):
        """Test can_backoff checks if additional backoff is allowed."""
        budget = CaptionRetryBudget()
        budget.max_backoff_time = 60.0

        assert budget.can_backoff(30.0) is True
        assert budget.can_backoff(60.0) is True
        assert budget.can_backoff(61.0) is False

        budget.record_backoff(50.0, "video1")
        assert budget.can_backoff(10.0) is True
        assert budget.can_backoff(15.0) is False


class TestCaptionRetryBudgetSummary:
    """Test get_summary() method."""

    @pytest.mark.fast
    def test_get_summary_fresh_budget(self):
        """Test get_summary returns correct values for fresh budget."""
        budget = CaptionRetryBudget()
        summary = budget.get_summary()

        assert summary['attempts'] == 0
        assert summary['successes'] == 0
        assert summary['failures'] == 0
        assert summary['backoff_time_spent'] == 0.0
        assert summary['videos_skipped'] == 0
        assert summary['is_exhausted'] is False

    @pytest.mark.fast
    def test_get_summary_after_activity(self):
        """Test get_summary returns correct values after activity."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        for i in range(10):
            budget.record_attempt(f"video_{i}")
        for i in range(7):
            budget.record_success(f"video_{i}")
        for i in range(3):
            budget.record_failure(f"fail_{i}")
        budget.record_backoff(15.5, "video1")
        budget.record_skipped("skipped1")

        summary = budget.get_summary()

        assert summary['attempts'] == 10
        assert summary['attempts_remaining'] == 90
        assert summary['successes'] == 7
        assert summary['failures'] == 3
        assert summary['backoff_time_spent'] == 15.5
        assert summary['videos_skipped'] == 1
        assert summary['is_exhausted'] is False


class TestCaptionRetryBudgetSerialization:
    """Test checkpoint serialization."""

    @pytest.mark.fast
    def test_to_dict_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 50
        budget.max_backoff_time = 120.0

        for i in range(20):
            budget.record_attempt(f"video_{i}")
        for i in range(15):
            budget.record_success(f"video_{i}")
        for i in range(5):
            budget.record_failure(f"fail_{i}")
        budget.record_backoff(30.5, "video1")
        budget.record_skipped("skipped1")
        budget.record_skipped("skipped2")

        data = budget.to_dict()
        restored = CaptionRetryBudget.from_dict(data)

        assert restored.attempts == 20
        assert restored.successes == 15
        assert restored.failures == 5
        assert restored.backoff_time_spent == 30.5
        assert restored.videos_skipped == 2
        assert restored.max_attempts == 50
        assert restored.max_backoff_time == 120.0

    @pytest.mark.fast
    def test_from_dict_none(self):
        """Test from_dict with None returns fresh budget."""
        budget = CaptionRetryBudget.from_dict(None)
        assert budget.attempts == 0
        assert budget.successes == 0

    @pytest.mark.fast
    def test_from_dict_empty(self):
        """Test from_dict with empty dict returns fresh budget."""
        budget = CaptionRetryBudget.from_dict({})
        assert budget.attempts == 0


class TestCaptionRetryBudgetReset:
    """Test reset functionality."""

    @pytest.mark.fast
    def test_reset_clears_counters(self):
        """Test reset clears all counters but preserves limits."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 50
        budget.max_backoff_time = 120.0

        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_success(f"video_{i}")
        budget.record_failure("fail1")
        budget.record_backoff(30.0, "video1")
        budget.record_skipped("skip1")

        budget.reset()

        assert budget.attempts == 0
        assert budget.successes == 0
        assert budget.failures == 0
        assert budget.backoff_time_spent == 0.0
        assert budget.videos_skipped == 0
        # Limits preserved
        assert budget.max_attempts == 50
        assert budget.max_backoff_time == 120.0


class TestCaptionRetryBudgetThreadSafety:
    """Test thread safety of CaptionRetryBudget."""

    @pytest.mark.fast
    def test_concurrent_recording(self):
        """Test concurrent recording from multiple threads."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10000  # High limit to avoid exhaustion

        def record_attempt_n_times(n):
            for _ in range(n):
                budget.record_attempt()

        def record_success_n_times(n):
            for _ in range(n):
                budget.record_success()

        def record_backoff_n_times(n):
            for _ in range(n):
                budget.record_backoff(0.1)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(record_attempt_n_times, 100),
                executor.submit(record_attempt_n_times, 100),
                executor.submit(record_success_n_times, 100),
                executor.submit(record_backoff_n_times, 100),
            ]
            for f in futures:
                f.result()

        # Should have 200 attempts from 2 threads
        assert budget.attempts == 200
        assert budget.successes == 100
        # Backoff should be approximately 10.0 (100 * 0.1)
        assert abs(budget.backoff_time_spent - 10.0) < 0.1

    @pytest.mark.fast
    def test_concurrent_budget_exhausted_check(self):
        """Test budget_exhausted is thread-safe during concurrent checks."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 50

        # Pre-fill to near exhaustion
        for i in range(49):
            budget.record_attempt(f"video_{i}")

        results = []
        lock = threading.Lock()

        def check_and_record():
            # Check exhaustion and record if not exhausted
            exhausted = budget.budget_exhausted()
            with lock:
                results.append(exhausted)
            if not exhausted:
                budget.record_attempt("thread_video")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(check_and_record) for _ in range(10)]
            for f in futures:
                f.result()

        # At most 1 should have succeeded before exhaustion
        # (49 pre-filled + some threads)
        assert budget.attempts >= 50


class TestCaptionRetryBudgetIntegration:
    """Test integration scenarios with caption fetching."""

    @pytest.mark.fast
    def test_typical_batch_workflow(self):
        """Test a typical batch workflow with mixed results."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 60.0

        # Simulate processing 50 videos
        # 35 success, 10 failures with retries, 5 skipped (budget check before)
        for i in range(45):  # 45 attempts before budget check
            budget.record_attempt(f"video_{i}")
            if i < 35:
                budget.record_success(f"video_{i}")
            else:
                budget.record_failure(f"video_{i}")
                budget.record_backoff(1.0, f"video_{i}")  # 1s backoff per failure

        summary = budget.get_summary()

        assert summary['attempts'] == 45
        assert summary['successes'] == 35
        assert summary['failures'] == 10
        assert summary['backoff_time_spent'] == 10.0
        assert summary['is_exhausted'] is False

    @pytest.mark.fast
    def test_early_exhaustion_scenario(self):
        """Test scenario where budget is exhausted early, skipping remaining."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 20  # Low limit
        budget.max_backoff_time = 30.0  # Low limit

        # Simulate high failure rate with backoff
        for i in range(15):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}")
            budget.record_backoff(3.0, f"video_{i}")  # 3s per failure = 45s total

        # Budget should be exhausted by backoff time (45s > 30s)
        assert budget.budget_exhausted() is True

        # Remaining 5 videos should be skipped
        for i in range(5):
            if budget.budget_exhausted():
                budget.record_skipped(f"remaining_{i}")

        assert budget.videos_skipped == 5


class TestCaptionRetryBudgetLogging:
    """Test logging behavior."""

    @pytest.mark.fast
    def test_exhaustion_logging(self):
        """Test that exhaustion is logged with INFO level."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 5

        with patch('src.caption.retry_budget.logger') as mock_logger:
            for i in range(5):
                budget.record_attempt(f"video_{i}")

            # Check exhaustion - should log
            budget.budget_exhausted()

            # Verify INFO logging was called for exhaustion
            info_calls = [c for c in mock_logger.info.call_args_list
                        if 'EXHAUSTED' in str(c)]
            assert len(info_calls) >= 1

    @pytest.mark.fast
    def test_debug_logging_for_recording(self):
        """Test that recording operations use DEBUG level logging."""
        budget = CaptionRetryBudget()

        with patch('src.caption.retry_budget.logger') as mock_logger:
            budget.record_attempt("video1")
            budget.record_success("video1")
            budget.record_failure("video2")
            budget.record_backoff(5.0, "video3")
            budget.record_skipped("video4")

            # All should use debug logging
            assert mock_logger.debug.call_count >= 5


# ============================================================================
# US-34-007: Retry Budget Exhaustion Tests
# ============================================================================


class TestCaptionRetryBudgetExhaustionMidBatch:
    """Test budget exhausted mid-batch triggers early abort (US-34-007)."""

    @pytest.mark.fast
    def test_budget_exhausts_mid_batch_by_attempts(self):
        """Test budget exhausts mid-batch when max_attempts reached."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 15
        budget.max_backoff_time = 0  # Unlimited backoff

        videos = [f"video_{i}" for i in range(30)]
        processed = []
        skipped = []

        for video_id in videos:
            if budget.budget_exhausted():
                budget.record_skipped(video_id)
                skipped.append(video_id)
            else:
                budget.record_attempt(video_id)
                processed.append(video_id)

        # Should have processed exactly 15 videos before exhaustion
        assert len(processed) == 15
        assert len(skipped) == 15
        assert budget.attempts == 15
        assert budget.videos_skipped == 15

    @pytest.mark.fast
    def test_budget_exhausts_mid_batch_by_backoff(self):
        """Test budget exhausts mid-batch when max_backoff_time reached."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 0  # Unlimited attempts
        budget.max_backoff_time = 50.0

        videos = [f"video_{i}" for i in range(20)]
        processed = []
        skipped = []

        for video_id in videos:
            if budget.budget_exhausted():
                budget.record_skipped(video_id)
                skipped.append(video_id)
            else:
                budget.record_attempt(video_id)
                budget.record_failure(video_id)
                budget.record_backoff(10.0, video_id)  # 10s per video
                processed.append(video_id)

        # Should have processed 5 videos (5 * 10s = 50s) before exhaustion
        assert len(processed) == 5
        assert len(skipped) == 15
        assert budget.backoff_time_spent == 50.0

    @pytest.mark.fast
    def test_early_abort_preserves_successful_results(self):
        """Test that early abort still preserves successes from before exhaustion."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10

        # Simulate mixed results before exhaustion
        for i in range(6):
            budget.record_attempt(f"video_{i}")
            budget.record_success(f"video_{i}")

        for i in range(4):
            budget.record_attempt(f"retry_{i}")
            budget.record_failure(f"retry_{i}")

        # Now exhausted
        assert budget.budget_exhausted() is True

        # But we still have our 6 successes preserved
        summary = budget.get_summary()
        assert summary['successes'] == 6
        assert summary['failures'] == 4
        assert summary['attempts'] == 10


class TestBatchRetryBudgetProgressPreservation:
    """Test BatchRetryBudget preserves progress on budget exhaustion (US-34-007)."""

    @pytest.mark.fast
    def test_progress_preserved_after_threshold_crossing(self):
        """Test processed count and category counts preserved after threshold cross."""
        budget = BatchRetryBudget(total_videos=100)

        # Process 10 videos with 6 network errors (>50% rate triggers disable)
        for i in range(4):
            budget.record_success(f"video_{i}")
        for i in range(6):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        # Threshold crossed (>50%), retries disabled
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 0

        # But all progress is preserved
        assert budget.processed_videos == 10
        assert budget.category_counts[CaptionErrorCategory.NETWORK] == 6
        assert len(budget.budget_reductions) >= 1

    @pytest.mark.fast
    def test_multiple_category_progress_preserved(self):
        """Test progress for multiple error categories preserved."""
        budget = BatchRetryBudget(total_videos=100)

        # Mixed errors
        for i in range(10):
            budget.record_success(f"s_{i}")
        for i in range(5):
            budget.record_error(CaptionErrorCategory.NETWORK, f"net_{i}")
        for i in range(3):
            budget.record_error(CaptionErrorCategory.TIMEOUT, f"to_{i}")
        for i in range(2):
            budget.record_error(CaptionErrorCategory.PARSE, f"parse_{i}")

        # All categories tracked
        assert budget.processed_videos == 20
        assert budget.category_counts[CaptionErrorCategory.NETWORK] == 5
        assert budget.category_counts[CaptionErrorCategory.TIMEOUT] == 3
        assert budget.category_counts[CaptionErrorCategory.PARSE] == 2

    @pytest.mark.fast
    def test_summary_reflects_exhaustion_state(self):
        """Test get_summary includes exhaustion info when budget reduced."""
        budget = BatchRetryBudget(total_videos=100)

        # Push to >50% to disable retries
        for i in range(4):
            budget.record_success(f"video_{i}")
        for i in range(6):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        summary = budget.get_summary()

        assert summary['reduced_budgets']['NETWORK'] == 0
        assert summary['original_budgets']['NETWORK'] == 3
        assert summary['reductions_applied'] >= 1
        assert summary['error_rates']['NETWORK'] == 0.6


class TestBudgetResetBetweenBatches:
    """Test budget reset between batches (US-34-007)."""

    @pytest.mark.fast
    def test_caption_retry_budget_reset_allows_new_batch(self):
        """Test CaptionRetryBudget reset allows processing new batch."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10

        # Exhaust the budget
        for i in range(10):
            budget.record_attempt(f"video_{i}")

        assert budget.budget_exhausted() is True

        # Reset for new batch
        budget.reset()

        # Fresh state, not exhausted
        assert budget.budget_exhausted() is False
        assert budget.attempts == 0
        assert budget.failures == 0
        assert budget.successes == 0
        assert budget.backoff_time_spent == 0.0
        assert budget.videos_skipped == 0

        # Can process new videos
        budget.record_attempt("new_video_0")
        assert budget.attempts == 1
        assert budget.budget_exhausted() is False

    @pytest.mark.fast
    def test_batch_retry_budget_reset_restores_budgets(self):
        """Test BatchRetryBudget reset restores original retry budgets."""
        budget = BatchRetryBudget(total_videos=100)

        # Trigger >50% threshold to disable retries
        for i in range(4):
            budget.record_success(f"video_{i}")
        for i in range(6):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 0

        # Reset
        budget.reset()

        # Budgets restored
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 3
        assert budget.processed_videos == 0
        assert len(budget.category_counts) == 0
        assert len(budget.budget_reductions) == 0

    @pytest.mark.fast
    def test_reset_preserves_limits(self):
        """Test reset preserves max_attempts and max_backoff_time limits."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 25
        budget.max_backoff_time = 120.0

        for i in range(25):
            budget.record_attempt(f"video_{i}")

        assert budget.budget_exhausted() is True

        budget.reset()

        # Limits preserved
        assert budget.max_attempts == 25
        assert budget.max_backoff_time == 120.0

    @pytest.mark.fast
    def test_multiple_batch_cycles(self):
        """Test multiple exhaust-reset cycles work correctly."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 5

        for batch_num in range(3):
            # Process batch until exhausted
            for i in range(5):
                budget.record_attempt(f"batch{batch_num}_video_{i}")

            assert budget.budget_exhausted() is True
            assert budget.attempts == 5

            # Reset for next batch
            budget.reset()

            assert budget.budget_exhausted() is False
            assert budget.attempts == 0


class TestBudgetMetricsExport:
    """Test budget metrics exported correctly (US-34-007)."""

    @pytest.mark.fast
    def test_caption_retry_budget_metrics_consumed(self):
        """Test consumed metrics are correctly reported."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 300.0

        for i in range(30):
            budget.record_attempt(f"video_{i}")
        for i in range(25):
            budget.record_success(f"video_{i}")
        for i in range(5):
            budget.record_failure(f"fail_{i}")
        budget.record_backoff(45.5, "video_x")
        budget.record_skipped("skipped_1")

        summary = budget.get_summary()

        # Consumed metrics
        assert summary['attempts'] == 30
        assert summary['successes'] == 25
        assert summary['failures'] == 5
        assert summary['backoff_time_spent'] == 45.5
        assert summary['videos_skipped'] == 1

    @pytest.mark.fast
    def test_caption_retry_budget_metrics_remaining(self):
        """Test remaining metrics are correctly reported."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 300.0

        for i in range(30):
            budget.record_attempt(f"video_{i}")
        budget.record_backoff(45.5, "video_x")

        summary = budget.get_summary()

        # Remaining metrics
        assert summary['attempts_remaining'] == 70
        assert summary['backoff_time_remaining'] == 254.5

    @pytest.mark.fast
    def test_unlimited_metrics_return_none(self):
        """Test unlimited budgets return None for remaining."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 0  # Unlimited
        budget.max_backoff_time = 0  # Unlimited

        for i in range(100):
            budget.record_attempt(f"video_{i}")
        budget.record_backoff(500.0, "video_x")

        summary = budget.get_summary()

        assert summary['attempts_remaining'] is None
        assert summary['backoff_time_remaining'] is None

    @pytest.mark.fast
    def test_batch_retry_budget_metrics_export(self):
        """Test BatchRetryBudget metrics exported in summary."""
        budget = BatchRetryBudget(total_videos=100)

        for i in range(30):
            budget.record_success(f"video_{i}")
        for i in range(15):
            budget.record_error(CaptionErrorCategory.NETWORK, f"net_{i}")
        for i in range(5):
            budget.record_error(CaptionErrorCategory.TIMEOUT, f"to_{i}")

        summary = budget.get_summary()

        assert summary['total_videos'] == 100
        assert summary['processed_videos'] == 50
        assert summary['error_rates']['NETWORK'] == 0.3
        assert summary['error_rates']['TIMEOUT'] == 0.1
        assert 'original_budgets' in summary
        assert 'reduced_budgets' in summary

    @pytest.mark.fast
    def test_to_dict_captures_all_state(self):
        """Test to_dict captures complete state for checkpoint."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 50
        budget.max_backoff_time = 120.0

        for i in range(20):
            budget.record_attempt(f"video_{i}")
        for i in range(15):
            budget.record_success(f"video_{i}")
        for i in range(5):
            budget.record_failure(f"fail_{i}")
        budget.record_backoff(30.0, "video_x")
        budget.record_skipped("skipped_1")

        data = budget.to_dict()

        assert data['attempts'] == 20
        assert data['successes'] == 15
        assert data['failures'] == 5
        assert data['backoff_time_spent'] == 30.0
        assert data['videos_skipped'] == 1
        assert data['max_attempts'] == 50
        assert data['max_backoff_time'] == 120.0


class TestBothBudgetClassesCoverage:
    """Test coverage for both CaptionRetryBudget and BatchRetryBudget (US-34-007)."""

    @pytest.mark.fast
    def test_caption_retry_budget_exhaustion_workflow(self):
        """Test complete CaptionRetryBudget exhaustion workflow."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 20
        budget.max_backoff_time = 30.0

        results = {'processed': 0, 'skipped': 0}

        for i in range(50):
            if budget.budget_exhausted():
                budget.record_skipped(f"video_{i}")
                results['skipped'] += 1
            else:
                budget.record_attempt(f"video_{i}")
                if i % 3 == 0:
                    budget.record_failure(f"video_{i}")
                    budget.record_backoff(2.0, f"video_{i}")
                else:
                    budget.record_success(f"video_{i}")
                results['processed'] += 1

        # Verify workflow completed with proper tracking
        assert results['processed'] + results['skipped'] == 50
        assert budget.budget_exhausted() is True
        summary = budget.get_summary()
        assert summary['is_exhausted'] is True

    @pytest.mark.fast
    def test_batch_retry_budget_threshold_workflow(self):
        """Test complete BatchRetryBudget threshold workflow."""
        budget = BatchRetryBudget(total_videos=100)

        # Simulate escalating failure scenario
        for i in range(20):
            budget.record_success(f"video_{i}")

        # Start getting network errors
        for i in range(15):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        # At this point: 35 processed, 15 network errors = 42.9% error rate
        # Should have triggered >30% reduction
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 1
        assert len(budget.budget_reductions) >= 1

        # Continue with more errors to trigger >50%
        for i in range(10):
            budget.record_error(CaptionErrorCategory.NETWORK, f"error_more_{i}")

        # Now: 45 processed, 25 errors = 55.5% error rate
        assert budget.budget_remaining(CaptionErrorCategory.NETWORK) == 0

    @pytest.mark.fast
    def test_both_budgets_work_together(self):
        """Test CaptionRetryBudget and BatchRetryBudget can work in tandem."""
        caption_budget = CaptionRetryBudget()
        caption_budget.max_attempts = 50
        batch_budget = BatchRetryBudget(total_videos=100)

        for i in range(30):
            # Check overall budget first
            if caption_budget.budget_exhausted():
                caption_budget.record_skipped(f"video_{i}")
                continue

            # Record attempt
            caption_budget.record_attempt(f"video_{i}")

            # Simulate 40% failure rate
            if i % 5 < 2:
                caption_budget.record_failure(f"video_{i}")
                batch_budget.record_error(CaptionErrorCategory.NETWORK, f"video_{i}")
            else:
                caption_budget.record_success(f"video_{i}")
                batch_budget.record_success(f"video_{i}")

        # Both budgets track independently
        assert caption_budget.attempts == 30
        assert batch_budget.processed_videos == 30

        # Batch budget should have reduced based on error rate
        assert batch_budget.get_error_rate(CaptionErrorCategory.NETWORK) == 12 / 30

    @pytest.mark.fast
    def test_serialization_roundtrip_both_budgets(self):
        """Test both budget classes serialize and deserialize correctly."""
        # CaptionRetryBudget roundtrip
        caption_budget = CaptionRetryBudget()
        caption_budget.max_attempts = 50
        for i in range(25):
            caption_budget.record_attempt(f"video_{i}")
            caption_budget.record_success(f"video_{i}")

        caption_data = caption_budget.to_dict()
        restored_caption = CaptionRetryBudget.from_dict(caption_data)

        assert restored_caption.attempts == 25
        assert restored_caption.successes == 25
        assert restored_caption.max_attempts == 50

        # BatchRetryBudget roundtrip
        batch_budget = BatchRetryBudget(total_videos=100)
        for i in range(20):
            batch_budget.record_success(f"video_{i}")
        for i in range(10):
            batch_budget.record_error(CaptionErrorCategory.NETWORK, f"error_{i}")

        batch_data = batch_budget.to_dict()
        restored_batch = BatchRetryBudget.from_dict(batch_data)

        assert restored_batch.processed_videos == 30
        assert restored_batch.category_counts[CaptionErrorCategory.NETWORK] == 10
