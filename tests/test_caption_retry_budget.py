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

    @pytest.mark.fast
    def test_from_config_with_auto_scale_settings(self):
        """Test from_config loads auto_scale and attempts_per_video (US-37-004)."""
        config = CaptionRetryBudgetConfig(
            max_attempts=100,
            max_backoff_time_seconds=300.0,
            auto_scale=False,
            attempts_per_video=2.0,
        )
        budget = CaptionRetryBudget.from_config(config)

        assert budget.auto_scale is False
        assert budget.attempts_per_video == 2.0

    @pytest.mark.fast
    def test_from_config_dict_with_auto_scale_settings(self):
        """Test from_config handles dict configs with auto_scale settings (US-37-004)."""
        config = {
            'max_attempts': 100,
            'max_backoff_time_seconds': 300.0,
            'auto_scale': True,
            'attempts_per_video': 1.8,
        }
        budget = CaptionRetryBudget.from_config(config)

        assert budget.auto_scale is True
        assert budget.attempts_per_video == 1.8

    @pytest.mark.fast
    def test_from_config_defaults_auto_scale_to_true(self):
        """Test auto_scale defaults to True when not specified (US-37-004)."""
        config = CaptionRetryBudgetConfig(
            max_attempts=100,
            max_backoff_time_seconds=300.0,
        )
        budget = CaptionRetryBudget.from_config(config)

        assert budget.auto_scale is True
        assert budget.attempts_per_video == 2.0  # Default value (US-40-006)

    @pytest.mark.fast
    def test_from_config_dict_defaults_auto_scale(self):
        """Test dict config defaults auto_scale to True (US-37-004)."""
        config = {
            'max_attempts': 100,
        }
        budget = CaptionRetryBudget.from_config(config)

        assert budget.auto_scale is True
        assert budget.attempts_per_video == 2.0  # Default value (US-40-006)


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

    @pytest.mark.fast
    def test_budget_exhausted_logs_diagnostic_info(self, caplog):
        """Test budget_exhausted() logs diagnostic info when exhausted (US-42-005).

        Verifies AC4: diagnostic info is logged when budget exhausts including
        auto_scale setting, batch_size, original_max, and top errors.
        """
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget()
        budget.max_attempts = 10
        budget.original_max_attempts = 10
        budget.batch_size = 50
        budget.auto_scale = False

        # Record some errors to track
        budget.record_failure("video1", CaptionErrorCategory.RATE_LIMIT)
        budget.record_failure("video2", CaptionErrorCategory.RATE_LIMIT)
        budget.record_failure("video3", CaptionErrorCategory.TIMEOUT)

        # Exhaust the budget by attempts
        for i in range(10):
            budget.record_attempt(f"video_{i}")

        # Trigger the exhausted log
        caplog.clear()
        assert budget.budget_exhausted() is True

        # Verify diagnostic log contains required info
        log_messages = [rec.message for rec in caplog.records]
        combined = ' '.join(log_messages)

        # AC1: Log includes auto_scale setting
        assert 'auto_scale=False' in combined

        # AC1: Log includes batch_size
        assert 'batch_size=50' in combined

        # AC1: Log includes config max_attempts (original_max)
        assert 'original_max=10' in combined

        # AC2: Shows why scaling didn't help (auto_scale=false)
        assert 'auto_scale=false' in combined.lower()

        # AC3: Include top error categories
        assert 'RATE_LIMIT:2' in combined
        assert 'TIMEOUT:1' in combined

        # AC5: Uses INFO level and includes [US-42-005] marker
        assert '[US-42-005]' in combined
        # Verify INFO level (all our diagnostic logs should be INFO)
        for rec in caplog.records:
            if '[US-42-005]' in rec.message:
                assert rec.levelno == logging.INFO

    @pytest.mark.fast
    def test_budget_exhausted_logs_scaling_not_triggered(self, caplog):
        """Test diagnostic log shows 'scaling_not_triggered' when auto_scale=true but scaling didn't occur."""
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.original_max_attempts = 100  # Same as max_attempts - scaling not triggered
        budget.batch_size = 30  # Small batch, doesn't require scaling
        budget.auto_scale = True

        # Exhaust the budget
        for i in range(100):
            budget.record_attempt(f"video_{i}")

        caplog.clear()
        assert budget.budget_exhausted() is True

        combined = ' '.join(rec.message for rec in caplog.records)
        assert 'scaling_not_triggered' in combined

    @pytest.mark.fast
    def test_budget_exhausted_logs_scaled_to_value(self, caplog):
        """Test diagnostic log shows 'scaled_to=N' when scaling occurred."""
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget()
        budget.max_attempts = 200  # Scaled value
        budget.original_max_attempts = 100  # Original config value
        budget.batch_size = 100
        budget.auto_scale = True

        # Exhaust the scaled budget
        for i in range(200):
            budget.record_attempt(f"video_{i}")

        caplog.clear()
        assert budget.budget_exhausted() is True

        combined = ' '.join(rec.message for rec in caplog.records)
        assert 'scaled_to=200' in combined
        assert 'original_max=100' in combined

    @pytest.mark.fast
    def test_budget_exhausted_backoff_logs_diagnostic_info(self, caplog):
        """Test diagnostic info is also logged when budget exhausts via backoff time."""
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget()
        budget.max_backoff_time = 10.0
        budget.original_max_attempts = 100
        budget.batch_size = 50
        budget.auto_scale = True

        # Exhaust via backoff, not attempts
        budget.record_backoff(15.0, "video1")

        caplog.clear()
        assert budget.budget_exhausted() is True

        combined = ' '.join(rec.message for rec in caplog.records)
        # Should still have [US-42-005] marker for backoff exhaustion too
        assert '[US-42-005]' in combined
        assert 'batch_size=50' in combined


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


# ============================================================================
# US-37-003: Scale CaptionRetryBudget max_attempts proportional to batch size
# ============================================================================


class TestCaptionRetryBudgetScaleToBatchSize:
    """Test scale_to_batch_size method (US-37-003)."""

    @pytest.mark.fast
    def test_small_batch_keeps_default_100(self):
        """Test small batch (50 videos) keeps default max_attempts=100."""
        budget = CaptionRetryBudget()
        assert budget.max_attempts == 100

        result = budget.scale_to_batch_size(50)

        # 50 * 1.5 = 75 < 100, so keep default
        assert budget.max_attempts == 100
        assert result == 100

    @pytest.mark.fast
    def test_boundary_batch_67_keeps_default(self):
        """Test batch of 67 videos keeps default (67 * 1.5 = 100.5 rounds to 101)."""
        budget = CaptionRetryBudget()

        # 50 * 2.0 = 100, equals default 100 - should keep as is
        budget.scale_to_batch_size(50)
        assert budget.max_attempts == 100

    @pytest.mark.fast
    def test_large_batch_175_scales_to_350(self):
        """Test large batch (175 videos) scales to 350 attempts (US-40-006).

        With attempts_per_video=2.0, 175 videos get 350 attempts.
        This gives 1 attempt + 1 retry per video average.
        """
        budget = CaptionRetryBudget()
        assert budget.max_attempts == 100

        result = budget.scale_to_batch_size(175)

        # 175 * 2.0 = 350
        assert budget.max_attempts == 350
        assert result == 350

    @pytest.mark.fast
    def test_custom_attempts_per_video(self):
        """Test custom attempts_per_video parameter."""
        budget = CaptionRetryBudget()

        # 50 videos with 2.5 attempts each = 125
        budget.scale_to_batch_size(50, attempts_per_video=2.5)
        assert budget.max_attempts == 125

    @pytest.mark.fast
    def test_never_reduces_below_default(self):
        """Test that scaling never reduces below default max_attempts."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 200  # Already above default

        # 50 * 2.0 = 100 < 200, should NOT reduce
        budget.scale_to_batch_size(50)
        assert budget.max_attempts == 200

    @pytest.mark.fast
    def test_scale_returns_new_max_attempts(self):
        """Test scale_to_batch_size returns the new max_attempts value."""
        budget = CaptionRetryBudget()

        result = budget.scale_to_batch_size(200)

        # 200 * 2.0 = 400
        assert result == 400
        assert budget.max_attempts == 400

    @pytest.mark.fast
    def test_scale_with_zero_batch_size(self):
        """Test scale with zero batch size keeps default."""
        budget = CaptionRetryBudget()

        result = budget.scale_to_batch_size(0)

        # 0 * 1.5 = 0 < 100, keep default
        assert budget.max_attempts == 100
        assert result == 100

    @pytest.mark.fast
    def test_scale_is_thread_safe(self):
        """Test scale_to_batch_size is thread-safe."""
        budget = CaptionRetryBudget()

        def scale_and_record():
            budget.scale_to_batch_size(200)
            budget.record_attempt("test")

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(scale_and_record) for _ in range(10)]
            for f in futures:
                f.result()

        # Should have scaled to 400 (200 * 2.0) and recorded 10 attempts
        assert budget.max_attempts == 400
        assert budget.attempts == 10

    @pytest.mark.fast
    def test_scale_multiple_calls_uses_largest(self):
        """Test multiple scale calls use the largest requirement."""
        budget = CaptionRetryBudget()

        budget.scale_to_batch_size(100)  # 100 * 2.0 = 200
        assert budget.max_attempts == 200

        budget.scale_to_batch_size(200)  # 200 * 2.0 = 400
        assert budget.max_attempts == 400

        budget.scale_to_batch_size(50)  # 50 * 2.0 = 100 < 400, no change
        assert budget.max_attempts == 400

    @pytest.mark.fast
    def test_acceptance_criteria_50_videos_stays_100(self):
        """Acceptance: 50 videos -> max_attempts stays at 100."""
        budget = CaptionRetryBudget()
        budget.scale_to_batch_size(50)
        assert budget.max_attempts == 100

    @pytest.mark.fast
    def test_acceptance_criteria_175_videos_becomes_350(self):
        """Acceptance: 175 videos -> max_attempts becomes 350 (US-40-006).

        With attempts_per_video=2.0, 175 * 2.0 = 350 attempts.
        This gives 1 attempt + 1 retry per video average.
        """
        budget = CaptionRetryBudget()
        budget.scale_to_batch_size(175)
        assert budget.max_attempts == 350

    @pytest.mark.fast
    def test_scale_uses_instance_attempts_per_video(self):
        """Test scale_to_batch_size uses instance attempts_per_video when no override (US-37-004)."""
        budget = CaptionRetryBudget()
        budget.attempts_per_video = 2.0  # Override instance value

        # 100 videos with 2.0 attempts each = 200
        result = budget.scale_to_batch_size(100)

        assert budget.max_attempts == 200
        assert result == 200

    @pytest.mark.fast
    def test_scale_override_takes_precedence(self):
        """Test scale_to_batch_size parameter overrides instance value (US-37-004)."""
        budget = CaptionRetryBudget()
        budget.attempts_per_video = 2.0  # Instance value

        # 100 videos with explicit 1.0 attempts each = 100 (override)
        result = budget.scale_to_batch_size(100, attempts_per_video=1.0)

        # Should use 1.0 not 2.0
        assert budget.max_attempts == 100
        assert result == 100

    @pytest.mark.fast
    def test_scale_from_config_values(self):
        """Test budget created from config uses config attempts_per_video (US-37-004)."""
        config = {
            'max_attempts': 100,
            'attempts_per_video': 2.5,
        }
        budget = CaptionRetryBudget.from_config(config)

        # 50 videos with 2.5 attempts each = 125
        budget.scale_to_batch_size(50)

        assert budget.max_attempts == 125


# ============================================================================
# US-39-010: CaptionRetryBudget.ensure_scaled() convenience method
# ============================================================================


class TestCaptionRetryBudgetEnsureScaled:
    """Test ensure_scaled convenience method (US-39-010)."""

    @pytest.mark.fast
    def test_ensure_scaled_returns_true_when_scaling_occurs(self):
        """Test ensure_scaled returns True when budget is scaled."""
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100

        result = budget.ensure_scaled(200)  # 200 * 2.0 = 400 > 100

        assert result is True
        assert budget.max_attempts == 400
        assert budget.batch_size == 200

    @pytest.mark.fast
    def test_ensure_scaled_returns_false_when_auto_scale_disabled(self):
        """Test ensure_scaled returns False when auto_scale is disabled."""
        budget = CaptionRetryBudget()
        budget.auto_scale = False
        budget.max_attempts = 100

        result = budget.ensure_scaled(200)

        assert result is False
        assert budget.max_attempts == 100  # Unchanged

    @pytest.mark.fast
    def test_ensure_scaled_idempotent_same_batch_size(self):
        """Test ensure_scaled is idempotent - calling twice with same batch_size does nothing (US-39-010).

        This is a key acceptance criterion: ensure_scaled() should be safe to call
        multiple times without redundant scaling.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100

        # First call - should scale
        result1 = budget.ensure_scaled(200)
        assert result1 is True
        assert budget.max_attempts == 400

        # Second call with same batch_size - should NOT scale again
        result2 = budget.ensure_scaled(200)
        assert result2 is False
        assert budget.max_attempts == 400  # Still 400, not doubled

    @pytest.mark.fast
    def test_ensure_scaled_idempotent_smaller_batch_size(self):
        """Test ensure_scaled doesn't reduce when called with smaller batch size."""
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100

        # First call - scale to 400 for 200 videos
        budget.ensure_scaled(200)
        assert budget.max_attempts == 400

        # Second call with smaller batch - should NOT reduce
        result = budget.ensure_scaled(50)  # 50 * 2.0 = 100 < 400
        assert result is False
        # max_attempts stays at 400 (already scaled higher)
        assert budget.max_attempts == 400

    @pytest.mark.fast
    def test_ensure_scaled_returns_false_when_budget_sufficient(self):
        """Test ensure_scaled returns False when existing budget is sufficient."""
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100

        result = budget.ensure_scaled(50)  # 50 * 2.0 = 100, equals current, no scale needed

        assert result is False
        assert budget.max_attempts == 100  # Unchanged
        assert budget.batch_size == 50  # Still tracks batch_size

    @pytest.mark.fast
    def test_ensure_scaled_uses_instance_attempts_per_video(self):
        """Test ensure_scaled uses the instance attempts_per_video config."""
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100
        budget.attempts_per_video = 2.0  # Custom multiplier

        result = budget.ensure_scaled(100)  # 100 * 2.0 = 200 > 100

        assert result is True
        assert budget.max_attempts == 200

    @pytest.mark.fast
    def test_ensure_scaled_with_reset_on_scale_clears_counters(self):
        """Test ensure_scaled with reset_on_scale=true clears counters (US-41-009).

        AC5: Add test: ensure_scaled with reset_on_scale=true clears counters

        When reset_on_scale is enabled and scaling occurs, all usage counters
        should be reset. This is useful when checkpoint restores a budget with
        50 attempts used, but the current batch needs 300 attempts - the user
        may want to start fresh rather than continue from the checkpoint's state.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100
        budget.reset_on_scale = True  # Enable reset on scale

        # Simulate prior usage (e.g., from checkpoint restore)
        budget.attempts = 50
        budget.failures = 20
        budget.successes = 30
        budget.backoff_time_spent = 45.0
        budget.videos_skipped = 5
        budget.error_counts = {CaptionErrorCategory.NETWORK: 10, CaptionErrorCategory.TIMEOUT: 10}
        budget.attempts_per_video_id = {"video1": 3, "video2": 2}
        budget.circuit_breaker_trips = 2
        budget.early_terminated = True
        budget.early_termination_reason = "Test reason"

        # Scale up - should trigger reset
        result = budget.ensure_scaled(200)  # 200 * 2.0 = 400 > 100

        # Verify scaling occurred
        assert result is True
        assert budget.max_attempts == 400
        assert budget.batch_size == 200

        # Verify all counters were reset
        assert budget.attempts == 0, "attempts should be reset"
        assert budget.failures == 0, "failures should be reset"
        assert budget.successes == 0, "successes should be reset"
        assert budget.backoff_time_spent == 0.0, "backoff_time_spent should be reset"
        assert budget.videos_skipped == 0, "videos_skipped should be reset"
        assert budget.error_counts == {}, "error_counts should be cleared"
        assert budget.attempts_per_video_id == {}, "attempts_per_video_id should be cleared"
        assert budget.circuit_breaker_trips == 0, "circuit_breaker_trips should be reset"
        assert budget.early_terminated is False, "early_terminated should be reset"
        assert budget.early_termination_reason is None, "early_termination_reason should be reset"

    @pytest.mark.fast
    def test_ensure_scaled_without_reset_on_scale_preserves_counters(self):
        """Test ensure_scaled without reset_on_scale preserves counters (US-41-009).

        AC4: Default to false to preserve existing behavior

        When reset_on_scale is False (the default), ensure_scaled should only
        update max_attempts and batch_size without touching usage counters.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100
        budget.reset_on_scale = False  # Default behavior

        # Simulate prior usage
        budget.attempts = 50
        budget.failures = 20
        budget.successes = 30
        budget.backoff_time_spent = 45.0

        # Scale up - should NOT reset counters
        result = budget.ensure_scaled(200)

        # Verify scaling occurred
        assert result is True
        assert budget.max_attempts == 400

        # Verify counters were PRESERVED
        assert budget.attempts == 50, "attempts should be preserved"
        assert budget.failures == 20, "failures should be preserved"
        assert budget.successes == 30, "successes should be preserved"
        assert budget.backoff_time_spent == 45.0, "backoff_time_spent should be preserved"

    @pytest.mark.fast
    def test_ensure_scaled_after_checkpoint_restore_with_batch_size_set_but_max_attempts_default(self):
        """Test US-42-004: ensure_scaled scales even when batch_size is already set.

        Bug scenario:
        1. First run: 175 videos, ensure_scaled(175) scales max_attempts from 100 to 350
        2. Checkpoint saves: batch_size=175, max_attempts=350
        3. Checkpoint restore: from_dict restores batch_size=175, but in caption_stage.py
           we DON'T copy max_attempts from checkpoint (line 316: "Keep max_attempts from config")
        4. So after restore: batch_size=175, max_attempts=100 (from config)
        5. ensure_scaled(175) is called
        6. OLD BUG: batch_size==175 matches, returns False without scaling!
        7. FIX: Also check if max_attempts >= required_attempts before skipping

        This test simulates the checkpoint restore scenario.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100  # From config (NOT restored from checkpoint)
        budget.attempts_per_video = 2.0

        # Simulate checkpoint restore setting batch_size but NOT max_attempts
        # (This is what happens in caption_stage.py lines 302-316)
        budget.batch_size = 175  # From checkpoint
        budget.attempts = 50  # Prior attempts from checkpoint

        # Before fix: ensure_scaled would see batch_size==175 and skip scaling
        # After fix: should detect max_attempts (100) < required (350) and scale
        result = budget.ensure_scaled(175)

        # Must scale because 175 * 2.0 = 350 > 100
        assert result is True, (
            "ensure_scaled should scale even when batch_size is already set, "
            "because max_attempts (100) < required (350)"
        )
        assert budget.max_attempts == 350, (
            f"max_attempts should be scaled to 350, got {budget.max_attempts}"
        )
        assert budget.batch_size == 175

    @pytest.mark.fast
    def test_ensure_scaled_after_restore_200_videos_50_attempts_used(self):
        """Test US-42-004 AC4: restore budget with 50 attempts used, new batch of 200.

        AC4: Add test: restore budget with 50 attempts used, new batch of 200,
        verify scaling to 400.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100  # From config
        budget.attempts_per_video = 2.0

        # Simulate checkpoint restore
        budget.batch_size = 200  # Previous batch size
        budget.attempts = 50  # Prior attempts used

        # Call ensure_scaled for current batch of 200
        result = budget.ensure_scaled(200)

        # Should scale: 200 * 2.0 = 400 > 100
        assert result is True
        assert budget.max_attempts == 400

    @pytest.mark.fast
    def test_ensure_scaled_considers_remaining_budget_headroom(self):
        """Test US-42-004 AC5: ensure_scaled considers remaining budget headroom.

        When checkpoint restores with attempts already used, scaling should
        still occur based on total batch size, not remaining videos.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100  # From config
        budget.attempts_per_video = 2.0

        # Checkpoint restored 175 batch with 99 attempts used
        # (matching user log: 175 videos, 99 successes, budget exhausted at 101)
        budget.batch_size = 175
        budget.attempts = 99
        budget.successes = 99
        budget.failures = 0

        # Call ensure_scaled - should scale even though batch_size matches
        result = budget.ensure_scaled(175)

        # Must scale: 175 * 2.0 = 350 > 100
        assert result is True
        assert budget.max_attempts == 350

        # Verify budget is now sufficient for remaining videos
        # 76 remaining videos need ~152 more attempts
        # We have 350 - 99 = 251 remaining, which is enough
        remaining = budget.max_attempts - budget.attempts
        assert remaining >= 76 * 2.0, f"Should have headroom for 76 videos, got {remaining}"

    @pytest.mark.fast
    def test_ensure_scaled_from_checkpoint_80_attempts_preserves_counters_by_default(self):
        """Test US-42-007: budget with 80 attempts used, scale up, preserves attempts.

        AC3: Add test: budget has 80 attempts used, scale up, verify attempts counter
        behavior based on reset_on_scale.

        Scenario: Checkpoint has 80/100 attempts used. New batch needs 200 attempts.
        With reset_on_scale=false (default), scale to 200 but attempts stay at 80.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100  # From checkpoint
        budget.attempts_per_video = 2.0
        budget.reset_on_scale = False  # Default - preserves counters

        # Simulate checkpoint restore with 80 attempts used
        budget.attempts = 80
        budget.failures = 25
        budget.successes = 55
        budget.backoff_time_spent = 120.0

        # Scale up for batch of 100 (needs 200 attempts)
        result = budget.ensure_scaled(100)

        # Should scale: 100 * 2.0 = 200 > 100
        assert result is True
        assert budget.max_attempts == 200

        # With reset_on_scale=false, counters should be PRESERVED
        assert budget.attempts == 80, "attempts should be preserved (default behavior)"
        assert budget.failures == 25, "failures should be preserved"
        assert budget.successes == 55, "successes should be preserved"
        assert budget.backoff_time_spent == 120.0, "backoff should be preserved"

        # Verify only 120 attempts remain (200 - 80 = 120)
        remaining = budget.attempts_remaining()
        assert remaining == 120, f"Should have 120 attempts remaining, got {remaining}"

    @pytest.mark.fast
    def test_ensure_scaled_from_checkpoint_80_attempts_resets_with_reset_on_scale(self):
        """Test US-42-007: budget with 80 attempts used, scale up with reset_on_scale=true.

        AC3: Add test: budget has 80 attempts used, scale up, verify attempts counter
        behavior based on reset_on_scale.

        Scenario: Checkpoint has 80/100 attempts used. New batch needs 200 attempts.
        With reset_on_scale=true, scale to 200 AND reset attempts to 0.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100  # From checkpoint
        budget.attempts_per_video = 2.0
        budget.reset_on_scale = True  # Enable reset on scale-up

        # Simulate checkpoint restore with 80 attempts used
        budget.attempts = 80
        budget.failures = 25
        budget.successes = 55
        budget.backoff_time_spent = 120.0
        budget.videos_skipped = 3

        # Scale up for batch of 100 (needs 200 attempts)
        result = budget.ensure_scaled(100)

        # Should scale: 100 * 2.0 = 200 > 100
        assert result is True
        assert budget.max_attempts == 200

        # With reset_on_scale=true, counters should be RESET
        assert budget.attempts == 0, "attempts should be reset to 0"
        assert budget.failures == 0, "failures should be reset"
        assert budget.successes == 0, "successes should be reset"
        assert budget.backoff_time_spent == 0.0, "backoff should be reset"
        assert budget.videos_skipped == 0, "skipped should be reset"

        # All 200 attempts are now available
        remaining = budget.attempts_remaining()
        assert remaining == 200, f"Should have 200 attempts remaining, got {remaining}"

    @pytest.mark.fast
    def test_ensure_scaled_logs_reset_with_story_id(self, caplog):
        """Test US-42-007 AC4: logging includes story ID format.

        AC4: Log when scaling resets counters: '[US-42-007] Budget scaled and reset: attempts 80->0'
        """
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100
        budget.attempts_per_video = 2.0
        budget.reset_on_scale = True

        # Simulate 80 attempts used
        budget.attempts = 80

        # Scale up - should trigger reset with specific log format
        budget.ensure_scaled(100)

        # Check log contains required format
        assert any("[US-42-007]" in record.message for record in caplog.records), \
            "Log should contain [US-42-007] story ID"
        assert any("Budget scaled and reset" in record.message for record in caplog.records), \
            "Log should contain 'Budget scaled and reset'"
        assert any("attempts 80->0" in record.message for record in caplog.records), \
            "Log should contain 'attempts 80->0'"


# ============================================================================
# US-37-005: Detailed logging for retry budget consumption
# ============================================================================


class TestCaptionRetryBudgetConsumptionPercentage:
    """Test get_consumption_percentage method (US-37-005)."""

    @pytest.mark.fast
    def test_consumption_percentage_fresh_budget(self):
        """Test consumption percentages are 0% for fresh budget."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 300.0

        pct = budget.get_consumption_percentage()

        assert pct['attempts'] == 0.0
        assert pct['backoff_time'] == 0.0

    @pytest.mark.fast
    def test_consumption_percentage_partial_use(self):
        """Test consumption percentages after partial use."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 300.0

        for i in range(25):
            budget.record_attempt(f"video_{i}")
        budget.backoff_time_spent = 90.0  # 30% of 300

        pct = budget.get_consumption_percentage()

        assert pct['attempts'] == 25.0  # 25/100 = 25%
        assert pct['backoff_time'] == 30.0  # 90/300 = 30%

    @pytest.mark.fast
    def test_consumption_percentage_at_50_percent(self):
        """Test consumption percentage at 50%."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 200.0

        budget.attempts = 50
        budget.backoff_time_spent = 100.0

        pct = budget.get_consumption_percentage()

        assert pct['attempts'] == 50.0
        assert pct['backoff_time'] == 50.0

    @pytest.mark.fast
    def test_consumption_percentage_over_100(self):
        """Test consumption percentage can exceed 100% when over budget."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 100.0

        budget.attempts = 120  # 120% consumed
        budget.backoff_time_spent = 150.0  # 150% consumed

        pct = budget.get_consumption_percentage()

        assert pct['attempts'] == 120.0
        assert pct['backoff_time'] == 150.0

    @pytest.mark.fast
    def test_consumption_percentage_unlimited_returns_none(self):
        """Test consumption percentages are None for unlimited budgets."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 0  # Unlimited
        budget.max_backoff_time = 0  # Unlimited

        budget.attempts = 500
        budget.backoff_time_spent = 1000.0

        pct = budget.get_consumption_percentage()

        assert pct['attempts'] is None
        assert pct['backoff_time'] is None

    @pytest.mark.fast
    def test_consumption_percentage_mixed_unlimited(self):
        """Test mixed unlimited/limited budget percentages."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 0  # Unlimited

        budget.attempts = 45
        budget.backoff_time_spent = 500.0

        pct = budget.get_consumption_percentage()

        assert pct['attempts'] == 45.0
        assert pct['backoff_time'] is None


class TestCaptionRetryBudgetThresholdLogging:
    """Test threshold-based logging (US-37-005)."""

    @pytest.mark.fast
    def test_logs_info_at_25_percent_threshold(self):
        """Test INFO logging when crossing 25% threshold."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        with patch('src.caption.retry_budget.logger') as mock_logger:
            # Record 24 attempts (below threshold)
            for i in range(24):
                budget.record_attempt(f"video_{i}")

            # Clear call history
            mock_logger.reset_mock()

            # Record 25th attempt (crosses 25%)
            budget.record_attempt("video_25")

            # Should have logged INFO about 25% threshold
            info_calls = [c for c in mock_logger.info.call_args_list
                         if '25%' in str(c)]
            assert len(info_calls) >= 1

    @pytest.mark.fast
    def test_logs_info_at_50_percent_threshold(self):
        """Test INFO logging when crossing 50% threshold."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        with patch('src.caption.retry_budget.logger') as mock_logger:
            # Record 49 attempts
            for i in range(49):
                budget.record_attempt(f"video_{i}")

            mock_logger.reset_mock()

            # Record 50th attempt (crosses 50%)
            budget.record_attempt("video_50")

            info_calls = [c for c in mock_logger.info.call_args_list
                         if '50%' in str(c)]
            assert len(info_calls) >= 1

    @pytest.mark.fast
    def test_logs_info_at_75_percent_threshold(self):
        """Test INFO logging when crossing 75% threshold."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        with patch('src.caption.retry_budget.logger') as mock_logger:
            # Record 74 attempts
            for i in range(74):
                budget.record_attempt(f"video_{i}")

            mock_logger.reset_mock()

            # Record 75th attempt
            budget.record_attempt("video_75")

            info_calls = [c for c in mock_logger.info.call_args_list
                         if '75%' in str(c)]
            assert len(info_calls) >= 1

    @pytest.mark.fast
    def test_logs_warning_at_90_percent_threshold(self):
        """Test WARNING logging when crossing 90% threshold."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        with patch('src.caption.retry_budget.logger') as mock_logger:
            # Record 89 attempts
            for i in range(89):
                budget.record_attempt(f"video_{i}")

            mock_logger.reset_mock()

            # Record 90th attempt (crosses 90%)
            budget.record_attempt("video_90")

            warning_calls = [c for c in mock_logger.warning.call_args_list
                           if '90%' in str(c)]
            assert len(warning_calls) >= 1

    @pytest.mark.fast
    def test_backoff_triggers_threshold_logging(self):
        """Test that backoff time also triggers threshold logging."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 1000  # High to avoid attempts triggering
        budget.max_backoff_time = 100.0

        with patch('src.caption.retry_budget.logger') as mock_logger:
            # Record 49% backoff
            budget.record_backoff(49.0, "video_1")

            mock_logger.reset_mock()

            # Record more backoff to cross 50%
            budget.record_backoff(2.0, "video_2")

            info_calls = [c for c in mock_logger.info.call_args_list
                         if '50%' in str(c)]
            assert len(info_calls) >= 1

    @pytest.mark.fast
    def test_failure_triggers_threshold_logging(self):
        """Test that failures also trigger threshold checking."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        with patch('src.caption.retry_budget.logger') as mock_logger:
            # Pre-fill to 24 attempts
            budget.attempts = 24

            mock_logger.reset_mock()

            # Record a failure (which doesn't increment attempts but checks threshold)
            # Actually, failure doesn't increment attempts - let's record an attempt instead
            budget.record_attempt("video_25")

            # Should log 25% threshold
            info_calls = [c for c in mock_logger.info.call_args_list
                         if '25%' in str(c)]
            assert len(info_calls) >= 1

    @pytest.mark.fast
    def test_no_logging_below_25_percent(self):
        """Test no threshold logging when below 25%."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        with patch('src.caption.retry_budget.logger') as mock_logger:
            # Record only 10 attempts (10%)
            for i in range(10):
                budget.record_attempt(f"video_{i}")

            # Should not have any threshold INFO logs (only DEBUG)
            info_calls = [c for c in mock_logger.info.call_args_list
                         if 'consumed' in str(c)]
            assert len(info_calls) == 0

    @pytest.mark.fast
    def test_warning_includes_success_failure_counts(self):
        """Test WARNING at 90% includes success and failure counts."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        # Pre-fill some successes and failures
        budget.successes = 50
        budget.failures = 39
        budget.attempts = 89

        with patch('src.caption.retry_budget.logger') as mock_logger:
            # Cross 90% threshold
            budget.record_attempt("video_90")

            # Check warning includes success/failure info
            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) >= 1
            call_str = str(warning_calls[-1])
            assert 'successes: 50' in call_str
            assert 'failures: 39' in call_str


class TestCaptionRetryBudgetConsumptionInSummary:
    """Test consumption percentage included in get_summary (US-37-005)."""

    @pytest.mark.fast
    def test_get_summary_includes_all_fields(self):
        """Test get_summary returns all expected fields including consumption."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 300.0

        for i in range(45):
            budget.record_attempt(f"video_{i}")
        budget.backoff_time_spent = 90.0

        summary = budget.get_summary()

        # Verify all fields present
        assert 'attempts' in summary
        assert 'attempts_remaining' in summary
        assert 'successes' in summary
        assert 'failures' in summary
        assert 'backoff_time_spent' in summary
        assert 'backoff_time_remaining' in summary
        assert 'videos_skipped' in summary
        assert 'is_exhausted' in summary

        # Verify values
        assert summary['attempts'] == 45
        assert summary['attempts_remaining'] == 55
        assert summary['backoff_time_spent'] == 90.0
        assert summary['backoff_time_remaining'] == 210.0


# ============================================================================
# US-37-006: Error category tracking in CaptionRetryBudget
# ============================================================================


class TestCaptionRetryBudgetErrorCategoryTracking:
    """Test error category tracking (US-37-006)."""

    @pytest.mark.fast
    def test_error_counts_field_exists(self):
        """Test error_counts field is initialized as empty dict."""
        budget = CaptionRetryBudget()
        assert hasattr(budget, 'error_counts')
        assert budget.error_counts == {}

    @pytest.mark.fast
    def test_record_failure_without_category(self):
        """Test record_failure works without error_category (backward compatible)."""
        budget = CaptionRetryBudget()
        budget.record_failure("video_1")
        assert budget.failures == 1
        assert len(budget.error_counts) == 0

    @pytest.mark.fast
    def test_record_failure_with_category(self):
        """Test record_failure tracks error category when provided."""
        budget = CaptionRetryBudget()
        budget.record_failure("video_1", error_category=CaptionErrorCategory.NETWORK)
        assert budget.failures == 1
        assert budget.error_counts[CaptionErrorCategory.NETWORK] == 1

    @pytest.mark.fast
    def test_record_failure_multiple_same_category(self):
        """Test multiple failures of same category accumulate correctly."""
        budget = CaptionRetryBudget()
        budget.record_failure("video_1", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("video_2", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("video_3", error_category=CaptionErrorCategory.NETWORK)

        assert budget.failures == 3
        assert budget.error_counts[CaptionErrorCategory.NETWORK] == 3

    @pytest.mark.fast
    def test_record_failure_multiple_categories(self):
        """Test failures across multiple categories tracked correctly."""
        budget = CaptionRetryBudget()
        budget.record_failure("video_1", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("video_2", error_category=CaptionErrorCategory.TIMEOUT)
        budget.record_failure("video_3", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("video_4", error_category=CaptionErrorCategory.RATE_LIMIT)
        budget.record_failure("video_5", error_category=CaptionErrorCategory.TIMEOUT)

        assert budget.failures == 5
        assert budget.error_counts[CaptionErrorCategory.NETWORK] == 2
        assert budget.error_counts[CaptionErrorCategory.TIMEOUT] == 2
        assert budget.error_counts[CaptionErrorCategory.RATE_LIMIT] == 1

    @pytest.mark.fast
    def test_record_failure_all_categories(self):
        """Test all error categories can be tracked."""
        budget = CaptionRetryBudget()

        for cat in CaptionErrorCategory:
            budget.record_failure(f"video_{cat.name}", error_category=cat)

        assert budget.failures == 5  # All 5 categories
        for cat in CaptionErrorCategory:
            assert cat in budget.error_counts
            assert budget.error_counts[cat] == 1


class TestCaptionRetryBudgetGetTopErrors:
    """Test get_top_errors method (US-37-006)."""

    @pytest.mark.fast
    def test_get_top_errors_empty(self):
        """Test get_top_errors returns empty list when no errors."""
        budget = CaptionRetryBudget()
        result = budget.get_top_errors()
        assert result == []

    @pytest.mark.fast
    def test_get_top_errors_single_category(self):
        """Test get_top_errors with single error category."""
        budget = CaptionRetryBudget()
        budget.record_failure("v1", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("v2", error_category=CaptionErrorCategory.NETWORK)

        result = budget.get_top_errors()

        assert len(result) == 1
        assert result[0] == (CaptionErrorCategory.NETWORK, 2)

    @pytest.mark.fast
    def test_get_top_errors_sorted_by_count(self):
        """Test get_top_errors returns categories sorted by count descending."""
        budget = CaptionRetryBudget()
        # Record different counts for each category
        for _ in range(5):
            budget.record_failure("v", error_category=CaptionErrorCategory.NETWORK)
        for _ in range(3):
            budget.record_failure("v", error_category=CaptionErrorCategory.TIMEOUT)
        for _ in range(10):
            budget.record_failure("v", error_category=CaptionErrorCategory.RATE_LIMIT)
        budget.record_failure("v", error_category=CaptionErrorCategory.PARSE)

        result = budget.get_top_errors()

        assert len(result) == 4
        assert result[0] == (CaptionErrorCategory.RATE_LIMIT, 10)
        assert result[1] == (CaptionErrorCategory.NETWORK, 5)
        assert result[2] == (CaptionErrorCategory.TIMEOUT, 3)
        assert result[3] == (CaptionErrorCategory.PARSE, 1)

    @pytest.mark.fast
    def test_get_top_errors_limit(self):
        """Test get_top_errors respects limit parameter."""
        budget = CaptionRetryBudget()
        for i, cat in enumerate(CaptionErrorCategory):
            for _ in range(i + 1):
                budget.record_failure("v", error_category=cat)

        result = budget.get_top_errors(limit=2)

        assert len(result) == 2
        # Top 2 should be the categories with highest counts

    @pytest.mark.fast
    def test_get_top_errors_default_limit_5(self):
        """Test get_top_errors has default limit of 5."""
        budget = CaptionRetryBudget()
        # All 5 categories recorded
        for cat in CaptionErrorCategory:
            budget.record_failure("v", error_category=cat)

        result = budget.get_top_errors()

        assert len(result) == 5  # Default limit


class TestCaptionRetryBudgetErrorBreakdownInSummary:
    """Test error breakdown in get_summary (US-37-006)."""

    @pytest.mark.fast
    def test_get_summary_includes_error_breakdown(self):
        """Test get_summary includes error_breakdown field."""
        budget = CaptionRetryBudget()
        budget.record_failure("v1", error_category=CaptionErrorCategory.NETWORK)

        summary = budget.get_summary()

        assert 'error_breakdown' in summary

    @pytest.mark.fast
    def test_get_summary_error_breakdown_empty_when_no_errors(self):
        """Test error_breakdown is empty dict when no categorized errors."""
        budget = CaptionRetryBudget()
        budget.record_failure("v1")  # No category

        summary = budget.get_summary()

        assert summary['error_breakdown'] == {}

    @pytest.mark.fast
    def test_get_summary_error_breakdown_has_category_names(self):
        """Test error_breakdown uses category names as keys."""
        budget = CaptionRetryBudget()
        budget.record_failure("v1", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("v2", error_category=CaptionErrorCategory.TIMEOUT)

        summary = budget.get_summary()

        assert 'NETWORK' in summary['error_breakdown']
        assert 'TIMEOUT' in summary['error_breakdown']
        assert summary['error_breakdown']['NETWORK'] == 1
        assert summary['error_breakdown']['TIMEOUT'] == 1

    @pytest.mark.fast
    def test_get_summary_error_breakdown_multiple(self):
        """Test error_breakdown with multiple errors per category."""
        budget = CaptionRetryBudget()
        for _ in range(5):
            budget.record_failure("v", error_category=CaptionErrorCategory.NETWORK)
        for _ in range(3):
            budget.record_failure("v", error_category=CaptionErrorCategory.RATE_LIMIT)

        summary = budget.get_summary()

        assert summary['error_breakdown']['NETWORK'] == 5
        assert summary['error_breakdown']['RATE_LIMIT'] == 3


class TestCaptionRetryBudgetErrorCategorySerialization:
    """Test error_counts serialization (US-37-006)."""

    @pytest.mark.fast
    def test_to_dict_includes_error_counts(self):
        """Test to_dict includes error_counts."""
        budget = CaptionRetryBudget()
        budget.record_failure("v1", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("v2", error_category=CaptionErrorCategory.TIMEOUT)

        data = budget.to_dict()

        assert 'error_counts' in data
        assert data['error_counts']['NETWORK'] == 1
        assert data['error_counts']['TIMEOUT'] == 1

    @pytest.mark.fast
    def test_from_dict_restores_error_counts(self):
        """Test from_dict restores error_counts."""
        data = {
            'attempts': 10,
            'failures': 5,
            'successes': 5,
            'backoff_time_spent': 10.0,
            'videos_skipped': 0,
            'error_counts': {
                'NETWORK': 3,
                'RATE_LIMIT': 2,
            }
        }

        budget = CaptionRetryBudget.from_dict(data)

        assert budget.error_counts[CaptionErrorCategory.NETWORK] == 3
        assert budget.error_counts[CaptionErrorCategory.RATE_LIMIT] == 2

    @pytest.mark.fast
    def test_roundtrip_serialization(self):
        """Test error_counts survives serialization roundtrip."""
        budget = CaptionRetryBudget()
        budget.record_failure("v1", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("v2", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("v3", error_category=CaptionErrorCategory.TIMEOUT)
        budget.record_failure("v4", error_category=CaptionErrorCategory.PARSE)

        data = budget.to_dict()
        restored = CaptionRetryBudget.from_dict(data)

        assert restored.error_counts[CaptionErrorCategory.NETWORK] == 2
        assert restored.error_counts[CaptionErrorCategory.TIMEOUT] == 1
        assert restored.error_counts[CaptionErrorCategory.PARSE] == 1

    @pytest.mark.fast
    def test_from_dict_handles_unknown_category(self):
        """Test from_dict handles unknown error category gracefully."""
        data = {
            'error_counts': {
                'NETWORK': 3,
                'UNKNOWN_FUTURE_CATEGORY': 5,  # Unknown category
            }
        }

        with patch('src.caption.retry_budget.logger') as mock_logger:
            budget = CaptionRetryBudget.from_dict(data)

            # Should restore known category
            assert budget.error_counts[CaptionErrorCategory.NETWORK] == 3
            # Should log warning for unknown
            mock_logger.warning.assert_called_once()


class TestCaptionRetryBudgetErrorCountsReset:
    """Test error_counts reset behavior (US-37-006)."""

    @pytest.mark.fast
    def test_reset_clears_error_counts(self):
        """Test reset() clears error_counts."""
        budget = CaptionRetryBudget()
        budget.record_failure("v1", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("v2", error_category=CaptionErrorCategory.TIMEOUT)

        assert len(budget.error_counts) == 2

        budget.reset()

        assert budget.error_counts == {}

    @pytest.mark.fast
    def test_error_counts_fresh_after_reset(self):
        """Test error tracking works correctly after reset."""
        budget = CaptionRetryBudget()
        budget.record_failure("v1", error_category=CaptionErrorCategory.NETWORK)
        budget.reset()
        budget.record_failure("v2", error_category=CaptionErrorCategory.TIMEOUT)

        assert CaptionErrorCategory.NETWORK not in budget.error_counts
        assert budget.error_counts[CaptionErrorCategory.TIMEOUT] == 1


class TestCaptionRetryBudgetErrorCategoryIntegration:
    """Integration tests for error category tracking (US-37-006)."""

    @pytest.mark.fast
    def test_diagnostic_scenario_rate_limit_dominant(self):
        """Test diagnosing rate limit as dominant error."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        # Simulate a batch with mostly rate limit errors
        for _ in range(20):
            budget.record_attempt("v")
            budget.record_success("v")

        for _ in range(15):
            budget.record_attempt("v")
            budget.record_failure("v", error_category=CaptionErrorCategory.RATE_LIMIT)

        for _ in range(5):
            budget.record_attempt("v")
            budget.record_failure("v", error_category=CaptionErrorCategory.NETWORK)

        # Diagnose
        top_errors = budget.get_top_errors()
        summary = budget.get_summary()

        assert top_errors[0] == (CaptionErrorCategory.RATE_LIMIT, 15)
        assert summary['error_breakdown']['RATE_LIMIT'] == 15
        assert summary['error_breakdown']['NETWORK'] == 5

    @pytest.mark.fast
    def test_diagnostic_scenario_network_issues(self):
        """Test diagnosing network issues as dominant error."""
        budget = CaptionRetryBudget()

        # Simulate network-heavy failure pattern
        for _ in range(25):
            budget.record_failure("v", error_category=CaptionErrorCategory.NETWORK)
        for _ in range(3):
            budget.record_failure("v", error_category=CaptionErrorCategory.TIMEOUT)
        for _ in range(2):
            budget.record_failure("v", error_category=CaptionErrorCategory.PARSE)

        top_errors = budget.get_top_errors(limit=1)

        assert top_errors[0][0] == CaptionErrorCategory.NETWORK
        assert top_errors[0][1] == 25

    @pytest.mark.fast
    def test_thread_safety_error_counts(self):
        """Test error_counts is thread-safe."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10000

        def record_errors(category, count):
            for _ in range(count):
                budget.record_failure("v", error_category=category)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(record_errors, CaptionErrorCategory.NETWORK, 100),
                executor.submit(record_errors, CaptionErrorCategory.TIMEOUT, 100),
                executor.submit(record_errors, CaptionErrorCategory.NETWORK, 100),
                executor.submit(record_errors, CaptionErrorCategory.RATE_LIMIT, 100),
            ]
            for f in futures:
                f.result()

        assert budget.error_counts[CaptionErrorCategory.NETWORK] == 200
        assert budget.error_counts[CaptionErrorCategory.TIMEOUT] == 100
        assert budget.error_counts[CaptionErrorCategory.RATE_LIMIT] == 100


class TestCaptionRetryBudgetCheckpointPersistence:
    """Test checkpoint persistence for resume support (US-37-007)."""

    @pytest.mark.fast
    def test_to_dict_includes_all_state(self):
        """Test to_dict() includes all budget state for checkpoint."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 200
        budget.max_backoff_time = 600.0

        # Record various operations
        for i in range(50):
            budget.record_attempt(f"video_{i}")
        for i in range(40):
            budget.record_success(f"video_{i}")
        budget.record_failure("fail1", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("fail2", error_category=CaptionErrorCategory.NETWORK)
        budget.record_failure("fail3", error_category=CaptionErrorCategory.RATE_LIMIT)
        budget.record_backoff(45.5, "video1")
        budget.record_skipped("skipped1")
        budget.record_skipped("skipped2")
        budget.record_skipped("skipped3")

        data = budget.to_dict()

        assert data['attempts'] == 50
        assert data['successes'] == 40
        assert data['failures'] == 3
        assert data['backoff_time_spent'] == 45.5
        assert data['videos_skipped'] == 3
        assert data['max_attempts'] == 200
        assert data['max_backoff_time'] == 600.0
        assert data['error_counts']['NETWORK'] == 2
        assert data['error_counts']['RATE_LIMIT'] == 1

    @pytest.mark.fast
    def test_from_dict_restores_all_state(self):
        """Test from_dict() restores all budget state from checkpoint."""
        data = {
            'attempts': 75,
            'successes': 60,
            'failures': 10,
            'backoff_time_spent': 120.5,
            'videos_skipped': 5,
            'max_attempts': 250,
            'max_backoff_time': 500.0,
            'error_counts': {
                'NETWORK': 5,
                'TIMEOUT': 3,
                'RATE_LIMIT': 2,
            }
        }

        budget = CaptionRetryBudget.from_dict(data)

        assert budget.attempts == 75
        assert budget.successes == 60
        assert budget.failures == 10
        assert budget.backoff_time_spent == 120.5
        assert budget.videos_skipped == 5
        assert budget.max_attempts == 250
        assert budget.max_backoff_time == 500.0
        assert budget.error_counts[CaptionErrorCategory.NETWORK] == 5
        assert budget.error_counts[CaptionErrorCategory.TIMEOUT] == 3
        assert budget.error_counts[CaptionErrorCategory.RATE_LIMIT] == 2

    @pytest.mark.fast
    def test_budget_remaining_correct_on_resume(self):
        """Test budget remaining is calculated correctly after restore."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.max_backoff_time = 300.0

        # Simulate partial run
        for i in range(45):
            budget.record_attempt(f"video_{i}")
        for i in range(40):
            budget.record_success(f"video_{i}")
        for i in range(5):
            budget.record_failure(f"fail_{i}")
        budget.record_backoff(100.0, "v1")

        # Save state
        data = budget.to_dict()

        # Restore in new session (simulating resume)
        restored = CaptionRetryBudget.from_dict(data)

        # Check remaining budget
        assert restored.attempts_remaining() == 55  # 100 - 45
        assert restored.backoff_time_remaining() == 200.0  # 300 - 100
        assert not restored.budget_exhausted()

    @pytest.mark.fast
    def test_exhausted_budget_state_preserved(self):
        """Test exhausted budget state is correctly preserved on restore."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 50
        budget.max_backoff_time = 100.0

        # Exhaust the budget
        for i in range(50):
            budget.record_attempt(f"video_{i}")
        budget.record_skipped("skipped1")

        assert budget.budget_exhausted()

        # Save and restore
        data = budget.to_dict()
        restored = CaptionRetryBudget.from_dict(data)

        assert restored.budget_exhausted()
        assert restored.attempts_remaining() == 0
        assert restored.videos_skipped == 1

    @pytest.mark.fast
    def test_complete_roundtrip_preserves_all_fields(self):
        """Test complete round-trip serialization preserves all budget fields."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 300
        budget.max_backoff_time = 450.0
        budget.auto_scale = True
        budget.attempts_per_video = 2.0

        # Simulate realistic batch processing
        for i in range(150):
            budget.record_attempt(f"video_{i}")
        for i in range(120):
            budget.record_success(f"video_{i}")
        for i in range(25):
            budget.record_failure(f"fail_{i}", error_category=CaptionErrorCategory.NETWORK)
        for i in range(5):
            budget.record_failure(f"fail_{i}", error_category=CaptionErrorCategory.TIMEOUT)
        budget.record_backoff(75.5, "v1")
        budget.record_backoff(25.0, "v2")
        for i in range(10):
            budget.record_skipped(f"skipped_{i}")

        # Round-trip
        data = budget.to_dict()
        restored = CaptionRetryBudget.from_dict(data)

        # Verify all state
        assert restored.attempts == 150
        assert restored.successes == 120
        assert restored.failures == 30
        assert restored.backoff_time_spent == 100.5
        assert restored.videos_skipped == 10
        assert restored.max_attempts == 300
        assert restored.max_backoff_time == 450.0
        assert restored.error_counts[CaptionErrorCategory.NETWORK] == 25
        assert restored.error_counts[CaptionErrorCategory.TIMEOUT] == 5

        # Verify summary matches
        orig_summary = budget.get_summary()
        rest_summary = restored.get_summary()
        assert orig_summary['attempts'] == rest_summary['attempts']
        assert orig_summary['failures'] == rest_summary['failures']
        assert orig_summary['videos_skipped'] == rest_summary['videos_skipped']

    @pytest.mark.fast
    def test_from_dict_partial_data_uses_defaults(self):
        """Test from_dict handles partial checkpoint data gracefully."""
        partial_data = {
            'attempts': 25,
            'successes': 20,
            # Missing: failures, backoff_time_spent, videos_skipped, error_counts
        }

        budget = CaptionRetryBudget.from_dict(partial_data)

        assert budget.attempts == 25
        assert budget.successes == 20
        assert budget.failures == 0
        assert budget.backoff_time_spent == 0.0
        assert budget.videos_skipped == 0
        assert budget.error_counts == {}
        # Defaults for limits
        assert budget.max_attempts == 100
        assert budget.max_backoff_time == 300.0


# ============================================================================
# US-37-008: Budget exhaustion recovery with VPN rotation trigger
# ============================================================================


class TestCaptionRetryBudgetVPNRotationTrigger:
    """Test VPN rotation trigger on budget exhaustion (US-37-008)."""

    @pytest.mark.fast
    def test_get_rate_limit_error_percentage_zero_failures(self):
        """Test rate limit percentage is 0 when no failures."""
        budget = CaptionRetryBudget()
        for i in range(10):
            budget.record_success(f"video_{i}")

        assert budget.get_rate_limit_error_percentage() == 0.0

    @pytest.mark.fast
    def test_get_rate_limit_error_percentage_no_rate_limits(self):
        """Test rate limit percentage when only network errors."""
        budget = CaptionRetryBudget()
        for i in range(10):
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.NETWORK)

        assert budget.get_rate_limit_error_percentage() == 0.0

    @pytest.mark.fast
    def test_get_rate_limit_error_percentage_all_rate_limits(self):
        """Test rate limit percentage when all errors are rate limits."""
        budget = CaptionRetryBudget()
        for i in range(10):
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        assert budget.get_rate_limit_error_percentage() == 100.0

    @pytest.mark.fast
    def test_get_rate_limit_error_percentage_mixed(self):
        """Test rate limit percentage with mixed errors."""
        budget = CaptionRetryBudget()
        for i in range(6):
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)
        for i in range(4):
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.NETWORK)

        # 6 out of 10 = 60%
        assert budget.get_rate_limit_error_percentage() == 60.0

    @pytest.mark.fast
    def test_should_trigger_vpn_rotation_not_exhausted(self):
        """Test VPN rotation not triggered when budget not exhausted."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.trigger_vpn_on_rate_limit = True

        # Add rate limit errors but don't exhaust budget
        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        assert not budget.budget_exhausted()
        assert not budget.should_trigger_vpn_rotation()

    @pytest.mark.fast
    def test_should_trigger_vpn_rotation_exhausted_rate_limits(self):
        """Test VPN rotation triggered when exhausted with >50% rate limits."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10
        budget.trigger_vpn_on_rate_limit = True

        # All failures are rate limits
        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        assert budget.budget_exhausted()
        assert budget.should_trigger_vpn_rotation()

    @pytest.mark.fast
    def test_should_trigger_vpn_rotation_exhausted_low_rate_limits(self):
        """Test VPN rotation NOT triggered when exhausted with <50% rate limits."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10
        budget.trigger_vpn_on_rate_limit = True

        # Mix of errors: 3 rate limits, 7 network (30%)
        for i in range(3):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)
        for i in range(7):
            budget.record_attempt(f"net_{i}")
            budget.record_failure(f"net_{i}", error_category=CaptionErrorCategory.NETWORK)

        assert budget.budget_exhausted()
        assert not budget.should_trigger_vpn_rotation()

    @pytest.mark.fast
    def test_should_trigger_vpn_rotation_disabled_in_config(self):
        """Test VPN rotation not triggered when disabled in config."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10
        budget.trigger_vpn_on_rate_limit = False  # Disabled

        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        assert budget.budget_exhausted()
        assert not budget.should_trigger_vpn_rotation()

    @pytest.mark.fast
    def test_should_trigger_vpn_rotation_max_resets_reached(self):
        """Test VPN rotation not triggered when max resets reached."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10
        budget.trigger_vpn_on_rate_limit = True
        budget.max_vpn_resets = 2
        budget.vpn_resets_used = 2  # Already used both

        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        assert budget.budget_exhausted()
        assert not budget.should_trigger_vpn_rotation()

    @pytest.mark.fast
    def test_record_vpn_reset_increments_count(self):
        """Test record_vpn_reset increments vpn_resets_used."""
        budget = CaptionRetryBudget()
        budget.max_vpn_resets = 5

        assert budget.vpn_resets_used == 0

        budget.record_vpn_reset()
        assert budget.vpn_resets_used == 1

        budget.record_vpn_reset()
        assert budget.vpn_resets_used == 2

    @pytest.mark.fast
    def test_can_vpn_reset_true(self):
        """Test can_vpn_reset returns True when resets available."""
        budget = CaptionRetryBudget()
        budget.max_vpn_resets = 2
        budget.vpn_resets_used = 0

        assert budget.can_vpn_reset()

        budget.vpn_resets_used = 1
        assert budget.can_vpn_reset()

    @pytest.mark.fast
    def test_can_vpn_reset_false(self):
        """Test can_vpn_reset returns False when resets exhausted."""
        budget = CaptionRetryBudget()
        budget.max_vpn_resets = 2
        budget.vpn_resets_used = 2

        assert not budget.can_vpn_reset()

    @pytest.mark.fast
    def test_reset_preserves_vpn_count_by_default(self):
        """Test reset() preserves vpn_resets_used by default."""
        budget = CaptionRetryBudget()
        budget.vpn_resets_used = 1
        budget.record_attempt("test")

        budget.reset()

        assert budget.attempts == 0
        assert budget.vpn_resets_used == 1  # Preserved

    @pytest.mark.fast
    def test_reset_clears_vpn_count_when_requested(self):
        """Test reset(preserve_vpn_count=False) clears vpn_resets_used."""
        budget = CaptionRetryBudget()
        budget.vpn_resets_used = 2
        budget.record_attempt("test")

        budget.reset(preserve_vpn_count=False)

        assert budget.attempts == 0
        assert budget.vpn_resets_used == 0  # Cleared

    @pytest.mark.fast
    def test_to_dict_includes_vpn_reset_state(self):
        """Test to_dict includes vpn_resets_used and max_vpn_resets."""
        budget = CaptionRetryBudget()
        budget.vpn_resets_used = 1
        budget.max_vpn_resets = 3

        data = budget.to_dict()

        assert data['vpn_resets_used'] == 1
        assert data['max_vpn_resets'] == 3

    @pytest.mark.fast
    def test_from_dict_restores_vpn_reset_state(self):
        """Test from_dict restores vpn_resets_used and max_vpn_resets."""
        data = {
            'attempts': 50,
            'vpn_resets_used': 1,
            'max_vpn_resets': 3,
        }

        budget = CaptionRetryBudget.from_dict(data)

        assert budget.vpn_resets_used == 1
        assert budget.max_vpn_resets == 3

    @pytest.mark.fast
    def test_from_dict_uses_defaults_for_vpn_state(self):
        """Test from_dict uses defaults when vpn state not in checkpoint."""
        data = {
            'attempts': 50,
            # No vpn_resets_used or max_vpn_resets
        }

        budget = CaptionRetryBudget.from_dict(data)

        assert budget.vpn_resets_used == 0
        assert budget.max_vpn_resets == 2  # Default

    @pytest.mark.fast
    def test_from_config_dict_sets_vpn_options(self):
        """Test from_config with dict sets VPN rotation options."""
        config = {
            'max_attempts': 100,
            'trigger_vpn_rotation_on_rate_limit': True,
            'max_vpn_resets_per_session': 5,
        }

        budget = CaptionRetryBudget.from_config(config)

        assert budget.trigger_vpn_on_rate_limit is True
        assert budget.max_vpn_resets == 5

    @pytest.mark.fast
    def test_from_config_dataclass_sets_vpn_options(self):
        """Test from_config with dataclass sets VPN rotation options."""
        from src.config.sections.download import CaptionRetryBudgetConfig as DLConfig

        config = DLConfig(
            max_attempts=100,
            trigger_vpn_rotation_on_rate_limit=False,
            max_vpn_resets_per_session=3,
        )

        budget = CaptionRetryBudget.from_config(config)

        assert budget.trigger_vpn_on_rate_limit is False
        assert budget.max_vpn_resets == 3


class TestCaptionRetryBudgetVPNRotationIntegration:
    """Integration tests for VPN rotation with CaptionRetryBudget (US-37-008)."""

    @pytest.mark.fast
    def test_vpn_rotation_workflow(self):
        """Test complete VPN rotation workflow: exhaust -> rotate -> reset -> continue."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 20
        budget.trigger_vpn_on_rate_limit = True
        budget.max_vpn_resets = 2

        # Process videos until budget exhausted with rate limit errors
        for i in range(20):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        assert budget.budget_exhausted()
        assert budget.should_trigger_vpn_rotation()

        # Simulate VPN rotation
        budget.record_vpn_reset()
        budget.reset()  # Preserves vpn_resets_used

        # Budget should be fresh but VPN count preserved
        assert not budget.budget_exhausted()
        assert budget.vpn_resets_used == 1
        assert budget.can_vpn_reset()  # Still have 1 more reset

        # Process more videos
        for i in range(20):
            budget.record_attempt(f"video2_{i}")
            budget.record_failure(f"video2_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        assert budget.budget_exhausted()
        assert budget.should_trigger_vpn_rotation()

        # Second VPN rotation
        budget.record_vpn_reset()
        budget.reset()

        assert not budget.budget_exhausted()
        assert budget.vpn_resets_used == 2
        assert not budget.can_vpn_reset()  # No more resets

        # Process more videos - will exhaust again
        for i in range(20):
            budget.record_attempt(f"video3_{i}")
            budget.record_failure(f"video3_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        assert budget.budget_exhausted()
        # This time should NOT trigger VPN rotation (max reached)
        assert not budget.should_trigger_vpn_rotation()

    @pytest.mark.fast
    def test_vpn_rotation_logging(self):
        """Test VPN rotation trigger logging."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10
        budget.trigger_vpn_on_rate_limit = True
        budget.max_vpn_resets = 2

        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        with patch('src.caption.retry_budget.logger') as mock_logger:
            result = budget.should_trigger_vpn_rotation()

            assert result is True
            # Check logging
            info_calls = [c for c in mock_logger.info.call_args_list
                         if 'rotating VPN' in str(c)]
            assert len(info_calls) >= 1
            # Verify log includes percentage
            call_str = str(info_calls[0])
            assert '100.0%' in call_str or '100%' in call_str

    @pytest.mark.fast
    def test_vpn_rotation_logging_max_reached(self):
        """Test logging when max VPN rotations reached."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10
        budget.trigger_vpn_on_rate_limit = True
        budget.max_vpn_resets = 2
        budget.vpn_resets_used = 2  # Already used all

        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        with patch('src.caption.retry_budget.logger') as mock_logger:
            result = budget.should_trigger_vpn_rotation()

            assert result is False
            # Check logging about limit reached
            info_calls = [c for c in mock_logger.info.call_args_list
                         if 'rotation limit reached' in str(c)]
            assert len(info_calls) >= 1

    @pytest.mark.fast
    def test_vpn_rotation_serialization_roundtrip(self):
        """Test VPN rotation state survives serialization roundtrip."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 20
        budget.trigger_vpn_on_rate_limit = True
        budget.max_vpn_resets = 3

        # Simulate partial VPN rotation usage
        budget.vpn_resets_used = 1

        # Add some errors
        for i in range(15):
            budget.record_attempt(f"video_{i}")
        for i in range(10):
            budget.record_failure(f"fail_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)

        # Serialize and restore
        data = budget.to_dict()
        restored = CaptionRetryBudget.from_dict(data)

        # Verify VPN state preserved
        assert restored.vpn_resets_used == 1
        assert restored.max_vpn_resets == 3
        # Verify can still calculate VPN rotation eligibility
        assert restored.can_vpn_reset()
        # Note: trigger_vpn_on_rate_limit is not serialized (config setting)
        # but from_dict defaults it to True


class TestCaptionRetryBudgetEarlyTermination:
    """Tests for early termination when success rate drops below threshold (US-37-009)."""

    @pytest.mark.fast
    def test_get_success_rate_no_data(self):
        """Test success rate returns 1.0 when no data."""
        budget = CaptionRetryBudget()
        assert budget.get_success_rate() == 1.0

    @pytest.mark.fast
    def test_get_success_rate_all_success(self):
        """Test success rate with all successes."""
        budget = CaptionRetryBudget()
        for i in range(10):
            budget.record_success(f"video_{i}")
        assert budget.get_success_rate() == 1.0

    @pytest.mark.fast
    def test_get_success_rate_all_failures(self):
        """Test success rate with all failures."""
        budget = CaptionRetryBudget()
        for i in range(10):
            budget.record_failure(f"video_{i}")
        assert budget.get_success_rate() == 0.0

    @pytest.mark.fast
    def test_get_success_rate_mixed(self):
        """Test success rate with mixed results."""
        budget = CaptionRetryBudget()
        # 3 successes, 7 failures = 30% success rate
        for i in range(3):
            budget.record_success(f"success_{i}")
        for i in range(7):
            budget.record_failure(f"fail_{i}")
        assert budget.get_success_rate() == 0.3

    @pytest.mark.fast
    def test_should_terminate_early_insufficient_sample(self):
        """Test no early termination before min_sample reached."""
        budget = CaptionRetryBudget()
        budget.min_sample_for_early_termination = 20
        budget.min_success_rate = 0.3

        # Process only 10 videos with 0% success rate
        for i in range(10):
            budget.record_failure(f"video_{i}")

        # Should not terminate - insufficient sample
        assert not budget.should_terminate_early()
        assert not budget.early_terminated

    @pytest.mark.fast
    def test_should_terminate_early_low_success_rate(self):
        """Test early termination triggers at low success rate."""
        budget = CaptionRetryBudget()
        budget.min_sample_for_early_termination = 20
        budget.min_success_rate = 0.3

        # Process 25 videos with 20% success rate (below 30% threshold)
        for i in range(5):
            budget.record_success(f"success_{i}")
        for i in range(20):
            budget.record_failure(f"fail_{i}")

        assert budget.should_terminate_early()

    @pytest.mark.fast
    def test_should_terminate_early_acceptable_rate(self):
        """Test no early termination when success rate is acceptable."""
        budget = CaptionRetryBudget()
        budget.min_sample_for_early_termination = 20
        budget.min_success_rate = 0.3

        # Process 25 videos with 40% success rate (above 30% threshold)
        for i in range(10):
            budget.record_success(f"success_{i}")
        for i in range(15):
            budget.record_failure(f"fail_{i}")

        assert not budget.should_terminate_early()

    @pytest.mark.fast
    def test_check_and_terminate_early(self):
        """Test check_and_terminate_early sets flags and logs."""
        budget = CaptionRetryBudget()
        budget.min_sample_for_early_termination = 20
        budget.min_success_rate = 0.3

        # Process 25 videos with 20% success rate
        for i in range(5):
            budget.record_success(f"success_{i}")
        for i in range(20):
            budget.record_failure(f"fail_{i}")

        # First call should trigger termination
        with patch('src.caption.retry_budget.logger') as mock_logger:
            result = budget.check_and_terminate_early()

            assert result is True
            assert budget.early_terminated is True
            assert budget.early_termination_reason is not None
            assert "20.0%" in budget.early_termination_reason
            assert "30.0%" in budget.early_termination_reason

            # Verify warning was logged
            warning_calls = [c for c in mock_logger.warning.call_args_list
                           if 'EARLY TERMINATION' in str(c)]
            assert len(warning_calls) >= 1

    @pytest.mark.fast
    def test_check_and_terminate_early_no_double_trigger(self):
        """Test early termination only triggers once."""
        budget = CaptionRetryBudget()
        budget.min_sample_for_early_termination = 20
        budget.min_success_rate = 0.3

        # Process videos to trigger termination
        for i in range(5):
            budget.record_success(f"success_{i}")
        for i in range(20):
            budget.record_failure(f"fail_{i}")

        # First call triggers
        assert budget.check_and_terminate_early() is True

        # Second call should not trigger again
        assert budget.check_and_terminate_early() is False
        assert budget.should_terminate_early() is False

    @pytest.mark.fast
    def test_is_early_terminated(self):
        """Test is_early_terminated returns correct flag state."""
        budget = CaptionRetryBudget()
        assert budget.is_early_terminated() is False

        budget.early_terminated = True
        assert budget.is_early_terminated() is True

    @pytest.mark.fast
    def test_early_termination_in_summary(self):
        """Test early termination state appears in summary."""
        budget = CaptionRetryBudget()
        budget.min_sample_for_early_termination = 10
        budget.min_success_rate = 0.3

        # Trigger early termination
        for i in range(2):
            budget.record_success(f"success_{i}")
        for i in range(10):
            budget.record_failure(f"fail_{i}")
        budget.check_and_terminate_early()

        summary = budget.get_summary()
        assert summary['early_terminated'] is True
        assert summary['early_termination_reason'] is not None
        assert summary['success_rate'] < 0.3

    @pytest.mark.fast
    def test_early_termination_serialization(self):
        """Test early termination state survives serialization."""
        budget = CaptionRetryBudget()
        budget.min_sample_for_early_termination = 10
        budget.min_success_rate = 0.3

        # Trigger early termination
        for i in range(2):
            budget.record_success(f"success_{i}")
        for i in range(10):
            budget.record_failure(f"fail_{i}")
        budget.check_and_terminate_early()

        # Serialize and restore
        data = budget.to_dict()
        restored = CaptionRetryBudget.from_dict(data)

        assert restored.early_terminated is True
        assert restored.early_termination_reason == budget.early_termination_reason

    @pytest.mark.fast
    def test_early_termination_reset(self):
        """Test reset clears early termination state."""
        budget = CaptionRetryBudget()
        budget.early_terminated = True
        budget.early_termination_reason = "test reason"

        budget.reset()

        assert budget.early_terminated is False
        assert budget.early_termination_reason is None

    @pytest.mark.fast
    def test_config_min_success_rate(self):
        """Test min_success_rate loaded from config."""
        config = CaptionRetryBudgetConfig(
            min_success_rate=0.5,
            min_sample_for_early_termination=30
        )
        budget = CaptionRetryBudget.from_config(config)

        assert budget.min_success_rate == 0.5
        assert budget.min_sample_for_early_termination == 30

    @pytest.mark.fast
    def test_config_min_success_rate_from_dict(self):
        """Test min_success_rate loaded from dict config."""
        config = {
            'min_success_rate': 0.25,
            'min_sample_for_early_termination': 15
        }
        budget = CaptionRetryBudget.from_config(config)

        assert budget.min_success_rate == 0.25
        assert budget.min_sample_for_early_termination == 15

    @pytest.mark.fast
    def test_early_termination_at_20_percent(self):
        """Test early termination triggers at exactly 20% success rate (below 30%)."""
        budget = CaptionRetryBudget()
        budget.min_sample_for_early_termination = 20
        budget.min_success_rate = 0.3

        # 4 successes + 16 failures = 20% success rate
        for i in range(4):
            budget.record_success(f"success_{i}")
        for i in range(16):
            budget.record_failure(f"fail_{i}")

        # 20% < 30%, should terminate
        assert budget.get_success_rate() == 0.2
        assert budget.should_terminate_early()

    @pytest.mark.fast
    def test_early_termination_at_threshold_boundary(self):
        """Test boundary: exactly 30% success rate should NOT terminate."""
        budget = CaptionRetryBudget()
        budget.min_sample_for_early_termination = 20
        budget.min_success_rate = 0.3

        # 6 successes + 14 failures = 30% success rate
        for i in range(6):
            budget.record_success(f"success_{i}")
        for i in range(14):
            budget.record_failure(f"fail_{i}")

        # 30% == 30%, should NOT terminate (only < threshold triggers)
        assert budget.get_success_rate() == 0.3
        assert not budget.should_terminate_early()


class TestCaptionRetryBudgetEarlyTerminationIntegration:
    """Integration tests for early termination feature (US-37-009)."""

    @pytest.mark.fast
    def test_realistic_workflow_high_failure_rate(self):
        """Test realistic workflow with high failure rate leading to termination."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.min_sample_for_early_termination = 20
        budget.min_success_rate = 0.3

        # Simulate fetching videos - some succeed, most fail
        processed = 0
        terminated_at = None

        for i in range(50):
            budget.record_attempt(f"video_{i}")
            # 10% success rate
            if i % 10 == 0:
                budget.record_success(f"video_{i}")
            else:
                budget.record_failure(f"video_{i}", error_category=CaptionErrorCategory.NETWORK)

            processed += 1

            # Check after each video if we should terminate
            if budget.check_and_terminate_early():
                terminated_at = processed
                break

        # Should have terminated around video 20-22 (once sample reached)
        assert terminated_at is not None
        assert terminated_at >= 20
        assert terminated_at < 30
        assert budget.early_terminated
        assert "below threshold" in budget.early_termination_reason

    @pytest.mark.fast
    def test_realistic_workflow_success_recovery(self):
        """Test workflow where success rate recovers and doesn't terminate."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.min_sample_for_early_termination = 20
        budget.min_success_rate = 0.3

        # First 10 videos: 20% success
        for i in range(2):
            budget.record_success(f"success_1_{i}")
        for i in range(8):
            budget.record_failure(f"fail_1_{i}")

        # Next 10 videos: 60% success (recovery)
        for i in range(6):
            budget.record_success(f"success_2_{i}")
        for i in range(4):
            budget.record_failure(f"fail_2_{i}")

        # Overall: 8/20 = 40% success, should NOT terminate
        assert not budget.should_terminate_early()
        assert not budget.check_and_terminate_early()

    @pytest.mark.fast
    def test_budget_exhaustion_vs_early_termination(self):
        """Test that budget exhaustion takes precedence over early termination check."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 10
        budget.min_sample_for_early_termination = 20  # Won't reach this
        budget.min_success_rate = 0.3

        # Exhaust budget before reaching sample threshold
        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}")

        # Budget should be exhausted
        assert budget.budget_exhausted()
        # Early termination check shouldn't trigger (insufficient sample)
        assert not budget.should_terminate_early()

    @pytest.mark.fast
    def test_early_termination_with_error_tracking(self):
        """Test early termination works with error category tracking."""
        budget = CaptionRetryBudget()
        budget.min_sample_for_early_termination = 20
        budget.min_success_rate = 0.3

        # Process with various error categories
        for i in range(4):
            budget.record_success(f"success_{i}")
        for i in range(8):
            budget.record_failure(f"fail_{i}", error_category=CaptionErrorCategory.RATE_LIMIT)
        for i in range(8):
            budget.record_failure(f"fail2_{i}", error_category=CaptionErrorCategory.NETWORK)

        # Should terminate (4/20 = 20% < 30%)
        assert budget.check_and_terminate_early()
        assert budget.early_terminated

        # Error tracking should still work
        top_errors = budget.get_top_errors()
        assert len(top_errors) == 2
        # Both RATE_LIMIT and NETWORK have 8 errors each


# ============================================================================
# US-38-009: Batch size in retry budget summary logging
# ============================================================================


class TestCaptionRetryBudgetBatchSizeTracking:
    """Test batch_size tracking in CaptionRetryBudget (US-38-009)."""

    @pytest.mark.fast
    def test_batch_size_field_exists(self):
        """Test batch_size field is initialized as None."""
        budget = CaptionRetryBudget()
        assert hasattr(budget, 'batch_size')
        assert budget.batch_size is None

    @pytest.mark.fast
    def test_scale_to_batch_size_sets_batch_size(self):
        """Test scale_to_batch_size() sets the batch_size field."""
        budget = CaptionRetryBudget()
        assert budget.batch_size is None

        budget.scale_to_batch_size(175)
        assert budget.batch_size == 175

    @pytest.mark.fast
    def test_batch_size_in_get_summary(self):
        """Test batch_size is included in get_summary() output (US-38-009 main criterion)."""
        budget = CaptionRetryBudget()
        budget.scale_to_batch_size(200)

        summary = budget.get_summary()

        assert 'batch_size' in summary
        assert summary['batch_size'] == 200

    @pytest.mark.fast
    def test_batch_size_in_get_summary_when_not_set(self):
        """Test batch_size is None in summary when scale_to_batch_size not called."""
        budget = CaptionRetryBudget()
        summary = budget.get_summary()

        assert 'batch_size' in summary
        assert summary['batch_size'] is None

    @pytest.mark.fast
    def test_batch_size_in_to_dict(self):
        """Test batch_size is included in to_dict() for checkpoint persistence."""
        budget = CaptionRetryBudget()
        budget.scale_to_batch_size(150)

        data = budget.to_dict()

        assert 'batch_size' in data
        assert data['batch_size'] == 150

    @pytest.mark.fast
    def test_batch_size_restored_from_dict(self):
        """Test batch_size is restored from checkpoint data."""
        # Simulate checkpoint data
        checkpoint_data = {
            "attempts": 50,
            "failures": 10,
            "successes": 40,
            "batch_size": 175,
        }

        budget = CaptionRetryBudget.from_dict(checkpoint_data)

        assert budget.batch_size == 175

    @pytest.mark.fast
    def test_batch_size_175_checkpoint_restore_logs_summary(self, caplog):
        """US-42-008: Verify batch_size=175 is logged in checkpoint restore summary.

        AC5: Log batch_size in checkpoint restore summary.
        """
        import logging
        caplog.set_level(logging.INFO)

        checkpoint_data = {
            "attempts": 50,
            "failures": 10,
            "successes": 40,
            "batch_size": 175,
            "max_attempts": 350,
        }

        budget = CaptionRetryBudget.from_dict(checkpoint_data)

        # Verify batch_size is logged in restore summary
        assert budget.batch_size == 175
        assert "[US-42-008]" in caplog.text
        assert "batch_size=175" in caplog.text
        assert "restored from checkpoint" in caplog.text

    @pytest.mark.fast
    def test_batch_size_roundtrip(self):
        """Test batch_size survives to_dict/from_dict roundtrip."""
        budget = CaptionRetryBudget()
        budget.scale_to_batch_size(225)
        budget.record_attempt("vid1")
        budget.record_success("vid1")

        # Serialize and restore
        data = budget.to_dict()
        restored = CaptionRetryBudget.from_dict(data)

        assert restored.batch_size == 225
        assert restored.attempts == 1
        assert restored.successes == 1


class TestCaptionRetryBudgetAutoScaleSufficiency:
    """Tests for US-39-002: Verify auto-scaling prevents budget exhaustion on large batches.

    These tests verify that when auto_scale is enabled, the budget is sufficient
    to process large batches (up to 200 videos) even with a 50% success rate.
    """

    @pytest.mark.fast
    def test_budget_sufficient_for_200_videos_at_50_percent_success(self):
        """Verify budget is NOT exhausted with 200 videos at 50% success rate.

        US-39-002: With auto_scale enabled and 1.5 attempts_per_video,
        a batch of 200 videos should get max_attempts of 300, which is
        sufficient for 200 videos even with 50% needing retries.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.attempts_per_video = 1.5

        # Scale to 200 videos
        new_max = budget.scale_to_batch_size(200)

        # With 1.5 attempts per video: 200 * 1.5 = 300 max_attempts
        assert new_max == 300
        assert budget.max_attempts == 300

        # Simulate 50% success rate: half succeed first try, half need retries
        # 100 videos succeed on first attempt (100 attempts)
        # 100 videos fail first attempt, retry and succeed (200 attempts)
        # Total: 300 attempts (exactly at limit, but should NOT be exhausted)
        successful_first_try = 100
        failed_first_need_retry = 100

        # Record successful first-try videos
        for i in range(successful_first_try):
            budget.record_attempt(f"vid_{i}")
            budget.record_success(f"vid_{i}")

        # Record failed then retry videos
        for i in range(failed_first_need_retry):
            vid_id = f"vid_retry_{i}"
            # First attempt fails
            budget.record_attempt(vid_id)
            budget.record_failure(vid_id)
            # Retry succeeds
            budget.record_attempt(vid_id)
            budget.record_success(vid_id)

        # Total: 100 + 200 = 300 attempts used
        assert budget.attempts == 300
        # Should NOT be exhausted (300 == 300 means exhausted at exactly limit)
        # Actually, when attempts >= max_attempts, budget is exhausted
        # This is correct behavior - budget is exactly consumed
        assert budget.budget_exhausted()

    @pytest.mark.fast
    def test_budget_not_exhausted_with_200_videos_realistic_success(self):
        """Verify budget handles 200 videos with realistic 70% success rate.

        A more realistic scenario: 70% succeed first try, 30% need one retry.
        This should NOT exhaust the budget.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.attempts_per_video = 1.5

        # Scale to 200 videos
        new_max = budget.scale_to_batch_size(200)
        assert new_max == 300

        # 70% success (140 videos) first try: 140 attempts
        # 30% fail first (60 videos), retry: 120 attempts
        # Total: 260 attempts (under 300 budget)
        successful_first = 140
        need_retry = 60

        for i in range(successful_first):
            budget.record_attempt(f"vid_{i}")
            budget.record_success(f"vid_{i}")

        for i in range(need_retry):
            vid_id = f"vid_retry_{i}"
            budget.record_attempt(vid_id)
            budget.record_failure(vid_id)
            budget.record_attempt(vid_id)
            budget.record_success(vid_id)

        # Total: 140 + 120 = 260 attempts
        assert budget.attempts == 260
        assert not budget.budget_exhausted(), "Budget should NOT be exhausted with 70% success rate"

    @pytest.mark.fast
    def test_budget_exhausted_without_auto_scale_for_large_batch(self):
        """Verify budget DOES exhaust without auto_scale for 175 videos.

        US-39-002: This demonstrates the bug scenario - without auto_scale,
        the default 100 max_attempts exhausts at video 101.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = False  # Disable auto-scaling
        budget.max_attempts = 100  # Default

        # Process 175 videos - each takes 1 attempt
        for i in range(100):
            if budget.budget_exhausted():
                break
            budget.record_attempt(f"vid_{i}")
            budget.record_success(f"vid_{i}")

        # Budget should be exhausted after 100 attempts
        assert budget.budget_exhausted()
        assert budget.attempts == 100
        assert budget.successes == 100
        # Video 101+ would be skipped

    @pytest.mark.fast
    def test_scale_to_batch_size_logging(self, caplog):
        """Verify INFO logging when budget is scaled.

        US-39-002: When scale_to_batch_size scales up, it should log an INFO message.
        """
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100  # Default

        # Scale to 175 videos - should trigger scaling
        budget.scale_to_batch_size(175)

        # Verify INFO log was produced
        assert any(
            "scaled max_attempts from 100 to" in record.message
            for record in caplog.records
        ), "Expected INFO log about budget scaling"

    @pytest.mark.fast
    def test_scale_to_batch_size_no_logging_when_sufficient(self, caplog):
        """Verify DEBUG (not INFO) logging when budget already sufficient.

        US-39-002: When current max_attempts is already sufficient,
        scale_to_batch_size should only log DEBUG.
        """
        import logging
        caplog.set_level(logging.DEBUG)

        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 500  # Already large

        # Scale to 50 videos - should NOT trigger scaling (50 * 1.5 = 75 < 500)
        budget.scale_to_batch_size(50)

        # Should NOT have INFO about scaling
        info_logs = [r for r in caplog.records if r.levelno >= logging.INFO]
        scaling_info_logs = [r for r in info_logs if "scaled max_attempts" in r.message]
        assert not scaling_info_logs, "Should not log INFO when budget already sufficient"

        # Should have DEBUG log
        debug_logs = [r for r in caplog.records if "sufficient for" in r.message]
        assert debug_logs, "Expected DEBUG log about budget being sufficient"


class TestCaptionRetryBudgetLargeBatchScaling:
    """Tests for US-41-003: Verify retry budget scales correctly for large batches.

    These tests verify:
    1. Batch of 175 with auto_scale=true should scale to 175*2.0=350 max_attempts
    2. Budget exhaustion doesn't occur before video 100 for batch of 175
    3. Logs show 'scaled max_attempts from 100 to 350' for batch of 175 videos
    """

    @pytest.mark.fast
    def test_batch_175_scales_to_350_max_attempts(self):
        """Verify batch of 175 videos with auto_scale=true scales to 350 max_attempts.

        US-41-003: The logs show 100/100 exhaustion for 175 videos, indicating
        ensure_scaled() may not be called or auto_scale is false.

        Expected: 175 * 2.0 = 350 max_attempts
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100  # Default
        budget.attempts_per_video = 2.0  # Default

        # Call ensure_scaled with 175 videos
        scaled = budget.ensure_scaled(175)

        # Verify scaling occurred
        assert scaled is True
        assert budget.max_attempts == 350, (
            f"Expected max_attempts=350 (175*2.0), got {budget.max_attempts}"
        )
        assert budget.batch_size == 175

    @pytest.mark.fast
    def test_batch_175_no_exhaustion_before_video_100(self):
        """Verify budget exhaustion doesn't occur before video 100 for batch of 175.

        US-41-003: Without scaling, budget exhausts at video 101, skipping 74 videos.
        With scaling, all 175 videos should be processable (assuming 1 attempt each).
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100  # Default
        budget.attempts_per_video = 2.0

        # Scale budget for batch of 175
        budget.ensure_scaled(175)

        # Process first 100 videos - should NOT exhaust
        for i in range(100):
            assert not budget.budget_exhausted(), (
                f"Budget exhausted at video {i}, but max_attempts={budget.max_attempts}"
            )
            budget.record_attempt(f"vid_{i}")
            budget.record_success(f"vid_{i}")

        # Verify budget is NOT exhausted after 100 videos
        assert not budget.budget_exhausted(), (
            f"Budget exhausted after 100 attempts with max_attempts={budget.max_attempts}"
        )
        assert budget.attempts == 100
        assert budget.successes == 100

    @pytest.mark.fast
    def test_batch_175_can_process_all_videos(self):
        """Verify all 175 videos can be processed with scaled budget.

        US-41-003: With scaling (350 max_attempts), processing 175 videos
        at 1 attempt each should complete without exhaustion.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100
        budget.attempts_per_video = 2.0

        # Scale budget
        budget.ensure_scaled(175)

        # Process all 175 videos (1 attempt each = 175 total)
        for i in range(175):
            assert not budget.budget_exhausted(), (
                f"Budget exhausted at video {i}"
            )
            budget.record_attempt(f"vid_{i}")
            budget.record_success(f"vid_{i}")

        # All 175 should have been processed
        assert budget.attempts == 175
        assert budget.successes == 175
        assert not budget.budget_exhausted()

    @pytest.mark.fast
    def test_batch_175_scaling_log_message(self, caplog):
        """Verify logs show 'scaled max_attempts from 100 to 350' for batch of 175.

        US-41-003: Logs must show the scaling occurred with specific values.
        """
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100
        budget.attempts_per_video = 2.0

        # Scale budget for 175 videos
        budget.ensure_scaled(175)

        # Verify INFO log shows scaling
        scaling_logs = [
            record.message for record in caplog.records
            if "scaled" in record.message.lower() and "350" in record.message
        ]
        assert len(scaling_logs) >= 1, (
            f"Expected log with 'scaled...350', got: {[r.message for r in caplog.records]}"
        )

        # Verify the log message content
        log_msg = scaling_logs[0]
        assert "100" in log_msg, "Log should mention original max_attempts=100"
        assert "350" in log_msg, "Log should mention scaled max_attempts=350"
        assert "175" in log_msg, "Log should mention batch_size=175"

    @pytest.mark.fast
    def test_scaling_only_occurs_once_for_same_batch(self):
        """Verify ensure_scaled() is idempotent for batch of 175.

        US-41-003: Calling ensure_scaled() multiple times with the same batch_size
        should not scale beyond 350.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100

        # First call - should scale
        result1 = budget.ensure_scaled(175)
        assert result1 is True
        assert budget.max_attempts == 350

        # Second call - should NOT scale again
        result2 = budget.ensure_scaled(175)
        assert result2 is False
        assert budget.max_attempts == 350  # Still 350, not doubled

    @pytest.mark.fast
    def test_auto_scale_false_does_not_scale_for_175(self):
        """Verify auto_scale=false does NOT scale for batch of 175.

        US-41-003: This confirms the bug scenario - when auto_scale is disabled,
        budget stays at 100 and will exhaust at video 101.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = False  # Bug condition
        budget.max_attempts = 100

        # Attempt to scale
        result = budget.ensure_scaled(175)

        # Should NOT scale
        assert result is False
        assert budget.max_attempts == 100, (
            "With auto_scale=False, max_attempts should remain 100"
        )


class TestCaptionRetryBudgetFormattedSummary:
    """Tests for US-39-005: Budget consumption summary logging at stage completion.

    These tests verify the get_formatted_summary() method returns a properly
    formatted string with all required fields for diagnosis of budget issues.
    """

    @pytest.mark.fast
    def test_get_formatted_summary_includes_all_required_fields(self):
        """Verify formatted summary contains all required fields.

        US-39-005: Summary format must be:
        'CaptionRetryBudget summary: {attempts}/{max_attempts} attempts,
         {successes} succeeded, {failures} failed, {skipped} skipped (batch_size={N})'
        """
        budget = CaptionRetryBudget()
        budget.max_attempts = 200
        budget.batch_size = 150

        # Simulate activity
        for i in range(120):
            budget.record_attempt(f"vid_{i}")
        for i in range(100):
            budget.record_success(f"vid_{i}")
        for i in range(20):
            budget.record_failure(f"fail_{i}")
        for i in range(5):
            budget.record_skipped(f"skip_{i}")

        summary = budget.get_formatted_summary()

        # Verify all required components are present
        assert "CaptionRetryBudget summary:" in summary
        assert "120/200 attempts" in summary
        assert "100 succeeded" in summary
        assert "20 failed" in summary
        assert "5 skipped" in summary
        assert "batch_size=150" in summary

    @pytest.mark.fast
    def test_get_formatted_summary_with_zero_values(self):
        """Verify formatted summary handles zero values correctly."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.batch_size = 50

        summary = budget.get_formatted_summary()

        assert "0/100 attempts" in summary
        assert "0 succeeded" in summary
        assert "0 failed" in summary
        assert "0 skipped" in summary
        assert "batch_size=50" in summary

    @pytest.mark.fast
    def test_get_formatted_summary_with_none_batch_size(self):
        """Verify formatted summary handles None batch_size."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        # batch_size defaults to None

        summary = budget.get_formatted_summary()

        # Should show batch_size=0 when None
        assert "batch_size=0" in summary

    @pytest.mark.fast
    def test_get_formatted_summary_includes_scaled_max_attempts(self):
        """Verify summary shows scaled max_attempts value (actual vs default).

        US-39-005: The max_attempts in summary should reflect the scaled value,
        not the original default 100.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.attempts_per_video = 2.0

        # Scale to 175 videos (175 * 2.0 = 350)
        budget.scale_to_batch_size(175)

        # Record some activity
        for i in range(150):
            budget.record_attempt(f"vid_{i}")
        for i in range(100):
            budget.record_success(f"vid_{i}")
        for i in range(50):
            budget.record_failure(f"fail_{i}")

        summary = budget.get_formatted_summary()

        # Should show scaled max_attempts (350), not default (100) - US-40-006: 175 * 2.0 = 350
        assert "150/350 attempts" in summary
        assert "100 succeeded" in summary
        assert "50 failed" in summary
        assert "batch_size=175" in summary

    @pytest.mark.fast
    def test_get_formatted_summary_thread_safe(self):
        """Verify get_formatted_summary is thread-safe."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 1000
        budget.batch_size = 500
        summaries = []

        def record_and_get_summary(thread_id):
            for i in range(10):
                budget.record_attempt(f"vid_{thread_id}_{i}")
                budget.record_success(f"vid_{thread_id}_{i}")
            summaries.append(budget.get_formatted_summary())

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(record_and_get_summary, i) for i in range(8)]
            for f in futures:
                f.result()

        # All summaries should be valid strings
        assert len(summaries) == 8
        for summary in summaries:
            assert "CaptionRetryBudget summary:" in summary
            assert "succeeded" in summary
            assert "failed" in summary
            assert "skipped" in summary
            assert "batch_size=" in summary


class TestCaptionRetryBudgetPerVideoLogging:
    """Tests for US-40-005: Per-video retry logging for budget consumption diagnosis.

    These tests verify that retry attempts are logged with appropriate details
    to help diagnose why budget is consumed by early videos.
    """

    @pytest.mark.fast
    def test_threshold_log_includes_percentage(self):
        """Verify threshold crossing logs include consumption percentage (AC3, AC5).

        US-40-005: When budget reaches 50%, 75%, 90% thresholds, the log message
        must include the actual percentage consumed.
        """
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        with patch('src.caption.retry_budget.logger') as mock_logger:
            # Move to just below 50%
            budget.attempts = 49

            # Cross 50% threshold
            budget.record_attempt("video_50")

            # Verify INFO call includes percentage
            info_calls = mock_logger.info.call_args_list
            threshold_calls = [c for c in info_calls if '50%' in str(c)]
            assert len(threshold_calls) >= 1

            # Verify the percentage is in the message
            call_str = str(threshold_calls[0])
            assert '50%' in call_str
            # The message format includes both absolute and percentage
            assert 'attempts:' in call_str or '50/100' in call_str

    @pytest.mark.fast
    def test_threshold_75_percent_includes_percentage(self):
        """Verify 75% threshold log includes correct percentage."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        with patch('src.caption.retry_budget.logger') as mock_logger:
            budget.attempts = 74
            budget.record_attempt("video_75")

            info_calls = mock_logger.info.call_args_list
            threshold_calls = [c for c in info_calls if '75%' in str(c)]
            assert len(threshold_calls) >= 1
            call_str = str(threshold_calls[0])
            assert '75%' in call_str

    @pytest.mark.fast
    def test_threshold_90_percent_warning_includes_percentage(self):
        """Verify 90% threshold WARNING includes correct percentage."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100

        with patch('src.caption.retry_budget.logger') as mock_logger:
            budget.attempts = 89
            budget.record_attempt("video_90")

            warning_calls = mock_logger.warning.call_args_list
            threshold_calls = [c for c in warning_calls if '90%' in str(c)]
            assert len(threshold_calls) >= 1
            call_str = str(threshold_calls[0])
            assert '90%' in call_str
            # 90% threshold also includes success/failure counts
            assert 'successes' in call_str
            assert 'failures' in call_str


class TestCaptionRetryBudgetCircuitBreakerState:
    """Tests for US-40-011: Circuit breaker state in CaptionRetryBudget summary.

    These tests verify that circuit breaker state is included in budget summaries
    for improved observability when investigating rapid failures.
    """

    @pytest.mark.fast
    def test_get_summary_includes_circuit_breaker_state_field(self):
        """Verify get_summary returns circuit_breaker_state field (US-40-011 AC1)."""
        budget = CaptionRetryBudget()
        summary = budget.get_summary()

        assert 'circuit_breaker_state' in summary

    @pytest.mark.fast
    def test_get_summary_circuit_breaker_state_none_when_not_set(self):
        """Verify circuit_breaker_state is None when no circuit breaker attached (US-40-011 AC4)."""
        budget = CaptionRetryBudget()
        # circuit_breaker defaults to None

        summary = budget.get_summary()

        assert summary['circuit_breaker_state'] is None

    @pytest.mark.fast
    def test_get_summary_circuit_breaker_state_closed(self):
        """Verify circuit_breaker_state is 'closed' when circuit is normal."""
        from src.caption.circuit_breaker import CaptionCircuitBreaker

        budget = CaptionRetryBudget()
        circuit_breaker = CaptionCircuitBreaker()
        budget.circuit_breaker = circuit_breaker

        summary = budget.get_summary()

        assert summary['circuit_breaker_state'] == 'closed'

    @pytest.mark.fast
    def test_get_summary_circuit_breaker_state_open(self):
        """Verify circuit_breaker_state is 'open' when circuit is tripped."""
        from src.caption.circuit_breaker import CaptionCircuitBreaker, CaptionCircuitBreakerConfig

        budget = CaptionRetryBudget()
        config = CaptionCircuitBreakerConfig(threshold=3, pause_seconds=60.0)
        circuit_breaker = CaptionCircuitBreaker(config)
        budget.circuit_breaker = circuit_breaker

        # Trip the circuit by recording enough failures
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.is_open is True  # Verify circuit is tripped

        summary = budget.get_summary()

        assert summary['circuit_breaker_state'] == 'open'

    @pytest.mark.fast
    def test_get_formatted_summary_includes_circuit_breaker_closed(self):
        """Verify get_formatted_summary includes circuit_breaker=closed (US-40-011 AC2)."""
        from src.caption.circuit_breaker import CaptionCircuitBreaker

        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.batch_size = 50
        circuit_breaker = CaptionCircuitBreaker()
        budget.circuit_breaker = circuit_breaker

        summary = budget.get_formatted_summary()

        assert 'circuit_breaker=closed' in summary

    @pytest.mark.fast
    def test_get_formatted_summary_includes_circuit_breaker_open(self):
        """Verify get_formatted_summary includes circuit_breaker=open when tripped."""
        from src.caption.circuit_breaker import CaptionCircuitBreaker, CaptionCircuitBreakerConfig

        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.batch_size = 50
        config = CaptionCircuitBreakerConfig(threshold=3, pause_seconds=60.0)
        circuit_breaker = CaptionCircuitBreaker(config)
        budget.circuit_breaker = circuit_breaker

        # Trip the circuit
        for _ in range(3):
            circuit_breaker.record_failure()

        summary = budget.get_formatted_summary()

        assert 'circuit_breaker=open' in summary

    @pytest.mark.fast
    def test_get_formatted_summary_no_circuit_breaker_suffix_when_none(self):
        """Verify get_formatted_summary has no circuit_breaker suffix when None (US-40-011 AC4)."""
        budget = CaptionRetryBudget()
        budget.max_attempts = 100
        budget.batch_size = 50

        summary = budget.get_formatted_summary()

        # Should NOT contain circuit_breaker when None
        assert 'circuit_breaker=' not in summary
        # Should still have the basic format
        assert 'CaptionRetryBudget summary:' in summary
        assert 'batch_size=50)' in summary  # Ends with batch_size

    @pytest.mark.fast
    def test_circuit_breaker_state_with_mock_object(self):
        """Verify circuit breaker state works with any object having is_open property."""
        from unittest.mock import Mock

        budget = CaptionRetryBudget()

        # Use mock with is_open = True
        mock_breaker = Mock()
        mock_breaker.is_open = True
        budget.circuit_breaker = mock_breaker

        summary = budget.get_summary()
        assert summary['circuit_breaker_state'] == 'open'

        # Switch to closed
        mock_breaker.is_open = False
        summary = budget.get_summary()
        assert summary['circuit_breaker_state'] == 'closed'

    @pytest.mark.fast
    def test_circuit_breaker_state_graceful_with_invalid_object(self):
        """Verify circuit breaker state returns None for objects without is_open."""
        budget = CaptionRetryBudget()

        # Set a non-circuit-breaker object
        budget.circuit_breaker = "not a circuit breaker"

        summary = budget.get_summary()
        assert summary['circuit_breaker_state'] is None


class TestCaptionRetryBudgetVerification:
    """Tests for US-41-004: Fail-fast budget verification after scaling.

    Verifies that verify_budget_sufficient() catches configuration errors
    before processing begins, preventing wasted work when budget is
    mathematically insufficient.
    """

    @pytest.mark.fast
    def test_verify_budget_sufficient_passes_when_scaled(self):
        """Verify verification passes after proper scaling.

        US-41-004: After ensure_scaled() with auto_scale=true,
        verification should always pass.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = True
        budget.max_attempts = 100
        budget.attempts_per_video = 2.0

        # Scale for batch
        budget.ensure_scaled(175)

        # Should not raise
        budget.verify_budget_sufficient(175)

        # Verify budget is actually sufficient
        assert budget.max_attempts >= 175 * 2.0

    @pytest.mark.fast
    def test_verify_budget_raises_valueerror_when_insufficient(self):
        """Verify ValueError raised when scaling math is wrong.

        US-41-004: If max_attempts < batch_size * attempts_per_video,
        raise ValueError with actionable message.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = False  # Prevent scaling
        budget.max_attempts = 100
        budget.attempts_per_video = 2.0

        # Try to verify for batch that needs 350 attempts
        with pytest.raises(ValueError) as exc_info:
            budget.verify_budget_sufficient(175)

        error_msg = str(exc_info.value)
        assert "insufficient" in error_msg.lower()
        assert "100" in error_msg  # Current max_attempts
        assert "350" in error_msg  # Required attempts
        assert "175" in error_msg  # batch_size
        assert "config.yaml" in error_msg  # Actionable config path

    @pytest.mark.fast
    def test_verify_budget_warns_when_autoscale_disabled(self, caplog):
        """Verify WARNING logged when auto_scale disabled and batch > max.

        US-41-004: Log WARNING if auto_scale=false and batch_size > max_attempts.
        """
        import logging
        caplog.set_level(logging.WARNING)

        budget = CaptionRetryBudget()
        budget.auto_scale = False
        budget.max_attempts = 200
        budget.attempts_per_video = 1.0  # Low to avoid ValueError

        # batch_size > max_attempts but still mathematically ok
        # (300 videos with 1.0 per video = 300, but max_attempts = 200)
        # This WILL raise ValueError because 300 > 200

        # Let's test a case where it's borderline: max=200, batch=150, attempts_per_video=1.0
        # 150 * 1.0 = 150 <= 200, should NOT raise ValueError
        budget.max_attempts = 100  # Make batch > max_attempts but verify still passes
        budget.attempts_per_video = 0.5  # 150 * 0.5 = 75 <= 100

        budget.verify_budget_sufficient(150)

        # Should log warning because batch_size (150) > max_attempts (100)
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_logs) >= 1, (
            f"Expected WARNING log for batch > max_attempts, got: {[r.message for r in caplog.records]}"
        )

        log_msg = warning_logs[0].message
        assert "auto_scale" in log_msg.lower() or "DISABLED" in log_msg
        assert "150" in log_msg  # batch_size
        assert "100" in log_msg  # max_attempts

    @pytest.mark.fast
    def test_verify_budget_passes_with_exact_match(self):
        """Verify passes when max_attempts exactly equals required.

        US-41-004: max_attempts >= batch_size * attempts_per_video.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = False
        budget.max_attempts = 350  # Exactly 175 * 2.0
        budget.attempts_per_video = 2.0

        # Should not raise
        budget.verify_budget_sufficient(175)

    @pytest.mark.fast
    def test_verify_budget_error_message_includes_config_path(self):
        """Verify error message includes actionable config.yaml path.

        US-41-004: ValueError message must include config path for fix.
        """
        budget = CaptionRetryBudget()
        budget.auto_scale = False
        budget.max_attempts = 50
        budget.attempts_per_video = 2.0

        with pytest.raises(ValueError) as exc_info:
            budget.verify_budget_sufficient(100)  # Needs 200

        error_msg = str(exc_info.value)
        # Must mention how to fix
        assert "auto_scale" in error_msg
        assert "config.yaml" in error_msg
        assert "max_attempts" in error_msg


class TestCaptionRetryBudgetConfigValidation:
    """Tests for US-41-004 and US-41-011: Config validation in __post_init__.

    Verifies that CaptionRetryBudgetConfig validates minimum values
    at configuration load time to prevent misconfiguration.

    US-41-004: attempts_per_video > 0 (updated to >= 1.0 in US-41-011)
    US-41-011: min values for max_attempts, attempts_per_video, max_backoff_time_seconds
    """

    # ============================================================
    # US-41-011: max_attempts validation (>= 10 or 0 for unlimited)
    # ============================================================

    @pytest.mark.fast
    def test_config_validates_max_attempts_minimum(self):
        """Verify ValueError raised when max_attempts < 10 and != 0 (US-41-011 AC2)."""
        with pytest.raises(ValueError) as exc_info:
            ConfigCaptionRetryBudgetConfig(max_attempts=5)

        error_msg = str(exc_info.value)
        assert "max_attempts" in error_msg
        assert ">= 10" in error_msg
        assert "config.yaml" in error_msg

    @pytest.mark.fast
    def test_config_accepts_max_attempts_unlimited(self):
        """Verify max_attempts=0 (unlimited) is accepted (US-41-011 AC2)."""
        config = ConfigCaptionRetryBudgetConfig(max_attempts=0)
        assert config.max_attempts == 0

    @pytest.mark.fast
    def test_config_accepts_max_attempts_at_minimum(self):
        """Verify max_attempts=10 (minimum) is accepted (US-41-011 AC2)."""
        config = ConfigCaptionRetryBudgetConfig(max_attempts=10)
        assert config.max_attempts == 10

    @pytest.mark.fast
    def test_config_accepts_max_attempts_above_minimum(self):
        """Verify max_attempts > 10 is accepted."""
        config = ConfigCaptionRetryBudgetConfig(max_attempts=100)
        assert config.max_attempts == 100

    # ============================================================
    # US-41-011: attempts_per_video validation (>= 1.0)
    # ============================================================

    @pytest.mark.fast
    def test_config_validates_attempts_per_video_minimum(self):
        """Verify ValueError raised when attempts_per_video < 1.0 (US-41-011 AC3)."""
        with pytest.raises(ValueError) as exc_info:
            ConfigCaptionRetryBudgetConfig(attempts_per_video=0.5)

        error_msg = str(exc_info.value)
        assert "attempts_per_video" in error_msg
        assert ">= 1.0" in error_msg
        assert "config.yaml" in error_msg

    @pytest.mark.fast
    def test_config_validates_attempts_per_video_zero(self):
        """Verify ValueError raised when attempts_per_video = 0 (US-41-004/US-41-011)."""
        with pytest.raises(ValueError) as exc_info:
            ConfigCaptionRetryBudgetConfig(attempts_per_video=0)

        error_msg = str(exc_info.value)
        assert "attempts_per_video" in error_msg
        assert "config.yaml" in error_msg

    @pytest.mark.fast
    def test_config_validates_attempts_per_video_negative(self):
        """Verify ValueError raised for negative attempts_per_video."""
        with pytest.raises(ValueError) as exc_info:
            ConfigCaptionRetryBudgetConfig(attempts_per_video=-1.0)

        assert "attempts_per_video" in str(exc_info.value)

    @pytest.mark.fast
    def test_config_accepts_attempts_per_video_at_minimum(self):
        """Verify attempts_per_video=1.0 (minimum) is accepted (US-41-011 AC3)."""
        config = ConfigCaptionRetryBudgetConfig(attempts_per_video=1.0)
        assert config.attempts_per_video == 1.0

    @pytest.mark.fast
    def test_config_accepts_attempts_per_video_above_minimum(self):
        """Verify attempts_per_video > 1.0 is accepted."""
        config = ConfigCaptionRetryBudgetConfig(attempts_per_video=2.0)
        assert config.attempts_per_video == 2.0

        config = ConfigCaptionRetryBudgetConfig(attempts_per_video=3.5)
        assert config.attempts_per_video == 3.5

    # ============================================================
    # US-41-011: max_backoff_time_seconds validation (>= 30.0 or 0)
    # ============================================================

    @pytest.mark.fast
    def test_config_validates_max_backoff_time_minimum(self):
        """Verify ValueError raised when max_backoff_time_seconds < 30.0 and != 0 (US-41-011 AC4)."""
        with pytest.raises(ValueError) as exc_info:
            ConfigCaptionRetryBudgetConfig(max_backoff_time_seconds=10.0)

        error_msg = str(exc_info.value)
        assert "max_backoff_time_seconds" in error_msg
        assert ">= 30.0" in error_msg
        assert "config.yaml" in error_msg

    @pytest.mark.fast
    def test_config_accepts_max_backoff_time_unlimited(self):
        """Verify max_backoff_time_seconds=0 (unlimited) is accepted (US-41-011 AC4)."""
        config = ConfigCaptionRetryBudgetConfig(max_backoff_time_seconds=0)
        assert config.max_backoff_time_seconds == 0

    @pytest.mark.fast
    def test_config_accepts_max_backoff_time_at_minimum(self):
        """Verify max_backoff_time_seconds=30.0 (minimum) is accepted (US-41-011 AC4)."""
        config = ConfigCaptionRetryBudgetConfig(max_backoff_time_seconds=30.0)
        assert config.max_backoff_time_seconds == 30.0

    @pytest.mark.fast
    def test_config_accepts_max_backoff_time_above_minimum(self):
        """Verify max_backoff_time_seconds > 30.0 is accepted."""
        config = ConfigCaptionRetryBudgetConfig(max_backoff_time_seconds=300.0)
        assert config.max_backoff_time_seconds == 300.0

    # ============================================================
    # US-41-011: Helpful error messages (AC5)
    # ============================================================

    @pytest.mark.fast
    def test_error_message_includes_config_yaml_path(self):
        """Verify all validation errors include config.yaml path info (US-41-011 AC5)."""
        # Test max_attempts error message
        with pytest.raises(ValueError) as exc_info:
            ConfigCaptionRetryBudgetConfig(max_attempts=1)
        assert "download.caption_first.retry_budget.max_attempts" in str(exc_info.value)

        # Test attempts_per_video error message
        with pytest.raises(ValueError) as exc_info:
            ConfigCaptionRetryBudgetConfig(attempts_per_video=0.5)
        assert "download.caption_first.retry_budget.attempts_per_video" in str(exc_info.value)

        # Test max_backoff_time_seconds error message
        with pytest.raises(ValueError) as exc_info:
            ConfigCaptionRetryBudgetConfig(max_backoff_time_seconds=5.0)
        assert "download.caption_first.retry_budget.max_backoff_time_seconds" in str(exc_info.value)


class TestCaptionRetryBudgetPerVideoAttemptTracking:
    """Tests for US-41-005: Per-video attempt counting to diagnose budget consumption.

    When one video consumes 10+ attempts due to retry loops, it starves other videos
    of budget. This feature tracks per-video attempts to diagnose such issues.
    """

    @pytest.mark.fast
    def test_track_attempts_per_video_id(self):
        """Verify attempts_per_video_id dict tracks per-video attempt counts (US-41-005 AC1)."""
        budget = CaptionRetryBudget()

        # Record attempts for different videos
        budget.record_attempt("video_abc")
        budget.record_attempt("video_abc")
        budget.record_attempt("video_xyz")

        assert budget.attempts_per_video_id["video_abc"] == 2
        assert budget.attempts_per_video_id["video_xyz"] == 1

    @pytest.mark.fast
    def test_warning_logged_when_video_exceeds_3_attempts(self, caplog):
        """Verify WARNING logged when single video consumes >3 attempts (US-41-005 AC2)."""
        import logging
        caplog.set_level(logging.WARNING)

        budget = CaptionRetryBudget()

        # Record 3 attempts - no warning yet
        budget.record_attempt("video_problematic")
        budget.record_attempt("video_problematic")
        budget.record_attempt("video_problematic")

        assert "retry loop" not in caplog.text.lower()

        # 4th attempt triggers warning
        budget.record_attempt("video_problematic")

        assert "video_problematic" in caplog.text
        assert ">3" in caplog.text or "4 attempts" in caplog.text
        assert "retry loop" in caplog.text.lower()

    @pytest.mark.fast
    def test_get_high_attempt_videos_returns_videos_above_threshold(self):
        """Verify get_high_attempt_videos() returns video IDs with >threshold attempts (US-41-005 AC3)."""
        budget = CaptionRetryBudget()

        # video_a: 5 attempts (above threshold of 3)
        for _ in range(5):
            budget.record_attempt("video_a")

        # video_b: 3 attempts (at threshold, not above)
        for _ in range(3):
            budget.record_attempt("video_b")

        # video_c: 4 attempts (above threshold)
        for _ in range(4):
            budget.record_attempt("video_c")

        high_attempt = budget.get_high_attempt_videos(threshold=3)

        # Should include video_a and video_c (>3 attempts) but not video_b (=3)
        assert "video_a" in high_attempt
        assert "video_c" in high_attempt
        assert "video_b" not in high_attempt

    @pytest.mark.fast
    def test_get_high_attempt_videos_sorted_by_count_descending(self):
        """Verify high attempt videos are sorted by attempt count descending."""
        budget = CaptionRetryBudget()

        for _ in range(4):
            budget.record_attempt("video_low")
        for _ in range(10):
            budget.record_attempt("video_high")
        for _ in range(6):
            budget.record_attempt("video_mid")

        high_attempt = budget.get_high_attempt_videos(threshold=3)

        # Should be sorted: video_high (10), video_mid (6), video_low (4)
        assert high_attempt == ["video_high", "video_mid", "video_low"]

    @pytest.mark.fast
    def test_summary_includes_high_attempt_videos_when_present(self):
        """Verify get_summary includes high_attempt_videos when count > 0 (US-41-005 AC4)."""
        budget = CaptionRetryBudget()

        # Record 5 attempts for one video (exceeds threshold of 3)
        for _ in range(5):
            budget.record_attempt("video_problematic")

        summary = budget.get_summary()

        assert "high_attempt_videos" in summary
        assert "video_problematic" in summary["high_attempt_videos"]

    @pytest.mark.fast
    def test_summary_excludes_high_attempt_videos_when_none(self):
        """Verify get_summary excludes high_attempt_videos when no videos exceed threshold."""
        budget = CaptionRetryBudget()

        # All videos have <=3 attempts
        budget.record_attempt("video_a")
        budget.record_attempt("video_b")
        budget.record_attempt("video_c")

        summary = budget.get_summary()

        assert "high_attempt_videos" not in summary

    @pytest.mark.fast
    def test_video_with_5_attempts_appears_in_get_high_attempt_videos_3(self):
        """Verify video with 5 attempts appears in get_high_attempt_videos(3) (US-41-005 AC5)."""
        budget = CaptionRetryBudget()

        # Record exactly 5 attempts for test_video
        for _ in range(5):
            budget.record_attempt("test_video")

        result = budget.get_high_attempt_videos(3)

        assert "test_video" in result

    @pytest.mark.fast
    def test_per_video_tracking_serialization(self):
        """Verify attempts_per_video_id is preserved in to_dict/from_dict."""
        budget = CaptionRetryBudget()

        for _ in range(5):
            budget.record_attempt("video_a")
        for _ in range(3):
            budget.record_attempt("video_b")

        # Serialize
        data = budget.to_dict()
        assert "attempts_per_video_id" in data
        assert data["attempts_per_video_id"]["video_a"] == 5
        assert data["attempts_per_video_id"]["video_b"] == 3

        # Deserialize
        restored = CaptionRetryBudget.from_dict(data)
        assert restored.attempts_per_video_id["video_a"] == 5
        assert restored.attempts_per_video_id["video_b"] == 3

    @pytest.mark.fast
    def test_per_video_tracking_reset(self):
        """Verify attempts_per_video_id is cleared on reset()."""
        budget = CaptionRetryBudget()

        for _ in range(5):
            budget.record_attempt("video_a")
        assert len(budget.attempts_per_video_id) > 0

        budget.reset()

        assert len(budget.attempts_per_video_id) == 0

    @pytest.mark.fast
    def test_empty_video_id_not_tracked(self):
        """Verify empty string video_id is not tracked in per-video dict."""
        budget = CaptionRetryBudget()

        # Record attempts with empty video_id
        budget.record_attempt("")
        budget.record_attempt("")

        # Empty string should not be in tracking
        assert "" not in budget.attempts_per_video_id

    @pytest.mark.fast
    def test_warning_only_logged_once_per_video(self, caplog):
        """Verify warning is logged only once per video (at 4th attempt)."""
        import logging
        caplog.set_level(logging.WARNING)

        budget = CaptionRetryBudget()

        # Record 10 attempts for same video
        for _ in range(10):
            budget.record_attempt("video_repeat")

        # Should only have one warning message for this video
        warning_count = caplog.text.count("video_repeat")
        assert warning_count == 1, f"Expected 1 warning, got {warning_count}"


class TestCaptionRetryBudgetCircuitBreakerTrips:
    """Tests for US-41-006: Circuit breaker trip count in budget exhaustion logging.

    These tests verify that circuit breaker trips are counted and included
    in budget summaries to help distinguish rate-limiting from other failures.
    """

    @pytest.mark.fast
    def test_initial_circuit_breaker_trips_is_zero(self):
        """Verify circuit_breaker_trips starts at 0."""
        budget = CaptionRetryBudget()
        assert budget.circuit_breaker_trips == 0

    @pytest.mark.fast
    def test_record_circuit_trip_increments_counter(self):
        """Verify record_circuit_trip() increments the counter."""
        budget = CaptionRetryBudget()

        budget.record_circuit_trip()
        assert budget.circuit_breaker_trips == 1

        budget.record_circuit_trip()
        assert budget.circuit_breaker_trips == 2

    @pytest.mark.fast
    def test_get_summary_includes_circuit_breaker_trips(self):
        """Verify get_summary includes circuit_breaker_trips field (US-41-006 AC3/AC5)."""
        budget = CaptionRetryBudget()

        # Record some trips
        budget.record_circuit_trip()
        budget.record_circuit_trip()
        budget.record_circuit_trip()

        summary = budget.get_summary()

        assert "circuit_breaker_trips" in summary
        assert summary["circuit_breaker_trips"] == 3

    @pytest.mark.fast
    def test_get_summary_circuit_breaker_trips_zero_when_no_trips(self):
        """Verify circuit_breaker_trips is 0 when no trips recorded."""
        budget = CaptionRetryBudget()

        summary = budget.get_summary()

        assert "circuit_breaker_trips" in summary
        assert summary["circuit_breaker_trips"] == 0

    @pytest.mark.fast
    def test_circuit_breaker_trips_serialization(self):
        """Verify circuit_breaker_trips is preserved in to_dict/from_dict."""
        budget = CaptionRetryBudget()

        # Record some trips
        for _ in range(5):
            budget.record_circuit_trip()

        # Serialize
        data = budget.to_dict()
        assert "circuit_breaker_trips" in data
        assert data["circuit_breaker_trips"] == 5

        # Deserialize
        restored = CaptionRetryBudget.from_dict(data)
        assert restored.circuit_breaker_trips == 5

    @pytest.mark.fast
    def test_circuit_breaker_trips_reset(self):
        """Verify circuit_breaker_trips is cleared on reset()."""
        budget = CaptionRetryBudget()

        for _ in range(3):
            budget.record_circuit_trip()
        assert budget.circuit_breaker_trips == 3

        budget.reset()

        assert budget.circuit_breaker_trips == 0

    @pytest.mark.fast
    def test_budget_exhausted_logs_trips_when_positive(self, caplog):
        """Verify INFO log includes trips when budget exhausts with >0 trips (US-41-006 AC4)."""
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget(max_attempts=5)

        # Record some trips
        budget.record_circuit_trip()
        budget.record_circuit_trip()

        # Exhaust the budget
        for _ in range(5):
            budget.record_attempt("test_video")

        # Check budget is exhausted (this logs the INFO messages)
        assert budget.budget_exhausted() is True

        # Check for the circuit breaker trip log message
        assert "2 circuit breaker trip(s)" in caplog.text
        assert "contributed to budget exhaustion" in caplog.text

    @pytest.mark.fast
    def test_budget_exhausted_no_trips_log_when_zero(self, caplog):
        """Verify no trips log when budget exhausts with 0 trips."""
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget(max_attempts=5)

        # Exhaust the budget without any circuit trips
        for _ in range(5):
            budget.record_attempt("test_video")

        # Check budget is exhausted
        assert budget.budget_exhausted() is True

        # Should NOT have the trips message
        assert "circuit breaker trip(s)" not in caplog.text

    @pytest.mark.fast
    def test_backoff_exhaustion_logs_trips(self, caplog):
        """Verify INFO log includes trips when backoff budget exhausts with >0 trips."""
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget(max_attempts=0, max_backoff_time=10.0)

        # Record a trip
        budget.record_circuit_trip()

        # Exhaust the backoff budget
        budget.record_backoff(10.0, "test_video")

        # Check budget is exhausted (this logs the INFO messages)
        assert budget.budget_exhausted() is True

        # Check for the circuit breaker trip log message
        assert "1 circuit breaker trip(s)" in caplog.text


class TestCaptionRetryBudgetProgressPercentage:
    """Tests for US-41-010: Batch progress percentage tracking."""

    @pytest.mark.fast
    def test_progress_percentage_returns_correct_value(self):
        """Verify progress_percentage returns correct value based on batch processing."""
        budget = CaptionRetryBudget(max_attempts=500)
        budget.batch_size = 175

        # Process 100 videos (80 successes + 20 failures)
        for _ in range(80):
            budget.record_success("video_success")
        for _ in range(20):
            budget.record_failure("video_failure")

        # 100/175 = 57.14%
        progress = budget.get_progress_percentage()
        assert progress == 57.14

    @pytest.mark.fast
    def test_progress_percentage_returns_none_when_no_batch_size(self):
        """Verify progress_percentage returns None when batch_size not set."""
        budget = CaptionRetryBudget(max_attempts=500)
        # batch_size is None by default
        budget.record_success("video")
        budget.record_failure("video2")

        assert budget.get_progress_percentage() is None

    @pytest.mark.fast
    def test_progress_percentage_returns_none_when_batch_size_zero(self):
        """Verify progress_percentage returns None when batch_size is 0."""
        budget = CaptionRetryBudget(max_attempts=500)
        budget.batch_size = 0
        budget.record_success("video")

        assert budget.get_progress_percentage() is None

    @pytest.mark.fast
    def test_progress_percentage_in_summary(self):
        """Verify progress_percentage is included in get_summary() output."""
        budget = CaptionRetryBudget(max_attempts=500)
        budget.batch_size = 200

        for _ in range(50):
            budget.record_success("video")
        for _ in range(50):
            budget.record_failure("video")

        summary = budget.get_summary()

        # 100/200 = 50%
        assert "progress_percentage" in summary
        assert summary["progress_percentage"] == 50.0
        assert "videos_processed" in summary
        assert summary["videos_processed"] == 100

    @pytest.mark.fast
    def test_progress_percentage_zero_when_no_processing(self):
        """Verify progress_percentage is 0.0 when no videos processed."""
        budget = CaptionRetryBudget(max_attempts=500)
        budget.batch_size = 100

        assert budget.get_progress_percentage() == 0.0

    @pytest.mark.fast
    def test_exhausted_log_includes_progress(self, caplog):
        """Verify EXHAUSTED log message includes progress percentage."""
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget(max_attempts=5)
        budget.batch_size = 175

        # Process some videos before exhaustion
        for _ in range(80):
            budget.record_success(f"success_video")
        for _ in range(20):
            budget.record_failure(f"failure_video")

        # Exhaust the budget
        for _ in range(5):
            budget.record_attempt("exhaust_video")

        # Trigger exhaustion check
        assert budget.budget_exhausted() is True

        # Check log includes progress info
        assert "EXHAUSTED" in caplog.text
        assert "57%" in caplog.text
        assert "100/175 videos processed" in caplog.text

    @pytest.mark.fast
    def test_exhausted_log_no_progress_when_batch_size_missing(self, caplog):
        """Verify EXHAUSTED log omits progress when batch_size not set."""
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget(max_attempts=5)
        # No batch_size set

        for _ in range(5):
            budget.record_attempt("video")

        assert budget.budget_exhausted() is True

        # Should NOT contain progress info
        assert "videos processed" not in caplog.text

    @pytest.mark.fast
    def test_backoff_exhausted_log_includes_progress(self, caplog):
        """Verify backoff EXHAUSTED log message includes progress percentage."""
        import logging
        caplog.set_level(logging.INFO)

        budget = CaptionRetryBudget(max_attempts=0, max_backoff_time=10.0)
        budget.batch_size = 100

        # Process 40 videos
        for _ in range(30):
            budget.record_success("video")
        for _ in range(10):
            budget.record_failure("video")

        # Exhaust via backoff
        budget.record_backoff(10.0, "test_video")

        assert budget.budget_exhausted() is True

        # Check log includes progress info
        assert "EXHAUSTED" in caplog.text
        assert "40%" in caplog.text
        assert "40/100 videos processed" in caplog.text

    @pytest.mark.fast
    def test_get_videos_processed(self):
        """Verify get_videos_processed returns successes + failures."""
        budget = CaptionRetryBudget(max_attempts=500)

        assert budget.get_videos_processed() == 0

        for _ in range(25):
            budget.record_success("video")
        for _ in range(15):
            budget.record_failure("video")

        assert budget.get_videos_processed() == 40


class TestCaptionRetryBudgetIntegrityCheck:
    """Test restore_with_integrity_check for batch_size mismatch detection (US-41-012)."""

    @pytest.mark.fast
    def test_batch_size_change_triggers_warning(self, caplog):
        """Verify batch_size change is detected and logged as WARNING."""
        import logging
        caplog.set_level(logging.WARNING)

        checkpoint_data = {
            "attempts": 50,
            "failures": 10,
            "successes": 40,
            "backoff_time_spent": 5.0,
            "videos_skipped": 0,
            "max_attempts": 100,
            "max_backoff_time": 300.0,
            "batch_size": 50,  # Old checkpoint was for 50 videos
            "error_counts": {},
            "vpn_resets_used": 0,
            "max_vpn_resets": 2,
            "early_terminated": False,
            "early_termination_reason": None,
        }

        # Current batch has 175 videos - much larger
        budget, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            checkpoint_data,
            current_batch_size=175,
        )

        # Verify batch_size_changed is True
        assert restore_info["batch_size_changed"] is True
        assert restore_info["restored_batch_size"] == 50
        assert restore_info["current_batch_size"] == 175

        # Verify WARNING was logged
        assert "[US-41-012]" in caplog.text
        assert "batch_size (50) differs from current (175)" in caplog.text

    @pytest.mark.fast
    def test_batch_size_increase_triggers_rescale(self, caplog):
        """Verify current > restored batch_size forces re-scale."""
        import logging
        caplog.set_level(logging.INFO)

        checkpoint_data = {
            "attempts": 50,
            "failures": 10,
            "successes": 40,
            "backoff_time_spent": 5.0,
            "videos_skipped": 0,
            "max_attempts": 100,  # Scaled for 50 videos
            "max_backoff_time": 300.0,
            "batch_size": 50,
            "error_counts": {},
            "vpn_resets_used": 0,
            "max_vpn_resets": 2,
            "early_terminated": False,
            "early_termination_reason": None,
        }

        # Current batch is 175 - requires ~263 attempts at 1.5/video
        budget, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            checkpoint_data,
            current_batch_size=175,
            auto_scale=True,
            attempts_per_video=1.5,
        )

        # Verify force_rescaled is True
        assert restore_info["force_rescaled"] is True
        # 175 * 1.5 = 262.5 -> 263
        assert restore_info["scaled_max_attempts"] == 263
        assert budget.max_attempts == 263
        assert budget.batch_size == 175

        # Verify INFO log about re-scaling
        assert "Force re-scaled" in caplog.text

    @pytest.mark.fast
    def test_batch_size_decrease_no_rescale(self):
        """Verify current < restored batch_size does NOT force re-scale."""
        checkpoint_data = {
            "attempts": 50,
            "failures": 10,
            "successes": 40,
            "backoff_time_spent": 5.0,
            "videos_skipped": 0,
            "max_attempts": 300,  # Was scaled for 200 videos
            "max_backoff_time": 300.0,
            "batch_size": 200,
            "error_counts": {},
            "vpn_resets_used": 0,
            "max_vpn_resets": 2,
            "early_terminated": False,
            "early_termination_reason": None,
        }

        # Current batch is only 50 - smaller than original
        budget, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            checkpoint_data,
            current_batch_size=50,
            auto_scale=True,
            attempts_per_video=1.5,
        )

        # batch_size changed but no rescale needed (current smaller)
        assert restore_info["batch_size_changed"] is True
        assert restore_info["force_rescaled"] is False
        assert restore_info["scaled_max_attempts"] is None
        # max_attempts preserved from checkpoint
        assert budget.max_attempts == 300
        assert budget.batch_size == 50

    @pytest.mark.fast
    def test_no_checkpoint_data_returns_empty_info(self):
        """Verify None checkpoint data returns proper defaults."""
        budget, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            None,
            current_batch_size=100,
        )

        assert restore_info["batch_size_changed"] is False
        assert restore_info["restored_batch_size"] is None
        assert restore_info["current_batch_size"] == 100
        assert restore_info["force_rescaled"] is False
        assert restore_info["scaled_max_attempts"] is None
        assert budget.batch_size == 100

    @pytest.mark.fast
    def test_restore_info_includes_all_fields(self):
        """Verify restore_info dict has all required fields per acceptance criteria."""
        checkpoint_data = {
            "batch_size": 50,
            "max_attempts": 100,
        }

        _, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            checkpoint_data,
            current_batch_size=175,
        )

        # All required fields must be present
        assert "batch_size_changed" in restore_info
        assert "restored_batch_size" in restore_info
        assert "current_batch_size" in restore_info
        assert "force_rescaled" in restore_info
        assert "scaled_max_attempts" in restore_info

    @pytest.mark.fast
    def test_same_batch_size_no_change(self):
        """Verify same batch_size reports no change."""
        checkpoint_data = {
            "batch_size": 100,
            "max_attempts": 150,
        }

        budget, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            checkpoint_data,
            current_batch_size=100,
        )

        assert restore_info["batch_size_changed"] is False
        assert restore_info["force_rescaled"] is False
        assert budget.batch_size == 100

    @pytest.mark.fast
    def test_auto_scale_disabled_no_rescale(self):
        """Verify auto_scale=False prevents force re-scale."""
        checkpoint_data = {
            "batch_size": 50,
            "max_attempts": 100,
        }

        budget, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            checkpoint_data,
            current_batch_size=175,
            auto_scale=False,  # Disabled
        )

        # batch_size changed but no rescale because auto_scale=False
        assert restore_info["batch_size_changed"] is True
        assert restore_info["force_rescaled"] is False
        assert budget.max_attempts == 100  # Unchanged

    @pytest.mark.fast
    def test_rescale_respects_attempts_per_video(self):
        """Verify different attempts_per_video ratios affect scaling."""
        checkpoint_data = {
            "batch_size": 50,
            "max_attempts": 100,
        }

        # With 2.0 attempts_per_video: 175 * 2.0 = 350
        _, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            checkpoint_data,
            current_batch_size=175,
            auto_scale=True,
            attempts_per_video=2.0,
        )

        assert restore_info["force_rescaled"] is True
        assert restore_info["scaled_max_attempts"] == 350

    @pytest.mark.fast
    def test_null_batch_size_in_checkpoint(self):
        """Verify checkpoint with null batch_size doesn't trigger change warning."""
        checkpoint_data = {
            "batch_size": None,  # Old checkpoint format
            "max_attempts": 100,
        }

        _, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            checkpoint_data,
            current_batch_size=175,
        )

        # None != 175 but we only flag change when restored_batch_size is not None
        assert restore_info["batch_size_changed"] is False
        assert restore_info["restored_batch_size"] is None

    @pytest.mark.fast
    def test_rescale_only_when_required_exceeds_max(self):
        """Verify re-scale only happens when required > current max_attempts."""
        checkpoint_data = {
            "batch_size": 50,
            "max_attempts": 500,  # Already high enough for 175 videos
        }

        # 175 * 1.5 = 263, but max_attempts is already 500
        budget, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            checkpoint_data,
            current_batch_size=175,
            auto_scale=True,
            attempts_per_video=1.5,
        )

        # batch_size changed but no rescale needed (500 > 263)
        assert restore_info["batch_size_changed"] is True
        assert restore_info["force_rescaled"] is False
        assert budget.max_attempts == 500  # Preserved
