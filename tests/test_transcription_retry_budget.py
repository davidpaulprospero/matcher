"""
Tests for src/transcription/retry_budget.py - TranscriptionRetryBudget (US-79-010).

Tests batch retry budget tracking for transcription, ensuring:
- Budget exhaustion stops retrying individual videos
- Budget summary is included in returned metrics
- Per-category retry tracking (US-137-002)
"""

import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription.retry_budget import TranscriptionRetryBudget, TranscriptionErrorCategory
from src.transcription.metrics import TranscriptionMetrics


class TestTranscriptionRetryBudget:
    """Tests for TranscriptionRetryBudget class."""

    def test_initial_state(self):
        """Budget starts with zero usage."""
        budget = TranscriptionRetryBudget()
        assert budget.attempts == 0
        assert budget.failures == 0
        assert budget.successes == 0
        assert budget.backoff_time_spent == 0.0
        assert budget.videos_skipped == 0
        assert not budget.is_exhausted()

    def test_record_attempt(self):
        """record_attempt increments attempts counter."""
        budget = TranscriptionRetryBudget()
        budget.record_attempt("video1")
        budget.record_attempt("video2")
        assert budget.attempts == 2

    def test_record_failure(self):
        """record_failure increments failures counter."""
        budget = TranscriptionRetryBudget()
        budget.record_failure("video1")
        assert budget.failures == 1

    def test_record_success(self):
        """record_success increments successes counter."""
        budget = TranscriptionRetryBudget()
        budget.record_success("video1")
        assert budget.successes == 1

    def test_record_backoff(self):
        """record_backoff accumulates backoff time."""
        budget = TranscriptionRetryBudget()
        budget.record_backoff(1.0)
        budget.record_backoff(2.0)
        assert budget.backoff_time_spent == 3.0

    def test_exhausted_by_max_attempts(self):
        """Budget exhausted when max_attempts reached."""
        budget = TranscriptionRetryBudget(max_attempts=3)
        budget.record_attempt()
        budget.record_attempt()
        assert not budget.is_exhausted()
        budget.record_attempt()
        assert budget.is_exhausted()
        assert "max_attempts" in budget.exhaustion_reason()

    def test_exhausted_by_max_backoff(self):
        """Budget exhausted when max_backoff_time reached."""
        budget = TranscriptionRetryBudget(max_backoff_time=5.0)
        budget.record_backoff(3.0)
        assert not budget.is_exhausted()
        budget.record_backoff(3.0)
        assert budget.is_exhausted()
        assert "max_backoff_time" in budget.exhaustion_reason()

    def test_unlimited_attempts(self):
        """max_attempts=0 means unlimited."""
        budget = TranscriptionRetryBudget(max_attempts=0)
        for _ in range(1000):
            budget.record_attempt()
        assert not budget.is_exhausted()

    def test_unlimited_backoff(self):
        """max_backoff_time=0 means unlimited."""
        budget = TranscriptionRetryBudget(max_backoff_time=0)
        budget.record_backoff(99999.0)
        assert not budget.is_exhausted()

    def test_record_skip(self):
        """record_skip tracks skipped videos."""
        budget = TranscriptionRetryBudget()
        budget.record_skip("video_abc")
        budget.record_skip("video_xyz")
        assert budget.videos_skipped == 2
        assert budget.skipped_video_ids == ["video_abc", "video_xyz"]

    def test_get_summary(self):
        """get_summary returns complete budget summary."""
        budget = TranscriptionRetryBudget(max_attempts=10, max_backoff_time=30.0)
        budget.record_attempt("v1")
        budget.record_success("v1")
        budget.record_attempt("v2")
        budget.record_failure("v2")
        budget.record_backoff(1.5)

        summary = budget.get_summary()
        assert summary['total_attempts'] == 2
        assert summary['failed_attempts'] == 1
        assert summary['successful_attempts'] == 1
        assert summary['backoff_time_spent'] == 1.5
        assert summary['videos_skipped'] == 0
        assert summary['max_attempts'] == 10
        assert summary['max_backoff_time'] == 30.0
        assert summary['is_exhausted'] is False
        assert summary['exhaustion_reason'] is None

    def test_not_exhausted_reason_is_none(self):
        """exhaustion_reason returns None when not exhausted."""
        budget = TranscriptionRetryBudget()
        assert budget.exhaustion_reason() is None


class TestTranscriptionRetryBudgetPerCategory:
    """Tests for per-category retry tracking (US-137-002)."""

    def test_record_attempt_with_category(self):
        """record_attempt tracks attempts per error category."""
        budget = TranscriptionRetryBudget()
        budget.record_attempt("video1", TranscriptionErrorCategory.GPU_OOM)
        budget.record_attempt("video2", TranscriptionErrorCategory.GPU_OOM)
        budget.record_attempt("video3", TranscriptionErrorCategory.TRANSIENT)

        assert budget.attempts == 3
        by_category = budget.get_attempts_by_category()
        assert by_category[TranscriptionErrorCategory.GPU_OOM] == 2
        assert by_category[TranscriptionErrorCategory.TRANSIENT] == 1

    def test_record_failure_with_category(self):
        """record_failure tracks failures per error category."""
        budget = TranscriptionRetryBudget()
        budget.record_failure("video1", TranscriptionErrorCategory.GPU_OOM)
        budget.record_failure("video2", TranscriptionErrorCategory.TIMEOUT)
        budget.record_failure("video3", TranscriptionErrorCategory.TIMEOUT)

        assert budget.failures == 3
        by_category = budget.get_failures_by_category()
        assert by_category[TranscriptionErrorCategory.GPU_OOM] == 1
        assert by_category[TranscriptionErrorCategory.TIMEOUT] == 2

    def test_record_attempt_without_category(self):
        """record_attempt works without error category (backwards compatible)."""
        budget = TranscriptionRetryBudget()
        budget.record_attempt("video1")
        budget.record_attempt("video2")

        assert budget.attempts == 2
        assert budget.get_attempts_by_category() == {}

    def test_record_failure_without_category(self):
        """record_failure works without error category (backwards compatible)."""
        budget = TranscriptionRetryBudget()
        budget.record_failure("video1")

        assert budget.failures == 1
        assert budget.get_failures_by_category() == {}

    def test_get_summary_includes_category_breakdown(self):
        """get_summary includes per-category breakdown."""
        budget = TranscriptionRetryBudget()
        budget.record_attempt("v1", TranscriptionErrorCategory.GPU_OOM)
        budget.record_attempt("v2", TranscriptionErrorCategory.GPU_OOM)
        budget.record_failure("v1", TranscriptionErrorCategory.GPU_OOM)
        budget.record_attempt("v3", TranscriptionErrorCategory.TRANSIENT)
        budget.record_failure("v3", TranscriptionErrorCategory.TRANSIENT)
        budget.record_attempt("v4", TranscriptionErrorCategory.TIMEOUT)

        summary = budget.get_summary()

        assert summary['attempts_by_category'] == {
            'gpu_oom': 2,
            'transient': 1,
            'timeout': 1,
        }
        assert summary['failures_by_category'] == {
            'gpu_oom': 1,
            'transient': 1,
        }

    def test_all_error_categories_tracked(self):
        """All error categories can be tracked independently."""
        budget = TranscriptionRetryBudget()

        for cat in TranscriptionErrorCategory:
            budget.record_attempt(f"v_{cat.value}", cat)
            budget.record_failure(f"v_{cat.value}", cat)

        by_category = budget.get_attempts_by_category()
        failures_by_category = budget.get_failures_by_category()

        assert len(by_category) == len(TranscriptionErrorCategory)
        assert len(failures_by_category) == len(TranscriptionErrorCategory)

        for cat in TranscriptionErrorCategory:
            assert by_category[cat] == 1
            assert failures_by_category[cat] == 1

    def test_category_distribution_logged_on_exhaustion(self, caplog):
        """Category distribution is logged when budget exhausts."""
        import logging
        caplog.set_level(logging.INFO)

        budget = TranscriptionRetryBudget(max_attempts=3)
        budget.record_attempt("v1", TranscriptionErrorCategory.GPU_OOM)
        budget.record_attempt("v2", TranscriptionErrorCategory.TRANSIENT)
        budget.record_attempt("v3", TranscriptionErrorCategory.GPU_OOM)

        # Trigger exhaustion check
        budget.is_exhausted()

        # Should log category distribution
        assert any(
            "Category distribution at exhaustion" in record.message
            for record in caplog.records
        )

    def test_no_category_logging_when_no_categories(self, caplog):
        """No category logging when no categories tracked."""
        import logging
        caplog.set_level(logging.INFO)

        budget = TranscriptionRetryBudget(max_attempts=2)
        budget.record_attempt("v1")
        budget.record_attempt("v2")

        # Trigger exhaustion check
        budget.is_exhausted()

        # Should NOT log category distribution (none tracked)
        assert not any(
            "Category distribution at exhaustion" in record.message
            for record in caplog.records
        )


class TestTranscriptionRetryBudgetInParallelProcessor:
    """Integration tests: budget stops batch processing when exhausted."""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.get_audio_duration')
    def test_batch_stops_retrying_when_budget_exhausted(
        self, mock_duration, mock_extract, mock_cache_cls, mock_whisper_cls
    ):
        """Batch processing stops retrying individual videos when budget is exhausted."""
        from src.transcription.parallel_processor import transcribe_videos_parallel
        from src.transcription.exceptions import TransientTranscriptionError

        # Setup mocks
        mock_cache = Mock()
        mock_cache.cache_dir = "/fake/cache"
        mock_cache_instance = mock_cache_cls.return_value
        mock_cache_instance.get.return_value = None  # No cache hits

        mock_whisper = mock_whisper_cls.return_value
        # Make transcription always fail with transient error
        mock_whisper.transcribe.side_effect = TransientTranscriptionError("GPU OOM")

        mock_extract.side_effect = lambda vp, td, timeout=60: f"/fake/audio/{Path(vp).stem}.wav"
        mock_duration.return_value = 10.0

        # Config with very small budget: max 5 attempts total
        config = Mock()
        config.transcription = Mock()
        config.transcription.model = "base"
        config.transcription.compute_type = "auto"
        config.transcription.language = "en"
        config.transcription.vad_filter = False
        config.transcription.min_silence_duration_ms = 200
        config.transcription.speech_pad_ms = 10
        config.transcription.audio_extraction_workers = 2
        config.transcription.auto_cleanup_after_batch = False
        config.transcription.gpu_transcription_timeout = 300
        config.transcription.audio_extraction_timeout = 60
        config.transcription.max_retries = 2  # 3 attempts per video
        config.transcription.whisper_num_workers = 1
        config.transcription.whisper_cpu_threads = 4
        config.transcription.progress_log_interval = 10
        config.transcription.retry_budget_max_attempts = 5  # Very small budget
        config.transcription.retry_budget_max_backoff_seconds = 180.0
        # US-137-011: Backoff config
        config.transcription.backoff_strategy = "jitter"
        config.transcription.backoff_jitter_factor = 0.3
        config.transcription.backoff_correlation_factor = 0.5
        config.transcription.backoff_max_jitter_cap = 10.0
        # Additional config needed by the pipeline
        config.transcription.worker_multiplier = 0.5
        config.transcription.dynamic_pipeline_depth = False
        config.transcription.pipeline_depth = 0  # Disable pipelining in tests

        # 10 videos - budget should exhaust before all are tried
        videos = [f"/fake/video_{i}.mp4" for i in range(10)]

        with patch('src.transcription.parallel_processor.time.sleep'):
            with patch('src.transcription.parallel_processor.Path') as mock_path:
                mock_path.return_value.mkdir.return_value = None
                mock_path.return_value.stem = "test"
                mock_path.return_value.unlink.return_value = None
                # Make Path(x).stem work for logging
                mock_path.side_effect = lambda x: Path(x)

                results, metrics = transcribe_videos_parallel(
                    video_paths=videos,
                    cache=mock_cache,
                    config=config,
                    show_progress=False,
                    return_metrics=True,
                )

        # All 10 videos should have results (empty lists for failed/skipped)
        assert len(results) == 10

        # Budget should have been exhausted
        summary = metrics.get_summary_dict()
        assert summary['budget_total_attempts'] > 0
        assert summary['budget_exhausted_count'] > 0
        # With 5 attempt budget and 3 attempts per video (1 + 2 retries),
        # the first video uses 3 attempts, second starts and budget hits at attempt 5,
        # remaining ~8 videos should be skipped
        assert summary['budget_exhausted_count'] >= 1

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.get_audio_duration')
    def test_budget_summary_in_metrics(
        self, mock_duration, mock_extract, mock_cache_cls, mock_whisper_cls
    ):
        """Budget summary is included in returned metrics even when no exhaustion."""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        # Setup mocks for successful transcription
        mock_cache = Mock()
        mock_cache.cache_dir = "/fake/cache"
        mock_cache_instance = mock_cache_cls.return_value
        mock_cache_instance.get.return_value = None

        mock_whisper = mock_whisper_cls.return_value
        mock_whisper.transcribe.return_value = [
            {'start': 0.0, 'end': 5.0, 'text': 'Test segment'}
        ]

        mock_extract.side_effect = lambda vp, td, timeout=60: f"/fake/audio/{Path(vp).stem}.wav"
        mock_duration.return_value = 10.0

        config = Mock()
        config.transcription = Mock()
        config.transcription.model = "base"
        config.transcription.compute_type = "auto"
        config.transcription.language = "en"
        config.transcription.vad_filter = False
        config.transcription.min_silence_duration_ms = 200
        config.transcription.speech_pad_ms = 10
        config.transcription.audio_extraction_workers = 2
        config.transcription.auto_cleanup_after_batch = False
        config.transcription.gpu_transcription_timeout = 300
        config.transcription.audio_extraction_timeout = 60
        config.transcription.max_retries = 2
        config.transcription.whisper_num_workers = 1
        config.transcription.whisper_cpu_threads = 4
        config.transcription.progress_log_interval = 10
        config.transcription.retry_budget_max_attempts = 50
        config.transcription.retry_budget_max_backoff_seconds = 180.0
        # US-137-011: Backoff config
        config.transcription.backoff_strategy = "jitter"
        config.transcription.backoff_jitter_factor = 0.3
        config.transcription.backoff_correlation_factor = 0.5
        config.transcription.backoff_max_jitter_cap = 10.0
        # Additional config needed by the pipeline
        config.transcription.worker_multiplier = 0.5
        config.transcription.dynamic_pipeline_depth = False
        config.transcription.pipeline_depth = 0  # Disable pipelining in tests

        videos = ["/fake/video_1.mp4", "/fake/video_2.mp4"]

        with patch('src.transcription.parallel_processor.Path') as mock_path:
            mock_path.side_effect = lambda x: Path(x)

            results, metrics = transcribe_videos_parallel(
                video_paths=videos,
                cache=mock_cache,
                config=config,
                show_progress=False,
                return_metrics=True,
            )

        # Budget summary should be present in metrics
        summary = metrics.get_summary_dict()
        assert 'budget_total_attempts' in summary
        assert 'budget_failed_attempts' in summary
        assert 'budget_exhausted_count' in summary
        # 2 successful videos = 2 attempts (1 each, no retries needed)
        assert summary['budget_total_attempts'] == 2
        assert summary['budget_failed_attempts'] == 0
        assert summary['budget_exhausted_count'] == 0


class TestTranscriptionMetricsBudgetFields:
    """Tests for budget fields in TranscriptionMetrics."""

    def test_initial_budget_fields(self):
        """Budget fields default to zero."""
        metrics = TranscriptionMetrics()
        assert metrics.budget_total_attempts == 0
        assert metrics.budget_failed_attempts == 0
        assert metrics.budget_exhausted_count == 0
        assert metrics.budget_summary is None

    def test_set_retry_budget_summary(self):
        """set_retry_budget_summary populates budget fields."""
        metrics = TranscriptionMetrics()
        summary = {
            'total_attempts': 25,
            'failed_attempts': 8,
            'successful_attempts': 17,
            'backoff_time_spent': 12.5,
            'videos_skipped': 3,
            'skipped_video_ids': ['v1', 'v2', 'v3'],
            'max_attempts': 50,
            'max_backoff_time': 180.0,
            'is_exhausted': False,
            'exhaustion_reason': None,
        }
        metrics.set_retry_budget_summary(summary)

        assert metrics.budget_total_attempts == 25
        assert metrics.budget_failed_attempts == 8
        assert metrics.budget_exhausted_count == 3
        assert metrics.budget_summary == summary

    def test_budget_in_summary_dict(self):
        """Budget fields appear in get_summary_dict."""
        metrics = TranscriptionMetrics()
        metrics.budget_total_attempts = 10
        metrics.budget_failed_attempts = 3
        metrics.budget_exhausted_count = 1

        d = metrics.get_summary_dict()
        assert d['budget_total_attempts'] == 10
        assert d['budget_failed_attempts'] == 3
        assert d['budget_exhausted_count'] == 1

    def test_budget_in_to_dict(self):
        """Budget fields appear in to_dict for serialization."""
        metrics = TranscriptionMetrics()
        metrics.budget_total_attempts = 5
        metrics.budget_failed_attempts = 2
        metrics.budget_exhausted_count = 0

        d = metrics.to_dict()
        assert d['budget_total_attempts'] == 5
        assert d['budget_failed_attempts'] == 2
        assert d['budget_exhausted_count'] == 0

    def test_budget_roundtrip_from_dict(self):
        """Budget fields survive to_dict/from_dict roundtrip."""
        metrics = TranscriptionMetrics()
        metrics.budget_total_attempts = 42
        metrics.budget_failed_attempts = 7
        metrics.budget_exhausted_count = 2

        restored = TranscriptionMetrics.from_dict(metrics.to_dict())
        assert restored.budget_total_attempts == 42
        assert restored.budget_failed_attempts == 7
        assert restored.budget_exhausted_count == 2


class TestTranscriptionBackoffManager:
    """Tests for TranscriptionBackoffManager (US-137-011)."""

    def test_initial_state(self):
        """Backoff manager starts with default values."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager()
        assert manager.base_delay == 1.0
        assert manager.jitter_factor == 0.3
        assert manager.correlation_factor == 0.5
        assert manager.strategy == BackoffStrategy.STANDARD
        assert manager.max_jitter_cap == 10.0

    def test_standard_backoff_no_jitter(self):
        """Standard strategy returns deterministic exponential backoff."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager(
            base_delay=1.0,
            strategy=BackoffStrategy.STANDARD
        )

        # Should be exactly: base_delay * 2^attempt
        assert manager.calculate_delay(0) == 1.0
        assert manager.calculate_delay(1) == 2.0
        assert manager.calculate_delay(2) == 4.0
        assert manager.calculate_delay(3) == 8.0

    def test_jitter_backoff_varies(self):
        """Jitter strategy adds randomness to delay."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager(
            base_delay=1.0,
            jitter_factor=0.5,  # 50% jitter range
            strategy=BackoffStrategy.JITTER
        )

        # With 50% jitter factor, delays should vary
        # For attempt=1: base_delay * 2^1 = 2.0
        # With 50% jitter: range is [2.0 - 1.0, 2.0 + 1.0] = [1.0, 3.0]
        delays = [manager.calculate_delay(1) for _ in range(100)]

        # All should be within [1.0, 3.0]
        assert all(1.0 <= d <= 3.0 for d in delays)
        # Not all should be the same (jitter is working)
        assert len(set(delays)) > 1

    def test_jitter_respects_max_cap(self):
        """Jitter strategy respects max_jitter_cap."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager(
            base_delay=1.0,
            jitter_factor=1.0,  # 100% jitter (would normally go to 2.0)
            max_jitter_cap=1.5,
            strategy=BackoffStrategy.JITTER
        )

        # Even with high jitter, should be capped at 1.5
        for _ in range(100):
            delay = manager.calculate_delay(0)
            assert delay <= 1.5

    def test_correlated_backoff_differs_by_worker(self):
        """Correlated strategy produces different delays for different workers."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager(
            base_delay=1.0,
            jitter_factor=0.1,  # Low jitter to isolate correlation effect
            correlation_factor=0.8,
            strategy=BackoffStrategy.CORRELATED
        )

        # Calculate delays for different workers at same attempt
        delays = [manager.calculate_delay(1, worker_id=i) for i in range(10)]

        # Delays should vary across workers (correlation offset)
        assert len(set(delays)) > 1

    def test_adaptive_strategy_selects_best(self):
        """Adaptive strategy selects best performing strategy."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager(strategy=BackoffStrategy.ADAPTIVE)

        # Record success for JITTER strategy
        manager._success_counts[BackoffStrategy.JITTER] = 10
        manager._failure_counts[BackoffStrategy.JITTER] = 0

        # Record poor results for STANDARD
        manager._success_counts[BackoffStrategy.STANDARD] = 1
        manager._failure_counts[BackoffStrategy.STANDARD] = 9

        # Adaptive should select JITTER (100% success rate)
        best = manager.get_best_strategy()
        assert best == BackoffStrategy.JITTER

    def test_success_rate_tracking(self):
        """Success rate is calculated correctly."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager()

        # Record some successes and failures
        manager.record_success(BackoffStrategy.JITTER)
        manager.record_success(BackoffStrategy.JITTER)
        manager.record_success(BackoffStrategy.JITTER)
        manager.record_failure(BackoffStrategy.JITTER)

        # 3 successes, 1 failure = 75% success rate
        rate = manager.get_success_rate(BackoffStrategy.JITTER)
        assert rate == 0.75

    def test_strategy_stats(self):
        """get_strategy_stats returns stats for all strategies."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager()

        manager.record_success(BackoffStrategy.STANDARD)
        manager.record_failure(BackoffStrategy.JITTER)

        stats = manager.get_strategy_stats()

        assert 'standard' in stats
        assert 'jitter' in stats
        assert 'correlated' in stats
        assert 'adaptive' in stats

        assert stats['standard']['successes'] == 1
        assert stats['jitter']['failures'] == 1

    def test_reset_stats(self):
        """reset_stats clears all historical data."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager()

        manager.record_success(BackoffStrategy.JITTER)
        manager.record_failure(BackoffStrategy.STANDARD)

        manager.reset_stats()

        # Stats should be cleared
        assert manager.get_success_rate(BackoffStrategy.JITTER) == 0.5  # Default neutral
        assert manager.get_success_rate(BackoffStrategy.STANDARD) == 0.5

    def test_backoff_with_zero_attempt(self):
        """Delay calculation works correctly for attempt 0."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager(
            base_delay=1.0,
            strategy=BackoffStrategy.STANDARD
        )

        # Attempt 0 should return base_delay
        assert manager.calculate_delay(0) == 1.0

    def test_backoff_with_high_attempt_caps_at_max(self):
        """High attempt numbers are capped at max_jitter_cap."""
        from src.transcription.retry_budget import TranscriptionBackoffManager, BackoffStrategy

        manager = TranscriptionBackoffManager(
            base_delay=1.0,
            max_jitter_cap=10.0,
            strategy=BackoffStrategy.STANDARD
        )

        # At attempt 10, 2^10 = 1024, should be capped at 10.0
        delay = manager.calculate_delay(10)
        assert delay == 10.0
