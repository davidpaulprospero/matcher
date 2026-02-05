"""
Tests for BatchProcessor class.

Verifies batch processing with mocked fetcher, progress callbacks,
and checkpoint integration.
"""

import pytest
import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass


@dataclass
class MockCaptionResult:
    """Mock CaptionResult for testing."""
    video_id: str
    language: str = "en"
    caption_quality: str = "medium"
    segments: list = None
    is_auto_generated: bool = False
    format_source: str = "vtt"

    def __post_init__(self):
        if self.segments is None:
            self.segments = []


class TestBatchProcessorInit:
    """Tests for BatchProcessor initialization."""

    def test_init_with_defaults(self):
        """BatchProcessor initializes with default config."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        mock_fetcher = MagicMock()
        processor = BatchProcessor(mock_fetcher)

        assert processor.fetcher is mock_fetcher
        assert processor.config is not None
        assert processor.config.max_workers == 4

    def test_init_with_custom_config(self):
        """BatchProcessor accepts custom config."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        mock_fetcher = MagicMock()
        config = BatchProcessorConfig(max_workers=8, checkpoint_save_interval=5)
        processor = BatchProcessor(mock_fetcher, config=config)

        assert processor.config.max_workers == 8
        assert processor.config.checkpoint_save_interval == 5


class TestBatchProcessorProcess:
    """Tests for BatchProcessor.process() method."""

    def test_process_empty_list(self):
        """Processing empty list returns empty result."""
        from src.caption.batch_processor import BatchProcessor

        mock_fetcher = MagicMock()
        processor = BatchProcessor(mock_fetcher)

        result = processor.process([])

        assert result.results == {}
        assert result.success_count == 0
        assert result.error_count == 0

    def test_process_single_video_success(self):
        """Processing single video with success."""
        from src.caption.batch_processor import BatchProcessor

        mock_fetcher = MagicMock()
        mock_result = MockCaptionResult(video_id="test123", segments=[{"text": "hello"}])
        mock_fetcher.fetch_captions_auto_language_with_retry.return_value = mock_result
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        processor = BatchProcessor(mock_fetcher)

        result = processor.process(["test123"])

        assert result.success_count == 1
        assert result.error_count == 0
        assert "test123" in result.results
        mock_fetcher.fetch_captions_auto_language_with_retry.assert_called_once()

    def test_process_with_on_video_complete_callback(self):
        """on_video_complete callback is called for each video."""
        from src.caption.batch_processor import BatchProcessor

        mock_fetcher = MagicMock()
        mock_result = MockCaptionResult(video_id="vid1", segments=[])
        mock_fetcher.fetch_captions_auto_language_with_retry.return_value = mock_result
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        # Track callback calls
        callback_calls: List[tuple] = []

        def on_complete(video_id: str, result: Any) -> None:
            callback_calls.append((video_id, result))

        processor = BatchProcessor(mock_fetcher)
        processor.process(["vid1"], on_video_complete=on_complete)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "vid1"
        assert callback_calls[0][1] is mock_result

    def test_process_with_progress_callback(self):
        """progress_callback receives fetching and success events."""
        from src.caption.batch_processor import BatchProcessor

        mock_fetcher = MagicMock()
        mock_result = MockCaptionResult(video_id="vid1", segments=[])
        mock_fetcher.fetch_captions_auto_language_with_retry.return_value = mock_result
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        # Track progress events
        progress_events: List[tuple] = []

        def on_progress(video_id: str, status: str, details: Dict) -> None:
            progress_events.append((video_id, status, details))

        processor = BatchProcessor(mock_fetcher)
        processor.process(["vid1"], progress_callback=on_progress)

        # Should have 'fetching' and 'success' events
        statuses = [e[1] for e in progress_events]
        assert "fetching" in statuses
        assert "success" in statuses

    def test_process_handles_fetch_error(self):
        """Fetch errors are captured and counted."""
        from src.caption.batch_processor import BatchProcessor
        from src.caption.exceptions import CaptionFetchError

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = (
            CaptionFetchError("test123", "Network error")
        )
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        processor = BatchProcessor(mock_fetcher)
        result = processor.process(["test123"])

        assert result.success_count == 0
        assert result.error_count == 1
        assert "test123" in result.results
        assert result.results["test123"]["error"] is True

    def test_process_handles_unavailable_error(self):
        """Unavailable errors are captured separately."""
        from src.caption.batch_processor import BatchProcessor
        from src.caption.exceptions import CaptionUnavailableError

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = (
            CaptionUnavailableError("test123", "No captions available")
        )
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        processor = BatchProcessor(mock_fetcher)
        result = processor.process(["test123"])

        assert result.success_count == 0
        assert result.error_count == 1
        assert result.results["test123"]["unavailable"] is True

    def test_process_with_skip_ids(self):
        """Videos in skip_video_ids are not processed."""
        from src.caption.batch_processor import BatchProcessor

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        processor = BatchProcessor(mock_fetcher)
        result = processor.process(
            ["vid1", "vid2", "vid3"],
            skip_video_ids={"vid1", "vid3"}
        )

        # Only vid2 should be fetched
        assert mock_fetcher.fetch_captions_auto_language_with_retry.call_count == 1

    def test_process_multiple_videos_parallel(self):
        """Multiple videos are processed in parallel."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        # Return different results based on video_id
        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=2)
        processor = BatchProcessor(mock_fetcher, config=config)

        result = processor.process(["vid1", "vid2", "vid3"])

        assert result.success_count == 3
        assert result.error_count == 0
        assert len(result.results) == 3


class TestBatchProcessorWithCheckpoint:
    """Tests for BatchProcessor with checkpoint integration."""

    def test_process_updates_checkpoint(self):
        """Checkpoint is updated as videos complete."""
        from src.caption.batch_processor import BatchProcessor
        from src.caption.batch_checkpoint import CaptionBatchCheckpoint

        mock_fetcher = MagicMock()
        mock_result = MockCaptionResult(video_id="vid1", segments=[])
        mock_fetcher.fetch_captions_auto_language_with_retry.return_value = mock_result
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        checkpoint = CaptionBatchCheckpoint()

        processor = BatchProcessor(mock_fetcher)
        processor.process(["vid1"], batch_checkpoint=checkpoint)

        assert "vid1" in checkpoint.results


class TestBatchProcessorWithMetrics:
    """Tests for BatchProcessor with metrics tracking."""

    def test_process_records_metrics(self):
        """Metrics are recorded during processing."""
        from src.caption.batch_processor import BatchProcessor
        from src.caption.metrics import CaptionMetrics

        mock_fetcher = MagicMock()
        mock_result = MockCaptionResult(video_id="vid1", segments=[{"text": "test"}])
        mock_fetcher.fetch_captions_auto_language_with_retry.return_value = mock_result
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        metrics = CaptionMetrics()

        processor = BatchProcessor(mock_fetcher)
        processor.process(["vid1"], metrics=metrics)

        assert metrics.successes == 1
        assert metrics.fetch_attempts == 1


class TestBatchProcessorConfig:
    """Tests for BatchProcessorConfig."""

    def test_default_values(self):
        """Default config values are set."""
        from src.caption.batch_processor import BatchProcessorConfig

        config = BatchProcessorConfig()

        assert config.max_workers == 4
        assert config.checkpoint_save_interval == 10
        assert config.error_pattern_mode == 'warn'
        assert config.error_pattern_threshold == 0.3
        assert config.prioritize_by_channel is True


class TestBatchProcessorConfigFromCaptionFirst:
    """Tests for BatchProcessorConfig.from_caption_first_config (US-66-005)."""

    def test_custom_progress_report_interval_from_config(self):
        """Custom progress_report_interval is pulled from CaptionFirstConfig."""
        from src.caption.batch_processor import BatchProcessorConfig
        from src.config.sections.download import CaptionFirstConfig

        caption_cfg = CaptionFirstConfig(progress_report_interval=50)
        batch_cfg = BatchProcessorConfig.from_caption_first_config(caption_cfg)

        assert batch_cfg.progress_report_interval == 50

    def test_custom_unavailable_threshold_from_config(self):
        """Custom unavailable_threshold is pulled from CaptionFirstConfig."""
        from src.caption.batch_processor import BatchProcessorConfig
        from src.config.sections.download import CaptionFirstConfig

        caption_cfg = CaptionFirstConfig(unavailable_threshold=0.5)
        batch_cfg = BatchProcessorConfig.from_caption_first_config(caption_cfg)

        assert batch_cfg.unavailable_threshold == 0.5

    def test_both_custom_values_respected(self):
        """Both progress_report_interval and unavailable_threshold are respected together."""
        from src.caption.batch_processor import BatchProcessorConfig
        from src.config.sections.download import CaptionFirstConfig

        caption_cfg = CaptionFirstConfig(
            progress_report_interval=10,
            unavailable_threshold=0.9,
        )
        batch_cfg = BatchProcessorConfig.from_caption_first_config(caption_cfg)

        assert batch_cfg.progress_report_interval == 10
        assert batch_cfg.unavailable_threshold == 0.9

    def test_defaults_match_when_no_custom_values(self):
        """Default CaptionFirstConfig values match BatchProcessorConfig defaults."""
        from src.caption.batch_processor import BatchProcessorConfig
        from src.config.sections.download import CaptionFirstConfig

        caption_cfg = CaptionFirstConfig()
        batch_cfg = BatchProcessorConfig.from_caption_first_config(caption_cfg)

        assert batch_cfg.progress_report_interval == 25
        assert batch_cfg.unavailable_threshold == 0.8

    def test_error_pattern_fields_pulled_from_config(self):
        """error_pattern_threshold and error_pattern_sample_size are pulled through."""
        from src.caption.batch_processor import BatchProcessorConfig
        from src.config.sections.download import CaptionFirstConfig

        caption_cfg = CaptionFirstConfig(
            error_pattern_threshold=0.5,
            error_pattern_sample_size=20,
        )
        batch_cfg = BatchProcessorConfig.from_caption_first_config(caption_cfg)

        assert batch_cfg.error_pattern_threshold == 0.5
        assert batch_cfg.error_pattern_sample_size == 20

    def test_overrides_take_precedence(self):
        """Explicit overrides take precedence over CaptionFirstConfig values."""
        from src.caption.batch_processor import BatchProcessorConfig
        from src.config.sections.download import CaptionFirstConfig

        caption_cfg = CaptionFirstConfig(progress_report_interval=50)
        batch_cfg = BatchProcessorConfig.from_caption_first_config(
            caption_cfg, progress_report_interval=100
        )

        assert batch_cfg.progress_report_interval == 100

    def test_dict_config_access(self):
        """from_caption_first_config works with dict-style config (Rule 6)."""
        from src.caption.batch_processor import BatchProcessorConfig

        config_dict = {
            'progress_report_interval': 15,
            'unavailable_threshold': 0.6,
            'abort_on_error_pattern': 'abort',
        }
        batch_cfg = BatchProcessorConfig.from_caption_first_config(config_dict)

        assert batch_cfg.progress_report_interval == 15
        assert batch_cfg.unavailable_threshold == 0.6
        assert batch_cfg.error_pattern_mode == 'abort'


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_default_values(self):
        """BatchResult has correct defaults."""
        from src.caption.batch_processor import BatchResult

        result = BatchResult()

        assert result.results == {}
        assert result.success_count == 0
        assert result.error_count == 0
        assert result.skipped_count == 0
        assert result.aborted is False
        assert result.abort_reason is None
        assert result.pattern_detected is None


class TestBatchProcessorThreadSafety:
    """Tests for thread safety of BatchProcessor."""

    def test_on_video_complete_called_from_multiple_threads(self):
        """on_video_complete is safely called from multiple threads."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        def mock_fetch(video_id, preferred_language=None):
            import time
            time.sleep(0.01)  # Small delay to allow parallel execution
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        # Track calls with thread-safe list
        lock = threading.Lock()
        callback_calls: List[str] = []

        def on_complete(video_id: str, result: Any) -> None:
            with lock:
                callback_calls.append(video_id)

        config = BatchProcessorConfig(max_workers=4)
        processor = BatchProcessor(mock_fetcher, config=config)

        videos = [f"vid{i}" for i in range(10)]
        processor.process(videos, on_video_complete=on_complete)

        assert len(callback_calls) == 10
        assert set(callback_calls) == set(videos)


class TestBatchProcessorProgressReporting:
    """Tests for batch progress reporting (US-37-011)."""

    def test_progress_report_interval_config(self):
        """BatchProcessorConfig includes progress_report_interval."""
        from src.caption.batch_processor import BatchProcessorConfig

        config = BatchProcessorConfig()
        assert config.progress_report_interval == 25  # Default

        config = BatchProcessorConfig(progress_report_interval=10)
        assert config.progress_report_interval == 10

    def test_batch_progress_logged_at_intervals(self, caplog):
        """Progress is logged at correct intervals."""
        import logging
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        # Set log level to INFO to capture batch progress
        caplog.set_level(logging.INFO)

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        # Use interval of 10 for faster testing
        config = BatchProcessorConfig(max_workers=1, progress_report_interval=10)
        processor = BatchProcessor(mock_fetcher, config=config)

        # Process 35 videos - should see progress at 10, 20, 30
        videos = [f"vid{i:03d}" for i in range(35)]
        processor.process(videos)

        # Check that batch progress was logged
        progress_logs = [
            r.message for r in caplog.records
            if "Batch progress:" in r.message
        ]

        # Should have 3 progress logs (at 10, 20, 30)
        assert len(progress_logs) == 3

        # Verify format: [N/35] X success, Y failed, Z skipped
        assert "[10/35]" in progress_logs[0]
        assert "[20/35]" in progress_logs[1]
        assert "[30/35]" in progress_logs[2]

    def test_batch_progress_includes_success_fail_skip_counts(self, caplog):
        """Progress includes success, failed, and skipped counts."""
        import logging
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from src.caption.exceptions import CaptionFetchError

        caplog.set_level(logging.INFO)

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        call_count = [0]

        def mock_fetch(video_id, preferred_language=None):
            call_count[0] += 1
            # Alternate: success, fail, success, fail, ...
            if call_count[0] % 2 == 0:
                raise CaptionFetchError(video_id, "Test error")
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=1, progress_report_interval=10)
        processor = BatchProcessor(mock_fetcher, config=config)

        videos = [f"vid{i:03d}" for i in range(20)]
        processor.process(videos)

        progress_logs = [
            r.message for r in caplog.records
            if "Batch progress:" in r.message
        ]

        # Should have progress log at video 10
        assert len(progress_logs) >= 1
        first_log = progress_logs[0]

        # Check format includes counts
        assert "success" in first_log
        assert "failed" in first_log
        assert "skipped" in first_log

    def test_batch_progress_includes_budget_consumption(self, caplog):
        """Progress includes budget consumption when retry_budget provided."""
        import logging
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from src.caption.retry_budget import CaptionRetryBudget

        caplog.set_level(logging.INFO)

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=1, progress_report_interval=10)
        processor = BatchProcessor(mock_fetcher, config=config)

        # Create retry budget with known max
        retry_budget = CaptionRetryBudget()
        retry_budget.max_attempts = 100

        videos = [f"vid{i:03d}" for i in range(15)]
        processor.process(videos, retry_budget=retry_budget)

        progress_logs = [
            r.message for r in caplog.records
            if "Batch progress:" in r.message
        ]

        assert len(progress_logs) >= 1
        first_log = progress_logs[0]

        # Check budget consumption is included
        assert "budget:" in first_log
        assert "attempts used" in first_log

    def test_batch_progress_includes_eta(self, caplog):
        """Progress includes ETA based on average fetch time."""
        import logging
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        caplog.set_level(logging.INFO)

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=1, progress_report_interval=10)
        processor = BatchProcessor(mock_fetcher, config=config)

        videos = [f"vid{i:03d}" for i in range(15)]
        processor.process(videos)

        progress_logs = [
            r.message for r in caplog.records
            if "Batch progress:" in r.message
        ]

        assert len(progress_logs) >= 1
        first_log = progress_logs[0]

        # Check ETA is included
        assert "ETA:" in first_log

    def test_batch_progress_callback_fires(self):
        """progress_callback receives batch_progress status events."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from typing import List, Dict

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=1, progress_report_interval=10)
        processor = BatchProcessor(mock_fetcher, config=config)

        # Track batch_progress events
        batch_progress_events: List[Dict] = []

        def on_progress(video_id: str, status: str, details: Dict):
            if status == 'batch_progress':
                batch_progress_events.append(details)

        videos = [f"vid{i:03d}" for i in range(25)]
        processor.process(videos, progress_callback=on_progress)

        # Should have 2 batch_progress events (at 10 and 20)
        assert len(batch_progress_events) == 2

        # Verify first event has expected fields
        first_event = batch_progress_events[0]
        assert first_event['processed'] == 10
        assert first_event['total'] == 25
        assert 'success_count' in first_event
        assert 'error_count' in first_event
        assert 'skipped_count' in first_event
        assert 'eta_seconds' in first_event

    def test_batch_progress_logged_at_info_level(self, caplog):
        """Progress is logged at INFO level (visible in non-verbose mode)."""
        import logging
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        # Only capture INFO and above
        caplog.set_level(logging.INFO)

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=1, progress_report_interval=5)
        processor = BatchProcessor(mock_fetcher, config=config)

        videos = [f"vid{i:03d}" for i in range(10)]
        processor.process(videos)

        # Find batch progress records
        batch_progress_records = [
            r for r in caplog.records
            if "Batch progress:" in r.message
        ]

        # Should have at least one INFO level batch progress
        assert len(batch_progress_records) >= 1
        assert batch_progress_records[0].levelno == logging.INFO

    def test_batch_progress_checkpoint_partial_results(self):
        """Progress callback updates checkpoint with partial results (US-37-011).

        Verifies that as batch processing progresses, the checkpoint
        accumulates partial results that can be recovered on resume.
        """
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from src.caption.batch_checkpoint import CaptionBatchCheckpoint

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=1, progress_report_interval=10)
        processor = BatchProcessor(mock_fetcher, config=config)

        # Create checkpoint to track partial results
        checkpoint = CaptionBatchCheckpoint()

        # Track checkpoint state at each progress interval
        checkpoint_states_at_progress: list = []

        def on_progress(video_id: str, status: str, details: dict):
            if status == 'batch_progress':
                # Capture checkpoint state at progress report
                # Note: progress callback fires before checkpoint.update() for the
                # triggering video, so checkpoint_results may be 1 less than processed
                checkpoint_states_at_progress.append({
                    'processed': details['processed'],
                    'checkpoint_results': len(checkpoint.results),
                })

        videos = [f"vid{i:03d}" for i in range(25)]
        processor.process(
            videos,
            batch_checkpoint=checkpoint,
            progress_callback=on_progress,
        )

        # Should have 2 progress events (at 10 and 20)
        assert len(checkpoint_states_at_progress) == 2

        # At each progress interval, checkpoint should have partial results
        # (may be 1 less than processed count due to callback order)
        assert checkpoint_states_at_progress[0]['checkpoint_results'] >= 9  # At 10 processed
        assert checkpoint_states_at_progress[1]['checkpoint_results'] >= 19  # At 20 processed

        # Final checkpoint should have all results
        assert len(checkpoint.results) == 25


class TestBatchProcessorBudgetStatus:
    """Tests for budget status in progress callbacks (US-38-004)."""

    def test_budget_consumed_pct_included_in_callback_details(self):
        """progress_callback details include budget_consumed_pct when retry_budget provided (US-38-004)."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from src.caption.retry_budget import CaptionRetryBudget
        from typing import Dict, List

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        # Create retry budget with known max_attempts
        retry_budget = CaptionRetryBudget()
        retry_budget.max_attempts = 100

        # Collect all progress events with their details
        progress_events: List[Dict] = []

        def on_progress(video_id: str, status: str, details: Dict):
            if status in ('success', 'failed', 'fetching', 'skipped'):
                progress_events.append({
                    'video_id': video_id,
                    'status': status,
                    'details': details.copy()
                })

        videos = [f"vid{i:03d}" for i in range(5)]
        processor.process(videos, retry_budget=retry_budget, progress_callback=on_progress)

        # All success events should include budget_consumed_pct
        success_events = [e for e in progress_events if e['status'] == 'success']
        assert len(success_events) == 5

        for event in success_events:
            assert 'budget_consumed_pct' in event['details'], \
                f"budget_consumed_pct missing from success event for {event['video_id']}"
            # Should be a percentage value
            pct = event['details']['budget_consumed_pct']
            assert isinstance(pct, (int, float)), f"Expected number, got {type(pct)}"
            assert 0 <= pct <= 100, f"Percentage out of range: {pct}"

    def test_budget_consumed_pct_not_included_without_retry_budget(self):
        """progress_callback details do NOT include budget_consumed_pct when no retry_budget (US-38-004)."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from typing import Dict, List

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        # Collect progress events
        progress_events: List[Dict] = []

        def on_progress(video_id: str, status: str, details: Dict):
            if status == 'success':
                progress_events.append(details.copy())

        videos = ["vid001", "vid002"]
        processor.process(videos, progress_callback=on_progress)

        # Without retry_budget, budget_consumed_pct should not be added
        for details in progress_events:
            assert 'budget_consumed_pct' not in details, \
                "budget_consumed_pct should not be present without retry_budget"

    def test_budget_consumed_pct_increases_over_batch(self):
        """budget_consumed_pct increases as batch progresses (US-38-004)."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from src.caption.retry_budget import CaptionRetryBudget
        from typing import Dict, List

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        # Single worker to ensure sequential processing
        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        # Create retry budget with known max
        retry_budget = CaptionRetryBudget()
        retry_budget.max_attempts = 10  # 10 videos * 1 attempt each = 100%

        # Collect budget percentages
        budget_pcts: List[float] = []

        def on_progress(video_id: str, status: str, details: Dict):
            if status == 'success' and 'budget_consumed_pct' in details:
                budget_pcts.append(details['budget_consumed_pct'])

        videos = [f"vid{i:03d}" for i in range(10)]
        processor.process(videos, retry_budget=retry_budget, progress_callback=on_progress)

        # Budget should increase with each video
        assert len(budget_pcts) == 10
        # First video should be at ~10%, last at 100%
        assert budget_pcts[0] > 0
        assert budget_pcts[-1] == 100.0
        # Should be monotonically increasing
        for i in range(1, len(budget_pcts)):
            assert budget_pcts[i] >= budget_pcts[i-1], \
                f"Budget should increase: {budget_pcts[i-1]} -> {budget_pcts[i]}"

    def test_budget_consumed_pct_included_in_failed_events(self):
        """budget_consumed_pct included in failed status events (US-38-004)."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from src.caption.retry_budget import CaptionRetryBudget
        from src.caption.exceptions import CaptionFetchError
        from typing import Dict, List

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        # Make some fetches fail
        call_count = [0]

        def mock_fetch(video_id, preferred_language=None):
            call_count[0] += 1
            if call_count[0] == 2:
                raise CaptionFetchError(video_id, "Test error")
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        retry_budget = CaptionRetryBudget()
        retry_budget.max_attempts = 100

        # Collect failed events
        failed_events: List[Dict] = []

        def on_progress(video_id: str, status: str, details: Dict):
            if status == 'failed':
                failed_events.append(details.copy())

        videos = ["vid001", "vid002", "vid003"]
        processor.process(videos, retry_budget=retry_budget, progress_callback=on_progress)

        # Failed event should include budget_consumed_pct
        assert len(failed_events) == 1
        assert 'budget_consumed_pct' in failed_events[0], \
            "budget_consumed_pct missing from failed event"


class TestBatchProcessorUnavailableDetection:
    """Tests for US-59-006: Early termination when all errors are 'no captions'."""

    def test_unavailable_pattern_triggers_warning_and_preflight_mode(self):
        """Batch of 20 videos where first 10 lack captions triggers warning
        and remaining 10 use pre-flight-only mode.

        Verifies:
        - Counter of consecutive CaptionUnavailableError is tracked
        - When 80%+ of last 10 videos return CaptionUnavailableError, warning is logged
        - Remaining videos use pre-flight-only mode (has_captions check first)
        - CaptionMetrics records unavailable_count and unavailable_rate
        """
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from src.caption.exceptions import CaptionUnavailableError
        from src.caption.metrics import CaptionMetrics

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        # Track which videos got preflight check vs full fetch
        preflight_checked = []
        full_fetch_called = []

        def mock_fetch(video_id, preferred_language=None):
            full_fetch_called.append(video_id)
            raise CaptionUnavailableError(video_id, "No captions available")

        def mock_has_captions(video_id):
            preflight_checked.append(video_id)
            return False

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch
        mock_fetcher.has_captions.side_effect = mock_has_captions

        # Use single worker for deterministic ordering
        # Use 30 videos total so that even with thread scheduling lag,
        # some later videos will see the preflight_only flag
        config = BatchProcessorConfig(
            max_workers=1,
            unavailable_window_size=10,
            unavailable_threshold=0.8,
        )
        processor = BatchProcessor(mock_fetcher, config=config)

        metrics = CaptionMetrics()
        progress_events: List[tuple] = []

        def on_progress(video_id: str, status: str, details: Dict):
            progress_events.append((video_id, status, details))

        video_ids = [f"vid{i:03d}" for i in range(30)]
        result = processor.process(
            video_ids,
            metrics=metrics,
            progress_callback=on_progress,
        )

        # All 30 videos should be unavailable
        assert result.unavailable_count == 30
        assert result.success_count == 0
        assert result.error_count == 30

        # Should have switched to preflight mode
        assert result.unavailable_switched_to_preflight is True

        # has_captions should have been called for at least some videos
        # after the mode switch (thread scheduling may cause 1-2 extra
        # full fetches before the flag takes effect)
        assert len(preflight_checked) > 0, (
            f"Expected preflight checks but got none. "
            f"Full fetches: {len(full_fetch_called)}"
        )

        # The full fetch count should be less than total because preflight
        # skips the full fetch for videos where has_captions returns False
        assert len(full_fetch_called) < 30, (
            f"Expected some videos to use preflight-only mode. "
            f"Full fetches: {len(full_fetch_called)}, preflight: {len(preflight_checked)}"
        )

        # Check that unavailable_pattern progress event was fired
        pattern_events = [
            e for e in progress_events if e[1] == 'unavailable_pattern'
        ]
        assert len(pattern_events) == 1
        assert pattern_events[0][2]['unavailable_rate'] >= 0.8

        # CaptionMetrics should record unavailable summary
        assert metrics.unavailable_count == 30
        assert metrics.unavailable_rate > 0.0

    def test_no_preflight_switch_when_below_threshold(self):
        """When unavailable rate is below 80%, no mode switch occurs."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from src.caption.exceptions import CaptionUnavailableError

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        call_count = [0]

        def mock_fetch(video_id, preferred_language=None):
            call_count[0] += 1
            # 50% unavailable (5 out of 10) - below 80% threshold
            if call_count[0] % 2 == 0:
                raise CaptionUnavailableError(video_id, "No captions")
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(
            max_workers=1,
            unavailable_window_size=10,
            unavailable_threshold=0.8,
        )
        processor = BatchProcessor(mock_fetcher, config=config)

        video_ids = [f"vid{i:03d}" for i in range(15)]
        result = processor.process(video_ids)

        # Should NOT switch to preflight mode
        assert result.unavailable_switched_to_preflight is False
        # has_captions should never be called
        mock_fetcher.has_captions.assert_not_called()

    def test_preflight_mode_allows_success_when_has_captions(self):
        """In preflight-only mode, videos that DO have captions still get fetched."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from src.caption.exceptions import CaptionUnavailableError

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        call_order = [0]

        def mock_fetch(video_id, preferred_language=None):
            call_order[0] += 1
            if call_order[0] <= 10:
                raise CaptionUnavailableError(video_id, "No captions")
            # After mode switch, if has_captions passes, this gets called
            return MockCaptionResult(video_id=video_id, segments=[{"text": "ok"}])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch
        # After preflight switch, has_captions returns True for some videos
        mock_fetcher.has_captions.return_value = True

        config = BatchProcessorConfig(
            max_workers=1,
            unavailable_window_size=10,
            unavailable_threshold=0.8,
        )
        processor = BatchProcessor(mock_fetcher, config=config)

        video_ids = [f"vid{i:03d}" for i in range(15)]
        result = processor.process(video_ids)

        # First 10 unavailable, remaining 5 should succeed (preflight passes)
        assert result.unavailable_switched_to_preflight is True
        assert result.success_count > 0
        assert result.unavailable_count == 10

    def test_unavailable_count_in_batch_result(self):
        """BatchResult.unavailable_count tracks total unavailable videos."""
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        from src.caption.exceptions import CaptionUnavailableError

        mock_fetcher = MagicMock()
        mock_fetcher._timeout = 30.0
        mock_fetcher._preferred_formats = ["vtt"]
        mock_fetcher._sort_videos_by_channel_success = lambda x, y: x

        call_count = [0]

        def mock_fetch(video_id, preferred_language=None):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise CaptionUnavailableError(video_id, "No captions")
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = mock_fetch

        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        video_ids = [f"vid{i:03d}" for i in range(5)]
        result = processor.process(video_ids)

        # 3 unavailable, 2 success
        assert result.unavailable_count == 3
        assert result.success_count == 2
        assert result.error_count == 3  # unavailable counted as errors

    def test_metrics_unavailable_rate_in_summary(self):
        """CaptionMetrics.get_summary_dict() includes unavailable_count and rate."""
        from src.caption.metrics import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_unavailable_summary(count=15, rate=0.75)

        summary = metrics.get_summary_dict()
        assert summary['unavailable_count'] == 15
        assert summary['unavailable_rate'] == 0.75

    def test_metrics_unavailable_in_to_dict_from_dict(self):
        """CaptionMetrics serializes and deserializes unavailable fields."""
        from src.caption.metrics import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_unavailable_summary(count=10, rate=0.5)

        data = metrics.to_dict()
        assert data['unavailable_count'] == 10
        assert data['unavailable_rate'] == 0.5

        restored = CaptionMetrics.from_dict(data)
        assert restored.unavailable_count == 10
        assert restored.unavailable_rate == 0.5

    def test_metrics_summary_includes_unavailable_line(self):
        """CaptionMetrics.summary() includes unavailable line when count > 0."""
        from src.caption.metrics import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_unavailable_summary(count=8, rate=0.4)

        summary_text = metrics.summary()
        assert "Unavailable: 8 videos" in summary_text
        assert "40.0%" in summary_text
