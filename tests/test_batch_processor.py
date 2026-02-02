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
