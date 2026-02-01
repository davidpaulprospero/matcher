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
