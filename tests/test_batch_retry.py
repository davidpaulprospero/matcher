"""Tests for batch-level retry queue for rate-limited videos.

Tests US-007 acceptance criteria:
1. New RetryQueue class collects rate-limited video IDs with their keywords
2. After batch completes, retry queue is processed with backoff
3. New config option download.batch_retry.enabled (default: true)
4. New config option download.batch_retry.delay_seconds (default: 120)
5. Maximum 2 batch retry passes per download session
6. Tests verify batch retry with multiple failed videos
"""

import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from src.downloader.retry_queue import RetryQueue, BatchRetryConfig, RetryItem


class TestBatchRetryConfig:
    """Test BatchRetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BatchRetryConfig()
        assert config.enabled is True
        assert config.delay_seconds == 120.0
        assert config.max_passes == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BatchRetryConfig(
            enabled=False,
            delay_seconds=60.0,
            max_passes=3
        )
        assert config.enabled is False
        assert config.delay_seconds == 60.0
        assert config.max_passes == 3


class TestRetryItem:
    """Test RetryItem dataclass."""

    def test_default_values(self):
        """Test RetryItem default values."""
        item = RetryItem(
            video_id="abc123",
            keyword="test keyword",
            tier="short",
            error_message="Rate limited"
        )
        assert item.video_id == "abc123"
        assert item.keyword == "test keyword"
        assert item.tier == "short"
        assert item.error_message == "Rate limited"
        assert item.retry_count == 0
        assert item.added_at > 0  # Should be set to current time

    def test_custom_retry_count(self):
        """Test RetryItem with custom retry count."""
        item = RetryItem(
            video_id="abc123",
            keyword="test",
            tier="short",
            error_message="Error",
            retry_count=2
        )
        assert item.retry_count == 2


class TestRetryQueueInitialization:
    """Test RetryQueue initialization."""

    def test_default_config(self):
        """Test initialization with default config."""
        queue = RetryQueue()
        assert queue.config.enabled is True
        assert queue.config.delay_seconds == 120.0
        assert queue.config.max_passes == 2
        assert len(queue.items) == 0

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = BatchRetryConfig(enabled=False, delay_seconds=60.0, max_passes=3)
        queue = RetryQueue(config)
        assert queue.config.enabled is False
        assert queue.config.delay_seconds == 60.0
        assert queue.config.max_passes == 3

    def test_disabled_queue(self):
        """Test disabled queue."""
        queue = RetryQueue(BatchRetryConfig(enabled=False))
        assert queue.is_enabled is False


class TestRetryQueueAdd:
    """Test adding items to retry queue."""

    def test_add_single_item(self):
        """Test adding a single item."""
        queue = RetryQueue()
        result = queue.add("video1", "keyword1", "short", "Rate limited")

        assert result is True
        assert len(queue.items) == 1
        assert "video1" in queue.items
        assert queue.items["video1"].keyword == "keyword1"
        assert queue.items["video1"].tier == "short"

    def test_add_multiple_items(self):
        """Test adding multiple items."""
        queue = RetryQueue()
        queue.add("video1", "keyword1", "short", "Error 1")
        queue.add("video2", "keyword2", "medium", "Error 2")
        queue.add("video3", "keyword1", "long", "Error 3")

        assert len(queue.items) == 3

    def test_add_duplicate_item(self):
        """Test adding duplicate item updates error but doesn't duplicate."""
        queue = RetryQueue()
        queue.add("video1", "keyword1", "short", "Error 1")
        result = queue.add("video1", "keyword1", "short", "Error 2 - updated")

        assert result is False
        assert len(queue.items) == 1
        assert queue.items["video1"].error_message == "Error 2 - updated"

    def test_add_disabled(self):
        """Test adding to disabled queue returns False."""
        queue = RetryQueue(BatchRetryConfig(enabled=False))
        result = queue.add("video1", "keyword1", "short", "Error")

        assert result is False
        assert len(queue.items) == 0

    def test_add_already_completed(self):
        """Test adding already completed item is skipped."""
        queue = RetryQueue()
        queue.add("video1", "keyword1", "short", "Error")
        queue.mark_success("video1")

        # Try to re-add
        result = queue.add("video1", "keyword1", "short", "New error")
        assert result is False


class TestRetryQueueState:
    """Test retry queue state management."""

    def test_has_pending(self):
        """Test has_pending method."""
        queue = RetryQueue()
        assert queue.has_pending() is False

        queue.add("video1", "keyword1", "short", "Error")
        assert queue.has_pending() is True

    def test_has_pending_after_max_passes(self):
        """Test has_pending is False after max passes."""
        queue = RetryQueue(BatchRetryConfig(max_passes=1))
        queue.add("video1", "keyword1", "short", "Error")

        assert queue.has_pending() is True
        assert queue.can_retry is True

        # Simulate one pass
        queue.current_pass = 1
        assert queue.can_retry is False
        assert queue.has_pending() is False

    def test_get_pending_items(self):
        """Test getting pending items."""
        queue = RetryQueue()
        queue.add("video1", "keyword1", "short", "Error 1")
        queue.add("video2", "keyword2", "medium", "Error 2")

        items = queue.get_pending_items()
        assert len(items) == 2
        assert all(isinstance(item, RetryItem) for item in items)


class TestRetryQueueMarking:
    """Test marking items as success/failed."""

    def test_mark_success(self):
        """Test marking item as success removes it from queue."""
        queue = RetryQueue()
        queue.add("video1", "keyword1", "short", "Error")
        queue.mark_success("video1")

        assert len(queue.items) == 0
        assert "video1" in queue._completed_ids

    def test_mark_success_nonexistent(self):
        """Test marking nonexistent item is a no-op."""
        queue = RetryQueue()
        queue.mark_success("video1")  # Should not raise
        assert len(queue._completed_ids) == 0

    def test_mark_failed(self):
        """Test marking item as failed increments retry count."""
        queue = RetryQueue()
        queue.add("video1", "keyword1", "short", "Error")
        queue.mark_failed("video1")

        assert queue.items["video1"].retry_count == 1


class TestRetryQueuePasses:
    """Test retry pass management."""

    def test_start_retry_pass(self):
        """Test starting a retry pass."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))  # Fast for testing
        queue.add("video1", "keyword1", "short", "Error")

        pass_num = queue.start_retry_pass()
        assert pass_num == 1
        assert queue.current_pass == 1

    def test_start_retry_pass_disabled(self):
        """Test start_retry_pass returns 0 when disabled."""
        queue = RetryQueue(BatchRetryConfig(enabled=False))
        queue.items["video1"] = RetryItem("video1", "kw", "short", "err")

        pass_num = queue.start_retry_pass()
        assert pass_num == 0

    def test_start_retry_pass_empty_queue(self):
        """Test start_retry_pass returns 0 when queue is empty."""
        queue = RetryQueue()
        pass_num = queue.start_retry_pass()
        assert pass_num == 0

    def test_finish_retry_pass_moves_to_failed(self):
        """Test finish_retry_pass moves exhausted items to failed."""
        queue = RetryQueue(BatchRetryConfig(max_passes=1))
        queue.add("video1", "keyword1", "short", "Error")
        queue.current_pass = 1  # At max passes

        queue.finish_retry_pass()

        assert len(queue.items) == 0
        assert "video1" in queue._failed_ids


class TestRetryQueueStats:
    """Test retry queue statistics."""

    def test_get_stats_empty(self):
        """Test stats on empty queue."""
        queue = RetryQueue()
        stats = queue.get_stats()

        assert stats['enabled'] is True
        assert stats['pending'] == 0
        assert stats['completed'] == 0
        assert stats['failed'] == 0
        assert stats['current_pass'] == 0

    def test_get_stats_with_items(self):
        """Test stats with items in queue."""
        queue = RetryQueue()
        queue.add("video1", "keyword1", "short", "Error")
        queue.add("video2", "keyword2", "medium", "Error")
        queue.mark_success("video1")

        stats = queue.get_stats()

        assert stats['pending'] == 1
        assert stats['completed'] == 1
        assert stats['total_added'] == 2


class TestRetryQueueCheckpoint:
    """Test checkpoint persistence."""

    def test_to_checkpoint_dict(self):
        """Test serializing to checkpoint dict."""
        queue = RetryQueue()
        queue.add("video1", "keyword1", "short", "Error 1")
        queue.add("video2", "keyword2", "medium", "Error 2")
        queue.current_pass = 1

        data = queue.to_checkpoint_dict()

        assert len(data['items']) == 2
        assert data['current_pass'] == 1
        assert 'completed_ids' in data
        assert 'failed_ids' in data

    def test_from_checkpoint_dict(self):
        """Test restoring from checkpoint dict."""
        queue = RetryQueue()
        data = {
            'items': [
                {'video_id': 'video1', 'keyword': 'kw1', 'tier': 'short', 'error_message': 'err', 'retry_count': 1}
            ],
            'current_pass': 1,
            'completed_ids': ['video2'],
            'failed_ids': ['video3'],
            'total_added': 3,
            'total_retried': 1,
        }

        queue.from_checkpoint_dict(data)

        assert len(queue.items) == 1
        assert queue.items['video1'].retry_count == 1
        assert queue.current_pass == 1
        assert 'video2' in queue._completed_ids
        assert 'video3' in queue._failed_ids

    def test_from_checkpoint_dict_empty(self):
        """Test restoring from empty/None checkpoint."""
        queue = RetryQueue()
        queue.from_checkpoint_dict(None)
        queue.from_checkpoint_dict({})

        assert len(queue.items) == 0

    def test_roundtrip_checkpoint(self):
        """Test save and restore roundtrip."""
        queue = RetryQueue()
        queue.add("video1", "keyword1", "short", "Error")
        queue.mark_success("video1")
        queue.add("video2", "keyword2", "medium", "Error")
        queue.current_pass = 1

        data = queue.to_checkpoint_dict()

        new_queue = RetryQueue()
        new_queue.from_checkpoint_dict(data)

        assert len(new_queue.items) == 1
        assert "video2" in new_queue.items


class TestRetryQueueClear:
    """Test clearing the queue."""

    def test_clear(self):
        """Test clearing all state."""
        queue = RetryQueue()
        queue.add("video1", "keyword1", "short", "Error")
        queue.mark_success("video1")
        queue.current_pass = 1

        queue.clear()

        assert len(queue.items) == 0
        assert len(queue._completed_ids) == 0
        assert queue.current_pass == 0


class TestVideoDownloaderBatchRetryIntegration:
    """Integration tests for batch retry in VideoDownloader."""

    def test_retry_queue_initialized(self):
        """Test retry queue is initialized in VideoDownloader."""
        from src.downloader.core import VideoDownloader
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.download.batch_retry = MagicMock()
        mock_config.download.batch_retry.enabled = True
        mock_config.download.batch_retry.delay_seconds = 60.0
        mock_config.download.batch_retry.max_passes = 2
        mock_config.download.circuit_breaker = MagicMock()
        mock_config.download.circuit_breaker.enabled = False
        mock_config.download.speed_tracking = MagicMock()
        mock_config.download.speed_tracking.enabled = False
        mock_config.download.cookie_rotation = None
        mock_config.download.vpn = None
        mock_config.download.cookies_from_browser = ''
        mock_config.download.cookies_path = ''
        mock_config.cache_dir = '/tmp/test_cache'
        mock_config.downloaded_videos_dir = '/tmp/test_videos'

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                downloader = VideoDownloader(mock_config)

        assert hasattr(downloader, 'retry_queue')
        assert downloader.retry_queue.is_enabled is True

    def test_retry_queue_disabled_when_config_disabled(self):
        """Test retry queue is disabled when config says so."""
        from src.downloader.core import VideoDownloader
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.download.batch_retry = MagicMock()
        mock_config.download.batch_retry.enabled = False  # Disabled
        mock_config.download.circuit_breaker = MagicMock()
        mock_config.download.circuit_breaker.enabled = False
        mock_config.download.speed_tracking = MagicMock()
        mock_config.download.speed_tracking.enabled = False
        mock_config.download.cookie_rotation = None
        mock_config.download.vpn = None
        mock_config.download.cookies_from_browser = ''
        mock_config.download.cookies_path = ''
        mock_config.cache_dir = '/tmp/test_cache'
        mock_config.downloaded_videos_dir = '/tmp/test_videos'

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                downloader = VideoDownloader(mock_config)

        assert downloader.retry_queue.is_enabled is False


class TestBatchRetryWithMultipleFailedVideos:
    """Test batch retry behavior with multiple failed videos."""

    def test_multiple_videos_queued_and_retried(self):
        """Test that multiple failed videos are queued and retried together."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01, max_passes=2))

        # Simulate multiple rate-limited videos
        queue.add("kw1|short", "keyword1", "short", "Rate limited")
        queue.add("kw2|medium", "keyword2", "medium", "Rate limited")
        queue.add("kw3|short", "keyword3", "short", "Rate limited")

        assert len(queue.items) == 3
        assert queue.has_pending() is True

        # First retry pass
        pass_num = queue.start_retry_pass()
        assert pass_num == 1

        # Simulate: kw1 succeeds, kw2 and kw3 still fail
        queue.mark_success("kw1|short")
        queue.mark_failed("kw2|medium")
        queue.mark_failed("kw3|short")

        queue.finish_retry_pass()

        assert len(queue.items) == 2  # kw2 and kw3 still pending
        assert len(queue._completed_ids) == 1  # kw1 succeeded

        # Second retry pass
        pass_num = queue.start_retry_pass()
        assert pass_num == 2

        # Simulate: kw2 succeeds, kw3 still fails
        queue.mark_success("kw2|medium")
        queue.mark_failed("kw3|short")

        queue.finish_retry_pass()

        # After max passes, remaining should be moved to failed
        assert len(queue.items) == 0
        assert len(queue._completed_ids) == 2  # kw1 and kw2
        assert len(queue._failed_ids) == 1  # kw3

        # No more retries allowed
        assert queue.has_pending() is False

    def test_batch_retry_respects_max_passes(self):
        """Test that batch retry stops after max_passes."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01, max_passes=1))

        queue.add("video1", "keyword1", "short", "Error")

        # First pass
        queue.start_retry_pass()
        queue.mark_failed("video1")
        queue.finish_retry_pass()

        # Should be moved to failed after max_passes
        assert len(queue.items) == 0
        assert "video1" in queue._failed_ids
        assert queue.has_pending() is False


class TestConfigYamlIntegration:
    """Test that config.yaml settings are properly loaded."""

    def test_config_section_exists_in_download_config(self):
        """Test BatchRetryConfig is part of DownloadConfig."""
        from src.config.sections.download import DownloadConfig, BatchRetryConfig

        config = DownloadConfig()
        assert hasattr(config, 'batch_retry')
        assert isinstance(config.batch_retry, BatchRetryConfig)

    def test_config_dict_conversion(self):
        """Test that dict is converted to BatchRetryConfig in __post_init__."""
        from src.config.sections.download import DownloadConfig

        config_dict = {
            'batch_retry': {
                'enabled': True,
                'delay_seconds': 90.0,
                'max_passes': 3
            }
        }

        # Simulate YAML loading by passing dict
        config = DownloadConfig(batch_retry=config_dict['batch_retry'])

        assert config.batch_retry.enabled is True
        assert config.batch_retry.delay_seconds == 90.0
        assert config.batch_retry.max_passes == 3
