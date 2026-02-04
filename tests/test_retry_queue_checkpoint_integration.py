"""Tests for retry queue persistence across pipeline restarts.

Story: US-51-010 - Add download retry queue persistence across pipeline restarts

Tests verify:
- Retry queue is saved to checkpoint data on stage completion
- Retry queue is loaded from checkpoint on resume
- max_retries_per_video is enforced on restore
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from src.downloader.retry_queue import RetryQueue, BatchRetryConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_state():
    """Mock PipelineState."""
    state = MagicMock()
    state.matches = [MagicMock(video_file='vid1'), MagicMock(video_file='vid2')]
    state.downloaded_segments = []
    return state


@pytest.fixture
def mock_checkpoint():
    """Mock CheckpointManager with download_segments data."""
    checkpoint = MagicMock()
    checkpoint.data = MagicMock()
    checkpoint.data.download_segments = {}
    return checkpoint


@pytest.fixture
def mock_config():
    """Mock Config."""
    config = MagicMock()
    config.downloaded_videos_dir = 'C:/fake/output'
    config.download = MagicMock()
    config.download.segment_buffer = 2.0
    config.download.batch_retry = BatchRetryConfig(
        enabled=True, delay_seconds=0.01, max_passes=2,
        max_retries_per_video=3, jitter_factor=0.0
    )
    return config


# =============================================================================
# Test: Retry queue is persisted to checkpoint on stage completion
# =============================================================================

@pytest.mark.fast
class TestRetryQueueSavedToCheckpoint:
    """Verify retry queue data is included in checkpoint on save."""

    def test_retry_queue_saved_in_checkpoint_data(self):
        """RetryQueue.to_checkpoint_dict() produces valid checkpoint data."""
        config = BatchRetryConfig(
            enabled=True, delay_seconds=0.01, max_passes=2,
            max_retries_per_video=3, jitter_factor=0.0
        )
        queue = RetryQueue(config)

        # Simulate failed downloads
        queue.add("vid1_10_20", "segment", "segment", "403 Forbidden",
                   error_category='bot_detection', escalation_tier=2)
        queue.add("vid2_30_40", "segment", "segment", "Connection timeout",
                   error_category='timeout', escalation_tier=1)

        # Serialize
        checkpoint_data = queue.to_checkpoint_dict()

        # Verify structure
        assert 'items' in checkpoint_data
        assert len(checkpoint_data['items']) == 2
        assert checkpoint_data['current_pass'] == 0

        # Verify each item has required fields from AC
        for item in checkpoint_data['items']:
            assert 'video_id' in item
            assert 'last_tier_attempted' in item  # AC: last_tier_attempted
            assert 'failure_reason' in item        # AC: failure_reason
            assert 'timestamp' in item             # AC: timestamp
            assert 'retry_count' in item

    def test_failed_ids_preserved_in_checkpoint(self):
        """Permanently failed video IDs are saved in checkpoint."""
        config = BatchRetryConfig(
            enabled=True, delay_seconds=0.01, max_passes=2,
            max_retries_per_video=2, jitter_factor=0.0
        )
        queue = RetryQueue(config)

        queue.add("vid1", "kw", "short", "Error")
        queue.items["vid1"].retry_count = 2  # At max
        queue.current_pass = 1
        queue.finish_retry_pass()

        checkpoint = queue.to_checkpoint_dict()

        assert 'vid1' in checkpoint['failed_ids']
        assert len(checkpoint['items']) == 0  # Removed from pending


# =============================================================================
# Test: Retry queue is loaded and processed on resume
# =============================================================================

@pytest.mark.fast
class TestRetryQueueLoadedOnResume:
    """Verify retry queue is restored from checkpoint on pipeline resume."""

    def test_retry_queue_loaded_from_checkpoint(self):
        """from_checkpoint_dict() restores pending items for retry."""
        config = BatchRetryConfig(
            enabled=True, delay_seconds=0.01, max_passes=2,
            max_retries_per_video=3, jitter_factor=0.0
        )
        queue = RetryQueue(config)

        checkpoint = {
            'items': [
                {
                    'video_id': 'vid1_10_20',
                    'keyword': 'segment',
                    'tier': 'segment',
                    'error_message': '403 Forbidden',
                    'failure_reason': '403 Forbidden',
                    'retry_count': 1,
                    'error_category': 'bot_detection',
                    'escalation_tier': 2,
                    'last_tier_attempted': 2,
                    'timestamp': 1706000000.0,
                },
            ],
            'current_pass': 0,
            'completed_ids': [],
            'failed_ids': [],
            'total_added': 1,
            'total_retried': 0,
        }

        queue.from_checkpoint_dict(checkpoint)

        assert len(queue.items) == 1
        assert 'vid1_10_20' in queue.items
        item = queue.items['vid1_10_20']
        assert item.retry_count == 1
        assert item.escalation_tier == 2
        assert item.error_message == '403 Forbidden'
        assert item.added_at == 1706000000.0

    def test_max_retries_per_video_enforced_on_restore(self):
        """Videos exceeding max_retries_per_video are skipped on restore."""
        config = BatchRetryConfig(
            enabled=True, delay_seconds=0.01, max_passes=2,
            max_retries_per_video=3, jitter_factor=0.0
        )
        queue = RetryQueue(config)

        checkpoint = {
            'items': [
                {'video_id': 'vid_ok', 'keyword': 'kw', 'tier': 'short',
                 'error_message': 'E', 'retry_count': 2},
                {'video_id': 'vid_exceeded', 'keyword': 'kw', 'tier': 'short',
                 'error_message': 'E', 'retry_count': 3},
            ],
            'current_pass': 0,
            'completed_ids': [],
            'failed_ids': [],
            'total_added': 2,
            'total_retried': 0,
        }

        queue.from_checkpoint_dict(checkpoint)

        assert 'vid_ok' in queue.items
        assert 'vid_exceeded' not in queue.items
        assert 'vid_exceeded' in queue._failed_ids

    def test_restore_method_stores_retry_queue_data(self, mock_config):
        """DownloadSegments.restore() stores retry queue data on state."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()

        # Use a simple namespace instead of MagicMock so setattr works normally
        class SimpleState:
            matches = []
            downloaded_segments = []

        state = SimpleState()

        # Set up checkpoint with retry queue data
        checkpoint = MagicMock()
        checkpoint.data = MagicMock()
        checkpoint.data.download_segments = {
            'retry_queue': {
                'items': [
                    {'video_id': 'vid1_10_20', 'keyword': 'segment',
                     'tier': 'segment', 'error_message': '403',
                     'retry_count': 1, 'escalation_tier': 2},
                ],
                'current_pass': 0,
                'completed_ids': [],
                'failed_ids': [],
                'total_added': 1,
                'total_retried': 0,
            }
        }

        # Mock output_dir to not exist (skip file scan)
        with patch.object(Path, 'exists', return_value=False):
            result = stage.restore(state, checkpoint, mock_config)

        assert result is True
        assert hasattr(state, '_restored_retry_queue')
        assert len(state._restored_retry_queue['items']) == 1

    def test_restore_method_no_retry_queue_in_checkpoint(self, mock_config):
        """DownloadSegments.restore() handles missing retry_queue gracefully."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()

        class SimpleState:
            matches = []
            downloaded_segments = []

        state = SimpleState()

        checkpoint = MagicMock()
        checkpoint.data = MagicMock()
        checkpoint.data.download_segments = {
            'segment_count': 10,
            # No 'retry_queue' key
        }

        with patch.object(Path, 'exists', return_value=False):
            result = stage.restore(state, checkpoint, mock_config)

        assert result is True
        # Should not set _restored_retry_queue
        assert not hasattr(state, '_restored_retry_queue')
