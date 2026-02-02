"""
Tests for DownloadVideoSegmentsStage.

Tests cover:
- Stage initialization and registration
- Skipping when no matches
- Segment collection from matches
- Checkpoint progress callback
- Deduplication of segments

Created: 2026-02-02 (Sprint 34 - Pipeline Focus)
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from src.stages.download_segments import DownloadVideoSegmentsStage
from src.state import PipelineState


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config(tmp_path):
    """Create mock config for download segments stage"""
    config = MagicMock()
    config.download = MagicMock()
    config.download.segment_buffer = 5.0
    config.downloaded_videos_dir = str(tmp_path / "test_segments")
    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint manager"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    checkpoint.save_intermediate = MagicMock()
    return checkpoint


@pytest.fixture
def mock_state_empty():
    """Create state with no matches"""
    state = PipelineState()
    state.matches = []
    return state


@pytest.fixture
def mock_state_with_matches():
    """Create state with matches having video_file attribute (plain Match structure)"""
    state = PipelineState()

    # Create mock matches with video_file attribute (plain Match, no primary_match)
    # Use spec to limit attributes and configure primary_match=None to skip that path
    match1 = Mock()
    match1.primary_match = None  # Explicitly set to skip primary_match path
    match1.video_file = "abc123"
    match1.video_start = 10.0
    match1.video_end = 25.0

    match2 = Mock()
    match2.primary_match = None
    match2.video_file = "def456"
    match2.video_start = 30.0
    match2.video_end = 45.0

    # Duplicate to test deduplication
    match3 = Mock()
    match3.primary_match = None
    match3.video_file = "abc123"
    match3.video_start = 10.0
    match3.video_end = 25.0

    state.matches = [match1, match2, match3]
    return state


@pytest.fixture
def mock_state_with_primary_matches():
    """Create state with MatchResult objects having primary_match"""
    state = PipelineState()

    # Create mock MatchResult with primary_match.video_segment
    match1 = MagicMock()
    match1.primary_match = MagicMock()
    match1.primary_match.video_segment = MagicMock()
    match1.primary_match.video_segment.source_file = "vid001"
    match1.primary_match.video_segment.start_time = 5.0
    match1.primary_match.video_segment.end_time = 20.0

    match2 = MagicMock()
    match2.primary_match = MagicMock()
    match2.primary_match.video_segment = MagicMock()
    match2.primary_match.video_segment.source_file = "vid002"
    match2.primary_match.video_segment.start_time = 0.0
    match2.primary_match.video_segment.end_time = 15.0

    state.matches = [match1, match2]
    return state


@pytest.fixture
def stage():
    """Create DownloadVideoSegmentsStage instance"""
    return DownloadVideoSegmentsStage()


# ============================================================================
# Stage Registration and Initialization
# ============================================================================

class TestStageRegistration:
    """Test stage registration and basic properties"""

    def test_stage_name(self, stage):
        """Stage has correct name"""
        assert stage.name == "DOWNLOAD_SEGMENTS"

    def test_stage_description(self, stage):
        """Stage has description"""
        assert stage.description == "Download matched video segments"


# ============================================================================
# Skipping Behavior
# ============================================================================

class TestDownloadSegmentsSkipping:
    """Test stage skipping when no matches"""

    def test_download_segments_skips_when_no_matches(
        self, stage, mock_state_empty, mock_config, mock_checkpoint
    ):
        """Stage returns ok with skipped=True when no matches exist"""
        result = stage.run(mock_state_empty, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_matches'

    def test_download_segments_skips_when_matches_is_none(
        self, stage, mock_config, mock_checkpoint
    ):
        """Stage handles None matches gracefully"""
        state = PipelineState()
        state.matches = None

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True


# ============================================================================
# Segment Collection
# ============================================================================

class TestCollectMatchedSegments:
    """Test _collect_matched_segments method"""

    def test_download_segments_collects_matched_segments(
        self, stage, mock_state_with_matches
    ):
        """Segments are collected from matches with video_file attribute"""
        segments = stage._collect_matched_segments(mock_state_with_matches)

        # Should have 2 segments (3rd is duplicate)
        assert len(segments) == 2

        # Check first segment
        assert segments[0]['video_id'] == "abc123"
        assert segments[0]['start'] == 10.0
        assert segments[0]['end'] == 25.0

        # Check second segment
        assert segments[1]['video_id'] == "def456"
        assert segments[1]['start'] == 30.0
        assert segments[1]['end'] == 45.0

    def test_collects_from_primary_match_structure(
        self, stage, mock_state_with_primary_matches
    ):
        """Segments are collected from MatchResult.primary_match.video_segment"""
        segments = stage._collect_matched_segments(mock_state_with_primary_matches)

        assert len(segments) == 2
        assert segments[0]['video_id'] == "vid001"
        assert segments[0]['start'] == 5.0
        assert segments[0]['end'] == 20.0

    def test_deduplicates_segments(self, stage, mock_state_with_matches):
        """Duplicate segments are filtered out"""
        # mock_state_with_matches has 3 matches but match1 and match3 are same
        segments = stage._collect_matched_segments(mock_state_with_matches)

        assert len(segments) == 2
        video_ids = [s['video_id'] for s in segments]
        assert video_ids == ["abc123", "def456"]

    def test_skips_matches_without_video_id(self, stage):
        """Matches with empty video_file are skipped"""
        state = PipelineState()

        match1 = Mock()
        match1.primary_match = None  # Skip primary_match path
        match1.video_file = ""  # Empty
        match1.video_start = 10.0
        match1.video_end = 25.0

        match2 = Mock()
        match2.primary_match = None  # Skip primary_match path
        match2.video_file = "valid123"
        match2.video_start = 0.0
        match2.video_end = 10.0

        state.matches = [match1, match2]

        segments = stage._collect_matched_segments(state)

        assert len(segments) == 1
        assert segments[0]['video_id'] == "valid123"


# ============================================================================
# Checkpoint Progress
# ============================================================================

class TestCheckpointProgress:
    """Test checkpoint progress callback during download"""

    def test_download_segments_checkpoints_progress(self, mock_checkpoint):
        """
        Test that checkpoint_progress callback saves correct data to checkpoint.

        This tests the callback mechanism directly by simulating what run() does:
        the callback receives (current, total, downloaded) and calls
        checkpoint.save_intermediate with the expected data structure.
        """
        # Simulate the checkpoint_progress callback as defined in run()
        def checkpoint_progress(current: int, total: int, downloaded: list):
            """Save progress checkpoint during download"""
            checkpoint_data = {
                'segment_count': len(downloaded),
                'segments_completed': current,
                'segments_total': total,
                'in_progress': current < total,
            }
            mock_checkpoint.save_intermediate('DOWNLOAD_SEGMENTS', checkpoint_data)

        # Simulate download progress for 2 segments
        downloaded = []
        downloaded.append({'file': '/tmp/abc.mp4'})  # segment 1
        checkpoint_progress(1, 2, downloaded)

        downloaded.append({'file': '/tmp/def.mp4'})  # segment 2
        checkpoint_progress(2, 2, downloaded)

        # Verify checkpoint was called twice
        assert mock_checkpoint.save_intermediate.call_count == 2

        # Verify first call (in-progress)
        first_call = mock_checkpoint.save_intermediate.call_args_list[0]
        assert first_call[0][0] == 'DOWNLOAD_SEGMENTS'
        assert first_call[0][1]['segment_count'] == 1
        assert first_call[0][1]['segments_completed'] == 1
        assert first_call[0][1]['segments_total'] == 2
        assert first_call[0][1]['in_progress'] is True

        # Verify second call (complete)
        second_call = mock_checkpoint.save_intermediate.call_args_list[1]
        assert second_call[0][0] == 'DOWNLOAD_SEGMENTS'
        assert second_call[0][1]['segment_count'] == 2
        assert second_call[0][1]['segments_completed'] == 2
        assert second_call[0][1]['segments_total'] == 2
        assert second_call[0][1]['in_progress'] is False

    def test_progress_callback_structure(self, stage, mock_checkpoint):
        """Verify progress callback receives correct structure"""
        # Create the callback similar to how run() creates it
        captured_data = []

        def capture_callback(current, total, downloaded):
            checkpoint_data = {
                'segment_count': len(downloaded),
                'segments_completed': current,
                'segments_total': total,
                'in_progress': current < total,
            }
            captured_data.append(checkpoint_data)

        # Simulate progress
        capture_callback(1, 3, ['seg1'])
        capture_callback(2, 3, ['seg1', 'seg2'])
        capture_callback(3, 3, ['seg1', 'seg2', 'seg3'])

        assert captured_data[0]['in_progress'] is True
        assert captured_data[1]['in_progress'] is True
        assert captured_data[2]['in_progress'] is False

        assert captured_data[2]['segment_count'] == 3
        assert captured_data[2]['segments_completed'] == 3


# ============================================================================
# Can Skip and Restore
# ============================================================================

class TestCanSkipAndRestore:
    """Test can_skip and restore methods"""

    def test_can_skip_delegates_to_checkpoint(self, stage, mock_checkpoint):
        """can_skip checks checkpoint manager"""
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, mock_checkpoint)

        assert result is True
        mock_checkpoint.should_skip_stage.assert_called_once_with("DOWNLOAD_SEGMENTS")

    def test_restore_scans_disk_for_segments(self, stage, mock_checkpoint, mock_config, tmp_path):
        """restore() scans output dir for existing segment files"""
        # Create mock segment files
        mock_config.downloaded_videos_dir = str(tmp_path)
        (tmp_path / "abc123_5_20.mp4").touch()
        (tmp_path / "def456_0_15.mp4").touch()

        state = PipelineState()
        state.downloaded_segments = []

        result = stage.restore(state, mock_checkpoint, mock_config)

        assert result is True
        assert len(state.downloaded_segments) == 2


# ============================================================================
# Validate Inputs
# ============================================================================

class TestValidateInputs:
    """Test validate_inputs method"""

    def test_validate_inputs_returns_error_when_no_matches(
        self, stage, mock_config
    ):
        """validate_inputs returns error message when matches empty"""
        state = PipelineState()
        state.matches = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "No matches" in error

    def test_validate_inputs_returns_none_when_matches_exist(
        self, stage, mock_config, mock_state_with_matches
    ):
        """validate_inputs returns None when matches exist"""
        error = stage.validate_inputs(mock_state_with_matches, mock_config)

        assert error is None


# ============================================================================
# Retry Queue Integration
# ============================================================================

class TestRetryQueueIntegration:
    """Test retry queue integration with DownloadSegmentsStage"""

    def test_failed_downloads_added_to_retry_queue(self, stage, tmp_path):
        """Failed downloads are added to the downloader's retry queue"""
        # Setup mock downloader with retry queue
        from src.downloader import RetryQueue, BatchRetryConfig

        mock_downloader = MagicMock()
        mock_downloader.retry_queue = RetryQueue(BatchRetryConfig(enabled=True))
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        # Create segments to download
        segments = [
            {'video_id': 'fail_vid_1', 'start': 0.0, 'end': 10.0},
            {'video_id': 'fail_vid_2', 'start': 5.0, 'end': 15.0},
        ]

        # Mock yt_dlp.YoutubeDL at the global module level (import happens inside function)
        # Also mock _process_retry_queue to prevent it from running during this test
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            # Set up the context manager to raise on download
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = Exception("HTTP Error 403: Forbidden")
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance
            mock_ydl_class.return_value.__exit__.return_value = False

            # Call download (will fail and add to retry queue)
            result = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        # Verify items were added to retry queue
        retry_queue = mock_downloader.retry_queue
        pending = retry_queue.get_pending_items()

        # Both failed videos should be in the retry queue
        assert len(pending) == 2
        video_ids = {item.video_id for item in pending}
        assert 'fail_vid_1_0_15' in video_ids  # 0-5 buffer, 10+5 buffer = 0_15
        assert 'fail_vid_2_0_20' in video_ids  # 5-5 buffer (clamped to 0), 15+5 buffer = 0_20

    def test_retry_queue_processes_pending_items(self, stage, tmp_path):
        """Retry queue is processed after initial download pass"""
        from src.downloader import RetryQueue, BatchRetryConfig

        # Setup retry queue with a pending item
        retry_queue = RetryQueue(BatchRetryConfig(
            enabled=True,
            delay_seconds=0,  # No delay for test
            max_passes=1
        ))
        retry_queue.add(
            video_id='retry_vid_0_10',
            keyword='segment',
            tier='segment',
            error_message='Initial failure'
        )

        mock_downloader = MagicMock()
        mock_downloader.retry_queue = retry_queue
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        # Create the output file to simulate successful retry
        output_file = tmp_path / 'retry_vid_0_10.mp4'
        output_file.touch()

        downloaded = []

        # Process retry queue
        stage._process_retry_queue(tmp_path, 5.0, downloaded, 1, None)

        # Item should be marked as success (file existed)
        assert 'retry_vid_0_10' not in [item.video_id for item in retry_queue.get_pending_items()]

    def test_retry_queue_not_used_when_downloader_none(self, stage, tmp_path):
        """Gracefully handles case when downloader is None"""
        stage.downloader = None

        segments = [
            {'video_id': 'test_vid', 'start': 0.0, 'end': 10.0},
        ]

        # Mock yt_dlp at the global module level (import happens inside function)
        # Also mock _process_retry_queue to prevent any processing
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = Exception("Test error")
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance
            mock_ydl_class.return_value.__exit__.return_value = False

            # Should not raise even without retry queue
            result = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        # No downloads (all failed), but no exception
        assert result == []

    def test_retry_queue_uses_downloader_impersonation(self, stage, tmp_path):
        """Retry queue downloads use downloader's impersonation manager"""
        from src.downloader import RetryQueue, BatchRetryConfig

        mock_impersonation = MagicMock()
        mock_impersonation.get_ydl_options.return_value = {'impersonate': 'Chrome-136:Macos-15'}

        mock_downloader = MagicMock()
        mock_downloader.retry_queue = RetryQueue(BatchRetryConfig(enabled=True))
        mock_downloader.impersonation_manager = mock_impersonation

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'imp_test', 'start': 0.0, 'end': 10.0},
        ]

        # Mock yt_dlp at the global module level to capture options
        # Also mock _process_retry_queue to prevent it from running
        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            # Create a mock context manager
            mock_ydl_instance = MagicMock()
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_ydl_instance)
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        # Verify impersonation was requested
        mock_impersonation.get_ydl_options.assert_called()

        # Verify impersonation options were applied
        assert len(captured_opts) > 0
        assert captured_opts[0].get('impersonate') == 'Chrome-136:Macos-15'
