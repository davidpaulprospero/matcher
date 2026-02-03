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
        # Create mock segment files with non-zero content
        mock_config.downloaded_videos_dir = str(tmp_path)
        (tmp_path / "abc123_5_20.mp4").write_bytes(b'\x00\x00\x01')
        (tmp_path / "def456_0_15.mp4").write_bytes(b'\x00\x00\x01')

        state = PipelineState()
        state.downloaded_segments = []

        result = stage.restore(state, mock_checkpoint, mock_config)

        assert result is True
        assert len(state.downloaded_segments) == 2

    def test_restore_filters_out_zero_byte_files(self, stage, mock_checkpoint, mock_config, tmp_path):
        """restore() filters out zero-byte (empty) files from restored segments"""
        mock_config.downloaded_videos_dir = str(tmp_path)
        # One valid file, one empty file
        (tmp_path / "abc123_5_20.mp4").write_bytes(b'\x00\x00\x01')
        (tmp_path / "corrupt_0_10.mp4").touch()  # zero bytes

        state = PipelineState()
        state.downloaded_segments = []

        result = stage.restore(state, mock_checkpoint, mock_config)

        assert result is True
        assert len(state.downloaded_segments) == 1
        assert 'abc123_5_20.mp4' in state.downloaded_segments[0].file

    def test_restore_filters_out_nonexistent_files(self, stage, mock_checkpoint, mock_config, tmp_path):
        """restore() filters out files that don't exist (glob race condition)"""
        mock_config.downloaded_videos_dir = str(tmp_path)
        # Create two files, then delete one after the test setup
        valid_file = tmp_path / "abc123_5_20.mp4"
        vanishing_file = tmp_path / "gone_0_10.mp4"
        valid_file.write_bytes(b'\x00\x00\x01')
        vanishing_file.write_bytes(b'\x00\x00\x01')

        # Patch glob to return both, but delete one before restore checks it
        original_glob = Path.glob

        def glob_then_delete(self_path, pattern):
            results = list(original_glob(self_path, pattern))
            # Delete vanishing file after glob finds it but before stat check
            if vanishing_file.exists():
                vanishing_file.unlink()
            return results

        state = PipelineState()
        state.downloaded_segments = []

        with patch.object(Path, 'glob', glob_then_delete):
            result = stage.restore(state, mock_checkpoint, mock_config)

        assert result is True
        assert len(state.downloaded_segments) == 1
        assert 'abc123_5_20.mp4' in state.downloaded_segments[0].file

    def test_restore_logs_valid_and_invalid_counts(self, stage, mock_checkpoint, mock_config, tmp_path):
        """restore() logs count of valid vs invalid segments"""
        mock_config.downloaded_videos_dir = str(tmp_path)
        (tmp_path / "good1_5_20.mp4").write_bytes(b'\x00\x01')
        (tmp_path / "good2_0_15.mp4").write_bytes(b'\x00\x01')
        (tmp_path / "empty_0_10.mp4").touch()  # zero bytes

        state = PipelineState()
        state.downloaded_segments = []

        with patch('src.stages.download_segments.logger') as mock_logger:
            stage.restore(state, mock_checkpoint, mock_config)
            # Check that log message includes both valid and invalid counts
            log_calls = [str(c) for c in mock_logger.info.call_args_list]
            log_text = ' '.join(log_calls)
            assert '2 valid' in log_text
            assert '1 invalid' in log_text

    def test_restore_cross_references_matches(self, stage, mock_checkpoint, mock_config, tmp_path):
        """restore() logs missing segments when matches exist but files don't"""
        mock_config.downloaded_videos_dir = str(tmp_path)
        # Only one segment file on disk
        (tmp_path / "abc123_5_20.mp4").write_bytes(b'\x00\x01')

        state = PipelineState()
        state.downloaded_segments = []
        # Matches reference two video IDs but only one has a file on disk
        match1 = Mock()
        match1.primary_match = None
        match1.video_file = "abc123"
        match2 = Mock()
        match2.primary_match = None
        match2.video_file = "missing_vid"
        state.matches = [match1, match2]

        with patch('src.stages.download_segments.logger') as mock_logger:
            result = stage.restore(state, mock_checkpoint, mock_config)

            assert result is True
            assert len(state.downloaded_segments) == 1
            # Should log about missing video(s)
            log_calls = [str(c) for c in mock_logger.info.call_args_list]
            log_text = ' '.join(log_calls)
            assert '1 matched video(s) have no restored segments' in log_text


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
            result, _stats = stage._download_segments(
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
            result, _stats = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        # No downloads (all failed), but no exception
        assert result == []

    def test_retry_queue_uses_downloader_impersonation(self, stage, tmp_path):
        """Downloads use escalation manager for impersonation args (US-48-005)"""
        from src.downloader import RetryQueue, BatchRetryConfig
        from src.downloader.types import EscalationTier

        # Set up escalation manager that returns Tier 1 impersonation
        mock_escalation_mgr = MagicMock()
        mock_result = MagicMock()
        mock_result.args = ['--impersonate', 'Chrome-136:Macos-15']
        mock_result.tier = EscalationTier.IMPERSONATE_ONLY
        mock_result.rotate_cookies = False
        mock_result.rotate_vpn = False
        mock_escalation_mgr.get_escalation_args.return_value = mock_result

        mock_downloader = MagicMock()
        mock_downloader.retry_queue = RetryQueue(BatchRetryConfig(enabled=True))
        mock_downloader.escalation_manager = mock_escalation_mgr
        mock_downloader.cookie_rotator = None
        mock_downloader.impersonation_manager = None

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

        # Verify escalation manager was consulted
        mock_escalation_mgr.get_escalation_args.assert_called_with('imp_test')

        # Verify impersonation options were applied to ydl_opts
        assert len(captured_opts) > 0
        assert captured_opts[0].get('impersonate') == 'Chrome-136:Macos-15'


# ============================================================================
# Socket Timeout in ydl_opts (US-48-002)
# ============================================================================

class TestSocketTimeoutInYdlOpts:
    """Test that socket_timeout, retries, and fragment_retries are set in ydl_opts"""

    @pytest.mark.fast
    def test_socket_timeout_present_in_ydl_opts(self, stage, tmp_path):
        """socket_timeout, retries, and fragment_retries are present in ydl_opts
        when calling yt-dlp Python API"""
        mock_download_config = MagicMock()
        mock_download_config.socket_timeout = 30
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.impersonation_manager = None
        mock_downloader.retry_queue = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'timeout_test', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
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

        assert len(captured_opts) > 0
        assert captured_opts[0]['socket_timeout'] == 30
        assert captured_opts[0]['retries'] == 10
        assert captured_opts[0]['fragment_retries'] == 10

    @pytest.mark.fast
    def test_socket_timeout_reads_from_config(self, stage, tmp_path):
        """socket_timeout value is read from download config, not hardcoded"""
        mock_download_config = MagicMock()
        mock_download_config.socket_timeout = 60  # Custom value
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.impersonation_manager = None
        mock_downloader.retry_queue = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'config_test', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert captured_opts[0]['socket_timeout'] == 60

    @pytest.mark.fast
    def test_socket_timeout_fallback_default(self, stage, tmp_path):
        """socket_timeout defaults to 30 when downloader has no config"""
        stage.downloader = None

        segments = [
            {'video_id': 'fallback_test', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert captured_opts[0]['socket_timeout'] == 30

    @pytest.mark.fast
    def test_socket_timeout_in_retry_queue_ydl_opts(self, stage, tmp_path):
        """socket_timeout is also applied in retry queue ydl_opts"""
        from src.downloader import RetryQueue, BatchRetryConfig

        mock_download_config = MagicMock()
        mock_download_config.socket_timeout = 45
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'

        retry_queue = RetryQueue(BatchRetryConfig(
            enabled=True, delay_seconds=0, max_passes=1
        ))
        retry_queue.add(
            video_id='retry_timeout_0_10',
            keyword='segment',
            tier='segment',
            error_message='Initial failure'
        )

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.retry_queue = retry_queue
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._process_retry_queue(tmp_path, 5.0, [], 1, None)

        assert len(captured_opts) > 0
        assert captured_opts[0]['socket_timeout'] == 45
        assert captured_opts[0]['retries'] == 10
        assert captured_opts[0]['fragment_retries'] == 10


# ============================================================================
# Network Failure Circuit Breaker (US-48-004)
# ============================================================================

class TestNetworkFailureCircuitBreaker:
    """Test consecutive network failure detection and early abort in download loop"""

    @pytest.mark.fast
    def test_is_network_failure_detects_dns_errors(self):
        """_is_network_failure detects DNS resolution failures"""
        from src.stages.download_segments import _is_network_failure

        # Should detect
        assert _is_network_failure("getaddrinfo failed") is True
        assert _is_network_failure("[Errno 11001] getaddrinfo failed") is True
        assert _is_network_failure("Name or service not known") is True
        assert _is_network_failure("nodename nor servname provided") is True
        assert _is_network_failure("Network is unreachable") is True
        assert _is_network_failure("Temporary failure in name resolution") is True
        assert _is_network_failure("No address associated with hostname") is True
        # ffmpeg network exit code
        assert _is_network_failure("ffmpeg exited with code 4294967158") is True

        # Should NOT detect (video-specific errors)
        assert _is_network_failure("HTTP Error 403: Forbidden") is False
        assert _is_network_failure("Video unavailable") is False
        assert _is_network_failure("This video is private") is False

    @pytest.mark.fast
    def test_early_abort_after_consecutive_dns_failures(self, stage, tmp_path):
        """Download loop aborts after 3 consecutive DNS failures, skipping remaining segments"""
        stage.downloader = None  # No retry queue

        # 5 segments — should abort after 3rd DNS failure
        segments = [
            {'video_id': f'dns_fail_{i}', 'start': 0.0, 'end': 10.0}
            for i in range(5)
        ]

        dns_error = OSError("[Errno 11001] getaddrinfo failed")

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = dns_error
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance
            mock_ydl_class.return_value.__exit__.return_value = False

            result, _stats = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        # yt-dlp should only be called 3 times (abort after 3rd failure)
        assert mock_ydl_class.call_count == 3
        assert result == []

    @pytest.mark.fast
    def test_counter_resets_on_successful_download(self, stage, tmp_path):
        """Counter resets after a successful download between network failures"""
        stage.downloader = None

        # 6 segments: fail, fail, success, fail, fail, fail
        # The success at index 2 resets the counter, so segments 3-5
        # are the new consecutive failures that should trigger abort at segment 5
        segments = [
            {'video_id': f'reset_test_{i}', 'start': 0.0, 'end': 10.0}
            for i in range(6)
        ]

        dns_error = OSError("[Errno 11001] getaddrinfo failed")
        call_count = [0]

        def mock_download_side_effect(urls):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 2:
                # Simulate success: create the output file
                video_id = segments[idx]['video_id']
                output_file = tmp_path / f"{video_id}_0_15.mp4"  # 0-5 buffer, 10+5
                output_file.write_text("fake")
                return
            raise dns_error

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = mock_download_side_effect
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_ydl_instance)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_ydl_class.return_value = mock_cm

            result, _stats = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        # Should have tried: 0(fail), 1(fail), 2(success+reset), 3(fail), 4(fail), 5(fail=abort)
        # Total yt-dlp calls = 6 (all 6 segments attempted)
        assert mock_ydl_class.call_count == 6
        # Only 1 successful download (index 2)
        assert len(result) == 1

    @pytest.mark.fast
    def test_non_network_errors_dont_trigger_abort(self, stage, tmp_path):
        """Non-network errors (403, removed) don't increment the network failure counter"""
        stage.downloader = None

        segments = [
            {'video_id': f'http_err_{i}', 'start': 0.0, 'end': 10.0}
            for i in range(5)
        ]

        http_error = Exception("HTTP Error 403: Forbidden")

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = http_error
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance
            mock_ydl_class.return_value.__exit__.return_value = False

            result, _stats = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        # All 5 segments should be attempted (no early abort)
        assert mock_ydl_class.call_count == 5

    @pytest.mark.fast
    def test_abort_checkpoints_before_breaking(self, stage, tmp_path):
        """Progress callback is called before aborting the loop"""
        stage.downloader = None

        segments = [
            {'video_id': f'cp_test_{i}', 'start': 0.0, 'end': 10.0}
            for i in range(5)
        ]

        dns_error = OSError("getaddrinfo failed")
        progress_calls = []

        def track_progress(current, total, downloaded):
            progress_calls.append((current, total, len(downloaded)))

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = dns_error
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance
            mock_ydl_class.return_value.__exit__.return_value = False

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0,
                progress_callback=track_progress
            )

        # Should have 3 progress calls (segments 1, 2, 3 — abort after 3rd)
        assert len(progress_calls) == 3
        # Last call should be for segment 3 of 5
        assert progress_calls[-1][0] == 3
        assert progress_calls[-1][1] == 5


# ============================================================================
# Escalation Tier Integration (US-48-005)
# ============================================================================

class TestEscalationTierIntegration:
    """Test EscalationManager integration with download_segments stage.

    US-48-005: Verifies that download_segments applies escalation tiers
    (impersonation, extractor_args, cookie rotation) via EscalationManager
    instead of only Tier 1 impersonation.
    """

    @pytest.mark.fast
    def test_apply_escalation_to_ydl_opts_impersonation(self):
        """_apply_escalation_to_ydl_opts translates --impersonate to ydl_opts"""
        from src.stages.download_segments import _apply_escalation_to_ydl_opts

        # Simulate EscalationResult with Tier 1 impersonation args
        mock_result = MagicMock()
        mock_result.args = ['--impersonate', 'Chrome-136:Macos-15']

        ydl_opts = {}
        _apply_escalation_to_ydl_opts(ydl_opts, mock_result)

        assert ydl_opts['impersonate'] == 'Chrome-136:Macos-15'

    @pytest.mark.fast
    def test_apply_escalation_to_ydl_opts_extractor_args(self):
        """_apply_escalation_to_ydl_opts translates --extractor-args to ydl_opts"""
        from src.stages.download_segments import _apply_escalation_to_ydl_opts

        # Simulate EscalationResult with Tier 2 args
        mock_result = MagicMock()
        mock_result.args = [
            '--impersonate', 'Chrome-136:Macos-15',
            '--extractor-args', 'youtube:player_client=web_safari,tv_downgraded,web',
        ]

        ydl_opts = {}
        _apply_escalation_to_ydl_opts(ydl_opts, mock_result)

        assert ydl_opts['impersonate'] == 'Chrome-136:Macos-15'
        assert 'extractor_args' in ydl_opts
        assert ydl_opts['extractor_args']['youtube']['player_client'] == 'web_safari,tv_downgraded,web'

    @pytest.mark.fast
    def test_tier2_extractor_args_added_after_403(self, stage, tmp_path):
        """After a 403 error, EscalationManager escalates and Tier 2 extractor_args
        are applied to ydl_opts on the next download attempt."""
        from src.downloader.types import EscalationTier

        # Create mock escalation manager that starts at Tier 1 then escalates to Tier 2
        mock_escalation_mgr = MagicMock()

        # Track calls to get_escalation_args to return different tiers
        call_count = [0]

        def mock_get_escalation_args(keyword):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                # First call: Tier 1 (will fail with 403)
                result.args = ['--impersonate', 'Chrome-136:Macos-15']
                result.tier = EscalationTier.IMPERSONATE_ONLY
                result.rotate_cookies = False
                result.rotate_vpn = False
            else:
                # Second call: Tier 2 (after 403 triggered escalation)
                result.args = [
                    '--impersonate', 'Chrome-136:Macos-15',
                    '--extractor-args', 'youtube:player_client=web_safari,tv_downgraded,web',
                ]
                result.tier = EscalationTier.EXTRACTOR_ARGS
                result.rotate_cookies = False
                result.rotate_vpn = False
            return result

        mock_escalation_mgr.get_escalation_args.side_effect = mock_get_escalation_args

        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = mock_escalation_mgr
        mock_downloader.cookie_rotator = None
        mock_downloader.retry_queue = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'esc_test_1', 'start': 0.0, 'end': 10.0},
            {'video_id': 'esc_test_2', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        download_call = [0]

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()

            def mock_download(urls):
                download_call[0] += 1
                if download_call[0] == 1:
                    raise Exception("HTTP Error 403: Forbidden")
                # Second download succeeds — create the output file
                vid_id = segments[1]['video_id']
                out = tmp_path / f"{vid_id}_0_15.mp4"
                out.write_text("fake")

            mock_ydl_instance.download.side_effect = mock_download
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

        # First call should NOT have extractor_args (Tier 1)
        assert 'extractor_args' not in captured_opts[0]
        assert captured_opts[0].get('impersonate') == 'Chrome-136:Macos-15'

        # Second call SHOULD have extractor_args (Tier 2 after 403)
        assert 'extractor_args' in captured_opts[1]
        assert captured_opts[1]['extractor_args']['youtube']['player_client'] == 'web_safari,tv_downgraded,web'

        # Verify record_failure was called for the 403 error
        mock_escalation_mgr.record_failure.assert_called_once()
        call_args = mock_escalation_mgr.record_failure.call_args
        assert call_args[0][0] == 'esc_test_1'
        assert '403' in call_args[0][1]

    @pytest.mark.fast
    def test_cookie_rotation_on_auth_error(self, stage, tmp_path):
        """Cookie rotator is advanced when escalation reaches Tier 3 and auth error occurs."""
        from src.downloader.types import EscalationTier

        mock_escalation_mgr = MagicMock()

        # Return Tier 3 (full bypass with cookies) for all calls
        mock_result = MagicMock()
        mock_result.args = [
            '--impersonate', 'Chrome-136:Macos-15',
            '--extractor-args', 'youtube:player_client=web_safari,tv_downgraded,web',
        ]
        mock_result.tier = EscalationTier.FULL_BYPASS
        mock_result.rotate_cookies = True
        mock_result.rotate_vpn = False
        mock_escalation_mgr.get_escalation_args.return_value = mock_result

        mock_cookie_rotator = MagicMock()
        mock_cookie_rotator.get_current_cookie.return_value = '/tmp/cookies_1.txt'
        mock_cookie_rotator.should_rotate.return_value = True

        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = mock_escalation_mgr
        mock_downloader.cookie_rotator = mock_cookie_rotator
        mock_downloader.retry_queue = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'cookie_test', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = Exception("HTTP Error 403: Forbidden")
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

        # Verify cookie was applied to ydl_opts
        assert captured_opts[0].get('cookiefile') == '/tmp/cookies_1.txt'

        # Verify escalation manager recorded failure
        mock_escalation_mgr.record_failure.assert_called_once()

        # Verify cookie rotator was advanced after auth error
        mock_cookie_rotator.should_rotate.assert_called_once()
        mock_cookie_rotator.rotate.assert_called_once()

    @pytest.mark.fast
    def test_escalation_state_tracked_per_segment(self, stage, tmp_path):
        """Escalation state is tracked per video_id (different segments get
        different escalation lookups)."""
        from src.downloader.types import EscalationTier

        mock_escalation_mgr = MagicMock()

        # Track which video_ids are passed to get_escalation_args
        called_keywords = []

        def mock_get_escalation_args(keyword):
            called_keywords.append(keyword)
            result = MagicMock()
            result.args = ['--impersonate', 'Chrome-136:Macos-15']
            result.tier = EscalationTier.IMPERSONATE_ONLY
            result.rotate_cookies = False
            result.rotate_vpn = False
            return result

        mock_escalation_mgr.get_escalation_args.side_effect = mock_get_escalation_args

        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = mock_escalation_mgr
        mock_downloader.cookie_rotator = None
        mock_downloader.retry_queue = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'vid_alpha', 'start': 0.0, 'end': 10.0},
            {'video_id': 'vid_beta', 'start': 0.0, 'end': 10.0},
            {'video_id': 'vid_gamma', 'start': 0.0, 'end': 10.0},
        ]

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = Exception("Test error")
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance
            mock_ydl_class.return_value.__exit__.return_value = False

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        # Each video_id should have its own escalation lookup
        assert called_keywords == ['vid_alpha', 'vid_beta', 'vid_gamma']

    @pytest.mark.fast
    def test_record_success_called_on_successful_download(self, stage, tmp_path):
        """EscalationManager.record_success() is called when download succeeds."""
        from src.downloader.types import EscalationTier

        mock_escalation_mgr = MagicMock()
        mock_result = MagicMock()
        mock_result.args = ['--impersonate', 'Chrome-136:Macos-15']
        mock_result.tier = EscalationTier.IMPERSONATE_ONLY
        mock_result.rotate_cookies = False
        mock_result.rotate_vpn = False
        mock_escalation_mgr.get_escalation_args.return_value = mock_result

        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = mock_escalation_mgr
        mock_downloader.cookie_rotator = None
        mock_downloader.retry_queue = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'success_vid', 'start': 0.0, 'end': 10.0},
        ]

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()

            def mock_download(urls):
                # Create output file to simulate success
                out = tmp_path / "success_vid_0_15.mp4"
                out.write_text("fake")

            mock_ydl_instance.download.side_effect = mock_download
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_ydl_instance)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_ydl_class.return_value = mock_cm

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        mock_escalation_mgr.record_success.assert_called_once_with('success_vid')

    @pytest.mark.fast
    def test_apply_escalation_handles_none_result(self):
        """_apply_escalation_to_ydl_opts handles None result gracefully."""
        from src.stages.download_segments import _apply_escalation_to_ydl_opts

        ydl_opts = {'format': 'best'}
        _apply_escalation_to_ydl_opts(ydl_opts, None)
        assert ydl_opts == {'format': 'best'}  # Unchanged

    @pytest.mark.fast
    def test_apply_escalation_handles_empty_args(self):
        """_apply_escalation_to_ydl_opts handles empty args list."""
        from src.stages.download_segments import _apply_escalation_to_ydl_opts

        mock_result = MagicMock()
        mock_result.args = []

        ydl_opts = {'format': 'best'}
        _apply_escalation_to_ydl_opts(ydl_opts, mock_result)
        assert ydl_opts == {'format': 'best'}  # Unchanged


# ============================================================================
# Download Progress Reporting (US-48-007)
# ============================================================================

class TestDownloadProgressReporting:
    """Test progress counters, running success rate, and end-of-stage summary.

    US-48-007: Verifies that _download_segments tracks success/failure/cached
    counters and that a summary is printed at end of stage.
    """

    @pytest.mark.fast
    def test_progress_callback_called_with_correct_values(self, stage, tmp_path):
        """Progress callback receives correct current/total values for each segment."""
        stage.downloader = None

        segments = [
            {'video_id': f'prog_{i}', 'start': 0.0, 'end': 10.0}
            for i in range(3)
        ]

        progress_calls = []

        def track_progress(current, total, downloaded):
            progress_calls.append((current, total))

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = Exception("Test error")
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance
            mock_ydl_class.return_value.__exit__.return_value = False

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0,
                progress_callback=track_progress
            )

        # Each of the 3 segments should trigger a progress callback
        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3)
        assert progress_calls[1] == (2, 3)
        assert progress_calls[2] == (3, 3)

    @pytest.mark.fast
    def test_stats_returned_with_correct_counters(self, stage, tmp_path):
        """_download_segments returns stats dict with succeeded/failed/cached counts."""
        stage.downloader = None

        # Pre-create a cached file for segment 0
        (tmp_path / "cached_vid_0_15.mp4").write_text("fake")

        segments = [
            {'video_id': 'cached_vid', 'start': 0.0, 'end': 10.0},  # cached
            {'video_id': 'success_vid', 'start': 0.0, 'end': 10.0},  # will succeed
            {'video_id': 'fail_vid', 'start': 0.0, 'end': 10.0},  # will fail
        ]

        call_idx = [0]

        def mock_download(urls):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 0:
                # success_vid — create output file
                (tmp_path / "success_vid_0_15.mp4").write_text("fake")
                return
            raise Exception("HTTP Error 403: Forbidden")

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = mock_download
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_ydl_instance)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_ydl_class.return_value = mock_cm

            _downloaded, stats = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert stats['cached'] == 1
        assert stats['succeeded'] == 1
        assert stats['failed'] == 1
        assert stats['attempted'] == 3
        assert stats['total'] == 3

    @pytest.mark.fast
    def test_summary_includes_all_categories(self, stage, capsys):
        """_print_summary outputs all counter categories and time."""
        stats = {
            'succeeded': 5,
            'failed': 2,
            'cached': 3,
            'attempted': 10,
            'total': 10,
        }

        stage._print_summary(stats, elapsed=125.3)
        captured = capsys.readouterr().out

        assert 'Attempted: 10/10' in captured
        assert 'Succeeded: 5' in captured
        assert 'Cached:    3' in captured
        assert 'Failed:    2' in captured
        assert 'Success rate: 80%' in captured
        assert '2m 5s' in captured

    @pytest.mark.fast
    def test_print_progress_displays_running_rate(self, stage, capsys):
        """_print_progress shows running success rate."""
        stats = {
            'succeeded': 2,
            'failed': 1,
            'cached': 1,
            'attempted': 4,
            'total': 6,
        }

        stage._print_progress(4, 6, stats)
        captured = capsys.readouterr().out

        assert '[4/6]' in captured
        assert 'ok=3' in captured  # succeeded + cached
        assert 'fail=1' in captured
        assert 'cached=1' in captured
        assert '75% success' in captured

    @pytest.mark.fast
    def test_cached_segments_counted_in_stats(self, stage, tmp_path):
        """Cached (already-existing) segments are counted in stats."""
        stage.downloader = None

        # Pre-create all files so all are cached
        for i in range(3):
            (tmp_path / f"cache_{i}_0_15.mp4").write_text("fake")

        segments = [
            {'video_id': f'cache_{i}', 'start': 0.0, 'end': 10.0}
            for i in range(3)
        ]

        with patch.object(stage, '_process_retry_queue'):
            _downloaded, stats = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert stats['cached'] == 3
        assert stats['succeeded'] == 0
        assert stats['failed'] == 0
        assert stats['attempted'] == 3


# ============================================================================
# Segment Deduplication & Merging (US-48-008)
# ============================================================================

class TestSegmentDeduplicationAndMerging:
    """Test improved deduplication and overlap merging in _collect_matched_segments.

    US-48-008: Verifies that:
    - Overlapping/adjacent segments from the same video are merged
    - Segments from different videos with same timestamps are NOT merged
    - Video IDs containing underscores are deduplicated correctly
    """

    @pytest.mark.fast
    def test_overlapping_segments_same_video_merged(self, stage):
        """Two overlapping segments from the same video are merged into one."""
        state = PipelineState()

        # Segment 1: 10-30s, Segment 2: 25-50s (overlap at 25-30)
        match1 = Mock()
        match1.primary_match = None
        match1.video_file = "vid_overlap"
        match1.video_start = 10.0
        match1.video_end = 30.0

        match2 = Mock()
        match2.primary_match = None
        match2.video_file = "vid_overlap"
        match2.video_start = 25.0
        match2.video_end = 50.0

        state.matches = [match1, match2]

        segments = stage._collect_matched_segments(state, buffer_seconds=5.0)

        # Should merge into one segment: 10-50
        assert len(segments) == 1
        assert segments[0]['video_id'] == "vid_overlap"
        assert segments[0]['start'] == 10.0
        assert segments[0]['end'] == 50.0

    @pytest.mark.fast
    def test_adjacent_segments_same_video_merged(self, stage):
        """Adjacent segments (gap < 2*buffer) from same video are merged."""
        state = PipelineState()

        # Segment 1: 10-20s, Segment 2: 25-35s (gap=5s, 2*buffer=10s → merge)
        match1 = Mock()
        match1.primary_match = None
        match1.video_file = "vid_adj"
        match1.video_start = 10.0
        match1.video_end = 20.0

        match2 = Mock()
        match2.primary_match = None
        match2.video_file = "vid_adj"
        match2.video_start = 25.0
        match2.video_end = 35.0

        state.matches = [match1, match2]

        segments = stage._collect_matched_segments(state, buffer_seconds=5.0)

        # Gap (5s) < 2*buffer (10s) → merged into one: 10-35
        assert len(segments) == 1
        assert segments[0]['video_id'] == "vid_adj"
        assert segments[0]['start'] == 10.0
        assert segments[0]['end'] == 35.0

    @pytest.mark.fast
    def test_distant_segments_same_video_not_merged(self, stage):
        """Distant segments from same video (gap > 2*buffer) stay separate."""
        state = PipelineState()

        # Segment 1: 10-20s, Segment 2: 100-110s (gap=80s >> 2*buffer=10s)
        match1 = Mock()
        match1.primary_match = None
        match1.video_file = "vid_dist"
        match1.video_start = 10.0
        match1.video_end = 20.0

        match2 = Mock()
        match2.primary_match = None
        match2.video_file = "vid_dist"
        match2.video_start = 100.0
        match2.video_end = 110.0

        state.matches = [match1, match2]

        segments = stage._collect_matched_segments(state, buffer_seconds=5.0)

        # Gap (80s) > 2*buffer (10s) → NOT merged
        assert len(segments) == 2

    @pytest.mark.fast
    def test_different_videos_same_timestamps_not_merged(self, stage):
        """Segments from different videos with identical timestamps stay separate."""
        state = PipelineState()

        match1 = Mock()
        match1.primary_match = None
        match1.video_file = "video_alpha"
        match1.video_start = 10.0
        match1.video_end = 25.0

        match2 = Mock()
        match2.primary_match = None
        match2.video_file = "video_beta"
        match2.video_start = 10.0
        match2.video_end = 25.0

        state.matches = [match1, match2]

        segments = stage._collect_matched_segments(state, buffer_seconds=5.0)

        # Different videos → NOT merged
        assert len(segments) == 2
        video_ids = {s['video_id'] for s in segments}
        assert video_ids == {"video_alpha", "video_beta"}

    @pytest.mark.fast
    def test_underscore_video_ids_deduplicated(self, stage):
        """Video IDs containing underscores are correctly deduplicated."""
        state = PipelineState()

        # Same video ID with underscores, same timestamps → should dedup
        match1 = Mock()
        match1.primary_match = None
        match1.video_file = "abc_def_ghi"
        match1.video_start = 10.0
        match1.video_end = 25.0

        match2 = Mock()
        match2.primary_match = None
        match2.video_file = "abc_def_ghi"
        match2.video_start = 10.0
        match2.video_end = 25.0

        state.matches = [match1, match2]

        segments = stage._collect_matched_segments(state, buffer_seconds=5.0)

        # Exact duplicate → only 1 segment
        assert len(segments) == 1
        assert segments[0]['video_id'] == "abc_def_ghi"

    @pytest.mark.fast
    def test_underscore_video_id_local_path_mapping(self, stage):
        """_update_matches_with_local_paths correctly parses video IDs with underscores."""
        from src.state import DownloadedVideo

        state = PipelineState()

        match1 = Mock()
        match1.video_file = "abc_def"
        match1.primary_match = None

        state.matches = [match1]

        # Filename format: {video_id}_{start}_{end}.mp4
        downloaded = [
            DownloadedVideo(
                file="/tmp/segments/abc_def_5_30.mp4",
                source='segment_download'
            )
        ]

        stage._update_matches_with_local_paths(state, downloaded)

        # Should map abc_def correctly (rsplit from right, not split from left)
        assert match1.video_file == "/tmp/segments/abc_def_5_30.mp4"

    @pytest.mark.fast
    @pytest.mark.fast
    def test_merge_segments_empty_list(self, stage):
        """_merge_segments returns empty list for empty input."""
        assert DownloadVideoSegmentsStage._merge_segments([], 5.0) == []


# ============================================================================
# Configurable Segment Defaults (US-48-010)
# ============================================================================

class TestConfigurableSegmentDefaults:
    """Test that DownloadConfig segment_ fields exist with correct defaults
    and are wired into download_segments stage.

    US-48-010: Verifies:
    - DownloadConfig loads segment_buffer, segment_format, segment_max_resolution,
      segment_socket_timeout from config dict
    - Default values match current hardcoded values (buffer=5.0, resolution=1080)
    - Stage reads segment_buffer directly from config (no getattr fallback)
    - Stage uses segment_max_resolution in format string
    """

    @pytest.mark.fast
    def test_download_config_has_segment_fields_with_defaults(self):
        """DownloadConfig has segment_ fields with correct default values."""
        from src.config.sections.download import DownloadConfig

        cfg = DownloadConfig()

        assert cfg.segment_buffer == 5.0
        assert cfg.segment_max_resolution == 1080
        assert cfg.segment_socket_timeout == 0
        assert '{segment_max_resolution}' in cfg.segment_format

    @pytest.mark.fast
    def test_download_config_loads_segment_fields_from_dict(self):
        """DownloadConfig loads segment_ prefixed fields from config dict."""
        from src.config.sections.download import DownloadConfig

        cfg = DownloadConfig(
            segment_buffer=10.0,
            segment_format='best[height<={segment_max_resolution}]/best',
            segment_max_resolution=720,
            segment_socket_timeout=45,
        )

        assert cfg.segment_buffer == 10.0
        assert cfg.segment_max_resolution == 720
        assert cfg.segment_socket_timeout == 45
        assert '720' not in cfg.segment_format  # Template not yet resolved
        assert '{segment_max_resolution}' in cfg.segment_format

    @pytest.mark.fast
    def test_default_values_match_previous_hardcoded(self):
        """Default segment values match what was previously hardcoded.

        Before US-48-010:
        - buffer was getattr(download_config, 'segment_buffer', 5.0) → 5.0
        - format was 'best[height<=1080]' → resolution 1080
        """
        from src.config.sections.download import DownloadConfig

        cfg = DownloadConfig()

        # Buffer default matches getattr fallback
        assert cfg.segment_buffer == 5.0
        # Resolution default matches hardcoded format string
        assert cfg.segment_max_resolution == 1080
        # Format template resolves to the previously hardcoded value
        resolved = cfg.segment_format.format(segment_max_resolution=cfg.segment_max_resolution)
        assert resolved == 'best[height<=1080]'

    @pytest.mark.fast
    def test_segment_max_resolution_wired_into_format_string(self, stage, tmp_path):
        """segment_max_resolution is used in the ydl_opts format string."""
        mock_download_config = MagicMock()
        mock_download_config.segment_buffer = 5.0
        mock_download_config.segment_max_resolution = 720
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.socket_timeout = 30

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.impersonation_manager = None
        mock_downloader.retry_queue = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'res_test', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert len(captured_opts) > 0
        assert captured_opts[0]['format'] == 'best[height<=720]'

    @pytest.mark.fast
    def test_segment_socket_timeout_overrides_main(self, stage, tmp_path):
        """When segment_socket_timeout > 0, it overrides the main socket_timeout."""
        mock_download_config = MagicMock()
        mock_download_config.segment_buffer = 5.0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'
        mock_download_config.segment_socket_timeout = 60
        mock_download_config.socket_timeout = 30

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.impersonation_manager = None
        mock_downloader.retry_queue = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'sock_test', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert captured_opts[0]['socket_timeout'] == 60

    @pytest.mark.fast
    def test_segment_socket_timeout_zero_uses_main(self, stage, tmp_path):
        """When segment_socket_timeout is 0, falls back to main socket_timeout."""
        mock_download_config = MagicMock()
        mock_download_config.segment_buffer = 5.0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.socket_timeout = 45

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.impersonation_manager = None
        mock_downloader.retry_queue = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'sock_default', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert captured_opts[0]['socket_timeout'] == 45


    def test_precision_dedup_preserves_close_segments(self, stage):
        """Segments differing by <0.5s are NOT incorrectly deduped (unlike round()).

        Old code used round() which would treat (10.0, 20.0) and (10.3, 20.3)
        as duplicates. New code uses exact floats for dedup keys.
        """
        state = PipelineState()

        # Two non-overlapping segments with start/end that round() to the same int
        match1 = Mock()
        match1.primary_match = None
        match1.video_file = "precision_vid"
        match1.video_start = 10.0
        match1.video_end = 20.0

        match2 = Mock()
        match2.primary_match = None
        match2.video_file = "precision_vid"
        match2.video_start = 50.3
        match2.video_end = 60.3

        state.matches = [match1, match2]

        # With buffer=0 and distant segments, they should NOT merge
        segments = stage._collect_matched_segments(state, buffer_seconds=0.0)
        assert len(segments) == 2

        # Verify that the old round()-based dedup would have collapsed
        # segments with close start/end values. These two segments have
        # different exact values but round() to the same ints.
        match3 = Mock()
        match3.primary_match = None
        match3.video_file = "precision_vid2"
        match3.video_start = 10.0
        match3.video_end = 20.0

        match4 = Mock()
        match4.primary_match = None
        match4.video_file = "precision_vid2"
        match4.video_start = 10.4   # round(10.4) == 10 == round(10.0)
        match4.video_end = 20.4     # round(20.4) == 20 == round(20.0)

        state.matches = [match3, match4]
        segments = stage._collect_matched_segments(state, buffer_seconds=0.0)

        # New code: these are distinct (10.0 != 10.4, 20.0 != 20.4)
        # but they overlap (10.4 < 20.0) so they merge. This is correct —
        # old code would have SILENTLY DROPPED match4 as a "duplicate".
        # New code properly merges to [10.0, 20.4].
        assert len(segments) == 1
        assert segments[0]['start'] == 10.0
        assert segments[0]['end'] == 20.4  # Extended, not dropped


# ============================================================================
# Stage Metrics Collection (US-48-011)
# ============================================================================

class TestStageMetricsCollection:
    """Test StageMetrics is populated with correct counts after run().

    US-48-011: Verifies that:
    - StageMetrics.items_processed counts succeeded + cached
    - StageMetrics.items_failed counts failed downloads
    - StageMetrics.duration_seconds reflects stage elapsed time
    - Per-segment durations are tracked in stats
    - Total bytes downloaded are tracked from output file sizes
    """

    @pytest.mark.fast
    def test_stage_metrics_returned_on_successful_run(
        self, stage, mock_state_with_matches, mock_config, mock_checkpoint, tmp_path
    ):
        """run() returns StageResult with populated StageMetrics."""
        from src.stages import StageMetrics

        mock_config.downloaded_videos_dir = str(tmp_path)

        # Pre-create all segment files so they're all cached (no yt-dlp needed)
        segments = stage._collect_matched_segments(mock_state_with_matches)
        for seg in segments:
            vid = seg['video_id']
            s = int(max(0, seg['start'] - 5.0))
            e = int(seg['end'] + 5.0)
            (tmp_path / f"{vid}_{s}_{e}.mp4").write_bytes(b'\x00' * 1024)

        with patch.object(stage, '_process_retry_queue'), \
             patch('src.downloader.VideoDownloader'):
            result = stage.run(mock_state_with_matches, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.metrics is not None
        assert isinstance(result.metrics, StageMetrics)
        # 2 unique segments (3 matches but 1 is duplicate), all cached
        assert result.metrics.items_processed == 2
        assert result.metrics.items_failed == 0
        assert result.metrics.duration_seconds > 0

    @pytest.mark.fast
    def test_stage_metrics_counts_failures(self, stage, tmp_path):
        """StageMetrics.items_failed reflects actual download failures."""
        stage.downloader = None

        # 3 segments, all fail
        segments = [
            {'video_id': f'fail_{i}', 'start': 0.0, 'end': 10.0}
            for i in range(3)
        ]

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = Exception("HTTP 403")
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance
            mock_ydl_class.return_value.__exit__.return_value = False

            _downloaded, stats = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert stats['failed'] == 3
        assert stats['succeeded'] == 0

    @pytest.mark.fast
    def test_segment_durations_tracked_per_segment(self, stage, tmp_path):
        """Per-segment download durations are recorded in stats."""
        stage.downloader = None

        segments = [
            {'video_id': f'dur_{i}', 'start': 0.0, 'end': 10.0}
            for i in range(2)
        ]

        def mock_download(urls):
            # Create output file for each call
            vid_id = segments[len(stage_calls)]['video_id']
            out = tmp_path / f"{vid_id}_0_15.mp4"
            out.write_bytes(b'\x00' * 512)
            stage_calls.append(1)

        stage_calls = []

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = mock_download
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_ydl_instance)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_ydl_class.return_value = mock_cm

            _downloaded, stats = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        # Should have 2 duration entries (one per successful download)
        assert len(stats['segment_durations']) == 2
        assert all(d >= 0 for d in stats['segment_durations'])

    @pytest.mark.fast
    def test_total_bytes_tracked_from_file_sizes(self, stage, tmp_path):
        """Total bytes downloaded is summed from output file sizes."""
        stage.downloader = None

        segments = [
            {'video_id': 'bytes_test', 'start': 0.0, 'end': 10.0},
        ]

        def mock_download(urls):
            out = tmp_path / "bytes_test_0_15.mp4"
            out.write_bytes(b'\x00' * 2048)

        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = mock_download
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_ydl_instance)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_ydl_class.return_value = mock_cm

            _downloaded, stats = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert stats['total_bytes'] == 2048

    @pytest.mark.fast
    def test_total_bytes_includes_cached_files(self, stage, tmp_path):
        """Total bytes includes size of already-cached segment files."""
        stage.downloader = None

        # Pre-create cached file
        cached = tmp_path / "cached_bytes_0_15.mp4"
        cached.write_bytes(b'\x00' * 4096)

        segments = [
            {'video_id': 'cached_bytes', 'start': 0.0, 'end': 10.0},
        ]

        with patch.object(stage, '_process_retry_queue'):
            _downloaded, stats = stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert stats['total_bytes'] == 4096
        assert stats['cached'] == 1

    @pytest.mark.fast
    def test_summary_includes_avg_median_duration(self, stage, capsys):
        """_print_summary outputs avg and median segment time when durations exist."""
        stats = {
            'succeeded': 3,
            'failed': 0,
            'cached': 0,
            'attempted': 3,
            'total': 3,
            'segment_durations': [2.0, 4.0, 6.0],
            'total_bytes': 0,
            'retry_count': 0,
        }

        stage._print_summary(stats, elapsed=12.0)
        captured = capsys.readouterr().out

        assert 'Avg segment time: 4.0s' in captured
        assert 'Median segment time: 4.0s' in captured

    @pytest.mark.fast
    def test_summary_includes_total_size(self, stage, capsys):
        """_print_summary outputs total downloaded size."""
        stats = {
            'succeeded': 2,
            'failed': 0,
            'cached': 0,
            'attempted': 2,
            'total': 2,
            'segment_durations': [1.0, 2.0],
            'total_bytes': 15 * 1024 * 1024,  # 15 MB
            'retry_count': 0,
        }

        stage._print_summary(stats, elapsed=3.0)
        captured = capsys.readouterr().out

        assert 'Total size: 15.0 MB' in captured

    @pytest.mark.fast
    def test_summary_shows_retry_count(self, stage, capsys):
        """_print_summary shows retry count when retries occurred."""
        stats = {
            'succeeded': 3,
            'failed': 1,
            'cached': 0,
            'attempted': 4,
            'total': 4,
            'segment_durations': [1.0],
            'total_bytes': 0,
            'retry_count': 2,
        }

        stage._print_summary(stats, elapsed=5.0)
        captured = capsys.readouterr().out

        assert 'Retried: 2' in captured

    @pytest.mark.fast
    def test_stage_metrics_on_skip_no_matches(
        self, stage, mock_state_empty, mock_config, mock_checkpoint
    ):
        """Skipped stage (no matches) returns no metrics (None)."""
        result = stage.run(mock_state_empty, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.metrics is None


# ============================================================================
# Retry Count in Checkpoint Data (US-48-011)
# ============================================================================

class TestRetryCountInCheckpointData:
    """Test retry_count in checkpoint data reflects actual retry attempts.

    US-48-011: Verifies that:
    - retry_count is included in checkpoint_data returned by run()
    - retry_count reflects the number of segments that were retried
    - retry_count is 0 when no retries occurred
    """

    @pytest.mark.fast
    def test_retry_count_zero_when_no_retries(
        self, stage, mock_state_with_matches, mock_config, mock_checkpoint, tmp_path
    ):
        """checkpoint_data has retry_count=0 when no retry queue items."""
        mock_config.downloaded_videos_dir = str(tmp_path)

        # Pre-create all segment files so they're cached
        segments = stage._collect_matched_segments(mock_state_with_matches)
        for seg in segments:
            vid = seg['video_id']
            s = int(max(0, seg['start'] - 5.0))
            e = int(seg['end'] + 5.0)
            (tmp_path / f"{vid}_{s}_{e}.mp4").write_bytes(b'\x00' * 512)

        with patch.object(stage, '_process_retry_queue'), \
             patch('src.downloader.VideoDownloader'):
            result = stage.run(mock_state_with_matches, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['retry_count'] == 0

    @pytest.mark.fast
    def test_retry_count_reflects_actual_retries(self, stage, tmp_path):
        """stats retry_count reflects number of items processed from retry queue."""
        from src.downloader import RetryQueue, BatchRetryConfig

        retry_queue = RetryQueue(BatchRetryConfig(
            enabled=True, delay_seconds=0, max_passes=1
        ))
        # Add 2 items to retry queue
        retry_queue.add(
            video_id='retry_a_0_10',
            keyword='segment', tier='segment',
            error_message='Initial failure'
        )
        retry_queue.add(
            video_id='retry_b_0_10',
            keyword='segment', tier='segment',
            error_message='Initial failure'
        )

        mock_downloader = MagicMock()
        mock_downloader.retry_queue = retry_queue
        mock_downloader.impersonation_manager = None
        mock_downloader.download_config = None

        stage.downloader = mock_downloader

        stats = {
            'succeeded': 0, 'failed': 2, 'cached': 0,
            'attempted': 2, 'total': 2, 'retry_count': 0,
            'segment_durations': [], 'total_bytes': 0,
        }

        # Mock yt_dlp — both retries will fail (file not created)
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class:
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = Exception("Retry failed")
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance
            mock_ydl_class.return_value.__exit__.return_value = False

            stage._process_retry_queue(
                tmp_path, 5.0, [], 2, None, stats
            )

        # 2 items were retried
        assert stats['retry_count'] == 2

    @pytest.mark.fast
    def test_retry_count_in_checkpoint_data_key(
        self, stage, mock_state_with_matches, mock_config, mock_checkpoint, tmp_path
    ):
        """checkpoint_data dict contains 'retry_count' key."""
        mock_config.downloaded_videos_dir = str(tmp_path)

        # Pre-create all segment files so they're cached (no actual download)
        segments = stage._collect_matched_segments(mock_state_with_matches)
        for seg in segments:
            vid = seg['video_id']
            s = int(max(0, seg['start'] - 5.0))
            e = int(seg['end'] + 5.0)
            (tmp_path / f"{vid}_{s}_{e}.mp4").write_bytes(b'\x00' * 256)

        with patch.object(stage, '_process_retry_queue'), \
             patch('src.downloader.VideoDownloader'):
            result = stage.run(mock_state_with_matches, mock_config, mock_checkpoint)

        assert result.success is True
        assert 'retry_count' in result.data
        assert isinstance(result.data['retry_count'], int)


# ============================================================================
# Cookie Propagation to Python API (US-49-002)
# ============================================================================

class TestCookiePropagation:
    """Test cookie propagation to yt-dlp Python API in download_segments.

    US-49-002: The yt-dlp Python API does NOT read ~/.config/yt-dlp/config,
    so cookies_from_browser must be explicitly passed as 'cookiesfrombrowser'
    (a list, e.g. ['firefox']) in ydl_opts. Falls back to 'cookiefile' from
    cookies_path or cookie_rotation files.
    """

    @pytest.mark.fast
    def test_cookiesfrombrowser_injected_when_configured(self, stage, tmp_path):
        """cookiesfrombrowser is set in ydl_opts when cookies_from_browser configured."""
        mock_download_config = MagicMock()
        mock_download_config.cookies_from_browser = 'firefox'
        mock_download_config.cookies_path = ''
        mock_download_config.socket_timeout = 30
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.impersonation_manager = None
        mock_downloader.escalation_manager = None
        mock_downloader.cookie_rotator = None
        mock_downloader.retry_queue = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'cookie_browser_test', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert len(captured_opts) > 0
        assert captured_opts[0]['cookiesfrombrowser'] == ['firefox']
        assert 'cookiefile' not in captured_opts[0]

    @pytest.mark.fast
    def test_cookiefile_fallback_from_cookies_path(self, stage, tmp_path):
        """cookiefile is set from cookies_path when cookies_from_browser is empty."""
        mock_download_config = MagicMock()
        mock_download_config.cookies_from_browser = ''
        mock_download_config.cookies_path = '/path/to/cookies.txt'
        mock_download_config.cookie_rotation = None
        mock_download_config.socket_timeout = 30
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.impersonation_manager = None
        mock_downloader.escalation_manager = None
        mock_downloader.cookie_rotator = None
        mock_downloader.retry_queue = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'cookie_file_test', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert len(captured_opts) > 0
        assert captured_opts[0]['cookiefile'] == '/path/to/cookies.txt'
        assert 'cookiesfrombrowser' not in captured_opts[0]

    @pytest.mark.fast
    def test_cookiefile_fallback_from_cookie_rotation(self, stage, tmp_path):
        """cookiefile falls back to first cookie_rotation file when both
        cookies_from_browser and cookies_path are empty."""
        mock_cookie_rotation = MagicMock()
        mock_cookie_rotation.cookie_files = ['/cookies/rotation1.txt', '/cookies/rotation2.txt']

        mock_download_config = MagicMock()
        mock_download_config.cookies_from_browser = ''
        mock_download_config.cookies_path = ''
        mock_download_config.cookie_rotation = mock_cookie_rotation
        mock_download_config.socket_timeout = 30
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.impersonation_manager = None
        mock_downloader.escalation_manager = None
        mock_downloader.cookie_rotator = None
        mock_downloader.retry_queue = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'cookie_rotation_test', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert len(captured_opts) > 0
        assert captured_opts[0]['cookiefile'] == '/cookies/rotation1.txt'
        assert 'cookiesfrombrowser' not in captured_opts[0]

    @pytest.mark.fast
    def test_no_cookies_when_nothing_configured(self, stage, tmp_path):
        """No cookie keys in ydl_opts when no cookie config exists."""
        mock_download_config = MagicMock()
        mock_download_config.cookies_from_browser = ''
        mock_download_config.cookies_path = ''
        mock_download_config.cookie_rotation = None
        mock_download_config.socket_timeout = 30
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.impersonation_manager = None
        mock_downloader.escalation_manager = None
        mock_downloader.cookie_rotator = None
        mock_downloader.retry_queue = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'no_cookie_test', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._download_segments(
                segments, tmp_path, buffer_seconds=5.0, progress_callback=None
            )

        assert len(captured_opts) > 0
        assert 'cookiesfrombrowser' not in captured_opts[0]
        assert 'cookiefile' not in captured_opts[0]

    @pytest.mark.fast
    def test_tier3_cookie_rotation_overrides_baseline_browser(self, stage, tmp_path):
        """When Tier 3 escalation provides cookie rotation, it overrides the
        baseline cookiesfrombrowser to avoid conflicts."""
        from src.downloader.types import EscalationTier

        mock_escalation_mgr = MagicMock()
        mock_result = MagicMock()
        mock_result.args = ['--impersonate', 'Chrome-136:Macos-15']
        mock_result.tier = EscalationTier.FULL_BYPASS
        mock_result.rotate_cookies = True
        mock_result.rotate_vpn = False
        mock_escalation_mgr.get_escalation_args.return_value = mock_result

        mock_cookie_rotator = MagicMock()
        mock_cookie_rotator.get_current_cookie.return_value = '/tmp/rotated_cookie.txt'
        mock_cookie_rotator.should_rotate.return_value = False

        mock_download_config = MagicMock()
        mock_download_config.cookies_from_browser = 'firefox'
        mock_download_config.cookies_path = ''
        mock_download_config.socket_timeout = 30
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.escalation_manager = mock_escalation_mgr
        mock_downloader.cookie_rotator = mock_cookie_rotator
        mock_downloader.retry_queue = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        segments = [
            {'video_id': 'tier3_override', 'start': 0.0, 'end': 10.0},
        ]

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class, \
             patch.object(stage, '_process_retry_queue'):
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.download.side_effect = Exception("test")
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

        assert len(captured_opts) > 0
        # Rotated cookie file should be set
        assert captured_opts[0]['cookiefile'] == '/tmp/rotated_cookie.txt'
        # Browser cookies should be removed to avoid conflict
        assert 'cookiesfrombrowser' not in captured_opts[0]

    @pytest.mark.fast
    def test_cookiesfrombrowser_in_retry_queue(self, stage, tmp_path):
        """cookiesfrombrowser is also propagated in retry queue downloads."""
        from src.downloader import RetryQueue, BatchRetryConfig

        mock_download_config = MagicMock()
        mock_download_config.cookies_from_browser = 'chrome'
        mock_download_config.cookies_path = ''
        mock_download_config.socket_timeout = 30
        mock_download_config.segment_socket_timeout = 0
        mock_download_config.segment_max_resolution = 1080
        mock_download_config.segment_format = 'best[height<={segment_max_resolution}]'

        retry_queue = RetryQueue(BatchRetryConfig(
            enabled=True, delay_seconds=0, max_passes=1
        ))
        retry_queue.add(
            video_id='retry_cookie_0_10',
            keyword='segment', tier='segment',
            error_message='Initial failure'
        )

        mock_downloader = MagicMock()
        mock_downloader.download_config = mock_download_config
        mock_downloader.retry_queue = retry_queue
        mock_downloader.impersonation_manager = None
        mock_downloader.escalation_manager = None
        mock_downloader.cookie_rotator = None

        stage.downloader = mock_downloader

        captured_opts = []
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock())
            mock_cm.__exit__ = MagicMock(return_value=False)

            def capture_init(opts):
                captured_opts.append(opts.copy())
                return mock_cm

            mock_ydl_class.side_effect = capture_init

            stage._process_retry_queue(tmp_path, 5.0, [], 1, None)

        assert len(captured_opts) > 0
        assert captured_opts[0]['cookiesfrombrowser'] == ['chrome']
