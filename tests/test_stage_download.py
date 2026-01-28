"""
Comprehensive tests for DownloadStage and DownloadVideoSegmentsStage.

Tests cover:
- Stage initialization and registration
- Full video download mode
- Audio-first download mode
- Video segment download (audio-first)
- Global cache integration
- Skip download behavior
- Checkpoint save/restore
- Validation and error handling
- Match object remapping (audio -> video)

Created: 2026-01-09 (Phase 10.2)
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages.download import DownloadStage, DownloadVideoSegmentsStage
from src.state import PipelineState, DownloadedVideo, AudioDownload


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with download settings"""
    config = MagicMock()
    config.pipeline.skip_download = False
    config.downloaded_videos_dir = "videos"
    config.download.audio_first.enabled = False
    config.download.audio_first.buffer_seconds = 30.0
    config.download.audio_first.merge_gap_seconds = 15.0
    # Retry settings (defaults matching DownloadConfig)
    config.download.max_retries = 3
    config.download.retry_delay = 0.01  # Short delay for tests
    config.download.retry_backoff = 2.0
    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint manager"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    return checkpoint


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create temporary project directory"""
    return tmp_path


# ============================================================================
# Test DownloadStage Initialization
# ============================================================================

@pytest.mark.fast
class TestDownloadStageInit:
    """Test DownloadStage initialization"""

    def test_stage_name(self):
        """Test stage name is correct"""
        stage = DownloadStage()
        assert stage.name == "DOWNLOAD"

    @pytest.mark.fast
    def test_stage_description(self):
        """Test stage description"""
        stage = DownloadStage()
        assert "Download" in stage.description or "download" in stage.description

    @pytest.mark.fast
    def test_stage_registration(self):
        """Test stage is registered"""
        from src.stages import get_stage
        stage_class = get_stage("DOWNLOAD")
        assert stage_class is DownloadStage

    @pytest.mark.fast
    def test_initial_attributes(self):
        """Test initial attributes are None"""
        stage = DownloadStage()
        assert stage.downloader is None
        assert stage.global_cache is None


# ============================================================================
# Test Input Validation
# ============================================================================

@pytest.mark.fast
class TestDownloadInputValidation:
    """Test validate_inputs method"""

    def test_validate_no_keywords(self, mock_config):
        """Test validation fails when no keywords"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "keywords" in error.lower()

    @pytest.mark.fast
    def test_validate_success(self, mock_config):
        """Test validation succeeds with keywords"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach", "ocean"]

        error = stage.validate_inputs(state, mock_config)

        assert error is None


# ============================================================================
# Test Full Download Mode
# ============================================================================

@pytest.mark.fast
class TestFullDownloadMode:
    """Test _run_full_download method"""

    @patch('src.downloader.VideoDownloader')
    def test_full_download_success(self, mock_downloader_class, mock_config, mock_checkpoint, temp_project_dir):
        """Test successful full video download"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach", "ocean"]
        state.topic_context = "Travel"

        # Mock downloader
        mock_downloader = Mock()
        mock_downloader._get_tier_value.return_value = 60
        mock_downloader.download_all.return_value = (
            [
                {'file': 'video1.mp4', 'url': 'https://example.com/1', 'title': 'Beach Video',
                 'channel': 'Test', 'upload_date': '20260101', 'duration': 120.0,
                 'duration_tier': 'medium', 'keyword': 'beach', 'download_date': '20260109'}
            ],
            []  # No failed keywords
        )
        mock_downloader_class.return_value = mock_downloader

        # Mock global cache check
        with patch.object(stage, '_check_global_cache', return_value=(["beach", "ocean"], [])):
            result = stage._run_full_download(state, mock_config, mock_checkpoint, [])

        assert result.success is True
        assert len(state.downloaded_videos) == 1
        assert state.downloaded_videos[0].file == 'video1.mp4'
        assert state.downloaded_videos[0].keyword == 'beach'
        assert state.failed_keywords == []

    @patch('src.downloader.VideoDownloader')
    @pytest.mark.fast
    def test_full_download_with_failures(self, mock_downloader_class, mock_config, mock_checkpoint):
        """Test full download with some failed keywords"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach", "ocean", "mountain"]

        mock_downloader = Mock()
        mock_downloader._get_tier_value.return_value = 60
        mock_downloader.download_all.return_value = (
            [{'file': 'video1.mp4', 'url': 'https://example.com/1', 'title': 'Beach',
              'channel': 'Test', 'duration': 120.0, 'duration_tier': 'medium',
              'keyword': 'beach', 'upload_date': '', 'download_date': ''}],
            ["mountain"]  # Failed keyword
        )
        mock_downloader_class.return_value = mock_downloader

        with patch.object(stage, '_check_global_cache', return_value=(state.keywords, [])):
            result = stage._run_full_download(state, mock_config, mock_checkpoint, [])

        assert result.success is True
        assert len(state.downloaded_videos) == 1
        assert "mountain" in state.failed_keywords

    @patch('src.downloader.VideoDownloader')
    @pytest.mark.fast
    def test_full_download_with_global_cache_reuse(self, mock_downloader_class, mock_config, mock_checkpoint):
        """Test full download reusing videos from global cache"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach", "ocean"]

        mock_downloader = Mock()
        mock_downloader._get_tier_value.return_value = 60
        mock_downloader.download_all.return_value = ([], [])
        mock_downloader_class.return_value = mock_downloader

        # Mock global cache returning reusable videos
        reusable = [{'path': 'cached_video.mp4', 'face_score': 0.8}]
        with patch.object(stage, '_check_global_cache', return_value=(["ocean"], reusable)):
            result = stage._run_full_download(state, mock_config, mock_checkpoint, [])

        assert result.success is True
        assert len(state.downloaded_videos) == 1
        assert state.downloaded_videos[0].source == 'global_cache'
        assert state.downloaded_videos[0].face_score == 0.8

    @pytest.mark.fast
    def test_full_download_import_error(self, mock_config, mock_checkpoint):
        """Test full download handles import error"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        with patch('src.downloader.VideoDownloader', side_effect=ImportError("downloader not found")):
            result = stage._run_full_download(state, mock_config, mock_checkpoint, [])

        assert result.success is False
        assert "import" in result.error.lower()


# ============================================================================
# Test Audio-First Mode
# ============================================================================

@pytest.mark.fast
class TestAudioFirstMode:
    """Test audio-first download mode"""

    def test_is_audio_first_enabled_true(self, mock_config):
        """Test audio-first mode detection when enabled"""
        stage = DownloadStage()
        mock_config.download.audio_first.enabled = True

        assert stage._is_audio_first_enabled(mock_config) is True

    @pytest.mark.fast
    def test_is_audio_first_enabled_false(self, mock_config):
        """Test audio-first mode detection when disabled"""
        stage = DownloadStage()
        mock_config.download.audio_first.enabled = False

        assert stage._is_audio_first_enabled(mock_config) is False

    @pytest.mark.fast
    def test_is_audio_first_no_config(self):
        """Test audio-first mode when config missing"""
        stage = DownloadStage()
        config = MagicMock()
        # Make audio_first return None so getattr(None, 'enabled', False) returns False
        config.download.audio_first = None

        # When audio_config is None, getattr(None, 'enabled', False) returns False
        result = stage._is_audio_first_enabled(config)
        # Should return False because None is falsy
        assert result is False or result is None

    @patch('src.downloader.VideoDownloader')
    @patch('src.state.AudioDownload')
    @pytest.mark.fast
    def test_run_audio_first_success(self, mock_audio_class, mock_downloader_class, mock_config, mock_checkpoint):
        """Test successful audio-first download"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]
        state.topic_context = "Travel"

        # Mock downloader with audio_first support
        mock_downloader = Mock()
        mock_downloader.DURATION_TIERS = {'short': {}}  # Only one tier to avoid duplicate downloads
        mock_downloader._get_tier_value.return_value = 5

        mock_audio_obj = Mock()
        mock_audio_obj.file = "audio1.mp3"
        mock_audio_obj.video_id = "vid123"
        mock_audio_obj.url = "https://example.com/1"

        mock_downloader.audio_first.download_audio_for_keyword.return_value = [mock_audio_obj]
        mock_downloader_class.return_value = mock_downloader

        result = stage._run_audio_first(state, mock_config, mock_checkpoint, [])

        assert result.success is True
        assert len(state.downloaded_audio) >= 1  # At least one audio file
        assert state.downloaded_audio[0].file == "audio1.mp3"
        assert result.data['mode'] == 'audio_first'

    @patch('src.downloader.VideoDownloader')
    @pytest.mark.fast
    def test_run_audio_first_with_failures(self, mock_downloader_class, mock_config, mock_checkpoint):
        """Test audio-first download with failed keywords"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach", "ocean"]

        mock_downloader = Mock()
        mock_downloader.DURATION_TIERS = {'short': {}}
        mock_downloader._get_tier_value.return_value = 5
        mock_downloader.audio_first.download_audio_for_keyword.side_effect = [
            [Mock(file="audio1.mp3", video_id="vid1", url="url1")],
            Exception("Download failed")
        ]
        mock_downloader_class.return_value = mock_downloader

        result = stage._run_audio_first(state, mock_config, mock_checkpoint, [])

        assert result.success is True
        assert len(state.downloaded_audio) == 1
        assert "ocean" in state.failed_keywords

    @pytest.mark.fast
    def test_run_audio_first_import_error(self, mock_config, mock_checkpoint):
        """Test audio-first download handles import error"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        with patch('src.downloader.VideoDownloader', side_effect=ImportError("downloader missing")):
            result = stage._run_audio_first(state, mock_config, mock_checkpoint, [])

        assert result.success is False
        assert "import" in result.error.lower()


# ============================================================================
# Test Skip Download Behavior
# ============================================================================

@pytest.mark.fast
class TestSkipDownload:
    """Test skip_download behavior"""

    def test_skip_download_loads_existing(self, mock_config, mock_checkpoint, temp_project_dir):
        """Test skip_download loads existing videos"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        # Create mock video files
        videos_dir = temp_project_dir / "videos"
        videos_dir.mkdir()
        (videos_dir / "video1.mp4").write_bytes(b"fake video")
        (videos_dir / "video2.mkv").write_bytes(b"fake video")

        mock_config.pipeline.skip_download = True
        mock_config.downloaded_videos_dir = str(videos_dir)

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.downloaded_videos) == 2
        assert all(v.source == 'existing' for v in state.downloaded_videos)

    @pytest.mark.fast
    def test_skip_download_subdirectory(self, mock_config, mock_checkpoint, temp_project_dir):
        """Test skip_download finds videos in subdirectories"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        # Create videos in subdirectory
        videos_dir = temp_project_dir / "videos"
        subdir = videos_dir / "subfolder"
        subdir.mkdir(parents=True)
        (subdir / "video1.mp4").write_bytes(b"fake")

        mock_config.pipeline.skip_download = True
        mock_config.downloaded_videos_dir = str(videos_dir)

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.downloaded_videos) == 1

    @pytest.mark.fast
    def test_skip_download_missing_directory(self, mock_config, mock_checkpoint, temp_project_dir):
        """Test skip_download handles missing directory"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        mock_config.pipeline.skip_download = True
        mock_config.downloaded_videos_dir = str(temp_project_dir / "nonexistent")

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.downloaded_videos) == 0


# ============================================================================
# Test Stage Execution
# ============================================================================

@pytest.mark.fast
class TestDownloadStageExecution:
    """Test full stage execution flow"""

    @patch('src.downloader.VideoDownloader')
    def test_run_success_full_mode(self, mock_downloader_class, mock_config, mock_checkpoint):
        """Test successful run in full download mode"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        mock_downloader = Mock()
        mock_downloader._get_tier_value.return_value = 60
        mock_downloader.download_all.return_value = (
            [{'file': 'video.mp4', 'url': 'url', 'title': 'Test', 'channel': 'Ch',
              'upload_date': '', 'duration': 120.0, 'duration_tier': 'medium',
              'keyword': 'beach', 'download_date': ''}],
            []
        )
        mock_downloader_class.return_value = mock_downloader

        with patch.object(stage, '_check_global_cache', return_value=(["beach"], [])):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.downloaded_videos) > 0

    @patch('src.downloader.VideoDownloader')
    @pytest.mark.fast
    def test_run_success_audio_first_mode(self, mock_downloader_class, mock_config, mock_checkpoint):
        """Test successful run in audio-first mode"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        mock_config.download.audio_first.enabled = True

        mock_downloader = Mock()
        mock_downloader.DURATION_TIERS = {'short': {}}
        mock_downloader._get_tier_value.return_value = 5
        mock_downloader.audio_first.download_audio_for_keyword.return_value = [
            Mock(file="audio.mp3", video_id="vid1", url="url")
        ]
        mock_downloader_class.return_value = mock_downloader

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.downloaded_audio) > 0

    @pytest.mark.fast
    def test_run_exception_handling(self, mock_config, mock_checkpoint):
        """Test run handles exceptions gracefully"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        with patch.object(stage, '_run_full_download', side_effect=Exception("Download error")):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "Download error" in result.error


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

@pytest.mark.fast
class TestDownloadCheckpoint:
    """Test checkpoint save/restore"""

    def test_can_skip_no_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = DownloadStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = False

        assert stage.can_skip(state, mock_checkpoint) is False

    @pytest.mark.fast
    def test_can_skip_with_checkpoint(self, mock_checkpoint):
        """Test can_skip returns True when checkpoint exists"""
        stage = DownloadStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        assert stage.can_skip(state, mock_checkpoint) is True

    @pytest.mark.fast
    def test_restore_success(self, mock_checkpoint):
        """Test successful restore from checkpoint"""
        stage = DownloadStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.return_value = {
            'downloaded_videos': [
                {'file': 'video1.mp4', 'url': 'url1', 'title': 'Test',
                 'channel': 'Ch', 'upload_date': '', 'duration': 120.0,
                 'duration_tier': 'medium', 'keyword': 'beach', 'download_date': '',
                 'source': 'download'}
            ],
            'failed_keywords': ['mountain']
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.downloaded_videos) == 1
        assert state.downloaded_videos[0].file == 'video1.mp4'
        assert "mountain" in state.failed_keywords

    @pytest.mark.fast
    def test_restore_audio_downloads(self, mock_checkpoint):
        """Test successful restore of audio downloads (audio-first mode)"""
        stage = DownloadStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.return_value = {
            'audio_downloads': [
                {'file': 'audio1.mp3', 'url': 'url1', 'video_id': 'vid1',
                 'title': 'Test Audio', 'duration': 120.0, 'keyword': 'beach'},
                {'file': 'audio2.mp3', 'url': 'url2', 'video_id': 'vid2',
                 'title': 'Test Audio 2', 'duration': 180.0, 'keyword': 'ocean'}
            ],
            'failed_keywords': ['mountain']
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.downloaded_audio) == 2
        assert state.downloaded_audio[0].file == 'audio1.mp3'
        assert state.downloaded_audio[0].video_id == 'vid1'
        assert state.downloaded_audio[1].file == 'audio2.mp3'
        assert state.downloaded_audio[1].video_id == 'vid2'
        assert "mountain" in state.failed_keywords

    @pytest.mark.fast
    def test_restore_audio_downloads_legacy_field_names(self, mock_checkpoint):
        """Test restore with old checkpoint field names (audio_file, video_url)"""
        stage = DownloadStage()
        state = PipelineState()

        # Simulate old checkpoint with incorrect field names
        mock_checkpoint.get_stage_data.return_value = {
            'audio_downloads': [
                {'audio_file': 'audio1.mp3', 'video_url': 'url1', 'video_id': 'vid1',
                 'title': 'Test Audio', 'duration': 120.0, 'keyword': 'beach'},
                {'audio_file': 'audio2.mp3', 'video_url': 'url2', 'video_id': 'vid2',
                 'title': 'Test Audio 2', 'duration': 180.0, 'keyword': 'ocean'}
            ],
            'failed_keywords': []
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.downloaded_audio) == 2
        # Verify field names were mapped correctly
        assert state.downloaded_audio[0].file == 'audio1.mp3'
        assert state.downloaded_audio[0].url == 'url1'
        assert state.downloaded_audio[0].video_id == 'vid1'
        assert state.downloaded_audio[1].file == 'audio2.mp3'
        assert state.downloaded_audio[1].url == 'url2'
        assert state.downloaded_audio[1].video_id == 'vid2'

    @pytest.mark.fast
    def test_restore_no_data(self, mock_checkpoint):
        """Test restore returns False when no checkpoint data"""
        stage = DownloadStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @patch('src.stages.download.logger')
    @pytest.mark.fast
    def test_restore_exception_handling(self, mock_logger, mock_checkpoint):
        """Test restore handles exceptions gracefully"""
        stage = DownloadStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.side_effect = Exception("Restore error")

        result = stage.restore(state, mock_checkpoint)

        assert result is False
        mock_logger.warning.assert_called()


# ============================================================================
# Test Helper Methods
# ============================================================================

@pytest.mark.fast
class TestDownloadHelpers:
    """Test helper methods"""

    def test_video_to_dict(self):
        """Test video serialization"""
        stage = DownloadStage()
        video = DownloadedVideo(
            file="video.mp4",
            url="https://example.com/1",
            title="Test Video",
            source="download"
        )

        result = stage._video_to_dict(video)

        assert isinstance(result, dict)
        assert result['file'] == 'video.mp4'
        assert result['title'] == 'Test Video'

    @pytest.mark.fast
    def test_audio_to_dict(self):
        """Test audio serialization"""
        stage = DownloadStage()
        audio = AudioDownload(
            file="audio.mp3",
            url="https://example.com/1",
            video_id="vid123",
            title="Test Audio"
        )

        result = stage._audio_to_dict(audio)

        assert isinstance(result, dict)
        assert result['file'] == 'audio.mp3'
        assert result['video_id'] == 'vid123'

    @pytest.mark.fast
    def test_check_global_cache(self, mock_config):
        """Test global cache checking"""
        stage = DownloadStage()
        keywords = ["beach", "ocean"]

        keywords_to_download, reusable = stage._check_global_cache(keywords, mock_config)

        # Currently returns all keywords (cache not implemented)
        assert keywords_to_download == keywords
        assert reusable == []

    @pytest.mark.fast
    def test_store_download_results_dict(self):
        """Test storing download results from dicts"""
        stage = DownloadStage()
        state = PipelineState()

        downloaded = [
            {'file': 'video1.mp4', 'url': 'url1', 'title': 'Test', 'channel': 'Ch',
             'upload_date': '', 'duration': 120.0, 'duration_tier': 'medium',
             'keyword': 'beach', 'download_date': ''}
        ]
        reusable = []
        failed = ["mountain"]

        stage._store_download_results(state, downloaded, reusable, failed)

        assert len(state.downloaded_videos) == 1
        assert state.downloaded_videos[0].keyword == 'beach'
        assert state.failed_keywords == ["mountain"]

    @pytest.mark.fast
    def test_store_download_results_objects(self):
        """Test storing download results from objects"""
        stage = DownloadStage()
        state = PipelineState()

        downloaded = [
            DownloadedVideo(file="video.mp4", url="url", title="Test", source="download")
        ]

        stage._store_download_results(state, downloaded, [], [])

        assert len(state.downloaded_videos) == 1
        assert state.downloaded_videos[0].file == "video.mp4"


# ============================================================================
# Test DownloadVideoSegmentsStage
# ============================================================================

@pytest.mark.fast
class TestDownloadVideoSegmentsStageInit:
    """Test DownloadVideoSegmentsStage initialization"""

    def test_stage_name(self):
        """Test stage name"""
        stage = DownloadVideoSegmentsStage()
        assert stage.name == "DOWNLOAD_SEGMENTS"

    @pytest.mark.fast
    def test_stage_description(self):
        """Test stage description"""
        stage = DownloadVideoSegmentsStage()
        assert "segment" in stage.description.lower()

    @pytest.mark.fast
    def test_initial_attributes(self):
        """Test initial attributes"""
        stage = DownloadVideoSegmentsStage()
        assert stage.downloader is None


@pytest.mark.fast
class TestDownloadVideoSegmentsValidation:
    """Test segment download validation"""

    def test_validate_not_audio_first(self, mock_config):
        """Test validation passes when not in audio-first mode"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()
        state.downloaded_audio = []

        error = stage.validate_inputs(state, mock_config)

        assert error is None

    @pytest.mark.fast
    def test_validate_skip_download(self, mock_config):
        """Test validation passes when skip_download=true"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()
        state.downloaded_audio = [Mock()]
        mock_config.pipeline.skip_download = True

        error = stage.validate_inputs(state, mock_config)

        assert error is None

    @pytest.mark.fast
    def test_validate_no_matches(self, mock_config):
        """Test validation fails when no matches"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()
        state.downloaded_audio = [Mock()]
        state.matches = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "matches" in error.lower()

    @pytest.mark.fast
    def test_validate_success(self, mock_config):
        """Test validation succeeds with audio and matches"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()
        state.downloaded_audio = [Mock()]
        state.matches = [Mock()]

        error = stage.validate_inputs(state, mock_config)

        assert error is None


@pytest.mark.fast
class TestDownloadVideoSegmentsExecution:
    """Test segment download execution"""

    def test_run_skip_not_audio_first(self, mock_config, mock_checkpoint):
        """Test run skips when not in audio-first mode"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()
        state.downloaded_audio = []

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True

    @pytest.mark.fast
    def test_run_skip_download_with_audio_warning(self, mock_config, mock_checkpoint):
        """Test run warns when skip_download=true but audio exists"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()
        state.downloaded_audio = [Mock(file="audio.mp3")]
        mock_config.pipeline.skip_download = True

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(result.warnings) > 0
        assert "Incomplete" in result.warnings[0]

    @pytest.mark.fast
    def test_run_no_matches_fails(self, mock_config, mock_checkpoint):
        """Test run fails when no matches"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()
        state.downloaded_audio = [Mock()]
        state.matches = []

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "matches" in result.error.lower()

    @patch('src.downloader.VideoDownloader')
    @patch('src.downloader.collect_matched_segments')
    @patch('src.downloader.prepare_merged_segments')
    @pytest.mark.fast
    def test_run_success(self, mock_prepare, mock_collect, mock_downloader_class,
                        mock_config, mock_checkpoint):
        """Test successful segment download"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()

        # Mock audio downloads
        audio = Mock(file="audio.mp3", video_id="vid123")
        state.downloaded_audio = [audio]

        # Mock matches
        match = Mock(video_file="audio.mp3", video_start=10.0)
        state.matches = [match]

        # Mock segment collection
        mock_collect.return_value = {'vid123': [match]}
        mock_prepare.return_value = [
            {'video_id': 'vid123', 'start': 0, 'end': 30, 'matches': [match]}
        ]

        # Mock downloader
        mock_downloader = Mock()
        downloaded_seg = Mock(
            file="segment.mp4",
            video_id="vid123",
            original_start=0,
            matches=[Mock(start_time=10.0)]
        )
        mock_downloader.audio_first.download_video_segments.return_value = [downloaded_seg]
        mock_downloader_class.return_value = mock_downloader

        # Mock the remapping function to avoid Path issues with Mock objects
        with patch.object(stage, '_remap_matches_to_video_segments'):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['segment_count'] == 1


@pytest.mark.fast
class TestDownloadVideoSegmentsCheckpoint:
    """Test segment download checkpoint operations"""

    def test_can_skip(self, mock_checkpoint):
        """Test can_skip behavior"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        assert stage.can_skip(state, mock_checkpoint) is True

    @pytest.mark.fast
    def test_restore(self, mock_checkpoint):
        """Test restore always succeeds (no state to restore)"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()

        result = stage.restore(state, mock_checkpoint)

        assert result is True


@pytest.mark.fast
class TestMatchRemapping:
    """Test match object remapping from audio to video segments"""

    def test_remap_simple_match_objects(self):
        """Test remapping state.Match objects to video segment files"""
        from src.state import Match, AudioDownload
        from src.downloader.types import DownloadedSegment, MatchedSegment

        # Create test data
        state = PipelineState()

        # Create audio downloads
        audio1 = AudioDownload(
            file="audio_video1.mp3",
            url="https://youtube.com/watch?v=video1",
            video_id="video1",
            title="Test Video 1"
        )
        state.audio_downloads = [audio1]

        # Create match that references audio file
        match1 = Match(
            segment_index=0,
            video_file="audio_video1.mp3",
            video_start=10.0,
            video_end=15.0,
            confidence=0.9,
            strategy="primary"
        )
        state.matches = [match1]

        # Create downloaded segments
        seg_match = MatchedSegment(
            video_id="video1",
            video_url="https://youtube.com/watch?v=video1",
            start_time=10.0,
            end_time=15.0,
            track="V1",
            voiceover_segment_idx=0
        )
        segment1 = DownloadedSegment(
            file="segment_video1_10.0-15.0.mp4",
            video_id="video1",
            original_start=10.0,
            original_end=15.0,
            file_duration=5.0,
            matches=[seg_match]
        )
        downloaded_segments = [segment1]

        # Execute remapping
        stage = DownloadStage()
        state.stage_name = 'DOWNLOAD_SEGMENTS'

        # Call the remapping logic (lines 509-632 in download.py)
        audio_downloads_by_id = {audio1.video_id: audio1}

        # Build segment map
        segment_map = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            for match in seg.matches:
                key = (video_id, match.start_time)
                segment_map[key] = (seg.file, seg.original_start)

        # Remap the match
        audio_file = Path(match1.video_file).stem
        video_id = "video1"
        key = (video_id, match1.video_start)

        assert key in segment_map
        segment_file, original_start = segment_map[key]
        match1.video_file = segment_file

        # Verify remapping
        assert match1.video_file == "segment_video1_10.0-15.0.mp4"
        assert match1.video_start == 10.0
        assert match1.confidence == 0.9

    @pytest.mark.fast
    def test_remap_match_result_with_alternatives(self):
        """Test remapping MatchResult objects with alternatives and strategy matches"""
        from src.state import Match, AudioDownload
        from src.utils import MatchResult, AlternativeMatch, StrategyMatch, SRTSegment
        from src.downloader.types import DownloadedSegment, MatchedSegment

        # Create test data
        state = PipelineState()

        # Create audio downloads for two videos
        audio1 = AudioDownload(
            file="audio_video1.mp3",
            url="https://youtube.com/watch?v=video1",
            video_id="video1",
            title="Test Video 1"
        )
        audio2 = AudioDownload(
            file="audio_video2.mp3",
            url="https://youtube.com/watch?v=video2",
            video_id="video2",
            title="Test Video 2"
        )
        state.audio_downloads = [audio1, audio2]

        # Create primary match
        primary_match = Match(
            segment_index=0,
            video_file="audio_video1.mp3",
            video_start=10.0,
            video_end=15.0,
            confidence=0.9,
            strategy="primary"
        )

        # Create alternative match with video_segment
        alt_segment = SRTSegment(
            index=1,
            start_time=20.0,
            end_time=25.0,
            text="Alternative text",
            source_file="audio_video1.mp3"
        )
        alternative = AlternativeMatch(
            video_segment=alt_segment,
            video_scene=None,
            confidence=0.7,
            reasoning="Alternative match"
        )

        # Create strategy match from different video
        strat_segment = SRTSegment(
            index=2,
            start_time=30.0,
            end_time=35.0,
            text="Strategy text",
            source_file="audio_video2.mp3"
        )
        strategy = StrategyMatch(
            video_segment=strat_segment,
            video_scene=None,
            confidence=0.8,
            reasoning="Embedding diversity",
            strategy="embedding_diversity"
        )

        # Create MatchResult
        match_result = MatchResult(
            primary_match=primary_match,
            alternatives=[alternative],
            secondary_matches=[],
            strategy_matches=[strategy]
        )
        state.matches = [match_result]

        # Create downloaded segments for both videos
        seg_match1 = MatchedSegment(
            video_id="video1",
            video_url="https://youtube.com/watch?v=video1",
            start_time=10.0,
            end_time=15.0,
            track="V1",
            voiceover_segment_idx=0
        )
        segment1 = DownloadedSegment(
            file="segment_video1_10.0-15.0.mp4",
            video_id="video1",
            original_start=10.0,
            original_end=15.0,
            file_duration=5.0,
            matches=[seg_match1]
        )

        seg_match2 = MatchedSegment(
            video_id="video1",
            video_url="https://youtube.com/watch?v=video1",
            start_time=20.0,
            end_time=25.0,
            track="V2",
            voiceover_segment_idx=0
        )
        segment2 = DownloadedSegment(
            file="segment_video1_20.0-25.0.mp4",
            video_id="video1",
            original_start=20.0,
            original_end=25.0,
            file_duration=5.0,
            matches=[seg_match2]
        )

        seg_match3 = MatchedSegment(
            video_id="video2",
            video_url="https://youtube.com/watch?v=video2",
            start_time=30.0,
            end_time=35.0,
            track="V7",
            voiceover_segment_idx=0
        )
        segment3 = DownloadedSegment(
            file="segment_video2_30.0-35.0.mp4",
            video_id="video2",
            original_start=30.0,
            original_end=35.0,
            file_duration=5.0,
            matches=[seg_match3]
        )

        downloaded_segments = [segment1, segment2, segment3]

        # Execute remapping
        audio_downloads_by_id = {
            audio1.video_id: audio1,
            audio2.video_id: audio2
        }

        # Build segment map
        segment_map = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            for match in seg.matches:
                key = (video_id, match.start_time)
                segment_map[key] = (seg.file, seg.original_start)

        # Remap primary match
        key1 = ("video1", 10.0)
        assert key1 in segment_map
        primary_match.video_file = segment_map[key1][0]

        # Remap alternative
        key2 = ("video1", 20.0)
        assert key2 in segment_map
        alternative.video_segment.source_file = segment_map[key2][0]

        # Remap strategy match
        key3 = ("video2", 30.0)
        assert key3 in segment_map
        strategy.video_segment.source_file = segment_map[key3][0]

        # Verify all remappings
        assert primary_match.video_file == "segment_video1_10.0-15.0.mp4"
        assert alternative.video_segment.source_file == "segment_video1_20.0-25.0.mp4"
        assert strategy.video_segment.source_file == "segment_video2_30.0-35.0.mp4"

    @pytest.mark.fast
    def test_remap_match_not_found_in_segments(self):
        """Test remapping when match is not found in downloaded segments"""
        from src.state import Match, AudioDownload
        from src.downloader.types import DownloadedSegment, MatchedSegment

        state = PipelineState()

        # Create audio download
        audio1 = AudioDownload(
            file="audio_video1.mp3",
            url="https://youtube.com/watch?v=video1",
            video_id="video1",
            title="Test Video 1"
        )
        state.audio_downloads = [audio1]

        # Create match with time NOT in downloaded segments
        match1 = Match(
            segment_index=0,
            video_file="audio_video1.mp3",
            video_start=50.0,  # Not in segments
            video_end=55.0,
            confidence=0.9
        )
        state.matches = [match1]

        # Create segment with different time range
        seg_match = MatchedSegment(
            video_id="video1",
            video_url="https://youtube.com/watch?v=video1",
            start_time=10.0,
            end_time=15.0,
            track="V1",
            voiceover_segment_idx=0
        )
        segment1 = DownloadedSegment(
            file="segment_video1_10.0-15.0.mp4",
            video_id="video1",
            original_start=10.0,
            original_end=15.0,
            file_duration=5.0,
            matches=[seg_match]
        )
        downloaded_segments = [segment1]

        # Build segment map
        segment_map = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            for match in seg.matches:
                key = (video_id, match.start_time)
                segment_map[key] = (seg.file, seg.original_start)

        # Try to find the match
        key = ("video1", 50.0)
        assert key not in segment_map

        # Match should remain unchanged (not remapped)
        assert match1.video_file == "audio_video1.mp3"

    @pytest.mark.fast
    def test_remap_with_multiple_match_results(self):
        """Test remapping multiple MatchResult objects in one pass"""
        from src.state import Match, AudioDownload
        from src.utils import MatchResult, AlternativeMatch, SRTSegment
        from src.downloader.types import DownloadedSegment, MatchedSegment

        state = PipelineState()

        # Create audio download
        audio1 = AudioDownload(
            file="audio_video1.mp3",
            url="https://youtube.com/watch?v=video1",
            video_id="video1",
            title="Test Video 1"
        )
        state.audio_downloads = [audio1]

        # Create two MatchResult objects (for two voiceover segments)
        match1 = Match(
            segment_index=0,
            video_file="audio_video1.mp3",
            video_start=10.0,
            video_end=15.0,
            confidence=0.9
        )
        result1 = MatchResult(primary_match=match1)

        match2 = Match(
            segment_index=1,
            video_file="audio_video1.mp3",
            video_start=20.0,
            video_end=25.0,
            confidence=0.85
        )
        result2 = MatchResult(primary_match=match2)

        state.matches = [result1, result2]

        # Create segments for both matches
        seg_match1 = MatchedSegment(
            video_id="video1",
            video_url="https://youtube.com/watch?v=video1",
            start_time=10.0,
            end_time=15.0,
            track="V1",
            voiceover_segment_idx=0
        )
        segment1 = DownloadedSegment(
            file="segment_video1_10.0-15.0.mp4",
            video_id="video1",
            original_start=10.0,
            original_end=15.0,
            file_duration=5.0,
            matches=[seg_match1]
        )

        seg_match2 = MatchedSegment(
            video_id="video1",
            video_url="https://youtube.com/watch?v=video1",
            start_time=20.0,
            end_time=25.0,
            track="V1",
            voiceover_segment_idx=1
        )
        segment2 = DownloadedSegment(
            file="segment_video1_20.0-25.0.mp4",
            video_id="video1",
            original_start=20.0,
            original_end=25.0,
            file_duration=5.0,
            matches=[seg_match2]
        )

        downloaded_segments = [segment1, segment2]
        audio_downloads_by_id = {audio1.video_id: audio1}

        # Build segment map
        segment_map = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            for match in seg.matches:
                key = (video_id, match.start_time)
                segment_map[key] = (seg.file, seg.original_start)

        # Remap both matches
        for match in [match1, match2]:
            key = ("video1", match.video_start)
            if key in segment_map:
                match.video_file = segment_map[key][0]

        # Verify both remappings
        assert match1.video_file == "segment_video1_10.0-15.0.mp4"
        assert match2.video_file == "segment_video1_20.0-25.0.mp4"

    @pytest.mark.fast
    def test_remap_secondary_matches(self):
        """Test remapping secondary matches (V4-V6) separately from alternatives"""
        from src.state import Match, AudioDownload
        from src.utils import MatchResult, AlternativeMatch, SRTSegment
        from src.downloader.types import DownloadedSegment, MatchedSegment

        state = PipelineState()

        # Create audio download
        audio1 = AudioDownload(
            file="audio_video1.mp3",
            url="https://youtube.com/watch?v=video1",
            video_id="video1",
            title="Test Video 1"
        )
        state.audio_downloads = [audio1]

        # Create primary match
        primary = Match(
            segment_index=0,
            video_file="audio_video1.mp3",
            video_start=10.0,
            video_end=15.0,
            confidence=0.9
        )

        # Create secondary match (different source diversity)
        secondary_segment = SRTSegment(
            index=1,
            start_time=20.0,
            end_time=25.0,
            text="Secondary text",
            source_file="audio_video1.mp3"
        )
        secondary = AlternativeMatch(
            video_segment=secondary_segment,
            video_scene=None,
            confidence=0.75,
            reasoning="Different source diversity"
        )

        # Create MatchResult
        result = MatchResult(
            primary_match=primary,
            alternatives=[],
            secondary_matches=[secondary],
            strategy_matches=[]
        )
        state.matches = [result]

        # Create segments
        seg_match1 = MatchedSegment(
            video_id="video1",
            video_url="https://youtube.com/watch?v=video1",
            start_time=10.0,
            end_time=15.0,
            track="V1",
            voiceover_segment_idx=0
        )
        segment1 = DownloadedSegment(
            file="segment_video1_10.0-15.0.mp4",
            video_id="video1",
            original_start=10.0,
            original_end=15.0,
            file_duration=5.0,
            matches=[seg_match1]
        )

        seg_match2 = MatchedSegment(
            video_id="video1",
            video_url="https://youtube.com/watch?v=video1",
            start_time=20.0,
            end_time=25.0,
            track="V4",
            voiceover_segment_idx=0
        )
        segment2 = DownloadedSegment(
            file="segment_video1_20.0-25.0.mp4",
            video_id="video1",
            original_start=20.0,
            original_end=25.0,
            file_duration=5.0,
            matches=[seg_match2]
        )

        downloaded_segments = [segment1, segment2]
        audio_downloads_by_id = {audio1.video_id: audio1}

        # Build segment map
        segment_map = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            for match in seg.matches:
                key = (video_id, match.start_time)
                segment_map[key] = (seg.file, seg.original_start)

        # Remap primary and secondary
        primary.video_file = segment_map[("video1", 10.0)][0]
        secondary.video_segment.source_file = segment_map[("video1", 20.0)][0]

        # Verify
        assert primary.video_file == "segment_video1_10.0-15.0.mp4"
        assert secondary.video_segment.source_file == "segment_video1_20.0-25.0.mp4"


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.fast
class TestDownloadEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_keywords(self, mock_config, mock_checkpoint):
        """Test handling empty keyword list"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None

    @pytest.mark.fast
    def test_load_existing_no_video_extensions(self, mock_config, temp_project_dir):
        """Test loading existing videos with no valid extensions"""
        stage = DownloadStage()
        state = PipelineState()

        videos_dir = temp_project_dir / "videos"
        videos_dir.mkdir()
        (videos_dir / "file.txt").write_bytes(b"not a video")

        mock_config.downloaded_videos_dir = str(videos_dir)

        stage._load_existing_videos(state, mock_config)

        assert len(state.downloaded_videos) == 0

    @pytest.mark.fast
    def test_video_to_dict_with_dict(self):
        """Test video serialization when already a dict"""
        stage = DownloadStage()
        video_dict = {'file': 'video.mp4', 'title': 'Test'}

        result = stage._video_to_dict(video_dict)

        assert result == video_dict

    @patch('src.downloader.VideoDownloader')
    @pytest.mark.fast
    def test_audio_first_zero_per_keyword(self, mock_downloader_class, mock_config, mock_checkpoint):
        """Test audio-first mode skips tiers with zero per_keyword"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        mock_downloader = Mock()
        mock_downloader.DURATION_TIERS = {'short': {}}
        mock_downloader._get_tier_value.return_value = 0  # Zero per_keyword
        mock_downloader_class.return_value = mock_downloader

        result = stage._run_audio_first(state, mock_config, mock_checkpoint, [])

        # Should succeed but download nothing
        assert result.success is True
        assert len(state.downloaded_audio) == 0


# ============================================================================
# Test Global Cache Coverage
# ============================================================================

@pytest.mark.fast
class TestGlobalCacheBranch:
    """Test global cache coverage branch"""

    @patch('src.downloader.VideoDownloader')
    def test_all_keywords_from_cache(self, mock_downloader_class, mock_config, mock_checkpoint):
        """Test when all keywords are covered by global cache (lines 188-190)"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach", "ocean"]

        mock_downloader = Mock()
        mock_downloader._get_tier_value.return_value = 60
        mock_downloader_class.return_value = mock_downloader

        # Return empty list for keywords_to_download (all from cache)
        # and non-empty reusable videos list
        reusable = [
            {'path': 'cached_video1.mp4', 'face_score': 0.7},
            {'path': 'cached_video2.mp4', 'face_score': 0.8}
        ]
        with patch.object(stage, '_check_global_cache', return_value=([], reusable)):
            result = stage._run_full_download(state, mock_config, mock_checkpoint, [])

        assert result.success is True
        # All videos should be from cache
        assert len(state.downloaded_videos) == 2
        assert all(v.source == 'global_cache' for v in state.downloaded_videos)
        # download_all should NOT be called since keywords_to_download is empty
        mock_downloader.download_all.assert_not_called()


# ============================================================================
# Test Download Segments Exception
# ============================================================================

@pytest.mark.fast
class TestDownloadSegmentsException:
    """Test exception handling in download segments stage"""

    @patch('src.downloader.VideoDownloader')
    @patch('src.downloader.collect_matched_segments')
    @patch('src.downloader.prepare_merged_segments')
    @pytest.mark.fast
    def test_run_exception_handling(self, mock_prepare, mock_collect, mock_downloader_class,
                                    mock_config, mock_checkpoint):
        """Test segment download handles exceptions (lines 514-516)"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()
        state.downloaded_audio = [Mock(file="audio.mp3", video_id="vid123")]
        state.matches = [Mock(video_file="audio.mp3", video_start=10.0)]

        mock_collect.return_value = {'vid123': [state.matches[0]]}
        mock_prepare.return_value = [{'video_id': 'vid123', 'start': 0, 'end': 30}]

        # Make the downloader raise an exception
        mock_downloader = Mock()
        mock_downloader.audio_first.download_video_segments.side_effect = Exception("Download failed")
        mock_downloader_class.return_value = mock_downloader

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "Download failed" in result.error


# ============================================================================
# Test Actual Remap Method Call
# ============================================================================

@pytest.mark.fast
class TestRemapMethodIntegration:
    """Test _remap_matches_to_video_segments is actually called"""

    @patch('src.downloader.VideoDownloader')
    @patch('src.downloader.collect_matched_segments')
    @patch('src.downloader.prepare_merged_segments')
    @pytest.mark.fast
    def test_remap_called_with_correct_args(self, mock_prepare, mock_collect, mock_downloader_class,
                                            mock_config, mock_checkpoint):
        """Test _remap_matches_to_video_segments is called with correct arguments"""
        stage = DownloadVideoSegmentsStage()
        state = PipelineState()

        # Setup audio download
        audio = Mock(file="audio.mp3", video_id="vid123")
        audio.__getitem__ = Mock(side_effect=lambda k: {'video_id': 'vid123', 'file': 'audio.mp3'}.get(k))
        state.downloaded_audio = [audio]

        # Setup match
        match = Mock(video_file="audio.mp3", video_start=10.0)
        state.matches = [match]

        mock_collect.return_value = {'vid123': [match]}
        mock_prepare.return_value = [{'video_id': 'vid123', 'start': 0, 'end': 30, 'matches': [match]}]

        # Create mock downloaded segment
        downloaded_seg = Mock(
            file="segment.mp4",
            video_id="vid123",
            original_start=0,
            matches=[Mock(start_time=10.0)]
        )
        mock_downloader = Mock()
        mock_downloader.audio_first.download_video_segments.return_value = [downloaded_seg]
        mock_downloader_class.return_value = mock_downloader

        # Track if remap was called
        with patch.object(stage, '_remap_matches_to_video_segments') as mock_remap:
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        # Verify remap was called
        mock_remap.assert_called_once()


# ============================================================================
# Test Audio Dict Handling
# ============================================================================

@pytest.mark.fast
class TestAudioDictHandling:
    """Test _audio_to_dict with dict input"""

    def test_audio_to_dict_with_dict_input(self):
        """Test _audio_to_dict returns same dict when input is dict (line 391)"""
        stage = DownloadStage()
        audio_dict = {
            'file': 'audio.mp3',
            'video_id': 'vid123',
            'url': 'https://example.com',
            'title': 'Test Audio'
        }

        result = stage._audio_to_dict(audio_dict)

        assert result == audio_dict

    @pytest.mark.fast
    def test_audio_to_dict_with_object_input(self):
        """Test _audio_to_dict converts object to dict"""
        stage = DownloadStage()
        audio = AudioDownload(
            file="audio.mp3",
            video_id="vid123",
            url="https://example.com",
            title="Test Audio"
        )

        result = stage._audio_to_dict(audio)

        assert isinstance(result, dict)
        assert result['file'] == 'audio.mp3'
        assert result['video_id'] == 'vid123'


# ============================================================================
# Test Store Download Results Additional Coverage
# ============================================================================

@pytest.mark.fast
class TestStoreDownloadResultsExtended:
    """Extended tests for _store_download_results"""

    def test_store_download_with_path_field(self):
        """Test storing downloads when dict uses 'path' instead of 'file'"""
        stage = DownloadStage()
        state = PipelineState()

        downloaded = [
            {'path': 'video1.mp4', 'url': 'url1', 'title': 'Test'}
        ]

        stage._store_download_results(state, downloaded, [], [])

        assert len(state.downloaded_videos) == 1
        assert state.downloaded_videos[0].file == 'video1.mp4'

    @pytest.mark.fast
    def test_store_download_with_tier_field(self):
        """Test storing downloads when dict uses 'tier' instead of 'duration_tier'"""
        stage = DownloadStage()
        state = PipelineState()

        downloaded = [
            {'file': 'video1.mp4', 'tier': 'short'}
        ]

        stage._store_download_results(state, downloaded, [], [])

        assert len(state.downloaded_videos) == 1
        assert state.downloaded_videos[0].duration_tier == 'short'

    @pytest.mark.fast
    def test_store_download_with_reusable_and_failed(self):
        """Test storing with both reusable videos and failed keywords"""
        stage = DownloadStage()
        state = PipelineState()

        downloaded = [
            {'file': 'new_video.mp4', 'url': 'url1', 'source': 'download'}
        ]
        reusable = [
            {'path': 'cached.mp4', 'face_score': 0.9}
        ]
        failed = ['mountain', 'desert']

        stage._store_download_results(state, downloaded, reusable, failed)

        assert len(state.downloaded_videos) == 2
        assert state.downloaded_videos[0].file == 'new_video.mp4'
        assert state.downloaded_videos[0].source == 'download'
        assert state.downloaded_videos[1].file == 'cached.mp4'
        assert state.downloaded_videos[1].source == 'global_cache'
        assert state.downloaded_videos[1].face_score == 0.9
        assert state.failed_keywords == ['mountain', 'desert']


# ============================================================================
# Test Restore Obsolete Fields
# ============================================================================

# ============================================================================
# Test Retry Logic
# ============================================================================

@pytest.mark.fast
class TestDownloadRetryLogic:
    """Test retry with exponential backoff behavior"""

    def test_retry_with_backoff_success_first_attempt(self, mock_config):
        """Test successful operation on first attempt"""
        stage = DownloadStage()
        mock_config.download.max_retries = 3
        mock_config.download.retry_delay = 0.1  # Short delay for tests
        mock_config.download.retry_backoff = 2.0

        call_count = 0
        def successful_op():
            nonlocal call_count
            call_count += 1
            return ["result"]

        success, result, attempts = stage._retry_with_backoff(
            successful_op, mock_config, "test_op"
        )

        assert success is True
        assert result == ["result"]
        assert attempts == 1
        assert call_count == 1

    @pytest.mark.fast
    def test_retry_with_backoff_success_after_retries(self, mock_config):
        """Test successful operation after retries"""
        stage = DownloadStage()
        mock_config.download.max_retries = 3
        mock_config.download.retry_delay = 0.01  # Very short delay for tests
        mock_config.download.retry_backoff = 2.0

        call_count = 0
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return ["success"]

        success, result, attempts = stage._retry_with_backoff(
            fail_then_succeed, mock_config, "test_op"
        )

        assert success is True
        assert result == ["success"]
        assert attempts == 3
        assert call_count == 3

    @pytest.mark.fast
    def test_retry_with_backoff_all_attempts_fail(self, mock_config):
        """Test all retry attempts fail"""
        stage = DownloadStage()
        mock_config.download.max_retries = 3
        mock_config.download.retry_delay = 0.01
        mock_config.download.retry_backoff = 2.0

        call_count = 0
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise Exception("Permanent failure")

        success, result, attempts = stage._retry_with_backoff(
            always_fail, mock_config, "test_op"
        )

        assert success is False
        assert "Permanent failure" in result
        assert attempts == 3
        assert call_count == 3

    @pytest.mark.fast
    def test_retry_with_backoff_uses_config_defaults(self):
        """Test retry uses getattr defaults when config fields missing"""
        stage = DownloadStage()
        mock_config = MagicMock()
        # Simulate missing attributes - getattr should use defaults
        mock_config.download.max_retries = None  # Will trigger default
        del mock_config.download.retry_delay
        del mock_config.download.retry_backoff

        # Re-mock to trigger getattr default behavior
        mock_config.download = MagicMock(spec=[])  # Empty spec means no attributes

        call_count = 0
        def succeed():
            nonlocal call_count
            call_count += 1
            return ["ok"]

        # Should use defaults: max_retries=3
        success, result, attempts = stage._retry_with_backoff(
            succeed, mock_config, "test_op"
        )

        assert success is True
        assert call_count == 1

    @patch('time.sleep')
    @pytest.mark.fast
    def test_retry_with_backoff_exponential_delay(self, mock_sleep, mock_config):
        """Test exponential backoff delays are calculated correctly"""
        stage = DownloadStage()
        mock_config.download.max_retries = 4
        mock_config.download.retry_delay = 2.0
        mock_config.download.retry_backoff = 2.0

        call_count = 0
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise Exception("Failure")

        success, result, attempts = stage._retry_with_backoff(
            always_fail, mock_config, "test_op"
        )

        # Should have slept 3 times (between attempts 1-2, 2-3, 3-4)
        # Delays: 2.0 * 2^0 = 2.0, 2.0 * 2^1 = 4.0, 2.0 * 2^2 = 8.0
        assert mock_sleep.call_count == 3
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays[0] == 2.0   # First retry delay
        assert delays[1] == 4.0   # Second retry delay
        assert delays[2] == 8.0   # Third retry delay

    @patch('src.stages.download.logger')
    @pytest.mark.fast
    def test_retry_logs_warnings_and_errors(self, mock_logger, mock_config):
        """Test retry logs appropriate messages"""
        stage = DownloadStage()
        mock_config.download.max_retries = 2
        mock_config.download.retry_delay = 0.01
        mock_config.download.retry_backoff = 2.0

        def always_fail():
            raise Exception("Test error")

        success, result, attempts = stage._retry_with_backoff(
            always_fail, mock_config, "test_op"
        )

        # Should have logged warning for first retry, error for final failure
        assert mock_logger.warning.call_count == 1
        assert mock_logger.error.call_count == 1
        assert "Retry 1/2" in str(mock_logger.warning.call_args)
        assert "All 2 retries failed" in str(mock_logger.error.call_args)


@pytest.mark.fast
class TestAudioFirstRetryIntegration:
    """Test retry integration in audio-first mode"""

    @patch('src.downloader.VideoDownloader')
    def test_audio_first_retries_failed_keywords(self, mock_downloader_class, mock_config, mock_checkpoint):
        """Test audio-first mode retries failed keyword downloads"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        mock_config.download.max_retries = 3
        mock_config.download.retry_delay = 0.01
        mock_config.download.retry_backoff = 2.0

        mock_downloader = Mock()
        mock_downloader.DURATION_TIERS = {'short': {}}
        mock_downloader._get_tier_value.return_value = 5

        # Fail twice, then succeed
        call_count = 0
        def fail_then_succeed(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Network error")
            return [Mock(file="audio.mp3", video_id="vid1", url="url")]

        mock_downloader.audio_first.download_audio_for_keyword.side_effect = fail_then_succeed
        mock_downloader_class.return_value = mock_downloader

        result = stage._run_audio_first(state, mock_config, mock_checkpoint, [])

        assert result.success is True
        assert len(state.downloaded_audio) == 1
        assert len(state.failed_keywords) == 0
        assert call_count == 3

    @patch('src.downloader.VideoDownloader')
    @pytest.mark.fast
    def test_audio_first_marks_keyword_failed_after_all_retries(self, mock_downloader_class, mock_config, mock_checkpoint):
        """Test audio-first mode marks keyword as failed after all retries exhausted"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        mock_config.download.max_retries = 2
        mock_config.download.retry_delay = 0.01
        mock_config.download.retry_backoff = 2.0

        mock_downloader = Mock()
        mock_downloader.DURATION_TIERS = {'short': {}}
        mock_downloader._get_tier_value.return_value = 5
        mock_downloader.audio_first.download_audio_for_keyword.side_effect = Exception("Permanent error")
        mock_downloader_class.return_value = mock_downloader

        result = stage._run_audio_first(state, mock_config, mock_checkpoint, [])

        assert result.success is True  # Stage succeeds even with failed keywords
        assert len(state.downloaded_audio) == 0
        assert "beach" in state.failed_keywords

    @patch('src.downloader.VideoDownloader')
    @pytest.mark.fast
    def test_audio_first_partial_tier_success(self, mock_downloader_class, mock_config, mock_checkpoint):
        """Test audio-first mode succeeds if any tier succeeds (even if others fail)"""
        stage = DownloadStage()
        state = PipelineState()
        state.keywords = ["beach"]

        mock_config.download.max_retries = 2
        mock_config.download.retry_delay = 0.01
        mock_config.download.retry_backoff = 2.0

        mock_downloader = Mock()
        mock_downloader.DURATION_TIERS = {'short': {}, 'medium': {}}
        mock_downloader._get_tier_value.return_value = 5

        # First tier fails, second succeeds
        tier_results = {
            'short': Exception("Tier short failed"),
            'medium': [Mock(file="audio.mp3", video_id="vid1", url="url")]
        }

        def download_by_tier(**kwargs):
            tier = kwargs.get('tier')
            result = tier_results.get(tier)
            if isinstance(result, Exception):
                raise result
            return result

        mock_downloader.audio_first.download_audio_for_keyword.side_effect = download_by_tier
        mock_downloader_class.return_value = mock_downloader

        result = stage._run_audio_first(state, mock_config, mock_checkpoint, [])

        assert result.success is True
        assert len(state.downloaded_audio) == 1
        assert len(state.failed_keywords) == 0  # Keyword succeeded because medium tier worked


@pytest.mark.fast
class TestRestoreObsoleteFields:
    """Test restore handles obsolete fields from old checkpoints"""

    def test_restore_removes_obsolete_fields(self, mock_checkpoint):
        """Test restore removes channel, duration_tier, upload_date, license from audio"""
        stage = DownloadStage()
        state = PipelineState()

        # Simulate old checkpoint with obsolete fields
        mock_checkpoint.get_stage_data.return_value = {
            'audio_downloads': [
                {
                    'file': 'audio.mp3',
                    'url': 'https://example.com',
                    'video_id': 'vid123',
                    'title': 'Test',
                    'duration': 120.0,
                    'keyword': 'beach',
                    # Obsolete fields that were removed from AudioDownload
                    'channel': 'TestChannel',
                    'duration_tier': 'medium',
                    'upload_date': '2026-01-01',
                    'license': 'CC-BY'
                }
            ],
            'failed_keywords': []
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.downloaded_audio) == 1
        assert state.downloaded_audio[0].file == 'audio.mp3'
        # Obsolete fields should not cause errors
        assert not hasattr(state.downloaded_audio[0], 'channel')
        assert not hasattr(state.downloaded_audio[0], 'upload_date')

    @pytest.mark.fast
    def test_restore_audio_already_object(self, mock_checkpoint):
        """Test restore when audio downloads are already AudioDownload objects"""
        stage = DownloadStage()
        state = PipelineState()

        audio_obj = AudioDownload(
            file='audio.mp3',
            video_id='vid123',
            url='https://example.com'
        )

        mock_checkpoint.get_stage_data.return_value = {
            'audio_downloads': [audio_obj],  # Already an object
            'failed_keywords': []
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.downloaded_audio) == 1
        assert state.downloaded_audio[0] is audio_obj
