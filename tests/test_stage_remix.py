"""
Test Suite for RemixStage

Tests the RemixStage class which handles:
- Keyword-based video/audio filtering
- Relevance scoring
- Audio-first mode support
- Mode detection (audio vs video)
- Interactive curation
- Checkpoint operations
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from src.stages.remix import RemixStage
from src.state import PipelineState, DownloadedVideo, AudioDownload


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with remix settings"""
    config = MagicMock()

    # Remix config
    remix = MagicMock()
    remix.enabled = True
    remix.trigger_after_download = True
    remix.min_relevance_score = 0.1
    remix.high_relevance_threshold = 0.5
    remix.max_files_to_process = 500
    remix.max_files_to_include = 100
    remix.fuzzy_match = True
    remix.case_sensitive = False
    remix.interactive_curation = False
    remix.auto_accept_filter = 'filtered'
    config.remix = remix

    config.downloaded_videos_dir = "/path/to/videos"
    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint manager"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    return checkpoint


@pytest.fixture
def mock_keywords():
    """Create mock keywords"""
    return ['ocean', 'waves', 'beach', 'sunset']


@pytest.fixture
def mock_downloaded_videos():
    """Create mock downloaded videos"""
    return [
        DownloadedVideo(
            file='/path/video1.mp4',
            source='youtube',
            keyword='ocean',
            url='http://example.com/1',
            title='Ocean Waves',
            duration=120.0
        ),
        DownloadedVideo(
            file='/path/video2.mp4',
            source='youtube',
            keyword='forest',
            url='http://example.com/2',
            title='Forest Sounds',
            duration=90.0
        ),
        DownloadedVideo(
            file='/path/video3.mp4',
            source='youtube',
            keyword='ocean',
            url='http://example.com/3',
            title='Beach Sunset',
            duration=150.0
        ),
    ]


@pytest.fixture
def mock_downloaded_audio():
    """Create mock downloaded audio"""
    return [
        AudioDownload(
            file='/path/audio1.mp3',
            url='http://example.com/1',
            video_id='vid1',
            title='Ocean Waves',
            duration=120.0,
            keyword='ocean'
        ),
        AudioDownload(
            file='/path/audio2.mp3',
            url='http://example.com/2',
            video_id='vid2',
            title='Forest Sounds',
            duration=90.0,
            keyword='forest'
        ),
    ]


@pytest.fixture
def mock_remix_result():
    """Create mock remix result"""
    @dataclass
    class RemixResult:
        total_files: int
        included_files: int
        avg_match_score: float
        processing_time_seconds: float

    return RemixResult(
        total_files=3,
        included_files=2,
        avg_match_score=0.75,
        processing_time_seconds=1.5
    )


# ============================================================================
# Test Stage Initialization
# ============================================================================

class TestRemixStageInit:
    """Test stage initialization"""

    def test_stage_name(self):
        """Test stage name"""
        stage = RemixStage()
        assert stage.name == "REMIX"

    def test_stage_description(self):
        """Test stage description"""
        stage = RemixStage()
        assert "keyword relevance" in stage.description.lower()

    def test_stage_registration(self):
        """Test stage is registered in stage registry"""
        from src.stages import get_stage
        stage_class = get_stage("REMIX")
        assert stage_class is RemixStage


# ============================================================================
# Test Input Validation
# ============================================================================

class TestInputValidation:
    """Test input validation"""

    def test_validate_no_hard_requirements(self, mock_config):
        """Test validation has no hard requirements (optional stage)"""
        stage = RemixStage()
        state = PipelineState()
        # Empty state should still pass validation
        result = stage.validate_inputs(state, mock_config)
        assert result is None


# ============================================================================
# Test Skip Conditions
# ============================================================================

class TestSkipConditions:
    """Test various skip conditions"""

    def test_skip_when_disabled(self, mock_config, mock_checkpoint):
        """Test skipping when remix disabled"""
        stage = RemixStage()
        state = PipelineState()
        mock_config.remix.enabled = False

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'disabled'

    def test_skip_when_no_remix_config(self, mock_config, mock_checkpoint):
        """Test skipping when remix config not present"""
        stage = RemixStage()
        state = PipelineState()
        mock_config.remix = None

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'disabled'

    def test_skip_when_trigger_disabled(self, mock_config, mock_checkpoint):
        """Test skipping when trigger_after_download disabled"""
        stage = RemixStage()
        state = PipelineState()
        mock_config.remix.trigger_after_download = False

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'trigger_disabled'

    def test_skip_when_no_keywords(self, mock_config, mock_checkpoint):
        """Test skipping when no keywords available"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = []

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'no_keywords'

    def test_skip_when_no_files(self, mock_config, mock_checkpoint, mock_keywords):
        """Test skipping when no files to remix"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []
        state.downloaded_audio = []

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'no_files'


# ============================================================================
# Test Mode Detection
# ============================================================================

class TestModeDetection:
    """Test audio vs video mode detection"""

    def test_detect_audio_mode(self, mock_downloaded_audio):
        """Test detection of audio-first mode"""
        stage = RemixStage()
        state = PipelineState()
        state.downloaded_audio = mock_downloaded_audio
        state.downloaded_videos = []

        mode = stage._detect_mode(state)

        assert mode == 'audio'

    def test_detect_video_mode(self, mock_downloaded_videos):
        """Test detection of video mode"""
        stage = RemixStage()
        state = PipelineState()
        state.downloaded_audio = []
        state.downloaded_videos = mock_downloaded_videos

        mode = stage._detect_mode(state)

        assert mode == 'video'

    def test_detect_none_mode(self):
        """Test detection when no files present"""
        stage = RemixStage()
        state = PipelineState()
        state.downloaded_audio = []
        state.downloaded_videos = []

        mode = stage._detect_mode(state)

        assert mode == 'none'

    def test_audio_mode_priority(self, mock_downloaded_audio, mock_downloaded_videos):
        """Test audio mode takes priority when both present"""
        stage = RemixStage()
        state = PipelineState()
        state.downloaded_audio = mock_downloaded_audio
        state.downloaded_videos = mock_downloaded_videos

        mode = stage._detect_mode(state)

        # Audio-first mode takes priority
        assert mode == 'audio'


# ============================================================================
# Test Video Mode Remix
# ============================================================================

class TestVideoModeRemix:
    """Test video file remixing"""

    @patch('src.keyword_remix.remix_downloaded_videos')
    def test_remix_videos_success(self, mock_remix, mock_config, mock_checkpoint,
                                  mock_keywords, mock_downloaded_videos, mock_remix_result):
        """Test successful video remix"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = mock_downloaded_videos
        state.downloaded_audio = []

        # Mock remix to include only first 2 videos
        included_paths = ['/path/video1.mp4', '/path/video3.mp4']
        mock_remix.return_value = (included_paths, mock_remix_result)

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.downloaded_videos) == 2
        assert state.downloaded_videos[0].file == '/path/video1.mp4'
        assert state.downloaded_videos[1].file == '/path/video3.mp4'
        assert result.data['included_count'] == 2
        assert result.data['excluded_count'] == 1
        assert result.data['mode'] == 'video'

    @patch('src.keyword_remix.remix_downloaded_videos')
    def test_remix_videos_config_passed(self, mock_remix, mock_config, mock_checkpoint,
                                       mock_keywords, mock_downloaded_videos, mock_remix_result):
        """Test RemixConfig is properly constructed and passed"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = mock_downloaded_videos
        state.downloaded_audio = []

        mock_remix.return_value = ([], mock_remix_result)

        result = stage.run(state, mock_config, mock_checkpoint)

        # Verify remix was called with config
        assert mock_remix.called
        call_kwargs = mock_remix.call_args[1]
        remix_config = call_kwargs['config']
        assert remix_config.min_relevance_score == 0.1
        assert remix_config.fuzzy_match is True


# ============================================================================
# Test Audio Mode Remix
# ============================================================================

class TestAudioModeRemix:
    """Test audio file remixing"""

    @patch('src.keyword_remix.remix_audio_files')
    def test_remix_audio_success(self, mock_remix, mock_config, mock_checkpoint,
                                mock_keywords, mock_downloaded_audio, mock_remix_result):
        """Test successful audio remix"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_audio = mock_downloaded_audio
        state.downloaded_videos = []

        # Mock remix to include only first audio
        included_paths = ['/path/audio1.mp3']
        mock_remix.return_value = (included_paths, mock_remix_result)

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.downloaded_audio) == 1
        assert state.downloaded_audio[0].file == '/path/audio1.mp3'
        assert result.data['included_count'] == 1
        assert result.data['excluded_count'] == 2  # total_files - included
        assert result.data['mode'] == 'audio'

    @patch('src.keyword_remix.remix_audio_files')
    def test_remix_audio_keywords_passed(self, mock_remix, mock_config, mock_checkpoint,
                                        mock_keywords, mock_downloaded_audio, mock_remix_result):
        """Test keywords are passed to remix function"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_audio = mock_downloaded_audio
        state.downloaded_videos = []

        mock_remix.return_value = ([], mock_remix_result)

        result = stage.run(state, mock_config, mock_checkpoint)

        # Verify keywords were passed
        call_args = mock_remix.call_args[0]
        passed_keywords = call_args[1]
        assert passed_keywords == mock_keywords


# ============================================================================
# Test Interactive Curation
# ============================================================================

class TestInteractiveCuration:
    """Test interactive curation mode"""

    @patch('src.keyword_remix.remix_downloaded_videos')
    def test_interactive_mode_passed(self, mock_remix, mock_config, mock_checkpoint,
                                    mock_keywords, mock_downloaded_videos, mock_remix_result):
        """Test interactive mode flag is passed"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = mock_downloaded_videos
        state.downloaded_audio = []

        mock_config.remix.interactive_curation = True
        mock_remix.return_value = ([], mock_remix_result)

        result = stage.run(state, mock_config, mock_checkpoint)

        # Verify interactive flag was passed
        call_kwargs = mock_remix.call_args[1]
        assert call_kwargs['interactive'] is True


# ============================================================================
# Test Stage Execution
# ============================================================================

class TestStageExecution:
    """Test full stage execution"""

    @patch('src.keyword_remix.remix_downloaded_videos')
    def test_run_success(self, mock_remix, mock_config, mock_checkpoint,
                        mock_keywords, mock_downloaded_videos, mock_remix_result):
        """Test successful full execution"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = mock_downloaded_videos
        state.downloaded_audio = []

        mock_remix.return_value = (['/path/video1.mp4'], mock_remix_result)

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert 'included_count' in result.data
        assert 'excluded_count' in result.data
        assert 'avg_score' in result.data
        assert 'mode' in result.data

    def test_run_import_error(self, mock_config, mock_checkpoint, mock_keywords,
                              mock_downloaded_videos):
        """Test handling import errors"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = mock_downloaded_videos
        state.downloaded_audio = []

        # Patch the import statement inside run()
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "import" in result.error.lower()

    @patch('src.keyword_remix.remix_downloaded_videos')
    def test_run_exception_handling(self, mock_remix, mock_config, mock_checkpoint,
                                   mock_keywords, mock_downloaded_videos):
        """Test exception handling in main run"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = mock_downloaded_videos
        state.downloaded_audio = []

        mock_remix.side_effect = Exception("Remix failed")

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "failed" in result.error.lower()


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

class TestCheckpointOperations:
    """Test checkpoint save/restore"""

    def test_can_skip_with_checkpoint(self, mock_checkpoint):
        """Test can_skip returns True when checkpoint exists"""
        stage = RemixStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, mock_checkpoint)

        assert result is True
        mock_checkpoint.should_skip_stage.assert_called_with("REMIX")

    def test_can_skip_no_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = RemixStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = False

        result = stage.can_skip(state, mock_checkpoint)

        assert result is False

    def test_restore_success(self, mock_checkpoint):
        """Test successful restore from checkpoint"""
        stage = RemixStage()
        state = PipelineState()

        # Mock checkpoint data
        checkpoint_data = {
            'included_count': 10,
            'excluded_count': 5,
            'mode': 'video',
            'avg_score': 0.75
        }
        mock_checkpoint.get_stage_data.return_value = checkpoint_data

        result = stage.restore(state, mock_checkpoint)

        # Restore just logs info, doesn't modify state
        assert result is True

    def test_restore_no_data(self, mock_checkpoint):
        """Test restore fails when no checkpoint data"""
        stage = RemixStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions"""

    @patch('src.keyword_remix.remix_downloaded_videos')
    def test_all_videos_excluded(self, mock_remix, mock_config, mock_checkpoint,
                                mock_keywords, mock_downloaded_videos, mock_remix_result):
        """Test handling when all videos are excluded"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = mock_downloaded_videos
        state.downloaded_audio = []

        # Mock remix to exclude all videos
        mock_remix.return_value = ([], mock_remix_result)

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.downloaded_videos) == 0
        assert result.data['included_count'] == 0

    @patch('src.keyword_remix.remix_downloaded_videos')
    def test_all_videos_included(self, mock_remix, mock_config, mock_checkpoint,
                                mock_keywords, mock_downloaded_videos, mock_remix_result):
        """Test handling when all videos are included"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = mock_downloaded_videos
        state.downloaded_audio = []

        # Mock remix to include all videos
        all_paths = [v.file for v in mock_downloaded_videos]
        mock_remix.return_value = (all_paths, mock_remix_result)

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.downloaded_videos) == 3
        assert result.data['excluded_count'] == 0

    @patch('src.keyword_remix.remix_downloaded_videos')
    def test_default_config_values(self, mock_remix, mock_config, mock_checkpoint,
                                   mock_keywords, mock_downloaded_videos, mock_remix_result):
        """Test default values when config attributes missing"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = mock_downloaded_videos
        state.downloaded_audio = []

        # Remove optional config attributes
        delattr(mock_config.remix, 'min_relevance_score')
        delattr(mock_config.remix, 'fuzzy_match')
        delattr(mock_config.remix, 'interactive_curation')

        mock_remix.return_value = ([], mock_remix_result)

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should use defaults (0.1, True, False)
        assert result.success is True

    @patch('src.keyword_remix.remix_downloaded_videos')
    def test_checkpoint_data_structure(self, mock_remix, mock_config, mock_checkpoint,
                                       mock_keywords, mock_downloaded_videos, mock_remix_result):
        """Test checkpoint data contains all required fields"""
        stage = RemixStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = mock_downloaded_videos
        state.downloaded_audio = []

        included_paths = ['/path/video1.mp4']
        mock_remix.return_value = (included_paths, mock_remix_result)

        result = stage.run(state, mock_config, mock_checkpoint)

        # Verify all checkpoint fields present
        assert 'included_count' in result.data
        assert 'excluded_count' in result.data
        assert 'avg_score' in result.data
        assert 'remixed_paths' in result.data
        assert 'mode' in result.data
        assert 'processing_time_sec' in result.data

        # Verify values
        assert result.data['included_count'] == 1
        assert result.data['excluded_count'] == 2
        assert result.data['remixed_paths'] == included_paths
        assert result.data['mode'] == 'video'
