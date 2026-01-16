"""
Unit tests for CaptionStage pipeline stage.

Tests caption fetching stage behavior, checkpoint handling, and state management.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages.caption import CaptionStage
from src.stages import StageResult


class TestCaptionStageBasic:
    """Basic tests for CaptionStage"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with caption-first disabled"""
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.enabled = False
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.cache = Mock()
        config.cache.cache_dir = tempfile.gettempdir()
        return config

    @pytest.fixture
    def mock_config_enabled(self):
        """Create mock config with caption-first enabled"""
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.enabled = True
        config.download.caption_first.languages = ["en", "en-US"]
        config.download.caption_first.prefer_manual_captions = True
        config.download.caption_first.fetch_timeout = 30
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.cache = Mock()
        config.cache.cache_dir = tempfile.gettempdir()
        return config

    @pytest.fixture
    def mock_state(self):
        """Create mock pipeline state"""
        state = Mock()
        state.downloaded_audio = []
        state.downloaded_videos = []
        state.caption_downloads = []
        state.videos_need_audio = []
        state.transcripts = {}
        state.video_candidates = []  # Added for _get_video_ids iteration
        return state

    @pytest.fixture
    def mock_checkpoint(self):
        """Create mock checkpoint manager"""
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False
        checkpoint.get_stage_data.return_value = None
        checkpoint.save = Mock()
        return checkpoint

    def test_stage_attributes(self):
        """Test CaptionStage has correct attributes"""
        stage = CaptionStage()
        assert stage.name == "CAPTION"
        assert stage.description == "Fetch YouTube captions for videos"

    def test_run_disabled_skips(self, mock_config, mock_state, mock_checkpoint):
        """Test stage skips when caption-first is disabled"""
        stage = CaptionStage()

        result = stage.run(mock_state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'not_enabled'

    def test_run_no_videos_skips(self, mock_config_enabled, mock_state, mock_checkpoint):
        """Test stage skips when no videos to process"""
        stage = CaptionStage()

        result = stage.run(mock_state, mock_config_enabled, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_videos'

    def test_can_skip_from_checkpoint(self, mock_state, mock_checkpoint):
        """Test can_skip returns checkpoint status"""
        stage = CaptionStage()

        mock_checkpoint.should_skip_stage.return_value = True
        assert stage.can_skip(mock_state, mock_checkpoint) is True

        mock_checkpoint.should_skip_stage.return_value = False
        assert stage.can_skip(mock_state, mock_checkpoint) is False


class TestCaptionStageVideoIdExtraction:
    """Tests for video ID extraction in CaptionStage"""

    @pytest.fixture
    def stage(self):
        return CaptionStage()

    def test_extract_video_id_from_url(self, stage):
        """Test extraction from YouTube URL"""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        vid = stage._extract_video_id(url)
        assert vid == "dQw4w9WgXcQ"

    def test_extract_video_id_from_short_url(self, stage):
        """Test extraction from youtu.be URL"""
        url = "https://youtu.be/dQw4w9WgXcQ"
        vid = stage._extract_video_id(url)
        assert vid == "dQw4w9WgXcQ"

    def test_extract_video_id_from_filename(self, stage):
        """Test extraction from filename"""
        filename = "dQw4w9WgXcQ.mp4"
        vid = stage._extract_video_id(filename)
        assert vid == "dQw4w9WgXcQ"

    def test_extract_video_id_from_path(self, stage):
        """Test extraction from full path"""
        path = "/videos/downloads/dQw4w9WgXcQ.webm"
        vid = stage._extract_video_id(path)
        assert vid == "dQw4w9WgXcQ"

    def test_extract_video_id_invalid(self, stage):
        """Test extraction returns None for invalid input"""
        # Very short strings without 11-char pattern
        assert stage._extract_video_id("short") is None
        assert stage._extract_video_id("") is None


class TestCaptionStageRestore:
    """Tests for CaptionStage restore functionality"""

    @pytest.fixture
    def stage(self):
        return CaptionStage()

    @pytest.fixture
    def mock_state(self):
        state = Mock()
        state.caption_downloads = []
        state.videos_need_audio = []
        state.transcripts = {}
        state.video_candidates = []  # Added for _get_video_ids iteration
        return state

    @pytest.fixture
    def mock_checkpoint(self):
        checkpoint = Mock()
        return checkpoint

    def test_restore_with_no_data(self, stage, mock_state, mock_checkpoint):
        """Test restore returns False when no checkpoint data"""
        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(mock_state, mock_checkpoint)

        assert result is False

    def test_restore_with_data(self, stage, mock_state, mock_checkpoint):
        """Test restore restores state from checkpoint"""
        mock_checkpoint.get_stage_data.return_value = {
            'caption_downloads': [
                {
                    'file': '/path/to/caption.srt',
                    'video_id': 'test123',
                    'url': 'https://youtube.com/watch?v=test123',
                    'title': 'Test Video',
                    'duration': 120.0,
                    'keyword': 'test',
                    'language': 'en',
                    'is_auto_generated': False,
                }
            ],
            'videos_need_audio': ['vid456', 'vid789'],
            'transcripts_from_captions': ['test123'],
        }

        result = stage.restore(mock_state, mock_checkpoint)

        assert result is True
        assert len(mock_state.caption_downloads) == 1
        assert mock_state.videos_need_audio == ['vid456', 'vid789']

    def test_restore_handles_exception(self, stage, mock_state, mock_checkpoint):
        """Test restore handles exceptions gracefully"""
        mock_checkpoint.get_stage_data.side_effect = Exception("Checkpoint error")

        result = stage.restore(mock_state, mock_checkpoint)

        assert result is False


class TestCaptionStageWithMockedFetcher:
    """Tests for CaptionStage with mocked CaptionFetcher"""

    @pytest.fixture
    def mock_config_enabled(self):
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.enabled = True
        config.download.caption_first.languages = ["en"]
        config.download.caption_first.prefer_manual_captions = True
        config.download.caption_first.fetch_timeout = 30
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.cache = Mock()
        config.cache.cache_dir = tempfile.gettempdir()
        return config

    @pytest.fixture
    def mock_state_with_audio(self):
        state = Mock()

        # Create mock audio download
        audio = Mock()
        audio.video_id = "test123"
        audio.url = "https://youtube.com/watch?v=test123"
        audio.title = "Test Video"
        audio.duration = 120.0
        audio.keyword = "test"

        state.downloaded_audio = [audio]
        state.downloaded_videos = []
        state.caption_downloads = []
        state.videos_need_audio = []
        state.transcripts = {}
        state.video_candidates = []  # Added for _get_video_ids iteration
        return state

    @pytest.fixture
    def mock_checkpoint(self):
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False
        checkpoint.save = Mock()
        return checkpoint

    @patch('src.downloader.caption_fetcher.CaptionFetcher')
    def test_run_with_captions_found(
        self, MockFetcher, mock_config_enabled, mock_state_with_audio, mock_checkpoint
    ):
        """Test run when captions are found"""
        # Setup mock fetcher
        mock_fetcher_instance = Mock()
        MockFetcher.return_value = mock_fetcher_instance

        # Mock successful caption fetch
        mock_result = Mock()
        mock_result.file = "/path/to/caption.srt"
        mock_result.language = "en"
        mock_result.is_auto_generated = False
        mock_fetcher_instance.fetch_captions.return_value = mock_result

        # Mock caption parsing
        mock_fetcher_instance.parse_caption_file.return_value = [
            {'text': 'Hello', 'start': 0.0, 'end': 2.0},
            {'text': 'World', 'start': 2.0, 'end': 4.0},
        ]

        stage = CaptionStage()
        result = stage.run(mock_state_with_audio, mock_config_enabled, mock_checkpoint)

        assert result.success is True
        assert result.data.get('caption_count') == 1
        assert result.data.get('fallback_count') == 0
        mock_checkpoint.save.assert_called_once()

    @patch('src.downloader.caption_fetcher.CaptionFetcher')
    def test_run_with_no_captions(
        self, MockFetcher, mock_config_enabled, mock_state_with_audio, mock_checkpoint
    ):
        """Test run when no captions are found"""
        mock_fetcher_instance = Mock()
        MockFetcher.return_value = mock_fetcher_instance
        mock_fetcher_instance.fetch_captions.return_value = None

        stage = CaptionStage()
        result = stage.run(mock_state_with_audio, mock_config_enabled, mock_checkpoint)

        assert result.success is True
        assert result.data.get('caption_count') == 0
        assert result.data.get('fallback_count') == 1
        assert len(result.warnings) > 0  # Should warn about no captions

    @patch('src.downloader.caption_fetcher.CaptionFetcher')
    def test_run_handles_fetcher_exception(
        self, MockFetcher, mock_config_enabled, mock_state_with_audio, mock_checkpoint
    ):
        """Test run handles CaptionFetcher exceptions"""
        MockFetcher.side_effect = Exception("Fetcher initialization failed")

        stage = CaptionStage()
        result = stage.run(mock_state_with_audio, mock_config_enabled, mock_checkpoint)

        assert result.success is False
        assert "Fetcher initialization failed" in result.error


class TestCaptionStageGetVideoIds:
    """Tests for _get_video_ids method"""

    @pytest.fixture
    def stage(self):
        return CaptionStage()

    @pytest.fixture
    def mock_config(self):
        return Mock()

    def test_get_video_ids_from_audio(self, stage, mock_config):
        """Test getting video IDs from audio downloads"""
        state = Mock()

        audio1 = Mock()
        audio1.video_id = "vid1"
        audio2 = Mock()
        audio2.video_id = "vid2"

        state.downloaded_audio = [audio1, audio2]
        state.downloaded_videos = []
        state.video_candidates = []  # Required for _get_video_ids iteration

        video_ids = stage._get_video_ids(state, mock_config)

        assert "vid1" in video_ids
        assert "vid2" in video_ids
        assert len(video_ids) == 2

    def test_get_video_ids_from_videos(self, stage, mock_config):
        """Test getting video IDs from video downloads"""
        state = Mock()
        state.downloaded_audio = []
        state.video_candidates = []  # Required for _get_video_ids iteration

        video1 = Mock()
        video1.url = "https://youtube.com/watch?v=abc12345678"
        video1.file = ""
        video2 = Mock()
        video2.url = ""
        video2.file = "def12345678.mp4"

        state.downloaded_videos = [video1, video2]

        video_ids = stage._get_video_ids(state, mock_config)

        assert "abc12345678" in video_ids
        assert "def12345678" in video_ids

    def test_get_video_ids_deduplicates(self, stage, mock_config):
        """Test that duplicate video IDs are removed"""
        state = Mock()

        audio1 = Mock()
        audio1.video_id = "duplicate123"
        audio2 = Mock()
        audio2.video_id = "duplicate123"

        state.downloaded_audio = [audio1, audio2]
        state.downloaded_videos = []
        state.video_candidates = []  # Required for _get_video_ids iteration

        video_ids = stage._get_video_ids(state, mock_config)

        assert len(video_ids) == 1
        assert video_ids[0] == "duplicate123"

    def test_get_video_ids_empty(self, stage, mock_config):
        """Test with empty state"""
        state = Mock()
        state.downloaded_audio = []
        state.downloaded_videos = []
        state.video_candidates = []  # Required for _get_video_ids iteration

        video_ids = stage._get_video_ids(state, mock_config)

        assert video_ids == []


class TestCaptionStageStageResult:
    """Tests for StageResult usage in CaptionStage"""

    def test_stage_result_ok(self):
        """Test StageResult.ok creation"""
        result = StageResult.ok({'key': 'value'}, ['warning1'])

        assert result.success is True
        assert result.data == {'key': 'value'}
        assert 'warning1' in result.warnings

    def test_stage_result_fail(self):
        """Test StageResult.fail creation"""
        result = StageResult.fail("Error message", ['warning'])

        assert result.success is False
        assert result.error == "Error message"


class TestCaptionStageIntegration:
    """Integration tests for CaptionStage"""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_stage_with_mock_fetcher(self, temp_cache_dir):
        """Test full stage execution flow"""
        # Create config
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.enabled = True
        config.download.caption_first.languages = ["en"]
        config.download.caption_first.prefer_manual_captions = True
        config.download.caption_first.fetch_timeout = 30
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.cache = Mock()
        config.cache.cache_dir = str(temp_cache_dir)

        # Create state
        state = Mock()
        audio = Mock()
        audio.video_id = "test123"
        audio.url = "https://youtube.com/watch?v=test123"
        audio.title = "Test"
        audio.duration = 60.0
        audio.keyword = "test"
        state.downloaded_audio = [audio]
        state.downloaded_videos = []
        state.caption_downloads = []
        state.videos_need_audio = []
        state.transcripts = {}
        state.video_candidates = []  # Required for _get_video_ids iteration

        # Create checkpoint
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False
        checkpoint.save = Mock()

        # Run stage with mocked fetcher
        with patch('src.downloader.caption_fetcher.CaptionFetcher') as MockFetcher:
            mock_fetcher = Mock()
            MockFetcher.return_value = mock_fetcher
            mock_fetcher.fetch_captions.return_value = None  # No captions

            stage = CaptionStage()
            result = stage.run(state, config, checkpoint)

        # Verify
        assert result.success is True
        assert state.videos_need_audio == ['test123']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
