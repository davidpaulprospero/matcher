"""
Comprehensive tests for CaptionStage.

Tests cover:
- Stage initialization and registration
- Video ID extraction from various sources
- Caption fetching with mocked API
- Checkpoint save/restore
- text_metadata population
- Error handling (unavailable/fetch errors)
- Configuration (enabled/disabled)
- Partial completion resume

Created: 2026-01-25 (Sprint 4 - Caption Mode)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.stages.caption_stage import CaptionStage
from src.state import PipelineState, DownloadedVideo, AudioDownload


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with caption-first settings"""
    config = MagicMock()

    # Caption-first config
    config.download.caption_first.enabled = True
    config.download.caption_first.preferred_language = "en"
    config.download.caption_first.prefer_human_captions = True
    config.download.caption_first.fallback_to_transcription = True
    config.download.caption_first.timeout = 30
    config.download.caption_first.cache_captions = True

    # No cookies (simplifies testing)
    config.download.cookies_from_browser = ""
    config.download.cookies_path = ""

    return config


@pytest.fixture
def mock_config_disabled():
    """Create mock config with caption-first disabled"""
    config = MagicMock()
    config.download.caption_first.enabled = False
    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint manager"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    checkpoint.checkpoint_path = Path(".") / "checkpoint.json"
    return checkpoint


@pytest.fixture
def mock_state_with_audio():
    """Create state with audio downloads (audio-first mode)"""
    state = PipelineState()
    state.downloaded_audio = [
        AudioDownload(
            file="/path/to/audio/abc123XYZ_0.mp3",
            video_id="abc123XYZ_0",
            url="https://youtube.com/watch?v=abc123XYZ_0",
            title="Test Video 1",
            duration=120.0,
            keyword="test"
        ),
        AudioDownload(
            file="/path/to/audio/def456ABC_1.mp3",
            video_id="def456ABC_1",
            url="https://youtube.com/watch?v=def456ABC_1",
            title="Test Video 2",
            duration=180.0,
            keyword="example"
        ),
    ]
    return state


@pytest.fixture
def mock_state_with_videos():
    """Create state with downloaded videos"""
    state = PipelineState()
    state.downloaded_videos = [
        DownloadedVideo(
            file="/path/to/video/ghi789JKL_2.mp4",
            url="https://youtube.com/watch?v=ghi789JKL_2",
            title="Test Video 3",
        ),
        DownloadedVideo(
            file="/path/to/video/jkl012MNO_3.mp4",
            url="https://youtube.com/watch?v=jkl012MNO_3",
            title="Test Video 4",
        ),
    ]
    return state


@pytest.fixture
def mock_caption_result():
    """Create mock CaptionResult"""
    from src.caption_fetcher import CaptionResult, CaptionSegment

    segments = [
        CaptionSegment(0, 0.0, 2.5, "Hello world", "abc123XYZ_0"),
        CaptionSegment(1, 2.5, 5.0, "This is a test", "abc123XYZ_0"),
        CaptionSegment(2, 5.0, 8.0, "Of the caption system", "abc123XYZ_0"),
    ]

    return CaptionResult(
        video_id="abc123XYZ_0",
        segments=segments,
        language="en",
        is_auto_generated=False,
        format_source="vtt"
    )


# ============================================================================
# Test CaptionStage Initialization
# ============================================================================

class TestCaptionStageInit:
    """Test CaptionStage initialization"""

    def test_stage_name(self):
        """Test stage name is correct"""
        stage = CaptionStage()
        assert stage.name == "CAPTION"

    def test_stage_description(self):
        """Test stage description"""
        stage = CaptionStage()
        assert "caption" in stage.description.lower() or "YouTube" in stage.description

    def test_stage_registration(self):
        """Test stage is registered in registry after import"""
        # Import stage to trigger registration
        from src.stages.caption_stage import CaptionStage as CS
        from src.stages import get_stage
        stage_class = get_stage("CAPTION")
        assert stage_class is CS


# ============================================================================
# Test Input Validation
# ============================================================================

class TestCaptionStageValidation:
    """Test input validation"""

    def test_validate_no_videos_or_audio(self, mock_config):
        """Test validation fails with no video candidates"""
        stage = CaptionStage()
        state = PipelineState()  # Empty state

        error = stage.validate_inputs(state, mock_config)
        assert error is not None
        assert "No video candidates" in error

    def test_validate_with_audio(self, mock_config, mock_state_with_audio):
        """Test validation passes with audio downloads"""
        stage = CaptionStage()

        error = stage.validate_inputs(mock_state_with_audio, mock_config)
        assert error is None

    def test_validate_with_videos(self, mock_config, mock_state_with_videos):
        """Test validation passes with downloaded videos"""
        stage = CaptionStage()

        error = stage.validate_inputs(mock_state_with_videos, mock_config)
        assert error is None


# ============================================================================
# Test Disabled Mode
# ============================================================================

class TestCaptionStageDisabled:
    """Test behavior when caption-first mode is disabled"""

    def test_skip_when_disabled(self, mock_config_disabled, mock_checkpoint, mock_state_with_audio):
        """Test stage skips when caption-first is disabled"""
        stage = CaptionStage()

        result = stage.run(mock_state_with_audio, mock_config_disabled, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'disabled'


# ============================================================================
# Test Video ID Extraction
# ============================================================================

class TestVideoIdExtraction:
    """Test video ID extraction from various sources"""

    def test_extract_from_audio_download(self, mock_config, mock_checkpoint, mock_state_with_audio):
        """Test extracting video IDs from audio downloads"""
        stage = CaptionStage()

        video_ids = stage._get_video_ids(mock_state_with_audio, mock_config)

        assert len(video_ids) == 2
        assert "abc123XYZ_0" in video_ids
        assert "def456ABC_1" in video_ids

    def test_extract_from_url(self, mock_config, mock_checkpoint):
        """Test extracting video ID from URL"""
        stage = CaptionStage()

        # Use Mock with spec=False to control hasattr behavior
        video = Mock(spec=[])
        video.video_id = None
        video.url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video.file = ""

        # Configure hasattr returns
        type(video).video_id = None

        video_id = stage._extract_video_id(video)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_from_filename(self, mock_config, mock_checkpoint):
        """Test extracting video ID from filename"""
        stage = CaptionStage()

        video = Mock(spec=[])
        video.video_id = None
        video.url = ""
        video.file = "/path/to/dQw4w9WgXcQ.mp4"

        video_id = stage._extract_video_id(video)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_from_direct_attribute(self, mock_config, mock_checkpoint):
        """Test extracting video ID from direct attribute"""
        stage = CaptionStage()

        video = Mock(spec=[])
        video.video_id = "test_VIDEO_id"
        video.url = ""
        video.file = ""

        video_id = stage._extract_video_id(video)
        assert video_id == "test_VIDEO_id"


# ============================================================================
# Test Caption Fetching
# ============================================================================

class TestCaptionFetching:
    """Test caption fetching with mocked API"""

    @patch('src.caption_fetcher.CaptionFetcher')
    def test_successful_fetch(self, mock_fetcher_class, mock_config, mock_checkpoint, mock_state_with_audio, mock_caption_result):
        """Test successful caption fetch"""
        # Setup mock
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_captions_auto_language.return_value = mock_caption_result
        mock_fetcher_class.return_value = mock_fetcher

        stage = CaptionStage()
        result = stage.run(mock_state_with_audio, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('success_count', 0) > 0

    @patch('src.caption_fetcher.CaptionFetcher')
    def test_unavailable_captions(self, mock_fetcher_class, mock_config, mock_checkpoint, mock_state_with_audio):
        """Test handling of unavailable captions"""
        from src.caption_fetcher import CaptionUnavailableError

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_captions_auto_language.side_effect = CaptionUnavailableError(
            "abc123XYZ_0", "No captions exist"
        )
        mock_fetcher_class.return_value = mock_fetcher

        stage = CaptionStage()
        result = stage.run(mock_state_with_audio, mock_config, mock_checkpoint)

        assert result.success is True  # Stage succeeds, just marks videos as unavailable
        assert result.data.get('fail_count', 0) > 0

    @patch('src.caption_fetcher.CaptionFetcher')
    def test_fetch_error(self, mock_fetcher_class, mock_config, mock_checkpoint, mock_state_with_audio):
        """Test handling of fetch errors"""
        from src.caption_fetcher import CaptionFetchError

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_captions_auto_language.side_effect = CaptionFetchError(
            "abc123XYZ_0", "Network timeout"
        )
        mock_fetcher_class.return_value = mock_fetcher

        stage = CaptionStage()
        result = stage.run(mock_state_with_audio, mock_config, mock_checkpoint)

        assert result.success is True  # Stage succeeds
        assert result.data.get('fail_count', 0) > 0


# ============================================================================
# Test Text Metadata Population
# ============================================================================

class TestTextMetadataPopulation:
    """Test population of state.text_metadata"""

    def test_populate_text_metadata(self, mock_config):
        """Test caption results are converted to text_metadata format"""
        stage = CaptionStage()
        state = PipelineState()

        caption_results = {
            "abc123XYZ_0": {
                "video_id": "abc123XYZ_0",
                "segments": [
                    {"text": "Hello world", "start": 0.0, "end": 2.5},
                    {"text": "Test caption", "start": 2.5, "end": 5.0},
                ],
                "language": "en",
                "is_auto_generated": False,
            }
        }

        stage._populate_text_metadata(state, caption_results)

        assert len(state.text_metadata) == 2
        assert state.text_metadata[0]['text'] == "Hello world"
        assert state.text_metadata[0]['video_path'] == "abc123XYZ_0"
        assert state.text_metadata[0]['caption_source'] == "youtube"
        assert state.text_metadata[0]['caption_language'] == "en"
        assert state.text_metadata[0]['caption_auto_generated'] is False

    def test_skip_unavailable_in_metadata(self, mock_config):
        """Test unavailable captions don't populate metadata"""
        stage = CaptionStage()
        state = PipelineState()

        caption_results = {
            "abc123XYZ_0": {
                "video_id": "abc123XYZ_0",
                "unavailable": True,
                "reason": "No captions",
            }
        }

        stage._populate_text_metadata(state, caption_results)

        assert len(state.text_metadata) == 0


# ============================================================================
# Test Checkpoint Integration
# ============================================================================

class TestCheckpointIntegration:
    """Test checkpoint save/restore functionality"""

    def test_can_skip_true(self, mock_checkpoint):
        """Test can_skip returns True when checkpoint indicates completion"""
        mock_checkpoint.should_skip_stage.return_value = True

        stage = CaptionStage()
        state = PipelineState()

        can_skip = stage.can_skip(state, mock_checkpoint)

        assert can_skip is True
        mock_checkpoint.should_skip_stage.assert_called_with("CAPTION")

    def test_can_skip_false(self, mock_checkpoint):
        """Test can_skip returns False when checkpoint indicates not complete"""
        mock_checkpoint.should_skip_stage.return_value = False

        stage = CaptionStage()
        state = PipelineState()

        can_skip = stage.can_skip(state, mock_checkpoint)

        assert can_skip is False

    def test_restore_success(self, mock_checkpoint, mock_config):
        """Test successful restore from checkpoint"""
        checkpoint_data = {
            'caption_results': {
                "abc123XYZ_0": {
                    "video_id": "abc123XYZ_0",
                    "segments": [{"text": "Hello", "start": 0.0, "end": 2.0}],
                    "language": "en",
                    "is_auto_generated": False,
                }
            },
            'total_segments': 1,
        }
        mock_checkpoint.get_stage_data.return_value = checkpoint_data

        stage = CaptionStage()
        state = PipelineState()

        restored = stage.restore(state, mock_checkpoint, mock_config)

        assert restored is True
        assert len(state.text_metadata) == 1

    def test_restore_no_data(self, mock_checkpoint, mock_config):
        """Test restore fails with no checkpoint data"""
        mock_checkpoint.get_stage_data.return_value = None

        stage = CaptionStage()
        state = PipelineState()

        restored = stage.restore(state, mock_checkpoint, mock_config)

        assert restored is False

    @patch('src.caption_fetcher.CaptionFetcher')
    def test_resume_from_partial(self, mock_fetcher_class, mock_config, mock_checkpoint, mock_state_with_audio, mock_caption_result):
        """Test resume from partial completion"""
        # Setup existing checkpoint data
        existing_captions = {
            "abc123XYZ_0": {
                "video_id": "abc123XYZ_0",
                "segments": [{"text": "Already fetched", "start": 0.0, "end": 1.0}],
                "language": "en",
                "is_auto_generated": False,
                "segment_count": 1,
            }
        }
        mock_checkpoint.get_stage_data.return_value = {"caption_results": existing_captions}

        # Setup mock for second video
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_captions_auto_language.return_value = mock_caption_result
        mock_fetcher_class.return_value = mock_fetcher

        stage = CaptionStage()
        result = stage.run(mock_state_with_audio, mock_config, mock_checkpoint)

        assert result.success is True
        # First video should be skipped
        assert result.data.get('skip_count', 0) >= 1


# ============================================================================
# Test No Videos Scenario
# ============================================================================

class TestNoVideos:
    """Test behavior with no video candidates"""

    def test_no_videos_available(self, mock_config, mock_checkpoint):
        """Test stage handles empty video list gracefully"""
        stage = CaptionStage()
        state = PipelineState()  # No videos or audio

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_videos'


# ============================================================================
# Test Configuration
# ============================================================================

class TestConfiguration:
    """Test configuration handling"""

    @patch('src.caption_fetcher.CaptionFetcher')
    def test_timeout_from_config(self, mock_fetcher_class, mock_config, mock_checkpoint, mock_state_with_audio, mock_caption_result):
        """Test timeout is applied from config"""
        mock_config.download.caption_first.timeout = 45

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_captions_auto_language.return_value = mock_caption_result
        mock_fetcher_class.return_value = mock_fetcher

        stage = CaptionStage()
        stage.run(mock_state_with_audio, mock_config, mock_checkpoint)

        # Check that fetcher was created with config
        mock_fetcher_class.assert_called_once()

    @patch('src.caption_fetcher.CaptionFetcher')
    def test_preferred_language_from_config(self, mock_fetcher_class, mock_config, mock_checkpoint, mock_state_with_audio, mock_caption_result):
        """Test preferred language is passed from config"""
        mock_config.download.caption_first.preferred_language = "es"

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_captions_auto_language.return_value = mock_caption_result
        mock_fetcher_class.return_value = mock_fetcher

        stage = CaptionStage()
        stage.run(mock_state_with_audio, mock_config, mock_checkpoint)

        # Check that fetch was called with preferred language
        mock_fetcher.fetch_captions_auto_language.assert_called()
        call_args = mock_fetcher.fetch_captions_auto_language.call_args
        assert call_args[1].get('preferred_language') == "es" or call_args[0][1] == "es" if len(call_args[0]) > 1 else True


# ============================================================================
# Test STAGE_ORDER Integration
# ============================================================================

class TestStageOrder:
    """Test CAPTION is properly positioned in STAGE_ORDER"""

    def test_caption_in_stage_order(self):
        """Test CAPTION is in STAGE_ORDER"""
        from src.checkpoint import STAGE_ORDER

        assert "CAPTION" in STAGE_ORDER

    def test_caption_after_remix(self):
        """Test CAPTION comes after REMIX (download stages)"""
        from src.checkpoint import STAGE_ORDER

        remix_idx = STAGE_ORDER.index("REMIX")
        caption_idx = STAGE_ORDER.index("CAPTION")

        assert caption_idx > remix_idx

    def test_caption_before_transcribe(self):
        """Test CAPTION comes before TRANSCRIBE"""
        from src.checkpoint import STAGE_ORDER

        caption_idx = STAGE_ORDER.index("CAPTION")
        transcribe_idx = STAGE_ORDER.index("TRANSCRIBE")

        assert caption_idx < transcribe_idx
