"""
Tests for VideoMetadataStage - caption-first video metadata fetching.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.state import PipelineState, VideoCandidate
from src.stages.video_metadata import VideoMetadataStage


class TestVideoCandidate:
    """Tests for VideoCandidate dataclass."""

    def test_create_basic(self):
        """Test basic VideoCandidate creation."""
        vc = VideoCandidate(
            video_id="dQw4w9WgXcQ",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        assert vc.video_id == "dQw4w9WgXcQ"
        assert vc.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert vc.has_captions is False
        assert vc.transcript_source == ""

    def test_create_full(self):
        """Test full VideoCandidate creation with all fields."""
        vc = VideoCandidate(
            video_id="abc123XYZ_-",
            url="https://www.youtube.com/watch?v=abc123XYZ_-",
            title="Test Video Title",
            channel="Test Channel",
            duration=300.5,
            duration_tier="medium",
            keyword="test keyword",
            upload_date="20240115",
            has_captions=True,
            caption_language="en",
            is_auto_caption=False,
            transcript_source="manual_caption"
        )
        assert vc.title == "Test Video Title"
        assert vc.duration == 300.5
        assert vc.has_captions is True
        assert vc.is_auto_caption is False
        assert vc.transcript_source == "manual_caption"

    def test_to_dict(self):
        """Test VideoCandidate serialization."""
        vc = VideoCandidate(
            video_id="test123",
            url="https://youtube.com/watch?v=test123",
            title="Test",
            has_captions=True
        )
        d = vc.to_dict()
        assert d['video_id'] == "test123"
        assert d['has_captions'] is True
        assert 'transcript_source' in d

    def test_from_dict(self):
        """Test VideoCandidate deserialization."""
        data = {
            'video_id': 'xyz789',
            'url': 'https://youtube.com/watch?v=xyz789',
            'title': 'Restored Video',
            'has_captions': True,
            'caption_language': 'es'
        }
        vc = VideoCandidate.from_dict(data)
        assert vc.video_id == 'xyz789'
        assert vc.title == 'Restored Video'
        assert vc.has_captions is True
        assert vc.caption_language == 'es'

    def test_from_dict_missing_fields(self):
        """Test VideoCandidate deserialization with missing optional fields."""
        data = {
            'video_id': 'min123',
            'url': 'https://youtube.com/watch?v=min123'
        }
        vc = VideoCandidate.from_dict(data)
        assert vc.video_id == 'min123'
        assert vc.title == ''  # Default
        assert vc.has_captions is False  # Default


class TestVideoMetadataStage:
    """Tests for VideoMetadataStage."""

    @pytest.fixture
    def stage(self):
        return VideoMetadataStage()

    @pytest.fixture
    def mock_config(self):
        """Create mock config with caption_first enabled."""
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.enabled = True
        config.download.search_pool_multiplier = 5
        config.download.max_search_pool = 50
        config.download.search_timeout = 30
        config.download.delay_between_keywords = 0
        config.download.title_blacklist = []
        config.download.cookies_from_browser = ""
        config.download.cookies_path = ""
        config.duration_tiers = None
        return config

    @pytest.fixture
    def mock_state(self):
        """Create mock pipeline state."""
        state = PipelineState()
        state.keywords = ["test keyword 1", "test keyword 2"]
        state.topic_context = "test topic"
        return state

    @pytest.fixture
    def mock_checkpoint(self):
        """Create mock checkpoint manager."""
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False
        checkpoint.get_stage_data.return_value = {}
        return checkpoint

    def test_stage_name(self, stage):
        """Test stage has correct name."""
        assert stage.name == "VIDEO_METADATA"

    def test_skips_when_caption_first_disabled(self, stage, mock_state, mock_checkpoint):
        """Test stage skips when caption-first mode is disabled."""
        config = Mock()
        config.download.caption_first = None

        result = stage.run(mock_state, config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'not_enabled'

    def test_skips_when_no_keywords(self, stage, mock_config, mock_checkpoint):
        """Test stage skips when no keywords available."""
        state = PipelineState()
        state.keywords = []

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_keywords'

    def test_is_blacklisted(self, stage):
        """Test title blacklist filtering."""
        blacklist = ["reaction", "compilation"]

        assert stage._is_blacklisted("My Reaction Video", blacklist) is True
        assert stage._is_blacklisted("REACTION to new song", blacklist) is True
        assert stage._is_blacklisted("Normal Video Title", blacklist) is False
        assert stage._is_blacklisted("Best compilation ever", blacklist) is True

    def test_get_duration_tier(self, stage):
        """Test duration tier classification."""
        tier_config = {
            'short': {'min': 10, 'max': 120},
            'medium': {'min': 120, 'max': 600},
            'long': {'min': 600, 'max': 1500},
        }

        assert stage._get_duration_tier(60, tier_config) == 'short'
        assert stage._get_duration_tier(300, tier_config) == 'medium'
        assert stage._get_duration_tier(900, tier_config) == 'long'
        assert stage._get_duration_tier(5, tier_config) == 'medium'  # Below short

    def test_can_skip(self, stage, mock_state, mock_checkpoint):
        """Test can_skip delegates to checkpoint."""
        mock_checkpoint.should_skip_stage.return_value = True
        assert stage.can_skip(mock_state, mock_checkpoint) is True

        mock_checkpoint.should_skip_stage.return_value = False
        assert stage.can_skip(mock_state, mock_checkpoint) is False

    def test_validate_inputs_caption_first_enabled(self, stage, mock_config):
        """Test validate_inputs when caption-first is enabled."""
        state = PipelineState()
        state.keywords = []

        error = stage.validate_inputs(state, mock_config)
        assert error is not None
        assert "keywords" in error.lower()

    def test_validate_inputs_caption_first_disabled(self, stage):
        """Test validate_inputs when caption-first is disabled."""
        config = Mock()
        config.download.caption_first = None

        state = PipelineState()
        state.keywords = []

        # Should pass validation since stage will be skipped
        error = stage.validate_inputs(state, config)
        assert error is None

    def test_restore_from_checkpoint(self, stage, mock_checkpoint, mock_config):
        """Test restore from checkpoint data."""
        mock_checkpoint.get_stage_data.return_value = {
            'video_candidates': [
                {
                    'video_id': 'vid1',
                    'url': 'https://youtube.com/watch?v=vid1',
                    'title': 'Video 1',
                    'has_captions': True
                },
                {
                    'video_id': 'vid2',
                    'url': 'https://youtube.com/watch?v=vid2',
                    'title': 'Video 2',
                    'has_captions': False
                }
            ]
        }

        state = PipelineState()
        result = stage.restore(state, mock_checkpoint, mock_config)

        assert result is True
        assert len(state.video_candidates) == 2
        assert state.video_candidates[0].video_id == 'vid1'
        assert state.video_candidates[0].has_captions is True

    def test_restore_fallback_to_video_ids(self, stage, mock_checkpoint, mock_config):
        """Test restore falls back to video_ids when full data not available."""
        mock_checkpoint.get_stage_data.return_value = {
            'video_ids': ['vid1', 'vid2', 'vid3']
        }

        state = PipelineState()
        result = stage.restore(state, mock_checkpoint, mock_config)

        assert result is True
        assert len(state.video_candidates) == 3
        assert state.video_candidates[0].video_id == 'vid1'


class TestVideoMetadataStageIntegration:
    """Integration tests for VideoMetadataStage with mocked yt-dlp."""

    @pytest.fixture
    def stage(self):
        return VideoMetadataStage()

    @patch('subprocess.run')
    def test_search_keyword_parses_ytdlp_output(self, mock_run, stage):
        """Test _search_keyword parses yt-dlp JSON output correctly."""
        # Mock yt-dlp output (newline-delimited JSON)
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"id": "vid1", "title": "Test Video 1", "duration": 300, "channel": "Channel1"}\n'
                   '{"id": "vid2", "title": "Test Video 2", "duration": 450, "channel": "Channel2"}\n',
            stderr=''
        )

        config = Mock()
        config.download = Mock()
        config.download.search_pool_multiplier = 5
        config.download.max_search_pool = 50
        config.download.search_timeout = 30
        config.download.title_blacklist = []

        tier_config = {
            'medium': {'min': 60, 'max': 600},
        }

        candidates = stage._search_keyword(
            keyword="test",
            topic="",
            tier_config=tier_config,
            config=config
        )

        assert len(candidates) == 2
        assert candidates[0].video_id == 'vid1'
        assert candidates[0].title == 'Test Video 1'
        assert candidates[0].duration == 300
        assert candidates[1].video_id == 'vid2'

    @patch('subprocess.run')
    def test_search_keyword_filters_blacklisted(self, mock_run, stage):
        """Test _search_keyword filters blacklisted titles."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"id": "vid1", "title": "Good Video", "duration": 300}\n'
                   '{"id": "vid2", "title": "Bad Reaction Video", "duration": 300}\n',
            stderr=''
        )

        config = Mock()
        config.download = Mock()
        config.download.search_pool_multiplier = 5
        config.download.max_search_pool = 50
        config.download.search_timeout = 30
        config.download.title_blacklist = ["reaction"]

        tier_config = {'medium': {'min': 60, 'max': 600}}

        candidates = stage._search_keyword(
            keyword="test",
            topic="",
            tier_config=tier_config,
            config=config
        )

        assert len(candidates) == 1
        assert candidates[0].video_id == 'vid1'

    @patch('subprocess.run')
    def test_search_keyword_handles_timeout(self, mock_run, stage):
        """Test _search_keyword handles subprocess timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="yt-dlp", timeout=30)

        config = Mock()
        config.download = Mock()
        config.download.search_pool_multiplier = 5
        config.download.max_search_pool = 50
        config.download.search_timeout = 30
        config.download.title_blacklist = []

        tier_config = {'medium': {'min': 60, 'max': 600}}

        candidates = stage._search_keyword(
            keyword="test",
            topic="",
            tier_config=tier_config,
            config=config
        )

        assert candidates == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
