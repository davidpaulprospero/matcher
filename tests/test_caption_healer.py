"""
Unit tests for CaptionHealer.

Tests caption-specific error recovery strategies.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.healers.caption import CaptionHealer
from src.agents.base import HealerAction


class TestCaptionHealerBasic:
    """Basic tests for CaptionHealer"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.enabled = True
        config.download.caption_first.fallback_to_audio = False
        config.download.caption_first.languages = ["en"]
        config.download.caption_first.fetch_timeout = 30
        config.cache = Mock()
        config.cache.cache_dir = tempfile.gettempdir()
        return config

    @pytest.fixture
    def healer(self, mock_config):
        """Create CaptionHealer instance"""
        return CaptionHealer(mock_config, Path(tempfile.gettempdir()))

    @pytest.fixture
    def mock_state(self):
        """Create mock pipeline state"""
        state = Mock()
        state.videos_need_audio = []
        state.caption_downloads = []
        return state

    def test_healer_attributes(self, healer):
        """Test CaptionHealer has correct attributes"""
        assert healer.name == "caption-healer"
        assert healer.description == "Fix caption fetching errors"
        assert "CAPTION" in healer.handled_stages

    def test_can_handle_caption_stage(self, healer):
        """Test can_handle returns True for CAPTION stage errors"""
        error = Exception("Caption fetch failed")
        assert healer.can_handle(error, "CAPTION") is True

    def test_can_handle_caption_keywords(self, healer):
        """Test can_handle returns True for caption-related errors"""
        errors = [
            Exception("No captions available"),
            Exception("Subtitle download failed"),
            Exception("Invalid SRT file"),
            Exception("VTT parse error"),
        ]
        for error in errors:
            assert healer.can_handle(error, "DOWNLOAD") is True

    def test_can_handle_ignores_download_errors(self, healer):
        """Test can_handle returns False for non-caption download errors"""
        error = Exception("Video download timeout")
        assert healer.can_handle(error, "DOWNLOAD") is False

    def test_error_patterns(self, healer):
        """Test all error patterns are lowercase for matching"""
        for pattern in healer.error_patterns:
            assert pattern == pattern.lower()


class TestCaptionHealerNoCaptions:
    """Tests for handling no captions available"""

    @pytest.fixture
    def healer(self):
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.fallback_to_audio = False
        return CaptionHealer(config, Path(tempfile.gettempdir()))

    @pytest.fixture
    def mock_state(self):
        state = Mock()
        state.videos_need_audio = []
        return state

    def test_enables_fallback(self, healer, mock_state):
        """Test that no captions error enables audio fallback"""
        error = Exception("No captions available for video")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        assert result.details.get('fallback_enabled') is True

    def test_extracts_video_id(self, healer, mock_state):
        """Test video ID extraction from error message"""
        # Use error with URL pattern that the extractor can match
        error = Exception("No captions for https://youtube.com/watch?v=dQw4w9WgXcQ")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        assert result.details.get('video_id') == 'dQw4w9WgXcQ'

    def test_adds_to_videos_need_audio(self, healer, mock_state):
        """Test video is added to fallback list"""
        # Use error with quote pattern that the extractor can match
        error = Exception("No captions available for 'dQw4w9WgXcQ'")

        healer.fix(error, mock_state, "CAPTION")

        assert 'dQw4w9WgXcQ' in mock_state.videos_need_audio


class TestCaptionHealerLanguageError:
    """Tests for handling language not available"""

    @pytest.fixture
    def healer(self):
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.languages = ["en"]
        config.download.caption_first.fallback_to_audio = True
        return CaptionHealer(config, Path(tempfile.gettempdir()))

    @pytest.fixture
    def mock_state(self):
        state = Mock()
        state.videos_need_audio = []
        return state

    def test_tries_alternate_languages(self, healer, mock_state):
        """Test healer tries alternate language codes"""
        error = Exception("Language 'en' not available")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        assert result.modified_config is True
        assert 'new_languages' in result.details

    def test_exhausts_languages_enables_fallback(self, healer, mock_state):
        """Test fallback is enabled when all languages exhausted"""
        error = Exception("Language not available")

        # Exhaust all language groups
        for _ in range(len(CaptionHealer.LANGUAGE_FALLBACKS) + 1):
            result = healer.fix(error, mock_state, "CAPTION")

        # Should eventually fall back to audio
        assert result.details.get('fallback_enabled') is True


class TestCaptionHealerParseError:
    """Tests for handling caption parse errors"""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def healer(self, temp_cache_dir):
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = str(temp_cache_dir)
        config.download = Mock()
        config.download.caption_first = None
        return CaptionHealer(config, temp_cache_dir)

    @pytest.fixture
    def mock_state(self):
        state = Mock()
        state.videos_need_audio = []
        return state

    def test_handles_parse_error(self, healer, mock_state):
        """Test handling of caption parse errors"""
        error = Exception("Parse error in caption file")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        assert result.action == HealerAction.RETRY

    def test_removes_corrupted_caption(self, healer, mock_state, temp_cache_dir):
        """Test corrupted caption file is removed"""
        # Create caption directory and file
        caption_dir = temp_cache_dir / "captions"
        caption_dir.mkdir()
        corrupted_file = caption_dir / "dQw4w9WgXcQ.en.srt"
        corrupted_file.write_text("corrupted content")

        error = Exception("Parse error in dQw4w9WgXcQ.srt")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        # File should be removed
        assert not corrupted_file.exists()


class TestCaptionHealerRateLimit:
    """Tests for handling rate limiting"""

    @pytest.fixture
    def healer(self):
        config = Mock()
        config.download = Mock()
        config.download.caption_first = None
        return CaptionHealer(config, Path(tempfile.gettempdir()))

    @pytest.fixture
    def mock_state(self):
        return Mock()

    @patch('time.sleep')
    def test_handles_rate_limit_with_backoff(self, mock_sleep, healer, mock_state):
        """Test rate limit handling with exponential backoff"""
        error = Exception("429 Too Many Requests")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        assert result.action == HealerAction.RETRY
        assert 'backoff_seconds' in result.details
        mock_sleep.assert_called_once()

    @patch('time.sleep')
    def test_backoff_increases(self, mock_sleep, healer, mock_state):
        """Test backoff time increases with retries"""
        error = Exception("Rate limit exceeded")

        initial_backoff = healer.backoff_time

        result1 = healer.fix(error, mock_state, "CAPTION")
        new_backoff = healer.backoff_time

        assert new_backoff > initial_backoff


class TestCaptionHealerTimeout:
    """Tests for handling timeout errors"""

    @pytest.fixture
    def healer(self):
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.fetch_timeout = 30
        return CaptionHealer(config, Path(tempfile.gettempdir()))

    @pytest.fixture
    def mock_state(self):
        return Mock()

    def test_increases_timeout(self, healer, mock_state):
        """Test timeout is increased on timeout error"""
        error = Exception("Caption fetch timeout")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        assert result.modified_config is True
        assert result.details.get('new_timeout', 0) > 30


class TestCaptionHealerGenericError:
    """Tests for handling generic errors"""

    @pytest.fixture
    def healer(self):
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.fallback_to_audio = False
        return CaptionHealer(config, Path(tempfile.gettempdir()))

    @pytest.fixture
    def mock_state(self):
        return Mock()

    @patch('time.sleep')
    def test_retries_on_generic_error(self, mock_sleep, healer, mock_state):
        """Test generic errors trigger retry"""
        error = Exception("Unknown caption error")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        assert result.action == HealerAction.RETRY

    @patch('time.sleep')
    def test_enables_fallback_after_retries(self, mock_sleep, healer, mock_state):
        """Test fallback is enabled after multiple retries"""
        error = Exception("Persistent caption error")

        # Retry multiple times
        for _ in range(5):
            result = healer.fix(error, mock_state, "CAPTION")

        assert result.details.get('fallback_enabled') is True


class TestCaptionHealerVideoIdExtraction:
    """Tests for video ID extraction"""

    @pytest.fixture
    def healer(self):
        config = Mock()
        config.download = Mock()
        config.download.caption_first = None
        return CaptionHealer(config, Path(tempfile.gettempdir()))

    def test_extract_from_url_pattern(self, healer):
        """Test extraction from URL-like pattern"""
        error_str = "Error for video https://youtube.com/watch?v=dQw4w9WgXcQ"
        vid = healer._extract_video_id(error_str)
        assert vid == "dQw4w9WgXcQ"

    def test_extract_from_filename(self, healer):
        """Test extraction from caption filename"""
        error_str = "Parse error in dQw4w9WgXcQ.srt"
        vid = healer._extract_video_id(error_str)
        assert vid == "dQw4w9WgXcQ"

    def test_extract_from_quoted_id(self, healer):
        """Test extraction from quoted video ID"""
        error_str = "Video ID: 'dQw4w9WgXcQ' not found"
        vid = healer._extract_video_id(error_str)
        assert vid == "dQw4w9WgXcQ"

    def test_returns_none_for_no_match(self, healer):
        """Test returns None when no video ID found"""
        error_str = "Generic error with no video ID"
        vid = healer._extract_video_id(error_str)
        assert vid is None


class TestCaptionHealerReset:
    """Tests for healer reset functionality"""

    @pytest.fixture
    def healer(self):
        config = Mock()
        config.download = Mock()
        config.download.caption_first = None
        return CaptionHealer(config, Path(tempfile.gettempdir()))

    def test_reset_clears_state(self, healer):
        """Test reset clears all healer state"""
        healer.backoff_time = 100
        healer.retry_count = 5
        healer.language_attempt_index = 3
        healer.skipped_videos.add("test123")

        healer.reset()

        assert healer.backoff_time == CaptionHealer.INITIAL_BACKOFF
        assert healer.retry_count == 0
        assert healer.language_attempt_index == 0
        # Note: skipped_videos is not reset by reset()

    def test_get_skipped_videos(self, healer):
        """Test get_skipped_videos returns copy"""
        healer.skipped_videos.add("vid1")
        healer.skipped_videos.add("vid2")

        skipped = healer.get_skipped_videos()

        assert skipped == {"vid1", "vid2"}
        # Verify it's a copy
        skipped.add("vid3")
        assert "vid3" not in healer.skipped_videos


class TestCaptionHealerIntegration:
    """Integration tests for CaptionHealer"""

    def test_healer_in_registry(self):
        """Test CaptionHealer is in the healer registry"""
        from src.agents import HEALER_REGISTRY, CaptionHealer

        healer_classes = [h.__name__ for h in HEALER_REGISTRY]
        assert 'CaptionHealer' in healer_classes

    def test_healer_before_download_healer(self):
        """Test CaptionHealer comes before DownloadHealer in registry"""
        from src.agents import HEALER_REGISTRY

        healer_names = [h.__name__ for h in HEALER_REGISTRY]
        caption_idx = healer_names.index('CaptionHealer')
        download_idx = healer_names.index('DownloadHealer')

        assert caption_idx < download_idx


class TestCheckpointHealerCaptionSupport:
    """Tests for CheckpointHealer captions cache detection"""

    @pytest.fixture
    def temp_project_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        return config

    def test_rebuild_detects_captions_cache(self, temp_project_dir, mock_config):
        """Test that _rebuild_checkpoint detects captions cache"""
        from src.agents.healers.checkpoint import CheckpointHealer

        # Create captions cache with a file
        caption_cache = temp_project_dir / ".cache" / "captions"
        caption_cache.mkdir(parents=True)
        (caption_cache / "dQw4w9WgXcQ.en.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nTest")

        healer = CheckpointHealer(mock_config, temp_project_dir)
        mock_state = Mock()

        error = Exception("Missing checkpoint field")
        result = healer._rebuild_checkpoint(error, mock_state)

        assert result.success is True
        assert "CAPTION" in result.details.get('cached_stages', [])

    def test_rebuild_without_captions_cache(self, temp_project_dir, mock_config):
        """Test rebuild without captions cache"""
        from src.agents.healers.checkpoint import CheckpointHealer

        # Create other caches but not captions
        trans_cache = temp_project_dir / ".cache" / "transcriptions"
        trans_cache.mkdir(parents=True)
        (trans_cache / "test.json").write_text("{}")

        healer = CheckpointHealer(mock_config, temp_project_dir)
        mock_state = Mock()

        error = Exception("Missing checkpoint field")
        result = healer._rebuild_checkpoint(error, mock_state)

        assert result.success is True
        assert "TRANSCRIBE" in result.details.get('cached_stages', [])
        assert "CAPTION" not in result.details.get('cached_stages', [])


class TestCheckpointMigrationCaption:
    """Tests for checkpoint migration with CAPTION stage"""

    @pytest.fixture
    def temp_project_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_migration_includes_caption_stage(self, temp_project_dir):
        """Test that migration maps CAPTION stage correctly"""
        from src.checkpoint import CheckpointManager

        # Create old format checkpoint with CAPTION data
        old_checkpoint = {
            'version': '0.9',
            'created_at': '2026-01-15T12:00:00',
            'last_completed_stage': 'CAPTION',
            'CAPTION': {
                'caption_count': 5,
                'fallback_count': 2
            }
        }

        checkpoint_path = temp_project_dir / "checkpoint.json"
        import json
        with open(checkpoint_path, 'w') as f:
            json.dump(old_checkpoint, f)

        manager = CheckpointManager(temp_project_dir)
        data = manager.load()

        assert data is not None
        assert data.version == '1.0'
        assert data.caption.get('caption_count') == 5
        assert data.caption.get('fallback_count') == 2

    def test_summary_includes_caption(self, temp_project_dir):
        """Test that get_summary includes CAPTION stage"""
        from src.checkpoint import CheckpointManager

        checkpoint_data = {
            'version': '1.0',
            'created_at': '2026-01-15T12:00:00',
            'updated_at': '2026-01-15T12:30:00',
            'last_completed_stage': 'TRANSCRIBE',
            'caption': {
                'caption_count': 10,
                'fallback_count': 3
            }
        }

        checkpoint_path = temp_project_dir / "checkpoint.json"
        import json
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(temp_project_dir)
        manager.load()
        summary = manager.get_summary()

        assert "CAPTION" in summary
        assert "10 captions" in summary
        assert "3 need audio fallback" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
