"""
Tests for individual healer classes (API, Checkpoint, Download, Disk, Path).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from src.agents.healers.api import APIHealer
from src.agents.healers.checkpoint import CheckpointHealer
from src.agents.healers.download import DownloadHealer
from src.agents.healers.disk import DiskHealer
from src.agents.healers.path import PathHealer
from src.agents.base import HealerResult, HealerAction


# =============================================================================
# API Healer Tests
# =============================================================================

class TestAPIHealer:
    """Tests for APIHealer."""

    @pytest.mark.fast
    def test_error_patterns(self, mock_config, project_dir):
        """Test healer has correct error patterns."""
        healer = APIHealer(mock_config, project_dir)

        assert "rate limit" in healer.error_patterns
        assert "429" in healer.error_patterns
        assert "timeout" in healer.error_patterns
        assert "gemini" in healer.error_patterns

    @pytest.mark.fast
    def test_can_handle_rate_limit(self, mock_config, project_dir):
        """Test can_handle matches rate limit errors."""
        healer = APIHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("Rate limit exceeded"), "MATCH")
        assert healer.can_handle(Exception("Error 429: Too many requests"), "MATCH")

    @pytest.mark.fast
    def test_can_handle_auth_errors(self, mock_config, project_dir):
        """Test can_handle matches authentication errors."""
        healer = APIHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("401 Unauthorized"), "MATCH")
        assert healer.can_handle(Exception("403 Forbidden"), "MATCH")
        assert healer.can_handle(Exception("Invalid API key"), "MATCH")

    @patch('time.sleep')
    @pytest.mark.fast
    def test_handle_rate_limit_backoff(self, mock_sleep, mock_config, project_dir):
        """Test rate limit triggers exponential backoff."""
        healer = APIHealer(mock_config, project_dir)
        state = Mock()

        result = healer._handle_rate_limit(Exception("rate limit"), state)

        assert result.success
        assert result.action == HealerAction.RETRY
        mock_sleep.assert_called()

    @patch('time.sleep')
    @pytest.mark.fast
    def test_backoff_increases(self, mock_sleep, mock_config, project_dir):
        """Test backoff time increases exponentially."""
        healer = APIHealer(mock_config, project_dir)
        state = Mock()

        initial_backoff = healer.backoff_time

        healer._handle_rate_limit(Exception("rate limit"), state)

        assert healer.backoff_time > initial_backoff
        assert healer.backoff_time <= healer.MAX_BACKOFF

    @pytest.mark.fast
    def test_reset_backoff(self, mock_config, project_dir):
        """Test reset_backoff resets state."""
        healer = APIHealer(mock_config, project_dir)

        # Increase backoff
        healer.backoff_time = 100.0
        healer.retry_count = 5

        healer.reset_backoff()

        assert healer.backoff_time == healer.INITIAL_BACKOFF
        assert healer.retry_count == 0

    @patch('os.environ.get')
    @pytest.mark.fast
    def test_is_provider_available(self, mock_env, mock_config, project_dir):
        """Test provider availability check."""
        healer = APIHealer(mock_config, project_dir)

        # Test with API key set
        mock_env.return_value = "test-key"
        assert healer._is_provider_available("gemini") is True

        # Test with no API key
        mock_env.return_value = None
        assert healer._is_provider_available("gemini") is False

        # Ollama doesn't need API key
        assert healer._is_provider_available("ollama") is True

    @pytest.mark.fast
    def test_handle_timeout_increases_config(self, mock_config, project_dir):
        """Test timeout handling increases timeout in config."""
        healer = APIHealer(mock_config, project_dir)
        state = Mock()

        mock_config.llm.timeout = 30

        result = healer._handle_timeout(Exception("request timed out"), state)

        assert result.success
        assert mock_config.llm.timeout > 30


# =============================================================================
# Checkpoint Healer Tests
# =============================================================================

class TestCheckpointHealer:
    """Tests for CheckpointHealer."""

    @pytest.mark.fast
    def test_error_patterns(self, mock_config, project_dir):
        """Test healer has correct error patterns."""
        healer = CheckpointHealer(mock_config, project_dir)

        assert "json" in healer.error_patterns
        assert "checkpoint" in healer.error_patterns
        assert "corrupt" in healer.error_patterns

    @pytest.mark.fast
    def test_can_handle_json_errors(self, mock_config, project_dir):
        """Test can_handle matches JSON decode errors."""
        healer = CheckpointHealer(mock_config, project_dir)

        assert healer.can_handle(json.JSONDecodeError("test", "doc", 0), "LOAD")
        assert healer.can_handle(Exception("JSON decode failed"), "LOAD")

    @pytest.mark.fast
    def test_restore_from_backup_no_backup(self, mock_config, project_dir):
        """Test restore fails gracefully when no backup exists."""
        healer = CheckpointHealer(mock_config, project_dir)
        state = Mock()

        result = healer._restore_from_backup(Exception("test"), state)

        # Should fall through to start_fresh
        assert isinstance(result, HealerResult)

    @pytest.mark.fast
    def test_restore_from_backup_success(self, mock_config, project_dir):
        """Test restore succeeds with valid backup."""
        healer = CheckpointHealer(mock_config, project_dir)
        state = Mock()

        # Create valid backup
        backup_path = project_dir / healer.BACKUP_FILE
        backup_data = {"stages": {"ANALYZE": {}, "DOWNLOAD": {}}}
        backup_path.write_text(json.dumps(backup_data))

        result = healer._restore_from_backup(Exception("test"), state)

        assert result.success
        assert result.action == HealerAction.RESTORE

    @pytest.mark.fast
    def test_restore_from_backup_corrupted(self, mock_config, project_dir):
        """Test restore handles corrupted backup."""
        healer = CheckpointHealer(mock_config, project_dir)
        state = Mock()

        # Create corrupted backup
        backup_path = project_dir / healer.BACKUP_FILE
        backup_path.write_text("{invalid json")

        result = healer._restore_from_backup(Exception("test"), state)

        # Should fall through to start_fresh
        assert isinstance(result, HealerResult)

    @pytest.mark.fast
    def test_handle_hash_mismatch(self, mock_config, project_dir):
        """Test hash mismatch continues with current config."""
        healer = CheckpointHealer(mock_config, project_dir)
        state = Mock()

        result = healer._handle_hash_mismatch(Exception("hash mismatch"), state)

        assert result.success
        assert result.action == HealerAction.RETRY

    @pytest.mark.fast
    def test_start_fresh_moves_corrupt(self, mock_config, project_dir):
        """Test start_fresh moves corrupted checkpoint."""
        healer = CheckpointHealer(mock_config, project_dir)
        state = Mock()

        # Create corrupted checkpoint
        checkpoint_path = project_dir / healer.CHECKPOINT_FILE
        checkpoint_path.write_text("{bad json")

        result = healer._start_fresh(Exception("test"), state)

        assert result.success
        assert not checkpoint_path.exists()
        assert (project_dir / "checkpoint.corrupted.json").exists()

    @pytest.mark.fast
    def test_create_backup(self, mock_config, project_dir):
        """Test create_backup creates backup file."""
        healer = CheckpointHealer(mock_config, project_dir)

        # Create checkpoint
        checkpoint_path = project_dir / healer.CHECKPOINT_FILE
        checkpoint_path.write_text('{"test": true}')

        result = healer.create_backup()

        assert result is True
        assert (project_dir / healer.BACKUP_FILE).exists()

    @pytest.mark.fast
    def test_create_backup_no_checkpoint(self, mock_config, project_dir):
        """Test create_backup returns False when no checkpoint."""
        healer = CheckpointHealer(mock_config, project_dir)

        result = healer.create_backup()

        assert result is False

    @pytest.mark.fast
    def test_rebuild_checkpoint_finds_cache(self, mock_config, project_dir):
        """Test rebuild_checkpoint finds cached stage data."""
        healer = CheckpointHealer(mock_config, project_dir)
        state = Mock()

        # Create cache directories with data (current pipeline caches)
        caption_cache = project_dir / ".cache" / "captions"
        caption_cache.mkdir(parents=True)
        (caption_cache / "test.json").write_text("{}")

        result = healer._rebuild_checkpoint(Exception("test"), state)

        assert result.success
        assert "CAPTION" in result.details.get("cached_stages", [])


# =============================================================================
# Download Healer Tests
# =============================================================================

class TestDownloadHealer:
    """Tests for DownloadHealer."""

    @pytest.mark.fast
    def test_error_patterns(self, mock_config, project_dir):
        """Test healer has correct error patterns."""
        healer = DownloadHealer(mock_config, project_dir)

        assert "yt-dlp" in healer.error_patterns
        assert "youtube" in healer.error_patterns
        assert "429" in healer.error_patterns
        assert "unavailable" in healer.error_patterns

    @pytest.mark.fast
    def test_can_handle_download_errors(self, mock_config, project_dir):
        """Test can_handle matches download errors."""
        healer = DownloadHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("yt-dlp error"), "DOWNLOAD")
        assert healer.can_handle(Exception("YouTube rate limit"), "DOWNLOAD")
        assert healer.can_handle(Exception("Video unavailable"), "DOWNLOAD")

    @pytest.mark.fast
    def test_can_handle_403_forbidden(self, mock_config, project_dir):
        """Test can_handle returns True for 403 Forbidden download errors."""
        healer = DownloadHealer(mock_config, project_dir)

        # 403 errors in real yt-dlp output include "download" or "yt-dlp" context
        assert healer.can_handle(Exception("yt-dlp: HTTP Error 403: Forbidden"), "DOWNLOAD")
        assert healer.can_handle(Exception("Download failed: 403 Forbidden"), "DOWNLOAD")
        # Error messages with "429" rate limit
        assert healer.can_handle(Exception("429 Too Many Requests"), "DOWNLOAD")

    @pytest.mark.fast
    def test_can_handle_extraction_failed(self, mock_config, project_dir):
        """Test can_handle returns True for yt-dlp extraction failures."""
        healer = DownloadHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("yt-dlp extraction failed for video"), "DOWNLOAD")
        assert healer.can_handle(Exception("Unable to extract video data"), "DOWNLOAD")

    @pytest.mark.fast
    def test_can_handle_false_for_config_errors(self, mock_config, project_dir):
        """Test can_handle returns False for config-related errors."""
        healer = DownloadHealer(mock_config, project_dir)

        assert not healer.can_handle(Exception("Invalid config key 'output.frame_rate'"), "OUTPUT")
        assert not healer.can_handle(Exception("Missing required config section 'llm'"), "ANALYZE")

    @pytest.mark.fast
    def test_can_handle_false_for_otio_errors(self, mock_config, project_dir):
        """Test can_handle returns False for OTIO-related errors."""
        healer = DownloadHealer(mock_config, project_dir)

        assert not healer.can_handle(Exception("OTIO serialization failed"), "OUTPUT")
        assert not healer.can_handle(Exception("opentimelineio import error"), "OUTPUT")
        assert not healer.can_handle(Exception("Track layout corruption detected"), "OUTPUT")

    @patch('time.sleep')
    @pytest.mark.fast
    def test_handle_rate_limit(self, mock_sleep, mock_config, project_dir):
        """Test rate limit triggers backoff."""
        healer = DownloadHealer(mock_config, project_dir)
        state = Mock()

        result = healer._handle_rate_limit(Exception("429"), state)

        assert result.success
        assert result.action == HealerAction.RETRY
        mock_sleep.assert_called()

    @pytest.mark.fast
    def test_handle_unavailable_skips_video(self, mock_config, project_dir):
        """Test unavailable video is skipped."""
        healer = DownloadHealer(mock_config, project_dir)
        state = Mock()

        # Use Video ID format that the regex can match
        result = healer._handle_unavailable(
            Exception("Video ID: dQw4w9WgXcQ is unavailable"),
            state
        )

        assert result.success
        assert "dQw4w9WgXcQ" in healer.skipped_videos

    @pytest.mark.fast
    def test_handle_format_error_changes_format(self, mock_config, project_dir):
        """Test format error changes format string."""
        healer = DownloadHealer(mock_config, project_dir)
        state = Mock()

        mock_config.download.format = "bestvideo+bestaudio/best"

        result = healer._handle_format_error(Exception("format error"), state)

        assert result.success
        assert mock_config.download.format != "bestvideo+bestaudio/best"

    @patch('time.sleep')
    @pytest.mark.fast
    def test_handle_network_error_increases_timeout(self, mock_sleep, mock_config, project_dir):
        """Test network error increases socket timeout."""
        healer = DownloadHealer(mock_config, project_dir)
        state = Mock()

        mock_config.download.socket_timeout = 30

        result = healer._handle_network_error(Exception("connection timeout"), state)

        assert result.success
        assert mock_config.download.socket_timeout > 30

    @pytest.mark.fast
    def test_handle_incomplete_enables_resume(self, mock_config, project_dir):
        """Test incomplete download enables resume."""
        healer = DownloadHealer(mock_config, project_dir)
        state = Mock()

        mock_config.download.continue_dl = False

        result = healer._handle_incomplete(Exception("partial download"), state)

        assert result.success
        assert mock_config.download.continue_dl is True

    @pytest.mark.fast
    def test_extract_video_id(self, mock_config, project_dir):
        """Test video ID extraction from error messages."""
        healer = DownloadHealer(mock_config, project_dir)

        # From URL
        vid_id = healer._extract_video_id(
            "Error with https://youtube.com/watch?v=dQw4w9WgXcQ"
        )
        assert vid_id == "dQw4w9WgXcQ"

        # From Video ID label
        vid_id = healer._extract_video_id("Video ID: abc123xyz78")
        assert vid_id == "abc123xyz78"

    @pytest.mark.fast
    def test_get_skipped_videos(self, mock_config, project_dir):
        """Test get_skipped_videos returns copy."""
        healer = DownloadHealer(mock_config, project_dir)
        healer.skipped_videos.add("test123")

        result = healer.get_skipped_videos()

        assert "test123" in result
        assert result is not healer.skipped_videos

    @pytest.mark.fast
    def test_reset_backoff(self, mock_config, project_dir):
        """Test reset_backoff resets state."""
        healer = DownloadHealer(mock_config, project_dir)

        healer.backoff_time = 300.0
        healer.retry_count = 10

        healer.reset_backoff()

        assert healer.backoff_time == healer.INITIAL_BACKOFF
        assert healer.retry_count == 0


# =============================================================================
# Disk Healer Tests
# =============================================================================

class TestDiskHealer:
    """Tests for DiskHealer."""

    @pytest.mark.fast
    def test_error_patterns(self, mock_config, project_dir):
        """Test healer has correct error patterns."""
        healer = DiskHealer(mock_config, project_dir)

        assert "disk full" in healer.error_patterns
        assert "no space" in healer.error_patterns
        assert "permission denied" in healer.error_patterns

    @pytest.mark.fast
    def test_can_handle_disk_errors(self, mock_config, project_dir):
        """Test can_handle matches disk errors."""
        healer = DiskHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("No space left on device"), "OUTPUT")
        assert healer.can_handle(Exception("Permission denied"), "OUTPUT")
        assert healer.can_handle(OSError("disk full"), "OUTPUT")

    @patch('shutil.disk_usage')
    @pytest.mark.fast
    def test_get_free_space(self, mock_usage, mock_config, project_dir):
        """Test getting free disk space."""
        healer = DiskHealer(mock_config, project_dir)

        mock_usage.return_value = Mock(free=10 * 1024**3)  # 10 GB

        free_bytes = healer._get_free_space(project_dir)

        assert free_bytes == 10 * 1024**3

    @pytest.mark.fast
    def test_check_space(self, mock_config, project_dir):
        """Test check_space method."""
        healer = DiskHealer(mock_config, project_dir)

        # Should return a tuple (has_enough, free_gb)
        has_enough, free_gb = healer.check_space(required_gb=0.001)

        assert isinstance(has_enough, bool)
        assert isinstance(free_gb, float)

    @pytest.mark.fast
    def test_format_size(self, mock_config, project_dir):
        """Test size formatting."""
        healer = DiskHealer(mock_config, project_dir)

        assert "GB" in healer._format_size(5 * 1024**3)
        assert "MB" in healer._format_size(500 * 1024**2)
        assert "KB" in healer._format_size(500 * 1024)


# =============================================================================
# Path Healer Tests
# =============================================================================

class TestPathHealer:
    """Tests for PathHealer."""

    @pytest.mark.fast
    def test_error_patterns(self, mock_config, project_dir):
        """Test healer has correct error patterns."""
        healer = PathHealer(mock_config, project_dir)

        assert "path too long" in healer.error_patterns
        assert "unicode" in healer.error_patterns
        assert "filename too long" in healer.error_patterns

    @pytest.mark.fast
    def test_can_handle_path_errors(self, mock_config, project_dir):
        """Test can_handle matches path errors."""
        healer = PathHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("path too long"), "OUTPUT")
        assert healer.can_handle(Exception("unicode encode error"), "OUTPUT")
        assert healer.can_handle(Exception("filename too long"), "OUTPUT")

    @pytest.mark.fast
    def test_sanitize_unicode(self, mock_config, project_dir):
        """Test unicode path sanitization."""
        healer = PathHealer(mock_config, project_dir)

        # Test unicode replacement
        result = healer._sanitize_unicode("/path/to/file\u2019s_name.mp4")
        assert "\u2019" not in result

        # Test invalid char removal
        result = healer._sanitize_unicode("/path/to/file<name>.mp4")
        assert "<" not in result
        assert ">" not in result

    @pytest.mark.fast
    def test_check_path_length(self, mock_config, project_dir):
        """Test path length check."""
        healer = PathHealer(mock_config, project_dir)

        short_path = "/path/to/file.mp4"
        long_path = "a" * 300

        assert healer.check_path_length(short_path) is True
        assert healer.check_path_length(long_path) is False

    @pytest.mark.fast
    def test_estimate_safe_filename_length(self, mock_config, project_dir):
        """Test safe filename length estimation."""
        healer = PathHealer(mock_config, project_dir)

        safe_len = healer.estimate_safe_filename_length(project_dir)

        assert safe_len >= 20
        assert safe_len <= 200
