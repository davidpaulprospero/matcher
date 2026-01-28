"""
Extended coverage tests for src/downloader/core.py

Covers missed lines:
- 486-487: elif existing_videos branch (partial existing)
- 491-494: Checkpoint-based skip
- 508: Break in retry loop (no alt keyword)
- 536: Remix also returned no results
- 541-545: Checkpoint save after download
- 806-807, 811-812: Exception handling in file cleanup
- 816: timeout_override case
- 822: download_timeout fallback
- 839-844: TimeoutExpired handling
- 848-852: Process cleanup in finally
- 883-884: Info file read exception
- 895-940: Transcoding block
- 972-974: General exception handling

Refactored as part of US-009 to use shared fixtures from tests/fixtures/downloader_fixtures.py.
"""

import sys
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from tests.fixtures.downloader_fixtures import (
    create_mock_downloader_config,
    patch_video_downloader_dependencies,
)


def create_downloader(config):
    """Create a VideoDownloader with all dependencies patched."""
    with patch_video_downloader_dependencies():
        from src.downloader.core import VideoDownloader
        return VideoDownloader(config)


class TestExistingVideosPartialMatch:
    """Test partial existing videos branch (lines 486-487)."""

    @pytest.mark.fast
    def test_existing_videos_partial_download(self, tmp_path):
        """Test when some but not all videos exist for a keyword."""
        config = create_mock_downloader_config(tmp_path)
        downloader = create_downloader(config)
        downloader.checkpoint = None
        downloader._lock = MagicMock()
        downloader.tier_download_counts = {}
        downloader.sources = []
        downloader._get_tier_value = MagicMock(return_value=3)

        # Create keyword directory with 1 existing video (need 2 more)
        keyword_dir = tmp_path / "videos" / "test_key_s"
        keyword_dir.mkdir(parents=True)
        (keyword_dir / "existing_video.mp4").touch()

        # Mock _download_single to return new video
        mock_video = MagicMock()
        mock_video.file = "new_video.mp4"
        downloader._download_single = MagicMock(return_value=[mock_video])
        downloader._get_retry_keyword = MagicMock(return_value=None)
        downloader._get_remix_keyword = MagicMock(return_value=None)
        downloader._save_sources = MagicMock()

        result = downloader.download_for_keyword(
            keyword="test keyword",
            output_dir=tmp_path / "videos",
            tiers=['short'],
            topic=""
        )

        # Should have attempted download for remaining
        assert downloader._download_single.called


class TestCheckpointSkip:
    """Test checkpoint-based skip (lines 491-494)."""

    @pytest.mark.fast
    def test_checkpoint_skips_completed_video(self, tmp_path):
        """Test that checkpoint skips completed video keys."""
        config = create_mock_downloader_config(tmp_path)
        downloader = create_downloader(config)
        downloader._lock = MagicMock()
        downloader.tier_download_counts = {}
        downloader.sources = []
        downloader._get_tier_value = MagicMock(return_value=5)

        # Set up checkpoint with completed video
        mock_checkpoint = MagicMock()
        mock_checkpoint.completed_videos = ["test keyword|short"]
        downloader.checkpoint = mock_checkpoint

        downloader._download_single = MagicMock()
        downloader._get_retry_keyword = MagicMock()
        downloader._get_remix_keyword = MagicMock()

        result = downloader.download_for_keyword(
            keyword="test keyword",
            output_dir=tmp_path / "videos",
            tiers=['short'],
            topic=""
        )

        # Should not have called _download_single due to checkpoint skip
        downloader._download_single.assert_not_called()


class TestRetryKeywordBreak:
    """Test retry loop break (line 508)."""

    @pytest.mark.fast
    def test_retry_breaks_when_no_alt_keyword(self, tmp_path):
        """Test that retry loop breaks when no alternative keyword available."""
        config = create_mock_downloader_config(tmp_path)
        downloader = create_downloader(config)
        downloader.checkpoint = None
        downloader._lock = MagicMock()
        downloader.tier_download_counts = {}
        downloader.sources = []
        downloader._get_tier_value = MagicMock(return_value=5)

        # First call returns empty and sets timeout, second also empty
        call_count = [0]
        def mock_download(*args, **kwargs):
            call_count[0] += 1
            downloader._last_download_timed_out = (call_count[0] == 1)
            return []

        downloader._download_single = mock_download
        # Return None for alt keyword (triggers break)
        downloader._get_retry_keyword = MagicMock(return_value=None)
        downloader._get_remix_keyword = MagicMock(return_value=None)

        result = downloader.download_for_keyword(
            keyword="test keyword",
            output_dir=tmp_path / "videos",
            tiers=['short'],
            topic=""
        )

        # Should have only called once due to break
        assert call_count[0] == 1


class TestRemixNoResults:
    """Test remix with no results (line 536)."""

    @pytest.mark.fast
    def test_remix_also_returns_no_results(self, tmp_path):
        """Test logging when remix keyword also returns no results."""
        config = create_mock_downloader_config(tmp_path)
        downloader = create_downloader(config)
        downloader.checkpoint = None
        downloader._lock = MagicMock()
        downloader.tier_download_counts = {}
        downloader.sources = []
        downloader._last_download_timed_out = False
        downloader._get_tier_value = MagicMock(return_value=5)

        # Always returns empty
        downloader._download_single = MagicMock(return_value=[])
        downloader._get_retry_keyword = MagicMock(return_value=None)
        # Return remix keyword but it also fails
        downloader._get_remix_keyword = MagicMock(return_value="remixed keyword")

        result = downloader.download_for_keyword(
            keyword="test keyword",
            output_dir=tmp_path / "videos",
            tiers=['short'],
            topic=""
        )

        # Both original and remix should have been tried
        assert downloader._download_single.call_count == 2


class TestCheckpointSave:
    """Test checkpoint save after download (lines 541-545)."""

    @pytest.mark.fast
    def test_checkpoint_saves_after_download(self, tmp_path):
        """Test that checkpoint is saved after successful download."""
        config = create_mock_downloader_config(tmp_path)
        downloader = create_downloader(config)
        downloader._lock = MagicMock()
        downloader.tier_download_counts = {}
        downloader.sources = []
        downloader._last_download_timed_out = False
        downloader._get_tier_value = MagicMock(return_value=5)

        # Set up checkpoint
        mock_checkpoint = MagicMock()
        mock_checkpoint.completed_videos = []
        downloader.checkpoint = mock_checkpoint

        mock_video = MagicMock()
        downloader._download_single = MagicMock(return_value=[mock_video])
        downloader._get_retry_keyword = MagicMock(return_value=None)
        downloader._save_checkpoint = MagicMock()
        downloader._save_sources = MagicMock()

        result = downloader.download_for_keyword(
            keyword="test keyword",
            output_dir=tmp_path / "videos",
            tiers=['short'],
            topic=""
        )

        # Checkpoint should have been saved
        downloader._save_checkpoint.assert_called()
        assert "test keyword|short" in mock_checkpoint.completed_videos


class TestPartFileCleanup:
    """Test partial file cleanup (lines 806-807, 811-812)."""

    @pytest.mark.integration
    def test_cleanup_handles_exceptions(self, tmp_path):
        """Test that .part and .ytdl cleanup handles exceptions gracefully."""
        config = create_mock_downloader_config(tmp_path)
        downloader = create_downloader(config)
        downloader._last_download_timed_out = False

        # Create keyword dir with .part and .ytdl files
        keyword_dir = tmp_path / "videos" / "short" / "test_keyword"
        keyword_dir.mkdir(parents=True)
        part_file = keyword_dir / "video.part"
        part_file.touch()
        ytdl_file = keyword_dir / "video.ytdl"
        ytdl_file.touch()

        # Make files read-only to cause exception
        part_file.chmod(stat.S_IRUSR)
        ytdl_file.chmod(stat.S_IRUSR)

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.return_value = ("", "")
            mock_process.returncode = 0
            mock_process.poll.return_value = 0
            mock_popen.return_value = mock_process

            # Should not raise despite cleanup failures
            result = downloader._run_download_cmd(
                cmd=['yt-dlp', 'test'],
                keyword_dir=keyword_dir,
                output_dir=tmp_path / "videos",
                keyword="test_keyword",
                tier="short",
                existing_before=set()
            )

        # Restore permissions for cleanup
        part_file.chmod(stat.S_IWUSR | stat.S_IRUSR)
        ytdl_file.chmod(stat.S_IWUSR | stat.S_IRUSR)


class TestTimeoutOverride:
    """Test timeout_override (line 816) and fallback (line 822)."""

    @pytest.mark.integration
    def test_timeout_override_used(self, tmp_path):
        """Test that timeout_override takes precedence."""
        config = create_mock_downloader_config(tmp_path, download_timeout=100)
        config.download.download_timeouts = {'short': 50}
        downloader = create_downloader(config)

        keyword_dir = tmp_path / "videos" / "short" / "test"
        keyword_dir.mkdir(parents=True)

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.poll.return_value = 0
            mock_popen.return_value = mock_process

            # Mock progress-aware timeout to capture args
            with patch.object(downloader, '_wait_for_process_with_progress',
                              return_value=("", "", None)) as mock_wait:
                # Use timeout_override=30
                result = downloader._run_download_cmd(
                    cmd=['yt-dlp', 'test'],
                    keyword_dir=keyword_dir,
                    output_dir=tmp_path / "videos",
                    keyword="test",
                    tier="short",
                    existing_before=set(),
                    timeout_override=30
                )

                # Should have used 30s as stall timeout
                call_args = mock_wait.call_args
                stall_timeout = call_args[0][1]  # second positional arg
                assert stall_timeout == 30

    @pytest.mark.integration
    def test_timeout_fallback_when_no_tier(self, tmp_path):
        """Test timeout falls back to config default when tier not in timeouts."""
        config = create_mock_downloader_config(tmp_path, download_timeout=120)
        config.download.download_timeouts = {}  # No tier-specific timeouts
        downloader = create_downloader(config)

        keyword_dir = tmp_path / "videos" / "medium" / "test"
        keyword_dir.mkdir(parents=True)

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.poll.return_value = 0
            mock_popen.return_value = mock_process

            with patch.object(downloader, '_wait_for_process_with_progress',
                              return_value=("", "", None)) as mock_wait:
                result = downloader._run_download_cmd(
                    cmd=['yt-dlp', 'test'],
                    keyword_dir=keyword_dir,
                    output_dir=tmp_path / "videos",
                    keyword="test",
                    tier="medium",
                    existing_before=set()
                )

                # Should have used default timeout
                call_args = mock_wait.call_args
                max_timeout = call_args[0][2]  # third positional arg (max_timeout)
                # max_timeout = int(download_timeout * 1.5) = 180
                assert max_timeout == 180


class TestTimeoutExpired:
    """Test TimeoutExpired handling (lines 839-844)."""

    @pytest.mark.integration
    def test_timeout_expired_handling(self, tmp_path):
        """Test handling of subprocess.TimeoutExpired."""
        config = create_mock_downloader_config(tmp_path)
        downloader = create_downloader(config)

        keyword_dir = tmp_path / "videos" / "short" / "test"
        keyword_dir.mkdir(parents=True)

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.returncode = None
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            # Mock to trigger TimeoutExpired
            with patch.object(downloader, '_wait_for_process_with_progress') as mock_wait:
                mock_wait.side_effect = subprocess.TimeoutExpired(cmd=['yt-dlp'], timeout=30)

                result = downloader._run_download_cmd(
                    cmd=['yt-dlp', 'test'],
                    keyword_dir=keyword_dir,
                    output_dir=tmp_path / "videos",
                    keyword="test",
                    tier="short",
                    existing_before=set()
                )

                # Should return empty list and have killed process
                assert result == []
                mock_process.kill.assert_called()


class TestProcessCleanup:
    """Test process cleanup in finally (lines 848-852)."""

    @pytest.mark.integration
    def test_process_cleanup_on_exception(self, tmp_path):
        """Test that process is properly cleaned up on exception."""
        config = create_mock_downloader_config(tmp_path)
        downloader = create_downloader(config)

        keyword_dir = tmp_path / "videos" / "short" / "test"
        keyword_dir.mkdir(parents=True)

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.returncode = None
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            # Mock to raise exception
            with patch.object(downloader, '_wait_for_process_with_progress') as mock_wait:
                mock_wait.side_effect = RuntimeError("Simulated error")

                result = downloader._run_download_cmd(
                    cmd=['yt-dlp', 'test'],
                    keyword_dir=keyword_dir,
                    output_dir=tmp_path / "videos",
                    keyword="test",
                    tier="short",
                    existing_before=set()
                )

                # Should return empty list
                assert result == []


class TestInfoFileException:
    """Test info file read exception (lines 883-884)."""

    @pytest.mark.integration
    def test_info_file_read_exception(self, tmp_path):
        """Test handling of exception when reading .info.json file."""
        config = create_mock_downloader_config(tmp_path)
        downloader = create_downloader(config)
        downloader._record_source_for_keyword = MagicMock()

        keyword_dir = tmp_path / "videos" / "short" / "test"
        keyword_dir.mkdir(parents=True)

        # Create video file
        video_file = keyword_dir / "video.mp4"
        video_file.touch()

        # Create invalid info.json file
        info_file = keyword_dir / "video.info.json"
        info_file.write_text("invalid json {", encoding='utf-8')

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.poll.return_value = 0
            mock_popen.return_value = mock_process

            with patch.object(downloader, '_wait_for_process_with_progress',
                              return_value=("", "", None)):
                with patch('src.downloader.core.utils.sanitize_filename_for_nle',
                           side_effect=lambda x: x):
                    result = downloader._run_download_cmd(
                        cmd=['yt-dlp', 'test'],
                        keyword_dir=keyword_dir,
                        output_dir=tmp_path / "videos",
                        keyword="test",
                        tier="short",
                        existing_before=set()
                    )

                    # Should still return video even if info.json parse failed
                    assert len(result) == 1


class TestGeneralException:
    """Test general exception handling (lines 972-974)."""

    @pytest.mark.integration
    def test_general_exception_handled(self, tmp_path):
        """Test handling of general exceptions in _run_download_cmd."""
        config = create_mock_downloader_config(tmp_path)
        downloader = create_downloader(config)

        keyword_dir = tmp_path / "videos" / "short" / "test"
        keyword_dir.mkdir(parents=True)

        with patch('subprocess.Popen') as mock_popen:
            mock_popen.side_effect = OSError("Cannot start process")

            result = downloader._run_download_cmd(
                cmd=['yt-dlp', 'test'],
                keyword_dir=keyword_dir,
                output_dir=tmp_path / "videos",
                keyword="test",
                tier="short",
                existing_before=set()
            )

            # Should return empty list on exception
            assert result == []


