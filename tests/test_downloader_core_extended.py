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
"""

import sys
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def create_mock_config(tmp_path, **overrides):
    """Create a mock config for testing."""
    mock_config = MagicMock()
    mock_config.cache_dir = str(tmp_path / ".cache")
    mock_config.downloaded_videos_dir = str(tmp_path / "videos")
    mock_config.download = MagicMock()
    mock_config.download.davinci_mode = False
    mock_config.download.cookies = None
    mock_config.download.cookies_from_browser = None
    mock_config.download.download_timeout = 120
    mock_config.download.download_timeouts = {}
    mock_config.download.stall_timeout = 0
    mock_config.download.delete_original = False
    mock_config.download.per_keyword = {'short': 2, 'medium': 2, 'long': 2}
    mock_config.download.use_llm_filter = False
    mock_config.download.use_speech_screening = False
    # Retry settings (added for US-001 exponential backoff)
    mock_config.download.max_retries = 3
    mock_config.download.retry_delay = 2.0
    mock_config.download.retry_backoff = 2.0
    mock_config.download.rate_limit_budget = None
    mock_config.download.cookie_rotation = None
    mock_config.download.vpn = None
    mock_config.llm = MagicMock()
    mock_config.llm.provider = 'gemini'
    mock_config.llm.model = 'gemini-pro'

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(mock_config.download, key):
            setattr(mock_config.download, key, value)
        elif hasattr(mock_config, key):
            setattr(mock_config, key, value)

    return mock_config


class TestExistingVideosPartialMatch:
    """Test partial existing videos branch (lines 486-487)."""

    def test_existing_videos_partial_download(self, tmp_path):
        """Test when some but not all videos exist for a keyword."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)
                                    downloader.checkpoint = None
                                    downloader._lock = MagicMock()
                                    downloader.tier_download_counts = {}
                                    downloader.sources = []
                                    # Mock _get_tier_value to return proper int
                                    downloader._get_tier_value = MagicMock(return_value=3)  # Need 3 videos

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

    def test_checkpoint_skips_completed_video(self, tmp_path):
        """Test that checkpoint skips completed video keys."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)
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

    def test_retry_breaks_when_no_alt_keyword(self, tmp_path):
        """Test that retry loop breaks when no alternative keyword available."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)
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

    def test_remix_also_returns_no_results(self, tmp_path):
        """Test logging when remix keyword also returns no results."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)
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

    def test_checkpoint_saves_after_download(self, tmp_path):
        """Test that checkpoint is saved after successful download."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)
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

    def test_cleanup_handles_exceptions(self, tmp_path):
        """Test that .part and .ytdl cleanup handles exceptions gracefully."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)
                                    downloader._last_download_timed_out = False

                                    # Create keyword dir with .part and .ytdl files
                                    keyword_dir = tmp_path / "videos" / "short" / "test_keyword"
                                    keyword_dir.mkdir(parents=True)
                                    part_file = keyword_dir / "video.part"
                                    part_file.touch()
                                    ytdl_file = keyword_dir / "video.ytdl"
                                    ytdl_file.touch()

                                    # Make files read-only to cause exception
                                    import stat
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

    def test_timeout_override_used(self, tmp_path):
        """Test that timeout_override takes precedence."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)
        config.download.download_timeout = 100
        config.download.download_timeouts = {'short': 50}

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)

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

    def test_timeout_fallback_when_no_tier(self, tmp_path):
        """Test timeout falls back to config default when tier not in timeouts."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)
        config.download.download_timeout = 120
        config.download.download_timeouts = {}  # Empty - no tier-specific timeouts

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)

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
                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test",
                                                tier="short",
                                                existing_before=set()
                                            )

                                            # Should have used fallback 120s as stall timeout
                                            call_args = mock_wait.call_args
                                            stall_timeout = call_args[0][1]  # second positional arg
                                            assert stall_timeout == 120


class TestDownloadTimeout:
    """Test TimeoutExpired handling (lines 839-844)."""

    def test_timeout_expired_handling(self, tmp_path):
        """Test handling of download timeout after all retries exhausted."""
        from src.downloader.core import VideoDownloader

        # Use max_retries=0 for simple timeout test (no retries)
        config = create_mock_config(tmp_path, max_retries=0)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)

                                    keyword_dir = tmp_path / "videos" / "short" / "test"
                                    keyword_dir.mkdir(parents=True)

                                    with patch('subprocess.Popen') as mock_popen:
                                        mock_process = MagicMock()
                                        mock_process.kill = MagicMock()
                                        mock_process.poll.return_value = 0
                                        mock_popen.return_value = mock_process

                                        # Simulate timeout via progress-aware monitor
                                        with patch.object(downloader, '_wait_for_process_with_progress',
                                                          return_value=("", "", 'stall')):
                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test_keyword",
                                                tier="short",
                                                existing_before=set()
                                            )

                                            # Should return empty list on timeout
                                            assert result == []
                                            # Should have set timeout flag
                                            assert downloader._last_download_timed_out


class TestProcessCleanup:
    """Test process cleanup on unexpected exceptions."""

    def test_process_cleanup_on_exception(self, tmp_path):
        """Test that process is killed if an unexpected exception occurs."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, max_retries=0)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)

                                    keyword_dir = tmp_path / "videos" / "short" / "test"
                                    keyword_dir.mkdir(parents=True)

                                    with patch('subprocess.Popen') as mock_popen:
                                        mock_process = MagicMock()
                                        mock_process.returncode = 0
                                        # poll() returns None = still running
                                        mock_process.poll.return_value = None
                                        mock_process.wait.side_effect = subprocess.TimeoutExpired('yt-dlp', 5)
                                        mock_popen.return_value = mock_process

                                        # Simulate unexpected exception during progress monitoring
                                        with patch.object(downloader, '_wait_for_process_with_progress',
                                                          side_effect=RuntimeError("unexpected")):
                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test",
                                                tier="short",
                                                existing_before=set()
                                            )

                                        # Process should have been killed in exception handler
                                        mock_process.kill.assert_called()


class TestInfoFileException:
    """Test info.json read exception (lines 883-884)."""

    def test_info_json_read_error(self, tmp_path):
        """Test handling of corrupted info.json files."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('src.downloader.core.utils.sanitize_filename_for_nle', side_effect=lambda x: x):
                                        downloader = VideoDownloader(config)
                                        downloader._record_source_for_keyword = MagicMock()

                                        keyword_dir = tmp_path / "videos" / "short" / "test"
                                        keyword_dir.mkdir(parents=True)

                                        # Create video file and corrupted info.json
                                        video_file = keyword_dir / "video.mp4"
                                        video_file.write_bytes(b"fake video content")
                                        info_file = keyword_dir / "video.info.json"
                                        info_file.write_text("not valid json{{{")

                                        with patch('subprocess.Popen') as mock_popen:
                                            mock_process = MagicMock()
                                            mock_process.communicate.return_value = ("", "")
                                            mock_process.returncode = 0
                                            mock_process.poll.return_value = 0
                                            mock_popen.return_value = mock_process

                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test",
                                                tier="short",
                                                existing_before=set()
                                            )

                                        # Should still return result with default metadata
                                        assert len(result) == 1
                                        assert result[0].title == "video.mp4"  # Falls back to filename


class TestTranscodingBlock:
    """Test transcoding block (lines 895-940)."""

    def test_transcoding_success(self, tmp_path):
        """Test successful transcoding when davinci_mode enabled."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)
        config.download.davinci_mode = True
        config.download.delete_original = False

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager') as mock_tm_class:
                mock_tm = MagicMock()
                mock_tm.needs_transcoding.return_value = (True, "needs dnxhd")
                mock_tm_class.return_value = mock_tm

                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('src.downloader.core.utils.sanitize_filename_for_nle', side_effect=lambda x: x):
                                        downloader = VideoDownloader(config)
                                        downloader._record_source_for_keyword = MagicMock()
                                        downloader._needs_transcoding = MagicMock(return_value=(True, "needs dnxhd"))

                                        keyword_dir = tmp_path / "videos" / "short" / "test"
                                        keyword_dir.mkdir(parents=True)

                                        # Create video file and temp output file
                                        video_file = keyword_dir / "video.mp4"
                                        video_file.write_bytes(b"fake video content")
                                        temp_output = keyword_dir / "video_davinci.mp4"
                                        temp_output.write_bytes(b"transcoded video content")

                                        # Mock get_ffmpeg_transcode_cmd to return path to temp file
                                        downloader._get_ffmpeg_transcode_cmd = MagicMock(return_value=(
                                            ['ffmpeg', '-i', str(video_file), str(temp_output)],
                                            str(temp_output)
                                        ))

                                        with patch('subprocess.Popen') as mock_popen:
                                            # First call for download, second for transcode
                                            download_process = MagicMock()
                                            download_process.communicate.return_value = ("", "")
                                            download_process.returncode = 0
                                            download_process.poll.return_value = 0

                                            transcode_process = MagicMock()
                                            transcode_process.communicate.return_value = ("", "")
                                            transcode_process.returncode = 0
                                            transcode_process.poll.return_value = 0

                                            mock_popen.side_effect = [download_process, transcode_process]

                                            # Include temp output in existing_before so only original triggers transcoding flow
                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test",
                                                tier="short",
                                                existing_before={'video_davinci.mp4'}
                                            )

                                            # Should return video(s) - transcoding was invoked
                                            assert len(result) >= 1
                                            # Transcoding method was called
                                            downloader._needs_transcoding.assert_called()

    def test_transcoding_timeout(self, tmp_path):
        """Test transcoding timeout handling."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)
        config.download.davinci_mode = True

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager') as mock_tm_class:
                mock_tm = MagicMock()
                mock_tm.needs_transcoding.return_value = (True, "needs dnxhd")
                mock_tm.get_ffmpeg_transcode_cmd.return_value = (
                    ['ffmpeg', '-i', 'input.mp4', 'output.mp4'],
                    str(tmp_path / "videos" / "output.mp4")
                )
                mock_tm_class.return_value = mock_tm

                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('src.downloader.core.utils.sanitize_filename_for_nle', side_effect=lambda x: x):
                                        downloader = VideoDownloader(config)
                                        downloader._record_source_for_keyword = MagicMock()
                                        downloader._needs_transcoding = mock_tm.needs_transcoding
                                        downloader._get_ffmpeg_transcode_cmd = mock_tm.get_ffmpeg_transcode_cmd

                                        keyword_dir = tmp_path / "videos" / "short" / "test"
                                        keyword_dir.mkdir(parents=True)
                                        video_file = keyword_dir / "video.mp4"
                                        video_file.write_bytes(b"fake video")

                                        with patch('subprocess.Popen') as mock_popen:
                                            download_process = MagicMock()
                                            download_process.communicate.return_value = ("", "")
                                            download_process.returncode = 0
                                            download_process.poll.return_value = 0

                                            transcode_process = MagicMock()
                                            transcode_process.communicate.side_effect = subprocess.TimeoutExpired('ffmpeg', 120)
                                            transcode_process.poll.return_value = None

                                            mock_popen.side_effect = [download_process, transcode_process]

                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test",
                                                tier="short",
                                                existing_before=set()
                                            )

                                            # Should still return original video
                                            assert len(result) == 1


class TestGeneralException:
    """Test general exception handling (lines 972-974)."""

    def test_general_exception_in_run_download_cmd(self, tmp_path):
        """Test handling of unexpected exceptions after all retries exhausted."""
        from src.downloader.core import VideoDownloader

        # Use max_retries=0 for simple exception test (no retries)
        config = create_mock_config(tmp_path, max_retries=0)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)

                                    keyword_dir = tmp_path / "videos" / "short" / "test"
                                    keyword_dir.mkdir(parents=True)

                                    with patch('subprocess.Popen') as mock_popen:
                                        # Raise unexpected exception
                                        mock_popen.side_effect = RuntimeError("Unexpected error")

                                        result = downloader._run_download_cmd(
                                            cmd=['yt-dlp', 'test'],
                                            keyword_dir=keyword_dir,
                                            output_dir=tmp_path / "videos",
                                            keyword="test_keyword",
                                            tier="short",
                                            existing_before=set()
                                        )

                                        # Should return empty list
                                        assert result == []


class TestTranscodingEdgeCases:
    """Additional transcoding edge cases."""

    def test_transcode_no_output_uses_original(self, tmp_path):
        """Test fallback to original when transcode produces no output."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)
        config.download.davinci_mode = True

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager') as mock_tm_class:
                mock_tm = MagicMock()
                mock_tm.needs_transcoding.return_value = (True, "needs dnxhd")
                mock_tm.get_ffmpeg_transcode_cmd.return_value = (
                    ['ffmpeg', '-i', 'input.mp4', 'output.mp4'],
                    str(tmp_path / "videos" / "output.mp4")
                )
                mock_tm_class.return_value = mock_tm

                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('src.downloader.core.utils.sanitize_filename_for_nle', side_effect=lambda x: x):
                                        downloader = VideoDownloader(config)
                                        downloader._record_source_for_keyword = MagicMock()
                                        downloader._needs_transcoding = mock_tm.needs_transcoding
                                        downloader._get_ffmpeg_transcode_cmd = mock_tm.get_ffmpeg_transcode_cmd

                                        keyword_dir = tmp_path / "videos" / "short" / "test"
                                        keyword_dir.mkdir(parents=True)
                                        video_file = keyword_dir / "video.mp4"
                                        video_file.write_bytes(b"fake video")

                                        with patch('subprocess.Popen') as mock_popen:
                                            download_process = MagicMock()
                                            download_process.communicate.return_value = ("", "")
                                            download_process.returncode = 0
                                            download_process.poll.return_value = 0

                                            transcode_process = MagicMock()
                                            transcode_process.communicate.return_value = ("", "")
                                            transcode_process.returncode = 0
                                            transcode_process.poll.return_value = 0

                                            mock_popen.side_effect = [download_process, transcode_process]

                                            # Mock temp_output to not exist (transcode failed)
                                            original_exists = Path.exists
                                            def mock_exists(self):
                                                if '_davinci' in str(self):
                                                    return False
                                                return original_exists(self)

                                            with patch.object(Path, 'exists', mock_exists):
                                                result = downloader._run_download_cmd(
                                                    cmd=['yt-dlp', 'test'],
                                                    keyword_dir=keyword_dir,
                                                    output_dir=tmp_path / "videos",
                                                    keyword="test",
                                                    tier="short",
                                                    existing_before=set()
                                                )

                                                # Should use original
                                                assert len(result) == 1

    def test_transcode_ffmpeg_error(self, tmp_path):
        """Test handling of FFmpeg errors during transcode."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path)
        config.download.davinci_mode = True

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager') as mock_tm_class:
                mock_tm = MagicMock()
                mock_tm.needs_transcoding.return_value = (True, "needs dnxhd")
                mock_tm.get_ffmpeg_transcode_cmd.return_value = (
                    ['ffmpeg', '-i', 'input.mp4', 'output.mp4'],
                    str(tmp_path / "videos" / "output.mp4")
                )
                mock_tm_class.return_value = mock_tm

                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('src.downloader.core.utils.sanitize_filename_for_nle', side_effect=lambda x: x):
                                        downloader = VideoDownloader(config)
                                        downloader._record_source_for_keyword = MagicMock()
                                        downloader._needs_transcoding = mock_tm.needs_transcoding
                                        downloader._get_ffmpeg_transcode_cmd = mock_tm.get_ffmpeg_transcode_cmd

                                        keyword_dir = tmp_path / "videos" / "short" / "test"
                                        keyword_dir.mkdir(parents=True)
                                        video_file = keyword_dir / "video.mp4"
                                        video_file.write_bytes(b"fake video")

                                        with patch('subprocess.Popen') as mock_popen:
                                            download_process = MagicMock()
                                            download_process.communicate.return_value = ("", "")
                                            download_process.returncode = 0
                                            download_process.poll.return_value = 0

                                            transcode_process = MagicMock()
                                            transcode_process.communicate.return_value = ("", "FFmpeg error: codec not found")
                                            transcode_process.returncode = 1  # Non-zero = error
                                            transcode_process.poll.return_value = 1

                                            mock_popen.side_effect = [download_process, transcode_process]

                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test",
                                                tier="short",
                                                existing_before=set()
                                            )
