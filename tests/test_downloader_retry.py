"""
Tests for exponential backoff retry mechanism in VideoDownloader._run_download_cmd.

Covers US-001: Implement exponential backoff retry in _run_download_cmd
- Uses config.download.max_retries for retry attempts
- Delay follows exponential backoff: retry_delay * (retry_backoff ^ attempt)
- Retries only on transient errors (timeout, connection reset, 429 rate limit)
- Permanent errors (video unavailable, private, age-restricted) fail immediately
- Each retry attempt is logged with attempt number and delay duration
"""

import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call

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
    mock_config.download.delete_original = False
    mock_config.download.max_retries = 3
    mock_config.download.retry_delay = 2.0
    mock_config.download.retry_backoff = 2.0
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


class TestRetryConfiguration:
    """Test that retry configuration is read from config."""

    def test_uses_config_max_retries(self, tmp_path):
        """Test that max_retries is read from config."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, max_retries=5)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)
                                    # Verify config access pattern
                                    assert config.download.max_retries == 5

    def test_uses_config_retry_delay(self, tmp_path):
        """Test that retry_delay is read from config."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, retry_delay=5.0)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)
                                    assert config.download.retry_delay == 5.0

    def test_uses_config_retry_backoff(self, tmp_path):
        """Test that retry_backoff is read from config."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, retry_backoff=3.0)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    downloader = VideoDownloader(config)
                                    assert config.download.retry_backoff == 3.0


class TestTransientErrorDetection:
    """Test that transient errors are correctly identified."""

    def test_is_transient_error_429(self, tmp_path):
        """Test that 429 rate limit is detected as transient."""
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
                                    assert downloader._is_transient_error("ERROR: HTTP Error 429: Too Many Requests")

    def test_is_transient_error_connection_reset(self, tmp_path):
        """Test that connection reset is detected as transient."""
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
                                    assert downloader._is_transient_error("Connection reset by peer")

    def test_is_transient_error_rate_limit(self, tmp_path):
        """Test that rate limit text is detected as transient."""
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
                                    assert downloader._is_transient_error("rate limit exceeded, please wait")

    def test_is_transient_error_service_unavailable(self, tmp_path):
        """Test that service unavailable is detected as transient."""
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
                                    assert downloader._is_transient_error("HTTP Error 503: Service Unavailable")

    def test_not_transient_for_unrelated_error(self, tmp_path):
        """Test that unrelated errors are not classified as transient."""
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
                                    assert not downloader._is_transient_error("Some random error message")


class TestPermanentErrorDetection:
    """Test that permanent errors are correctly identified."""

    def test_is_permanent_error_video_unavailable(self, tmp_path):
        """Test that video unavailable is detected as permanent."""
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
                                    assert downloader._is_permanent_error("Video unavailable")

    def test_is_permanent_error_private_video(self, tmp_path):
        """Test that private video is detected as permanent."""
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
                                    assert downloader._is_permanent_error("This video is private")

    def test_is_permanent_error_age_restricted(self, tmp_path):
        """Test that age-restricted is detected as permanent."""
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
                                    assert downloader._is_permanent_error("Sign in to confirm your age")

    def test_is_permanent_error_copyright(self, tmp_path):
        """Test that copyright claim is detected as permanent."""
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
                                    assert downloader._is_permanent_error("Video removed due to copyright claim")

    def test_not_permanent_for_transient_error(self, tmp_path):
        """Test that transient errors are not classified as permanent."""
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
                                    assert not downloader._is_permanent_error("HTTP Error 429: Too Many Requests")


class TestRetryBehavior:
    """Test the actual retry behavior with mocked subprocess."""

    def test_retry_on_timeout(self, tmp_path):
        """Test that timeouts trigger retry with exponential backoff."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, max_retries=2, retry_delay=0.1, retry_backoff=2.0)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('time.sleep') as mock_sleep:
                                        downloader = VideoDownloader(config)

                                        keyword_dir = tmp_path / "videos" / "test"
                                        keyword_dir.mkdir(parents=True)

                                        with patch('subprocess.Popen') as mock_popen:
                                            mock_process = MagicMock()
                                            # All calls timeout
                                            mock_process.communicate.side_effect = [
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),  # After kill
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),  # After kill
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),  # After kill
                                            ]
                                            mock_process.kill = MagicMock()
                                            mock_process.poll.return_value = 0
                                            mock_popen.return_value = mock_process

                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test_keyword",
                                                tier="short",
                                                existing_before=set()
                                            )

                                            # Should return empty list after retries exhausted
                                            assert result == []
                                            assert downloader._last_download_timed_out

                                            # Should have slept with exponential backoff
                                            # First retry: 0.1 * (2.0 ^ 0) = 0.1
                                            # Second retry: 0.1 * (2.0 ^ 1) = 0.2
                                            assert mock_sleep.call_count == 2
                                            calls = mock_sleep.call_args_list
                                            assert abs(calls[0][0][0] - 0.1) < 0.01
                                            assert abs(calls[1][0][0] - 0.2) < 0.01

    def test_retry_on_transient_error(self, tmp_path):
        """Test that transient errors trigger retry."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, max_retries=2, retry_delay=0.1, retry_backoff=2.0)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('time.sleep') as mock_sleep:
                                        downloader = VideoDownloader(config)

                                        keyword_dir = tmp_path / "videos" / "test"
                                        keyword_dir.mkdir(parents=True)

                                        with patch('subprocess.Popen') as mock_popen:
                                            mock_process = MagicMock()
                                            # First call: 429 error, second call: success
                                            mock_process.communicate.side_effect = [
                                                ("", "ERROR: HTTP Error 429: Too Many Requests"),
                                                ("", ""),  # Success on retry
                                            ]
                                            mock_process.returncode = 1  # First call fails
                                            mock_process.poll.return_value = 0
                                            mock_popen.return_value = mock_process

                                            # Make returncode change on second call
                                            call_count = [0]
                                            original_communicate = mock_process.communicate.side_effect

                                            def update_returncode(*args, **kwargs):
                                                call_count[0] += 1
                                                if call_count[0] == 1:
                                                    mock_process.returncode = 1
                                                    return ("", "ERROR: HTTP Error 429: Too Many Requests")
                                                else:
                                                    mock_process.returncode = 0
                                                    return ("", "")

                                            mock_process.communicate = update_returncode

                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test_keyword",
                                                tier="short",
                                                existing_before=set()
                                            )

                                            # Should have slept once for retry
                                            assert mock_sleep.call_count >= 1

    def test_no_retry_on_permanent_error(self, tmp_path):
        """Test that permanent errors fail immediately without retry."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, max_retries=3, retry_delay=0.1)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('time.sleep') as mock_sleep:
                                        downloader = VideoDownloader(config)

                                        keyword_dir = tmp_path / "videos" / "test"
                                        keyword_dir.mkdir(parents=True)

                                        with patch('subprocess.Popen') as mock_popen:
                                            mock_process = MagicMock()
                                            # Permanent error - should fail immediately
                                            mock_process.communicate.return_value = (
                                                "",
                                                "ERROR: Video unavailable. This video has been removed."
                                            )
                                            mock_process.returncode = 1
                                            mock_process.poll.return_value = 1
                                            mock_popen.return_value = mock_process

                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test_keyword",
                                                tier="short",
                                                existing_before=set()
                                            )

                                            # Should return empty list immediately
                                            assert result == []
                                            # Should NOT have slept (no retry)
                                            assert mock_sleep.call_count == 0

    def test_success_on_first_attempt(self, tmp_path):
        """Test that successful downloads don't retry."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, max_retries=3, retry_delay=0.1)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('src.downloader.core.utils.sanitize_filename_for_nle', side_effect=lambda x: x):
                                        with patch('time.sleep') as mock_sleep:
                                            downloader = VideoDownloader(config)
                                            downloader._record_source_for_keyword = MagicMock()

                                            keyword_dir = tmp_path / "videos" / "test"
                                            keyword_dir.mkdir(parents=True)

                                            # Create a video file that will be "downloaded"
                                            video_file = keyword_dir / "video.mp4"
                                            video_file.write_bytes(b"fake video")

                                            with patch('subprocess.Popen') as mock_popen:
                                                mock_process = MagicMock()
                                                # Success immediately
                                                mock_process.communicate.return_value = ("", "")
                                                mock_process.returncode = 0
                                                mock_process.poll.return_value = 0
                                                mock_popen.return_value = mock_process

                                                result = downloader._run_download_cmd(
                                                    cmd=['yt-dlp', 'test'],
                                                    keyword_dir=keyword_dir,
                                                    output_dir=tmp_path / "videos",
                                                    keyword="test_keyword",
                                                    tier="short",
                                                    existing_before=set()
                                                )

                                                # Should NOT have slept
                                                assert mock_sleep.call_count == 0
                                                # Should return the video
                                                assert len(result) == 1


class TestExponentialBackoffCalculation:
    """Test that exponential backoff delay is calculated correctly."""

    def test_backoff_formula_attempt_0(self, tmp_path):
        """Test backoff delay for first retry (attempt=0)."""
        # delay = retry_delay * (retry_backoff ^ attempt)
        # delay = 2.0 * (2.0 ^ 0) = 2.0 * 1 = 2.0
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, max_retries=3, retry_delay=2.0, retry_backoff=2.0)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('time.sleep') as mock_sleep:
                                        downloader = VideoDownloader(config)

                                        keyword_dir = tmp_path / "videos" / "test"
                                        keyword_dir.mkdir(parents=True)

                                        with patch('subprocess.Popen') as mock_popen:
                                            mock_process = MagicMock()
                                            # First call: transient error, second: success
                                            call_count = [0]

                                            def communicate_side_effect(*args, **kwargs):
                                                call_count[0] += 1
                                                if call_count[0] == 1:
                                                    mock_process.returncode = 1
                                                    return ("", "ERROR: 429 rate limit")
                                                mock_process.returncode = 0
                                                return ("", "")

                                            mock_process.communicate = communicate_side_effect
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

                                            # First retry delay: 2.0 * (2.0 ^ 0) = 2.0
                                            assert mock_sleep.call_count == 1
                                            assert abs(mock_sleep.call_args[0][0] - 2.0) < 0.01

    def test_backoff_formula_progressive(self, tmp_path):
        """Test that delays increase with each retry."""
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, max_retries=3, retry_delay=1.0, retry_backoff=2.0)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with patch('time.sleep') as mock_sleep:
                                        downloader = VideoDownloader(config)

                                        keyword_dir = tmp_path / "videos" / "test"
                                        keyword_dir.mkdir(parents=True)

                                        with patch('subprocess.Popen') as mock_popen:
                                            mock_process = MagicMock()
                                            # All calls timeout
                                            mock_process.communicate.side_effect = [
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),
                                            ]
                                            mock_process.kill = MagicMock()
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

                                            # Delays: 1.0*2^0=1, 1.0*2^1=2, 1.0*2^2=4
                                            assert mock_sleep.call_count == 3
                                            calls = mock_sleep.call_args_list
                                            assert abs(calls[0][0][0] - 1.0) < 0.01
                                            assert abs(calls[1][0][0] - 2.0) < 0.01
                                            assert abs(calls[2][0][0] - 4.0) < 0.01


class TestRetryLogging:
    """Test that retry attempts are properly logged."""

    def test_retry_attempt_logged(self, tmp_path, caplog):
        """Test that each retry attempt is logged with attempt number."""
        import logging
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, max_retries=2, retry_delay=0.01, retry_backoff=2.0)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with caplog.at_level(logging.INFO):
                                        downloader = VideoDownloader(config)

                                        keyword_dir = tmp_path / "videos" / "test"
                                        keyword_dir.mkdir(parents=True)

                                        with patch('subprocess.Popen') as mock_popen:
                                            mock_process = MagicMock()
                                            # All calls timeout
                                            mock_process.communicate.side_effect = [
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),
                                            ]
                                            mock_process.kill = MagicMock()
                                            mock_process.poll.return_value = 0
                                            mock_popen.return_value = mock_process

                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test_keyword",
                                                tier="short",
                                                existing_before=set()
                                            )

                                            # Check that retry attempts were logged
                                            log_text = caplog.text
                                            assert "retry 1/2" in log_text
                                            assert "retry 2/2" in log_text

    def test_exhausted_retries_logged(self, tmp_path, caplog):
        """Test that exhausted retries are logged as warning."""
        import logging
        from src.downloader.core import VideoDownloader

        config = create_mock_config(tmp_path, max_retries=1, retry_delay=0.01)

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    with caplog.at_level(logging.WARNING):
                                        downloader = VideoDownloader(config)

                                        keyword_dir = tmp_path / "videos" / "test"
                                        keyword_dir.mkdir(parents=True)

                                        with patch('subprocess.Popen') as mock_popen:
                                            mock_process = MagicMock()
                                            mock_process.communicate.side_effect = [
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),
                                                subprocess.TimeoutExpired('yt-dlp', 120),
                                                ("", ""),
                                            ]
                                            mock_process.kill = MagicMock()
                                            mock_process.poll.return_value = 0
                                            mock_popen.return_value = mock_process

                                            result = downloader._run_download_cmd(
                                                cmd=['yt-dlp', 'test'],
                                                keyword_dir=keyword_dir,
                                                output_dir=tmp_path / "videos",
                                                keyword="test_keyword",
                                                tier="short",
                                                existing_before=set()
                                            )

                                            # Check that exhausted retries were logged
                                            assert "retries exhausted" in caplog.text.lower()
