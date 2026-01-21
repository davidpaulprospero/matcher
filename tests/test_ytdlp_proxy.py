"""
Tests for yt-dlp proxy integration in VideoDownloader.

Tests that:
- Rate limit handler is initialized
- Proxy is added to yt-dlp commands when available
- Rate limit detection works in output parsing
- Success is reported to handler after download
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.downloader.core import VideoDownloader


class TestVideoDownloaderProxyInit:
    """Tests for VideoDownloader rate limit handler initialization."""

    def test_initializes_rate_limit_handler(self):
        """Should initialize rate limit handler from config."""
        with patch('src.downloader.core.get_rate_limit_handler') as mock_get:
            mock_handler = Mock()
            mock_get.return_value = mock_handler

            with patch('src.downloader.core.get_config') as mock_config:
                mock_cfg = MagicMock()
                mock_cfg.cache_dir = "/tmp/cache"
                mock_cfg.downloaded_videos_dir = "/tmp/videos"
                mock_cfg.download = MagicMock()
                mock_config.return_value = mock_cfg

                # Patch other managers to avoid side effects
                with patch('src.downloader.core.CheckpointManager'):
                    with patch('src.downloader.core.TranscodingManager'):
                        with patch('src.downloader.core.TitleFilter'):
                            with patch('src.downloader.core.SpeechScreener'):
                                with patch('src.downloader.core.SearchOptimizer'):
                                    with patch('src.downloader.core.YouTubeSearchCache'):
                                        with patch('src.downloader.core.AudioFirstPipeline'):
                                            downloader = VideoDownloader()

                mock_get.assert_called_once()
                assert downloader.rate_limit_handler is mock_handler


class TestAddCookiesToCmd:
    """Tests for _add_cookies_to_cmd proxy injection."""

    def test_adds_proxy_when_available(self):
        """Should add --proxy argument when handler has proxy."""
        with patch('src.downloader.core.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.cache_dir = "/tmp/cache"
            mock_cfg.downloaded_videos_dir = "/tmp/videos"
            mock_cfg.download = MagicMock()
            mock_cfg.download.rate_limit_bypass = None
            mock_cfg.download.cookies_from_browser = ''
            mock_config.return_value = mock_cfg

            with patch('src.downloader.core.CheckpointManager'):
                with patch('src.downloader.core.TranscodingManager'):
                    with patch('src.downloader.core.TitleFilter'):
                        with patch('src.downloader.core.SpeechScreener'):
                            with patch('src.downloader.core.SearchOptimizer'):
                                with patch('src.downloader.core.YouTubeSearchCache'):
                                    with patch('src.downloader.core.AudioFirstPipeline'):
                                        downloader = VideoDownloader()

            # Set up handler with proxy
            downloader.rate_limit_handler = Mock()
            downloader.rate_limit_handler.has_proxy = True
            downloader.rate_limit_handler.get_proxy.return_value = "socks5://127.0.0.1:1080"

            cmd = ['yt-dlp', 'URL']
            downloader._add_cookies_to_cmd(cmd)

            assert '--proxy' in cmd
            assert 'socks5://127.0.0.1:1080' in cmd

    def test_no_proxy_when_handler_has_no_proxy(self):
        """Should not add --proxy when handler has no proxy."""
        with patch('src.downloader.core.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.cache_dir = "/tmp/cache"
            mock_cfg.downloaded_videos_dir = "/tmp/videos"
            mock_cfg.download = MagicMock()
            mock_cfg.download.rate_limit_bypass = None
            mock_cfg.download.cookies_from_browser = ''
            mock_config.return_value = mock_cfg

            with patch('src.downloader.core.CheckpointManager'):
                with patch('src.downloader.core.TranscodingManager'):
                    with patch('src.downloader.core.TitleFilter'):
                        with patch('src.downloader.core.SpeechScreener'):
                            with patch('src.downloader.core.SearchOptimizer'):
                                with patch('src.downloader.core.YouTubeSearchCache'):
                                    with patch('src.downloader.core.AudioFirstPipeline'):
                                        downloader = VideoDownloader()

            # Set up handler without proxy
            downloader.rate_limit_handler = Mock()
            downloader.rate_limit_handler.has_proxy = False

            cmd = ['yt-dlp', 'URL']
            downloader._add_cookies_to_cmd(cmd)

            assert '--proxy' not in cmd

    def test_no_proxy_when_no_handler(self):
        """Should not add --proxy when no handler."""
        with patch('src.downloader.core.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.cache_dir = "/tmp/cache"
            mock_cfg.downloaded_videos_dir = "/tmp/videos"
            mock_cfg.download = MagicMock()
            mock_cfg.download.rate_limit_bypass = None
            mock_cfg.download.cookies_from_browser = ''
            mock_config.return_value = mock_cfg

            with patch('src.downloader.core.CheckpointManager'):
                with patch('src.downloader.core.TranscodingManager'):
                    with patch('src.downloader.core.TitleFilter'):
                        with patch('src.downloader.core.SpeechScreener'):
                            with patch('src.downloader.core.SearchOptimizer'):
                                with patch('src.downloader.core.YouTubeSearchCache'):
                                    with patch('src.downloader.core.AudioFirstPipeline'):
                                        downloader = VideoDownloader()

            downloader.rate_limit_handler = None

            cmd = ['yt-dlp', 'URL']
            downloader._add_cookies_to_cmd(cmd)

            assert '--proxy' not in cmd


class TestRateLimitDetection:
    """Tests for rate limit detection in yt-dlp output."""

    def test_detects_429_in_stderr(self):
        """Should detect HTTP 429 in stderr and call handler."""
        with patch('src.downloader.core.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.cache_dir = "/tmp/cache"
            mock_cfg.downloaded_videos_dir = "/tmp/videos"
            mock_cfg.download = MagicMock()
            mock_cfg.download.rate_limit_bypass = None
            mock_cfg.download.download_timeout = 60
            mock_cfg.download.first_byte_timeout = 10
            mock_config.return_value = mock_cfg

            with patch('src.downloader.core.CheckpointManager'):
                with patch('src.downloader.core.TranscodingManager'):
                    with patch('src.downloader.core.TitleFilter'):
                        with patch('src.downloader.core.SpeechScreener'):
                            with patch('src.downloader.core.SearchOptimizer'):
                                with patch('src.downloader.core.YouTubeSearchCache'):
                                    with patch('src.downloader.core.AudioFirstPipeline'):
                                        downloader = VideoDownloader()

            # Set up mock handler
            downloader.rate_limit_handler = Mock()

            # Mock subprocess to return 429 error
            mock_process = Mock()
            mock_process.communicate.return_value = (
                "",
                "ERROR: HTTP Error 429: Too Many Requests"
            )

            with patch('src.downloader.core.subprocess.Popen', return_value=mock_process):
                with patch.object(downloader, '_process_downloaded_files', return_value=[]):
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        downloader._run_download_cmd(
                            cmd=['yt-dlp', 'URL'],
                            keyword_dir=Path(tmpdir),
                            output_dir=Path(tmpdir),
                            keyword='test',
                            tier='short',
                            existing_before=set()
                        )

            downloader.rate_limit_handler.on_rate_limit.assert_called_once_with('ytdlp_short')

    def test_detects_rate_limit_text(self):
        """Should detect 'rate limit' text in output and call handler."""
        with patch('src.downloader.core.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.cache_dir = "/tmp/cache"
            mock_cfg.downloaded_videos_dir = "/tmp/videos"
            mock_cfg.download = MagicMock()
            mock_cfg.download.rate_limit_bypass = None
            mock_cfg.download.download_timeout = 60
            mock_cfg.download.first_byte_timeout = 10
            mock_config.return_value = mock_cfg

            with patch('src.downloader.core.CheckpointManager'):
                with patch('src.downloader.core.TranscodingManager'):
                    with patch('src.downloader.core.TitleFilter'):
                        with patch('src.downloader.core.SpeechScreener'):
                            with patch('src.downloader.core.SearchOptimizer'):
                                with patch('src.downloader.core.YouTubeSearchCache'):
                                    with patch('src.downloader.core.AudioFirstPipeline'):
                                        downloader = VideoDownloader()

            downloader.rate_limit_handler = Mock()

            mock_process = Mock()
            mock_process.communicate.return_value = (
                "WARNING: Rate limit reached, waiting...",
                ""
            )

            with patch('src.downloader.core.subprocess.Popen', return_value=mock_process):
                with patch.object(downloader, '_process_downloaded_files', return_value=[]):
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        downloader._run_download_cmd(
                            cmd=['yt-dlp', 'URL'],
                            keyword_dir=Path(tmpdir),
                            output_dir=Path(tmpdir),
                            keyword='test',
                            tier='medium',
                            existing_before=set()
                        )

            downloader.rate_limit_handler.on_rate_limit.assert_called_once_with('ytdlp_medium')

    def test_reports_success_on_successful_download(self):
        """Should call on_success when download succeeds."""
        with patch('src.downloader.core.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.cache_dir = "/tmp/cache"
            mock_cfg.downloaded_videos_dir = "/tmp/videos"
            mock_cfg.download = MagicMock()
            mock_cfg.download.rate_limit_bypass = None
            mock_cfg.download.download_timeout = 60
            mock_cfg.download.first_byte_timeout = 10
            mock_config.return_value = mock_cfg

            with patch('src.downloader.core.CheckpointManager'):
                with patch('src.downloader.core.TranscodingManager'):
                    with patch('src.downloader.core.TitleFilter'):
                        with patch('src.downloader.core.SpeechScreener'):
                            with patch('src.downloader.core.SearchOptimizer'):
                                with patch('src.downloader.core.YouTubeSearchCache'):
                                    with patch('src.downloader.core.AudioFirstPipeline'):
                                        downloader = VideoDownloader()

            downloader.rate_limit_handler = Mock()

            mock_process = Mock()
            mock_process.communicate.return_value = (
                "[download] 100% of 10.5MiB",
                ""
            )

            # Return a mock video to indicate success
            mock_video = Mock()

            with patch('src.downloader.core.subprocess.Popen', return_value=mock_process):
                with patch.object(downloader, '_process_downloaded_files', return_value=[mock_video]):
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        result = downloader._run_download_cmd(
                            cmd=['yt-dlp', 'URL'],
                            keyword_dir=Path(tmpdir),
                            output_dir=Path(tmpdir),
                            keyword='test',
                            tier='short',
                            existing_before=set()
                        )

            assert len(result) == 1
            downloader.rate_limit_handler.on_success.assert_called_once()
            downloader.rate_limit_handler.on_rate_limit.assert_not_called()
