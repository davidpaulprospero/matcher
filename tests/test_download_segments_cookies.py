"""Integration tests for download_segments cookie propagation (US-50-010).

Validates that ydl_opts passed to YoutubeDL correctly propagate cookie
configuration from DownloadConfig:
- cookies_from_browser -> cookiesfrombrowser (as list)
- cookies_path -> cookiefile (as string)
- Priority: cookies_from_browser wins over cookies_path
- Neither set: no cookie keys in ydl_opts

All tests mock YoutubeDL to avoid actual network calls.

Pytest marker: fast
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.stages.download_segments import DownloadVideoSegmentsStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_downloader(
    *,
    cookies_from_browser: str = '',
    cookies_path: str = '',
    cookie_rotation=None,
):
    """Create a mock downloader with configurable cookie settings."""
    mock_dl = MagicMock()
    mock_dl.escalation_manager = None
    mock_dl.circuit_breaker = None
    mock_dl.cookie_rotator = None
    mock_dl.retry_queue = MagicMock()
    mock_dl.retry_queue.has_pending.return_value = False
    mock_dl.impersonation_manager = None

    mock_dl.download_config = MagicMock()
    mock_dl.download_config.bot_detection_tier_floor_threshold = 100
    mock_dl.download_config.bot_detection_abort_threshold = 100
    mock_dl.download_config.segment_socket_timeout = 30
    mock_dl.download_config.socket_timeout = 30
    mock_dl.download_config.segment_max_resolution = 1080
    mock_dl.download_config.segment_format = 'best[height<={segment_max_resolution}]'
    mock_dl.download_config.segment_stall_timeout = 0
    mock_dl.download_config.cookies_from_browser = cookies_from_browser
    mock_dl.download_config.cookies_path = cookies_path
    mock_dl.download_config.cookie_rotation = cookie_rotation
    mock_dl.download_config.ffmpeg_location = ''
    return mock_dl


def _make_segments():
    """Return a single segment for testing."""
    return [{'video_id': 'test_vid_001', 'start': 0.0, 'end': 10.0}]


def _yt_dlp_mock_context():
    """Return a patch context for yt_dlp.YoutubeDL that captures opts.

    The mock simulates a successful download by writing a dummy output file.

    Returns:
        (patch_context, captured_opts_list)
    """
    captured_opts: List[Dict[str, Any]] = []

    class _FakeYDL:
        def __init__(self, opts):
            captured_opts.append(dict(opts))
            self._opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def download(self, urls):
            outtmpl = self._opts.get('outtmpl', '')
            if outtmpl:
                Path(outtmpl).write_bytes(b'\x00' * 1024)
            return None

    ctx = patch('yt_dlp.YoutubeDL', _FakeYDL)
    return ctx, captured_opts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestCookiePropagation:
    """Integration tests for cookie config -> ydl_opts propagation."""

    def test_cookies_from_browser_sets_cookiesfrombrowser_as_list(self, tmp_path):
        """AC1: cookies_from_browser='firefox' -> cookiesfrombrowser=['firefox']."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(cookies_from_browser='firefox')

        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        ctx, captured = _yt_dlp_mock_context()
        with ctx:
            stage._download_segments(
                segments=_make_segments(),
                output_dir=output_dir,
                buffer_seconds=2.0,
                progress_callback=None,
            )

        assert len(captured) == 1
        opts = captured[0]
        assert 'cookiesfrombrowser' in opts
        assert opts['cookiesfrombrowser'] == ['firefox']
        assert isinstance(opts['cookiesfrombrowser'], list)
        # cookiefile should NOT be set when browser cookies are used
        assert 'cookiefile' not in opts

    def test_cookies_path_sets_cookiefile(self, tmp_path):
        """AC2: cookies_path without cookies_from_browser -> cookiefile."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            cookies_from_browser='',
            cookies_path='/path/to/cookies.txt',
        )

        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        ctx, captured = _yt_dlp_mock_context()
        with ctx:
            stage._download_segments(
                segments=_make_segments(),
                output_dir=output_dir,
                buffer_seconds=2.0,
                progress_callback=None,
            )

        assert len(captured) == 1
        opts = captured[0]
        assert 'cookiefile' in opts
        assert opts['cookiefile'] == '/path/to/cookies.txt'
        # cookiesfrombrowser should NOT be set
        assert 'cookiesfrombrowser' not in opts

    def test_cookies_from_browser_takes_priority_over_cookies_path(self, tmp_path):
        """AC3: When both are set, cookies_from_browser wins."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            cookies_from_browser='chrome',
            cookies_path='/path/to/cookies.txt',
        )

        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        ctx, captured = _yt_dlp_mock_context()
        with ctx:
            stage._download_segments(
                segments=_make_segments(),
                output_dir=output_dir,
                buffer_seconds=2.0,
                progress_callback=None,
            )

        assert len(captured) == 1
        opts = captured[0]
        # Browser cookies take priority
        assert 'cookiesfrombrowser' in opts
        assert opts['cookiesfrombrowser'] == ['chrome']
        # cookiefile should NOT be set — browser cookies override
        assert 'cookiefile' not in opts

    def test_no_cookies_configured_no_cookie_keys(self, tmp_path):
        """AC4: Neither cookies_from_browser nor cookies_path -> no cookie keys."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            cookies_from_browser='',
            cookies_path='',
            cookie_rotation=None,
        )

        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        ctx, captured = _yt_dlp_mock_context()
        with ctx:
            stage._download_segments(
                segments=_make_segments(),
                output_dir=output_dir,
                buffer_seconds=2.0,
                progress_callback=None,
            )

        assert len(captured) == 1
        opts = captured[0]
        assert 'cookiesfrombrowser' not in opts
        assert 'cookiefile' not in opts

    def test_cookie_rotation_fallback_when_no_browser_or_path(self, tmp_path):
        """Bonus: cookie_rotation files used as fallback when no direct config."""
        stage = DownloadVideoSegmentsStage()

        # Set up cookie_rotation with a file list
        cookie_rotation = MagicMock()
        cookie_rotation.cookie_files = ['/rotated/cookies1.txt', '/rotated/cookies2.txt']

        stage.downloader = _make_mock_downloader(
            cookies_from_browser='',
            cookies_path='',
            cookie_rotation=cookie_rotation,
        )

        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        ctx, captured = _yt_dlp_mock_context()
        with ctx:
            stage._download_segments(
                segments=_make_segments(),
                output_dir=output_dir,
                buffer_seconds=2.0,
                progress_callback=None,
            )

        assert len(captured) == 1
        opts = captured[0]
        # Should fall back to first cookie_rotation file
        assert 'cookiefile' in opts
        assert opts['cookiefile'] == '/rotated/cookies1.txt'
        assert 'cookiesfrombrowser' not in opts
