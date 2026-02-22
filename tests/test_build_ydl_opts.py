"""Unit tests for DownloadVideoSegmentsStage._build_ydl_opts (US-52-005).

Validates that the extracted builder method correctly constructs ydl_opts:
- Base options: format, outtmpl, socket_timeout, retries, fragment_retries
- Cookie propagation: cookiesfrombrowser / cookiefile fallback
- Escalation application: _apply_escalation_to_ydl_opts called within builder
- Progress hooks forwarding

Pytest marker: fast
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.stages.download_segments import DownloadVideoSegmentsStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_download_config(
    *,
    cookies_from_browser: str = '',
    cookies_path: str = '',
    cookie_rotation=None,
    socket_timeout: int = 30,
    segment_socket_timeout: int = 0,
    segment_max_resolution: int = 1080,
    segment_format: str = 'best[height<={segment_max_resolution}]',
):
    """Create a mock download config with configurable settings."""
    cfg = MagicMock()
    cfg.cookies_from_browser = cookies_from_browser
    cfg.cookies_path = cookies_path
    cfg.cookie_rotation = cookie_rotation
    cfg.socket_timeout = socket_timeout
    cfg.segment_socket_timeout = segment_socket_timeout
    cfg.segment_max_resolution = segment_max_resolution
    cfg.segment_format = segment_format
    return cfg


def _make_stage(*, downloader=None):
    """Create a DownloadVideoSegmentsStage with optional mock downloader."""
    stage = DownloadVideoSegmentsStage()
    stage.downloader = downloader
    return stage


# ---------------------------------------------------------------------------
# Tests: Base options
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestBuildYdlOptsBaseOptions:
    """Verify base ydl_opts fields from config."""

    def test_includes_socket_timeout_from_config(self, tmp_path):
        """AC6: socket_timeout from config appears in ydl_opts."""
        stage = _make_stage()
        cfg = _make_download_config(socket_timeout=45)

        opts, esc = stage._build_ydl_opts(
            video_id='abc123',
            start=10.0,
            end=20.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
        )

        assert opts['socket_timeout'] == 45
        assert esc is None

    def test_segment_socket_timeout_overrides_socket_timeout(self, tmp_path):
        """segment_socket_timeout takes priority over socket_timeout."""
        stage = _make_stage()
        cfg = _make_download_config(socket_timeout=30, segment_socket_timeout=60)

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
        )

        assert opts['socket_timeout'] == 60

    def test_includes_retries_from_defaults(self, tmp_path):
        """AC6: retries and fragment_retries are set."""
        stage = _make_stage()
        cfg = _make_download_config()

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
        )

        assert opts['retries'] == 10
        assert opts['fragment_retries'] == 10

    def test_includes_format_string_from_config(self, tmp_path):
        """AC6: format string uses config values."""
        stage = _make_stage()
        cfg = _make_download_config(
            segment_max_resolution=720,
            segment_format='bestvideo[height<={segment_max_resolution}]+bestaudio',
        )

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
        )

        assert opts['format'] == 'bestvideo[height<=720]+bestaudio/best/bestvideo+bestaudio'

    def test_default_format_when_no_config(self, tmp_path):
        """Falls back to defaults when download_config is None."""
        stage = _make_stage()

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=5.0,
            end=15.0,
            output_file=tmp_path / 'out.mp4',
            download_config=None,
        )

        assert opts['format'] == 'best[height<=1080]/best/bestvideo+bestaudio'
        assert opts['socket_timeout'] == 30
        assert opts['retries'] == 10

    def test_outtmpl_set_to_output_file(self, tmp_path):
        """outtmpl is the string representation of output_file."""
        stage = _make_stage()
        out = tmp_path / 'segment.mp4'

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=out,
            download_config=None,
        )

        assert opts['outtmpl'] == str(out)

    def test_download_ranges_returns_correct_times(self, tmp_path):
        """download_ranges lambda returns correct start/end."""
        stage = _make_stage()

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=30.5,
            end=45.2,
            output_file=tmp_path / 'out.mp4',
            download_config=None,
        )

        ranges = opts['download_ranges'](None, None)
        assert len(ranges) == 1
        assert ranges[0]['start_time'] == 30.5
        assert ranges[0]['end_time'] == 45.2

    def test_progress_hooks_forwarded(self, tmp_path):
        """Progress hooks list is included when provided."""
        stage = _make_stage()
        hook = MagicMock()

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=None,
            progress_hooks=[hook],
        )

        assert opts['progress_hooks'] == [hook]

    def test_no_progress_hooks_when_not_provided(self, tmp_path):
        """progress_hooks key is absent when not provided."""
        stage = _make_stage()

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=None,
        )

        assert 'progress_hooks' not in opts


# ---------------------------------------------------------------------------
# Tests: Cookie propagation
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestBuildYdlOptsCookies:
    """AC5: Verify cookie propagation in the builder."""

    def test_cookiesfrombrowser_when_config_provides_cookies_from_browser(self, tmp_path):
        """AC5: cookiesfrombrowser set as list when cookies_from_browser configured."""
        stage = _make_stage()
        cfg = _make_download_config(cookies_from_browser='firefox')

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
        )

        assert opts['cookiesfrombrowser'] == ['firefox']
        assert 'cookiefile' not in opts

    def test_cookiefile_when_cookies_path_set(self, tmp_path):
        """cookiefile set when cookies_path configured (no browser)."""
        stage = _make_stage()
        cfg = _make_download_config(cookies_path='/path/to/cookies.txt')

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
        )

        assert opts['cookiefile'] == '/path/to/cookies.txt'
        assert 'cookiesfrombrowser' not in opts

    def test_cookiefile_from_cookie_rotation_fallback(self, tmp_path):
        """cookiefile set from first cookie_rotation file when no direct path."""
        stage = _make_stage()
        rotation = MagicMock()
        rotation.cookie_files = ['/rotated/cookies_1.txt', '/rotated/cookies_2.txt']
        cfg = _make_download_config(cookie_rotation=rotation)

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
        )

        assert opts['cookiefile'] == '/rotated/cookies_1.txt'

    def test_no_cookie_keys_when_nothing_configured(self, tmp_path):
        """No cookie keys when no cookie source configured."""
        stage = _make_stage()
        cfg = _make_download_config()

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
        )

        assert 'cookiesfrombrowser' not in opts
        assert 'cookiefile' not in opts


# ---------------------------------------------------------------------------
# Tests: Escalation application
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestBuildYdlOptsEscalation:
    """AC4: Verify escalation is applied within the builder."""

    def test_escalation_result_returned(self, tmp_path):
        """Builder returns escalation_result from escalation_mgr."""
        stage = _make_stage()
        cfg = _make_download_config()

        mock_esc_result = MagicMock()
        mock_esc_result.args = []
        mock_esc_result.rotate_cookies = False
        mock_esc_result.tier.value = 1

        mock_esc_mgr = MagicMock()
        mock_esc_mgr.get_escalation_args.return_value = mock_esc_result

        opts, esc = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
            escalation_mgr=mock_esc_mgr,
        )

        assert esc is mock_esc_result
        mock_esc_mgr.get_escalation_args.assert_called_once_with('abc123')

    @patch('src.stages.download_segments._apply_escalation_to_ydl_opts')
    def test_apply_escalation_called_within_builder(self, mock_apply, tmp_path):
        """AC4: _apply_escalation_to_ydl_opts is called within the builder."""
        stage = _make_stage()
        cfg = _make_download_config()

        mock_esc_result = MagicMock()
        mock_esc_result.args = ['--impersonate', 'Chrome-131']
        mock_esc_result.rotate_cookies = False

        mock_esc_mgr = MagicMock()
        mock_esc_mgr.get_escalation_args.return_value = mock_esc_result

        stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
            escalation_mgr=mock_esc_mgr,
        )

        mock_apply.assert_called_once()
        call_args = mock_apply.call_args
        assert call_args[0][1] is mock_esc_result

    def test_cookie_rotation_overrides_browser_cookies(self, tmp_path):
        """Tier 3 cookie rotation removes cookiesfrombrowser."""
        stage = _make_stage()
        cfg = _make_download_config(cookies_from_browser='chrome')

        mock_esc_result = MagicMock()
        mock_esc_result.args = []
        mock_esc_result.rotate_cookies = True
        mock_esc_result.tier.value = 3

        mock_esc_mgr = MagicMock()
        mock_esc_mgr.get_escalation_args.return_value = mock_esc_result

        mock_rotator = MagicMock()
        mock_rotator.get_current_cookie.return_value = '/rotated/cookie.txt'

        opts, _ = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
            escalation_mgr=mock_esc_mgr,
            cookie_rotator=mock_rotator,
        )

        assert opts['cookiefile'] == '/rotated/cookie.txt'
        assert 'cookiesfrombrowser' not in opts

    def test_impersonation_fallback_without_escalation_mgr(self, tmp_path):
        """Fallback impersonation when no escalation manager."""
        mock_imp = MagicMock()
        mock_imp.get_impersonate_args.return_value = ['--impersonate', 'Chrome-131']

        mock_dl = MagicMock()
        mock_dl.impersonation_manager = mock_imp

        stage = _make_stage(downloader=mock_dl)
        cfg = _make_download_config()

        opts, esc = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
            escalation_mgr=None,
        )

        assert opts['impersonate'] == 'Chrome-131'
        assert esc is None

    def test_no_escalation_when_no_mgr_and_no_impersonation(self, tmp_path):
        """No escalation keys when neither escalation_mgr nor impersonation_manager."""
        stage = _make_stage()
        cfg = _make_download_config()

        opts, esc = stage._build_ydl_opts(
            video_id='abc123',
            start=0.0,
            end=10.0,
            output_file=tmp_path / 'out.mp4',
            download_config=cfg,
        )

        assert 'impersonate' not in opts
        assert 'extractor_args' not in opts
        assert esc is None
