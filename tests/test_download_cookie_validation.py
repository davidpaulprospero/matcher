"""Unit tests for download config cookie validation (US-50-012).

Validates that DownloadVideoSegmentsStage._validate_cookie_config()
emits appropriate warnings when:
- No cookie source is configured (AC1, AC2, AC4)
- cookies_from_browser is set to a browser not on PATH (AC3)
- cookies_from_browser is properly configured (AC5)
- cookies_path is properly configured (AC5)

Pytest marker: fast
"""

from __future__ import annotations

import logging
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
):
    """Create a mock download config with cookie settings."""
    cfg = MagicMock()
    cfg.cookies_from_browser = cookies_from_browser
    cfg.cookies_path = cookies_path
    return cfg


# ---------------------------------------------------------------------------
# AC1 + AC2: Warning when no cookie source configured
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestNoCookieWarning:
    """AC1/AC2: Warn when neither cookies_from_browser nor cookies_path is set."""

    def test_warning_emitted_when_no_cookie_config(self, caplog):
        """AC1: WARNING log emitted when no cookie source is configured."""
        cfg = _make_download_config()

        with caplog.at_level(logging.WARNING):
            DownloadVideoSegmentsStage._validate_cookie_config(cfg)

        assert any(
            'No cookie source configured' in rec.message
            for rec in caplog.records
        ), f"Expected 'No cookie source configured' warning, got: {[r.message for r in caplog.records]}"

    def test_warning_includes_actionable_guidance(self, caplog):
        """AC2: Warning includes guidance about cookies_from_browser."""
        cfg = _make_download_config()

        with caplog.at_level(logging.WARNING):
            DownloadVideoSegmentsStage._validate_cookie_config(cfg)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        combined = ' '.join(warning_msgs)
        assert 'cookies_from_browser' in combined, (
            f"Warning should mention cookies_from_browser, got: {combined}"
        )
        assert 'config.yaml' in combined, (
            f"Warning should mention config.yaml, got: {combined}"
        )

    def test_warning_mentions_likely_blocking(self, caplog):
        """AC1: Warning explains YouTube will likely block requests."""
        cfg = _make_download_config()

        with caplog.at_level(logging.WARNING):
            DownloadVideoSegmentsStage._validate_cookie_config(cfg)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        combined = ' '.join(warning_msgs)
        assert '403' in combined or 'block' in combined, (
            f"Warning should mention blocking/403, got: {combined}"
        )


# ---------------------------------------------------------------------------
# AC3: Warning for uninstalled browser (best-effort)
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestBrowserNotInstalled:
    """AC3: Warn when cookies_from_browser points to uninstalled browser."""

    @patch('shutil.which', return_value=None)
    def test_warning_for_missing_browser(self, mock_which, caplog):
        """AC3: WARNING when browser is not found on PATH."""
        cfg = _make_download_config(cookies_from_browser='firefox')

        with caplog.at_level(logging.WARNING):
            DownloadVideoSegmentsStage._validate_cookie_config(cfg)

        assert any(
            'does not appear to be installed' in rec.message
            for rec in caplog.records
        ), f"Expected browser-not-installed warning, got: {[r.message for r in caplog.records]}"

    @patch('shutil.which', return_value=None)
    def test_warning_includes_browser_name(self, mock_which, caplog):
        """AC3: Warning mentions the configured browser name."""
        cfg = _make_download_config(cookies_from_browser='chrome')

        with caplog.at_level(logging.WARNING):
            DownloadVideoSegmentsStage._validate_cookie_config(cfg)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        combined = ' '.join(warning_msgs)
        assert 'chrome' in combined, f"Warning should mention 'chrome', got: {combined}"


# ---------------------------------------------------------------------------
# AC5: No warning when properly configured
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestNoWarningWhenConfigured:
    """AC5: No warning when cookies_from_browser or cookies_path is set."""

    @patch('shutil.which', return_value='/usr/bin/firefox')
    def test_no_warning_with_cookies_from_browser(self, mock_which, caplog):
        """AC5: No warning when cookies_from_browser is set and browser exists."""
        cfg = _make_download_config(cookies_from_browser='firefox')

        with caplog.at_level(logging.WARNING):
            DownloadVideoSegmentsStage._validate_cookie_config(cfg)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warning_msgs, f"Expected no warnings, got: {warning_msgs}"

    def test_no_warning_with_cookies_path(self, caplog):
        """AC5: No warning when cookies_path is set."""
        cfg = _make_download_config(cookies_path='cookies.txt')

        with caplog.at_level(logging.WARNING):
            DownloadVideoSegmentsStage._validate_cookie_config(cfg)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warning_msgs, f"Expected no warnings, got: {warning_msgs}"

    @patch('shutil.which', return_value='/usr/bin/chrome')
    def test_no_warning_with_both_configured(self, mock_which, caplog):
        """AC5: No warning when both cookie sources are configured."""
        cfg = _make_download_config(
            cookies_from_browser='chrome',
            cookies_path='cookies.txt',
        )

        with caplog.at_level(logging.WARNING):
            DownloadVideoSegmentsStage._validate_cookie_config(cfg)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warning_msgs, f"Expected no warnings, got: {warning_msgs}"
