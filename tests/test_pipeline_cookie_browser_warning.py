"""Unit tests for pipeline startup cookies_from_browser PATH validation (US-57-004).

Validates that PipelineOrchestrator._warn_cookies_from_browser() emits a
WARNING when cookies_from_browser is configured but the browser executable
is not findable on PATH, and emits no warning when the browser is found.

Pytest marker: fast
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(cookies_from_browser: str = ''):
    """Create a mock Config with download.cookies_from_browser set."""
    config = MagicMock()
    config.download.cookies_from_browser = cookies_from_browser
    # Defaults so _validate_config doesn't blow up on other checks
    config.cache.cache_dir = ''
    return config


def _create_pipeline(config, tmp_path):
    """Instantiate PipelineOrchestrator with validation patches to isolate our test."""
    with patch.object(PipelineOrchestrator, '_validate_config', return_value=[]):
        with patch('src.pipeline.CheckpointManager'):
            pipeline = PipelineOrchestrator(config, tmp_path)
    return pipeline


# ---------------------------------------------------------------------------
# AC4: Warning emitted when browser NOT on PATH
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestCookiesBrowserNotOnPath:
    """Warning emitted when cookies_from_browser is set but browser not on PATH."""

    def test_warning_when_browser_not_found(self, tmp_path, caplog):
        """AC4: shutil.which returns None → warning emitted."""
        config = _make_config(cookies_from_browser='firefox')
        pipeline = _create_pipeline(config, tmp_path)

        with patch('src.pipeline_validator.shutil.which', return_value=None):
            with caplog.at_level(logging.WARNING):
                pipeline._warn_cookies_from_browser()

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_msgs) == 1, f"Expected exactly 1 warning, got {len(warning_msgs)}: {warning_msgs}"
        msg = warning_msgs[0]
        assert 'cookies_from_browser is set to "firefox"' in msg
        assert 'not found on PATH' in msg
        assert 'Tier 3 cookie extraction will fail' in msg
        assert 'download.cookies_path' in msg
        assert 'install firefox' in msg

    def test_warning_for_chrome(self, tmp_path, caplog):
        """AC4: Chrome browser also triggers warning when not found."""
        config = _make_config(cookies_from_browser='chrome')
        pipeline = _create_pipeline(config, tmp_path)

        with patch('src.pipeline_validator.shutil.which', return_value=None):
            with caplog.at_level(logging.WARNING):
                pipeline._warn_cookies_from_browser()

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('"chrome"' in m and 'not found on PATH' in m for m in warning_msgs)

    def test_warning_for_edge_maps_to_msedge(self, tmp_path, caplog):
        """AC4: Edge maps to msedge executable; warning when neither found."""
        config = _make_config(cookies_from_browser='edge')
        pipeline = _create_pipeline(config, tmp_path)

        which_calls = []

        def mock_which(name):
            which_calls.append(name)
            return None

        with patch('src.pipeline_validator.shutil.which', side_effect=mock_which):
            with caplog.at_level(logging.WARNING):
                pipeline._warn_cookies_from_browser()

        # Should check msedge (mapped) AND edge (raw)
        assert 'msedge' in which_calls
        assert 'edge' in which_calls
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_msgs) == 1


# ---------------------------------------------------------------------------
# AC5: No warning when browser IS on PATH
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestCookiesBrowserOnPath:
    """No warning emitted when cookies_from_browser browser is found on PATH."""

    def test_no_warning_when_browser_found(self, tmp_path, caplog):
        """AC5: shutil.which returns a path → no warning emitted."""
        config = _make_config(cookies_from_browser='firefox')
        pipeline = _create_pipeline(config, tmp_path)

        with patch('src.pipeline_validator.shutil.which', return_value='/usr/bin/firefox'):
            with caplog.at_level(logging.WARNING):
                pipeline._warn_cookies_from_browser()

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_msgs) == 0, f"Expected no warnings, got: {warning_msgs}"

    def test_no_warning_when_no_cookies_from_browser(self, tmp_path, caplog):
        """AC5: Empty cookies_from_browser → no warning (nothing to validate)."""
        config = _make_config(cookies_from_browser='')
        pipeline = _create_pipeline(config, tmp_path)

        with caplog.at_level(logging.WARNING):
            pipeline._warn_cookies_from_browser()

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_msgs) == 0, f"Expected no warnings, got: {warning_msgs}"

    def test_no_warning_when_no_download_config(self, tmp_path, caplog):
        """AC5: No download config at all → no warning."""
        config = MagicMock()
        config.download = None
        config.cache.cache_dir = ''
        pipeline = _create_pipeline(config, tmp_path)

        with caplog.at_level(logging.WARNING):
            pipeline._warn_cookies_from_browser()

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_msgs) == 0


# ---------------------------------------------------------------------------
# Integration: Validation runs during _validate_config
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestValidationIntegration:
    """_warn_cookies_from_browser is called from _validate_config."""

    def test_validate_config_calls_browser_warning(self, tmp_path):
        """_validate_config invokes _warn_cookies_from_browser (via PipelineValidator)."""
        config = _make_config(cookies_from_browser='firefox')
        pipeline = _create_pipeline(config, tmp_path)

        with patch.object(pipeline._validator, '_warn_cookies_from_browser') as mock_warn:
            pipeline._validate_config()

        mock_warn.assert_called_once()

    def test_validation_does_not_block_startup(self, tmp_path, caplog):
        """Warning does NOT appear in the returned error list."""
        config = _make_config(cookies_from_browser='firefox')
        pipeline = _create_pipeline(config, tmp_path)

        with patch('src.pipeline_validator.shutil.which', return_value=None):
            with caplog.at_level(logging.WARNING):
                errors = pipeline._validate_config()

        # The browser warning should NOT be in errors (which would block startup)
        browser_errors = [e for e in errors if 'cookies_from_browser' in e]
        assert len(browser_errors) == 0, f"Browser warning should not be an error: {browser_errors}"

        # But the warning should be in the log
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('cookies_from_browser' in m for m in warning_msgs)
