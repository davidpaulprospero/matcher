"""Tests for Windows subprocess console suppression helpers."""

from unittest.mock import Mock, patch

import pytest

import src.utils as utils_module
from src.caption_fetcher import CaptionFetcher
from src.otio.entities import _get_video_duration_frames as otio_get_video_duration_frames
from src.otio.utils import _get_media_duration as otio_get_media_duration
from src.otio_builder import (
    _get_media_duration as legacy_get_media_duration,
    _get_video_duration_frames as legacy_get_video_duration_frames,
)
from src.utils import windows_no_window_kwargs, windows_pythonw_executable


@pytest.mark.fast
class TestWindowsNoWindowKwargs:
    """Verify Windows-only subprocess kwargs are generated correctly."""

    def test_returns_empty_dict_off_windows(self, monkeypatch):
        """Non-Windows platforms should not add subprocess flags."""
        monkeypatch.setattr(utils_module, "IS_WINDOWS", False)

        assert windows_no_window_kwargs() == {}

    def test_returns_create_no_window_flag_on_windows(self, monkeypatch):
        """Windows should add CREATE_NO_WINDOW when available."""
        monkeypatch.setattr(utils_module, "IS_WINDOWS", True)
        monkeypatch.setattr(
            utils_module.subprocess,
            "CREATE_NO_WINDOW",
            0x08000000,
            raising=False,
        )

        assert windows_no_window_kwargs() == {"creationflags": 0x08000000}

    def test_returns_pythonw_executable_on_windows(self, monkeypatch, tmp_path):
        """Windows worker processes should prefer pythonw.exe when available."""
        python_exe = tmp_path / "python.exe"
        pythonw_exe = tmp_path / "pythonw.exe"
        python_exe.write_text("", encoding="utf-8")
        pythonw_exe.write_text("", encoding="utf-8")

        monkeypatch.setattr(utils_module, "IS_WINDOWS", True)
        monkeypatch.setattr(utils_module.sys, "executable", str(python_exe))

        assert windows_pythonw_executable() == str(pythonw_exe)

    def test_returns_none_when_pythonw_missing(self, monkeypatch, tmp_path):
        """Worker helper should gracefully fall back when pythonw.exe is unavailable."""
        python_exe = tmp_path / "python.exe"
        python_exe.write_text("", encoding="utf-8")

        monkeypatch.setattr(utils_module, "IS_WINDOWS", True)
        monkeypatch.setattr(utils_module.sys, "executable", str(python_exe))

        assert windows_pythonw_executable() is None


@pytest.mark.fast
class TestCaptionFetcherNoWindowIntegration:
    """Ensure CaptionFetcher forwards the Windows subprocess kwargs."""

    def test_list_available_languages_passes_creationflags(self):
        """Caption preflight should pass through no-window kwargs to subprocess.run."""
        fetcher = CaptionFetcher(list_subs_cache=None, preflight_cache=None)
        mock_result = Mock(
            returncode=0,
            stdout=(
                "[info] Available subtitles for dQw4w9WgXcQ:\n"
                "Language  Name                 Formats\n"
                "en        English              vtt, ttml, srv3, srv2, srv1, json3\n"
            ),
            stderr="",
        )

        with patch(
            "src.caption_fetcher.windows_no_window_kwargs",
            return_value={"creationflags": 12345},
        ), patch("src.caption_fetcher.subprocess.run", return_value=mock_result) as mock_run:
            fetcher.list_available_languages("dQw4w9WgXcQ")

        assert mock_run.call_args.kwargs["creationflags"] == 12345


@pytest.mark.fast
class TestOtioNoWindowIntegration:
    """Ensure output-stage ffprobe helpers suppress Windows console windows."""

    def test_otio_media_duration_passes_creationflags(self):
        """Refactored OTIO helper should forward no-window kwargs to ffprobe."""
        mock_result = Mock(returncode=0, stdout="12.5\n", stderr="")

        with patch(
            "src.otio.utils.windows_no_window_kwargs",
            return_value={"creationflags": 12345},
        ), patch("src.otio.utils.subprocess.run", return_value=mock_result) as mock_run:
            assert otio_get_media_duration("D:/tmp/test.mp4") == 12.5

        assert mock_run.call_args.kwargs["creationflags"] == 12345

    def test_otio_entity_duration_passes_creationflags(self):
        """Entity-track ffprobe helper should forward no-window kwargs."""
        mock_result = Mock(returncode=0, stdout="10\n", stderr="")

        with patch(
            "src.otio.entities.windows_no_window_kwargs",
            return_value={"creationflags": 12345},
        ), patch("subprocess.run", return_value=mock_result) as mock_run:
            assert otio_get_video_duration_frames("D:/tmp/test.mp4", frame_rate=30.0) == 300

        assert mock_run.call_args.kwargs["creationflags"] == 12345

    def test_legacy_otio_builder_ffprobe_passes_creationflags(self):
        """Legacy OTIO builder helpers should also suppress ffprobe windows."""
        mock_result = Mock(returncode=0, stdout="9\n", stderr="")

        with patch(
            "src.otio_builder.windows_no_window_kwargs",
            return_value={"creationflags": 12345},
        ), patch("subprocess.run", return_value=mock_result) as mock_run:
            assert legacy_get_media_duration("D:/tmp/test.mp4") == 9.0
            assert legacy_get_video_duration_frames("D:/tmp/test.mp4", frame_rate=30.0) == 270

        assert mock_run.call_args.kwargs["creationflags"] == 12345
