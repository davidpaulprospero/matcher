"""
Tests for CLI config_utils root directory validation and path resolution.

Covers validate_root_directories() and make_paths_project_relative().
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from src.cli.config_utils import validate_root_directories, make_paths_project_relative


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(download_root=None, image_root=None):
    """Build a minimal mock Config with download.root_dir and image_search.root_dir."""
    config = MagicMock()
    config.download = MagicMock()
    config.download.root_dir = download_root
    config.image_search = MagicMock()
    config.image_search.root_dir = image_root
    return config


# ===========================================================================
# AC-1: validate_root_directories() creates missing directories
# ===========================================================================

class TestValidateRootDirCreatesMissing:
    """Test that validate_root_directories() calls mkdir for missing absolute dirs."""

    @pytest.mark.fast
    def test_creates_download_root_dir(self, tmp_path):
        """mkdir is called for download.root_dir when the path is absolute but doesn't exist."""
        missing = tmp_path / "videos"
        config = _make_config(download_root=str(missing))

        validate_root_directories(config)

        assert missing.exists(), "download.root_dir should have been created"

    @pytest.mark.fast
    def test_creates_image_search_root_dir(self, tmp_path):
        """mkdir is called for image_search.root_dir when the path is absolute but doesn't exist."""
        missing = tmp_path / "images"
        config = _make_config(image_root=str(missing))

        validate_root_directories(config)

        assert missing.exists(), "image_search.root_dir should have been created"

    @pytest.mark.fast
    def test_creates_both_root_dirs(self, tmp_path):
        """Both download and image root dirs are created when both missing."""
        vid = tmp_path / "v"
        img = tmp_path / "i"
        config = _make_config(download_root=str(vid), image_root=str(img))

        validate_root_directories(config)

        assert vid.exists()
        assert img.exists()

    @pytest.mark.fast
    def test_existing_dir_no_error(self, tmp_path):
        """An already-existing root dir is accepted without error."""
        existing = tmp_path / "already"
        existing.mkdir()
        config = _make_config(download_root=str(existing))

        # Should not raise
        validate_root_directories(config)

    @pytest.mark.fast
    def test_creates_nested_dirs(self, tmp_path):
        """mkdir(parents=True) creates intermediate directories."""
        nested = tmp_path / "a" / "b" / "c"
        config = _make_config(download_root=str(nested))

        validate_root_directories(config)

        assert nested.exists()


# ===========================================================================
# AC-2: validate_root_directories() rejects relative paths
# ===========================================================================

class TestValidateRootDirRejectsRelative:
    """Test that relative paths cause SystemExit."""

    @pytest.mark.fast
    def test_relative_download_root_exits(self):
        """SystemExit when download.root_dir is relative (e.g., './videos')."""
        config = _make_config(download_root="./videos")

        with pytest.raises(SystemExit):
            validate_root_directories(config)

    @pytest.mark.fast
    def test_relative_image_root_exits(self):
        """SystemExit when image_search.root_dir is relative."""
        config = _make_config(image_root="images")

        with pytest.raises(SystemExit):
            validate_root_directories(config)

    @pytest.mark.fast
    def test_relative_both_exits(self):
        """SystemExit when both root_dirs are relative."""
        config = _make_config(download_root="./v", image_root="./i")

        with pytest.raises(SystemExit):
            validate_root_directories(config)

    @pytest.mark.fast
    def test_error_message_mentions_path(self, capsys):
        """Error output includes the offending relative path."""
        config = _make_config(download_root="./videos")

        with pytest.raises(SystemExit):
            validate_root_directories(config)

        captured = capsys.readouterr()
        assert "./videos" in captured.out


# ===========================================================================
# AC-3: validate_root_directories() handles None root_dir gracefully
# ===========================================================================

class TestValidateRootDirHandlesNone:
    """Test no error when root_dir is None or empty string."""

    @pytest.mark.fast
    def test_none_download_root(self):
        """No error when download.root_dir is None."""
        config = _make_config(download_root=None, image_root=None)

        # Should not raise
        validate_root_directories(config)

    @pytest.mark.fast
    def test_empty_string_download_root(self):
        """No error when download.root_dir is empty string."""
        config = _make_config(download_root="", image_root="")

        # Should not raise
        validate_root_directories(config)

    @pytest.mark.fast
    def test_none_image_root_with_valid_download(self, tmp_path):
        """None image_root doesn't interfere with valid download root."""
        vid = tmp_path / "v"
        config = _make_config(download_root=str(vid), image_root=None)

        validate_root_directories(config)

        assert vid.exists()


# ===========================================================================
# AC-4: validate_root_directories() reports creation failure
# ===========================================================================

class TestValidateRootDirReportsFailure:
    """Test error message includes path when mkdir raises PermissionError."""

    @pytest.mark.fast
    def test_permission_error_includes_path(self, capsys):
        """Error message includes path when mkdir fails with PermissionError."""
        bad_path = "Z:/nonexistent/protected"
        config = _make_config(download_root=bad_path)

        with patch.object(Path, 'exists', return_value=False):
            with patch.object(Path, 'is_absolute', return_value=True):
                with patch.object(Path, 'mkdir', side_effect=PermissionError("Access denied")):
                    with pytest.raises(SystemExit):
                        validate_root_directories(config)

        captured = capsys.readouterr()
        assert bad_path in captured.out

    @pytest.mark.fast
    def test_permission_error_image_root(self, capsys):
        """Error message includes path for image_search.root_dir failure."""
        bad_path = "Z:/protected/images"
        config = _make_config(image_root=bad_path)

        with patch.object(Path, 'exists', return_value=False):
            with patch.object(Path, 'is_absolute', return_value=True):
                with patch.object(Path, 'mkdir', side_effect=PermissionError("Access denied")):
                    with pytest.raises(SystemExit):
                        validate_root_directories(config)

        captured = capsys.readouterr()
        assert bad_path in captured.out

    @pytest.mark.fast
    def test_oserror_reported(self, capsys):
        """Generic OSError is also reported with path."""
        bad_path = "X:/broken"
        config = _make_config(download_root=bad_path)

        with patch.object(Path, 'exists', return_value=False):
            with patch.object(Path, 'is_absolute', return_value=True):
                with patch.object(Path, 'mkdir', side_effect=OSError("Disk failure")):
                    with pytest.raises(SystemExit):
                        validate_root_directories(config)

        captured = capsys.readouterr()
        assert bad_path in captured.out


# ===========================================================================
# AC-5: make_paths_project_relative() converts paths
# ===========================================================================

class TestMakePathsProjectRelative:
    """Test that make_paths_project_relative() delegates to _resolve_paths()."""

    @pytest.mark.fast
    def test_output_dir_resolved(self, tmp_path):
        """Relative output_dir becomes project-relative."""
        from src.config.base import Config

        config = Config()
        config.output.output_dir = "output"

        result = make_paths_project_relative(config, tmp_path)

        expected = str(tmp_path / "output")
        assert result.output.output_dir == expected

    @pytest.mark.fast
    def test_transcription_cache_dir_resolved(self, tmp_path):
        """Relative transcription.cache_dir becomes project-relative."""
        from src.config.base import Config

        config = Config()
        config.transcription.cache_dir = "my_cache"

        result = make_paths_project_relative(config, tmp_path)

        expected = str(tmp_path / "my_cache")
        assert result.transcription.cache_dir == expected

    @pytest.mark.fast
    def test_cache_dir_resolved(self, tmp_path):
        """Relative cache.cache_dir becomes project-relative."""
        from src.config.base import Config

        config = Config()
        config.cache.cache_dir = ".cache"

        result = make_paths_project_relative(config, tmp_path)

        assert str(tmp_path) in result.cache.cache_dir
        assert ".cache" in result.cache.cache_dir

    @pytest.mark.fast
    def test_absolute_paths_not_changed(self, tmp_path):
        """Absolute paths are NOT converted to project-relative."""
        from src.config.base import Config

        abs_path = str(tmp_path / "absolute_output")
        config = Config()
        config.output.output_dir = abs_path

        result = make_paths_project_relative(config, tmp_path)

        # Absolute path should remain unchanged
        assert result.output.output_dir == abs_path

    @pytest.mark.fast
    def test_returns_config_object(self, tmp_path):
        """make_paths_project_relative returns the config object."""
        from src.config.base import Config

        config = Config()

        result = make_paths_project_relative(config, tmp_path)

        assert result is config
