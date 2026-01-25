"""Unit tests for PathHealer."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agents.healers.path import PathHealer
from src.agents.base import HealerResult, HealerAction


class TestPathHealerInit:
    """Test PathHealer initialization."""

    def test_init_stores_config_and_project(self):
        """Test that PathHealer stores config and project_dir."""
        config = MagicMock()
        healer = PathHealer(config, "/path/to/project")

        assert healer.config is config
        assert healer.project_dir == "/path/to/project"

    def test_init_has_correct_name(self):
        """Test that PathHealer has correct name."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        assert healer.name == "path-healer"

    def test_init_has_max_path_length_constant(self):
        """Test that PathHealer has MAX_PATH_LENGTH constant."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        assert hasattr(healer, 'MAX_PATH_LENGTH')
        assert healer.MAX_PATH_LENGTH == 260  # Windows MAX_PATH

    def test_init_has_short_roots_list(self):
        """Test that PathHealer has SHORT_ROOTS list."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        assert hasattr(healer, 'SHORT_ROOTS')
        assert isinstance(healer.SHORT_ROOTS, list)
        assert len(healer.SHORT_ROOTS) > 0


class TestPathHealerCanHandle:
    """Test PathHealer.can_handle() method."""

    def test_can_handle_path_too_long(self):
        """Test that PathHealer can handle path too long errors."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        error = Exception("Path too long")
        assert healer.can_handle(error, "OUTPUT") is True

        error = Exception("Filename too long: very_long_filename.mp4")
        assert healer.can_handle(error, "OUTPUT") is True

    def test_can_handle_errno_63(self):
        """Test that PathHealer can handle ENAMETOOLONG (errno 63)."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        error = OSError("[Errno 63] File name too long")
        assert healer.can_handle(error, "OUTPUT") is True

    def test_can_handle_errno_36(self):
        """Test that PathHealer can handle macOS ENAMETOOLONG (errno 36)."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        error = OSError("[Errno 36] File name too long")
        assert healer.can_handle(error, "OUTPUT") is True

    def test_can_handle_errno_206(self):
        """Test that PathHealer can handle Windows path too long (errno 206)."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        error = OSError("[Errno 206] The filename or extension is too long")
        assert healer.can_handle(error, "OUTPUT") is True

    def test_can_handle_unicode_errors(self):
        """Test that PathHealer can handle unicode encoding errors."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        # UnicodeEncodeError requires a str, not bytes for second argument
        error = UnicodeEncodeError("utf-8", "test", 0, 1, "invalid")
        assert healer.can_handle(error, "OUTPUT") is True

        error = Exception("Unicode decode error in filename")
        assert healer.can_handle(error, "OUTPUT") is True

    def test_can_handle_invalid_path(self):
        """Test that PathHealer can handle invalid path errors."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        error = Exception("Invalid path: contains illegal characters")
        assert healer.can_handle(error, "OUTPUT") is True

    def test_cannot_handle_unrelated_error(self):
        """Test that PathHealer doesn't handle unrelated errors."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        error = Exception("Disk full")
        assert healer.can_handle(error, "OUTPUT") is False

        error = Exception("Network timeout")
        assert healer.can_handle(error, "OUTPUT") is False


class TestPathHealerHandlePathTooLong:
    """Test PathHealer._handle_path_too_long() method."""

    def test_path_too_long_tries_short_roots(self, tmp_path):
        """Test that path too long handler tries short roots."""
        config = MagicMock()
        config.download = MagicMock()
        config.download.root_dir = str(tmp_path / "very" / "long" / "path")

        healer = PathHealer(config, str(tmp_path))
        state = MagicMock()

        # Mock finding an available short root
        with patch.object(healer, '_find_available_short_root', return_value="E:/v"):
            result = healer._handle_path_too_long(Exception("path too long"), state)

        assert result.success is True
        assert result.action == HealerAction.MODIFY_CONFIG
        assert "E:/v" in result.message

    def test_path_too_long_falls_back_to_truncate(self, tmp_path):
        """Test that path too long falls back to filename truncation."""
        config = MagicMock()
        config.download = MagicMock()

        healer = PathHealer(config, str(tmp_path))
        state = MagicMock()

        # Mock no available short roots
        with patch.object(healer, '_find_available_short_root', return_value=None):
            result = healer._handle_path_too_long(Exception("path too long"), state)

        # Should try truncation
        assert isinstance(result, HealerResult)


class TestPathHealerHandleUnicodeError:
    """Test PathHealer._handle_unicode_error() method."""

    def test_unicode_error_sanitizes_downloads(self, tmp_path):
        """Test that unicode handler sanitizes download paths."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        state = MagicMock()
        download1 = MagicMock()
        download1.file = "/path/to/file\u2019s_name.mp4"  # Smart quote
        download2 = MagicMock()
        download2.file = "/path/to/normal_file.mp4"
        state.downloads = [download1, download2]

        result = healer._handle_unicode_error(Exception("unicode"), state)

        assert result.success is True
        assert "\u2019" not in download1.file

    def test_unicode_error_fails_when_no_downloads(self, tmp_path):
        """Test that unicode handler fails when no downloads in state."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        state = MagicMock()
        state.downloads = []

        result = healer._handle_unicode_error(Exception("unicode"), state)

        assert result.success is False


class TestPathHealerSanitizeUnicode:
    """Test PathHealer._sanitize_unicode() method."""

    def test_sanitize_unicode_replaces_smart_quotes(self, tmp_path):
        """Test that _sanitize_unicode replaces smart quotes."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        result = healer._sanitize_unicode("/path/to/file\u2019s_name.mp4")

        assert "\u2019" not in result
        assert "'" in result

    def test_sanitize_unicode_replaces_smart_double_quotes(self, tmp_path):
        """Test that _sanitize_unicode replaces smart double quotes."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        result = healer._sanitize_unicode("/path/to/\u201cquoted\u201d.mp4")

        assert "\u201c" not in result
        assert "\u201d" not in result

    def test_sanitize_unicode_replaces_dashes(self, tmp_path):
        """Test that _sanitize_unicode replaces en-dash and em-dash."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        result = healer._sanitize_unicode("/path/to/file\u2013name\u2014test.mp4")

        assert "\u2013" not in result
        assert "\u2014" not in result
        assert "-" in result

    def test_sanitize_unicode_replaces_ellipsis(self, tmp_path):
        """Test that _sanitize_unicode replaces ellipsis character."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        result = healer._sanitize_unicode("/path/to/file\u2026.mp4")

        assert "\u2026" not in result
        assert "..." in result

    def test_sanitize_unicode_removes_windows_invalid_chars(self, tmp_path):
        """Test that _sanitize_unicode removes Windows-invalid characters."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        result = healer._sanitize_unicode("/path/to/file<name>:test?.mp4")

        assert "<" not in result
        assert ">" not in result
        assert "?" not in result

    def test_sanitize_unicode_handles_empty_string(self, tmp_path):
        """Test that _sanitize_unicode handles empty string."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        result = healer._sanitize_unicode("")

        assert result == ""

    def test_sanitize_unicode_handles_none(self, tmp_path):
        """Test that _sanitize_unicode handles None."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        result = healer._sanitize_unicode(None)

        assert result is None


class TestPathHealerCheckPathLength:
    """Test PathHealer.check_path_length() method."""

    def test_check_path_length_short_path(self):
        """Test check_path_length with short path."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        assert healer.check_path_length("/short/path/file.mp4") is True

    def test_check_path_length_long_path(self):
        """Test check_path_length with path exceeding MAX_PATH."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        long_path = "a" * 300
        assert healer.check_path_length(long_path) is False

    def test_check_path_length_exactly_max(self):
        """Test check_path_length with path exactly at MAX_PATH."""
        config = MagicMock()
        healer = PathHealer(config, "/tmp")

        max_path = "a" * 260
        assert healer.check_path_length(max_path) is True


class TestPathHealerEstimateSafeFilenameLength:
    """Test PathHealer.estimate_safe_filename_length() method."""

    def test_estimate_safe_filename_length_short_dir(self, tmp_path):
        """Test estimate with short directory path."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        safe_len = healer.estimate_safe_filename_length(Path("E:/v"))

        assert safe_len >= 20
        assert safe_len <= 200

    def test_estimate_safe_filename_length_long_dir(self, tmp_path):
        """Test estimate with long directory path."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        long_dir = Path("E:/Projects/very/long/path/structure/that/goes/on")
        safe_len = healer.estimate_safe_filename_length(long_dir)

        assert safe_len >= 20  # Minimum guaranteed

    def test_estimate_safe_filename_length_respects_max(self, tmp_path):
        """Test that estimate doesn't exceed 200 chars."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        short_dir = Path("/")
        safe_len = healer.estimate_safe_filename_length(short_dir)

        assert safe_len <= 200


class TestPathHealerFindAvailableShortRoot:
    """Test PathHealer._find_available_short_root() method."""

    def test_find_available_short_root_returns_writable(self, tmp_path):
        """Test that _find_available_short_root returns a writable path."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        # Create a writable short root
        short_root = tmp_path / "v"
        short_root.mkdir()

        # Mock SHORT_ROOTS to include our test path
        with patch.object(PathHealer, 'SHORT_ROOTS', [str(short_root)]):
            result = healer._find_available_short_root()

        assert result == str(short_root)

    def test_find_available_short_root_none_available(self, tmp_path):
        """Test that _find_available_short_root returns None when none available."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        # Mock SHORT_ROOTS with nonexistent paths
        with patch.object(PathHealer, 'SHORT_ROOTS', ["/nonexistent/path"]):
            result = healer._find_available_short_root()

        assert result is None


class TestPathHealerTruncateFilenames:
    """Test PathHealer._truncate_filenames() method."""

    def test_truncate_filenames_sets_max_length(self, tmp_path):
        """Test that _truncate_filenames sets max_filename_length in config."""
        config = MagicMock()
        config.download = MagicMock()

        healer = PathHealer(config, str(tmp_path))
        state = MagicMock()

        result = healer._truncate_filenames(Exception("path too long"), state)

        assert result.success is True
        assert result.action == HealerAction.MODIFY_CONFIG
        assert config.download.max_filename_length == 50

    def test_truncate_filenames_handles_dict_config(self, tmp_path):
        """Test that _truncate_filenames handles dict-style config."""
        config = MagicMock()
        # Non-empty dict to pass truthiness check
        config.download = {"format": "bestvideo+bestaudio"}

        healer = PathHealer(config, str(tmp_path))
        state = MagicMock()

        result = healer._truncate_filenames(Exception("path too long"), state)

        assert result.success is True
        assert config.download['max_filename_length'] == 50


class TestPathHealerFix:
    """Test PathHealer.fix() method routing."""

    def test_fix_routes_path_too_long(self, tmp_path):
        """Test that fix() routes path too long errors correctly."""
        config = MagicMock()
        config.download = MagicMock()

        healer = PathHealer(config, str(tmp_path))
        state = MagicMock()

        with patch.object(healer, '_find_available_short_root', return_value=None):
            result = healer.fix(Exception("Path too long"), state, "OUTPUT")

        assert isinstance(result, HealerResult)

    def test_fix_routes_unicode_error(self, tmp_path):
        """Test that fix() routes unicode errors correctly."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        state = MagicMock()
        state.downloads = []

        result = healer.fix(Exception("Unicode encode error"), state, "OUTPUT")

        assert isinstance(result, HealerResult)

    def test_fix_routes_invalid_chars(self, tmp_path):
        """Test that fix() routes invalid character errors correctly."""
        config = MagicMock()
        healer = PathHealer(config, str(tmp_path))

        state = MagicMock()
        state.downloads = []

        result = healer.fix(Exception("Invalid path: illegal character"), state, "OUTPUT")

        assert isinstance(result, HealerResult)

    def test_fix_routes_errno_63(self, tmp_path):
        """Test that fix() routes ENAMETOOLONG correctly."""
        config = MagicMock()
        config.download = MagicMock()

        healer = PathHealer(config, str(tmp_path))
        state = MagicMock()

        with patch.object(healer, '_find_available_short_root', return_value=None):
            result = healer.fix(OSError("[Errno 63] File name too long"), state, "OUTPUT")

        assert isinstance(result, HealerResult)
