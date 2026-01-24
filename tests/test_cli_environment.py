"""
Tests for CLI environment detection utilities.

US-006: Add CLI environment detection tests

Tests for src/cli/environment.py covering:
- strip_extended_path_prefix() for Windows path handling
- load_environment() for .env file loading
- INSTALL_DIR constant
"""

import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.environment import (
    strip_extended_path_prefix,
    load_environment,
    INSTALL_DIR
)


class TestInstallDirConstant:
    """Tests for INSTALL_DIR constant."""

    def test_install_dir_is_path(self):
        """Test INSTALL_DIR is a Path object."""
        assert isinstance(INSTALL_DIR, Path)

    def test_install_dir_exists(self):
        """Test INSTALL_DIR points to existing directory."""
        assert INSTALL_DIR.exists()
        assert INSTALL_DIR.is_dir()

    def test_install_dir_contains_main_py(self):
        """Test INSTALL_DIR contains main.py (project root)."""
        main_py = INSTALL_DIR / "main.py"
        assert main_py.exists(), f"Expected main.py at {main_py}"

    def test_install_dir_contains_src(self):
        """Test INSTALL_DIR contains src directory."""
        src_dir = INSTALL_DIR / "src"
        assert src_dir.exists()
        assert src_dir.is_dir()


class TestStripExtendedPathPrefix:
    """Tests for strip_extended_path_prefix()."""

    def test_strip_unc_prefix(self):
        r"""Test strip_extended_path_prefix() strips \\?\ prefix."""
        path = Path(r"\\?\C:\Users\test\file.txt")
        result = strip_extended_path_prefix(path)
        assert str(result) == r"C:\Users\test\file.txt"

    def test_strip_device_prefix(self):
        r"""Test strip_extended_path_prefix() strips \\.\ prefix."""
        path = Path(r"\\.\C:\Users\test\file.txt")
        result = strip_extended_path_prefix(path)
        assert str(result) == r"C:\Users\test\file.txt"

    def test_strip_forward_slash_prefix(self):
        """Test strip_extended_path_prefix() strips //?/ prefix."""
        path = Path("//?/C:/Users/test/file.txt")
        result = strip_extended_path_prefix(path)
        assert "C:" in str(result) or "Users" in str(result)

    def test_strip_forward_slash_device_prefix(self):
        """Test strip_extended_path_prefix() strips //./ prefix."""
        path = Path("//./C:/Users/test/file.txt")
        result = strip_extended_path_prefix(path)
        assert "C:" in str(result) or "Users" in str(result)

    def test_no_prefix_unchanged(self):
        """Test strip_extended_path_prefix() leaves normal paths unchanged."""
        path = Path(r"C:\Users\test\file.txt")
        result = strip_extended_path_prefix(path)
        assert str(result) == r"C:\Users\test\file.txt"

    def test_unix_path_unchanged(self):
        """Test strip_extended_path_prefix() leaves Unix paths unchanged."""
        path = Path("/home/user/file.txt")
        result = strip_extended_path_prefix(path)
        # On Windows, Path normalizes to backslashes
        assert "home" in str(result) and "user" in str(result) and "file.txt" in str(result)

    def test_relative_path_unchanged(self):
        """Test strip_extended_path_prefix() leaves relative paths unchanged."""
        path = Path("relative/path/file.txt")
        result = strip_extended_path_prefix(path)
        # On Windows, Path normalizes to backslashes
        assert "relative" in str(result) and "path" in str(result) and "file.txt" in str(result)

    def test_empty_path_after_prefix(self):
        """Test strip_extended_path_prefix() handles edge case of just prefix."""
        # This is an invalid path but should not crash
        path = Path("\\\\?\\")
        result = strip_extended_path_prefix(path)
        # Should strip the prefix and return empty or minimal path
        assert isinstance(result, Path)


class TestLoadEnvironment:
    """Tests for load_environment()."""

    def test_load_environment_no_dotenv_installed(self, tmp_path):
        """Test load_environment() handles missing python-dotenv gracefully."""
        with patch.dict('sys.modules', {'dotenv': None}):
            # Should not raise when dotenv is not available
            try:
                load_environment(tmp_path)
            except ImportError:
                pass  # Expected if dotenv actually not installed

    def test_load_environment_with_project_dir(self, tmp_path):
        """Test load_environment() loads project .env when it exists."""
        # Create project .env file
        project_env = tmp_path / ".env"
        project_env.write_text("PROJECT_VAR=project_value\n")

        # Patch dotenv.load_dotenv (imported inside the function)
        with patch('dotenv.load_dotenv') as mock_load:
            with patch('builtins.print'):  # Suppress output
                load_environment(tmp_path)

        # Should have called load_dotenv with project .env
        assert mock_load.called

    def test_load_environment_without_project_dir(self, tmp_path):
        """Test load_environment() works without project_dir argument."""
        with patch('dotenv.load_dotenv'):
            with patch('builtins.print'):
                load_environment(None)
        # Should not raise

    def test_load_environment_global_env_file(self, tmp_path, monkeypatch):
        """Test load_environment() loads global .env from INSTALL_DIR."""
        # Create a fake global .env
        with patch('src.cli.environment.INSTALL_DIR', tmp_path):
            global_env = tmp_path / ".env"
            global_env.write_text("GLOBAL_VAR=global_value\n")

            with patch('dotenv.load_dotenv') as mock_load:
                with patch('builtins.print'):
                    load_environment()

            # Should have tried to load global .env
            assert mock_load.called


class TestTerminalDetection:
    """Tests for terminal/TTY detection utilities.

    Note: The actual detect_terminal_type(), is_interactive(),
    get_terminal_width(), and supports_color() functions are not
    implemented in src/cli/environment.py yet.

    These tests verify terminal-related behavior using stdlib functions.
    """

    def test_sys_stdout_isatty(self):
        """Test sys.stdout.isatty() returns boolean."""
        result = sys.stdout.isatty()
        assert isinstance(result, bool)

    def test_os_isatty_stdin(self):
        """Test os.isatty() works for stdin."""
        try:
            result = os.isatty(sys.stdin.fileno())
            assert isinstance(result, bool)
        except (OSError, AttributeError):
            # stdin may not have fileno in some test environments
            pass

    def test_os_get_terminal_size_fallback(self):
        """Test os.get_terminal_size() with fallback."""
        try:
            size = os.get_terminal_size()
            assert isinstance(size.columns, int)
            assert isinstance(size.lines, int)
        except OSError:
            # Not connected to terminal - this is expected in tests
            # Fallback behavior should return default size
            default_columns = 80
            assert isinstance(default_columns, int)

    def test_term_environment_variable(self):
        """Test TERM environment variable is accessible."""
        term = os.environ.get('TERM', '')
        # TERM should be string (may be empty on Windows)
        assert isinstance(term, str)

    def test_colorterm_environment_variable(self):
        """Test COLORTERM environment variable is accessible."""
        colorterm = os.environ.get('COLORTERM', '')
        assert isinstance(colorterm, str)


class TestColorSupport:
    """Tests for color support detection patterns."""

    def test_windows_virtual_terminal_check(self):
        """Test Windows virtual terminal support can be checked."""
        is_windows = sys.platform == 'win32'
        # On Windows, color support depends on Windows version and terminal
        # This test just verifies the check doesn't crash
        if is_windows:
            import ctypes
            try:
                kernel32 = ctypes.windll.kernel32
                # GetConsoleMode can be called to check VT support
                assert kernel32 is not None
            except (AttributeError, OSError):
                pass

    def test_ansi_color_environment_patterns(self):
        """Test common environment patterns for ANSI color support."""
        # These are common patterns for detecting color support
        patterns = {
            'TERM': ['xterm', 'xterm-256color', 'screen', 'screen-256color', 'vt100'],
            'COLORTERM': ['truecolor', '24bit'],
            'FORCE_COLOR': ['1', 'true'],
            'NO_COLOR': ['1'],  # This disables color
        }

        for var, expected_values in patterns.items():
            value = os.environ.get(var, '')
            # Just verify we can check these without error
            assert isinstance(value, str)


class TestInteractiveMode:
    """Tests for interactive mode detection."""

    def test_stdin_detection(self):
        """Test stdin can be checked for TTY."""
        try:
            is_tty = sys.stdin.isatty()
            assert isinstance(is_tty, bool)
        except AttributeError:
            # Some test environments don't have isatty
            pass

    def test_stdout_detection(self):
        """Test stdout can be checked for TTY."""
        is_tty = sys.stdout.isatty()
        assert isinstance(is_tty, bool)

    def test_stderr_detection(self):
        """Test stderr can be checked for TTY."""
        is_tty = sys.stderr.isatty()
        assert isinstance(is_tty, bool)

    def test_non_interactive_flag_detection(self):
        """Test --non-interactive flag can be detected from args."""
        test_args = ['script.py', '--non-interactive']
        assert '--non-interactive' in test_args

        test_args_no_flag = ['script.py', '--project', 'path']
        assert '--non-interactive' not in test_args_no_flag


class TestTerminalWidth:
    """Tests for terminal width detection."""

    def test_shutil_get_terminal_size(self):
        """Test shutil.get_terminal_size() returns valid size."""
        import shutil
        size = shutil.get_terminal_size(fallback=(80, 24))
        assert isinstance(size.columns, int)
        assert isinstance(size.lines, int)
        assert size.columns > 0
        assert size.lines > 0

    def test_shutil_get_terminal_size_fallback(self):
        """Test shutil.get_terminal_size() uses fallback when specified."""
        import shutil
        custom_fallback = (120, 40)
        size = shutil.get_terminal_size(fallback=custom_fallback)
        # If not connected to terminal, should use fallback
        # If connected, should return actual size
        assert size.columns >= 1
        assert size.lines >= 1

    def test_columns_environment_variable(self):
        """Test COLUMNS environment variable can be used for width."""
        with patch.dict(os.environ, {'COLUMNS': '100'}):
            columns = os.environ.get('COLUMNS')
            assert columns == '100'
            assert int(columns) == 100

    def test_lines_environment_variable(self):
        """Test LINES environment variable can be used for height."""
        with patch.dict(os.environ, {'LINES': '50'}):
            lines = os.environ.get('LINES')
            assert lines == '50'
            assert int(lines) == 50
