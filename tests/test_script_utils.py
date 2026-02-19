#!/usr/bin/env python3
"""
Tests for scripts/script_utils.py

Tests the common utility functions including:
- Verbosity level management
- print_header: section headers
- print_ok: success messages
- print_warn: warning messages
- print_error: error messages
- print_info: info messages
- File operation utilities
"""

import os
import sys
import tempfile
import shutil
import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import the module under test
from script_utils import (
    set_verbosity,
    get_verbosity,
    print_header,
    print_ok,
    print_warn,
    print_error,
    print_info,
    safe_read_file,
    safe_write_file,
    ensure_directory,
    find_files,
    get_file_hash,
)


@pytest.mark.script
class TestVerbosity:
    """Test verbosity level management"""

    def test_default_verbosity_is_one(self):
        """Default verbosity should be 1 (normal)"""
        # Reset to default first
        set_verbosity(1)
        assert get_verbosity() == 1

    def test_set_verbosity_clamp_to_zero(self):
        """Setting verbosity below 0 clamps to 0"""
        set_verbosity(-5)
        assert get_verbosity() == 0

    def test_set_verbosity_clamp_to_two(self):
        """Setting verbosity above 2 clamps to 2"""
        set_verbosity(10)
        assert get_verbosity() == 2

    def test_set_verbosity_valid_range(self):
        """Setting verbosity within range keeps it"""
        set_verbosity(0)
        assert get_verbosity() == 0

        set_verbosity(1)
        assert get_verbosity() == 1

        set_verbosity(2)
        assert get_verbosity() == 2


@pytest.mark.script
class TestPrintHeader:
    """Test print_header function"""

    def test_prints_header_at_normal_verbosity(self):
        """Header is printed at normal verbosity (1)"""
        set_verbosity(1)
        output = StringIO()

        with patch('sys.stdout', output):
            print_header("Test Header")

        result = output.getvalue()
        assert "Test Header" in result
        assert "=" in result

    def test_prints_header_at_verbose_verbosity(self):
        """Header is printed at verbose verbosity (2)"""
        set_verbosity(2)
        output = StringIO()

        with patch('sys.stdout', output):
            print_header("Verbose Header")

        result = output.getvalue()
        assert "Verbose Header" in result

    def test_suppresses_header_at_quiet_verbosity(self):
        """Header is suppressed at quiet verbosity (0)"""
        set_verbosity(0)
        output = StringIO()

        with patch('sys.stdout', output):
            print_header("Quiet Header")

        result = output.getvalue()
        assert "Quiet Header" not in result


@pytest.mark.script
class TestPrintOk:
    """Test print_ok function"""

    def test_prints_ok_message_at_normal_verbosity(self):
        """OK message is printed at normal verbosity (1)"""
        set_verbosity(1)
        output = StringIO()

        with patch('sys.stdout', output):
            print_ok("Operation completed")

        result = output.getvalue()
        assert "[OK]" in result
        assert "Operation completed" in result

    def test_prints_ok_message_at_verbose_verbosity(self):
        """OK message is printed at verbose verbosity (2)"""
        set_verbosity(2)
        output = StringIO()

        with patch('sys.stdout', output):
            print_ok("Verbose success")

        result = output.getvalue()
        assert "[OK]" in result

    def test_suppresses_ok_at_quiet_verbosity(self):
        """OK message is suppressed at quiet verbosity (0)"""
        set_verbosity(0)
        output = StringIO()

        with patch('sys.stdout', output):
            print_ok("Quiet success")

        result = output.getvalue()
        assert "Quiet success" not in result


@pytest.mark.script
class TestPrintWarn:
    """Test print_warn function"""

    def test_prints_warning_at_normal_verbosity(self):
        """Warning is printed at normal verbosity (1)"""
        set_verbosity(1)
        output = StringIO()

        with patch('sys.stdout', output):
            print_warn("This is a warning")

        result = output.getvalue()
        assert "[WARN]" in result
        assert "This is a warning" in result

    def test_prints_warning_at_verbose_verbosity(self):
        """Warning is printed at verbose verbosity (2)"""
        set_verbosity(2)
        output = StringIO()

        with patch('sys.stdout', output):
            print_warn("Verbose warning")

        result = output.getvalue()
        assert "[WARN]" in result

    def test_suppresses_warning_at_quiet_verbosity(self):
        """Warning is suppressed at quiet verbosity (0)"""
        set_verbosity(0)
        output = StringIO()

        with patch('sys.stdout', output):
            print_warn("Quiet warning")

        result = output.getvalue()
        assert "Quiet warning" not in result


@pytest.mark.script
class TestPrintError:
    """Test print_error function"""

    def test_prints_error_without_exit(self):
        """Error message printed without exiting when exit_code is None"""
        set_verbosity(1)
        output = StringIO()

        with patch('sys.stdout', output):
            print_error("Error occurred", exit_code=None)

        result = output.getvalue()
        assert "[ERROR]" in result
        assert "Error occurred" in result

    def test_prints_error_with_exit_code(self):
        """Error message printed and exit called with code"""
        set_verbosity(1)
        output = StringIO()

        with patch('sys.stdout', output), \
             patch('sys.exit') as mock_exit:

            print_error("Fatal error", exit_code=1)

        result = output.getvalue()
        assert "[ERROR]" in result
        assert "Fatal error" in result
        mock_exit.assert_called_once_with(1)

    def test_error_always_shown_at_quiet_verbosity(self):
        """Error is shown even at quiet verbosity"""
        set_verbosity(0)
        output = StringIO()

        with patch('sys.stdout', output), \
             patch('sys.exit'):

            print_error("Critical error")

        result = output.getvalue()
        assert "[ERROR]" in result


@pytest.mark.script
class TestPrintInfo:
    """Test print_info function"""

    def test_suppresses_info_at_normal_verbosity(self):
        """Info is suppressed at normal verbosity (1)"""
        set_verbosity(1)
        output = StringIO()

        with patch('sys.stdout', output):
            print_info("Some info")

        result = output.getvalue()
        assert "Some info" not in result

    def test_prints_info_at_verbose_verbosity(self):
        """Info is printed at verbose verbosity (2)"""
        set_verbosity(2)
        output = StringIO()

        with patch('sys.stdout', output):
            print_info("Detailed info")

        result = output.getvalue()
        assert "[INFO]" in result
        assert "Detailed info" in result

    def test_suppresses_info_at_quiet_verbosity(self):
        """Info is suppressed at quiet verbosity (0)"""
        set_verbosity(0)
        output = StringIO()

        with patch('sys.stdout', output):
            print_info("Quiet info")

        result = output.getvalue()
        assert "Quiet info" not in result


@pytest.mark.script
class TestOutputFormat:
    """Test output formatting"""

    def test_format_uses_two_space_indent(self):
        """Output should use 2-space indent"""
        set_verbosity(1)
        output = StringIO()

        with patch('sys.stdout', output):
            print_ok("Test message")

        result = output.getvalue()
        # Should start with "  " (2 spaces) before [OK]
        assert result.startswith("  ")

    def test_no_emoji_in_output(self):
        """Output should not contain emoji"""
        set_verbosity(1)
        output = StringIO()

        with patch('sys.stdout', output):
            print_ok("Test")
            print_warn("Test")
            print_error("Test")
            print_info("Test")

        result = output.getvalue()
        # Check for common emoji patterns - should not have any
        assert "😀" not in result
        assert "✅" not in result
        assert "⚠️" not in result
        assert "❌" not in result


@pytest.mark.script
class TestVerbosityBehavior:
    """Test complete verbosity behavior"""

    @pytest.mark.fast
    def test_quiet_mode_shows_only_errors(self):
        """At verbosity 0, only errors are shown"""
        set_verbosity(0)
        output = StringIO()

        with patch('sys.stdout', output):
            print_header("Header")
            print_ok("Success")
            print_warn("Warning")
            print_error("Error")
            print_info("Info")

        result = output.getvalue()
        assert "[ERROR]" in result
        assert "Header" not in result
        assert "[OK]" not in result
        assert "[WARN]" not in result
        assert "[INFO]" not in result

    @pytest.mark.fast
    def test_normal_mode_shows_header_ok_warn(self):
        """At verbosity 1, header, ok, and warn are shown"""
        set_verbosity(1)
        output = StringIO()

        with patch('sys.stdout', output):
            print_header("Header")
            print_ok("Success")
            print_warn("Warning")
            print_error("Error")
            print_info("Info")

        result = output.getvalue()
        assert "Header" in result
        assert "[OK]" in result
        assert "[WARN]" in result
        assert "[ERROR]" in result
        assert "[INFO]" not in result

    @pytest.mark.fast
    def test_verbose_mode_shows_everything(self):
        """At verbosity 2, all messages are shown"""
        set_verbosity(2)
        output = StringIO()

        with patch('sys.stdout', output):
            print_header("Header")
            print_ok("Success")
            print_warn("Warning")
            print_error("Error")
            print_info("Info")

        result = output.getvalue()
        assert "Header" in result
        assert "[OK]" in result
        assert "[WARN]" in result
        assert "[ERROR]" in result
        assert "[INFO]" in result


@pytest.mark.script
class TestCliHelpersFormatSize:
    """Test format_size from cli_helpers (imported by script_utils users)"""

    def test_format_size_bytes(self):
        """Test bytes formatting"""
        from utils.cli_helpers import format_size
        assert format_size(0) == "0.0 B"
        assert format_size(1) == "1.0 B"
        assert format_size(512) == "512.0 B"

    def test_format_size_kilobytes(self):
        """Test kilobytes formatting"""
        from utils.cli_helpers import format_size
        assert format_size(1024) == "1.0 KB"
        assert format_size(1536) == "1.5 KB"

    def test_format_size_megabytes(self):
        """Test megabytes formatting"""
        from utils.cli_helpers import format_size
        assert format_size(1048576) == "1.0 MB"
        assert format_size(1572864) == "1.5 MB"

    def test_format_size_gigabytes(self):
        """Test gigabytes formatting"""
        from utils.cli_helpers import format_size
        assert format_size(1073741824) == "1.0 GB"


@pytest.mark.script
class TestImportGuidelines:
    """Test import pattern documentation is correct"""

    def test_import_pattern_documented(self):
        """Verify import guidelines are documented in module"""
        # Read the module file to verify documentation
        module_path = scripts_dir / "script_utils.py"
        content = module_path.read_text()

        # Check key import guidelines are documented
        assert "IMPORT GUIDELINES" in content
        assert "from script_utils import" in content
        assert "sys.path.insert" in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# FILE OPERATION UTILITY TESTS
# =============================================================================

@pytest.mark.script
class TestSafeReadFile:
    """Test safe_read_file function"""

    def test_read_text_file(self, tmp_path):
        """Read a text file with default UTF-8 encoding"""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content, encoding='utf-8')

        result = safe_read_file(test_file)
        assert result == test_content

    def test_read_text_file_with_path_string(self, tmp_path):
        """Read file using string path"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        result = safe_read_file(str(test_file))
        assert result == "Test content"

    def test_read_binary_file(self, tmp_path):
        """Read a binary file"""
        test_file = tmp_path / "test.bin"
        test_content = b"\x00\x01\x02\x03"
        test_file.write_bytes(test_content)

        result = safe_read_file(test_file, binary=True)
        assert result == test_content
        assert isinstance(result, bytes)

    def test_read_nonexistent_file_exits(self, tmp_path):
        """Reading nonexistent file should exit with error"""
        nonexistent = tmp_path / "nonexistent.txt"

        with patch('sys.exit') as mock_exit:
            safe_read_file(nonexistent)
            mock_exit.assert_called_once_with(1)

    def test_read_with_custom_encoding(self, tmp_path):
        """Read file with custom encoding"""
        test_file = tmp_path / "test.txt"
        # Write with latin-1 encoding
        test_file.write_bytes("café".encode('latin-1'))

        result = safe_read_file(test_file, encoding='latin-1')
        assert result == "café"


@pytest.mark.script
class TestSafeWriteFile:
    """Test safe_write_file function"""

    def test_write_text_file(self, tmp_path):
        """Write text to a file"""
        test_file = tmp_path / "output.txt"
        content = "Hello, World!"

        result = safe_write_file(test_file, content)
        assert result is True
        assert test_file.read_text() == content

    def test_write_binary_file(self, tmp_path):
        """Write binary data to a file"""
        test_file = tmp_path / "output.bin"
        content = b"\x00\x01\x02\x03"

        result = safe_write_file(test_file, content, atomic=False)
        assert result is True
        assert test_file.read_bytes() == content

    def test_write_creates_parent_directories(self, tmp_path):
        """Write should create parent directories"""
        test_file = tmp_path / "subdir" / "nested" / "output.txt"

        result = safe_write_file(test_file, "content")
        assert result is True
        assert test_file.exists()

    def test_write_atomic_creates_temp_file(self, tmp_path):
        """Atomic write should use temp file"""
        test_file = tmp_path / "output.txt"

        result = safe_write_file(test_file, "content", atomic=True)
        assert result is True
        # Check no temp files left behind
        temp_files = list(tmp_path.glob('.tmp_*'))
        assert len(temp_files) == 0


@pytest.mark.script
class TestEnsureDirectory:
    """Test ensure_directory function"""

    def test_create_new_directory(self, tmp_path):
        """Create a new directory"""
        new_dir = tmp_path / "new_dir"

        result = ensure_directory(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_existing_directory_returns_path(self, tmp_path):
        """Existing directory should be returned without error"""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensure_directory(existing_dir)
        assert result == existing_dir

    def test_create_nested_directories(self, tmp_path):
        """Create nested directory structure"""
        nested = tmp_path / "a" / "b" / "c"

        result = ensure_directory(nested)
        assert result.exists()
        assert result.is_dir()


@pytest.mark.script
class TestFindFiles:
    """Test find_files function"""

    def test_find_all_files(self, tmp_path):
        """Find all files with default pattern"""
        # Create test files
        (tmp_path / "file1.txt").write_text("test")
        (tmp_path / "file2.txt").write_text("test")
        (tmp_path / "file3.py").write_text("test")

        files = find_files(tmp_path, "*", recursive=False)
        assert len(files) == 3

    def test_find_with_extension_filter(self, tmp_path):
        """Find files with specific extension"""
        (tmp_path / "file1.txt").write_text("test")
        (tmp_path / "file2.txt").write_text("test")
        (tmp_path / "file3.py").write_text("test")

        txt_files = find_files(tmp_path, "*.txt", recursive=False)
        assert len(txt_files) == 2
        assert all(f.suffix == '.txt' for f in txt_files)

    def test_find_recursive(self, tmp_path):
        """Find files recursively"""
        # Create nested structure
        (tmp_path / "file1.txt").write_text("test")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file2.txt").write_text("test")

        files = find_files(tmp_path, "*.txt", recursive=True)
        assert len(files) == 2

    def test_find_excludes_hidden_files_by_default(self, tmp_path):
        """Hidden files should be excluded by default"""
        (tmp_path / "visible.txt").write_text("test")
        (tmp_path / ".hidden.txt").write_text("test")

        files = find_files(tmp_path, "*.txt", recursive=False)
        assert len(files) == 1
        assert files[0].name == "visible.txt"

    def test_find_nonexistent_directory(self, tmp_path):
        """Find in nonexistent directory returns empty list"""
        nonexistent = tmp_path / "nonexistent"
        files = find_files(nonexistent, "*.txt")
        assert files == []


@pytest.mark.script
class TestGetFileHash:
    """Test get_file_hash function"""

    def test_sha256_hash(self, tmp_path):
        """Calculate SHA256 hash of a file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_result = get_file_hash(test_file, algorithm='sha256')
        # SHA256 of "Hello, World!"
        assert hash_result == "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"

    def test_md5_hash(self, tmp_path):
        """Calculate MD5 hash of a file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        hash_result = get_file_hash(test_file, algorithm='md5')
        # MD5 of "test"
        assert hash_result == "098f6bcd4621d373cade4e832627b4f6"

    def test_hash_nonexistent_file(self, tmp_path):
        """Hash of nonexistent file returns None"""
        nonexistent = tmp_path / "nonexistent.txt"
        hash_result = get_file_hash(nonexistent)
        assert hash_result is None

    def test_hash_binary_file(self, tmp_path):
        """Hash binary file correctly"""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        hash_result = get_file_hash(test_file, algorithm='sha256')
        assert hash_result is not None
        assert len(hash_result) == 64  # SHA256 produces 64 hex chars


@pytest.mark.script
class TestFileUtilitiesIntegration:
    """Integration tests for file utilities"""

    def test_read_write_roundtrip(self, tmp_path):
        """Test read/write roundtrip"""
        test_file = tmp_path / "roundtrip.txt"
        original_content = "Line 1\nLine 2\nLine 3"

        # Write
        safe_write_file(test_file, original_content)

        # Read
        content = safe_read_file(test_file)

        assert content == original_content

    def test_directory_workflow(self, tmp_path):
        """Test complete directory workflow"""
        work_dir = tmp_path / "workflow"

        # Ensure directory exists
        ensure_directory(work_dir)

        # Write files
        for i in range(3):
            safe_write_file(work_dir / f"file{i}.txt", f"Content {i}")

        # Find files
        files = find_files(work_dir, "*.txt")
        assert len(files) == 3

        # Hash files
        for f in files:
            h = get_file_hash(f)
            assert h is not None

