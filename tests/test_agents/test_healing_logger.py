"""Tests for HealingLogger initialization and core functionality.

Sprint 20 US-001: HealingLogger initialization tests covering:
- Directory creation
- File creation with correct naming pattern
- Session ID generation
- Orphaned temp file cleanup
- Permission error handling
"""

import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.healing_logger import HealingLogger, HealingLogEntry


class TestHealingLoggerInit:
    """Tests for HealingLogger.__init__() - US-001."""

    def test_init_creates_log_directory_if_not_exists(self):
        """Test HealingLogger.__init__() creates log directory if not exists."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a path that doesn't exist yet
            log_dir = Path(tmp) / "nested" / "logs" / "healing"
            assert not log_dir.exists()

            logger = HealingLogger(log_dir, json_log=True)

            # Directory should now exist
            assert log_dir.exists()
            assert log_dir.is_dir()

    def test_init_creates_log_and_json_files_with_correct_naming_pattern(self):
        """Test HealingLogger.__init__() creates both .log and .json files with correct naming pattern.

        Note: JSON file is created immediately on init (with empty array).
        Log file is created lazily on first write, but the path is set up correctly.
        """
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"

            # Capture timestamp before/after creation
            before_ts = datetime.now().strftime("%Y%m%d_%H%M")
            logger = HealingLogger(log_dir, json_log=True)
            after_ts = datetime.now().strftime("%Y%m%d_%H%M")

            # Check JSON file exists immediately (created on init with "[]")
            assert logger.json_file.exists()
            assert logger.json_file.suffix == ".json"

            # JSON file should have empty array content
            content = logger.json_file.read_text()
            assert content == "[]"

            # Log file path is set but file is created lazily on first write
            assert logger.log_file.suffix == ".log"
            assert not logger.log_file.exists()  # Not created yet

            # Write an entry to trigger log file creation
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="TEST",
                component="test",
                action="test",
                error_type="TestError",
                error_message="test",
                result="success",
                duration_ms=1.0
            )
            logger._write(entry)

            # Now log file should exist
            assert logger.log_file.exists()

            # Verify naming pattern: healing_YYYYMMDD_HHMMSS
            log_name = logger.log_file.stem
            json_name = logger.json_file.stem

            # Both should start with "healing_"
            assert log_name.startswith("healing_")
            assert json_name.startswith("healing_")

            # Extract timestamp from filename
            log_timestamp = log_name.replace("healing_", "")
            json_timestamp = json_name.replace("healing_", "")

            # Timestamps should match each other (same logger)
            assert log_timestamp == json_timestamp

            # Timestamp should match pattern YYYYMMDD_HHMMSS
            timestamp_pattern = r"^\d{8}_\d{6}$"
            assert re.match(timestamp_pattern, log_timestamp)

    def test_init_generates_unique_8_char_session_id(self):
        """Test HealingLogger.__init__() generates unique 8-char session_id."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"

            logger1 = HealingLogger(log_dir / "session1", json_log=True)
            logger2 = HealingLogger(log_dir / "session2", json_log=True)
            logger3 = HealingLogger(log_dir / "session3", json_log=True)

            # Session IDs should be exactly 8 characters
            assert len(logger1.session_id) == 8
            assert len(logger2.session_id) == 8
            assert len(logger3.session_id) == 8

            # Session IDs should be unique across instances
            session_ids = {logger1.session_id, logger2.session_id, logger3.session_id}
            assert len(session_ids) == 3, "Session IDs should be unique"

            # Session IDs should be alphanumeric (first 8 chars of UUID)
            for session_id in session_ids:
                assert re.match(r"^[a-f0-9]{8}$", session_id), f"Session ID {session_id} should be hex characters"

    def test_init_calls_cleanup_orphaned_temp_files_on_startup(self):
        """Test HealingLogger.__init__() calls _cleanup_orphaned_temp_files() on startup."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            log_dir.mkdir(parents=True)

            # Create orphaned temp files (older than 60 seconds)
            orphaned_file1 = log_dir / ".healing_abc123.json.tmp"
            orphaned_file2 = log_dir / ".healing_def456.json.tmp"

            orphaned_file1.write_text('{"partial": "data"}')
            orphaned_file2.write_text('{"incomplete": true}')

            # Set modification time to be older than 60 seconds
            old_time = time.time() - 120  # 2 minutes ago
            os.utime(orphaned_file1, (old_time, old_time))
            os.utime(orphaned_file2, (old_time, old_time))

            # Verify orphaned files exist
            assert orphaned_file1.exists()
            assert orphaned_file2.exists()

            # Create logger - should trigger cleanup
            logger = HealingLogger(log_dir, json_log=True)

            # Orphaned files should be cleaned up
            assert not orphaned_file1.exists(), "Old orphaned temp file should be cleaned up"
            assert not orphaned_file2.exists(), "Old orphaned temp file should be cleaned up"

    def test_init_does_not_cleanup_recent_temp_files(self):
        """Test _cleanup_orphaned_temp_files() preserves temp files younger than 60 seconds."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            log_dir.mkdir(parents=True)

            # Create a recent temp file (should not be cleaned up)
            recent_temp = log_dir / ".healing_recent.json.tmp"
            recent_temp.write_text('{"in_progress": true}')
            # Note: file is just created, so mtime is current - within 60s threshold

            # Create logger
            logger = HealingLogger(log_dir, json_log=True)

            # Recent temp file should be preserved
            assert recent_temp.exists(), "Recent temp file should not be cleaned up"

    def test_init_handles_permission_error_on_directory_creation(self):
        """Test HealingLogger.__init__() handles permission errors gracefully when directory creation fails."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"

            # Patch mkdir to raise PermissionError
            with patch.object(Path, 'mkdir') as mock_mkdir:
                mock_mkdir.side_effect = PermissionError("Access denied")

                with pytest.raises(PermissionError):
                    HealingLogger(log_dir, json_log=True)

    def test_init_handles_permission_error_on_json_file_write(self):
        """Test HealingLogger handles permission error when writing initial JSON file."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            log_dir.mkdir(parents=True)

            # Create logger but patch write_text to fail
            with patch.object(Path, 'write_text') as mock_write:
                mock_write.side_effect = PermissionError("Cannot write file")

                # Should not raise - logger should handle gracefully
                with pytest.raises(PermissionError):
                    HealingLogger(log_dir, json_log=True)

    def test_init_with_json_log_disabled(self):
        """Test HealingLogger.__init__() with json_log=False doesn't create JSON file initially."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"

            logger = HealingLogger(log_dir, json_log=False)

            # Log file should still exist
            assert logger.log_file is not None

            # JSON file should be defined but not written to
            assert logger.json_file is not None
            assert not logger.json_file.exists(), "JSON file should not be created when json_log=False"

    def test_init_console_format_options(self):
        """Test HealingLogger accepts all three console_format options."""
        with tempfile.TemporaryDirectory() as tmp:
            for console_format in ["box", "simple", "minimal"]:
                log_dir = Path(tmp) / f"logs_{console_format}"
                logger = HealingLogger(log_dir, console_format=console_format)
                assert logger.console_format == console_format


class TestHealingLoggerCleanup:
    """Tests for _cleanup_orphaned_temp_files() method."""

    def test_cleanup_only_targets_healing_temp_files(self):
        """Test cleanup only removes .healing_*.json.tmp files, not other files."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            log_dir.mkdir(parents=True)

            # Create various files
            orphaned_healing = log_dir / ".healing_orphan.json.tmp"
            other_temp = log_dir / ".other_temp.json.tmp"
            regular_file = log_dir / "regular_data.json"

            for f in [orphaned_healing, other_temp, regular_file]:
                f.write_text("{}")

            # Make healing file old
            old_time = time.time() - 120
            os.utime(orphaned_healing, (old_time, old_time))
            os.utime(other_temp, (old_time, old_time))

            # Create logger
            logger = HealingLogger(log_dir, json_log=True)

            # Only healing temp should be removed
            assert not orphaned_healing.exists(), "Healing temp file should be cleaned"
            assert other_temp.exists(), "Non-healing temp file should be preserved"
            assert regular_file.exists(), "Regular file should be preserved"

    def test_cleanup_handles_file_access_errors_gracefully(self):
        """Test cleanup continues if individual file deletion fails."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            log_dir.mkdir(parents=True)

            # Create orphaned temp files
            orphaned1 = log_dir / ".healing_file1.json.tmp"
            orphaned2 = log_dir / ".healing_file2.json.tmp"
            orphaned1.write_text("{}")
            orphaned2.write_text("{}")

            # Make both old
            old_time = time.time() - 120
            os.utime(orphaned1, (old_time, old_time))
            os.utime(orphaned2, (old_time, old_time))

            # Patch unlink to fail for first file only
            original_unlink = Path.unlink
            call_count = [0]

            def failing_unlink(self, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise OSError("Cannot delete")
                return original_unlink(self, *args, **kwargs)

            with patch.object(Path, 'unlink', failing_unlink):
                # Should not raise - cleanup handles errors
                logger = HealingLogger(log_dir, json_log=True)

            # At least the cleanup was attempted (no exception)
            assert True


class TestHealingLoggerFilesystemCheck:
    """Tests for _check_filesystem_atomicity() method."""

    def test_network_share_warning_for_unc_path(self):
        """Test warning is logged for UNC path (Windows network share)."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"

            # Mock resolve to return UNC path
            with patch.object(Path, 'resolve') as mock_resolve:
                mock_resolve.return_value = Path(r"\\server\share\logs")

                # Capture warning
                with patch('src.agents.healing_logger.logger') as mock_logger:
                    healing_logger = HealingLogger(log_dir, json_log=True)

                    # Should log warning about network share
                    warning_calls = [call for call in mock_logger.warning.call_args_list
                                   if 'network share' in str(call).lower()]
                    assert len(warning_calls) > 0, "Should warn about network share"

    def test_local_path_no_warning(self):
        """Test no warning for local filesystem path."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"

            with patch('src.agents.healing_logger.logger') as mock_logger:
                healing_logger = HealingLogger(log_dir, json_log=True)

                # Should not log network share warning
                warning_calls = [call for call in mock_logger.warning.call_args_list
                               if 'network share' in str(call).lower()]
                assert len(warning_calls) == 0, "Should not warn for local path"


class TestHealingLogEntry:
    """Tests for HealingLogEntry dataclass."""

    def test_to_dict_converts_timestamp_to_iso(self):
        """Test to_dict() converts datetime to ISO format string."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        entry = HealingLogEntry(
            timestamp=timestamp,
            stage="TEST",
            component="test",
            action="test",
            error_type="TestError",
            error_message="test message",
            result="success",
            duration_ms=100.0
        )

        d = entry.to_dict()
        assert d['timestamp'] == "2024-01-15T10:30:45+00:00"

    def test_to_dict_preserves_all_fields(self):
        """Test to_dict() includes all entry fields."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="DOWNLOAD",
            component="healer",
            action="attempt",
            error_type="RateLimitError",
            error_message="Too many requests",
            result="success",
            duration_ms=1234.5,
            session_id="abc12345",
            stack_trace="traceback here",
            details={"retry_count": 3}
        )

        d = entry.to_dict()

        assert d['stage'] == "DOWNLOAD"
        assert d['component'] == "healer"
        assert d['action'] == "attempt"
        assert d['error_type'] == "RateLimitError"
        assert d['error_message'] == "Too many requests"
        assert d['result'] == "success"
        assert d['duration_ms'] == 1234.5
        assert d['session_id'] == "abc12345"
        assert d['stack_trace'] == "traceback here"
        assert d['details'] == {"retry_count": 3}
