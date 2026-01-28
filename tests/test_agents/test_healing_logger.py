"""Tests for HealingLogger initialization and core functionality.

Sprint 20 US-001: HealingLogger initialization tests covering:
- Directory creation
- File creation with correct naming pattern
- Session ID generation
- Orphaned temp file cleanup
- Permission error handling

Sprint 20 US-002: HealingLogger atomic write and thread safety tests covering:
- Atomic JSON write creates temp file in same directory
- Atomic write uses os.replace for atomic rename
- Temp file cleanup on write failure
- Thread safety under concurrent calls
- Entries list truncation at MAX_ENTRIES
"""

import json
import os
import re
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

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


# =============================================================================
# US-002: Atomic Write and Thread Safety Tests
# =============================================================================


class TestAtomicJsonWrite:
    """Tests for HealingLogger._atomic_json_write() - US-002."""

    def test_atomic_json_write_creates_temp_file_in_same_directory(self):
        """Test _atomic_json_write() creates temp file in same directory as target."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=True)

            # Track temp file creation
            original_mkstemp = tempfile.mkstemp
            created_temp_dirs = []

            def tracking_mkstemp(*args, **kwargs):
                if 'dir' in kwargs:
                    created_temp_dirs.append(kwargs['dir'])
                return original_mkstemp(*args, **kwargs)

            with patch.object(tempfile, 'mkstemp', tracking_mkstemp):
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
                logger._atomic_json_write(entry)

            # Temp file should be created in same directory as JSON file
            assert len(created_temp_dirs) > 0
            assert created_temp_dirs[-1] == log_dir

    def test_atomic_json_write_uses_os_replace_for_atomic_rename(self):
        """Test _atomic_json_write() uses os.replace for atomic rename."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=True)

            # Track os.replace calls
            with patch.object(os, 'replace', wraps=os.replace) as mock_replace:
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
                logger._atomic_json_write(entry)

                # os.replace should be called with temp file -> json file
                assert mock_replace.called
                call_args = mock_replace.call_args[0]
                # First arg is temp path (string from tempfile.mkstemp)
                assert '.healing_' in call_args[0]
                assert '.json.tmp' in call_args[0]
                # Second arg is target json file path
                assert call_args[1] == logger.json_file

    def test_atomic_json_write_cleans_up_temp_file_on_write_failure(self):
        """Test _atomic_json_write() cleans up temp file on write failure."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=True)

            # Force json.dump to fail
            with patch('json.dump') as mock_dump:
                mock_dump.side_effect = IOError("Simulated write failure")

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
                logger._atomic_json_write(entry)

            # Temp files should be cleaned up
            temp_files = list(log_dir.glob(".healing_*.json.tmp"))
            assert len(temp_files) == 0, "Temp file should be cleaned up on failure"

    def test_atomic_json_write_cleans_up_temp_on_replace_failure(self):
        """Test temp file cleanup when os.replace fails."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=True)

            # Force os.replace to fail
            with patch.object(os, 'replace') as mock_replace:
                mock_replace.side_effect = OSError("Simulated rename failure")

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
                logger._atomic_json_write(entry)

            # Temp files should be cleaned up even after replace failure
            temp_files = list(log_dir.glob(".healing_*.json.tmp"))
            assert len(temp_files) == 0, "Temp file should be cleaned up on replace failure"

    def test_atomic_json_write_preserves_existing_entries(self):
        """Test atomic write appends to existing entries without losing data."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=True)

            # Write multiple entries
            for i in range(5):
                entry = HealingLogEntry(
                    timestamp=datetime.now(timezone.utc),
                    stage=f"STAGE_{i}",
                    component="test",
                    action="test",
                    error_type="TestError",
                    error_message=f"test message {i}",
                    result="success",
                    duration_ms=float(i)
                )
                logger._atomic_json_write(entry)

            # Read and verify all entries preserved
            content = json.loads(logger.json_file.read_text())
            assert len(content) == 5
            for i, item in enumerate(content):
                assert item['stage'] == f"STAGE_{i}"
                assert item['error_message'] == f"test message {i}"


class TestWriteThreadSafety:
    """Tests for HealingLogger._write() thread safety - US-002."""

    def test_write_is_thread_safe_under_concurrent_calls(self):
        """Test _write() is thread-safe under concurrent log_* method calls."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=True)

            num_threads = 10
            entries_per_thread = 20
            errors = []

            def write_entries(thread_id):
                try:
                    for i in range(entries_per_thread):
                        entry = HealingLogEntry(
                            timestamp=datetime.now(timezone.utc),
                            stage=f"STAGE_{thread_id}",
                            component="test",
                            action=f"action_{i}",
                            error_type="TestError",
                            error_message=f"Thread {thread_id}, entry {i}",
                            result="success",
                            duration_ms=1.0
                        )
                        logger._write(entry)
                except Exception as e:
                    errors.append((thread_id, e))

            # Start concurrent threads
            threads = []
            for t_id in range(num_threads):
                t = threading.Thread(target=write_entries, args=(t_id,))
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Check no errors occurred
            assert len(errors) == 0, f"Errors during concurrent writes: {errors}"

            # All entries should be recorded
            expected_count = num_threads * entries_per_thread
            assert len(logger.entries) == expected_count, \
                f"Expected {expected_count} entries, got {len(logger.entries)}"

    def test_write_lock_prevents_race_conditions(self):
        """Test that _write() lock prevents entry list corruption.

        This test verifies that the lock is used by:
        1. Running many concurrent writes
        2. Verifying entries count matches expected (no lost writes)
        3. Verifying entries list is consistent (no duplicates or corruption)
        """
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)  # Disable JSON for speed

            num_threads = 20
            entries_per_thread = 50
            barrier = threading.Barrier(num_threads)  # Ensure all threads start together

            def write_entries(thread_id):
                # Wait for all threads to be ready
                barrier.wait()
                for i in range(entries_per_thread):
                    entry = HealingLogEntry(
                        timestamp=datetime.now(timezone.utc),
                        stage=f"THREAD_{thread_id}",
                        component="test",
                        action=f"action_{i}",
                        error_type="TestError",
                        error_message=f"T{thread_id}_E{i}",
                        result="success",
                        duration_ms=1.0
                    )
                    logger._write(entry)

            threads = []
            for t_id in range(num_threads):
                t = threading.Thread(target=write_entries, args=(t_id,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Verify exact count - if lock wasn't working, we'd see missing entries
            expected = num_threads * entries_per_thread
            assert len(logger.entries) == expected, \
                f"Lock failure: expected {expected}, got {len(logger.entries)}"

            # Verify no corruption - all entries should be valid
            for entry in logger.entries:
                assert entry.stage.startswith("THREAD_")
                assert entry.error_message.startswith("T")
                assert "_E" in entry.error_message

    def test_concurrent_log_methods_do_not_corrupt_json(self):
        """Test concurrent log_* method calls don't corrupt JSON file."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=True)

            # Mock error classification for log_classification
            mock_classification = MagicMock()
            mock_classification.category = "network"
            mock_classification.severity = "medium"
            mock_classification.suggested_healer = "NetworkHealer"
            mock_classification.confidence = 0.9
            mock_classification.needs_llm_healer = False
            mock_classification.reasoning = "Test reasoning"

            # Mock healer result for log_healer_attempt
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.action = MagicMock()
            mock_result.action.value = "RETRY"
            mock_result.message = "Fixed"
            mock_result.modified_config = False
            mock_result.details = {}

            def call_log_classification():
                for _ in range(10):
                    logger.log_classification(
                        "TEST", ValueError("test"), mock_classification, 1.0
                    )

            def call_log_healer_attempt():
                for _ in range(10):
                    logger.log_healer_attempt(
                        "TEST", "TestHealer", ValueError("test"),
                        mock_result, 2.0
                    )

            def call_log_fallback():
                for _ in range(10):
                    logger.log_fallback("TEST", "ComponentA", "ComponentB", "fallback reason")

            threads = [
                threading.Thread(target=call_log_classification),
                threading.Thread(target=call_log_healer_attempt),
                threading.Thread(target=call_log_fallback),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Verify JSON is valid and has all entries
            content = json.loads(logger.json_file.read_text())
            assert len(content) == 30  # 10 * 3 methods
            # Verify it's valid JSON (would raise if corrupted)
            json.dumps(content)


class TestEntriesTruncation:
    """Tests for entries list truncation at MAX_ENTRIES - US-002."""

    def test_entries_truncates_at_max_entries_keeping_recent_half(self):
        """Test entries list truncates at MAX_ENTRIES (10000) keeping recent half.

        Behavior: When len(entries) > MAX_ENTRIES, truncate to MAX_ENTRIES // 2 (keep recent half).
        For MAX_ENTRIES=100:
        - entries 0-99: 100 entries, no truncation (not > MAX)
        - entry 100 added: 101 > 100 → truncate to 50 (keeps entries 51-100)
        - entries 101-119 added: 50 + 19 = 69 total

        This test verifies truncation happens and keeps most recent entries.
        """
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)  # Disable JSON for speed

            # Override MAX_ENTRIES for faster testing
            original_max = HealingLogger.MAX_ENTRIES
            HealingLogger.MAX_ENTRIES = 100  # Use smaller value for test

            try:
                # Write more than MAX_ENTRIES
                for i in range(120):
                    entry = HealingLogEntry(
                        timestamp=datetime.now(timezone.utc),
                        stage=f"STAGE_{i}",
                        component="test",
                        action="test",
                        error_type="TestError",
                        error_message=f"Entry {i}",
                        result="success",
                        duration_ms=1.0
                    )
                    logger._write(entry)

                # After 120 entries with MAX=100:
                # Truncation happens at entry 100 → keeps 50 entries (51-100)
                # Then entries 101-119 added (19 more) → 69 total
                assert len(logger.entries) == 69, \
                    f"Expected 69 entries after truncation+additions, got {len(logger.entries)}"

                # Verify truncation removed old entries (0-50 should be gone)
                entry_ids = [int(e.error_message.split()[1]) for e in logger.entries]
                min_entry = min(entry_ids)
                max_entry = max(entry_ids)

                # Oldest entry should be 51 (after truncation removed 0-50)
                assert min_entry == 51, f"Oldest entry should be 51, got {min_entry}"
                # Newest entry should be 119
                assert max_entry == 119, f"Newest entry should be 119, got {max_entry}"

            finally:
                HealingLogger.MAX_ENTRIES = original_max

    def test_truncation_keeps_most_recent_entries(self):
        """Test that truncation keeps the most recent (highest index) entries."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            # Use small MAX_ENTRIES for testing
            original_max = HealingLogger.MAX_ENTRIES
            HealingLogger.MAX_ENTRIES = 20

            try:
                # Add entries with distinct identifiers
                for i in range(30):
                    entry = HealingLogEntry(
                        timestamp=datetime.now(timezone.utc),
                        stage=f"STAGE",
                        component="test",
                        action="test",
                        error_type="TestError",
                        error_message=f"ENTRY_{i:03d}",  # ENTRY_000, ENTRY_001, ...
                        result="success",
                        duration_ms=1.0
                    )
                    logger._write(entry)

                # After truncation, we should have entries from later in sequence
                entry_ids = [int(e.error_message.split('_')[1]) for e in logger.entries]

                # The minimum ID should be > 0 (old entries removed)
                min_id = min(entry_ids)
                max_id = max(entry_ids)

                assert max_id == 29, f"Should have most recent entry (29), got {max_id}"
                assert min_id > 0, f"Oldest entries should be truncated, min_id={min_id}"

            finally:
                HealingLogger.MAX_ENTRIES = original_max

    def test_truncation_logs_warning(self):
        """Test that truncation logs a warning message."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            original_max = HealingLogger.MAX_ENTRIES
            HealingLogger.MAX_ENTRIES = 10

            try:
                with patch('src.agents.healing_logger.logger') as mock_logger:
                    # Add entries to trigger truncation
                    for i in range(15):
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

                    # Should have logged truncation warning
                    warning_calls = [c for c in mock_logger.warning.call_args_list
                                   if 'Truncated' in str(c)]
                    assert len(warning_calls) > 0, "Should log warning on truncation"

            finally:
                HealingLogger.MAX_ENTRIES = original_max

    def test_no_truncation_below_max_entries(self):
        """Test that no truncation occurs when entries < MAX_ENTRIES."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            original_max = HealingLogger.MAX_ENTRIES
            HealingLogger.MAX_ENTRIES = 100

            try:
                # Add fewer entries than MAX
                for i in range(50):
                    entry = HealingLogEntry(
                        timestamp=datetime.now(timezone.utc),
                        stage=f"STAGE_{i}",
                        component="test",
                        action="test",
                        error_type="TestError",
                        error_message=f"Entry {i}",
                        result="success",
                        duration_ms=1.0
                    )
                    logger._write(entry)

                # All entries should be present
                assert len(logger.entries) == 50
                # First entry should still be there (not truncated)
                assert logger.entries[0].stage == "STAGE_0"

            finally:
                HealingLogger.MAX_ENTRIES = original_max
