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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
    def test_init_handles_permission_error_on_directory_creation(self):
        """Test HealingLogger.__init__() handles permission errors gracefully when directory creation fails."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"

            # Patch mkdir to raise PermissionError
            with patch.object(Path, 'mkdir') as mock_mkdir:
                mock_mkdir.side_effect = PermissionError("Access denied")

                with pytest.raises(PermissionError):
                    HealingLogger(log_dir, json_log=True)

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
    def test_init_console_format_options(self):
        """Test HealingLogger accepts all three console_format options."""
        with tempfile.TemporaryDirectory() as tmp:
            for console_format in ["box", "simple", "minimal"]:
                log_dir = Path(tmp) / f"logs_{console_format}"
                logger = HealingLogger(log_dir, console_format=console_format)
                assert logger.console_format == console_format


class TestHealingLoggerCleanup:
    """Tests for _cleanup_orphaned_temp_files() method."""

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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


# =============================================================================
# US-003: Log Method Tests for All Entry Types
# =============================================================================


class TestLogClassification:
    """Tests for HealingLogger.log_classification() - US-003."""

    @pytest.mark.integration
    def test_log_classification_creates_entry_with_correct_component_and_action(self):
        """Test log_classification() creates entry with correct component='watcher' and action='classify'."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            # Create mock classification
            mock_classification = MagicMock()
            mock_classification.category = "network"
            mock_classification.severity = "medium"
            mock_classification.suggested_healer = "NetworkHealer"
            mock_classification.confidence = 0.85
            mock_classification.needs_llm_healer = False
            mock_classification.reasoning = "Connection timeout detected"

            # Log classification
            error = ConnectionError("Connection refused")
            logger.log_classification("DOWNLOAD", error, mock_classification, 15.5)

            # Verify entry created with correct values
            assert len(logger.entries) == 1
            entry = logger.entries[0]
            assert entry.component == "watcher"
            assert entry.action == "classify"
            assert entry.stage == "DOWNLOAD"
            assert entry.error_type == "ConnectionError"
            assert "Connection refused" in entry.error_message
            assert entry.result == "success"
            assert entry.duration_ms == 15.5

    @pytest.mark.integration
    def test_log_classification_captures_classification_details(self):
        """Test log_classification() stores classification details correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            mock_classification = MagicMock()
            mock_classification.category = "rate_limit"
            mock_classification.severity = "high"
            mock_classification.suggested_healer = "RateLimitHealer"
            mock_classification.confidence = 0.95
            mock_classification.needs_llm_healer = True
            mock_classification.reasoning = "HTTP 429 response detected with Retry-After header"

            error = Exception("Rate limited")
            logger.log_classification("API_CALL", error, mock_classification, 5.0)

            entry = logger.entries[0]
            details = entry.details

            assert details["category"] == "rate_limit"
            assert details["severity"] == "high"
            assert details["suggested_healer"] == "RateLimitHealer"
            assert details["confidence"] == 0.95
            assert details["needs_llm_healer"] == True
            assert "HTTP 429" in details["reasoning"]

    @pytest.mark.integration
    def test_log_classification_truncates_long_reasoning(self):
        """Test log_classification() truncates reasoning to 100 characters."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            mock_classification = MagicMock()
            mock_classification.category = "unknown"
            mock_classification.severity = "low"
            mock_classification.suggested_healer = None
            mock_classification.confidence = 0.3
            mock_classification.needs_llm_healer = True
            mock_classification.reasoning = "A" * 200  # Long reasoning

            error = Exception("Unknown error")
            logger.log_classification("TEST", error, mock_classification, 1.0)

            entry = logger.entries[0]
            assert len(entry.details["reasoning"]) == 100


class TestLogHealerAttempt:
    """Tests for HealingLogger.log_healer_attempt() - US-003."""

    @pytest.mark.integration
    def test_log_healer_attempt_captures_healer_name_and_result(self):
        """Test log_healer_attempt() captures healer_name, result, and stack_trace in details."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            # Create mock healer result
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.action = MagicMock()
            mock_result.action.value = "RETRY"
            mock_result.message = "Backoff applied, retrying"
            mock_result.modified_config = True
            mock_result.details = {"backoff_seconds": 30}

            error = TimeoutError("Request timed out")
            stack_trace = "Traceback (most recent call last):\n  File ..."

            logger.log_healer_attempt(
                "TRANSCRIBE", "TimeoutHealer", error, mock_result, 250.0, stack_trace
            )

            assert len(logger.entries) == 1
            entry = logger.entries[0]

            # Verify basic fields
            assert entry.component == "healer"
            assert entry.action == "attempt"
            assert entry.stage == "TRANSCRIBE"
            assert entry.error_type == "TimeoutError"
            assert entry.result == "success"
            assert entry.duration_ms == 250.0
            assert entry.stack_trace == stack_trace

            # Verify details contain healer info
            assert entry.details["healer"] == "TimeoutHealer"
            assert entry.details["action"] == "RETRY"
            assert entry.details["message"] == "Backoff applied, retrying"
            assert entry.details["modified_config"] == True
            assert entry.details["healer_details"]["backoff_seconds"] == 30

    @pytest.mark.integration
    def test_log_healer_attempt_records_failed_result(self):
        """Test log_healer_attempt() correctly records failed result."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            mock_result = MagicMock()
            mock_result.success = False
            mock_result.action = MagicMock()
            mock_result.action.value = "ABORT"
            mock_result.message = "Cannot fix this error"
            mock_result.modified_config = False
            mock_result.details = {}

            error = ValueError("Invalid data")
            logger.log_healer_attempt("MATCH", "DataHealer", error, mock_result, 100.0)

            entry = logger.entries[0]
            assert entry.result == "failed"
            assert entry.details["action"] == "ABORT"

    @pytest.mark.integration
    def test_log_healer_attempt_handles_action_without_value_attr(self):
        """Test log_healer_attempt() handles action that's a string not enum."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.action = "SKIP"  # String, not enum
            mock_result.message = "Skipping"
            mock_result.modified_config = False
            mock_result.details = {}

            error = Exception("Test")
            logger.log_healer_attempt("TEST", "TestHealer", error, mock_result, 10.0)

            entry = logger.entries[0]
            assert entry.details["action"] == "SKIP"


class TestLogFallback:
    """Tests for HealingLogger.log_fallback() - US-003."""

    @pytest.mark.integration
    def test_log_fallback_records_component_transition(self):
        """Test log_fallback() records from_component and to_component transition correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            logger.log_fallback(
                "DOWNLOAD",
                from_component="WatcherWithLLM",
                to_component="PatternMatcher",
                reason="LLM API unavailable"
            )

            assert len(logger.entries) == 1
            entry = logger.entries[0]

            # Verify basic fields
            assert entry.component == "fallback"
            assert entry.action == "fallback"
            assert entry.stage == "DOWNLOAD"
            assert entry.error_type == "FallbackTriggered"
            assert entry.error_message == "LLM API unavailable"
            assert entry.result == "degraded"
            assert entry.duration_ms == 0

            # Verify transition details
            assert entry.details["from"] == "WatcherWithLLM"
            assert entry.details["to"] == "PatternMatcher"
            assert entry.details["reason"] == "LLM API unavailable"

    @pytest.mark.integration
    def test_log_fallback_multiple_transitions(self):
        """Test log_fallback() can record multiple fallback transitions."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            # First fallback
            logger.log_fallback("STAGE1", "ComponentA", "ComponentB", "Reason 1")
            # Second fallback
            logger.log_fallback("STAGE2", "ComponentB", "ComponentC", "Reason 2")

            assert len(logger.entries) == 2

            # First transition
            assert logger.entries[0].details["from"] == "ComponentA"
            assert logger.entries[0].details["to"] == "ComponentB"

            # Second transition
            assert logger.entries[1].details["from"] == "ComponentB"
            assert logger.entries[1].details["to"] == "ComponentC"


class TestLogEscalation:
    """Tests for HealingLogger.log_escalation() - US-003."""

    @pytest.mark.integration
    def test_log_escalation_distinguishes_to_user(self):
        """Test log_escalation() correctly handles to_user=True escalation."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            error = RuntimeError("Critical system failure")
            logger.log_escalation(
                "OUTPUT",
                error,
                "All automated fixes exhausted",
                to_user=True
            )

            assert len(logger.entries) == 1
            entry = logger.entries[0]

            # Verify basic fields
            assert entry.component == "orchestrator"
            assert entry.action == "escalate"
            assert entry.stage == "OUTPUT"
            assert entry.error_type == "RuntimeError"
            assert entry.result == "escalated"

            # Verify escalation type
            assert entry.details["to_user"] == True
            assert entry.details["to_llm_healer"] == False
            assert entry.details["reason"] == "All automated fixes exhausted"

    @pytest.mark.integration
    def test_log_escalation_distinguishes_to_llm_healer(self):
        """Test log_escalation() correctly handles to_user=False (to LLM healer) escalation."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            error = ValueError("Complex parsing error")
            logger.log_escalation(
                "ANALYZE",
                error,
                "Pattern matchers insufficient",
                to_user=False
            )

            entry = logger.entries[0]

            # Verify escalation type
            assert entry.details["to_user"] == False
            assert entry.details["to_llm_healer"] == True
            assert entry.details["reason"] == "Pattern matchers insufficient"

    @pytest.mark.integration
    def test_log_escalation_default_is_to_llm_healer(self):
        """Test log_escalation() defaults to LLM healer (to_user=False)."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            error = Exception("Test error")
            # Don't specify to_user, should default to False
            logger.log_escalation("TEST", error, "Testing default")

            entry = logger.entries[0]
            assert entry.details["to_user"] == False
            assert entry.details["to_llm_healer"] == True


class TestLogSelfHeal:
    """Tests for HealingLogger.log_self_heal() - US-003."""

    @pytest.mark.integration
    def test_log_self_heal_captures_attempt_progression(self):
        """Test log_self_heal() captures attempt count and max_attempts progression."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            # Simulate 3 self-heal attempts
            for attempt in range(1, 4):
                logger.log_self_heal(
                    healer="GeminiHealer",
                    attempt=attempt,
                    max_attempts=3,
                    error_type="APIError",
                    action=f"Retry with modified prompt (attempt {attempt})"
                )

            assert len(logger.entries) == 3

            # Verify first attempt
            entry1 = logger.entries[0]
            assert entry1.stage == "SELF_HEAL"
            assert entry1.component == "GeminiHealer"
            assert entry1.action == "self_heal"
            assert entry1.error_type == "APIError"
            assert entry1.result == "retrying"
            assert entry1.details["attempt"] == 1
            assert entry1.details["max_attempts"] == 3

            # Verify progression
            assert logger.entries[1].details["attempt"] == 2
            assert logger.entries[2].details["attempt"] == 3

            # All should have same max_attempts
            for entry in logger.entries:
                assert entry.details["max_attempts"] == 3

    @pytest.mark.integration
    def test_log_self_heal_records_action_as_error_message(self):
        """Test log_self_heal() stores action description in error_message field."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            logger.log_self_heal(
                healer="AnthropicHealer",
                attempt=1,
                max_attempts=5,
                error_type="RateLimitError",
                action="Applying exponential backoff"
            )

            entry = logger.entries[0]
            assert entry.error_message == "Applying exponential backoff"
            assert entry.component == "AnthropicHealer"

    @pytest.mark.integration
    def test_log_self_heal_different_healers(self):
        """Test log_self_heal() correctly identifies different healer types."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=False)

            healers = ["GeminiHealer", "AnthropicHealer", "OllamaHealer"]
            for healer in healers:
                logger.log_self_heal(healer, 1, 3, "TestError", "Testing")

            assert len(logger.entries) == 3
            for i, healer in enumerate(healers):
                assert logger.entries[i].component == healer


# =============================================================================
# US-004: Report Generation and Finalize Tests
# =============================================================================


class TestGenerateReportCounts:
    """Tests for HealingLogger.generate_report() result counts - US-004."""

    @pytest.mark.integration
    def test_generate_report_returns_correct_success_failed_counts(self):
        """Test generate_report() returns correct success/failed counts from entries."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False)

            # Add entries with different results
            success_entries = 5
            failed_entries = 3

            for i in range(success_entries):
                entry = HealingLogEntry(
                    timestamp=datetime.now(timezone.utc),
                    stage="TEST",
                    component="healer",
                    action="attempt",
                    error_type="TestError",
                    error_message=f"Success entry {i}",
                    result="success",
                    duration_ms=10.0
                )
                healing_logger._write(entry)

            for i in range(failed_entries):
                entry = HealingLogEntry(
                    timestamp=datetime.now(timezone.utc),
                    stage="TEST",
                    component="healer",
                    action="attempt",
                    error_type="TestError",
                    error_message=f"Failed entry {i}",
                    result="failed",
                    duration_ms=10.0
                )
                healing_logger._write(entry)

            report = healing_logger.generate_report()

            # Verify counts appear in report
            assert f"Successful heals: {success_entries}" in report
            assert f"Failed heals: {failed_entries}" in report
            assert f"Total entries: {success_entries + failed_entries}" in report

    @pytest.mark.integration
    def test_generate_report_counts_watcher_classifications(self):
        """Test generate_report() counts watcher classifications correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False)

            # Add watcher classification entries
            for i in range(4):
                entry = HealingLogEntry(
                    timestamp=datetime.now(timezone.utc),
                    stage="DOWNLOAD",
                    component="watcher",
                    action="classify",
                    error_type="NetworkError",
                    error_message=f"Classification {i}",
                    result="success",
                    duration_ms=5.0
                )
                healing_logger._write(entry)

            # Add non-watcher entry
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="DOWNLOAD",
                component="healer",
                action="attempt",
                error_type="NetworkError",
                error_message="Healer attempt",
                result="success",
                duration_ms=50.0
            )
            healing_logger._write(entry)

            report = healing_logger.generate_report()
            assert "Watcher classifications: 4" in report

    @pytest.mark.integration
    def test_generate_report_counts_llm_healer_invocations(self):
        """Test generate_report() counts LLM healer invocations correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False)

            # Add llm-healer attempt entries
            for i in range(3):
                entry = HealingLogEntry(
                    timestamp=datetime.now(timezone.utc),
                    stage="ANALYZE",
                    component="llm-healer",
                    action="attempt",
                    error_type="ParseError",
                    error_message=f"LLM attempt {i}",
                    result="success",
                    duration_ms=100.0
                )
                healing_logger._write(entry)

            # Add llm-healer non-attempt entry (shouldn't be counted)
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="SELF_HEAL",
                component="llm-healer",
                action="provider_switch",
                error_type="ProviderSwitch",
                error_message="Switch provider",
                result="retrying",
                duration_ms=0
            )
            healing_logger._write(entry)

            report = healing_logger.generate_report()
            assert "LLM healer invocations: 3" in report


class TestGenerateReportGrouping:
    """Tests for HealingLogger.generate_report() grouping by stage - US-004."""

    @pytest.mark.integration
    def test_generate_report_groups_entries_by_stage(self):
        """Test generate_report() groups entries by stage and counts healer activity."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False)

            # Add healer entries to different stages
            stages_data = [
                ("DOWNLOAD", 3, 1),     # 3 success, 1 failed
                ("TRANSCRIBE", 2, 2),   # 2 success, 2 failed
                ("MATCH", 4, 0),        # 4 success, 0 failed
            ]

            for stage, success_count, failed_count in stages_data:
                for _ in range(success_count):
                    entry = HealingLogEntry(
                        timestamp=datetime.now(timezone.utc),
                        stage=stage,
                        component="healer",
                        action="attempt",
                        error_type="TestError",
                        error_message="test",
                        result="success",
                        duration_ms=10.0
                    )
                    healing_logger._write(entry)

                for _ in range(failed_count):
                    entry = HealingLogEntry(
                        timestamp=datetime.now(timezone.utc),
                        stage=stage,
                        component="healer",
                        action="attempt",
                        error_type="TestError",
                        error_message="test",
                        result="failed",
                        duration_ms=10.0
                    )
                    healing_logger._write(entry)

            report = healing_logger.generate_report()

            # Verify stage groupings appear
            assert "By stage:" in report
            assert "DOWNLOAD: 3/4 healed" in report
            assert "TRANSCRIBE: 2/4 healed" in report
            assert "MATCH: 4/4 healed" in report

    @pytest.mark.integration
    def test_generate_report_excludes_self_heal_and_preflight_from_stages(self):
        """Test generate_report() excludes SELF_HEAL and PREFLIGHT from stage listing."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False)

            # Add healer entries to regular stage
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="DOWNLOAD",
                component="healer",
                action="attempt",
                error_type="TestError",
                error_message="test",
                result="success",
                duration_ms=10.0
            )
            healing_logger._write(entry)

            # Add SELF_HEAL entries (should be excluded from stage grouping)
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="SELF_HEAL",
                component="healer",
                action="attempt",
                error_type="TestError",
                error_message="test",
                result="success",
                duration_ms=10.0
            )
            healing_logger._write(entry)

            # Add PREFLIGHT entries (should be excluded)
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="PREFLIGHT",
                component="healer",
                action="attempt",
                error_type="TestError",
                error_message="test",
                result="success",
                duration_ms=10.0
            )
            healing_logger._write(entry)

            report = healing_logger.generate_report()

            # DOWNLOAD should appear
            assert "DOWNLOAD:" in report
            # SELF_HEAL and PREFLIGHT should NOT appear in stage listing
            assert "SELF_HEAL:" not in report
            assert "PREFLIGHT:" not in report

    @pytest.mark.integration
    def test_generate_report_only_shows_stages_with_healer_activity(self):
        """Test generate_report() only lists stages that have healer activity."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False)

            # Add watcher-only entry (no healer)
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="ANALYZE",
                component="watcher",
                action="classify",
                error_type="TestError",
                error_message="test",
                result="success",
                duration_ms=5.0
            )
            healing_logger._write(entry)

            # Add healer entry to different stage
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="DOWNLOAD",
                component="healer",
                action="attempt",
                error_type="TestError",
                error_message="test",
                result="success",
                duration_ms=10.0
            )
            healing_logger._write(entry)

            report = healing_logger.generate_report()

            # DOWNLOAD should appear (has healer)
            assert "DOWNLOAD:" in report
            # ANALYZE should NOT appear (no healer, only watcher)
            assert "ANALYZE:" not in report


class TestGenerateReportEmptyEntries:
    """Tests for HealingLogger.generate_report() empty entries case - US-004."""

    @pytest.mark.integration
    def test_generate_report_returns_no_activity_when_entries_empty(self):
        """Test generate_report() returns 'No healing activity recorded' when entries empty."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False)

            # No entries added
            report = healing_logger.generate_report()
            assert report == "No healing activity recorded."

    @pytest.mark.integration
    def test_generate_report_includes_session_id(self):
        """Test generate_report() includes session ID in report."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False)

            # Add one entry
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="TEST",
                component="healer",
                action="attempt",
                error_type="TestError",
                error_message="test",
                result="success",
                duration_ms=10.0
            )
            healing_logger._write(entry)

            report = healing_logger.generate_report()
            assert f"Session ID: {healing_logger.session_id}" in report


class TestFinalize:
    """Tests for HealingLogger.finalize() - US-004."""

    @pytest.mark.integration
    def test_finalize_writes_final_json_with_session_summary(self):
        """Test finalize() writes final JSON with session summary."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=True)

            # Add some entries
            for i in range(3):
                entry = HealingLogEntry(
                    timestamp=datetime.now(timezone.utc),
                    stage=f"STAGE_{i}",
                    component="healer",
                    action="attempt",
                    error_type="TestError",
                    error_message=f"Entry {i}",
                    result="success" if i % 2 == 0 else "failed",
                    duration_ms=10.0
                )
                healing_logger._write(entry)

            # Finalize
            healing_logger.finalize()

            # Read and verify JSON structure
            content = json.loads(healing_logger.json_file.read_text())

            assert "session_id" in content
            assert content["session_id"] == healing_logger.session_id
            assert "total_entries" in content
            assert content["total_entries"] == 3
            assert "entries" in content
            assert len(content["entries"]) == 3

    @pytest.mark.integration
    def test_finalize_json_contains_all_entry_data(self):
        """Test finalize() JSON contains complete entry data."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=True)

            # Add entry with all fields
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="TEST_STAGE",
                component="test_component",
                action="test_action",
                error_type="TestError",
                error_message="Test message",
                result="success",
                duration_ms=123.45,
                stack_trace="Test stack trace",
                details={"key": "value"}
            )
            healing_logger._write(entry)

            healing_logger.finalize()

            content = json.loads(healing_logger.json_file.read_text())
            entry_data = content["entries"][0]

            assert entry_data["stage"] == "TEST_STAGE"
            assert entry_data["component"] == "test_component"
            assert entry_data["action"] == "test_action"
            assert entry_data["error_type"] == "TestError"
            assert entry_data["error_message"] == "Test message"
            assert entry_data["result"] == "success"
            assert entry_data["duration_ms"] == 123.45
            assert entry_data["stack_trace"] == "Test stack trace"
            assert entry_data["details"] == {"key": "value"}

    @pytest.mark.integration
    def test_finalize_with_empty_entries_writes_empty_entries_array(self):
        """Test finalize() writes empty entries array when no entries."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=True)

            # No entries added
            healing_logger.finalize()

            content = json.loads(healing_logger.json_file.read_text())

            assert "session_id" in content
            assert content["total_entries"] == 0
            assert content["entries"] == []

    @pytest.mark.integration
    def test_finalize_calls_generate_report(self):
        """Test finalize() calls generate_report() for logging."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=True)

            # Add an entry
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="TEST",
                component="healer",
                action="attempt",
                error_type="TestError",
                error_message="Test",
                result="success",
                duration_ms=10.0
            )
            healing_logger._write(entry)

            # Mock generate_report to verify it's called
            with patch.object(healing_logger, 'generate_report', wraps=healing_logger.generate_report) as mock_report:
                healing_logger.finalize()
                mock_report.assert_called_once()


class TestPrintBox:
    """Tests for HealingLogger.print_box() - US-004."""

    @pytest.mark.integration
    def test_print_box_handles_minimal_format(self, capsys):
        """Test print_box() handles minimal console_format option."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False, console_format="minimal")

            lines = ["Line 1", "Line 2", "Line 3"]
            healing_logger.print_box("Test Title", lines)

            captured = capsys.readouterr()

            # Minimal format: just indented lines, no box or title
            assert "  Line 1" in captured.out
            assert "  Line 2" in captured.out
            assert "  Line 3" in captured.out
            # Should NOT have box characters
            assert "┌" not in captured.out
            assert "└" not in captured.out
            assert "===" not in captured.out

    @pytest.mark.integration
    def test_print_box_handles_simple_format(self, capsys):
        """Test print_box() handles simple console_format option."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False, console_format="simple")

            lines = ["Line 1", "Line 2"]
            healing_logger.print_box("Test Title", lines)

            captured = capsys.readouterr()

            # Simple format: === Title === header
            assert "=== Test Title ===" in captured.out
            assert "  Line 1" in captured.out
            assert "  Line 2" in captured.out
            # Should NOT have Unicode box characters
            assert "┌" not in captured.out
            assert "└" not in captured.out

    @pytest.mark.integration
    def test_print_box_handles_box_format(self, capsys):
        """Test print_box() handles box console_format option (default)."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False, console_format="box")

            lines = ["Line 1", "Line 2"]
            healing_logger.print_box("Test Title", lines)

            captured = capsys.readouterr()

            # Box format: Unicode box characters
            assert "┌" in captured.out
            assert "└" in captured.out
            assert "│" in captured.out
            assert "├" in captured.out
            assert "Test Title" in captured.out
            assert "Line 1" in captured.out
            assert "Line 2" in captured.out

    @pytest.mark.integration
    def test_print_box_truncates_long_lines(self, capsys):
        """Test print_box() truncates lines longer than width."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False, console_format="box")

            # Create a line longer than default width (65)
            long_line = "A" * 100
            healing_logger.print_box("Title", [long_line], width=65)

            captured = capsys.readouterr()

            # Line should be truncated with "..."
            assert "..." in captured.out
            # Original long line should not appear in full
            assert "A" * 100 not in captured.out

    @pytest.mark.integration
    def test_print_box_custom_width(self, capsys):
        """Test print_box() respects custom width parameter."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            healing_logger = HealingLogger(log_dir, json_log=False, console_format="box")

            healing_logger.print_box("Title", ["Short line"], width=80)

            captured = capsys.readouterr()

            # Box should be 80 chars wide (excluding │ characters)
            lines = captured.out.split('\n')
            # Find a line with box border
            for line in lines:
                if line.startswith('┌'):
                    # Total width includes ┌ + 80 dashes + ┐ = 82 chars
                    assert len(line) == 82
                    break
