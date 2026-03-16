"""
Tests for checkpoint automatic cleanup (US-115-010).

Covers:
- AC1: Clean up old checkpoint.backup.*.json files older than max_age_days
- AC2: Clean up orphaned .tmp files from crashed saves
- AC3: cleanup_report includes files_removed and bytes_freed
- AC4: Configurable enabled (default true) and max_age_days (default 7)
"""

import json
import logging
import os
import time
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData


class _FakeAutoCleanupConfig:
    """Minimal auto_cleanup config stub for testing."""
    def __init__(self, enabled=True, max_age_days=7):
        self.enabled = enabled
        self.max_age_days = max_age_days


class _FakePipelineConfig:
    """Minimal pipeline config stub for testing."""
    def __init__(self, backup_count=3, auto_cleanup_enabled=True, max_age_days=7):
        self.checkpoint_backup_count = backup_count
        self.auto_cleanup = _FakeAutoCleanupConfig(enabled=auto_cleanup_enabled, max_age_days=max_age_days)


class _FakeConfig:
    """Minimal config stub for testing."""
    def __init__(self, backup_count=3, auto_cleanup_enabled=True, max_age_days=7):
        self.pipeline = _FakePipelineConfig(backup_count, auto_cleanup_enabled, max_age_days)


class TestCheckpointCleanup:
    """US-115-010: Test automatic stale checkpoint cleanup."""

    @pytest.mark.fast
    def test_cleanup_removes_old_backup_files(self, tmp_path):
        """Old checkpoint.backup.*.json files should be removed on cleanup."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(max_age_days=7))

        # Create some old backup files (old = 10 days ago)
        old_time = time.time() - (10 * 24 * 3600)
        backup1 = tmp_path / "checkpoint.backup.1.json"
        backup2 = tmp_path / "checkpoint.backup.2.json"
        backup3 = tmp_path / "checkpoint.backup.3.json"

        backup1.write_text('{"test": 1}')
        backup2.write_text('{"test": 2}')
        backup3.write_text('{"test": 3}')

        # Set file modification time to 10 days ago
        os.utime(backup1, (old_time, old_time))
        os.utime(backup2, (old_time, old_time))
        os.utime(backup3, (old_time, old_time))

        # Create a recent backup (should NOT be deleted)
        recent_backup = tmp_path / "checkpoint.backup.json"
        recent_backup.write_text('{"test": 0}')

        # Run cleanup
        report = manager.cleanup_stale_backups()

        # Should have removed 3 old backups
        assert report["backup_files_removed"] == 3
        assert report["files_removed"] == 3
        assert report["bytes_freed"] > 0

        # Recent backup should still exist
        assert recent_backup.exists()
        # Old backups should be gone
        assert not backup1.exists()
        assert not backup2.exists()
        assert not backup3.exists()

    @pytest.mark.fast
    def test_cleanup_removes_orphaned_temp_files(self, tmp_path):
        """Orphaned .tmp files from crashed saves should be removed."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig())

        # Create temp files
        temp1 = tmp_path / "checkpoint.tmp"
        temp2 = tmp_path / "something.tmp"

        temp1.write_text('{"in_progress": true}')
        temp2.write_text('{"other": true}')

        # Run cleanup
        report = manager.cleanup_stale_backups()

        # Should have removed both temp files
        assert report["temp_files_removed"] == 2
        assert report["files_removed"] == 2

        # Temp files should be gone
        assert not temp1.exists()
        assert not temp2.exists()

    @pytest.mark.fast
    def test_cleanup_respects_max_age_days(self, tmp_path):
        """Files younger than max_age_days should NOT be removed."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(max_age_days=7))

        # Create a backup that's 5 days old (younger than 7 day max)
        recent_time = time.time() - (5 * 24 * 3600)
        recent_backup = tmp_path / "checkpoint.backup.json"
        recent_backup.write_text('{"test": 1}')
        os.utime(recent_backup, (recent_time, recent_time))

        # Run cleanup
        report = manager.cleanup_stale_backups()

        # Should NOT have removed any files
        assert report["backup_files_removed"] == 0
        assert report["files_removed"] == 0
        assert report["bytes_freed"] == 0

        # Recent backup should still exist
        assert recent_backup.exists()

    @pytest.mark.fast
    def test_cleanup_disabled_when_config_disabled(self, tmp_path):
        """When auto_cleanup.enabled=false, no files should be removed."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(auto_cleanup_enabled=False))

        # Create old backup and temp files
        old_time = time.time() - (10 * 24 * 3600)
        backup = tmp_path / "checkpoint.backup.1.json"
        backup.write_text('{"test": 1}')
        os.utime(backup, (old_time, old_time))

        temp = tmp_path / "checkpoint.tmp"
        temp.write_text('{"test": 1}')

        # Run cleanup
        report = manager.cleanup_stale_backups()

        # Should NOT have removed any files
        assert report["backup_files_removed"] == 0
        assert report["temp_files_removed"] == 0
        assert report["files_removed"] == 0

        # Files should still exist
        assert backup.exists()
        assert temp.exists()

    @pytest.mark.fast
    def test_cleanup_report_includes_bytes_freed(self, tmp_path):
        """Cleanup report should include bytes_freed."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig())

        # Create temp file with known size
        temp = tmp_path / "checkpoint.tmp"
        content = '{"test": "data"}'  # 18 bytes
        temp.write_text(content)

        # Run cleanup
        report = manager.cleanup_stale_backups()

        # Should have freed exactly the bytes from the temp file
        assert report["bytes_freed"] == len(content)
        assert report["temp_files_removed"] == 1

    @pytest.mark.fast
    def test_cleanup_handles_nonexistent_directory(self, tmp_path):
        """Cleanup should handle missing files gracefully."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig())

        # No files exist - should return empty report
        report = manager.cleanup_stale_backups()

        assert report["files_removed"] == 0
        assert report["bytes_freed"] == 0
        assert report["errors"] == []

    @pytest.mark.fast
    def test_cleanup_preserves_main_checkpoint(self, tmp_path):
        """Main checkpoint.json should never be deleted by cleanup."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(max_age_days=7))

        # Create main checkpoint
        main_checkpoint = tmp_path / "checkpoint.json"
        main_checkpoint.write_text('{"version": "2.0"}')

        # Set to old time
        old_time = time.time() - (10 * 24 * 3600)
        os.utime(main_checkpoint, (old_time, old_time))

        # Run cleanup
        report = manager.cleanup_stale_backups()

        # Main checkpoint should still exist
        assert main_checkpoint.exists()
        # No files should have been removed (main checkpoint isn't a backup or temp)
        assert report["files_removed"] == 0
