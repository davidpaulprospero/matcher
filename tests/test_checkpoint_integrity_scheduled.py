"""US-115-012: Test scheduled checkpoint integrity verification.

Tests verify that:
- verify_on_save option verifies checksum after each write
- schedule_verification() starts periodic integrity checks
- verify_all_backups() checks all rotated backups
- integrity_check_result includes last_verified_at, all_backups_valid
- Scheduled verification can be started and stopped
"""

import json
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.checkpoint import CheckpointManager, CheckpointData


class TestCheckpointIntegrityScheduled:
    """Test scheduled checkpoint integrity verification."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_config_with_integrity(self):
        """Create a mock config with integrity verification enabled."""
        config = MagicMock()
        pipeline = MagicMock()

        # Compression config
        compression = MagicMock()
        compression.enabled = True
        compression.compression_level = 6

        # Auto repair config
        auto_repair = MagicMock()
        auto_repair.enabled = True
        auto_repair.log_repairs = True
        auto_repair.max_repair_attempts = 3

        # Auto cleanup config
        auto_cleanup = MagicMock()
        auto_cleanup.enabled = True
        auto_cleanup.max_age_days = 7

        # Integrity config
        integrity = MagicMock()
        integrity.verify_on_save = True
        integrity.schedule_verification = False
        integrity.cron_expression = "0 * * * *"
        integrity.verify_backups = True

        pipeline.checkpoint_compression = compression
        pipeline.checkpoint_auto_repair = auto_repair
        pipeline.auto_cleanup = auto_cleanup
        pipeline.checkpoint_integrity = integrity
        pipeline.checkpoint_backup_count = 3
        pipeline.min_rotation_interval_seconds = 60

        config.pipeline = pipeline
        return config

    @pytest.fixture
    def mock_config_without_integrity(self):
        """Create a mock config with integrity verification disabled."""
        config = MagicMock()
        pipeline = MagicMock()

        compression = MagicMock()
        compression.enabled = True
        compression.compression_level = 6

        auto_repair = MagicMock()
        auto_repair.enabled = True

        auto_cleanup = MagicMock()
        auto_cleanup.enabled = True

        integrity = MagicMock()
        integrity.verify_on_save = False
        integrity.schedule_verification = False
        integrity.cron_expression = "0 * * * *"
        integrity.verify_backups = False

        pipeline.checkpoint_compression = compression
        pipeline.checkpoint_auto_repair = auto_repair
        pipeline.auto_cleanup = auto_cleanup
        pipeline.checkpoint_integrity = integrity
        pipeline.checkpoint_backup_count = 3
        pipeline.min_rotation_interval_seconds = 60

        config.pipeline = pipeline
        return config

    def test_verify_on_save_enabled(self, temp_project_dir, mock_config_with_integrity):
        """Test that verify_on_save is correctly configured."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_with_integrity)

        assert cm._verify_on_save is True
        assert cm._schedule_verification is False
        assert cm._verify_backups is True
        assert cm._cron_expression == "0 * * * *"

    def test_verify_on_save_disabled(self, temp_project_dir, mock_config_without_integrity):
        """Test that verify_on_save can be disabled."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_without_integrity)

        assert cm._verify_on_save is False
        assert cm._schedule_verification is False
        assert cm._verify_backups is False

    def test_verify_on_save_verifies_checksum(self, temp_project_dir, mock_config_with_integrity):
        """Test that verify_on_save computes checksum after save."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_with_integrity)

        # Create checkpoint data
        cm.data = CheckpointData(
            version="2.1",
            created_at="2026-02-16T10:00:00",
            config_hash="abc123"
        )
        cm.data.analyze = {"keywords": ["test"]}

        # Save with verify_on_save enabled
        cm.save(stage="ANALYZE", stage_data=cm.data.analyze)

        # Verify that integrity was checked
        assert cm._last_verified_at is not None
        integrity = cm.verify_integrity()
        assert integrity['is_valid'] is True
        assert integrity['checksum'] != ''

    def test_verify_all_backups_main_and_rotated(self, temp_project_dir, mock_config_with_integrity):
        """Test verify_all_backups checks both main and rotated backups."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_with_integrity)

        # Create and save checkpoint multiple times to create rotated backups
        cm.data = CheckpointData(
            version="2.1",
            created_at="2026-02-16T10:00:00",
            config_hash="abc123"
        )
        cm.data.analyze = {"keywords": ["test"]}
        cm.save(stage="ANALYZE", stage_data=cm.data.analyze, force_full=True)
        time.sleep(0.1)

        cm.data.analyze = {"keywords": ["test2"]}
        cm.save(stage="ANALYZE", stage_data=cm.data.analyze, force_full=True)
        time.sleep(0.1)

        cm.data.analyze = {"keywords": ["test3"]}
        cm.save(stage="ANALYZE", stage_data=cm.data.analyze, force_full=True)

        # Verify all backups
        result = cm.verify_all_backups()

        assert result['backup_count'] >= 1  # At least main checkpoint
        assert result['valid_count'] >= 1
        assert result['all_valid'] is True
        assert len(result['backup_results']) >= 1

    def test_verify_all_backups_detects_invalid(self, temp_project_dir, mock_config_with_integrity):
        """Test verify_all_backups detects invalid backups."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_with_integrity)

        # Create a valid checkpoint and force backup rotation
        cm.data = CheckpointData(
            version="2.1",
            created_at="2026-02-16T10:00:00",
            config_hash="abc123"
        )
        cm.save(stage="ANALYZE", stage_data={"keywords": ["test"]}, force_full=True)
        time.sleep(0.1)
        # Second save triggers backup rotation
        cm.save(stage="ANALYZE", stage_data={"keywords": ["test2"]}, force_full=True)

        # Corrupt a backup file (index 0 maps to checkpoint.backup.json)
        backup_path = temp_project_dir / "checkpoint.backup.json"
        if backup_path.exists():
            with open(backup_path, 'w') as f:
                f.write("{ invalid json }")

        # Verify should detect corruption
        result = cm.verify_all_backups()

        # Should detect the invalid backup
        assert result['all_valid'] is False
        assert len(result['issues']) > 0

    def test_schedule_verification_starts_thread(self, temp_project_dir, mock_config_with_integrity):
        """Test that schedule_verification starts a background thread."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_with_integrity)

        # Enable schedule verification in config for this test
        cm._schedule_verification = True

        # Start scheduled verification
        result = cm.schedule_verification(cron_expression="0 * * * *")

        assert result['scheduled'] is True
        assert result['cron_expression'] == "0 * * * *"

        # Wait a bit for thread to start
        time.sleep(0.2)

        assert cm._scheduled_verification_thread is not None
        assert cm._scheduled_verification_thread.is_alive()

        # Stop scheduled verification
        cm.stop_scheduled_verification()

        time.sleep(0.2)
        assert not cm._scheduled_verification_thread.is_alive()

    def test_schedule_verification_disabled(self, temp_project_dir, mock_config_without_integrity):
        """Test that schedule_verification returns disabled when not configured."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_without_integrity)

        result = cm.schedule_verification()

        assert result['scheduled'] is False
        assert 'disabled' in result['message'].lower()

    def test_get_integrity_status(self, temp_project_dir, mock_config_with_integrity):
        """Test get_integrity_status returns correct status."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_with_integrity)

        # Initially no verification done
        status = cm.get_integrity_status()

        assert status['verify_on_save_enabled'] is True
        assert status['schedule_verification_enabled'] is False
        assert status['schedule_verification_running'] is False
        assert status['last_verified_at'] is None
        assert status['all_backups_valid'] is True  # Default
        assert status['history_count'] == 0

        # After verification
        cm.save(stage="ANALYZE", stage_data={"keywords": ["test"]})  # This triggers verify_on_save

        status = cm.get_integrity_status()

        assert status['last_verified_at'] is not None
        assert status['history_count'] > 0

    def test_integrity_check_history(self, temp_project_dir, mock_config_with_integrity):
        """Test that integrity checks are recorded in history."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_with_integrity)

        # Save checkpoint (triggers verify_on_save)
        cm.data = CheckpointData(
            version="2.1",
            created_at="2026-02-16T10:00:00",
            config_hash="abc123"
        )
        cm.save(stage="ANALYZE", stage_data={"keywords": ["test"]})

        # Verify all backups
        cm.verify_all_backups()

        # Check history
        assert len(cm._integrity_check_history) >= 2  # At least one on_save + one all_backups

        # Check history entries have expected fields
        for entry in cm._integrity_check_history:
            assert 'timestamp' in entry
            assert 'type' in entry
            assert entry['type'] in ['on_save', 'all_backups']

    def test_verify_all_backups_when_disabled(self, temp_project_dir, mock_config_without_integrity):
        """Test verify_all_backups when backup verification is disabled."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_without_integrity)

        # Create checkpoint
        cm.data = CheckpointData(
            version="2.1",
            created_at="2026-02-16T10:00:00",
            config_hash="abc123"
        )
        cm.save(stage="ANALYZE", stage_data={"keywords": ["test"]})

        # Verify with backups disabled
        result = cm.verify_all_backups()

        # Should return success with disabled message
        assert result['all_valid'] is True
        assert 'disabled' in result['issues'][0].lower()

    def test_checkpoint_info_includes_integrity_check_result(self, temp_project_dir, mock_config_with_integrity):
        """Test that checkpoint info includes integrity_check_result fields."""
        cm = CheckpointManager(temp_project_dir, config_hash="abc123", config=mock_config_with_integrity)

        # Create and save checkpoint
        cm.data = CheckpointData(
            version="2.1",
            created_at="2026-02-16T10:00:00",
            config_hash="abc123"
        )
        cm.save(stage="ANALYZE", stage_data={"keywords": ["test"]})

        # Get integrity status
        status = cm.get_integrity_status()

        # Verify all expected fields are present
        assert 'last_verified_at' in status
        assert 'all_backups_valid' in status
        assert status['last_verified_at'] is not None
        assert isinstance(status['all_backups_valid'], bool)


class TestCheckpointIntegrityConfig:
    """Test CheckpointIntegrityConfig dataclass."""

    def test_integrity_config_defaults(self):
        """Test default values for CheckpointIntegrityConfig."""
        from src.config.sections.infrastructure import CheckpointIntegrityConfig

        config = CheckpointIntegrityConfig()

        assert config.verify_on_save is False
        assert config.schedule_verification is False
        assert config.cron_expression == "0 * * * *"
        assert config.verify_backups is True

    def test_integrity_config_validation(self):
        """Test CheckpointIntegrityConfig validation."""
        from src.config.sections.infrastructure import CheckpointIntegrityConfig

        # Valid config
        config = CheckpointIntegrityConfig(
            verify_on_save=True,
            schedule_verification=True,
            cron_expression="30 * * * *",
            verify_backups=True
        )
        assert config.verify_on_save is True
        assert config.schedule_verification is True
        assert config.cron_expression == "30 * * * *"

        # Invalid cron expression
        with pytest.raises(ValueError):
            CheckpointIntegrityConfig(cron_expression="")
