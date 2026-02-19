"""
Tests for enhanced checkpoint validation (US-106-004).

Tests:
- AC1: Version compatibility validation with CURRENT_CHECKPOINT_VERSION
- AC2: Stage data fields validation (non-empty dicts when stage complete)
- AC3: Migration logic for version upgrades (v1 -> v2)
- AC4: Corrupted checkpoint file handling (JSON parse errors)
- AC5: Backup before validation/migration
- AC6: Tests for various invalid states
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import (
    CheckpointManager, CheckpointData, STAGE_ORDER,
    CURRENT_CHECKPOINT_VERSION
)
from src.checkpoint_migrator import CheckpointMigrator


# ============================================================================
# AC1: Version compatibility validation
# ============================================================================

class TestVersionCompatibilityValidation:
    """AC1: Test version compatibility validation with CURRENT_CHECKPOINT_VERSION"""

    @pytest.mark.fast
    def test_validate_warns_about_old_version(self):
        """Checkpoint with old version triggers migration warning"""
        data = CheckpointData(
            version="0.9",
            last_completed_stage="MATCH",
            match={"match_count": 5}
        )

        warnings = data.validate()

        # Should warn about version migration
        assert any("0.9" in w and "migrate" in w.lower() for w in warnings)

    @pytest.mark.fast
    def test_validate_warns_about_v1_version(self):
        """Checkpoint with v1.0 triggers migration warning"""
        data = CheckpointData(
            version="1.0",
            last_completed_stage="MATCH",
            match={"match_count": 5}
        )

        warnings = data.validate()

        # Should warn about version migration
        assert any("1.0" in w and "migrate" in w.lower() for w in warnings)

    @pytest.mark.fast
    def test_validate_passes_with_current_version(self):
        """Checkpoint with current version doesn't warn about migration"""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            match={"match_count": 5}
        )

        warnings = data.validate()

        # Should NOT warn about migration
        migration_warnings = [w for w in warnings if "migrate" in w.lower()]
        assert len(migration_warnings) == 0

    @pytest.mark.fast
    def test_validate_fails_with_incompatible_version(self):
        """Checkpoint with incompatible version fails validation"""
        data = CheckpointData(
            version="99.0",  # Far future version
            last_completed_stage="MATCH",
            match={"match_count": 5}
        )

        warnings = data.validate()

        # Should warn about incompatible version
        assert any("not compatible" in w.lower() for w in warnings)


# ============================================================================
# AC2: Stage data fields validation (non-empty dicts)
# ============================================================================

class TestStageDataValidation:
    """AC2: Validate that stage data fields are non-empty dicts when stage complete"""

    @pytest.mark.fast
    def test_validate_warns_about_empty_stage_data(self):
        """Stage marked complete but with empty data dict triggers warning"""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            analyze={"keywords": ["test"]},  # Has data
            video_search={"video_ids": ["vid1"]},  # Has data
            match={},  # EMPTY - should warn
        )

        warnings = data.validate()

        # Should warn about no data (empty dict)
        assert any("no data" in w.lower() and "match" in w.lower() for w in warnings)

    @pytest.mark.fast
    def test_validate_warns_about_missing_stage_data(self):
        """Stage marked complete but with no data triggers warning"""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="CAPTION",
            analyze={"keywords": ["test"]},
            video_search={"video_ids": ["vid1"]},
            caption={},  # Empty
        )

        warnings = data.validate()

        # Should warn about missing caption data
        assert any("caption" in w.lower() and ("no data" in w.lower() or "empty" in w.lower()) for w in warnings)

    @pytest.mark.fast
    def test_validate_warns_about_non_dict_stage_data(self):
        """Stage data that is not a dict triggers warning"""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            match="not a dict"  # Wrong type
        )

        warnings = data.validate()

        # Should warn about type mismatch
        assert any("str" in w.lower() or "dict" in w.lower() for w in warnings)

    @pytest.mark.fast
    def test_validate_passes_with_populated_stage_data(self):
        """Stage with populated data passes validation"""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            analyze={"keywords": ["test"]},
            video_search={"video_ids": ["vid1"]},
            caption={"caption_count": 3},
            match={"match_count": 5, "matches": []}
        )

        warnings = data.validate()

        # Should NOT warn about stage data
        stage_warnings = [w for w in warnings if "stage" in w.lower() and ("empty" in w.lower() or "no data" in w.lower())]
        assert len(stage_warnings) == 0


# ============================================================================
# AC3: Migration logic for version upgrades (v1 -> v2)
# ============================================================================

class TestCheckpointMigration:
    """AC3: Test checkpoint migration logic for version upgrades"""

    @pytest.mark.fast
    def test_migrator_migrates_0_9_to_1_0(self):
        """Migrator correctly handles v0.9 -> v1.0"""
        migrator = CheckpointMigrator()

        old_data = {
            "version": "0.9",
            "created_at": "2024-01-01T00:00:00",
            "last_completed_stage": "DOWNLOAD",
            "ANALYZE": {"keywords": ["test"]}
        }

        migrated = migrator.migrate(old_data, "0.9", "1.0")

        assert migrated["version"] == "1.0"
        assert "analyze" in migrated  # Should be lowercase

    @pytest.mark.fast
    def test_migrator_migrates_1_0_to_2_0(self):
        """Migrator correctly handles v1.0 -> v2.0"""
        migrator = CheckpointMigrator()

        # Use a standard 11-char YouTube video ID
        old_data = {
            "version": "1.0",
            "created_at": "2024-01-01T00:00:00",
            "last_completed_stage": "DOWNLOAD",
            "download": {"downloaded_videos": [{"url": "https://youtube.com/watch?v=dQw4w9WgXcQ"}]}
        }

        migrated = migrator.migrate(old_data, "1.0", "2.0")

        assert migrated["version"] == "2.0"
        assert "video_search" in migrated
        assert "dQw4w9WgXcQ" in migrated["video_search"].get("video_ids", [])

    @pytest.mark.fast
    def test_migrator_chains_multiple_versions(self):
        """Migrator chains migrations correctly"""
        migrator = CheckpointMigrator()

        old_data = {
            "version": "0.9",
            "created_at": "2024-01-01T00:00:00",
            "last_completed_stage": "DOWNLOAD",
            "DOWNLOAD": {"downloaded_videos": [{"url": "https://youtube.com/watch?v=abc123"}]},
            "ANALYZE": {"keywords": ["test"]}
        }

        migrated = migrator.migrate(old_data, "0.9", "2.0")

        assert migrated["version"] == "2.0"
        assert "analyze" in migrated
        assert "video_search" in migrated

    @pytest.mark.fast
    def test_migrator_handles_unknown_stage(self):
        """Migrator handles unknown stage names gracefully"""
        migrator = CheckpointMigrator()

        old_data = {
            "version": "1.0",
            "created_at": "2024-01-01T00:00:00",
            "last_completed_stage": "UNKNOWN_STAGE"
        }

        migrated = migrator.migrate(old_data, "1.0", "2.0")

        # Unknown stage should be reset to empty
        assert migrated["last_completed_stage"] == ""


# ============================================================================
# AC4: Corrupted checkpoint file handling
# ============================================================================

class TestCorruptedCheckpointHandling:
    """AC4: Test detection and handling of corrupted checkpoint files"""

    @pytest.mark.fast
    def test_load_handles_invalid_json(self, tmp_path):
        """Checkpoint manager handles invalid JSON gracefully"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Write invalid JSON
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text("this is not valid json {")

        manager = CheckpointManager(project_dir)
        loaded = manager.load()

        # Should return None for corrupt file
        assert loaded is None

    @pytest.mark.fast
    def test_load_handles_empty_file(self, tmp_path):
        """Checkpoint manager handles empty file gracefully"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Write empty file
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text("")

        manager = CheckpointManager(project_dir)
        loaded = manager.load()

        # Should return None for empty file
        assert loaded is None

    @pytest.mark.fast
    def test_load_handles_truncated_json(self, tmp_path):
        """Checkpoint manager handles truncated JSON gracefully"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Write truncated JSON
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text('{"version": "2.0", "last_completed_stage": "MATCH", "match": {')

        manager = CheckpointManager(project_dir)
        loaded = manager.load()

        # Should return None for truncated JSON
        assert loaded is None

    @pytest.mark.fast
    def test_load_handles_non_dict_json(self, tmp_path):
        """Checkpoint manager handles non-dict JSON gracefully"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Write JSON that is not a dict
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text('"just a string"')

        manager = CheckpointManager(project_dir)
        loaded = manager.load()

        # Should return None for non-dict JSON
        assert loaded is None


# ============================================================================
# AC5: Backup before validation/migration
# ============================================================================

class TestBackupBeforeValidation:
    """AC5: Test backup creation before validation/migration"""

    @pytest.mark.fast
    def test_backup_created_before_migration(self, tmp_path):
        """Backup is created before migration modifies checkpoint"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create old version checkpoint
        checkpoint_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_completed_stage": "MATCH",
            "match": {"match_count": 5}
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(project_dir)
        manager.load()

        # Should have created pre-migration backup
        pre_migration_backups = list(project_dir.glob("checkpoint.pre_migration_*.json"))
        assert len(pre_migration_backups) > 0

    @pytest.mark.fast
    def test_backup_created_before_healing(self, tmp_path):
        """Backup is created before healing modifies checkpoint"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create checkpoint with corruption that needs healing - string instead of dict
        checkpoint_data = {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "last_completed_stage": "MATCH",
            "match": "string instead of dict"  # This will trigger healing
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(project_dir)
        manager.load()

        # Should have created pre-validation backup
        pre_validation_backups = list(project_dir.glob("checkpoint.pre_validation_*.json"))
        assert len(pre_validation_backups) > 0

    @pytest.mark.fast
    def test_cleanup_removes_old_pre_modification_backups(self, tmp_path):
        """Old pre-modification backups are cleaned up"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create some old backup files (need to be older by modifying mtime)
        import time
        old_backup = project_dir / "checkpoint.pre_migration_1000000000.json"
        old_backup.write_text('{"version": "2.0"}')

        # Create a manager and create a recent backup to test cleanup
        manager = CheckpointManager(project_dir)

        # Manually trigger the backup logic to set project_dir context
        # Then create multiple old backups
        recent_backup = project_dir / "checkpoint.pre_migration_9999999999.json"
        recent_backup.write_text('{"version": "2.0"}')

        old_backup2 = project_dir / "checkpoint.pre_migration_9999999998.json"
        old_backup2.write_text('{"version": "2.0"}')

        # Trigger cleanup with keep=2 (should remove oldest)
        manager._cleanup_old_backups(prefix="checkpoint.pre_migration_", keep=2)

        # Oldest backup should be removed
        assert not old_backup.exists()
        # Recent ones should remain
        assert recent_backup.exists()
        assert old_backup2.exists()


# ============================================================================
# AC6: Tests for various invalid states
# ============================================================================

class TestVariousInvalidStates:
    """AC6: Test checkpoint validation with various invalid states"""

    @pytest.mark.fast
    def test_validate_with_all_empty_stages(self):
        """Checkpoint with all empty stages triggers appropriate warnings"""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            analyze={},
            video_search={},
            caption={},
            match={},
            iterative_match={}
        )

        warnings = data.validate()

        # Should have multiple warnings about empty stages
        assert len(warnings) > 0

    @pytest.mark.fast
    def test_validate_with_invalid_stage_name(self):
        """Checkpoint with invalid stage name triggers warning"""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="INVALID_STAGE"
        )

        warnings = data.validate()

        # Should warn about unknown stage
        assert any("unknown" in w.lower() for w in warnings)

    @pytest.mark.fast
    def test_validate_with_no_last_completed_stage(self):
        """Checkpoint with no last_completed_stage is valid but minimal"""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage=""
        )

        warnings = data.validate()

        # Should return empty warnings (valid but no progress)
        assert len(warnings) == 0

    @pytest.mark.fast
    def test_validate_with_string_stage_data(self):
        """Checkpoint with string instead of dict for stage data"""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            match="string instead of dict"
        )

        warnings = data.validate()

        # Should warn about type mismatch
        assert any("str" in w.lower() for w in warnings)

    @pytest.mark.fast
    def test_validate_with_list_stage_data(self):
        """Checkpoint with list instead of dict for stage data"""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            match=[{"item": 1}, {"item": 2}]
        )

        warnings = data.validate()

        # Should warn about type mismatch
        assert any("list" in w.lower() for w in warnings)


# ============================================================================
# Integration tests
# ============================================================================

class TestValidationMigrationIntegration:
    """Integration tests for validation and migration working together"""

    @pytest.mark.fast
    def test_full_validation_flow(self, tmp_path):
        """Test full validation flow with migration"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create old version checkpoint needing migration
        checkpoint_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_completed_stage": "MATCH",
            "match": {"match_count": 5}
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(project_dir)
        loaded = manager.load()

        # Should have migrated
        assert loaded.version == CURRENT_CHECKPOINT_VERSION

    @pytest.mark.fast
    def test_validation_reports_all_issues(self):
        """Validation reports all issues, not just the first one"""
        data = CheckpointData(
            version="0.9",  # Old version
            last_completed_stage="MATCH",
            match={}  # Empty
        )

        warnings = data.validate()

        # Should have both version warning AND empty stage warning
        assert len(warnings) >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
