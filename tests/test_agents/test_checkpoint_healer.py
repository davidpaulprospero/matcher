"""Unit tests for CheckpointHealer."""

import json
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agents.healers.checkpoint import CheckpointHealer
from src.agents.base import HealerResult, HealerAction


class TestCheckpointHealerInit:
    """Test CheckpointHealer initialization."""

    @pytest.mark.fast
    def test_init_stores_config_and_project(self):
        """Test that CheckpointHealer stores config and project_dir."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/path/to/project")

        assert healer.config is config
        assert healer.project_dir == "/path/to/project"

    @pytest.mark.fast
    def test_init_has_correct_name(self):
        """Test that CheckpointHealer has correct name."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/tmp")

        assert healer.name == "checkpoint-healer"

    @pytest.mark.fast
    def test_init_has_checkpoint_file_constant(self):
        """Test that CheckpointHealer has CHECKPOINT_FILE constant."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/tmp")

        assert hasattr(healer, 'CHECKPOINT_FILE')
        assert healer.CHECKPOINT_FILE == "checkpoint.json"

    @pytest.mark.fast
    def test_init_has_backup_file_constant(self):
        """Test that CheckpointHealer has BACKUP_FILE constant."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/tmp")

        assert hasattr(healer, 'BACKUP_FILE')
        assert healer.BACKUP_FILE == "checkpoint.backup.json"

    @pytest.mark.fast
    def test_init_has_exception_types(self):
        """Test that CheckpointHealer has exception_types list."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/tmp")

        assert hasattr(healer, 'exception_types')
        assert json.JSONDecodeError in healer.exception_types


class TestCheckpointHealerCanHandle:
    """Test CheckpointHealer.can_handle() method."""

    @pytest.mark.fast
    def test_can_handle_json_decode_error(self):
        """Test that CheckpointHealer can handle JSONDecodeError."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/tmp")

        error = json.JSONDecodeError("Expecting value", "doc", 0)
        assert healer.can_handle(error, "LOAD") is True

    @pytest.mark.fast
    def test_can_handle_json_in_message(self):
        """Test that CheckpointHealer can handle errors with 'json' in message."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/tmp")

        error = Exception("JSON decode failed at line 42")
        assert healer.can_handle(error, "LOAD") is True

    @pytest.mark.fast
    def test_can_handle_checkpoint_in_message(self):
        """Test that CheckpointHealer can handle checkpoint-related errors."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/tmp")

        error = Exception("Checkpoint file is corrupted")
        assert healer.can_handle(error, "LOAD") is True

    @pytest.mark.fast
    def test_can_handle_corrupt_in_message(self):
        """Test that CheckpointHealer can handle corruption errors."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/tmp")

        error = Exception("File is corrupt or incomplete")
        assert healer.can_handle(error, "LOAD") is True

    @pytest.mark.fast
    def test_can_handle_hash_in_message(self):
        """Test that CheckpointHealer can handle hash mismatch errors."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/tmp")

        error = Exception("Hash mismatch: config has changed")
        assert healer.can_handle(error, "LOAD") is True

    @pytest.mark.fast
    def test_cannot_handle_unrelated_error(self):
        """Test that CheckpointHealer doesn't handle unrelated errors."""
        config = MagicMock()
        healer = CheckpointHealer(config, "/tmp")

        error = Exception("Rate limit exceeded")
        assert healer.can_handle(error, "LOAD") is False

        error = Exception("Disk full")
        assert healer.can_handle(error, "LOAD") is False


class TestCheckpointHealerRestoreFromBackup:
    """Test CheckpointHealer._restore_from_backup() method."""

    @pytest.mark.fast
    def test_restore_from_backup_success(self, tmp_path):
        """Test successful restore from backup."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create valid backup
        backup_data = {"stages": {"ANALYZE": {}, "DOWNLOAD": {}}}
        backup_path = project_dir / "checkpoint.backup.json"
        backup_path.write_text(json.dumps(backup_data))

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._restore_from_backup(Exception("json error"), state)

        assert result.success is True
        assert result.action == HealerAction.RESTORE
        assert (project_dir / "checkpoint.json").exists()

    @pytest.mark.fast
    def test_restore_from_backup_no_backup_exists(self, tmp_path):
        """Test restore fails gracefully when no backup exists."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._restore_from_backup(Exception("json error"), state)

        # Should fall through to start_fresh
        assert isinstance(result, HealerResult)

    @pytest.mark.fast
    def test_restore_from_backup_corrupted_backup(self, tmp_path):
        """Test restore handles corrupted backup."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create corrupted backup
        backup_path = project_dir / "checkpoint.backup.json"
        backup_path.write_text("{invalid json")

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._restore_from_backup(Exception("json error"), state)

        # Should fall through to start_fresh
        assert isinstance(result, HealerResult)


class TestCheckpointHealerRebuildCheckpoint:
    """Test CheckpointHealer._rebuild_checkpoint() method."""

    @pytest.mark.fast
    def test_rebuild_checkpoint_finds_transcriptions(self, tmp_path):
        """Test rebuild finds cached transcriptions."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create transcriptions cache
        trans_cache = project_dir / ".cache" / "transcriptions"
        trans_cache.mkdir(parents=True)
        (trans_cache / "test.json").write_text('{}')

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._rebuild_checkpoint(Exception("missing"), state)

        assert result.success is True
        assert "TRANSCRIBE" in result.details.get("cached_stages", [])

    @pytest.mark.fast
    def test_rebuild_checkpoint_finds_embeddings(self, tmp_path):
        """Test rebuild finds cached embeddings."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create embeddings cache
        embed_cache = project_dir / ".cache" / "embeddings"
        embed_cache.mkdir(parents=True)
        (embed_cache / "test.npy").write_bytes(b'\x00')

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._rebuild_checkpoint(Exception("missing"), state)

        assert result.success is True
        assert "EMBEDDINGS" in result.details.get("cached_stages", [])

    @pytest.mark.fast
    def test_rebuild_checkpoint_finds_scene_detection(self, tmp_path):
        """Test rebuild finds cached scene detection data."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create scene detection cache
        scene_cache = project_dir / ".cache" / "scene_detection"
        scene_cache.mkdir(parents=True)
        (scene_cache / "test.json").write_text('{}')

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._rebuild_checkpoint(Exception("missing"), state)

        assert result.success is True
        assert "SCENE_DETECTION" in result.details.get("cached_stages", [])

    @pytest.mark.fast
    def test_rebuild_checkpoint_no_cache(self, tmp_path):
        """Test rebuild with no cache directory."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._rebuild_checkpoint(Exception("missing"), state)

        # Should fall through to start_fresh
        assert isinstance(result, HealerResult)

    @pytest.mark.fast
    def test_rebuild_checkpoint_empty_cache(self, tmp_path):
        """Test rebuild with empty cache directories."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create empty cache directories
        (project_dir / ".cache" / "transcriptions").mkdir(parents=True)
        (project_dir / ".cache" / "embeddings").mkdir(parents=True)

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._rebuild_checkpoint(Exception("missing"), state)

        # Should fall through to start_fresh (empty dirs have no files)
        assert isinstance(result, HealerResult)


class TestCheckpointHealerHandleHashMismatch:
    """Test CheckpointHealer._handle_hash_mismatch() method."""

    @pytest.mark.fast
    def test_handle_hash_mismatch_continues(self, tmp_path):
        """Test that hash mismatch continues with current config."""
        config = MagicMock()
        healer = CheckpointHealer(config, str(tmp_path))
        state = MagicMock()

        result = healer._handle_hash_mismatch(Exception("hash mismatch"), state)

        assert result.success is True
        assert result.action == HealerAction.RETRY
        assert result.details.get("hash_mismatch_ignored") is True


class TestCheckpointHealerStartFresh:
    """Test CheckpointHealer._start_fresh() method."""

    @pytest.mark.fast
    def test_start_fresh_moves_corrupted(self, tmp_path):
        """Test that start_fresh moves corrupted checkpoint."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create corrupted checkpoint
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text("{bad json")

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._start_fresh(Exception("corrupt"), state)

        assert result.success is True
        assert result.details.get("fresh_start") is True
        assert not checkpoint_path.exists()
        assert (project_dir / "checkpoint.corrupted.json").exists()

    @pytest.mark.fast
    def test_start_fresh_no_checkpoint(self, tmp_path):
        """Test start_fresh when no checkpoint exists."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._start_fresh(Exception("missing"), state)

        assert result.success is True
        assert result.details.get("fresh_start") is True


class TestCheckpointHealerCreateBackup:
    """Test CheckpointHealer.create_backup() method."""

    @pytest.mark.fast
    def test_create_backup_success(self, tmp_path):
        """Test successful backup creation."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create valid checkpoint
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text('{"test": true}')

        healer = CheckpointHealer(config, str(project_dir))

        result = healer.create_backup()

        assert result is True
        assert (project_dir / "checkpoint.backup.json").exists()

    @pytest.mark.fast
    def test_create_backup_no_checkpoint(self, tmp_path):
        """Test create_backup returns False when no checkpoint exists."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = CheckpointHealer(config, str(project_dir))

        result = healer.create_backup()

        assert result is False

    @pytest.mark.fast
    def test_create_backup_preserves_content(self, tmp_path):
        """Test that backup preserves checkpoint content."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create checkpoint with specific content
        checkpoint_data = {"stages": {"ANALYZE": {"keywords": ["test"]}}}
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint_data))

        healer = CheckpointHealer(config, str(project_dir))
        healer.create_backup()

        # Verify backup content matches
        backup_path = project_dir / "checkpoint.backup.json"
        backup_content = json.loads(backup_path.read_text())
        assert backup_content == checkpoint_data


class TestCheckpointHealerFix:
    """Test CheckpointHealer.fix() method routing."""

    @pytest.mark.fast
    def test_fix_routes_json_decode_error(self, tmp_path):
        """Test that fix() routes JSONDecodeError correctly."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        error = json.JSONDecodeError("Expecting value", "doc", 0)
        result = healer.fix(error, state, "LOAD")

        assert isinstance(result, HealerResult)

    @pytest.mark.fast
    def test_fix_routes_corrupt_error(self, tmp_path):
        """Test that fix() routes corruption errors correctly."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer.fix(Exception("File is corrupt"), state, "LOAD")

        assert isinstance(result, HealerResult)

    @pytest.mark.fast
    def test_fix_routes_missing_error(self, tmp_path):
        """Test that fix() routes missing field errors correctly."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer.fix(Exception("Missing required field"), state, "LOAD")

        assert isinstance(result, HealerResult)

    @pytest.mark.fast
    def test_fix_routes_hash_mismatch(self, tmp_path):
        """Test that fix() routes hash mismatch errors correctly."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer.fix(Exception("Hash mismatch detected"), state, "LOAD")

        assert result.success is True
        assert result.details.get("hash_mismatch_ignored") is True


class TestCheckpointHealerIntegration:
    """Integration tests for CheckpointHealer."""

    @pytest.mark.fast
    def test_full_recovery_cycle(self, tmp_path):
        """Test full recovery cycle: create, corrupt, restore."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create valid checkpoint and backup
        checkpoint_data = {
            "stages": {"ANALYZE": {"keywords": ["test"]}, "DOWNLOAD": {}},
            "last_completed_stage": "DOWNLOAD"
        }
        checkpoint_path = project_dir / "checkpoint.json"
        backup_path = project_dir / "checkpoint.backup.json"

        checkpoint_path.write_text(json.dumps(checkpoint_data))
        backup_path.write_text(json.dumps(checkpoint_data))

        # Corrupt the main checkpoint
        checkpoint_path.write_text("{corrupted")

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        # Attempt to heal
        result = healer.fix(
            json.JSONDecodeError("test", "doc", 0),
            state,
            "LOAD"
        )

        assert result.success is True
        assert result.action == HealerAction.RESTORE

        # Verify checkpoint is restored
        restored_data = json.loads(checkpoint_path.read_text())
        assert restored_data == checkpoint_data

    @pytest.mark.fast
    def test_cascade_to_fresh_start(self, tmp_path):
        """Test cascade from restore to fresh start when backup is also bad."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create corrupted checkpoint
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text("{bad")

        # Create corrupted backup
        backup_path = project_dir / "checkpoint.backup.json"
        backup_path.write_text("{also bad")

        healer = CheckpointHealer(config, str(project_dir))
        state = MagicMock()

        result = healer.fix(
            json.JSONDecodeError("test", "doc", 0),
            state,
            "LOAD"
        )

        assert result.success is True
        assert result.details.get("fresh_start") is True
        assert not checkpoint_path.exists()
        assert (project_dir / "checkpoint.corrupted.json").exists()
