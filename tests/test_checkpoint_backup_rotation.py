"""
Tests for checkpoint backup rotation (US-51-007).

Covers:
- AC1: Checkpoint save rotates up to N backup files (default 3)
- AC2: Each save rotates existing backups (2->3 deleted, 1->2, current->1, new backup)
- AC3: Backup rotation count is configurable via infrastructure config
- AC4: When checkpoint.json is corrupted, restore tries backup files in order
- AC5: Unit test verifies backup rotation creates correct number of files
- AC6: Unit test verifies restore from backup when primary checkpoint is corrupted
"""

import json
import logging
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData


def _make_checkpoint(stage: str, timestamp: str, extra: dict = None) -> dict:
    """Helper to create a valid checkpoint dict."""
    data = {
        "version": "2.0",
        "created_at": timestamp,
        "updated_at": timestamp,
        "last_completed_stage": stage,
        "config_hash": "testhash",
        "voiceover_path": "",
        "voiceover_hash": "",
        "analyze": {},
        "video_search": {},
        "caption": {},
        "match": {},
        "iterative_match": {},
        "download_segments": {},
    }
    if extra:
        data.update(extra)
    return data


class _FakePipelineConfig:
    """Minimal pipeline config stub for testing."""
    def __init__(self, backup_count=3):
        self.checkpoint_backup_count = backup_count


class _FakeConfig:
    """Minimal config stub for testing."""
    def __init__(self, backup_count=3):
        self.pipeline = _FakePipelineConfig(backup_count)


class TestBackupRotationCreatesFiles:
    """AC1/AC5: Checkpoint save rotates up to N backup files."""

    @pytest.mark.fast
    def test_four_saves_create_three_backups(self, tmp_path):
        """After 4 saves (with backup_count=3), all 3 backup slots should be filled."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))

        # Save 1: no backup (nothing to back up yet)
        # Save 2: backup.json created
        # Save 3: backup.json -> backup.1.json, new backup.json
        # Save 4: backup.1 -> backup.2, backup -> backup.1, new backup
        for i, stage in enumerate(["ANALYZE", "VIDEO_SEARCH", "CAPTION", "MATCH"]):
            manager.save(stage, {"step": i})

        # Primary checkpoint should exist
        assert (tmp_path / "checkpoint.json").exists()

        # All 3 backup slots should exist
        assert (tmp_path / "checkpoint.backup.json").exists()
        assert (tmp_path / "checkpoint.backup.1.json").exists()
        assert (tmp_path / "checkpoint.backup.2.json").exists()

    @pytest.mark.fast
    def test_first_save_creates_no_backup(self, tmp_path):
        """First save has nothing to back up."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))
        manager.save("ANALYZE", {"keywords": ["test"]})

        assert (tmp_path / "checkpoint.json").exists()
        # No backup yet since there was no previous checkpoint
        assert not (tmp_path / "checkpoint.backup.json").exists()

    @pytest.mark.fast
    def test_second_save_creates_one_backup(self, tmp_path):
        """Second save creates the primary backup."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))
        manager.save("ANALYZE", {"keywords": ["test"]})
        manager.save("VIDEO_SEARCH", {"video_ids": ["v1"]})

        assert (tmp_path / "checkpoint.json").exists()
        assert (tmp_path / "checkpoint.backup.json").exists()
        assert not (tmp_path / "checkpoint.backup.1.json").exists()

    @pytest.mark.fast
    def test_backup_count_caps_at_n(self, tmp_path):
        """Excess saves beyond N don't create extra backup files."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))

        # Perform 6 saves
        stages = ["ANALYZE", "VIDEO_SEARCH", "CAPTION", "MATCH",
                  "ITERATIVE_MATCH", "DOWNLOAD_SEGMENTS"]
        for i, stage in enumerate(stages):
            manager.save(stage, {"step": i})

        # Only 3 backups should exist (not 5)
        assert (tmp_path / "checkpoint.backup.json").exists()
        assert (tmp_path / "checkpoint.backup.1.json").exists()
        assert (tmp_path / "checkpoint.backup.2.json").exists()
        assert not (tmp_path / "checkpoint.backup.3.json").exists()


class TestBackupRotationOrder:
    """AC2: Each save rotates backups in correct order."""

    @pytest.mark.fast
    def test_rotation_preserves_history(self, tmp_path):
        """Backups should contain progressively older data."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))

        # Save stages sequentially
        manager.save("ANALYZE", {"stage_num": 1})
        manager.save("VIDEO_SEARCH", {"stage_num": 2})
        manager.save("CAPTION", {"stage_num": 3})
        manager.save("MATCH", {"stage_num": 4})

        # Primary should be MATCH (latest)
        primary = json.loads((tmp_path / "checkpoint.json").read_text())
        assert primary["last_completed_stage"] == "MATCH"

        # backup.json should be CAPTION (previous save)
        backup0 = json.loads((tmp_path / "checkpoint.backup.json").read_text())
        assert backup0["last_completed_stage"] == "CAPTION"

        # backup.1.json should be VIDEO_SEARCH
        backup1 = json.loads((tmp_path / "checkpoint.backup.1.json").read_text())
        assert backup1["last_completed_stage"] == "VIDEO_SEARCH"

        # backup.2.json should be ANALYZE (oldest)
        backup2 = json.loads((tmp_path / "checkpoint.backup.2.json").read_text())
        assert backup2["last_completed_stage"] == "ANALYZE"

    @pytest.mark.fast
    def test_oldest_backup_is_deleted_on_rotation(self, tmp_path):
        """When a new save occurs and N backups exist, oldest is dropped."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=2))

        manager.save("ANALYZE", {"n": 1})
        manager.save("VIDEO_SEARCH", {"n": 2})
        manager.save("CAPTION", {"n": 3})
        # At this point: backup.json=VIDEO_SEARCH, backup.1.json=ANALYZE

        manager.save("MATCH", {"n": 4})
        # backup.json=CAPTION, backup.1.json=VIDEO_SEARCH
        # ANALYZE should be gone

        backup1 = json.loads((tmp_path / "checkpoint.backup.1.json").read_text())
        assert backup1["last_completed_stage"] == "VIDEO_SEARCH"

        # Only 2 backups should exist
        assert not (tmp_path / "checkpoint.backup.2.json").exists()


class TestConfigurableBackupCount:
    """AC3: Backup rotation count is configurable."""

    @pytest.mark.fast
    def test_backup_count_1(self, tmp_path):
        """With count=1, only one backup should exist."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=1))

        manager.save("ANALYZE", {"n": 1})
        manager.save("VIDEO_SEARCH", {"n": 2})
        manager.save("CAPTION", {"n": 3})

        assert (tmp_path / "checkpoint.backup.json").exists()
        assert not (tmp_path / "checkpoint.backup.1.json").exists()

    @pytest.mark.fast
    def test_backup_count_5(self, tmp_path):
        """With count=5, up to 5 backups should exist."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=5))

        stages = ["ANALYZE", "VIDEO_SEARCH", "CAPTION", "MATCH",
                  "ITERATIVE_MATCH", "DOWNLOAD_SEGMENTS"]
        for i, stage in enumerate(stages):
            manager.save(stage, {"step": i})

        # 5 backups should exist (6 saves - 1 first save = 5 backups)
        for i in range(5):
            path = manager._get_backup_path(i)
            assert path.exists(), f"Expected backup at {path.name}"
        assert not (tmp_path / "checkpoint.backup.5.json").exists()

    @pytest.mark.fast
    def test_default_backup_count_without_config(self, tmp_path):
        """Without config, default backup count of 3 is used."""
        manager = CheckpointManager(tmp_path)
        assert manager._backup_count == 3

    @pytest.mark.fast
    def test_backup_count_from_config(self, tmp_path):
        """Backup count reads from config.pipeline.checkpoint_backup_count."""
        config = _FakeConfig(backup_count=7)
        manager = CheckpointManager(tmp_path, config=config)
        assert manager._backup_count == 7


class TestRestoreFromRotatedBackup:
    """AC4/AC6: When primary is corrupted, restore tries backup files in order."""

    @pytest.mark.fast
    def test_restore_from_primary_backup(self, tmp_path):
        """When main is corrupt, primary backup is used (existing behavior)."""
        # Set up corrupt main + valid primary backup
        (tmp_path / "checkpoint.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.json").write_text(
            json.dumps(_make_checkpoint("CAPTION", "2026-01-20T10:00:00"))
        )

        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))
        result = manager.load()

        assert result is not None
        assert result.last_completed_stage == "CAPTION"

    @pytest.mark.fast
    def test_restore_from_second_backup_when_primary_corrupt(self, tmp_path):
        """When main and primary backup are corrupt, backup.1 is tried."""
        (tmp_path / "checkpoint.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.json").write_text("{also corrupt!!")
        (tmp_path / "checkpoint.backup.1.json").write_text(
            json.dumps(_make_checkpoint(
                "VIDEO_SEARCH", "2026-01-20T08:00:00",
                {"video_search": {"video_ids": ["v1", "v2"]}}
            ))
        )

        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))
        result = manager.load()

        assert result is not None
        assert result.last_completed_stage == "VIDEO_SEARCH"
        assert result.video_search.get("video_ids") == ["v1", "v2"]

    @pytest.mark.fast
    def test_restore_from_third_backup_when_others_corrupt(self, tmp_path):
        """When main, backup, and backup.1 are corrupt, backup.2 is tried."""
        (tmp_path / "checkpoint.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.1.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.2.json").write_text(
            json.dumps(_make_checkpoint("ANALYZE", "2026-01-20T06:00:00"))
        )

        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))
        result = manager.load()

        assert result is not None
        assert result.last_completed_stage == "ANALYZE"

    @pytest.mark.fast
    def test_all_backups_corrupt_returns_none(self, tmp_path):
        """When all files are corrupt, load() returns None."""
        (tmp_path / "checkpoint.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.1.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.2.json").write_text("{corrupt!!")

        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))
        result = manager.load()

        assert result is None

    @pytest.mark.fast
    def test_restore_logs_rotated_backup_name(self, tmp_path, caplog):
        """Restoring from rotated backup logs which file was used."""
        (tmp_path / "checkpoint.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.1.json").write_text(
            json.dumps(_make_checkpoint("VIDEO_SEARCH", "2026-01-20T08:00:00"))
        )

        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))
        with caplog.at_level(logging.WARNING):
            result = manager.load()

        assert result is not None
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("checkpoint.backup.1.json" in msg for msg in warning_messages), \
            f"Expected rotated backup filename in log, got: {warning_messages}"

    @pytest.mark.fast
    def test_restore_copies_rotated_backup_to_main(self, tmp_path):
        """When restoring from rotated backup, main file is overwritten."""
        (tmp_path / "checkpoint.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.json").write_text("{corrupt!!")
        (tmp_path / "checkpoint.backup.1.json").write_text(
            json.dumps(_make_checkpoint("VIDEO_SEARCH", "2026-01-20T08:00:00"))
        )

        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))
        manager.load()

        # Main file should now be valid
        restored = json.loads((tmp_path / "checkpoint.json").read_text())
        assert restored["last_completed_stage"] == "VIDEO_SEARCH"


class TestClearRemovesRotatedBackups:
    """clear() should remove all rotated backup files."""

    @pytest.mark.fast
    def test_clear_removes_all_backups(self, tmp_path):
        """clear() removes primary checkpoint and all rotated backups."""
        manager = CheckpointManager(tmp_path, config=_FakeConfig(backup_count=3))

        # Create checkpoint with backups
        stages = ["ANALYZE", "VIDEO_SEARCH", "CAPTION", "MATCH"]
        for i, stage in enumerate(stages):
            manager.save(stage, {"step": i})

        # Verify files exist
        assert (tmp_path / "checkpoint.json").exists()
        assert (tmp_path / "checkpoint.backup.json").exists()
        assert (tmp_path / "checkpoint.backup.1.json").exists()
        assert (tmp_path / "checkpoint.backup.2.json").exists()

        # Clear
        manager.clear()

        # All should be gone
        assert not (tmp_path / "checkpoint.json").exists()
        assert not (tmp_path / "checkpoint.backup.json").exists()
        assert not (tmp_path / "checkpoint.backup.1.json").exists()
        assert not (tmp_path / "checkpoint.backup.2.json").exists()
