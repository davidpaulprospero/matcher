"""
Tests for checkpoint corruption recovery (US-005, Sprint 15).

Covers:
- AC1: load() falls back to backup when primary is corrupted JSON
- AC2: load() returns None when both primary and backup are corrupted
- AC3: save() creates backup of previous checkpoint before overwriting
- AC4: is_stale() correctly identifies checkpoints older than max_age_hours
- AC5: save() with non-serializable stage data handled gracefully
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData, STAGE_ORDER


class TestBackupFallbackOnCorruption:
    """AC1: load() falls back to backup when primary checkpoint.json is corrupted JSON."""

    @pytest.mark.fast
    def test_load_from_backup_when_primary_corrupted(self, tmp_path):
        """Verify data loaded from checkpoint.backup.json matches saved state."""
        # Create corrupted primary checkpoint
        primary = tmp_path / "checkpoint.json"
        primary.write_text("{corrupted json !!@#$%")

        # Create valid backup with known state
        backup = tmp_path / "checkpoint.backup.json"
        backup_state = {
            "version": "2.0",
            "created_at": "2026-01-15T10:30:00",
            "updated_at": "2026-01-15T12:00:00",
            "last_completed_stage": "VIDEO_SEARCH",
            "config_hash": "abc123",
            "voiceover_path": "/project/script.srt",
            "voiceover_hash": "deadbeef",
            "analyze": {"keywords": ["travel", "adventure"], "segment_count": 12},
            "video_search": {"video_ids": ["vid1", "vid2"]},
        }
        backup.write_text(json.dumps(backup_state))

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is not None
        assert result.last_completed_stage == "VIDEO_SEARCH"
        assert result.config_hash == "abc123"
        assert result.voiceover_path == "/project/script.srt"
        assert result.voiceover_hash == "deadbeef"
        assert result.analyze["keywords"] == ["travel", "adventure"]
        assert result.analyze["segment_count"] == 12
        assert result.video_search["video_ids"] == ["vid1", "vid2"]

    @pytest.mark.fast
    def test_backup_fallback_restores_primary_file(self, tmp_path):
        """Verify that after backup fallback, primary file is restored from backup."""
        primary = tmp_path / "checkpoint.json"
        primary.write_text("not valid json at all")

        backup = tmp_path / "checkpoint.backup.json"
        backup_data = {
            "version": "2.0",
            "created_at": "2026-01-15T10:00:00",
            "updated_at": "2026-01-15T10:00:00",
            "last_completed_stage": "CAPTION",
        }
        backup.write_text(json.dumps(backup_data))

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is not None
        assert result.last_completed_stage == "CAPTION"
        # Primary should now be restored from backup
        restored = json.loads(primary.read_text())
        assert restored["last_completed_stage"] == "CAPTION"

    @pytest.mark.fast
    def test_backup_fallback_with_empty_primary(self, tmp_path):
        """Verify backup fallback works when primary is empty (0 bytes)."""
        primary = tmp_path / "checkpoint.json"
        primary.write_text("")

        backup = tmp_path / "checkpoint.backup.json"
        backup_data = {
            "version": "2.0",
            "created_at": "2026-01-15T08:00:00",
            "updated_at": "2026-01-15T08:00:00",
            "last_completed_stage": "ANALYZE",
            "analyze": {"keywords": ["nature"]},
        }
        backup.write_text(json.dumps(backup_data))

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is not None
        assert result.last_completed_stage == "ANALYZE"
        assert result.analyze["keywords"] == ["nature"]

    @pytest.mark.fast
    def test_backup_fallback_with_truncated_primary(self, tmp_path):
        """Verify backup fallback works when primary is truncated JSON."""
        primary = tmp_path / "checkpoint.json"
        # Truncated JSON - common corruption from interrupted writes
        primary.write_text('{"version": "2.0", "last_completed_stage": "MAT')

        backup = tmp_path / "checkpoint.backup.json"
        backup_data = {
            "version": "2.0",
            "created_at": "2026-01-15T09:00:00",
            "updated_at": "2026-01-15T09:00:00",
            "last_completed_stage": "MATCH",
            "match": {"match_count": 15, "avg_confidence": 0.82},
        }
        backup.write_text(json.dumps(backup_data))

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is not None
        assert result.last_completed_stage == "MATCH"
        assert result.match["match_count"] == 15


class TestBothCorrupted:
    """AC2: load() returns None when both checkpoint.json and backup are corrupted."""

    @pytest.mark.fast
    def test_both_corrupted_json_returns_none(self, tmp_path):
        """Verify no exception raised, returns None gracefully."""
        primary = tmp_path / "checkpoint.json"
        primary.write_text("{broken json!!!}")

        backup = tmp_path / "checkpoint.backup.json"
        backup.write_text("also broken json @#$")

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is None
        assert manager.data is None

    @pytest.mark.fast
    def test_both_empty_returns_none(self, tmp_path):
        """Verify None when both files are empty."""
        primary = tmp_path / "checkpoint.json"
        primary.write_text("")

        backup = tmp_path / "checkpoint.backup.json"
        backup.write_text("")

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is None

    @pytest.mark.fast
    def test_primary_corrupted_no_backup_returns_none(self, tmp_path):
        """Verify None when primary is corrupted and backup doesn't exist."""
        primary = tmp_path / "checkpoint.json"
        primary.write_text("not json!!")

        # No backup file at all
        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is None

    @pytest.mark.fast
    def test_both_json_arrays_returns_none(self, tmp_path):
        """Verify None when both files contain JSON arrays (not dicts)."""
        primary = tmp_path / "checkpoint.json"
        primary.write_text('["not", "a", "dict", "structure", "at", "all"]')

        backup = tmp_path / "checkpoint.backup.json"
        backup.write_text('[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]')

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is None

    @pytest.mark.fast
    def test_both_too_small_returns_none(self, tmp_path):
        """Verify None when both files are too small (< 10 chars)."""
        primary = tmp_path / "checkpoint.json"
        primary.write_text("{}")  # 2 chars

        backup = tmp_path / "checkpoint.backup.json"
        backup.write_text("{}")  # 2 chars

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is None


class TestSaveCreatesBackup:
    """AC3: save() creates backup of previous checkpoint before overwriting."""

    @pytest.mark.fast
    def test_backup_contains_pre_save_state(self, tmp_path):
        """Verify checkpoint.backup.json contains the pre-save state after save() call."""
        manager = CheckpointManager(tmp_path, config_hash="hash1")

        # First save: ANALYZE stage with keywords
        manager.save("ANALYZE", {"keywords": ["python", "coding"], "segment_count": 5})

        # Read what's in checkpoint.json after first save
        first_state = json.loads((tmp_path / "checkpoint.json").read_text())
        assert first_state["last_completed_stage"] == "ANALYZE"

        # Second save: VIDEO_SEARCH stage
        manager.save("VIDEO_SEARCH", {"video_ids": ["vid1"]})

        # Backup should contain the FIRST save's state (ANALYZE)
        assert (tmp_path / "checkpoint.backup.json").exists()
        backup_data = json.loads((tmp_path / "checkpoint.backup.json").read_text())
        assert backup_data["last_completed_stage"] == "ANALYZE"
        assert backup_data["analyze"]["keywords"] == ["python", "coding"]

        # Current checkpoint should have VIDEO_SEARCH
        current_data = json.loads((tmp_path / "checkpoint.json").read_text())
        assert current_data["last_completed_stage"] == "VIDEO_SEARCH"

    @pytest.mark.fast
    def test_backup_updates_on_each_save(self, tmp_path):
        """Verify backup is updated on each successive save."""
        manager = CheckpointManager(tmp_path)

        # Save three stages sequentially
        manager.save("ANALYZE", {"keywords": ["a"]})
        manager.save("VIDEO_SEARCH", {"count": 10})
        manager.save("CAPTION", {"fetched": 5})

        # Backup should contain the state from BEFORE the last save (VIDEO_SEARCH stage)
        backup_data = json.loads((tmp_path / "checkpoint.backup.json").read_text())
        assert backup_data["last_completed_stage"] == "VIDEO_SEARCH"

        # Current should be CAPTION
        current = json.loads((tmp_path / "checkpoint.json").read_text())
        assert current["last_completed_stage"] == "CAPTION"

    @pytest.mark.fast
    def test_first_save_no_backup(self, tmp_path):
        """Verify no backup exists after first save (nothing to back up)."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})

        # No backup should exist after the very first save
        assert not (tmp_path / "checkpoint.backup.json").exists()

    @pytest.mark.fast
    def test_save_intermediate_also_creates_backup(self, tmp_path):
        """Verify save_intermediate() creates backup when rotation interval allows it."""
        manager = CheckpointManager(tmp_path)
        # US-85-003: Set interval to 0 so intermediate save always rotates
        manager._min_rotation_interval = 0
        manager.save("ANALYZE", {"keywords": ["test"]})

        # Now call save_intermediate, which should back up the current state
        manager.save_intermediate("VIDEO_SEARCH", {"progress": 50})

        assert (tmp_path / "checkpoint.backup.json").exists()
        backup_data = json.loads((tmp_path / "checkpoint.backup.json").read_text())
        # Backup should have the ANALYZE state (before intermediate save)
        assert backup_data["last_completed_stage"] == "ANALYZE"


class TestIsStaleThresholds:
    """AC4: is_stale() correctly identifies checkpoints older than max_age_hours."""

    @pytest.mark.fast
    def test_25_hour_old_checkpoint_is_stale(self, tmp_path):
        """Verify True for 25-hour-old checkpoint with default 24h threshold."""
        manager = CheckpointManager(tmp_path)
        old_time = datetime.now() - timedelta(hours=25)
        manager.data = CheckpointData(
            updated_at=old_time.isoformat()
        )

        assert manager.is_stale(max_age_hours=24.0) is True

    @pytest.mark.fast
    def test_23_hour_old_checkpoint_is_not_stale(self, tmp_path):
        """Verify False for 23-hour-old checkpoint with default 24h threshold."""
        manager = CheckpointManager(tmp_path)
        recent_time = datetime.now() - timedelta(hours=23)
        manager.data = CheckpointData(
            updated_at=recent_time.isoformat()
        )

        assert manager.is_stale(max_age_hours=24.0) is False

    @pytest.mark.fast
    def test_exactly_24_hour_boundary(self, tmp_path):
        """Verify behavior right at the 24-hour boundary (slightly over)."""
        manager = CheckpointManager(tmp_path)
        # Exactly 24 hours and 1 minute ago
        boundary_time = datetime.now() - timedelta(hours=24, minutes=1)
        manager.data = CheckpointData(
            updated_at=boundary_time.isoformat()
        )

        assert manager.is_stale(max_age_hours=24.0) is True

    @pytest.mark.fast
    def test_just_under_24_hour_boundary(self, tmp_path):
        """Verify behavior just under the 24-hour boundary."""
        manager = CheckpointManager(tmp_path)
        # 23 hours and 59 minutes ago
        boundary_time = datetime.now() - timedelta(hours=23, minutes=59)
        manager.data = CheckpointData(
            updated_at=boundary_time.isoformat()
        )

        assert manager.is_stale(max_age_hours=24.0) is False

    @pytest.mark.fast
    def test_custom_max_age_1_hour(self, tmp_path):
        """Verify is_stale with custom 1-hour threshold."""
        manager = CheckpointManager(tmp_path)
        two_hours_ago = datetime.now() - timedelta(hours=2)
        manager.data = CheckpointData(
            updated_at=two_hours_ago.isoformat()
        )

        assert manager.is_stale(max_age_hours=1.0) is True

    @pytest.mark.fast
    def test_custom_max_age_1_hour_not_stale(self, tmp_path):
        """Verify is_stale with custom 1-hour threshold for fresh checkpoint."""
        manager = CheckpointManager(tmp_path)
        half_hour_ago = datetime.now() - timedelta(minutes=30)
        manager.data = CheckpointData(
            updated_at=half_hour_ago.isoformat()
        )

        assert manager.is_stale(max_age_hours=1.0) is False

    @pytest.mark.fast
    def test_stale_uses_created_at_when_no_updated_at(self, tmp_path):
        """Verify is_stale falls back to created_at when updated_at is empty."""
        manager = CheckpointManager(tmp_path)
        old_time = datetime.now() - timedelta(hours=48)
        manager.data = CheckpointData(
            updated_at="",
            created_at=old_time.isoformat()
        )

        assert manager.is_stale(max_age_hours=24.0) is True

    @pytest.mark.fast
    def test_no_timestamp_is_stale(self, tmp_path):
        """Verify checkpoint with no timestamps is considered stale."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            updated_at="",
            created_at=""
        )

        assert manager.is_stale() is True


class TestNonSerializableStageData:
    """AC5: save() with stage data containing non-serializable objects."""

    @pytest.mark.fast
    def test_save_with_datetime_in_stage_data(self, tmp_path):
        """Verify datetime objects are converted via default=str, no crash."""
        manager = CheckpointManager(tmp_path)
        stage_data = {
            "started_at": datetime(2026, 1, 15, 10, 30, 0),
            "keywords": ["test"],
            "segment_count": 5,
        }

        # Should NOT raise - json.dump uses default=str
        manager.save("ANALYZE", stage_data)

        # Verify file was saved
        assert manager.checkpoint_path.exists()
        saved = json.loads(manager.checkpoint_path.read_text())
        assert "2026-01-15" in saved["analyze"]["started_at"]

    @pytest.mark.fast
    def test_save_with_path_objects_in_stage_data(self, tmp_path):
        """Verify Path objects are converted via default=str, no crash."""
        manager = CheckpointManager(tmp_path)
        stage_data = {
            "video_ids": ["vid1", "vid2"],
            "output_dir": Path("/output/project"),
        }

        # Should NOT raise
        manager.save("VIDEO_SEARCH", stage_data)

        saved = json.loads(manager.checkpoint_path.read_text())
        # Path objects should be serialized as strings
        assert "vid1" in saved["video_search"]["video_ids"][0]

    @pytest.mark.fast
    def test_save_with_set_in_stage_data(self, tmp_path):
        """Verify set objects are converted via default=str, no crash."""
        manager = CheckpointManager(tmp_path)
        stage_data = {
            "unique_keywords": {"travel", "adventure", "nature"},
            "count": 3,
        }

        # default=str converts sets to their string representation
        manager.save("ANALYZE", stage_data)

        assert manager.checkpoint_path.exists()

    @pytest.mark.fast
    def test_save_with_bytes_in_stage_data(self, tmp_path):
        """Verify bytes objects are converted via default=str, no crash."""
        manager = CheckpointManager(tmp_path)
        stage_data = {
            "file_hash": b"\xde\xad\xbe\xef",
            "content_type": "video/mp4",
        }

        # Should NOT raise
        manager.save("VIDEO_SEARCH", stage_data)

        assert manager.checkpoint_path.exists()

    @pytest.mark.fast
    def test_save_with_mixed_non_serializable_types(self, tmp_path):
        """Verify mixed non-serializable types all handled gracefully."""
        manager = CheckpointManager(tmp_path)
        stage_data = {
            "timestamp": datetime.now(),
            "path": Path("/some/path"),
            "normal_data": {"key": "value"},
            "numbers": [1, 2.5, 3],
            "nested": {
                "inner_date": datetime(2026, 6, 15),
                "inner_path": Path("/inner/path"),
            },
        }

        # Should NOT raise
        manager.save("ANALYZE", stage_data)

        # Verify the file is valid JSON
        saved = json.loads(manager.checkpoint_path.read_text())
        assert saved["last_completed_stage"] == "ANALYZE"
        assert saved["analyze"]["normal_data"] == {"key": "value"}
        assert saved["analyze"]["numbers"] == [1, 2.5, 3]

    @pytest.mark.fast
    def test_save_preserves_serializable_data_alongside_non_serializable(self, tmp_path):
        """Verify that serializable data is preserved correctly even when non-serializable types present."""
        manager = CheckpointManager(tmp_path)
        stage_data = {
            "keywords": ["python", "coding", "tutorial"],
            "segment_count": 42,
            "avg_confidence": 0.87,
            "metadata": {"source": "youtube", "language": "en"},
            "non_serializable_timestamp": datetime(2026, 1, 28, 14, 30),
        }

        manager.save("ANALYZE", stage_data)

        saved = json.loads(manager.checkpoint_path.read_text())
        assert saved["analyze"]["keywords"] == ["python", "coding", "tutorial"]
        assert saved["analyze"]["segment_count"] == 42
        assert saved["analyze"]["avg_confidence"] == 0.87
        assert saved["analyze"]["metadata"] == {"source": "youtube", "language": "en"}
