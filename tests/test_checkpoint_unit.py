"""
Unit tests for checkpoint module (pytest format).

Complements existing test_checkpoint.py integration tests.
Tests checkpoint dataclasses, manager initialization, and utilities.
"""

import pytest
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import (
    SavedKeywords,
    CheckpointData,
    CheckpointManager,
    STAGE_ORDER
)


class TestSavedKeywordsDataclass:
    """Test SavedKeywords dataclass."""

    def test_create_with_minimal_fields(self):
        """Test creating SavedKeywords with minimal fields."""
        keywords = SavedKeywords()

        assert keywords.name == ""
        assert keywords.keywords == []
        assert keywords.num_keywords == 0

    def test_create_with_all_fields(self):
        """Test creating SavedKeywords with all fields."""
        keywords = SavedKeywords(
            name="travel_preset",
            created_at="2025-01-08T12:00:00",
            voiceover_hash="abc123",
            keywords=["paris", "tokyo", "travel"],
            topic_context="travel vlog",
            entities=[{"name": "Paris", "type": "LOCATION"}],
            num_keywords=3
        )

        assert keywords.name == "travel_preset"
        assert len(keywords.keywords) == 3
        assert keywords.num_keywords == 3
        assert keywords.topic_context == "travel vlog"

    def test_to_dict_conversion(self):
        """Test converting to dictionary."""
        keywords = SavedKeywords(
            name="test",
            keywords=["kw1", "kw2"],
            num_keywords=2
        )

        data = keywords.to_dict()

        assert isinstance(data, dict)
        assert "name" in data
        assert "keywords" in data
        assert data["name"] == "test"

    def test_from_dict_conversion(self):
        """Test creating from dictionary."""
        data = {
            "name": "preset1",
            "keywords": ["keyword1", "keyword2"],
            "num_keywords": 2,
            "topic_context": "test topic"
        }

        keywords = SavedKeywords.from_dict(data)

        assert keywords.name == "preset1"
        assert len(keywords.keywords) == 2
        assert keywords.topic_context == "test topic"

    def test_round_trip_serialization(self):
        """Test to_dict -> from_dict round trip."""
        original = SavedKeywords(
            name="original",
            keywords=["a", "b", "c"],
            num_keywords=3,
            topic_context="context"
        )

        data = original.to_dict()
        restored = SavedKeywords.from_dict(data)

        assert restored.name == original.name
        assert restored.keywords == original.keywords
        assert restored.num_keywords == original.num_keywords


class TestCheckpointDataDataclass:
    """Test CheckpointData dataclass."""

    def test_create_default(self):
        """Test creating with defaults."""
        checkpoint = CheckpointData()

        assert checkpoint.version == "1.0"
        assert checkpoint.last_completed_stage == ""
        assert checkpoint.analyze == {}
        assert checkpoint.download == {}

    def test_create_with_stage_data(self):
        """Test creating with stage data."""
        checkpoint = CheckpointData(
            last_completed_stage="DOWNLOAD",
            download={"videos_count": 50, "duration": 3600}
        )

        assert checkpoint.last_completed_stage == "DOWNLOAD"
        assert checkpoint.download["videos_count"] == 50

    def test_all_stage_fields_exist(self):
        """Test all expected stage fields are present."""
        checkpoint = CheckpointData()

        # Verify all stage storage fields
        assert hasattr(checkpoint, "analyze")
        assert hasattr(checkpoint, "entity_images")
        assert hasattr(checkpoint, "entity_videos")
        assert hasattr(checkpoint, "download")
        assert hasattr(checkpoint, "stock")
        assert hasattr(checkpoint, "remix")
        assert hasattr(checkpoint, "transcribe")
        assert hasattr(checkpoint, "match")

    def test_to_dict_includes_all_fields(self):
        """Test to_dict includes all fields."""
        checkpoint = CheckpointData(
            config_hash="hash123",
            voiceover_path="/path/to/vo.srt"
        )

        data = checkpoint.to_dict()

        assert "version" in data
        assert "config_hash" in data
        assert "voiceover_path" in data
        assert "analyze" in data

    def test_from_dict_with_partial_data(self):
        """Test from_dict handles partial data."""
        data = {
            "version": "1.0",
            "last_completed_stage": "TRANSCRIBE",
            "unknown_field": "should be ignored"
        }

        checkpoint = CheckpointData.from_dict(data)

        assert checkpoint.version == "1.0"
        assert checkpoint.last_completed_stage == "TRANSCRIBE"
        # Unknown fields should be ignored
        assert not hasattr(checkpoint, "unknown_field")


class TestCheckpointManagerInit:
    """Test CheckpointManager initialization."""

    def test_create_manager(self, tmp_path):
        """Test creating checkpoint manager."""
        manager = CheckpointManager(
            project_dir=tmp_path,
            config_hash="test_hash"
        )

        assert manager.project_dir == tmp_path
        assert manager.config_hash == "test_hash"
        assert manager.data is None

    def test_checkpoint_paths(self, tmp_path):
        """Test checkpoint file paths."""
        manager = CheckpointManager(project_dir=tmp_path)

        assert manager.checkpoint_path == tmp_path / "checkpoint.json"
        assert manager.backup_path == tmp_path / "checkpoint.backup.json"

    def test_exists_no_file(self, tmp_path):
        """Test exists() returns False when no checkpoint."""
        manager = CheckpointManager(project_dir=tmp_path)

        assert not manager.exists()

    def test_exists_with_file(self, tmp_path):
        """Test exists() returns True when checkpoint exists."""
        # Create checkpoint file
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('{"version": "1.0"}')

        manager = CheckpointManager(project_dir=tmp_path)

        assert manager.exists()


class TestStageOrder:
    """Test STAGE_ORDER constant."""

    def test_stage_order_exists(self):
        """Test STAGE_ORDER is defined."""
        assert STAGE_ORDER is not None
        assert isinstance(STAGE_ORDER, (list, tuple))

    def test_stage_order_has_expected_stages(self):
        """Test STAGE_ORDER contains expected stages."""
        expected_stages = ["ANALYZE", "DOWNLOAD", "TRANSCRIBE", "MATCH", "OUTPUT"]

        for stage in expected_stages:
            assert stage in STAGE_ORDER

    def test_stage_order_is_sequential(self):
        """Test stages are in correct order."""
        # ANALYZE should come before DOWNLOAD
        assert STAGE_ORDER.index("ANALYZE") < STAGE_ORDER.index("DOWNLOAD")

        # DOWNLOAD should come before TRANSCRIBE
        assert STAGE_ORDER.index("DOWNLOAD") < STAGE_ORDER.index("TRANSCRIBE")

        # TRANSCRIBE should come before MATCH
        assert STAGE_ORDER.index("TRANSCRIBE") < STAGE_ORDER.index("MATCH")

        # MATCH should come before OUTPUT
        assert STAGE_ORDER.index("MATCH") < STAGE_ORDER.index("OUTPUT")


class TestCheckpointAge:
    """Test checkpoint age tracking."""

    def test_is_stale_no_data(self, tmp_path):
        """Test is_stale() with no checkpoint data."""
        manager = CheckpointManager(project_dir=tmp_path)

        assert not manager.is_stale()

    def test_is_stale_recent_checkpoint(self, tmp_path):
        """Test is_stale() with recent checkpoint."""
        manager = CheckpointManager(project_dir=tmp_path)
        manager.data = CheckpointData(
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

        # Recent checkpoint should not be stale
        assert not manager.is_stale(max_age_hours=24.0)

    def test_is_stale_old_checkpoint(self, tmp_path):
        """Test is_stale() with old checkpoint."""
        old_time = datetime.now() - timedelta(hours=48)
        manager = CheckpointManager(project_dir=tmp_path)
        manager.data = CheckpointData(
            created_at=old_time.isoformat(),
            updated_at=old_time.isoformat()
        )

        # 48-hour old checkpoint should be stale (max 24 hours)
        assert manager.is_stale(max_age_hours=24.0)

    def test_is_stale_no_timestamp(self, tmp_path):
        """Test is_stale() with no timestamp."""
        manager = CheckpointManager(project_dir=tmp_path)
        manager.data = CheckpointData()  # No timestamps

        # No timestamp = stale
        assert manager.is_stale()

    def test_get_age_hours_no_data(self, tmp_path):
        """Test get_age_hours() with no data."""
        manager = CheckpointManager(project_dir=tmp_path)

        age = manager.get_age_hours()

        assert age == 0.0

    def test_get_age_hours_recent(self, tmp_path):
        """Test get_age_hours() with recent checkpoint."""
        manager = CheckpointManager(project_dir=tmp_path)
        manager.data = CheckpointData(
            created_at=datetime.now().isoformat()
        )

        age = manager.get_age_hours()

        # Should be close to 0 hours
        assert age < 0.1  # Less than 6 minutes


class TestCheckpointValidation:
    """Test checkpoint validation."""

    def test_checkpoint_with_valid_stage(self):
        """Test checkpoint with valid stage name."""
        checkpoint = CheckpointData(
            last_completed_stage="DOWNLOAD"
        )

        assert checkpoint.last_completed_stage in STAGE_ORDER

    def test_checkpoint_stage_progression(self):
        """Test stages progress in order."""
        stages = ["ANALYZE", "DOWNLOAD", "TRANSCRIBE"]

        for i, stage in enumerate(stages):
            checkpoint = CheckpointData(last_completed_stage=stage)
            assert checkpoint.last_completed_stage == stage


class TestCheckpointBackup:
    """Test checkpoint backup functionality."""

    def test_backup_path_different_from_main(self, tmp_path):
        """Test backup path is different from main checkpoint."""
        manager = CheckpointManager(project_dir=tmp_path)

        assert manager.checkpoint_path != manager.backup_path

    def test_backup_path_in_same_directory(self, tmp_path):
        """Test backup is in same directory as main checkpoint."""
        manager = CheckpointManager(project_dir=tmp_path)

        assert manager.checkpoint_path.parent == manager.backup_path.parent


class TestConfigHash:
    """Test config hash functionality."""

    def test_config_hash_stored(self, tmp_path):
        """Test config hash is stored."""
        manager = CheckpointManager(
            project_dir=tmp_path,
            config_hash="abc123"
        )

        assert manager.config_hash == "abc123"

    def test_checkpoint_data_stores_hash(self):
        """Test CheckpointData can store config hash."""
        checkpoint = CheckpointData(config_hash="xyz789")

        assert checkpoint.config_hash == "xyz789"


class TestVoiceoverTracking:
    """Test voiceover file tracking."""

    def test_voiceover_path_storage(self):
        """Test storing voiceover path."""
        checkpoint = CheckpointData(
            voiceover_path="/project/voiceover.srt"
        )

        assert checkpoint.voiceover_path == "/project/voiceover.srt"

    def test_voiceover_hash_storage(self):
        """Test storing voiceover hash."""
        checkpoint = CheckpointData(
            voiceover_hash="file_hash_abc123"
        )

        assert checkpoint.voiceover_hash == "file_hash_abc123"

    def test_both_voiceover_fields(self):
        """Test storing both path and hash."""
        checkpoint = CheckpointData(
            voiceover_path="/vo.srt",
            voiceover_hash="hash123"
        )

        assert checkpoint.voiceover_path == "/vo.srt"
        assert checkpoint.voiceover_hash == "hash123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
