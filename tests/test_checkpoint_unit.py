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

    @pytest.mark.fast
    def test_create_with_minimal_fields(self):
        """Test creating SavedKeywords with minimal fields."""
        keywords = SavedKeywords()

        assert keywords.name == ""
        assert keywords.keywords == []
        assert keywords.num_keywords == 0

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_create_default(self):
        """Test creating with defaults."""
        checkpoint = CheckpointData()

        assert checkpoint.version == "2.0"
        assert checkpoint.last_completed_stage == ""
        assert checkpoint.analyze == {}
        assert checkpoint.video_search == {}

    @pytest.mark.fast
    def test_create_with_stage_data(self):
        """Test creating with stage data."""
        checkpoint = CheckpointData(
            last_completed_stage="VIDEO_SEARCH",
            video_search={"video_ids": ["abc123"], "count": 50}
        )

        assert checkpoint.last_completed_stage == "VIDEO_SEARCH"
        assert checkpoint.video_search["count"] == 50

    @pytest.mark.fast
    def test_all_stage_fields_exist(self):
        """Test all expected stage fields are present (7-stage pipeline)."""
        checkpoint = CheckpointData()

        # Verify all 7-stage storage fields
        assert hasattr(checkpoint, "analyze")
        assert hasattr(checkpoint, "video_search")
        assert hasattr(checkpoint, "caption")
        assert hasattr(checkpoint, "match")
        assert hasattr(checkpoint, "iterative_match")
        assert hasattr(checkpoint, "download_segments")

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_from_dict_with_partial_data(self):
        """Test from_dict handles partial data."""
        data = {
            "version": "2.0",
            "last_completed_stage": "CAPTION",
            "unknown_field": "should be ignored"
        }

        checkpoint = CheckpointData.from_dict(data)

        assert checkpoint.version == "2.0"
        assert checkpoint.last_completed_stage == "CAPTION"
        # Unknown fields should be ignored
        assert not hasattr(checkpoint, "unknown_field")


class TestCheckpointManagerInit:
    """Test CheckpointManager initialization."""

    @pytest.mark.fast
    def test_create_manager(self, tmp_path):
        """Test creating checkpoint manager."""
        manager = CheckpointManager(
            project_dir=tmp_path,
            config_hash="test_hash"
        )

        assert manager.project_dir == tmp_path
        assert manager.config_hash == "test_hash"
        assert manager.data is None

    @pytest.mark.fast
    def test_checkpoint_paths(self, tmp_path):
        """Test checkpoint file paths."""
        manager = CheckpointManager(project_dir=tmp_path)

        assert manager.checkpoint_path == tmp_path / "checkpoint.json"
        assert manager.backup_path == tmp_path / "checkpoint.backup.json"

    @pytest.mark.fast
    def test_exists_no_file(self, tmp_path):
        """Test exists() returns False when no checkpoint."""
        manager = CheckpointManager(project_dir=tmp_path)

        assert not manager.exists()

    @pytest.mark.fast
    def test_exists_with_file(self, tmp_path):
        """Test exists() returns True when checkpoint exists."""
        # Create checkpoint file
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('{"version": "2.0"}')

        manager = CheckpointManager(project_dir=tmp_path)

        assert manager.exists()


class TestStageOrder:
    """Test STAGE_ORDER constant for 7-stage pipeline."""

    @pytest.mark.fast
    def test_stage_order_exists(self):
        """Test STAGE_ORDER is defined."""
        assert STAGE_ORDER is not None
        assert isinstance(STAGE_ORDER, (list, tuple))

    @pytest.mark.fast
    def test_stage_order_has_expected_stages(self):
        """Test STAGE_ORDER contains expected 7 stages."""
        expected_stages = ["ANALYZE", "VIDEO_SEARCH", "CAPTION", "MATCH", "ITERATIVE_MATCH", "DOWNLOAD_SEGMENTS", "OUTPUT"]

        for stage in expected_stages:
            assert stage in STAGE_ORDER

    @pytest.mark.fast
    def test_stage_order_is_sequential(self):
        """Test stages are in correct order (7-stage pipeline)."""
        # ANALYZE should come before VIDEO_SEARCH
        assert STAGE_ORDER.index("ANALYZE") < STAGE_ORDER.index("VIDEO_SEARCH")

        # VIDEO_SEARCH should come before CAPTION
        assert STAGE_ORDER.index("VIDEO_SEARCH") < STAGE_ORDER.index("CAPTION")

        # CAPTION should come before MATCH
        assert STAGE_ORDER.index("CAPTION") < STAGE_ORDER.index("MATCH")

        # MATCH should come before OUTPUT
        assert STAGE_ORDER.index("MATCH") < STAGE_ORDER.index("OUTPUT")


class TestCheckpointAge:
    """Test checkpoint age tracking."""

    @pytest.mark.fast
    def test_is_stale_no_data(self, tmp_path):
        """Test is_stale() with no checkpoint data."""
        manager = CheckpointManager(project_dir=tmp_path)

        assert not manager.is_stale()

    @pytest.mark.fast
    def test_is_stale_recent_checkpoint(self, tmp_path):
        """Test is_stale() with recent checkpoint."""
        manager = CheckpointManager(project_dir=tmp_path)
        manager.data = CheckpointData(
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

        # Recent checkpoint should not be stale
        assert not manager.is_stale(max_age_hours=24.0)

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_is_stale_no_timestamp(self, tmp_path):
        """Test is_stale() with no timestamp."""
        manager = CheckpointManager(project_dir=tmp_path)
        manager.data = CheckpointData()  # No timestamps

        # No timestamp = stale
        assert manager.is_stale()

    @pytest.mark.fast
    def test_get_age_hours_no_data(self, tmp_path):
        """Test get_age_hours() with no data."""
        manager = CheckpointManager(project_dir=tmp_path)

        age = manager.get_age_hours()

        assert age == 0.0

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_checkpoint_with_valid_stage(self):
        """Test checkpoint with valid stage name."""
        checkpoint = CheckpointData(
            last_completed_stage="MATCH"
        )

        assert checkpoint.last_completed_stage in STAGE_ORDER

    @pytest.mark.fast
    def test_checkpoint_stage_progression(self):
        """Test stages progress in order (7-stage pipeline)."""
        stages = ["ANALYZE", "VIDEO_SEARCH", "CAPTION"]

        for i, stage in enumerate(stages):
            checkpoint = CheckpointData(last_completed_stage=stage)
            assert checkpoint.last_completed_stage == stage


class TestCheckpointBackup:
    """Test checkpoint backup functionality."""

    @pytest.mark.fast
    def test_backup_path_different_from_main(self, tmp_path):
        """Test backup path is different from main checkpoint."""
        manager = CheckpointManager(project_dir=tmp_path)

        assert manager.checkpoint_path != manager.backup_path

    @pytest.mark.fast
    def test_backup_path_in_same_directory(self, tmp_path):
        """Test backup is in same directory as main checkpoint."""
        manager = CheckpointManager(project_dir=tmp_path)

        assert manager.checkpoint_path.parent == manager.backup_path.parent


class TestConfigHash:
    """Test config hash functionality."""

    @pytest.mark.fast
    def test_config_hash_stored(self, tmp_path):
        """Test config hash is stored."""
        manager = CheckpointManager(
            project_dir=tmp_path,
            config_hash="abc123"
        )

        assert manager.config_hash == "abc123"

    @pytest.mark.fast
    def test_checkpoint_data_stores_hash(self):
        """Test CheckpointData can store config hash."""
        checkpoint = CheckpointData(config_hash="xyz789")

        assert checkpoint.config_hash == "xyz789"


class TestVoiceoverTracking:
    """Test voiceover file tracking."""

    @pytest.mark.fast
    def test_voiceover_path_storage(self):
        """Test storing voiceover path."""
        checkpoint = CheckpointData(
            voiceover_path="/project/voiceover.srt"
        )

        assert checkpoint.voiceover_path == "/project/voiceover.srt"

    @pytest.mark.fast
    def test_voiceover_hash_storage(self):
        """Test storing voiceover hash."""
        checkpoint = CheckpointData(
            voiceover_hash="file_hash_abc123"
        )

        assert checkpoint.voiceover_hash == "file_hash_abc123"

    @pytest.mark.fast
    def test_both_voiceover_fields(self):
        """Test storing both path and hash."""
        checkpoint = CheckpointData(
            voiceover_path="/vo.srt",
            voiceover_hash="hash123"
        )

        assert checkpoint.voiceover_path == "/vo.srt"
        assert checkpoint.voiceover_hash == "hash123"


class TestPipelineStateValidation:
    """Test PipelineState.validate_state_attributes() after checkpoint restoration."""

    @pytest.mark.fast
    def test_validate_state_attributes_all_present(self):
        """Test validation when all required fields already exist."""
        from src.state import PipelineState

        state = PipelineState()
        # All defaults should be present
        initialized = state.validate_state_attributes()

        # Nothing should be initialized since all fields have defaults
        assert initialized == []
        # Verify fields still exist
        assert state.text_metadata == []
        assert state.caption_results == {}
        assert state.video_ids == []

    @pytest.mark.fast
    def test_validate_state_attributes_missing_text_metadata(self):
        """Test validation initializes missing text_metadata."""
        from src.state import PipelineState

        state = PipelineState()
        # Simulate incomplete state from checkpoint (set to None)
        state.text_metadata = None

        initialized = state.validate_state_attributes()

        assert 'text_metadata' in initialized
        assert state.text_metadata == []

    @pytest.mark.fast
    def test_validate_state_attributes_missing_caption_results(self):
        """Test validation initializes missing caption_results."""
        from src.state import PipelineState

        state = PipelineState()
        # Simulate incomplete state from checkpoint (set to None)
        state.caption_results = None

        initialized = state.validate_state_attributes()

        assert 'caption_results' in initialized
        assert state.caption_results == {}

    @pytest.mark.fast
    def test_validate_state_attributes_missing_video_ids(self):
        """Test validation initializes missing video_ids."""
        from src.state import PipelineState

        state = PipelineState()
        # Simulate incomplete state from checkpoint (set to None)
        state.video_ids = None

        initialized = state.validate_state_attributes()

        assert 'video_ids' in initialized
        assert state.video_ids == []

    @pytest.mark.fast
    def test_validate_state_attributes_multiple_missing(self):
        """Test validation initializes multiple missing fields."""
        from src.state import PipelineState

        state = PipelineState()
        # Simulate multiple missing fields
        state.text_metadata = None
        state.caption_results = None
        state.video_ids = None

        initialized = state.validate_state_attributes()

        # All three should be initialized
        assert len(initialized) == 3
        assert 'text_metadata' in initialized
        assert 'caption_results' in initialized
        assert 'video_ids' in initialized

        # Verify defaults
        assert state.text_metadata == []
        assert state.caption_results == {}
        assert state.video_ids == []

    @pytest.mark.fast
    def test_validate_state_attributes_logs_warning(self, caplog):
        """Test validation logs WARNING for each restored field."""
        import logging
        from src.state import PipelineState

        state = PipelineState()
        state.text_metadata = None

        with caplog.at_level(logging.WARNING):
            state.validate_state_attributes()

        # Should have logged a warning
        assert 'Restored missing text_metadata after checkpoint load' in caplog.text


class TestCheckpointManagerRestoreState:
    """Test CheckpointManager.restore_state() for US-40-003."""

    @pytest.mark.fast
    def test_restore_state_creates_new_state_if_none(self, tmp_path):
        """Test restore_state() creates a new PipelineState if none provided."""
        from src.state import PipelineState

        manager = CheckpointManager(tmp_path)
        state = manager.restore_state()

        assert isinstance(state, PipelineState)

    @pytest.mark.fast
    def test_restore_state_uses_provided_state(self, tmp_path):
        """Test restore_state() uses provided state object."""
        from src.state import PipelineState

        manager = CheckpointManager(tmp_path)
        existing_state = PipelineState()
        existing_state.keywords = ["test", "keyword"]

        result = manager.restore_state(existing_state)

        assert result is existing_state
        assert result.keywords == ["test", "keyword"]

    @pytest.mark.fast
    def test_restore_state_initializes_text_metadata_if_missing(self, tmp_path):
        """AC3: Test text_metadata is initialized to [] if missing from checkpoint."""
        from src.state import PipelineState

        manager = CheckpointManager(tmp_path)
        state = PipelineState()
        state.text_metadata = None  # Simulate missing from checkpoint

        result = manager.restore_state(state)

        assert result.text_metadata == []

    @pytest.mark.fast
    def test_restore_state_initializes_caption_results_if_missing(self, tmp_path):
        """AC4: Test caption_results is initialized to {} if missing from checkpoint."""
        from src.state import PipelineState

        manager = CheckpointManager(tmp_path)
        state = PipelineState()
        state.caption_results = None  # Simulate missing from checkpoint

        result = manager.restore_state(state)

        assert result.caption_results == {}

    @pytest.mark.fast
    def test_restore_state_initializes_video_ids_if_missing(self, tmp_path):
        """AC5: Test video_ids is initialized to [] if missing from checkpoint."""
        from src.state import PipelineState

        manager = CheckpointManager(tmp_path)
        state = PipelineState()
        state.video_ids = None  # Simulate missing from checkpoint

        result = manager.restore_state(state)

        assert result.video_ids == []

    @pytest.mark.fast
    def test_restore_state_logs_warning_for_missing_fields(self, tmp_path, caplog):
        """AC2: Test WARNING is logged for each missing/None attribute."""
        import logging
        from src.state import PipelineState

        manager = CheckpointManager(tmp_path)
        state = PipelineState()
        state.text_metadata = None
        state.caption_results = None

        with caplog.at_level(logging.WARNING):
            manager.restore_state(state)

        # Should have logged warnings for missing fields
        assert 'Checkpoint missing text_metadata' in caplog.text
        assert 'Checkpoint missing caption_results' in caplog.text

    @pytest.mark.fast
    def test_restore_state_preserves_existing_values(self, tmp_path):
        """Test restore_state() preserves existing non-None values."""
        from src.state import PipelineState

        manager = CheckpointManager(tmp_path)
        state = PipelineState()
        state.text_metadata = [{'text': 'existing'}]
        state.video_ids = ['video1', 'video2']

        result = manager.restore_state(state)

        # Existing values should be preserved
        assert result.text_metadata == [{'text': 'existing'}]
        assert result.video_ids == ['video1', 'video2']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
