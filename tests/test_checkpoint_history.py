"""
Test checkpoint history tracking (US-115-007).

Tests that checkpoint saves record metadata including timestamp, stage,
video_count, match_count, and size_bytes for run history tracking.
"""
import json
import pytest
import shutil
import tempfile
import time
from pathlib import Path
from datetime import datetime

from src.checkpoint import CheckpointManager, CheckpointData


class TestCheckpointHistory:
    """Tests for checkpoint history tracking."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def checkpoint_manager(self, temp_project_dir):
        """Create a CheckpointManager with no existing checkpoint."""
        return CheckpointManager(temp_project_dir, config_hash="test123")

    def test_history_file_created_after_save(self, checkpoint_manager):
        """Test that history file is created after first save."""
        # Initially no history
        assert not checkpoint_manager.history_path.exists()

        # Initialize checkpoint data
        checkpoint_manager.data = CheckpointData(
            created_at=datetime.now().isoformat(),
            config_hash="test123"
        )
        checkpoint_manager.data.updated_at = datetime.now().isoformat()
        checkpoint_manager.data.last_completed_stage = "ANALYZE"

        # Manually call record since we need data
        checkpoint_manager._record_history_entry("ANALYZE", is_intermediate=False)

        # History file should now exist
        assert checkpoint_manager.history_path.exists()

    def test_history_entry_structure(self, checkpoint_manager):
        """Test that history entries contain required fields."""
        # Initialize checkpoint data with video data
        checkpoint_manager.data = CheckpointData(
            created_at=datetime.now().isoformat(),
            config_hash="test123"
        )
        checkpoint_manager.data.updated_at = datetime.now().isoformat()
        checkpoint_manager.data.last_completed_stage = "VIDEO_SEARCH"

        # Add some mock video search data
        checkpoint_manager.data.video_search = {
            "videos": [{"id": f"vid_{i}"} for i in range(10)]
        }

        checkpoint_manager._record_history_entry("VIDEO_SEARCH", is_intermediate=False)

        # Check history content
        history = checkpoint_manager.get_checkpoint_history(limit=10)

        assert len(history) == 1
        entry = history[0]

        # Verify required fields
        assert "timestamp" in entry
        assert "stage" in entry
        assert "video_count" in entry
        assert "match_count" in entry
        assert "size_bytes" in entry

        assert entry["stage"] == "VIDEO_SEARCH"
        assert entry["video_count"] == 10
        assert entry["match_count"] == 0

    def test_history_multiple_runs(self, checkpoint_manager):
        """Test that history tracks multiple save events."""
        # First run - ANALYZE stage
        checkpoint_manager.data = CheckpointData(
            created_at=datetime.now().isoformat(),
            config_hash="test123"
        )
        checkpoint_manager.data.updated_at = datetime.now().isoformat()
        checkpoint_manager.data.last_completed_stage = "ANALYZE"
        checkpoint_manager._record_history_entry("ANALYZE", is_intermediate=False)

        # Wait a bit
        time.sleep(0.01)

        # Second run - VIDEO_SEARCH stage
        checkpoint_manager.data.last_completed_stage = "VIDEO_SEARCH"
        checkpoint_manager.data.video_search = {"videos": [{"id": f"vid_{i}"} for i in range(15)]}
        checkpoint_manager._record_history_entry("VIDEO_SEARCH", is_intermediate=False)

        # Third run - MATCH stage
        checkpoint_manager.data.last_completed_stage = "MATCH"
        checkpoint_manager.data.match = {"segments": [{"id": f"seg_{i}"} for i in range(5)]}
        checkpoint_manager._record_history_entry("MATCH", is_intermediate=False)

        # Check history
        history = checkpoint_manager.get_checkpoint_history(limit=10)

        assert len(history) == 3
        assert history[0]["stage"] == "ANALYZE"
        assert history[1]["stage"] == "VIDEO_SEARCH"
        assert history[2]["stage"] == "MATCH"

        assert history[1]["video_count"] == 15
        assert history[2]["match_count"] == 5

    def test_history_limit_parameter(self, checkpoint_manager):
        """Test that limit parameter works correctly."""
        # Create 5 history entries
        for i in range(5):
            checkpoint_manager.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash="test123"
            )
            checkpoint_manager.data.updated_at = datetime.now().isoformat()
            checkpoint_manager.data.last_completed_stage = f"STAGE_{i}"
            checkpoint_manager._record_history_entry(f"STAGE_{i}", is_intermediate=False)
            time.sleep(0.01)

        # Get last 3 only
        history = checkpoint_manager.get_checkpoint_history(limit=3)

        assert len(history) == 3
        # Should be the most recent 3
        assert history[0]["stage"] == "STAGE_2"
        assert history[2]["stage"] == "STAGE_4"

    def test_history_empty_checkpoint(self, checkpoint_manager):
        """Test that history works with no checkpoint data."""
        # Don't set any data
        history = checkpoint_manager.get_checkpoint_history(limit=10)

        assert history == []

    def test_history_persists_across_manager_instances(self, temp_project_dir):
        """Test that history is persisted and can be loaded by new manager."""
        # First manager - create history
        manager1 = CheckpointManager(temp_project_dir, config_hash="test123")
        manager1.data = CheckpointData(
            created_at=datetime.now().isoformat(),
            config_hash="test123"
        )
        manager1.data.updated_at = datetime.now().isoformat()
        manager1.data.last_completed_stage = "CAPTION"
        manager1._record_history_entry("CAPTION", is_intermediate=False)

        # Second manager - should load existing history
        manager2 = CheckpointManager(temp_project_dir, config_hash="test456")
        history = manager2.get_checkpoint_history(limit=10)

        assert len(history) == 1
        assert history[0]["stage"] == "CAPTION"

    def test_history_max_entries(self, checkpoint_manager):
        """Test that history is limited to 100 entries."""
        # Create more than 100 entries
        for i in range(105):
            checkpoint_manager.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash="test123"
            )
            checkpoint_manager.data.updated_at = datetime.now().isoformat()
            checkpoint_manager.data.last_completed_stage = f"STAGE_{i}"
            checkpoint_manager._record_history_entry(f"STAGE_{i}", is_intermediate=False)

        # History should be capped at 100
        history = checkpoint_manager.get_checkpoint_history(limit=200)

        assert len(history) == 100
        # Should have the most recent entries
        assert history[0]["stage"] == "STAGE_5"
        assert history[99]["stage"] == "STAGE_104"

    def test_history_video_count_from_match_data(self, checkpoint_manager):
        """Test that video_count is properly counted from match data."""
        checkpoint_manager.data = CheckpointData(
            created_at=datetime.now().isoformat(),
            config_hash="test123"
        )
        checkpoint_manager.data.updated_at = datetime.now().isoformat()
        checkpoint_manager.data.last_completed_stage = "DOWNLOAD_SEGMENTS"

        # Add match data with segments
        checkpoint_manager.data.match = {
            "segments": [
                {"video_id": "vid_1", "start": 0, "end": 10},
                {"video_id": "vid_2", "start": 10, "end": 20},
                {"video_id": "vid_1", "start": 20, "end": 30},
            ]
        }

        checkpoint_manager._record_history_entry("DOWNLOAD_SEGMENTS", is_intermediate=False)

        history = checkpoint_manager.get_checkpoint_history(limit=10)

        assert len(history) == 1
        # match_count should be 3 (number of segments)
        assert history[0]["match_count"] == 3

    def test_intermediate_saves_not_recorded(self, checkpoint_manager):
        """Test that intermediate saves don't create history entries."""
        checkpoint_manager.data = CheckpointData(
            created_at=datetime.now().isoformat(),
            config_hash="test123"
        )
        checkpoint_manager.data.updated_at = datetime.now().isoformat()
        checkpoint_manager.data.last_completed_stage = "MATCH"

        # Record intermediate
        checkpoint_manager._record_history_entry("DOWNLOAD_SEGMENTS", is_intermediate=True)

        # Should have no history
        history = checkpoint_manager.get_checkpoint_history(limit=10)
        assert history == []
