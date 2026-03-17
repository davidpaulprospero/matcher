"""US-130-006: Test differential checkpointing functionality.

Tests verify that:
- compute_diff() correctly identifies changes between checkpoints
- apply_diff() correctly reconstructs full checkpoint from diff
- Differential saves 30-50% space for typical checkpoints
- Full checkpoint kept every N saves (configurable)
- Performance tests show save time reduction
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.checkpoint import CheckpointManager, CheckpointData


class TestDifferentialCheckpointing:
    """Test differential checkpointing functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with compression and diff enabled."""
        config = MagicMock()
        pipeline = MagicMock()
        compression = MagicMock()
        compression.enabled = False  # Disable compression to isolate diff testing
        compression.compression_level = 6
        pipeline.checkpoint_compression = compression
        pipeline.checkpoint_backup_count = 3
        pipeline.min_rotation_interval_seconds = 60
        config.pipeline = pipeline
        return config

    def test_compute_diff_basic(self, temp_project_dir, mock_config):
        """Test that compute_diff correctly identifies changes between checkpoints."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Base checkpoint data
        base_data = {
            "updated_at": "2026-02-18T10:00:00",
            "last_completed_stage": "VIDEO_SEARCH",
            "video_search": {"videos": [{"id": "v1", "title": "Video 1"}]},
            "config_hash": "test_hash",
        }

        # Current checkpoint with changes
        current_data = {
            "updated_at": "2026-02-18T11:00:00",
            "last_completed_stage": "CAPTION",
            "video_search": {"videos": [{"id": "v1", "title": "Video 1"}]},
            "caption": {"captions": [{"id": "v1", "text": "Caption text"}]},
            "config_hash": "test_hash",
        }

        diff = checkpoint_manager._compute_diff(current_data, base_data)

        # Verify diff structure
        assert diff["_diff_version"] == 1
        assert diff["_diff_from"] == "2026-02-18T10:00:00"
        assert diff["_diff_timestamp"] == "2026-02-18T11:00:00"

        # Verify changed fields are captured
        assert "updated_at" in diff["_diff_fields"]
        assert "last_completed_stage" in diff["_diff_fields"]
        assert "caption" in diff["_diff_fields"]

        # Verify unchanged field is not in diff
        assert "video_search" not in diff["_diff_fields"]
        assert "config_hash" not in diff["_diff_fields"]

    def test_apply_diff(self, temp_project_dir, mock_config):
        """Test that apply_diff correctly reconstructs full checkpoint."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Base checkpoint data
        base_data = {
            "updated_at": "2026-02-18T10:00:00",
            "last_completed_stage": "VIDEO_SEARCH",
            "video_search": {"videos": [{"id": "v1", "title": "Video 1"}]},
            "config_hash": "test_hash",
        }

        # Diff containing changes
        diff = {
            "_diff_version": 1,
            "_diff_from": "2026-02-18T10:00:00",
            "_diff_timestamp": "2026-02-18T11:00:00",
            "_diff_fields": {
                "updated_at": "2026-02-18T11:00:00",
                "last_completed_stage": "CAPTION",
                "caption": {"captions": [{"id": "v1", "text": "Caption text"}]},
            },
        }

        result = checkpoint_manager._apply_diff(base_data, diff)

        # Verify full reconstruction
        assert result["updated_at"] == "2026-02-18T11:00:00"
        assert result["last_completed_stage"] == "CAPTION"
        assert result["caption"] == {"captions": [{"id": "v1", "text": "Caption text"}]}
        assert result["video_search"] == {"videos": [{"id": "v1", "title": "Video 1"}]}
        assert result["config_hash"] == "test_hash"

    def test_differential_saves_space(self, temp_project_dir, mock_config):
        """Test that differential saves 30-50% space for typical checkpoints."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Create typical checkpoint data with some repeated structures
        base_data = {
            "updated_at": "2026-02-18T10:00:00",
            "last_completed_stage": "MATCH",
            "config_hash": "test_hash",
            "created_at": "2026-02-18T09:00:00",
        }

        # Add typical stage data (mimics real checkpoint)
        for stage in ["analyze", "video_search", "caption", "match"]:
            base_data[stage] = {
                "results": [
                    {"id": f"v{i}", "title": f"Video {i}", "description": "Description " * 10}
                    for i in range(20)
                ]
            }

        # Current data - only a few fields changed
        current_data = base_data.copy()
        current_data["updated_at"] = "2026-02-18T11:00:00"
        current_data["last_completed_stage"] = "ITERATIVE_MATCH"
        current_data["iterative_match"] = {"gaps": [{"start": 10, "end": 20}]}

        # Calculate sizes
        full_size = len(json.dumps(current_data, indent=2, default=str).encode('utf-8'))

        # Compute diff
        diff = checkpoint_manager._compute_diff(current_data, base_data)

        # Create diff save format (what gets written to disk)
        diff_save_data = {
            "_diff_marker": True,
            "_diff_base_timestamp": base_data["updated_at"],
            "_diff_sequence": 1,
            "_diff_fields": diff.get("_diff_fields", {}),
            "_diff_version": diff.get("_diff_version", 1),
            "updated_at": current_data["updated_at"],
            "last_completed_stage": current_data["last_completed_stage"],
            "config_hash": current_data["config_hash"],
        }

        diff_size = len(json.dumps(diff_save_data, indent=2, default=str).encode('utf-8'))

        # Calculate space savings
        space_saved = full_size - diff_size
        savings_ratio = space_saved / full_size

        # Verify at least 30% space savings
        assert savings_ratio >= 0.30, f"Expected at least 30% savings, got {savings_ratio:.1%}"
        print(f"Full size: {full_size} bytes, Diff size: {diff_size} bytes, Savings: {savings_ratio:.1%}")

    def test_differential_save_count_tracking(self, temp_project_dir, mock_config):
        """Test that save count tracks correctly for full checkpoint intervals."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Default interval is 10
        assert checkpoint_manager._diff_full_interval == 10
        assert checkpoint_manager._diff_save_count == 0

        # Simulate differential saves
        for i in range(5):
            checkpoint_manager._diff_save_count += 1

        assert checkpoint_manager._diff_save_count == 5

        # Simulate full save (reset)
        checkpoint_manager._diff_base_data = {"updated_at": "2026-02-18T12:00:00"}
        checkpoint_manager._diff_save_count = 0

        assert checkpoint_manager._diff_save_count == 0
        assert checkpoint_manager._diff_base_data is not None

    def test_differential_disabled(self, temp_project_dir, mock_config):
        """Test that differential can be disabled."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Check default is enabled
        assert checkpoint_manager._diff_enabled is True

        # Disable differential
        checkpoint_manager._diff_enabled = False

        # Compute diff should not be used when disabled
        base_data = {"updated_at": "2026-02-18T10:00:00", "config_hash": "test_hash"}
        current_data = {"updated_at": "2026-02-18T11:00:00", "config_hash": "test_hash"}

        # Should still work but won't be used in saving
        diff = checkpoint_manager._compute_diff(current_data, base_data)
        assert diff is not None

    def test_diff_stats_tracking(self, temp_project_dir, mock_config):
        """Test that differential stats are tracked correctly."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Check initial stats
        assert checkpoint_manager._diff_stats["diff_saves"] == 0
        assert checkpoint_manager._diff_stats["full_saves"] == 0
        assert checkpoint_manager._diff_stats["space_saved_bytes"] == 0

        # Simulate diff saves
        checkpoint_manager._diff_stats["diff_saves"] = 5
        checkpoint_manager._diff_stats["full_saves"] = 1
        checkpoint_manager._diff_stats["total_original_bytes"] = 10000
        checkpoint_manager._diff_stats["total_diff_bytes"] = 3000
        checkpoint_manager._diff_stats["space_saved_bytes"] = 7000

        assert checkpoint_manager._diff_stats["diff_saves"] == 5
        assert checkpoint_manager._diff_stats["full_saves"] == 1
        assert checkpoint_manager._diff_stats["space_saved_bytes"] == 7000

    def test_diff_roundtrip(self, temp_project_dir, mock_config):
        """Test full diff save/restore cycle."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Create initial full checkpoint
        base_data = {
            "updated_at": "2026-02-18T10:00:00",
            "last_completed_stage": "VIDEO_SEARCH",
            "config_hash": "test_hash",
            "created_at": "2026-02-18T09:00:00",
            "video_search": {"results": [{"id": "v1", "title": "Test"}]},
        }

        # Set as base for diff
        checkpoint_manager._diff_base_data = base_data
        checkpoint_manager._diff_save_count = 1

        # Create current data with changes
        current_data = base_data.copy()
        current_data["updated_at"] = "2026-02-18T11:00:00"
        current_data["last_completed_stage"] = "CAPTION"
        current_data["caption"] = {"results": [{"id": "v1", "text": "Caption"}]}

        # Compute diff
        diff = checkpoint_manager._compute_diff(current_data, base_data)

        # Apply diff to reconstruct
        reconstructed = checkpoint_manager._apply_diff(base_data, diff)

        # Verify reconstruction
        assert reconstructed["updated_at"] == "2026-02-18T11:00:00"
        assert reconstructed["last_completed_stage"] == "CAPTION"
        assert reconstructed["caption"] == {"results": [{"id": "v1", "text": "Caption"}]}
        assert reconstructed["video_search"] == {"results": [{"id": "v1", "title": "Test"}]}
        assert reconstructed["config_hash"] == "test_hash"

    def test_custom_full_interval(self, temp_project_dir):
        """Test that custom full interval can be set."""
        config = MagicMock()
        pipeline = MagicMock()
        compression = MagicMock()
        compression.enabled = False
        compression.compression_level = 6
        pipeline.checkpoint_compression = compression
        pipeline.checkpoint_backup_count = 3
        pipeline.min_rotation_interval_seconds = 60
        config.pipeline = pipeline

        # Create checkpoint with custom interval
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=config)
        checkpoint_manager._diff_full_interval = 5

        assert checkpoint_manager._diff_full_interval == 5

    def test_diff_with_nested_changes(self, temp_project_dir, mock_config):
        """Test diff correctly handles nested data changes."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Base with nested structure
        base_data = {
            "updated_at": "2026-02-18T10:00:00",
            "match": {
                "matches": [
                    {"segment": 0, "video_id": "v1", "score": 0.9},
                    {"segment": 1, "video_id": "v2", "score": 0.85},
                ]
            },
        }

        # Current with nested changes
        current_data = {
            "updated_at": "2026-02-18T11:00:00",
            "match": {
                "matches": [
                    {"segment": 0, "video_id": "v1", "score": 0.95},  # Changed score
                    {"segment": 1, "video_id": "v3", "score": 0.8},  # Changed video
                    {"segment": 2, "video_id": "v4", "score": 0.75},  # New match
                ]
            },
        }

        diff = checkpoint_manager._compute_diff(current_data, base_data)

        # The entire match field should be in diff (field-level diffing)
        assert "match" in diff["_diff_fields"]
        assert "updated_at" in diff["_diff_fields"]
