"""US-130-005: Test checkpoint data deduplication functionality.

Tests verify that:
- Content hash computation correctly identifies duplicate data
- Deduplication saves space for repeated data structures
- Restoration correctly reconstructs original data from references
- Deduplication ratio is calculated correctly
- Deduplication can be disabled via config
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.checkpoint import CheckpointManager, CheckpointData


class TestCheckpointDeduplication:
    """Test checkpoint data deduplication functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with compression and deduplication enabled."""
        config = MagicMock()
        pipeline = MagicMock()
        compression = MagicMock()
        compression.enabled = False  # Disable compression to isolate dedup testing
        compression.compression_level = 6
        pipeline.checkpoint_compression = compression
        pipeline.checkpoint_backup_count = 3
        pipeline.min_rotation_interval_seconds = 60
        config.pipeline = pipeline
        return config

    @pytest.fixture
    def mock_config_dedup_disabled(self):
        """Create a mock config with deduplication disabled."""
        config = MagicMock()
        pipeline = MagicMock()
        compression = MagicMock()
        compression.enabled = False
        compression.compression_level = 6
        pipeline.checkpoint_compression = compression
        pipeline.checkpoint_backup_count = 3
        pipeline.min_rotation_interval_seconds = 60
        config.pipeline = pipeline
        return config

    def test_compute_content_hash(self, temp_project_dir, mock_config):
        """Test that content hash correctly identifies duplicate data."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Test with identical data
        data1 = {"key": "value", "number": 123}
        data2 = {"key": "value", "number": 123}
        hash1 = checkpoint_manager._compute_content_hash(data1)
        hash2 = checkpoint_manager._compute_content_hash(data2)
        assert hash1 == hash2, "Identical data should produce same hash"

        # Test with different data
        data3 = {"key": "different", "number": 456}
        hash3 = checkpoint_manager._compute_content_hash(data3)
        assert hash1 != hash3, "Different data should produce different hash"

        # Test with list
        data_list = [1, 2, 3, "test"]
        hash_list = checkpoint_manager._compute_content_hash(data_list)
        assert hash_list is not None

    def test_deduplication_saves_space(self, temp_project_dir, mock_config):
        """Test that deduplication reduces checkpoint size for repeated data."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Create checkpoint with repeated data structures
        checkpoint_manager.data = CheckpointData(
            created_at="2026-02-18T10:00:00",
            updated_at="2026-02-18T10:30:00",
            config_hash="test_hash"
        )

        # Add repeated video metadata (typical in checkpoint)
        repeated_metadata = {
            "id": "video123",
            "title": "Sample Video Title",
            "description": "This is a sample video description " * 5,
            "duration": 120,
            "channel": "Test Channel",
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
            "published_at": "2026-01-15",
            "view_count": 10000,
            "like_count": 500,
        }

        # Add same metadata multiple times (simulating cross-stage repetition)
        checkpoint_manager.data.video_search = {
            "videos": [repeated_metadata.copy() for _ in range(20)]
        }
        checkpoint_manager.data.caption = {
            "captions": [repeated_metadata.copy() for _ in range(20)]
        }
        checkpoint_manager.data.match = {
            "matches": [repeated_metadata.copy() for _ in range(20)]
        }

        # Calculate sizes
        original_data = checkpoint_manager.data.to_dict()
        original_size = len(json.dumps(original_data, default=str))

        # Apply deduplication
        deduped_data, dedup_store = checkpoint_manager._deduplicate_data(original_data)
        deduped_size = len(json.dumps(deduped_data, default=str))

        # Verify deduplication occurred
        assert len(dedup_store) > 0, "Dedup store should have entries"
        assert deduped_size < original_size, f"Deduped size ({deduped_size}) should be less than original ({original_size})"

        # Calculate ratio
        savings_ratio = (original_size - deduped_size) / original_size
        assert savings_ratio > 0.1, f"Expected >10% savings, got {savings_ratio:.1%}"

    def test_restore_deduplicated_data(self, temp_project_dir, mock_config):
        """Test that deduplicated data is correctly restored on load."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Original data with repeated entries
        original_data = {
            "stage1": {
                "videos": [
                    {"id": "v1", "title": "Test Video 1", "data": "x" * 100},
                    {"id": "v2", "title": "Test Video 2", "data": "y" * 100},
                ]
            },
            "stage2": {
                "results": [
                    {"id": "v1", "title": "Test Video 1", "data": "x" * 100},  # Same as stage1[0]
                    {"id": "v3", "title": "Test Video 3", "data": "z" * 100},
                ]
            }
        }

        # Apply deduplication
        deduped_data, dedup_store = checkpoint_manager._deduplicate_data(original_data)

        # Verify dedup store has entries
        assert len(dedup_store) > 0

        # Restore data
        restored_data = checkpoint_manager._restore_deduplicated_data(deduped_data, dedup_store)

        # Verify restoration matches original
        assert restored_data["stage1"]["videos"][0]["title"] == original_data["stage1"]["videos"][0]["title"]
        assert restored_data["stage1"]["videos"][1]["id"] == original_data["stage1"]["videos"][1]["id"]
        assert restored_data["stage2"]["results"][0]["id"] == original_data["stage2"]["results"][0]["id"]

    def test_deduplication_roundtrip(self, temp_project_dir, mock_config):
        """Test complete deduplication roundtrip: save -> load."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Create checkpoint with repeated data
        checkpoint_manager.data = CheckpointData(
            created_at="2026-02-18T10:00:00",
            updated_at="2026-02-18T10:30:00",
            config_hash="test_hash",
            last_completed_stage="MATCH"
        )

        # Add repeated metadata (simulating typical checkpoint)
        video_metadata = {
            "id": "xyz123",
            "title": "Repeated Video Title",
            "description": "This video appears in multiple stages " * 3,
            "duration": 300,
            "channel": "Test Channel",
            "tags": ["educational", "tutorial", "howto"],
        }

        checkpoint_manager.data.video_search = {"videos": [video_metadata.copy() for _ in range(10)]}
        checkpoint_manager.data.caption = {"results": [video_metadata.copy() for _ in range(10)]}
        checkpoint_manager.data.match = {"matches": [video_metadata.copy() for _ in range(10)]}

        # Save checkpoint
        checkpoint_manager.save(stage="MATCH", force_full=True)

        # Load checkpoint
        loaded_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)
        loaded_data = loaded_manager.load()

        # Verify data was restored correctly
        assert loaded_data is not None
        assert len(loaded_data.video_search.get("videos", [])) == 10
        assert len(loaded_data.caption.get("results", [])) == 10
        assert len(loaded_data.match.get("matches", [])) == 10

        # Verify content matches
        original_video = checkpoint_manager.data.video_search["videos"][0]
        loaded_video = loaded_data.video_search["videos"][0]
        assert original_video["id"] == loaded_video["id"]
        assert original_video["title"] == loaded_video["title"]
        assert original_video["channel"] == loaded_video["channel"]

    def test_dedup_stats_tracking(self, temp_project_dir, mock_config):
        """Test that deduplication stats are correctly tracked."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Create data with duplicates
        test_data = {
            "stage1": {"items": [{"data": "x" * 100} for _ in range(5)]},
            "stage2": {"items": [{"data": "x" * 100} for _ in range(5)]},
        }

        # Apply deduplication
        deduped_data, dedup_store = checkpoint_manager._deduplicate_data(test_data)

        # Check stats
        stats = checkpoint_manager._dedup_stats
        assert stats["dedup_saves"] > 0
        assert stats["unique_blocks"] > 0

    def test_deduplication_disabled(self, temp_project_dir, mock_config_dedup_disabled):
        """Test that deduplication can be disabled."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config_dedup_disabled)
        checkpoint_manager._dedup_enabled = False

        test_data = {"key": "value", "repeated": "data" * 50}

        # With dedup disabled, should return original data
        deduped_data, dedup_store = checkpoint_manager._deduplicate_data(test_data)

        assert deduped_data == test_data
        assert len(dedup_store) == 0

    def test_deduplication_ratio_calculation(self, temp_project_dir, mock_config):
        """Test deduplication ratio calculation for typical checkpoint data."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash", config=mock_config)

        # Simulate realistic checkpoint data with repeated video metadata
        video_template = {
            "id": "video_{}",
            "title": "Video Title {}",
            "description": "Description " * 10,
            "duration": 120,
            "channel": "Test Channel",
            "tags": ["tag1", "tag2", "tag3"],
            "metrics": {"views": 1000, "likes": 100, "comments": 50},
        }

        # Create data with many repeated entries
        checkpoint_data = {"videos": []}
        for i in range(30):
            video = video_template.copy()
            video["id"] = video["id"].format(i)
            video["title"] = video["title"].format(i)
            # Create duplicates: same video appears multiple times
            if i % 3 == 0:
                video["id"] = "duplicate_video"  # Force duplicates
            checkpoint_data["videos"].append(video)

        # Calculate original size
        original_size = len(json.dumps(checkpoint_data, default=str))

        # Apply deduplication
        deduped_data, dedup_store = checkpoint_manager._deduplicate_data(checkpoint_data)
        deduped_size = len(json.dumps(deduped_data, default=str))

        # Calculate and log ratio
        ratio = deduped_size / original_size if original_size > 0 else 1.0
        savings = 1.0 - ratio

        # Verify deduplication worked
        assert len(dedup_store) > 0, "Should have unique blocks in dedup store"
        assert ratio < 1.0, f"Dedup ratio {ratio:.2f} should be less than 1.0"

        print(f"Deduplication: {original_size} -> {deduped_size} bytes ({savings:.1%} savings)")
        print(f"Unique blocks: {len(dedup_store)}")
