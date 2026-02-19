"""US-115-003: Test checkpoint compression functionality.

Tests verify that:
- Compression reduces checkpoint file size
- Data integrity is maintained after compression/decompression
- Compression stats are correctly calculated
- Both compressed and uncompressed files can be loaded
- Compression can be disabled via config
"""

import gzip
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.checkpoint import CheckpointManager, CheckpointData


class TestCheckpointCompression:
    """Test checkpoint compression functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_config_with_compression(self):
        """Create a mock config with compression enabled."""
        config = MagicMock()
        pipeline = MagicMock()
        compression = MagicMock()
        compression.enabled = True
        compression.compression_level = 6
        pipeline.checkpoint_compression = compression
        pipeline.checkpoint_backup_count = 3
        pipeline.min_rotation_interval_seconds = 60
        config.pipeline = pipeline
        return config

    @pytest.fixture
    def mock_config_without_compression(self):
        """Create a mock config with compression disabled."""
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

    def test_compression_reduces_file_size(self, temp_project_dir, mock_config_with_compression):
        """Test that compression actually reduces file size."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash123", config=mock_config_with_compression)

        # Create checkpoint with substantial data
        checkpoint_manager.data = CheckpointData(
            created_at="2026-02-16T10:00:00",
            updated_at="2026-02-16T10:30:00",
            config_hash="test_hash123"
        )
        # Add substantial data to see compression effect
        checkpoint_manager.data.analyze = {"keywords": ["keyword" + str(i) for i in range(100)]}
        checkpoint_manager.data.match = {"matches": [{"id": i, "data": "x" * 100} for i in range(50)]}
        checkpoint_manager.data.video_search = {"videos": [{"id": i, "title": "Video " + str(i)} for i in range(100)]}

        # Save checkpoint
        checkpoint_manager.save(stage="TEST", force_full=True)

        # Read saved file
        saved_path = temp_project_dir / "checkpoint.json"
        assert saved_path.exists()

        # Check that file is gzip compressed (check magic number)
        with open(saved_path, 'rb') as f:
            header = f.read(2)
        assert header == b'\x1f\x8b', "File should be gzip compressed"

        # Verify compression stats
        assert checkpoint_manager._compression_stats is not None
        assert checkpoint_manager._compression_stats['compressed_size'] < checkpoint_manager._compression_stats['original_size']
        assert checkpoint_manager._compression_stats['ratio'] < 1.0

    def test_compression_data_integrity(self, temp_project_dir, mock_config_with_compression):
        """Test that data is correctly preserved after compression/decompression."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash123", config=mock_config_with_compression)

        # Create checkpoint with specific data
        original_data = CheckpointData(
            created_at="2026-02-16T10:00:00",
            updated_at="2026-02-16T10:30:00",
            config_hash="test_hash123",
            last_completed_stage="MATCH"
        )
        original_data.analyze = {"keywords": ["test1", "test2"], "segment_count": 5}
        original_data.match = {"match_count": 3, "avg_confidence": 0.85}
        original_data.video_search = {"video_ids": ["abc123", "def456"]}

        checkpoint_manager.data = original_data
        checkpoint_manager.save(stage="TEST", force_full=True)

        # Load checkpoint
        loaded = checkpoint_manager.load()
        assert loaded is not None

        # Verify data integrity
        assert loaded.created_at == original_data.created_at
        assert loaded.updated_at == original_data.updated_at
        assert loaded.config_hash == original_data.config_hash
        assert loaded.last_completed_stage == original_data.last_completed_stage
        assert loaded.analyze == original_data.analyze
        assert loaded.match == original_data.match
        assert loaded.video_search == original_data.video_search

    def test_compression_stats_in_summary(self, temp_project_dir, mock_config_with_compression):
        """Test that compression stats appear in checkpoint summary."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash123", config=mock_config_with_compression)

        checkpoint_manager.data = CheckpointData(
            created_at="2026-02-16T10:00:00",
            updated_at="2026-02-16T10:30:00",
            config_hash="test_hash123"
        )
        checkpoint_manager.data.analyze = {"keywords": ["test"]}

        checkpoint_manager.save(stage="TEST", force_full=True)

        summary = checkpoint_manager.get_summary()
        assert "Compression:" in summary

    def test_no_compression_when_disabled(self, temp_project_dir, mock_config_without_compression):
        """Test that checkpoint is not compressed when disabled in config."""
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash123", config=mock_config_without_compression)

        checkpoint_manager.data = CheckpointData(
            created_at="2026-02-16T10:00:00",
            updated_at="2026-02-16T10:30:00",
            config_hash="test_hash123"
        )
        checkpoint_manager.data.analyze = {"keywords": ["test"]}

        checkpoint_manager.save(stage="TEST", force_full=True)

        # Read saved file
        saved_path = temp_project_dir / "checkpoint.json"
        assert saved_path.exists()

        # Check that file is NOT gzip compressed
        with open(saved_path, 'rb') as f:
            header = f.read(2)
        assert header != b'\x1f\x8b', "File should NOT be gzip compressed when disabled"

        # Verify compression stats show no compression
        assert checkpoint_manager._compression_stats['ratio'] == 1.0

    def test_load_compressed_checkpoint(self, temp_project_dir, mock_config_with_compression):
        """Test that compressed checkpoint can be loaded."""
        # First, create and save a compressed checkpoint
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash123", config=mock_config_with_compression)
        checkpoint_manager.data = CheckpointData(
            created_at="2026-02-16T10:00:00",
            updated_at="2026-02-16T10:30:00",
            config_hash="test_hash123"
        )
        checkpoint_manager.data.match = {"match_count": 5}
        checkpoint_manager.save(stage="TEST", force_full=True)

        # Create a new checkpoint manager and load
        checkpoint_manager2 = CheckpointManager(temp_project_dir, config_hash="test_hash123", config=mock_config_with_compression)
        loaded = checkpoint_manager2.load()

        assert loaded is not None
        assert loaded.match == {"match_count": 5}

    def test_load_uncompressed_checkpoint(self, temp_project_dir, mock_config_without_compression):
        """Test that uncompressed checkpoint can still be loaded (backwards compatibility)."""
        # First, save an uncompressed checkpoint
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash123", config=mock_config_without_compression)
        checkpoint_manager.data = CheckpointData(
            created_at="2026-02-16T10:00:00",
            updated_at="2026-02-16T10:30:00",
            config_hash="test_hash123"
        )
        checkpoint_manager.data.match = {"match_count": 3}
        checkpoint_manager.save(stage="TEST", force_full=True)

        # Create a new checkpoint manager and load
        checkpoint_manager2 = CheckpointManager(temp_project_dir, config_hash="test_hash123", config=mock_config_without_compression)
        loaded = checkpoint_manager2.load()

        assert loaded is not None
        assert loaded.match == {"match_count": 3}

    def test_compression_level_config(self, temp_project_dir):
        """Test that different compression levels work correctly."""
        for level in [1, 6, 9]:
            config = MagicMock()
            pipeline = MagicMock()
            compression = MagicMock()
            compression.enabled = True
            compression.compression_level = level
            pipeline.checkpoint_compression = compression
            pipeline.checkpoint_backup_count = 3
            pipeline.min_rotation_interval_seconds = 60
            config.pipeline = pipeline

            checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash123", config=config)

            checkpoint_manager.data = CheckpointData(
                created_at="2026-02-16T10:00:00",
                updated_at="2026-02-16T10:30:00",
                config_hash="test_hash123"
            )
            checkpoint_manager.data.analyze = {"keywords": ["test" + str(i) for i in range(50)]}

            checkpoint_manager.save(stage="TEST", force_full=True)

            assert checkpoint_manager._compression_stats is not None
            assert checkpoint_manager._compression_level == level

    def test_compressed_checkpoint_field_exists(self):
        """Test that CheckpointData has compressed_checkpoint field."""
        data = CheckpointData()
        assert hasattr(data, 'compressed_checkpoint')
        assert data.compressed_checkpoint == {}

    def test_default_compression_enabled(self, temp_project_dir):
        """Test that compression is enabled by default when no config provided."""
        # Create checkpoint manager without config (should default to compression enabled)
        checkpoint_manager = CheckpointManager(temp_project_dir, config_hash="test_hash123")

        assert checkpoint_manager._compression_enabled == True
        assert checkpoint_manager._compression_level == 6
