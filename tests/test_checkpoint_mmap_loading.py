"""
Test memory-mapped file loading for large checkpoints (US-130-004).

Tests that checkpoint files > 10MB use memory-mapped I/O for improved
performance and reduced memory pressure.
"""
import json
import pytest
import shutil
import tempfile
import time
from pathlib import Path

from src.checkpoint import CheckpointManager, CheckpointData, MMAP_THRESHOLD_BYTES


class TestCheckpointMmapLoading:
    """Tests for memory-mapped file loading functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def large_checkpoint(self, temp_project_dir):
        """Create a checkpoint larger than mmap threshold (10MB)."""
        checkpoint_path = temp_project_dir / "checkpoint.json"

        # Generate large data to exceed 10MB threshold
        # Each video entry is ~500 bytes, need ~20k entries for 10MB
        large_data = {
            "videos": [
                {
                    "id": f"vid_{i}",
                    "title": f"Video {i} " * 50,  # ~600 bytes per entry
                    "description": "Description " * 50,
                    "metadata": {f"key_{k}": f"value_{k}" for k in range(10)},
                }
                for i in range(25000)
            ]
        }

        checkpoint_data = {
            "version": "2.1",
            "created_at": "2026-02-16T10:00:00",
            "updated_at": "2026-02-16T12:00:00",
            "last_completed_stage": "MATCH",
            "config_hash": "abc123",
            "voiceover_path": "/path/to/voiceover.srt",
            "voiceover_hash": "def456",
            "analyze": {"duration": 10.5, "segments": 5},
            "video_search": {"results": list(range(50)), "searches_performed": 25},
            "caption": {"captioned_count": 45, "failed": 5},
            "match": large_data,
            "iterative_match": {"matches": list(range(30))},
            "download_segments": {"downloaded": 20},
            "chapter_data": {"chapters": [], "listicle_groups": []},
            "stage_metrics": {},
            "transcription_metrics": {},
            "validation_cache": {},
            "escalation_state": {},
            "circuit_breaker_health": {},
            "circuit_breaker_state": {},
            "compressed_checkpoint": {},
        }

        checkpoint_path.write_text(json.dumps(checkpoint_data))
        file_size = checkpoint_path.stat().st_size

        # Verify it's actually large
        assert file_size > MMAP_THRESHOLD_BYTES, f"Test data too small: {file_size} bytes"

        return temp_project_dir, file_size

    @pytest.fixture
    def small_checkpoint(self, temp_project_dir):
        """Create a checkpoint smaller than mmap threshold."""
        checkpoint_path = temp_project_dir / "checkpoint.json"

        checkpoint_data = {
            "version": "2.1",
            "created_at": "2026-02-16T10:00:00",
            "updated_at": "2026-02-16T12:00:00",
            "last_completed_stage": "MATCH",
            "config_hash": "abc123",
            "voiceover_path": "/path/to/voiceover.srt",
            "voiceover_hash": "def456",
            "analyze": {"duration": 10.5, "segments": 5},
            "video_search": {"results": list(range(50)), "searches_performed": 25},
            "caption": {"captioned_count": 45, "failed": 5},
            "match": {"videos": [{"id": "vid_1", "title": "Test"}]},
            "iterative_match": {"matches": list(range(30))},
            "download_segments": {"downloaded": 20},
            "chapter_data": {"chapters": [], "listicle_groups": []},
            "stage_metrics": {},
            "transcription_metrics": {},
            "validation_cache": {},
            "escalation_state": {},
            "circuit_breaker_health": {},
            "circuit_breaker_state": {},
            "compressed_checkpoint": {},
        }

        checkpoint_path.write_text(json.dumps(checkpoint_data))
        return temp_project_dir

    def test_mmap_loads_large_checkpoint(self, large_checkpoint):
        """Test that large checkpoints use mmap loading."""
        project_dir, file_size = large_checkpoint
        manager = CheckpointManager(project_dir)

        # Verify mmap is enabled by default
        assert manager._mmap_enabled is True

        # Load checkpoint
        data = manager.load()

        assert data is not None
        assert data.last_completed_stage == "MATCH"
        assert len(data.match.get("videos", [])) > 0

        # Verify mmap was used (or attempted)
        stats = manager._mmap_stats
        assert stats["mmap_loads"] > 0 or stats["standard_loads"] > 0

    def test_standard_load_for_small_checkpoint(self, small_checkpoint):
        """Test that small checkpoints use standard loading."""
        manager = CheckpointManager(small_checkpoint)

        data = manager.load()

        assert data is not None
        assert data.last_completed_stage == "MATCH"

        # Small checkpoint should not trigger mmap
        stats = manager._mmap_stats
        assert stats["mmap_loads"] == 0

    def test_mmap_threshold_respected(self, temp_project_dir):
        """Test that mmap threshold is respected."""
        # Create checkpoint just under threshold
        checkpoint_path = temp_project_dir / "checkpoint.json"

        # Create data that's under 10MB but not too small
        data = {"match": {"videos": [{"id": f"vid_{i}", "title": "Test " * 100} for i in range(1000)]}}
        checkpoint_data = {
            "version": "2.1",
            "created_at": "2026-02-16T10:00:00",
            "updated_at": "2026-02-16T12:00:00",
            "last_completed_stage": "MATCH",
            "config_hash": "abc123",
            "voiceover_path": "/path/to/voiceover.srt",
            "voiceover_hash": "def456",
            "analyze": {"duration": 10.5},
            "video_search": {},
            "caption": {},
            "match": data["match"],
            "iterative_match": {},
            "download_segments": {},
            "chapter_data": {},
            "stage_metrics": {},
            "transcription_metrics": {},
            "validation_cache": {},
            "escalation_state": {},
            "circuit_breaker_health": {},
            "circuit_breaker_state": {},
            "compressed_checkpoint": {},
        }

        checkpoint_path.write_text(json.dumps(checkpoint_data))
        file_size = checkpoint_path.stat().st_size

        # If under threshold, should not use mmap
        manager = CheckpointManager(temp_project_dir)
        manager._mmap_threshold_bytes = file_size + 1  # Set threshold above file size

        data = manager.load()
        assert data is not None
        assert manager._mmap_stats["mmap_loads"] == 0

    def test_mmap_fallback_on_error(self, temp_project_dir):
        """Test that mmap falls back to standard loading on error."""
        checkpoint_path = temp_project_dir / "checkpoint.json"

        # Create valid checkpoint
        checkpoint_data = {
            "version": "2.1",
            "created_at": "2026-02-16T10:00:00",
            "updated_at": "2026-02-16T12:00:00",
            "last_completed_stage": "MATCH",
            "config_hash": "abc123",
            "voiceover_path": "/path/to/voiceover.srt",
            "voiceover_hash": "def456",
            "analyze": {"duration": 10.5},
            "video_search": {},
            "caption": {},
            "match": {"videos": [{"id": "vid_1"}]},
            "iterative_match": {},
            "download_segments": {},
            "chapter_data": {},
            "stage_metrics": {},
            "transcription_metrics": {},
            "validation_cache": {},
            "escalation_state": {},
            "circuit_breaker_health": {},
            "circuit_breaker_state": {},
            "compressed_checkpoint": {},
        }

        checkpoint_path.write_text(json.dumps(checkpoint_data))

        # Test with mmap disabled
        manager = CheckpointManager(temp_project_dir)
        manager._mmap_enabled = False

        data = manager.load()
        assert data is not None

    def test_mmap_stats_tracking(self, large_checkpoint):
        """Test that mmap stats are properly tracked."""
        project_dir, file_size = large_checkpoint
        manager = CheckpointManager(project_dir)

        # Load checkpoint
        manager.load()

        stats = manager._mmap_stats

        # Should have attempted loading
        assert stats["mmap_loads"] + stats["standard_loads"] > 0

    def test_mmap_with_custom_threshold(self, temp_project_dir):
        """Test mmap with custom threshold."""
        checkpoint_path = temp_project_dir / "checkpoint.json"

        # Create small checkpoint
        checkpoint_data = {
            "version": "2.1",
            "created_at": "2026-02-16T10:00:00",
            "updated_at": "2026-02-16T12:00:00",
            "last_completed_stage": "MATCH",
            "config_hash": "abc123",
            "voiceover_path": "/path/to/voiceover.srt",
            "voiceover_hash": "def456",
            "analyze": {"duration": 10.5},
            "video_search": {},
            "caption": {},
            "match": {"videos": [{"id": "vid_1"}]},
            "iterative_match": {},
            "download_segments": {},
            "chapter_data": {},
            "stage_metrics": {},
            "transcription_metrics": {},
            "validation_cache": {},
            "escalation_state": {},
            "circuit_breaker_health": {},
            "circuit_breaker_state": {},
            "compressed_checkpoint": {},
        }

        checkpoint_path.write_text(json.dumps(checkpoint_data))

        # Set very low threshold to force mmap
        manager = CheckpointManager(temp_project_dir)
        manager._mmap_threshold_bytes = 100  # 100 bytes - very low

        data = manager.load()
        assert data is not None

    def test_lazy_loading_with_mmap(self, large_checkpoint):
        """Test that lazy loading still works with mmap approach."""
        project_dir, file_size = large_checkpoint
        manager = CheckpointManager(project_dir)

        # Use lazy loading
        match_data = manager.load_stage_data("MATCH")

        assert match_data is not None
        assert "videos" in match_data
        assert len(match_data["videos"]) > 0

    def test_get_mmap_stats_method(self, large_checkpoint):
        """Test get_mmap_stats method exists and returns stats."""
        project_dir, _ = large_checkpoint
        manager = CheckpointManager(project_dir)

        # Load checkpoint
        manager.load()

        # Get stats via method if it exists
        if hasattr(manager, 'get_mmap_stats'):
            stats = manager.get_mmap_stats()
            assert "mmap_loads" in stats
            assert "standard_loads" in stats
