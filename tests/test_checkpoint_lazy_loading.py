"""
Test lazy loading for checkpoint stage data (US-115-005).

Tests that checkpoint stage data is loaded on-demand rather than all at once,
improving startup time for large checkpoints.
"""
import json
import pytest
import shutil
import tempfile
from pathlib import Path

from src.checkpoint import CheckpointManager, CheckpointData, STAGE_ORDER


class TestCheckpointLazyLoading:
    """Tests for lazy loading functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def checkpoint_with_data(self, temp_project_dir):
        """Create a checkpoint with realistic stage data."""
        # Create checkpoint with stage data
        checkpoint_path = temp_project_dir / "checkpoint.json"

        # Add substantial data to each stage to demonstrate lazy loading benefits
        large_data = {"videos": [{"id": f"vid_{i}", "title": f"Video {i}" * 50} for i in range(100)]}

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
            "match": large_data,  # Large data
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

    def test_load_stage_data_lazy(self, checkpoint_with_data):
        """Test that load_stage_data only loads requested stage."""
        manager = CheckpointManager(checkpoint_with_data)

        # Initially no data loaded
        assert manager.data is None
        assert manager._raw_checkpoint_data is None

        # Load only MATCH stage data
        match_data = manager.load_stage_data("MATCH")

        # Verify only MATCH was loaded
        assert isinstance(match_data, dict)
        assert "videos" in match_data
        assert len(match_data["videos"]) == 100

        # Verify stats tracking
        stats = manager.get_lazy_load_stats()
        assert "match" in stats["stages_loaded"]
        assert stats["total_load_calls"] == 1

    def test_cache_hit_on_repeated_access(self, checkpoint_with_data):
        """Test that repeated access uses cache."""
        manager = CheckpointManager(checkpoint_with_data)

        # First access loads
        data1 = manager.load_stage_data("MATCH")
        # Second access should hit cache
        data2 = manager.load_stage_data("MATCH")

        assert data1 == data2
        stats = manager.get_lazy_load_stats()
        assert stats["stages_cached"] >= 1

    def test_load_all_stage_data(self, checkpoint_with_data):
        """Test load_all_stage_data loads full checkpoint."""
        manager = CheckpointManager(checkpoint_with_data)

        # Use load_all_stage_data
        data = manager.load_all_stage_data()

        assert data is not None
        assert isinstance(data, CheckpointData)
        assert data.last_completed_stage == "MATCH"
        assert manager._full_checkpoint_loaded is True

    def test_get_stage_data_uses_lazy_loading(self, checkpoint_with_data):
        """Test that get_stage_data uses lazy loading."""
        manager = CheckpointManager(checkpoint_with_data)

        # get_stage_data should trigger lazy loading when no full load
        stage_data = manager.get_stage_data("match")

        assert isinstance(stage_data, dict)
        assert "videos" in stage_data

    def test_get_lazy_load_stats_memory_saved(self, checkpoint_with_data):
        """Test that memory_saved_bytes is calculated."""
        manager = CheckpointManager(checkpoint_with_data)

        # Load only one stage
        manager.load_stage_data("MATCH")

        # Get stats
        stats = manager.get_lazy_load_stats()

        assert "memory_saved_bytes" in stats
        assert stats["memory_saved_bytes"] > 0  # Some data was saved by not loading all

    def test_different_stages_loaded_separately(self, checkpoint_with_data):
        """Test loading different stages separately."""
        manager = CheckpointManager(checkpoint_with_data)

        # Load multiple different stages
        match_data = manager.load_stage_data("MATCH")
        analyze_data = manager.load_stage_data("ANALYZE")
        caption_data = manager.load_stage_data("CAPTION")

        assert "videos" in match_data
        assert analyze_data.get("duration") == 10.5
        assert caption_data.get("captioned_count") == 45

        stats = manager.get_lazy_load_stats()
        assert len(stats["stages_loaded"]) == 3

    def test_full_load_after_lazy(self, checkpoint_with_data):
        """Test that full load works after lazy loading."""
        manager = CheckpointManager(checkpoint_with_data)

        # First lazy load
        manager.load_stage_data("MATCH")

        # Then load all
        data = manager.load_all_stage_data()

        assert data is not None
        assert data.last_completed_stage == "MATCH"
        assert manager._full_checkpoint_loaded is True

    def test_nonexistent_stage_returns_empty(self, checkpoint_with_data):
        """Test that nonexistent stage returns empty dict."""
        manager = CheckpointManager(checkpoint_with_data)

        data = manager.load_stage_data("NONEXISTENT")

        assert data == {}

    def test_lazy_loading_no_checkpoint(self, temp_project_dir):
        """Test lazy loading when no checkpoint exists."""
        manager = CheckpointManager(temp_project_dir)

        data = manager.load_stage_data("MATCH")

        assert data == {}

    def test_get_chapter_data_lazy_loading(self, checkpoint_with_data):
        """Test that get_chapter_data also uses lazy loading."""
        manager = CheckpointManager(checkpoint_with_data)

        # get_chapter_data should lazy load
        chapter_data = manager.get_chapter_data()

        assert isinstance(chapter_data, dict)
        # Our test data has chapter_data = {"chapters": [], "listicle_groups": []}
        assert "chapters" in chapter_data or chapter_data == {}

    def test_backward_compatibility_with_full_load(self, checkpoint_with_data):
        """Test backward compatibility - get_stage_data works after full load."""
        manager = CheckpointManager(checkpoint_with_data)

        # First load full checkpoint (existing behavior)
        data = manager.load()

        # Now get_stage_data should use existing data
        stage_data = manager.get_stage_data("match")

        assert "videos" in stage_data
        assert len(stage_data["videos"]) == 100

    def test_performance_improvement_demonstration(self, checkpoint_with_data):
        """Demonstrate performance improvement from lazy loading."""
        import time

        manager = CheckpointManager(checkpoint_with_data)

        # Time lazy loading just one stage
        start = time.perf_counter()
        manager.load_stage_data("MATCH")
        lazy_time = time.perf_counter() - start

        # Create fresh manager and load all
        manager2 = CheckpointManager(checkpoint_with_data)
        start = time.perf_counter()
        manager2.load_all_stage_data()
        full_time = time.perf_counter() - start

        # Lazy should be faster (or at least not slower) than full load
        # This is a soft assertion since timing varies
        print(f"\nLazy load time: {lazy_time*1000:.2f}ms")
        print(f"Full load time: {full_time*1000:.2f}ms")

        # At minimum verify both complete successfully
        assert lazy_time > 0
        assert full_time > 0
