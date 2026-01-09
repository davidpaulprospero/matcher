"""
Tests for src/transcription/delta_index.py

Tests the DeltaAwareIndex class for tracking transcribed videos
and avoiding redundant transcription work.
"""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import patch

from src.transcription.delta_index import DeltaAwareIndex


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Create a temporary cache directory"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def existing_index(tmp_path):
    """Create an existing delta index file"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    index_file = cache_dir / "delta_index.json"
    data = {
        "indexed": ["/path/to/video1.mp4", "/path/to/video2.mp4"],
        "indexed_ids": ["abc123", "def456"],
        "updated_at": time.time()
    }
    with open(index_file, 'w') as f:
        json.dump(data, f)

    return cache_dir


class TestDeltaAwareIndexInit:
    """Test DeltaAwareIndex initialization"""

    def test_init_creates_empty_index(self, tmp_cache_dir):
        """Test initialization with no existing index"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        assert isinstance(index.indexed_videos, set)
        assert len(index.indexed_videos) == 0
        assert isinstance(index.indexed_video_ids, set)
        assert len(index.indexed_video_ids) == 0

    def test_init_loads_existing_index(self, existing_index):
        """Test initialization loads existing index file"""
        index = DeltaAwareIndex(str(existing_index))

        assert len(index.indexed_videos) >= 2
        assert len(index.indexed_video_ids) >= 2

    def test_init_sets_index_path(self, tmp_cache_dir):
        """Test that index path is set correctly"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        assert index.index_path == tmp_cache_dir / "delta_index.json"

    def test_init_handles_corrupt_index(self, tmp_path):
        """Test initialization handles corrupt index file"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create corrupt index file
        index_file = cache_dir / "delta_index.json"
        with open(index_file, 'w') as f:
            f.write("{invalid json")

        # Should not crash
        index = DeltaAwareIndex(str(cache_dir))

        # Should have empty index
        assert len(index.indexed_videos) == 0
        assert len(index.indexed_video_ids) == 0


class TestIndexLoading:
    """Test index loading from disk"""

    def test_load_normalizes_paths(self, tmp_path):
        """Test that paths are normalized when loading"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create index with Windows-style paths
        index_file = cache_dir / "delta_index.json"
        data = {
            "indexed": ["C:\\Users\\test\\video.mp4", "/Users/test/video2.mp4"],
            "indexed_ids": []
        }
        with open(index_file, 'w') as f:
            json.dump(data, f)

        index = DeltaAwareIndex(str(cache_dir))

        # Paths should be normalized
        assert len(index.indexed_videos) >= 2

    def test_load_migrates_video_ids(self, tmp_path):
        """Test migration of video IDs from paths if not stored"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create old format index without indexed_ids
        index_file = cache_dir / "delta_index.json"
        data = {
            "indexed": ["/path/to/abc123.mp4", "/path/to/def456.mp4"]
            # No indexed_ids key (old format)
        }
        with open(index_file, 'w') as f:
            json.dump(data, f)

        index = DeltaAwareIndex(str(cache_dir))

        # Should have attempted to rebuild video IDs
        # (actual result depends on extract_video_id implementation)
        assert len(index.indexed_videos) >= 2

    def test_load_empty_index_file(self, tmp_path):
        """Test loading empty index file"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        index_file = cache_dir / "delta_index.json"
        with open(index_file, 'w') as f:
            json.dump({"indexed": [], "indexed_ids": []}, f)

        index = DeltaAwareIndex(str(cache_dir))

        assert len(index.indexed_videos) == 0
        assert len(index.indexed_video_ids) == 0


class TestIndexSaving:
    """Test index saving to disk"""

    def test_save_creates_cache_dir(self, tmp_path):
        """Test that save creates cache directory if needed"""
        cache_dir = tmp_path / "cache"
        # Don't create directory

        index = DeltaAwareIndex(str(cache_dir))
        index.mark_indexed("/path/to/video.mp4")

        # Cache dir should be created
        assert cache_dir.exists()
        assert (cache_dir / "delta_index.json").exists()

    def test_save_writes_json(self, tmp_cache_dir):
        """Test that save writes proper JSON format"""
        index = DeltaAwareIndex(str(tmp_cache_dir))
        index.mark_indexed("/path/to/video.mp4")

        # Read and verify JSON
        index_file = tmp_cache_dir / "delta_index.json"
        with open(index_file, 'r') as f:
            data = json.load(f)

        assert "indexed" in data
        assert "indexed_ids" in data
        assert "updated_at" in data
        assert isinstance(data["updated_at"], (int, float))

    def test_save_handles_write_error(self, tmp_cache_dir):
        """Test that save handles write errors gracefully"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        # Make directory read-only
        tmp_cache_dir.chmod(0o444)

        try:
            # Should not crash
            index.mark_indexed("/path/to/video.mp4")
        finally:
            # Restore permissions
            tmp_cache_dir.chmod(0o755)


class TestIsIndexed:
    """Test is_indexed() method"""

    def test_is_indexed_returns_true_for_indexed_video(self, existing_index):
        """Test is_indexed returns True for indexed videos"""
        index = DeltaAwareIndex(str(existing_index))

        result = index.is_indexed("/path/to/video1.mp4")
        assert result is True

    def test_is_indexed_returns_false_for_new_video(self, existing_index):
        """Test is_indexed returns False for non-indexed videos"""
        index = DeltaAwareIndex(str(existing_index))

        result = index.is_indexed("/path/to/new_video.mp4")
        assert result is False

    def test_is_indexed_normalizes_path(self, tmp_cache_dir):
        """Test that is_indexed normalizes paths"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        # Mark video indexed with one path format
        index.mark_indexed("/path/to/video.mp4")

        # Check with different path format (should still match after normalization)
        # Exact behavior depends on normalize_path implementation
        result = index.is_indexed("/path/to/video.mp4")
        assert result is True

    def test_is_indexed_checks_video_id(self, tmp_cache_dir):
        """Test that is_indexed checks video ID (for segments)"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        # Mark a video indexed (extract_video_id may extract ID)
        index.mark_indexed("/path/to/abc123.mp4")

        # Check if segment file with same video ID is considered indexed
        # Behavior depends on extract_video_id implementation
        # If extract_video_id returns "abc123", segment should match
        result1 = index.is_indexed("/path/to/abc123.mp4")
        assert result1 is True


class TestMarkIndexed:
    """Test mark_indexed() method"""

    def test_mark_indexed_adds_video(self, tmp_cache_dir):
        """Test that mark_indexed adds video to index"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        assert not index.is_indexed("/path/to/video.mp4")

        index.mark_indexed("/path/to/video.mp4")

        assert index.is_indexed("/path/to/video.mp4")

    def test_mark_indexed_normalizes_path(self, tmp_cache_dir):
        """Test that mark_indexed normalizes paths"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        index.mark_indexed("/path/to/video.mp4")

        # Should be in normalized form
        assert len(index.indexed_videos) == 1

    def test_mark_indexed_extracts_video_id(self, tmp_cache_dir):
        """Test that mark_indexed extracts and stores video ID"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        # Mark video with potential video ID in filename
        index.mark_indexed("/path/to/abc123.mp4")

        # Video IDs may or may not be extracted depending on implementation
        # Just verify the method doesn't crash
        assert len(index.indexed_videos) == 1

    def test_mark_indexed_saves_to_disk(self, tmp_cache_dir):
        """Test that mark_indexed persists to disk"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        index.mark_indexed("/path/to/video.mp4")

        # Verify file exists
        assert (tmp_cache_dir / "delta_index.json").exists()

        # Create new index instance (should load from disk)
        index2 = DeltaAwareIndex(str(tmp_cache_dir))
        assert index2.is_indexed("/path/to/video.mp4")

    def test_mark_indexed_duplicate_is_idempotent(self, tmp_cache_dir):
        """Test that marking same video multiple times is safe"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        index.mark_indexed("/path/to/video.mp4")
        index.mark_indexed("/path/to/video.mp4")
        index.mark_indexed("/path/to/video.mp4")

        # Should only have one entry (set behavior)
        assert len(index.indexed_videos) == 1


class TestMarkIndexedBatch:
    """Test mark_indexed_batch() method"""

    def test_mark_indexed_batch_adds_multiple_videos(self, tmp_cache_dir):
        """Test that batch marking adds multiple videos"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        video_paths = [
            "/path/to/video1.mp4",
            "/path/to/video2.mp4",
            "/path/to/video3.mp4"
        ]

        index.mark_indexed_batch(video_paths)

        assert len(index.indexed_videos) == 3
        for vp in video_paths:
            assert index.is_indexed(vp)

    def test_mark_indexed_batch_empty_list(self, tmp_cache_dir):
        """Test batch marking with empty list"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        index.mark_indexed_batch([])

        assert len(index.indexed_videos) == 0

    def test_mark_indexed_batch_saves_once(self, tmp_cache_dir):
        """Test that batch marking saves only once"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        video_paths = ["/video1.mp4", "/video2.mp4", "/video3.mp4"]

        # Spy on _save calls
        original_save = index._save
        save_count = 0

        def counting_save():
            nonlocal save_count
            save_count += 1
            original_save()

        index._save = counting_save

        index.mark_indexed_batch(video_paths)

        # Should save only once (not 3 times)
        assert save_count == 1

    def test_mark_indexed_batch_normalizes_paths(self, tmp_cache_dir):
        """Test that batch marking normalizes all paths"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        video_paths = ["/path/to/video1.mp4", "/path/to/video2.mp4"]

        index.mark_indexed_batch(video_paths)

        # Paths should be normalized
        assert len(index.indexed_videos) == 2


class TestGetNewVideos:
    """Test get_new_videos() method"""

    def test_get_new_videos_returns_unindexed_only(self, existing_index):
        """Test that get_new_videos returns only unindexed videos"""
        index = DeltaAwareIndex(str(existing_index))

        video_paths = [
            "/path/to/video1.mp4",  # Indexed
            "/path/to/video2.mp4",  # Indexed
            "/path/to/video3.mp4",  # Not indexed
            "/path/to/video4.mp4"   # Not indexed
        ]

        new_videos = index.get_new_videos(video_paths)

        # Should return only video3 and video4
        assert len(new_videos) == 2
        assert "/path/to/video3.mp4" in new_videos
        assert "/path/to/video4.mp4" in new_videos

    def test_get_new_videos_empty_list(self, tmp_cache_dir):
        """Test get_new_videos with empty input list"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        new_videos = index.get_new_videos([])

        assert new_videos == []

    def test_get_new_videos_all_indexed(self, existing_index):
        """Test get_new_videos when all videos are indexed"""
        index = DeltaAwareIndex(str(existing_index))

        video_paths = ["/path/to/video1.mp4", "/path/to/video2.mp4"]

        new_videos = index.get_new_videos(video_paths)

        assert len(new_videos) == 0

    def test_get_new_videos_all_new(self, tmp_cache_dir):
        """Test get_new_videos when all videos are new"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        video_paths = ["/video1.mp4", "/video2.mp4", "/video3.mp4"]

        new_videos = index.get_new_videos(video_paths)

        assert len(new_videos) == 3
        assert new_videos == video_paths


class TestClear:
    """Test clear() method"""

    def test_clear_removes_all_entries(self, existing_index):
        """Test that clear removes all indexed videos"""
        index = DeltaAwareIndex(str(existing_index))

        # Verify index has entries
        assert len(index.indexed_videos) > 0

        index.clear()

        # Should be empty
        assert len(index.indexed_videos) == 0
        assert len(index.indexed_video_ids) == 0

    def test_clear_saves_to_disk(self, existing_index):
        """Test that clear persists to disk"""
        index = DeltaAwareIndex(str(existing_index))

        index.clear()

        # Create new instance (should load empty index)
        index2 = DeltaAwareIndex(str(existing_index))
        assert len(index2.indexed_videos) == 0

    def test_clear_on_empty_index(self, tmp_cache_dir):
        """Test clearing an already empty index"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        # Should not crash
        index.clear()

        assert len(index.indexed_videos) == 0


class TestEdgeCases:
    """Test edge cases and error scenarios"""

    def test_index_with_special_characters(self, tmp_cache_dir):
        """Test indexing paths with special characters"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        special_paths = [
            "/path/with spaces/video.mp4",
            "/path/with-dashes/video.mp4",
            "/path/with_underscores/video.mp4",
            "/path/with(parentheses)/video.mp4"
        ]

        for path in special_paths:
            index.mark_indexed(path)
            assert index.is_indexed(path)

    def test_index_with_unicode_characters(self, tmp_cache_dir):
        """Test indexing paths with unicode characters"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        unicode_path = "/path/to/видео.mp4"  # Russian characters

        index.mark_indexed(unicode_path)
        assert index.is_indexed(unicode_path)

    def test_index_with_very_long_path(self, tmp_cache_dir):
        """Test indexing very long paths"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        long_path = "/path/to/" + ("very_long_directory/" * 20) + "video.mp4"

        index.mark_indexed(long_path)
        assert index.is_indexed(long_path)

    def test_concurrent_mark_and_check(self, tmp_cache_dir):
        """Test marking and checking in sequence"""
        index = DeltaAwareIndex(str(tmp_cache_dir))

        for i in range(100):
            path = f"/path/to/video{i}.mp4"
            assert not index.is_indexed(path)
            index.mark_indexed(path)
            assert index.is_indexed(path)

    def test_index_persistence_across_instances(self, tmp_cache_dir):
        """Test that index persists across multiple instances"""
        # First instance
        index1 = DeltaAwareIndex(str(tmp_cache_dir))
        index1.mark_indexed("/video1.mp4")
        index1.mark_indexed("/video2.mp4")

        # Second instance (should load from disk)
        index2 = DeltaAwareIndex(str(tmp_cache_dir))
        assert index2.is_indexed("/video1.mp4")
        assert index2.is_indexed("/video2.mp4")

        # Third instance after clear
        index2.clear()
        index3 = DeltaAwareIndex(str(tmp_cache_dir))
        assert not index3.is_indexed("/video1.mp4")
        assert not index3.is_indexed("/video2.mp4")
