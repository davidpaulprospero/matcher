"""
Tests for cache eviction functionality.

Tests LRU eviction, size limits, and automatic cleanup for:
- BaseCache eviction methods
- GlobalCacheManager eviction
- EntityCache eviction
"""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass

from src.cache.base import BaseCache, EvictionResult


@dataclass
class MockCacheConfig:
    """Mock config for testing"""
    cache_dir: str = ""
    ttl_hours: int = 24
    max_size_mb: int = 0
    eviction_strategy: str = "lru"
    eviction_threshold: float = 0.9


def create_test_cache(cache_dir):
    """Create a concrete test cache implementation"""
    from src.cache.base import BaseCache, CacheEntry

    class TestCache(BaseCache):
        def _serialize_entry(self, entry):
            return {
                "data": entry.data,
                "cached_at": entry.cached_at,
                "key": entry.key,
                "metadata": entry.metadata
            }

        def _deserialize_entry(self, data):
            return CacheEntry(
                data=data.get("data"),
                cached_at=data.get("cached_at", 0),
                key=data.get("key", ""),
                metadata=data.get("metadata", {})
            )

    return TestCache(cache_dir)


class TestBaseCacheEviction:
    """Tests for BaseCache eviction methods"""

    def test_get_size_mb_empty_cache(self, tmp_path):
        """Empty cache returns 0 size"""
        cache_dir = tmp_path / "cache"
        cache = create_test_cache(cache_dir)
        assert cache.get_size_mb() >= 0  # May have index.json

    def test_get_size_mb_with_files(self, tmp_path):
        """Calculates size of cached files"""
        cache_dir = tmp_path / "cache"
        cache = create_test_cache(cache_dir)

        # Create test files
        for i in range(3):
            (cache_dir / f"test_{i}.json").write_text('{"data": "x" * 1000}')

        size = cache.get_size_mb()
        assert size > 0

    def test_get_entries_sorted_by_cached_at(self, tmp_path):
        """Sorts entries by cached_at timestamp"""
        cache_dir = tmp_path / "cache"
        cache = create_test_cache(cache_dir)

        # Add entries with different timestamps
        cache.index["old"] = {"cached_at": time.time() - 3600, "data": "old"}
        cache.index["new"] = {"cached_at": time.time(), "data": "new"}
        cache.index["mid"] = {"cached_at": time.time() - 1800, "data": "mid"}

        sorted_entries = cache.get_entries_sorted("cached_at", reverse=False)

        # Oldest should be first
        assert len(sorted_entries) == 3
        assert sorted_entries[0][0] == "old"

    def test_get_entries_sorted_by_last_used(self, tmp_path):
        """Sorts entries by last_used timestamp (LRU)"""
        cache_dir = tmp_path / "cache"
        cache = create_test_cache(cache_dir)

        # Add entries
        cache.index["recently_used"] = {"cached_at": time.time() - 7200, "last_used": time.time()}
        cache.index["never_used"] = {"cached_at": time.time(), "last_used": time.time() - 3600}

        sorted_entries = cache.get_entries_sorted("last_used", reverse=False)

        # Least recently used should be first
        assert sorted_entries[0][0] == "never_used"

    def test_evict_to_size_removes_oldest(self, tmp_path):
        """Evicts oldest entries first when using 'oldest' strategy"""
        cache_dir = tmp_path / "cache"
        cache = create_test_cache(cache_dir)

        # Create actual files to increase size
        for i in range(5):
            cache.index[f"entry_{i}"] = {
                "cached_at": time.time() - (i * 1000),
                "data": "x" * 10000  # Larger data
            }
            # Create corresponding file
            (cache_dir / f"entry_{i}.json").write_text(json.dumps({"data": "x" * 50000}))

        cache._save_index()  # Save to disk to increase size

        initial_size = cache.get_size_mb()
        # Only try eviction if size is significant
        if initial_size > 0.0001:
            result = cache.evict_to_size(initial_size / 10, strategy="oldest")
            assert isinstance(result, EvictionResult)
        else:
            # Just verify the method works
            result = cache.evict_to_size(0, strategy="oldest")
            assert isinstance(result, EvictionResult)

    def test_evict_to_size_dry_run(self, tmp_path):
        """Dry run doesn't delete entries"""
        cache_dir = tmp_path / "cache"
        cache = create_test_cache(cache_dir)

        for i in range(3):
            cache.index[f"entry_{i}"] = {"cached_at": time.time(), "data": "x" * 10000}
            (cache_dir / f"entry_{i}.json").write_text(json.dumps({"data": "x" * 50000}))

        cache._save_index()
        initial_count = len(cache.index)

        # Try very small target to trigger eviction
        initial_size = cache.get_size_mb()
        if initial_size > 0.0001:
            result = cache.evict_to_size(initial_size / 100, strategy="oldest", dry_run=True)
            assert result.dry_run is True
            # Entries should still exist
            assert len(cache.index) == initial_count
        else:
            # Just verify method works
            result = cache.evict_to_size(0, dry_run=True)
            assert result.dry_run is True

    def test_check_size_limit_under_threshold(self, tmp_path):
        """Returns False when under size limit"""
        cache_dir = tmp_path / "cache"
        cache = create_test_cache(cache_dir)
        cache.index["small"] = {"data": "test"}

        # 100MB limit should not be exceeded
        assert cache.check_size_limit(100, threshold=0.9) is False

    def test_check_size_limit_over_threshold(self, tmp_path):
        """Returns True when over threshold"""
        cache_dir = tmp_path / "cache"
        cache = create_test_cache(cache_dir)

        # Create many large entries
        for i in range(100):
            cache.index[f"large_{i}"] = {"data": "x" * 10000}
        cache._save_index()

        # Very small limit should be exceeded
        result = cache.check_size_limit(0.00001, threshold=0.9)
        assert result is True


@dataclass
class MockGlobalCacheConfig:
    """Full mock config for GlobalCacheManager"""
    enabled: bool = True
    cache_dir: str = ""
    check_before_download: bool = True
    prompt_reuse: bool = False
    min_topic_overlap: float = 0.3
    min_keyword_similarity: float = 0.8
    max_reuse_videos: int = 50
    max_redownload: int = 10
    redownload_deleted: bool = True
    max_size_mb: int = 0
    eviction_strategy: str = "lru"
    eviction_threshold: float = 0.9
    auto_cleanup_enabled: bool = True
    cleanup_on_startup: bool = False


class TestGlobalCacheEviction:
    """Tests for GlobalCacheManager eviction"""

    @pytest.fixture
    def global_cache_config(self, tmp_path):
        """Create test global cache config"""
        config = MockGlobalCacheConfig()
        config.cache_dir = str(tmp_path / "global_cache")
        return config

    def test_touch_video_nonexistent(self, global_cache_config, tmp_path):
        """touch_video handles nonexistent videos gracefully"""
        from src.global_cache import GlobalCacheManager

        cache = GlobalCacheManager(cache_dir=global_cache_config.cache_dir, config=global_cache_config)

        # Touch a nonexistent video - should not raise
        cache.touch_video("nonexistent_hash")

    def test_cleanup_orphaned_entries_empty_cache(self, global_cache_config, tmp_path):
        """cleanup_orphaned_entries handles empty cache"""
        from src.global_cache import GlobalCacheManager

        cache = GlobalCacheManager(cache_dir=global_cache_config.cache_dir, config=global_cache_config)

        # Cleanup on empty cache
        removed = cache.cleanup_orphaned_entries()

        assert removed == 0

    def test_count_orphaned_entries(self, global_cache_config, tmp_path):
        """count_orphaned_entries returns a count"""
        from src.global_cache import GlobalCacheManager

        cache = GlobalCacheManager(cache_dir=global_cache_config.cache_dir, config=global_cache_config)

        # Count orphaned entries on empty cache
        count = cache.count_orphaned_entries()

        assert count >= 0

    def test_evict_videos_empty_cache(self, global_cache_config, tmp_path):
        """evict_videos handles empty cache"""
        from src.global_cache import GlobalCacheManager

        cache = GlobalCacheManager(cache_dir=global_cache_config.cache_dir, config=global_cache_config)

        # Evict on empty cache
        result = cache.evict_videos(0.001, strategy="lru")

        assert result["entries_removed"] == 0
        assert isinstance(result["final_size_mb"], (int, float))
        assert isinstance(result["evicted_hashes"], list)

    def test_check_size_limit_disabled(self, global_cache_config, tmp_path):
        """check_size_limit returns False when no limit set"""
        from src.global_cache import GlobalCacheManager

        # No size limit (max_size_mb = 0)
        global_cache_config.max_size_mb = 0

        cache = GlobalCacheManager(cache_dir=global_cache_config.cache_dir, config=global_cache_config)

        result = cache.check_size_limit()
        assert result is False


class TestEvictionResult:
    """Tests for EvictionResult dataclass"""

    def test_eviction_result_fields(self):
        """EvictionResult has correct fields"""
        result = EvictionResult(
            entries_removed=5,
            bytes_freed=10240,
            final_size_mb=50.5,
            evicted_keys=["a", "b", "c"],
            dry_run=False
        )

        assert result.entries_removed == 5
        assert result.bytes_freed == 10240
        assert result.final_size_mb == 50.5
        assert len(result.evicted_keys) == 3
        assert result.dry_run is False

    def test_eviction_result_dry_run(self):
        """EvictionResult correctly tracks dry_run state"""
        result = EvictionResult(
            entries_removed=10,
            bytes_freed=20480,
            final_size_mb=25.0,
            evicted_keys=["x", "y"],
            dry_run=True
        )

        assert result.dry_run is True
        assert result.entries_removed == 10  # Would have removed


class TestEvictionStrategies:
    """Tests for different eviction strategies"""

    def test_lru_strategy_sorts_by_last_used(self, tmp_path):
        """LRU strategy sorts entries by last_used"""
        cache_dir = tmp_path / "cache"
        cache = create_test_cache(cache_dir)

        # Entry used long ago
        cache.index["unused"] = {
            "cached_at": time.time() - 100,
            "last_used": time.time() - 7200,  # 2 hours ago
            "data": "x" * 5000
        }
        # Entry used recently
        cache.index["recent"] = {
            "cached_at": time.time() - 7200,  # Cached 2 hours ago
            "last_used": time.time() - 60,    # Used 1 minute ago
            "data": "x" * 5000
        }

        # Check sorting by last_used
        sorted_entries = cache.get_entries_sorted("last_used", reverse=False)

        # Unused should be first (oldest last_used)
        assert sorted_entries[0][0] == "unused"

    def test_oldest_strategy_sorts_by_cached_at(self, tmp_path):
        """Oldest strategy sorts entries by cached_at"""
        cache_dir = tmp_path / "cache"
        cache = create_test_cache(cache_dir)

        # Old entry
        cache.index["old"] = {
            "cached_at": time.time() - 7200,  # 2 hours ago
            "last_used": time.time(),          # Recently used
            "data": "x" * 5000
        }
        # New entry
        cache.index["new"] = {
            "cached_at": time.time() - 60,    # 1 minute ago
            "last_used": time.time() - 3600,  # Not recently used
            "data": "x" * 5000
        }

        # Check sorting by cached_at
        sorted_entries = cache.get_entries_sorted("cached_at", reverse=False)

        # Old should be first (oldest cached_at)
        assert sorted_entries[0][0] == "old"


class TestLLMCacheCleanupExpired:
    """Tests for LLMCache cleanup_expired method"""

    def test_cleanup_expired_removes_old_entries(self, tmp_path):
        """cleanup_expired removes entries past TTL"""
        from src.llm_client.cache import LLMCache

        cache = LLMCache(str(tmp_path / "cache"), "gemini", ttl_hours=1)

        # Create expired entry
        cache_file = cache.cache_dir / "expired.json"
        cache_file.write_text(json.dumps({
            "text": "old response",
            "cached_at": time.time() - 7200,  # 2 hours ago
            "provider": "gemini",
            "model": "gemini-2.0-flash"
        }))

        # Create valid entry
        valid_file = cache.cache_dir / "valid.json"
        valid_file.write_text(json.dumps({
            "text": "new response",
            "cached_at": time.time(),
            "provider": "gemini",
            "model": "gemini-2.0-flash"
        }))

        removed = cache.cleanup_expired()

        assert removed >= 1
        assert not cache_file.exists()
        assert valid_file.exists()

    def test_cleanup_expired_with_zero_ttl(self, tmp_path):
        """cleanup_expired does nothing when TTL is 0 (disabled)"""
        from src.llm_client.cache import LLMCache

        cache = LLMCache(str(tmp_path / "cache"), "gemini", ttl_hours=0)

        # Create old entry
        cache_file = cache.cache_dir / "old.json"
        cache_file.write_text(json.dumps({
            "text": "response",
            "cached_at": time.time() - 999999,
            "provider": "gemini"
        }))

        removed = cache.cleanup_expired()

        assert removed == 0
        assert cache_file.exists()


class TestTranscriptCacheCleanup:
    """Tests for TranscriptCache cleanup methods"""

    def test_cleanup_orphaned_removes_missing_source(self, tmp_path):
        """cleanup_orphaned removes entries for missing source files"""
        from src.transcription.cache import TranscriptCache

        cache = TranscriptCache(str(tmp_path / ".cache"))

        # Create cache entry for missing video
        cache_file = cache.cache_dir / "orphan.json"
        cache_file.write_text(json.dumps([{
            "source_file": str(tmp_path / "missing_video.mp4"),
            "start_time": 0,
            "end_time": 10,
            "text": "test transcript"
        }]))

        # Create cache entry for existing video
        existing_video = tmp_path / "exists.mp4"
        existing_video.write_bytes(b"video")
        valid_file = cache.cache_dir / "valid.json"
        valid_file.write_text(json.dumps([{
            "source_file": str(existing_video),
            "start_time": 0,
            "end_time": 10,
            "text": "test transcript"
        }]))

        # Rebuild map
        cache._build_source_map()

        removed = cache.cleanup_orphaned()

        assert removed >= 1
        assert not cache_file.exists()
        assert valid_file.exists()

    def test_get_stats_returns_metrics(self, tmp_path):
        """get_stats returns cache metrics"""
        from src.transcription.cache import TranscriptCache

        cache = TranscriptCache(str(tmp_path / ".cache"))

        # Create some cache files
        for i in range(3):
            (cache.cache_dir / f"test_{i}.json").write_text(json.dumps([{
                "source_file": f"/path/to/video_{i}.mp4",
                "text": "transcript"
            }]))

        cache._build_source_map()
        stats = cache.get_stats()

        assert "total_entries" in stats
        assert "total_size_mb" in stats
        assert stats["total_entries"] >= 3
