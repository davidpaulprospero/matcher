"""
Unit tests for VideoContextCache (US-134-011).

Tests the video context caching functionality that avoids rebuilding
video context from metadata across multiple segment comparisons.
"""

import pytest
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.similarity_cache import (
    VideoContextCache,
    get_video_context_cache,
    clear_video_context_cache,
    video_context_cache_key,
)


class TestVideoContextCache:
    """Test VideoContextCache functionality."""

    @pytest.mark.fast
    def test_cache_basic_get_put(self):
        """Test basic cache put and get operations."""
        cache = VideoContextCache(max_size=100, ttl_seconds=3600)

        # Put a value
        cache.put("video1", "Video context: Title: Test Video")

        # Get should return the value
        result = cache.get("video1")
        assert result == "Video context: Title: Test Video"

    @pytest.mark.fast
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = VideoContextCache(max_size=100, ttl_seconds=3600)

        # Non-existent key should return None
        result = cache.get("nonexistent")
        assert result is None

    @pytest.mark.fast
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = VideoContextCache(max_size=100, ttl_seconds=3600)

        # Add some entries
        cache.put("video1", "Context 1")
        cache.get("video1")  # Hit
        cache.get("video2")  # Miss

        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['size'] == 1
        assert stats['hit_rate'] == 0.5

    @pytest.mark.fast
    def test_cache_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = VideoContextCache(max_size=100, ttl_seconds=1)  # 1 second TTL

        # Put a value
        cache.put("video1", "Context")

        # Should be available immediately
        result = cache.get("video1")
        assert result == "Context"

        # Wait for TTL to expire
        time.sleep(1.5)

        # Should be expired now
        result = cache.get("video1")
        assert result is None

        stats = cache.get_stats()
        assert stats['expirations'] == 1

    @pytest.mark.fast
    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = VideoContextCache(max_size=100, ttl_seconds=3600)

        cache.put("video1", "Context 1")
        cache.put("video2", "Context 2")

        assert cache.get("video1") is not None

        cache.clear()

        assert cache.get("video1") is None
        assert cache.get("video2") is None
        assert cache.get_stats()['size'] == 0

    @pytest.mark.fast
    def test_cache_max_size_eviction(self):
        """Test cache eviction when max size is reached."""
        cache = VideoContextCache(max_size=3, ttl_seconds=3600)

        cache.put("v1", "Context 1")
        cache.put("v2", "Context 2")
        cache.put("v3", "Context 3")
        cache.put("v4", "Context 4")  # Should trigger eviction

        stats = cache.get_stats()
        assert stats['evictions'] >= 1


class TestVideoContextCacheKey:
    """Test video_context_cache_key function."""

    @pytest.mark.fast
    def test_video_id_key(self):
        """Test cache key generation with video ID."""
        key = video_context_cache_key("abc123", title="Test", description="Description")

        assert key.startswith("vid:")
        assert "abc123" in key

    @pytest.mark.fast
    def test_metadata_key(self):
        """Test cache key generation with metadata only."""
        key = video_context_cache_key(
            None,
            title="Test Video",
            description="A test video description",
            tags=["tag1", "tag2"],
            chapters=[{"title": "Chapter 1"}]
        )

        assert key.startswith("meta:")
        assert len(key) > 10

    @pytest.mark.fast
    def test_empty_key(self):
        """Test cache key for empty metadata."""
        key = video_context_cache_key(None, None, None, None, None)

        assert key == "vid:empty"


class TestGlobalCacheInstance:
    """Test global cache instance functions."""

    @pytest.mark.fast
    def test_get_global_cache(self):
        """Test getting global cache instance."""
        # Clear any existing cache first
        clear_video_context_cache()

        cache1 = get_video_context_cache(ttl_seconds=1800)
        cache2 = get_video_context_cache(ttl_seconds=3600)

        # Should return the same instance (ttl only used on first call)
        assert cache1 is cache2

    @pytest.mark.fast
    def test_clear_global_cache(self):
        """Test clearing global cache."""
        cache = get_video_context_cache(ttl_seconds=3600)
        cache.put("test", "value")

        clear_video_context_cache()

        assert cache.get("test") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
