"""
Unit tests for LLM cache functionality.
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from src.llm_client.cache import LLMCache
from src.llm_client.base import LLMRequest, LLMResponse, ResponseFormat


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestLLMCache:
    """Test LLM cache functionality."""

    def test_initialization(self, temp_cache_dir):
        """Test cache can be initialized."""
        cache = LLMCache(temp_cache_dir, provider="test", ttl_hours=24)

        assert cache.cache_dir.exists()
        assert cache.provider == "test"
        assert cache.ttl_seconds == 24 * 3600

    def test_cache_directory_creation(self, temp_cache_dir):
        """Test cache directory is created if it doesn't exist."""
        cache_path = Path(temp_cache_dir) / "test_provider"
        assert not cache_path.exists()

        cache = LLMCache(temp_cache_dir, provider="test_provider")

        assert cache.cache_dir.exists()
        assert cache.cache_dir == cache_path

    def test_cache_key_generation(self, temp_cache_dir):
        """Test cache key is generated consistently."""
        cache = LLMCache(temp_cache_dir, provider="test")

        request1 = LLMRequest(prompt="test prompt", max_tokens=1000)
        request2 = LLMRequest(prompt="test prompt", max_tokens=1000)
        request3 = LLMRequest(prompt="different prompt", max_tokens=1000)

        key1 = cache._cache_key(request1)
        key2 = cache._cache_key(request2)
        key3 = cache._cache_key(request3)

        # Same requests should have same key
        assert key1 == key2
        # Different requests should have different keys
        assert key1 != key3
        # Keys should be 16 characters (MD5 hash truncated)
        assert len(key1) == 16

    def test_cache_key_includes_parameters(self, temp_cache_dir):
        """Test cache key includes all relevant parameters."""
        cache = LLMCache(temp_cache_dir, provider="test")

        request1 = LLMRequest(prompt="test", temperature=0.7)
        request2 = LLMRequest(prompt="test", temperature=0.9)

        key1 = cache._cache_key(request1)
        key2 = cache._cache_key(request2)

        # Different temperature should result in different key
        assert key1 != key2

    def test_set_and_get(self, temp_cache_dir):
        """Test setting and getting cache entries."""
        cache = LLMCache(temp_cache_dir, provider="test")

        request = LLMRequest(prompt="test prompt")
        response = LLMResponse(
            text="test response",
            parsed_data={"result": "success"},
            provider="test",
            model="test-model"
        )

        # Set cache
        cache.set(request, response)

        # Get cache
        cached = cache.get(request)

        assert cached is not None
        assert cached["text"] == "test response"
        assert cached["parsed_data"] == {"result": "success"}
        assert cached["provider"] == "test"
        assert cached["model"] == "test-model"
        assert "cached_at" in cached

    def test_cache_miss(self, temp_cache_dir):
        """Test cache returns None for miss."""
        cache = LLMCache(temp_cache_dir, provider="test")

        request = LLMRequest(prompt="nonexistent")
        cached = cache.get(request)

        assert cached is None

    def test_ttl_expiration(self, temp_cache_dir):
        """Test cache entries expire based on TTL."""
        # Use very short TTL for testing (1 second = 1/3600 hours)
        cache = LLMCache(temp_cache_dir, provider="test", ttl_hours=1/3600)

        request = LLMRequest(prompt="test")
        response = LLMResponse(text="test")

        # Set cache
        cache.set(request, response)

        # Should be cached immediately
        cached = cache.get(request)
        assert cached is not None

        # Wait for expiration (1.5 seconds to be safe)
        time.sleep(1.5)

        # Should be expired now
        cached = cache.get(request)
        assert cached is None

    def test_ttl_zero_never_expires(self, temp_cache_dir):
        """Test TTL=0 means never expire."""
        cache = LLMCache(temp_cache_dir, provider="test", ttl_hours=0)

        request = LLMRequest(prompt="test")
        response = LLMResponse(text="test")

        cache.set(request, response)

        # Should be cached
        cached = cache.get(request)
        assert cached is not None

    def test_cache_with_prefix(self, temp_cache_dir):
        """Test cache respects prefix in file naming."""
        cache = LLMCache(temp_cache_dir, provider="test")

        request = LLMRequest(
            prompt="test",
            cache_key_prefix="my_feature"
        )
        response = LLMResponse(text="test")

        cache.set(request, response)

        # Check file exists with correct prefix
        cache_files = list(cache.cache_dir.glob("my_feature_*.json"))
        assert len(cache_files) == 1

    def test_clear_all(self, temp_cache_dir):
        """Test clearing all cache entries."""
        cache = LLMCache(temp_cache_dir, provider="test")

        # Add multiple entries
        for i in range(3):
            request = LLMRequest(prompt=f"test {i}")
            response = LLMResponse(text=f"response {i}")
            cache.set(request, response)

        # Verify entries exist
        assert len(list(cache.cache_dir.glob("*.json"))) == 3

        # Clear all
        cache.clear()

        # All should be gone
        assert len(list(cache.cache_dir.glob("*.json"))) == 0

    def test_clear_by_prefix(self, temp_cache_dir):
        """Test clearing cache entries by prefix."""
        cache = LLMCache(temp_cache_dir, provider="test")

        # Add entries with different prefixes
        request1 = LLMRequest(prompt="test1", cache_key_prefix="prefix_a")
        request2 = LLMRequest(prompt="test2", cache_key_prefix="prefix_b")
        response = LLMResponse(text="test")

        cache.set(request1, response)
        cache.set(request2, response)

        # Clear only prefix_a
        cache.clear(prefix="prefix_a")

        # prefix_a should be gone, prefix_b should remain
        assert len(list(cache.cache_dir.glob("prefix_a_*.json"))) == 0
        assert len(list(cache.cache_dir.glob("prefix_b_*.json"))) == 1

    def test_clear_by_age(self, temp_cache_dir):
        """Test clearing cache entries older than specified age."""
        cache = LLMCache(temp_cache_dir, provider="test")

        # Add an entry
        request = LLMRequest(prompt="old")
        response = LLMResponse(text="test")
        cache.set(request, response)

        # Wait a bit
        time.sleep(0.5)

        # Add another entry
        request2 = LLMRequest(prompt="new")
        cache.set(request2, response)

        # Clear entries older than 0.25 hours (15 minutes)
        # The first entry should NOT be cleared (too recent)
        cache.clear(older_than_hours=0.25)

        # Both should still exist (not old enough)
        assert len(list(cache.cache_dir.glob("*.json"))) == 2

    def test_stats(self, temp_cache_dir):
        """Test cache statistics."""
        cache = LLMCache(temp_cache_dir, provider="test")

        # Initially empty
        stats = cache.stats()
        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0
        assert stats["provider"] == "test"

        # Add some entries
        for i in range(3):
            request = LLMRequest(prompt=f"test {i}")
            response = LLMResponse(text=f"response {i}" * 100)  # Make it bigger
            cache.set(request, response)

        # Check stats again
        stats = cache.stats()
        assert stats["total_files"] == 3
        assert stats["total_size_mb"] > 0
        assert stats["oldest_entry_hours"] is not None
        assert stats["oldest_entry_hours"] >= 0

    def test_corrupt_cache_file_handling(self, temp_cache_dir):
        """Test handling of corrupt cache files."""
        cache = LLMCache(temp_cache_dir, provider="test")

        request = LLMRequest(prompt="test")
        response = LLMResponse(text="test")

        # Set cache
        cache.set(request, response)

        # Corrupt the cache file
        cache_key = cache._cache_key(request)
        cache_file = cache.cache_dir / f"{request.cache_key_prefix}_{cache_key}.json"

        with open(cache_file, 'w') as f:
            f.write("not valid json{{{")

        # Getting corrupted cache should return None and delete file
        cached = cache.get(request)
        assert cached is None
        assert not cache_file.exists()

    def test_cache_with_images(self, temp_cache_dir):
        """Test cache key generation includes image hash."""
        cache = LLMCache(temp_cache_dir, provider="test")

        image1 = b"image data 1"
        image2 = b"image data 2"

        request1 = LLMRequest(prompt="test", images=[image1])
        request2 = LLMRequest(prompt="test", images=[image2])

        key1 = cache._cache_key(request1)
        key2 = cache._cache_key(request2)

        # Different images should result in different keys
        assert key1 != key2
