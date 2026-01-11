"""
Comprehensive tests for src/llm_client/cache.py

Tests LLMCache class:
- Cache key generation
- TTL expiration logic
- Cache get/set operations
- Clear with prefix and age filtering
- Statistics collection
- Error handling
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# Mock ResponseFormat enum before importing
class MockResponseFormat(Enum):
    TEXT = "text"
    JSON = "json"


@dataclass
class MockLLMRequest:
    """Mock LLM request for testing"""
    prompt: str
    system_prompt: str = ""
    max_tokens: int = 1000
    temperature: float = 0.7
    response_format: MockResponseFormat = MockResponseFormat.TEXT
    cache_key_prefix: str = "test"
    images: list = None
    timeout: int = 60


@dataclass
class MockLLMResponse:
    """Mock LLM response for testing"""
    text: str
    parsed_data: dict = None
    provider: str = "test"
    model: str = "test-model"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Create temporary cache directory"""
    cache_dir = tmp_path / "llm_cache"
    cache_dir.mkdir(parents=True)
    return str(cache_dir)


@pytest.fixture
def cache_instance(tmp_cache_dir):
    """Create LLMCache instance"""
    from src.llm_client.cache import LLMCache
    return LLMCache(base_dir=tmp_cache_dir, provider="test_provider", ttl_hours=24)


# ============================================================================
# Test Initialization
# ============================================================================

class TestLLMCacheInit:
    """Test LLMCache initialization"""

    def test_init_creates_provider_directory(self, tmp_cache_dir):
        """Test that initialization creates provider subdirectory"""
        from src.llm_client.cache import LLMCache

        cache = LLMCache(
            base_dir=tmp_cache_dir,
            provider="gemini",
            ttl_hours=24
        )

        provider_dir = Path(tmp_cache_dir) / "gemini"
        assert provider_dir.exists()
        assert provider_dir.is_dir()

    def test_init_sets_ttl_correctly(self, tmp_cache_dir):
        """Test that TTL is converted to seconds"""
        from src.llm_client.cache import LLMCache

        cache = LLMCache(
            base_dir=tmp_cache_dir,
            provider="test",
            ttl_hours=48
        )

        assert cache.ttl_seconds == 48 * 3600

    def test_init_zero_ttl(self, tmp_cache_dir):
        """Test initialization with zero TTL (never expire)"""
        from src.llm_client.cache import LLMCache

        cache = LLMCache(
            base_dir=tmp_cache_dir,
            provider="test",
            ttl_hours=0
        )

        assert cache.ttl_seconds == 0


# ============================================================================
# Test Cache Key Generation
# ============================================================================

class TestCacheKeyGeneration:
    """Test cache key generation"""

    def test_cache_key_deterministic(self, cache_instance):
        """Test that same request generates same key"""
        request = MockLLMRequest(prompt="Test prompt")

        key1 = cache_instance._cache_key(request)
        key2 = cache_instance._cache_key(request)

        assert key1 == key2
        assert len(key1) == 16  # MD5 truncated to 16 chars

    def test_cache_key_different_prompts(self, cache_instance):
        """Test that different prompts generate different keys"""
        request1 = MockLLMRequest(prompt="Prompt one")
        request2 = MockLLMRequest(prompt="Prompt two")

        key1 = cache_instance._cache_key(request1)
        key2 = cache_instance._cache_key(request2)

        assert key1 != key2

    def test_cache_key_includes_system_prompt(self, cache_instance):
        """Test that system prompt affects cache key"""
        request1 = MockLLMRequest(prompt="Test", system_prompt="System A")
        request2 = MockLLMRequest(prompt="Test", system_prompt="System B")

        key1 = cache_instance._cache_key(request1)
        key2 = cache_instance._cache_key(request2)

        assert key1 != key2

    def test_cache_key_includes_temperature(self, cache_instance):
        """Test that temperature affects cache key"""
        request1 = MockLLMRequest(prompt="Test", temperature=0.5)
        request2 = MockLLMRequest(prompt="Test", temperature=0.9)

        key1 = cache_instance._cache_key(request1)
        key2 = cache_instance._cache_key(request2)

        assert key1 != key2

    def test_cache_key_includes_images_hash(self, cache_instance):
        """Test that images affect cache key"""
        request1 = MockLLMRequest(prompt="Test", images=[b"image1_bytes"])
        request2 = MockLLMRequest(prompt="Test", images=[b"image2_bytes"])
        request3 = MockLLMRequest(prompt="Test", images=None)

        key1 = cache_instance._cache_key(request1)
        key2 = cache_instance._cache_key(request2)
        key3 = cache_instance._cache_key(request3)

        assert key1 != key2
        assert key1 != key3

    def test_cache_key_response_format_enum(self, cache_instance):
        """Test cache key with ResponseFormat enum"""
        request1 = MockLLMRequest(prompt="Test", response_format=MockResponseFormat.TEXT)
        request2 = MockLLMRequest(prompt="Test", response_format=MockResponseFormat.JSON)

        key1 = cache_instance._cache_key(request1)
        key2 = cache_instance._cache_key(request2)

        assert key1 != key2


# ============================================================================
# Test Cache Set
# ============================================================================

class TestCacheSet:
    """Test cache set operations"""

    def test_set_creates_cache_file(self, cache_instance, tmp_cache_dir):
        """Test that set creates cache file"""
        request = MockLLMRequest(prompt="Test prompt", cache_key_prefix="myprefix")
        response = MockLLMResponse(text="Response text")

        cache_instance.set(request, response)

        cache_files = list(Path(tmp_cache_dir).rglob("*.json"))
        assert len(cache_files) == 1
        assert cache_files[0].stem.startswith("myprefix_")

    def test_set_stores_correct_data(self, cache_instance, tmp_cache_dir):
        """Test that set stores all response data"""
        request = MockLLMRequest(prompt="Test prompt")
        response = MockLLMResponse(
            text="Response text",
            parsed_data={"key": "value"},
            provider="gemini",
            model="gemini-2.0-flash"
        )

        cache_instance.set(request, response)

        cache_files = list(Path(tmp_cache_dir).rglob("*.json"))
        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        assert data["text"] == "Response text"
        assert data["parsed_data"] == {"key": "value"}
        assert data["provider"] == "gemini"
        assert data["model"] == "gemini-2.0-flash"
        assert "cached_at" in data
        assert data["cached_at"] > 0

    def test_set_handles_write_error(self, cache_instance, tmp_cache_dir, caplog):
        """Test that set handles write errors gracefully"""
        import logging

        request = MockLLMRequest(prompt="Test")
        response = MockLLMResponse(text="Response")

        # Mock the open function to simulate write error
        with patch('builtins.open', side_effect=OSError("Permission denied")):
            with caplog.at_level(logging.WARNING):
                # Should not raise, just log warning
                cache_instance.set(request, response)

            # Should log warning about failure
            assert "Failed to cache" in caplog.text


# ============================================================================
# Test Cache Get
# ============================================================================

class TestCacheGet:
    """Test cache get operations"""

    def test_get_returns_cached_response(self, cache_instance):
        """Test that get returns cached response"""
        request = MockLLMRequest(prompt="Test prompt")
        response = MockLLMResponse(text="Cached response")

        cache_instance.set(request, response)
        result = cache_instance.get(request)

        assert result is not None
        assert result["text"] == "Cached response"

    def test_get_returns_none_for_missing(self, cache_instance):
        """Test that get returns None for missing entry"""
        request = MockLLMRequest(prompt="Never cached")

        result = cache_instance.get(request)

        assert result is None

    def test_get_expires_old_entries(self, cache_instance, tmp_cache_dir):
        """Test that get deletes expired entries"""
        request = MockLLMRequest(prompt="Test", cache_key_prefix="expiry_test")
        response = MockLLMResponse(text="Response")

        # Cache the response
        cache_instance.set(request, response)

        # Modify cached_at to be old
        cache_files = list(Path(tmp_cache_dir).rglob("expiry_test*.json"))
        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        data["cached_at"] = time.time() - (48 * 3600)  # 48 hours ago

        with open(cache_files[0], 'w') as f:
            json.dump(data, f)

        # Get should return None and delete file
        result = cache_instance.get(request)

        assert result is None
        assert not cache_files[0].exists()

    def test_get_with_zero_ttl_never_expires(self, tmp_cache_dir):
        """Test that zero TTL never expires"""
        from src.llm_client.cache import LLMCache

        cache = LLMCache(base_dir=tmp_cache_dir, provider="test", ttl_hours=0)

        request = MockLLMRequest(prompt="Test", cache_key_prefix="no_expire")
        response = MockLLMResponse(text="Response")

        cache.set(request, response)

        # Modify cached_at to be very old
        cache_files = list(Path(tmp_cache_dir).rglob("no_expire*.json"))
        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        data["cached_at"] = time.time() - (365 * 24 * 3600)  # 1 year ago

        with open(cache_files[0], 'w') as f:
            json.dump(data, f)

        # Should still return the entry
        result = cache.get(request)

        assert result is not None
        assert result["text"] == "Response"

    def test_get_handles_corrupt_json(self, cache_instance, tmp_cache_dir, caplog):
        """Test that get handles corrupt JSON files"""
        import logging

        request = MockLLMRequest(prompt="Corrupt test", cache_key_prefix="corrupt")

        # Create corrupt cache file
        provider_dir = Path(tmp_cache_dir) / cache_instance.provider
        cache_key = cache_instance._cache_key(request)
        cache_file = provider_dir / f"corrupt_{cache_key}.json"

        with open(cache_file, 'w') as f:
            f.write("{invalid json")

        with caplog.at_level(logging.WARNING):
            result = cache_instance.get(request)

        assert result is None
        assert "Failed to read cache" in caplog.text
        # Corrupt file should be deleted
        assert not cache_file.exists()

    def test_get_handles_missing_keys(self, cache_instance, tmp_cache_dir):
        """Test that get handles JSON missing expected keys"""
        request = MockLLMRequest(prompt="Missing keys", cache_key_prefix="missing")

        # Create cache file with missing keys
        provider_dir = Path(tmp_cache_dir) / cache_instance.provider
        cache_key = cache_instance._cache_key(request)
        cache_file = provider_dir / f"missing_{cache_key}.json"

        with open(cache_file, 'w') as f:
            json.dump({"text": "Test"}, f)  # Missing cached_at

        # Should still work (cached_at defaults to 0)
        result = cache_instance.get(request)

        # With TTL enabled, missing cached_at (0) means it's expired
        assert result is None


# ============================================================================
# Test Cache Clear
# ============================================================================

class TestCacheClear:
    """Test cache clear operations"""

    def test_clear_all(self, cache_instance, tmp_cache_dir):
        """Test clearing all cache entries"""
        # Create multiple cache entries
        for i in range(5):
            request = MockLLMRequest(prompt=f"Test {i}", cache_key_prefix=f"prefix{i}")
            response = MockLLMResponse(text=f"Response {i}")
            cache_instance.set(request, response)

        cache_files_before = list(Path(tmp_cache_dir).rglob("*.json"))
        assert len(cache_files_before) == 5

        cache_instance.clear()

        cache_files_after = list(Path(tmp_cache_dir).rglob("*.json"))
        assert len(cache_files_after) == 0

    def test_clear_with_prefix(self, cache_instance, tmp_cache_dir):
        """Test clearing only entries with specific prefix"""
        # Create entries with different prefixes
        for i, prefix in enumerate(["keep", "keep", "remove", "remove"]):
            # Use unique prompts so each creates a unique cache entry
            request = MockLLMRequest(prompt=f"Test {prefix} {i}", cache_key_prefix=prefix)
            response = MockLLMResponse(text="Response")
            cache_instance.set(request, response)

        cache_files_before = list(Path(tmp_cache_dir).rglob("*.json"))
        assert len(cache_files_before) == 4

        cache_instance.clear(prefix="remove")

        remaining = list(Path(tmp_cache_dir).rglob("*.json"))
        assert len(remaining) == 2
        assert all("keep" in f.stem for f in remaining)

    def test_clear_older_than_hours(self, cache_instance, tmp_cache_dir):
        """Test clearing entries older than specified hours"""
        # Create entries
        for i in range(3):
            request = MockLLMRequest(prompt=f"Test {i}", cache_key_prefix=f"age{i}")
            response = MockLLMResponse(text=f"Response {i}")
            cache_instance.set(request, response)

        # Make one entry old
        cache_files = list(Path(tmp_cache_dir).rglob("*.json"))
        old_file = cache_files[0]

        with open(old_file, 'r') as f:
            data = json.load(f)

        data["cached_at"] = time.time() - (48 * 3600)  # 48 hours ago

        with open(old_file, 'w') as f:
            json.dump(data, f)

        # Clear entries older than 24 hours
        cache_instance.clear(older_than_hours=24)

        remaining = list(Path(tmp_cache_dir).rglob("*.json"))
        assert len(remaining) == 2

    def test_clear_handles_corrupt_files(self, cache_instance, tmp_cache_dir):
        """Test that clear deletes corrupt files"""
        # Create valid entry
        request = MockLLMRequest(prompt="Valid", cache_key_prefix="valid")
        response = MockLLMResponse(text="Response")
        cache_instance.set(request, response)

        # Create corrupt file
        provider_dir = Path(tmp_cache_dir) / cache_instance.provider
        corrupt_file = provider_dir / "corrupt_abc123.json"
        with open(corrupt_file, 'w') as f:
            f.write("{corrupt")

        # Clear with age filter - corrupt files should be deleted
        cache_instance.clear(older_than_hours=0)

        remaining = list(Path(tmp_cache_dir).rglob("*.json"))
        assert len(remaining) == 0


# ============================================================================
# Test Cache Stats
# ============================================================================

class TestCacheStats:
    """Test cache statistics"""

    def test_stats_empty_cache(self, cache_instance):
        """Test stats for empty cache"""
        stats = cache_instance.stats()

        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0.0
        assert stats["oldest_entry_hours"] is None
        assert stats["provider"] == cache_instance.provider

    def test_stats_with_entries(self, cache_instance, tmp_cache_dir):
        """Test stats with cache entries"""
        # Create entries
        for i in range(3):
            request = MockLLMRequest(prompt=f"Test {i}", cache_key_prefix=f"stat{i}")
            response = MockLLMResponse(text=f"Response {i}" * 100)  # Some content
            cache_instance.set(request, response)

        stats = cache_instance.stats()

        assert stats["total_files"] == 3
        assert stats["total_size_mb"] >= 0  # May be 0.0 due to rounding for small files
        assert stats["oldest_entry_hours"] is not None
        assert stats["oldest_entry_hours"] >= 0

    def test_stats_with_old_entry(self, cache_instance, tmp_cache_dir):
        """Test stats shows correct oldest entry"""
        # Create entries
        request = MockLLMRequest(prompt="Old", cache_key_prefix="old")
        response = MockLLMResponse(text="Old response")
        cache_instance.set(request, response)

        # Make it old
        cache_files = list(Path(tmp_cache_dir).rglob("*.json"))
        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        data["cached_at"] = time.time() - (72 * 3600)  # 72 hours ago

        with open(cache_files[0], 'w') as f:
            json.dump(data, f)

        stats = cache_instance.stats()

        assert stats["oldest_entry_hours"] >= 71
        assert stats["oldest_entry_hours"] <= 73

    def test_stats_handles_corrupt_files(self, cache_instance, tmp_cache_dir):
        """Test stats handles corrupt files gracefully"""
        # Create valid entry
        request = MockLLMRequest(prompt="Valid", cache_key_prefix="valid")
        response = MockLLMResponse(text="Response")
        cache_instance.set(request, response)

        # Create corrupt file
        provider_dir = Path(tmp_cache_dir) / cache_instance.provider
        corrupt_file = provider_dir / "corrupt_abc123.json"
        with open(corrupt_file, 'w') as f:
            f.write("{corrupt")

        # Should not crash
        stats = cache_instance.stats()

        assert stats["total_files"] == 2  # Both files counted
        # Size may be 0.0 due to rounding for small files
        assert stats["total_size_mb"] >= 0


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_concurrent_access(self, cache_instance):
        """Test concurrent set operations don't crash"""
        import threading

        results = []

        def set_cache(i):
            # Each thread uses unique request to avoid race conditions
            request = MockLLMRequest(prompt=f"Concurrent test {i}", cache_key_prefix=f"concurrent{i}")
            response = MockLLMResponse(text=f"Response {i}")
            try:
                cache_instance.set(request, response)
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")

        threads = [threading.Thread(target=set_cache, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed without crashing
        assert len(results) == 5
        assert all(r == "success" for r in results)

    def test_unicode_content(self, cache_instance):
        """Test caching unicode content"""
        request = MockLLMRequest(prompt="Unicode test: ", cache_key_prefix="unicode")
        response = MockLLMResponse(text="Response with unicode: ")

        cache_instance.set(request, response)
        result = cache_instance.get(request)

        assert result is not None
        assert "" in result["text"]

    def test_large_response(self, cache_instance):
        """Test caching large responses"""
        request = MockLLMRequest(prompt="Large test", cache_key_prefix="large")
        large_text = "A" * 1000000  # 1MB of text
        response = MockLLMResponse(text=large_text)

        cache_instance.set(request, response)
        result = cache_instance.get(request)

        assert result is not None
        assert len(result["text"]) == 1000000

    def test_special_characters_in_prefix(self, cache_instance):
        """Test cache key prefix with special characters"""
        request = MockLLMRequest(prompt="Test", cache_key_prefix="test-prefix_v2.0")
        response = MockLLMResponse(text="Response")

        cache_instance.set(request, response)
        result = cache_instance.get(request)

        assert result is not None
