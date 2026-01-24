"""
Tests for BaseCache statistics tracking (US-003).

Tests cache hit/miss tracking, bytes_saved, and hit_rate calculation.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache.base import BaseCache, CacheEntry

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


class ConcreteCache(BaseCache[str]):
    """Concrete implementation of BaseCache for testing."""

    def _serialize_entry(self, entry: CacheEntry[str]) -> Dict[str, Any]:
        return {
            'data': entry.data,
            'cached_at': entry.cached_at,
            'key': entry.key,
            'metadata': entry.metadata
        }

    def _deserialize_entry(self, data: Dict[str, Any]) -> CacheEntry[str]:
        return CacheEntry(
            data=data['data'],
            cached_at=data['cached_at'],
            key=data['key'],
            metadata=data.get('metadata', {})
        )


class TestCacheStatsCounters:
    """Test that hits, misses, and bytes_saved counters exist and work."""

    def test_counters_initialized_to_zero(self, tmp_path):
        """Counters should be initialized to zero."""
        cache = ConcreteCache(tmp_path / "cache")
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._bytes_saved == 0

    def test_hit_counter_incremented_on_cache_hit(self, tmp_path):
        """Hit counter should increment when get() returns a cached entry."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "value1")

        # First get should be a hit
        result = cache.get("key1")
        assert result is not None
        assert cache._hits == 1
        assert cache._misses == 0

    def test_miss_counter_incremented_on_cache_miss(self, tmp_path):
        """Miss counter should increment when get() returns None."""
        cache = ConcreteCache(tmp_path / "cache")

        # Get non-existent key should be a miss
        result = cache.get("nonexistent")
        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0

    def test_multiple_hits_and_misses(self, tmp_path):
        """Counters should track multiple hits and misses correctly."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # 2 hits
        cache.get("key1")
        cache.get("key2")

        # 3 misses
        cache.get("missing1")
        cache.get("missing2")
        cache.get("missing3")

        assert cache._hits == 2
        assert cache._misses == 3

    def test_bytes_saved_incremented_on_hit(self, tmp_path):
        """bytes_saved should increase when cache hit occurs."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "some_value_data")

        initial_bytes = cache._bytes_saved
        cache.get("key1")

        assert cache._bytes_saved > initial_bytes


class TestCacheStatsGetStats:
    """Test get_stats() method returns correct statistics."""

    def test_get_stats_returns_hits(self, tmp_path):
        """get_stats() should return hits count."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "value1")
        cache.get("key1")

        stats = cache.get_stats()
        assert 'hits' in stats
        assert stats['hits'] == 1

    def test_get_stats_returns_misses(self, tmp_path):
        """get_stats() should return misses count."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert 'misses' in stats
        assert stats['misses'] == 1

    def test_get_stats_returns_hit_rate(self, tmp_path):
        """get_stats() should return hit_rate."""
        cache = ConcreteCache(tmp_path / "cache")

        stats = cache.get_stats()
        assert 'hit_rate' in stats

    def test_get_stats_returns_bytes_saved(self, tmp_path):
        """get_stats() should return bytes_saved."""
        cache = ConcreteCache(tmp_path / "cache")

        stats = cache.get_stats()
        assert 'bytes_saved' in stats
        assert stats['bytes_saved'] == 0


class TestCacheHitRateCalculation:
    """Test hit_rate calculation is correct."""

    def test_hit_rate_zero_with_no_requests(self, tmp_path):
        """hit_rate should be 0.0 when no requests made."""
        cache = ConcreteCache(tmp_path / "cache")

        stats = cache.get_stats()
        assert stats['hit_rate'] == 0.0

    def test_hit_rate_one_with_all_hits(self, tmp_path):
        """hit_rate should be 1.0 when all requests are hits."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.get("key1")
        cache.get("key2")

        stats = cache.get_stats()
        assert stats['hit_rate'] == 1.0

    def test_hit_rate_zero_with_all_misses(self, tmp_path):
        """hit_rate should be 0.0 when all requests are misses."""
        cache = ConcreteCache(tmp_path / "cache")

        cache.get("missing1")
        cache.get("missing2")

        stats = cache.get_stats()
        assert stats['hit_rate'] == 0.0

    def test_hit_rate_calculation_mixed(self, tmp_path):
        """hit_rate should be hits / (hits + misses)."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "value1")

        # 2 hits (same key twice)
        cache.get("key1")
        cache.get("key1")

        # 2 misses
        cache.get("missing1")
        cache.get("missing2")

        stats = cache.get_stats()
        # 2 hits / (2 hits + 2 misses) = 0.5
        assert stats['hit_rate'] == 0.5

    def test_hit_rate_calculation_75_percent(self, tmp_path):
        """hit_rate should correctly calculate 75%."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "value1")

        # 3 hits
        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        # 1 miss
        cache.get("missing")

        stats = cache.get_stats()
        # 3 hits / (3 hits + 1 miss) = 0.75
        assert stats['hit_rate'] == 0.75


class TestCacheStatsResetStats:
    """Test reset_stats() method."""

    def test_reset_stats_clears_hits(self, tmp_path):
        """reset_stats() should clear hits counter."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "value1")
        cache.get("key1")
        assert cache._hits == 1

        cache.reset_stats()
        assert cache._hits == 0

    def test_reset_stats_clears_misses(self, tmp_path):
        """reset_stats() should clear misses counter."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.get("missing")
        assert cache._misses == 1

        cache.reset_stats()
        assert cache._misses == 0

    def test_reset_stats_clears_bytes_saved(self, tmp_path):
        """reset_stats() should clear bytes_saved counter."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "value1")
        cache.get("key1")
        assert cache._bytes_saved > 0

        cache.reset_stats()
        assert cache._bytes_saved == 0

    def test_reset_stats_affects_get_stats(self, tmp_path):
        """reset_stats() should affect values returned by get_stats()."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("missing")

        cache.reset_stats()

        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['bytes_saved'] == 0
        assert stats['hit_rate'] == 0.0


class TestCacheStatsExpiredEntries:
    """Test stats tracking with expired entries."""

    def test_expired_entry_counts_as_miss(self, tmp_path):
        """Getting an expired entry should count as a miss."""
        cache = ConcreteCache(tmp_path / "cache", ttl_seconds=1)
        cache.set("key1", "value1")

        # Manually expire the entry by modifying cached_at
        cache.index["key1"]["cached_at"] = 0  # Very old timestamp

        result = cache.get("key1")
        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0


class TestCacheStatsExistingStats:
    """Test that existing stats fields are still present."""

    def test_get_stats_still_has_total_entries(self, tmp_path):
        """get_stats() should still return total_entries."""
        cache = ConcreteCache(tmp_path / "cache")
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()
        assert 'total_entries' in stats
        assert stats['total_entries'] == 2

    def test_get_stats_still_has_cache_dir(self, tmp_path):
        """get_stats() should still return cache_dir."""
        cache = ConcreteCache(tmp_path / "cache")

        stats = cache.get_stats()
        assert 'cache_dir' in stats

    def test_get_stats_still_has_ttl_seconds(self, tmp_path):
        """get_stats() should still return ttl_seconds."""
        cache = ConcreteCache(tmp_path / "cache", ttl_seconds=3600)

        stats = cache.get_stats()
        assert 'ttl_seconds' in stats
        assert stats['ttl_seconds'] == 3600
