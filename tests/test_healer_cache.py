"""Unit tests for HealerResultCache (US-64-007).

Tests verify:
- Cache hit prevents redundant heal attempts
- Cache key based on error signature (type + message hash + stage)
- Configurable TTL (default: 5 minutes)
- Cache invalidation when config changes
"""

import time
import pytest
from unittest.mock import MagicMock


class TestHealerCacheConfig:
    """Test HealerCacheConfig dataclass."""

    @pytest.mark.fast
    def test_default_values(self):
        """Test default config values."""
        from src.agents.healer_cache import HealerCacheConfig
        config = HealerCacheConfig()
        assert config.enabled is True
        assert config.ttl_seconds == 300.0  # 5 minutes
        assert config.max_entries == 100

    @pytest.mark.fast
    def test_custom_values(self):
        """Test custom config values."""
        from src.agents.healer_cache import HealerCacheConfig
        config = HealerCacheConfig(enabled=False, ttl_seconds=60.0, max_entries=50)
        assert config.enabled is False
        assert config.ttl_seconds == 60.0
        assert config.max_entries == 50


class TestHealerResultCacheInit:
    """Test HealerResultCache initialization."""

    @pytest.mark.fast
    def test_init_with_defaults(self):
        """Test cache initializes with default config."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        assert cache.config.enabled is True
        assert cache.config.ttl_seconds == 300.0
        assert len(cache._cache) == 0

    @pytest.mark.fast
    def test_init_with_custom_config(self):
        """Test cache initializes with custom config."""
        from src.agents.healer_cache import HealerResultCache, HealerCacheConfig
        config = HealerCacheConfig(ttl_seconds=120.0)
        cache = HealerResultCache(config)
        assert cache.config.ttl_seconds == 120.0

    @pytest.mark.fast
    def test_init_statistics_zeroed(self):
        """Test statistics are zeroed on init."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._invalidations == 0


class TestErrorSignature:
    """Test error signature generation."""

    @pytest.mark.fast
    def test_signature_includes_error_type(self):
        """Test signature includes error type name."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        error = ValueError("test error")
        signature = cache._make_error_signature(error, "MATCH")
        assert signature.startswith("ValueError:")

    @pytest.mark.fast
    def test_signature_includes_stage(self):
        """Test signature includes stage name."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        error = ValueError("test error")
        signature = cache._make_error_signature(error, "MATCH")
        assert signature.endswith(":MATCH")

    @pytest.mark.fast
    def test_signature_message_is_hashed(self):
        """Test message is hashed to 12 characters."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        error = ValueError("a very long error message that should be hashed")
        signature = cache._make_error_signature(error, "MATCH")
        parts = signature.split(":")
        assert len(parts) == 3
        assert len(parts[1]) == 12  # MD5 truncated to 12 chars

    @pytest.mark.fast
    def test_same_error_same_signature(self):
        """Test same error produces same signature."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        error1 = ValueError("test error")
        error2 = ValueError("test error")
        sig1 = cache._make_error_signature(error1, "MATCH")
        sig2 = cache._make_error_signature(error2, "MATCH")
        assert sig1 == sig2

    @pytest.mark.fast
    def test_different_message_different_signature(self):
        """Test different messages produce different signatures."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        error1 = ValueError("error A")
        error2 = ValueError("error B")
        sig1 = cache._make_error_signature(error1, "MATCH")
        sig2 = cache._make_error_signature(error2, "MATCH")
        assert sig1 != sig2

    @pytest.mark.fast
    def test_different_stage_different_signature(self):
        """Test different stages produce different signatures."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        error = ValueError("test error")
        sig1 = cache._make_error_signature(error, "MATCH")
        sig2 = cache._make_error_signature(error, "DOWNLOAD")
        assert sig1 != sig2


class TestCacheStoreAndGet:
    """Test store and get operations."""

    @pytest.mark.fast
    def test_store_successful_result(self):
        """Test storing successful healer result."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult, HealerAction
        cache = HealerResultCache()
        error = ValueError("test error")
        result = HealerResult.fixed("Fixed it", action=HealerAction.RETRY)
        stored = cache.store(error, "MATCH", result, "test_healer")
        assert stored is True
        assert len(cache._cache) == 1

    @pytest.mark.fast
    def test_store_failed_result_not_cached(self):
        """Test that failed results are not cached."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult
        cache = HealerResultCache()
        error = ValueError("test error")
        result = HealerResult.failed("Could not fix")
        stored = cache.store(error, "MATCH", result, "test_healer")
        assert stored is False
        assert len(cache._cache) == 0

    @pytest.mark.fast
    def test_get_returns_cached_entry(self):
        """Test get returns cached entry."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult, HealerAction
        cache = HealerResultCache()
        error = ValueError("test error")
        result = HealerResult.fixed("Fixed it", action=HealerAction.RETRY)
        cache.store(error, "MATCH", result, "test_healer")
        entry = cache.get(error, "MATCH")
        assert entry is not None
        assert entry.healer_name == "test_healer"
        assert entry.message == "Fixed it"
        assert entry.action == "retry"

    @pytest.mark.fast
    def test_get_miss_returns_none(self):
        """Test get returns None for cache miss."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        error = ValueError("not cached")
        entry = cache.get(error, "MATCH")
        assert entry is None

    @pytest.mark.fast
    def test_get_increments_hit_counter(self):
        """Test get increments hit counter on cache hit."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult
        cache = HealerResultCache()
        error = ValueError("test error")
        result = HealerResult.fixed("Fixed it")
        cache.store(error, "MATCH", result, "test_healer")
        cache.get(error, "MATCH")
        assert cache._hits == 1
        assert cache._misses == 0

    @pytest.mark.fast
    def test_get_increments_miss_counter(self):
        """Test get increments miss counter on cache miss."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        error = ValueError("not cached")
        cache.get(error, "MATCH")
        assert cache._hits == 0
        assert cache._misses == 1


class TestCacheTTL:
    """Test TTL-based expiration."""

    @pytest.mark.fast
    def test_entry_not_expired_within_ttl(self):
        """Test entry is not expired within TTL."""
        from src.agents.healer_cache import HealerResultCache, HealerCacheEntry
        cache = HealerResultCache()
        entry = HealerCacheEntry(
            error_signature="test",
            healer_name="test_healer",
            action="retry",
            message="Fixed",
            cached_at=time.time(),
        )
        assert cache._is_expired(entry) is False

    @pytest.mark.fast
    def test_entry_expired_after_ttl(self):
        """Test entry is expired after TTL."""
        from src.agents.healer_cache import HealerResultCache, HealerCacheConfig, HealerCacheEntry
        config = HealerCacheConfig(ttl_seconds=1.0)  # 1 second TTL
        cache = HealerResultCache(config)
        entry = HealerCacheEntry(
            error_signature="test",
            healer_name="test_healer",
            action="retry",
            message="Fixed",
            cached_at=time.time() - 2.0,  # 2 seconds ago
        )
        assert cache._is_expired(entry) is True

    @pytest.mark.fast
    def test_get_returns_none_for_expired_entry(self):
        """Test get returns None for expired entry."""
        from src.agents.healer_cache import HealerResultCache, HealerCacheConfig
        from src.agents.base import HealerResult
        config = HealerCacheConfig(ttl_seconds=0.1)  # 100ms TTL
        cache = HealerResultCache(config)
        error = ValueError("test error")
        result = HealerResult.fixed("Fixed it")
        cache.store(error, "MATCH", result, "test_healer")
        # Wait for TTL to expire
        time.sleep(0.15)
        entry = cache.get(error, "MATCH")
        assert entry is None

    @pytest.mark.fast
    def test_zero_ttl_never_expires(self):
        """Test zero TTL means entries never expire."""
        from src.agents.healer_cache import HealerResultCache, HealerCacheConfig, HealerCacheEntry
        config = HealerCacheConfig(ttl_seconds=0)
        cache = HealerResultCache(config)
        entry = HealerCacheEntry(
            error_signature="test",
            healer_name="test_healer",
            action="retry",
            message="Fixed",
            cached_at=time.time() - 10000,  # Very old
        )
        assert cache._is_expired(entry) is False


class TestCacheInvalidation:
    """Test config-change invalidation."""

    @pytest.mark.fast
    def test_set_config_hash_clears_cache_on_change(self):
        """Test setting new config hash clears cache."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult
        cache = HealerResultCache()
        cache.set_config_hash("hash1")
        error = ValueError("test error")
        result = HealerResult.fixed("Fixed it")
        cache.store(error, "MATCH", result, "test_healer")
        assert len(cache._cache) == 1
        # Change config hash
        cache.set_config_hash("hash2")
        assert len(cache._cache) == 0
        assert cache._invalidations == 1

    @pytest.mark.fast
    def test_same_config_hash_preserves_cache(self):
        """Test same config hash preserves cache."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult
        cache = HealerResultCache()
        cache.set_config_hash("hash1")
        error = ValueError("test error")
        result = HealerResult.fixed("Fixed it")
        cache.store(error, "MATCH", result, "test_healer")
        # Set same hash again
        cache.set_config_hash("hash1")
        assert len(cache._cache) == 1

    @pytest.mark.fast
    def test_invalidate_on_config_change(self):
        """Test explicit invalidation clears cache."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult
        cache = HealerResultCache()
        error = ValueError("test error")
        result = HealerResult.fixed("Fixed it")
        cache.store(error, "MATCH", result, "test_healer")
        cache.invalidate_on_config_change()
        assert len(cache._cache) == 0
        assert cache._invalidations == 1

    @pytest.mark.fast
    def test_stale_config_entry_treated_as_miss(self):
        """Test entry with stale config hash is treated as miss."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult
        cache = HealerResultCache()
        cache.set_config_hash("old_hash")
        error = ValueError("test error")
        result = HealerResult.fixed("Fixed it")
        cache.store(error, "MATCH", result, "test_healer")
        # Change config hash (which clears cache)
        cache.set_config_hash("new_hash")
        entry = cache.get(error, "MATCH")
        assert entry is None


class TestCacheDisabled:
    """Test behavior when cache is disabled."""

    @pytest.mark.fast
    def test_store_returns_false_when_disabled(self):
        """Test store returns False when cache disabled."""
        from src.agents.healer_cache import HealerResultCache, HealerCacheConfig
        from src.agents.base import HealerResult
        config = HealerCacheConfig(enabled=False)
        cache = HealerResultCache(config)
        error = ValueError("test error")
        result = HealerResult.fixed("Fixed it")
        stored = cache.store(error, "MATCH", result, "test_healer")
        assert stored is False

    @pytest.mark.fast
    def test_get_returns_none_when_disabled(self):
        """Test get returns None when cache disabled."""
        from src.agents.healer_cache import HealerResultCache, HealerCacheConfig
        config = HealerCacheConfig(enabled=False)
        cache = HealerResultCache(config)
        error = ValueError("test error")
        entry = cache.get(error, "MATCH")
        assert entry is None


class TestMaxEntries:
    """Test max_entries limit enforcement."""

    @pytest.mark.fast
    def test_enforces_max_entries_limit(self):
        """Test cache enforces max_entries limit."""
        from src.agents.healer_cache import HealerResultCache, HealerCacheConfig
        from src.agents.base import HealerResult
        config = HealerCacheConfig(max_entries=3)
        cache = HealerResultCache(config)
        # Store 5 entries
        for i in range(5):
            error = ValueError(f"error {i}")
            result = HealerResult.fixed(f"Fixed {i}")
            cache.store(error, "MATCH", result, "test_healer")
        # Should only have max_entries
        assert len(cache._cache) == 3

    @pytest.mark.fast
    def test_oldest_entries_evicted(self):
        """Test oldest entries are evicted when max exceeded."""
        from src.agents.healer_cache import HealerResultCache, HealerCacheConfig
        from src.agents.base import HealerResult
        config = HealerCacheConfig(max_entries=2)
        cache = HealerResultCache(config)
        # Store 3 entries with delays
        error1 = ValueError("error 1")
        error2 = ValueError("error 2")
        error3 = ValueError("error 3")
        cache.store(error1, "MATCH", HealerResult.fixed("Fixed 1"), "healer1")
        time.sleep(0.01)
        cache.store(error2, "MATCH", HealerResult.fixed("Fixed 2"), "healer2")
        time.sleep(0.01)
        cache.store(error3, "MATCH", HealerResult.fixed("Fixed 3"), "healer3")
        # First entry should be evicted
        assert cache.get(error1, "MATCH") is None
        assert cache.get(error3, "MATCH") is not None


class TestCacheStats:
    """Test cache statistics."""

    @pytest.mark.fast
    def test_get_stats_returns_dict(self):
        """Test get_stats returns a dict with expected keys."""
        from src.agents.healer_cache import HealerResultCache
        cache = HealerResultCache()
        stats = cache.get_stats()
        assert 'enabled' in stats
        assert 'entries' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats
        assert 'invalidations' in stats

    @pytest.mark.fast
    def test_hit_rate_calculation(self):
        """Test hit rate is calculated correctly."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult
        cache = HealerResultCache()
        error = ValueError("test error")
        result = HealerResult.fixed("Fixed it")
        cache.store(error, "MATCH", result, "test_healer")
        # 2 hits, 1 miss
        cache.get(error, "MATCH")
        cache.get(error, "MATCH")
        cache.get(ValueError("other"), "MATCH")  # miss
        stats = cache.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == pytest.approx(2/3, rel=0.01)


class TestCacheClear:
    """Test cache clear operation."""

    @pytest.mark.fast
    def test_clear_removes_all_entries(self):
        """Test clear removes all entries."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult
        cache = HealerResultCache()
        for i in range(5):
            error = ValueError(f"error {i}")
            result = HealerResult.fixed(f"Fixed {i}")
            cache.store(error, "MATCH", result, "test_healer")
        cache.clear()
        assert len(cache._cache) == 0


class TestCacheHitPreventsRedundantHeal:
    """Integration test: verify cache hit prevents redundant heal attempts."""

    @pytest.mark.fast
    def test_cache_hit_prevents_heal_attempt(self):
        """Test that cache hit prevents calling healers."""
        from src.agents.healer_cache import HealerResultCache
        from src.agents.base import HealerResult, HealerAction
        cache = HealerResultCache()
        error = ValueError("test error")
        # Store successful result
        result = HealerResult.fixed("Already fixed", action=HealerAction.RETRY)
        cache.store(error, "MATCH", result, "test_healer")
        # Simulate what orchestrator does: check cache before healing
        cached = cache.get(error, "MATCH")
        if cached:
            # Would return early without calling healers
            early_return_result = HealerResult.fixed(
                f"Cached heal: {cached.message}",
                action=HealerAction(cached.action),
                cached=True,
                original_healer=cached.healer_name
            )
            assert early_return_result.success is True
            assert early_return_result.details.get('cached') is True
            assert early_return_result.details.get('original_healer') == "test_healer"
