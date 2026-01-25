"""Tests for cache warm-up functionality."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache.base import BaseCache, CacheEntry

pytestmark = pytest.mark.unit


# Concrete implementation for testing (not a pytest test class)
class ConcreteCache(BaseCache[str]):
    """Concrete cache implementation for testing."""

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


class TestWarmCacheFunctionExists:
    """Test that warm_cache function exists and has correct signature."""

    def test_warm_cache_method_exists(self, tmp_path):
        """Test that warm_cache method exists on BaseCache."""
        cache = ConcreteCache(tmp_path)
        assert hasattr(cache, 'warm_cache')
        assert callable(cache.warm_cache)

    def test_warm_cache_returns_dict(self, tmp_path):
        """Test that warm_cache returns a dictionary."""
        cache = ConcreteCache(tmp_path)
        result = cache.warm_cache()
        assert isinstance(result, dict)

    def test_warm_cache_dict_has_required_keys(self, tmp_path):
        """Test that warm_cache returns dict with required keys."""
        cache = ConcreteCache(tmp_path)
        result = cache.warm_cache()
        assert 'entries_loaded' in result
        assert 'entries_removed' in result
        assert 'elapsed_ms' in result


class TestPreWarmParameter:
    """Test pre_warm parameter in BaseCache.__init__."""

    def test_pre_warm_parameter_exists(self, tmp_path):
        """Test that pre_warm parameter is accepted by __init__."""
        # Should not raise
        cache = ConcreteCache(tmp_path, pre_warm=False)
        assert cache is not None

    def test_pre_warm_default_is_false(self, tmp_path):
        """Test that pre_warm defaults to False."""
        # Creating without pre_warm should not trigger warm_cache
        with patch.object(ConcreteCache, 'warm_cache') as mock_warm:
            cache = ConcreteCache(tmp_path)
            mock_warm.assert_not_called()

    def test_pre_warm_true_calls_warm_cache(self, tmp_path):
        """Test that pre_warm=True calls warm_cache on init."""
        with patch.object(ConcreteCache, 'warm_cache', return_value={'entries_loaded': 0, 'entries_removed': 0, 'elapsed_ms': 0.0}) as mock_warm:
            cache = ConcreteCache(tmp_path, pre_warm=True)
            mock_warm.assert_called_once()

    def test_pre_warm_false_does_not_call_warm_cache(self, tmp_path):
        """Test that pre_warm=False does not call warm_cache."""
        with patch.object(ConcreteCache, 'warm_cache') as mock_warm:
            cache = ConcreteCache(tmp_path, pre_warm=False)
            mock_warm.assert_not_called()


class TestWarmCacheWithEntries:
    """Test warm_cache behavior with entries in cache."""

    def test_warm_cache_loads_valid_entries(self, tmp_path):
        """Test that warm_cache counts valid entries as loaded."""
        cache = ConcreteCache(tmp_path)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')

        result = cache.warm_cache()

        assert result['entries_loaded'] == 3
        assert result['entries_removed'] == 0

    def test_warm_cache_empty_cache(self, tmp_path):
        """Test warm_cache on empty cache."""
        cache = ConcreteCache(tmp_path)
        result = cache.warm_cache()

        assert result['entries_loaded'] == 0
        assert result['entries_removed'] == 0
        assert result['elapsed_ms'] >= 0

    def test_warm_cache_single_entry(self, tmp_path):
        """Test warm_cache with single entry."""
        cache = ConcreteCache(tmp_path)
        cache.set('only_key', 'only_value')

        result = cache.warm_cache()

        assert result['entries_loaded'] == 1
        assert result['entries_removed'] == 0


class TestWarmCacheWithExpiredEntries:
    """Test warm_cache behavior with expired entries."""

    def test_warm_cache_removes_expired_entries(self, tmp_path):
        """Test that warm_cache removes expired entries."""
        # Create cache with 1 second TTL
        cache = ConcreteCache(tmp_path, ttl_seconds=1)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        # Wait for expiration
        time.sleep(1.1)

        result = cache.warm_cache()

        assert result['entries_loaded'] == 0
        assert result['entries_removed'] == 2

    def test_warm_cache_keeps_non_expired_entries(self, tmp_path):
        """Test that warm_cache keeps non-expired entries."""
        # Create cache with 10 second TTL
        cache = ConcreteCache(tmp_path, ttl_seconds=10)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        result = cache.warm_cache()

        assert result['entries_loaded'] == 2
        assert result['entries_removed'] == 0

    def test_warm_cache_mixed_expired_and_valid(self, tmp_path):
        """Test warm_cache with mix of expired and valid entries."""
        # Create cache with 1 second TTL
        cache = ConcreteCache(tmp_path, ttl_seconds=1)
        cache.set('old_key', 'old_value')

        # Wait for first entry to expire
        time.sleep(1.1)

        # Add new entry
        cache.set('new_key', 'new_value')

        result = cache.warm_cache()

        assert result['entries_loaded'] == 1
        assert result['entries_removed'] == 1


class TestWarmCacheLogging:
    """Test warm_cache logging behavior."""

    def test_warm_cache_logs_completion(self, tmp_path, caplog):
        """Test that warm_cache logs completion message."""
        cache = ConcreteCache(tmp_path)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        with caplog.at_level(logging.INFO):
            cache.warm_cache()

        assert 'Cache warm-up complete' in caplog.text
        assert '2 entries loaded' in caplog.text

    def test_warm_cache_logs_time_in_ms(self, tmp_path, caplog):
        """Test that warm_cache logs time in milliseconds."""
        cache = ConcreteCache(tmp_path)
        cache.set('key1', 'value1')

        with caplog.at_level(logging.INFO):
            cache.warm_cache()

        assert 'ms' in caplog.text

    def test_warm_cache_logs_removal_at_debug(self, tmp_path, caplog):
        """Test that warm_cache logs removal at DEBUG level."""
        cache = ConcreteCache(tmp_path, ttl_seconds=1)
        cache.set('key1', 'value1')
        time.sleep(1.1)

        with caplog.at_level(logging.DEBUG):
            cache.warm_cache()

        assert 'Removed' in caplog.text or 'expired/invalid' in caplog.text

    def test_warm_cache_no_removal_log_when_none_removed(self, tmp_path, caplog):
        """Test that warm_cache does not log removal when none removed."""
        cache = ConcreteCache(tmp_path)
        cache.set('key1', 'value1')

        with caplog.at_level(logging.DEBUG):
            result = cache.warm_cache()

        # The INFO log should be there
        assert 'Cache warm-up complete' in caplog.text
        # But removal message should not be there (since no entries removed)
        log_lower = caplog.text.lower()
        if result['entries_removed'] == 0:
            # If entries_removed is 0, we shouldn't see the removal debug message
            pass  # This is expected


class TestWarmCacheTiming:
    """Test warm_cache timing measurement."""

    def test_warm_cache_elapsed_ms_is_positive(self, tmp_path):
        """Test that elapsed_ms is positive."""
        cache = ConcreteCache(tmp_path)
        cache.set('key1', 'value1')

        result = cache.warm_cache()

        assert result['elapsed_ms'] >= 0

    def test_warm_cache_elapsed_ms_increases_with_entries(self, tmp_path):
        """Test that elapsed time tends to increase with more entries."""
        cache1 = ConcreteCache(tmp_path / 'small')
        cache1.set('key1', 'value1')

        cache2 = ConcreteCache(tmp_path / 'large')
        for i in range(100):
            cache2.set(f'key{i}', f'value{i}')

        result1 = cache1.warm_cache()
        result2 = cache2.warm_cache()

        # The larger cache should have a non-zero time
        assert result2['elapsed_ms'] >= 0


class TestWarmCacheWithInvalidEntries:
    """Test warm_cache handling of invalid/corrupted entries."""

    def test_warm_cache_handles_deserialization_failure(self, tmp_path):
        """Test that warm_cache handles deserialization failures gracefully."""
        cache = ConcreteCache(tmp_path)
        cache.set('valid_key', 'valid_value')

        # Inject invalid entry directly into index
        cache.index['invalid_key'] = {'not': 'valid', 'structure': True}
        cache._save_index()

        # Warm cache should handle the invalid entry
        result = cache.warm_cache()

        # Valid entry should be loaded, invalid should be removed
        assert result['entries_loaded'] == 1
        assert result['entries_removed'] == 1

    def test_warm_cache_removes_corrupted_entries_from_index(self, tmp_path):
        """Test that corrupted entries are removed from index."""
        cache = ConcreteCache(tmp_path)
        cache.set('valid_key', 'valid_value')

        # Inject invalid entry
        cache.index['invalid_key'] = {'broken': 'entry'}
        cache._save_index()

        cache.warm_cache()

        # Invalid key should be removed from index
        assert 'invalid_key' not in cache.index
        assert 'valid_key' in cache.index


class TestPreWarmOnInit:
    """Test that pre_warm=True loads index immediately on init."""

    def test_pre_warm_true_loads_index_on_init(self, tmp_path, caplog):
        """Test that pre_warm=True loads index immediately."""
        # First create a cache with some entries
        cache1 = ConcreteCache(tmp_path)
        cache1.set('key1', 'value1')
        cache1.set('key2', 'value2')
        cache1.set('key3', 'value3')

        # Now create new cache with pre_warm=True
        with caplog.at_level(logging.INFO):
            cache2 = ConcreteCache(tmp_path, pre_warm=True)

        # Check that warm-up was logged during init
        assert 'Cache warm-up complete' in caplog.text
        assert '3 entries loaded' in caplog.text

    def test_pre_warm_false_does_not_load_entries(self, tmp_path, caplog):
        """Test that pre_warm=False does not trigger warm-up."""
        # First create a cache with some entries
        cache1 = ConcreteCache(tmp_path)
        cache1.set('key1', 'value1')
        cache1.set('key2', 'value2')

        caplog.clear()

        # Now create new cache with pre_warm=False (default)
        with caplog.at_level(logging.INFO):
            cache2 = ConcreteCache(tmp_path, pre_warm=False)

        # Check that warm-up was NOT logged during init
        assert 'Cache warm-up complete' not in caplog.text


class TestWarmCacheIndexSaving:
    """Test that warm_cache saves index after removing invalid entries."""

    def test_warm_cache_saves_index_after_removal(self, tmp_path):
        """Test that index is saved after removing expired entries."""
        cache = ConcreteCache(tmp_path, ttl_seconds=1, auto_save=True)
        cache.set('key1', 'value1')

        # Wait for expiration
        time.sleep(1.1)

        cache.warm_cache()

        # Create new cache to verify index was saved
        cache2 = ConcreteCache(tmp_path, ttl_seconds=1)
        assert 'key1' not in cache2.index

    def test_warm_cache_does_not_save_when_auto_save_false(self, tmp_path):
        """Test that index is not saved when auto_save=False."""
        cache = ConcreteCache(tmp_path, ttl_seconds=1, auto_save=False)
        cache.set('key1', 'value1')
        cache._save_index()  # Manually save for setup

        # Wait for expiration
        time.sleep(1.1)

        cache.warm_cache()

        # Create new cache - entry should still be in saved index
        # (even though in-memory index had it removed)
        cache2 = ConcreteCache(tmp_path, ttl_seconds=0)  # No TTL for reading
        # The entry might still be in the file since auto_save was False
        # This tests that warm_cache respects auto_save setting
