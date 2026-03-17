"""
Tests for src/cache/types.py

Tests CacheStats and CacheConfig dataclasses.
"""

import pytest
from src.cache.types import CacheStats, CacheConfig


class TestCacheStats:
    """Test CacheStats dataclass"""

    @pytest.mark.fast
    def test_cache_stats_creation(self):
        """Test creating CacheStats with required fields"""
        stats = CacheStats(
            total_entries=100,
            cache_size_mb=25.5,
            cache_dir="/path/to/cache",
            index_file="/path/to/index.json",
            ttl_seconds=3600
        )
        assert stats.total_entries == 100
        assert stats.cache_size_mb == 25.5
        assert stats.cache_dir == "/path/to/cache"
        assert stats.index_file == "/path/to/index.json"
        assert stats.ttl_seconds == 3600
        assert stats.oldest_entry is None
        assert stats.newest_entry is None

    @pytest.mark.fast
    def test_cache_stats_with_optional_fields(self):
        """Test CacheStats with optional entry timestamps"""
        stats = CacheStats(
            total_entries=50,
            cache_size_mb=10.0,
            cache_dir="/cache",
            index_file="/cache/index.json",
            ttl_seconds=7200,
            oldest_entry="2025-01-01T00:00:00",
            newest_entry="2025-01-10T12:00:00"
        )
        assert stats.oldest_entry == "2025-01-01T00:00:00"
        assert stats.newest_entry == "2025-01-10T12:00:00"

    @pytest.mark.fast
    def test_cache_stats_str(self):
        """Test CacheStats string representation"""
        stats = CacheStats(
            total_entries=42,
            cache_size_mb=15.75,
            cache_dir="/cache",
            index_file="/cache/index.json",
            ttl_seconds=1800
        )
        result = str(stats)

        assert "CacheStats" in result
        assert "entries=42" in result
        assert "size=15.75MB" in result
        assert "ttl=1800s" in result

    @pytest.mark.fast
    def test_cache_stats_str_formatting(self):
        """Test CacheStats string formats size to 2 decimal places"""
        stats = CacheStats(
            total_entries=0,
            cache_size_mb=1.12345,
            cache_dir="/cache",
            index_file="/cache/index.json",
            ttl_seconds=60
        )
        result = str(stats)

        # Should format to 2 decimal places
        assert "size=1.12MB" in result


class TestCacheConfig:
    """Test CacheConfig dataclass"""

    @pytest.mark.fast
    def test_cache_config_defaults(self):
        """Test CacheConfig with default values"""
        config = CacheConfig(cache_dir="/cache")

        assert config.cache_dir == "/cache"
        assert config.enabled is True
        assert config.ttl_hours == 24
        assert config.max_size_mb == 0
        assert config.auto_cleanup is True

    @pytest.mark.fast
    def test_cache_config_custom_values(self):
        """Test CacheConfig with custom values"""
        config = CacheConfig(
            cache_dir="/custom/cache",
            enabled=False,
            ttl_hours=48,
            max_size_mb=100,
            auto_cleanup=False
        )

        assert config.cache_dir == "/custom/cache"
        assert config.enabled is False
        assert config.ttl_hours == 48
        assert config.max_size_mb == 100
        assert config.auto_cleanup is False

    @pytest.mark.fast
    def test_cache_config_ttl_seconds_property(self):
        """Test ttl_seconds property converts hours to seconds"""
        config = CacheConfig(cache_dir="/cache", ttl_hours=24)
        assert config.ttl_seconds == 24 * 3600  # 86400 seconds

        config2 = CacheConfig(cache_dir="/cache", ttl_hours=1)
        assert config2.ttl_seconds == 3600

        config3 = CacheConfig(cache_dir="/cache", ttl_hours=0)
        assert config3.ttl_seconds == 0

    @pytest.mark.fast
    def test_cache_config_negative_ttl_raises_error(self):
        """Test that negative ttl_hours raises ValueError"""
        with pytest.raises(ValueError, match="ttl_hours must be non-negative"):
            CacheConfig(cache_dir="/cache", ttl_hours=-1)

    @pytest.mark.fast
    def test_cache_config_negative_max_size_raises_error(self):
        """Test that negative max_size_mb raises ValueError"""
        with pytest.raises(ValueError, match="max_size_mb must be non-negative"):
            CacheConfig(cache_dir="/cache", max_size_mb=-100)

    @pytest.mark.fast
    def test_cache_config_both_negative_raises_first_error(self):
        """Test validation order when both values are negative"""
        # ttl_hours is validated first
        with pytest.raises(ValueError, match="ttl_hours must be non-negative"):
            CacheConfig(cache_dir="/cache", ttl_hours=-5, max_size_mb=-50)

    @pytest.mark.fast
    def test_cache_config_zero_values_valid(self):
        """Test that zero values are valid (edge case)"""
        config = CacheConfig(
            cache_dir="/cache",
            ttl_hours=0,
            max_size_mb=0
        )
        assert config.ttl_hours == 0
        assert config.max_size_mb == 0
        assert config.ttl_seconds == 0
