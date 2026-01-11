"""
Tests for src/config/sections/entity.py coverage gaps.

Targets:
- Line 137: stock_video dict conversion in __post_init__
- Line 139: entity_cache dict conversion in __post_init__
"""

import pytest
from src.config.sections.entity import (
    ImageSearchConfig,
    StockVideoConfig,
    EntityCacheConfig,
    SilentVideoConfig,
)


class TestImageSearchConfigPostInit:
    """Test ImageSearchConfig __post_init__ dict conversions."""

    def test_stock_video_dict_converted(self):
        """Test line 137: stock_video dict converted to StockVideoConfig."""
        config = ImageSearchConfig(
            stock_video={"min_duration": 5.0, "max_duration": 60.0, "prefer_hd": False}
        )

        assert isinstance(config.stock_video, StockVideoConfig)
        assert config.stock_video.min_duration == 5.0
        assert config.stock_video.max_duration == 60.0
        assert config.stock_video.prefer_hd is False

    def test_entity_cache_dict_converted(self):
        """Test line 139: entity_cache dict converted to EntityCacheConfig."""
        config = ImageSearchConfig(
            entity_cache={
                "enabled": True,
                "cache_dir": "/custom/cache",
                "fuzzy_threshold": 0.9
            }
        )

        assert isinstance(config.entity_cache, EntityCacheConfig)
        assert config.entity_cache.enabled is True
        assert config.entity_cache.cache_dir == "/custom/cache"
        assert config.entity_cache.fuzzy_threshold == 0.9

    def test_both_dicts_converted(self):
        """Test both nested configs converted from dicts."""
        config = ImageSearchConfig(
            stock_video={"min_duration": 2.0},
            entity_cache={"enabled": True, "max_age_days": 30}
        )

        assert isinstance(config.stock_video, StockVideoConfig)
        assert isinstance(config.entity_cache, EntityCacheConfig)
        assert config.stock_video.min_duration == 2.0
        assert config.entity_cache.max_age_days == 30

    def test_dataclass_objects_unchanged(self):
        """Test that dataclass instances are not modified."""
        stock_config = StockVideoConfig(min_duration=10.0)
        cache_config = EntityCacheConfig(enabled=True)

        config = ImageSearchConfig(
            stock_video=stock_config,
            entity_cache=cache_config
        )

        # Should still be the same objects
        assert config.stock_video is stock_config
        assert config.entity_cache is cache_config


class TestStockVideoConfig:
    """Test StockVideoConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = StockVideoConfig()

        assert config.min_duration == 3.0
        assert config.max_duration == 30.0
        assert config.prefer_hd is True

    def test_custom_values(self):
        """Test custom values."""
        config = StockVideoConfig(
            min_duration=1.0,
            max_duration=120.0,
            prefer_hd=False
        )

        assert config.min_duration == 1.0
        assert config.max_duration == 120.0
        assert config.prefer_hd is False


class TestEntityCacheConfig:
    """Test EntityCacheConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = EntityCacheConfig()

        assert config.enabled is False
        assert config.cache_dir == "~/.matcher_entity_cache"
        assert config.fuzzy_threshold == 0.85
        assert config.max_age_days == 0
        assert config.cache_strategy == "copy"

    def test_custom_values(self):
        """Test custom values."""
        config = EntityCacheConfig(
            enabled=True,
            cache_dir="/my/cache",
            fuzzy_threshold=0.95,
            max_age_days=7,
            cache_strategy="symlink"
        )

        assert config.enabled is True
        assert config.cache_dir == "/my/cache"
        assert config.fuzzy_threshold == 0.95
        assert config.max_age_days == 7
        assert config.cache_strategy == "symlink"


class TestSilentVideoConfig:
    """Test SilentVideoConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = SilentVideoConfig()

        assert config.enabled is True
        assert config.min_words_threshold == 10
        assert config.use_vision_api is True
        assert config.use_llm_fallback is True
        assert config.cache_descriptions is True

    def test_custom_values(self):
        """Test custom values."""
        config = SilentVideoConfig(
            enabled=False,
            min_words_threshold=5,
            use_vision_api=False
        )

        assert config.enabled is False
        assert config.min_words_threshold == 5
        assert config.use_vision_api is False


class TestImageSearchConfig:
    """Test ImageSearchConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = ImageSearchConfig()

        assert config.enabled is True
        assert config.root_dir == ""
        assert config.images_per_entity == 5
        assert config.videos_per_entity == 3
        assert config.use_google is True
        assert config.use_bing is False
        assert config.use_stock_apis is True
        assert "PERSON" in config.entity_types

    def test_entity_types_default(self):
        """Test default entity types."""
        config = ImageSearchConfig()

        expected = ["PERSON", "GPE", "ORG", "DATE", "EVENT"]
        assert config.entity_types == expected

    def test_custom_entity_types(self):
        """Test custom entity types."""
        config = ImageSearchConfig(entity_types=["PERSON", "GPE"])

        assert config.entity_types == ["PERSON", "GPE"]
