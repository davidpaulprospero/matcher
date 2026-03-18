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
    SilentVideoConfig,
)


class TestImageSearchConfigPostInit:
    """Test ImageSearchConfig __post_init__ dict conversions."""

    @pytest.mark.fast
    def test_stock_video_dict_converted(self):
        """Test line 137: stock_video dict converted to StockVideoConfig."""
        config = ImageSearchConfig(
            stock_video={"min_duration": 5.0, "max_duration": 60.0, "prefer_hd": False}
        )

        assert isinstance(config.stock_video, StockVideoConfig)
        assert config.stock_video.min_duration == 5.0
        assert config.stock_video.max_duration == 60.0
        assert config.stock_video.prefer_hd is False

    @pytest.mark.fast
    def test_dataclass_objects_unchanged(self):
        """Test that dataclass instances are not modified."""
        stock_config = StockVideoConfig(min_duration=10.0)

        config = ImageSearchConfig(
            stock_video=stock_config,
        )

        # Should still be the same objects
        assert config.stock_video is stock_config


class TestStockVideoConfig:
    """Test StockVideoConfig dataclass."""

    @pytest.mark.fast
    def test_defaults(self):
        """Test default values."""
        config = StockVideoConfig()

        assert config.min_duration == 3.0
        assert config.max_duration == 30.0
        assert config.prefer_hd is True

    @pytest.mark.fast
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


class TestSilentVideoConfig:
    """Test SilentVideoConfig dataclass."""

    @pytest.mark.fast
    def test_defaults(self):
        """Test default values."""
        config = SilentVideoConfig()

        assert config.enabled is True
        assert config.min_words_threshold == 10
        assert config.use_vision_api is True
        assert config.use_llm_fallback is True
        assert config.cache_descriptions is True

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_entity_types_default(self):
        """Test default entity types."""
        config = ImageSearchConfig()

        expected = ["PERSON", "GPE", "ORG", "DATE", "EVENT"]
        assert config.entity_types == expected

    @pytest.mark.fast
    def test_custom_entity_types(self):
        """Test custom entity types."""
        config = ImageSearchConfig(entity_types=["PERSON", "GPE"])

        assert config.entity_types == ["PERSON", "GPE"]
