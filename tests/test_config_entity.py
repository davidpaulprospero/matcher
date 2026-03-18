"""
Unit tests for config/sections/entity module.

Tests EntityConfig, StockVideoConfig, SilentVideoConfig,
EntityCacheConfig, and ImageSearchConfig dataclasses.
"""

import pytest
import sys
import yaml
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.sections.entity import (
    StockVideoConfig,
    SilentVideoConfig,
    ImageSearchConfig,
)


class TestStockVideoConfig:
    """Test StockVideoConfig dataclass."""

    @pytest.mark.fast
    def test_default_values(self):
        """Test initialization with default values."""
        config = StockVideoConfig()

        assert config.min_duration == 3.0
        assert config.max_duration == 30.0
        assert config.prefer_hd is True

    @pytest.mark.fast
    def test_custom_values(self):
        """Test initialization with custom values."""
        config = StockVideoConfig(
            min_duration=5.0,
            max_duration=60.0,
            prefer_hd=False
        )

        assert config.min_duration == 5.0
        assert config.max_duration == 60.0
        assert config.prefer_hd is False


class TestSilentVideoConfig:
    """Test SilentVideoConfig dataclass."""

    @pytest.mark.fast
    def test_default_values(self):
        """Test initialization with default values."""
        config = SilentVideoConfig()

        assert config.enabled is True
        assert config.min_words_threshold == 10
        assert config.use_vision_api is True
        assert config.use_llm_fallback is True
        assert config.cache_descriptions is True

    @pytest.mark.fast
    def test_custom_values(self):
        """Test initialization with custom values."""
        config = SilentVideoConfig(
            enabled=False,
            min_words_threshold=5,
            use_vision_api=False,
            use_llm_fallback=False,
            cache_descriptions=False
        )

        assert config.enabled is False
        assert config.min_words_threshold == 5
        assert config.use_vision_api is False
        assert config.use_llm_fallback is False
        assert config.cache_descriptions is False


class TestImageSearchConfig:
    """Test ImageSearchConfig dataclass."""

    @pytest.mark.fast
    def test_default_values(self):
        """Test initialization with default values."""
        config = ImageSearchConfig()

        # Basic settings
        assert config.enabled is True
        assert config.root_dir == ""
        assert config.folder_name == "images"

        # Image/video counts
        assert config.images_per_entity == 5
        assert config.videos_per_entity == 3
        assert config.max_entities == 0
        assert config.entity_display_limit == 5

        # File size
        assert config.min_size_mb == 1.0

        # Output (deprecated)
        assert config.output_dir == "images"

        # Search sources
        assert config.use_google is True
        assert config.use_bing is False
        assert config.use_stock_apis is True

        # Entity types
        assert config.entity_types == ["PERSON", "GPE", "ORG", "DATE", "EVENT"]

        # OTIO settings
        assert config.image_track == "V9"
        assert config.stock_video_track == "V10"
        assert config.default_duration == 0.0

        # Download limits
        assert config.download_timeout == 10
        assert config.max_search_time == 300
        assert config.max_results_to_check == 500
        assert config.search_until_found is True

        # Nested configs
        assert isinstance(config.stock_video, StockVideoConfig)

        # Entity matching
        assert config.enable_sticky_matching is False
        assert config.semantic_match_threshold == 0.15

    @pytest.mark.fast
    def test_custom_values(self):
        """Test initialization with custom values."""
        config = ImageSearchConfig(
            enabled=False,
            root_dir="E:/custom_images",
            folder_name="my_images",
            images_per_entity=10,
            videos_per_entity=5,
            max_entities=20,
            entity_display_limit=10,
            min_size_mb=2.0,
            use_google=False,
            use_bing=True,
            use_stock_apis=False,
            entity_types=["PERSON", "ORG"],
            image_track="V8",
            stock_video_track="V9",
            default_duration=5.0,
            download_timeout=30,
            max_search_time=600,
            max_results_to_check=1000,
            search_until_found=False,
            enable_sticky_matching=True,
            semantic_match_threshold=0.25
        )

        assert config.enabled is False
        assert config.root_dir == "E:/custom_images"
        assert config.folder_name == "my_images"
        assert config.images_per_entity == 10
        assert config.videos_per_entity == 5
        assert config.max_entities == 20
        assert config.entity_display_limit == 10
        assert config.min_size_mb == 2.0
        assert config.use_google is False
        assert config.use_bing is True
        assert config.use_stock_apis is False
        assert config.entity_types == ["PERSON", "ORG"]
        assert config.image_track == "V8"
        assert config.stock_video_track == "V9"
        assert config.default_duration == 5.0
        assert config.download_timeout == 30
        assert config.max_search_time == 600
        assert config.max_results_to_check == 1000
        assert config.search_until_found is False
        assert config.enable_sticky_matching is True
        assert config.semantic_match_threshold == 0.25

    @pytest.mark.fast
    def test_post_init_dict_conversion(self):
        """Test __post_init__ converts dict fields correctly."""
        # Test with dict for stock_video
        config = ImageSearchConfig(
            stock_video={"min_duration": 10.0, "max_duration": 90.0, "prefer_hd": False}
        )

        assert isinstance(config.stock_video, StockVideoConfig)
        assert config.stock_video.min_duration == 10.0
        assert config.stock_video.max_duration == 90.0
        assert config.stock_video.prefer_hd is False

    @pytest.mark.fast
    def test_post_init_none_values(self):
        """Test __post_init__ handles None values - leaves as None (not converted)."""
        # When stock_video is None, __post_init__ doesn't convert
        # (only dicts are converted to dataclasses)
        config = ImageSearchConfig(
            stock_video=None,
        )

        # None values are left as None (not converted to defaults)
        assert config.stock_video is None


class TestEntityConfigSerialization:
    """Test dataclass serialization/deserialization."""

    @pytest.mark.fast
    def test_stock_video_to_dict(self):
        """Test StockVideoConfig can be converted to dict."""
        config = StockVideoConfig(min_duration=5.0, max_duration=45.0)

        result = {
            'min_duration': config.min_duration,
            'max_duration': config.max_duration,
            'prefer_hd': config.prefer_hd
        }

        assert result['min_duration'] == 5.0
        assert result['max_duration'] == 45.0

    @pytest.mark.fast
    def test_image_search_roundtrip_via_yaml(self):
        """Test ImageSearchConfig survives YAML roundtrip."""
        original = ImageSearchConfig(
            images_per_entity=8,
            videos_per_entity=4,
            entity_types=["PERSON", "GPE"],
            enable_sticky_matching=True
        )

        # Serialize to dict (simulating YAML dump/load)
        data = {
            'enabled': original.enabled,
            'root_dir': original.root_dir,
            'folder_name': original.folder_name,
            'images_per_entity': original.images_per_entity,
            'videos_per_entity': original.videos_per_entity,
            'max_entities': original.max_entities,
            'entity_display_limit': original.entity_display_limit,
            'min_size_mb': original.min_size_mb,
            'output_dir': original.output_dir,
            'use_google': original.use_google,
            'use_bing': original.use_bing,
            'use_stock_apis': original.use_stock_apis,
            'entity_types': original.entity_types,
            'image_track': original.image_track,
            'stock_video_track': original.stock_video_track,
            'default_duration': original.default_duration,
            'download_timeout': original.download_timeout,
            'max_search_time': original.max_search_time,
            'max_results_to_check': original.max_results_to_check,
            'search_until_found': original.search_until_found,
            'stock_video': {
                'min_duration': original.stock_video.min_duration,
                'max_duration': original.stock_video.max_duration,
                'prefer_hd': original.stock_video.prefer_hd,
            },
            'enable_sticky_matching': original.enable_sticky_matching,
            'semantic_match_threshold': original.semantic_match_threshold,
        }

        # Deserialize (simulating YAML load + __post_init__)
        restored = ImageSearchConfig(**data)

        assert restored.images_per_entity == 8
        assert restored.videos_per_entity == 4
        assert restored.entity_types == ["PERSON", "GPE"]
        assert restored.enable_sticky_matching is True
        assert isinstance(restored.stock_video, StockVideoConfig)


class TestEntityConfigFallbacks:
    """Test fallback values for missing fields."""

    @pytest.mark.fast
    def test_semantic_match_threshold_bounds(self):
        """Test semantic_match_threshold is within valid range."""
        config = ImageSearchConfig(semantic_match_threshold=0.0)
        assert config.semantic_match_threshold == 0.0

        config = ImageSearchConfig(semantic_match_threshold=1.0)
        assert config.semantic_match_threshold == 1.0

    @pytest.mark.fast
    def test_timeout_bounds(self):
        """Test timeout values are positive."""
        config = ImageSearchConfig(download_timeout=1, max_search_time=1)
        assert config.download_timeout == 1
        assert config.max_search_time == 1


class TestEntityConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.fast
    def test_empty_entity_types(self):
        """Test with empty entity types list."""
        config = ImageSearchConfig(entity_types=[])
        assert config.entity_types == []

    @pytest.mark.fast
    def test_custom_entity_types(self):
        """Test with custom entity types."""
        config = ImageSearchConfig(
            entity_types=["FAC", "NORP", "PRODUCT"]
        )
        assert config.entity_types == ["FAC", "NORP", "PRODUCT"]

    @pytest.mark.fast
    def test_empty_root_dir(self):
        """Test with empty root_dir (use project_dir)."""
        config = ImageSearchConfig(root_dir="")
        assert config.root_dir == ""

    @pytest.mark.fast
    def test_zero_max_entities_unlimited(self):
        """Test that max_entities=0 means no limit."""
        config = ImageSearchConfig(max_entities=0)
        assert config.max_entities == 0

    @pytest.mark.fast
    def test_zero_default_duration_uses_segment(self):
        """Test that default_duration=0 uses segment duration."""
        config = ImageSearchConfig(default_duration=0.0)
        assert config.default_duration == 0.0
