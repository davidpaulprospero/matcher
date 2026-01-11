"""
Tests for config dataclasses in src/config/

Tests dataclass initialization and validation for:
- DurationTierConfig - Single duration tier
- DurationTiersConfig - All duration tiers
- StockFootageConfig - Stock footage API settings
- DeduplicationConfig - Deduplication settings
- VarietyConfig - Track variety enforcement
- OutputConfig - Output generation settings
- MultiStyleConfig - Multi-style OTIO
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.sections.duration import (
    DurationTierConfig,
    DurationTiersConfig,
    StockFootageConfig
)
from src.config.sections.output import (
    DeduplicationConfig,
    VarietyConfig,
    OutputConfig,
    MultiStyleConfig
)


class TestDurationTierConfig:
    """Test DurationTierConfig dataclass"""

    def test_duration_tier_default_initialization(self):
        """Test default initialization"""
        tier = DurationTierConfig()

        assert tier.min_seconds == 0
        assert tier.max_seconds == 120
        assert tier.videos_per_keyword == 5
        assert tier.max_total == 0

    def test_duration_tier_custom_values(self):
        """Test with custom values"""
        tier = DurationTierConfig(
            min_seconds=120,
            max_seconds=600,
            videos_per_keyword=10,
            max_total=50
        )

        assert tier.min_seconds == 120
        assert tier.max_seconds == 600
        assert tier.videos_per_keyword == 10
        assert tier.max_total == 50

    def test_duration_tier_zero_max_total(self):
        """Test max_total=0 means no limit"""
        tier = DurationTierConfig(max_total=0)
        assert tier.max_total == 0  # No limit

    def test_duration_tier_large_values(self):
        """Test with large duration values"""
        tier = DurationTierConfig(
            min_seconds=3000,
            max_seconds=10000,
            videos_per_keyword=100
        )

        assert tier.min_seconds == 3000
        assert tier.max_seconds == 10000


class TestDurationTiersConfig:
    """Test DurationTiersConfig dataclass"""

    def test_duration_tiers_default_initialization(self):
        """Test default initialization with factory functions"""
        tiers = DurationTiersConfig()

        # Short tier
        assert tiers.short.min_seconds == 20
        assert tiers.short.max_seconds == 120
        assert tiers.short.videos_per_keyword == 8

        # Medium tier
        assert tiers.medium.min_seconds == 120
        assert tiers.medium.max_seconds == 600

        # Long tier
        assert tiers.long.min_seconds == 600
        assert tiers.long.max_seconds == 1500

        # Longer tier (limited to 1 per project)
        assert tiers.longer.min_seconds == 1500
        assert tiers.longer.max_seconds == 3000
        assert tiers.longer.max_total == 1

    def test_duration_tiers_custom_short(self):
        """Test with custom short tier"""
        custom_short = DurationTierConfig(10, 100, 5, 0)
        tiers = DurationTiersConfig(short=custom_short)

        assert tiers.short.min_seconds == 10
        assert tiers.short.max_seconds == 100
        assert tiers.short.videos_per_keyword == 5

    def test_duration_tiers_all_custom(self):
        """Test with all custom tiers"""
        tiers = DurationTiersConfig(
            short=DurationTierConfig(15, 90, 6, 0),
            medium=DurationTierConfig(90, 500, 7, 0),
            long=DurationTierConfig(500, 1200, 4, 0),
            longer=DurationTierConfig(1200, 2500, 2, 2)
        )

        assert tiers.short.max_seconds == 90
        assert tiers.medium.max_seconds == 500
        assert tiers.long.max_seconds == 1200
        assert tiers.longer.max_total == 2


class TestStockFootageConfig:
    """Test StockFootageConfig dataclass"""

    def test_stock_footage_default_initialization(self):
        """Test default initialization"""
        config = StockFootageConfig()

        assert config.enabled is True
        assert config.pexels_enabled is True
        assert config.pixabay_enabled is True
        assert config.per_keyword == 3
        assert config.min_duration == 5
        assert config.max_duration == 60
        assert config.min_height == 720
        assert config.prefer_landscape is True
        assert config.request_interval == 0.5

    def test_stock_footage_all_disabled(self):
        """Test with all APIs disabled"""
        config = StockFootageConfig(
            enabled=False,
            pexels_enabled=False,
            pixabay_enabled=False
        )

        assert config.enabled is False
        assert config.pexels_enabled is False
        assert config.pixabay_enabled is False

    def test_stock_footage_custom_duration(self):
        """Test with custom duration limits"""
        config = StockFootageConfig(
            min_duration=10,
            max_duration=120
        )

        assert config.min_duration == 10
        assert config.max_duration == 120

    def test_stock_footage_high_resolution(self):
        """Test with 4K resolution requirement"""
        config = StockFootageConfig(
            min_height=2160  # 4K
        )

        assert config.min_height == 2160

    def test_stock_footage_rate_limiting(self):
        """Test request interval for rate limiting"""
        config = StockFootageConfig(request_interval=1.0)
        assert config.request_interval == 1.0


class TestDeduplicationConfig:
    """Test DeduplicationConfig dataclass"""

    def test_deduplication_default_initialization(self):
        """Test default initialization"""
        config = DeduplicationConfig()

        assert config.enabled is True
        assert config.hash_threshold == 10
        assert config.use_first_frame is True
        assert config.auto_delete is True
        assert config.generate_report is True
        assert config.frame_timeout == 30

    def test_deduplication_disabled(self):
        """Test with deduplication disabled"""
        config = DeduplicationConfig(enabled=False)
        assert config.enabled is False

    def test_deduplication_strict_threshold(self):
        """Test with strict threshold"""
        config = DeduplicationConfig(hash_threshold=5)
        assert config.hash_threshold == 5

    def test_deduplication_lenient_threshold(self):
        """Test with lenient threshold"""
        config = DeduplicationConfig(hash_threshold=20)
        assert config.hash_threshold == 20

    def test_deduplication_no_auto_delete(self):
        """Test with auto_delete disabled"""
        config = DeduplicationConfig(auto_delete=False)
        assert config.auto_delete is False

    def test_deduplication_custom_timeout(self):
        """Test with custom frame extraction timeout"""
        config = DeduplicationConfig(frame_timeout=60)
        assert config.frame_timeout == 60


class TestVarietyConfig:
    """Test VarietyConfig dataclass"""

    def test_variety_default_initialization(self):
        """Test default initialization"""
        config = VarietyConfig()

        assert config.require_different_source is True
        assert config.exclude_same_clip is True
        assert config.min_time_distance == 10.0
        assert config.min_embedding_distance == 0.3
        assert config.enforce_timeline_variety is True
        assert config.timeline_variety_window == 600.0
        assert config.max_source_repeats_in_window == 1

    def test_variety_allow_same_source(self):
        """Test with same source allowed"""
        config = VarietyConfig(require_different_source=False)
        assert config.require_different_source is False

    def test_variety_custom_time_distance(self):
        """Test with custom time distance"""
        config = VarietyConfig(min_time_distance=30.0)
        assert config.min_time_distance == 30.0

    def test_variety_custom_embedding_distance(self):
        """Test with custom embedding distance"""
        config = VarietyConfig(min_embedding_distance=0.5)
        assert config.min_embedding_distance == 0.5

    def test_variety_timeline_enforcement_disabled(self):
        """Test with timeline variety enforcement disabled"""
        config = VarietyConfig(enforce_timeline_variety=False)
        assert config.enforce_timeline_variety is False

    def test_variety_custom_window(self):
        """Test with custom timeline variety window"""
        config = VarietyConfig(
            timeline_variety_window=1200.0,  # 20 minutes
            max_source_repeats_in_window=2
        )

        assert config.timeline_variety_window == 1200.0
        assert config.max_source_repeats_in_window == 2


class TestOutputConfig:
    """Test OutputConfig dataclass"""

    def test_output_default_initialization(self):
        """Test default initialization"""
        config = OutputConfig()

        assert config.output_dir == "output"
        assert config.generate_otio is True
        assert config.split_otio is True
        assert config.otio_clips_per_file == 10
        assert config.generate_edl is True
        assert config.generate_xml is True
        assert config.xml_parts == 2
        assert config.generate_report is True
        assert config.frame_rate == 30.0
        assert config.timeline_start_tc == "01:00:00:00"
        assert config.num_alternatives == 2
        assert config.include_alternatives is True
        assert config.include_strategy_tracks is True
        assert isinstance(config.variety, VarietyConfig)

    def test_output_custom_output_dir(self):
        """Test with custom output directory"""
        config = OutputConfig(output_dir="/custom/output")
        assert config.output_dir == "/custom/output"

    def test_output_disable_all_formats(self):
        """Test with all formats disabled"""
        config = OutputConfig(
            generate_otio=False,
            generate_edl=False,
            generate_xml=False,
            generate_report=False
        )

        assert config.generate_otio is False
        assert config.generate_edl is False
        assert config.generate_xml is False
        assert config.generate_report is False

    def test_output_custom_frame_rate(self):
        """Test with custom frame rate"""
        config = OutputConfig(frame_rate=60.0)
        assert config.frame_rate == 60.0

    def test_output_custom_timecode(self):
        """Test with custom start timecode"""
        config = OutputConfig(timeline_start_tc="10:00:00:00")
        assert config.timeline_start_tc == "10:00:00:00"

    def test_output_variety_dict_conversion(self):
        """Test __post_init__ converts dict to VarietyConfig"""
        # Create config with variety as dict
        variety_dict = {
            'require_different_source': False,
            'min_time_distance': 20.0
        }
        config = OutputConfig(variety=variety_dict)

        # Should be converted to VarietyConfig
        assert isinstance(config.variety, VarietyConfig)
        assert config.variety.require_different_source is False
        assert config.variety.min_time_distance == 20.0

    def test_output_variety_object(self):
        """Test with variety as VarietyConfig object"""
        variety = VarietyConfig(min_time_distance=15.0)
        config = OutputConfig(variety=variety)

        assert isinstance(config.variety, VarietyConfig)
        assert config.variety.min_time_distance == 15.0

    def test_output_custom_alternatives(self):
        """Test with custom number of alternatives"""
        config = OutputConfig(num_alternatives=3)
        assert config.num_alternatives == 3

    def test_output_strategy_tracks(self):
        """Test default strategy tracks"""
        config = OutputConfig()

        assert len(config.strategy_tracks) == 2
        assert "embedding_diversity" in config.strategy_tracks
        assert "broll_only" in config.strategy_tracks

    def test_output_custom_strategy_tracks(self):
        """Test with custom strategy tracks"""
        custom_tracks = ["custom_strategy_1", "custom_strategy_2"]
        config = OutputConfig(strategy_tracks=custom_tracks)

        assert config.strategy_tracks == custom_tracks
        assert len(config.strategy_tracks) == 2

    def test_output_split_otio_settings(self):
        """Test OTIO splitting configuration"""
        config = OutputConfig(
            split_otio=True,
            otio_clips_per_file=5
        )

        assert config.split_otio is True
        assert config.otio_clips_per_file == 5


class TestMultiStyleConfig:
    """Test MultiStyleConfig dataclass"""

    def test_multi_style_default_initialization(self):
        """Test default initialization"""
        config = MultiStyleConfig()

        assert config.enabled is False
        assert config.styles == ["default", "strict"]

    def test_multi_style_enabled(self):
        """Test with multi-style enabled"""
        config = MultiStyleConfig(enabled=True)
        assert config.enabled is True

    def test_multi_style_custom_styles(self):
        """Test with custom styles"""
        custom_styles = ["conservative", "aggressive", "balanced"]
        config = MultiStyleConfig(
            enabled=True,
            styles=custom_styles
        )

        assert config.enabled is True
        assert config.styles == custom_styles
        assert len(config.styles) == 3

    def test_multi_style_single_style(self):
        """Test with single style"""
        config = MultiStyleConfig(styles=["default"])
        assert len(config.styles) == 1


class TestConfigEdgeCases:
    """Test edge cases for config dataclasses"""

    def test_duration_tier_negative_values(self):
        """Test duration tier with negative values (invalid but allowed)"""
        tier = DurationTierConfig(min_seconds=-10, max_seconds=-5)

        assert tier.min_seconds == -10
        assert tier.max_seconds == -5

    def test_stock_footage_zero_duration(self):
        """Test stock footage with zero duration"""
        config = StockFootageConfig(min_duration=0, max_duration=0)

        assert config.min_duration == 0
        assert config.max_duration == 0

    def test_deduplication_zero_threshold(self):
        """Test deduplication with zero threshold (identical only)"""
        config = DeduplicationConfig(hash_threshold=0)
        assert config.hash_threshold == 0

    def test_variety_zero_distances(self):
        """Test variety with zero distances"""
        config = VarietyConfig(
            min_time_distance=0.0,
            min_embedding_distance=0.0
        )

        assert config.min_time_distance == 0.0
        assert config.min_embedding_distance == 0.0

    def test_output_empty_strategy_tracks(self):
        """Test output with empty strategy tracks"""
        config = OutputConfig(strategy_tracks=[])
        assert config.strategy_tracks == []

    def test_multi_style_empty_styles(self):
        """Test multi-style with empty styles list"""
        config = MultiStyleConfig(styles=[])
        assert config.styles == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
