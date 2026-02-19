"""
Unit tests for config/sections/keywords module.

Tests KeywordConfig and ListDetectionConfig dataclasses from src/config/sections/keywords.py.
"""

import pytest
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.sections.keywords import KeywordConfig, ListDetectionConfig


class TestListDetectionConfigDefaults:
    """Test ListDetectionConfig initialization with default values."""

    @pytest.mark.fast
    def test_default_values(self):
        """Test initialization with default values."""
        config = ListDetectionConfig()

        assert config.enabled is True
        assert config.download_first is True
        assert config.skip_if_entity_covered is True
        assert config.keyword_suffix == "footage"

    @pytest.mark.fast
    def test_all_defaults_match_docstring(self):
        """Test that defaults match documented values."""
        config = ListDetectionConfig()

        assert config.enabled is True
        assert config.download_first is True  # Download list keywords before general keywords
        assert config.skip_if_entity_covered is True  # Skip if already in entity extraction
        assert config.keyword_suffix == "footage"  # Suffix for generated keywords


class TestListDetectionConfigCustom:
    """Test ListDetectionConfig with custom values."""

    @pytest.mark.fast
    def test_custom_values(self):
        """Test initialization with custom values."""
        config = ListDetectionConfig(
            enabled=False,
            download_first=False,
            skip_if_entity_covered=False,
            keyword_suffix="video"
        )

        assert config.enabled is False
        assert config.download_first is False
        assert config.skip_if_entity_covered is False
        assert config.keyword_suffix == "video"

    @pytest.mark.fast
    def test_custom_suffix_variations(self):
        """Test with various keyword suffixes."""
        suffixes = ["clip", "video", "broll", "footage", "media"]
        for suffix in suffixes:
            config = ListDetectionConfig(keyword_suffix=suffix)
            assert config.keyword_suffix == suffix


class TestListDetectionConfigSerialization:
    """Test ListDetectionConfig serialization/deserialization."""

    @pytest.mark.fast
    def test_to_dict(self):
        """Test ListDetectionConfig can be converted to dict."""
        config = ListDetectionConfig(
            enabled=True,
            download_first=False,
            skip_if_entity_covered=True,
            keyword_suffix="clip"
        )

        result = {
            'enabled': config.enabled,
            'download_first': config.download_first,
            'skip_if_entity_covered': config.skip_if_entity_covered,
            'keyword_suffix': config.keyword_suffix
        }

        assert result['enabled'] is True
        assert result['download_first'] is False
        assert result['skip_if_entity_covered'] is True
        assert result['keyword_suffix'] == "clip"

    @pytest.mark.fast
    def test_roundtrip_via_dict(self):
        """Test ListDetectionConfig survives dict roundtrip."""
        original = ListDetectionConfig(
            enabled=True,
            download_first=True,
            skip_if_entity_covered=False,
            keyword_suffix="broll"
        )

        # Serialize to dict
        data = {
            'enabled': original.enabled,
            'download_first': original.download_first,
            'skip_if_entity_covered': original.skip_if_entity_covered,
            'keyword_suffix': original.keyword_suffix
        }

        # Deserialize back to dataclass
        restored = ListDetectionConfig(**data)

        assert restored.enabled is True
        assert restored.download_first is True
        assert restored.skip_if_entity_covered is False
        assert restored.keyword_suffix == "broll"

    @pytest.mark.fast
    def test_yaml_roundtrip(self):
        """Test ListDetectionConfig survives YAML roundtrip."""
        original = ListDetectionConfig(
            enabled=False,
            download_first=True,
            skip_if_entity_covered=True,
            keyword_suffix="video"
        )

        # Serialize to YAML
        yaml_str = yaml.dump({
            'enabled': original.enabled,
            'download_first': original.download_first,
            'skip_if_entity_covered': original.skip_if_entity_covered,
            'keyword_suffix': original.keyword_suffix
        })

        # Deserialize from YAML
        data = yaml.safe_load(yaml_str)
        restored = ListDetectionConfig(**data)

        assert restored.enabled is False
        assert restored.download_first is True
        assert restored.skip_if_entity_covered is True
        assert restored.keyword_suffix == "video"


class TestKeywordConfigDefaults:
    """Test KeywordConfig initialization with default values."""

    @pytest.mark.fast
    def test_default_values(self):
        """Test initialization with default values."""
        config = KeywordConfig()

        # Provider
        assert config.provider == "gemini"

        # Extraction settings
        assert config.max_keywords == 30
        assert config.min_keyword_length == 3
        assert config.max_keyword_words == 8
        assert config.segments_per_query == 3

        # List detection (should be converted to ListDetectionConfig in __post_init__)
        assert config.list_detection is not None
        assert isinstance(config.list_detection, ListDetectionConfig)

        # Entity extraction
        assert config.extract_entities is True
        assert config.entity_types == ["PERSON", "GPE", "ORG", "DATE", "EVENT"]

        # TF-IDF fallback
        assert config.use_tfidf_weights is True
        assert config.tfidf_max_features == 100

        # Footage suffixes
        assert config.add_footage_suffixes is True
        assert config.footage_suffixes == [
            "4K footage", "news footage", "drone footage",
            "aerial footage", "stock footage", "documentary footage"
        ]

        # Batch processing
        assert config.batch_size == 50

    @pytest.mark.fast
    def test_all_defaults_match_docstring(self):
        """Test that defaults match documented values."""
        config = KeywordConfig()

        # Provider
        assert config.provider == "gemini"  # gemini, anthropic, tfidf

        # Extraction settings
        assert config.max_keywords == 30
        assert config.min_keyword_length == 3
        assert config.max_keyword_words == 8  # Max words per keyword
        assert config.segments_per_query == 3  # Number of segments grouped per search query


class TestKeywordConfigCustom:
    """Test KeywordConfig with custom values."""

    @pytest.mark.fast
    def test_custom_provider(self):
        """Test with custom provider."""
        config = KeywordConfig(provider="anthropic")
        assert config.provider == "anthropic"

        config = KeywordConfig(provider="tfidf")
        assert config.provider == "tfidf"

    @pytest.mark.fast
    def test_custom_extraction_settings(self):
        """Test with custom extraction settings."""
        config = KeywordConfig(
            max_keywords=50,
            min_keyword_length=5,
            max_keyword_words=10,
            segments_per_query=5
        )

        assert config.max_keywords == 50
        assert config.min_keyword_length == 5
        assert config.max_keyword_words == 10
        assert config.segments_per_query == 5

    @pytest.mark.fast
    def test_custom_entity_types(self):
        """Test with custom entity types."""
        config = KeywordConfig(entity_types=["PERSON", "ORG", "FAC"])
        assert config.entity_types == ["PERSON", "ORG", "FAC"]

    @pytest.mark.fast
    def test_custom_footage_suffixes(self):
        """Test with custom footage suffixes."""
        custom_suffixes = ["4k video", "stock video", "broll"]
        config = KeywordConfig(footage_suffixes=custom_suffixes)
        assert config.footage_suffixes == custom_suffixes

    @pytest.mark.fast
    def test_custom_batch_size(self):
        """Test with custom batch size."""
        config = KeywordConfig(batch_size=100)
        assert config.batch_size == 100

    @pytest.mark.fast
    def test_toggle_entity_extraction(self):
        """Test toggling entity extraction."""
        config = KeywordConfig(extract_entities=False)
        assert config.extract_entities is False

    @pytest.mark.fast
    def test_toggle_tfidf_weights(self):
        """Test toggling TF-IDF weights."""
        config = KeywordConfig(use_tfidf_weights=False)
        assert config.use_tfidf_weights is False

    @pytest.mark.fast
    def test_toggle_footage_suffixes(self):
        """Test toggling footage suffixes."""
        config = KeywordConfig(add_footage_suffixes=False)
        assert config.add_footage_suffixes is False


class TestKeywordConfigPostInit:
    """Test KeywordConfig __post_init__ dict conversion."""

    @pytest.mark.fast
    def test_post_init_converts_dict_list_detection(self):
        """Test __post_init__ converts dict list_detection to ListDetectionConfig."""
        # Create config with list_detection as dict (like YAML would provide)
        config = KeywordConfig(
            provider="anthropic",
            list_detection={
                'enabled': False,
                'download_first': False,
                'keyword_suffix': "video"
            }
        )

        # Should convert dict to ListDetectionConfig
        assert isinstance(config.list_detection, ListDetectionConfig)
        assert config.list_detection.enabled is False
        assert config.list_detection.download_first is False
        assert config.list_detection.keyword_suffix == "video"

    @pytest.mark.fast
    def test_post_init_handles_none_list_detection(self):
        """Test __post_init__ handles None list_detection."""
        config = KeywordConfig(list_detection=None)

        # Should create default list_detection
        assert isinstance(config.list_detection, ListDetectionConfig)
        assert config.list_detection.enabled is True

    @pytest.mark.fast
    def test_post_init_handles_empty_dict(self):
        """Test __post_init__ handles empty dict list_detection."""
        config = KeywordConfig(list_detection={})

        # Should create list_detection with defaults
        assert isinstance(config.list_detection, ListDetectionConfig)
        assert config.list_detection.enabled is True

    @pytest.mark.fast
    def test_post_init_preserves_list_detection_object(self):
        """Test __post_init__ preserves existing ListDetectionConfig object."""
        original = ListDetectionConfig(enabled=False, keyword_suffix="clip")
        config = KeywordConfig(list_detection=original)

        # Should preserve the object
        assert config.list_detection is original
        assert config.list_detection.enabled is False
        assert config.list_detection.keyword_suffix == "clip"


class TestKeywordConfigSerialization:
    """Test KeywordConfig serialization/deserialization."""

    @pytest.mark.fast
    def test_to_dict_basic(self):
        """Test KeywordConfig can be converted to dict."""
        config = KeywordConfig(
            provider="anthropic",
            max_keywords=40,
            batch_size=75
        )

        result = {
            'provider': config.provider,
            'max_keywords': config.max_keywords,
            'batch_size': config.batch_size,
            'list_detection': {
                'enabled': config.list_detection.enabled,
                'download_first': config.list_detection.download_first,
                'skip_if_entity_covered': config.list_detection.skip_if_entity_covered,
                'keyword_suffix': config.list_detection.keyword_suffix
            }
        }

        assert result['provider'] == "anthropic"
        assert result['max_keywords'] == 40
        assert result['batch_size'] == 75
        assert result['list_detection']['enabled'] is True

    @pytest.mark.fast
    def test_roundtrip_via_dict(self):
        """Test KeywordConfig survives dict roundtrip."""
        original = KeywordConfig(
            provider="tfidf",
            max_keywords=25,
            min_keyword_length=4,
            extract_entities=False
        )

        # Serialize to dict
        data = {
            'provider': original.provider,
            'max_keywords': original.max_keywords,
            'min_keyword_length': original.min_keyword_length,
            'extract_entities': original.extract_entities,
            'list_detection': {
                'enabled': original.list_detection.enabled,
                'download_first': original.list_detection.download_first,
                'skip_if_entity_covered': original.list_detection.skip_if_entity_covered,
                'keyword_suffix': original.list_detection.keyword_suffix
            }
        }

        # Deserialize back to dataclass
        restored = KeywordConfig(**data)

        assert restored.provider == "tfidf"
        assert restored.max_keywords == 25
        assert restored.min_keyword_length == 4
        assert restored.extract_entities is False

    @pytest.mark.fast
    def test_yaml_roundtrip(self):
        """Test KeywordConfig survives YAML roundtrip."""
        original = KeywordConfig(
            provider="anthropic",
            max_keywords=35,
            batch_size=60,
            list_detection={'enabled': False, 'keyword_suffix': "video"}
        )

        # Serialize to YAML
        yaml_str = yaml.dump({
            'provider': original.provider,
            'max_keywords': original.max_keywords,
            'batch_size': original.batch_size,
            'list_detection': {
                'enabled': original.list_detection.enabled,
                'download_first': original.list_detection.download_first,
                'skip_if_entity_covered': original.list_detection.skip_if_entity_covered,
                'keyword_suffix': original.list_detection.keyword_suffix
            }
        })

        # Deserialize from YAML
        data = yaml.safe_load(yaml_str)
        restored = KeywordConfig(**data)

        assert restored.provider == "anthropic"
        assert restored.max_keywords == 35
        assert restored.batch_size == 60
        assert restored.list_detection.enabled is False
        assert restored.list_detection.keyword_suffix == "video"


class TestKeywordConfigBoundaries:
    """Test boundary conditions and edge cases."""

    @pytest.mark.fast
    def test_zero_max_keywords(self):
        """Test with zero max keywords."""
        config = KeywordConfig(max_keywords=0)
        assert config.max_keywords == 0

    @pytest.mark.fast
    def test_zero_min_keyword_length(self):
        """Test with zero min keyword length."""
        config = KeywordConfig(min_keyword_length=0)
        assert config.min_keyword_length == 0

    @pytest.mark.fast
    def test_zero_max_keyword_words(self):
        """Test with zero max keyword words."""
        config = KeywordConfig(max_keyword_words=0)
        assert config.max_keyword_words == 0

    @pytest.mark.fast
    def test_zero_segments_per_query(self):
        """Test with zero segments per query."""
        config = KeywordConfig(segments_per_query=0)
        assert config.segments_per_query == 0

    @pytest.mark.fast
    def test_zero_batch_size(self):
        """Test with zero batch size."""
        config = KeywordConfig(batch_size=0)
        assert config.batch_size == 0

    @pytest.mark.fast
    def test_large_max_keywords(self):
        """Test with very large max keywords."""
        config = KeywordConfig(max_keywords=1000)
        assert config.max_keywords == 1000

    @pytest.mark.fast
    def test_large_tfidf_max_features(self):
        """Test with very large TF-IDF max features."""
        config = KeywordConfig(tfidf_max_features=10000)
        assert config.tfidf_max_features == 10000


class TestKeywordConfigEquality:
    """Test dataclass equality and comparison."""

    @pytest.mark.fast
    def test_equal_configs(self):
        """Test that identical configs are equal."""
        config1 = KeywordConfig(provider="gemini", max_keywords=30)
        config2 = KeywordConfig(provider="gemini", max_keywords=30)

        assert config1 == config2

    @pytest.mark.fast
    def test_unequal_configs(self):
        """Test that different configs are not equal."""
        config1 = KeywordConfig(provider="gemini")
        config2 = KeywordConfig(provider="anthropic")

        assert config1 != config2


class TestKeywordConfigImmutability:
    """Test that dataclass behaves correctly (no frozen=True, so mutable)."""

    @pytest.mark.fast
    def test_fields_are_modifiable(self):
        """Test that fields can be modified after creation."""
        config = KeywordConfig()
        config.provider = "anthropic"
        config.max_keywords = 50

        assert config.provider == "anthropic"
        assert config.max_keywords == 50


class TestKeywordConfigFallbacks:
    """Test fallback values for missing fields."""

    @pytest.mark.fast
    def test_empty_entity_types_uses_default(self):
        """Test empty entity_types list gets converted to default in __post_init__."""
        # When loading from YAML, empty list might be None or empty dict
        config = KeywordConfig(entity_types=[])
        # The field should use the default_factory
        assert config.entity_types == []

    @pytest.mark.fast
    def test_empty_footage_suffixes_uses_default(self):
        """Test empty footage_suffixes list gets converted to default."""
        config = KeywordConfig(footage_suffixes=[])
        assert config.footage_suffixes == []

    @pytest.mark.fast
    def test_partial_dict_list_detection_uses_defaults(self):
        """Test partial dict for list_detection uses defaults for missing fields."""
        config = KeywordConfig(list_detection={'enabled': False})

        # Should convert to ListDetectionConfig with defaults for missing fields
        assert isinstance(config.list_detection, ListDetectionConfig)
        assert config.list_detection.enabled is False
        assert config.list_detection.download_first is True  # default
        assert config.list_detection.keyword_suffix == "footage"  # default
