"""
Tests for _build_dataclass section-aware error handling.

US-65-005: Replace silent config loading fallback with explicit error logging.

Covers:
- AC1: Warning logged with failed field names when falling back to empty defaults
- AC2: Critical config sections raise ConfigError
- AC3: Optional config sections fall back silently with debug log
- AC4: CRITICAL_SECTIONS list exists and distinguishes critical from optional
- AC5: Malformed critical section raises ConfigError with descriptive message
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pytest

from src.config.base import Config, ConfigError, CRITICAL_SECTIONS


# ---------------------------------------------------------------------------
# Helpers — dataclasses used only in tests
# ---------------------------------------------------------------------------

@dataclass
class _RequiredFieldConfig:
    """A config that requires a positional argument (no default)."""
    name: str  # required — no default


@dataclass
class _OptionalFieldConfig:
    """A config where every field has a default."""
    name: str = "default"
    count: int = 0


# ---------------------------------------------------------------------------
# AC4: CRITICAL_SECTIONS list exists and distinguishes critical from optional
# ---------------------------------------------------------------------------

class TestCriticalSectionsList:
    """Verify CRITICAL_SECTIONS constant."""

    def test_critical_sections_is_frozenset(self):
        assert isinstance(CRITICAL_SECTIONS, frozenset)

    @pytest.mark.parametrize("section", [
        "download", "downloading", "matching", "output",
        "transcription", "embedding", "pipeline", "llm",
    ])
    def test_core_pipeline_sections_are_critical(self, section):
        assert section in CRITICAL_SECTIONS

    @pytest.mark.parametrize("section", [
        "image_search", "stock_footage", "vision",
        "scene_detection", "audio_analysis", "remix",
        "zero_download_remix", "multi_style", "deduplication",
        "negative_matching",
    ])
    def test_optional_sections_not_critical(self, section):
        assert section not in CRITICAL_SECTIONS


# ---------------------------------------------------------------------------
# AC2 / AC5: Critical config sections raise ConfigError
# ---------------------------------------------------------------------------

class TestCriticalSectionRaisesConfigError:
    """When _build_dataclass fails for a critical section, ConfigError is raised."""

    def test_critical_section_raises_config_error(self):
        """Malformed critical section data that triggers TypeError → ConfigError."""
        # _RequiredFieldConfig needs 'name', but we pass a key that gets
        # filtered out, leaving no 'name' → TypeError in __init__.
        bad_data = {"nonexistent_field": "value"}
        with pytest.raises(ConfigError, match="Critical config section 'download'"):
            Config._build_dataclass(
                _RequiredFieldConfig, bad_data, section_name="download"
            )

    def test_config_error_includes_section_name(self):
        bad_data = {"nonexistent_field": "value"}
        with pytest.raises(ConfigError, match="'matching'"):
            Config._build_dataclass(
                _RequiredFieldConfig, bad_data, section_name="matching"
            )

    def test_config_error_includes_class_name(self):
        bad_data = {"nonexistent_field": "value"}
        with pytest.raises(ConfigError, match="_RequiredFieldConfig"):
            Config._build_dataclass(
                _RequiredFieldConfig, bad_data, section_name="output"
            )

    def test_config_error_includes_field_names(self):
        bad_data = {"nonexistent_field": "value", "another": 42}
        with pytest.raises(ConfigError, match="nonexistent_field"):
            Config._build_dataclass(
                _RequiredFieldConfig, bad_data, section_name="download"
            )

    def test_config_error_is_chained_from_type_error(self):
        bad_data = {"nonexistent_field": "value"}
        with pytest.raises(ConfigError) as exc_info:
            Config._build_dataclass(
                _RequiredFieldConfig, bad_data, section_name="pipeline"
            )
        assert isinstance(exc_info.value.__cause__, TypeError)


# ---------------------------------------------------------------------------
# AC3: Optional config sections fall back with debug-level log
# ---------------------------------------------------------------------------

class TestOptionalSectionFallback:
    """Optional sections return empty defaults and log at DEBUG level."""

    def test_optional_section_debug_log(self, caplog):
        """Optional section logs at DEBUG when falling back."""
        @dataclass
        class _StrictConfig:
            value: int = 0
            def __post_init__(self):
                if not isinstance(self.value, int):
                    raise TypeError("value must be int")

        bad_data = {"value": "not_an_int"}
        with caplog.at_level(logging.DEBUG, logger="src.config.base"):
            result = Config._build_dataclass(
                _StrictConfig, bad_data, section_name="image_search"
            )
        assert result.value == 0  # fell back to default
        assert any("Optional config section" in r.message for r in caplog.records)
        assert any("image_search" in r.message for r in caplog.records)

    def test_no_section_name_treated_as_optional(self, caplog):
        """When section_name is None (e.g. nested build), falls back to defaults."""
        @dataclass
        class _StrictConfig:
            value: int = 0
            def __post_init__(self):
                if not isinstance(self.value, int):
                    raise TypeError("value must be int")

        bad_data = {"value": "not_an_int"}
        with caplog.at_level(logging.DEBUG, logger="src.config.base"):
            result = Config._build_dataclass(_StrictConfig, bad_data)
        assert result.value == 0


# ---------------------------------------------------------------------------
# AC1: Warning includes specific field names that failed
# ---------------------------------------------------------------------------

class TestFailedFieldNamesInLog:
    """The log/error message includes the specific field names that failed."""

    def test_critical_error_lists_provided_fields(self):
        bad_data = {"name": 123, "extra_key": "ignored"}
        # _RequiredFieldConfig only has 'name', 'extra_key' gets filtered.
        # So filtered_data = {'name': 123} → succeeds because name accepts any.
        # Use a config that rejects in __post_init__ instead:
        @dataclass
        class _ValidatingConfig:
            x: int = 0
            def __post_init__(self):
                if self.x < 0:
                    raise TypeError("x must be non-negative")

        bad_data = {"x": -1, "unknown": "val"}
        with pytest.raises(ConfigError, match="Fields provided:.*x.*unknown"):
            Config._build_dataclass(
                _ValidatingConfig, bad_data, section_name="download"
            )

    def test_optional_debug_log_lists_failed_fields(self, caplog):
        @dataclass
        class _ValidatingConfig:
            x: int = 0
            def __post_init__(self):
                if self.x < 0:
                    raise TypeError("x must be non-negative")

        bad_data = {"x": -1, "stray_key": "data"}
        with caplog.at_level(logging.DEBUG, logger="src.config.base"):
            Config._build_dataclass(
                _ValidatingConfig, bad_data, section_name="vision"
            )
        log_text = " ".join(r.message for r in caplog.records)
        assert "Fields that failed" in log_text or "failed" in log_text.lower()


# ---------------------------------------------------------------------------
# Integration: _from_dict propagates section_name correctly
# ---------------------------------------------------------------------------

class TestFromDictSectionPropagation:
    """Verify _from_dict passes section names through to _build_dataclass."""

    def test_valid_config_loads_normally(self):
        """Normal config data should still load without errors."""
        data = {
            "download": {"quality": "720p"},
            "matching": {"min_confidence": 0.5},
            "output": {"frame_rate": 30},
        }
        config = Config._from_dict(data)
        assert config.download.quality == "720p"
        assert config.matching.min_confidence == 0.5
        assert config.output.frame_rate == 30


# ---------------------------------------------------------------------------
# US-69-010: duration_tiers loading uses consistent error handling
# ---------------------------------------------------------------------------

class TestDurationTiersErrorHandling:
    """Malformed duration_tiers raises ConfigError instead of raw exceptions."""

    def test_malformed_duration_tiers_string_raises_config_error(self):
        """Non-dict duration_tiers value raises ConfigError."""
        data = {"duration_tiers": "not_a_dict"}
        with pytest.raises(ConfigError, match="duration_tiers"):
            Config._from_dict(data)

    def test_malformed_duration_tiers_list_raises_config_error(self):
        """List value for duration_tiers raises ConfigError."""
        data = {"duration_tiers": [1, 2, 3]}
        with pytest.raises(ConfigError, match="duration_tiers"):
            Config._from_dict(data)

    def test_malformed_tier_value_raises_config_error(self):
        """Non-dict tier value inside duration_tiers raises ConfigError."""
        data = {"duration_tiers": {"short": "bad_value"}}
        with pytest.raises(ConfigError, match="duration_tiers"):
            Config._from_dict(data)

    def test_config_error_is_chained_from_attribute_error(self):
        """ConfigError preserves the original exception as __cause__ for tier errors."""
        data = {"duration_tiers": {"short": "bad_value"}}
        with pytest.raises(ConfigError) as exc_info:
            Config._from_dict(data)
        assert exc_info.value.__cause__ is not None

    def test_valid_duration_tiers_loads_correctly(self):
        """Valid duration_tiers dict populates DurationTiersConfig correctly."""
        data = {
            "duration_tiers": {
                "short": {"min": 10, "max": 60, "count": 3},
                "medium": {"min": 60, "max": 300, "count": 5},
            }
        }
        config = Config._from_dict(data)
        assert config.duration_tiers.short.min_seconds == 10
        assert config.duration_tiers.short.max_seconds == 60
        assert config.duration_tiers.short.videos_per_keyword == 3
        assert config.duration_tiers.medium.min_seconds == 60
        assert config.duration_tiers.medium.max_seconds == 300
        assert config.duration_tiers.medium.videos_per_keyword == 5

    def test_valid_duration_tiers_preserves_defaults_for_missing_tiers(self):
        """Tiers not specified in data keep their defaults."""
        data = {
            "duration_tiers": {
                "short": {"min": 5, "max": 30, "count": 2},
            }
        }
        config = Config._from_dict(data)
        # short is overridden
        assert config.duration_tiers.short.min_seconds == 5
        # long keeps its default (600, 1500, 5, 0)
        assert config.duration_tiers.long.min_seconds == 600
        assert config.duration_tiers.long.max_seconds == 1500
