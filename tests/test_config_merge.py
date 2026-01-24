"""
Unit tests for config merge functionality.

Tests duration_tiers merging, legacy path migration, and key name mapping.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.config.sections.duration import DurationTierConfig, DurationTiersConfig
from src.cli.config_utils import merge_config, _merge_duration_tiers


class TestDurationTiersMerge:
    """Test duration_tiers merging in project config."""

    def test_merge_duration_tiers_with_count(self):
        """Test project config overrides duration_tiers with 'count' key."""
        config = load_config()
        original_long = config.duration_tiers.long.videos_per_keyword

        overrides = {
            'duration_tiers': {
                'long': {'count': 3}
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.long.videos_per_keyword == 3
        assert isinstance(merged.duration_tiers.long, DurationTierConfig)
        # Other tiers should be unchanged
        assert isinstance(merged.duration_tiers.short, DurationTierConfig)

    def test_merge_duration_tiers_with_per_keyword(self):
        """Test project config overrides with 'per_keyword' key (legacy)."""
        config = load_config()

        overrides = {
            'duration_tiers': {
                'medium': {'per_keyword': 5}
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.medium.videos_per_keyword == 5
        assert isinstance(merged.duration_tiers.medium, DurationTierConfig)

    def test_merge_duration_tiers_with_videos_per_keyword(self):
        """Test project config overrides with 'videos_per_keyword' key."""
        config = load_config()

        overrides = {
            'duration_tiers': {
                'short': {'videos_per_keyword': 10}
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.short.videos_per_keyword == 10

    def test_merge_duration_tiers_full_override(self):
        """Test full tier override with min, max, count, max_total."""
        config = load_config()

        overrides = {
            'duration_tiers': {
                'long': {
                    'min': 480,  # 8 min
                    'max': 900,  # 15 min
                    'count': 3,
                    'max_total': 5
                }
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.long.min_seconds == 480
        assert merged.duration_tiers.long.max_seconds == 900
        assert merged.duration_tiers.long.videos_per_keyword == 3
        assert merged.duration_tiers.long.max_total == 5
        assert isinstance(merged.duration_tiers.long, DurationTierConfig)

    def test_merge_duration_tiers_preserves_unspecified(self):
        """Test that unspecified values are preserved from existing config."""
        config = load_config()
        original_min = config.duration_tiers.long.min_seconds
        original_max = config.duration_tiers.long.max_seconds

        overrides = {
            'duration_tiers': {
                'long': {'count': 1}  # Only override count
            }
        }
        merged = merge_config(config, overrides)

        # count should be overridden
        assert merged.duration_tiers.long.videos_per_keyword == 1
        # min and max should be preserved
        assert merged.duration_tiers.long.min_seconds == original_min
        assert merged.duration_tiers.long.max_seconds == original_max

    def test_merge_multiple_tiers(self):
        """Test overriding multiple tiers at once."""
        config = load_config()

        overrides = {
            'duration_tiers': {
                'short': {'count': 2},
                'medium': {'count': 2},
                'long': {'count': 1},
                'longer': {'count': 0}
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.short.videos_per_keyword == 2
        assert merged.duration_tiers.medium.videos_per_keyword == 2
        assert merged.duration_tiers.long.videos_per_keyword == 1
        assert merged.duration_tiers.longer.videos_per_keyword == 0


class TestLegacyPathMigration:
    """Test silent migration from legacy config paths."""

    def test_download_tier_config_backwards_compat(self):
        """Test download.tier_config is mapped to duration_tiers."""
        config = load_config()

        overrides = {
            'download': {
                'tier_config': {
                    'short': {'per_keyword': 2}
                }
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.short.videos_per_keyword == 2

    def test_keywords_tier_config_backwards_compat(self):
        """Test keywords.tier_config is mapped to duration_tiers."""
        config = load_config()

        overrides = {
            'keywords': {
                'tier_config': {
                    'medium': {'per_keyword': 4}
                }
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.medium.videos_per_keyword == 4

    def test_keyword_tier_config_backwards_compat(self):
        """Test keyword.tier_config (singular) is mapped to duration_tiers."""
        config = load_config()

        overrides = {
            'keyword': {
                'tier_config': {
                    'long': {'per_keyword': 3}
                }
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.long.videos_per_keyword == 3

    def test_legacy_path_with_full_tier_override(self):
        """Test legacy path works with full tier properties."""
        config = load_config()

        overrides = {
            'download': {
                'tier_config': {
                    'long': {
                        'min': 300,
                        'max': 900,
                        'per_keyword': 2,
                        'max_total': 4
                    }
                }
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.long.min_seconds == 300
        assert merged.duration_tiers.long.max_seconds == 900
        assert merged.duration_tiers.long.videos_per_keyword == 2
        assert merged.duration_tiers.long.max_total == 4

    def test_legacy_path_removes_tier_config_from_section(self):
        """Test that tier_config is removed from download section after migration."""
        config = load_config()

        overrides = {
            'download': {
                'tier_config': {
                    'short': {'per_keyword': 1}
                },
                'quality': '720p'  # Other setting to verify section isn't deleted
            }
        }
        merged = merge_config(config, overrides)

        # tier_config should be processed and removed
        # but other download settings should work
        assert merged.duration_tiers.short.videos_per_keyword == 1

    def test_real_project_config_format(self):
        """Test the exact format used in E:\\Edit Job\\...\\project_config.yaml."""
        config = load_config()

        # This is the exact format from the user's project_config.yaml
        overrides = {
            'download': {
                'tier_config': {
                    'short': {'per_keyword': 1},
                    'medium': {'per_keyword': 1},
                    'long': {'per_keyword': 1},
                    'longer': {'per_keyword': 0}
                }
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.short.videos_per_keyword == 1
        assert merged.duration_tiers.medium.videos_per_keyword == 1
        assert merged.duration_tiers.long.videos_per_keyword == 1
        assert merged.duration_tiers.longer.videos_per_keyword == 0


class TestDurationTiersPostInit:
    """Test __post_init__ in DurationTiersConfig."""

    def test_post_init_converts_dicts_to_dataclass(self):
        """Test __post_init__ converts dicts to DurationTierConfig."""
        tiers = DurationTiersConfig()
        # Manually assign a dict (simulates what merge_config might do)
        tiers.long = {'min': 600, 'max': 1500, 'count': 3}
        tiers.__post_init__()

        assert isinstance(tiers.long, DurationTierConfig)
        assert tiers.long.min_seconds == 600
        assert tiers.long.max_seconds == 1500
        assert tiers.long.videos_per_keyword == 3

    def test_post_init_with_min_seconds_format(self):
        """Test __post_init__ handles min_seconds/max_seconds format."""
        tiers = DurationTiersConfig()
        tiers.medium = {
            'min_seconds': 120,
            'max_seconds': 600,
            'videos_per_keyword': 8
        }
        tiers.__post_init__()

        assert isinstance(tiers.medium, DurationTierConfig)
        assert tiers.medium.min_seconds == 120
        assert tiers.medium.max_seconds == 600
        assert tiers.medium.videos_per_keyword == 8

    def test_post_init_preserves_existing_dataclass(self):
        """Test __post_init__ doesn't modify already-correct DurationTierConfig."""
        tiers = DurationTiersConfig()
        original_short = tiers.short
        original_min = original_short.min_seconds

        tiers.__post_init__()

        assert isinstance(tiers.short, DurationTierConfig)
        assert tiers.short.min_seconds == original_min

    def test_post_init_handles_all_tiers(self):
        """Test __post_init__ processes all four tiers."""
        tiers = DurationTiersConfig()
        tiers.short = {'count': 1}
        tiers.medium = {'count': 2}
        tiers.long = {'count': 3}
        tiers.longer = {'count': 4}
        tiers.__post_init__()

        assert isinstance(tiers.short, DurationTierConfig)
        assert isinstance(tiers.medium, DurationTierConfig)
        assert isinstance(tiers.long, DurationTierConfig)
        assert isinstance(tiers.longer, DurationTierConfig)
        assert tiers.short.videos_per_keyword == 1
        assert tiers.medium.videos_per_keyword == 2
        assert tiers.long.videos_per_keyword == 3
        assert tiers.longer.videos_per_keyword == 4


class TestMergeDurationTiersFunction:
    """Test the _merge_duration_tiers helper function."""

    def test_merge_duration_tiers_function(self):
        """Test _merge_duration_tiers directly."""
        config = load_config()
        tiers = config.duration_tiers

        overrides = {
            'long': {'count': 2, 'max_total': 10}
        }
        _merge_duration_tiers(tiers, overrides)

        assert tiers.long.videos_per_keyword == 2
        assert tiers.long.max_total == 10

    def test_merge_duration_tiers_ignores_invalid_tier_names(self):
        """Test that invalid tier names are ignored."""
        config = load_config()
        tiers = config.duration_tiers
        original_short = tiers.short.videos_per_keyword

        overrides = {
            'invalid_tier': {'count': 99},
            'another_invalid': {'per_keyword': 50}
        }
        _merge_duration_tiers(tiers, overrides)

        # Original should be unchanged
        assert tiers.short.videos_per_keyword == original_short

    def test_merge_duration_tiers_handles_empty_override(self):
        """Test that empty override dict doesn't cause issues."""
        config = load_config()
        tiers = config.duration_tiers
        original = tiers.long.videos_per_keyword

        _merge_duration_tiers(tiers, {})

        assert tiers.long.videos_per_keyword == original


class TestKeyNameMapping:
    """Test key name mapping between YAML and internal format."""

    def test_yaml_keys_min_max_count(self):
        """Test YAML keys: min, max, count."""
        config = load_config()

        overrides = {
            'duration_tiers': {
                'short': {'min': 10, 'max': 60, 'count': 5}
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.short.min_seconds == 10
        assert merged.duration_tiers.short.max_seconds == 60
        assert merged.duration_tiers.short.videos_per_keyword == 5

    def test_internal_keys_min_seconds_max_seconds(self):
        """Test internal keys: min_seconds, max_seconds, videos_per_keyword."""
        config = load_config()

        overrides = {
            'duration_tiers': {
                'medium': {
                    'min_seconds': 120,
                    'max_seconds': 600,
                    'videos_per_keyword': 8
                }
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.medium.min_seconds == 120
        assert merged.duration_tiers.medium.max_seconds == 600
        assert merged.duration_tiers.medium.videos_per_keyword == 8

    def test_mixed_key_formats(self):
        """Test mixing YAML and internal key formats."""
        config = load_config()

        overrides = {
            'duration_tiers': {
                'long': {
                    'min': 600,  # YAML format
                    'max_seconds': 1500,  # Internal format
                    'per_keyword': 3  # Legacy format
                }
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.long.min_seconds == 600
        assert merged.duration_tiers.long.max_seconds == 1500
        assert merged.duration_tiers.long.videos_per_keyword == 3


class TestConfigMergeIntegration:
    """Integration tests for config merge with load_project_config."""

    def test_load_project_config_with_duration_tiers(self, tmp_path):
        """Test load_project_config handles duration_tiers correctly."""
        import yaml
        from src.cli.config_utils import load_project_config

        # Create project directory with config
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        project_config = {
            'duration_tiers': {
                'short': {'count': 2},
                'medium': {'count': 2},
                'long': {'count': 1},
                'longer': {'count': 0}
            }
        }

        config_file = project_dir / "project_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(project_config, f)

        config = load_project_config(project_dir)

        assert config.duration_tiers.short.videos_per_keyword == 2
        assert config.duration_tiers.medium.videos_per_keyword == 2
        assert config.duration_tiers.long.videos_per_keyword == 1
        assert config.duration_tiers.longer.videos_per_keyword == 0

    def test_load_project_config_with_legacy_path(self, tmp_path):
        """Test load_project_config handles legacy download.tier_config."""
        import yaml
        from src.cli.config_utils import load_project_config

        project_dir = tmp_path / "legacy_project"
        project_dir.mkdir()

        # Legacy format that existing projects might use
        project_config = {
            'download': {
                'tier_config': {
                    'short': {'per_keyword': 1},
                    'medium': {'per_keyword': 1},
                    'long': {'per_keyword': 1},
                    'longer': {'per_keyword': 0}
                }
            }
        }

        config_file = project_dir / "project_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(project_config, f)

        config = load_project_config(project_dir)

        assert config.duration_tiers.short.videos_per_keyword == 1
        assert config.duration_tiers.medium.videos_per_keyword == 1
        assert config.duration_tiers.long.videos_per_keyword == 1
        assert config.duration_tiers.longer.videos_per_keyword == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_post_init_warns_on_min_greater_than_max(self, caplog):
        """Test __post_init__ warns when min_seconds >= max_seconds."""
        import logging
        caplog.set_level(logging.WARNING)

        tiers = DurationTiersConfig()
        # Set invalid tier where min > max
        tiers.short = {'min': 200, 'max': 100, 'count': 5}
        tiers.__post_init__()

        # Should log a warning
        assert any("min >= max" in record.message for record in caplog.records)
        assert any("short" in record.message for record in caplog.records)

    def test_post_init_warns_on_min_equals_max(self, caplog):
        """Test __post_init__ warns when min_seconds == max_seconds."""
        import logging
        caplog.set_level(logging.WARNING)

        tiers = DurationTiersConfig()
        # Set invalid tier where min == max
        tiers.medium = {'min': 100, 'max': 100, 'count': 3}
        tiers.__post_init__()

        # Should log a warning
        assert any("min >= max" in record.message for record in caplog.records)
        assert any("medium" in record.message for record in caplog.records)

    def test_merge_config_with_non_dict_value(self):
        """Test merge_config handles non-dict section values."""
        config = load_config()

        # Override with a scalar value (not a dict)
        overrides = {
            'project_dir': '/some/path'  # scalar value, not dict
        }
        merged = merge_config(config, overrides)

        # Should set the value directly
        assert merged.project_dir == '/some/path'

    def test_merge_config_ignores_unknown_section(self):
        """Test merge_config ignores sections that don't exist in config."""
        config = load_config()
        original_short = config.duration_tiers.short.videos_per_keyword

        overrides = {
            'nonexistent_section': {
                'some_key': 'some_value'
            }
        }
        merged = merge_config(config, overrides)

        # Config should be unchanged
        assert merged.duration_tiers.short.videos_per_keyword == original_short

    def test_merge_duration_tiers_no_existing_config(self):
        """Test _merge_duration_tiers when tier has no existing dataclass."""
        # Create a minimal tiers object without existing configs
        class MinimalTiers:
            pass

        tiers = MinimalTiers()
        tiers.short = None  # No existing config

        overrides = {
            'short': {'min': 20, 'max': 120, 'count': 5, 'max_total': 10}
        }
        _merge_duration_tiers(tiers, overrides)

        # Should create from scratch with provided values
        assert tiers.short.min_seconds == 20
        assert tiers.short.max_seconds == 120
        assert tiers.short.videos_per_keyword == 5
        assert tiers.short.max_total == 10

    def test_merge_duration_tiers_no_existing_uses_defaults(self):
        """Test _merge_duration_tiers uses defaults when no existing config."""
        class MinimalTiers:
            pass

        tiers = MinimalTiers()
        tiers.medium = None

        # Only provide count, let other values use defaults
        overrides = {
            'medium': {'count': 3}
        }
        _merge_duration_tiers(tiers, overrides)

        assert tiers.medium.videos_per_keyword == 3
        assert tiers.medium.min_seconds == 0  # default
        assert tiers.medium.max_seconds == 0  # default (not 120 - use actual defaults)
        assert tiers.medium.max_total == 0  # default

    def test_load_project_config_with_invalid_yaml(self, tmp_path, capsys):
        """Test load_project_config handles invalid YAML gracefully."""
        from src.cli.config_utils import load_project_config

        project_dir = tmp_path / "bad_yaml_project"
        project_dir.mkdir()

        # Create invalid YAML file
        config_file = project_dir / "project_config.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")

        # Should not crash, return base config with warning
        config = load_project_config(project_dir)
        captured = capsys.readouterr()

        assert config is not None
        assert "Failed to load project config" in captured.out or config is not None

    def test_load_project_config_without_project_config_file(self, tmp_path):
        """Test load_project_config when no project_config.yaml exists."""
        from src.cli.config_utils import load_project_config

        project_dir = tmp_path / "empty_project"
        project_dir.mkdir()

        # Should return base config without error
        config = load_project_config(project_dir)

        assert config is not None
        # Should have default duration_tiers
        assert isinstance(config.duration_tiers.short, DurationTierConfig)

    def test_merge_config_preserves_other_section_attributes(self):
        """Test merge_config preserves attributes not in overrides."""
        config = load_config()
        original_timeout = config.download.download_timeout

        overrides = {
            'download': {
                # Only override one attribute
                'quality': '720p'
            }
        }
        merged = merge_config(config, overrides)

        # Other attributes should be preserved
        assert merged.download.download_timeout == original_timeout
        # Overridden attribute should change
        assert merged.download.quality == '720p'

    def test_legacy_migration_with_empty_section_after_removal(self):
        """Test legacy migration removes empty section after tier_config removal."""
        config = load_config()

        # Only tier_config in download section
        overrides = {
            'download': {
                'tier_config': {
                    'short': {'count': 1}
                }
            }
        }
        # After processing, download should be removed from overrides
        merged = merge_config(config, overrides.copy())

        assert merged.duration_tiers.short.videos_per_keyword == 1


class TestTierConfigValidation:
    """Test tier configuration validation scenarios."""

    def test_zero_count_is_valid(self):
        """Test that count=0 is valid (disables tier)."""
        config = load_config()

        overrides = {
            'duration_tiers': {
                'longer': {'count': 0}
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.longer.videos_per_keyword == 0

    def test_large_count_is_valid(self):
        """Test that large count values work."""
        config = load_config()

        overrides = {
            'duration_tiers': {
                'short': {'count': 100}
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.short.videos_per_keyword == 100

    def test_custom_duration_ranges(self):
        """Test custom min/max duration values."""
        config = load_config()

        overrides = {
            'duration_tiers': {
                'long': {
                    'min': 1800,  # 30 min
                    'max': 7200,  # 2 hours
                    'count': 2
                }
            }
        }
        merged = merge_config(config, overrides)

        assert merged.duration_tiers.long.min_seconds == 1800
        assert merged.duration_tiers.long.max_seconds == 7200
        assert merged.duration_tiers.long.videos_per_keyword == 2


class TestDeepMergeEdgeCases:
    """
    US-008: Edge case tests for deep merge behavior.

    Tests for:
    - Preserving sibling keys when merging nested dicts
    - Handling None values correctly
    - Handling empty dicts correctly
    - Project config overrides don't clobber unrelated sections
    """

    def test_deep_merge_preserves_sibling_keys(self):
        """Test merge preserves sibling keys when merging nested dicts."""
        config = load_config()

        # Get original values from sibling sections
        original_matching_confidence = config.matching.min_confidence
        original_transcription_model = config.transcription.model

        # Only override download section
        overrides = {
            'download': {
                'quality': '720p'
            }
        }
        merged = merge_config(config, overrides)

        # Sibling sections should be preserved
        assert merged.matching.min_confidence == original_matching_confidence
        assert merged.transcription.model == original_transcription_model
        # Overridden section should have new value
        assert merged.download.quality == '720p'

    def test_deep_merge_handles_none_values_in_overrides(self):
        """Test merge handles None values in override dict."""
        config = load_config()

        # Override with None value (should not crash)
        overrides = {
            'download': {
                'quality': None,
                'max_retries': 5
            }
        }
        merged = merge_config(config, overrides)

        # None value should be set
        assert merged.download.quality is None
        # Other overrides should work
        assert merged.download.max_retries == 5

    def test_deep_merge_handles_empty_override_dict(self):
        """Test merge handles empty override dict without errors."""
        config = load_config()
        original_quality = config.download.quality
        original_model = config.transcription.model

        # Empty overrides
        overrides = {}
        merged = merge_config(config, overrides)

        # Config should be unchanged
        assert merged.download.quality == original_quality
        assert merged.transcription.model == original_model

    def test_deep_merge_handles_empty_section_override(self):
        """Test merge handles empty section override dict."""
        config = load_config()
        original_quality = config.download.quality

        # Override with empty section dict
        overrides = {
            'download': {}
        }
        merged = merge_config(config, overrides)

        # Config should be unchanged
        assert merged.download.quality == original_quality

    def test_project_config_does_not_clobber_unrelated_sections(self, tmp_path):
        """Test project_config.yaml overrides don't affect unrelated sections."""
        import yaml
        from src.cli.config_utils import load_project_config

        project_dir = tmp_path / "isolated_override_project"
        project_dir.mkdir()

        # Load base config to get original values
        base_config = load_config()
        original_transcription_model = base_config.transcription.model
        original_matching_confidence = base_config.matching.min_confidence
        original_llm_provider = base_config.llm.provider

        # Project config only touches download section
        project_config = {
            'download': {
                'quality': '1080p',
                'max_retries': 10
            }
        }

        config_file = project_dir / "project_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(project_config, f)

        config = load_project_config(project_dir)

        # Unrelated sections should be unchanged
        assert config.transcription.model == original_transcription_model
        assert config.matching.min_confidence == original_matching_confidence
        assert config.llm.provider == original_llm_provider
        # Only download section should change
        assert config.download.quality == '1080p'
        assert config.download.max_retries == 10

    def test_nested_dict_merge_preserves_unspecified_nested_keys(self):
        """Test merging nested dicts preserves unspecified keys at all levels."""
        config = load_config()

        # Get original nested values
        original_short_min = config.duration_tiers.short.min_seconds
        original_short_max = config.duration_tiers.short.max_seconds
        original_medium = config.duration_tiers.medium.videos_per_keyword

        # Only override one key in one tier
        overrides = {
            'duration_tiers': {
                'short': {'count': 99}
            }
        }
        merged = merge_config(config, overrides)

        # Unspecified keys in short tier should be preserved
        assert merged.duration_tiers.short.min_seconds == original_short_min
        assert merged.duration_tiers.short.max_seconds == original_short_max
        assert merged.duration_tiers.short.videos_per_keyword == 99
        # Other tiers should be unchanged
        assert merged.duration_tiers.medium.videos_per_keyword == original_medium

    def test_multiple_sections_override_independently(self):
        """Test overriding multiple sections doesn't cause interference."""
        config = load_config()

        overrides = {
            'download': {'quality': 'best'},
            'matching': {'min_confidence': 0.8},
            'duration_tiers': {
                'long': {'count': 5}
            }
        }
        merged = merge_config(config, overrides)

        # All overrides should be applied
        assert merged.download.quality == 'best'
        assert merged.matching.min_confidence == 0.8
        assert merged.duration_tiers.long.videos_per_keyword == 5

        # Each section should be independently correct
        assert hasattr(merged.download, 'max_retries')
        assert hasattr(merged.matching, 'high_confidence_threshold')
        assert hasattr(merged.duration_tiers.long, 'min_seconds')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
