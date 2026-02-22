"""
Tests for config profile system (US-142-004).

Verifies that named profiles can inherit from config.yaml and override specific values.
"""

import os
import pytest
from pathlib import Path
import sys
import tempfile
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.base import Config


class TestConfigProfiles:
    """Test config profile functionality."""

    @pytest.fixture
    def temp_profile_dir(self, tmp_path):
        """Create a temporary profile directory with test profiles."""
        profile_dir = tmp_path / "profiles"
        profile_dir.mkdir()

        # Create a test profile
        profile_data = {
            'name': 'test',
            'description': 'Test profile',
            'inherits_from': 'config.yaml',
            'logging': {
                'log_level': 'DEBUG'
            },
            'matching': {
                'min_confidence': 0.25,
                'high_confidence_threshold': 0.7,
                'ambiguous_threshold': 0.2
            },
            'rate_limit': {
                'slots_per_second': 10.0
            }
        }

        with open(profile_dir / "test.yaml", 'w') as f:
            yaml.dump(profile_data, f)

        # Create another profile that inherits from test
        profile_data2 = {
            'name': 'test2',
            'description': 'Test profile 2',
            'inherits_from': 'config.yaml',
            'matching': {
                'min_confidence': 0.15
            }
        }

        with open(profile_dir / "test2.yaml", 'w') as f:
            yaml.dump(profile_data2, f)

        return profile_dir


    def test_profile_merge_logic(self):
        """Test that profile values override base config."""
        # Test with actual dev profile in the repo
        config = Config.load_with_profile('config.yaml', 'dev', skip_final_validation=True)

        # Dev profile should override these values
        assert config._profile_applied == 'dev'
        assert config.logging.log_level == 'DEBUG'
        assert config.rate_limit.slots_per_second == 5.0
        assert config.healing.strategy == 'aggressive'
        assert config.video_search.max_total_results == 50
        assert config.matching.min_confidence == 0.3

    def test_prod_profile_override(self):
        """Test that prod profile values are applied correctly."""
        config = Config.load_with_profile('config.yaml', 'prod', skip_final_validation=True)

        assert config._profile_applied == 'prod'
        assert config.logging.log_level == 'WARNING'
        assert config.rate_limit.slots_per_second == 0.5
        assert config.healing.strategy == 'conservative'
        assert config.video_search.max_total_results == 300
        assert config.matching.min_confidence == 0.6

    def test_profile_nonexistent(self):
        """Test that loading nonexistent profile raises error."""
        with pytest.raises(FileNotFoundError) as exc_info:
            Config.load_with_profile('config.yaml', 'nonexistent')

        assert "not found" in str(exc_info.value).lower()
        assert "nonexistent" in str(exc_info.value)

    def test_profile_deep_merge(self):
        """Test that nested values are deeply merged."""
        # Load dev profile which should deeply merge
        config = Config.load_with_profile('config.yaml', 'dev', skip_final_validation=True)

        # Profile should override specific nested values while keeping others from base
        # Rate limit has both slots_per_second and jitter_factor
        # Dev profile only specifies slots_per_second, so jitter_factor should come from base
        # But in this case, let's check other nested values

        # Video search has many fields - profile only overrides some
        assert config.video_search.max_total_results == 50  # from profile

        # Check that other video_search fields still exist from base
        assert hasattr(config.video_search, 'results_per_keyword')

    def test_profile_staging(self):
        """Test staging profile values."""
        config = Config.load_with_profile('config.yaml', 'staging', skip_final_validation=True)

        assert config._profile_applied == 'staging'
        assert config.logging.log_level == 'INFO'
        assert config.rate_limit.slots_per_second == 2.0
        assert config.healing.strategy == 'conservative'
        assert config.video_search.max_total_results == 150
        assert config.matching.min_confidence == 0.5


class TestProfileMerge:
    """Test the profile merge logic directly."""

    def test_deep_merge(self):
        """Test _deep_merge function."""
        base = {
            'a': 1,
            'b': {
                'c': 2,
                'd': 3
            }
        }

        override = {
            'b': {
                'c': 20,
                'e': 4
            },
            'f': 5
        }

        result = Config._deep_merge(base, override)

        assert result['a'] == 1
        assert result['b']['c'] == 20  # overridden
        assert result['b']['d'] == 3  # kept from base
        assert result['b']['e'] == 4  # added from override
        assert result['f'] == 5

    def test_merge_profile_with_config(self):
        """Test _merge_profile_with_config function."""
        base_data = {
            'logging': {
                'log_level': 'INFO',
                'log_api_calls': False
            },
            'matching': {
                'min_confidence': 0.5,
                'high_confidence_threshold': 0.8
            }
        }

        profile_data = {
            'name': 'test',
            'description': 'Test profile',
            'logging': {
                'log_level': 'DEBUG'
            },
            'matching': {
                'min_confidence': 0.3
            }
        }

        result = Config._merge_profile_with_config(base_data, profile_data)

        # Profile values should override base
        assert result['logging']['log_level'] == 'DEBUG'
        # Base values should be kept when not overridden
        assert result['logging']['log_api_calls'] == False
        # Matching should be merged
        assert result['matching']['min_confidence'] == 0.3
        assert result['matching']['high_confidence_threshold'] == 0.8
        # Profile metadata should be added
        assert result['_profile_applied'] == 'test'
