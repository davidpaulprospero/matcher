"""
Tests for environment variable overrides (US-128-002).

Verifies that MATCHER_* environment variables can override config.yaml values.
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


class TestEnvVarOverrides:
    """Test environment variable override functionality."""

    @pytest.fixture(autouse=True)
    def setup_env_vars(self, monkeypatch):
        """Clear MATCHER_* env vars before each test."""
        # Clear any existing MATCHER_ env vars
        for key in list(os.environ.keys()):
            if key.startswith('MATCHER_'):
                monkeypatch.delenv(key, raising=False)

    def test_env_override_simple_field(self, monkeypatch):
        """Test simple field override (MATCHER_SECTION_FIELD)."""
        monkeypatch.setenv('MATCHER_MATCHING_MIN_CONFIDENCE', '0.85')

        config = Config()
        config._apply_env_overrides()

        assert config.matching.min_confidence == 0.85
        assert isinstance(config.matching.min_confidence, float)

    def test_env_override_nested_field(self, monkeypatch):
        """Test nested field override (MATCHER_SECTION_NESTED_FIELD)."""
        monkeypatch.setenv('MATCHER_DOWNLOAD_MULLVAD_ENABLED', 'true')

        config = Config()
        config._apply_env_overrides()

        assert config.download.mullvad.enabled is True
        assert isinstance(config.download.mullvad.enabled, bool)

    def test_env_override_underscore_section(self, monkeypatch):
        """Test section with underscore (e.g., VIDEO_SEARCH)."""
        monkeypatch.setenv('MATCHER_VIDEO_SEARCH_MAX_TOTAL_RESULTS', '150')

        config = Config()
        config._apply_env_overrides()

        assert config.video_search.max_total_results == 150
        assert isinstance(config.video_search.max_total_results, int)

    def test_env_override_takes_precedence(self, monkeypatch):
        """Test that env var takes precedence over config.yaml."""
        # Set required API keys for validation
        monkeypatch.setenv('GEMINI_API_KEY', 'test_key')
        monkeypatch.setenv('PEXELS_API_KEY', 'test_key')
        monkeypatch.setenv('PIXABAY_API_KEY', 'test_key')

        # Create temp config file with minimal valid config
        # Also set high_confidence_threshold to avoid validation error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'matching': {'min_confidence': 0.5, 'high_confidence_threshold': 0.95},
                'project': {'version': '4.0.0'},
            }, f)
            config_path = f.name

        try:
            monkeypatch.setenv('MATCHER_MATCHING_MIN_CONFIDENCE', '0.9')

            config = Config.from_yaml(config_path)

            # Env var should override config file
            assert config.matching.min_confidence == 0.9
        finally:
            os.unlink(config_path)

    def test_env_override_int_type(self, monkeypatch):
        """Test integer type coercion."""
        monkeypatch.setenv('MATCHER_MATCHING_MAX_RETRIES', '10')

        config = Config()
        config._apply_env_overrides()

        assert config.matching.max_retries == 10
        assert isinstance(config.matching.max_retries, int)

    def test_env_override_bool_true(self, monkeypatch):
        """Test boolean true value coercion."""
        monkeypatch.setenv('MATCHER_MATCHING_ADAPTIVE_THRESHOLD_ENABLED', 'false')

        config = Config()
        config._apply_env_overrides()

        assert config.matching.adaptive_threshold_enabled is False

    def test_multiple_env_overrides(self, monkeypatch):
        """Test multiple env var overrides at once."""
        monkeypatch.setenv('MATCHER_MATCHING_MIN_CONFIDENCE', '0.8')
        monkeypatch.setenv('MATCHER_DOWNLOAD_MULLVAD_ENABLED', 'true')
        monkeypatch.setenv('MATCHER_VIDEO_SEARCH_ENABLE_CHANNEL_DIVERSITY', 'false')

        config = Config()
        config._apply_env_overrides()

        assert config.matching.min_confidence == 0.8
        assert config.download.mullvad.enabled is True
        assert config.video_search.enable_channel_diversity is False

    def test_env_override_no_matcher_prefix(self, monkeypatch):
        """Test that non-MATCHER env vars are ignored."""
        monkeypatch.setenv('MIN_CONFIDENCE', '0.99')

        config = Config()
        original_value = config.matching.min_confidence
        config._apply_env_overrides()

        # Should not be changed
        assert config.matching.min_confidence == original_value

    def test_env_override_invalid_section(self, monkeypatch, caplog):
        """Test invalid section name is handled gracefully."""
        monkeypatch.setenv('MATCHER_UNKNOWN_MIN_CONFIDENCE', '0.5')

        config = Config()
        config._apply_env_overrides()

        # Should not raise, just log warning
        assert config.matching.min_confidence != 0.5  # Should remain default


class TestValueSourceTracking:
    """Test config value source tracking (US-142-010)."""

    @pytest.fixture(autouse=True)
    def setup_env_vars(self, monkeypatch):
        """Clear MATCHER_* env vars before each test."""
        # Clear any existing MATCHER_ env vars
        for key in list(os.environ.keys()):
            if key.startswith('MATCHER_'):
                monkeypatch.delenv(key, raising=False)

    def test_yaml_source_tracking(self, tmp_path, monkeypatch):
        """Test that values from YAML are tracked as 'yaml' source."""
        # Create minimal config YAML
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
matching:
  min_confidence: 0.75
video_search:
  max_results: 30
""")

        config = Config.from_yaml(str(config_yaml))

        # Check sources are tracked
        source = config.get_value_source('matching.min_confidence')
        assert source is not None
        assert source['source'] == 'yaml'
        assert source['value'] == 0.75

        source2 = config.get_value_source('video_search.max_results')
        assert source2 is not None
        assert source2['source'] == 'yaml'
        assert source2['value'] == 30

    def test_env_source_tracking(self, monkeypatch):
        """Test that env overrides are tracked as 'env' source."""
        monkeypatch.setenv('MATCHER_MATCHING_MIN_CONFIDENCE', '0.9')

        config = Config()
        config._apply_env_overrides()

        # Check source is tracked as env
        source = config.get_value_source('matching.min_confidence')
        assert source is not None
        assert source['source'] == 'env'
        assert source['value'] == 0.9

    def test_default_source_tracking(self):
        """Test that default values are tracked correctly."""
        config = Config()

        # Fields not in YAML should show default
        # Check if there's any field that's only default
        all_sources = config.get_all_sources()

        # At least some sources should be tracked
        assert len(all_sources) > 0

    def test_get_all_sources(self, tmp_path, monkeypatch):
        """Test get_all_sources returns all tracked fields."""
        monkeypatch.setenv('MATCHER_MATCHING_MIN_CONFIDENCE', '0.85')

        config = Config()
        config._apply_env_overrides()

        all_sources = config.get_all_sources()

        # Should have at least the env override tracked
        assert 'matching.min_confidence' in all_sources
        assert all_sources['matching.min_confidence']['source'] == 'env'

    def test_unknown_field_returns_none(self):
        """Test that unknown field returns None."""
        config = Config()

        source = config.get_value_source('nonexistent.field')
        assert source is None

    def test_dot_notation_support(self, tmp_path):
        """Test that dot notation works for nested fields."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
download:
  mullvad:
    enabled: true
""")

        config = Config.from_yaml(str(config_yaml))

        # Check nested field source
        source = config.get_value_source('download.mullvad.enabled')
        assert source is not None
        assert source['source'] == 'yaml'
        assert source['value'] is True
