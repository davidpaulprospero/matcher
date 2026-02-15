"""
Tests for US-99-011: Config validation tests.

Tests that verify Config.validate() catches:
- Negative results_per_keyword
- max_total_results=0
- Invalid confidence (at validate() time)
- Deprecated options warnings
- Valid config passes validation
"""

import pytest
import logging
import os
import sys
from unittest.mock import patch

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.base import Config


# Helper function to create a minimal valid config with API keys
def create_minimal_valid_config_text() -> str:
    """Create a minimal valid config that passes API key and constraint validation."""
    return """
# Minimal config with valid API keys (test values)
api_keys:
  gemini_api_key: "test_gemini_key_12345"
  anthropic_api_key: "test_anthropic_key_12345"
  pexels_api_key: "test_pexels_key_12345"
  pixabay_api_key: "test_pixabay_key_12345"

# Required sections with valid values
matching:
  min_confidence: 0.5
  max_clip_reuse: 3
  primary_provider: "gemini"
  secondary_provider: "gemini"
  # Threshold values must satisfy:
  # low_confidence_threshold <= ambiguous_threshold <= min_confidence <= high_confidence_threshold
  low_confidence_threshold: 0.3
  ambiguous_threshold: 0.4
  high_confidence_threshold: 0.7

transcription:
  max_workers: 4

embedding:
  provider: "gemini"
  batch_size: 32

keyword:
  max_keywords: 50

video_search:
  results_per_keyword: 20
  max_total_results: 200
"""


# =============================================================================
# Test: Validation catches invalid confidence (at validate() stage)
# =============================================================================

@pytest.mark.fast
class TestConfidenceValidation:
    """Test that min_confidence validation catches invalid values at validate() stage."""

    def test_confidence_warning_for_high_value(self, tmp_path, caplog):
        """High min_confidence (>1) gets clamped with warning at config load time."""
        config_file = tmp_path / "high_confidence.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
matching:
  min_confidence: 1.5
  low_confidence_threshold: 0.3
  ambiguous_threshold: 0.4
  high_confidence_threshold: 0.7
""")
        # Config loading produces a warning about clamping
        with caplog.at_level(logging.WARNING):
            config = Config.from_yaml(str(config_file))

        # The config should have been clamped to 1.0
        assert config.matching.min_confidence == 1.0
        # And there should be a warning about clamping
        assert any("clamped to 1.0" in msg for msg in caplog.messages)

    def test_valid_confidence_passes(self, tmp_path):
        """Valid min_confidence (0-1) passes validation."""
        config_file = tmp_path / "good_confidence.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
matching:
  min_confidence: 0.7
  low_confidence_threshold: 0.3
  ambiguous_threshold: 0.4
  high_confidence_threshold: 0.85
video_search:
  results_per_keyword: 20
  max_total_results: 200
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should have no validation errors for confidence
        confidence_errors = [e for e in errors if "min_confidence" in e]
        assert len(confidence_errors) == 0


# =============================================================================
# Test: Validation catches negative results_per_keyword
# =============================================================================

@pytest.mark.fast
class TestVideoSearchValidation:
    """Test that video_search validation catches invalid values."""

    def test_negative_results_per_keyword(self, tmp_path):
        """Validation catches negative results_per_keyword."""
        config_file = tmp_path / "bad_video_search.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: -5
  max_total_results: 200
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert any("results_per_keyword must be > 0" in err for err in errors)

    def test_zero_results_per_keyword(self, tmp_path):
        """Validation catches zero results_per_keyword."""
        config_file = tmp_path / "bad_video_search.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: 0
  max_total_results: 200
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert any("results_per_keyword must be > 0" in err for err in errors)

    def test_negative_max_total_results(self, tmp_path):
        """Validation catches negative max_total_results."""
        config_file = tmp_path / "bad_video_search.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: 20
  max_total_results: -10
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert any("max_total_results must be > 0" in err for err in errors)

    def test_zero_max_total_results(self, tmp_path):
        """Validation catches max_total_results=0."""
        config_file = tmp_path / "bad_video_search.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: 20
  max_total_results: 0
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert any("max_total_results must be > 0" in err for err in errors)

    def test_valid_video_search_passes(self, tmp_path):
        """Valid video_search values pass validation."""
        config_file = tmp_path / "good_video_search.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: 20
  max_total_results: 200
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        video_search_errors = [e for e in errors if "video_search" in e]
        assert len(video_search_errors) == 0


# =============================================================================
# Test: Validation warns on deprecated options
# =============================================================================

@pytest.mark.fast
class TestDeprecatedOptions:
    """Test that validation emits warnings for deprecated options.

    Deprecation warnings are emitted during validate().
    """

    def test_deprecated_image_search_output_dir_warns(self, tmp_path, caplog):
        """Validation warns on deprecated image_search.output_dir."""
        config_file = tmp_path / "deprecated_output_dir.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
image_search:
  output_dir: "custom_images"
""")
        config = Config.from_yaml(str(config_file))

        # Capture warnings during validate()
        with caplog.at_level(logging.WARNING):
            config.validate()

        # Check using text property which captures all log output
        assert "output_dir' is deprecated" in caplog.text

    def test_deprecated_caption_first_warns(self, tmp_path, caplog):
        """Validation warns on deprecated download.caption_first.enabled."""
        config_file = tmp_path / "deprecated_caption_first.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
download:
  caption_first:
    enabled: true
""")
        config = Config.from_yaml(str(config_file))

        # Capture warnings during validate()
        with caplog.at_level(logging.WARNING):
            config.validate()

        # Check using text property which captures all log output
        assert "caption_first.enabled' is deprecated" in caplog.text

    def test_deprecated_negative_cache_ttl_warns(self, tmp_path, caplog):
        """Validation warns on deprecated download.caption_first.negative_cache_ttl_hours."""
        config_file = tmp_path / "deprecated_negative_cache.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
download:
  caption_first:
    negative_cache_ttl_hours: 24
""")
        config = Config.from_yaml(str(config_file))

        # Capture warnings during validate()
        with caplog.at_level(logging.WARNING):
            config.validate()

        # Check using text property which captures all log output
        assert "negative_cache_ttl_hours' is deprecated" in caplog.text


# =============================================================================
# Test: Validation passes for valid config.yaml
# =============================================================================

@pytest.mark.fast
class TestValidConfigPasses:
    """Test that validation passes for valid config."""

    def test_valid_config_yaml_passes(self, tmp_path):
        """Valid config.yaml passes validation without errors."""
        # Create a minimal valid config
        config_file = tmp_path / "valid_config.yaml"
        config_file.write_text(create_minimal_valid_config_text())
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should have no errors
        assert len(errors) == 0

    def test_default_config_passes(self):
        """Default Config() passes validation when API keys are provided."""
        # Create config with default values but with valid API keys
        config = Config()
        # Set required API keys to pass validation
        config.api_keys.gemini_api_key = "test_key"
        config.api_keys.anthropic_api_key = "test_key"
        config.api_keys.pexels_api_key = "test_key"
        config.api_keys.pixabay_api_key = "test_key"
        errors = config.validate()

        # Default values should be valid
        assert len(errors) == 0


# =============================================================================
# Integration test: CLI --validate-config flag
# =============================================================================

@pytest.mark.fast
class TestValidateConfigCli:
    """Test --validate-config CLI flag integration."""

    def test_validate_config_flag_exists(self):
        """Verify --validate-config flag is available."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--validate-config']):
            args = parse_arguments()
            assert hasattr(args, 'validate_config')
            assert args.validate_config is True

    def test_validate_config_detects_errors(self, tmp_path):
        """--validate-config catches validation errors."""
        from src.cli.args import parse_arguments

        # Create invalid config
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: -5
""")

        with patch.object(sys, 'argv', ['main.py', '--config', str(config_file), '--validate-config']):
            args = parse_arguments()

        # This should return errors
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert len(errors) > 0
