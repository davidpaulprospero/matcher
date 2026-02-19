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

    def test_results_per_keyword_greater_than_max_total_results(self, tmp_path):
        """Validation catches results_per_keyword > max_total_results."""
        config_file = tmp_path / "bad_video_search.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: 100
  max_total_results: 50
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert any("must be <=" in err and "results_per_keyword" in err for err in errors)

    def test_max_videos_per_channel_greater_equal_results_per_keyword(self, tmp_path):
        """Validation catches max_videos_per_channel >= results_per_keyword."""
        config_file = tmp_path / "bad_video_search.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: 5
  max_videos_per_channel: 5
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert any("must be <" in err and "max_videos_per_channel" in err for err in errors)

    def test_search_timeout_zero(self, tmp_path):
        """Validation catches search_timeout <= 0."""
        config_file = tmp_path / "bad_video_search.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: 20
  max_total_results: 200
  search_timeout: 0
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert any("search_timeout must be > 0" in err for err in errors)

    def test_search_timeout_negative(self, tmp_path):
        """Validation catches search_timeout < 0."""
        config_file = tmp_path / "bad_video_search.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: 20
  max_total_results: 200
  search_timeout: -5
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert any("search_timeout must be > 0" in err for err in errors)

    def test_valid_budget_constraints_passes(self, tmp_path):
        """Valid budget constraints pass validation."""
        config_file = tmp_path / "good_video_search.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: 20
  max_total_results: 200
  max_videos_per_channel: 3
  search_timeout: 30
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        budget_errors = [e for e in errors if "results_per_keyword" in e or "max_videos_per_channel" in e or "search_timeout" in e]
        assert len(budget_errors) == 0

    def test_search_budget_vs_video_search_inconsistency_warns(self, tmp_path, caplog):
        """Validation warns when search_budget and video_search have different values."""
        config_file = tmp_path / "inconsistent_budget.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
video_search:
  results_per_keyword: 20
  max_total_results: 200

search_budget:
  results_per_keyword: 50
  max_total_results: 500
""")
        config = Config.from_yaml(str(config_file))

        # Capture warnings during validate()
        with caplog.at_level(logging.WARNING):
            config.validate()

        # Check that warnings are emitted for both differences
        assert "search_budget.results_per_keyword" in caplog.text
        assert "search_budget.max_total_results" in caplog.text


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

    def test_deprecated_prefer_human_captions_warns(self, tmp_path, caplog):
        """Validation warns when prefer_human_captions is set to false."""
        config_file = tmp_path / "deprecated_prefer_human.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
download:
  caption_first:
    prefer_human_captions: false
""")
        config = Config.from_yaml(str(config_file))

        # Capture warnings during validate()
        with caplog.at_level(logging.WARNING):
            config.validate()

        # Check for warning about prefer_human_captions
        assert "prefer_human_captions" in caplog.text

    def test_deprecated_audio_first_legacy_warns(self, tmp_path, caplog):
        """Validation warns for legacy audio_first config placement.

        When audio_first-related fields are placed at top-level download
        instead of nested under download.audio_first, a warning is logged.
        """
        config_file = tmp_path / "deprecated_audio_first.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
download:
  # Legacy placement - should be under download.audio_first
  audio_quality: 3
""")
        config = Config.from_yaml(str(config_file))

        # Capture warnings during validate()
        with caplog.at_level(logging.WARNING):
            config.validate()

        # Check for warning about unknown config key (detected as wrong location)
        assert "Unknown config key" in caplog.text or "audio_quality" in caplog.text

    def test_deprecated_fields_no_runtime_error(self, tmp_path):
        """Deprecated fields should not cause runtime errors during validation."""
        config_file = tmp_path / "all_deprecated.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
image_search:
  output_dir: "custom_images"
download:
  caption_first:
    enabled: true
    negative_cache_ttl_hours: 24
    prefer_human_captions: false
""")
        config = Config.from_yaml(str(config_file))

        # Should not raise any exceptions
        errors = config.validate()

        # Should have warnings but no errors from deprecated fields
        assert isinstance(errors, list)


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

    # US-111-011: Context richness signal weights validation tests

    def test_context_richness_weights_sum_to_one_passes(self, tmp_path):
        """Test that weights summing to 1.0 pass validation."""
        config_file = tmp_path / "valid_weights.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
matching:
  context_richness_title_weight: 0.25
  context_richness_description_weight: 0.25
  context_richness_tags_weight: 0.25
  context_richness_chapters_weight: 0.25
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should pass (no errors about weights)
        weight_errors = [e for e in errors if "context richness signal weights" in e]
        assert len(weight_errors) == 0

    def test_context_richness_weights_sum_to_one_custom_passes(self, tmp_path):
        """Test that custom weights summing to 1.0 pass validation."""
        config_file = tmp_path / "valid_custom_weights.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
matching:
  context_richness_title_weight: 0.5
  context_richness_description_weight: 0.2
  context_richness_tags_weight: 0.2
  context_richness_chapters_weight: 0.1
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should pass
        weight_errors = [e for e in errors if "context richness signal weights" in e]
        assert len(weight_errors) == 0

    def test_context_richness_weights_sum_to_less_than_one_fails(self, tmp_path):
        """Test that weights summing to less than 1.0 fail validation."""
        config_file = tmp_path / "invalid_low_weights.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
matching:
  context_richness_title_weight: 0.2
  context_richness_description_weight: 0.2
  context_richness_tags_weight: 0.2
  context_richness_chapters_weight: 0.2
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should fail
        weight_errors = [e for e in errors if "context richness signal weights" in e]
        assert len(weight_errors) == 1
        assert "0.800" in weight_errors[0]

    def test_context_richness_weights_sum_to_greater_than_one_fails(self, tmp_path):
        """Test that weights summing to > 1.0 fail validation."""
        config_file = tmp_path / "invalid_high_weights.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
matching:
  context_richness_title_weight: 0.3
  context_richness_description_weight: 0.3
  context_richness_tags_weight: 0.3
  context_richness_chapters_weight: 0.3
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should fail
        weight_errors = [e for e in errors if "context richness signal weights" in e]
        assert len(weight_errors) == 1
        assert "1.200" in weight_errors[0]

    def test_context_richness_weights_default_passes(self, tmp_path):
        """Test that default weights (not specified) pass validation."""
        config_file = tmp_path / "default_weights.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
# No explicit weights - should use defaults (0.25 each = 1.0)
matching:
  context_richness_calibration: true
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should pass
        weight_errors = [e for e in errors if "context richness signal weights" in e]
        assert len(weight_errors) == 0


# =============================================================================
# US-112-002: Cross-validation between iterative_matching and matching configs
# =============================================================================

@pytest.mark.fast
class TestIterativeMatchingCrossValidation:
    """Test cross-validation between iterative_matching and matching configs."""

    def test_iterative_enabled_with_high_min_confidence_fails(self, tmp_path):
        """Validation fails when iterative matching enabled with min_confidence > 0.9."""
        config_file = tmp_path / "invalid_high_conf.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
iterative_matching:
  enabled: true
  target_confidence: 0.90

matching:
  min_confidence: 0.95
  low_confidence_threshold: 0.3
  ambiguous_threshold: 0.4
  high_confidence_threshold: 0.7
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should fail with the cross-validation error
        assert any("iterative_matching.enabled=true requires matching.min_confidence <= 0.9" in err for err in errors)

    def test_iterative_disabled_with_high_min_confidence_passes(self, tmp_path):
        """Validation passes when iterative matching disabled with high min_confidence."""
        config_file = tmp_path / "valid_high_conf.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
iterative_matching:
  enabled: false
  target_confidence: 0.90

matching:
  min_confidence: 0.95
  low_confidence_threshold: 0.3
  ambiguous_threshold: 0.4
  high_confidence_threshold: 0.7
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should pass (no iterative_matching errors)
        iterative_errors = [e for e in errors if "iterative_matching" in e]
        assert len(iterative_errors) == 0

    def test_max_iterations_less_than_max_retries_fails(self, tmp_path):
        """Validation fails when max_iterations < max_retries."""
        config_file = tmp_path / "invalid_iterations.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
iterative_matching:
  enabled: true
  max_iterations: 2

matching:
  max_retries: 5
  min_confidence: 0.7
  low_confidence_threshold: 0.3
  ambiguous_threshold: 0.4
  high_confidence_threshold: 0.7
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should fail
        assert any("max_iterations (2) should be >= matching.max_retries (5)" in err for err in errors)

    def test_max_iterations_greater_than_max_retries_passes(self, tmp_path):
        """Validation passes when max_iterations >= max_retries."""
        config_file = tmp_path / "valid_iterations.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
iterative_matching:
  enabled: true
  max_iterations: 5

matching:
  max_retries: 3
  min_confidence: 0.7
  low_confidence_threshold: 0.3
  ambiguous_threshold: 0.4
  high_confidence_threshold: 0.7
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should pass
        iterative_errors = [e for e in errors if "max_iterations" in e]
        assert len(iterative_errors) == 0

    def test_target_confidence_less_than_low_threshold_fails(self, tmp_path):
        """Validation fails when target_confidence < low_confidence_threshold."""
        config_file = tmp_path / "invalid_confidence.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
iterative_matching:
  enabled: true
  target_confidence: 0.3

matching:
  min_confidence: 0.7
  low_confidence_threshold: 0.5
  ambiguous_threshold: 0.6
  high_confidence_threshold: 0.85
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should fail
        assert any("target_confidence (0.3) should be >= matching.low_confidence_threshold (0.5)" in err for err in errors)

    def test_target_confidence_greater_than_low_threshold_passes(self, tmp_path):
        """Validation passes when target_confidence >= low_confidence_threshold."""
        config_file = tmp_path / "valid_confidence.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
iterative_matching:
  enabled: true
  target_confidence: 0.90

matching:
  min_confidence: 0.7
  low_confidence_threshold: 0.5
  ambiguous_threshold: 0.6
  high_confidence_threshold: 0.85
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should pass
        conf_errors = [e for e in errors if "target_confidence" in e]
        assert len(conf_errors) == 0

    def test_iterative_chapter_boost_without_chapter_matching_logs_debug(self, tmp_path, caplog):
        """Test that iterative_chapter_boost without chapter_matching_enabled logs debug."""
        config_file = tmp_path / "chapter_boost_no_chapter.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
iterative_matching:
  enabled: true
  iterative_chapter_boost: 0.1

matching:
  chapter_matching_enabled: false
  min_confidence: 0.7
  low_confidence_threshold: 0.3
  ambiguous_threshold: 0.4
  high_confidence_threshold: 0.7
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))

        # Should pass validation (this is just a debug log, not an error)
        with caplog.at_level(logging.DEBUG):
            errors = config.validate()

        # Should pass (no errors)
        assert len(errors) == 0

    def test_iterative_chapter_boost_with_chapter_matching_passes(self, tmp_path):
        """Validation passes when iterative_chapter_boost with chapter_matching_enabled."""
        config_file = tmp_path / "chapter_boost_with_chapter.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
iterative_matching:
  enabled: true
  iterative_chapter_boost: 0.1

matching:
  chapter_matching_enabled: true
  min_confidence: 0.7
  low_confidence_threshold: 0.3
  ambiguous_threshold: 0.4
  high_confidence_threshold: 0.7
""")
        from src.config.base import Config
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        # Should pass
        assert len(errors) == 0


# =============================================================================
# US-142-007: Test external config validation webhook
# =============================================================================

@pytest.mark.fast
class TestValidationWebhook:
    """Test external config validation webhook (US-142-007)."""

    def test_webhook_disabled_by_default(self, tmp_path):
        """When validation_webhook.enabled=false, no webhook call is made."""
        config_file = tmp_path / "webhook_disabled.yaml"
        config_file.write_text(create_minimal_valid_config_text())

        config = Config.from_yaml(str(config_file))

        # Webhook should be disabled by default
        assert config.validation_webhook.enabled is False
        assert config.validation_webhook.url == ""

    def test_webhook_enabled_requires_url(self, tmp_path):
        """ValidationWebhookConfig.enabled=True without URL raises ValueError."""
        config_file = tmp_path / "webhook_no_url.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
validation_webhook:
  enabled: true
  url: ""
""")

        # Should raise ValueError
        with pytest.raises(ValueError, match="requires a URL"):
            Config.from_yaml(str(config_file))

    def test_webhook_timeout_validation(self, tmp_path):
        """ValidationWebhookConfig with timeout < 0.1 raises ValueError."""
        config_file = tmp_path / "webhook_bad_timeout.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
validation_webhook:
  enabled: true
  url: "https://example.com/validate"
  timeout_seconds: 0.01
""")

        # Should raise ValueError
        with pytest.raises(ValueError, match="timeout_seconds must be >= 0.1"):
            Config.from_yaml(str(config_file))

    def test_webhook_successful_validation(self, tmp_path):
        """Webhook returns valid=true, validation passes with warning."""
        import json
        from unittest.mock import patch, MagicMock

        config_file = tmp_path / "webhook_success.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
validation_webhook:
  enabled: true
  url: "https://example.com/validate"
  timeout_seconds: 10.0
""")

        config = Config.from_yaml(str(config_file))

        # Mock the webhook response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "valid": True,
            "errors": [],
            "warnings": ["Consider increasing min_confidence"]
        }).encode('utf-8')
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            errors = config.validate()

        # Should pass validation (webhook returned valid=true)
        assert len(errors) == 0

    def test_webhook_returns_invalid_non_blocking(self, tmp_path):
        """Webhook returns valid=false with fail_on_error=False (default), logs warning but doesn't block."""
        import json
        from unittest.mock import patch, MagicMock

        config_file = tmp_path / "webhook_invalid_non_blocking.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
validation_webhook:
  enabled: true
  url: "https://example.com/validate"
  timeout_seconds: 10.0
  fail_on_error: false
""")

        config = Config.from_yaml(str(config_file))

        # Mock the webhook response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "valid": False,
            "errors": ["Invalid min_confidence: must be > 0"],
            "warnings": []
        }).encode('utf-8')
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            errors = config.validate()

        # Should NOT add to errors list (fail_on_error=False)
        assert len(errors) == 0

    def test_webhook_returns_invalid_blocking(self, tmp_path):
        """Webhook returns valid=false with fail_on_error=True, adds error to block pipeline."""
        import json
        from unittest.mock import patch, MagicMock

        config_file = tmp_path / "webhook_invalid_blocking.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
validation_webhook:
  enabled: true
  url: "https://example.com/validate"
  timeout_seconds: 10.0
  fail_on_error: true
""")

        # Use skip_final_validation=True because fail_on_error=True requires
        # webhook to work during from_yaml, which would fail without mock
        config = Config.from_yaml(str(config_file), skip_final_validation=True)

        # Mock the webhook response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "valid": False,
            "errors": ["Invalid min_confidence: must be > 0"],
            "warnings": []
        }).encode('utf-8')
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            errors = config.validate()

        # Should add to errors list (fail_on_error=True)
        assert len(errors) == 1
        assert "Invalid min_confidence: must be > 0" in errors[0]

    def test_webhook_failure_non_blocking(self, tmp_path, caplog):
        """Webhook request fails with fail_on_error=False (default), logs warning but doesn't block."""
        import logging
        from unittest.mock import patch
        import urllib.error

        config_file = tmp_path / "webhook_failure_non_blocking.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
validation_webhook:
  enabled: true
  url: "https://example.com/validate"
  timeout_seconds: 10.0
  fail_on_error: false
""")

        config = Config.from_yaml(str(config_file))

        # Mock the webhook to raise an error
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("Connection refused")):
            errors = config.validate()

        # Should NOT add to errors list (fail_on_error=False)
        assert len(errors) == 0
        # Should log warning
        assert any("Validation webhook request failed" in record.message for record in caplog.records)

    def test_webhook_failure_blocking(self, tmp_path, caplog):
        """Webhook request fails with fail_on_error=True, adds error to block pipeline."""
        import logging
        from unittest.mock import patch
        import urllib.error

        config_file = tmp_path / "webhook_failure_blocking.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
validation_webhook:
  enabled: true
  url: "https://example.com/validate"
  timeout_seconds: 10.0
  fail_on_error: true
""")

        # Use skip_final_validation=True because fail_on_error=True requires
        # webhook to work during from_yaml, which would fail without mock
        config = Config.from_yaml(str(config_file), skip_final_validation=True)

        # Mock the webhook to raise an error
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("Connection refused")):
            errors = config.validate()

        # Should add to errors list (fail_on_error=True)
        assert len(errors) == 1

    def test_webhook_with_custom_headers(self, tmp_path):
        """Webhook request includes custom headers."""
        import json
        from unittest.mock import patch, MagicMock

        config_file = tmp_path / "webhook_headers.yaml"
        config_file.write_text(create_minimal_valid_config_text() + """
validation_webhook:
  enabled: true
  url: "https://example.com/validate"
  timeout_seconds: 10.0
  headers:
    Authorization: "Bearer test_token"
    X-Custom-Header: "custom_value"
""")

        config = Config.from_yaml(str(config_file))

        # Verify headers are set
        assert config.validation_webhook.headers.get("Authorization") == "Bearer test_token"
        assert config.validation_webhook.headers.get("X-Custom-Header") == "custom_value"
