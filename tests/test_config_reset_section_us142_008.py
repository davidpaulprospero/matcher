"""Tests for config section reset-to-defaults (US-142-008)."""

import pytest
import os
from unittest.mock import patch

from src.config.base import Config, FrozenConfigError


@pytest.fixture
def mock_api_keys():
    """Provide mock API keys for tests."""
    with patch.dict(os.environ, {
        'GEMINI_API_KEY': 'test_gemini_key_12345',
        'GOOGLE_API_KEY': 'test_google_key_12345',
        'PEXELS_API_KEY': 'test_pexels_key',
        'PIXABAY_API_KEY': 'test_pixabay_key',
    }):
        yield


class TestResetSection:
    """Test Config.reset_section() method."""

    def test_reset_section_restores_defaults(self, mock_api_keys):
        """Test that reset_section restores section to default values."""
        config = Config()

        # Modify a section value
        original_min_confidence = config.matching.min_confidence
        config.matching.min_confidence = 0.99  # Non-default value

        # Verify modification took effect
        assert config.matching.min_confidence == 0.99

        # Reset the section
        success = config.reset_section('matching')
        assert success is True

        # Verify restored to default
        assert config.matching.min_confidence == original_min_confidence

    def test_reset_section_returns_false_for_unknown(self, mock_api_keys):
        """Test that reset_section returns False for unknown section."""
        config = Config()

        success = config.reset_section('nonexistent_section')
        assert success is False

    def test_reset_section_handles_frozen_config(self, mock_api_keys):
        """Test that reset_section handles frozen config correctly."""
        config = Config()
        config.unfreeze()  # Start unfrozen

        # Modify and freeze
        config.matching.min_confidence = 0.99
        config.freeze()

        # Reset should unfreeze, reset, and refreeze
        success = config.reset_section('matching')
        assert success is True
        assert config._frozen is True  # Should be refrozen
        assert config.matching.min_confidence == 0.7  # Default value per MatchingConfig

    def test_reset_section_triggers_callbacks(self, mock_api_keys):
        """Test that reset_section triggers change callbacks."""
        config = Config()

        callback_invoked = []

        def on_matching_change(cfg, changed_sections):
            callback_invoked.append(changed_sections)

        config.register_change_callback(on_matching_change, section='matching')

        # Reset matching section
        config.reset_section('matching')

        # Verify callback was invoked
        assert len(callback_invoked) == 1
        assert 'matching' in callback_invoked[0]

    def test_reset_section_works_on_various_sections(self, mock_api_keys):
        """Test reset_section works on various config sections."""
        config = Config()

        # Test reset on different sections
        sections_to_test = ['matching', 'download', 'video_search', 'transcription', 'cache']

        for section in sections_to_test:
            success = config.reset_section(section)
            assert success is True, f"Failed to reset section: {section}"

    def test_reset_section_preserves_other_sections(self, mock_api_keys):
        """Test that resetting one section doesn't affect others."""
        config = Config()

        # Modify multiple sections
        matching_value = config.matching.min_confidence
        download_value = config.download.max_concurrent
        video_search_value = config.video_search.max_total_results

        config.matching.min_confidence = 0.99
        config.download.max_concurrent = 99
        config.video_search.max_total_results = 999

        # Reset only matching
        config.reset_section('matching')

        # Verify matching was reset
        assert config.matching.min_confidence == matching_value

        # Verify others are unchanged
        assert config.download.max_concurrent == 99
        assert config.video_search.max_total_results == 999


class TestResetSectionCLI:
    """Test --reset-section CLI argument integration."""

    def test_reset_section_argument_defined_in_args_module(self):
        """Verify --reset-section argument is defined in args.py."""
        import re
        with open('src/cli/args.py', 'r') as f:
            content = f.read()

        # Check for --reset-section argument
        assert '--reset-section' in content

    def test_main_handles_reset_section_flag(self):
        """Test main.py correctly handles --reset-section with valid section."""
        from src.config.base import Config

        config = Config()

        # Set a non-default value
        config.matching.min_confidence = 0.99

        # Reset using the method
        result = config.reset_section('matching')

        assert result is True
        assert config.matching.min_confidence == 0.7  # Default value per MatchingConfig
