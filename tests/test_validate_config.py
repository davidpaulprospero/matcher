"""
Tests for scripts/validate_config.py

Tests validation logic including:
- Config loading with mock config
- validate_config_structure with missing sections
- validate_value_ranges with out-of-range values
- --json output produces valid JSON
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is in path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "scripts"))

# Import the validate_config functions
from scripts.validate_config import (
    validate_config_structure,
    validate_config_structure_json,
    validate_value_ranges,
    validate_value_ranges_json,
    run_json_validation,
)


# =============================================================================
# Helper: Create mock config object
# =============================================================================

def create_mock_config(**overrides):
    """Create a mock config object with required attributes."""
    mock = MagicMock()

    # Project config
    mock.project.name = "test-project"
    mock.project.version = "1.0.0"

    # Matching config
    mock.matching.min_confidence = 0.7
    mock.matching.high_confidence_threshold = 0.8
    mock.matching.max_clip_reuse = 3
    mock.matching.primary_provider = "gemini"
    mock.matching.secondary_provider = "anthropic"

    # Transcription config
    mock.transcription.max_workers = 4
    mock.transcription.model = "base"

    # Embedding config
    mock.embedding.provider = "gemini"
    mock.embedding.batch_size = 32

    # Keyword config
    mock.keyword.max_keywords = 50

    # Enhanced config
    mock.enhanced.min_confidence = 0.5

    # Stock footage config
    mock.stock_footage.pexels_enabled = True
    mock.stock_footage.pixabay_enabled = False

    # Paths
    mock.project_dir = "/tmp/test_project"
    mock.downloaded_videos_dir = "/tmp/test_videos"
    mock.otio_output_dir = "/tmp/test_output"
    mock.cache.cache_dir = "/tmp/test_cache"
    mock.logging.log_dir = "/tmp/test_logs"

    # Config metadata
    mock._config_path = "config.yaml"
    mock._config_hash = "abc123"

    # Apply any overrides
    for key, value in overrides.items():
        setattr(mock, key, value)

    return mock


# =============================================================================
# Test: validate_config_structure with present sections
# =============================================================================

class StrictMockConfig:
    """A strict mock that only has explicitly set attributes - no auto-magic."""

    def __init__(self, section_names):
        object.__setattr__(self, '_config_path', "config.yaml")
        object.__setattr__(self, '_config_hash', "abc123")
        # Only add the sections specified - use plain MagicMock for sections
        for section in section_names:
            object.__setattr__(self, section, MagicMock())


def create_strict_mock_config(section_names):
    """Create a mock config with only the specified sections."""
    return StrictMockConfig(section_names)


@pytest.mark.fast
@pytest.mark.script
class TestValidateConfigStructure:
    """Test validate_config_structure function."""

    def test_all_sections_present_returns_true(self):
        """When all required sections exist, validate_config_structure returns True."""
        sections = [
            'project', 'transcription', 'embedding', 'indexing', 'vision',
            'scene_detection', 'matching', 'keyword', 'enhanced',
            'downloading', 'output', 'logging', 'cache', 'pipeline'
        ]
        mock = create_strict_mock_config(sections)

        # Mock print functions to capture output
        with patch('scripts.validate_config.print_header'), \
             patch('scripts.validate_config.print_ok') as mock_ok, \
             patch('scripts.validate_config.print_error'):

            result = validate_config_structure(mock)

        assert result is True

    def test_missing_section_returns_false(self):
        """When a required section is missing, validate_config_structure returns False."""
        # All sections except 'matching'
        sections = [
            'project', 'transcription', 'embedding', 'indexing', 'vision',
            'scene_detection', 'keyword', 'enhanced',
            'downloading', 'output', 'logging', 'cache', 'pipeline'
        ]
        mock = create_strict_mock_config(sections)

        with patch('scripts.validate_config.print_header'), \
             patch('scripts.validate_config.print_ok'), \
             patch('scripts.validate_config.print_error') as mock_error:

            result = validate_config_structure(mock)

        assert result is False
        # Verify error was printed for missing section
        assert mock_error.called

    def test_missing_multiple_sections_returns_false(self):
        """When multiple sections are missing, returns False."""
        # Only a few sections
        sections = ['project', 'transcription', 'matching']
        mock = create_strict_mock_config(sections)

        with patch('scripts.validate_config.print_header'), \
             patch('scripts.validate_config.print_ok'), \
             patch('scripts.validate_config.print_error'):

            result = validate_config_structure(mock)

        assert result is False


# =============================================================================
# Test: validate_config_structure_json
# =============================================================================

@pytest.mark.fast
@pytest.mark.script
class TestValidateConfigStructureJson:
    """Test validate_config_structure_json function."""

    def test_json_output_with_all_sections(self):
        """JSON output contains all sections when present."""
        sections = [
            'project', 'transcription', 'embedding', 'indexing', 'vision',
            'scene_detection', 'matching', 'keyword', 'enhanced',
            'downloading', 'output', 'logging', 'cache', 'pipeline'
        ]
        mock = create_strict_mock_config(sections)

        result = validate_config_structure_json(mock)

        assert result['valid'] is True
        assert result['total'] == 14
        assert result['present'] == 14
        assert len(result['sections']) == 14

    def test_json_output_with_missing_sections(self):
        """JSON output correctly identifies missing sections."""
        # Only some sections
        sections = ['project', 'transcription', 'matching']
        mock = create_strict_mock_config(sections)

        result = validate_config_structure_json(mock)

        assert result['valid'] is False
        assert result['present'] == 3

        # Check that missing sections have valid=False
        sections_by_name = {s['name']: s for s in result['sections']}
        assert sections_by_name['project']['valid'] is True
        assert sections_by_name['embedding']['valid'] is False


# =============================================================================
# Test: validate_value_ranges with valid values
# =============================================================================

@pytest.mark.fast
@pytest.mark.script
class TestValidateValueRanges:
    """Test validate_value_ranges function."""

    def test_valid_ranges_returns_true(self):
        """When all values are in valid ranges, returns True."""
        mock = create_mock_config(
            matching=MagicMock(
                min_confidence=0.7,
                high_confidence_threshold=0.8,
                max_clip_reuse=3
            ),
            transcription=MagicMock(max_workers=4),
            embedding=MagicMock(batch_size=32),
            keyword=MagicMock(max_keywords=50),
            enhanced=MagicMock(min_confidence=0.5)
        )

        with patch('scripts.validate_config.print_header'), \
             patch('scripts.validate_config.print_ok'), \
             patch('scripts.validate_config.print_error'):

            result = validate_value_ranges(mock)

        assert result is True

    def test_invalid_confidence_returns_false(self):
        """When min_confidence is out of range, returns False."""
        mock = create_mock_config(
            matching=MagicMock(
                min_confidence=1.5,  # Invalid: > 1
                high_confidence_threshold=0.8,
                max_clip_reuse=3
            ),
            transcription=MagicMock(max_workers=4),
            embedding=MagicMock(batch_size=32),
            keyword=MagicMock(max_keywords=50),
            enhanced=MagicMock(min_confidence=0.5)
        )

        with patch('scripts.validate_config.print_header'), \
             patch('scripts.validate_config.print_ok'), \
             patch('scripts.validate_config.print_error') as mock_error:

            result = validate_value_ranges(mock)

        assert result is False

    def test_negative_max_clip_reuse_returns_false(self):
        """When max_clip_reuse is negative, returns False."""
        mock = create_mock_config(
            matching=MagicMock(
                min_confidence=0.7,
                high_confidence_threshold=0.8,
                max_clip_reuse=-1  # Invalid: < 0
            ),
            transcription=MagicMock(max_workers=4),
            embedding=MagicMock(batch_size=32),
            keyword=MagicMock(max_keywords=50),
            enhanced=MagicMock(min_confidence=0.5)
        )

        with patch('scripts.validate_config.print_header'), \
             patch('scripts.validate_config.print_ok'), \
             patch('scripts.validate_config.print_error'):

            result = validate_value_ranges(mock)

        assert result is False

    def test_zero_max_workers_returns_false(self):
        """When max_workers is 0, returns False."""
        mock = create_mock_config(
            matching=MagicMock(
                min_confidence=0.7,
                high_confidence_threshold=0.8,
                max_clip_reuse=3
            ),
            transcription=MagicMock(max_workers=0),  # Invalid: < 1
            embedding=MagicMock(batch_size=32),
            keyword=MagicMock(max_keywords=50),
            enhanced=MagicMock(min_confidence=0.5)
        )

        with patch('scripts.validate_config.print_header'), \
             patch('scripts.validate_config.print_ok'), \
             patch('scripts.validate_config.print_error'):

            result = validate_value_ranges(mock)

        assert result is False


# =============================================================================
# Test: validate_value_ranges_json
# =============================================================================

@pytest.mark.fast
@pytest.mark.script
class TestValidateValueRangesJson:
    """Test validate_value_ranges_json function."""

    def test_valid_ranges_json_output(self):
        """JSON output correctly reports valid ranges."""
        mock = create_mock_config(
            matching=MagicMock(
                min_confidence=0.7,
                high_confidence_threshold=0.8,
                max_clip_reuse=3
            ),
            transcription=MagicMock(max_workers=4),
            embedding=MagicMock(batch_size=32),
            keyword=MagicMock(max_keywords=50),
            enhanced=MagicMock(min_confidence=0.5)
        )

        result = validate_value_ranges_json(mock)

        assert result['valid'] is True
        assert result['total'] == 7
        assert result['passed'] == 7

    def test_invalid_confidence_json_output(self):
        """JSON output correctly identifies out-of-range confidence."""
        mock = create_mock_config(
            matching=MagicMock(
                min_confidence=-0.5,  # Invalid: < 0
                high_confidence_threshold=0.8,
                max_clip_reuse=3
            ),
            transcription=MagicMock(max_workers=4),
            embedding=MagicMock(batch_size=32),
            keyword=MagicMock(max_keywords=50),
            enhanced=MagicMock(min_confidence=0.5)
        )

        result = validate_value_ranges_json(mock)

        assert result['valid'] is False
        assert result['passed'] == 6  # One failed

        # Find the failed check
        failed_checks = [c for c in result['checks'] if not c['valid']]
        assert len(failed_checks) == 1
        assert failed_checks[0]['name'] == 'matching.min_confidence'


# =============================================================================
# Test: --json output produces valid JSON
# =============================================================================

@pytest.mark.fast
@pytest.mark.script
class TestJsonOutput:
    """Test --json output functionality."""

    def test_run_json_validation_returns_valid_json_structure(self):
        """run_json_validation returns a structure that can be JSON serialized."""
        # Mock at the src.config module level where it's imported
        mock_config = create_mock_config()
        mock_config.validate = MagicMock(return_value=[])
        mock_config.get_nested = MagicMock(return_value="test_value")

        with patch('src.config.load_config', return_value=mock_config), \
             patch('src.config.get_config_metrics', return_value={'load_count': 1, 'cache_hits': 0}):

            # Need to reimport to pick up the patch
            import importlib
            import scripts.validate_config as vc
            importlib.reload(vc)

            result = vc.run_json_validation()

        # Verify it's valid JSON-serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)

        # Verify expected top-level keys
        assert 'valid' in parsed
        assert 'config_file' in parsed
        assert 'project' in parsed
        assert 'sections' in parsed
        assert 'value_ranges' in parsed

    def test_json_output_contains_all_validation_results(self):
        """JSON output contains results from all validation functions."""
        mock_config = create_mock_config()
        mock_config.validate = MagicMock(return_value=[])
        mock_config.get_nested = MagicMock(return_value="test_value")

        with patch('src.config.load_config', return_value=mock_config), \
             patch('src.config.get_config_metrics', return_value={'load_count': 1, 'cache_hits': 0}):

            import importlib
            import scripts.validate_config as vc
            importlib.reload(vc)

            result = vc.run_json_validation()

        # All validation sections should be present
        assert 'sections' in result
        assert 'api_keys' in result
        assert 'paths' in result
        assert 'value_ranges' in result
        assert 'nested_access' in result
        assert 'validation' in result

    def test_json_output_with_invalid_config_returns_valid_false(self):
        """When config has invalid values, JSON output has valid=False."""
        # Create mock with invalid values
        mock_config = MagicMock()
        mock_config.matching = MagicMock(
            min_confidence=1.5,  # Invalid
            high_confidence_threshold=0.8,
            max_clip_reuse=3
        )
        mock_config.transcription = MagicMock(max_workers=4)
        mock_config.embedding = MagicMock(
            provider="gemini",
            batch_size=32
        )
        mock_config.keyword = MagicMock(max_keywords=50)
        mock_config.enhanced = MagicMock(min_confidence=0.5)
        mock_config.stock_footage = MagicMock(pexels_enabled=False, pixabay_enabled=False)
        mock_config.project = MagicMock(name="test", version="1.0")
        mock_config.project_dir = "/tmp"
        mock_config.downloaded_videos_dir = "/tmp"
        mock_config.otio_output_dir = "/tmp"
        mock_config.cache = MagicMock(cache_dir="/tmp")
        mock_config.logging = MagicMock(log_dir="/tmp")
        mock_config.api_keys = MagicMock(
            gemini_api_key="test",
            anthropic_api_key="test",
            pexels_api_key="test",
            pixabay_api_key="test"
        )
        mock_config._config_path = "config.yaml"
        mock_config._config_hash = "abc123"
        mock_config.validate = MagicMock(return_value=[])
        mock_config.get_nested = MagicMock(return_value="test_value")

        with patch('src.config.load_config', return_value=mock_config), \
             patch('src.config.get_config_metrics', return_value={'load_count': 1, 'cache_hits': 0}):

            import importlib
            import scripts.validate_config as vc
            importlib.reload(vc)

            result = vc.run_json_validation()

        assert result['valid'] is False
        assert result['value_ranges']['valid'] is False


# =============================================================================
# Test: Integration with real config
# =============================================================================

@pytest.mark.fast
@pytest.mark.script
class TestValidateConfigIntegration:
    """Integration tests with real config."""

    def test_validate_config_with_real_config(self):
        """Test that validate_config_structure works with real config."""
        # This test uses the actual config.yaml file
        from src.config import load_config

        try:
            config = load_config("config.yaml")

            # Mock print functions
            with patch('scripts.validate_config.print_header'), \
                 patch('scripts.validate_config.print_ok'), \
                 patch('scripts.validate_config.print_error'):

                result = validate_config_structure(config)

            # Should pass with real config
            assert result is True

        except Exception as e:
            pytest.skip(f"Could not load config.yaml: {e}")

    def test_value_ranges_with_real_config(self):
        """Test that validate_value_ranges works with real config."""
        from src.config import load_config

        try:
            config = load_config("config.yaml")

            with patch('scripts.validate_config.print_header'), \
                 patch('scripts.validate_config.print_ok'), \
                 patch('scripts.validate_config.print_error'):

                result = validate_value_ranges(config)

            # Real config should have valid ranges
            assert result is True

        except Exception as e:
            pytest.skip(f"Could not load config.yaml: {e}")
