"""
Tests for config value interpolation (US-142-005).

Verifies that ${section.field} and ${ENV_VAR} interpolations work correctly.
"""

import os
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.base import Config


class TestValueInterpolations:
    """Test config value interpolation functionality."""

    def test_resolve_section_field_interpolation(self):
        """Test ${section.field} syntax resolves correctly."""
        data = {
            'project': {'version': '4.0.0'},
            'video_search': {'results_per_keyword': 50},
            'matching': {'max_results': '${video_search.results_per_keyword}'},
        }

        Config._resolve_value_interpolations(data)

        # Type coercion should convert to int
        assert data['matching']['max_results'] == 50
        assert isinstance(data['matching']['max_results'], int)

    def test_resolve_env_var_interpolation(self, monkeypatch):
        """Test ${ENV_VAR} syntax resolves correctly."""
        monkeypatch.setenv('TEST_INTERP_VAR', 'test_value')

        data = {
            'project': {'version': '4.0.0'},
            'matching': {'description': 'Using env: ${TEST_INTERP_VAR}'},
        }

        Config._resolve_value_interpolations(data)

        assert data['matching']['description'] == 'Using env: test_value'

    def test_resolve_multiple_interpolations(self):
        """Test multiple interpolations in single value."""
        data = {
            'project': {'version': '4.0.0'},
            'video_search': {'results_per_keyword': 50},
            'video_search_2': {'fallback': 25},
            'matching': {'max_results': '${video_search.results_per_keyword}', 'min_confidence': 0.5},
            'iterative': {
                'max_gap_results': '${video_search.results_per_keyword}',
                'fallback_results': '${video_search_2.fallback}',
            },
        }

        Config._resolve_value_interpolations(data)

        # Type coercion should convert to int
        assert data['matching']['max_results'] == 50
        assert data['iterative']['max_gap_results'] == 50
        assert data['iterative']['fallback_results'] == 25

    def test_circular_reference_detection(self):
        """Test circular references are detected and raise error."""
        data = {
            'project': {'version': '4.0.0'},
            'matching': {'a': '${matching.b}', 'b': '${matching.a}'},
        }

        with pytest.raises(Exception) as exc_info:
            Config._resolve_value_interpolations(data)

        assert 'Circular reference' in str(exc_info.value) or 'cycle' in str(exc_info.value)

    def test_interpolation_in_nested_dict(self):
        """Test interpolations work in deeply nested dicts."""
        data = {
            'project': {'version': '4.0.0'},
            'video_search': {'results_per_keyword': 50},
            'matching': {
                'scoring': {
                    'threshold': '${video_search.results_per_keyword}'
                }
            },
        }

        Config._resolve_value_interpolations(data)

        # Type coercion should convert to int
        assert data['matching']['scoring']['threshold'] == 50

    def test_interpolation_with_list_values(self):
        """Test interpolations work in list values."""
        data = {
            'project': {'version': '4.0.0'},
            'video_search': {'results_per_keyword': 50},
            'matching': {
                'tiers': [1, 2, '${video_search.results_per_keyword}']
            },
        }

        Config._resolve_value_interpolations(data)

        assert data['matching']['tiers'] == [1, 2, '50']

    def test_interpolation_with_non_string_no_change(self):
        """Test non-string values are not modified."""
        data = {
            'project': {'version': '4.0.0'},
            'matching': {
                'min_confidence': 0.5,
                'enabled': True,
                'count': 100,
            },
        }

        original = data['matching'].copy()
        Config._resolve_value_interpolations(data)

        assert data['matching']['min_confidence'] == 0.5
        assert data['matching']['enabled'] is True
        assert data['matching']['count'] == 100

    def test_interpolation_in_from_yaml(self, monkeypatch, tmp_path):
        """Test interpolations are resolved during from_yaml()."""
        # Create a test config file with interpolations (using valid fields)
        config_content = """
project:
  version: "4.0.0"
  name: test_project

video_search:
  results_per_keyword: 50

matching:
  min_confidence: 0.5
  max_retries: "${video_search.results_per_keyword}"

pipeline:
  enabled: true
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        # Load config with from_yaml (skip validation to avoid API key requirements)
        config = Config.from_yaml(str(config_file), skip_final_validation=True)

        # Verify interpolation resolved with type coercion
        assert config.matching.max_retries == 50
        assert isinstance(config.matching.max_retries, int)
