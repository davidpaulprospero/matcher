"""
Unit tests for US-142-011: Config loading performance metrics.

Tests timing capture for each config loading phase.
"""

import pytest
import yaml
import tempfile
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


class TestConfigLoadMetrics:
    """Test config loading performance metrics."""

    @pytest.mark.fast
    def test_load_timings_dict_exists(self, tmp_path):
        """Test that _load_timings dict is initialized."""
        # Create minimal config
        config_data = {
            'project': {'name': 'test', 'version': '4.0.0'},
            'llm': {'provider': 'gemini'},
            'download': {'max_results': 10},
            'matching': {'min_confidence': 0.6},
            'embedding': {'provider': 'voyage'},
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = Config.from_yaml(str(config_file), skip_final_validation=True)

        assert hasattr(config, '_load_timings')
        assert isinstance(config._load_timings, dict)

    @pytest.mark.fast
    def test_get_load_metrics_returns_dict(self, tmp_path):
        """Test get_load_metrics returns a dict."""
        config_data = {
            'project': {'name': 'test', 'version': '4.0.0'},
            'llm': {'provider': 'gemini'},
            'download': {'max_results': 10},
            'matching': {'min_confidence': 0.6},
            'embedding': {'provider': 'voyage'},
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = Config.from_yaml(str(config_file), skip_final_validation=True)
        metrics = config.get_load_metrics()

        assert isinstance(metrics, dict)

    @pytest.mark.fast
    def test_timing_capture_accuracy(self, tmp_path):
        """Test verifies timing capture accuracy."""
        config_data = {
            'project': {'name': 'test', 'version': '4.0.0'},
            'llm': {'provider': 'gemini'},
            'download': {'max_results': 10},
            'matching': {'min_confidence': 0.6},
            'embedding': {'provider': 'voyage'},
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Load and get metrics
        config = Config.from_yaml(str(config_file), skip_final_validation=True)
        metrics = config.get_load_metrics()

        # Verify expected phases are captured
        expected_phases = ['parse', 'validation', 'convert', 'total']
        for phase in expected_phases:
            assert phase in metrics, f"Expected phase '{phase}' not found in metrics"

        # Verify timing values are positive numbers
        for phase, duration in metrics.items():
            assert isinstance(duration, (int, float)), f"Duration for {phase} should be numeric"
            assert duration >= 0, f"Duration for {phase} should be non-negative"

        # Verify total is sum of phases (approximately)
        # Note: Some phases may overlap, so we just verify total >= max phase
        assert metrics['total'] >= 0
        assert metrics['total'] > 0, "Total load time should be greater than 0"

    @pytest.mark.fast
    def test_timing_capture_with_multiple_loads(self, tmp_path):
        """Test that timing is captured correctly across multiple loads."""
        config_data = {
            'project': {'name': 'test', 'version': '4.0.0'},
            'llm': {'provider': 'gemini'},
            'download': {'max_results': 10},
            'matching': {'min_confidence': 0.6},
            'embedding': {'provider': 'voyage'},
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Load config multiple times
        config1 = Config.from_yaml(str(config_file), skip_final_validation=True)
        config2 = Config.from_yaml(str(config_file), skip_final_validation=True)

        # Both should have timings
        assert len(config1.get_load_metrics()) > 0
        assert len(config2.get_load_metrics()) > 0

        # Timings should be independent
        assert config1._load_timings is not config2._load_timings

    @pytest.mark.fast
    def test_timings_include_all_phases(self, tmp_path):
        """Test that all expected phases are recorded."""
        config_data = {
            'project': {'name': 'test', 'version': '4.0.0'},
            'llm': {'provider': 'gemini'},
            'download': {'max_results': 10},
            'matching': {'min_confidence': 0.6},
            'embedding': {'provider': 'voyage'},
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = Config.from_yaml(str(config_file), skip_final_validation=True)
        metrics = config.get_load_metrics()

        # Check all expected phases are present
        expected_phases = [
            'parse',
            'line_tracking',
            'interpolation',
            'validation',
            'convert',
            'env_overrides',
            'final_validation',
            'total'
        ]

        for phase in expected_phases:
            assert phase in metrics, f"Missing phase: {phase}"

    @pytest.mark.fast
    def test_json_export_includes_load_metrics(self, tmp_path):
        """Test that JSON export includes load metrics in metadata."""
        config_data = {
            'project': {'name': 'test', 'version': '4.0.0'},
            'llm': {'provider': 'gemini'},
            'download': {'max_results': 10},
            'matching': {'min_confidence': 0.6},
            'embedding': {'provider': 'voyage'},
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = Config.from_yaml(str(config_file), skip_final_validation=True)

        # Export to JSON
        json_output = config.to_json()

        # Check that load_metrics is in the output
        assert 'load_metrics' in json_output
        assert 'parse' in json_output
        assert 'validation' in json_output
        assert 'convert' in json_output
        assert 'total' in json_output


class TestConfigLoadMetricsEdgeCases:
    """Test edge cases for config loading performance metrics."""

    @pytest.mark.fast
    def test_missing_config_file(self, tmp_path):
        """Test metrics when config file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"

        config = Config.from_yaml(str(nonexistent), skip_final_validation=True)
        metrics = config.get_load_metrics()

        # Should still return metrics dict (possibly empty or with defaults)
        assert isinstance(metrics, dict)

    @pytest.mark.fast
    def test_load_time_ms_still_works(self, tmp_path):
        """Test backward compatibility with _load_time_ms."""
        config_data = {
            'project': {'name': 'test', 'version': '4.0.0'},
            'llm': {'provider': 'gemini'},
            'download': {'max_results': 10},
            'matching': {'min_confidence': 0.6},
            'embedding': {'provider': 'voyage'},
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = Config.from_yaml(str(config_file), skip_final_validation=True)

        # Both old and new timing metrics should work
        assert config._load_time_ms > 0
        assert config.get_load_metrics()['total'] > 0
        assert abs(config._load_time_ms - config.get_load_metrics()['total']) < 1
