"""Tests for pipeline resource monitoring (US-106-011).

Verifies memory/CPU tracking, thresholds, warnings,
resource summary in completion output.
"""

import pytest
from pathlib import Path

from src.config import Config
from src.pipeline import PipelineOrchestrator


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary project directory."""
    return tmp_path / "test_project"


class TestResourceTracking:
    """Test resource tracking implementation."""

    def test_default_memory_threshold(self):
        """DEFAULT_MEMORY_THRESHOLD is 80.0."""
        assert PipelineOrchestrator.DEFAULT_MEMORY_THRESHOLD == 80.0

    def test_resource_history_initialized(self, temp_dir):
        """_resource_history is initialized as empty list."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        assert pipeline._resource_history == []

    def test_get_summary_includes_resource_usage_key(self, temp_dir):
        """get_summary() includes resource_usage key."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        summary = pipeline.get_summary()
        assert 'resource_usage' in summary
        assert summary['resource_usage'] is None  # No history yet

    def test_track_stage_resources_method_exists(self, temp_dir):
        """_track_stage_resources method exists."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        assert hasattr(pipeline, '_track_stage_resources')
        assert callable(pipeline._track_stage_resources)

    def test_track_stage_resources_returns_dict(self, temp_dir):
        """_track_stage_resources returns a dict with expected keys."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        # Track resources (will use real psutil if available)
        result = pipeline._track_stage_resources('TEST', 'before')

        assert isinstance(result, dict)
        assert 'phase' in result
        assert 'stage_name' in result
        assert result['phase'] == 'before'
        assert result['stage_name'] == 'TEST'

    def test_track_stage_resources_accumulates_history(self, temp_dir):
        """_track_stage_resources results can be accumulated in _resource_history."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        # Track resources multiple times and manually add to history
        # (this is how it's used in the pipeline code)
        pipeline._resource_history.append(pipeline._track_stage_resources('STAGE1', 'before'))
        pipeline._resource_history.append(pipeline._track_stage_resources('STAGE1', 'after'))
        pipeline._resource_history.append(pipeline._track_stage_resources('STAGE2', 'before'))

        assert len(pipeline._resource_history) == 3

    def test_get_memory_usage_method_exists(self, temp_dir):
        """_get_memory_usage method exists."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        assert hasattr(pipeline, '_get_memory_usage')
        assert callable(pipeline._get_memory_usage)

    def test_get_cpu_usage_method_exists(self, temp_dir):
        """_get_cpu_usage method exists."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        assert hasattr(pipeline, '_get_cpu_usage')
        assert callable(pipeline._get_cpu_usage)

    def test_emit_resource_warning_method_exists(self, temp_dir):
        """_emit_resource_warning method exists."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        assert hasattr(pipeline, '_emit_resource_warning')
        assert callable(pipeline._emit_resource_warning)


class TestResourceSummary:
    """Test resource summary in get_summary()."""

    def test_summary_calculates_averages(self, temp_dir):
        """Summary calculates avg/max from resource history."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        # Add mock resource data directly to history
        pipeline._resource_history = [
            {'memory_percent': 50.0, 'cpu_percent': 30.0},
            {'memory_percent': 60.0, 'cpu_percent': 40.0},
            {'memory_percent': 70.0, 'cpu_percent': 50.0},
        ]

        summary = pipeline.get_summary()

        assert summary['resource_usage'] is not None
        assert summary['resource_usage']['memory_percent_avg'] == 60.0
        assert summary['resource_usage']['memory_percent_max'] == 70.0
        assert summary['resource_usage']['cpu_percent_avg'] == 40.0
        assert summary['resource_usage']['cpu_percent_max'] == 50.0

    def test_summary_handles_partial_data(self, temp_dir):
        """Summary handles history with some missing fields."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        # Add data with only memory or only cpu
        pipeline._resource_history = [
            {'memory_percent': 50.0},  # No CPU
            {'cpu_percent': 40.0},  # No memory
            {'memory_percent': 60.0, 'cpu_percent': 50.0},
        ]

        summary = pipeline.get_summary()

        assert summary['resource_usage'] is not None
        # Only entries with memory_percent should be averaged
        assert summary['resource_usage']['memory_percent_avg'] == 55.0
        # Only entries with cpu_percent should be averaged
        assert summary['resource_usage']['cpu_percent_avg'] == 45.0


class TestResourceMetricsExport:
    """Test resource metrics export functionality (US-125-003)."""

    def test_export_resource_metrics_method_exists(self, temp_dir):
        """export_resource_metrics method exists on PipelineOrchestrator."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        assert hasattr(pipeline, 'export_resource_metrics')
        assert callable(pipeline.export_resource_metrics)

    def test_export_returns_dict(self, temp_dir):
        """export_resource_metrics returns a dict."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        result = pipeline.export_resource_metrics()
        assert isinstance(result, dict)

    def test_export_contains_required_keys(self, temp_dir):
        """export_resource_metrics returns dict with required keys."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        result = pipeline.export_resource_metrics()
        assert 'format' in result
        assert 'generated_at' in result
        assert 'history' in result
        assert 'summary' in result
        assert 'per_stage' in result

    def test_export_empty_history(self, temp_dir):
        """Export handles empty resource history."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        result = pipeline.export_resource_metrics()
        assert result['history'] == []
        assert result['summary']['stages_tracked'] == 0

    def test_export_with_resource_data(self, temp_dir):
        """Export calculates correct aggregates from resource history."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        # Add mock resource data
        pipeline._resource_history = [
            {
                'phase': 'before',
                'stage_name': 'VIDEO_SEARCH',
                'memory_percent': 50.0,
                'cpu_percent': 30.0,
                'memory_available_gb': 8.0,
            },
            {
                'phase': 'after',
                'stage_name': 'VIDEO_SEARCH',
                'memory_percent': 70.0,
                'cpu_percent': 50.0,
                'memory_available_gb': 6.0,
            },
            {
                'phase': 'before',
                'stage_name': 'MATCH',
                'memory_percent': 65.0,
                'cpu_percent': 45.0,
                'memory_available_gb': 7.0,
            },
            {
                'phase': 'after',
                'stage_name': 'MATCH',
                'memory_percent': 80.0,
                'cpu_percent': 60.0,
                'memory_available_gb': 5.0,
            },
        ]

        result = pipeline.export_resource_metrics()

        # Check summary aggregates
        summary = result['summary']
        assert summary['stages_tracked'] == 2  # VIDEO_SEARCH and MATCH
        assert summary['total_measurements'] == 4

        # Memory: avg=(50+70+65+80)/4=66.25, max=80
        assert summary['memory_percent_avg'] == 66.25
        assert summary['memory_percent_max'] == 80.0

        # CPU: avg=(30+50+45+60)/4=46.25, max=60
        assert summary['cpu_percent_avg'] == 46.25
        assert summary['cpu_percent_max'] == 60.0

        # Memory available min
        assert summary['memory_available_gb_min'] == 5.0

    def test_export_per_stage_aggregates(self, temp_dir):
        """Export includes per-stage aggregates."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        pipeline._resource_history = [
            {'phase': 'before', 'stage_name': 'VIDEO_SEARCH', 'memory_percent': 50.0, 'cpu_percent': 30.0},
            {'phase': 'after', 'stage_name': 'VIDEO_SEARCH', 'memory_percent': 70.0, 'cpu_percent': 50.0},
        ]

        result = pipeline.export_resource_metrics()
        per_stage = result['per_stage']

        assert 'VIDEO_SEARCH' in per_stage
        assert per_stage['VIDEO_SEARCH']['measurements'] == 2
        assert per_stage['VIDEO_SEARCH']['memory_percent_avg'] == 60.0
        assert per_stage['VIDEO_SEARCH']['memory_percent_max'] == 70.0
        assert per_stage['VIDEO_SEARCH']['cpu_percent_avg'] == 40.0

    def test_export_json_format(self, temp_dir):
        """Export respects format parameter."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        result = pipeline.export_resource_metrics(format='json')
        assert result['format'] == 'json'

    def test_export_json_serializable(self, temp_dir):
        """Export result is JSON serializable."""
        import json

        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        pipeline._resource_history = [
            {'phase': 'before', 'stage_name': 'TEST', 'memory_percent': 50.0},
        ]

        result = pipeline.export_resource_metrics()
        # Should not raise
        json_str = json.dumps(result)
        assert json_str is not None
