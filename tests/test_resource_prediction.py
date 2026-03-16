"""Tests for resource prediction functionality (US-138-009)."""

import json
import tempfile
import pytest
from pathlib import Path

from src.pipeline_history import (
    append_resource_usage,
    get_resource_history,
    predict_memory_usage,
    get_stage_resource_stats,
    save_history,
    load_history,
)


class TestResourcePrediction:
    """Tests for resource usage prediction functions."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_append_resource_usage(self, temp_project_dir):
        """Test appending resource usage records."""
        # Append a resource usage record
        append_resource_usage(
            temp_project_dir,
            "MATCH",
            memory_mb=512.5,
            cpu_percent=45.2,
            voiceover_duration_seconds=300.0,
            video_count=50,
            segment_count=20,
        )

        # Verify it was saved
        history = load_history(temp_project_dir)
        assert "resource_usage" in history
        assert "MATCH" in history["resource_usage"]
        assert len(history["resource_usage"]["MATCH"]) == 1

        record = history["resource_usage"]["MATCH"][0]
        assert record["memory_mb"] == 512.5
        assert record["cpu_percent"] == 45.2
        assert record["voiceover_duration_seconds"] == 300.0
        assert record["video_count"] == 50
        assert record["segment_count"] == 20

    def test_append_resource_usage_multiple_stages(self, temp_project_dir):
        """Test appending resource usage for multiple stages."""
        # Append for different stages
        append_resource_usage(temp_project_dir, "CAPTION", memory_mb=256.0, cpu_percent=30.0)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=512.0, cpu_percent=45.0)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=550.0, cpu_percent=48.0)

        history = load_history(temp_project_dir)
        assert len(history["resource_usage"]["CAPTION"]) == 1
        assert len(history["resource_usage"]["MATCH"]) == 2

    def test_get_resource_history(self, temp_project_dir):
        """Test retrieving resource usage history."""
        # Add some records
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=512.0, cpu_percent=45.0)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=550.0, cpu_percent=48.0)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=530.0, cpu_percent=46.0)

        # Get all history
        history = get_resource_history(temp_project_dir)
        assert "MATCH" in history
        assert len(history["MATCH"]) == 3

        # Get with window
        history = get_resource_history(temp_project_dir, window=2)
        assert len(history["MATCH"]) == 2

        # Get specific stage
        history = get_resource_history(temp_project_dir, stage_name="CAPTION")
        assert "CAPTION" in history

    def test_predict_memory_usage_formula_only(self, temp_project_dir):
        """Test prediction with no historical data (formula only)."""
        # No historical data - should use formula only
        result = predict_memory_usage(
            temp_project_dir,
            voiceover_duration_seconds=600.0,  # 10 minutes
            video_count=100,
            segment_count=50,
        )

        assert result is not None
        assert result["predicted_memory_mb"] > 0
        assert result["sources"] == ["formula"]
        assert result["confidence"] == "low"

        # Check formula: 10min * 50 + 100*5 + 50*2 = 500 + 500 + 100 = 1100
        # With no history, predicted = formula_based_mb
        assert result["formula_based_mb"] == 1100.0

    def test_predict_memory_usage_with_history(self, temp_project_dir):
        """Test prediction with historical data."""
        # Add historical data
        for i in range(5):
            append_resource_usage(
                temp_project_dir,
                "MATCH",
                memory_mb=400.0 + i * 20,  # 400, 420, 440, 460, 480
                cpu_percent=40.0,
                voiceover_duration_seconds=300.0,
                video_count=50,
                segment_count=25,
            )

        # Now predict - should use weighted average
        result = predict_memory_usage(
            temp_project_dir,
            voiceover_duration_seconds=600.0,  # 10 minutes
            video_count=100,
            segment_count=50,
        )

        assert result is not None
        assert "historical_avg_mb" in result
        assert result["historical_avg_mb"] == 440.0  # Average of 400-480
        assert "formula_based_mb" in result
        assert "formula" in result["sources"]
        assert "historical" in result["sources"]

    def test_predict_memory_usage_confidence_levels(self, temp_project_dir):
        """Test prediction confidence levels."""
        # Low confidence: < 3 samples
        result = predict_memory_usage(temp_project_dir, 300, 50, 20)
        assert result["confidence"] == "low"

        # Add more samples for low confidence (with 2 samples it uses the second branch)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=400.0, cpu_percent=40.0)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=420.0, cpu_percent=42.0)

        result = predict_memory_usage(temp_project_dir, 300, 50, 20)
        # With 2 samples, confidence is "low"
        assert result["confidence"] == "low"

        # Add one more for >= 3 samples (now goes to first branch - medium)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=440.0, cpu_percent=44.0)

        result = predict_memory_usage(temp_project_dir, 300, 50, 20)
        # With 3 samples, confidence is "medium" (3-4 samples)
        assert result["confidence"] == "medium"

        # Add 2 more for >= 5 samples (high confidence)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=460.0, cpu_percent=46.0)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=480.0, cpu_percent=48.0)

        result = predict_memory_usage(temp_project_dir, 300, 50, 20)
        assert result["confidence"] == "high"

    def test_get_stage_resource_stats(self, temp_project_dir):
        """Test getting resource statistics for a stage."""
        # Add resource data for MATCH stage
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=400.0, cpu_percent=40.0)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=450.0, cpu_percent=45.0)
        append_resource_usage(temp_project_dir, "MATCH", memory_mb=500.0, cpu_percent=50.0)

        stats = get_stage_resource_stats(temp_project_dir, "MATCH")

        assert stats is not None
        assert stats["avg_memory_mb"] == 450.0
        assert stats["max_memory_mb"] == 500.0
        assert stats["min_memory_mb"] == 400.0
        assert stats["avg_cpu_percent"] == 45.0
        assert stats["max_cpu_percent"] == 50.0
        assert stats["sample_count"] == 3

    def test_get_stage_resource_stats_no_data(self, temp_project_dir):
        """Test getting stats for stage with no data."""
        stats = get_stage_resource_stats(temp_project_dir, "NONEXISTENT")
        assert stats is None

    def test_resource_prediction_with_custom_params(self, temp_project_dir):
        """Test prediction with custom memory parameters."""
        result = predict_memory_usage(
            temp_project_dir,
            voiceover_duration_seconds=300.0,  # 5 minutes
            video_count=100,
            segment_count=50,
            base_memory_per_vo_minute=100.0,  # Custom: 100MB per minute
            memory_per_video=10.0,  # Custom: 10MB per video
            memory_per_segment=5.0,  # Custom: 5MB per segment
        )

        # Formula: 5 * 100 + 100 * 10 + 50 * 5 = 500 + 1000 + 250 = 1750
        assert result["formula_based_mb"] == 1750.0

    def test_history_trimming(self, temp_project_dir):
        """Test that resource history is trimmed to max size."""
        # Add many records (more than MAX_RESOURCE_HISTORY = 20)
        for i in range(25):
            append_resource_usage(
                temp_project_dir,
                "MATCH",
                memory_mb=400.0 + i,
                cpu_percent=40.0,
            )

        history = get_resource_history(temp_project_dir, stage_name="MATCH")
        # Should be trimmed to 20
        assert len(history["MATCH"]) <= 20


class TestResourcePredictionConfig:
    """Tests for ResourcePredictionConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        from src.config.sections.infrastructure import ResourcePredictionConfig

        config = ResourcePredictionConfig(
            enabled=True,
            warning_threshold_percent=75.0,
            critical_threshold_percent=90.0,
            history_window=5,
            base_memory_per_vo_minute=50.0,
            memory_per_video=5.0,
            memory_per_segment=2.0,
        )

        assert config.enabled is True
        assert config.warning_threshold_percent == 75.0
        assert config.critical_threshold_percent == 90.0

    def test_invalid_warning_threshold(self):
        """Test invalid warning threshold."""
        from src.config.sections.infrastructure import ResourcePredictionConfig

        with pytest.raises(ValueError, match="warning_threshold_percent must be 0-100"):
            ResourcePredictionConfig(warning_threshold_percent=150.0)

    def test_invalid_critical_threshold(self):
        """Test invalid critical threshold."""
        from src.config.sections.infrastructure import ResourcePredictionConfig

        with pytest.raises(ValueError, match="critical_threshold_percent must be 0-100"):
            ResourcePredictionConfig(critical_threshold_percent=-10.0)

    def test_warning_above_critical(self):
        """Test warning threshold above critical threshold."""
        from src.config.sections.infrastructure import ResourcePredictionConfig

        with pytest.raises(ValueError, match="warning_threshold_percent.*must be <= critical_threshold_percent"):
            ResourcePredictionConfig(
                warning_threshold_percent=90.0,
                critical_threshold_percent=75.0,
            )

    def test_invalid_history_window(self):
        """Test invalid history window."""
        from src.config.sections.infrastructure import ResourcePredictionConfig

        with pytest.raises(ValueError, match="history_window must be >= 1"):
            ResourcePredictionConfig(history_window=0)

    def test_negative_memory_params(self):
        """Test negative memory parameters."""
        from src.config.sections.infrastructure import ResourcePredictionConfig

        with pytest.raises(ValueError, match="base_memory_per_vo_minute must be >= 0"):
            ResourcePredictionConfig(base_memory_per_vo_minute=-10.0)

        with pytest.raises(ValueError, match="memory_per_video must be >= 0"):
            ResourcePredictionConfig(memory_per_video=-5.0)

        with pytest.raises(ValueError, match="memory_per_segment must be >= 0"):
            ResourcePredictionConfig(memory_per_segment=-2.0)
