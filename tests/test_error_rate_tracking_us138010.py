"""Tests for US-138-010: Per-stage error rate tracking and alerting.

Tests cover:
  - StageMetrics.error_rate computation
  - StageMetrics serialization (to_dict, from_dict)
  - ErrorRateThresholdConfig configuration
  - ErrorRateTrackingConfig threshold checking
  - Pipeline error rate event emission
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.stages import StageMetrics
from src.config.sections.infrastructure import (
    ErrorRateThresholdConfig,
    ErrorRateTrackingConfig,
)


# ---------------------------------------------------------------------------
# Test: StageMetrics error_rate computation
# ---------------------------------------------------------------------------

class TestStageMetricsErrorRate:
    """US-138-010: StageMetrics.error_rate calculation."""

    def test_error_rate_zero_when_no_items(self):
        """Error rate is 0 when items_processed is 0."""
        metrics = StageMetrics(items_processed=0, items_failed=0)
        metrics.compute_error_rate()
        assert metrics.error_rate == 0.0

    def test_error_rate_calculation(self):
        """Error rate is correctly calculated as items_failed / items_processed."""
        metrics = StageMetrics(items_processed=100, items_failed=20)
        metrics.compute_error_rate()
        assert metrics.error_rate == 0.2

    def test_error_rate_full_failure(self):
        """Error rate is 1.0 when all items fail."""
        metrics = StageMetrics(items_processed=50, items_failed=50)
        metrics.compute_error_rate()
        assert metrics.error_rate == 1.0

    def test_error_rate_no_failures(self):
        """Error rate is 0 when no items fail."""
        metrics = StageMetrics(items_processed=100, items_failed=0)
        metrics.compute_error_rate()
        assert metrics.error_rate == 0.0


class TestStageMetricsErrorRateSerialization:
    """US-138-010: StageMetrics error_rate serialization."""

    def test_to_dict_includes_error_rate(self):
        """to_dict includes error_rate when > 0."""
        metrics = StageMetrics(items_processed=100, items_failed=20, error_rate=0.2)
        d = metrics.to_dict()
        assert 'error_rate' in d
        assert d['error_rate'] == 0.2

    def test_to_dict_excludes_zero_error_rate(self):
        """to_dict excludes error_rate when 0."""
        metrics = StageMetrics(items_processed=100, items_failed=0, error_rate=0.0)
        d = metrics.to_dict()
        assert 'error_rate' not in d

    def test_from_dict_includes_error_rate(self):
        """from_dict includes error_rate."""
        data = {'items_processed': 100, 'items_failed': 20, 'error_rate': 0.2}
        metrics = StageMetrics.from_dict(data)
        assert metrics.error_rate == 0.2

    def test_from_dict_defaults_error_rate(self):
        """from_dict defaults error_rate to 0.0."""
        data = {'items_processed': 100, 'items_failed': 20}
        metrics = StageMetrics.from_dict(data)
        assert metrics.error_rate == 0.0


# ---------------------------------------------------------------------------
# Test: ErrorRateThresholdConfig
# ---------------------------------------------------------------------------

class TestErrorRateThresholdConfig:
    """US-138-010: ErrorRateThresholdConfig validation."""

    def test_default_values(self):
        """Default thresholds are 20% warning, 50% critical."""
        config = ErrorRateThresholdConfig()
        assert config.warning_threshold == 0.2
        assert config.critical_threshold == 0.5

    def test_custom_values(self):
        """Custom threshold values are set correctly."""
        config = ErrorRateThresholdConfig(warning_threshold=0.15, critical_threshold=0.4)
        assert config.warning_threshold == 0.15
        assert config.critical_threshold == 0.4


# ---------------------------------------------------------------------------
# Test: ErrorRateTrackingConfig
# ---------------------------------------------------------------------------

class TestErrorRateTrackingConfig:
    """US-138-010: ErrorRateTrackingConfig threshold checking."""

    def test_default_values(self):
        """Default config has correct values."""
        config = ErrorRateTrackingConfig()
        assert config.enabled is True
        assert config.default_warning_threshold == 0.2
        assert config.default_critical_threshold == 0.5

    def test_get_threshold_default(self):
        """get_threshold returns defaults for unknown stage type."""
        config = ErrorRateTrackingConfig()
        threshold = config.get_threshold('UNKNOWN_STAGE')
        assert threshold.warning_threshold == 0.2
        assert threshold.critical_threshold == 0.5

    def test_get_threshold_custom(self):
        """get_threshold returns custom threshold for configured stage type."""
        config = ErrorRateTrackingConfig(
            stage_thresholds={
                'VIDEO_SEARCH': ErrorRateThresholdConfig(warning_threshold=0.1, critical_threshold=0.3)
            }
        )
        threshold = config.get_threshold('VIDEO_SEARCH')
        assert threshold.warning_threshold == 0.1
        assert threshold.critical_threshold == 0.3

    def test_check_threshold_warning(self):
        """check_threshold returns warning when threshold exceeded."""
        config = ErrorRateTrackingConfig()
        status, threshold = config.check_threshold('STAGE', 0.25)
        assert status == 'warning'
        assert threshold == 0.2

    def test_check_threshold_critical(self):
        """check_threshold returns critical when critical threshold exceeded."""
        config = ErrorRateTrackingConfig()
        status, threshold = config.check_threshold('STAGE', 0.6)
        assert status == 'critical'
        assert threshold == 0.5

    def test_check_threshold_ok(self):
        """check_threshold returns ok when within thresholds."""
        config = ErrorRateTrackingConfig()
        status, threshold = config.check_threshold('STAGE', 0.1)
        assert status == 'ok'
        assert threshold == 0.0


class TestErrorRateTrackingConfigPostInit:
    """US-138-010: ErrorRateTrackingConfig __post_init__ conversion."""

    def test_post_init_converts_dict(self):
        """__post_init__ converts dict stage_thresholds to config objects."""
        config = ErrorRateTrackingConfig(
            stage_thresholds={
                'DOWNLOAD': {'warning_threshold': 0.1, 'critical_threshold': 0.3}
            }
        )
        threshold = config.get_threshold('DOWNLOAD')
        assert threshold.warning_threshold == 0.1
        assert threshold.critical_threshold == 0.3


# ---------------------------------------------------------------------------
# Test: Pipeline error rate integration
# ---------------------------------------------------------------------------

class TestPipelineErrorRate:
    """US-138-010: Pipeline error rate computation."""

    def test_compute_error_rate_method_exists(self):
        """Pipeline has _compute_and_check_error_rate method."""
        from src.pipeline import PipelineOrchestrator
        assert hasattr(PipelineOrchestrator, '_compute_and_check_error_rate')

    @patch('src.pipeline.PipelineOrchestrator.__init__')
    def test_error_rate_emits_event(self, mock_init):
        """Error rate exceeding threshold emits warning event."""
        from src.pipeline import PipelineOrchestrator
        from src.stages import StageMetrics
        from src.pipeline_events import EVENT_ERROR_RATE_THRESHOLD

        # Create mock pipeline
        mock_init.return_value = None
        pipeline = PipelineOrchestrator.__new__(PipelineOrchestrator)
        pipeline.config = MagicMock()
        pipeline.config.pipeline_config = MagicMock()
        pipeline.config.pipeline_config.error_rate_tracking = ErrorRateTrackingConfig(
            enabled=True,
            default_warning_threshold=0.2,
            default_critical_threshold=0.5
        )
        pipeline.stage_metrics = {}
        pipeline._event_bus = MagicMock()
        pipeline.emit_event = MagicMock()

        # Create metrics with high error rate
        metrics = StageMetrics(items_processed=100, items_failed=30)
        metrics.compute_error_rate()

        # Set stage metrics
        pipeline.stage_metrics['TEST_STAGE'] = metrics

        # Call the method
        pipeline._compute_and_check_error_rate('TEST_STAGE')

        # Verify event was emitted
        pipeline.emit_event.assert_called_once()
        call_args = pipeline.emit_event.call_args
        event = call_args[0][0]
        assert event.event_type == EVENT_ERROR_RATE_THRESHOLD
        assert event.data['status'] == 'warning'
        assert event.data['error_rate'] == 0.3

    @patch('src.pipeline.PipelineOrchestrator.__init__')
    def test_error_rate_no_warning_when_ok(self, mock_init):
        """No warning event when error rate below threshold."""
        from src.pipeline import PipelineOrchestrator
        from src.stages import StageMetrics

        # Create mock pipeline
        mock_init.return_value = None
        pipeline = PipelineOrchestrator.__new__(PipelineOrchestrator)
        pipeline.config = MagicMock()
        pipeline.config.pipeline_config = MagicMock()
        pipeline.config.pipeline_config.error_rate_tracking = ErrorRateTrackingConfig(
            enabled=True,
            default_warning_threshold=0.2,
            default_critical_threshold=0.5
        )
        pipeline.stage_metrics = {}
        pipeline._event_bus = MagicMock()
        pipeline.emit_event = MagicMock()

        # Create metrics with low error rate
        metrics = StageMetrics(items_processed=100, items_failed=10)
        metrics.compute_error_rate()

        # Set stage metrics
        pipeline.stage_metrics['TEST_STAGE'] = metrics

        # Call the method
        pipeline._compute_and_check_error_rate('TEST_STAGE')

        # Verify no event was emitted (error_rate 0.1 < warning threshold 0.2)
        pipeline.emit_event.assert_not_called()

    @patch('src.pipeline.PipelineOrchestrator.__init__')
    def test_error_rate_disabled_config(self, mock_init):
        """No action when error rate tracking is disabled."""
        from src.pipeline import PipelineOrchestrator
        from src.stages import StageMetrics

        # Create mock pipeline
        mock_init.return_value = None
        pipeline = PipelineOrchestrator.__new__(PipelineOrchestrator)
        pipeline.config = MagicMock()
        pipeline.config.pipeline_config = MagicMock()
        pipeline.config.pipeline_config.error_rate_tracking = ErrorRateTrackingConfig(
            enabled=False
        )
        pipeline.stage_metrics = {}
        pipeline._event_bus = MagicMock()
        pipeline.emit_event = MagicMock()

        # Create metrics
        metrics = StageMetrics(items_processed=100, items_failed=30)
        metrics.compute_error_rate()
        pipeline.stage_metrics['TEST_STAGE'] = metrics

        # Call the method
        pipeline._compute_and_check_error_rate('TEST_STAGE')

        # Verify no event was emitted (disabled)
        pipeline.emit_event.assert_not_called()
