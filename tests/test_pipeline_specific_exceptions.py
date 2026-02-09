"""
Tests for US-85-002: Verify broad except-Exception blocks in pipeline.py
are replaced with specific exception types.

Ensures that:
- Known exception types (OSError, TypeError, etc.) are caught and logged
- Unexpected exceptions (e.g., SystemExit, KeyboardInterrupt) propagate
- Callback errors don't crash the pipeline
- Metrics collection failures are logged, not silently swallowed
"""

import json
import logging
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PipelineOrchestrator, _collect_escalation_metrics
from src.state import PipelineState
from src.stages import Stage, StageResult


class SimpleStage(Stage):
    """Minimal stage for testing callback error handling."""

    def __init__(self, name: str = "TEST"):
        self.name = name

    def can_skip(self, state, checkpoint):
        return False

    def restore(self, state, checkpoint, config=None):
        return False

    def validate_inputs(self, state, config):
        return None

    def run(self, state, config, checkpoint):
        return StageResult(success=True, data={"done": True})


@pytest.fixture
def pipeline_with_stage(tmp_path):
    """Create a pipeline with a single test stage."""
    config = MagicMock()
    config.healing = MagicMock()
    config.healing.enabled = False
    # Satisfy config validation
    config._config_hash = ''

    stages = [SimpleStage("TEST")]

    # Patch _validate_config to avoid real validation in tests
    with patch.object(PipelineOrchestrator, '_validate_config', return_value=[]):
        pipeline = PipelineOrchestrator(
            config=config,
            project_dir=tmp_path,
            stages=stages,
        )

    return pipeline


class TestCallbackSpecificExceptions:
    """Verify callback error handling catches specific types, not all."""

    def test_on_stage_start_catches_type_error(self, pipeline_with_stage, caplog):
        """TypeError in on_stage_start callback is caught and logged."""
        pipeline = pipeline_with_stage

        def bad_callback(name):
            raise TypeError("bad argument")

        with caplog.at_level(logging.WARNING):
            pipeline.run(on_stage_start=bad_callback)

        assert "on_stage_start callback failed" in caplog.text
        assert "bad argument" in caplog.text

    def test_on_stage_complete_catches_runtime_error(self, pipeline_with_stage, caplog):
        """RuntimeError in on_stage_complete callback is caught and logged."""
        pipeline = pipeline_with_stage

        def bad_callback(name, result, elapsed):
            raise RuntimeError("callback broke")

        with caplog.at_level(logging.WARNING):
            pipeline.run(on_stage_complete=bad_callback)

        assert "on_stage_complete callback failed" in caplog.text

    def test_callback_keyboard_interrupt_propagates(self, pipeline_with_stage):
        """KeyboardInterrupt in a callback must NOT be caught."""
        pipeline = pipeline_with_stage

        def interrupting_callback(name):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            pipeline.run(on_stage_start=interrupting_callback)

    def test_callback_system_exit_propagates(self, pipeline_with_stage):
        """SystemExit in a callback must NOT be caught."""
        pipeline = pipeline_with_stage

        def exiting_callback(name):
            raise SystemExit(1)

        with pytest.raises(SystemExit):
            pipeline.run(on_stage_start=exiting_callback)


class TestEstimateTimingSpecificExceptions:
    """Verify _log_stage_estimate and _save_stage_timing use specific types."""

    def test_log_estimate_catches_os_error(self, pipeline_with_stage, caplog):
        """OSError during duration estimation is caught."""
        pipeline = pipeline_with_stage

        with patch("src.pipeline.estimate_duration", side_effect=OSError("disk full")):
            with caplog.at_level(logging.DEBUG):
                pipeline._log_stage_estimate("TEST")

        assert "Could not estimate duration" in caplog.text
        assert "disk full" in caplog.text

    def test_log_estimate_catches_json_decode_error(self, pipeline_with_stage, caplog):
        """JSONDecodeError during estimation is caught."""
        pipeline = pipeline_with_stage

        err = json.JSONDecodeError("bad json", "", 0)
        with patch("src.pipeline.estimate_duration", side_effect=err):
            with caplog.at_level(logging.DEBUG):
                pipeline._log_stage_estimate("TEST")

        assert "Could not estimate duration" in caplog.text

    def test_save_timing_catches_os_error(self, pipeline_with_stage, caplog):
        """OSError during timing save is caught."""
        pipeline = pipeline_with_stage

        with patch("src.pipeline.append_stage_timing", side_effect=OSError("read-only")):
            with caplog.at_level(logging.DEBUG):
                pipeline._save_stage_timing("TEST", 1.5)

        assert "Could not save stage timing" in caplog.text

    def test_estimate_unexpected_exception_propagates(self, pipeline_with_stage):
        """An unexpected exception type (e.g., RuntimeError) propagates."""
        pipeline = pipeline_with_stage

        with patch("src.pipeline.estimate_duration", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError, match="unexpected"):
                pipeline._log_stage_estimate("TEST")


class TestCollectEscalationMetricsSpecificExceptions:
    """Verify _collect_escalation_metrics uses specific exception types."""

    def test_catches_attribute_error(self, caplog):
        """AttributeError from missing downloader attributes is caught."""
        pipeline = MagicMock()
        stage = MagicMock()
        stage.downloader = MagicMock()
        stage.downloader.escalation_manager = MagicMock()
        stage.downloader.escalation_manager.get_metrics.side_effect = AttributeError("no attr")
        pipeline.stages = [stage]
        orchestrator = MagicMock()

        with caplog.at_level(logging.DEBUG):
            _collect_escalation_metrics(pipeline, orchestrator)

        assert "escalation metrics collection failed" in caplog.text

    def test_catches_key_error(self, caplog):
        """KeyError from missing metrics keys is caught."""
        pipeline = MagicMock()
        stage = MagicMock()
        stage.downloader = MagicMock()
        stage.downloader.escalation_manager = MagicMock()
        stage.downloader.escalation_manager.get_metrics.side_effect = KeyError("missing_key")
        pipeline.stages = [stage]
        orchestrator = MagicMock()

        with caplog.at_level(logging.DEBUG):
            _collect_escalation_metrics(pipeline, orchestrator)

        assert "escalation metrics collection failed" in caplog.text

    def test_unexpected_exception_propagates(self):
        """RuntimeError should NOT be caught by _collect_escalation_metrics."""
        pipeline = MagicMock()
        stage = MagicMock()
        stage.downloader = MagicMock()
        stage.downloader.escalation_manager = MagicMock()
        stage.downloader.escalation_manager.get_metrics.side_effect = RuntimeError("boom")
        pipeline.stages = [stage]
        orchestrator = MagicMock()

        with pytest.raises(RuntimeError, match="boom"):
            _collect_escalation_metrics(pipeline, orchestrator)
