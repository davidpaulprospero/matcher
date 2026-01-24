"""
Unit tests for PipelineOrchestrator.

US-004: Add PipelineOrchestrator unit tests

Focused unit tests for PipelineOrchestrator covering:
- add_stage() fluent interface
- load_checkpoint() with valid and invalid checkpoints
- run() stage filtering with skip_stages and only_stages
- run() resume behavior with checkpoint
- stage timing recording in stage_timings dict
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.pipeline import PipelineOrchestrator
from src.state import PipelineState
from src.stages import Stage, StageResult
from src.config import Config


class MockStage(Stage):
    """Mock stage for testing PipelineOrchestrator."""

    def __init__(
        self,
        name: str,
        should_fail: bool = False,
        validation_error: str = None,
        can_skip_value: bool = False,
        restore_success: bool = True,
        run_delay: float = 0.0
    ):
        self.name = name
        self._should_fail = should_fail
        self._validation_error = validation_error
        self._can_skip_value = can_skip_value
        self._restore_success = restore_success
        self._run_delay = run_delay
        self._run_called = False
        self._restore_called = False
        self._can_skip_called = False

    def can_skip(self, state: PipelineState, checkpoint) -> bool:
        """Check if stage can be skipped via checkpoint."""
        self._can_skip_called = True
        return self._can_skip_value

    def restore(self, state: PipelineState, checkpoint, config=None) -> bool:
        """Restore stage state from checkpoint."""
        self._restore_called = True
        return self._restore_success

    def validate_inputs(self, state: PipelineState, config: Config) -> str:
        """Validate inputs before running."""
        return self._validation_error

    def run(self, state: PipelineState, config: Config, checkpoint) -> StageResult:
        """Execute the stage."""
        import time
        if self._run_delay > 0:
            time.sleep(self._run_delay)

        self._run_called = True

        if self._should_fail:
            return StageResult.fail(f"{self.name} failed intentionally")

        return StageResult.ok(data={'stage': self.name, 'completed': True})


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Create a minimal config object."""
    config = Config()
    return config


class TestAddStageFluent:
    """Test add_stage() fluent interface returns self."""

    def test_add_stage_returns_self(self, temp_project_dir, mock_config):
        """Test that add_stage() returns the PipelineOrchestrator instance."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage = MockStage("test_stage")

        result = pipeline.add_stage(stage)

        assert result is pipeline, "add_stage() should return self"

    def test_add_stage_chaining(self, temp_project_dir, mock_config):
        """Test that add_stage() can be chained multiple times."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        result = (
            pipeline
            .add_stage(MockStage("stage1"))
            .add_stage(MockStage("stage2"))
            .add_stage(MockStage("stage3"))
        )

        assert result is pipeline
        assert len(pipeline.stages) == 3
        assert [s.name for s in pipeline.stages] == ["stage1", "stage2", "stage3"]

    def test_add_stage_appends_to_list(self, temp_project_dir, mock_config):
        """Test that add_stage() appends stage to stages list."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage1 = MockStage("first")
        stage2 = MockStage("second")

        pipeline.add_stage(stage1)
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0] is stage1

        pipeline.add_stage(stage2)
        assert len(pipeline.stages) == 2
        assert pipeline.stages[1] is stage2


class TestLoadCheckpoint:
    """Test load_checkpoint() with valid and invalid checkpoints."""

    def test_load_checkpoint_no_file(self, temp_project_dir, mock_config):
        """Test load_checkpoint() returns False when no checkpoint exists."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        result = pipeline.load_checkpoint()

        assert result is False
        assert pipeline.resume_mode is False

    def test_load_checkpoint_returns_none(self, temp_project_dir, mock_config):
        """Test load_checkpoint() returns False when load() returns None."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        # Mock checkpoint to exist but return None on load
        pipeline.checkpoint.exists = Mock(return_value=True)
        pipeline.checkpoint.load = Mock(return_value=None)

        result = pipeline.load_checkpoint()

        assert result is False
        assert pipeline.resume_mode is False

    def test_load_checkpoint_invalid_validation(self, temp_project_dir, mock_config):
        """Test load_checkpoint() returns False when validation fails."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        pipeline.checkpoint.exists = Mock(return_value=True)
        pipeline.checkpoint.load = Mock(return_value={'data': 'test'})
        pipeline.checkpoint.validate = Mock(return_value={
            'valid': False,
            'errors': ['Checkpoint corrupted', 'Missing required field'],
            'warnings': []
        })

        with patch('src.pipeline.logger') as mock_logger:
            result = pipeline.load_checkpoint()

        assert result is False
        assert pipeline.resume_mode is False
        # Should log both errors
        assert mock_logger.error.call_count == 2

    def test_load_checkpoint_valid(self, temp_project_dir, mock_config):
        """Test load_checkpoint() returns True for valid checkpoint."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        pipeline.checkpoint.exists = Mock(return_value=True)
        pipeline.checkpoint.load = Mock(return_value={'data': 'test'})
        pipeline.checkpoint.validate = Mock(return_value={
            'valid': True,
            'errors': [],
            'warnings': []
        })

        result = pipeline.load_checkpoint()

        assert result is True
        assert pipeline.resume_mode is True

    def test_load_checkpoint_valid_with_warnings(self, temp_project_dir, mock_config):
        """Test load_checkpoint() logs warnings but still returns True."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        pipeline.checkpoint.exists = Mock(return_value=True)
        pipeline.checkpoint.load = Mock(return_value={'data': 'test'})
        pipeline.checkpoint.validate = Mock(return_value={
            'valid': True,
            'errors': [],
            'warnings': ['Config hash changed', 'Voiceover modified']
        })

        with patch('src.pipeline.logger') as mock_logger:
            result = pipeline.load_checkpoint()

        assert result is True
        assert pipeline.resume_mode is True
        # Should log both warnings
        assert mock_logger.warning.call_count == 2


class TestRunStageFiltering:
    """Test run() stage filtering with skip_stages and only_stages."""

    def test_skip_stages_single(self, temp_project_dir, mock_config):
        """Test skipping a single stage via skip_stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stages = [MockStage("A"), MockStage("B"), MockStage("C")]
        for s in stages:
            pipeline.add_stage(s)

        result = pipeline.run(resume=False, skip_stages=["B"])

        assert result is True
        assert stages[0]._run_called is True
        assert stages[1]._run_called is False  # Skipped
        assert stages[2]._run_called is True

    def test_skip_stages_multiple(self, temp_project_dir, mock_config):
        """Test skipping multiple stages via skip_stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stages = [MockStage("A"), MockStage("B"), MockStage("C"), MockStage("D")]
        for s in stages:
            pipeline.add_stage(s)

        result = pipeline.run(resume=False, skip_stages=["A", "C"])

        assert result is True
        assert stages[0]._run_called is False
        assert stages[1]._run_called is True
        assert stages[2]._run_called is False
        assert stages[3]._run_called is True

    def test_only_stages_single(self, temp_project_dir, mock_config):
        """Test running only a single stage via only_stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stages = [MockStage("A"), MockStage("B"), MockStage("C")]
        for s in stages:
            pipeline.add_stage(s)

        result = pipeline.run(resume=False, only_stages=["B"])

        assert result is True
        assert stages[0]._run_called is False
        assert stages[1]._run_called is True
        assert stages[2]._run_called is False

    def test_only_stages_multiple(self, temp_project_dir, mock_config):
        """Test running only specific stages via only_stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stages = [MockStage("A"), MockStage("B"), MockStage("C"), MockStage("D")]
        for s in stages:
            pipeline.add_stage(s)

        result = pipeline.run(resume=False, only_stages=["A", "D"])

        assert result is True
        assert stages[0]._run_called is True
        assert stages[1]._run_called is False
        assert stages[2]._run_called is False
        assert stages[3]._run_called is True

    def test_skip_stages_takes_precedence(self, temp_project_dir, mock_config):
        """Test that skip_stages takes precedence when combined with only_stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stages = [MockStage("A"), MockStage("B"), MockStage("C")]
        for s in stages:
            pipeline.add_stage(s)

        # Stage B is in only_stages but also in skip_stages
        result = pipeline.run(
            resume=False,
            skip_stages=["B"],
            only_stages=["A", "B", "C"]
        )

        assert result is True
        assert stages[0]._run_called is True
        assert stages[1]._run_called is False  # skip_stages wins
        assert stages[2]._run_called is True

    def test_skip_stages_not_tracked_in_timings(self, temp_project_dir, mock_config):
        """Test that skipped stages don't appear in stage_timings."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        for s in [MockStage("A"), MockStage("B"), MockStage("C")]:
            pipeline.add_stage(s)

        pipeline.run(resume=False, skip_stages=["B"])

        assert "A" in pipeline.stage_timings
        assert "B" not in pipeline.stage_timings
        assert "C" in pipeline.stage_timings


class TestRunResumeWithCheckpoint:
    """Test run() resume behavior with checkpoint."""

    def test_resume_triggers_load_checkpoint(self, temp_project_dir, mock_config):
        """Test that resume=True triggers load_checkpoint()."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        with patch.object(pipeline, 'load_checkpoint') as mock_load:
            mock_load.return_value = False
            pipeline.run(resume=True)

        mock_load.assert_called_once()

    def test_resume_false_skips_load_checkpoint(self, temp_project_dir, mock_config):
        """Test that resume=False does not trigger load_checkpoint()."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        with patch.object(pipeline, 'load_checkpoint') as mock_load:
            pipeline.run(resume=False)

        mock_load.assert_not_called()

    def test_resume_skips_completed_stages(self, temp_project_dir, mock_config):
        """Test that resume mode skips stages that can_skip returns True."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage1 = MockStage("stage1", can_skip_value=True)
        stage2 = MockStage("stage2", can_skip_value=False)
        pipeline.add_stage(stage1).add_stage(stage2)

        # Enable resume mode manually
        pipeline.resume_mode = True

        pipeline.run(resume=False)  # resume=False since we set resume_mode manually

        assert stage1._can_skip_called is True
        assert stage1._run_called is False  # Skipped
        assert stage1._restore_called is True  # Restored from checkpoint
        assert stage2._run_called is True

    def test_resume_restore_failure_logs_warning(self, temp_project_dir, mock_config):
        """Test that restore failure during resume logs a warning."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage = MockStage("stage1", can_skip_value=True, restore_success=False)
        pipeline.add_stage(stage)
        pipeline.resume_mode = True

        with patch('src.pipeline.logger') as mock_logger:
            result = pipeline.run(resume=False)

        assert result is True  # Pipeline continues despite restore failure
        mock_logger.warning.assert_called_with(
            "Failed to restore stage1 from checkpoint"
        )

    def test_resume_mode_not_set_when_no_checkpoint(self, temp_project_dir, mock_config):
        """Test resume_mode stays False when no valid checkpoint."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage = MockStage("stage1", can_skip_value=True)
        pipeline.add_stage(stage)

        # resume_mode is False by default (no checkpoint)
        pipeline.run(resume=True)

        # can_skip should not be called since resume_mode is False
        assert stage._can_skip_called is False
        assert stage._run_called is True


class TestStageTimingRecording:
    """Test stage timing is recorded in stage_timings dict."""

    def test_timing_recorded_for_single_stage(self, temp_project_dir, mock_config):
        """Test timing is recorded for a single stage."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("test_stage"))

        pipeline.run(resume=False)

        assert "test_stage" in pipeline.stage_timings
        assert isinstance(pipeline.stage_timings["test_stage"], float)
        assert pipeline.stage_timings["test_stage"] >= 0

    def test_timing_recorded_for_multiple_stages(self, temp_project_dir, mock_config):
        """Test timing is recorded for multiple stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        for name in ["A", "B", "C"]:
            pipeline.add_stage(MockStage(name))

        pipeline.run(resume=False)

        assert len(pipeline.stage_timings) == 3
        for name in ["A", "B", "C"]:
            assert name in pipeline.stage_timings
            assert pipeline.stage_timings[name] >= 0

    def test_timing_reflects_actual_duration(self, temp_project_dir, mock_config):
        """Test timing reflects actual stage execution time."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("slow_stage", run_delay=0.05))

        pipeline.run(resume=False)

        # Stage with 50ms delay should have timing >= 50ms
        assert pipeline.stage_timings["slow_stage"] >= 0.05

    def test_timing_synced_with_state(self, temp_project_dir, mock_config):
        """Test that stage_timings is synced to state.stage_timings."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("synced_stage"))

        pipeline.run(resume=False)

        assert pipeline.stage_timings["synced_stage"] == pipeline.state.stage_timings["synced_stage"]

    def test_timing_not_recorded_for_failed_stage(self, temp_project_dir, mock_config):
        """Test that timing IS recorded even for failed stages (before failure)."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("fail_stage", should_fail=True))

        with patch('src.pipeline.logger'):
            pipeline.run(resume=False)

        # Timing should still be recorded since stage ran (just failed)
        assert "fail_stage" in pipeline.stage_timings

    def test_timing_not_recorded_for_validation_failure(self, temp_project_dir, mock_config):
        """Test that timing is NOT recorded when validation fails."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("invalid_stage", validation_error="Missing input"))

        with patch('src.pipeline.logger'):
            pipeline.run(resume=False)

        # Timing should NOT be recorded since stage never ran
        assert "invalid_stage" not in pipeline.stage_timings


class TestCheckpointSaving:
    """Test checkpoint is saved after successful stage execution."""

    def test_checkpoint_saved_after_stage(self, temp_project_dir, mock_config):
        """Test checkpoint.save() is called after successful stage."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("save_test"))
        pipeline.checkpoint.save = Mock()

        pipeline.run(resume=False)

        pipeline.checkpoint.save.assert_called_once()
        args = pipeline.checkpoint.save.call_args[0]
        assert args[0] == "save_test"  # Stage name
        assert args[1] == {'stage': 'save_test', 'completed': True}  # Stage data

    def test_checkpoint_not_saved_on_failure(self, temp_project_dir, mock_config):
        """Test checkpoint.save() is NOT called when stage fails."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("fail_save", should_fail=True))
        pipeline.checkpoint.save = Mock()

        with patch('src.pipeline.logger'):
            pipeline.run(resume=False)

        pipeline.checkpoint.save.assert_not_called()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_pipeline(self, temp_project_dir, mock_config):
        """Test running an empty pipeline succeeds."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        result = pipeline.run(resume=False)

        assert result is True
        assert pipeline.stage_timings == {}

    def test_current_stage_cleared_after_run(self, temp_project_dir, mock_config):
        """Test current_stage is None after pipeline completes."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("test"))

        pipeline.run(resume=False)

        assert pipeline.current_stage is None

    def test_all_stages_after_failure_not_run(self, temp_project_dir, mock_config):
        """Test that stages after a failed stage are not run."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stages = [
            MockStage("A"),
            MockStage("B", should_fail=True),
            MockStage("C"),
            MockStage("D")
        ]
        for s in stages:
            pipeline.add_stage(s)

        with patch('src.pipeline.logger'):
            result = pipeline.run(resume=False)

        assert result is False
        assert stages[0]._run_called is True
        assert stages[1]._run_called is True
        assert stages[2]._run_called is False
        assert stages[3]._run_called is False

    def test_clear_checkpoint(self, temp_project_dir, mock_config):
        """Test clear_checkpoint() calls checkpoint.clear() and resets resume_mode."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.resume_mode = True
        pipeline.checkpoint.clear = Mock()

        pipeline.clear_checkpoint()

        pipeline.checkpoint.clear.assert_called_once()
        assert pipeline.resume_mode is False

    def test_get_summary_empty(self, temp_project_dir, mock_config):
        """Test get_summary() for empty pipeline."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        summary = pipeline.get_summary()

        assert summary['stages_run'] == []
        assert summary['total_time'] == 0
        assert 'state' in summary

    def test_get_summary_after_run(self, temp_project_dir, mock_config):
        """Test get_summary() after running pipeline."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A")).add_stage(MockStage("B"))

        pipeline.run(resume=False)
        summary = pipeline.get_summary()

        assert summary['stages_run'] == ["A", "B"]
        assert summary['total_time'] >= 0
        assert len(summary['stage_timings']) == 2


class TestStageProgressCallbacks:
    """Test on_stage_start and on_stage_complete callbacks."""

    def test_on_stage_start_called_for_each_stage(self, temp_project_dir, mock_config):
        """Test that on_stage_start is called for each stage in order."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stages = [MockStage("A"), MockStage("B"), MockStage("C")]
        for s in stages:
            pipeline.add_stage(s)

        started_stages = []

        def on_start(stage_name):
            started_stages.append(stage_name)

        result = pipeline.run(resume=False, on_stage_start=on_start)

        assert result is True
        assert started_stages == ["A", "B", "C"]

    def test_on_stage_complete_called_for_each_stage(self, temp_project_dir, mock_config):
        """Test that on_stage_complete is called for each stage with result and timing."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stages = [MockStage("A"), MockStage("B")]
        for s in stages:
            pipeline.add_stage(s)

        completed_stages = []

        def on_complete(stage_name, result, elapsed):
            completed_stages.append({
                'name': stage_name,
                'success': result.success,
                'elapsed': elapsed
            })

        result = pipeline.run(resume=False, on_stage_complete=on_complete)

        assert result is True
        assert len(completed_stages) == 2
        assert completed_stages[0]['name'] == "A"
        assert completed_stages[0]['success'] is True
        assert completed_stages[0]['elapsed'] >= 0
        assert completed_stages[1]['name'] == "B"
        assert completed_stages[1]['success'] is True

    def test_callbacks_called_in_correct_order(self, temp_project_dir, mock_config):
        """Test that start callback is called before complete for each stage."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A")).add_stage(MockStage("B"))

        events = []

        def on_start(stage_name):
            events.append(f"start:{stage_name}")

        def on_complete(stage_name, result, elapsed):
            events.append(f"complete:{stage_name}")

        pipeline.run(resume=False, on_stage_start=on_start, on_stage_complete=on_complete)

        assert events == [
            "start:A", "complete:A",
            "start:B", "complete:B"
        ]

    def test_on_stage_start_exception_does_not_stop_pipeline(self, temp_project_dir, mock_config):
        """Test that exception in on_stage_start callback does not stop the pipeline."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A")).add_stage(MockStage("B"))

        def failing_start(stage_name):
            if stage_name == "A":
                raise ValueError("Callback error!")

        completed = []

        def track_complete(stage_name, result, elapsed):
            completed.append(stage_name)

        with patch('src.pipeline.logger') as mock_logger:
            result = pipeline.run(
                resume=False,
                on_stage_start=failing_start,
                on_stage_complete=track_complete
            )

        assert result is True
        assert completed == ["A", "B"]  # Both stages completed despite callback failure
        # Should have logged a warning
        assert any("on_stage_start callback failed" in str(call) for call in mock_logger.warning.call_args_list)

    def test_on_stage_complete_exception_does_not_stop_pipeline(self, temp_project_dir, mock_config):
        """Test that exception in on_stage_complete callback does not stop the pipeline."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A")).add_stage(MockStage("B"))

        started = []

        def track_start(stage_name):
            started.append(stage_name)

        def failing_complete(stage_name, result, elapsed):
            if stage_name == "A":
                raise RuntimeError("Complete callback error!")

        with patch('src.pipeline.logger') as mock_logger:
            result = pipeline.run(
                resume=False,
                on_stage_start=track_start,
                on_stage_complete=failing_complete
            )

        assert result is True
        assert started == ["A", "B"]  # Both stages started despite callback failure
        # Should have logged a warning
        assert any("on_stage_complete callback failed" in str(call) for call in mock_logger.warning.call_args_list)

    def test_callbacks_receive_failed_stage_result(self, temp_project_dir, mock_config):
        """Test that on_stage_complete receives failed result for failed stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A", should_fail=True))

        received_results = []

        def on_complete(stage_name, result, elapsed):
            received_results.append({
                'name': stage_name,
                'success': result.success,
                'error': result.error
            })

        with patch('src.pipeline.logger'):
            result = pipeline.run(resume=False, on_stage_complete=on_complete)

        assert result is False
        assert len(received_results) == 1
        assert received_results[0]['name'] == "A"
        assert received_results[0]['success'] is False
        assert "failed intentionally" in received_results[0]['error']

    def test_callbacks_not_called_for_skipped_stages(self, temp_project_dir, mock_config):
        """Test that callbacks are not called for stages skipped via skip_stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A")).add_stage(MockStage("B")).add_stage(MockStage("C"))

        started = []
        completed = []

        def on_start(stage_name):
            started.append(stage_name)

        def on_complete(stage_name, result, elapsed):
            completed.append(stage_name)

        pipeline.run(
            resume=False,
            skip_stages=["B"],
            on_stage_start=on_start,
            on_stage_complete=on_complete
        )

        assert started == ["A", "C"]
        assert completed == ["A", "C"]

    def test_callbacks_not_called_for_checkpoint_skipped_stages(self, temp_project_dir, mock_config):
        """Test that callbacks are not called for stages skipped via checkpoint resume."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A", can_skip_value=True))
        pipeline.add_stage(MockStage("B"))
        pipeline.resume_mode = True

        started = []
        completed = []

        def on_start(stage_name):
            started.append(stage_name)

        def on_complete(stage_name, result, elapsed):
            completed.append(stage_name)

        pipeline.run(
            resume=False,
            on_stage_start=on_start,
            on_stage_complete=on_complete
        )

        assert started == ["B"]  # A was skipped via checkpoint
        assert completed == ["B"]

    def test_on_stage_complete_receives_timing_info(self, temp_project_dir, mock_config):
        """Test that on_stage_complete receives accurate timing information."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("slow_stage", run_delay=0.05))

        timing_info = {}

        def on_complete(stage_name, result, elapsed):
            timing_info[stage_name] = elapsed

        pipeline.run(resume=False, on_stage_complete=on_complete)

        assert "slow_stage" in timing_info
        assert timing_info["slow_stage"] >= 0.05  # Should be at least 50ms

    def test_only_on_stage_start_callback(self, temp_project_dir, mock_config):
        """Test running with only on_stage_start callback."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A"))

        started = []

        def on_start(stage_name):
            started.append(stage_name)

        result = pipeline.run(resume=False, on_stage_start=on_start)

        assert result is True
        assert started == ["A"]

    def test_only_on_stage_complete_callback(self, temp_project_dir, mock_config):
        """Test running with only on_stage_complete callback."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A"))

        completed = []

        def on_complete(stage_name, result, elapsed):
            completed.append(stage_name)

        result = pipeline.run(resume=False, on_stage_complete=on_complete)

        assert result is True
        assert completed == ["A"]

    def test_no_callbacks_provided(self, temp_project_dir, mock_config):
        """Test that pipeline runs correctly when no callbacks are provided."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A")).add_stage(MockStage("B"))

        result = pipeline.run(resume=False)

        assert result is True
        assert "A" in pipeline.stage_timings
        assert "B" in pipeline.stage_timings
