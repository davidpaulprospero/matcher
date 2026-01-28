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
from src.stages import Stage, StageResult, StageMetrics
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
        run_delay: float = 0.0,
        items_processed: int = 0,
        items_failed: int = 0
    ):
        self.name = name
        self._should_fail = should_fail
        self._validation_error = validation_error
        self._can_skip_value = can_skip_value
        self._restore_success = restore_success
        self._run_delay = run_delay
        self._items_processed = items_processed
        self._items_failed = items_failed
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

        # Return metrics if items were specified
        metrics = None
        if self._items_processed > 0 or self._items_failed > 0:
            metrics = StageMetrics(
                items_processed=self._items_processed,
                items_failed=self._items_failed
            )

        return StageResult.ok(
            data={'stage': self.name, 'completed': True},
            metrics=metrics
        )


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


@pytest.mark.fast
class TestAddStageFluent:
    """Test add_stage() fluent interface returns self."""

    def test_add_stage_returns_self(self, temp_project_dir, mock_config):
        """Test that add_stage() returns the PipelineOrchestrator instance."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage = MockStage("test_stage")

        result = pipeline.add_stage(stage)

        assert result is pipeline, "add_stage() should return self"

    @pytest.mark.fast
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

    @pytest.mark.fast
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


@pytest.mark.fast
class TestLoadCheckpoint:
    """Test load_checkpoint() with valid and invalid checkpoints."""

    def test_load_checkpoint_no_file(self, temp_project_dir, mock_config):
        """Test load_checkpoint() returns False when no checkpoint exists."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        result = pipeline.load_checkpoint()

        assert result is False
        assert pipeline.resume_mode is False

    @pytest.mark.fast
    def test_load_checkpoint_returns_none(self, temp_project_dir, mock_config):
        """Test load_checkpoint() returns False when load() returns None."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        # Mock checkpoint to exist but return None on load
        pipeline.checkpoint.exists = Mock(return_value=True)
        pipeline.checkpoint.load = Mock(return_value=None)

        result = pipeline.load_checkpoint()

        assert result is False
        assert pipeline.resume_mode is False

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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


@pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_skip_stages_not_tracked_in_timings(self, temp_project_dir, mock_config):
        """Test that skipped stages don't appear in stage_timings."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        for s in [MockStage("A"), MockStage("B"), MockStage("C")]:
            pipeline.add_stage(s)

        pipeline.run(resume=False, skip_stages=["B"])

        assert "A" in pipeline.stage_timings
        assert "B" not in pipeline.stage_timings
        assert "C" in pipeline.stage_timings


@pytest.mark.fast
class TestRunResumeWithCheckpoint:
    """Test run() resume behavior with checkpoint."""

    def test_resume_triggers_load_checkpoint(self, temp_project_dir, mock_config):
        """Test that resume=True triggers load_checkpoint()."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        with patch.object(pipeline, 'load_checkpoint') as mock_load:
            mock_load.return_value = False
            pipeline.run(resume=True)

        mock_load.assert_called_once()

    @pytest.mark.fast
    def test_resume_false_skips_load_checkpoint(self, temp_project_dir, mock_config):
        """Test that resume=False does not trigger load_checkpoint()."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        with patch.object(pipeline, 'load_checkpoint') as mock_load:
            pipeline.run(resume=False)

        mock_load.assert_not_called()

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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


@pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_timing_reflects_actual_duration(self, temp_project_dir, mock_config):
        """Test timing reflects actual stage execution time."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("slow_stage", run_delay=0.05))

        pipeline.run(resume=False)

        # Stage with 50ms delay should have timing >= 50ms
        assert pipeline.stage_timings["slow_stage"] >= 0.05

    @pytest.mark.fast
    def test_timing_synced_with_state(self, temp_project_dir, mock_config):
        """Test that stage_timings is synced to state.stage_timings."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("synced_stage"))

        pipeline.run(resume=False)

        assert pipeline.stage_timings["synced_stage"] == pipeline.state.stage_timings["synced_stage"]

    @pytest.mark.fast
    def test_timing_not_recorded_for_failed_stage(self, temp_project_dir, mock_config):
        """Test that timing IS recorded even for failed stages (before failure)."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("fail_stage", should_fail=True))

        with patch('src.pipeline.logger'):
            pipeline.run(resume=False)

        # Timing should still be recorded since stage ran (just failed)
        assert "fail_stage" in pipeline.stage_timings

    @pytest.mark.fast
    def test_timing_not_recorded_for_validation_failure(self, temp_project_dir, mock_config):
        """Test that timing is NOT recorded when validation fails."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("invalid_stage", validation_error="Missing input"))

        with patch('src.pipeline.logger'):
            pipeline.run(resume=False)

        # Timing should NOT be recorded since stage never ran
        assert "invalid_stage" not in pipeline.stage_timings


@pytest.mark.fast
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

    @pytest.mark.fast
    def test_checkpoint_not_saved_on_failure(self, temp_project_dir, mock_config):
        """Test checkpoint.save() is NOT called when stage fails."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("fail_save", should_fail=True))
        pipeline.checkpoint.save = Mock()

        with patch('src.pipeline.logger'):
            pipeline.run(resume=False)

        pipeline.checkpoint.save.assert_not_called()


@pytest.mark.fast
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_pipeline(self, temp_project_dir, mock_config):
        """Test running an empty pipeline succeeds."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        result = pipeline.run(resume=False)

        assert result is True
        assert pipeline.stage_timings == {}

    @pytest.mark.fast
    def test_current_stage_cleared_after_run(self, temp_project_dir, mock_config):
        """Test current_stage is None after pipeline completes."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("test"))

        pipeline.run(resume=False)

        assert pipeline.current_stage is None

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_clear_checkpoint(self, temp_project_dir, mock_config):
        """Test clear_checkpoint() calls checkpoint.clear() and resets resume_mode."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.resume_mode = True
        pipeline.checkpoint.clear = Mock()

        pipeline.clear_checkpoint()

        pipeline.checkpoint.clear.assert_called_once()
        assert pipeline.resume_mode is False

    @pytest.mark.fast
    def test_get_summary_empty(self, temp_project_dir, mock_config):
        """Test get_summary() for empty pipeline."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        summary = pipeline.get_summary()

        assert summary['stages_run'] == []
        assert summary['total_time'] == 0
        assert 'state' in summary

    @pytest.mark.fast
    def test_get_summary_after_run(self, temp_project_dir, mock_config):
        """Test get_summary() after running pipeline."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A")).add_stage(MockStage("B"))

        pipeline.run(resume=False)
        summary = pipeline.get_summary()

        assert summary['stages_run'] == ["A", "B"]
        assert summary['total_time'] >= 0
        assert len(summary['stage_timings']) == 2


@pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_no_callbacks_provided(self, temp_project_dir, mock_config):
        """Test that pipeline runs correctly when no callbacks are provided."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A")).add_stage(MockStage("B"))

        result = pipeline.run(resume=False)

        assert result is True
        assert "A" in pipeline.stage_timings
        assert "B" in pipeline.stage_timings


@pytest.mark.fast
class TestStageMetricsCollection:
    """Test stage metrics collection and aggregation."""

    def test_metrics_collected_for_single_stage(self, temp_project_dir, mock_config):
        """Test that metrics are collected for a single stage."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("test_stage", items_processed=10, items_failed=2))

        pipeline.run(resume=False)

        assert "test_stage" in pipeline.stage_metrics
        metrics = pipeline.stage_metrics["test_stage"]
        assert metrics.items_processed == 10
        assert metrics.items_failed == 2
        assert metrics.duration_seconds >= 0

    @pytest.mark.fast
    def test_metrics_collected_for_multiple_stages(self, temp_project_dir, mock_config):
        """Test that metrics are collected for multiple stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A", items_processed=5, items_failed=1))
        pipeline.add_stage(MockStage("B", items_processed=10, items_failed=3))
        pipeline.add_stage(MockStage("C", items_processed=15, items_failed=0))

        pipeline.run(resume=False)

        assert len(pipeline.stage_metrics) == 3
        assert pipeline.stage_metrics["A"].items_processed == 5
        assert pipeline.stage_metrics["B"].items_processed == 10
        assert pipeline.stage_metrics["C"].items_processed == 15

    @pytest.mark.fast
    def test_default_metrics_when_stage_returns_none(self, temp_project_dir, mock_config):
        """Test that default metrics are created when stage returns no metrics."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        # Stage without items_processed/items_failed returns no metrics
        pipeline.add_stage(MockStage("no_metrics_stage"))

        pipeline.run(resume=False)

        assert "no_metrics_stage" in pipeline.stage_metrics
        metrics = pipeline.stage_metrics["no_metrics_stage"]
        assert metrics.items_processed == 0
        assert metrics.items_failed == 0
        assert metrics.duration_seconds >= 0

    @pytest.mark.fast
    def test_get_metrics_returns_aggregated_totals(self, temp_project_dir, mock_config):
        """Test that get_metrics() returns aggregated totals."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A", items_processed=5, items_failed=1))
        pipeline.add_stage(MockStage("B", items_processed=10, items_failed=2))
        pipeline.add_stage(MockStage("C", items_processed=15, items_failed=0))

        pipeline.run(resume=False)

        metrics = pipeline.get_metrics()

        assert metrics['total_items_processed'] == 30  # 5 + 10 + 15
        assert metrics['total_items_failed'] == 3      # 1 + 2 + 0
        assert metrics['total_duration_seconds'] >= 0
        assert 'stages' in metrics
        assert len(metrics['stages']) == 3

    @pytest.mark.fast
    def test_get_metrics_empty_pipeline(self, temp_project_dir, mock_config):
        """Test get_metrics() for empty pipeline."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)

        pipeline.run(resume=False)

        metrics = pipeline.get_metrics()

        assert metrics['total_items_processed'] == 0
        assert metrics['total_items_failed'] == 0
        assert metrics['total_duration_seconds'] == 0.0
        assert metrics['stages'] == {}

    @pytest.mark.fast
    def test_metrics_duration_updated_from_actual_elapsed(self, temp_project_dir, mock_config):
        """Test that duration_seconds is updated to actual elapsed time."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("slow_stage", items_processed=10, run_delay=0.05))

        pipeline.run(resume=False)

        metrics = pipeline.stage_metrics["slow_stage"]
        # Duration should be at least 50ms
        assert metrics.duration_seconds >= 0.05

    @pytest.mark.fast
    def test_get_metrics_includes_stage_dict(self, temp_project_dir, mock_config):
        """Test that get_metrics() includes the stages dict."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A", items_processed=5, items_failed=1))

        pipeline.run(resume=False)

        metrics = pipeline.get_metrics()

        assert 'stages' in metrics
        assert "A" in metrics['stages']
        assert isinstance(metrics['stages']["A"], StageMetrics)
        assert metrics['stages']["A"].items_processed == 5

    @pytest.mark.fast
    def test_metrics_not_collected_for_skipped_stages(self, temp_project_dir, mock_config):
        """Test that metrics are not collected for skipped stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A", items_processed=5))
        pipeline.add_stage(MockStage("B", items_processed=10))
        pipeline.add_stage(MockStage("C", items_processed=15))

        pipeline.run(resume=False, skip_stages=["B"])

        assert "A" in pipeline.stage_metrics
        assert "B" not in pipeline.stage_metrics
        assert "C" in pipeline.stage_metrics

    @pytest.mark.fast
    def test_metrics_collected_for_failed_stage(self, temp_project_dir, mock_config):
        """Test that metrics ARE collected for failed stages (before failure)."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("fail_stage", should_fail=True))

        with patch('src.pipeline.logger'):
            pipeline.run(resume=False)

        # Failed stages still get recorded in stage_metrics
        assert "fail_stage" in pipeline.stage_metrics
        # Default metrics since MockStage doesn't return metrics when failing
        assert pipeline.stage_metrics["fail_stage"].items_processed == 0

    @pytest.mark.fast
    def test_stage_metrics_dataclass_to_dict(self, temp_project_dir, mock_config):
        """Test StageMetrics.to_dict() method."""
        metrics = StageMetrics(items_processed=10, items_failed=2, duration_seconds=5.5)

        result = metrics.to_dict()

        assert result == {
            'items_processed': 10,
            'items_failed': 2,
            'duration_seconds': 5.5
        }

    @pytest.mark.fast
    def test_stage_metrics_dataclass_from_dict(self, temp_project_dir, mock_config):
        """Test StageMetrics.from_dict() classmethod."""
        data = {
            'items_processed': 15,
            'items_failed': 3,
            'duration_seconds': 7.2
        }

        metrics = StageMetrics.from_dict(data)

        assert metrics.items_processed == 15
        assert metrics.items_failed == 3
        assert metrics.duration_seconds == 7.2

    @pytest.mark.fast
    def test_stage_result_ok_with_metrics(self, temp_project_dir, mock_config):
        """Test StageResult.ok() with metrics parameter."""
        metrics = StageMetrics(items_processed=5, items_failed=1)
        result = StageResult.ok(data={'test': 1}, metrics=metrics)

        assert result.success is True
        assert result.metrics is metrics
        assert result.metrics.items_processed == 5

    @pytest.mark.fast
    def test_stage_result_fail_with_metrics(self, temp_project_dir, mock_config):
        """Test StageResult.fail() with metrics parameter."""
        metrics = StageMetrics(items_processed=3, items_failed=2)
        result = StageResult.fail(error="Test error", metrics=metrics)

        assert result.success is False
        assert result.metrics is metrics
        assert result.metrics.items_failed == 2


@pytest.mark.fast
class TestResumeLogic:
    """
    US-007 (Sprint 15): Pipeline orchestrator resume logic tests.

    Tests specifically targeting acceptance criteria for resume behavior:
    - Resume skips stages before checkpoint's last_completed_stage
    - Fresh run executes all stages in order
    - Halts on stage failure (subsequent stages NOT called)
    - Stage timing recorded for each completed stage
    - Stage callbacks fire with correct stage names
    """

    @pytest.mark.fast
    def test_resume_skips_stages_before_checkpoint_3_stages(self, temp_project_dir, mock_config):
        """AC1: Mock 3 stages, set checkpoint to stage 1, verify only stages 2 and 3 execute."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage1 = MockStage("ANALYZE", can_skip_value=True)
        stage2 = MockStage("DOWNLOAD", can_skip_value=False)
        stage3 = MockStage("OUTPUT", can_skip_value=False)
        pipeline.add_stage(stage1).add_stage(stage2).add_stage(stage3)

        # Simulate checkpoint resume: stage1 already completed
        pipeline.resume_mode = True

        result = pipeline.run(resume=False)

        assert result is True
        # Stage 1 skipped (restored from checkpoint), stages 2 and 3 executed
        assert stage1._run_called is False
        assert stage1._restore_called is True
        assert stage2._run_called is True
        assert stage3._run_called is True

    @pytest.mark.fast
    def test_resume_skips_first_two_of_three(self, temp_project_dir, mock_config):
        """AC1 variant: Both first stages completed, only last executes."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage1 = MockStage("ANALYZE", can_skip_value=True)
        stage2 = MockStage("DOWNLOAD", can_skip_value=True)
        stage3 = MockStage("OUTPUT", can_skip_value=False)
        pipeline.add_stage(stage1).add_stage(stage2).add_stage(stage3)

        pipeline.resume_mode = True
        result = pipeline.run(resume=False)

        assert result is True
        assert stage1._run_called is False
        assert stage2._run_called is False
        assert stage3._run_called is True

    @pytest.mark.fast
    def test_fresh_run_executes_all_stages_in_order(self, temp_project_dir, mock_config):
        """AC2: No checkpoint exists — verify all stage execute() methods called in order."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage1 = MockStage("ANALYZE")
        stage2 = MockStage("DOWNLOAD")
        stage3 = MockStage("OUTPUT")
        pipeline.add_stage(stage1).add_stage(stage2).add_stage(stage3)

        execution_order = []

        def on_start(name):
            execution_order.append(name)

        result = pipeline.run(resume=False, on_stage_start=on_start)

        assert result is True
        # All three stages executed
        assert stage1._run_called is True
        assert stage2._run_called is True
        assert stage3._run_called is True
        # Correct order
        assert execution_order == ["ANALYZE", "DOWNLOAD", "OUTPUT"]

    @pytest.mark.fast
    def test_fresh_run_no_can_skip_called(self, temp_project_dir, mock_config):
        """AC2: Fresh run does not invoke can_skip on any stage."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage1 = MockStage("ANALYZE")
        stage2 = MockStage("DOWNLOAD")
        pipeline.add_stage(stage1).add_stage(stage2)

        # resume_mode is False (fresh), so can_skip should never be checked
        result = pipeline.run(resume=False)

        assert result is True
        assert stage1._can_skip_called is False
        assert stage2._can_skip_called is False

    @pytest.mark.fast
    def test_halts_on_failure_subsequent_not_called(self, temp_project_dir, mock_config):
        """AC3: Stage failure halts pipeline — subsequent stages NOT called."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        stage1 = MockStage("ANALYZE")
        stage2 = MockStage("DOWNLOAD", should_fail=True)
        stage3 = MockStage("OUTPUT")
        pipeline.add_stage(stage1).add_stage(stage2).add_stage(stage3)

        with patch('src.pipeline.logger'):
            result = pipeline.run(resume=False)

        assert result is False
        assert stage1._run_called is True
        assert stage2._run_called is True   # Ran but failed
        assert stage3._run_called is False   # Never reached

    @pytest.mark.fast
    def test_halts_on_failure_returns_false(self, temp_project_dir, mock_config):
        """AC3: Pipeline returns False when a stage fails."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("FAIL", should_fail=True))

        with patch('src.pipeline.logger'):
            result = pipeline.run(resume=False)

        assert result is False

    @pytest.mark.fast
    def test_timing_recorded_per_completed_stage(self, temp_project_dir, mock_config):
        """AC4: elapsed_seconds populated for each completed stage after run."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("ANALYZE"))
        pipeline.add_stage(MockStage("DOWNLOAD"))
        pipeline.add_stage(MockStage("OUTPUT"))

        pipeline.run(resume=False)

        # All 3 stages have timing entries
        assert len(pipeline.stage_timings) == 3
        for name in ["ANALYZE", "DOWNLOAD", "OUTPUT"]:
            assert name in pipeline.stage_timings
            assert isinstance(pipeline.stage_timings[name], float)
            assert pipeline.stage_timings[name] >= 0

    @pytest.mark.fast
    def test_timing_not_recorded_for_stages_after_failure(self, temp_project_dir, mock_config):
        """AC4: Stages after failure have no timing entry."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("ANALYZE"))
        pipeline.add_stage(MockStage("DOWNLOAD", should_fail=True))
        pipeline.add_stage(MockStage("OUTPUT"))

        with patch('src.pipeline.logger'):
            pipeline.run(resume=False)

        assert "ANALYZE" in pipeline.stage_timings
        assert "DOWNLOAD" in pipeline.stage_timings  # Ran but failed — still timed
        assert "OUTPUT" not in pipeline.stage_timings  # Never ran

    @pytest.mark.fast
    def test_callbacks_fire_with_correct_stage_names(self, temp_project_dir, mock_config):
        """AC5: on_stage_start and on_stage_complete called with correct stage name."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("ANALYZE"))
        pipeline.add_stage(MockStage("DOWNLOAD"))
        pipeline.add_stage(MockStage("OUTPUT"))

        started = []
        completed = []

        def on_start(name):
            started.append(name)

        def on_complete(name, result, elapsed):
            completed.append(name)

        pipeline.run(resume=False, on_stage_start=on_start, on_stage_complete=on_complete)

        assert started == ["ANALYZE", "DOWNLOAD", "OUTPUT"]
        assert completed == ["ANALYZE", "DOWNLOAD", "OUTPUT"]

    @pytest.mark.fast
    def test_callbacks_interleave_start_complete(self, temp_project_dir, mock_config):
        """AC5: start fires before complete for each stage, interleaved correctly."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("A"))
        pipeline.add_stage(MockStage("B"))

        events = []

        def on_start(name):
            events.append(f"start:{name}")

        def on_complete(name, result, elapsed):
            events.append(f"complete:{name}")

        pipeline.run(resume=False, on_stage_start=on_start, on_stage_complete=on_complete)

        assert events == ["start:A", "complete:A", "start:B", "complete:B"]

    @pytest.mark.fast
    def test_callbacks_receive_success_result(self, temp_project_dir, mock_config):
        """AC5: on_stage_complete receives StageResult with success=True for passing stages."""
        pipeline = PipelineOrchestrator(mock_config, temp_project_dir)
        pipeline.add_stage(MockStage("ANALYZE"))

        results = []

        def on_complete(name, result, elapsed):
            results.append({'name': name, 'success': result.success, 'elapsed': elapsed})

        pipeline.run(resume=False, on_stage_complete=on_complete)

        assert len(results) == 1
        assert results[0]['name'] == "ANALYZE"
        assert results[0]['success'] is True
        assert results[0]['elapsed'] >= 0
