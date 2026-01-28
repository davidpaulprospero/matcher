"""
Tests for pipeline orchestration module.

Covers:
- PipelineOrchestrator initialization
- Stage execution flow
- Checkpoint integration
- Resume functionality
- Stage filtering (skip_stages, only_stages)
- Error handling and validation
- Timing tracking
- Factory functions (create_default_pipeline, create_match_only_pipeline)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    PipelineOrchestrator,
    create_default_pipeline,
    create_match_only_pipeline
)
from src.state import PipelineState
from src.stages import Stage, StageResult
from src.config import Config


class MockStage(Stage):
    """Mock stage for testing pipeline orchestration."""

    def __init__(self, name: str, should_fail: bool = False, validation_error: str = None):
        self.name = name
        self._should_fail = should_fail
        self._validation_error = validation_error
        self._run_called = False
        self._restore_called = False

    def can_skip(self, state: PipelineState, checkpoint) -> bool:
        """Default: can skip if checkpoint has data for this stage"""
        return checkpoint.get(self.name) is not None

    def restore(self, state: PipelineState, checkpoint, config=None) -> bool:
        """Restore from checkpoint"""
        self._restore_called = True
        data = checkpoint.get(self.name)
        if data:
            # Restore some test data
            state.voiceover_segments = data.get('segments', [])
            return True
        return False

    def validate_inputs(self, state: PipelineState, config: Config) -> str:
        """Validate inputs before running"""
        return self._validation_error

    def run(self, state: PipelineState, config: Config, checkpoint) -> StageResult:
        """Execute the stage"""
        self._run_called = True

        if self._should_fail:
            return StageResult(
                success=False,
                error=f"{self.name} failed intentionally",
                warnings=["Warning 1", "Warning 2"]
            )

        # Modify state to simulate work
        state.voiceover_segments.append(f"data_from_{self.name}")

        # Return checkpoint data
        return StageResult(
            success=True,
            data={
                'stage': self.name,
                'segments': state.voiceover_segments.copy()
            },
            warnings=[]
        )


class TestPipelineOrchestratorInit:
    """Test PipelineOrchestrator initialization."""

    @pytest.mark.fast
    def test_init_with_minimal_args(self, temp_dir):
        """Test initialization with minimal arguments."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        assert pipeline.config == config
        assert pipeline.project_dir == Path(temp_dir)
        assert isinstance(pipeline.state, PipelineState)
        assert pipeline.stages == []
        assert pipeline.checkpoint is not None
        assert pipeline.resume_mode is False
        assert pipeline.current_stage is None
        assert pipeline.stage_timings == {}

    @pytest.mark.fast
    def test_init_with_stages(self, temp_dir):
        """Test initialization with pre-defined stages."""
        config = Config()
        stages = [MockStage("stage1"), MockStage("stage2")]
        pipeline = PipelineOrchestrator(config, temp_dir, stages=stages)

        assert len(pipeline.stages) == 2
        assert pipeline.stages[0].name == "stage1"
        assert pipeline.stages[1].name == "stage2"

    @pytest.mark.fast
    def test_add_stage_fluent_interface(self, temp_dir):
        """Test adding stages using fluent interface."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        result = pipeline.add_stage(MockStage("stage1"))
        assert result == pipeline  # Fluent interface returns self

        pipeline.add_stage(MockStage("stage2")).add_stage(MockStage("stage3"))
        assert len(pipeline.stages) == 3


class TestPipelineCheckpoint:
    """Test checkpoint loading and validation."""

    @pytest.mark.fast
    def test_load_checkpoint_not_exists(self, temp_dir):
        """Test loading checkpoint when it doesn't exist."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        result = pipeline.load_checkpoint()

        assert result is False
        assert pipeline.resume_mode is False

    @pytest.mark.fast
    def test_load_checkpoint_invalid(self, temp_dir):
        """Test loading checkpoint when validation fails."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        # Mock checkpoint methods
        pipeline.checkpoint.exists = Mock(return_value=True)
        pipeline.checkpoint.load = Mock(return_value={'test': 'data'})
        pipeline.checkpoint.validate = Mock(return_value={
            'valid': False,
            'errors': ['Invalid checkpoint format'],
            'warnings': []
        })

        with patch('src.pipeline.logger') as mock_logger:
            result = pipeline.load_checkpoint()

            assert result is False
            assert pipeline.resume_mode is False
            mock_logger.error.assert_called_once()

    @pytest.mark.fast
    def test_load_checkpoint_valid_with_warnings(self, temp_dir):
        """Test loading valid checkpoint with warnings."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        # Mock checkpoint methods
        pipeline.checkpoint.exists = Mock(return_value=True)
        pipeline.checkpoint.load = Mock(return_value={'test': 'data'})
        pipeline.checkpoint.validate = Mock(return_value={
            'valid': True,
            'errors': [],
            'warnings': ['Config hash mismatch']
        })

        with patch('src.pipeline.logger') as mock_logger:
            result = pipeline.load_checkpoint()

            assert result is True
            assert pipeline.resume_mode is True
            mock_logger.warning.assert_called_once()

    @pytest.mark.fast
    def test_clear_checkpoint(self, temp_dir):
        """Test clearing checkpoint."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)
        pipeline.resume_mode = True

        pipeline.checkpoint.clear = Mock()
        pipeline.clear_checkpoint()

        pipeline.checkpoint.clear.assert_called_once()
        assert pipeline.resume_mode is False


class TestPipelineRun:
    """Test pipeline execution logic."""

    @pytest.mark.fast
    def test_run_empty_pipeline(self, temp_dir):
        """Test running pipeline with no stages."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        result = pipeline.run(resume=False)

        assert result is True
        assert pipeline.stage_timings == {}

    @pytest.mark.fast
    def test_run_single_stage_success(self, temp_dir):
        """Test running pipeline with one successful stage."""
        config = Config()
        stage = MockStage("test_stage")
        pipeline = PipelineOrchestrator(config, temp_dir, stages=[stage])

        result = pipeline.run(resume=False)

        assert result is True
        assert stage._run_called is True
        assert "test_stage" in pipeline.stage_timings
        assert pipeline.stage_timings["test_stage"] >= 0  # Can be 0.0 for very fast execution
        assert pipeline.current_stage is None  # Cleared after run

    @pytest.mark.fast
    def test_run_multiple_stages_success(self, temp_dir):
        """Test running pipeline with multiple successful stages."""
        config = Config()
        stages = [
            MockStage("stage1"),
            MockStage("stage2"),
            MockStage("stage3")
        ]
        pipeline = PipelineOrchestrator(config, temp_dir, stages=stages)

        result = pipeline.run(resume=False)

        assert result is True
        assert all(s._run_called for s in stages)
        assert len(pipeline.stage_timings) == 3
        assert "stage1" in pipeline.stage_timings
        assert "stage2" in pipeline.stage_timings
        assert "stage3" in pipeline.stage_timings

    @pytest.mark.fast
    def test_run_stage_failure(self, temp_dir):
        """Test pipeline stops on stage failure."""
        config = Config()
        stages = [
            MockStage("stage1"),
            MockStage("stage2", should_fail=True),
            MockStage("stage3")
        ]
        pipeline = PipelineOrchestrator(config, temp_dir, stages=stages)

        with patch('src.pipeline.logger') as mock_logger:
            result = pipeline.run(resume=False)

            assert result is False
            assert stages[0]._run_called is True
            assert stages[1]._run_called is True
            assert stages[2]._run_called is False  # Should not run
            mock_logger.error.assert_called()
            mock_logger.warning.assert_called()  # Warnings from failed stage

    @pytest.mark.fast
    def test_run_validation_failure(self, temp_dir):
        """Test pipeline stops if stage validation fails."""
        config = Config()
        stages = [
            MockStage("stage1"),
            MockStage("stage2", validation_error="Missing required input"),
            MockStage("stage3")
        ]
        pipeline = PipelineOrchestrator(config, temp_dir, stages=stages)

        with patch('src.pipeline.logger') as mock_logger:
            result = pipeline.run(resume=False)

            assert result is False
            assert stages[0]._run_called is True
            assert stages[1]._run_called is False  # Validation failed before run
            assert stages[2]._run_called is False
            mock_logger.error.assert_called_with(
                "Stage stage2 validation failed: Missing required input"
            )


class TestPipelineStageFiltering:
    """Test stage filtering with skip_stages and only_stages."""

    @pytest.mark.fast
    def test_skip_stages(self, temp_dir):
        """Test skipping specific stages."""
        config = Config()
        stages = [
            MockStage("stage1"),
            MockStage("stage2"),
            MockStage("stage3")
        ]
        pipeline = PipelineOrchestrator(config, temp_dir, stages=stages)

        result = pipeline.run(resume=False, skip_stages=["stage2"])

        assert result is True
        assert stages[0]._run_called is True
        assert stages[1]._run_called is False  # Skipped
        assert stages[2]._run_called is True
        assert "stage2" not in pipeline.stage_timings

    @pytest.mark.fast
    def test_only_stages(self, temp_dir):
        """Test running only specific stages."""
        config = Config()
        stages = [
            MockStage("stage1"),
            MockStage("stage2"),
            MockStage("stage3")
        ]
        pipeline = PipelineOrchestrator(config, temp_dir, stages=stages)

        result = pipeline.run(resume=False, only_stages=["stage1", "stage3"])

        assert result is True
        assert stages[0]._run_called is True
        assert stages[1]._run_called is False  # Not in only_stages
        assert stages[2]._run_called is True
        assert "stage2" not in pipeline.stage_timings

    @pytest.mark.fast
    def test_skip_stages_and_only_stages(self, temp_dir):
        """Test combining skip_stages and only_stages."""
        config = Config()
        stages = [
            MockStage("stage1"),
            MockStage("stage2"),
            MockStage("stage3"),
            MockStage("stage4")
        ]
        pipeline = PipelineOrchestrator(config, temp_dir, stages=stages)

        result = pipeline.run(
            resume=False,
            skip_stages=["stage2"],
            only_stages=["stage1", "stage2", "stage3"]
        )

        assert result is True
        assert stages[0]._run_called is True
        assert stages[1]._run_called is False  # Skipped via skip_stages
        assert stages[2]._run_called is True
        assert stages[3]._run_called is False  # Not in only_stages


class TestPipelineResume:
    """Test resume functionality with checkpoint."""

    @pytest.mark.fast
    def test_resume_skip_completed_stage(self, temp_dir):
        """Test resuming skips stages already in checkpoint."""
        config = Config()
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")
        pipeline = PipelineOrchestrator(config, temp_dir, stages=[stage1, stage2])

        # Mock checkpoint to have stage1 data
        pipeline.checkpoint.get = Mock(side_effect=lambda name:
            {'segments': ['stage1_data']} if name == "stage1" else None
        )
        pipeline.resume_mode = True

        result = pipeline.run(resume=False)  # resume=False because we set resume_mode manually

        assert result is True
        assert stage1._run_called is False  # Skipped due to checkpoint
        assert stage1._restore_called is True  # Restored from checkpoint
        assert stage2._run_called is True  # Should run

    @pytest.mark.fast
    def test_resume_restore_failure(self, temp_dir):
        """Test handling restore failure during resume."""
        config = Config()
        stage1 = MockStage("stage1")
        stage1.restore = Mock(return_value=False)  # Simulate restore failure
        pipeline = PipelineOrchestrator(config, temp_dir, stages=[stage1])

        pipeline.checkpoint.get = Mock(return_value={'test': 'data'})
        pipeline.resume_mode = True

        with patch('src.pipeline.logger') as mock_logger:
            result = pipeline.run(resume=False)

            # Should continue despite restore failure
            assert result is True
            mock_logger.warning.assert_called()


class TestPipelineSummary:
    """Test pipeline summary generation."""

    @pytest.mark.fast
    def test_get_summary_empty_pipeline(self, temp_dir):
        """Test summary for pipeline that hasn't run."""
        config = Config()
        pipeline = PipelineOrchestrator(config, temp_dir)

        summary = pipeline.get_summary()

        assert summary['stages_run'] == []
        assert summary['total_time'] == 0
        assert summary['stage_timings'] == {}
        assert 'state' in summary

    @pytest.mark.fast
    def test_get_summary_after_run(self, temp_dir):
        """Test summary after running pipeline."""
        config = Config()
        stages = [MockStage("stage1"), MockStage("stage2")]
        pipeline = PipelineOrchestrator(config, temp_dir, stages=stages)

        pipeline.run(resume=False)
        summary = pipeline.get_summary()

        assert summary['stages_run'] == ["stage1", "stage2"]
        assert summary['total_time'] >= 0  # Can be 0.0 for very fast execution
        assert len(summary['stage_timings']) == 2
        assert summary['stage_timings']['stage1'] >= 0
        assert summary['stage_timings']['stage2'] >= 0


class TestPipelineFactories:
    """Test factory functions for creating pipelines."""

    @pytest.mark.fast
    def test_create_default_pipeline(self, temp_dir):
        """Test creating default pipeline."""
        config = Config()

        pipeline = create_default_pipeline(config, temp_dir, audio_first_mode=False)

        assert isinstance(pipeline, PipelineOrchestrator)
        assert len(pipeline.stages) > 0
        # Should have standard stages: ANALYZE, DOWNLOAD, TRANSCRIBE, MATCH, OUTPUT
        stage_names = [s.name for s in pipeline.stages]
        assert "ANALYZE" in stage_names
        assert "DOWNLOAD" in stage_names
        assert "TRANSCRIBE" in stage_names
        assert "MATCH" in stage_names
        assert "OUTPUT" in stage_names

    @pytest.mark.fast
    def test_create_default_pipeline_audio_first(self, temp_dir):
        """Test creating default pipeline with audio-first mode."""
        config = Config()

        pipeline = create_default_pipeline(config, temp_dir, audio_first_mode=True)

        stage_names = [s.name for s in pipeline.stages]
        # Should have DOWNLOAD_SEGMENTS stage after MATCH
        assert "DOWNLOAD_SEGMENTS" in stage_names

        # DOWNLOAD_SEGMENTS should come after MATCH and before OUTPUT
        match_idx = stage_names.index("MATCH")
        download_segments_idx = stage_names.index("DOWNLOAD_SEGMENTS")
        output_idx = stage_names.index("OUTPUT")
        assert match_idx < download_segments_idx < output_idx

    @pytest.mark.fast
    def test_create_match_only_pipeline(self, temp_dir):
        """Test creating match-only pipeline."""
        config = Config()

        pipeline = create_match_only_pipeline(config, temp_dir)

        assert isinstance(pipeline, PipelineOrchestrator)
        stage_names = [s.name for s in pipeline.stages]

        # Should have prerequisite stages for restoration
        assert "ANALYZE" in stage_names
        assert "DOWNLOAD" in stage_names
        assert "TRANSCRIBE" in stage_names

        # Should have MATCH and OUTPUT for actual execution
        assert "MATCH" in stage_names
        assert "OUTPUT" in stage_names

        # Should NOT have DOWNLOAD_SEGMENTS (audio-first only)
        assert "DOWNLOAD_SEGMENTS" not in stage_names


class TestPipelineCheckpointSaving:
    """Test checkpoint saving during pipeline execution."""

    @pytest.mark.fast
    def test_checkpoint_saved_after_stage(self, temp_dir):
        """Test that checkpoint is saved after each successful stage."""
        config = Config()
        stage = MockStage("test_stage")
        pipeline = PipelineOrchestrator(config, temp_dir, stages=[stage])

        pipeline.checkpoint.save = Mock()

        pipeline.run(resume=False)

        # Should save checkpoint with stage data
        pipeline.checkpoint.save.assert_called_once()
        call_args = pipeline.checkpoint.save.call_args
        assert call_args[0][0] == "test_stage"
        assert call_args[0][1]['stage'] == "test_stage"
        assert 'segments' in call_args[0][1]

    @pytest.mark.fast
    def test_checkpoint_not_saved_on_failure(self, temp_dir):
        """Test that checkpoint is not saved if stage fails."""
        config = Config()
        stage = MockStage("test_stage", should_fail=True)
        pipeline = PipelineOrchestrator(config, temp_dir, stages=[stage])

        pipeline.checkpoint.save = Mock()

        with patch('src.pipeline.logger'):
            pipeline.run(resume=False)

        # Should not save checkpoint for failed stage
        pipeline.checkpoint.save.assert_not_called()


class TestPipelineTimingTracking:
    """Test stage timing tracking."""

    @pytest.mark.fast
    def test_timing_tracked_per_stage(self, temp_dir):
        """Test that timing is tracked for each stage."""
        config = Config()
        stages = [MockStage("stage1"), MockStage("stage2")]
        pipeline = PipelineOrchestrator(config, temp_dir, stages=stages)

        pipeline.run(resume=False)

        # Timings should be in both pipeline and state
        assert len(pipeline.stage_timings) == 2
        assert len(pipeline.state.stage_timings) == 2
        assert pipeline.stage_timings['stage1'] >= 0  # Can be 0.0 for very fast execution
        assert pipeline.stage_timings['stage2'] >= 0
        assert pipeline.state.stage_timings['stage1'] == pipeline.stage_timings['stage1']
        assert pipeline.state.stage_timings['stage2'] == pipeline.stage_timings['stage2']

    @pytest.mark.fast
    def test_timing_not_tracked_for_skipped_stages(self, temp_dir):
        """Test that timing is not tracked for skipped stages."""
        config = Config()
        stages = [MockStage("stage1"), MockStage("stage2")]
        pipeline = PipelineOrchestrator(config, temp_dir, stages=stages)

        pipeline.run(resume=False, skip_stages=["stage2"])

        assert "stage1" in pipeline.stage_timings
        assert "stage2" not in pipeline.stage_timings


# ============================================================================
# Coverage Tests - Lines 83, 119
# ============================================================================

class TestPipelineCoverage:
    """Test pipeline coverage gaps."""

    @pytest.mark.fast
    def test_load_checkpoint_returns_false_when_data_is_none(self):
        """Test line 83: Returns False when checkpoint.load() returns None."""
        from src.pipeline import PipelineOrchestrator
        from unittest.mock import Mock, MagicMock

        # Create a mock pipeline with checkpoint that returns None on load
        pipeline = MagicMock(spec=PipelineOrchestrator)
        pipeline.checkpoint = Mock()
        pipeline.checkpoint.exists.return_value = True
        pipeline.checkpoint.load.return_value = None  # Triggers line 83

        # Call the actual method
        result = PipelineOrchestrator.load_checkpoint(pipeline)

        assert result is False

    @pytest.mark.fast
    def test_run_calls_load_checkpoint_when_resume_true(self, temp_dir):
        """Test line 119: load_checkpoint() is called when resume=True."""
        from src.pipeline import PipelineOrchestrator
        from src.config import Config
        from unittest.mock import patch

        # Create a minimal config
        config = Config()
        config.project_dir = str(temp_dir)
        config.cache_dir = str(temp_dir / ".cache")

        # Create orchestrator with no stages
        pipeline = PipelineOrchestrator(config, temp_dir, stages=[])

        # Mock load_checkpoint to track calls
        with patch.object(pipeline, 'load_checkpoint') as mock_load:
            mock_load.return_value = False  # Simulate no checkpoint found
            pipeline.run(resume=True)

        # load_checkpoint should have been called
        mock_load.assert_called_once()


# Pytest fixtures

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)
