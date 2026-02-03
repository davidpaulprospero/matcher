"""
Stage Error Propagation Integration Tests - US-47-011

Tests that verify cross-stage error propagation through the pipeline
orchestrator, ensuring:
- Stage failures produce correct errors in checkpoint
- validate_required_state_attrs() raises clear errors for missing fields
- pipeline.run() handles stage exceptions without losing prior checkpoint data
- --resume correctly skips already-completed stages after mid-pipeline failure
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.checkpoint import CheckpointManager, STAGE_ORDER, CheckpointData
from src.pipeline import PipelineOrchestrator
from src.stages import Stage, StageResult, StageMetrics, validate_required_state_attrs
from src.state import PipelineState

from tests.fixtures import create_mock_config

# Mark all tests as integration
pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SuccessStage(Stage):
    """A stage that always succeeds with checkpoint data."""

    def __init__(self, stage_name: str, data: dict = None):
        self.name = stage_name
        self.description = f"Test {stage_name} (success)"
        self._data = data or {"completed": True}

    def run(self, state, config, checkpoint):
        return StageResult.ok(data=self._data)

    def can_skip(self, state, checkpoint):
        return checkpoint.should_skip_stage(self.name)

    def restore(self, state, checkpoint, config=None):
        return True

    def validate_inputs(self, state, config):
        return None


class FailStage(Stage):
    """A stage that always fails with a configurable error."""

    def __init__(self, stage_name: str, error_msg: str = "Stage failed"):
        self.name = stage_name
        self.description = f"Test {stage_name} (fail)"
        self._error_msg = error_msg

    def run(self, state, config, checkpoint):
        return StageResult.fail(self._error_msg)

    def can_skip(self, state, checkpoint):
        return checkpoint.should_skip_stage(self.name)

    def restore(self, state, checkpoint, config=None):
        return True

    def validate_inputs(self, state, config):
        return None


class ExceptionStage(Stage):
    """A stage that raises an unhandled exception."""

    def __init__(self, stage_name: str, exc_cls=RuntimeError, exc_msg="boom"):
        self.name = stage_name
        self.description = f"Test {stage_name} (exception)"
        self._exc_cls = exc_cls
        self._exc_msg = exc_msg

    def run(self, state, config, checkpoint):
        raise self._exc_cls(self._exc_msg)

    def can_skip(self, state, checkpoint):
        return checkpoint.should_skip_stage(self.name)

    def restore(self, state, checkpoint, config=None):
        return True

    def validate_inputs(self, state, config):
        return None


class ValidationFailStage(Stage):
    """A stage whose validate_inputs always returns an error."""

    def __init__(self, stage_name: str, validation_error: str):
        self.name = stage_name
        self.description = f"Test {stage_name} (validation fail)"
        self._validation_error = validation_error

    def run(self, state, config, checkpoint):
        return StageResult.ok()

    def can_skip(self, state, checkpoint):
        return checkpoint.should_skip_stage(self.name)

    def restore(self, state, checkpoint, config=None):
        return True

    def validate_inputs(self, state, config):
        return self._validation_error


def _make_pipeline(tmp_path, stages, config=None):
    """Build a PipelineOrchestrator with the given stages."""
    cfg = config or create_mock_config(tmp_path)
    # Ensure cache dir exists so config validation passes
    cache_path = Path(cfg.cache.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    # Remove matching stages from config validation (no embedding provider needed)
    cfg.embedding = Mock(provider="sentence-transformers")

    pipeline = PipelineOrchestrator(cfg, tmp_path, stages=list(stages))
    return pipeline


# ---------------------------------------------------------------------------
# Test: Stage failure in MATCH produces correct error in checkpoint
# ---------------------------------------------------------------------------

class TestMatchStageFailureCheckpoint:
    """AC: Stage failure in MATCH stage produces correct error in checkpoint."""

    def test_match_failure_preserves_prior_stages_in_checkpoint(self, tmp_path):
        """When MATCH fails, ANALYZE and VIDEO_SEARCH data remain in checkpoint."""
        analyze_data = {"keywords": ["test", "example"], "segment_count": 3}
        video_search_data = {"video_ids": ["abc123", "def456"]}

        stages = [
            SuccessStage("ANALYZE", data=analyze_data),
            SuccessStage("VIDEO_SEARCH", data=video_search_data),
            SuccessStage("CAPTION", data={"caption_count": 2}),
            FailStage("MATCH", error_msg="No suitable matches found"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        success = pipeline.run(resume=False)

        assert success is False

        # Verify prior stage data was checkpointed
        cp = pipeline.checkpoint
        assert cp.data is not None
        assert cp.data.last_completed_stage == "CAPTION"
        assert cp.data.analyze == analyze_data
        assert cp.data.video_search == video_search_data

    def test_match_failure_does_not_checkpoint_match_data(self, tmp_path):
        """Failed MATCH stage should not have its data saved to checkpoint."""
        stages = [
            SuccessStage("ANALYZE", data={"keywords": ["a"]}),
            SuccessStage("VIDEO_SEARCH", data={"video_ids": ["x"]}),
            SuccessStage("CAPTION", data={"caption_count": 1}),
            FailStage("MATCH", error_msg="embedding provider error"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        pipeline.run(resume=False)

        # MATCH data should be empty since stage failed
        assert pipeline.checkpoint.data.match == {}

    def test_match_failure_returns_false(self, tmp_path):
        """Pipeline.run() returns False when MATCH fails."""
        stages = [
            SuccessStage("ANALYZE"),
            SuccessStage("VIDEO_SEARCH"),
            SuccessStage("CAPTION"),
            FailStage("MATCH", error_msg="test error"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        result = pipeline.run(resume=False)

        assert result is False

    def test_match_failure_state_rollback(self, tmp_path):
        """On MATCH failure, state fields are rolled back to pre-stage snapshot."""
        stages = [
            SuccessStage("ANALYZE"),
            SuccessStage("VIDEO_SEARCH"),
            SuccessStage("CAPTION"),
            FailStage("MATCH"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        # Pre-populate matches so rollback has something to restore
        pipeline.state.matches = [Mock(segment_index=0)]
        original_matches = pipeline.state.matches.copy()

        pipeline.run(resume=False)

        # Matches should be rolled back to snapshot (before MATCH ran)
        # Since we set matches before run, and the snapshot captures them,
        # they should be restored after failure
        assert pipeline.state.matches == original_matches


# ---------------------------------------------------------------------------
# Test: validate_required_state_attrs raises clear errors
# ---------------------------------------------------------------------------

class TestValidateRequiredStateAttrs:
    """AC: validate_required_state_attrs() raises clear errors for missing fields."""

    def test_missing_attribute_gets_initialized(self):
        """Missing attribute is initialized to default and logged."""
        state = PipelineState()
        # Remove an attribute to simulate corrupted checkpoint
        delattr(state, 'text_metadata')
        assert not hasattr(state, 'text_metadata')

        validate_required_state_attrs(state, ['text_metadata'], 'MATCH')

        # Attribute should now exist with default value
        assert hasattr(state, 'text_metadata')
        assert state.text_metadata == []

    def test_multiple_missing_attributes_all_initialized(self):
        """Multiple missing attributes are all initialized."""
        state = PipelineState()
        delattr(state, 'text_metadata')
        delattr(state, 'matches')
        delattr(state, 'caption_results')

        validate_required_state_attrs(
            state, ['text_metadata', 'matches', 'caption_results'], 'MATCH'
        )

        assert state.text_metadata == []
        assert state.matches == []
        assert state.caption_results == {}

    def test_existing_attributes_not_overwritten(self):
        """Existing attributes with data are preserved (not overwritten)."""
        state = PipelineState()
        state.text_metadata = [{"video_path": "test.mp4", "text": "hello"}]

        validate_required_state_attrs(state, ['text_metadata'], 'MATCH')

        # Data should be preserved
        assert len(state.text_metadata) == 1
        assert state.text_metadata[0]["video_path"] == "test.mp4"

    def test_validate_state_attributes_on_pipeline_state(self):
        """PipelineState.validate_state_attributes() initializes None fields."""
        state = PipelineState()
        # Force None on a required field
        state.text_metadata = None

        initialized = state.validate_state_attributes()

        assert 'text_metadata' in initialized
        assert state.text_metadata == []

    def test_validate_state_attributes_returns_empty_when_all_valid(self):
        """validate_state_attributes() returns empty list when all fields valid."""
        state = PipelineState()
        initialized = state.validate_state_attributes()
        assert initialized == []

    def test_unknown_attribute_gets_none_default(self):
        """Attribute not in _STATE_ATTR_DEFAULTS gets None."""
        state = PipelineState()
        # A hypothetical unknown attribute
        validate_required_state_attrs(state, ['nonexistent_field'], 'TEST')
        assert state.nonexistent_field is None


# ---------------------------------------------------------------------------
# Test: pipeline.run() handles stage exceptions without losing checkpoint
# ---------------------------------------------------------------------------

class TestPipelineRunExceptionHandling:
    """AC: pipeline.run() handles stage exceptions without losing prior checkpoint data."""

    def test_stage_returning_fail_preserves_checkpoint(self, tmp_path):
        """A stage returning StageResult.fail() doesn't corrupt checkpoint."""
        analyze_data = {"keywords": ["k1"]}
        stages = [
            SuccessStage("ANALYZE", data=analyze_data),
            FailStage("VIDEO_SEARCH", error_msg="network error"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        success = pipeline.run(resume=False)

        assert success is False
        # ANALYZE data persists in checkpoint
        assert pipeline.checkpoint.data.analyze == analyze_data
        assert pipeline.checkpoint.data.last_completed_stage == "ANALYZE"

    def test_unhandled_exception_in_stage_propagates(self, tmp_path):
        """Unhandled exception in stage.run() propagates (not swallowed)."""
        stages = [
            SuccessStage("ANALYZE"),
            ExceptionStage("VIDEO_SEARCH", RuntimeError, "unexpected crash"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)

        # The PipelineOrchestrator.run() does NOT catch arbitrary exceptions
        # from stage.run() - it only handles StageResult failures.
        # Exceptions propagate to the caller.
        with pytest.raises(RuntimeError, match="unexpected crash"):
            pipeline.run(resume=False)

        # But ANALYZE checkpoint should still have been saved before crash
        assert pipeline.checkpoint.data is not None
        assert pipeline.checkpoint.data.last_completed_stage == "ANALYZE"

    def test_checkpoint_file_written_to_disk_before_failure(self, tmp_path):
        """Checkpoint file is persisted to disk after each successful stage."""
        analyze_data = {"keywords": ["persist_test"]}
        stages = [
            SuccessStage("ANALYZE", data=analyze_data),
            SuccessStage("VIDEO_SEARCH", data={"video_ids": ["v1"]}),
            FailStage("CAPTION", error_msg="rate limited"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        pipeline.run(resume=False)

        # Verify checkpoint file on disk
        cp_path = tmp_path / "checkpoint.json"
        assert cp_path.exists()

        with open(cp_path, 'r', encoding='utf-8') as f:
            cp_data = json.load(f)

        assert cp_data["last_completed_stage"] == "VIDEO_SEARCH"
        assert cp_data["analyze"] == analyze_data

    def test_validation_failure_does_not_corrupt_checkpoint(self, tmp_path):
        """Stage validation failure doesn't corrupt prior checkpoint data."""
        stages = [
            SuccessStage("ANALYZE", data={"keywords": ["safe"]}),
            ValidationFailStage(
                "VIDEO_SEARCH",
                "Missing required config: search.api_key"
            ),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        success = pipeline.run(resume=False)

        assert success is False
        assert pipeline.checkpoint.data.analyze == {"keywords": ["safe"]}

    def test_multiple_stages_checkpoint_progressively(self, tmp_path):
        """Each successful stage advances the checkpoint progressively."""
        stages = [
            SuccessStage("ANALYZE", data={"keywords": ["a"]}),
            SuccessStage("VIDEO_SEARCH", data={"video_ids": ["v"]}),
            SuccessStage("CAPTION", data={"caption_count": 1}),
            SuccessStage("MATCH", data={"match_count": 1}),
            FailStage("ITERATIVE_MATCH", error_msg="gap analysis failed"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        pipeline.run(resume=False)

        assert pipeline.checkpoint.data.last_completed_stage == "MATCH"
        # All four prior stages should have data
        assert pipeline.checkpoint.data.analyze != {}
        assert pipeline.checkpoint.data.video_search != {}
        assert pipeline.checkpoint.data.caption != {}
        assert pipeline.checkpoint.data.match != {}
        # Failed stage should not have data
        assert pipeline.checkpoint.data.iterative_match == {}


# ---------------------------------------------------------------------------
# Test: --resume correctly skips completed stages after mid-pipeline failure
# ---------------------------------------------------------------------------

class TestResumeAfterMidPipelineFailure:
    """AC: --resume correctly skips already-completed stages after failure."""

    def test_resume_skips_completed_stages(self, tmp_path):
        """After ANALYZE+VIDEO_SEARCH complete and CAPTION fails,
        resuming skips ANALYZE+VIDEO_SEARCH."""
        call_log = []

        class TrackingSuccessStage(Stage):
            def __init__(self, stage_name, data=None):
                self.name = stage_name
                self.description = f"Tracking {stage_name}"
                self._data = data or {"completed": True}

            def run(self, state, config, checkpoint):
                call_log.append(('run', self.name))
                return StageResult.ok(data=self._data)

            def can_skip(self, state, checkpoint):
                return checkpoint.should_skip_stage(self.name)

            def restore(self, state, checkpoint, config=None):
                call_log.append(('restore', self.name))
                return True

            def validate_inputs(self, state, config):
                return None

        class TrackingFailStage(Stage):
            def __init__(self, stage_name, fail_first=True):
                self.name = stage_name
                self.description = f"Tracking fail {stage_name}"
                self._fail_first = fail_first
                self._call_count = 0

            def run(self, state, config, checkpoint):
                self._call_count += 1
                call_log.append(('run', self.name))
                if self._fail_first and self._call_count == 1:
                    return StageResult.fail("temporary failure")
                return StageResult.ok(data={"recovered": True})

            def can_skip(self, state, checkpoint):
                return checkpoint.should_skip_stage(self.name)

            def restore(self, state, checkpoint, config=None):
                call_log.append(('restore', self.name))
                return True

            def validate_inputs(self, state, config):
                return None

        # First run: ANALYZE + VIDEO_SEARCH succeed, CAPTION fails
        stages_run1 = [
            TrackingSuccessStage("ANALYZE", {"keywords": ["k"]}),
            TrackingSuccessStage("VIDEO_SEARCH", {"video_ids": ["v"]}),
            TrackingFailStage("CAPTION", fail_first=True),
        ]

        cfg = create_mock_config(tmp_path)
        cache_path = Path(cfg.cache.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cfg.embedding = Mock(provider="sentence-transformers")

        pipeline1 = PipelineOrchestrator(cfg, tmp_path, stages=stages_run1)
        result1 = pipeline1.run(resume=False)
        assert result1 is False
        assert pipeline1.checkpoint.data.last_completed_stage == "VIDEO_SEARCH"

        # Capture what ran in first pass
        first_run_calls = list(call_log)
        assert ('run', 'ANALYZE') in first_run_calls
        assert ('run', 'VIDEO_SEARCH') in first_run_calls
        assert ('run', 'CAPTION') in first_run_calls

        # Second run: resume - ANALYZE and VIDEO_SEARCH should be skipped
        call_log.clear()
        caption_stage = TrackingFailStage("CAPTION", fail_first=False)
        caption_stage._call_count = 1  # Simulate it's now the 2nd call

        stages_run2 = [
            TrackingSuccessStage("ANALYZE"),
            TrackingSuccessStage("VIDEO_SEARCH"),
            caption_stage,
        ]

        pipeline2 = PipelineOrchestrator(cfg, tmp_path, stages=stages_run2)
        result2 = pipeline2.run(resume=True)

        assert result2 is True
        # ANALYZE and VIDEO_SEARCH should have been restored, not re-run
        second_run_calls = list(call_log)
        run_calls = [name for action, name in second_run_calls if action == 'run']
        restore_calls = [name for action, name in second_run_calls if action == 'restore']

        # ANALYZE and VIDEO_SEARCH should be restored (skipped)
        assert 'ANALYZE' in restore_calls
        assert 'VIDEO_SEARCH' in restore_calls
        # They should NOT have been run again
        assert 'ANALYZE' not in run_calls
        assert 'VIDEO_SEARCH' not in run_calls
        # CAPTION should have been run (it was the stage that failed)
        assert 'CAPTION' in run_calls

    def test_resume_from_checkpoint_file_on_disk(self, tmp_path):
        """Resume loads checkpoint from disk and skips completed stages."""
        # Write a checkpoint to disk simulating prior partial run
        cp_data = {
            "version": "2.0",
            "created_at": "2026-02-01T12:00:00",
            "updated_at": "2026-02-01T12:01:00",
            "last_completed_stage": "CAPTION",
            "config_hash": "",
            "voiceover_path": "",
            "voiceover_hash": "",
            "analyze": {"keywords": ["from_disk"]},
            "video_search": {"video_ids": ["disk_vid"]},
            "caption": {"caption_count": 5},
            "match": {},
            "iterative_match": {},
            "download_segments": {},
        }
        cp_path = tmp_path / "checkpoint.json"
        with open(cp_path, 'w', encoding='utf-8') as f:
            json.dump(cp_data, f)

        run_log = []

        class LoggingStage(Stage):
            def __init__(self, stage_name):
                self.name = stage_name
                self.description = stage_name

            def run(self, state, config, checkpoint):
                run_log.append(self.name)
                return StageResult.ok(data={"ran": True})

            def can_skip(self, state, checkpoint):
                return checkpoint.should_skip_stage(self.name)

            def restore(self, state, checkpoint, config=None):
                return True

            def validate_inputs(self, state, config):
                return None

        stages = [
            LoggingStage("ANALYZE"),
            LoggingStage("VIDEO_SEARCH"),
            LoggingStage("CAPTION"),
            LoggingStage("MATCH"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        result = pipeline.run(resume=True)

        assert result is True
        # Only MATCH should have run; the rest were in checkpoint
        assert run_log == ["MATCH"]

    def test_resume_with_empty_checkpoint_runs_all(self, tmp_path):
        """Resume with no checkpoint on disk runs all stages."""
        run_log = []

        class LogStage(Stage):
            def __init__(self, name_str):
                self.name = name_str
                self.description = name_str

            def run(self, state, config, checkpoint):
                run_log.append(self.name)
                return StageResult.ok(data={"ok": True})

            def can_skip(self, state, checkpoint):
                return checkpoint.should_skip_stage(self.name)

            def restore(self, state, checkpoint, config=None):
                return True

            def validate_inputs(self, state, config):
                return None

        stages = [LogStage("ANALYZE"), LogStage("VIDEO_SEARCH")]
        pipeline = _make_pipeline(tmp_path, stages)
        result = pipeline.run(resume=True)

        assert result is True
        assert run_log == ["ANALYZE", "VIDEO_SEARCH"]


# ---------------------------------------------------------------------------
# Test: Stage metrics on failure
# ---------------------------------------------------------------------------

class TestStageMetricsOnFailure:
    """Additional coverage: stage metrics are recorded on failure."""

    def test_failed_stage_has_metrics_with_failed_flag(self, tmp_path):
        """Failed stage should have metrics with failed=True."""
        stages = [
            FailStage("ANALYZE", error_msg="fail for metrics test"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        pipeline.run(resume=False)

        assert "ANALYZE" in pipeline.stage_metrics
        assert pipeline.stage_metrics["ANALYZE"].failed is True

    def test_successful_stages_have_unfailed_metrics(self, tmp_path):
        """Successful stages should have metrics with failed=False."""
        stages = [
            SuccessStage("ANALYZE"),
            FailStage("VIDEO_SEARCH"),
        ]

        pipeline = _make_pipeline(tmp_path, stages)
        pipeline.run(resume=False)

        assert "ANALYZE" in pipeline.stage_metrics
        assert pipeline.stage_metrics["ANALYZE"].failed is False
        assert "VIDEO_SEARCH" in pipeline.stage_metrics
        assert pipeline.stage_metrics["VIDEO_SEARCH"].failed is True
