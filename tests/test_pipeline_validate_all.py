"""
Pipeline validate_all() Tests - US-81-008

Tests that verify:
- Pipeline.validate_all() runs validate_inputs() for ALL stages, collecting errors
- validate_all() returns structured StageValidationResult list
- validate_all() checks config validity per stage without executing
- validate_all() catches a missing voiceover file for ANALYZE stage
- validate_all() integrates with --dry-run flag
"""

import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.checkpoint import CheckpointManager
from src.pipeline import PipelineOrchestrator, StageValidationResult
from src.stages import Stage, StageResult
from src.state import PipelineState

from tests.fixtures import create_mock_config


# ---------------------------------------------------------------------------
# Test Stage Implementations
# ---------------------------------------------------------------------------

class _DummyStage(Stage):
    """A stage that tracks whether run() was called."""

    def __init__(self, name, validation_error=None, can_skip_result=False):
        self.name = name
        self._validation_error = validation_error
        self._can_skip_result = can_skip_result
        self.run_called = False

    def run(self, state, config, checkpoint=None):
        self.run_called = True
        return StageResult(success=True)

    def can_skip(self, state, checkpoint):
        return self._can_skip_result

    def restore(self, state, checkpoint, config=None):
        return True

    def validate_inputs(self, state, config):
        return self._validation_error


def _make_orchestrator(stages, tmp_path, checkpoint_exists=False, resume_mode=False):
    """Create PipelineOrchestrator with given stages (bypassing __init__ I/O)."""
    config = create_mock_config(tmp_path=tmp_path)
    config.project_dir = str(tmp_path)
    config.downloaded_videos_dir = str(tmp_path / 'downloads')
    config.non_interactive = True

    state = PipelineState()
    state.voiceover_path = str(tmp_path / 'test.srt')
    state.project_dir = str(tmp_path)

    checkpoint = Mock(spec=CheckpointManager)
    checkpoint.exists.return_value = checkpoint_exists
    checkpoint.load.return_value = {} if checkpoint_exists else None
    checkpoint.validate.return_value = {'valid': True, 'errors': [], 'warnings': []}
    checkpoint.is_stale.return_value = False
    checkpoint.should_skip_stage.return_value = True
    checkpoint.get_stage_data.return_value = {}
    checkpoint.save.return_value = None

    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.stages = stages
    orch.state = state
    orch.config = config
    orch.checkpoint = checkpoint
    orch.resume_mode = resume_mode
    orch.healing_enabled = False
    orch.current_stage = None
    orch.stage_timings = {}
    orch.stage_metrics = {}
    orch._validate_config = Mock(return_value=[])

    return orch


# ===========================================================================
# validate_all() returns structured results
# ===========================================================================

class TestValidateAllStructuredOutput:
    """Test that validate_all() returns StageValidationResult list."""

    def test_returns_list_of_stage_validation_results(self, tmp_path):
        """validate_all() returns a list of StageValidationResult."""
        stages = [_DummyStage('ANALYZE'), _DummyStage('MATCH')]
        orch = _make_orchestrator(stages, tmp_path)

        results = orch.validate_all()

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, StageValidationResult)

    def test_result_fields_populated(self, tmp_path):
        """Each result has stage_name, status, and message fields."""
        stages = [_DummyStage('ANALYZE')]
        orch = _make_orchestrator(stages, tmp_path)

        results = orch.validate_all()

        assert results[0].stage_name == 'ANALYZE'
        assert results[0].status == 'run'
        assert results[0].message == 'inputs valid'

    def test_error_result_includes_validation_message(self, tmp_path):
        """Error results include the validation error message."""
        stages = [_DummyStage('MATCH', validation_error='Missing text_metadata')]
        orch = _make_orchestrator(stages, tmp_path)

        results = orch.validate_all()

        assert results[0].stage_name == 'MATCH'
        assert results[0].status == 'error'
        assert 'Missing text_metadata' in results[0].message


# ===========================================================================
# validate_all() collects ALL errors (does not short-circuit)
# ===========================================================================

class TestValidateAllCollectsAllErrors:
    """validate_all() must collect errors from ALL stages, not fail-fast."""

    def test_collects_multiple_errors(self, tmp_path):
        """All stage errors are collected, not just the first one."""
        stages = [
            _DummyStage('ANALYZE', validation_error='No voiceover'),
            _DummyStage('VIDEO_SEARCH', validation_error='No keywords'),
            _DummyStage('MATCH'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        results = orch.validate_all()

        error_results = [r for r in results if r.status == 'error']
        assert len(error_results) == 2
        assert error_results[0].stage_name == 'ANALYZE'
        assert error_results[1].stage_name == 'VIDEO_SEARCH'

    def test_valid_stages_still_included_after_errors(self, tmp_path):
        """Stages after an error are still validated."""
        stages = [
            _DummyStage('ANALYZE', validation_error='Missing voiceover'),
            _DummyStage('VIDEO_SEARCH'),
            _DummyStage('MATCH', validation_error='No matches'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        results = orch.validate_all()

        assert len(results) == 3
        assert results[0].status == 'error'
        assert results[1].status == 'run'
        assert results[2].status == 'error'


# ===========================================================================
# validate_all() respects skip/only filters
# ===========================================================================

class TestValidateAllFilters:
    """validate_all() respects skip_stages and only_stages."""

    def test_skip_stages(self, tmp_path):
        """Skipped stages get status='skip'."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH'),
            _DummyStage('MATCH'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        results = orch.validate_all(skip_stages=['VIDEO_SEARCH'])

        video_result = [r for r in results if r.stage_name == 'VIDEO_SEARCH'][0]
        assert video_result.status == 'skip'
        assert 'skip_stages' in video_result.message

    def test_only_stages(self, tmp_path):
        """Stages not in only_stages get status='skip'."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH'),
            _DummyStage('MATCH'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        results = orch.validate_all(only_stages=['MATCH'])

        non_match = [r for r in results if r.stage_name != 'MATCH']
        for r in non_match:
            assert r.status == 'skip'

    def test_checkpoint_stages(self, tmp_path):
        """Checkpoint-restorable stages get status='checkpoint'."""
        stages = [
            _DummyStage('ANALYZE', can_skip_result=True),
            _DummyStage('MATCH', can_skip_result=False),
        ]
        orch = _make_orchestrator(
            stages, tmp_path, checkpoint_exists=True, resume_mode=True
        )

        results = orch.validate_all(resume=True)

        analyze_result = [r for r in results if r.stage_name == 'ANALYZE'][0]
        assert analyze_result.status == 'checkpoint'


# ===========================================================================
# validate_all() includes config-level errors
# ===========================================================================

class TestValidateAllConfigErrors:
    """validate_all() checks config validity and reports CONFIG-level errors."""

    def test_config_error_reported(self, tmp_path):
        """Config validation errors appear as CONFIG entries in results."""
        stages = [_DummyStage('ANALYZE')]
        orch = _make_orchestrator(stages, tmp_path)
        orch._validate_config = Mock(return_value=[
            'Embedding provider not configured'
        ])

        results = orch.validate_all()

        config_errors = [r for r in results if r.stage_name == 'CONFIG']
        assert len(config_errors) == 1
        assert config_errors[0].status == 'error'
        assert 'Embedding provider' in config_errors[0].message

    def test_config_error_plus_stage_error(self, tmp_path):
        """Both config and stage errors are collected."""
        stages = [_DummyStage('ANALYZE', validation_error='No voiceover')]
        orch = _make_orchestrator(stages, tmp_path)
        orch._validate_config = Mock(return_value=['Cache dir not writable'])

        results = orch.validate_all()

        errors = [r for r in results if r.status == 'error']
        assert len(errors) == 2
        names = {r.stage_name for r in errors}
        assert 'CONFIG' in names
        assert 'ANALYZE' in names


# ===========================================================================
# validate_all() does NOT execute stages
# ===========================================================================

class TestValidateAllNoExecution:
    """validate_all() must not call stage.run()."""

    def test_no_run_called(self, tmp_path):
        """run() must not be called on any stage."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        orch.validate_all()

        for stage in stages:
            assert not stage.run_called


# ===========================================================================
# Real AnalyzeStage: catches missing voiceover file
# ===========================================================================

class TestValidateAllMissingVoiceover:
    """Test that validate_all() catches a missing voiceover file via AnalyzeStage."""

    def test_missing_voiceover_file_detected(self, tmp_path):
        """
        validate_all() reports ANALYZE stage error for a non-existent voiceover file,
        without running any stage.
        """
        from unittest.mock import patch as _patch
        from src.stages.analyze import AnalyzeStage

        analyze_stage = AnalyzeStage()
        stages = [analyze_stage]
        orch = _make_orchestrator(stages, tmp_path)

        # Set voiceover_path to a file that does NOT exist
        orch.state.voiceover_path = str(tmp_path / 'nonexistent_voiceover.srt')

        # Patch run() to track if it's called (real stage has no run_called attr)
        with _patch.object(analyze_stage, 'run', wraps=analyze_stage.run) as mock_run:
            results = orch.validate_all()

            # Should have exactly one result for ANALYZE, with status='error'
            assert len(results) >= 1
            analyze_results = [r for r in results if r.stage_name == 'ANALYZE']
            assert len(analyze_results) == 1
            assert analyze_results[0].status == 'error'
            assert 'not found' in analyze_results[0].message.lower() or 'voiceover' in analyze_results[0].message.lower()

            # Stage.run() must NOT have been called
            mock_run.assert_not_called()

    def test_empty_voiceover_path_detected(self, tmp_path):
        """validate_all() reports error when voiceover_path is empty."""
        from src.stages.analyze import AnalyzeStage

        stages = [AnalyzeStage()]
        orch = _make_orchestrator(stages, tmp_path)
        orch.state.voiceover_path = ''

        results = orch.validate_all()

        analyze_results = [r for r in results if r.stage_name == 'ANALYZE']
        assert len(analyze_results) == 1
        assert analyze_results[0].status == 'error'


# ===========================================================================
# dry_run flag uses validate_all()
# ===========================================================================

class TestDryRunUsesValidateAll:
    """Test that --dry-run delegates to validate_all()."""

    def test_dry_run_uses_validate_all(self, tmp_path, caplog):
        """Dry-run mode should call validate_all() internally."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH', validation_error='No video_ids'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        with caplog.at_level(logging.INFO):
            result = orch.run(dry_run=True, resume=False)

        # Should return False because of VIDEO_SEARCH error
        assert result is False
        # Summary table should appear
        assert 'Stage Execution Plan' in caplog.text
        assert 'No video_ids' in caplog.text
        assert 'DRY-RUN COMPLETE' in caplog.text

    def test_dry_run_all_valid_returns_true(self, tmp_path):
        """Dry-run with all valid stages returns True."""
        stages = [_DummyStage('ANALYZE'), _DummyStage('MATCH')]
        orch = _make_orchestrator(stages, tmp_path)

        result = orch.run(dry_run=True, resume=False)
        assert result is True
