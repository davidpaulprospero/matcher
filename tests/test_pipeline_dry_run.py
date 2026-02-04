"""
Pipeline Dry-Run Mode Tests - US-51-009

Tests that verify:
- Pipeline.run(dry_run=True) validates all stage inputs without executing any stage
- Dry-run reports which stages would run, skip (checkpoint), or have validation errors
- Dry-run output is logged at INFO level with a summary table
- Dry-run does not modify state or checkpoint
- Dry-run correctly identifies stages that would fail validation
"""

import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.checkpoint import CheckpointManager
from src.pipeline import PipelineOrchestrator
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
        self.state_before_run = None

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
# Dry-run does NOT execute stages
# ===========================================================================

class TestDryRunNoExecution:
    """Test that dry-run mode never calls stage.run()."""

    def test_no_stage_run_called(self, tmp_path):
        """Dry-run must not call run() on any stage."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH'),
            _DummyStage('MATCH'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        orch.run(dry_run=True, resume=False)

        for stage in stages:
            assert not stage.run_called, f"{stage.name}.run() should NOT be called in dry-run"

    def test_no_stage_run_called_with_errors(self, tmp_path):
        """Dry-run must not call run() even when validation errors exist."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH', validation_error='Missing video_ids'),
            _DummyStage('MATCH'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        orch.run(dry_run=True, resume=False)

        for stage in stages:
            assert not stage.run_called, f"{stage.name}.run() should NOT be called in dry-run"


# ===========================================================================
# Dry-run does NOT modify state or checkpoint
# ===========================================================================

class TestDryRunNoStateModification:
    """Test that dry-run does not modify state or checkpoint."""

    def test_state_unchanged(self, tmp_path):
        """Pipeline state should be unchanged after dry-run."""
        stages = [_DummyStage('ANALYZE'), _DummyStage('MATCH')]
        orch = _make_orchestrator(stages, tmp_path)

        # Capture state before
        state_keywords_before = orch.state.keywords.copy() if hasattr(orch.state, 'keywords') and orch.state.keywords else []
        state_matches_before = orch.state.matches.copy() if hasattr(orch.state, 'matches') and orch.state.matches else []

        orch.run(dry_run=True, resume=False)

        # State should be identical after dry-run
        state_keywords_after = orch.state.keywords if hasattr(orch.state, 'keywords') and orch.state.keywords else []
        state_matches_after = orch.state.matches if hasattr(orch.state, 'matches') and orch.state.matches else []
        assert state_keywords_before == list(state_keywords_after)
        assert state_matches_before == list(state_matches_after)

    def test_checkpoint_not_saved(self, tmp_path):
        """Checkpoint.save() should not be called during dry-run."""
        stages = [_DummyStage('ANALYZE'), _DummyStage('MATCH')]
        orch = _make_orchestrator(stages, tmp_path)

        orch.run(dry_run=True, resume=False)

        orch.checkpoint.save.assert_not_called()


# ===========================================================================
# Dry-run reports stage actions correctly
# ===========================================================================

class TestDryRunStageActions:
    """Test that dry-run correctly categorizes stages."""

    def test_all_stages_would_run(self, tmp_path, caplog):
        """When no checkpoint, all stages should be marked as 'run'."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH'),
            _DummyStage('MATCH'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        with caplog.at_level(logging.INFO):
            result = orch.run(dry_run=True, resume=False)

        assert result is True
        # All three stages should appear as "run"
        assert caplog.text.count('run') >= 3  # at minimum in the summary table

    def test_skipped_stages_reported(self, tmp_path, caplog):
        """Stages in skip_stages should be marked as 'skip'."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH'),
            _DummyStage('MATCH'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        with caplog.at_level(logging.INFO):
            result = orch.run(dry_run=True, resume=False, skip_stages=['VIDEO_SEARCH'])

        assert result is True
        assert 'VIDEO_SEARCH' in caplog.text
        assert 'skip' in caplog.text.lower()

    def test_checkpoint_stages_reported(self, tmp_path, caplog):
        """Stages that would be skipped by checkpoint should be marked as 'checkpoint'."""
        stages = [
            _DummyStage('ANALYZE', can_skip_result=True),
            _DummyStage('VIDEO_SEARCH', can_skip_result=False),
        ]
        orch = _make_orchestrator(stages, tmp_path, checkpoint_exists=True, resume_mode=True)

        with caplog.at_level(logging.INFO):
            result = orch.run(dry_run=True, resume=True)

        assert result is True
        assert 'checkpoint' in caplog.text.lower()

    def test_validation_errors_reported(self, tmp_path, caplog):
        """Stages with validation errors should be marked as 'error'."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH', validation_error='No video_ids available'),
            _DummyStage('MATCH'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        with caplog.at_level(logging.INFO):
            result = orch.run(dry_run=True, resume=False)

        assert result is False  # Validation errors -> False
        assert 'error' in caplog.text.lower()
        assert 'No video_ids available' in caplog.text


# ===========================================================================
# Dry-run return value
# ===========================================================================

class TestDryRunReturnValue:
    """Test that dry-run returns correct boolean."""

    def test_returns_true_when_all_valid(self, tmp_path):
        """Returns True when all stages pass validation."""
        stages = [_DummyStage('ANALYZE'), _DummyStage('MATCH')]
        orch = _make_orchestrator(stages, tmp_path)

        result = orch.run(dry_run=True, resume=False)
        assert result is True

    def test_returns_false_when_validation_errors(self, tmp_path):
        """Returns False when any stage has validation errors."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('MATCH', validation_error='Missing matches data'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        result = orch.run(dry_run=True, resume=False)
        assert result is False

    def test_returns_true_when_only_skipped_stages(self, tmp_path):
        """Returns True when all non-skipped stages pass validation."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH', validation_error='Would fail'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        # Skip the failing stage
        result = orch.run(dry_run=True, resume=False, skip_stages=['VIDEO_SEARCH'])
        assert result is True


# ===========================================================================
# Dry-run summary logging
# ===========================================================================

class TestDryRunSummaryLogging:
    """Test that dry-run produces structured summary at INFO level."""

    def test_summary_table_logged(self, tmp_path, caplog):
        """Dry-run should log a summary table with Stage/Action/Detail columns."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        with caplog.at_level(logging.INFO):
            orch.run(dry_run=True, resume=False)

        # Check for table structure
        assert 'Stage Execution Plan' in caplog.text
        assert 'ANALYZE' in caplog.text
        assert 'VIDEO_SEARCH' in caplog.text
        assert 'Summary:' in caplog.text

    def test_dry_run_complete_message(self, tmp_path, caplog):
        """Dry-run should log completion message."""
        stages = [_DummyStage('ANALYZE')]
        orch = _make_orchestrator(stages, tmp_path)

        with caplog.at_level(logging.INFO):
            orch.run(dry_run=True, resume=False)

        assert 'DRY-RUN COMPLETE' in caplog.text

    def test_error_summary_logged_at_error_level(self, tmp_path, caplog):
        """Validation errors should be logged at ERROR level."""
        stages = [
            _DummyStage('ANALYZE', validation_error='Missing keywords'),
        ]
        orch = _make_orchestrator(stages, tmp_path)

        with caplog.at_level(logging.DEBUG):
            orch.run(dry_run=True, resume=False)

        # Find error-level records
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) >= 1, "Validation errors should produce ERROR-level log"
        error_text = ' '.join(r.message for r in error_records)
        assert 'Missing keywords' in error_text


# ===========================================================================
# CLI integration
# ===========================================================================

class TestDryRunCLIFlag:
    """Test that --dry-run CLI flag exists and is parsed correctly."""

    def test_dry_run_flag_parsed(self):
        """--dry-run flag should set args.dry_run to True."""
        from src.cli.args import parse_arguments
        import sys

        with patch.object(sys, 'argv', ['main.py', '--dry-run', '--voiceover', 'test.srt']):
            args = parse_arguments()

        assert args.dry_run is True

    def test_dry_run_flag_default_false(self):
        """--dry-run should default to False."""
        from src.cli.args import parse_arguments
        import sys

        with patch.object(sys, 'argv', ['main.py', '--voiceover', 'test.srt']):
            args = parse_arguments()

        assert args.dry_run is False
