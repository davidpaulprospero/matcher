"""
Stage Restore Validation Tests - US-51-008

Tests that verify:
- Each stage's restore() validates checkpoint data schema/keys before restoring
- Missing required keys cause restore() to return False with WARNING log
- Type mismatches cause restore() to return False with WARNING log
- Pipeline orchestrator re-runs stage when restore() returns False
"""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

from src.checkpoint import CheckpointManager
from src.pipeline import PipelineOrchestrator
from src.stages import Stage, StageResult
from src.state import PipelineState

from tests.fixtures import create_mock_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checkpoint_mock(stage_data):
    """Create a mock CheckpointManager that returns stage_data for get_stage_data."""
    cp = Mock(spec=CheckpointManager)
    cp.get_stage_data.return_value = stage_data
    cp.should_skip_stage.return_value = True
    return cp


def _make_state():
    """Create a minimal PipelineState for testing."""
    state = Mock(spec=PipelineState)
    state.keywords = []
    state.voiceover_segments = []
    state.topic_context = ''
    state.extracted_entities = []
    state.location_chapters = []
    state.video_ids = []
    state.video_search_results = []
    state.search_failed_keywords = []
    state.matches = []
    state.downloaded_segments = []
    state.text_metadata = {}
    state.caption_results = {}
    return state


# ===========================================================================
# ANALYZE Stage Restore Validation
# ===========================================================================

class TestAnalyzeRestoreValidation:
    """Test ANALYZE stage restore validates checkpoint data."""

    def _make_stage(self):
        from src.stages.analyze import AnalyzeStage
        stage = AnalyzeStage.__new__(AnalyzeStage)
        stage.name = 'ANALYZE'
        stage.description = 'Test analyze'
        return stage

    def test_missing_keywords_key_returns_false(self, caplog):
        """restore() returns False when 'keywords' key is missing."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({'segments': []})
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'missing required keys' in caplog.text.lower()
        assert 'keywords' in caplog.text

    def test_missing_segments_key_returns_false(self, caplog):
        """restore() returns False when 'segments' key is missing."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({'keywords': ['test']})
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'missing required keys' in caplog.text.lower()
        assert 'segments' in caplog.text

    def test_keywords_wrong_type_returns_false(self, caplog):
        """restore() returns False when 'keywords' is not a list."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({'keywords': 'not-a-list', 'segments': []})
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected list' in caplog.text.lower()

    def test_segments_wrong_type_returns_false(self, caplog):
        """restore() returns False when 'segments' is not a list."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({'keywords': [], 'segments': 'bad'})
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected list' in caplog.text.lower()

    def test_data_not_dict_returns_false(self, caplog):
        """restore() returns False when checkpoint data is not a dict."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock('not-a-dict')
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected dict' in caplog.text.lower()

    def test_valid_data_restores_successfully(self):
        """restore() returns True with valid data."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({
            'keywords': ['travel', 'nature'],
            'segments': [{'index': 0, 'start': 0.0, 'end': 1.0, 'text': 'hello'}],
            'topic_context': 'test',
            'entities': [],
        })
        state = _make_state()

        result = stage.restore(state, cp)

        assert result is True


# ===========================================================================
# VIDEO_SEARCH Stage Restore Validation
# ===========================================================================

class TestVideoSearchRestoreValidation:
    """Test VIDEO_SEARCH stage restore validates checkpoint data."""

    def _make_stage(self):
        from src.stages.video_search import VideoSearchStage
        stage = VideoSearchStage.__new__(VideoSearchStage)
        stage.name = 'VIDEO_SEARCH'
        stage.description = 'Test video search'
        return stage

    def test_missing_video_ids_key_returns_false(self, caplog):
        """restore() returns False when 'video_ids' key is missing."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({'search_results': []})
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'missing required keys' in caplog.text.lower()
        assert 'video_ids' in caplog.text

    def test_video_ids_wrong_type_returns_false(self, caplog):
        """restore() returns False when 'video_ids' is not a list."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({'video_ids': 'not-a-list'})
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected list' in caplog.text.lower()

    def test_data_not_dict_returns_false(self, caplog):
        """restore() returns False when checkpoint data is not a dict."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock([1, 2, 3])
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected dict' in caplog.text.lower()

    def test_valid_data_restores_successfully(self):
        """restore() returns True with valid data."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({
            'video_ids': ['vid1', 'vid2'],
            'search_results': [],
        })
        state = _make_state()

        # _to_search_results needs to exist
        stage._to_search_results = Mock(return_value=[])

        result = stage.restore(state, cp)

        assert result is True


# ===========================================================================
# CAPTION Stage Restore Validation
# ===========================================================================

class TestCaptionRestoreValidation:
    """Test CAPTION stage restore validates checkpoint data."""

    def _make_stage(self):
        from src.stages.caption_stage import CaptionStage
        stage = CaptionStage.__new__(CaptionStage)
        stage.name = 'CAPTION'
        stage.description = 'Test caption'
        stage._ensure_state_attributes = Mock()
        return stage

    def test_data_not_dict_returns_false(self, caplog):
        """restore() returns False when checkpoint data is not a dict."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock('string-data')
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected dict' in caplog.text.lower()

    def test_caption_results_wrong_type_returns_false(self, caplog):
        """restore() returns False when 'caption_results' is not a dict."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({'caption_results': ['not', 'a', 'dict']})
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected dict' in caplog.text.lower()

    def test_no_data_returns_true(self):
        """restore() returns True when no stage data (valid no-op)."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock(None)
        state = _make_state()

        result = stage.restore(state, cp)

        assert result is True

    def test_valid_data_restores_successfully(self):
        """restore() returns True with valid data."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({
            'caption_results': {'vid1': {'text': 'hello'}},
            'total_segments': 5,
        })
        state = _make_state()
        stage._populate_text_metadata = Mock()

        result = stage.restore(state, cp)

        assert result is True


# ===========================================================================
# MATCH Stage Restore Validation
# ===========================================================================

class TestMatchRestoreValidation:
    """Test MATCH stage restore validates checkpoint data."""

    def _make_stage(self):
        from src.stages.match import MatchStage
        stage = MatchStage.__new__(MatchStage)
        stage.name = 'MATCH'
        stage.description = 'Test match'
        return stage

    def test_data_not_dict_returns_false(self, caplog):
        """restore() returns False when checkpoint data is not a dict."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock('bad-data')
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected dict' in caplog.text.lower()

    def test_matches_wrong_type_returns_false(self, caplog):
        """restore() returns False when 'matches' is not a list."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({'matches': 'not-a-list'})
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected list' in caplog.text.lower()

    def test_valid_data_no_matches_restores(self):
        """restore() returns True with valid data and no matches."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({'some_metadata': True})
        state = _make_state()

        result = stage.restore(state, cp)

        assert result is True


# ===========================================================================
# ITERATIVE_MATCH Stage Restore Validation
# ===========================================================================

class TestIterativeMatchRestoreValidation:
    """Test ITERATIVE_MATCH stage restore validates checkpoint data."""

    def _make_stage(self):
        from src.stages.iterative_match import IterativeMatchStage
        stage = IterativeMatchStage.__new__(IterativeMatchStage)
        stage.name = 'ITERATIVE_MATCH'
        stage.description = 'Test iterative match'
        return stage

    def test_data_not_dict_returns_false(self, caplog):
        """restore() returns False when checkpoint data is not a dict."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock(42)
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected dict' in caplog.text.lower()

    def test_matches_wrong_type_returns_false(self, caplog):
        """restore() returns False when 'matches' is not a list."""
        stage = self._make_stage()
        cp = _make_checkpoint_mock({'matches': {'wrong': 'type'}})
        state = _make_state()

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, cp)

        assert result is False
        assert 'expected list' in caplog.text.lower()


# ===========================================================================
# Pipeline Re-run on Restore Failure
# ===========================================================================

class _RerunTrackingStage(Stage):
    """A stage that tracks whether run() was called and has controllable restore()."""

    def __init__(self, name: str, restore_result: bool = True):
        self.name = name
        self.description = f'Test {name}'
        self._restore_result = restore_result
        self.run_called = False

    def run(self, state, config, checkpoint):
        self.run_called = True
        return StageResult.ok(data={'completed': True})

    def can_skip(self, state, checkpoint):
        return True  # Always skippable (checkpoint says done)

    def restore(self, state, checkpoint, config=None):
        return self._restore_result

    def validate_inputs(self, state, config):
        return None


class TestPipelineRerunOnRestoreFailure:
    """Test that pipeline re-runs stage when restore() returns False."""

    def _make_orchestrator(self, stages, tmp_path):
        """Create PipelineOrchestrator with given stages."""
        import time
        config = create_mock_config(tmp_path=tmp_path)
        config.project_dir = str(tmp_path)
        config.downloaded_videos_dir = str(tmp_path / 'downloads')
        config.non_interactive = True

        state = Mock(spec=PipelineState)
        state.voiceover_path = str(tmp_path / 'test.srt')
        state.project_dir = str(tmp_path)
        state.validate_state_attributes = Mock()
        state.stage_timings = {}

        checkpoint = Mock(spec=CheckpointManager)
        checkpoint.should_skip_stage.return_value = True
        checkpoint.get_stage_data.return_value = {}
        checkpoint.save.return_value = None
        checkpoint.exists.return_value = True
        checkpoint.load.return_value = {}
        checkpoint.validate.return_value = {'valid': True, 'errors': [], 'warnings': []}
        checkpoint.is_stale.return_value = False
        checkpoint.restore_state.return_value = state

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.stages = stages
        orch.state = state
        orch.config = config
        orch.checkpoint = checkpoint
        orch.resume_mode = True
        orch.healing_enabled = False
        orch.current_stage = None
        orch.stage_timings = {}
        orch.stage_metrics = {}
        orch._validate_config = Mock(return_value=[])
        orch._snapshot_state = Mock(return_value={})
        orch._rollback_state = Mock()
        orch._get_state_summary = Mock(return_value='')
        orch._get_recovery_suggestion = Mock(return_value='')

        return orch

    def test_restore_true_skips_stage(self, tmp_path):
        """When restore() returns True, stage.run() is NOT called."""
        stage = _RerunTrackingStage('ANALYZE', restore_result=True)
        orch = self._make_orchestrator([stage], tmp_path)

        orch.run(resume=True)

        assert not stage.run_called, "Stage should NOT have been run when restore succeeds"

    def test_restore_false_reruns_stage(self, tmp_path):
        """When restore() returns False, stage.run() IS called (re-run)."""
        stage = _RerunTrackingStage('ANALYZE', restore_result=False)
        orch = self._make_orchestrator([stage], tmp_path)

        orch.run(resume=True)

        assert stage.run_called, "Stage SHOULD have been run when restore fails"

    def test_restore_false_logs_warning(self, tmp_path, caplog):
        """When restore() fails, pipeline logs a warning about re-running."""
        stage = _RerunTrackingStage('ANALYZE', restore_result=False)
        orch = self._make_orchestrator([stage], tmp_path)

        with caplog.at_level(logging.WARNING):
            orch.run(resume=True)

        assert 're-running stage' in caplog.text.lower() or 're-running' in caplog.text.lower()
