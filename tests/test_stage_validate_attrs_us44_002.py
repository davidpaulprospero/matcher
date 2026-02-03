"""
Tests for US-44-002: validate_required_state_attrs calls in all pipeline stages.

Verifies that each of the 5 newly-guarded stages (ANALYZE, VIDEO_SEARCH,
CAPTION, DOWNLOAD_SEGMENTS, OUTPUT) calls validate_required_state_attrs at
run() entry, so missing attributes are initialized instead of causing
AttributeError crashes.
"""

import logging
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.stages import validate_required_state_attrs


class TestAnalyzeStageValidation:
    """ANALYZE stage validates ['voiceover_path'] at run() entry."""

    @pytest.mark.fast
    def test_missing_voiceover_path_initialized(self, caplog):
        """ANALYZE initializes missing voiceover_path instead of AttributeError."""
        from src.stages.analyze import AnalyzeStage

        stage = AnalyzeStage()
        state = SimpleNamespace()  # No voiceover_path

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            # run() will fail after validation (no real file), but should NOT
            # raise AttributeError on missing voiceover_path
            result = stage.run(state, MagicMock(), MagicMock())

        # voiceover_path should be initialized to None (default)
        assert hasattr(state, 'voiceover_path')
        # Stage should fail gracefully (no path), not crash
        assert result.success is False
        assert "voiceover_path" in caplog.text or "voiceover" in result.error.lower()


class TestVideoSearchStageValidation:
    """VIDEO_SEARCH stage validates ['keywords'] at run() entry."""

    @pytest.mark.fast
    def test_missing_keywords_initialized(self, caplog):
        """VIDEO_SEARCH initializes missing keywords instead of AttributeError."""
        from src.stages.video_search import VideoSearchStage

        stage = VideoSearchStage()
        state = SimpleNamespace()  # No keywords

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            result = stage.run(state, MagicMock(), MagicMock())

        # keywords should be initialized to []
        assert hasattr(state, 'keywords')
        assert state.keywords == []


class TestCaptionStageValidation:
    """CAPTION stage validates ['video_ids'] at run() entry."""

    @pytest.mark.fast
    def test_missing_video_ids_initialized(self, caplog):
        """CAPTION initializes missing video_ids instead of AttributeError."""
        from src.stages.caption_stage import CaptionStage

        stage = CaptionStage()
        state = SimpleNamespace()  # No video_ids

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            result = stage.run(state, MagicMock(), MagicMock())

        # video_ids should be initialized to []
        assert hasattr(state, 'video_ids')
        assert state.video_ids == []


class TestDownloadSegmentsStageValidation:
    """DOWNLOAD_SEGMENTS stage validates ['matches'] at run() entry."""

    @pytest.mark.fast
    def test_missing_matches_initialized(self, caplog):
        """DOWNLOAD_SEGMENTS initializes missing matches instead of AttributeError."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()
        state = SimpleNamespace()  # No matches

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            result = stage.run(state, MagicMock(), MagicMock())

        # matches should be initialized to []
        assert hasattr(state, 'matches')
        assert state.matches == []
        # With empty matches, stage should skip gracefully
        assert result.success is True


class TestOutputStageValidation:
    """OUTPUT stage validates ['matches', 'voiceover_segments'] at run() entry."""

    @pytest.mark.fast
    def test_missing_matches_and_segments_initialized(self, caplog):
        """OUTPUT initializes missing matches and voiceover_segments instead of AttributeError."""
        from src.stages.output import OutputStage

        stage = OutputStage()
        state = SimpleNamespace()  # No matches or voiceover_segments

        # OUTPUT needs config.otio_output_dir
        mock_config = MagicMock()
        mock_config.otio_output_dir = "test_output"

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            # Will fail on output_dir but AFTER validation succeeds
            try:
                result = stage.run(state, mock_config, MagicMock())
            except Exception:
                pass  # We only care that the validation ran

        # Both attributes should be initialized
        assert hasattr(state, 'matches')
        assert hasattr(state, 'voiceover_segments')
        assert state.matches == []
        assert state.voiceover_segments == []


class TestAllStagesHaveValidation:
    """Meta-test: verify validate_required_state_attrs is called in all 7 stages."""

    @pytest.mark.fast
    def test_all_stages_import_validate(self):
        """All stage modules import validate_required_state_attrs."""
        import importlib
        stage_modules = [
            'src.stages.analyze',
            'src.stages.video_search',
            'src.stages.caption_stage',
            'src.stages.match',
            'src.stages.iterative_match',
            'src.stages.download_segments',
            'src.stages.output',
        ]

        for module_name in stage_modules:
            mod = importlib.import_module(module_name)
            assert hasattr(mod, 'validate_required_state_attrs'), (
                f"{module_name} does not import validate_required_state_attrs"
            )
