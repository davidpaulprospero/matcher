"""
Tests for US-45-008: State snapshot before stage execution for rollback capability.

Verifies that PipelineOrchestrator snapshots critical state fields before each stage
and restores them when a stage fails, preventing state corruption.
"""

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import PipelineOrchestrator
from src.stages import StageResult, StageMetrics
from src.state import PipelineState, Match, DownloadedVideo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    """Create a minimal mock Config that passes validation."""
    config = MagicMock()
    config.cache.cache_dir = str(Path(__file__).parent / ".test_cache")
    config.embedding.provider = "sentence-transformers"
    config._config_hash = "test"
    return config


def _make_match(index: int = 0) -> Match:
    return Match(
        segment_index=index,
        video_file=f"video_{index}.mp4",
        video_start=0.0,
        video_end=10.0,
        confidence=0.8,
        strategy="test",
    )


def _make_downloaded(index: int = 0) -> DownloadedVideo:
    return DownloadedVideo(
        file=f"segment_{index}.mp4",
        url=f"https://youtube.com/watch?v=test{index}",
    )


class _FakeStage:
    """A stage whose run() behaviour is controlled by the caller."""

    def __init__(self, name: str, result: StageResult):
        self.name = name
        self._result = result

    # Stage protocol methods
    def can_skip(self, state, checkpoint):
        return False

    def restore(self, state, checkpoint, config):
        return True

    def validate_inputs(self, state, config):
        return None  # no error

    def run(self, state, config, checkpoint):
        return self._result


class _CorruptingStage(_FakeStage):
    """A stage that mutates state then reports failure."""

    def __init__(self, name: str):
        super().__init__(name, StageResult(success=False, error="boom"))

    def run(self, state, config, checkpoint):
        # Corrupt state before failing
        state.matches.append(_make_match(99))
        state.alternatives[99] = [_make_match(99)]
        state.downloaded_segments.append(_make_downloaded(99))
        state.output_files.append(Path("/fake/output.otio"))
        return self._result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStateRollback:
    """Verify state snapshot/rollback on stage failure."""

    def test_state_restored_after_failure(self, tmp_path):
        """Core acceptance test: state is clean after a corrupting stage fails."""
        config = _make_config()
        pipeline = PipelineOrchestrator(config, tmp_path)

        # Seed state with known-good data
        pipeline.state.matches = [_make_match(0), _make_match(1)]
        pipeline.state.alternatives = {0: [_make_match(0)]}
        pipeline.state.downloaded_segments = [_make_downloaded(0)]
        pipeline.state.output_files = [Path("/real/output.otio")]

        original_matches = list(pipeline.state.matches)
        original_alts = dict(pipeline.state.alternatives)
        original_segments = list(pipeline.state.downloaded_segments)
        original_outputs = list(pipeline.state.output_files)

        # Add a corrupting stage
        pipeline.add_stage(_CorruptingStage("FAIL_STAGE"))

        result = pipeline.run(resume=False)

        assert result is False, "Pipeline should report failure"

        # State should be restored to pre-stage values
        assert len(pipeline.state.matches) == len(original_matches)
        assert pipeline.state.matches == original_matches
        assert pipeline.state.alternatives == original_alts
        assert len(pipeline.state.downloaded_segments) == len(original_segments)
        assert pipeline.state.output_files == original_outputs

    def test_state_unchanged_on_success(self, tmp_path):
        """Successful stages should leave state as-is (no rollback)."""
        config = _make_config()
        pipeline = PipelineOrchestrator(config, tmp_path)

        pipeline.state.matches = [_make_match(0)]

        # A successful stage that adds a match
        class _GoodStage(_FakeStage):
            def run(self, state, config, checkpoint):
                state.matches.append(_make_match(1))
                return StageResult(success=True, data={"test": True})

        pipeline.add_stage(_GoodStage("GOOD_STAGE", StageResult(success=True)))

        result = pipeline.run(resume=False)

        assert result is True
        assert len(pipeline.state.matches) == 2, "Successful stage mutations should persist"

    def test_snapshot_is_shallow_copy(self, tmp_path):
        """Snapshot uses shallow copy so the list identity differs from the original."""
        config = _make_config()
        pipeline = PipelineOrchestrator(config, tmp_path)

        pipeline.state.matches = [_make_match(0)]
        snapshot = pipeline._snapshot_state()

        # Different list object
        assert snapshot['matches'] is not pipeline.state.matches
        # But same content
        assert snapshot['matches'] == pipeline.state.matches

    def test_rollback_logs_warning(self, tmp_path, caplog):
        """Rollback should log which fields were restored."""
        config = _make_config()
        pipeline = PipelineOrchestrator(config, tmp_path)

        pipeline.state.matches = [_make_match(0)]
        pipeline.add_stage(_CorruptingStage("BAD_STAGE"))

        import logging
        with caplog.at_level(logging.WARNING):
            pipeline.run(resume=False)

        assert any("State rollback after BAD_STAGE failure" in r.message for r in caplog.records)
        assert any("restored fields" in r.message for r in caplog.records)

    def test_empty_state_rollback(self, tmp_path):
        """Rollback works correctly when state starts empty."""
        config = _make_config()
        pipeline = PipelineOrchestrator(config, tmp_path)

        # State starts with empty lists (defaults)
        assert pipeline.state.matches == []
        assert pipeline.state.alternatives == {}

        pipeline.add_stage(_CorruptingStage("CORRUPT"))

        result = pipeline.run(resume=False)
        assert result is False

        # Should be back to empty
        assert pipeline.state.matches == []
        assert pipeline.state.alternatives == {}
        assert pipeline.state.downloaded_segments == []
        assert pipeline.state.output_files == []
