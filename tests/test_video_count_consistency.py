"""
Tests for US-44-010: Video count consistency assertion between
video_ids, caption_results, and text_metadata.

Verifies that MATCH stage validate_inputs() warns on drift between
video_ids and caption_results counts, but does not fail the pipeline.
"""

import logging
import pytest

from src.state import PipelineState, VoiceoverSegment


def _make_state(
    num_video_ids: int = 0,
    num_caption_results: int = 0,
    populate_text_metadata: bool = False,
    num_voiceover_segments: int = 3,
) -> PipelineState:
    """Build a PipelineState with specified video/caption counts."""
    state = PipelineState()
    state.voiceover_segments = [
        VoiceoverSegment(index=i, start=float(i), end=float(i + 1), text=f"seg {i}")
        for i in range(num_voiceover_segments)
    ]
    state.video_ids = [f"vid_{i}" for i in range(num_video_ids)]
    state.caption_results = {
        f"vid_{i}": {
            "segments": [{"text": f"caption {i}", "start": 0, "end": 5}],
            "language": "en",
        }
        for i in range(num_caption_results)
    }
    if populate_text_metadata:
        state.text_metadata = [
            {"text": f"meta {i}", "video_path": f"vid_{i}", "start_time": 0, "end_time": 5}
            for i in range(num_caption_results)
        ]
    return state


def _get_match_stage():
    """Import and return a MatchStage instance."""
    from src.stages.match import MatchStage
    return MatchStage()


class TestVideoCountConsistency:
    """Tests for video_ids / caption_results drift warning in validate_inputs."""

    def test_warning_when_large_drift(self, caplog):
        """100 video_ids but only 50 caption_results (50% drift) should warn."""
        state = _make_state(
            num_video_ids=100,
            num_caption_results=50,
            populate_text_metadata=True,
        )
        stage = _get_match_stage()

        with caplog.at_level(logging.WARNING, logger="src.stages.match"):
            result = stage.validate_inputs(state, None)

        assert result is None, "Consistency check must not fail the pipeline"
        assert any("Video count drift detected" in msg for msg in caplog.messages), (
            "Expected drift warning when video_ids=100, caption_results=50"
        )
        # Verify the warning contains useful debug info
        warning_msg = [m for m in caplog.messages if "Video count drift" in m][0]
        assert "video_ids=100" in warning_msg
        assert "caption_results=50" in warning_msg

    def test_no_warning_within_threshold(self, caplog):
        """95 caption_results for 100 video_ids (5% drift) — no warning."""
        state = _make_state(
            num_video_ids=100,
            num_caption_results=95,
            populate_text_metadata=True,
        )
        stage = _get_match_stage()

        with caplog.at_level(logging.WARNING, logger="src.stages.match"):
            result = stage.validate_inputs(state, None)

        assert result is None
        assert not any("Video count drift detected" in msg for msg in caplog.messages), (
            "Should NOT warn when drift is within 10% threshold"
        )

    def test_no_warning_at_exact_threshold(self, caplog):
        """90 caption_results for 100 video_ids (exactly 10%) — no warning."""
        state = _make_state(
            num_video_ids=100,
            num_caption_results=90,
            populate_text_metadata=True,
        )
        stage = _get_match_stage()

        with caplog.at_level(logging.WARNING, logger="src.stages.match"):
            result = stage.validate_inputs(state, None)

        assert result is None
        assert not any("Video count drift detected" in msg for msg in caplog.messages), (
            "Should NOT warn at exactly 10% — threshold is > 10%"
        )

    def test_warning_just_over_threshold(self, caplog):
        """89 caption_results for 100 video_ids (11% drift) — should warn."""
        state = _make_state(
            num_video_ids=100,
            num_caption_results=89,
            populate_text_metadata=True,
        )
        stage = _get_match_stage()

        with caplog.at_level(logging.WARNING, logger="src.stages.match"):
            result = stage.validate_inputs(state, None)

        assert result is None
        assert any("Video count drift detected" in msg for msg in caplog.messages)

    def test_no_warning_when_both_empty(self, caplog):
        """No video_ids and no caption_results — just fails has_text_data check."""
        state = _make_state(num_video_ids=0, num_caption_results=0)
        stage = _get_match_stage()

        with caplog.at_level(logging.WARNING, logger="src.stages.match"):
            result = stage.validate_inputs(state, None)

        # Should fail on missing data, not on consistency
        assert result is not None
        assert "Missing text_metadata" in result
        assert not any("Video count drift detected" in msg for msg in caplog.messages)

    def test_no_warning_when_video_ids_empty(self, caplog):
        """caption_results present but no video_ids — skip drift check."""
        state = _make_state(
            num_video_ids=0,
            num_caption_results=10,
            populate_text_metadata=True,
        )
        stage = _get_match_stage()

        with caplog.at_level(logging.WARNING, logger="src.stages.match"):
            result = stage.validate_inputs(state, None)

        assert result is None
        assert not any("Video count drift detected" in msg for msg in caplog.messages)

    def test_does_not_block_pipeline(self):
        """Even with massive drift, validate_inputs returns None (warn-only)."""
        state = _make_state(
            num_video_ids=200,
            num_caption_results=10,
            populate_text_metadata=True,
        )
        stage = _get_match_stage()
        result = stage.validate_inputs(state, None)
        assert result is None, "Consistency check must never block the pipeline"


class TestTextMetadataWithoutCaptionResults:
    """Tests for warning when text_metadata is populated but caption_results is empty."""

    def test_warning_text_metadata_without_caption_results(self, caplog):
        """text_metadata populated but caption_results empty — warn about legacy."""
        state = _make_state(
            num_video_ids=10,
            num_caption_results=0,
        )
        # Manually set text_metadata without caption_results
        state.text_metadata = [
            {"text": "some text", "video_path": "vid_0", "start_time": 0, "end_time": 5}
        ]
        stage = _get_match_stage()

        with caplog.at_level(logging.WARNING, logger="src.stages.match"):
            result = stage.validate_inputs(state, None)

        assert result is None
        assert any("caption_results is empty" in msg for msg in caplog.messages)

    def test_no_warning_when_both_populated(self, caplog):
        """Both text_metadata and caption_results populated — no legacy warning."""
        state = _make_state(
            num_video_ids=10,
            num_caption_results=10,
            populate_text_metadata=True,
        )
        stage = _get_match_stage()

        with caplog.at_level(logging.WARNING, logger="src.stages.match"):
            result = stage.validate_inputs(state, None)

        assert result is None
        assert not any("caption_results is empty" in msg for msg in caplog.messages)
