"""
Regression Tests: Match Stage Stale State References (US-47-002)

Verifies that _run_matching() in MatchStage does NOT access stale attributes
from PipelineState (e.g., state.embeddings, state.embedding_index) and instead
computes embeddings locally and passes embedding_index=None to match_all_segments.

Background: A previous bug had the match stage reading pre-computed embeddings
and embedding_index from PipelineState, which could be stale after checkpoint
restore or stage re-runs. The fix computes them locally in _run_matching().
"""

import inspect
import textwrap
from dataclasses import fields
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.stages.match import MatchStage
from src.state import PipelineState, VoiceoverSegment


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def match_stage():
    return MatchStage()


@pytest.fixture
def minimal_state():
    """PipelineState with only dataclass-defined fields (no extras)."""
    return PipelineState(
        voiceover_segments=[
            VoiceoverSegment(index=0, start=0.0, end=3.0, text="Hello world"),
        ],
        face_preference="neutral",
        location_chapters=[{"name": "intro", "start": 0.0}],
    )


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.cache.cache_dir = ".cache"
    return config


@pytest.fixture
def dummy_embeddings():
    return np.random.rand(3, 128).tolist()


# ============================================================================
# AC1: _run_matching does NOT access state.embeddings or state.embedding_index
# ============================================================================

class TestNoStaleStateAccess:
    """Verify _run_matching never reads stale embedding fields from state."""

    def test_source_does_not_reference_state_embeddings(self):
        """Inspect _run_matching source to confirm no state.embeddings access."""
        source = inspect.getsource(MatchStage._run_matching)
        # Should NOT contain state.embeddings (reading stale embeddings)
        # But state.face_preference is fine
        lines_with_state_embeddings = [
            line.strip() for line in source.splitlines()
            if "state.embeddings" in line
            and "state.embedding" not in line.replace("state.embeddings", "")
            # Exclude comments
            and not line.strip().startswith("#")
        ]
        assert lines_with_state_embeddings == [], (
            f"_run_matching accesses state.embeddings (stale reference): "
            f"{lines_with_state_embeddings}"
        )

    def test_source_does_not_reference_state_embedding_index(self):
        """Inspect _run_matching source to confirm no state.embedding_index access."""
        source = inspect.getsource(MatchStage._run_matching)
        lines_with_state_embedding_index = [
            line.strip() for line in source.splitlines()
            if "state.embedding_index" in line
            and not line.strip().startswith("#")
        ]
        assert lines_with_state_embedding_index == [], (
            f"_run_matching accesses state.embedding_index (stale reference): "
            f"{lines_with_state_embedding_index}"
        )

    def test_state_attribute_access_is_limited(self):
        """Only state.face_preference and state.location_chapters should be accessed."""
        source = inspect.getsource(MatchStage._run_matching)
        # Find all state.X attribute accesses (excluding comments)
        import re
        state_accesses = set()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for match in re.finditer(r'state\.(\w+)', stripped):
                state_accesses.add(match.group(1))

        # Only these are allowed
        allowed = {"face_preference", "location_chapters"}
        unexpected = state_accesses - allowed
        assert not unexpected, (
            f"_run_matching accesses unexpected state attributes: {unexpected}. "
            f"Only {allowed} should be accessed from state."
        )


# ============================================================================
# AC2: Video embeddings computed locally (not read from PipelineState)
# ============================================================================

class TestLocalEmbeddingComputation:
    """Verify embeddings are computed within _run_matching, not from state."""

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    def test_compute_embeddings_called_for_videos(
        self, mock_cache_cls, mock_provider, mock_compute, mock_match,
        match_stage, minimal_state, mock_config
    ):
        """
        Verify compute_embeddings is called with video text, not state embeddings.
        """
        vo_segs = [MagicMock(text="hello")]
        vid_segs = [MagicMock(text="world")]
        fake_emb = np.array([[0.1, 0.2]])

        mock_compute.return_value = fake_emb
        mock_match.return_value = []
        mock_provider.return_value = MagicMock()

        result = match_stage._run_matching(
            vo_segments=vo_segs,
            video_segments=vid_segs,
            all_video_paths=["video1.mp4"],
            state=minimal_state,
            config=mock_config,
            delta_enabled=False,
            force_rematch=False,
        )

        # compute_embeddings should be called twice: once for voiceover, once for video
        assert mock_compute.call_count == 2, (
            f"compute_embeddings called {mock_compute.call_count} times, expected 2 "
            f"(once for voiceover, once for video)"
        )

        # Second call should use video segment texts
        video_call = mock_compute.call_args_list[1]
        assert video_call.kwargs.get("texts") == ["world"], (
            "Second compute_embeddings call should use video segment texts"
        )

    def test_pipeline_state_has_no_embeddings_field(self):
        """PipelineState should not have an 'embeddings' field (removed)."""
        field_names = {f.name for f in fields(PipelineState)}
        assert "embeddings" not in field_names, (
            "PipelineState still has an 'embeddings' field - embeddings should be "
            "computed locally in _run_matching, not stored on state"
        )

    def test_pipeline_state_has_no_embedding_index_field(self):
        """PipelineState should not have an 'embedding_index' field (removed)."""
        field_names = {f.name for f in fields(PipelineState)}
        assert "embedding_index" not in field_names, (
            "PipelineState still has an 'embedding_index' field - it should not "
            "be stored on state"
        )


# ============================================================================
# AC3: match_all_segments receives embedding_index=None and location_chapters
# ============================================================================

class TestMatchAllSegmentsParameters:
    """Verify correct parameters are passed to match_all_segments."""

    def _call_run_matching(self, state, config):
        """Helper to call _run_matching with mocked dependencies and return mock_match."""
        stage = MatchStage()
        vo_segs = [MagicMock(text="hello")]
        vid_segs = [MagicMock(text="world")]
        fake_emb = np.array([[0.1, 0.2]])

        with (
            patch("src.matching.match_all_segments", return_value=[]) as mock_match,
            patch("src.embeddings.compute_embeddings", return_value=fake_emb),
            patch("src.embeddings.get_embedding_provider", return_value=MagicMock()),
            patch("src.utils.CacheManager"),
        ):
            stage._run_matching(
                vo_segments=vo_segs,
                video_segments=vid_segs,
                all_video_paths=["v1.mp4"],
                state=state,
                config=config,
                delta_enabled=False,
                force_rematch=False,
            )
            return mock_match

    def test_embedding_index_is_none(self, minimal_state, mock_config):
        """match_all_segments must receive embedding_index=None."""
        mock_match = self._call_run_matching(minimal_state, mock_config)
        mock_match.assert_called_once()
        call_kwargs = mock_match.call_args
        # Check keyword arg or positional
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("embedding_index") is None, (
                "embedding_index must be None (computed locally, not from state)"
            )
        else:
            # embedding_index is the 8th positional arg (0-indexed: 7)
            assert call_kwargs.args[7] is None

    def test_location_chapters_from_state(self, mock_config):
        """match_all_segments must receive location_chapters from state."""
        chapters = [{"name": "chapter1", "start": 0.0}]
        state = PipelineState(
            voiceover_segments=[
                VoiceoverSegment(index=0, start=0.0, end=3.0, text="test"),
            ],
            location_chapters=chapters,
        )
        mock_match = self._call_run_matching(state, mock_config)
        mock_match.assert_called_once()
        call_kwargs = mock_match.call_args
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("location_chapters") == chapters
        else:
            # location_chapters is the 10th positional arg (0-indexed: 9)
            assert call_kwargs.args[10] == chapters

    def test_location_chapters_fallback_when_missing(self, mock_config):
        """getattr fallback should return None if location_chapters somehow missing."""
        state = PipelineState(
            voiceover_segments=[
                VoiceoverSegment(index=0, start=0.0, end=3.0, text="test"),
            ],
        )
        # location_chapters defaults to [] via dataclass, so getattr returns []
        # The code uses getattr(state, 'location_chapters', None) which will
        # return the actual value (empty list), not None
        mock_match = self._call_run_matching(state, mock_config)
        mock_match.assert_called_once()
        call_kwargs = mock_match.call_args
        if call_kwargs.kwargs:
            lc = call_kwargs.kwargs.get("location_chapters")
            # Should be either None or empty list (both valid for "no chapters")
            assert lc is None or lc == [], (
                f"Expected None or [] for location_chapters, got {lc}"
            )

    def test_face_preference_from_state(self, mock_config):
        """match_all_segments must receive face_preference from state."""
        state = PipelineState(
            voiceover_segments=[
                VoiceoverSegment(index=0, start=0.0, end=3.0, text="test"),
            ],
            face_preference="low",
        )
        mock_match = self._call_run_matching(state, mock_config)
        mock_match.assert_called_once()
        call_kwargs = mock_match.call_args
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("face_preference") == "low"


# ============================================================================
# AC4: _run_matching succeeds with minimal PipelineState
# ============================================================================

class TestMinimalPipelineState:
    """Verify _run_matching works with a bare-minimum PipelineState."""

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    def test_succeeds_with_defaults_only_state(
        self, mock_cache_cls, mock_provider, mock_compute, mock_match, mock_config
    ):
        """
        _run_matching should succeed with a PipelineState that has no extra
        fields beyond the dataclass definition (all defaults).
        """
        state = PipelineState(
            voiceover_segments=[
                VoiceoverSegment(index=0, start=0.0, end=2.0, text="Minimal test"),
            ],
        )
        stage = MatchStage()
        fake_emb = np.array([[0.5, 0.5]])
        mock_compute.return_value = fake_emb
        mock_match.return_value = []
        mock_provider.return_value = MagicMock()

        result = stage._run_matching(
            vo_segments=[MagicMock(text="minimal")],
            video_segments=[MagicMock(text="test")],
            all_video_paths=["v.mp4"],
            state=state,
            config=mock_config,
            delta_enabled=False,
            force_rematch=False,
        )

        assert result == [], "Should return empty list from mocked match_all_segments"
        mock_match.assert_called_once()

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    def test_no_attribute_error_with_fresh_state(
        self, mock_cache_cls, mock_provider, mock_compute, mock_match, mock_config
    ):
        """
        A fresh PipelineState() with zero customization should not cause
        AttributeError in _run_matching.
        """
        state = PipelineState()
        stage = MatchStage()
        fake_emb = np.array([[0.1]])
        mock_compute.return_value = fake_emb
        mock_match.return_value = []
        mock_provider.return_value = MagicMock()

        # Should not raise AttributeError for missing state fields
        result = stage._run_matching(
            vo_segments=[MagicMock(text="a")],
            video_segments=[MagicMock(text="b")],
            all_video_paths=[],
            state=state,
            config=mock_config,
            delta_enabled=False,
            force_rematch=False,
        )
        assert isinstance(result, list)

    def test_no_extra_attributes_on_dataclass(self):
        """
        PipelineState should not have stale attributes like 'embeddings'
        or 'embedding_index' that could tempt usage in match stage.
        """
        field_names = {f.name for f in fields(PipelineState)}
        stale_candidates = {"embeddings", "embedding_index", "video_embeddings"}
        present_stale = stale_candidates & field_names
        # voiceover_embeddings is OK (used elsewhere), but the above should not exist
        assert not present_stale, (
            f"PipelineState contains potentially stale fields: {present_stale}"
        )
