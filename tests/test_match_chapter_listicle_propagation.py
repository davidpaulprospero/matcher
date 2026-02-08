"""
Tests for US-75-010: Propagate chapter and listicle state from detection to match stage.

Verifies that:
- match_all_segments receives listicle_groups parameter from match stage
- TieredMatcher constructor accepts and stores listicle_groups
- segment_chapter_map is passed to TieredMatcher for per-segment chapter lookups
- chapter_source_counts is tracked across segments and passed to scoring
"""

import inspect
from unittest.mock import MagicMock, patch

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
def mock_config():
    config = MagicMock()
    config.cache.cache_dir = ".cache"
    return config


@pytest.fixture
def state_with_listicle_groups():
    """PipelineState with listicle_groups set."""
    state = PipelineState(
        voiceover_segments=[
            VoiceoverSegment(index=0, start=0.0, end=3.0, text="Number one fact"),
            VoiceoverSegment(index=1, start=3.0, end=6.0, text="Number two fact"),
        ],
        face_preference="neutral",
    )
    # Simulate listicle groups detected in _detect_and_store_listicle_groups
    mock_group = MagicMock()
    mock_group.item_label = "fact"
    mock_group.segment_indices = [0, 1]
    state.listicle_groups = [mock_group]
    return state


# ============================================================================
# AC1: match_all_segments receives listicle_groups parameter from match stage
# ============================================================================

class TestListicleGroupsPropagation:
    """Verify listicle_groups flows from match stage to match_all_segments."""

    def _call_run_matching(self, state, config):
        """Helper to call _run_matching and return the mock of match_all_segments."""
        stage = MatchStage()
        vo_segs = [MagicMock(text="hello")]
        vid_segs = [MagicMock(text="world")]
        fake_emb = np.array([[0.1, 0.2]])

        mock_emb_cache = MagicMock()
        mock_emb_cache.get_cached_embeddings.return_value = ([], [], [])

        with (
            patch("src.matching.match_all_segments", return_value=[]) as mock_match,
            patch("src.embeddings.compute_embeddings", return_value=fake_emb),
            patch("src.embeddings.get_embedding_provider", return_value=MagicMock()),
            patch("src.utils.CacheManager"),
            patch("src.embeddings.EmbeddingCache", return_value=mock_emb_cache),
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

    def test_listicle_groups_passed_to_match_all_segments(
        self, state_with_listicle_groups, mock_config
    ):
        """match_all_segments must receive listicle_groups from state."""
        mock_match = self._call_run_matching(
            state_with_listicle_groups, mock_config
        )
        mock_match.assert_called_once()
        call_kwargs = mock_match.call_args.kwargs
        assert "listicle_groups" in call_kwargs, (
            "match_all_segments not called with listicle_groups parameter"
        )
        assert call_kwargs["listicle_groups"] is not None
        assert len(call_kwargs["listicle_groups"]) == 1

    def test_listicle_groups_none_when_not_set(self, mock_config):
        """When state has no listicle_groups, None should be passed."""
        state = PipelineState(
            voiceover_segments=[
                VoiceoverSegment(index=0, start=0.0, end=3.0, text="test"),
            ],
        )
        mock_match = self._call_run_matching(state, mock_config)
        mock_match.assert_called_once()
        call_kwargs = mock_match.call_args.kwargs
        # listicle_groups defaults to [] on PipelineState, getattr returns []
        lgs = call_kwargs.get("listicle_groups")
        assert lgs is None or lgs == [], (
            f"Expected None or [] for listicle_groups when not set, got {lgs}"
        )

    def test_match_all_segments_signature_accepts_listicle_groups(self):
        """match_all_segments function signature must include listicle_groups."""
        from src.matching.main import match_all_segments
        sig = inspect.signature(match_all_segments)
        assert "listicle_groups" in sig.parameters, (
            "match_all_segments missing 'listicle_groups' parameter"
        )


# ============================================================================
# AC2: TieredMatcher constructor accepts and stores listicle_groups
# ============================================================================

class TestTieredMatcherListicleGroups:
    """Verify TieredMatcher receives and stores chapter/listicle state."""

    def test_constructor_accepts_listicle_groups(self):
        """TieredMatcher.__init__ must accept listicle_groups parameter."""
        from src.matching.tiered_matcher import TieredMatcher
        sig = inspect.signature(TieredMatcher.__init__)
        assert "listicle_groups" in sig.parameters, (
            "TieredMatcher.__init__ missing 'listicle_groups' parameter"
        )

    @patch("src.matching.tiered_matcher.get_config")
    def test_stores_listicle_groups(self, mock_get_config):
        """TieredMatcher should store listicle_groups as instance attribute."""
        from src.matching.tiered_matcher import TieredMatcher

        # Build minimal mock config
        mock_cfg = MagicMock()
        mock_cfg.matching.max_clip_reuse = 3
        mock_cfg.matching.reuse_penalty = 0.1
        mock_get_config.return_value = mock_cfg

        fake_groups = [MagicMock(item_label="step")]
        matcher = TieredMatcher(config=mock_cfg, listicle_groups=fake_groups)

        assert matcher.listicle_groups == fake_groups, (
            "TieredMatcher did not store listicle_groups"
        )

    @patch("src.matching.tiered_matcher.get_config")
    def test_defaults_listicle_groups_to_empty_list(self, mock_get_config):
        """TieredMatcher should default listicle_groups to [] when not provided."""
        from src.matching.tiered_matcher import TieredMatcher

        mock_cfg = MagicMock()
        mock_cfg.matching.max_clip_reuse = 3
        mock_cfg.matching.reuse_penalty = 0.1
        mock_get_config.return_value = mock_cfg

        matcher = TieredMatcher(config=mock_cfg)
        assert matcher.listicle_groups == [], (
            "TieredMatcher should default listicle_groups to []"
        )


# ============================================================================
# AC3: segment_chapter_map is passed to TieredMatcher
# ============================================================================

class TestSegmentChapterMapPropagation:
    """Verify segment_chapter_map flows from match_all_segments to TieredMatcher."""

    def test_tiered_matcher_has_segment_chapter_map_attribute(self):
        """TieredMatcher should have segment_chapter_map instance attribute."""
        from src.matching.tiered_matcher import TieredMatcher
        sig = inspect.signature(TieredMatcher.__init__)
        # Check via __init__ source or just instantiate
        mock_cfg = MagicMock()
        mock_cfg.matching.max_clip_reuse = 3
        mock_cfg.matching.reuse_penalty = 0.1
        with patch("src.matching.tiered_matcher.get_config", return_value=mock_cfg):
            matcher = TieredMatcher(config=mock_cfg)
        assert hasattr(matcher, "segment_chapter_map"), (
            "TieredMatcher missing 'segment_chapter_map' attribute"
        )

    def test_segment_chapter_map_defaults_to_empty_dict(self):
        """segment_chapter_map should default to empty dict."""
        from src.matching.tiered_matcher import TieredMatcher
        mock_cfg = MagicMock()
        mock_cfg.matching.max_clip_reuse = 3
        mock_cfg.matching.reuse_penalty = 0.1
        with patch("src.matching.tiered_matcher.get_config", return_value=mock_cfg):
            matcher = TieredMatcher(config=mock_cfg)
        assert matcher.segment_chapter_map == {}, (
            "segment_chapter_map should default to empty dict"
        )

    def test_match_all_segments_assigns_segment_chapter_map(self):
        """match_all_segments source should assign segment_chapter_map to matcher."""
        from src.matching.main import match_all_segments
        source = inspect.getsource(match_all_segments)
        assert "matcher.segment_chapter_map" in source, (
            "match_all_segments does not assign segment_chapter_map to TieredMatcher"
        )


# ============================================================================
# AC4: chapter_source_counts is tracked across segments and passed to scoring
# ============================================================================

class TestChapterSourceCountsTracking:
    """Verify chapter_source_counts is tracked in TieredMatcher."""

    def test_chapter_source_counts_initialized(self):
        """TieredMatcher should initialize _chapter_source_counts as empty dict."""
        from src.matching.tiered_matcher import TieredMatcher
        mock_cfg = MagicMock()
        mock_cfg.matching.max_clip_reuse = 3
        mock_cfg.matching.reuse_penalty = 0.1
        with patch("src.matching.tiered_matcher.get_config", return_value=mock_cfg):
            matcher = TieredMatcher(config=mock_cfg)
        assert hasattr(matcher, "_chapter_source_counts"), (
            "TieredMatcher missing '_chapter_source_counts' attribute"
        )
        assert matcher._chapter_source_counts == {}

    def test_update_chapter_source_counts_method_exists(self):
        """TieredMatcher should have _update_chapter_source_counts method."""
        from src.matching.tiered_matcher import TieredMatcher
        assert hasattr(TieredMatcher, "_update_chapter_source_counts"), (
            "TieredMatcher missing '_update_chapter_source_counts' method"
        )

    def test_chapter_source_counts_used_in_scoring(self):
        """TieredMatcher source should pass _chapter_source_counts to scoring."""
        from src.matching.tiered_matcher import TieredMatcher
        source = inspect.getsource(TieredMatcher)
        assert "chapter_source_counts=self._chapter_source_counts" in source, (
            "TieredMatcher does not pass _chapter_source_counts to scoring functions"
        )


# ============================================================================
# AC5: listicle_groups flows end-to-end from TieredMatcher to scoring
# ============================================================================

class TestListicleGroupsScoringIntegration:
    """Verify listicle_groups is used in TieredMatcher scoring calls."""

    def test_listicle_groups_passed_to_scoring(self):
        """TieredMatcher source should pass listicle_groups to scoring."""
        from src.matching.tiered_matcher import TieredMatcher
        source = inspect.getsource(TieredMatcher)
        assert "listicle_groups=self.listicle_groups" in source, (
            "TieredMatcher does not pass listicle_groups to scoring functions"
        )

    def test_match_all_segments_passes_listicle_groups_to_constructor(self):
        """match_all_segments should pass listicle_groups to TieredMatcher constructor."""
        from src.matching.main import match_all_segments
        source = inspect.getsource(match_all_segments)
        assert "listicle_groups=listicle_groups" in source, (
            "match_all_segments does not pass listicle_groups to TieredMatcher"
        )
