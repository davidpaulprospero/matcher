"""
Contract validation tests for match_all_segments().

Validates the parameter contract of the public match_all_segments() API:
- location_chapters=None and video_locations=None are truly optional
- embedding_index=None works without error
- Empty voiceover_segments returns empty list
- Empty video_segments handled gracefully
- face_preference parameter flows through to TieredMatcher

Uses mocked TieredMatcher and EmbeddingSearch to verify parameter flow
without requiring actual embeddings or LLM calls.
"""

import pytest
from unittest.mock import Mock, patch

from src.utils import SRTSegment, Match, MatchResult

# Patch targets:
# - TieredMatcher: imported locally inside match_all_segments (line 86 of main.py)
#   so we patch at its origin: src.matching.tiered_matcher.TieredMatcher
# - EmbeddingSearch and StrategyMatcher: imported at module level in main.py
#   so we patch on the main module: src.matching.main.EmbeddingSearch / StrategyMatcher
PATCH_TIERED_MATCHER = "src.matching.tiered_matcher.TieredMatcher"
PATCH_EMBEDDING_SEARCH = "src.matching.main.EmbeddingSearch"
PATCH_STRATEGY_MATCHER = "src.matching.main.StrategyMatcher"


def _make_segment(index: int, text: str = "test", source_file: str = "video1.mp4",
                  start_time: float = 0.0, end_time: float = 5.0) -> SRTSegment:
    """Create a minimal SRTSegment for testing."""
    return SRTSegment(
        index=index,
        start_time=start_time,
        end_time=end_time,
        text=text,
        source_file=source_file,
    )


def _make_match_result(vo_seg: SRTSegment, vid_seg: SRTSegment) -> MatchResult:
    """Create a minimal MatchResult for mock returns."""
    primary = Match(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        video_scene=None,
        confidence=0.8,
        reasoning="mock match",
    )
    return MatchResult(primary_match=primary)


def _make_mock_config():
    """Create a mock Config object with all attributes match_all_segments accesses."""
    config = Mock()

    # matching config
    mc = Mock()
    mc.embedding_candidates = 5
    mc.llm_rerank_candidates = 3
    mc.max_clip_reuse = 2
    mc.reuse_penalty = 0.1
    mc.min_confidence = 0.3
    mc.context_window = 1
    mc.duration_scoring_enabled = False
    mc.clip_hard_block = False
    mc.use_local_for_review = False
    config.matching = mc

    # output config
    oc = Mock()
    oc.include_strategy_tracks = False

    # variety config (as object with getattr)
    vc = Mock()
    vc.enforce_timeline_variety = False
    vc.timeline_variety_window = 600.0
    vc.max_source_repeats_in_window = 1
    oc.variety = vc

    config.output = oc

    return config


def _make_mock_cache():
    """Create a mock CacheManager."""
    return Mock()


def _setup_mocks(MockTieredMatcher, MockEmbeddingSearch, vo_seg, vid_seg):
    """Common mock setup for most tests."""
    mock_matcher = MockTieredMatcher.return_value
    mock_matcher.match_segment.return_value = _make_match_result(vo_seg, vid_seg)
    mock_matcher.local_provider = None
    # US-77-007: diversity pass-through (returns input unchanged)
    mock_matcher.enforce_chapter_source_diversity.side_effect = lambda x: x

    mock_search = MockEmbeddingSearch.from_matching_config.return_value
    mock_search.search.return_value = [(vid_seg, 0.9)]

    return mock_matcher, mock_search


class TestLocationParamsOptional:
    """Contract: location_chapters=None and video_locations=None must work."""

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_none_location_chapters_and_video_locations(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """match_all_segments succeeds with location_chapters=None and video_locations=None."""
        config = _make_mock_config()
        cache = _make_mock_cache()

        vo_seg = _make_segment(0, "Hello world", start_time=0.0, end_time=3.0)
        vid_seg = _make_segment(0, "Video text", source_file="v1.mp4")
        mock_matcher, _ = _setup_mocks(MockTieredMatcher, MockEmbeddingSearch, vo_seg, vid_seg)

        from src.matching.main import match_all_segments

        results = match_all_segments(
            voiceover_segments=[vo_seg],
            video_segments=[vid_seg],
            voiceover_embeddings=[[0.1, 0.2]],
            video_embeddings=[[0.3, 0.4]],
            scenes=None,
            config=config,
            cache=cache,
            embedding_index=None,
            face_preference="neutral",
            video_topics=None,
            location_chapters=None,
            video_locations=None,
        )

        assert isinstance(results, list)
        assert len(results) == 1

        # Verify set_location_chapters was NOT called (None should skip it)
        mock_matcher.set_location_chapters.assert_not_called()
        mock_matcher.set_video_locations.assert_not_called()


class TestEmbeddingIndexNone:
    """Contract: embedding_index=None must work (brute-force fallback)."""

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_none_embedding_index_succeeds(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """match_all_segments succeeds with embedding_index=None."""
        config = _make_mock_config()
        cache = _make_mock_cache()

        vo_seg = _make_segment(0, "Test segment", start_time=0.0, end_time=3.0)
        vid_seg = _make_segment(0, "Video", source_file="v1.mp4")
        _setup_mocks(MockTieredMatcher, MockEmbeddingSearch, vo_seg, vid_seg)

        from src.matching.main import match_all_segments

        results = match_all_segments(
            voiceover_segments=[vo_seg],
            video_segments=[vid_seg],
            voiceover_embeddings=[[0.1, 0.2]],
            video_embeddings=[[0.3, 0.4]],
            scenes=None,
            config=config,
            cache=cache,
            embedding_index=None,
        )

        assert isinstance(results, list)
        assert len(results) == 1

        # Verify EmbeddingSearch.from_matching_config was called with None index
        MockEmbeddingSearch.from_matching_config.assert_called_once()
        call_args = MockEmbeddingSearch.from_matching_config.call_args
        # 4th positional arg is embedding_index
        assert call_args[0][3] is None


class TestEmptyVoiceoverSegments:
    """Contract: empty voiceover_segments returns empty results list."""

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_empty_voiceover_segments_returns_empty(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """match_all_segments with empty voiceover_segments returns empty list."""
        config = _make_mock_config()
        cache = _make_mock_cache()

        vid_seg = _make_segment(0, "Video", source_file="v1.mp4")

        mock_matcher = MockTieredMatcher.return_value
        mock_matcher.local_provider = None
        mock_matcher.enforce_chapter_source_diversity.side_effect = lambda x: x

        from src.matching.main import match_all_segments

        results = match_all_segments(
            voiceover_segments=[],
            video_segments=[vid_seg],
            voiceover_embeddings=[],
            video_embeddings=[[0.3, 0.4]],
            scenes=None,
            config=config,
            cache=cache,
        )

        assert isinstance(results, list)
        assert len(results) == 0

        # TieredMatcher.match_segment should never be called
        mock_matcher.match_segment.assert_not_called()


class TestEmptyVideoSegments:
    """Contract: empty video_segments handled gracefully."""

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_empty_video_segments_no_crash(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """match_all_segments with empty video_segments doesn't raise."""
        config = _make_mock_config()
        cache = _make_mock_cache()

        vo_seg = _make_segment(0, "Hello world", start_time=0.0, end_time=3.0)

        mock_matcher = MockTieredMatcher.return_value
        mock_matcher.local_provider = None
        mock_matcher.enforce_chapter_source_diversity.side_effect = lambda x: x
        mock_match = _make_match_result(
            vo_seg, _make_segment(0, "placeholder", source_file="fake.mp4")
        )
        mock_matcher.match_segment.return_value = mock_match

        mock_search = MockEmbeddingSearch.from_matching_config.return_value
        mock_search.search.return_value = []

        from src.matching.main import match_all_segments

        results = match_all_segments(
            voiceover_segments=[vo_seg],
            video_segments=[],
            voiceover_embeddings=[[0.1, 0.2]],
            video_embeddings=[],
            scenes=None,
            config=config,
            cache=cache,
        )

        assert isinstance(results, list)
        assert len(results) == 1


class TestFacePreferenceFlowThrough:
    """Contract: face_preference parameter flows through to TieredMatcher."""

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_face_preference_more_set_on_matcher(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """face_preference='more' is assigned to matcher.face_preference."""
        config = _make_mock_config()
        cache = _make_mock_cache()

        vo_seg = _make_segment(0, "Test", start_time=0.0, end_time=3.0)
        vid_seg = _make_segment(0, "Video", source_file="v1.mp4")
        mock_matcher, _ = _setup_mocks(MockTieredMatcher, MockEmbeddingSearch, vo_seg, vid_seg)

        from src.matching.main import match_all_segments

        match_all_segments(
            voiceover_segments=[vo_seg],
            video_segments=[vid_seg],
            voiceover_embeddings=[[0.1, 0.2]],
            video_embeddings=[[0.3, 0.4]],
            scenes=None,
            config=config,
            cache=cache,
            face_preference="more",
        )

        assert mock_matcher.face_preference == "more"

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_face_preference_defaults_to_neutral(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """face_preference defaults to 'neutral' when not provided."""
        config = _make_mock_config()
        cache = _make_mock_cache()

        vo_seg = _make_segment(0, "Test", start_time=0.0, end_time=3.0)
        vid_seg = _make_segment(0, "Video", source_file="v1.mp4")
        mock_matcher, _ = _setup_mocks(MockTieredMatcher, MockEmbeddingSearch, vo_seg, vid_seg)

        from src.matching.main import match_all_segments

        match_all_segments(
            voiceover_segments=[vo_seg],
            video_segments=[vid_seg],
            voiceover_embeddings=[[0.1, 0.2]],
            video_embeddings=[[0.3, 0.4]],
            scenes=None,
            config=config,
            cache=cache,
        )

        assert mock_matcher.face_preference == "neutral"

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_face_preference_none_value(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """face_preference='none' (avoid faces) flows through correctly."""
        config = _make_mock_config()
        cache = _make_mock_cache()

        vo_seg = _make_segment(0, "Test", start_time=0.0, end_time=3.0)
        vid_seg = _make_segment(0, "Video", source_file="v1.mp4")
        mock_matcher, _ = _setup_mocks(MockTieredMatcher, MockEmbeddingSearch, vo_seg, vid_seg)

        from src.matching.main import match_all_segments

        match_all_segments(
            voiceover_segments=[vo_seg],
            video_segments=[vid_seg],
            voiceover_embeddings=[[0.1, 0.2]],
            video_embeddings=[[0.3, 0.4]],
            scenes=None,
            config=config,
            cache=cache,
            face_preference="none",
        )

        assert mock_matcher.face_preference == "none"
