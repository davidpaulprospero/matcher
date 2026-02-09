"""
Tests for US-85-005: Within-stage resumption support for MATCH stage.

Validates:
- Intermediate checkpointing saves partial results every N segments
- _restore_partial_matches loads partial results and returns correct start_index
- Interrupted MATCH resumes from last checkpointed segment, not from scratch
- Final match results are identical whether run to completion or interrupted+resumed
- partial_matches cleared after successful completion
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from copy import deepcopy

from src.utils import SRTSegment, Match, MatchResult
from src.matching.serialization import serialize_match_for_match_stage

# Patch targets
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


def _make_match_result(vo_seg: SRTSegment, vid_seg: SRTSegment,
                       confidence: float = 0.8) -> MatchResult:
    """Create a minimal MatchResult for mock returns."""
    primary = Match(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        video_scene=None,
        confidence=confidence,
        reasoning="mock match",
    )
    return MatchResult(primary_match=primary)


def _make_mock_config():
    """Create a mock Config object with all match_all_segments attributes."""
    config = Mock()
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

    oc = Mock()
    oc.include_strategy_tracks = False
    vc = Mock()
    vc.enforce_timeline_variety = False
    vc.timeline_variety_window = 600.0
    vc.max_source_repeats_in_window = 1
    oc.variety = vc
    config.output = oc

    return config


def _make_mock_checkpoint(stage_data=None):
    """Create a mock CheckpointManager."""
    cp = Mock()
    _data = stage_data or {}
    cp.get_stage_data.return_value = _data
    cp.save_intermediate = Mock()
    return cp


def _setup_mocks_multi(MockTieredMatcher, MockEmbeddingSearch, vo_segs, vid_segs):
    """Set up mocks that return different matches per segment call."""
    mock_matcher = MockTieredMatcher.return_value
    results = []
    for vo, vid in zip(vo_segs, vid_segs):
        results.append(_make_match_result(vo, vid, confidence=0.7 + 0.05 * vo.index))
    mock_matcher.match_segment.side_effect = results
    mock_matcher.local_provider = None
    mock_matcher.enforce_chapter_source_diversity.side_effect = lambda x: x

    mock_search = MockEmbeddingSearch.from_matching_config.return_value
    mock_search.search.side_effect = [[(vid, 0.9)] for vid in vid_segs]

    return mock_matcher, mock_search


class TestProgressCallback:
    """Test that match_all_segments invokes progress_callback at correct intervals."""

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_callback_invoked_every_n_segments(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """progress_callback is called at the correct interval."""
        config = _make_mock_config()
        cache = Mock()

        num_segments = 10
        vo_segs = [_make_segment(i, f"seg {i}", start_time=float(i * 5), end_time=float(i * 5 + 5))
                   for i in range(num_segments)]
        vid_segs = [_make_segment(i, f"vid {i}", source_file=f"v{i}.mp4")
                   for i in range(num_segments)]

        _setup_mocks_multi(MockTieredMatcher, MockEmbeddingSearch, vo_segs, vid_segs)

        callback = Mock()

        from src.matching.main import match_all_segments
        results = match_all_segments(
            voiceover_segments=vo_segs,
            video_segments=vid_segs,
            voiceover_embeddings=[[0.1]] * num_segments,
            video_embeddings=[[0.2]] * num_segments,
            scenes=None,
            config=config,
            cache=cache,
            progress_callback=callback,
        )

        assert len(results) == num_segments
        # Callback should be called for every segment (filtering is done by the callback itself)
        assert callback.call_count == num_segments
        # Verify the callback receives index and results list
        for call_idx, c in enumerate(callback.call_args_list):
            assert c[0][0] == call_idx  # first arg is segment index
            assert isinstance(c[0][1], list)  # second arg is results list


class TestStartIndexResumption:
    """Test that start_index parameter skips already-processed segments."""

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_start_index_skips_segments(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """Segments before start_index are skipped."""
        config = _make_mock_config()
        cache = Mock()

        num_segments = 6
        vo_segs = [_make_segment(i, f"seg {i}", start_time=float(i * 5), end_time=float(i * 5 + 5))
                   for i in range(num_segments)]
        vid_segs = [_make_segment(i, f"vid {i}", source_file=f"v{i}.mp4")
                   for i in range(num_segments)]

        # Only set up mocks for segments 3-5 (the ones that will actually be processed)
        mock_matcher = MockTieredMatcher.return_value
        remaining_results = [
            _make_match_result(vo_segs[i], vid_segs[i]) for i in range(3, num_segments)
        ]
        mock_matcher.match_segment.side_effect = remaining_results
        mock_matcher.local_provider = None
        mock_matcher.enforce_chapter_source_diversity.side_effect = lambda x: x

        mock_search = MockEmbeddingSearch.from_matching_config.return_value
        mock_search.search.side_effect = [[(vid_segs[i], 0.9)] for i in range(3, num_segments)]

        # Provide prior results for segments 0-2
        prior = [_make_match_result(vo_segs[i], vid_segs[i]) for i in range(3)]

        from src.matching.main import match_all_segments
        results = match_all_segments(
            voiceover_segments=vo_segs,
            video_segments=vid_segs,
            voiceover_embeddings=[[0.1]] * num_segments,
            video_embeddings=[[0.2]] * num_segments,
            scenes=None,
            config=config,
            cache=cache,
            start_index=3,
            prior_results=prior,
        )

        # Total results = 3 prior + 3 new
        assert len(results) == num_segments
        # match_segment should have been called only 3 times (segments 3, 4, 5)
        assert mock_matcher.match_segment.call_count == 3


class TestRestorePartialMatches:
    """Test _restore_partial_matches method of MatchStage."""

    def test_no_partial_data_returns_zero(self):
        """Returns (0, None) when no partial_matches in checkpoint."""
        from src.stages.match import MatchStage
        stage = MatchStage()
        cp = _make_mock_checkpoint({})
        start_idx, prior = stage._restore_partial_matches(cp)
        assert start_idx == 0
        assert prior is None

    def test_partial_data_restores_correctly(self):
        """Returns correct start_index and restored matches."""
        from src.stages.match import MatchStage
        stage = MatchStage()

        # Build serialized partial matches
        vo_seg = _make_segment(0, "test vo", start_time=0.0, end_time=5.0)
        vid_seg = _make_segment(0, "test vid", source_file="v1.mp4")
        match_result = _make_match_result(vo_seg, vid_seg)

        serialized = [serialize_match_for_match_stage(match_result, 0)]

        partial_data = {
            'partial_matches': {
                'last_completed_index': 0,
                'match_count': 1,
                'matches': serialized,
            }
        }
        cp = _make_mock_checkpoint(partial_data)

        start_idx, prior = stage._restore_partial_matches(cp)
        assert start_idx == 1  # Resume from segment 1
        assert prior is not None
        assert len(prior) == 1

    def test_corrupt_partial_data_returns_zero(self):
        """Returns (0, None) when partial_matches has invalid data."""
        from src.stages.match import MatchStage
        stage = MatchStage()

        partial_data = {
            'partial_matches': {
                'last_completed_index': -1,
                'matches': [],
            }
        }
        cp = _make_mock_checkpoint(partial_data)

        start_idx, prior = stage._restore_partial_matches(cp)
        assert start_idx == 0
        assert prior is None


class TestClearPartialMatches:
    """Test _clear_partial_matches removes partial data."""

    def test_clears_partial_matches_key(self):
        """partial_matches key is removed after successful completion."""
        from src.stages.match import MatchStage
        stage = MatchStage()

        data_with_partial = {
            'match_count': 5,
            'partial_matches': {'last_completed_index': 2, 'matches': []},
        }
        cp = _make_mock_checkpoint(data_with_partial)

        stage._clear_partial_matches(cp)

        # save_intermediate should have been called with data sans partial_matches
        cp.save_intermediate.assert_called_once()
        saved_data = cp.save_intermediate.call_args[0][1]
        assert 'partial_matches' not in saved_data
        assert saved_data['match_count'] == 5


class TestMakeCheckpointCallback:
    """Test _make_checkpoint_callback creates correct callback."""

    def test_callback_only_fires_at_interval(self):
        """Callback only saves at multiples of interval."""
        from src.stages.match import MatchStage
        stage = MatchStage()

        cp = _make_mock_checkpoint()
        callback = stage._make_checkpoint_callback(cp, interval=3)

        # Create some mock match results
        vo_seg = _make_segment(0, "test")
        vid_seg = _make_segment(0, "vid", source_file="v1.mp4")
        mock_results = [_make_match_result(vo_seg, vid_seg)]

        # Call at indices 0, 1, 2, 3, 4, 5
        for i in range(6):
            callback(i, mock_results)

        # Only indices 2 and 5 should trigger save (index+1 % 3 == 0)
        assert cp.save_intermediate.call_count == 2


class TestInterruptAndResume:
    """Integration test: simulate MATCH interruption and verify resumption."""

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_interrupted_match_resumes_correctly(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """Simulate interruption at segment 3 of 6 and verify resume produces same results."""
        config = _make_mock_config()
        cache = Mock()

        num_segments = 6
        vo_segs = [_make_segment(i, f"seg {i}", start_time=float(i * 5), end_time=float(i * 5 + 5))
                   for i in range(num_segments)]
        vid_segs = [_make_segment(i, f"vid {i}", source_file=f"v{i}.mp4")
                   for i in range(num_segments)]

        # Define deterministic match results
        all_match_results = [
            _make_match_result(vo_segs[i], vid_segs[i], confidence=0.7 + 0.02 * i)
            for i in range(num_segments)
        ]

        from src.matching.main import match_all_segments

        # --- Full run (no interruption) ---
        mock_matcher = MockTieredMatcher.return_value
        mock_matcher.match_segment.side_effect = deepcopy(all_match_results)
        mock_matcher.local_provider = None
        mock_matcher.enforce_chapter_source_diversity.side_effect = lambda x: x

        mock_search = MockEmbeddingSearch.from_matching_config.return_value
        mock_search.search.side_effect = [[(vid_segs[i], 0.9)] for i in range(num_segments)]

        full_results = match_all_segments(
            voiceover_segments=vo_segs,
            video_segments=vid_segs,
            voiceover_embeddings=[[0.1]] * num_segments,
            video_embeddings=[[0.2]] * num_segments,
            scenes=None,
            config=config,
            cache=cache,
        )

        assert len(full_results) == num_segments

        # --- Simulated interrupted run (only first 3 segments) ---
        interrupt_at = 3
        mock_matcher.match_segment.side_effect = deepcopy(all_match_results[:interrupt_at])
        mock_search.search.side_effect = [[(vid_segs[i], 0.9)] for i in range(interrupt_at)]

        # Capture the partial results by running with a callback that records them
        captured_results = []

        def capture_callback(idx, results):
            if idx == interrupt_at - 1:
                captured_results.extend(deepcopy(results))

        partial_results = match_all_segments(
            voiceover_segments=vo_segs[:interrupt_at],
            video_segments=vid_segs,
            voiceover_embeddings=[[0.1]] * interrupt_at,
            video_embeddings=[[0.2]] * num_segments,
            scenes=None,
            config=config,
            cache=cache,
            progress_callback=capture_callback,
        )

        assert len(partial_results) == interrupt_at

        # --- Resumed run (segments 3-5) ---
        mock_matcher.match_segment.side_effect = deepcopy(all_match_results[interrupt_at:])
        mock_matcher.enforce_chapter_source_diversity.side_effect = lambda x: x
        mock_search.search.side_effect = [
            [(vid_segs[i], 0.9)] for i in range(interrupt_at, num_segments)
        ]

        resumed_results = match_all_segments(
            voiceover_segments=vo_segs,
            video_segments=vid_segs,
            voiceover_embeddings=[[0.1]] * num_segments,
            video_embeddings=[[0.2]] * num_segments,
            scenes=None,
            config=config,
            cache=cache,
            start_index=interrupt_at,
            prior_results=partial_results,
        )

        # Same total count
        assert len(resumed_results) == len(full_results)

        # Verify match confidences are identical
        for i in range(num_segments):
            assert (
                resumed_results[i].primary_match.confidence
                == full_results[i].primary_match.confidence
            ), f"Confidence mismatch at segment {i}"


class TestResultsIdenticalFullVsResumed:
    """Verify final results are identical whether run to completion or interrupted+resumed."""

    @patch(PATCH_EMBEDDING_SEARCH)
    @patch(PATCH_TIERED_MATCHER)
    @patch(PATCH_STRATEGY_MATCHER)
    def test_results_match_across_runs(
        self, MockStrategyMatcher, MockTieredMatcher, MockEmbeddingSearch
    ):
        """Full run and interrupted+resumed run produce identical match results."""
        config = _make_mock_config()
        cache = Mock()

        num_segments = 8
        vo_segs = [_make_segment(i, f"seg {i}", start_time=float(i * 5), end_time=float(i * 5 + 5))
                   for i in range(num_segments)]
        vid_segs = [_make_segment(i, f"vid {i}", source_file=f"v{i}.mp4")
                   for i in range(num_segments)]

        all_match_results = [
            _make_match_result(vo_segs[i], vid_segs[i], confidence=0.6 + 0.04 * i)
            for i in range(num_segments)
        ]

        from src.matching.main import match_all_segments

        # Full run
        MockTieredMatcher.return_value.match_segment.side_effect = deepcopy(all_match_results)
        MockTieredMatcher.return_value.local_provider = None
        MockTieredMatcher.return_value.enforce_chapter_source_diversity.side_effect = lambda x: x
        MockEmbeddingSearch.from_matching_config.return_value.search.side_effect = [
            [(vid_segs[i], 0.9)] for i in range(num_segments)
        ]

        full_results = match_all_segments(
            voiceover_segments=vo_segs,
            video_segments=vid_segs,
            voiceover_embeddings=[[0.1]] * num_segments,
            video_embeddings=[[0.2]] * num_segments,
            scenes=None,
            config=config,
            cache=cache,
        )

        # Interrupted at segment 4, then resumed
        midpoint = 4
        prior = deepcopy(full_results[:midpoint])

        MockTieredMatcher.return_value.match_segment.side_effect = deepcopy(
            all_match_results[midpoint:]
        )
        MockTieredMatcher.return_value.enforce_chapter_source_diversity.side_effect = lambda x: x
        MockEmbeddingSearch.from_matching_config.return_value.search.side_effect = [
            [(vid_segs[i], 0.9)] for i in range(midpoint, num_segments)
        ]

        resumed_results = match_all_segments(
            voiceover_segments=vo_segs,
            video_segments=vid_segs,
            voiceover_embeddings=[[0.1]] * num_segments,
            video_embeddings=[[0.2]] * num_segments,
            scenes=None,
            config=config,
            cache=cache,
            start_index=midpoint,
            prior_results=prior,
        )

        assert len(resumed_results) == len(full_results)
        for i in range(num_segments):
            assert (
                resumed_results[i].primary_match.confidence
                == full_results[i].primary_match.confidence
            ), f"Mismatch at segment {i}: {resumed_results[i].primary_match.confidence} != {full_results[i].primary_match.confidence}"
            assert (
                resumed_results[i].primary_match.video_segment.source_file
                == full_results[i].primary_match.video_segment.source_file
            ), f"Source file mismatch at segment {i}"
