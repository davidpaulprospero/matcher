"""
Tests for cross-chapter relevance matrix and boost (US-71-005).

Tests:
- compute_relevance_matrix: Jaccard similarity over keyword sets
- apply_cross_chapter_relevance_boost: proportional boost from relevance score
"""

import pytest
from unittest.mock import Mock

from src.matching.scoring import compute_relevance_matrix, MatchScoring
from src.utils import SRTSegment


def _mock_config(relevance_boost_weight=0.1, enabled=True):
    """Create a minimal mock config for MatchScoring with chapter grouping."""
    config = Mock()
    matching = Mock()
    matching.multimodal_enabled = True
    matching.multimodal_weights = None
    matching.pool_normalization_enabled = True
    matching.chapter_matching_enabled = False
    matching.topic_mismatch_penalty = 0.15
    matching.broll_boost = 0.1
    matching.caption_quality_adjustment_enabled = True
    matching.caption_quality_high_boost = 0.05
    matching.caption_quality_low_penalty = 0.1
    matching.apply_timing_penalty = True
    matching.skip_llm_threshold = 0.85

    # Scoring config with confidence_floor (needed for duck-type check)
    scoring = Mock()
    scoring.confidence_floor = 0.05
    scoring.low_confidence_warning_threshold = 0.15
    matching.scoring = scoring

    # Chapter grouping config
    chapter_grouping = Mock()
    chapter_grouping.enabled = enabled
    chapter_grouping.source_consistency_boost = 0.03
    chapter_grouping.coherence_penalty_threshold = 5
    chapter_grouping.relevance_boost_weight = relevance_boost_weight
    matching.chapter_grouping = chapter_grouping

    config.matching = matching

    global_cache = Mock()
    global_cache.current_project_boost = 0.1
    config.global_cache = global_cache

    return config


# ============================================================================
# compute_relevance_matrix tests
# ============================================================================


class TestComputeRelevanceMatrix:
    """Tests for compute_relevance_matrix()."""

    def test_identical_keywords_gives_score_1(self):
        """When both chapters share the same keywords, score = 1.0."""
        vo = [['cats', 'dogs', 'pets']]
        vid = [['cats', 'dogs', 'pets']]
        matrix = compute_relevance_matrix(vo, vid)
        assert len(matrix) == 1
        assert len(matrix[0]) == 1
        assert matrix[0][0] == pytest.approx(1.0)

    def test_disjoint_keywords_gives_score_0(self):
        """When chapters share no keywords, score = 0.0."""
        vo = [['cats', 'dogs']]
        vid = [['cars', 'trucks']]
        matrix = compute_relevance_matrix(vo, vid)
        assert matrix[0][0] == pytest.approx(0.0)

    def test_partial_overlap_jaccard(self):
        """Partial overlap: intersection=1, union=3 -> 1/3."""
        vo = [['cats', 'dogs']]
        vid = [['dogs', 'fish']]
        matrix = compute_relevance_matrix(vo, vid)
        # intersection = {'dogs'}, union = {'cats','dogs','fish'}
        assert matrix[0][0] == pytest.approx(1.0 / 3.0)

    def test_case_insensitive(self):
        """Keywords should be compared case-insensitively."""
        vo = [['CATS', 'Dogs']]
        vid = [['cats', 'dogs']]
        matrix = compute_relevance_matrix(vo, vid)
        assert matrix[0][0] == pytest.approx(1.0)

    def test_multi_chapter_matrix_shape(self):
        """Matrix shape should be (vo_chapters x vid_chapters)."""
        vo = [['a', 'b'], ['c', 'd'], ['e']]
        vid = [['a'], ['c', 'e']]
        matrix = compute_relevance_matrix(vo, vid)
        assert len(matrix) == 3  # 3 vo chapters
        assert all(len(row) == 2 for row in matrix)  # 2 vid chapters

    def test_multi_chapter_specific_values(self):
        """Verify specific cells in a multi-chapter matrix."""
        vo = [['paris', 'france', 'eiffel'], ['tokyo', 'japan', 'sushi']]
        vid = [['paris', 'france', 'wine'], ['tokyo', 'ramen', 'japan']]
        matrix = compute_relevance_matrix(vo, vid)

        # vo[0] vs vid[0]: intersection={paris,france}, union={paris,france,eiffel,wine} = 2/4
        assert matrix[0][0] == pytest.approx(2.0 / 4.0)
        # vo[0] vs vid[1]: intersection={}, union={paris,france,eiffel,tokyo,ramen,japan} = 0
        assert matrix[0][1] == pytest.approx(0.0)
        # vo[1] vs vid[0]: intersection={}, union={tokyo,japan,sushi,paris,france,wine} = 0
        assert matrix[1][0] == pytest.approx(0.0)
        # vo[1] vs vid[1]: intersection={tokyo,japan}, union={tokyo,japan,sushi,ramen} = 2/4
        assert matrix[1][1] == pytest.approx(2.0 / 4.0)

    def test_empty_inputs(self):
        """Empty input lists return empty matrix."""
        assert compute_relevance_matrix([], [['a']]) == []
        assert compute_relevance_matrix([['a']], []) == []
        assert compute_relevance_matrix([], []) == []

    def test_empty_keyword_lists(self):
        """Empty keyword lists within chapters produce 0.0."""
        matrix = compute_relevance_matrix([[]], [['a']])
        assert matrix[0][0] == pytest.approx(0.0)

        matrix2 = compute_relevance_matrix([['a']], [[]])
        assert matrix2[0][0] == pytest.approx(0.0)

        matrix3 = compute_relevance_matrix([[]], [[]])
        assert matrix3[0][0] == pytest.approx(0.0)

    def test_all_values_normalized_0_to_1(self):
        """All values in the matrix must be in [0.0, 1.0]."""
        vo = [['a', 'b', 'c'], ['d'], ['e', 'f']]
        vid = [['a', 'x'], ['b', 'c', 'd', 'e'], ['f']]
        matrix = compute_relevance_matrix(vo, vid)
        for row in matrix:
            for val in row:
                assert 0.0 <= val <= 1.0, f"Value {val} out of range"


# ============================================================================
# apply_cross_chapter_relevance_boost tests
# ============================================================================


class TestApplyCrossChapterRelevanceBoost:
    """Tests for MatchScoring.apply_cross_chapter_relevance_boost()."""

    def test_boost_proportional_to_relevance(self):
        """Boost = relevance_score * weight."""
        config = _mock_config(relevance_boost_weight=0.1)
        scorer = MatchScoring(config)

        matrix = [[0.5, 0.0], [0.0, 0.8]]
        conf, reason = scorer.apply_cross_chapter_relevance_boost(
            0.7, current_chapter_index=0, video_chapter_index=0, relevance_matrix=matrix
        )
        assert conf == pytest.approx(0.7 + 0.5 * 0.1)
        assert 'cross_chapter_relevance' in reason

    def test_zero_relevance_no_boost(self):
        """Zero relevance score produces no boost."""
        config = _mock_config(relevance_boost_weight=0.1)
        scorer = MatchScoring(config)

        matrix = [[0.0]]
        conf, reason = scorer.apply_cross_chapter_relevance_boost(
            0.7, current_chapter_index=0, video_chapter_index=0, relevance_matrix=matrix
        )
        assert conf == pytest.approx(0.7)
        assert reason == ""

    def test_disabled_chapter_grouping(self):
        """No boost when chapter grouping is disabled."""
        config = _mock_config(enabled=False)
        scorer = MatchScoring(config)

        matrix = [[1.0]]
        conf, reason = scorer.apply_cross_chapter_relevance_boost(
            0.7, current_chapter_index=0, video_chapter_index=0, relevance_matrix=matrix
        )
        assert conf == pytest.approx(0.7)
        assert reason == ""

    def test_negative_chapter_index_no_boost(self):
        """No boost when chapter indices are -1."""
        config = _mock_config()
        scorer = MatchScoring(config)

        matrix = [[1.0]]
        conf, reason = scorer.apply_cross_chapter_relevance_boost(
            0.7, current_chapter_index=-1, video_chapter_index=0, relevance_matrix=matrix
        )
        assert conf == pytest.approx(0.7)

        conf2, reason2 = scorer.apply_cross_chapter_relevance_boost(
            0.7, current_chapter_index=0, video_chapter_index=-1, relevance_matrix=matrix
        )
        assert conf2 == pytest.approx(0.7)

    def test_out_of_bounds_no_crash(self):
        """Out-of-bounds chapter indices return no boost."""
        config = _mock_config()
        scorer = MatchScoring(config)

        matrix = [[0.5]]  # 1x1 matrix
        conf, reason = scorer.apply_cross_chapter_relevance_boost(
            0.7, current_chapter_index=5, video_chapter_index=0, relevance_matrix=matrix
        )
        assert conf == pytest.approx(0.7)

    def test_no_matrix_no_boost(self):
        """No boost when relevance_matrix is None or empty."""
        config = _mock_config()
        scorer = MatchScoring(config)

        conf, reason = scorer.apply_cross_chapter_relevance_boost(
            0.7, current_chapter_index=0, video_chapter_index=0, relevance_matrix=None
        )
        assert conf == pytest.approx(0.7)

        conf2, reason2 = scorer.apply_cross_chapter_relevance_boost(
            0.7, current_chapter_index=0, video_chapter_index=0, relevance_matrix=[]
        )
        assert conf2 == pytest.approx(0.7)

    def test_custom_weight(self):
        """Custom relevance_boost_weight from config is respected."""
        config = _mock_config(relevance_boost_weight=0.2)
        scorer = MatchScoring(config)

        matrix = [[0.5]]
        conf, reason = scorer.apply_cross_chapter_relevance_boost(
            0.7, current_chapter_index=0, video_chapter_index=0, relevance_matrix=matrix
        )
        assert conf == pytest.approx(0.7 + 0.5 * 0.2)
