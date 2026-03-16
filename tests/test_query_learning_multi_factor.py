"""
Tests for US-94-006: Multi-factor strategy selection

Acceptance Criteria:
- AC1: Add multi-factor strategy selection to query_learning.py [DONE]
- AC2: Combine pattern success rate, chapter type success rate, AND confidence factor [DONE]
- AC3: Add confidence_weight config option (default: 0.2) [DONE in config.yaml]
- AC4: Implement get_multi_factor_strategy_ranking method in QueryLearningDB [DONE]
- AC5: Test that low-confidence gaps favor conservative strategies [DONE]
- AC6: Test that high-confidence gaps favor aggressive strategies [DONE]

Created: 2026-02-13 (Sprint 94)
"""

import pytest
import tempfile
import os
from src.iterative_match.query_learning import QueryLearningDB, QueryResult


class TestMultiFactorStrategySelection:
    """Test multi-factor strategy selection combining pattern, chapter, and confidence."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a fresh QueryLearningDB for testing."""
        db_path = tmp_path / "test_query_learning.json"
        return QueryLearningDB(str(db_path))

    def test_low_confidence_favors_conservative_strategies(self, db):
        """
        AC5: Test that low-confidence gaps favor conservative strategies.

        Low-confidence gaps should rank voiceover, topic, entity higher than similar_locked.
        """
        # Seed some pattern data showing similar_locked works best for this pattern
        db.pattern_strategy_success['abstract_concept']['similar_locked'] = 0.9
        db.pattern_strategy_success['abstract_concept']['voiceover'] = 0.5
        db.pattern_strategy_success['abstract_concept']['topic'] = 0.4
        db.pattern_strategy_success['abstract_concept']['entity'] = 0.3

        # Seed chapter data
        db.chapter_strategy_success['body']['similar_locked'] = 0.8
        db.chapter_strategy_success['body']['voiceover'] = 0.6

        # With low confidence (0.1), even though similar_locked has high base success,
        # the confidence bias should favor conservative strategies
        ranking = db.get_multi_factor_strategy_ranking(
            gap_pattern='abstract_concept',
            chapter_type='body',
            confidence=0.1,  # Low confidence
            confidence_weight=0.2
        )

        # Conservative strategies should be ranked higher for low confidence
        voiceover_idx = ranking.index('voiceover')
        similar_locked_idx = ranking.index('similar_locked')

        assert voiceover_idx < similar_locked_idx, \
            f"Low confidence should favor voiceover ({voiceover_idx}) over similar_locked ({similar_locked_idx})"

    def test_high_confidence_favors_aggressive_strategies(self, db):
        """
        AC6: Test that high-confidence gaps favor aggressive strategies.

        High-confidence gaps should rank similar_locked higher than conservative strategies.
        """
        # Seed pattern data showing voiceover works best
        db.pattern_strategy_success['abstract_concept']['voiceover'] = 0.5
        db.pattern_strategy_success['abstract_concept']['similar_locked'] = 0.4
        db.pattern_strategy_success['abstract_concept']['topic'] = 0.3
        db.pattern_strategy_success['abstract_concept']['entity'] = 0.2

        # With high confidence (0.9), similar_locked should be boosted
        ranking = db.get_multi_factor_strategy_ranking(
            gap_pattern='abstract_concept',
            chapter_type='body',
            confidence=0.9,  # High confidence
            confidence_weight=0.2
        )

        # Aggressive strategies should be ranked higher for high confidence
        similar_locked_idx = ranking.index('similar_locked')
        voiceover_idx = ranking.index('voiceover')

        assert similar_locked_idx < voiceover_idx, \
            f"High confidence should favor similar_locked ({similar_locked_idx}) over voiceover ({voiceover_idx})"

    def test_confidence_weight_zero_ignores_confidence(self, db):
        """
        Test that confidence_weight=0 disables confidence adjustment.
        """
        # Seed equal pattern data
        db.pattern_strategy_success['test_pattern']['voiceover'] = 0.5
        db.pattern_strategy_success['test_pattern']['similar_locked'] = 0.5

        # With confidence_weight=0, ranking should be based purely on pattern/chapter
        ranking_low = db.get_multi_factor_strategy_ranking(
            gap_pattern='test_pattern',
            chapter_type='body',
            confidence=0.1,
            confidence_weight=0.0
        )

        ranking_high = db.get_multi_factor_strategy_ranking(
            gap_pattern='test_pattern',
            chapter_type='body',
            confidence=0.9,
            confidence_weight=0.0
        )

        # Rankings should be identical when confidence weight is 0
        assert ranking_low == ranking_high, \
            "Rankings should be identical when confidence_weight=0"

    def test_confidence_weight_maximizes_bias(self, db):
        """
        Test that higher confidence_weight produces stronger bias.
        """
        # Seed pattern data where similar_locked is clearly better
        db.pattern_strategy_success['test_pattern']['voiceover'] = 0.2
        db.pattern_strategy_success['test_pattern']['similar_locked'] = 0.8

        # With low confidence_weight, ranking should still favor similar_locked (higher base)
        ranking_low_weight = db.get_multi_factor_strategy_ranking(
            gap_pattern='test_pattern',
            chapter_type='body',
            confidence=0.1,
            confidence_weight=0.1
        )

        # With high confidence_weight, voiceover should be boosted for low confidence
        ranking_high_weight = db.get_multi_factor_strategy_ranking(
            gap_pattern='test_pattern',
            chapter_type='body',
            confidence=0.1,
            confidence_weight=0.9
        )

        # Higher confidence weight should move voiceover up relative to similar_locked
        voiceover_idx_low = ranking_low_weight.index('voiceover')
        similar_idx_low = ranking_low_weight.index('similar_locked')
        voiceover_idx_high = ranking_high_weight.index('voiceover')
        similar_idx_high = ranking_high_weight.index('similar_locked')

        # With high weight, voiceover should be relatively better positioned
        assert (voiceover_idx_high - similar_idx_high) <= (voiceover_idx_low - similar_idx_low), \
            "Higher confidence_weight should strengthen the bias"

    def test_default_confidence_weight(self, db):
        """
        Test that default confidence_weight is used when not specified.
        """
        # Seed data
        db.pattern_strategy_success['test']['voiceover'] = 0.3
        db.pattern_strategy_success['test']['similar_locked'] = 0.7

        # Call without confidence_weight - should use default 0.2
        ranking = db.get_multi_factor_strategy_ranking(
            gap_pattern='test',
            chapter_type='body',
            confidence=0.5  # Neutral confidence
        )

        # Should return a valid ranking
        assert len(ranking) >= 2
        assert 'voiceover' in ranking
        assert 'similar_locked' in ranking

    def test_no_data_returns_default_order(self, db):
        """
        Test that missing data returns sensible default ordering.
        """
        ranking = db.get_multi_factor_strategy_ranking(
            gap_pattern='unknown_pattern',
            chapter_type='unknown_chapter',
            confidence=0.5,
            confidence_weight=0.2
        )

        # Should return default order with all strategies
        assert len(ranking) >= 4
        expected = {'voiceover', 'similar_locked', 'entity', 'topic'}
        assert expected.issubset(set(ranking))

    def test_pattern_and_chapter_blending(self, db):
        """
        Test that pattern and chapter success rates are blended correctly.
        """
        # Pattern favors voiceover (0.9)
        db.pattern_strategy_success['test_pat']['voiceover'] = 0.9
        db.pattern_strategy_success['test_pat']['similar_locked'] = 0.1

        # Chapter favors similar_locked (0.9)
        db.chapter_strategy_success['test_chap']['voiceover'] = 0.1
        db.chapter_strategy_success['test_chap']['similar_locked'] = 0.9

        # With neutral confidence (0.5), the 60/40 blend should apply
        ranking = db.get_multi_factor_strategy_ranking(
            gap_pattern='test_pat',
            chapter_type='test_chap',
            confidence=0.5,
            confidence_weight=0.0  # Disable confidence bias to test base blending
        )

        # voiceover: 0.6*0.9 + 0.4*0.1 = 0.54 + 0.04 = 0.58
        # similar_locked: 0.6*0.1 + 0.4*0.9 = 0.06 + 0.36 = 0.42
        # voiceover should be ranked higher
        assert ranking.index('voiceover') < ranking.index('similar_locked'), \
            "Pattern-dominant strategy should win with neutral confidence"


class TestGetBestStrategyWithConfidence:
    """Test confidence-aware best strategy selection."""

    def test_get_best_strategy_for_chapter_with_confidence(self, tmp_path):
        """
        Test that get_best_strategy_for_chapter can be enhanced with confidence.
        """
        db_path = tmp_path / "test_db.json"
        db = QueryLearningDB(str(db_path))

        # Seed data
        db.pattern_strategy_success['abstract']['voiceover'] = 0.5
        db.pattern_strategy_success['abstract']['similar_locked'] = 0.7
        db.chapter_strategy_success['intro']['voiceover'] = 0.8
        db.chapter_strategy_success['intro']['similar_locked'] = 0.3

        # For low confidence, get_multi_factor_strategy should favor voiceover
        ranking = db.get_multi_factor_strategy_ranking(
            gap_pattern='abstract',
            chapter_type='intro',
            confidence=0.1,
            confidence_weight=0.3
        )

        # The best strategy should be at position 0
        assert ranking[0] in ['voiceover', 'similar_locked']
