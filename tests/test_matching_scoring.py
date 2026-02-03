"""
Tests for MatchScoring class (US-33-005).

Tests the extracted scoring logic that was refactored from TieredMatcher.
Verifies the MatchScoring composition class provides the same functionality
as the original standalone functions.

Created: 2026-02-01 (Sprint 33 - US-33-005)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import MatchScoring
from src.utils import SRTSegment


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Mock config with matching settings"""
    config = Mock()

    # Matching config
    matching = Mock()
    matching.multimodal_enabled = True
    matching.multimodal_weights = None  # Use defaults
    matching.pool_normalization_enabled = True
    matching.chapter_matching_enabled = False
    matching.topic_mismatch_penalty = 0.15
    matching.broll_boost = 0.1
    matching.caption_quality_adjustment_enabled = True
    matching.caption_quality_high_boost = 0.05
    matching.caption_quality_low_penalty = 0.1
    matching.apply_timing_penalty = True
    matching.skip_llm_threshold = 0.85

    config.matching = matching

    # Global cache config
    global_cache = Mock()
    global_cache.current_project_boost = 0.1
    config.global_cache = global_cache

    return config


@pytest.fixture
def sample_vo_segment():
    """Sample voiceover segment"""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="Sample voiceover text about Tokyo",
        source_file="voiceover.srt"
    )
    seg.keywords = ["tokyo", "japan", "travel"]
    seg.entities = [{"text": "Tokyo", "type": "LOCATION"}]
    return seg


@pytest.fixture
def sample_video_segment():
    """Sample video segment"""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="Video footage of Tokyo skyline",
        source_file="/videos/tokyo.mp4"
    )
    seg.keywords = ["tokyo", "skyline", "city"]
    seg.entities = [{"text": "Tokyo", "type": "LOCATION"}]
    return seg


# ============================================================================
# Test MatchScoring Initialization
# ============================================================================

class TestMatchScoringInit:
    """Test MatchScoring class initialization"""

    @pytest.mark.fast
    def test_init_with_config(self, mock_config):
        """Test initialization with config"""
        scoring = MatchScoring(mock_config)

        assert scoring.config == mock_config
        assert scoring._mc == mock_config.matching

    @pytest.mark.fast
    def test_init_without_config(self):
        """Test initialization without config uses defaults"""
        scoring = MatchScoring(None)

        assert scoring.config is None
        assert scoring._mc is None


# ============================================================================
# Test calculate_confidence
# ============================================================================

class TestCalculateConfidence:
    """Test multimodal confidence calculation"""

    @pytest.mark.fast
    def test_calculate_confidence_with_all_scores(self, mock_config):
        """Test confidence calculation with all component scores"""
        scoring = MatchScoring(mock_config)

        confidence, reason, components = scoring.calculate_confidence(
            embedding_similarity=0.8,
            keyword_score=0.6,
            entity_score=0.5,
            visual_score=0.4
        )

        assert 0.0 <= confidence <= 1.0
        assert "multimodal" in reason or confidence == 0.8  # If multimodal disabled
        assert 'embedding_similarity' in components

    @pytest.mark.fast
    def test_calculate_confidence_embedding_only(self, mock_config):
        """Test confidence with only embedding similarity"""
        scoring = MatchScoring(mock_config)

        confidence, reason, components = scoring.calculate_confidence(
            embedding_similarity=0.9,
            keyword_score=0.0,
            entity_score=0.0,
            visual_score=0.0
        )

        # With zero other scores, confidence is reduced from embedding alone
        assert 0.0 <= confidence <= 1.0
        assert components['embedding_similarity'] == 0.9

    @pytest.mark.fast
    def test_calculate_confidence_multimodal_disabled(self):
        """Test confidence when multimodal is disabled"""
        config = Mock()
        config.matching = Mock()
        config.matching.multimodal_enabled = False
        config.matching.multimodal_weights = None

        scoring = MatchScoring(config)

        confidence, reason, components = scoring.calculate_confidence(
            embedding_similarity=0.85,
            keyword_score=0.5,
            entity_score=0.5,
            visual_score=0.5
        )

        # When disabled, returns embedding similarity directly
        assert confidence == 0.85
        assert "multimodal_disabled" in reason


# ============================================================================
# Test normalize_score
# ============================================================================

class TestNormalizeScore:
    """Test pool normalization"""

    @pytest.mark.fast
    def test_normalize_score_medium_pool(self, mock_config):
        """Test normalization with medium pool size"""
        scoring = MatchScoring(mock_config)

        normalized, reason = scoring.normalize_score(
            confidence=0.8,
            pool_size=50,  # Reference size
            candidates=None
        )

        # With pool_size=50 (reference), factor should be ~1.0
        assert 0.75 <= normalized <= 0.85
        assert "medium_pool" in reason

    @pytest.mark.fast
    def test_normalize_score_small_pool(self, mock_config):
        """Test normalization with small pool size"""
        scoring = MatchScoring(mock_config)

        # Create candidates with clear winner
        seg1 = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="A", source_file="a.mp4")
        seg2 = SRTSegment(index=2, start_time=0.0, end_time=5.0, text="B", source_file="b.mp4")
        candidates = [(seg1, 0.95), (seg2, 0.70)]  # Clear winner

        normalized, reason = scoring.normalize_score(
            confidence=0.8,
            pool_size=5,
            candidates=candidates
        )

        # Small pool with clear winner should boost confidence
        assert "small_pool" in reason

    @pytest.mark.fast
    def test_normalize_score_large_pool_tight_margin(self, mock_config):
        """Test normalization with large pool and tight margins"""
        scoring = MatchScoring(mock_config)

        # Create candidates with tight margin
        seg1 = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="A", source_file="a.mp4")
        seg2 = SRTSegment(index=2, start_time=0.0, end_time=5.0, text="B", source_file="b.mp4")
        candidates = [(seg1, 0.80), (seg2, 0.78)]  # Tight margin

        normalized, reason = scoring.normalize_score(
            confidence=0.8,
            pool_size=150,
            candidates=candidates
        )

        # Large pool with tight margin should reduce confidence
        assert normalized < 0.8
        assert "large_pool" in reason or "tight_margin" in reason


# ============================================================================
# Test apply_boost
# ============================================================================

class TestApplyBoost:
    """Test specific boost application"""

    @pytest.mark.fast
    def test_apply_broll_boost(self, mock_config, sample_video_segment):
        """Test B-roll boost application"""
        scoring = MatchScoring(mock_config)
        sample_video_segment.is_broll = True

        boosted, reason = scoring.apply_boost(
            confidence=0.7,
            video_segment=sample_video_segment,
            boost_type='broll'
        )

        assert abs(boosted - 0.8) < 0.001  # 0.7 + 0.1
        assert "B-roll boost" in reason

    @pytest.mark.fast
    def test_apply_broll_boost_non_broll(self, mock_config, sample_video_segment):
        """Test B-roll boost on non-B-roll segment"""
        scoring = MatchScoring(mock_config)
        sample_video_segment.is_broll = False

        boosted, reason = scoring.apply_boost(
            confidence=0.7,
            video_segment=sample_video_segment,
            boost_type='broll'
        )

        assert boosted == 0.7  # No change
        assert reason == ""

    @pytest.mark.fast
    def test_apply_project_boost_global_cache(self, mock_config, sample_video_segment):
        """Test project boost on global cache segment"""
        scoring = MatchScoring(mock_config)
        sample_video_segment.source = 'global_cache'

        boosted, reason = scoring.apply_boost(
            confidence=0.8,
            video_segment=sample_video_segment,
            boost_type='project'
        )

        assert abs(boosted - 0.7) < 0.001  # 0.8 - 0.1 penalty
        assert "global cache" in reason

    @pytest.mark.fast
    def test_apply_unknown_boost_type(self, mock_config, sample_video_segment):
        """Test unknown boost type returns unchanged"""
        scoring = MatchScoring(mock_config)

        boosted, reason = scoring.apply_boost(
            confidence=0.8,
            video_segment=sample_video_segment,
            boost_type='unknown'
        )

        assert boosted == 0.8
        assert reason == ""


# ============================================================================
# Test apply_penalty
# ============================================================================

class TestApplyPenalty:
    """Test specific penalty application"""

    @pytest.mark.fast
    def test_apply_timing_penalty(self, mock_config, sample_video_segment):
        """Test timing penalty application"""
        scoring = MatchScoring(mock_config)
        sample_video_segment.timing_penalty = 0.9

        penalized, reason = scoring.apply_penalty(
            confidence=0.8,
            video_segment=sample_video_segment,
            penalty_type='timing'
        )

        assert abs(penalized - 0.72) < 0.001  # 0.8 * 0.9
        assert "timing penalty" in reason

    @pytest.mark.fast
    def test_apply_caption_quality_penalty_low(self, mock_config, sample_video_segment):
        """Test caption quality penalty for low quality"""
        scoring = MatchScoring(mock_config)
        sample_video_segment.caption_quality = 'low'

        penalized, reason = scoring.apply_penalty(
            confidence=0.8,
            video_segment=sample_video_segment,
            penalty_type='caption_quality'
        )

        assert abs(penalized - 0.7) < 0.001  # 0.8 - 0.1
        assert "caption quality low" in reason

    @pytest.mark.fast
    def test_apply_topic_penalty_without_vo_segment(self, mock_config, sample_video_segment):
        """Test topic penalty without vo_segment returns unchanged"""
        scoring = MatchScoring(mock_config)

        penalized, reason = scoring.apply_penalty(
            confidence=0.8,
            video_segment=sample_video_segment,
            penalty_type='topic',
            vo_segment=None
        )

        assert penalized == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_apply_unknown_penalty_type(self, mock_config, sample_video_segment):
        """Test unknown penalty type returns unchanged"""
        scoring = MatchScoring(mock_config)

        penalized, reason = scoring.apply_penalty(
            confidence=0.8,
            video_segment=sample_video_segment,
            penalty_type='unknown'
        )

        assert penalized == 0.8
        assert reason == ""


# ============================================================================
# Test apply_all_adjustments
# ============================================================================

class TestApplyAllAdjustments:
    """Test combined adjustment application"""

    @pytest.mark.fast
    def test_apply_all_adjustments_basic(self, mock_config, sample_vo_segment, sample_video_segment):
        """Test applying all adjustments together"""
        scoring = MatchScoring(mock_config)

        adjusted, reason = scoring.apply_all_adjustments(
            confidence=0.8,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
            video_topics=None,
            chapter_matching_enabled=False,
            topic_mismatch_penalty=0.15
        )

        # With no special flags set, confidence should be unchanged
        assert 0.7 <= adjusted <= 0.9
        # Reason may be empty or contain adjustments

    @pytest.mark.fast
    def test_apply_all_adjustments_with_broll(self, mock_config, sample_vo_segment, sample_video_segment):
        """Test all adjustments with B-roll boost"""
        scoring = MatchScoring(mock_config)
        sample_video_segment.is_broll = True

        adjusted, reason = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment
        )

        # Should get B-roll boost (+0.1)
        assert adjusted >= 0.75
        assert "B-roll boost" in reason

    @pytest.mark.fast
    def test_apply_all_adjustments_with_multiple(self, mock_config, sample_vo_segment, sample_video_segment):
        """Test all adjustments with multiple adjustments active"""
        scoring = MatchScoring(mock_config)
        sample_video_segment.is_broll = True
        sample_video_segment.timing_penalty = 0.95
        sample_video_segment.caption_quality = 'high'

        adjusted, reason = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment
        )

        # Multiple adjustments should compound
        assert 0.0 <= adjusted <= 1.0
        # Should have multiple reasons
        assert "B-roll boost" in reason or "timing penalty" in reason or "caption quality" in reason


# ============================================================================
# Test calculate_adaptive_threshold
# ============================================================================

class TestCalculateAdaptiveThreshold:
    """Test adaptive threshold calculation"""

    @pytest.mark.fast
    def test_adaptive_threshold_normal(self, mock_config, sample_video_segment):
        """Test adaptive threshold with normal text"""
        scoring = MatchScoring(mock_config)
        candidates = [(sample_video_segment, 0.8)]

        threshold, reason = scoring.calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover segment about Tokyo travel",
            candidates=candidates
        )

        assert 0.5 <= threshold <= 0.99
        # Normal text should have minimal adjustment

    @pytest.mark.fast
    def test_adaptive_threshold_short_text(self, mock_config, sample_video_segment):
        """Test adaptive threshold with short text"""
        scoring = MatchScoring(mock_config)
        candidates = [(sample_video_segment, 0.8)]

        threshold, reason = scoring.calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="Short",  # < 20 chars
            candidates=candidates
        )

        # Short text should increase threshold
        assert threshold >= 0.85
        assert "short_vo" in reason


# ============================================================================
# Test Helper Methods
# ============================================================================

class TestHelperMethods:
    """Test helper extraction methods"""

    @pytest.mark.fast
    def test_extract_entity_texts(self, mock_config, sample_vo_segment):
        """Test entity text extraction"""
        scoring = MatchScoring(mock_config)

        entities = scoring.extract_entity_texts(sample_vo_segment)

        assert "Tokyo" in entities

    @pytest.mark.fast
    def test_extract_entity_texts_empty(self, mock_config):
        """Test entity extraction from segment without entities"""
        scoring = MatchScoring(mock_config)
        seg = SRTSegment(
            index=1, start_time=0.0, end_time=5.0,
            text="No entities", source_file="test.srt"
        )

        entities = scoring.extract_entity_texts(seg)

        assert entities == []

    @pytest.mark.fast
    def test_calculate_keyword_overlap(self, mock_config):
        """Test keyword overlap calculation"""
        scoring = MatchScoring(mock_config)

        score, matched = scoring.calculate_keyword_overlap(
            vo_keywords=["tokyo", "japan", "travel"],
            video_keywords=["tokyo", "city", "skyline"]
        )

        assert score > 0
        assert "tokyo" in matched

    @pytest.mark.fast
    def test_calculate_keyword_overlap_no_match(self, mock_config):
        """Test keyword overlap with no matches"""
        scoring = MatchScoring(mock_config)

        score, matched = scoring.calculate_keyword_overlap(
            vo_keywords=["paris", "france"],
            video_keywords=["tokyo", "japan"]
        )

        assert score == 0.0
        assert matched == []

    @pytest.mark.fast
    def test_calculate_entity_match(self, mock_config):
        """Test entity match calculation"""
        scoring = MatchScoring(mock_config)

        score, matched = scoring.calculate_entity_match(
            vo_entities=["Tokyo", "Japan"],
            video_entities=["Tokyo", "Shibuya"]
        )

        assert score > 0
        assert "Tokyo" in matched

    @pytest.mark.fast
    def test_calculate_entity_match_no_match(self, mock_config):
        """Test entity match with no matches"""
        scoring = MatchScoring(mock_config)

        score, matched = scoring.calculate_entity_match(
            vo_entities=["Paris", "France"],
            video_entities=["Tokyo", "Japan"]
        )

        assert score == 0.0
        assert matched == []


# ============================================================================
# Test Integration with TieredMatcher
# ============================================================================

class TestMatchScoringIntegration:
    """Test MatchScoring integration with TieredMatcher"""

    @pytest.mark.fast
    def test_tiered_matcher_has_scoring_instance(self, mock_config):
        """Test that TieredMatcher creates MatchScoring instance"""
        from src.matching.tiered_matcher import TieredMatcher

        with patch('src.matching.tiered_matcher.GeminiMatcher'):
            matcher = TieredMatcher(mock_config)

            assert hasattr(matcher, 'scoring')
            assert isinstance(matcher.scoring, MatchScoring)
            assert matcher.scoring.config == mock_config


# ============================================================================
# Test Cascading Confidence Penalties (US-46-004)
# ============================================================================

class TestCascadingConfidencePenalties:
    """Test guards against cascading confidence penalties"""

    @pytest.mark.fast
    def test_confidence_floor_prevents_near_zero(self, mock_config, sample_vo_segment, sample_video_segment):
        """
        Test that cascading penalties cannot reduce confidence below the floor.

        Scenario: confidence 0.8 * caption_quality 0.9 * timing 0.6 = 0.432
        This should stay above the confidence floor (0.05).
        """
        scoring = MatchScoring(mock_config)

        # Set up multiplicative caption quality weights
        mock_config.matching.caption_quality_weights = {'low': 0.5, 'medium': 0.9, 'high': 1.0}
        sample_video_segment.caption_quality = 'low'  # x0.5 multiplier
        sample_video_segment.timing_penalty = 0.3      # x0.3 multiplier (severe)

        adjusted, reason = scoring.apply_all_adjustments(
            confidence=0.8,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment
        )

        # 0.8 * 0.5 * 0.3 = 0.12, which is above floor but below warning threshold
        assert adjusted >= scoring.CONFIDENCE_FLOOR
        assert adjusted > 0.0

    @pytest.mark.fast
    def test_confidence_floor_enforced_at_extreme_penalties(self, mock_config, sample_vo_segment, sample_video_segment):
        """
        Test confidence floor is enforced when extreme penalties would reduce to near-zero.
        """
        scoring = MatchScoring(mock_config)

        # Set up extreme multiplicative penalties
        mock_config.matching.caption_quality_weights = {'low': 0.1}
        sample_video_segment.caption_quality = 'low'    # x0.1
        sample_video_segment.timing_penalty = 0.1       # x0.1
        sample_video_segment.source = 'global_cache'     # -0.1 additive

        adjusted, reason = scoring.apply_all_adjustments(
            confidence=0.5,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment
        )

        # 0.5 * 0.1 * 0.1 = 0.005 < floor, should be clamped to 0.05
        assert adjusted >= scoring.CONFIDENCE_FLOOR
        assert "confidence floor applied" in reason

    @pytest.mark.fast
    def test_caption_quality_09_timing_06_stays_above_floor(self, mock_config, sample_vo_segment, sample_video_segment):
        """
        AC4: caption quality 0.9 multiplier + timing penalty 0.6 should not
        reduce 0.8 confidence below floor.
        """
        scoring = MatchScoring(mock_config)

        mock_config.matching.caption_quality_weights = {'medium': 0.9, 'high': 1.0, 'low': 0.75}
        sample_video_segment.caption_quality = 'medium'  # x0.9
        sample_video_segment.timing_penalty = 0.6        # x0.6

        adjusted, reason = scoring.apply_all_adjustments(
            confidence=0.8,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment
        )

        # 0.8 * 0.9 = 0.72, then 0.72 * 0.6 = 0.432
        assert adjusted >= scoring.CONFIDENCE_FLOOR
        # The result should be approximately 0.432
        assert 0.40 <= adjusted <= 0.46
        assert "caption quality" in reason
        assert "timing penalty" in reason

    @pytest.mark.fast
    def test_low_confidence_warning_logged(self, mock_config, sample_vo_segment, sample_video_segment):
        """
        Test that a warning is logged when confidence drops below 0.15 after penalties.
        """
        scoring = MatchScoring(mock_config)

        mock_config.matching.caption_quality_weights = {'low': 0.3}
        sample_video_segment.caption_quality = 'low'  # x0.3
        sample_video_segment.timing_penalty = 0.5     # x0.5

        with patch('src.matching.scoring.logger') as mock_logger:
            adjusted, reason = scoring.apply_all_adjustments(
                confidence=0.8,
                vo_segment=sample_vo_segment,
                video_segment=sample_video_segment
            )

            # 0.8 * 0.3 = 0.24, then 0.24 * 0.5 = 0.12 < 0.15 threshold
            assert adjusted < scoring.LOW_CONFIDENCE_WARNING_THRESHOLD
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Over-penalized match" in warning_msg

    @pytest.mark.fast
    def test_no_floor_when_original_already_below(self, mock_config, sample_vo_segment, sample_video_segment):
        """
        Test that floor is NOT applied when original confidence is already below the floor.
        This prevents artificially boosting genuinely low-confidence matches.
        """
        scoring = MatchScoring(mock_config)

        adjusted, reason = scoring.apply_all_adjustments(
            confidence=0.03,  # Already below floor
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment
        )

        # Should NOT be raised to the floor - original was already low
        assert adjusted <= 0.05
        assert "confidence floor applied" not in reason


class TestCaptionQualityModeExclusivity:
    """Test that only one caption quality mode applies per match (US-46-004)"""

    @pytest.mark.fast
    def test_multiplicative_mode_takes_precedence(self, mock_config, sample_video_segment):
        """
        AC5: When caption_quality_weights is set (multiplicative mode),
        the additive mode should NOT also apply.
        """
        from src.matching.scoring import apply_caption_quality_adjustment

        # Configure both modes
        mock_config.matching.caption_quality_weights = {'high': 1.0, 'medium': 0.9, 'low': 0.75}
        mock_config.matching.caption_quality_high_boost = 0.05
        mock_config.matching.caption_quality_low_penalty = 0.1

        sample_video_segment.caption_quality = 'low'

        adjusted, reason = apply_caption_quality_adjustment(
            confidence=0.8,
            video_segment=sample_video_segment,
            config=mock_config
        )

        # Should use multiplicative (0.8 * 0.75 = 0.6), NOT additive (0.8 - 0.1 = 0.7)
        assert abs(adjusted - 0.6) < 0.001
        assert "multiplicative" in reason
        assert "additive" not in reason

    @pytest.mark.fast
    def test_additive_mode_only_when_no_weights(self, mock_config, sample_video_segment):
        """
        When caption_quality_weights is None, additive mode applies.
        """
        from src.matching.scoring import apply_caption_quality_adjustment

        mock_config.matching.caption_quality_weights = None
        mock_config.matching.caption_quality_low_penalty = 0.1

        sample_video_segment.caption_quality = 'low'

        adjusted, reason = apply_caption_quality_adjustment(
            confidence=0.8,
            video_segment=sample_video_segment,
            config=mock_config
        )

        # Should use additive (0.8 - 0.1 = 0.7)
        assert abs(adjusted - 0.7) < 0.001
        assert "additive" in reason
        assert "multiplicative" not in reason

    @pytest.mark.fast
    def test_multiplicative_and_additive_never_both_apply(self, mock_config, sample_video_segment):
        """
        Verify that the reason string never contains both 'multiplicative' and 'additive'.
        """
        from src.matching.scoring import apply_caption_quality_adjustment

        for quality in ['high', 'medium', 'low']:
            sample_video_segment.caption_quality = quality

            # Test with weights set (multiplicative mode)
            mock_config.matching.caption_quality_weights = {'high': 1.0, 'medium': 0.9, 'low': 0.75}
            _, reason_mult = apply_caption_quality_adjustment(0.8, sample_video_segment, mock_config)

            # Test without weights (additive mode)
            mock_config.matching.caption_quality_weights = None
            _, reason_add = apply_caption_quality_adjustment(0.8, sample_video_segment, mock_config)

            # Neither reason should contain both modes
            if reason_mult:
                assert "additive" not in reason_mult, f"Multiplicative reason contains 'additive': {reason_mult}"
            if reason_add:
                assert "multiplicative" not in reason_add, f"Additive reason contains 'multiplicative': {reason_add}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
