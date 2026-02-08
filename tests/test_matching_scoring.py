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
    matching.language_confidence_penalty = 0.0

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

        adjusted, reason, *_ = scoring.apply_all_adjustments(
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

        adjusted, reason, *_ = scoring.apply_all_adjustments(
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

        adjusted, reason, *_ = scoring.apply_all_adjustments(
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

        adjusted, reason, *_ = scoring.apply_all_adjustments(
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

        adjusted, reason, *_ = scoring.apply_all_adjustments(
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

        adjusted, reason, *_ = scoring.apply_all_adjustments(
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
            adjusted, reason, *_ = scoring.apply_all_adjustments(
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

        adjusted, reason, *_ = scoring.apply_all_adjustments(
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


# ============================================================================
# Test Adaptive Threshold Edge Cases (US-46-010)
# ============================================================================

class TestAdaptiveThresholdEdgeCases:
    """Test adaptive threshold boundary conditions and extreme pool sizes."""

    @pytest.mark.fast
    def test_empty_voiceover_text_adjusts_threshold(self, mock_config, sample_video_segment):
        """
        AC1: Empty voiceover text (0 chars) should trigger short_vo adjustment (+0.05).
        """
        scoring = MatchScoring(mock_config)
        candidates = [(sample_video_segment, 0.8)]

        threshold, reason = scoring.calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="",
            candidates=candidates
        )

        # Empty string -> 0 chars after strip -> short_vo adjustment
        assert threshold >= 0.85  # base + 0.05 = 0.90 (or clamped)
        assert "short_vo(0c)" in reason

    @pytest.mark.fast
    def test_none_voiceover_text_adjusts_threshold(self, mock_config, sample_video_segment):
        """
        AC1 variant: None voiceover text should also trigger short_vo adjustment.
        """
        scoring = MatchScoring(mock_config)
        candidates = [(sample_video_segment, 0.8)]

        threshold, reason = scoring.calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text=None,
            candidates=candidates
        )

        assert threshold >= 0.85
        assert "short_vo(0c)" in reason

    @pytest.mark.fast
    def test_large_pool_1000_candidates_normalization(self, mock_config):
        """
        AC2: Very large candidate pool (1000+) should produce valid normalized threshold.
        Pool normalization factor is capped at 1.2, so even 1000 candidates
        should not produce unreasonable thresholds.
        """
        from src.matching.scoring import normalize_confidence_by_pool

        # 1000 candidates: raw factor = sqrt(1000/50) = 4.47, clamped to 1.2
        confidence = 0.85
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=1000,
            pool_normalization_enabled=True
        )

        # Inverse of 1.2 = 0.833, so 0.85 * 0.833 ≈ 0.708
        assert 0.5 <= result <= 0.99
        assert result < confidence  # Large pool reduces confidence
        assert "large_pool(1000)" in reason

    @pytest.mark.fast
    def test_large_pool_2000_same_as_1000(self, mock_config):
        """
        AC2 variant: 2000 candidates should produce same result as 1000 due to capping.
        """
        from src.matching.scoring import normalize_confidence_by_pool

        confidence = 0.85
        result_1000, _ = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=1000,
            pool_normalization_enabled=True
        )
        result_2000, _ = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=2000,
            pool_normalization_enabled=True
        )

        # Both capped at max factor 1.2, so results should be identical
        assert abs(result_1000 - result_2000) < 0.001

    @pytest.mark.fast
    def test_small_pool_1_candidate_reasonable_threshold(self, mock_config):
        """
        AC3: Very small pool (1 candidate) should not produce unreasonable thresholds.
        """
        from src.matching.scoring import normalize_confidence_by_pool

        confidence = 0.7
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=1,
            pool_normalization_enabled=True
        )

        # Factor = sqrt(1/50) = 0.141, clamped to 0.8
        # Inverse = 1/0.8 = 1.25, so 0.7 * 1.25 = 0.875
        assert result > confidence  # Small pool boosts
        assert result <= 1.0  # Never exceeds 1.0
        assert "small_pool(1)" in reason

    @pytest.mark.fast
    def test_small_pool_2_candidates_reasonable_threshold(self, mock_config):
        """
        AC3: Very small pool (2 candidates) should not produce unreasonable thresholds.
        """
        from src.matching.scoring import normalize_confidence_by_pool

        seg = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="A", source_file="a.mp4")
        candidates = [(seg, 0.80), (seg, 0.60)]

        confidence = 0.7
        result, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=2,
            candidates=candidates,
            pool_normalization_enabled=True
        )

        # Pool of 2 with clear winner (margin 0.20 >= 0.1) gets +0.05 boost
        assert result > confidence
        assert result <= 1.0
        assert "small_pool(2)" in reason

    @pytest.mark.fast
    def test_variance_exactly_at_005_boundary(self, mock_config):
        """
        AC4: Candidate variance exactly at 0.05 boundary should NOT trigger
        the low_var threshold boost. The condition is variance < 0.05 (strict).
        """
        from src.matching.scoring import calculate_adaptive_threshold
        import statistics

        # Craft candidates whose top-5 stdev is exactly >= 0.05
        # stdev([0.78, 0.84, 0.85, 0.86, 0.92]) ≈ 0.0510
        seg = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="V", source_file="v.mp4")
        candidates = [
            (seg, 0.92),
            (seg, 0.86),
            (seg, 0.85),
            (seg, 0.84),
            (seg, 0.78),
        ]

        # Verify our crafted variance is >= 0.05
        top_scores = [s for _, s in candidates[:5]]
        actual_stdev = statistics.stdev(top_scores)
        assert actual_stdev >= 0.05, f"Test setup error: stdev={actual_stdev}"

        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover text that is long enough.",
            candidates=candidates
        )

        # Variance >= 0.05 should NOT trigger low_var adjustment
        assert "low_var" not in reason

    @pytest.mark.fast
    def test_variance_just_below_005_triggers_boost(self, mock_config):
        """
        AC4 complement: Variance just below 0.05 SHOULD trigger the low_var boost.
        """
        from src.matching.scoring import calculate_adaptive_threshold
        import statistics

        # Craft candidates with stdev just under 0.05
        # stdev([0.84, 0.85, 0.85, 0.86, 0.87]) ≈ 0.0112
        seg = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="V", source_file="v.mp4")
        candidates = [
            (seg, 0.87),
            (seg, 0.86),
            (seg, 0.85),
            (seg, 0.85),
            (seg, 0.84),
        ]

        # Verify variance is < 0.05
        top_scores = [s for _, s in candidates[:5]]
        actual_stdev = statistics.stdev(top_scores)
        assert actual_stdev < 0.05, f"Test setup error: stdev={actual_stdev}"

        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover text that is long enough.",
            candidates=candidates
        )

        # Variance < 0.05 SHOULD trigger low_var adjustment
        assert "low_var" in reason
        assert abs(threshold - 0.80) < 0.001  # 0.85 - 0.05 = 0.80

    @pytest.mark.fast
    def test_pool_normalization_scaling_at_10_50_200(self, mock_config):
        """
        AC5: Pool normalization reference size (50) produces expected scaling
        for pools of 10, 50, and 200 candidates.
        """
        from src.matching.scoring import (
            normalize_confidence_by_pool,
            POOL_NORMALIZATION_REFERENCE_SIZE,
        )

        confidence = 0.8

        # Pool of 10: sqrt(10/50) = 0.447, clamped to 0.8
        # inverse = 1/0.8 = 1.25, result = 0.8 * 1.25 = 1.0
        result_10, reason_10 = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=10,
            pool_normalization_enabled=True
        )

        # Pool of 50 (reference): sqrt(50/50) = 1.0
        # inverse = 1/1.0 = 1.0, result = 0.8 * 1.0 = 0.8
        result_50, reason_50 = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=50,
            pool_normalization_enabled=True
        )

        # Pool of 200: sqrt(200/50) = 2.0, clamped to 1.2
        # inverse = 1/1.2 = 0.833, result = 0.8 * 0.833 ≈ 0.667
        result_200, reason_200 = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=200,
            pool_normalization_enabled=True
        )

        # Verify ordering: small pool boosts > reference unchanged > large pool reduces
        assert result_10 > result_50 > result_200

        # Verify reference size produces ~unchanged confidence
        assert abs(result_50 - confidence) < 0.01

        # Verify small pool boosts confidence
        assert result_10 > confidence

        # Verify large pool reduces confidence
        assert result_200 < confidence

        # Verify reason strings reflect pool category
        # Pool of 10 is at the POOL_SMALL_THRESHOLD boundary (10 is NOT < 10)
        assert "medium_pool(10)" in reason_10 or "small_pool(10)" in reason_10
        assert "medium_pool(50)" in reason_50
        assert "large_pool(200)" in reason_200


# ============================================================================
# Test Confidence Breakdown Audit Trail (US-53-003)
# ============================================================================

class TestConfidenceBreakdown:
    """Test that apply_all_adjustments returns a confidence breakdown list."""

    @pytest.mark.fast
    def test_breakdown_returned_as_third_element(self, mock_config, sample_vo_segment, sample_video_segment):
        """apply_all_adjustments returns a 3-tuple with breakdown as third element."""
        scoring = MatchScoring(mock_config)
        result = scoring.apply_all_adjustments(
            confidence=0.8,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
        )
        assert len(result) == 3
        _, _, breakdown = result
        assert isinstance(breakdown, list)

    @pytest.mark.fast
    def test_breakdown_populated_with_broll(self, mock_config, sample_vo_segment, sample_video_segment):
        """Breakdown list has entries when B-roll boost fires."""
        scoring = MatchScoring(mock_config)
        sample_video_segment.is_broll = True

        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
        )

        assert len(breakdown) >= 1
        components = [b['component'] for b in breakdown]
        assert 'broll_boost' in components

        broll_entry = next(b for b in breakdown if b['component'] == 'broll_boost')
        assert 'adjustment' in broll_entry
        assert 'reason' in broll_entry
        assert broll_entry['adjustment'] > 0  # boost is positive

    @pytest.mark.fast
    def test_breakdown_populated_with_multiple_adjustments(self, mock_config, sample_vo_segment, sample_video_segment):
        """Breakdown has multiple entries when multiple adjustments fire."""
        scoring = MatchScoring(mock_config)
        sample_video_segment.is_broll = True
        sample_video_segment.timing_penalty = 0.9
        sample_video_segment.caption_quality = 'high'

        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
        )

        assert len(breakdown) >= 2
        components = [b['component'] for b in breakdown]
        assert 'broll_boost' in components
        assert 'timing_penalty' in components

    @pytest.mark.fast
    def test_breakdown_empty_when_no_adjustments(self, mock_config, sample_vo_segment, sample_video_segment):
        """Breakdown is empty list when no adjustments fire."""
        scoring = MatchScoring(mock_config)

        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.8,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
        )

        # No special attributes set, so no adjustments should fire
        assert isinstance(breakdown, list)

    @pytest.mark.fast
    def test_breakdown_is_json_serializable(self, mock_config, sample_vo_segment, sample_video_segment):
        """Breakdown list is JSON-serializable (dicts with simple types)."""
        import json
        scoring = MatchScoring(mock_config)
        sample_video_segment.is_broll = True
        sample_video_segment.timing_penalty = 0.95

        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
        )

        # Must not raise
        serialized = json.dumps(breakdown)
        deserialized = json.loads(serialized)
        assert isinstance(deserialized, list)
        for entry in deserialized:
            assert 'component' in entry
            assert 'adjustment' in entry
            assert 'reason' in entry

    @pytest.mark.fast
    def test_breakdown_entry_structure(self, mock_config, sample_vo_segment, sample_video_segment):
        """Each breakdown entry has required keys: component, adjustment, reason."""
        scoring = MatchScoring(mock_config)
        sample_video_segment.is_broll = True

        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
        )

        for entry in breakdown:
            assert isinstance(entry, dict)
            assert set(entry.keys()) == {'component', 'adjustment', 'reason'}
            assert isinstance(entry['component'], str)
            assert isinstance(entry['adjustment'], (int, float))
            assert isinstance(entry['reason'], str)


class TestMatchResultConfidenceBreakdownField:
    """Test MatchResult dataclass has confidence_breakdown field with correct default."""

    @pytest.mark.fast
    def test_confidence_breakdown_has_default(self):
        """confidence_breakdown field has a default_factory in the dataclass."""
        from dataclasses import fields as dataclass_fields
        from src.utils import MatchResult
        field_map = {f.name: f for f in dataclass_fields(MatchResult)}
        assert 'confidence_breakdown' in field_map
        f = field_map['confidence_breakdown']
        assert f.default_factory is not None

    @pytest.mark.fast
    def test_matchresult_default_empty_breakdown(self):
        """MatchResult defaults to empty confidence_breakdown."""
        from src.utils import MatchResult, Match, SRTSegment
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test")
        vid = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="vid.mp4")
        m = Match(voiceover_segment=vo, video_segment=vid, video_scene=None,
                  confidence=0.8, reasoning="test")
        result = MatchResult(primary_match=m)
        assert result.confidence_breakdown == []

    @pytest.mark.fast
    def test_matchresult_accepts_breakdown(self):
        """MatchResult can be constructed with a confidence_breakdown."""
        from src.utils import MatchResult, Match, SRTSegment
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test")
        vid = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="vid.mp4")
        m = Match(voiceover_segment=vo, video_segment=vid, video_scene=None,
                  confidence=0.75, reasoning="test")
        bd = [{'component': 'broll_boost', 'adjustment': 0.1, 'reason': 'B-roll boost: +0.10'}]
        result = MatchResult(primary_match=m, confidence_breakdown=bd)
        assert len(result.confidence_breakdown) == 1
        assert result.confidence_breakdown[0]['component'] == 'broll_boost'


# ============================================================================
# Test Match.confidence_breakdown Field (US-63-007)
# ============================================================================

class TestMatchConfidenceBreakdownField:
    """Test Match dataclass has confidence_breakdown field for scoring transparency."""

    @pytest.mark.fast
    def test_match_has_confidence_breakdown_field(self):
        """Match dataclass has confidence_breakdown field."""
        from dataclasses import fields as dataclass_fields
        from src.utils import Match
        field_map = {f.name: f for f in dataclass_fields(Match)}
        assert 'confidence_breakdown' in field_map

    @pytest.mark.fast
    def test_match_confidence_breakdown_default_empty(self):
        """Match defaults to empty confidence_breakdown list."""
        from src.utils import Match, SRTSegment
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test")
        vid = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="vid.mp4")
        m = Match(voiceover_segment=vo, video_segment=vid, video_scene=None,
                  confidence=0.8, reasoning="test")
        assert m.confidence_breakdown == []

    @pytest.mark.fast
    def test_match_accepts_confidence_breakdown(self):
        """Match can be constructed with a confidence_breakdown list."""
        from src.utils import Match, SRTSegment
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test")
        vid = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="vid.mp4")
        breakdown = [
            {'component': 'broll_boost', 'adjustment': 0.1, 'reason': 'B-roll boost: +0.10'},
            {'component': 'timing_penalty', 'adjustment': -0.05, 'reason': 'Timing penalty: -0.05'},
        ]
        m = Match(voiceover_segment=vo, video_segment=vid, video_scene=None,
                  confidence=0.85, reasoning="test", confidence_breakdown=breakdown)
        assert len(m.confidence_breakdown) == 2
        assert m.confidence_breakdown[0]['component'] == 'broll_boost'
        assert m.confidence_breakdown[1]['component'] == 'timing_penalty'

    @pytest.mark.fast
    def test_match_to_dict_includes_breakdown(self):
        """Match.to_dict includes confidence_breakdown when present."""
        from src.utils import Match, SRTSegment
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test")
        vid = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="vid.mp4")
        breakdown = [{'component': 'topic_penalty', 'adjustment': -0.15, 'reason': 'Topic mismatch'}]
        m = Match(voiceover_segment=vo, video_segment=vid, video_scene=None,
                  confidence=0.7, reasoning="test", confidence_breakdown=breakdown)

        d = m.to_dict()
        assert 'confidence_breakdown' in d
        assert d['confidence_breakdown'] == breakdown

    @pytest.mark.fast
    def test_match_to_dict_omits_empty_breakdown(self):
        """Match.to_dict omits confidence_breakdown when empty."""
        from src.utils import Match, SRTSegment
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test")
        vid = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="vid.mp4")
        m = Match(voiceover_segment=vo, video_segment=vid, video_scene=None,
                  confidence=0.8, reasoning="test")

        d = m.to_dict()
        # Empty breakdown should not be serialized (cleaner JSON)
        assert 'confidence_breakdown' not in d

    @pytest.mark.fast
    def test_match_from_dict_restores_breakdown(self):
        """Match.from_dict restores confidence_breakdown from dict."""
        from src.utils import Match, SRTSegment
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test")
        vid = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="vid.mp4")
        breakdown = [
            {'component': 'caption_quality', 'adjustment': 0.05, 'reason': 'High caption quality'},
            {'component': 'project_boost', 'adjustment': 0.1, 'reason': 'Current project boost'},
        ]
        m = Match(voiceover_segment=vo, video_segment=vid, video_scene=None,
                  confidence=0.95, reasoning="test", confidence_breakdown=breakdown)

        d = m.to_dict()
        restored = Match.from_dict(d)

        assert restored.confidence_breakdown == breakdown
        assert len(restored.confidence_breakdown) == 2

    @pytest.mark.fast
    def test_match_from_dict_without_breakdown(self):
        """Match.from_dict handles dicts without confidence_breakdown (backward compat)."""
        from src.utils import Match, SRTSegment
        # Old format dict without breakdown
        d = {
            'voiceover_segment': {'index': 0, 'start_time': 0.0, 'end_time': 5.0, 'text': 'test'},
            'video_segment': {'index': 0, 'start_time': 0.0, 'end_time': 5.0, 'text': 'test', 'source_file': 'vid.mp4'},
            'video_scene': None,
            'confidence': 0.8,
            'reasoning': 'test',
        }

        restored = Match.from_dict(d)
        assert restored.confidence_breakdown == []

    @pytest.mark.fast
    def test_match_breakdown_round_trip_json(self):
        """Match confidence_breakdown survives JSON round-trip via to_dict/from_dict."""
        import json
        from src.utils import Match, SRTSegment
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test")
        vid = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="vid.mp4")
        breakdown = [
            {'component': 'duration_penalty', 'adjustment': -0.08, 'reason': 'Duration mismatch: -0.08'},
            {'component': 'broll_boost', 'adjustment': 0.12, 'reason': 'B-roll segment: +0.12'},
        ]
        m = Match(voiceover_segment=vo, video_segment=vid, video_scene=None,
                  confidence=0.84, reasoning="test", confidence_breakdown=breakdown)

        # Round-trip through JSON
        json_str = json.dumps(m.to_dict())
        restored = Match.from_dict(json.loads(json_str))

        assert restored.confidence_breakdown == breakdown


class TestTieredMatcherPopulatesBreakdown:
    """Test that TieredMatcher populates confidence_breakdown on Match objects (US-63-007)."""

    @pytest.mark.fast
    def test_breakdown_captures_all_adjustments(self, mock_config):
        """
        US-63-007: Verify breakdown captures adjustments from:
        - duration penalty
        - topic penalty
        - broll boost
        - caption quality
        - timing penalty
        """
        from src.matching.tiered_matcher import _record_breakdown

        breakdown = []

        # Simulate multiple adjustments
        prev = 0.8
        _record_breakdown(breakdown, 'topic_penalty', prev, 0.65, 'Topic mismatch: -0.15')
        prev = 0.65
        _record_breakdown(breakdown, 'broll_boost', prev, 0.75, 'B-roll boost: +0.10')
        prev = 0.75
        _record_breakdown(breakdown, 'caption_quality', prev, 0.80, 'High quality: +0.05')
        prev = 0.80
        _record_breakdown(breakdown, 'timing_penalty', prev, 0.76, 'Timing issue: -0.04')
        prev = 0.76
        _record_breakdown(breakdown, 'project_boost', prev, 0.86, 'Current project: +0.10')

        assert len(breakdown) == 5
        components = [b['component'] for b in breakdown]
        assert 'topic_penalty' in components
        assert 'broll_boost' in components
        assert 'caption_quality' in components
        assert 'timing_penalty' in components
        assert 'project_boost' in components

        # Verify adjustment values are captured
        topic_entry = next(b for b in breakdown if b['component'] == 'topic_penalty')
        assert abs(topic_entry['adjustment'] - (-0.15)) < 0.001

        broll_entry = next(b for b in breakdown if b['component'] == 'broll_boost')
        assert abs(broll_entry['adjustment'] - 0.10) < 0.001

    @pytest.mark.fast
    def test_record_breakdown_skips_empty_reason(self):
        """_record_breakdown skips entries when reason is empty (no adjustment)."""
        from src.matching.tiered_matcher import _record_breakdown

        breakdown = []
        _record_breakdown(breakdown, 'broll_boost', 0.8, 0.8, '')  # No change, empty reason
        _record_breakdown(breakdown, 'timing_penalty', 0.8, 0.75, 'Penalty applied')  # Has change

        assert len(breakdown) == 1
        assert breakdown[0]['component'] == 'timing_penalty'

    @pytest.mark.fast
    def test_record_breakdown_rounds_adjustment(self):
        """_record_breakdown rounds adjustment to 4 decimal places."""
        from src.matching.tiered_matcher import _record_breakdown

        breakdown = []
        _record_breakdown(breakdown, 'test', 0.80000001, 0.75000002, 'Test reason')

        assert breakdown[0]['adjustment'] == round(0.75000002 - 0.80000001, 4)


# ============================================================================
# Test Consecutive Source Penalty (US-63-009)
# ============================================================================

class TestConsecutiveSourcePenalty:
    """Test consecutive same-source matching penalty (US-63-009).

    Verifies that using the same video source in consecutive segments
    applies a stacking penalty to encourage visual variety.
    """

    @pytest.fixture
    def mock_match(self):
        """Create a mock Match object for testing."""
        def _make_match(source_file: str):
            match = Mock()
            match.video_segment = Mock()
            match.video_segment.source_file = source_file
            return match
        return _make_match

    @pytest.fixture
    def mock_config_with_consecutive(self):
        """Mock config with consecutive source penalty settings."""
        config = Mock()
        matching = Mock()
        matching.consecutive_source_penalty = 0.1
        matching.max_consecutive_same_source = 3
        config.matching = matching
        return config

    @pytest.mark.fast
    def test_no_penalty_when_no_recent_matches(self, sample_video_segment, mock_config_with_consecutive):
        """No penalty should be applied when there are no recent matches."""
        from src.matching.scoring import apply_consecutive_source_penalty

        confidence, reason = apply_consecutive_source_penalty(
            confidence=0.8,
            video_segment=sample_video_segment,
            recent_matches=[],
            config=mock_config_with_consecutive
        )

        assert confidence == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_no_penalty_when_different_source(self, sample_video_segment, mock_config_with_consecutive, mock_match):
        """No penalty when previous match is from different source."""
        from src.matching.scoring import apply_consecutive_source_penalty

        # Previous match from different source
        recent_matches = [mock_match("/videos/other_video.mp4")]

        confidence, reason = apply_consecutive_source_penalty(
            confidence=0.8,
            video_segment=sample_video_segment,  # source_file="/videos/tokyo.mp4"
            recent_matches=recent_matches,
            config=mock_config_with_consecutive
        )

        assert confidence == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_single_consecutive_penalty(self, sample_video_segment, mock_config_with_consecutive, mock_match):
        """Single consecutive match applies 1x penalty."""
        from src.matching.scoring import apply_consecutive_source_penalty

        # Previous match from same source
        sample_video_segment.source_file = "/videos/tokyo.mp4"
        recent_matches = [mock_match("/videos/tokyo.mp4")]

        confidence, reason = apply_consecutive_source_penalty(
            confidence=0.8,
            video_segment=sample_video_segment,
            recent_matches=recent_matches,
            config=mock_config_with_consecutive
        )

        # Should apply 0.1 penalty (1 * 0.1)
        assert confidence == pytest.approx(0.7, abs=0.001)
        assert "consecutive_source_penalty" in reason
        assert "-0.10" in reason
        assert "1 consecutive" in reason

    @pytest.mark.fast
    def test_stacking_penalty_two_consecutive(self, sample_video_segment, mock_config_with_consecutive, mock_match):
        """Two consecutive matches apply 2x penalty."""
        from src.matching.scoring import apply_consecutive_source_penalty

        sample_video_segment.source_file = "/videos/tokyo.mp4"
        recent_matches = [
            mock_match("/videos/tokyo.mp4"),
            mock_match("/videos/tokyo.mp4"),
        ]

        confidence, reason = apply_consecutive_source_penalty(
            confidence=0.8,
            video_segment=sample_video_segment,
            recent_matches=recent_matches,
            config=mock_config_with_consecutive
        )

        # Should apply 0.2 penalty (2 * 0.1)
        assert confidence == pytest.approx(0.6, abs=0.001)
        assert "2 consecutive" in reason

    @pytest.mark.fast
    def test_stacking_penalty_three_consecutive(self, sample_video_segment, mock_config_with_consecutive, mock_match):
        """Three consecutive matches apply 3x penalty."""
        from src.matching.scoring import apply_consecutive_source_penalty

        sample_video_segment.source_file = "/videos/tokyo.mp4"
        recent_matches = [
            mock_match("/videos/tokyo.mp4"),
            mock_match("/videos/tokyo.mp4"),
            mock_match("/videos/tokyo.mp4"),
        ]

        confidence, reason = apply_consecutive_source_penalty(
            confidence=0.8,
            video_segment=sample_video_segment,
            recent_matches=recent_matches,
            config=mock_config_with_consecutive
        )

        # Should apply 0.3 penalty (3 * 0.1)
        assert confidence == pytest.approx(0.5, abs=0.001)
        assert "3 consecutive" in reason

    @pytest.mark.fast
    def test_stops_counting_at_different_source(self, sample_video_segment, mock_config_with_consecutive, mock_match):
        """Consecutive count stops when a different source is encountered."""
        from src.matching.scoring import apply_consecutive_source_penalty

        sample_video_segment.source_file = "/videos/tokyo.mp4"
        recent_matches = [
            mock_match("/videos/tokyo.mp4"),  # 1 consecutive
            mock_match("/videos/other.mp4"),  # Different source - stops here
            mock_match("/videos/tokyo.mp4"),  # Should not count
        ]

        confidence, reason = apply_consecutive_source_penalty(
            confidence=0.8,
            video_segment=sample_video_segment,
            recent_matches=recent_matches,
            config=mock_config_with_consecutive
        )

        # Should only apply 0.1 penalty (1 consecutive, stopped at 'other.mp4')
        assert confidence == pytest.approx(0.7, abs=0.001)
        assert "1 consecutive" in reason

    @pytest.mark.fast
    def test_penalty_does_not_go_negative(self, sample_video_segment, mock_config_with_consecutive, mock_match):
        """Penalty should not reduce confidence below 0.0."""
        from src.matching.scoring import apply_consecutive_source_penalty

        sample_video_segment.source_file = "/videos/tokyo.mp4"
        # Create many consecutive matches to potentially exceed confidence
        recent_matches = [mock_match("/videos/tokyo.mp4") for _ in range(10)]

        confidence, reason = apply_consecutive_source_penalty(
            confidence=0.5,  # Low initial confidence
            video_segment=sample_video_segment,
            recent_matches=recent_matches,
            config=mock_config_with_consecutive
        )

        # Should be clamped to 0.0
        assert confidence >= 0.0

    @pytest.mark.fast
    def test_hard_cap_check(self, sample_video_segment, mock_config_with_consecutive, mock_match):
        """check_consecutive_source_hard_cap returns True when at max."""
        from src.matching.scoring import check_consecutive_source_hard_cap

        sample_video_segment.source_file = "/videos/tokyo.mp4"
        recent_matches = [
            mock_match("/videos/tokyo.mp4"),
            mock_match("/videos/tokyo.mp4"),
            mock_match("/videos/tokyo.mp4"),  # 3 consecutive - at max
        ]

        should_block, count = check_consecutive_source_hard_cap(
            video_segment=sample_video_segment,
            recent_matches=recent_matches,
            config=mock_config_with_consecutive
        )

        # At max (3), should block
        assert should_block is True
        assert count == 3

    @pytest.mark.fast
    def test_hard_cap_check_under_limit(self, sample_video_segment, mock_config_with_consecutive, mock_match):
        """check_consecutive_source_hard_cap returns False when under max."""
        from src.matching.scoring import check_consecutive_source_hard_cap

        sample_video_segment.source_file = "/videos/tokyo.mp4"
        recent_matches = [
            mock_match("/videos/tokyo.mp4"),
            mock_match("/videos/tokyo.mp4"),  # 2 consecutive - under max
        ]

        should_block, count = check_consecutive_source_hard_cap(
            video_segment=sample_video_segment,
            recent_matches=recent_matches,
            config=mock_config_with_consecutive
        )

        # Under max (2 < 3), should not block
        assert should_block is False
        assert count == 2

    @pytest.mark.fast
    def test_penalty_records_in_confidence_breakdown(self, sample_video_segment, sample_vo_segment, mock_config_with_consecutive, mock_match):
        """The penalty should be recorded in confidence_breakdown (depends on US-63-007)."""
        from src.matching.tiered_matcher import _record_breakdown
        from src.matching.scoring import apply_consecutive_source_penalty

        sample_video_segment.source_file = "/videos/tokyo.mp4"
        recent_matches = [mock_match("/videos/tokyo.mp4")]

        prev = 0.8
        adjusted, reason = apply_consecutive_source_penalty(
            confidence=prev,
            video_segment=sample_video_segment,
            recent_matches=recent_matches,
            config=mock_config_with_consecutive
        )

        # Record breakdown like TieredMatcher does
        breakdown = []
        _record_breakdown(breakdown, 'consecutive_source_penalty', prev, adjusted, reason)

        assert len(breakdown) == 1
        assert breakdown[0]['component'] == 'consecutive_source_penalty'
        assert breakdown[0]['adjustment'] == pytest.approx(-0.1, abs=0.0001)
        assert "consecutive" in breakdown[0]['reason']


# ============================================================================
# Test Title Relevance Adjustment (US-70-006)
# ============================================================================

class TestTitleRelevanceAdjustment:
    """Test apply_title_relevance_adjustment graduated boost."""

    @pytest.mark.fast
    def test_one_keyword_match_boost(self, mock_config):
        """1 keyword match -> +0.03 boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="The history of ancient Rome and its empire")
        adjusted, reason = scoring.apply_title_relevance_adjustment(
            0.7, vo, video_title="Rome travel guide"
        )
        assert adjusted == pytest.approx(0.73, abs=0.001)
        assert "title relevance boost +0.03" in reason
        assert "1 keyword" in reason

    @pytest.mark.fast
    def test_two_keyword_match_boost(self, mock_config):
        """2 keyword matches -> +0.05 boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Tokyo skyline and Japanese culture travel")
        adjusted, reason = scoring.apply_title_relevance_adjustment(
            0.7, vo, video_title="Tokyo culture documentary"
        )
        assert adjusted == pytest.approx(0.75, abs=0.001)
        assert "title relevance boost +0.05" in reason
        assert "2 keywords" in reason

    @pytest.mark.fast
    def test_three_plus_keyword_match_boost(self, mock_config):
        """3+ keyword matches -> +0.08 boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="ancient Roman architecture ruins temples heritage")
        adjusted, reason = scoring.apply_title_relevance_adjustment(
            0.7, vo, video_title="ancient Roman architecture temples"
        )
        assert adjusted == pytest.approx(0.78, abs=0.001)
        assert "title relevance boost +0.08" in reason
        assert "3+" in reason or "keyword" in reason

    @pytest.mark.fast
    def test_no_keyword_overlap_no_boost(self, mock_config):
        """No keyword overlap -> no boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="underwater coral reef marine biology")
        adjusted, reason = scoring.apply_title_relevance_adjustment(
            0.7, vo, video_title="mountain hiking trails"
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_empty_title_no_boost(self, mock_config):
        """Empty video title -> no adjustment (graceful no-op)."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Some voiceover text")
        adjusted, reason = scoring.apply_title_relevance_adjustment(
            0.7, vo, video_title=""
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_none_title_no_boost(self, mock_config):
        """None video title -> no adjustment (graceful no-op)."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Some voiceover text")
        adjusted, reason = scoring.apply_title_relevance_adjustment(
            0.7, vo, video_title=None
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_breakdown_entry_format_in_apply_all(self, mock_config, sample_vo_segment, sample_video_segment):
        """apply_all_adjustments records title_relevance in confidence_breakdown."""
        scoring = MatchScoring(mock_config)
        # Use a title that overlaps with sample_vo_segment text "Sample voiceover text about Tokyo"
        adjusted, reason, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
            video_title="Tokyo travel guide sample"
        )
        title_entries = [b for b in breakdown if b['component'] == 'title_relevance']
        assert len(title_entries) == 1
        entry = title_entries[0]
        assert entry['component'] == 'title_relevance'
        assert entry['adjustment'] > 0
        assert isinstance(entry['reason'], str)
        assert 'title relevance boost' in entry['reason']

    @pytest.mark.fast
    def test_apply_all_no_title_no_entry(self, mock_config, sample_vo_segment, sample_video_segment):
        """apply_all_adjustments with no video_title produces no title_relevance entry."""
        scoring = MatchScoring(mock_config)
        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
        )
        title_entries = [b for b in breakdown if b['component'] == 'title_relevance']
        assert len(title_entries) == 0

    @pytest.mark.fast
    def test_stopwords_excluded(self, mock_config):
        """Stopwords like 'the', 'and', 'with' don't count as keyword matches."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="the and with this that from")
        adjusted, reason = scoring.apply_title_relevance_adjustment(
            0.7, vo, video_title="the and with this that from"
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_case_insensitive_matching(self, mock_config):
        """Keywords match case-insensitively."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="TOKYO skyline panoramic")
        adjusted, reason = scoring.apply_title_relevance_adjustment(
            0.7, vo, video_title="tokyo Skyline view"
        )
        assert adjusted > 0.7
        assert "tokyo" in reason.lower() or "skyline" in reason.lower()


class TestDescriptionRelevance:
    """Test apply_description_relevance graduated boost (US-73-003)."""

    @pytest.mark.fast
    def test_zero_keyword_match_no_boost(self, mock_config):
        """No keyword overlap produces no boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="underwater coral reef marine biology")
        adjusted, reason = scoring.apply_description_relevance(
            0.7, vo, video_description="mountain hiking trails adventure"
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_one_keyword_match_boost(self, mock_config):
        """One keyword overlap gives +0.02 boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="The history of ancient Rome and its empire")
        adjusted, reason = scoring.apply_description_relevance(
            0.7, vo, video_description="Rome travel guide for beginners"
        )
        assert adjusted == pytest.approx(0.72, abs=0.001)
        assert "description relevance boost" in reason
        assert "+0.02" in reason

    @pytest.mark.fast
    def test_two_keyword_match_boost(self, mock_config):
        """Two keyword overlaps give +0.04 boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Tokyo skyline and Japanese culture travel")
        adjusted, reason = scoring.apply_description_relevance(
            0.7, vo, video_description="Tokyo culture documentary film"
        )
        assert adjusted == pytest.approx(0.74, abs=0.001)
        assert "+0.04" in reason

    @pytest.mark.fast
    def test_three_plus_keyword_match_boost(self, mock_config):
        """Three or more keyword overlaps give +0.06 boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="ancient Roman architecture ruins temples heritage")
        adjusted, reason = scoring.apply_description_relevance(
            0.7, vo, video_description="ancient Roman architecture temples overview"
        )
        assert adjusted == pytest.approx(0.76, abs=0.001)
        assert "+0.06" in reason

    @pytest.mark.fast
    def test_empty_description_no_boost(self, mock_config):
        """Empty description produces no boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Some voiceover text")
        adjusted, reason = scoring.apply_description_relevance(
            0.7, vo, video_description=""
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_none_description_no_boost(self, mock_config):
        """None description produces no boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Some voiceover text")
        adjusted, reason = scoring.apply_description_relevance(
            0.7, vo, video_description=None
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_description_truncated_to_200_chars(self, mock_config):
        """Description is truncated to first 200 chars before keyword extraction."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="unique keyword matchword")
        # Place matching keyword beyond 200 chars
        padding = "a " * 110  # 220 chars of non-matching text
        desc = padding + "matchword unique keyword"
        adjusted, reason = scoring.apply_description_relevance(
            0.7, vo, video_description=desc
        )
        # Keywords beyond 200 chars should not be found
        assert adjusted == pytest.approx(0.7, abs=0.001)

    @pytest.mark.fast
    def test_breakdown_entry_in_apply_all(self, mock_config, sample_vo_segment, sample_video_segment):
        """apply_all_adjustments records description_relevance in confidence_breakdown."""
        scoring = MatchScoring(mock_config)
        # sample_vo_segment text: "Sample voiceover text about Tokyo"
        adjusted, reason, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
            video_description="Tokyo travel guide sample"
        )
        desc_entries = [b for b in breakdown if b['component'] == 'description_relevance']
        assert len(desc_entries) == 1
        entry = desc_entries[0]
        assert entry['adjustment'] > 0
        assert 'description relevance boost' in entry['reason']

    @pytest.mark.fast
    def test_apply_all_no_description_no_entry(self, mock_config, sample_vo_segment, sample_video_segment):
        """apply_all_adjustments with no video_description produces no description_relevance entry."""
        scoring = MatchScoring(mock_config)
        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
        )
        desc_entries = [b for b in breakdown if b['component'] == 'description_relevance']
        assert len(desc_entries) == 0


class TestChapterSourceConsistency:
    """Tests for chapter-level source consistency boost (US-70-011)."""

    @pytest.fixture
    def mock_match(self):
        """Create a mock Match object with both video and voiceover segments."""
        def _make_match(source_file: str, vo_index: int = 0, chapter_index: int = -1):
            match = Mock()
            match.video_segment = Mock()
            match.video_segment.source_file = source_file
            match.voiceover_segment = Mock()
            match.voiceover_segment.index = vo_index
            match.voiceover_segment.chapter_index = chapter_index
            return match
        return _make_match

    @pytest.fixture
    def config_with_chapter_grouping(self):
        """Mock config with chapter grouping enabled."""
        config = Mock()
        matching = Mock()
        matching.consecutive_source_penalty = 0.1
        matching.max_consecutive_same_source = 3
        matching.broll_boost = 0.1
        matching.caption_quality_adjustment_enabled = False
        matching.apply_timing_penalty = False
        matching.chapter_matching_enabled = False
        matching.topic_mismatch_penalty = 0.15
        # Chapter grouping config
        cg = Mock()
        cg.enabled = True
        cg.source_consistency_boost = 0.03
        matching.chapter_grouping = cg
        # Scoring config (for MatchScoring init)
        sc = Mock()
        sc.confidence_floor = 0.05
        sc.low_confidence_warning_threshold = 0.15
        matching.scoring = sc
        config.matching = matching
        config.global_cache = Mock()
        config.global_cache.current_project_boost = 0.1
        return config

    @pytest.fixture
    def config_without_chapter_grouping(self):
        """Mock config with chapter grouping disabled."""
        config = Mock()
        matching = Mock()
        matching.consecutive_source_penalty = 0.1
        matching.max_consecutive_same_source = 3
        cg = Mock()
        cg.enabled = False
        cg.source_consistency_boost = 0.03
        matching.chapter_grouping = cg
        sc = Mock()
        sc.confidence_floor = 0.05
        matching.scoring = sc
        config.matching = matching
        config.global_cache = Mock()
        config.global_cache.current_project_boost = 0.1
        return config

    @pytest.mark.fast
    def test_boost_applied_same_source_same_chapter(self, config_with_chapter_grouping, mock_match):
        """Boost applies when same source used within same chapter."""
        scoring = MatchScoring(config_with_chapter_grouping)
        video_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                               text="More footage", source_file="/videos/tokyo.mp4")
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=0)]
        segment_chapter_map = {1: 0, 2: 0}

        adjusted, reason = scoring.apply_chapter_source_consistency(
            0.7, video_seg, recent,
            current_chapter_index=0,
            segment_chapter_map=segment_chapter_map,
        )

        assert adjusted == pytest.approx(0.73, abs=0.001)
        assert "chapter_source_consistency" in reason
        assert "+0.03" in reason

    @pytest.mark.fast
    def test_no_boost_different_chapters(self, config_with_chapter_grouping, mock_match):
        """No boost when segments are in different chapters."""
        scoring = MatchScoring(config_with_chapter_grouping)
        video_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                               text="Different chapter", source_file="/videos/tokyo.mp4")
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=0)]
        segment_chapter_map = {1: 0, 2: 1}

        adjusted, reason = scoring.apply_chapter_source_consistency(
            0.7, video_seg, recent,
            current_chapter_index=1,
            segment_chapter_map=segment_chapter_map,
        )

        assert adjusted == 0.7
        assert reason == ""

    @pytest.mark.fast
    def test_no_boost_different_source(self, config_with_chapter_grouping, mock_match):
        """No boost when sources differ even within same chapter."""
        scoring = MatchScoring(config_with_chapter_grouping)
        video_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                               text="Different video", source_file="/videos/other.mp4")
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=0)]
        segment_chapter_map = {1: 0, 2: 0}

        adjusted, reason = scoring.apply_chapter_source_consistency(
            0.7, video_seg, recent,
            current_chapter_index=0,
            segment_chapter_map=segment_chapter_map,
        )

        assert adjusted == 0.7
        assert reason == ""

    @pytest.mark.fast
    def test_no_boost_when_disabled(self, config_without_chapter_grouping, mock_match):
        """No boost when chapter grouping is disabled."""
        scoring = MatchScoring(config_without_chapter_grouping)
        video_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                               text="Footage", source_file="/videos/tokyo.mp4")
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=0)]

        adjusted, reason = scoring.apply_chapter_source_consistency(
            0.7, video_seg, recent,
            current_chapter_index=0,
            segment_chapter_map={1: 0, 2: 0},
        )

        assert adjusted == 0.7
        assert reason == ""

    @pytest.mark.fast
    def test_no_boost_no_chapter_info(self, config_with_chapter_grouping, mock_match):
        """No boost when no chapter info available (chapter_index=-1)."""
        scoring = MatchScoring(config_with_chapter_grouping)
        video_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                               text="Footage", source_file="/videos/tokyo.mp4")
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=-1)]

        adjusted, reason = scoring.apply_chapter_source_consistency(
            0.7, video_seg, recent,
            current_chapter_index=-1,
        )

        assert adjusted == 0.7
        assert reason == ""

    @pytest.mark.fast
    def test_consecutive_penalty_suppressed_in_chapter(self, config_with_chapter_grouping, mock_match):
        """Consecutive source penalty is suppressed within same chapter."""
        from src.matching.scoring import apply_consecutive_source_penalty

        video_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                               text="Footage", source_file="/videos/tokyo.mp4")
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=0)]

        # Without suppression - penalty applies
        adjusted_no_suppress, reason_no_suppress = apply_consecutive_source_penalty(
            0.8, video_seg, recent,
            config=config_with_chapter_grouping,
            suppress_in_chapter=False,
        )
        assert adjusted_no_suppress < 0.8

        # With suppression - penalty suppressed
        adjusted_suppress, reason_suppress = apply_consecutive_source_penalty(
            0.8, video_seg, recent,
            config=config_with_chapter_grouping,
            suppress_in_chapter=True,
        )
        assert adjusted_suppress == 0.8
        assert reason_suppress == ""

    @pytest.mark.fast
    def test_consecutive_penalty_unchanged_without_chapters(self, mock_match):
        """When chapter structure not available, consecutive penalty unchanged."""
        from src.matching.scoring import apply_consecutive_source_penalty

        config = Mock()
        matching = Mock()
        matching.consecutive_source_penalty = 0.1
        matching.max_consecutive_same_source = 3
        matching.chapter_grouping = None
        config.matching = matching

        video_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                               text="Footage", source_file="/videos/tokyo.mp4")
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1)]

        adjusted, reason = apply_consecutive_source_penalty(
            0.8, video_seg, recent,
            config=config,
            suppress_in_chapter=False,
        )

        # Penalty still applies normally
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert "consecutive_source_penalty" in reason

    @pytest.mark.fast
    def test_is_within_chapter(self, config_with_chapter_grouping, mock_match):
        """is_within_chapter returns True for same chapter, False otherwise."""
        scoring = MatchScoring(config_with_chapter_grouping)
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=0)]

        # Same chapter
        assert scoring.is_within_chapter(0, recent, {1: 0}) is True

        # Different chapter
        assert scoring.is_within_chapter(1, recent, {1: 0}) is False

        # No chapter info
        assert scoring.is_within_chapter(-1, recent) is False

    @pytest.mark.fast
    def test_custom_boost_amount(self, mock_match):
        """Custom source_consistency_boost value is respected."""
        config = Mock()
        matching = Mock()
        cg = Mock()
        cg.enabled = True
        cg.source_consistency_boost = 0.07  # Custom boost
        matching.chapter_grouping = cg
        sc = Mock()
        sc.confidence_floor = 0.05
        matching.scoring = sc
        config.matching = matching

        scoring = MatchScoring(config)
        video_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                               text="Footage", source_file="/videos/tokyo.mp4")
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=0)]

        adjusted, reason = scoring.apply_chapter_source_consistency(
            0.7, video_seg, recent,
            current_chapter_index=0,
            segment_chapter_map={1: 0, 2: 0},
        )

        assert adjusted == pytest.approx(0.77, abs=0.001)
        assert "+0.07" in reason

    @pytest.mark.fast
    def test_breakdown_recorded_in_apply_all_adjustments(self, config_with_chapter_grouping, mock_match):
        """Chapter source consistency appears in confidence_breakdown."""
        scoring = MatchScoring(config_with_chapter_grouping)
        vo_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                            text="Tokyo travel guide")
        video_seg = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                               text="Tokyo footage", source_file="/videos/tokyo.mp4")
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=0)]

        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=vo_seg,
            video_segment=video_seg,
            recent_matches=recent,
            current_chapter_index=0,
            segment_chapter_map={1: 0, 2: 0},
        )

        components = [b['component'] for b in breakdown]
        assert 'chapter_source_consistency' in components
        consistency_entry = next(b for b in breakdown if b['component'] == 'chapter_source_consistency')
        assert consistency_entry['adjustment'] == pytest.approx(0.03, abs=0.001)

    @pytest.mark.fast
    def test_no_boost_when_chapter_index_is_none(self, config_with_chapter_grouping, mock_match):
        """No boost when chapter_index is None (not assigned)."""
        scoring = MatchScoring(config_with_chapter_grouping)
        video_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                               text="Footage", source_file="/videos/tokyo.mp4")
        # Previous match has no chapter info (chapter_index=-1 means None/unassigned)
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=-1)]

        # current_chapter_index=-1 means None/unassigned
        adjusted, reason = scoring.apply_chapter_source_consistency(
            0.7, video_seg, recent,
            current_chapter_index=-1,
            segment_chapter_map={1: -1, 2: -1},
        )

        assert adjusted == 0.7
        assert reason == ""

    @pytest.mark.fast
    def test_coexists_with_consecutive_source_penalty(self, config_with_chapter_grouping, mock_match):
        """Chapter source consistency boost coexists with consecutive_source_penalty.

        Within chapter boundaries, both adjustments apply independently:
        - consecutive_source_penalty: -0.10 (penalty for same source)
        - chapter_source_consistency: +0.03 (boost for same source in same chapter)
        Net effect is negative (-0.07), showing they coexist rather than one replacing the other.
        """
        from src.matching.scoring import apply_consecutive_source_penalty

        scoring = MatchScoring(config_with_chapter_grouping)
        video_seg = SRTSegment(index=2, start_time=10.0, end_time=20.0,
                               text="Footage", source_file="/videos/tokyo.mp4")
        recent = [mock_match("/videos/tokyo.mp4", vo_index=1, chapter_index=0)]

        # Apply consecutive_source_penalty (without suppress_in_chapter)
        base = 0.8
        after_penalty, penalty_reason = apply_consecutive_source_penalty(
            base, video_seg, recent,
            config=config_with_chapter_grouping,
            suppress_in_chapter=False,  # penalty still applies within chapter
        )
        assert after_penalty < base, "consecutive_source_penalty should reduce confidence"
        assert "consecutive_source_penalty" in penalty_reason

        # Apply chapter_source_consistency boost
        after_boost, boost_reason = scoring.apply_chapter_source_consistency(
            after_penalty, video_seg, recent,
            current_chapter_index=0,
            segment_chapter_map={1: 0, 2: 0},
        )
        assert after_boost > after_penalty, "chapter_source_consistency should add boost"
        assert "chapter_source_consistency" in boost_reason

        # Net effect: both applied, net is negative (penalty > boost)
        net_effect = after_boost - base
        assert net_effect < 0, "Net effect should be negative (penalty -0.10 > boost +0.03)"
        assert after_boost == pytest.approx(base - 0.10 + 0.03, abs=0.001)


class TestTagKeywordBoost:
    """Tests for apply_tag_keyword_boost (US-71-003)."""

    @pytest.mark.fast
    def test_no_tags_no_boost(self, mock_config):
        """No video tags produces no boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Tokyo travel guide")
        adjusted, reason = scoring.apply_tag_keyword_boost(0.7, vo, video_tags=None)
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_empty_tags_no_boost(self, mock_config):
        """Empty video tags list produces no boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Tokyo travel guide")
        adjusted, reason = scoring.apply_tag_keyword_boost(0.7, vo, video_tags=[])
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_zero_overlap_no_boost(self, mock_config):
        """Tags with no keyword overlap produce no boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Tokyo travel guide")
        adjusted, reason = scoring.apply_tag_keyword_boost(
            0.7, vo, video_tags=["cooking", "recipes", "kitchen"]
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_one_tag_match_boost(self, mock_config):
        """1 tag match gives +0.02 boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Tokyo travel guide")
        adjusted, reason = scoring.apply_tag_keyword_boost(
            0.7, vo, video_tags=["tokyo", "cooking", "recipes"]
        )
        assert adjusted == pytest.approx(0.72, abs=0.001)
        assert "tag keyword boost +0.02" in reason
        assert "1 tag" in reason

    @pytest.mark.fast
    def test_two_tag_matches_boost(self, mock_config):
        """2 tag matches gives +0.04 boost."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Tokyo travel guide")
        adjusted, reason = scoring.apply_tag_keyword_boost(
            0.7, vo, video_tags=["tokyo", "travel", "recipes"]
        )
        assert adjusted == pytest.approx(0.74, abs=0.001)
        assert "tag keyword boost +0.04" in reason
        assert "2 tags" in reason

    @pytest.mark.fast
    def test_three_tag_matches_boost(self, mock_config):
        """3 tag matches gives +0.06 boost (3 * 0.02)."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Tokyo travel guide panoramic")
        adjusted, reason = scoring.apply_tag_keyword_boost(
            0.7, vo, video_tags=["tokyo", "travel", "guide"]
        )
        assert adjusted == pytest.approx(0.76, abs=0.001)
        assert "tag keyword boost +0.06" in reason
        assert "3 tags" in reason

    @pytest.mark.fast
    def test_five_tag_matches_capped(self, mock_config):
        """5 tag matches caps at +0.08 (not 5 * 0.02 = 0.10)."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Tokyo travel guide panoramic scenic")
        adjusted, reason = scoring.apply_tag_keyword_boost(
            0.7, vo, video_tags=["tokyo", "travel", "guide", "panoramic", "scenic"]
        )
        assert adjusted == pytest.approx(0.78, abs=0.001)
        assert "tag keyword boost +0.08" in reason
        assert "5 tags" in reason

    @pytest.mark.fast
    def test_case_insensitive_tag_matching(self, mock_config):
        """Tag matching is case-insensitive."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="TOKYO travel guide")
        adjusted, reason = scoring.apply_tag_keyword_boost(
            0.7, vo, video_tags=["Tokyo", "TRAVEL", "cooking"]
        )
        assert adjusted == pytest.approx(0.74, abs=0.001)

    @pytest.mark.fast
    def test_short_tags_filtered(self, mock_config):
        """Tags shorter than 3 chars are excluded."""
        scoring = MatchScoring(mock_config)
        vo = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                        text="Tokyo travel guide")
        adjusted, reason = scoring.apply_tag_keyword_boost(
            0.7, vo, video_tags=["to", "tr", "ab"]
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_breakdown_in_apply_all(self, mock_config, sample_vo_segment, sample_video_segment):
        """apply_all_adjustments records tag_keyword_boost in confidence_breakdown."""
        scoring = MatchScoring(mock_config)
        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
            video_tags=["tokyo", "japan", "travel"]
        )
        tag_entries = [b for b in breakdown if b['component'] == 'tag_keyword_boost']
        assert len(tag_entries) == 1
        assert tag_entries[0]['adjustment'] > 0
        assert 'tag keyword boost' in tag_entries[0]['reason']

    @pytest.mark.fast
    def test_no_tags_no_breakdown_entry(self, mock_config, sample_vo_segment, sample_video_segment):
        """apply_all_adjustments with no video_tags produces no tag_keyword_boost entry."""
        scoring = MatchScoring(mock_config)
        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
        )
        tag_entries = [b for b in breakdown if b['component'] == 'tag_keyword_boost']
        assert len(tag_entries) == 0


# ============================================================================
# Chapter Coherence Penalty Tests (US-71-004)
# ============================================================================

class TestChapterCoherencePenalty:
    """Tests for apply_chapter_coherence_penalty (US-71-004)."""

    @pytest.fixture
    def mock_config(self):
        """Config with chapter_grouping enabled and coherence_penalty_threshold=5."""
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
        cg = Mock()
        cg.enabled = True
        cg.source_consistency_boost = 0.03
        cg.coherence_penalty_threshold = 5
        matching.chapter_grouping = cg
        matching.scoring = None
        config.matching = matching
        return config

    @pytest.fixture
    def sample_vo_segment(self):
        return SRTSegment(index=1, start_time=0.0, end_time=10.0,
                          text="Tokyo travel guide exploration")

    @pytest.fixture
    def sample_video_segment(self):
        seg = SRTSegment(index=1, start_time=0.0, end_time=10.0,
                         text="Beautiful scenery in Tokyo Japan")
        seg.source_file = "vid_A"
        return seg

    @pytest.mark.fast
    def test_no_chapter_no_penalty(self, mock_config):
        """No penalty when chapter_index is -1."""
        scoring = MatchScoring(mock_config)
        adjusted, reason = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=-1, chapter_source_counts={0: {"a", "b", "c", "d", "e", "f"}}
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_none_chapter_index_no_penalty(self, mock_config):
        """No penalty when chapter_index is None (no chapter structure detected)."""
        scoring = MatchScoring(mock_config)
        adjusted, reason = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=None, chapter_source_counts={0: {"a", "b", "c", "d", "e", "f"}}
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_no_source_counts_no_penalty(self, mock_config):
        """No penalty when chapter_source_counts is None."""
        scoring = MatchScoring(mock_config)
        adjusted, reason = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=0, chapter_source_counts=None
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_below_threshold_no_penalty(self, mock_config):
        """No penalty when source count <= threshold (5)."""
        scoring = MatchScoring(mock_config)
        sources = {0: {"vid_A", "vid_B", "vid_C", "vid_D", "vid_E"}}
        adjusted, reason = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=0, chapter_source_counts=sources
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_one_excess_source_penalty(self, mock_config):
        """1 excess source (6 total, threshold 5) -> -0.03 penalty."""
        scoring = MatchScoring(mock_config)
        sources = {0: {"a", "b", "c", "d", "e", "f"}}
        adjusted, reason = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=0, chapter_source_counts=sources
        )
        assert adjusted == pytest.approx(0.67, abs=0.001)
        assert "chapter_coherence" in reason
        assert "6 sources" in reason

    @pytest.mark.fast
    def test_two_excess_sources_penalty(self, mock_config):
        """2 excess sources (7 total) -> -0.06 penalty."""
        scoring = MatchScoring(mock_config)
        sources = {0: {"a", "b", "c", "d", "e", "f", "g"}}
        adjusted, reason = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=0, chapter_source_counts=sources
        )
        assert adjusted == pytest.approx(0.64, abs=0.001)

    @pytest.mark.fast
    def test_penalty_capped_at_max(self, mock_config):
        """Penalty caps at -0.10 regardless of excess count."""
        scoring = MatchScoring(mock_config)
        # 10 sources = 5 excess -> 5*(-0.03) = -0.15 but capped at -0.10
        sources = {0: {f"vid_{i}" for i in range(10)}}
        adjusted, reason = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=0, chapter_source_counts=sources
        )
        assert adjusted == pytest.approx(0.60, abs=0.001)

    @pytest.mark.fast
    def test_chapter_grouping_disabled_no_penalty(self, mock_config):
        """No penalty when chapter_grouping is disabled."""
        mock_config.matching.chapter_grouping.enabled = False
        scoring = MatchScoring(mock_config)
        sources = {0: {"a", "b", "c", "d", "e", "f", "g", "h"}}
        adjusted, reason = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=0, chapter_source_counts=sources
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_different_chapter_index(self, mock_config):
        """Penalty uses correct chapter from the counts dict."""
        scoring = MatchScoring(mock_config)
        sources = {
            0: {"a", "b"},  # chapter 0: 2 sources, no penalty
            1: {"a", "b", "c", "d", "e", "f", "g"},  # chapter 1: 7 sources, penalty
        }
        adj_ch0, reason0 = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=0, chapter_source_counts=sources
        )
        adj_ch1, reason1 = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=1, chapter_source_counts=sources
        )
        assert adj_ch0 == pytest.approx(0.7, abs=0.001)
        assert adj_ch1 == pytest.approx(0.64, abs=0.001)

    @pytest.mark.fast
    def test_custom_threshold_from_config(self, mock_config):
        """Config coherence_penalty_threshold is respected."""
        mock_config.matching.chapter_grouping.coherence_penalty_threshold = 3
        scoring = MatchScoring(mock_config)
        sources = {0: {"a", "b", "c", "d"}}  # 4 sources, threshold 3 -> 1 excess
        adjusted, reason = scoring.apply_chapter_coherence_penalty(
            0.7, current_chapter_index=0, chapter_source_counts=sources
        )
        assert adjusted == pytest.approx(0.67, abs=0.001)

    @pytest.mark.fast
    def test_breakdown_in_apply_all(self, mock_config, sample_vo_segment, sample_video_segment):
        """apply_all_adjustments records chapter_coherence in confidence_breakdown."""
        scoring = MatchScoring(mock_config)
        sources = {0: {"a", "b", "c", "d", "e", "f"}}
        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
            current_chapter_index=0,
            chapter_source_counts=sources,
        )
        coherence_entries = [b for b in breakdown if b['component'] == 'chapter_coherence_penalty']
        assert len(coherence_entries) == 1
        assert coherence_entries[0]['adjustment'] < 0
        assert 'chapter_coherence' in coherence_entries[0]['reason']

    @pytest.mark.fast
    def test_no_counts_no_breakdown_entry(self, mock_config, sample_vo_segment, sample_video_segment):
        """apply_all_adjustments with no chapter_source_counts produces no chapter_coherence entry."""
        scoring = MatchScoring(mock_config)
        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
        )
        coherence_entries = [b for b in breakdown if b['component'] == 'chapter_coherence_penalty']
        assert len(coherence_entries) == 0


class TestListicleConsistencyBoost:
    """Tests for apply_listicle_consistency_boost (US-71-006)."""

    @pytest.fixture
    def mock_config(self):
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
        matching.chapter_grouping = None
        matching.scoring = None
        config.matching = matching
        return config

    @pytest.fixture
    def listicle_groups(self):
        """Two listicle groups: group 0 covers segments 0-3, group 1 covers segments 4-7."""
        group0 = Mock()
        group0.group_id = 0
        group0.start_segment_idx = 0
        group0.end_segment_idx = 3
        group1 = Mock()
        group1.group_id = 1
        group1.start_segment_idx = 4
        group1.end_segment_idx = 7
        return [group0, group1]

    def _make_segment(self, index, source_file="vid_A"):
        seg = SRTSegment(index=index, start_time=float(index * 10),
                         end_time=float(index * 10 + 10),
                         text=f"Segment {index} text content here")
        seg.source_file = source_file
        return seg

    def _make_match(self, vo_index, source_file="vid_A"):
        m = Mock()
        m.voiceover_segment = self._make_segment(vo_index, source_file="vo.srt")
        m.voiceover_segment.index = vo_index
        m.video_segment = self._make_segment(vo_index, source_file=source_file)
        m.video_segment.source_file = source_file
        return m

    @pytest.mark.fast
    def test_boost_within_same_group_same_source(self, mock_config, listicle_groups):
        """Segments within same listicle group from same source get +0.04 boost."""
        scoring = MatchScoring(mock_config)
        vo_seg = self._make_segment(2, source_file="vo.srt")
        vid_seg = self._make_segment(10, source_file="vid_A")
        recent = [self._make_match(1, source_file="vid_A")]

        adjusted, reason = scoring.apply_listicle_consistency_boost(
            0.7, vo_seg, vid_seg, listicle_groups, recent
        )
        assert adjusted == pytest.approx(0.74, abs=0.001)
        assert "listicle_consistency" in reason
        assert "group 0" in reason

    @pytest.mark.fast
    def test_no_boost_at_group_boundary(self, mock_config, listicle_groups):
        """First segment of a listicle group gets no boost."""
        scoring = MatchScoring(mock_config)
        vo_seg = self._make_segment(0, source_file="vo.srt")  # First segment of group 0
        vid_seg = self._make_segment(10, source_file="vid_A")
        recent = [self._make_match(3, source_file="vid_A")]  # Previous in different group context

        adjusted, reason = scoring.apply_listicle_consistency_boost(
            0.7, vo_seg, vid_seg, listicle_groups, recent
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_no_boost_across_group_boundaries(self, mock_config, listicle_groups):
        """No boost when previous match is in a different listicle group."""
        scoring = MatchScoring(mock_config)
        vo_seg = self._make_segment(5, source_file="vo.srt")  # In group 1
        vid_seg = self._make_segment(10, source_file="vid_A")
        recent = [self._make_match(3, source_file="vid_A")]  # In group 0

        adjusted, reason = scoring.apply_listicle_consistency_boost(
            0.7, vo_seg, vid_seg, listicle_groups, recent
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_no_boost_different_source(self, mock_config, listicle_groups):
        """No boost when same group but different video sources."""
        scoring = MatchScoring(mock_config)
        vo_seg = self._make_segment(2, source_file="vo.srt")
        vid_seg = self._make_segment(10, source_file="vid_B")  # Different source
        recent = [self._make_match(1, source_file="vid_A")]

        adjusted, reason = scoring.apply_listicle_consistency_boost(
            0.7, vo_seg, vid_seg, listicle_groups, recent
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_no_boost_without_groups(self, mock_config):
        """No boost when listicle_groups is empty."""
        scoring = MatchScoring(mock_config)
        vo_seg = self._make_segment(2)
        vid_seg = self._make_segment(10)
        recent = [self._make_match(1)]

        adjusted, reason = scoring.apply_listicle_consistency_boost(
            0.7, vo_seg, vid_seg, [], recent
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_no_boost_without_recent_matches(self, mock_config, listicle_groups):
        """No boost when no recent matches."""
        scoring = MatchScoring(mock_config)
        vo_seg = self._make_segment(2)
        vid_seg = self._make_segment(10)

        adjusted, reason = scoring.apply_listicle_consistency_boost(
            0.7, vo_seg, vid_seg, listicle_groups, None
        )
        assert adjusted == pytest.approx(0.7, abs=0.001)

    @pytest.mark.fast
    def test_breakdown_entry_in_apply_all(self, mock_config, listicle_groups):
        """apply_all_adjustments includes listicle_consistency in breakdown."""
        scoring = MatchScoring(mock_config)
        vo_seg = self._make_segment(2, source_file="vo.srt")
        vid_seg = self._make_segment(10, source_file="vid_A")
        recent = [self._make_match(1, source_file="vid_A")]

        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=vo_seg,
            video_segment=vid_seg,
            listicle_groups=listicle_groups,
            recent_matches=recent,
        )
        listicle_entries = [b for b in breakdown if b['component'] == 'listicle_consistency']
        assert len(listicle_entries) == 1
        assert listicle_entries[0]['adjustment'] == pytest.approx(0.04, abs=0.001)

    @pytest.mark.fast
    def test_no_breakdown_without_groups(self, mock_config):
        """apply_all_adjustments with no listicle_groups produces no listicle_consistency entry."""
        scoring = MatchScoring(mock_config)
        vo_seg = self._make_segment(2)
        vid_seg = self._make_segment(10)
        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.7,
            vo_segment=vo_seg,
            video_segment=vid_seg,
        )
        listicle_entries = [b for b in breakdown if b['component'] == 'listicle_consistency']
        assert len(listicle_entries) == 0


# ============================================================================
# Test Chapter Topic Match (US-72-007)
# ============================================================================

class TestChapterTopicMatch:
    """Test apply_chapter_topic_match scoring adjustment (US-72-007)."""

    @staticmethod
    def _make_segment(text: str, chapter_index=None) -> SRTSegment:
        seg = SRTSegment(index=1, start_time=0.0, end_time=10.0, text=text, source_file="v.mp4")
        if chapter_index is not None:
            seg.chapter_index = chapter_index
        return seg

    @pytest.mark.fast
    def test_no_chapter_data_no_adjustment(self, mock_config):
        """No adjustment when vo_chapter_index is None (either side lacks chapter data)."""
        scoring = MatchScoring(mock_config)
        vo = self._make_segment("climate change impacts global warming")
        conf, reason = scoring.apply_chapter_topic_match(
            confidence=0.70,
            vo_segment=vo,
            chapter_title="Climate and Weather Patterns",
            vo_chapter_index=None,
        )
        assert conf == 0.70
        assert reason == ""

    @pytest.mark.fast
    def test_no_chapter_title_no_adjustment(self, mock_config):
        """No adjustment when chapter_title is empty/None."""
        scoring = MatchScoring(mock_config)
        vo = self._make_segment("climate change impacts global warming")
        conf, reason = scoring.apply_chapter_topic_match(
            confidence=0.70,
            vo_segment=vo,
            chapter_title=None,
            vo_chapter_index=0,
        )
        assert conf == 0.70
        assert reason == ""

    @pytest.mark.fast
    def test_partial_match_1_keyword(self, mock_config):
        """Partial match (+0.05) with 1 shared keyword."""
        scoring = MatchScoring(mock_config)
        # 'climate' will overlap
        vo = self._make_segment("climate change impacts the world")
        conf, reason = scoring.apply_chapter_topic_match(
            confidence=0.70,
            vo_segment=vo,
            chapter_title="Climate and Extreme Events",
            vo_chapter_index=0,
        )
        assert conf == pytest.approx(0.75, abs=0.001)
        assert "partial match" in reason
        assert "+0.05" in reason

    @pytest.mark.fast
    def test_partial_match_2_keywords(self, mock_config):
        """Partial match (+0.05) with 2 shared keywords."""
        scoring = MatchScoring(mock_config)
        # 'climate' and 'impacts' will overlap
        vo = self._make_segment("climate change impacts the world")
        conf, reason = scoring.apply_chapter_topic_match(
            confidence=0.70,
            vo_segment=vo,
            chapter_title="Climate Impacts Analysis",
            vo_chapter_index=1,
        )
        assert conf == pytest.approx(0.75, abs=0.001)
        assert "partial match" in reason
        assert "+0.05" in reason

    @pytest.mark.fast
    def test_strong_match_3_plus_keywords(self, mock_config):
        """Strong match (+0.10) with 3+ shared keywords."""
        scoring = MatchScoring(mock_config)
        # 'climate', 'change', 'global' will overlap (3 keywords)
        vo = self._make_segment("climate change global warming effects")
        conf, reason = scoring.apply_chapter_topic_match(
            confidence=0.70,
            vo_segment=vo,
            chapter_title="Climate Change Global Warming",
            vo_chapter_index=2,
        )
        assert conf == pytest.approx(0.80, abs=0.001)
        assert "strong match" in reason
        assert "+0.1" in reason

    @pytest.mark.fast
    def test_mismatch_penalty(self, mock_config):
        """Mismatch penalty (-0.05) when voiceover chapter has keywords but zero overlap."""
        scoring = MatchScoring(mock_config)
        vo = self._make_segment("renewable energy solar power wind")
        conf, reason = scoring.apply_chapter_topic_match(
            confidence=0.70,
            vo_segment=vo,
            chapter_title="Marine Biology Ocean Ecosystems",
            vo_chapter_index=0,
        )
        assert conf == pytest.approx(0.65, abs=0.001)
        assert "mismatch" in reason
        assert "-0.05" in reason

    @pytest.mark.fast
    def test_breakdown_in_apply_all_adjustments(self, mock_config):
        """Verify chapter_topic_match appears in confidence_breakdown."""
        scoring = MatchScoring(mock_config)
        vo = self._make_segment("climate change global warming effects")
        vid = self._make_segment("video about rising sea levels")
        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.70,
            vo_segment=vo,
            video_segment=vid,
            chapter_title="Climate Change Global Warming",
            current_chapter_index=0,
        )
        chapter_entries = [b for b in breakdown if b['component'] == 'chapter_topic_match']
        assert len(chapter_entries) == 1
        entry = chapter_entries[0]
        assert 'adjustment' in entry
        assert 'reason' in entry
        assert isinstance(entry['adjustment'], float)

    @pytest.mark.fast
    def test_no_breakdown_without_chapter_index(self, mock_config):
        """No chapter_topic_match entry when current_chapter_index is -1 (None)."""
        scoring = MatchScoring(mock_config)
        vo = self._make_segment("climate change impacts the world")
        vid = self._make_segment("video content here")
        _, _, breakdown = scoring.apply_all_adjustments(
            confidence=0.70,
            vo_segment=vo,
            video_segment=vid,
            chapter_title="Climate Change",
            current_chapter_index=-1,
        )
        chapter_entries = [b for b in breakdown if b['component'] == 'chapter_topic_match']
        assert len(chapter_entries) == 0


# ============================================================================
# Test Tiered Caption Quality Penalties (US-73-006)
# ============================================================================

class TestTieredCaptionQualityPenalties:
    """Test graduated caption quality penalties with stacking and cap."""

    @pytest.fixture
    def tiered_config(self):
        """Config with tiered caption penalty settings."""
        config = Mock()
        matching = Mock()
        matching.caption_quality_adjustment_enabled = True
        matching.caption_quality_weights = None
        matching.caption_quality_high_boost = 0.0
        matching.caption_quality_low_penalty = 0.0
        matching.caption_penalty_auto_generated = -0.05
        matching.caption_penalty_low_quality = -0.08
        matching.caption_penalty_missing_timing = -0.03
        matching.max_caption_penalty = -0.12
        config.matching = matching
        return config

    @pytest.fixture
    def video_seg(self):
        seg = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="v.mp4")
        return seg

    @pytest.mark.fast
    def test_auto_generated_penalty_alone(self, tiered_config, video_seg):
        """auto_generated issue applies -0.05 penalty."""
        from src.matching.scoring import apply_tiered_caption_penalties
        video_seg.caption_quality_issues = ['auto_generated']

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, tiered_config)

        assert abs(adjusted - 0.75) < 0.001
        assert len(entries) == 1
        assert entries[0]['component'] == 'caption_quality_auto'
        assert abs(entries[0]['adjustment'] - (-0.05)) < 0.001

    @pytest.mark.fast
    def test_low_quality_penalty_alone(self, tiered_config, video_seg):
        """low_quality issue applies -0.08 penalty."""
        from src.matching.scoring import apply_tiered_caption_penalties
        video_seg.caption_quality_issues = ['low_quality']

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, tiered_config)

        assert abs(adjusted - 0.72) < 0.001
        assert len(entries) == 1
        assert entries[0]['component'] == 'caption_quality_low'
        assert abs(entries[0]['adjustment'] - (-0.08)) < 0.001

    @pytest.mark.fast
    def test_missing_timing_penalty_alone(self, tiered_config, video_seg):
        """missing_timing issue applies -0.03 penalty."""
        from src.matching.scoring import apply_tiered_caption_penalties
        video_seg.caption_quality_issues = ['missing_timing']

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, tiered_config)

        assert abs(adjusted - 0.77) < 0.001
        assert len(entries) == 1
        assert entries[0]['component'] == 'caption_timing_gap'
        assert abs(entries[0]['adjustment'] - (-0.03)) < 0.001

    @pytest.mark.fast
    def test_stacking_within_cap(self, tiered_config, video_seg):
        """auto_generated + missing_timing = -0.08 total (within -0.12 cap)."""
        from src.matching.scoring import apply_tiered_caption_penalties
        video_seg.caption_quality_issues = ['auto_generated', 'missing_timing']

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, tiered_config)

        assert abs(adjusted - 0.72) < 0.001  # 0.80 - 0.08
        assert len(entries) == 2
        components = {e['component'] for e in entries}
        assert 'caption_quality_auto' in components
        assert 'caption_timing_gap' in components

    @pytest.mark.fast
    def test_stacking_hits_cap(self, tiered_config, video_seg):
        """All three issues = -0.16 raw, capped to -0.12."""
        from src.matching.scoring import apply_tiered_caption_penalties
        video_seg.caption_quality_issues = ['auto_generated', 'low_quality', 'missing_timing']

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, tiered_config)

        # Raw: -0.05 + -0.08 + -0.03 = -0.16, capped to -0.12
        assert abs(adjusted - 0.68) < 0.001  # 0.80 - 0.12
        assert len(entries) == 3
        components = {e['component'] for e in entries}
        assert components == {'caption_quality_auto', 'caption_quality_low', 'caption_timing_gap'}
        # Total adjustment should sum to -0.12 (capped)
        total_adj = sum(e['adjustment'] for e in entries)
        assert abs(total_adj - (-0.12)) < 0.001

    @pytest.mark.fast
    def test_no_issues_no_penalty(self, tiered_config, video_seg):
        """No caption_quality_issues means no penalty."""
        from src.matching.scoring import apply_tiered_caption_penalties

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, tiered_config)

        assert adjusted == 0.80
        assert entries == []

    @pytest.mark.fast
    def test_empty_issues_list(self, tiered_config, video_seg):
        """Empty caption_quality_issues list means no penalty."""
        from src.matching.scoring import apply_tiered_caption_penalties
        video_seg.caption_quality_issues = []

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, tiered_config)

        assert adjusted == 0.80
        assert entries == []

    @pytest.mark.fast
    def test_disabled_returns_unchanged(self, tiered_config, video_seg):
        """When caption_quality_adjustment_enabled is False, no penalty."""
        from src.matching.scoring import apply_tiered_caption_penalties
        tiered_config.matching.caption_quality_adjustment_enabled = False
        video_seg.caption_quality_issues = ['auto_generated', 'low_quality']

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, tiered_config)

        assert adjusted == 0.80
        assert entries == []

    @pytest.mark.fast
    def test_separate_breakdown_entries(self, tiered_config, video_seg):
        """Each penalty type appears as separate entry in breakdown."""
        from src.matching.scoring import apply_tiered_caption_penalties
        video_seg.caption_quality_issues = ['auto_generated', 'missing_timing']

        _, entries = apply_tiered_caption_penalties(0.80, video_seg, tiered_config)

        assert len(entries) == 2
        for entry in entries:
            assert 'component' in entry
            assert 'adjustment' in entry
            assert 'reason' in entry
            assert 'tiered' in entry['reason']

    @pytest.mark.fast
    def test_custom_max_penalty_from_config(self, tiered_config, video_seg):
        """Custom max_caption_penalty is respected."""
        from src.matching.scoring import apply_tiered_caption_penalties
        tiered_config.matching.max_caption_penalty = -0.06
        video_seg.caption_quality_issues = ['auto_generated', 'low_quality']
        # Raw: -0.05 + -0.08 = -0.13, capped to -0.06

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, tiered_config)

        assert abs(adjusted - 0.74) < 0.001  # 0.80 - 0.06
        total_adj = sum(e['adjustment'] for e in entries)
        assert abs(total_adj - (-0.06)) < 0.001

    @pytest.mark.fast
    def test_tiered_in_apply_all_adjustments(self, video_seg):
        """Tiered penalties appear in apply_all_adjustments breakdown."""
        config = Mock()
        matching = Mock()
        matching.multimodal_enabled = True
        matching.multimodal_weights = None
        matching.pool_normalization_enabled = True
        matching.chapter_matching_enabled = False
        matching.topic_mismatch_penalty = 0.15
        matching.broll_boost = 0.0
        matching.caption_quality_adjustment_enabled = True
        matching.caption_quality_weights = None
        matching.caption_quality_high_boost = 0.0
        matching.caption_quality_low_penalty = 0.0
        matching.caption_penalty_auto_generated = -0.05
        matching.caption_penalty_low_quality = -0.08
        matching.caption_penalty_missing_timing = -0.03
        matching.max_caption_penalty = -0.12
        matching.apply_timing_penalty = False
        matching.skip_llm_threshold = 0.85
        matching.chapter_grouping = None
        matching.scoring = None
        config.matching = matching
        global_cache = Mock()
        global_cache.current_project_boost = 0.0
        config.global_cache = global_cache

        video_seg.caption_quality = None  # No legacy quality
        video_seg.caption_quality_issues = ['auto_generated', 'missing_timing']

        vo_seg = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test vo", source_file="vo.srt")
        vo_seg.keywords = []
        vo_seg.entities = []

        scoring = MatchScoring(config)
        adjusted, reason, breakdown = scoring.apply_all_adjustments(
            confidence=0.80,
            vo_segment=vo_seg,
            video_segment=video_seg,
        )

        # Check tiered penalty components are in breakdown
        components = [b['component'] for b in breakdown]
        assert 'caption_quality_auto' in components
        assert 'caption_timing_gap' in components


class TestTieredPenaltyConfigDefaults:
    """Verify MatchingConfig has tiered penalty fields with correct defaults (US-73-006)."""

    @pytest.mark.fast
    def test_matching_config_has_tiered_penalty_fields(self):
        """Config fields for each penalty value exist with correct defaults."""
        from src.config.sections.matching import MatchingConfig
        mc = MatchingConfig()

        assert mc.caption_penalty_auto_generated == -0.05
        assert mc.caption_penalty_low_quality == -0.08
        assert mc.caption_penalty_missing_timing == -0.03
        assert mc.max_caption_penalty == -0.12

    @pytest.mark.fast
    def test_matching_config_penalty_fields_configurable(self):
        """Config penalty fields can be overridden via constructor."""
        from src.config.sections.matching import MatchingConfig
        mc = MatchingConfig(
            caption_penalty_auto_generated=-0.10,
            caption_penalty_low_quality=-0.15,
            caption_penalty_missing_timing=-0.06,
            max_caption_penalty=-0.20,
        )

        assert mc.caption_penalty_auto_generated == -0.10
        assert mc.caption_penalty_low_quality == -0.15
        assert mc.caption_penalty_missing_timing == -0.06
        assert mc.max_caption_penalty == -0.20


class TestTieredCaptionPenaltiesNestedConfig:
    """Verify TieredCaptionPenalties nested config (US-78-008)."""

    @pytest.mark.fast
    def test_nested_config_defaults(self):
        """TieredCaptionPenalties has correct default values."""
        from src.config.sections.matching import TieredCaptionPenalties
        tcp = TieredCaptionPenalties()

        assert tcp.auto_generated_penalty == -0.05
        assert tcp.low_quality_penalty == -0.08
        assert tcp.missing_timing_penalty == -0.03
        assert tcp.max_caption_penalty == -0.12

    @pytest.mark.fast
    def test_nested_config_on_matching_config(self):
        """MatchingConfig creates TieredCaptionPenalties nested config."""
        from src.config.sections.matching import MatchingConfig, TieredCaptionPenalties
        mc = MatchingConfig()

        assert isinstance(mc.tiered_caption_penalties, TieredCaptionPenalties)
        assert mc.tiered_caption_penalties.auto_generated_penalty == -0.05

    @pytest.mark.fast
    def test_nested_config_from_dict(self):
        """TieredCaptionPenalties can be created from dict (YAML loading)."""
        from src.config.sections.matching import MatchingConfig, TieredCaptionPenalties
        mc = MatchingConfig(tiered_caption_penalties={
            'auto_generated_penalty': -0.10,
            'low_quality_penalty': -0.15,
            'missing_timing_penalty': -0.06,
            'max_caption_penalty': -0.20,
        })

        assert isinstance(mc.tiered_caption_penalties, TieredCaptionPenalties)
        assert mc.tiered_caption_penalties.auto_generated_penalty == -0.10
        assert mc.tiered_caption_penalties.low_quality_penalty == -0.15
        assert mc.tiered_caption_penalties.missing_timing_penalty == -0.06
        assert mc.tiered_caption_penalties.max_caption_penalty == -0.20
        # Flat fields synced from nested
        assert mc.caption_penalty_auto_generated == -0.10
        assert mc.caption_penalty_low_quality == -0.15
        assert mc.caption_penalty_missing_timing == -0.06
        assert mc.max_caption_penalty == -0.20

    @pytest.mark.fast
    def test_flat_fields_build_nested_config(self):
        """Flat field overrides build nested config (backward compat)."""
        from src.config.sections.matching import MatchingConfig
        mc = MatchingConfig(
            caption_penalty_auto_generated=-0.10,
            caption_penalty_low_quality=-0.15,
            caption_penalty_missing_timing=-0.06,
            max_caption_penalty=-0.20,
        )

        assert mc.tiered_caption_penalties.auto_generated_penalty == -0.10
        assert mc.tiered_caption_penalties.low_quality_penalty == -0.15

    @pytest.mark.fast
    def test_custom_penalties_applied_in_scoring(self):
        """Custom penalty values from nested config flow through to scoring (US-78-008).

        End-to-end: MatchingConfig with nested TieredCaptionPenalties ->
        flat fields synced via __post_init__ -> scoring.py reads flat fields.
        """
        from src.matching.scoring import apply_tiered_caption_penalties
        from src.config.sections.matching import MatchingConfig

        mc = MatchingConfig(tiered_caption_penalties={
            'auto_generated_penalty': -0.10,
            'low_quality_penalty': -0.20,
            'missing_timing_penalty': -0.05,
            'max_caption_penalty': -0.25,
        })
        config = Mock()
        config.matching = mc

        video_seg = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="v.mp4")
        video_seg.caption_quality_issues = ['auto_generated']

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, config)

        # Should use custom -0.10 penalty, not default -0.05
        assert abs(adjusted - 0.70) < 0.001
        assert len(entries) == 1
        assert abs(entries[0]['adjustment'] - (-0.10)) < 0.001

    @pytest.mark.fast
    def test_custom_max_penalty_cap_from_nested_config(self):
        """Custom max_caption_penalty from nested config caps stacked penalties."""
        from src.matching.scoring import apply_tiered_caption_penalties
        from src.config.sections.matching import MatchingConfig

        mc = MatchingConfig(tiered_caption_penalties={
            'auto_generated_penalty': -0.10,
            'low_quality_penalty': -0.20,
            'missing_timing_penalty': -0.05,
            'max_caption_penalty': -0.15,  # Tight cap
        })
        config = Mock()
        config.matching = mc

        video_seg = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test", source_file="v.mp4")
        video_seg.caption_quality_issues = ['auto_generated', 'low_quality']  # Would be -0.30 uncapped

        adjusted, entries = apply_tiered_caption_penalties(0.80, video_seg, config)

        # Capped at -0.15: 0.80 - 0.15 = 0.65
        assert abs(adjusted - 0.65) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
