"""Integration tests for multi-factor scoring pipeline (US-117-012).

Tests that all scoring adjustments stack correctly without exceeding bounds,
interact properly between caption quality, context richness, and voiceover context,
and handle missing metadata gracefully.
"""

import pytest
from unittest.mock import Mock
from src.matching.scoring import (
    MatchScoring,
    apply_context_richness_calibration,
    apply_voiceover_context_calibration,
)
from src.utils import SRTSegment


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_matching_config():
    """Mock matching config with all scoring options enabled."""
    config = Mock()
    matching = Mock()
    matching.multimodal_enabled = True
    matching.multimodal_weights = None  # Use default weights
    matching.pool_normalization_enabled = True
    matching.chapter_matching_enabled = False
    matching.topic_mismatch_penalty = 0.15
    matching.broll_boost = 0.1
    matching.caption_quality_adjustment_enabled = True
    matching.caption_quality_high_boost = 0.05
    matching.caption_quality_low_penalty = 0.1
    matching.caption_quality_weights = None  # Use additive mode
    matching.apply_timing_penalty = True
    matching.skip_llm_threshold = 0.85
    matching.language_confidence_penalty = 0.0

    # Context richness config (US-95-010)
    matching.context_richness_enabled = True
    matching.context_richness_boost_max = 0.08
    matching.context_richness_penalty_max = 0.05

    # Voiceover context config (US-111-010)
    matching.voiceover_context_enabled = True
    matching.voiceover_context_boost_max = 0.05
    matching.voiceover_context_penalty_max = 0.03

    config.matching = matching
    return config


@pytest.fixture
def sample_vo_segment():
    """Sample voiceover segment with rich context."""
    seg = SRTSegment(
        index=5,
        start_time=50.0,
        end_time=60.0,
        text="The beautiful Tokyo skyline at night with neon lights",
        source_file="voiceover.srt"
    )
    seg.keywords = ["tokyo", "night", "skyline", "neon", "japan"]
    seg.entities = [{"text": "Tokyo", "type": "LOCATION"}]
    return seg


@pytest.fixture
def sample_video_segment():
    """Sample video segment."""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="Video footage of Tokyo city at night",
        source_file="/videos/tokyo_night.mp4"
    )
    seg.keywords = ["tokyo", "city", "night", "lights"]
    seg.entities = [{"text": "Tokyo", "type": "LOCATION"}]
    return seg


# ============================================================================
# Integration Tests: All Scoring Adjustments in Sequence
# ============================================================================

class TestMultiFactorScoringPipeline:
    """Integration tests for multi-factor scoring pipeline."""

    @pytest.mark.fast
    def test_all_adjustments_sequence_bounds(self, mock_matching_config, sample_vo_segment, sample_video_segment):
        """Test that all adjustments applied in sequence stay within [0.0, 1.0] bounds."""
        scoring = MatchScoring(mock_matching_config)

        # Set up video with all adjustments possible
        sample_video_segment.is_broll = False
        sample_video_segment.timing_penalty = 0.98  # High timing penalty
        sample_video_segment.caption_quality = 'high'

        # Apply all adjustments with base confidence at extremes
        for base_confidence in [0.0, 0.1, 0.5, 0.9, 1.0]:
            adjusted, reason, breakdown = scoring.apply_all_adjustments(
                confidence=base_confidence,
                vo_segment=sample_vo_segment,
                video_segment=sample_video_segment,
                video_topics=None,
                chapter_matching_enabled=False,
                topic_mismatch_penalty=0.15,
                has_prev_segment=True,
                has_next_segment=True,
                voiceover_length=10,
            )

            # Verify bounds
            assert 0.0 <= adjusted <= 1.0, f"Confidence {adjusted} out of bounds for base {base_confidence}"

    @pytest.mark.fast
    def test_caption_quality_context_richness_interaction(self, mock_matching_config, sample_vo_segment):
        """Test interaction between caption quality and context richness."""
        scoring = MatchScoring(mock_matching_config)

        # Rich context with high caption quality
        video_rich = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Video with rich metadata", source_file="v.mp4"
        )
        video_rich.caption_quality = 'high'
        video_rich.is_broll = False

        # Get base confidence from scoring
        base_conf, _, _ = scoring.calculate_confidence(
            embedding_similarity=0.85,
            keyword_score=0.7,
            entity_score=0.6,
            visual_score=0.0
        )

        # Apply context richness
        conf1, reason1 = apply_context_richness_calibration(
            confidence=base_conf,
            video_title="Python Tutorial",
            video_description="Learn Python programming",
            video_tags=["python", "programming"],
            video_chapter="Introduction",
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # Apply caption quality via MatchScoring
        conf2, reason2, _ = scoring.apply_all_adjustments(
            confidence=conf1,
            vo_segment=sample_vo_segment,
            video_segment=video_rich,
            video_topics=None,
        )

        # Both should boost confidence
        assert conf2 > base_conf, "Rich context + high caption should boost confidence"
        assert conf2 <= 1.0, "Should not exceed 1.0"

    @pytest.mark.fast
    def test_caption_quality_context_richness_voiceover_interaction(
        self, mock_matching_config, sample_vo_segment
    ):
        """Test interaction between caption quality, context richness, and voiceover context."""
        scoring = MatchScoring(mock_matching_config)

        # Base confidence from scoring
        base_confidence, _, _ = scoring.calculate_confidence(
            embedding_similarity=0.8,
            keyword_score=0.7,
            entity_score=0.6,
            visual_score=0.0
        )

        # Step 1: Apply context richness (rich context -> boost)
        conf1, _ = apply_context_richness_calibration(
            confidence=base_confidence,
            video_title="Amazing Travel Video",
            video_description="Explore the world",
            video_tags=["travel", "adventure"],
            video_chapter="Intro",
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # Create video with caption quality
        video = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Travel video", source_file="v.mp4"
        )
        video.caption_quality = 'high'
        video.is_broll = False

        # Step 2: Apply caption quality via apply_all_adjustments
        conf2, _, _ = scoring.apply_all_adjustments(
            confidence=conf1,
            vo_segment=sample_vo_segment,
            video_segment=video,
            video_topics=None,
            has_prev_segment=True,
            has_next_segment=True,
            voiceover_length=10,
        )

        # All three should compound to boost confidence
        assert conf2 > base_confidence, "All three factors should boost confidence"
        assert 0.0 <= conf2 <= 1.0, "Should stay within bounds"

    @pytest.mark.fast
    def test_confidence_stacks_without_exceeding_bounds_max_boost(self):
        """Test that max boosts from all factors don't exceed 1.0."""
        base_confidence = 0.95  # Near max

        # Apply max boost from context richness
        conf1, _ = apply_context_richness_calibration(
            confidence=base_confidence,
            video_title="Title", video_description="Desc",
            video_tags=["tag1"], video_chapter="Chapter",
            enabled=True, boost_max=0.08, penalty_max=0.05,
        )

        # Apply max boost from voiceover context
        conf3, _ = apply_voiceover_context_calibration(
            confidence=conf1, has_prev_segment=True, has_next_segment=True,
            enabled=True, boost_max=0.05, penalty_max=0.03,
            voiceover_length=10,
        )

        # Should not exceed 1.0
        assert conf3 <= 1.0, f"Max boost should not exceed 1.0, got {conf3}"

    @pytest.mark.fast
    def test_confidence_stays_above_zero_max_penalty(self):
        """Test that max penalties from all factors don't go below 0.0."""
        base_confidence = 0.05  # Near min

        # Apply max penalty from context richness (sparse context)
        conf1, _ = apply_context_richness_calibration(
            confidence=base_confidence,
            video_title=None, video_description=None,
            video_tags=None, video_chapter=None,
            enabled=True, boost_max=0.08, penalty_max=0.05,
        )

        # Apply max penalty from voiceover context (no adjacent segments)
        conf3, _ = apply_voiceover_context_calibration(
            confidence=conf1, has_prev_segment=False, has_next_segment=False,
            enabled=True, boost_max=0.05, penalty_max=0.03,
            voiceover_length=10,
        )

        # Should not go below 0.0
        assert conf3 >= 0.0, f"Max penalty should not go below 0.0, got {conf3}"


# ============================================================================
# Test Score Variance with Different Config Combinations
# ============================================================================

class TestScoreVarianceWithConfigCombinations:
    """Test score variance with different config combinations."""

    @pytest.mark.fast
    def test_variance_rich_vs_sparse_context(self):
        """Test score variance between rich and sparse context."""
        base_confidence = 0.70

        # Rich context
        rich_conf, _ = apply_context_richness_calibration(
            confidence=base_confidence,
            video_title="Title", video_description="Description",
            video_tags=["tag1", "tag2"], video_chapter="Chapter",
            enabled=True, boost_max=0.08, penalty_max=0.05,
        )

        # Sparse context
        sparse_conf, _ = apply_context_richness_calibration(
            confidence=base_confidence,
            video_title=None, video_description=None,
            video_tags=None, video_chapter=None,
            enabled=True, boost_max=0.08, penalty_max=0.05,
        )

        # Rich should be higher than sparse
        variance = rich_conf - sparse_conf
        assert variance > 0, "Rich context should produce higher confidence than sparse"
        # Should be approximately 0.13 (0.08 boost - 0.05 penalty)
        assert 0.10 <= variance <= 0.15, f"Variance should be ~0.13, got {variance}"

    @pytest.mark.fast
    def test_variance_high_vs_low_caption_quality(self, mock_matching_config, sample_vo_segment):
        """Test score variance between high and low caption quality."""
        scoring = MatchScoring(mock_matching_config)
        base_confidence = 0.70

        # High caption quality
        video_high = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="High quality", source_file="h.mp4")
        video_high.caption_quality = 'high'
        video_high.is_broll = False

        conf_high, _, _ = scoring.apply_all_adjustments(
            confidence=base_confidence,
            vo_segment=sample_vo_segment,
            video_segment=video_high,
        )

        # Low caption quality
        video_low = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Low quality", source_file="l.mp4")
        video_low.caption_quality = 'low'
        video_low.is_broll = False

        conf_low, _, _ = scoring.apply_all_adjustments(
            confidence=base_confidence,
            vo_segment=sample_vo_segment,
            video_segment=video_low,
        )

        # High should be higher than low
        variance = conf_high - conf_low
        assert variance > 0, "High quality should produce higher confidence than low"
        # Should be exactly 0.15 (0.05 boost + 0.10 penalty)
        assert 0.14 <= variance <= 0.16, f"Variance should be ~0.15, got {variance}"

    @pytest.mark.fast
    def test_variance_with_vs_without_voiceover_context(self):
        """Test score variance with and without voiceover context."""
        base_confidence = 0.70

        # Has both adjacent segments
        with_context, _ = apply_voiceover_context_calibration(
            confidence=base_confidence,
            has_prev_segment=True,
            has_next_segment=True,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=10,
        )

        # No adjacent segments
        without_context, _ = apply_voiceover_context_calibration(
            confidence=base_confidence,
            has_prev_segment=False,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=10,
        )

        # With context should be higher than without
        variance = with_context - without_context
        assert variance > 0, "Voiceover context should boost confidence"
        # Should be 0.08 (0.05 boost + 0.03 penalty)
        assert variance == pytest.approx(0.08, abs=0.001)

    @pytest.mark.fast
    def test_combined_config_variance(self, mock_matching_config, sample_vo_segment):
        """Test variance with different config combinations."""
        scoring = MatchScoring(mock_matching_config)
        base = 0.70

        # Configuration A: All positive factors (rich context, high caption, voiceover context)
        video_a = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="A", source_file="a.mp4")
        video_a.caption_quality = 'high'
        video_a.is_broll = False

        conf_a, _, _ = scoring.apply_all_adjustments(
            confidence=base,
            vo_segment=sample_vo_segment,
            video_segment=video_a,
            video_topics=["topic1"],
            has_prev_segment=True,
            has_next_segment=True,
            video_title="Title",
            video_description="Description",
            video_tags=["tag1"],
            chapter_title="Chapter",
        )

        # Configuration B: All negative factors (sparse context, low caption, no voiceover context)
        video_b = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="B", source_file="b.mp4")
        video_b.caption_quality = 'low'
        video_b.is_broll = False

        conf_b, _, _ = scoring.apply_all_adjustments(
            confidence=base,
            vo_segment=sample_vo_segment,
            video_segment=video_b,
            video_topics=None,
            has_prev_segment=False,
            has_next_segment=False,
            video_title=None,
            video_description=None,
            video_tags=None,
            chapter_title=None,
        )

        # Configuration C: Mixed factors
        video_c = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="C", source_file="c.mp4")
        video_c.caption_quality = 'low'  # Negative
        video_c.is_broll = False

        conf_c, _, _ = scoring.apply_all_adjustments(
            confidence=base,
            vo_segment=sample_vo_segment,
            video_segment=video_c,
            video_topics=None,
            has_prev_segment=True,  # One positive
            has_next_segment=False,
            video_title="Title",  # Rich context - positive
            video_description=None,
            video_tags=None,
            chapter_title=None,
        )

        # A should be highest, B lowest
        assert conf_a > conf_c > conf_b, f"All positive ({conf_a}) > mixed ({conf_c}) > all negative ({conf_b})"

        # Variance should be significant
        total_variance = conf_a - conf_b
        assert total_variance > 0.15, f"Total variance should be > 0.15, got {total_variance}"


# ============================================================================
# Test Missing Metadata Handling (Graceful)
# ============================================================================

class TestMissingMetadataGracefulHandling:
    """Ensure scoring functions handle missing metadata gracefully (no crashes)."""

    @pytest.mark.fast
    def test_context_richness_all_none(self):
        """Test context richness with all None values - should not crash."""
        confidence = 0.80

        # Should not raise any exception
        adjusted, reason = apply_context_richness_calibration(
            confidence=confidence,
            video_title=None,
            video_description=None,
            video_tags=None,
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        assert 0.0 <= adjusted <= 1.0
        assert "sparse" in reason.lower() or "none" in reason.lower()

    @pytest.mark.fast
    def test_context_richness_disabled(self):
        """Test context richness when disabled - should return unchanged."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence=confidence,
            video_title=None,
            video_description=None,
            video_tags=None,
            video_chapter=None,
            enabled=False,  # Disabled
            boost_max=0.08,
            penalty_max=0.05,
        )

        assert adjusted == confidence
        assert reason == ""

    @pytest.mark.fast
    def test_voiceover_context_all_false(self):
        """Test voiceover context with no adjacent segments - should not crash."""
        confidence = 0.80

        adjusted, reason = apply_voiceover_context_calibration(
            confidence=confidence,
            has_prev_segment=False,
            has_next_segment=False,
            enabled=True,
            boost_max=0.05,
            penalty_max=0.03,
            voiceover_length=0,
        )

        assert 0.0 <= adjusted <= 1.0
        assert "no adjacent segments" in reason.lower()

    @pytest.mark.fast
    def test_apply_all_adjustments_missing_metadata(self, mock_matching_config, sample_vo_segment):
        """Test apply_all_adjustments with missing metadata - should not crash."""
        scoring = MatchScoring(mock_matching_config)

        # Video segment with minimal metadata
        video = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Minimal video", source_file="v.mp4"
        )
        # Don't set any optional attributes

        # Should not raise any exception
        adjusted, reason, breakdown = scoring.apply_all_adjustments(
            confidence=0.80,
            vo_segment=sample_vo_segment,
            video_segment=video,
            video_topics=None,
            chapter_matching_enabled=False,
            topic_mismatch_penalty=0.15,
        )

        assert 0.0 <= adjusted <= 1.0


# ============================================================================
# Test MatchScoring Integration
# ============================================================================

class TestMatchScoringIntegration:
    """Integration tests for MatchScoring class with all factors."""

    @pytest.mark.fast
    def test_match_scoring_full_pipeline(self, mock_matching_config, sample_vo_segment):
        """Test MatchScoring with full pipeline of adjustments."""
        scoring = MatchScoring(mock_matching_config)

        # Create video with all factors
        video = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Tokyo night cityscape", source_file="tokyo.mp4"
        )
        video.is_broll = False
        video.timing_penalty = 0.95
        video.caption_quality = 'high'

        # Calculate base confidence
        base_conf, _, _ = scoring.calculate_confidence(
            embedding_similarity=0.85,
            keyword_score=0.7,
            entity_score=0.6,
            visual_score=0.0
        )

        # Apply all adjustments
        adjusted, reason, breakdown = scoring.apply_all_adjustments(
            confidence=base_conf,
            vo_segment=sample_vo_segment,
            video_segment=video,
            video_topics=["tokyo", "travel"],
            chapter_matching_enabled=True,
            topic_mismatch_penalty=0.15,
            has_prev_segment=True,
            has_next_segment=True,
            voiceover_length=10,
            video_title="Tokyo Travel Guide",
            video_description="Explore Tokyo",
            video_tags=["travel", "japan"],
            chapter_title="Introduction",
        )

        # Verify bounds
        assert 0.0 <= adjusted <= 1.0, f"Adjusted {adjusted} out of bounds"

        # Breakdown should contain info
        assert breakdown is not None

    @pytest.mark.fast
    def test_match_scoring_with_normalize_pool(self, mock_matching_config, sample_vo_segment):
        """Test MatchScoring with pool normalization included."""
        scoring = MatchScoring(mock_matching_config)

        video = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test video", source_file="test.mp4"
        )

        # Calculate with pool normalization
        base_conf, _, _ = scoring.calculate_confidence(
            embedding_similarity=0.8,
            keyword_score=0.6,
            entity_score=0.5,
            visual_score=0.0
        )

        adjusted, reason, breakdown = scoring.apply_all_adjustments(
            confidence=base_conf,
            vo_segment=sample_vo_segment,
            video_segment=video,
        )

        # Normalize with pool
        normalized, norm_reason = scoring.normalize_score(
            confidence=adjusted,
            pool_size=5,  # Small pool
            candidates=[(video, adjusted)]
        )

        assert 0.0 <= normalized <= 1.0

    @pytest.mark.fast
    def test_obvious_match_detection_with_confidence_floor(
        self, mock_matching_config, sample_vo_segment
    ):
        """Test obvious match detection respects confidence floor."""
        scoring = MatchScoring(mock_matching_config)

        video = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Perfect match video", source_file="perfect.mp4"
        )
        video.is_broll = False

        # Very high similarity - use values that will result in high final confidence
        base_conf, _, _ = scoring.calculate_confidence(
            embedding_similarity=0.98,
            keyword_score=0.95,
            entity_score=0.95,
            visual_score=0.0
        )

        # Provide adjacent segments to avoid voiceover context penalty
        # Also add caption quality high to boost
        video.caption_quality = 'high'

        adjusted, reason, breakdown = scoring.apply_all_adjustments(
            confidence=base_conf,
            vo_segment=sample_vo_segment,
            video_segment=video,
            has_prev_segment=True,
            has_next_segment=True,
            voiceover_length=10,
        )

        # Should be high and within bounds - multimodal + caption quality boost
        assert adjusted >= 0.80, f"High similarity should result in high confidence, got {adjusted}"
        assert adjusted <= 1.0, "Should not exceed 1.0"
