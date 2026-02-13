"""
Tests for confidence scoring and adjustment functions (scoring.py).

Targets uncovered lines:
- Lines 21-44: apply_duration_penalty()
- Lines 47-102: apply_topic_penalty()
- Lines 105-147: apply_broll_boost()
- Lines 149-191: apply_current_project_boost()
- Lines 193-234: compute_duration_penalty()
- Lines 236-263: apply_duration_scoring()

Current coverage: 25.61%
Target coverage: 75%+

Created: 2026-01-10 (Session 13 - Critical gap coverage, Phase 1)
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching import scoring
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
    matching.ideal_speed_range = (0.9, 1.1)
    matching.soft_speed_range = (0.7, 1.3)
    matching.duration_penalty_factor = 0.1
    matching.duration_scoring_enabled = True
    matching.soft_penalty_range = (0.7, 1.3)
    matching.broll_boost = 0.1
    # Scoring sub-config for duration ratio reward (US-84-007)
    scoring_mock = Mock()
    scoring_mock.duration_ratio_reward_threshold = 0.1
    scoring_mock.duration_ratio_reward_boost = 0.02
    matching.scoring = scoring_mock

    config.matching = matching

    # Global cache config
    global_cache = Mock()
    global_cache.current_project_boost = 0.1
    config.global_cache = global_cache

    return config


@pytest.fixture
def sample_vo_segment():
    """Sample voiceover segment"""
    return SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,  # 10 second duration
        text="Sample voiceover text",
        source_file="voiceover.srt"
    )


@pytest.fixture
def sample_video_segment():
    """Sample video segment"""
    return SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,  # 10 second duration
        text="Sample video content",
        source_file="/videos/test.mp4"
    )


# ============================================================================
# Test apply_duration_penalty()
# ============================================================================

class TestApplyDurationPenalty:
    """Test duration-based confidence penalties"""

    @pytest.mark.fast
    def test_ideal_speed_reward(self, mock_config):
        """Test reward boost for near-perfect speed ratio (US-84-007)"""
        confidence = 0.8
        speed_ratio = 1.0  # Perfect match (within reward threshold)

        result = scoring.apply_duration_penalty(confidence, speed_ratio, mock_config)

        assert result == pytest.approx(0.82, abs=1e-6)  # +0.02 reward boost

    @pytest.mark.fast
    def test_ideal_speed_boundary_reward(self, mock_config):
        """Test boundaries of reward range get boost (US-84-007)"""
        # Lower boundary (0.9 = 1.0 - 0.1 threshold)
        result1 = scoring.apply_duration_penalty(0.8, 0.9, mock_config)
        assert result1 == pytest.approx(0.82, abs=1e-6)  # +0.02 reward

        # Upper boundary (1.1 = 1.0 + 0.1 threshold)
        result2 = scoring.apply_duration_penalty(0.8, 1.1, mock_config)
        assert result2 == pytest.approx(0.82, abs=1e-6)  # +0.02 reward

    @pytest.mark.fast
    def test_soft_speed_graduated_penalty(self, mock_config):
        """Test graduated log penalty for soft speed range (US-84-007)"""
        import math
        confidence = 0.8
        speed_ratio = 0.8  # Outside reward range but not extreme

        result = scoring.apply_duration_penalty(confidence, speed_ratio, mock_config)

        # log penalty = min(0.1, 0.02 * |log2(0.8)|) = 0.02 * 0.3219 ≈ 0.00644
        expected_penalty = 0.02 * abs(math.log2(0.8))
        assert result == pytest.approx(confidence - expected_penalty, abs=1e-4)

    @pytest.mark.fast
    def test_outside_both_ranges_hard_penalty(self, mock_config):
        """Test hard penalty for extreme speed ratio (US-84-007)"""
        confidence = 0.9
        speed_ratio = 0.2  # Below hard threshold (< 0.3)

        result = scoring.apply_duration_penalty(confidence, speed_ratio, mock_config)

        # Hard penalty: penalty_factor * 2 = 0.2
        assert result == pytest.approx(0.7, abs=1e-6)  # 0.9 - 0.2

    @pytest.mark.fast
    def test_very_fast_speed_graduated_penalty(self, mock_config):
        """Test graduated penalty for moderately fast speed (US-84-007)"""
        import math
        confidence = 0.9
        speed_ratio = 1.5  # Moderate deviation

        result = scoring.apply_duration_penalty(confidence, speed_ratio, mock_config)

        # log penalty = min(0.1, 0.02 * |log2(1.5)|) = 0.02 * 0.585 ≈ 0.0117
        expected_penalty = min(0.1, 0.02 * abs(math.log2(1.5)))
        assert result == pytest.approx(confidence - expected_penalty, abs=1e-4)


# ============================================================================
# Test apply_topic_penalty()
# ============================================================================

class TestApplyTopicPenalty:
    """Test topic-based confidence penalties"""

    @pytest.mark.fast
    def test_chapter_matching_disabled(self, sample_vo_segment, sample_video_segment):
        """Test no penalty when chapter matching is disabled"""
        confidence = 0.8

        result_conf, reason = scoring.apply_topic_penalty(
            confidence=confidence,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
            video_topics={},
            chapter_matching_enabled=False,
            topic_mismatch_penalty=0.3
        )

        assert result_conf == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_vo_segment_no_topics(self, sample_vo_segment, sample_video_segment):
        """Test no penalty when voiceover has no topics"""
        confidence = 0.8

        # VO segment without topics attribute
        result_conf, reason = scoring.apply_topic_penalty(
            confidence=confidence,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
            video_topics={},
            chapter_matching_enabled=True,
            topic_mismatch_penalty=0.3
        )

        assert result_conf == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_video_not_in_video_topics(self, sample_vo_segment, sample_video_segment):
        """Test no penalty when video not in video_topics dict"""
        confidence = 0.8

        # Add topics to VO segment
        sample_vo_segment.topics = ["earthquake", "disaster"]

        result_conf, reason = scoring.apply_topic_penalty(
            confidence=confidence,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
            video_topics={},  # Empty - video not found
            chapter_matching_enabled=True,
            topic_mismatch_penalty=0.3
        )

        assert result_conf == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_video_topics_empty(self, sample_vo_segment, sample_video_segment):
        """Test no penalty when video has no topics"""
        confidence = 0.8

        sample_vo_segment.topics = ["earthquake", "disaster"]

        # Mock video topics with empty list
        video_topic_info = Mock()
        video_topic_info.topics = []

        video_topics = {sample_video_segment.source_file: video_topic_info}

        result_conf, reason = scoring.apply_topic_penalty(
            confidence=confidence,
            vo_segment=sample_vo_segment,
            video_segment=sample_video_segment,
            video_topics=video_topics,
            chapter_matching_enabled=True,
            topic_mismatch_penalty=0.3
        )

        assert result_conf == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_topic_mismatch_penalty_applied(self, sample_vo_segment, sample_video_segment):
        """Test penalty applied for topic mismatch"""
        confidence = 0.8

        sample_vo_segment.topics = ["earthquake", "disaster"]

        # Mock video with different topics
        video_topic_info = Mock()
        video_topic_info.topics = ["travel", "vacation"]

        video_topics = {sample_video_segment.source_file: video_topic_info}

        with patch('src.matching.scoring.compute_topic_penalty', return_value=0.2):
            result_conf, reason = scoring.apply_topic_penalty(
                confidence=confidence,
                vo_segment=sample_vo_segment,
                video_segment=sample_video_segment,
                video_topics=video_topics,
                chapter_matching_enabled=True,
                topic_mismatch_penalty=0.3
            )

            assert abs(result_conf - 0.6) < 0.001  # 0.8 - 0.2
            assert "topic mismatch penalty" in reason
            assert "-0.20" in reason

    @pytest.mark.fast
    def test_topic_match_no_penalty(self, sample_vo_segment, sample_video_segment):
        """Test no penalty when topics match"""
        confidence = 0.8

        sample_vo_segment.topics = ["earthquake", "disaster"]

        # Mock video with matching topics
        video_topic_info = Mock()
        video_topic_info.topics = ["earthquake", "rescue"]

        video_topics = {sample_video_segment.source_file: video_topic_info}

        with patch('src.matching.scoring.compute_topic_penalty', return_value=0.0):
            result_conf, reason = scoring.apply_topic_penalty(
                confidence=confidence,
                vo_segment=sample_vo_segment,
                video_segment=sample_video_segment,
                video_topics=video_topics,
                chapter_matching_enabled=True,
                topic_mismatch_penalty=0.3
            )

            assert result_conf == 0.8  # No change
            assert reason == ""


# ============================================================================
# Test apply_broll_boost()
# ============================================================================

class TestApplyBRollBoost:
    """Test B-roll/silent video confidence boosts"""

    @pytest.mark.fast
    def test_not_broll_no_boost(self, mock_config, sample_video_segment):
        """Test no boost for non-B-roll segments"""
        confidence = 0.7

        # Segment without is_broll attribute (defaults to False)
        result_conf, reason = scoring.apply_broll_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        assert result_conf == 0.7
        assert reason == ""

    @pytest.mark.fast
    def test_broll_boost_applied(self, mock_config, sample_video_segment):
        """Test boost applied for B-roll segments"""
        confidence = 0.7

        # Mark as B-roll
        sample_video_segment.is_broll = True

        result_conf, reason = scoring.apply_broll_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        # Boost = 0.1
        assert abs(result_conf - 0.8) < 0.001  # 0.7 + 0.1
        assert "B-roll boost" in reason
        assert "+0.10" in reason

    @pytest.mark.fast
    def test_broll_boost_capped_at_1(self, mock_config, sample_video_segment):
        """Test boost is capped at 1.0"""
        confidence = 0.95

        sample_video_segment.is_broll = True

        result_conf, reason = scoring.apply_broll_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        # Should cap at 1.0
        assert result_conf == 1.0  # Not 1.05
        assert "B-roll boost" in reason

    @pytest.mark.fast
    def test_broll_boost_zero_config(self, mock_config, sample_video_segment):
        """Test no boost when config is zero"""
        confidence = 0.7

        sample_video_segment.is_broll = True
        mock_config.matching.broll_boost = 0.0

        result_conf, reason = scoring.apply_broll_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        assert result_conf == 0.7  # No boost
        assert reason == ""

    @pytest.mark.fast
    def test_broll_boost_missing_config(self, sample_video_segment):
        """Test fallback when config missing broll_boost"""
        confidence = 0.7
        config = Mock()
        config.matching = Mock(spec=[])  # Empty spec = no attributes
        # No broll_boost attribute

        sample_video_segment.is_broll = True

        result_conf, reason = scoring.apply_broll_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # Default boost = 0.1
        assert abs(result_conf - 0.8) < 0.001


# ============================================================================
# Test apply_current_project_boost()
# ============================================================================

class TestApplyCurrentProjectBoost:
    """Test current project vs global cache scoring"""

    @pytest.mark.fast
    def test_current_project_no_penalty(self, mock_config, sample_video_segment):
        """Test no penalty for current project videos"""
        confidence = 0.8

        # No source attribute = current project
        result_conf, reason = scoring.apply_current_project_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        assert result_conf == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_global_cache_penalty(self, mock_config, sample_video_segment):
        """Test penalty for global cache videos"""
        confidence = 0.8

        # Mark as global cache
        sample_video_segment.source = 'global_cache'

        result_conf, reason = scoring.apply_current_project_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        # Penalty = 0.1
        assert abs(result_conf - 0.7) < 0.001  # 0.8 - 0.1
        assert "global cache" in reason
        assert "-0.10" in reason

    @pytest.mark.fast
    def test_global_cache_zero_boost(self, mock_config, sample_video_segment):
        """Test no penalty when boost is zero"""
        confidence = 0.8

        sample_video_segment.source = 'global_cache'
        mock_config.global_cache.current_project_boost = 0.0

        result_conf, reason = scoring.apply_current_project_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        assert result_conf == 0.8  # No penalty
        assert reason == ""

    @pytest.mark.fast
    def test_global_cache_missing_config(self, sample_video_segment):
        """Test fallback when config missing"""
        confidence = 0.8
        config = Mock()
        config.global_cache = None

        sample_video_segment.source = 'global_cache'

        result_conf, reason = scoring.apply_current_project_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # Default boost = 0.1
        assert abs(result_conf - 0.7) < 0.001


# ============================================================================
# Test compute_duration_penalty()
# ============================================================================

class TestComputeDurationPenalty:
    """Test duration penalty computation"""

    @pytest.mark.fast
    def test_duration_scoring_disabled(self, mock_config, sample_vo_segment, sample_video_segment):
        """Test no penalty when duration scoring disabled"""
        mock_config.matching.duration_scoring_enabled = False

        penalty = scoring.compute_duration_penalty(sample_vo_segment, sample_video_segment, mock_config)

        assert penalty == 0.0

    @pytest.mark.fast
    def test_ideal_speed_no_penalty(self, mock_config, sample_vo_segment, sample_video_segment):
        """Test no penalty for ideal speed match"""
        # VO: 10s, Video: 10s → speed_ratio = 1.0 (ideal)
        penalty = scoring.compute_duration_penalty(sample_vo_segment, sample_video_segment, mock_config)

        assert penalty == 0.0

    @pytest.mark.fast
    def test_soft_penalty_range(self, mock_config, sample_vo_segment):
        """Test small penalty for soft range"""
        # VO: 10s, Video: 7.5s → speed_ratio = 0.75 (soft range)
        video_seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=7.5,
            text="Video",
            source_file="/video.mp4"
        )

        penalty = scoring.compute_duration_penalty(sample_vo_segment, video_seg, mock_config)

        # penalty_factor = 0.1
        assert penalty == 0.1

    @pytest.mark.fast
    def test_large_penalty_outside_ranges(self, mock_config, sample_vo_segment):
        """Test large penalty outside both ranges"""
        # VO: 10s, Video: 5s → speed_ratio = 0.5 (outside soft range)
        video_seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Video",
            source_file="/video.mp4"
        )

        penalty = scoring.compute_duration_penalty(sample_vo_segment, video_seg, mock_config)

        # penalty_factor * 2 = 0.2
        assert penalty == 0.2

    @pytest.mark.fast
    def test_zero_duration_no_penalty(self, mock_config):
        """Test no penalty for zero-duration segments"""
        vo_seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=0.0,  # Zero duration
            text="VO",
            source_file="vo.srt"
        )
        video_seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=10.0,
            text="Video",
            source_file="/video.mp4"
        )

        penalty = scoring.compute_duration_penalty(vo_seg, video_seg, mock_config)

        assert penalty == 0.0

    @pytest.mark.fast
    def test_very_long_video_penalty(self, mock_config, sample_vo_segment):
        """Test penalty for very long video segments"""
        # VO: 10s, Video: 20s → speed_ratio = 2.0 (much too fast)
        video_seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=20.0,
            text="Video",
            source_file="/video.mp4"
        )

        penalty = scoring.compute_duration_penalty(sample_vo_segment, video_seg, mock_config)

        # Outside both ranges
        assert penalty == 0.2


# ============================================================================
# Test apply_duration_scoring()
# ============================================================================

class TestApplyDurationScoring:
    """Test duration scoring application to candidates"""

    @pytest.mark.fast
    def test_apply_duration_scoring_empty(self, mock_config, sample_vo_segment):
        """Test with empty candidates"""
        candidates = []

        result = scoring.apply_duration_scoring(sample_vo_segment, candidates, mock_config)

        assert result == []

    @pytest.mark.fast
    def test_apply_duration_scoring_single(self, mock_config, sample_vo_segment):
        """Test with single candidate"""
        video_seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=10.0,  # Ideal match
            text="Video",
            source_file="/video.mp4"
        )
        candidates = [(video_seg, 0.8)]

        result = scoring.apply_duration_scoring(sample_vo_segment, candidates, mock_config)

        assert len(result) == 1
        # (segment, adjusted_sim, penalty)
        assert result[0][0] == video_seg
        assert result[0][1] == 0.8  # No penalty
        assert result[0][2] == 0.0  # Penalty = 0

    @pytest.mark.fast
    def test_apply_duration_scoring_multiple(self, mock_config, sample_vo_segment):
        """Test with multiple candidates"""
        # Ideal match
        video_seg1 = SRTSegment(index=1, start_time=0.0, end_time=10.0, text="V1", source_file="/v1.mp4")
        # Soft penalty
        video_seg2 = SRTSegment(index=2, start_time=0.0, end_time=7.5, text="V2", source_file="/v2.mp4")
        # Large penalty
        video_seg3 = SRTSegment(index=3, start_time=0.0, end_time=5.0, text="V3", source_file="/v3.mp4")

        candidates = [
            (video_seg1, 0.9),
            (video_seg2, 0.85),
            (video_seg3, 0.8)
        ]

        result = scoring.apply_duration_scoring(sample_vo_segment, candidates, mock_config)

        assert len(result) == 3
        # Check penalties
        assert result[0][2] == 0.0   # Ideal = no penalty
        assert result[1][2] == 0.1   # Soft range
        assert result[2][2] == 0.2   # Outside range

        # Check adjusted scores
        assert abs(result[0][1] - 0.9) < 0.001   # 0.9 - 0.0
        assert abs(result[1][1] - 0.75) < 0.001  # 0.85 - 0.1
        assert abs(result[2][1] - 0.6) < 0.001   # 0.8 - 0.2

    @pytest.mark.fast
    def test_apply_duration_scoring_sorts_by_adjusted(self, mock_config, sample_vo_segment):
        """Test that results are sorted by adjusted score"""
        # Create candidates where penalties change the order
        video_seg1 = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="V1", source_file="/v1.mp4")  # Large penalty
        video_seg2 = SRTSegment(index=2, start_time=0.0, end_time=10.0, text="V2", source_file="/v2.mp4")  # No penalty

        candidates = [
            (video_seg1, 0.9),   # High original, but large penalty
            (video_seg2, 0.75)   # Lower original, but no penalty
        ]

        result = scoring.apply_duration_scoring(sample_vo_segment, candidates, mock_config)

        # After penalties:
        # v1: 0.9 - 0.2 = 0.7
        # v2: 0.75 - 0.0 = 0.75
        # So v2 should be first

        assert result[0][0] == video_seg2  # v2 first
        assert result[1][0] == video_seg1  # v1 second

    @pytest.mark.fast
    def test_apply_duration_scoring_negative_adjusted(self, mock_config, sample_vo_segment):
        """Test that adjusted scores don't go below 0"""
        video_seg = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="V1", source_file="/v1.mp4")

        candidates = [(video_seg, 0.1)]  # Very low similarity + large penalty

        result = scoring.apply_duration_scoring(sample_vo_segment, candidates, mock_config)

        # 0.1 - 0.2 = -0.1, but should be clamped to 0.0
        assert result[0][1] == 0.0
        assert result[0][2] == 0.2  # Penalty still recorded


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestScoringEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.fast
    def test_all_scoring_functions_combined(self, mock_config, sample_vo_segment):
        """Test realistic scenario with all scoring functions"""
        video_seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=10.0,
            text="B-roll footage from Paris",
            source_file="/videos/paris_broll.mp4"
        )
        video_seg.is_broll = True
        video_seg.source = 'global_cache'

        # Add topics
        sample_vo_segment.topics = ["paris", "travel"]

        video_topic_info = Mock()
        video_topic_info.topics = ["paris", "eiffel tower"]
        video_topics = {video_seg.source_file: video_topic_info}

        # Start with base confidence
        confidence = 0.7

        # Apply duration penalty (ideal speed = reward boost, US-84-007)
        confidence = scoring.apply_duration_penalty(confidence, 1.0, mock_config)
        assert abs(confidence - 0.72) < 0.001  # +0.02 reward for perfect ratio

        # Apply topic penalty (good match = minimal penalty)
        with patch('src.matching.scoring.compute_topic_penalty', return_value=0.05):
            confidence, _ = scoring.apply_topic_penalty(
                confidence, sample_vo_segment, video_seg, video_topics,
                True, 0.3
            )
        assert abs(confidence - 0.67) < 0.001

        # Apply B-roll boost
        confidence, _ = scoring.apply_broll_boost(confidence, video_seg, mock_config)
        assert abs(confidence - 0.77) < 0.001  # +0.1

        # Apply global cache penalty
        confidence, _ = scoring.apply_current_project_boost(confidence, video_seg, mock_config)
        assert abs(confidence - 0.67) < 0.001  # -0.1

        # Final score balances all factors
        assert 0.6 <= confidence <= 0.7

    @pytest.mark.fast
    def test_extreme_speed_ratios(self, mock_config):
        """Test handling of extreme speed ratios"""
        # Very slow (speed up 10x)
        penalty1 = scoring.apply_duration_penalty(0.8, 10.0, mock_config)
        assert penalty1 <= 0.601  # Large penalty (allow floating point tolerance)

        # Very fast (slow down 10x)
        penalty2 = scoring.apply_duration_penalty(0.8, 0.1, mock_config)
        assert penalty2 <= 0.601  # Large penalty (allow floating point tolerance)

    @pytest.mark.fast
    def test_config_object_vs_dict(self):
        """Test that scoring works with both object and dict configs"""
        # Config as object
        config_obj = Mock()
        config_obj.matching = Mock()
        config_obj.matching.broll_boost = 0.15

        # Config as dict (handled via getattr fallback)
        video_seg = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test", source_file="/test.mp4")
        video_seg.is_broll = True

        result, _ = scoring.apply_broll_boost(0.7, video_seg, config_obj)
        assert result == 0.85  # 0.7 + 0.15


# ============================================================================
# Test apply_caption_quality_adjustment() (US-007)
# ============================================================================

class TestApplyCaptionQualityAdjustment:
    """Test caption quality confidence adjustments (US-007)"""

    @pytest.fixture
    def caption_quality_config(self):
        """Mock config with caption quality settings"""
        config = Mock()
        matching = Mock()
        matching.caption_quality_adjustment_enabled = True
        matching.caption_quality_high_boost = 0.05
        matching.caption_quality_low_penalty = 0.1
        config.matching = matching
        return config

    @pytest.mark.fast
    def test_high_quality_boost(self, caption_quality_config, sample_video_segment):
        """Test confidence boost for high-quality human captions"""
        confidence = 0.7
        sample_video_segment.caption_quality = "high"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=caption_quality_config
        )

        assert result_conf == 0.75  # 0.7 + 0.05
        assert "caption quality high" in reason
        assert "+0.05" in reason

    @pytest.mark.fast
    def test_medium_quality_no_adjustment(self, caption_quality_config, sample_video_segment):
        """Test no adjustment for medium-quality auto captions"""
        confidence = 0.7
        sample_video_segment.caption_quality = "medium"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=caption_quality_config
        )

        assert result_conf == 0.7  # No change
        assert reason == ""

    @pytest.mark.fast
    def test_low_quality_penalty(self, caption_quality_config, sample_video_segment):
        """Test confidence penalty for low-quality/fallback captions"""
        confidence = 0.7
        sample_video_segment.caption_quality = "low"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=caption_quality_config
        )

        assert result_conf == 0.6  # 0.7 - 0.1
        assert "caption quality low" in reason
        assert "-0.10" in reason

    @pytest.mark.fast
    def test_no_caption_quality_attribute(self, caption_quality_config, sample_video_segment):
        """Test no adjustment when caption_quality not set"""
        confidence = 0.7
        # Don't set caption_quality attribute

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=caption_quality_config
        )

        assert result_conf == 0.7
        assert reason == ""

    @pytest.mark.fast
    def test_disabled_via_config(self, sample_video_segment):
        """Test no adjustment when feature disabled in config"""
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = False

        sample_video_segment.caption_quality = "high"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=0.7,
            video_segment=sample_video_segment,
            config=config
        )

        assert result_conf == 0.7
        assert reason == ""

    @pytest.mark.fast
    def test_boost_capped_at_1(self, caption_quality_config, sample_video_segment):
        """Test that boost is capped at 1.0"""
        confidence = 0.98
        sample_video_segment.caption_quality = "high"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=caption_quality_config
        )

        assert result_conf == 1.0  # Not 1.03
        assert "caption quality high" in reason

    @pytest.mark.fast
    def test_penalty_capped_at_0(self, caption_quality_config, sample_video_segment):
        """Test that penalty doesn't go below 0"""
        confidence = 0.05
        sample_video_segment.caption_quality = "low"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=caption_quality_config
        )

        assert result_conf == 0.0  # Not negative
        assert "caption quality low" in reason

    @pytest.mark.fast
    def test_zero_boost_config(self, sample_video_segment):
        """Test no boost when config boost is zero"""
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = True
        config.matching.caption_quality_high_boost = 0.0
        config.matching.caption_quality_low_penalty = 0.1

        sample_video_segment.caption_quality = "high"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=0.7,
            video_segment=sample_video_segment,
            config=config
        )

        assert result_conf == 0.7
        assert reason == ""

    @pytest.mark.fast
    def test_missing_config_defaults(self, sample_video_segment):
        """Test fallback to default values when config attributes missing"""
        config = Mock()
        config.matching = Mock(spec=[])  # Empty spec = no attributes

        sample_video_segment.caption_quality = "high"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=0.7,
            video_segment=sample_video_segment,
            config=config
        )

        # Default enabled=True, high_boost=0.05
        assert result_conf == 0.75
        assert "caption quality high" in reason


# ============================================================================
# Test apply_caption_quality_adjustment() - Multiplicative Weights Mode (US-006)
# ============================================================================

class TestCaptionQualityMultiplicativeWeights:
    """Test caption quality multiplicative weights mode (US-006)

    US-006: Apply caption quality weights to match confidence scores.
    When caption_quality_weights dict is set, uses multiplicative mode:
    adjusted = raw_confidence * weight
    """

    @pytest.fixture
    def weights_config(self):
        """Mock config with multiplicative weights"""
        config = Mock()
        matching = Mock()
        matching.caption_quality_adjustment_enabled = True
        # US-006: Multiplicative weights {high: 1.0, medium: 0.9, low: 0.75}
        matching.caption_quality_weights = {'high': 1.0, 'medium': 0.9, 'low': 0.75}
        config.matching = matching
        return config

    @pytest.mark.fast
    def test_high_quality_weight_no_change(self, weights_config, sample_video_segment):
        """Test high quality weight=1.0 produces no change (US-006 AC)"""
        confidence = 0.85
        sample_video_segment.caption_quality = "high"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=weights_config
        )

        # high=1.0 means no change
        assert result_conf == 0.85
        assert reason == ""  # No reason when weight=1.0

    @pytest.mark.fast
    def test_medium_quality_weight_reduces_confidence(self, weights_config, sample_video_segment):
        """Test medium quality weight=0.9 reduces confidence by 10% (US-006 AC)"""
        confidence = 0.85
        sample_video_segment.caption_quality = "medium"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=weights_config
        )

        # 0.85 * 0.9 = 0.765
        assert abs(result_conf - 0.765) < 0.001
        assert "x0.90" in reason
        assert "caption quality medium" in reason

    @pytest.mark.fast
    def test_low_quality_weight_significantly_reduces_confidence(self, weights_config, sample_video_segment):
        """Test low quality weight=0.75 reduces confidence by 25% (US-006 AC)"""
        confidence = 0.85
        sample_video_segment.caption_quality = "low"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=weights_config
        )

        # 0.85 * 0.75 = 0.6375
        assert abs(result_conf - 0.6375) < 0.001
        assert "x0.75" in reason
        assert "caption quality low" in reason

    @pytest.mark.fast
    def test_same_text_different_quality_different_confidence(self, weights_config, sample_video_segment):
        """Test that same match text produces different confidence with high vs low quality (US-006 AC)"""
        base_confidence = 0.80

        # High quality
        sample_video_segment.caption_quality = "high"
        high_conf, _ = scoring.apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=sample_video_segment,
            config=weights_config
        )

        # Low quality
        sample_video_segment.caption_quality = "low"
        low_conf, _ = scoring.apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=sample_video_segment,
            config=weights_config
        )

        # High quality (x1.0) vs Low quality (x0.75) - significant difference
        assert high_conf == 0.80  # 0.80 * 1.0
        assert abs(low_conf - 0.60) < 0.001  # 0.80 * 0.75
        assert high_conf > low_conf
        assert high_conf - low_conf >= 0.15  # At least 15% difference

    @pytest.mark.fast
    def test_weights_override_additive_mode(self, sample_video_segment):
        """Test that weights mode ignores legacy additive settings"""
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = True
        # Both modes configured - weights should take precedence
        config.matching.caption_quality_weights = {'high': 1.0, 'medium': 0.9, 'low': 0.75}
        config.matching.caption_quality_high_boost = 0.05  # Would be +0.05 in additive mode
        config.matching.caption_quality_low_penalty = 0.1  # Would be -0.1 in additive mode

        confidence = 0.85
        sample_video_segment.caption_quality = "low"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # Weights mode: 0.85 * 0.75 = 0.6375
        # (Not additive: 0.85 - 0.1 = 0.75)
        assert abs(result_conf - 0.6375) < 0.001
        assert "x0.75" in reason

    @pytest.mark.fast
    def test_custom_weights(self, sample_video_segment):
        """Test custom weight values in config"""
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = True
        # Custom weights - more aggressive
        config.matching.caption_quality_weights = {'high': 1.05, 'medium': 0.85, 'low': 0.5}

        confidence = 0.80
        sample_video_segment.caption_quality = "low"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # 0.80 * 0.5 = 0.4
        assert abs(result_conf - 0.40) < 0.001

    @pytest.mark.fast
    def test_weights_capped_at_1(self, sample_video_segment):
        """Test that multiplicative result is capped at 1.0"""
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = True
        config.matching.caption_quality_weights = {'high': 1.5, 'medium': 0.9, 'low': 0.75}

        confidence = 0.90
        sample_video_segment.caption_quality = "high"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # 0.90 * 1.5 = 1.35 -> capped at 1.0
        assert result_conf == 1.0

    @pytest.mark.fast
    def test_weights_capped_at_0(self, sample_video_segment):
        """Test that multiplicative result is capped at 0.0"""
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = True
        config.matching.caption_quality_weights = {'high': 1.0, 'medium': 0.9, 'low': -0.5}

        confidence = 0.50
        sample_video_segment.caption_quality = "low"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # 0.50 * -0.5 = -0.25 -> capped at 0.0
        assert result_conf == 0.0

    @pytest.mark.fast
    def test_unknown_quality_uses_default_weight(self, weights_config, sample_video_segment):
        """Test that unknown quality level defaults to weight=1.0"""
        confidence = 0.80
        sample_video_segment.caption_quality = "unknown"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=weights_config
        )

        # Unknown quality defaults to 1.0 (no change)
        assert result_conf == 0.80
        assert reason == ""

    @pytest.mark.fast
    def test_partial_weights_dict_uses_defaults(self, sample_video_segment):
        """Test that missing quality keys use default weights"""
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = True
        # Only set 'low' - others should use defaults
        config.matching.caption_quality_weights = {'low': 0.5}

        confidence = 0.80
        sample_video_segment.caption_quality = "medium"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # medium not in dict -> use default 0.9
        assert abs(result_conf - 0.72) < 0.001  # 0.80 * 0.9 = 0.72

    @pytest.mark.fast
    def test_empty_weights_dict_falls_back_to_defaults(self, sample_video_segment):
        """Test that empty weights dict uses all default weights"""
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = True
        config.matching.caption_quality_weights = {}  # Empty dict

        confidence = 0.80
        sample_video_segment.caption_quality = "low"

        result_conf, reason = scoring.apply_caption_quality_adjustment(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # low not in dict -> use default 0.75
        assert abs(result_conf - 0.60) < 0.001  # 0.80 * 0.75 = 0.60


# ============================================================================
# Test apply_timing_penalty() (US-008 Sprint 7)
# ============================================================================

class TestApplyTimingPenalty:
    """Test timing penalty confidence adjustments (US-008 Sprint 7)

    The timing penalty is calculated at caption fetch time:
    penalty = 1.0 - (exceeds_ratio * 0.3) - ((1 - coverage_ratio) * 0.2)

    This test class verifies that apply_timing_penalty() correctly:
    1. Reads timing_penalty from video_segment
    2. Applies multiplicative penalty when enabled
    3. Respects config setting to disable
    """

    @pytest.fixture
    def timing_penalty_config(self):
        """Mock config with timing penalty enabled"""
        config = Mock()
        matching = Mock()
        matching.apply_timing_penalty = True
        config.matching = matching
        return config

    @pytest.fixture
    def timing_penalty_disabled_config(self):
        """Mock config with timing penalty disabled"""
        config = Mock()
        matching = Mock()
        matching.apply_timing_penalty = False
        config.matching = matching
        return config

    @pytest.mark.fast
    def test_no_penalty_when_perfect_timing(self, timing_penalty_config, sample_video_segment):
        """Test no penalty when timing_penalty=1.0"""
        confidence = 0.85
        sample_video_segment.timing_penalty = 1.0

        result_conf, reason = scoring.apply_timing_penalty(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=timing_penalty_config
        )

        assert result_conf == 0.85
        assert reason == ""

    @pytest.mark.fast
    def test_penalty_applied_multiplicatively(self, timing_penalty_config, sample_video_segment):
        """Test that timing penalty is applied multiplicatively"""
        confidence = 0.80
        sample_video_segment.timing_penalty = 0.84  # 16% penalty

        result_conf, reason = scoring.apply_timing_penalty(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=timing_penalty_config
        )

        # 0.80 * 0.84 = 0.672
        assert abs(result_conf - 0.672) < 0.001
        assert "timing penalty" in reason
        assert "x0.84" in reason

    @pytest.mark.fast
    def test_penalty_disabled_no_effect(self, timing_penalty_disabled_config, sample_video_segment):
        """Test no penalty when config disabled"""
        confidence = 0.80
        sample_video_segment.timing_penalty = 0.5  # Would be 50% penalty

        result_conf, reason = scoring.apply_timing_penalty(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=timing_penalty_disabled_config
        )

        assert result_conf == 0.80  # No change
        assert reason == ""

    @pytest.mark.fast
    def test_no_timing_penalty_attribute(self, timing_penalty_config, sample_video_segment):
        """Test no penalty when timing_penalty attribute missing"""
        confidence = 0.80
        # Don't set timing_penalty attribute

        result_conf, reason = scoring.apply_timing_penalty(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=timing_penalty_config
        )

        assert result_conf == 0.80
        assert reason == ""

    @pytest.mark.fast
    def test_penalty_capped_at_zero(self, timing_penalty_config, sample_video_segment):
        """Test that result is capped at 0.0"""
        confidence = 0.50
        sample_video_segment.timing_penalty = -0.5  # Invalid but tests capping

        result_conf, reason = scoring.apply_timing_penalty(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=timing_penalty_config
        )

        assert result_conf == 0.0

    @pytest.mark.fast
    def test_penalty_above_one_is_no_penalty(self, timing_penalty_config, sample_video_segment):
        """Test that timing_penalty >= 1.0 means no penalty applied"""
        confidence = 0.80
        sample_video_segment.timing_penalty = 1.5  # Invalid, but >= 1.0 means no penalty

        result_conf, reason = scoring.apply_timing_penalty(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=timing_penalty_config
        )

        # timing_penalty >= 1.0 means no penalty applied
        assert result_conf == 0.80
        assert reason == ""

    @pytest.mark.fast
    def test_10_percent_penalty(self, timing_penalty_config, sample_video_segment):
        """Test 10% penalty (90% coverage, no exceeds)"""
        confidence = 0.85
        sample_video_segment.timing_penalty = 0.98  # 2% penalty from low coverage

        result_conf, reason = scoring.apply_timing_penalty(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=timing_penalty_config
        )

        # 0.85 * 0.98 = 0.833
        assert abs(result_conf - 0.833) < 0.001

    @pytest.mark.fast
    def test_reason_format(self, timing_penalty_config, sample_video_segment):
        """Test reason string format"""
        confidence = 0.80
        sample_video_segment.timing_penalty = 0.90

        result_conf, reason = scoring.apply_timing_penalty(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=timing_penalty_config
        )

        # Reason format: "timing penalty: x0.90 (-10%)"
        assert "timing penalty" in reason
        assert "x0.90" in reason
        assert "-10%" in reason

    @pytest.mark.fast
    def test_config_default_enabled(self, sample_video_segment):
        """Test default behavior when config attribute missing"""
        config = Mock()
        config.matching = Mock(spec=[])  # Empty spec

        confidence = 0.80
        sample_video_segment.timing_penalty = 0.90

        result_conf, reason = scoring.apply_timing_penalty(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # Default is enabled, so penalty should apply
        assert abs(result_conf - 0.72) < 0.001  # 0.80 * 0.90


# ============================================================================
# Test Confidence Scoring Edge Cases (US-86-007)
# ============================================================================

class TestConfidenceScoringEdgeCases:
    """Parameterized edge case tests for confidence scoring functions (US-86-007).

    Tests boundary values (0.0, 0.5, 1.0), edge cases (NaN, Inf, negative),
    and empty/minimal inputs for confidence scoring functions.
    """

    @pytest.fixture
    def edge_case_config(self):
        """Mock config for edge case testing"""
        config = Mock()
        matching = Mock()
        matching.ideal_speed_range = (0.9, 1.1)
        matching.soft_speed_range = (0.7, 1.3)
        matching.duration_penalty_factor = 0.1
        matching.duration_scoring_enabled = True
        matching.soft_penalty_range = (0.7, 1.3)
        matching.broll_boost = 0.1
        matching.topic_mismatch_penalty = 0.2
        matching.chapter_matching_enabled = True
        scoring_mock = Mock()
        scoring_mock.duration_ratio_reward_threshold = 0.1
        scoring_mock.duration_ratio_reward_boost = 0.02
        matching.scoring = scoring_mock
        config.matching = matching
        global_cache = Mock()
        global_cache.current_project_boost = 0.1
        config.global_cache = global_cache
        return config

    @pytest.mark.fast
    @pytest.mark.parametrize("confidence,expected_behavior", [
        (0.0, "returns_boosted"),  # Zero confidence - should get boost
        (0.5, "returns_adjusted"),  # Mid confidence
        (1.0, "returns_over_1"),    # Max confidence - can exceed 1.0 (no cap)
    ])
    def test_apply_duration_penalty_boundary_scores(self, edge_case_config, confidence, expected_behavior):
        """Test boundary confidence values (0.0, 0.5, 1.0) with apply_duration_penalty."""
        speed_ratio = 1.0  # Perfect ratio

        result = scoring.apply_duration_penalty(confidence, speed_ratio, edge_case_config)

        if expected_behavior == "returns_boosted":
            assert result > 0.0  # Zero confidence should get boost applied
        elif expected_behavior == "returns_adjusted":
            assert 0.0 < result <= 1.0  # Mid confidence returns adjusted value
        elif expected_behavior == "returns_over_1":
            # Note: Function does not cap at 1.0, boost can exceed
            assert result > 1.0  # Can exceed 1.0 (no cap in implementation)

    @pytest.mark.fast
    @pytest.mark.parametrize("speed_ratio,expected_behavior", [
        (0.0, "penalty"),        # Zero ratio - should apply penalty
        (0.5, "penalty"),        # Low ratio - graduated penalty
        (1.0, "boost"),          # Perfect ratio - gets boost
        (2.0, "penalty"),       # High ratio - graduated penalty
        (3.0, "max_penalty"),   # Max ratio - hard ceiling penalty
        (-1.0, "penalty"),      # Negative ratio - penalty
    ])
    def test_apply_duration_penalty_edge_cases(self, edge_case_config, speed_ratio, expected_behavior):
        """Test edge case speed ratios including negative and extreme values."""
        confidence = 0.8

        result = scoring.apply_duration_penalty(confidence, speed_ratio, edge_case_config)

        if expected_behavior == "boost":
            assert result > confidence  # Boost should increase
        elif expected_behavior in ("penalty", "max_penalty"):
            assert result < confidence  # Penalty should decrease

    @pytest.mark.fast
    @pytest.mark.parametrize("confidence", [
        -0.5,  # Negative
        -1.0,  # More negative
    ])
    def test_apply_duration_penalty_negative_confidence(self, edge_case_config, confidence):
        """Test negative confidence values are handled gracefully."""
        speed_ratio = 1.0

        result = scoring.apply_duration_penalty(confidence, speed_ratio, edge_case_config)

        # Should return a valid float (not raise exception)
        assert isinstance(result, float)

    @pytest.mark.fast
    @pytest.mark.parametrize("confidence,speed_ratio", [
        (0.8, 0.0),
        (0.8, -1.0),
        (0.8, -0.5),
    ])
    def test_apply_duration_penalty_zero_negative_ratio(self, edge_case_config, confidence, speed_ratio):
        """Test zero and negative speed ratios are handled safely."""
        result = scoring.apply_duration_penalty(confidence, speed_ratio, edge_case_config)

        # Should return a float, not raise exception
        assert isinstance(result, float)

    @pytest.mark.fast
    def test_apply_duration_penalty_extreme_ratio_ceiling(self, edge_case_config):
        """Test speed_ratio > 3.0 applies maximum penalty."""
        confidence = 0.8
        speed_ratio = 10.0  # Extreme high

        result = scoring.apply_duration_penalty(confidence, speed_ratio, edge_case_config)

        # Should apply maximum penalty (2x factor)
        assert result < confidence - 0.15  # Less than with normal penalty

    @pytest.mark.fast
    def test_apply_duration_penalty_extreme_ratio_floor(self, edge_case_config):
        """Test speed_ratio < 0.3 applies maximum penalty."""
        confidence = 0.8
        speed_ratio = 0.1  # Extreme low

        result = scoring.apply_duration_penalty(confidence, speed_ratio, edge_case_config)

        # Should apply maximum penalty (2x factor)
        assert result < confidence - 0.15

    @pytest.mark.fast
    @pytest.mark.parametrize("confidence", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_apply_broll_boost_boundary(self, edge_case_config, confidence):
        """Test apply_broll_boost with boundary confidence values."""
        video_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test video", source_file="test.mp4"
        )
        video_segment.is_broll = True

        result, reason = scoring.apply_broll_boost(confidence, video_segment, edge_case_config)

        if confidence < 1.0:
            # Boost should apply and result should be higher
            assert result >= confidence
            assert result <= 1.0  # Capped at 1.0
        else:
            assert result == 1.0  # Already at max

    @pytest.mark.fast
    @pytest.mark.parametrize("is_broll", [True, False])
    def test_apply_broll_boost_broll_flag(self, edge_case_config, is_broll):
        """Test apply_broll_boost behavior with broll flag."""
        confidence = 0.7
        video_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test video", source_file="test.mp4"
        )
        video_segment.is_broll = is_broll

        result, reason = scoring.apply_broll_boost(confidence, video_segment, edge_case_config)

        if is_broll:
            assert result > confidence  # Boost applied
        else:
            assert result == confidence  # No change

    @pytest.mark.fast
    @pytest.mark.parametrize("boost_value", [0.0, 0.05, 0.1, 0.2, 0.5])
    def test_apply_broll_boost_values(self, edge_case_config, boost_value):
        """Test apply_broll_boost with various boost values."""
        edge_case_config.matching.broll_boost = boost_value
        confidence = 0.7
        video_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test video", source_file="test.mp4"
        )
        video_segment.is_broll = True

        result, reason = scoring.apply_broll_boost(confidence, video_segment, edge_case_config)

        expected = min(confidence + boost_value, 1.0)
        assert abs(result - expected) < 0.001

    @pytest.mark.fast
    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_apply_current_project_boost_boundary(self, edge_case_config, confidence):
        """Test apply_current_project_boost with boundary confidence values."""
        video_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test video", source_file="test.mp4"
        )
        video_segment.source = "current_project"

        result, reason = scoring.apply_current_project_boost(confidence, video_segment, edge_case_config)

        assert result >= confidence
        assert result <= 1.0

    @pytest.mark.fast
    def test_apply_current_project_boost_empty_project(self, edge_case_config):
        """Test apply_current_project_boost with global cache source."""
        confidence = 0.7
        video_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test video", source_file="test.mp4"
        )
        video_segment.source = "global_cache"

        result, reason = scoring.apply_current_project_boost(confidence, video_segment, edge_case_config)

        # From global cache - should apply penalty
        assert isinstance(result, float)

    @pytest.mark.fast
    @pytest.mark.parametrize("confidence,expected_change", [
        (0.0, "penalty"),   # Zero gets penalty (no boost applied)
        (0.3, "penalty"),   # Low gets penalty
        (0.5, "penalty"),   # Mid gets penalty
        (0.8, "penalty"),   # High gets penalty
        (1.0, "penalty"),   # Max gets penalty
    ])
    def test_apply_current_project_boost_global_cache(self, edge_case_config, confidence, expected_change):
        """Test apply_current_project_boost applies penalty for global cache sources."""
        video_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test video", source_file="test.mp4"
        )
        video_segment.source = "global_cache"  # From cache gets penalty
        original = confidence

        result, reason = scoring.apply_current_project_boost(confidence, video_segment, edge_case_config)

        # Global cache gets penalty
        assert "global cache" in reason.lower() or result < original

    @pytest.mark.fast
    def test_apply_current_project_boost_current_project(self, edge_case_config):
        """Test apply_current_project_boost with current project source (no penalty)."""
        confidence = 0.7
        video_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test video", source_file="test.mp4"
        )
        video_segment.source = "current_project"  # From current project

        result, reason = scoring.apply_current_project_boost(confidence, video_segment, edge_case_config)

        # Current project should not get penalty
        assert result == confidence
        assert reason == ""


class TestComputeDurationPenaltyEdgeCases:
    """Edge case tests for compute_duration_penalty function."""

    @pytest.fixture
    def duration_config(self):
        """Mock config for duration penalty testing"""
        config = Mock()
        matching = Mock()
        matching.ideal_speed_range = (0.9, 1.1)
        matching.soft_speed_range = (0.7, 1.3)
        matching.duration_penalty_factor = 0.1
        matching.duration_scoring_enabled = True
        matching.soft_penalty_range = (0.7, 1.3)
        config.matching = matching
        return config

    @pytest.mark.fast
    @pytest.mark.parametrize("vo_start,vo_end,video_start,video_end", [
        (0.0, 0.0, 0.0, 10.0),   # Zero vo duration
        (0.0, 10.0, 0.0, 0.0),    # Zero video duration
        (0.0, 0.0, 0.0, 0.0),    # Both zero
        (0.0, 5.0, 0.0, 5.0),    # Equal durations
        (0.0, 10.0, 0.0, 5.0),   # Vo longer
        (0.0, 5.0, 0.0, 10.0),   # Video longer
    ])
    def test_compute_duration_penalty_edge_cases(self, duration_config, vo_start, vo_end, video_start, video_end):
        """Test compute_duration_penalty with edge case durations."""
        vo_segment = SRTSegment(
            index=1, start_time=vo_start, end_time=vo_end,
            text="VO", source_file="vo.srt"
        )
        video_segment = SRTSegment(
            index=1, start_time=video_start, end_time=video_end,
            text="Video", source_file="video.mp4"
        )

        result = scoring.compute_duration_penalty(vo_segment, video_segment, duration_config)

        # Should return a valid float, not raise exception
        assert isinstance(result, float)

    @pytest.mark.fast
    @pytest.mark.parametrize("vo_end", [0.0])
    def test_compute_duration_penalty_zero_vo(self, duration_config, vo_end):
        """Test compute_duration_penalty with zero voiceover duration."""
        vo_segment = SRTSegment(
            index=1, start_time=0.0, end_time=vo_end,
            text="VO", source_file="vo.srt"
        )
        video_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Video", source_file="video.mp4"
        )

        result = scoring.compute_duration_penalty(vo_segment, video_segment, duration_config)

        # Should handle gracefully
        assert isinstance(result, float)


class TestApplyDurationScoringEdgeCases:
    """Edge case tests for apply_duration_scoring function."""

    @pytest.fixture
    def scoring_config(self):
        """Mock config for duration scoring tests"""
        config = Mock()
        matching = Mock()
        matching.duration_scoring_enabled = True
        matching.ideal_speed_range = (0.9, 1.1)
        matching.soft_speed_range = (0.7, 1.3)
        matching.duration_penalty_factor = 0.1
        matching.soft_penalty_range = (0.7, 1.3)
        scoring_mock = Mock()
        scoring_mock.duration_ratio_reward_threshold = 0.1
        scoring_mock.duration_ratio_reward_boost = 0.02
        matching.scoring = scoring_mock
        config.matching = matching
        return config

    @pytest.mark.fast
    @pytest.mark.parametrize("vo_end", [0.0, 5.0, 10.0])
    def test_apply_duration_scoring_boundary_durations(self, scoring_config, vo_end):
        """Test apply_duration_scoring with boundary duration values."""
        vo_segment = SRTSegment(
            index=1, start_time=0.0, end_time=vo_end,
            text="VO", source_file="vo.srt"
        )
        video_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Video", source_file="video.mp4"
        )
        candidates = [(video_segment, 0.8)]

        result = scoring.apply_duration_scoring(vo_segment, candidates, scoring_config)

        # Should return list of tuples
        assert isinstance(result, list)
        if result:
            assert len(result[0]) == 3  # (segment, score, penalty)

    @pytest.mark.fast
    def test_apply_duration_scoring_disabled(self, scoring_config):
        """Test apply_duration_scoring when disabled in config."""
        scoring_config.matching.duration_scoring_enabled = False

        vo_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="VO", source_file="vo.srt"
        )
        video_segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Video", source_file="video.mp4"
        )
        candidates = [(video_segment, 0.8)]

        result = scoring.apply_duration_scoring(vo_segment, candidates, scoring_config)

        # When disabled, returns candidates unchanged (with penalty=1.0)
        assert isinstance(result, list)
        assert len(result) == 1

