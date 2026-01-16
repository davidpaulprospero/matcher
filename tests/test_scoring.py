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

    def test_ideal_speed_no_penalty(self, mock_config):
        """Test no penalty for ideal speed ratio"""
        confidence = 0.8
        speed_ratio = 1.0  # Perfect match (within 0.9-1.1)

        result = scoring.apply_duration_penalty(confidence, speed_ratio, mock_config)

        assert result == 0.8  # No change

    def test_ideal_speed_boundary_no_penalty(self, mock_config):
        """Test boundaries of ideal range"""
        # Lower boundary
        result1 = scoring.apply_duration_penalty(0.8, 0.9, mock_config)
        assert result1 == 0.8

        # Upper boundary
        result2 = scoring.apply_duration_penalty(0.8, 1.1, mock_config)
        assert result2 == 0.8

    def test_soft_speed_small_penalty(self, mock_config):
        """Test small penalty for soft speed range"""
        confidence = 0.8
        speed_ratio = 0.8  # In soft range (0.7-1.3) but outside ideal (0.9-1.1)

        result = scoring.apply_duration_penalty(confidence, speed_ratio, mock_config)

        # penalty_factor = 0.1
        assert abs(result - 0.7) < 0.001  # 0.8 - 0.1 (allow floating point imprecision)

    def test_outside_both_ranges_large_penalty(self, mock_config):
        """Test large penalty for speed outside both ranges"""
        confidence = 0.9
        speed_ratio = 0.5  # Outside soft range (< 0.7)

        result = scoring.apply_duration_penalty(confidence, speed_ratio, mock_config)

        # penalty_factor * 2 = 0.2
        assert result == 0.7  # 0.9 - 0.2

    def test_very_fast_speed_large_penalty(self, mock_config):
        """Test penalty for very fast speeds"""
        confidence = 0.9
        speed_ratio = 1.5  # Much faster than ideal

        result = scoring.apply_duration_penalty(confidence, speed_ratio, mock_config)

        assert result == 0.7  # 0.9 - 0.2


# ============================================================================
# Test apply_topic_penalty()
# ============================================================================

class TestApplyTopicPenalty:
    """Test topic-based confidence penalties"""

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

    def test_duration_scoring_disabled(self, mock_config, sample_vo_segment, sample_video_segment):
        """Test no penalty when duration scoring disabled"""
        mock_config.matching.duration_scoring_enabled = False

        penalty = scoring.compute_duration_penalty(sample_vo_segment, sample_video_segment, mock_config)

        assert penalty == 0.0

    def test_ideal_speed_no_penalty(self, mock_config, sample_vo_segment, sample_video_segment):
        """Test no penalty for ideal speed match"""
        # VO: 10s, Video: 10s → speed_ratio = 1.0 (ideal)
        penalty = scoring.compute_duration_penalty(sample_vo_segment, sample_video_segment, mock_config)

        assert penalty == 0.0

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

    def test_apply_duration_scoring_empty(self, mock_config, sample_vo_segment):
        """Test with empty candidates"""
        candidates = []

        result = scoring.apply_duration_scoring(sample_vo_segment, candidates, mock_config)

        assert result == []

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

        # Apply duration penalty (ideal speed = no penalty)
        confidence = scoring.apply_duration_penalty(confidence, 1.0, mock_config)
        assert confidence == 0.7

        # Apply topic penalty (good match = minimal penalty)
        with patch('src.matching.scoring.compute_topic_penalty', return_value=0.05):
            confidence, _ = scoring.apply_topic_penalty(
                confidence, sample_vo_segment, video_seg, video_topics,
                True, 0.3
            )
        assert abs(confidence - 0.65) < 0.001

        # Apply B-roll boost
        confidence, _ = scoring.apply_broll_boost(confidence, video_seg, mock_config)
        assert abs(confidence - 0.75) < 0.001  # +0.1

        # Apply global cache penalty
        confidence, _ = scoring.apply_current_project_boost(confidence, video_seg, mock_config)
        assert abs(confidence - 0.65) < 0.001  # -0.1

        # Final score balances all factors
        assert 0.6 <= confidence <= 0.7

    def test_extreme_speed_ratios(self, mock_config):
        """Test handling of extreme speed ratios"""
        # Very slow (speed up 10x)
        penalty1 = scoring.apply_duration_penalty(0.8, 10.0, mock_config)
        assert penalty1 <= 0.601  # Large penalty (allow floating point tolerance)

        # Very fast (slow down 10x)
        penalty2 = scoring.apply_duration_penalty(0.8, 0.1, mock_config)
        assert penalty2 <= 0.601  # Large penalty (allow floating point tolerance)

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
# Test apply_caption_boost()
# ============================================================================

class TestApplyCaptionBoost:
    """Test manual caption confidence boosts"""

    def test_not_manual_caption_no_boost(self, mock_config, sample_video_segment):
        """Test no boost for non-caption segments"""
        confidence = 0.7

        # Segment without transcript_source (defaults to empty)
        result_conf, reason = scoring.apply_caption_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        assert result_conf == 0.7
        assert reason == ""

    def test_whisper_no_boost(self, mock_config, sample_video_segment):
        """Test no boost for whisper transcripts"""
        confidence = 0.7
        sample_video_segment.transcript_source = "whisper"

        result_conf, reason = scoring.apply_caption_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        assert result_conf == 0.7
        assert reason == ""

    def test_auto_caption_no_boost(self, mock_config, sample_video_segment):
        """Test no boost for auto-generated captions"""
        confidence = 0.7
        sample_video_segment.transcript_source = "auto_caption"

        result_conf, reason = scoring.apply_caption_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        assert result_conf == 0.7
        assert reason == ""

    def test_manual_caption_boost_applied(self, mock_config, sample_video_segment):
        """Test boost applied for manual captions"""
        confidence = 0.7
        sample_video_segment.transcript_source = "manual_caption"

        # Set up caption_first config
        mock_config.download = Mock()
        mock_config.download.caption_first = Mock()
        mock_config.download.caption_first.confidence_boost_manual = 0.1

        result_conf, reason = scoring.apply_caption_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        assert abs(result_conf - 0.8) < 0.001  # 0.7 + 0.1
        assert "manual caption boost" in reason
        assert "+0.10" in reason

    def test_manual_caption_boost_capped_at_1(self, mock_config, sample_video_segment):
        """Test boost doesn't exceed 1.0"""
        confidence = 0.95
        sample_video_segment.transcript_source = "manual_caption"

        mock_config.download = Mock()
        mock_config.download.caption_first = Mock()
        mock_config.download.caption_first.confidence_boost_manual = 0.1

        result_conf, reason = scoring.apply_caption_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        assert result_conf == 1.0  # Capped at 1.0
        assert "manual caption boost" in reason

    def test_manual_caption_boost_zero_config(self, mock_config, sample_video_segment):
        """Test no boost when configured to 0"""
        confidence = 0.7
        sample_video_segment.transcript_source = "manual_caption"

        mock_config.download = Mock()
        mock_config.download.caption_first = Mock()
        mock_config.download.caption_first.confidence_boost_manual = 0.0

        result_conf, reason = scoring.apply_caption_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=mock_config
        )

        assert result_conf == 0.7
        assert reason == ""

    def test_manual_caption_boost_missing_config(self, sample_video_segment):
        """Test default boost when caption config is missing"""
        confidence = 0.7
        sample_video_segment.transcript_source = "manual_caption"

        # Config without download.caption_first
        config = Mock()
        config.download = None

        result_conf, reason = scoring.apply_caption_boost(
            confidence=confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # Default boost should be 0.1
        assert abs(result_conf - 0.8) < 0.001
        assert "manual caption boost" in reason
