"""
Tests for US-75-002: Description relevance scoring adjustment.

Verifies:
- Standalone apply_description_relevance_adjustment function exists
- Graduated boost: 1 match -> +0.02, 2 matches -> +0.04, 3+ matches -> +0.06
- description_relevance key in confidence_breakdown after adjustment
- Integration with apply_all_adjustments
"""

import pytest
from unittest.mock import Mock

from src.matching.scoring import (
    apply_description_relevance_adjustment,
    MatchScoring,
)
from src.utils import SRTSegment


@pytest.fixture
def vo_segment():
    """Voiceover segment about Tokyo culture and food."""
    return SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="Tokyo is known for its amazing culture and traditional food scene",
        source_file="voiceover.srt",
    )


@pytest.fixture
def video_segment():
    """Video segment for pairing."""
    return SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="A documentary about Japanese cities",
        source_file="video123",
    )


@pytest.fixture
def mock_config():
    """Mock config with matching settings - all adjustments neutral."""
    config = Mock()
    matching = Mock()
    matching.multimodal_enabled = False
    matching.multimodal_weights = None
    matching.pool_normalization_enabled = False
    matching.broll_boost = 0.0
    matching.caption_quality_adjustment_enabled = False
    matching.entity_match_boost = 0.0
    matching.language_confidence_penalty = 0.0
    matching.timing_penalty_enabled = False
    scoring = Mock()
    scoring.confidence_floor = 0.05
    scoring.low_confidence_warning_threshold = 0.15
    matching.scoring = scoring
    config.matching = matching
    global_cache = Mock()
    global_cache.current_project_boost = 0.0
    config.global_cache = global_cache
    return config


@pytest.fixture
def score_manager(mock_config):
    """MatchScoring with mock config."""
    return MatchScoring(config=mock_config)


class TestApplyDescriptionRelevanceAdjustmentGraduated:
    """Test graduated boost values for 0, 1, 2, 3+ keyword matches."""

    def test_zero_matches_no_boost(self, vo_segment):
        """No boost when description shares no keywords with voiceover."""
        adjusted, reason = apply_description_relevance_adjustment(
            0.70, vo_segment, video_description="completely unrelated content xyz"
        )
        assert adjusted == pytest.approx(0.70, abs=0.001)
        assert reason == ""

    def test_one_keyword_match_boost_002(self):
        """1 keyword match -> +0.02 boost."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="The history of Tokyo architecture",
            source_file="vo.srt",
        )
        # "tokyo" overlaps
        adjusted, reason = apply_description_relevance_adjustment(
            0.70, vo, video_description="Tokyo travel guide for visitors"
        )
        assert adjusted == pytest.approx(0.72, abs=0.001)
        assert "+0.02" in reason
        assert "1 keyword" in reason

    def test_two_keyword_matches_boost_004(self):
        """2 keyword matches -> +0.04 boost."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="The culture and history of Tokyo",
            source_file="vo.srt",
        )
        # "tokyo" and "culture" overlap
        adjusted, reason = apply_description_relevance_adjustment(
            0.70, vo, video_description="Tokyo culture and modern lifestyle"
        )
        assert adjusted == pytest.approx(0.74, abs=0.001)
        assert "+0.04" in reason
        assert "2 keywords" in reason

    def test_three_plus_keyword_matches_boost_006(self):
        """3+ keyword matches -> +0.06 boost."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Tokyo culture food traditional Japanese cuisine",
            source_file="vo.srt",
        )
        # "tokyo", "culture", "food" all overlap
        adjusted, reason = apply_description_relevance_adjustment(
            0.70, vo, video_description="Tokyo culture food documentary highlights"
        )
        assert adjusted == pytest.approx(0.76, abs=0.001)
        assert "+0.06" in reason
        assert "3 keywords" in reason

    def test_no_description_no_boost(self, vo_segment):
        """No boost when description is None."""
        adjusted, reason = apply_description_relevance_adjustment(
            0.70, vo_segment, video_description=None
        )
        assert adjusted == pytest.approx(0.70, abs=0.001)
        assert reason == ""

    def test_empty_description_no_boost(self, vo_segment):
        """No boost when description is empty string."""
        adjusted, reason = apply_description_relevance_adjustment(
            0.70, vo_segment, video_description=""
        )
        assert adjusted == pytest.approx(0.70, abs=0.001)
        assert reason == ""

    def test_confidence_capped_at_1(self):
        """Boost should not exceed 1.0."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Tokyo culture food traditional",
            source_file="vo.srt",
        )
        adjusted, reason = apply_description_relevance_adjustment(
            0.98, vo, video_description="Tokyo culture food documentary"
        )
        assert adjusted <= 1.0


class TestDescriptionRelevanceInBreakdown:
    """Test that description_relevance appears in confidence_breakdown."""

    def test_description_relevance_key_in_breakdown(self, score_manager, vo_segment, video_segment):
        """description_relevance entry exists in confidence_breakdown after adjustment."""
        confidence, reason, breakdown = score_manager.apply_all_adjustments(
            confidence=0.70,
            vo_segment=vo_segment,
            video_segment=video_segment,
            video_description="Tokyo culture food documentary highlights",
        )

        # Find description_relevance in breakdown
        desc_entries = [b for b in breakdown if b['component'] == 'description_relevance']
        assert len(desc_entries) == 1, f"Expected description_relevance in breakdown, got components: {[b['component'] for b in breakdown]}"

        entry = desc_entries[0]
        assert isinstance(entry['adjustment'], (int, float))
        assert entry['adjustment'] > 0  # Should be a positive boost
        assert 'reason' in entry

    def test_no_description_relevance_without_description(self, score_manager, vo_segment, video_segment):
        """No description_relevance entry when no description provided."""
        confidence, reason, breakdown = score_manager.apply_all_adjustments(
            confidence=0.70,
            vo_segment=vo_segment,
            video_segment=video_segment,
        )

        desc_entries = [b for b in breakdown if b['component'] == 'description_relevance']
        assert len(desc_entries) == 0
