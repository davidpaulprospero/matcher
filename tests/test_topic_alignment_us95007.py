"""
Tests for topic alignment boost (US-95-007).

Verifies that topic alignment between voiceover and video segments
boosts confidence when topics align.

Created: 2026-02-13 (Sprint 95 - US-95-007)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_topic_alignment_boost
from src.topic_extraction import compute_topic_alignment_boost, VideoTopics
from src.utils import SRTSegment


# ============================================================================
# Fixtures
# ============================================================================

def _make_vo_segment(text: str, topics: list = None) -> SRTSegment:
    """Create a mock voiceover segment."""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=5.0,
        text=text,
    )
    if topics:
        seg.topics = topics
    return seg


def _make_video_segment(source_file: str = "video_123") -> SRTSegment:
    """Create a mock video segment."""
    return SRTSegment(
        index=1,
        start_time=0.0,
        end_time=5.0,
        text="Video text content",
        source_file=source_file,
    )


def _make_video_topics(topics: list) -> dict:
    """Create a mock video_topics dict."""
    return {
        "video_123": VideoTopics(
            video_path="video_123",
            topics=topics,
            confidence=0.9
        )
    }


# ============================================================================
# Tests for compute_topic_alignment_boost
# ============================================================================

class TestComputeTopicAlignmentBoost:
    """Test the compute_topic_alignment_boost function."""

    @pytest.mark.fast
    def test_strong_overlap_full_boost(self):
        """3+ shared keywords gives full boost."""
        vo_topics = ["paris", "eiffel tower", "france", "travel"]
        video_topics = ["paris", "eiffel tower", "france", "tourism"]

        boost = compute_topic_alignment_boost(
            vo_topics, video_topics, max_boost=0.1, min_overlap=1
        )
        assert boost == pytest.approx(0.1)

    @pytest.mark.fast
    def test_partial_overlap_half_boost(self):
        """1-2 shared keywords gives 50% boost."""
        vo_topics = ["travel", "food", "restaurant"]
        video_topics = ["paris", "food", "restaurant", "cooking"]

        boost = compute_topic_alignment_boost(
            vo_topics, video_topics, max_boost=0.1, min_overlap=1
        )
        assert boost == pytest.approx(0.05)  # 50% of 0.1

    @pytest.mark.fast
    def test_no_overlap_no_boost(self):
        """No shared keywords gives no boost."""
        vo_topics = ["cooking", "recipe", "kitchen"]
        video_topics = ["car", "driving", "road"]

        boost = compute_topic_alignment_boost(
            vo_topics, video_topics, max_boost=0.1, min_overlap=1
        )
        assert boost == 0.0

    @pytest.mark.fast
    def test_empty_topics_no_boost(self):
        """Empty topic lists give no boost."""
        boost = compute_topic_alignment_boost(
            [], ["paris", "france"], max_boost=0.1
        )
        assert boost == 0.0

        boost = compute_topic_alignment_boost(
            ["travel"], [], max_boost=0.1
        )
        assert boost == 0.0


# ============================================================================
# Tests for apply_topic_alignment_boost
# ============================================================================

class TestApplyTopicAlignmentBoost:
    """Test the apply_topic_alignment_boost scoring function."""

    @pytest.mark.fast
    def test_aligned_topics_boost_confidence(self):
        """Aligned topics boost confidence."""
        vo_seg = _make_vo_segment("Voiceover text", topics=["paris", "france", "travel"])
        video_seg = _make_video_segment("video_123")
        video_topics = _make_video_topics(["paris", "eiffel tower", "france", "travel"])

        confidence, reason = apply_topic_alignment_boost(
            confidence=0.80,
            vo_segment=vo_seg,
            video_segment=video_seg,
            video_topics=video_topics,
            topic_alignment_weight=0.1
        )

        assert confidence == pytest.approx(0.90)  # 0.80 + 0.10
        assert "topic alignment boost" in reason

    @pytest.mark.fast
    def test_misaligned_topics_no_boost(self):
        """Misaligned topics don't boost confidence."""
        vo_seg = _make_vo_segment("Voiceover text", topics=["cooking", "recipe"])
        video_seg = _make_video_segment("video_123")
        video_topics = _make_video_topics(["car", "driving", "road"])

        confidence, reason = apply_topic_alignment_boost(
            confidence=0.80,
            vo_segment=vo_seg,
            video_segment=video_seg,
            video_topics=video_topics,
            topic_alignment_weight=0.1
        )

        assert confidence == 0.80  # No change
        assert reason == ""

    @pytest.mark.fast
    def test_no_vo_topics_no_boost(self):
        """Voiceover without topics doesn't get boost."""
        vo_seg = _make_vo_segment("Voiceover text")  # No topics
        video_seg = _make_video_segment("video_123")
        video_topics = _make_video_topics(["paris", "france"])

        confidence, reason = apply_topic_alignment_boost(
            confidence=0.80,
            vo_segment=vo_seg,
            video_segment=video_seg,
            video_topics=video_topics,
            topic_alignment_weight=0.1
        )

        assert confidence == 0.80
        assert reason == ""

    @pytest.mark.fast
    def test_no_video_topics_no_boost(self):
        """Video without topics doesn't get boost."""
        vo_seg = _make_vo_segment("Voiceover text", topics=["paris", "france"])
        video_seg = _make_video_segment("video_123")
        video_topics = {}  # Empty

        confidence, reason = apply_topic_alignment_boost(
            confidence=0.80,
            vo_segment=vo_seg,
            video_segment=video_seg,
            video_topics=video_topics,
            topic_alignment_weight=0.1
        )

        assert confidence == 0.80
        assert reason == ""

    @pytest.mark.fast
    def test_zero_weight_no_change(self):
        """Zero topic_alignment_weight doesn't change confidence."""
        vo_seg = _make_vo_segment("Voiceover text", topics=["paris", "france"])
        video_seg = _make_video_segment("video_123")
        video_topics = _make_video_topics(["paris", "france"])

        confidence, reason = apply_topic_alignment_boost(
            confidence=0.80,
            vo_segment=vo_seg,
            video_segment=video_seg,
            video_topics=video_topics,
            topic_alignment_weight=0.0  # Disabled
        )

        assert confidence == 0.80
        assert reason == ""

    @pytest.mark.fast
    def test_confidence_capped_at_one(self):
        """Confidence boost doesn't exceed 1.0."""
        vo_seg = _make_vo_segment("Voiceover text", topics=["paris", "france", "travel"])
        video_seg = _make_video_segment("video_123")
        video_topics = _make_video_topics(["paris", "eiffel tower", "france", "travel"])

        confidence, reason = apply_topic_alignment_boost(
            confidence=0.95,
            vo_segment=vo_seg,
            video_segment=video_seg,
            video_topics=video_topics,
            topic_alignment_weight=0.1
        )

        assert confidence == 1.0  # Capped at 1.0
        assert "topic alignment boost" in reason
