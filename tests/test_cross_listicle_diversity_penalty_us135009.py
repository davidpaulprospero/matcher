"""
Tests for apply_cross_listicle_diversity_penalty standalone function (US-135-009).

Verifies that:
- Same source for 2+ consecutive listicle items: small penalty (-0.02 per repeat)
- Same source for 3+ consecutive: larger penalty (-0.05 per repeat after 2)
- Penalty does NOT apply when listicle items are thematically related (same topic keywords)
- Penalty is skipped when listicle_diversity_penalty_enabled is False
- No penalty when no listicle groups or recent matches
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_cross_listicle_diversity_penalty, _compute_topic_overlap
from src.utils import SRTSegment
from src.chapter_detection.models import ListicleGroup


def _make_segment(index: int, text: str = "segment text", source_file: str = "") -> SRTSegment:
    """Create a minimal SRTSegment with given index and optional source."""
    seg = SRTSegment(
        index=index,
        start_time=float(index * 5),
        end_time=float(index * 5 + 5),
        text=text,
        source_file=source_file,
    )
    return seg


def _make_match(video_source_file: str, vo_index: int = 0):
    """Create a mock Match object with video_segment and voiceover_segment."""
    match = MagicMock()
    match.video_segment = MagicMock()
    match.video_segment.source_file = video_source_file
    match.voiceover_segment = MagicMock()
    match.voiceover_segment.index = vo_index
    return match


def _make_listicle_group(group_id: int, start: int, end: int, topic_keywords: list = None) -> ListicleGroup:
    """Create a ListicleGroup with the given segment range."""
    return ListicleGroup(
        group_id=group_id,
        item_label=f"item {group_id + 1}",
        start_segment_idx=start,
        end_segment_idx=end,
        topic_keywords=topic_keywords or ["topic"],
    )


class TestComputeTopicOverlap:
    """Tests for the topic overlap helper function."""

    def test_full_overlap(self):
        """Complete overlap returns 1.0."""
        keywords1 = ["python", "tutorial", "beginner"]
        keywords2 = ["python", "tutorial", "beginner"]
        assert _compute_topic_overlap(keywords1, keywords2) == pytest.approx(1.0, abs=0.01)

    def test_partial_overlap(self):
        """Partial overlap returns ratio."""
        keywords1 = ["python", "tutorial", "beginner"]
        keywords2 = ["python", "coding", "advanced"]
        # 1 common / 3 max = 0.333
        assert _compute_topic_overlap(keywords1, keywords2) == pytest.approx(0.333, abs=0.01)

    def test_no_overlap(self):
        """No overlap returns 0.0."""
        keywords1 = ["python", "tutorial"]
        keywords2 = ["javascript", "react"]
        assert _compute_topic_overlap(keywords1, keywords2) == pytest.approx(0.0, abs=0.01)

    def test_empty_keywords(self):
        """Empty list returns 0.0."""
        assert _compute_topic_overlap([], ["python"]) == 0.0
        assert _compute_topic_overlap(["python"], []) == 0.0
        assert _compute_topic_overlap([], []) == 0.0

    def test_case_insensitive(self):
        """Keywords are compared case-insensitively."""
        keywords1 = ["Python", "TUTORIAL"]
        keywords2 = ["python", "tutorial"]
        assert _compute_topic_overlap(keywords1, keywords2) == pytest.approx(1.0, abs=0.01)


class TestApplyCrossListicleDiversityPenalty:
    """Tests for the cross-listicle diversity penalty function."""

    def test_penalty_2_consecutive_listicle_items(self):
        """Same source for 2+ consecutive listicle items: -0.02 penalty."""
        # Group 0: segments 0-2, Group 1: segments 3-5
        groups = [
            _make_listicle_group(0, start=0, end=2, topic_keywords=["topic_a"]),
            _make_listicle_group(1, start=3, end=5, topic_keywords=["topic_b"]),
        ]
        # Previous match: segment 2 (group 0) matched to video_A
        prev_match = _make_match("video_A", vo_index=2)
        # Current: segment 3 (group 1), candidate from video_A (same source as group 0)
        vo_seg = _make_segment(index=3)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        # 2 consecutive listicle items with same source = -0.02
        assert conf == pytest.approx(0.78, abs=0.001)
        assert "cross_listicle_diversity_penalty" in reason

    def test_penalty_3_consecutive_listicle_items(self):
        """Same source for 3+ consecutive: -0.05 per repeat after 2."""
        # Group 0: segments 0-2, Group 1: segments 3-5, Group 2: segments 6-8
        groups = [
            _make_listicle_group(0, start=0, end=2, topic_keywords=["topic_a"]),
            _make_listicle_group(1, start=3, end=5, topic_keywords=["topic_b"]),
            _make_listicle_group(2, start=6, end=8, topic_keywords=["topic_c"]),
        ]
        # Previous matches: group 1 and group 0 both matched to video_A
        prev_match1 = _make_match("video_A", vo_index=4)  # group 1 - most recent
        prev_match2 = _make_match("video_A", vo_index=1)  # group 0 - older
        # Current: segment 7 (group 2), candidate from video_A
        vo_seg = _make_segment(index=7)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match1, prev_match2],
        )

        # 3 consecutive = -0.02 - 0.05 = -0.07
        assert conf == pytest.approx(0.73, abs=0.001)
        assert "cross_listicle_diversity_penalty" in reason

    def test_no_penalty_when_topics_related(self):
        """Penalty does NOT apply when listicle items are thematically related."""
        # Group 0: segments 0-2, Group 1: segments 3-5 - both have overlapping topics
        groups = [
            _make_listicle_group(0, start=0, end=2, topic_keywords=["python", "tutorial"]),
            _make_listicle_group(1, start=3, end=5, topic_keywords=["python", "coding"]),
        ]
        # Previous match: segment 2 (group 0) matched to video_A
        prev_match = _make_match("video_A", vo_index=2)
        # Current: segment 3 (group 1), candidate from video_A
        vo_seg = _make_segment(index=3)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        # No penalty due to topic overlap (python is in both)
        assert conf == 0.80
        assert reason == ""

    def test_no_penalty_when_disabled(self):
        """Penalty is skipped when listicle_diversity_penalty_enabled is False."""
        groups = [
            _make_listicle_group(0, start=0, end=2, topic_keywords=["topic_a"]),
            _make_listicle_group(1, start=3, end=5, topic_keywords=["topic_b"]),
        ]
        prev_match = _make_match("video_A", vo_index=4)
        vo_seg = _make_segment(index=5)
        vid_seg = _make_segment(index=0, source_file="video_A")

        # Create mock config with disabled
        mock_config = MagicMock()
        mock_config.matching = MagicMock()
        mock_config.matching.listicle_diversity_penalty_enabled = False

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
            config=mock_config,
        )

        assert conf == 0.80
        assert reason == ""

    def test_no_penalty_no_listicle_groups(self):
        """No penalty when no listicle groups provided."""
        vo_seg = _make_segment(index=5)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.80, vo_seg, vid_seg,
            listicle_groups=None,
            recent_matches=[],
        )

        assert conf == 0.80
        assert reason == ""

    def test_no_penalty_no_recent_matches(self):
        """No penalty when no recent matches."""
        groups = [_make_listicle_group(0, start=0, end=2)]
        vo_seg = _make_segment(index=5)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=None,
        )

        assert conf == 0.80
        assert reason == ""

    def test_no_penalty_different_sources(self):
        """No penalty when different sources across listicle items."""
        groups = [
            _make_listicle_group(0, start=0, end=2),
            _make_listicle_group(1, start=3, end=5),
        ]
        # Previous match used different source
        prev_match = _make_match("video_B", vo_index=4)
        vo_seg = _make_segment(index=5)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf == 0.80
        assert reason == ""

    def test_no_penalty_outside_listicle_group(self):
        """No penalty when current segment is not in a listicle group."""
        groups = [
            _make_listicle_group(0, start=0, end=2),
            _make_listicle_group(1, start=3, end=5),
        ]
        # Segment 10 is outside any group
        prev_match = _make_match("video_A", vo_index=4)
        vo_seg = _make_segment(index=10)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf == 0.80
        assert reason == ""

    def test_no_penalty_same_listicle_group(self):
        """No penalty when matches are within the SAME listicle group (different mechanism)."""
        # Both segments in the same group - this is handled by listicle_consistency boost
        groups = [
            _make_listicle_group(0, start=0, end=5),
        ]
        prev_match = _make_match("video_A", vo_index=2)
        vo_seg = _make_segment(index=3)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        # Same group should not trigger cross-listicle penalty
        assert conf == 0.80
        assert reason == ""

    def test_config_values_used(self):
        """Config values are used when provided."""
        groups = [
            _make_listicle_group(0, start=0, end=2, topic_keywords=["topic_a"]),
            _make_listicle_group(1, start=3, end=5, topic_keywords=["topic_b"]),
        ]
        # Previous match is in a different group (group 0)
        prev_match = _make_match("video_A", vo_index=2)
        # Current is in group 1
        vo_seg = _make_segment(index=3)
        vid_seg = _make_segment(index=0, source_file="video_A")

        # Custom config values
        mock_config = MagicMock()
        mock_config.matching = MagicMock()
        mock_config.matching.listicle_diversity_penalty_enabled = True
        mock_config.matching.listicle_diversity_penalty_2_consecutive = 0.05
        mock_config.matching.listicle_diversity_penalty_3_plus = 0.10
        mock_config.matching.listicle_diversity_topic_overlap_threshold = 0.1

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
            config=mock_config,
        )

        # Should use custom penalty values: 2 consecutive = -0.05
        assert conf == pytest.approx(0.75, abs=0.001)

    def test_confidence_not_below_zero(self):
        """Penalty does not reduce confidence below 0."""
        groups = [
            _make_listicle_group(0, start=0, end=2),
            _make_listicle_group(1, start=3, end=5),
        ]
        prev_match = _make_match("video_A", vo_index=4)
        vo_seg = _make_segment(index=5)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_cross_listicle_diversity_penalty(
            0.01, vo_seg, vid_seg,  # Very low confidence
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf >= 0.0
