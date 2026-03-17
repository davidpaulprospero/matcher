"""
Tests for apply_listicle_consistency standalone function (US-75-007).

Verifies that:
- Segments within the same listicle group that share the same video source
  get a +0.04 consistency boost
- Segments outside any listicle group get no adjustment
- Boundary segments (first in a group) get no adjustment
- Segments with no recent matches get no adjustment
- The adjustment appears in confidence_breakdown when active
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_listicle_consistency
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


def _make_listicle_group(group_id: int, start: int, end: int) -> ListicleGroup:
    """Create a ListicleGroup with the given segment range."""
    return ListicleGroup(
        group_id=group_id,
        item_label=f"item {group_id + 1}",
        start_segment_idx=start,
        end_segment_idx=end,
        topic_keywords=["topic"],
    )


class TestApplyListicleConsistency:
    """Tests for the standalone apply_listicle_consistency function (US-75-007)."""

    def test_boost_when_same_source_in_same_group(self):
        """Segments within a listicle group prefer same video source as other segments in group."""
        groups = [_make_listicle_group(0, start=0, end=4)]
        # Previous match: segment 1 matched to video_A
        prev_match = _make_match("video_A", vo_index=1)
        # Current: segment 2, candidate from video_A
        vo_seg = _make_segment(index=2)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_listicle_consistency(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf == pytest.approx(0.84, abs=0.001)
        assert "listicle_consistency" in reason
        assert "same source" in reason
        assert "group 0" in reason

    def test_no_boost_when_outside_any_group(self):
        """Segments outside any listicle group get no listicle_consistency adjustment."""
        groups = [_make_listicle_group(0, start=0, end=2)]
        prev_match = _make_match("video_A", vo_index=4)
        # Segment 5 is outside the group (0-2)
        vo_seg = _make_segment(index=5)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_listicle_consistency(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf == 0.80
        assert reason == ""

    def test_no_boost_at_group_boundary(self):
        """First segment of a listicle group gets no boost (boundary segment)."""
        groups = [_make_listicle_group(0, start=3, end=6)]
        prev_match = _make_match("video_A", vo_index=2)
        # Segment 3 is the first in the group
        vo_seg = _make_segment(index=3)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_listicle_consistency(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf == 0.80
        assert reason == ""

    def test_no_boost_when_different_source(self):
        """No boost when candidate is from a different video source."""
        groups = [_make_listicle_group(0, start=0, end=4)]
        prev_match = _make_match("video_A", vo_index=1)
        vo_seg = _make_segment(index=2)
        vid_seg = _make_segment(index=0, source_file="video_B")  # different source

        conf, reason = apply_listicle_consistency(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf == 0.80
        assert reason == ""

    def test_no_boost_when_no_listicle_groups(self):
        """No adjustment when listicle_groups is None or empty."""
        prev_match = _make_match("video_A", vo_index=1)
        vo_seg = _make_segment(index=2)
        vid_seg = _make_segment(index=0, source_file="video_A")

        # None
        conf, reason = apply_listicle_consistency(
            0.80, vo_seg, vid_seg, listicle_groups=None, recent_matches=[prev_match]
        )
        assert conf == 0.80
        assert reason == ""

        # Empty list
        conf, reason = apply_listicle_consistency(
            0.80, vo_seg, vid_seg, listicle_groups=[], recent_matches=[prev_match]
        )
        assert conf == 0.80
        assert reason == ""

    def test_no_boost_when_no_recent_matches(self):
        """No adjustment when recent_matches is None or empty."""
        groups = [_make_listicle_group(0, start=0, end=4)]
        vo_seg = _make_segment(index=2)
        vid_seg = _make_segment(index=0, source_file="video_A")

        # None
        conf, reason = apply_listicle_consistency(
            0.80, vo_seg, vid_seg, listicle_groups=groups, recent_matches=None
        )
        assert conf == 0.80
        assert reason == ""

        # Empty
        conf, reason = apply_listicle_consistency(
            0.80, vo_seg, vid_seg, listicle_groups=groups, recent_matches=[]
        )
        assert conf == 0.80
        assert reason == ""

    def test_no_boost_when_prev_match_in_different_group(self):
        """No boost when previous match's voiceover segment is in a different group."""
        groups = [
            _make_listicle_group(0, start=0, end=2),
            _make_listicle_group(1, start=3, end=5),
        ]
        # Previous match in group 0 (segment 2)
        prev_match = _make_match("video_A", vo_index=2)
        # Current segment in group 1 (segment 4, not boundary)
        vo_seg = _make_segment(index=4)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_listicle_consistency(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf == 0.80
        assert reason == ""

    def test_boost_with_dict_groups(self):
        """Works with dict-based listicle groups (backward compatibility)."""
        groups = [{"group_id": 0, "start_segment_idx": 0, "end_segment_idx": 4}]
        prev_match = _make_match("video_A", vo_index=1)
        vo_seg = _make_segment(index=2)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_listicle_consistency(
            0.80, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf == pytest.approx(0.84, abs=0.001)
        assert "listicle_consistency" in reason

    def test_confidence_capped_at_1(self):
        """Boost doesn't push confidence above 1.0."""
        groups = [_make_listicle_group(0, start=0, end=4)]
        prev_match = _make_match("video_A", vo_index=1)
        vo_seg = _make_segment(index=2)
        vid_seg = _make_segment(index=0, source_file="video_A")

        conf, reason = apply_listicle_consistency(
            0.99, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf == 1.0
        assert "listicle_consistency" in reason

    def test_multiple_groups_correct_group_selected(self):
        """When multiple groups exist, the correct group is matched."""
        groups = [
            _make_listicle_group(0, start=0, end=2),
            _make_listicle_group(1, start=3, end=6),
            _make_listicle_group(2, start=7, end=9),
        ]
        # Previous in group 1, segment 4
        prev_match = _make_match("video_B", vo_index=4)
        # Current in group 1, segment 5
        vo_seg = _make_segment(index=5)
        vid_seg = _make_segment(index=0, source_file="video_B")

        conf, reason = apply_listicle_consistency(
            0.75, vo_seg, vid_seg,
            listicle_groups=groups,
            recent_matches=[prev_match],
        )

        assert conf == pytest.approx(0.79, abs=0.001)
        assert "group 1" in reason
