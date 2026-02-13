"""
Tests for apply_chapter_boundary_penalty standalone function (US-95-004).

Verifies that:
- When enforce_boundaries is False, no penalty is applied
- When both vo_segment and video_segment have the same chapter_index, no penalty
- When vo_segment and video_segment have different chapter_index, penalty is applied
- Penalty is not applied when either segment has no chapter_index
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_chapter_boundary_penalty
from src.utils import SRTSegment


def _make_segment(text: str, chapter_index=None, source_file=None) -> SRTSegment:
    """Create a minimal SRTSegment with given text and optional chapter info."""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=5.0,
        text=text,
    )
    if chapter_index is not None:
        seg.chapter_index = chapter_index
    if source_file is not None:
        seg.source_file = source_file
    return seg


class TestApplyChapterBoundaryPenalty:
    """Tests for the standalone apply_chapter_boundary_penalty function."""

    def test_disabled_when_enforce_boundaries_off(self):
        """No adjustment when enforce_boundaries is False."""
        vo_seg = _make_segment("Test voiceover", chapter_index=0)
        vid_seg = _make_segment("Test video", chapter_index=1)
        conf, reason = apply_chapter_boundary_penalty(
            0.80, vo_seg, vid_seg, enforce_boundaries=False
        )
        assert conf == 0.80
        assert reason == ""

    def test_same_chapter_no_penalty(self):
        """No penalty when voiceover and video are in the same chapter."""
        vo_seg = _make_segment("Test voiceover", chapter_index=2)
        vid_seg = _make_segment("Test video", chapter_index=2)
        conf, reason = apply_chapter_boundary_penalty(
            0.80, vo_seg, vid_seg, enforce_boundaries=True, penalty=0.1
        )
        assert conf == 0.80
        assert reason == ""

    def test_cross_chapter_penalty_applied(self):
        """Penalty applied when voiceover and video chapters differ."""
        vo_seg = _make_segment("Test voiceover", chapter_index=0)
        vid_seg = _make_segment("Test video", chapter_index=1)
        conf, reason = apply_chapter_boundary_penalty(
            0.80, vo_seg, vid_seg, enforce_boundaries=True, penalty=0.1
        )
        assert abs(conf - 0.70) < 0.001
        assert "cross_chapter_boundary" in reason
        assert "vo_ch=0" in reason
        assert "vid_ch=1" in reason

    def test_cross_chapter_penalty_respects_minimum(self):
        """Penalty doesn't reduce confidence below 0.0."""
        vo_seg = _make_segment("Test voiceover", chapter_index=0)
        vid_seg = _make_segment("Test video", chapter_index=1)
        conf, reason = apply_chapter_boundary_penalty(
            0.05, vo_seg, vid_seg, enforce_boundaries=True, penalty=0.1
        )
        assert conf == 0.0
        assert "cross_chapter_boundary" in reason

    def test_no_vo_chapter_index_no_penalty(self):
        """No penalty when voiceover segment has no chapter_index."""
        vo_seg = _make_segment("Test voiceover")  # No chapter_index
        vid_seg = _make_segment("Test video", chapter_index=1)
        conf, reason = apply_chapter_boundary_penalty(
            0.80, vo_seg, vid_seg, enforce_boundaries=True, penalty=0.1
        )
        assert conf == 0.80
        assert reason == ""

    def test_no_video_chapter_index_no_penalty(self):
        """No penalty when video segment has no chapter_index."""
        vo_seg = _make_segment("Test voiceover", chapter_index=0)
        vid_seg = _make_segment("Test video")  # No chapter_index
        conf, reason = apply_chapter_boundary_penalty(
            0.80, vo_seg, vid_seg, enforce_boundaries=True, penalty=0.1
        )
        assert conf == 0.80
        assert reason == ""

    def test_negative_chapter_index_no_penalty(self):
        """No penalty when chapter_index is negative (unset)."""
        vo_seg = _make_segment("Test voiceover", chapter_index=-1)
        vid_seg = _make_segment("Test video", chapter_index=1)
        conf, reason = apply_chapter_boundary_penalty(
            0.80, vo_seg, vid_seg, enforce_boundaries=True, penalty=0.1
        )
        assert conf == 0.80
        assert reason == ""

    def test_custom_penalty_amount(self):
        """Penalty amount is configurable."""
        vo_seg = _make_segment("Test voiceover", chapter_index=0)
        vid_seg = _make_segment("Test video", chapter_index=1)
        conf, reason = apply_chapter_boundary_penalty(
            0.80, vo_seg, vid_seg, enforce_boundaries=True, penalty=0.25
        )
        assert conf == 0.55
        assert "-0.250" in reason

    def test_default_penalty_value(self):
        """Default penalty of 0.1 is applied when not specified."""
        vo_seg = _make_segment("Test voiceover", chapter_index=0)
        vid_seg = _make_segment("Test video", chapter_index=1)
        conf, reason = apply_chapter_boundary_penalty(
            0.80, vo_seg, vid_seg, enforce_boundaries=True
        )
        assert abs(conf - 0.70) < 0.001
