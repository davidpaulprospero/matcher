"""
Tests for apply_chapter_coherence_penalty and apply_cross_chapter_relevance_boost
standalone functions (US-75-006).

Verifies that:
- chapter_coherence_penalty fires when >5 unique video sources used in a voiceover chapter
- cross_chapter_relevance boost is applied when relevance_matrix entry is high
- Both adjustments appear in confidence_breakdown with descriptive reasons
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_chapter_coherence_penalty, apply_cross_chapter_relevance_boost
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


class TestApplyChapterCoherencePenalty:
    """Tests for the standalone apply_chapter_coherence_penalty function."""

    def test_disabled_when_chapter_matching_off(self):
        """No adjustment when chapter_matching_enabled is False."""
        seg = _make_segment("Some text", chapter_index=0)
        sources = {0: {"vid_a", "vid_b", "vid_c", "vid_d", "vid_e", "vid_f"}}
        conf, reason = apply_chapter_coherence_penalty(
            0.70, seg, chapter_source_counts=sources, chapter_matching_enabled=False
        )
        assert conf == 0.70
        assert reason == ""

    def test_no_chapter_index_no_penalty(self):
        """No penalty when voiceover segment has no chapter_index."""
        seg = _make_segment("Some text")
        sources = {0: {"vid_a", "vid_b", "vid_c", "vid_d", "vid_e", "vid_f"}}
        conf, reason = apply_chapter_coherence_penalty(
            0.70, seg, chapter_source_counts=sources, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_no_source_counts_no_penalty(self):
        """No penalty when chapter_source_counts is None."""
        seg = _make_segment("Some text", chapter_index=0)
        conf, reason = apply_chapter_coherence_penalty(
            0.70, seg, chapter_source_counts=None, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_under_threshold_no_penalty(self):
        """No penalty when source count <= 5 (threshold)."""
        seg = _make_segment("Some text", chapter_index=0)
        sources = {0: {"vid_a", "vid_b", "vid_c", "vid_d", "vid_e"}}  # exactly 5
        conf, reason = apply_chapter_coherence_penalty(
            0.70, seg, chapter_source_counts=sources, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_six_sources_gets_penalty(self):
        """6 sources (1 over threshold) should get -0.03 penalty."""
        seg = _make_segment("Some text", chapter_index=0)
        sources = {0: {"vid_a", "vid_b", "vid_c", "vid_d", "vid_e", "vid_f"}}
        conf, reason = apply_chapter_coherence_penalty(
            0.70, seg, chapter_source_counts=sources, chapter_matching_enabled=True
        )
        # 1 excess * -0.03 = -0.03
        assert conf == pytest.approx(0.67, abs=0.01)
        assert "chapter_coherence_penalty" in reason
        assert "6 sources" in reason

    def test_eight_sources_gets_capped_penalty(self):
        """8 sources (3 over threshold) should be capped at -0.10."""
        seg = _make_segment("Some text", chapter_index=0)
        sources = {0: {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"}}
        conf, reason = apply_chapter_coherence_penalty(
            0.70, seg, chapter_source_counts=sources, chapter_matching_enabled=True
        )
        # 3 excess * -0.03 = -0.09, cap is -0.10, max(-0.10, -0.09) = -0.09
        assert conf == pytest.approx(0.61, abs=0.01)
        assert "chapter_coherence_penalty" in reason
        assert "8 sources" in reason

    def test_ten_sources_hits_cap(self):
        """10 sources (5 over threshold): 5 * -0.03 = -0.15, capped at -0.10."""
        seg = _make_segment("Some text", chapter_index=0)
        sources = {0: {f"v{i}" for i in range(10)}}
        conf, reason = apply_chapter_coherence_penalty(
            0.70, seg, chapter_source_counts=sources, chapter_matching_enabled=True
        )
        # 5 excess * -0.03 = -0.15, cap is -0.10, max(-0.10, -0.15) = -0.10
        assert conf == pytest.approx(0.60, abs=0.01)
        assert "chapter_coherence_penalty" in reason

    def test_negative_chapter_index_no_penalty(self):
        """No penalty when chapter_index is -1."""
        seg = _make_segment("Some text", chapter_index=-1)
        sources = {-1: {f"v{i}" for i in range(10)}}
        conf, reason = apply_chapter_coherence_penalty(
            0.70, seg, chapter_source_counts=sources, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_different_chapter_no_penalty(self):
        """No penalty when segment's chapter doesn't have excessive sources."""
        seg = _make_segment("Some text", chapter_index=1)
        # Chapter 0 has many sources, but segment is in chapter 1
        sources = {0: {f"v{i}" for i in range(10)}, 1: {"vid_a", "vid_b"}}
        conf, reason = apply_chapter_coherence_penalty(
            0.70, seg, chapter_source_counts=sources, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""


class TestApplyCrossChapterRelevanceBoost:
    """Tests for the standalone apply_cross_chapter_relevance_boost function."""

    def test_disabled_when_chapter_matching_off(self):
        """No boost when chapter_matching_enabled is False."""
        vo_seg = _make_segment("Text", chapter_index=0)
        vid_seg = _make_segment("Video text", chapter_index=1)
        vid_seg.source_file = "vid_a"
        matrix = [[0.0, 0.8], [0.5, 0.0]]
        conf, reason = apply_cross_chapter_relevance_boost(
            0.70, vo_seg, vid_seg, relevance_matrix=matrix, chapter_matching_enabled=False
        )
        assert conf == 0.70
        assert reason == ""

    def test_no_vo_chapter_index_no_boost(self):
        """No boost when voiceover segment has no chapter_index."""
        vo_seg = _make_segment("Text")
        vid_seg = _make_segment("Video text", chapter_index=1)
        vid_seg.source_file = "vid_a"
        matrix = [[0.0, 0.8], [0.5, 0.0]]
        conf, reason = apply_cross_chapter_relevance_boost(
            0.70, vo_seg, vid_seg, relevance_matrix=matrix, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_no_vid_chapter_index_no_boost(self):
        """No boost when video segment has no chapter_index."""
        vo_seg = _make_segment("Text", chapter_index=0)
        vid_seg = _make_segment("Video text")
        vid_seg.source_file = "vid_a"
        matrix = [[0.0, 0.8], [0.5, 0.0]]
        conf, reason = apply_cross_chapter_relevance_boost(
            0.70, vo_seg, vid_seg, relevance_matrix=matrix, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_no_matrix_no_boost(self):
        """No boost when relevance_matrix is None."""
        vo_seg = _make_segment("Text", chapter_index=0)
        vid_seg = _make_segment("Video text", chapter_index=1)
        vid_seg.source_file = "vid_a"
        conf, reason = apply_cross_chapter_relevance_boost(
            0.70, vo_seg, vid_seg, relevance_matrix=None, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_zero_relevance_no_boost(self):
        """No boost when relevance score is 0."""
        vo_seg = _make_segment("Text", chapter_index=0)
        vid_seg = _make_segment("Video text", chapter_index=1)
        vid_seg.source_file = "vid_a"
        matrix = [[0.0, 0.0], [0.0, 0.0]]
        conf, reason = apply_cross_chapter_relevance_boost(
            0.70, vo_seg, vid_seg, relevance_matrix=matrix, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_high_relevance_gets_boost(self):
        """High relevance score (0.8) with weight 0.1 produces +0.08 boost."""
        vo_seg = _make_segment("Text", chapter_index=0)
        vid_seg = _make_segment("Video text", chapter_index=1)
        vid_seg.source_file = "vid_a"
        matrix = [[0.0, 0.8], [0.5, 0.0]]
        conf, reason = apply_cross_chapter_relevance_boost(
            0.70, vo_seg, vid_seg, relevance_matrix=matrix, chapter_matching_enabled=True
        )
        # 0.8 * 0.1 = 0.08
        assert conf == pytest.approx(0.78, abs=0.01)
        assert "cross_chapter_relevance" in reason
        assert "vo_ch=0" in reason
        assert "vid_ch=1" in reason

    def test_moderate_relevance_boost(self):
        """Moderate relevance score (0.5) with weight 0.1 produces +0.05 boost."""
        vo_seg = _make_segment("Text", chapter_index=1)
        vid_seg = _make_segment("Video text", chapter_index=0)
        vid_seg.source_file = "vid_a"
        matrix = [[0.0, 0.8], [0.5, 0.0]]
        conf, reason = apply_cross_chapter_relevance_boost(
            0.70, vo_seg, vid_seg, relevance_matrix=matrix, chapter_matching_enabled=True
        )
        # 0.5 * 0.1 = 0.05
        assert conf == pytest.approx(0.75, abs=0.01)
        assert "cross_chapter_relevance" in reason

    def test_out_of_bounds_vo_chapter_no_boost(self):
        """No boost when vo chapter index exceeds matrix dimensions."""
        vo_seg = _make_segment("Text", chapter_index=5)
        vid_seg = _make_segment("Video text", chapter_index=0)
        vid_seg.source_file = "vid_a"
        matrix = [[0.0, 0.8], [0.5, 0.0]]
        conf, reason = apply_cross_chapter_relevance_boost(
            0.70, vo_seg, vid_seg, relevance_matrix=matrix, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_out_of_bounds_vid_chapter_no_boost(self):
        """No boost when video chapter index exceeds matrix dimensions."""
        vo_seg = _make_segment("Text", chapter_index=0)
        vid_seg = _make_segment("Video text", chapter_index=5)
        vid_seg.source_file = "vid_a"
        matrix = [[0.0, 0.8], [0.5, 0.0]]
        conf, reason = apply_cross_chapter_relevance_boost(
            0.70, vo_seg, vid_seg, relevance_matrix=matrix, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_negative_chapter_index_no_boost(self):
        """No boost when chapter_index is negative."""
        vo_seg = _make_segment("Text", chapter_index=-1)
        vid_seg = _make_segment("Video text", chapter_index=0)
        vid_seg.source_file = "vid_a"
        matrix = [[0.0, 0.8], [0.5, 0.0]]
        conf, reason = apply_cross_chapter_relevance_boost(
            0.70, vo_seg, vid_seg, relevance_matrix=matrix, chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_capped_at_1(self):
        """Boost should not push confidence above 1.0."""
        vo_seg = _make_segment("Text", chapter_index=0)
        vid_seg = _make_segment("Video text", chapter_index=1)
        vid_seg.source_file = "vid_a"
        matrix = [[0.0, 1.0], [0.5, 0.0]]
        conf, reason = apply_cross_chapter_relevance_boost(
            0.98, vo_seg, vid_seg, relevance_matrix=matrix, chapter_matching_enabled=True
        )
        # 1.0 * 0.1 = 0.1, 0.98 + 0.1 = 1.08, but capped at 1.0
        assert conf <= 1.0
        assert "cross_chapter_relevance" in reason
