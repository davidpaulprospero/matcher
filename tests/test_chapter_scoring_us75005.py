"""
Tests for apply_chapter_topic_match and apply_chapter_source_consistency
standalone functions (US-75-005).

Verifies that:
- chapter_topic_match boosts/penalizes based on keyword overlap between
  voiceover text and video chapter title
- chapter_source_consistency applies +0.03 boost when same source reused
  within the same voiceover chapter
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import (
    apply_chapter_topic_match,
    apply_chapter_source_consistency,
    apply_chapter_coherence_penalty,
    _compute_chapter_confidence_weight,
)
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


def _make_match(video_source_file: str, vo_chapter_index=None):
    """Create a mock Match object with video_segment and voiceover_segment."""
    match = MagicMock()
    match.video_segment = MagicMock()
    match.video_segment.source_file = video_source_file
    match.voiceover_segment = MagicMock()
    match.voiceover_segment.chapter_index = vo_chapter_index
    return match


class TestApplyChapterTopicMatch:
    """Tests for the standalone apply_chapter_topic_match function."""

    def test_disabled_when_chapter_matching_off(self):
        """No adjustment when chapter_matching_enabled is False."""
        seg = _make_segment("Ancient Roman architecture", chapter_index=0)
        conf, reason = apply_chapter_topic_match(0.70, seg, "Roman History", chapter_matching_enabled=False)
        assert conf == 0.70
        assert reason == ""

    def test_no_chapter_index_no_adjustment(self):
        """No adjustment when voiceover segment has no chapter_index."""
        seg = _make_segment("Ancient Roman architecture")
        conf, reason = apply_chapter_topic_match(0.70, seg, "Roman History", chapter_matching_enabled=True)
        assert conf == 0.70
        assert reason == ""

    def test_no_chapter_title_no_adjustment(self):
        """No adjustment when chapter_title is None."""
        seg = _make_segment("Ancient Roman architecture", chapter_index=0)
        conf, reason = apply_chapter_topic_match(0.70, seg, None, chapter_matching_enabled=True)
        assert conf == 0.70
        assert reason == ""

    def test_strong_match_boost(self):
        """3+ keyword overlap produces +0.10 strong match boost."""
        seg = _make_segment("Roman architecture and ancient history exploration", chapter_index=0)
        # Keywords: roman, architecture, ancient, history, exploration
        chapter_title = "Roman Architecture and Ancient History"
        # chapter keywords: roman, architecture, ancient, history => 4 overlap
        conf, reason = apply_chapter_topic_match(0.70, seg, chapter_title, chapter_matching_enabled=True)
        assert conf == pytest.approx(0.80, abs=0.01)
        assert "strong match" in reason
        assert "+0.1" in reason

    def test_partial_match_boost(self):
        """1-2 keyword overlap produces +0.05 partial match boost."""
        seg = _make_segment("The modern technology revolution changes everything", chapter_index=1)
        chapter_title = "Technology Innovations"
        # overlap: "technology" => 1 keyword
        conf, reason = apply_chapter_topic_match(0.70, seg, chapter_title, chapter_matching_enabled=True)
        assert conf == pytest.approx(0.75, abs=0.01)
        assert "partial match" in reason
        assert "+0.05" in reason

    def test_mismatch_penalty(self):
        """0 keyword overlap produces -0.05 mismatch penalty."""
        seg = _make_segment("The weather forecast predicts heavy rainfall", chapter_index=2)
        chapter_title = "Cooking Recipes for Beginners"
        conf, reason = apply_chapter_topic_match(0.70, seg, chapter_title, chapter_matching_enabled=True)
        assert conf == pytest.approx(0.65, abs=0.01)
        assert "mismatch" in reason

    def test_chapter_topic_appears_in_breakdown(self):
        """Reason string contains the adjustment label 'chapter topic'."""
        seg = _make_segment("Ancient Roman temples and columns", chapter_index=0)
        conf, reason = apply_chapter_topic_match(0.70, seg, "Roman Temples", chapter_matching_enabled=True)
        assert "chapter topic" in reason

    def test_graduated_values(self):
        """Boost values are graduated: partial < strong."""
        seg_partial = _make_segment("Modern technology overview", chapter_index=0)
        seg_strong = _make_segment("Roman architecture ancient history temples", chapter_index=0)

        conf_partial, _ = apply_chapter_topic_match(0.70, seg_partial, "Technology Overview", chapter_matching_enabled=True)
        conf_strong, _ = apply_chapter_topic_match(0.70, seg_strong, "Roman Architecture Ancient History Temples", chapter_matching_enabled=True)

        assert conf_strong > conf_partial


class TestApplyChapterSourceConsistency:
    """Tests for the standalone apply_chapter_source_consistency function."""

    def test_disabled_when_chapter_matching_off(self):
        """No adjustment when chapter_matching_enabled is False."""
        video_seg = _make_segment("some content", source_file="vid_A")
        vo_seg = _make_segment("voiceover text", chapter_index=0)
        prev_match = _make_match("vid_A", vo_chapter_index=0)

        conf, reason = apply_chapter_source_consistency(
            0.70, video_seg, vo_seg, [prev_match], chapter_matching_enabled=False
        )
        assert conf == 0.70
        assert reason == ""

    def test_no_chapter_index_no_boost(self):
        """No boost when voiceover segment has no chapter_index."""
        video_seg = _make_segment("some content", source_file="vid_A")
        vo_seg = _make_segment("voiceover text")  # no chapter_index
        prev_match = _make_match("vid_A", vo_chapter_index=0)

        conf, reason = apply_chapter_source_consistency(
            0.70, video_seg, vo_seg, [prev_match], chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_same_source_same_chapter_gets_boost(self):
        """Same video source reused within same chapter gets +0.03 boost."""
        video_seg = _make_segment("some content", source_file="vid_A")
        vo_seg = _make_segment("voiceover text", chapter_index=2)
        prev_match = _make_match("vid_A", vo_chapter_index=2)

        conf, reason = apply_chapter_source_consistency(
            0.70, video_seg, vo_seg, [prev_match], chapter_matching_enabled=True
        )
        assert conf == pytest.approx(0.73, abs=0.01)
        assert "chapter_source_consistency" in reason
        assert "+0.03" in reason
        assert "chapter 2" in reason

    def test_different_source_no_boost(self):
        """Different video source in same chapter gets no boost."""
        video_seg = _make_segment("some content", source_file="vid_B")
        vo_seg = _make_segment("voiceover text", chapter_index=2)
        prev_match = _make_match("vid_A", vo_chapter_index=2)

        conf, reason = apply_chapter_source_consistency(
            0.70, video_seg, vo_seg, [prev_match], chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_same_source_different_chapter_no_boost(self):
        """Same video source but different chapters gets no boost."""
        video_seg = _make_segment("some content", source_file="vid_A")
        vo_seg = _make_segment("voiceover text", chapter_index=3)
        prev_match = _make_match("vid_A", vo_chapter_index=2)

        conf, reason = apply_chapter_source_consistency(
            0.70, video_seg, vo_seg, [prev_match], chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_no_recent_matches_no_boost(self):
        """Empty recent matches produces no boost."""
        video_seg = _make_segment("some content", source_file="vid_A")
        vo_seg = _make_segment("voiceover text", chapter_index=0)

        conf, reason = apply_chapter_source_consistency(
            0.70, video_seg, vo_seg, [], chapter_matching_enabled=True
        )
        assert conf == 0.70
        assert reason == ""

    def test_consecutive_matches_within_chapter(self):
        """Consecutive matches from same source within a chapter get consistency boost."""
        video_seg = _make_segment("another segment from vid_A", source_file="vid_A")
        vo_seg = _make_segment("continuing voiceover", chapter_index=1)
        prev_match = _make_match("vid_A", vo_chapter_index=1)

        conf, reason = apply_chapter_source_consistency(
            0.80, video_seg, vo_seg, [prev_match], chapter_matching_enabled=True
        )
        assert conf > 0.80
        assert "chapter_source_consistency" in reason


class TestChapterConfidenceWeight:
    """Tests for chapter boundary confidence weighting in scoring (US-76-012)."""

    def test_weight_high_confidence(self):
        """Confidence >= 0.8 produces weight 1.0."""
        assert _compute_chapter_confidence_weight(0.8) == pytest.approx(1.0)
        assert _compute_chapter_confidence_weight(0.9) == pytest.approx(1.0)
        assert _compute_chapter_confidence_weight(1.0) == pytest.approx(1.0)

    def test_weight_low_confidence(self):
        """Confidence <= 0.5 produces weight 0.5."""
        assert _compute_chapter_confidence_weight(0.5) == pytest.approx(0.5)
        assert _compute_chapter_confidence_weight(0.4) == pytest.approx(0.5)
        assert _compute_chapter_confidence_weight(0.0) == pytest.approx(0.5)

    def test_weight_mid_confidence_interpolation(self):
        """Confidence 0.65 (midpoint) produces ~0.75 weight via linear interpolation."""
        weight = _compute_chapter_confidence_weight(0.65)
        assert weight == pytest.approx(0.75, abs=0.01)

    def test_topic_match_high_confidence_full_boost(self):
        """Chapter with confidence 0.9 applies full chapter_topic_match boost."""
        seg = _make_segment("Roman architecture ancient history exploration", chapter_index=0)
        chapter_title = "Roman Architecture and Ancient History"
        conf, reason = apply_chapter_topic_match(
            0.70, seg, chapter_title,
            chapter_matching_enabled=True,
            chapter_confidence=0.9,
        )
        # High confidence => weight 1.0 => full +0.10 boost
        assert conf == pytest.approx(0.80, abs=0.01)
        assert "strong match" in reason

    def test_topic_match_low_confidence_half_boost(self):
        """Chapter with confidence 0.4 applies 50% of chapter_topic_match boost."""
        seg = _make_segment("Roman architecture ancient history exploration", chapter_index=0)
        chapter_title = "Roman Architecture and Ancient History"
        conf, reason = apply_chapter_topic_match(
            0.70, seg, chapter_title,
            chapter_matching_enabled=True,
            chapter_confidence=0.4,
        )
        # Low confidence => weight 0.5 => +0.05 boost (50% of 0.10)
        assert conf == pytest.approx(0.75, abs=0.01)
        assert "strong match" in reason

    def test_topic_match_mid_confidence_interpolated_boost(self):
        """Chapter with confidence 0.65 applies ~75% of chapter_topic_match boost."""
        seg = _make_segment("Roman architecture ancient history exploration", chapter_index=0)
        chapter_title = "Roman Architecture and Ancient History"
        conf, reason = apply_chapter_topic_match(
            0.70, seg, chapter_title,
            chapter_matching_enabled=True,
            chapter_confidence=0.65,
        )
        # Mid confidence 0.65 => weight ~0.75 => +0.075 boost (75% of 0.10)
        assert conf == pytest.approx(0.775, abs=0.01)
        assert "strong match" in reason

    def test_source_consistency_weighted_by_confidence(self):
        """Source consistency boost is scaled by chapter confidence."""
        video_seg = _make_segment("some content", source_file="vid_A")
        vo_seg = _make_segment("voiceover text", chapter_index=2)
        prev_match = _make_match("vid_A", vo_chapter_index=2)

        # High confidence: full 0.03 boost
        conf_high, _ = apply_chapter_source_consistency(
            0.70, video_seg, vo_seg, [prev_match],
            chapter_matching_enabled=True, chapter_confidence=0.9,
        )
        assert conf_high == pytest.approx(0.73, abs=0.01)

        # Low confidence: 50% of 0.03 = 0.015
        conf_low, _ = apply_chapter_source_consistency(
            0.70, video_seg, vo_seg, [prev_match],
            chapter_matching_enabled=True, chapter_confidence=0.4,
        )
        assert conf_low == pytest.approx(0.715, abs=0.01)

    def test_coherence_penalty_weighted_by_confidence(self):
        """Coherence penalty is scaled by chapter confidence."""
        vo_seg = _make_segment("voiceover text", chapter_index=0)
        # 7 sources = 2 excess over threshold(5) => raw penalty = 2 * -0.03 = -0.06
        sources = {0: {f"vid_{i}" for i in range(7)}}

        # High confidence: full -0.06 penalty
        conf_high, _ = apply_chapter_coherence_penalty(
            0.70, vo_seg, chapter_source_counts=sources,
            chapter_matching_enabled=True, chapter_confidence=0.9,
        )
        assert conf_high == pytest.approx(0.64, abs=0.01)

        # Low confidence: 50% of -0.06 = -0.03
        conf_low, _ = apply_chapter_coherence_penalty(
            0.70, vo_seg, chapter_source_counts=sources,
            chapter_matching_enabled=True, chapter_confidence=0.4,
        )
        assert conf_low == pytest.approx(0.67, abs=0.01)

    def test_default_confidence_preserves_original_behavior(self):
        """Default chapter_confidence=1.0 preserves original unweighted behavior."""
        seg = _make_segment("Roman architecture ancient history exploration", chapter_index=0)
        chapter_title = "Roman Architecture and Ancient History"

        # No chapter_confidence arg (default 1.0)
        conf_default, _ = apply_chapter_topic_match(
            0.70, seg, chapter_title, chapter_matching_enabled=True,
        )
        # Explicit 1.0
        conf_explicit, _ = apply_chapter_topic_match(
            0.70, seg, chapter_title, chapter_matching_enabled=True, chapter_confidence=1.0,
        )
        assert conf_default == conf_explicit
