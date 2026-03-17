"""
Tests for apply_tag_keyword_boost standalone function (US-75-004).

Verifies that voiceover keywords overlapping with video tags produce
a positive confidence boost, and empty/None tags produce zero adjustment.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_tag_keyword_boost
from src.utils import SRTSegment


def _make_segment(text: str) -> SRTSegment:
    """Create a minimal SRTSegment with given text."""
    return SRTSegment(
        index=1,
        start_time=0.0,
        end_time=5.0,
        text=text,
    )


class TestApplyTagKeywordBoost:
    """Tests for the standalone apply_tag_keyword_boost function."""

    def test_matching_tags_give_positive_boost(self):
        """Candidate with tags matching voiceover keywords gets positive boost."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        tags = ["roman", "architecture", "history", "travel"]
        confidence = 0.70
        new_conf, reason = apply_tag_keyword_boost(confidence, seg, tags)
        assert new_conf > confidence
        assert "tag keyword boost" in reason
        assert "roman" in reason or "architecture" in reason

    def test_multiple_matching_tags_higher_boost(self):
        """More matching tags produce a larger boost (up to cap)."""
        seg = _make_segment("Modern technology and science drive innovation forward")
        tags_1 = ["technology"]
        tags_3 = ["technology", "science", "innovation"]
        confidence = 0.60

        conf_1, _ = apply_tag_keyword_boost(confidence, seg, tags_1)
        conf_3, _ = apply_tag_keyword_boost(confidence, seg, tags_3)
        assert conf_3 > conf_1

    def test_boost_capped_at_maximum(self):
        """Boost is capped at 0.08 regardless of match count."""
        seg = _make_segment("alpha bravo charlie delta echo foxtrot golf hotel india juliet")
        tags = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
        confidence = 0.50
        new_conf, reason = apply_tag_keyword_boost(confidence, seg, tags)
        boost = new_conf - confidence
        assert boost <= 0.08 + 1e-9  # Allow float precision

    def test_no_tags_zero_boost(self):
        """Empty tags list produces zero adjustment."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        confidence = 0.70
        new_conf, reason = apply_tag_keyword_boost(confidence, seg, [])
        assert new_conf == confidence
        assert reason == ""

    def test_none_tags_zero_boost(self):
        """None tags produces zero adjustment."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        confidence = 0.70
        new_conf, reason = apply_tag_keyword_boost(confidence, seg, None)
        assert new_conf == confidence
        assert reason == ""

    def test_no_overlap_zero_boost(self):
        """Tags that don't overlap with voiceover keywords produce zero boost."""
        seg = _make_segment("The weather forecast predicts heavy rainfall")
        tags = ["cooking", "recipes", "kitchen"]
        confidence = 0.70
        new_conf, reason = apply_tag_keyword_boost(confidence, seg, tags)
        assert new_conf == confidence
        assert reason == ""

    def test_short_tags_ignored(self):
        """Tags shorter than 3 characters are filtered out."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        tags = ["an", "to", "it"]  # All too short
        confidence = 0.70
        new_conf, reason = apply_tag_keyword_boost(confidence, seg, tags)
        assert new_conf == confidence
        assert reason == ""

    def test_case_insensitive_matching(self):
        """Tag matching is case-insensitive."""
        seg = _make_segment("Technology drives modern innovation")
        tags = ["TECHNOLOGY", "Innovation"]
        confidence = 0.60
        new_conf, reason = apply_tag_keyword_boost(confidence, seg, tags)
        assert new_conf > confidence
        assert "tag keyword boost" in reason

    def test_confidence_capped_at_1(self):
        """Boost doesn't push confidence above 1.0."""
        seg = _make_segment("Technology science innovation research")
        tags = ["technology", "science", "innovation", "research"]
        confidence = 0.98
        new_conf, _ = apply_tag_keyword_boost(confidence, seg, tags)
        assert new_conf <= 1.0
