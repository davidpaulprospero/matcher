"""
Tests for US-57-005: create_gap_match factory function.

Verifies:
- create_gap_match returns a Match with confidence=0.0 and match_type='gap'
- The gap match's video_segment is the same object as voiceover_segment (self-reference)
"""

import pytest

from src.utils import SRTSegment, Match
from src.matching.tiered_matcher import create_gap_match


def _make_segment(text: str = "Test voiceover segment") -> SRTSegment:
    """Create a minimal SRTSegment for testing."""
    return SRTSegment(
        index=0,
        start_time=0.0,
        end_time=5.0,
        text=text,
        source_file="video1.mp4",
    )


class TestCreateGapMatch:
    """Tests for the create_gap_match factory function."""

    def test_returns_match_with_zero_confidence_and_gap_type(self):
        """AC4: create_gap_match returns a Match with confidence=0.0 and match_type='gap'."""
        vo_seg = _make_segment()
        result = create_gap_match(vo_seg, "No candidates available")

        assert isinstance(result, Match)
        assert result.confidence == 0.0
        assert result.match_type == 'gap'

    def test_video_segment_is_same_object_as_voiceover_segment(self):
        """AC5: The gap match's video_segment is the same object as voiceover_segment."""
        vo_seg = _make_segment()
        result = create_gap_match(vo_seg, "All candidates filtered")

        assert result.video_segment is result.voiceover_segment
        assert result.video_segment is vo_seg

    def test_reasoning_is_set(self):
        """The reasoning field reflects the provided reason string."""
        vo_seg = _make_segment()
        reason = "No valid candidates after filtering"
        result = create_gap_match(vo_seg, reason)

        assert result.reasoning == reason

    def test_video_scene_is_none(self):
        """Gap matches have no associated video scene."""
        vo_seg = _make_segment()
        result = create_gap_match(vo_seg, "test reason")

        assert result.video_scene is None

    def test_optional_fields_have_defaults(self):
        """Gap matches have default values for optional quality indicators."""
        vo_seg = _make_segment()
        result = create_gap_match(vo_seg, "test reason")

        assert result.is_keyword_match is False
        assert result.is_visual_match is False
        assert result.embedding_similarity == 0.0
        assert result.clip_reuse_count == 0

    def test_gap_match_serialization_includes_match_type(self):
        """The match_type='gap' field survives to_dict serialization."""
        vo_seg = _make_segment()
        result = create_gap_match(vo_seg, "No candidates")
        d = result.to_dict()

        assert d['match_type'] == 'gap'
        assert d['confidence'] == 0.0

    def test_gap_match_roundtrip_via_dict(self):
        """match_type survives to_dict -> from_dict roundtrip."""
        vo_seg = _make_segment()
        result = create_gap_match(vo_seg, "No candidates")
        d = result.to_dict()
        restored = Match.from_dict(d)

        assert restored.match_type == 'gap'
        assert restored.confidence == 0.0
