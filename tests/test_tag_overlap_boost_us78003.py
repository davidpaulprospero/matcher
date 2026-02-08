"""
Tests for apply_tag_overlap_boost standalone function and MatchScoring integration (US-78-003).

Verifies graduated boost values (1 tag -> +0.02, 2 -> +0.04, 3+ -> +0.06)
and that zero-overlap produces no adjustment.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_tag_overlap_boost, MatchScoring
from src.utils import SRTSegment


def _make_segment(text: str) -> SRTSegment:
    """Create a minimal SRTSegment with given text."""
    return SRTSegment(
        index=1,
        start_time=0.0,
        end_time=5.0,
        text=text,
    )


class TestApplyTagOverlapBoostStandalone:
    """Tests for the standalone apply_tag_overlap_boost function."""

    def test_one_tag_match_gives_002(self):
        """Single tag match produces +0.02 boost."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        tags = ["roman", "cooking", "gardening"]
        confidence = 0.70
        new_conf, reason = apply_tag_overlap_boost(confidence, seg, tags)
        assert new_conf == pytest.approx(0.72, abs=1e-6)
        assert "tag overlap boost +0.02" in reason
        assert "1 tag" in reason

    def test_two_tag_matches_gives_004(self):
        """Two tag matches produce +0.04 boost."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        tags = ["roman", "architecture", "gardening"]
        confidence = 0.70
        new_conf, reason = apply_tag_overlap_boost(confidence, seg, tags)
        assert new_conf == pytest.approx(0.74, abs=1e-6)
        assert "tag overlap boost +0.04" in reason
        assert "2 tags" in reason

    def test_three_plus_tag_matches_gives_006(self):
        """Three or more tag matches produce +0.06 boost (capped)."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        tags = ["roman", "architecture", "ancient", "history", "travel"]
        confidence = 0.70
        new_conf, reason = apply_tag_overlap_boost(confidence, seg, tags)
        assert new_conf == pytest.approx(0.76, abs=1e-6)
        assert "tag overlap boost +0.06" in reason
        assert "tags" in reason

    def test_zero_overlap_no_adjustment(self):
        """No tag overlap produces no adjustment."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        tags = ["cooking", "gardening", "pets"]
        confidence = 0.70
        new_conf, reason = apply_tag_overlap_boost(confidence, seg, tags)
        assert new_conf == confidence
        assert reason == ""

    def test_none_tags_no_adjustment(self):
        """None video_tags produces no adjustment."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        new_conf, reason = apply_tag_overlap_boost(0.70, seg, None)
        assert new_conf == 0.70
        assert reason == ""

    def test_empty_tags_no_adjustment(self):
        """Empty tag list produces no adjustment."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        new_conf, reason = apply_tag_overlap_boost(0.70, seg, [])
        assert new_conf == 0.70
        assert reason == ""

    def test_short_tags_filtered_out(self):
        """Tags shorter than 3 chars are filtered out."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        tags = ["ab", "cd"]  # too short
        new_conf, reason = apply_tag_overlap_boost(0.70, seg, tags)
        assert new_conf == 0.70
        assert reason == ""

    def test_confidence_capped_at_1(self):
        """Boost does not exceed 1.0."""
        seg = _make_segment("The ancient Roman architecture still stands today")
        tags = ["roman", "architecture", "ancient"]
        confidence = 0.98
        new_conf, reason = apply_tag_overlap_boost(confidence, seg, tags)
        assert new_conf == 1.0

    def test_four_matches_still_006(self):
        """Four matches still produce +0.06 (same as 3+)."""
        seg = _make_segment("Modern technology and science drive innovation forward rapidly")
        tags = ["modern", "technology", "science", "innovation"]
        confidence = 0.60
        new_conf, reason = apply_tag_overlap_boost(confidence, seg, tags)
        assert new_conf == pytest.approx(0.66, abs=1e-6)
        assert "+0.06" in reason


class TestTagOverlapInApplyAllAdjustments:
    """Tests for tag_overlap integration in MatchScoring.apply_all_adjustments."""

    def _make_config(self, extract_video_tags: bool = True):
        """Create a minimal config mock with context_enrichment."""
        config = Mock()
        matching = Mock()
        matching.multimodal_enabled = True
        matching.multimodal_weights = None
        matching.pool_normalization_enabled = True
        matching.chapter_matching_enabled = False
        matching.topic_mismatch_penalty = 0.15
        matching.broll_boost = 0.1
        matching.caption_quality_adjustment_enabled = True
        matching.caption_quality_high_boost = 0.05
        matching.caption_quality_low_penalty = 0.1
        matching.apply_timing_penalty = True
        matching.skip_llm_threshold = 0.85
        matching.language_confidence_penalty = 0.0

        ce = Mock()
        ce.extract_video_tags = extract_video_tags
        matching.context_enrichment = ce

        scoring = Mock()
        scoring.confidence_floor = 0.1
        scoring.low_confidence_warning_threshold = 0.15
        matching.scoring = scoring

        config.matching = matching
        config.global_cache = Mock()
        config.global_cache.current_project_boost = 0.1
        return config

    def test_tag_overlap_in_breakdown(self):
        """tag_overlap entry appears in confidence_breakdown."""
        scoring = MatchScoring(self._make_config())
        vo_seg = _make_segment("The ancient Roman architecture still stands today")
        vid_seg = _make_segment("Roman ruins from the ancient empire")
        tags = ["roman", "architecture", "ancient"]

        _, _, breakdown = scoring.apply_all_adjustments(
            0.70, vo_seg, vid_seg, video_tags=tags,
        )
        tag_overlap_entries = [b for b in breakdown if b['component'] == 'tag_overlap']
        assert len(tag_overlap_entries) == 1
        entry = tag_overlap_entries[0]
        assert 'adjustment' in entry
        assert 'reason' in entry
        assert entry['adjustment'] > 0

    def test_tag_overlap_gated_by_config_false(self):
        """When extract_video_tags is False, no tag_overlap entry in breakdown."""
        scoring = MatchScoring(self._make_config(extract_video_tags=False))
        vo_seg = _make_segment("The ancient Roman architecture still stands today")
        vid_seg = _make_segment("Roman ruins from the ancient empire")
        tags = ["roman", "architecture", "ancient"]

        _, _, breakdown = scoring.apply_all_adjustments(
            0.70, vo_seg, vid_seg, video_tags=tags,
        )
        tag_overlap_entries = [b for b in breakdown if b['component'] == 'tag_overlap']
        assert len(tag_overlap_entries) == 0

    def test_tag_overlap_gated_by_config_true(self):
        """When extract_video_tags is True (default), tag_overlap applies."""
        scoring = MatchScoring(self._make_config(extract_video_tags=True))
        vo_seg = _make_segment("Modern technology and science drive innovation")
        vid_seg = _make_segment("Tech and science news")
        tags = ["technology", "science"]

        _, _, breakdown = scoring.apply_all_adjustments(
            0.70, vo_seg, vid_seg, video_tags=tags,
        )
        tag_overlap_entries = [b for b in breakdown if b['component'] == 'tag_overlap']
        assert len(tag_overlap_entries) == 1
        assert tag_overlap_entries[0]['adjustment'] == pytest.approx(0.04, abs=1e-3)
