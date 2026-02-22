"""
Tests for low-confidence segment reporting with confidence_breakdown.

US-77-012: Report segments below LOW_CONFIDENCE_WARNING_THRESHOLD (0.15)
with top negative adjustments from confidence_breakdown.
"""

import logging
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.matching.tiered_matcher import report_low_confidence_segments
from src.utils import Match, MatchResult, SRTSegment


# ============================================================================
# Helpers
# ============================================================================

def _make_vo_segment(index: int, text: str = "Default voiceover text for testing") -> SRTSegment:
    return SRTSegment(
        index=index,
        start_time=index * 5.0,
        end_time=(index + 1) * 5.0,
        text=text,
        source_file="",
        keywords=[],
        entities=[],
    )


def _make_match_result(
    index: int,
    confidence: float,
    text: str = "Default voiceover text for testing",
    breakdown: List[Dict[str, Any]] = None,
) -> MatchResult:
    vo_seg = _make_vo_segment(index, text)
    vid_seg = SRTSegment(
        index=index,
        start_time=index * 3.0,
        end_time=(index + 1) * 3.0,
        text="Video segment text",
        source_file="video.mp4",
        keywords=[],
    )
    primary = Match(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        video_scene=None,
        confidence=confidence,
        reasoning="test",
    )
    return MatchResult(
        primary_match=primary,
        alternatives=[],
        secondary_matches=[],
        strategy_matches=[],
        has_gap=False,
        gap_reason="",
        confidence_variance=0.0,
        matched_keywords=[],
        confidence_breakdown=breakdown or [],
    )


# ============================================================================
# Test: segment at 0.10 triggers warning
# ============================================================================

class TestLowConfidenceWarningTrigger:

    @pytest.mark.fast
    def test_segment_at_0_10_triggers_warning(self, caplog):
        """Segment with confidence 0.10 (below 0.15) should trigger WARNING."""
        results = [
            _make_match_result(0, 0.10, text="Low confidence voiceover segment text here"),
        ]
        with caplog.at_level(logging.WARNING):
            low = report_low_confidence_segments(results)

        assert len(low) == 1
        assert low[0]['confidence'] == 0.10
        assert "Low-confidence segment #0" in caplog.text
        assert "conf=0.100" in caplog.text

    @pytest.mark.fast
    def test_segment_at_0_20_does_not_trigger(self, caplog):
        """Segment with confidence 0.20 (above 0.15) should NOT trigger warning."""
        results = [
            _make_match_result(0, 0.20),
        ]
        with caplog.at_level(logging.WARNING):
            low = report_low_confidence_segments(results)

        assert len(low) == 0
        assert "Low-confidence segment" not in caplog.text

    @pytest.mark.fast
    def test_segment_exactly_at_threshold_not_triggered(self, caplog):
        """Segment at exactly 0.15 should NOT trigger (uses >= comparison)."""
        results = [
            _make_match_result(0, 0.15),
        ]
        with caplog.at_level(logging.WARNING):
            low = report_low_confidence_segments(results)

        assert len(low) == 0


# ============================================================================
# Test: warning includes segment index, text preview, confidence, top adjustments
# ============================================================================

class TestWarningContent:

    @pytest.mark.fast
    def test_warning_includes_segment_index(self, caplog):
        results = [
            _make_match_result(0, 0.85),  # high - not warned
            _make_match_result(1, 0.08, text="This segment has very low confidence"),
        ]
        with caplog.at_level(logging.WARNING):
            report_low_confidence_segments(results)

        assert "#1" in caplog.text

    @pytest.mark.fast
    def test_warning_includes_text_preview_first_50_chars(self, caplog):
        long_text = "A" * 100
        results = [_make_match_result(0, 0.05, text=long_text)]
        with caplog.at_level(logging.WARNING):
            low = report_low_confidence_segments(results)

        assert low[0]['text_preview'] == "A" * 50
        assert "A" * 50 in caplog.text

    @pytest.mark.fast
    def test_warning_includes_confidence_value(self, caplog):
        results = [_make_match_result(0, 0.07)]
        with caplog.at_level(logging.WARNING):
            report_low_confidence_segments(results)

        assert "conf=0.070" in caplog.text

    @pytest.mark.fast
    def test_warning_includes_top_3_negative_adjustments(self, caplog):
        breakdown = [
            {'component': 'topic_penalty', 'adjustment': -0.05, 'reason': 'mismatch'},
            {'component': 'timing_penalty', 'adjustment': -0.03, 'reason': 'slow'},
            {'component': 'broll_boost', 'adjustment': 0.02, 'reason': 'broll'},
            {'component': 'caption_quality', 'adjustment': -0.01, 'reason': 'poor'},
            {'component': 'consecutive_source_penalty', 'adjustment': -0.08, 'reason': 'repeat'},
        ]
        results = [_make_match_result(0, 0.05, breakdown=breakdown)]
        with caplog.at_level(logging.WARNING):
            low = report_low_confidence_segments(results)

        # Top 3 negative: consecutive_source_penalty (-0.08), topic_penalty (-0.05), timing_penalty (-0.03)
        negs = low[0]['top_negative_adjustments']
        assert len(negs) == 3
        assert negs[0]['component'] == 'consecutive_source_penalty'
        assert negs[1]['component'] == 'topic_penalty'
        assert negs[2]['component'] == 'timing_penalty'

    @pytest.mark.fast
    def test_no_negative_adjustments_shows_none(self, caplog):
        breakdown = [
            {'component': 'broll_boost', 'adjustment': 0.02, 'reason': 'broll'},
        ]
        results = [_make_match_result(0, 0.05, breakdown=breakdown)]
        with caplog.at_level(logging.WARNING):
            report_low_confidence_segments(results)

        assert "none" in caplog.text


# ============================================================================
# Test: >20% threshold triggers summary warning
# ============================================================================

class TestSummaryWarning:

    @pytest.mark.fast
    def test_over_20_percent_triggers_summary(self, caplog):
        """When >20% of segments are low-confidence, summary WARNING fires."""
        # 3 out of 10 = 30% low-confidence
        results = [_make_match_result(i, 0.05) for i in range(3)]
        results += [_make_match_result(i + 3, 0.50) for i in range(7)]
        with caplog.at_level(logging.WARNING):
            report_low_confidence_segments(results)

        assert "Consider enabling iterative matching" in caplog.text
        assert "30%" in caplog.text

    @pytest.mark.fast
    def test_exactly_20_percent_no_summary(self, caplog):
        """Exactly 20% should NOT trigger summary (needs >20%)."""
        # 2 out of 10 = 20% exactly
        results = [_make_match_result(i, 0.05) for i in range(2)]
        results += [_make_match_result(i + 2, 0.50) for i in range(8)]
        with caplog.at_level(logging.WARNING):
            report_low_confidence_segments(results)

        assert "Consider enabling iterative matching" not in caplog.text

    @pytest.mark.fast
    def test_below_20_percent_no_summary(self, caplog):
        """Below 20% should NOT trigger summary."""
        # 1 out of 10 = 10%
        results = [_make_match_result(0, 0.05)]
        results += [_make_match_result(i + 1, 0.50) for i in range(9)]
        with caplog.at_level(logging.WARNING):
            report_low_confidence_segments(results)

        assert "Consider enabling iterative matching" not in caplog.text


# ============================================================================
# Test: empty match list produces no warnings
# ============================================================================

class TestEmptyResults:

    @pytest.mark.fast
    def test_empty_list_no_warnings(self, caplog):
        """Empty match list should produce no warnings."""
        with caplog.at_level(logging.WARNING):
            low = report_low_confidence_segments([])

        assert len(low) == 0
        assert "Low-confidence segment" not in caplog.text
        assert "Consider enabling" not in caplog.text

    @pytest.mark.fast
    def test_all_high_confidence_no_warnings(self, caplog):
        """All high confidence results should produce no warnings."""
        results = [_make_match_result(i, 0.80) for i in range(5)]
        with caplog.at_level(logging.WARNING):
            low = report_low_confidence_segments(results)

        assert len(low) == 0
        assert "Low-confidence segment" not in caplog.text


# ============================================================================
# Test: confidence_breakdown integration
# ============================================================================

class TestBreakdownIntegration:

    @pytest.mark.fast
    def test_empty_breakdown_handled(self, caplog):
        """Segment with empty breakdown should still be reported."""
        results = [_make_match_result(0, 0.05, breakdown=[])]
        with caplog.at_level(logging.WARNING):
            low = report_low_confidence_segments(results)

        assert len(low) == 1
        assert low[0]['top_negative_adjustments'] == []

    @pytest.mark.fast
    def test_fewer_than_3_negatives(self, caplog):
        """When fewer than 3 negative adjustments exist, report all of them."""
        breakdown = [
            {'component': 'topic_penalty', 'adjustment': -0.04, 'reason': 'mismatch'},
        ]
        results = [_make_match_result(0, 0.05, breakdown=breakdown)]
        with caplog.at_level(logging.WARNING):
            low = report_low_confidence_segments(results)

        assert len(low[0]['top_negative_adjustments']) == 1

    @pytest.mark.fast
    def test_returns_list_of_dicts(self):
        """Return value should be list of dicts with expected keys."""
        results = [_make_match_result(0, 0.05)]
        low = report_low_confidence_segments(results)
        assert isinstance(low, list)
        assert len(low) == 1
        entry = low[0]
        assert 'segment_index' in entry
        assert 'text_preview' in entry
        assert 'confidence' in entry
        assert 'top_negative_adjustments' in entry
