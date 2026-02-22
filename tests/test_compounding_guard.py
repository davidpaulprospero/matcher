"""
Tests for confidence adjustment compounding guard (US-84-002).

Verifies that when cumulative negative adjustments exceed a configurable threshold,
remaining negative adjustments are dampened by 50% to prevent cascading false negatives.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field

from src.matching.scoring import MatchScoring
from src.utils import SRTSegment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class MockScoringConfig:
    """Mock scoring config with compounding guard fields."""
    confidence_floor: float = 0.05
    low_confidence_warning_threshold: float = 0.15
    max_cumulative_negative_adjustment: float = -0.30
    compounding_dampening_factor: float = 0.50
    adaptive_confidence_floor_enabled: bool = False
    adaptive_confidence_floor: dict = field(default_factory=lambda: {
        'intro': 0.03, 'conclusion': 0.03, 'body': 0.05,
    })


def _make_config(scoring_config=None):
    """Create a mock config with all required attributes."""
    config = Mock()
    matching = Mock()
    matching.scoring = scoring_config or MockScoringConfig()
    matching.chapter_matching_enabled = False
    matching.topic_mismatch_penalty = 0.15
    matching.caption_quality_adjustment_enabled = False
    matching.apply_timing_penalty = False
    matching.timing_penalty_enabled = False
    matching.broll_boost = 0.0
    matching.language_confidence_penalty = 0.0
    matching.chapter_grouping = None
    matching.context_enrichment = None
    config.matching = matching
    config.global_cache = Mock(current_project_boost=0.0)
    return config


def _make_segment(index=1, text="test text", source_file="test.mp4"):
    seg = SRTSegment(
        index=index,
        start_time=0.0,
        end_time=10.0,
        text=text,
        source_file=source_file,
    )
    seg.keywords = []
    seg.entities = []
    return seg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompoundingGuardActivation:
    """Test that dampening activates when 5+ negative adjustments compound past threshold."""

    def test_dampening_activates_with_many_negative_adjustments(self):
        """When cumulative negatives exceed -0.30, remaining negatives are dampened."""
        config = _make_config()
        scoring = MatchScoring(config)
        vo = _make_segment(text="Tokyo travel adventure")
        vid = _make_segment(text="unrelated content", source_file="vid.mp4")

        # Patch multiple adjustment functions to each return a -0.08 penalty
        # 5 adjustments of -0.08 = -0.40, which exceeds -0.30
        penalties = [
            ('src.matching.scoring.apply_topic_penalty', 'topic penalty'),
            ('src.matching.scoring.apply_broll_boost', 'broll penalty'),
            ('src.matching.scoring.apply_caption_quality_adjustment', 'caption penalty'),
            ('src.matching.scoring.apply_tiered_caption_penalties', None),  # returns list
            ('src.matching.scoring.apply_language_confidence_penalty', 'lang penalty'),
            ('src.matching.scoring.apply_timing_penalty', 'timing penalty'),
            ('src.matching.scoring.apply_current_project_boost', 'project penalty'),
            ('src.matching.scoring.apply_duration_ratio_calibration', 'duration penalty'),
        ]

        def make_penalty_fn(amount, reason_text):
            """Create a function that applies a fixed penalty."""
            def fn(conf, *args, **kwargs):
                return conf + amount, reason_text
            return fn

        def make_tiered_penalty_fn(amount):
            """For tiered penalties that return list of entries."""
            def fn(conf, *args, **kwargs):
                new_conf = conf + amount
                entries = [{'component': 'tiered', 'adjustment': amount, 'reason': 'tiered penalty'}]
                return new_conf, entries
            return fn

        patches = {}
        for path, reason in penalties:
            if reason is None:
                patches[path] = make_tiered_penalty_fn(-0.08)
            else:
                patches[path] = make_penalty_fn(-0.08, reason)

        with patch(penalties[0][0], patches[penalties[0][0]]), \
             patch(penalties[1][0], patches[penalties[1][0]]), \
             patch(penalties[2][0], patches[penalties[2][0]]), \
             patch(penalties[3][0], patches[penalties[3][0]]), \
             patch(penalties[4][0], patches[penalties[4][0]]), \
             patch(penalties[5][0], patches[penalties[5][0]]), \
             patch(penalties[6][0], patches[penalties[6][0]]), \
             patch(penalties[7][0], patches[penalties[7][0]]):

            result_conf, reason_str, breakdown = scoring.apply_all_adjustments(
                confidence=0.90,
                vo_segment=vo,
                video_segment=vid,
            )

        # Find the compounding breakdown entry
        compounding_entry = None
        for entry in breakdown:
            if entry['component'] == 'adjustment_compounding':
                compounding_entry = entry
                break

        assert compounding_entry is not None, "adjustment_compounding entry missing from breakdown"
        assert 'dampening_applied=True' in compounding_entry['reason'], \
            f"Dampening should be active but got: {compounding_entry['reason']}"
        assert 'dampened_count=' in compounding_entry['reason']

        # Extract dampened_count - should be > 0
        import re
        count_match = re.search(r'dampened_count=(\d+)', compounding_entry['reason'])
        assert count_match, "dampened_count not found in reason"
        dampened_count = int(count_match.group(1))
        assert dampened_count > 0, f"Expected dampened_count > 0, got {dampened_count}"

        # The final confidence should be higher than if no dampening applied
        # Without dampening: 0.90 - 8*0.08 = 0.90 - 0.64 = 0.26
        # With dampening: some penalties are halved after threshold
        assert result_conf > 0.26, \
            f"Dampening should preserve some confidence, got {result_conf}"


class TestCompoundingGuardNoActivation:
    """Test that dampening does NOT activate when total negative adjustment is within threshold."""

    def test_no_dampening_within_threshold(self):
        """When cumulative negatives stay within -0.30, no dampening occurs."""
        config = _make_config()
        scoring = MatchScoring(config)
        vo = _make_segment(text="Tokyo travel")
        vid = _make_segment(text="Tokyo footage", source_file="vid.mp4")

        # Only 2 small penalties: -0.10 each = -0.20, within -0.30 threshold
        def small_penalty(conf, *args, **kwargs):
            return conf - 0.10, "small penalty"

        def no_penalty(conf, *args, **kwargs):
            return conf, ""

        def no_tiered_penalty(conf, *args, **kwargs):
            return conf, []

        with patch('src.matching.scoring.apply_topic_penalty', small_penalty), \
             patch('src.matching.scoring.apply_broll_boost', no_penalty), \
             patch('src.matching.scoring.apply_caption_quality_adjustment', small_penalty), \
             patch('src.matching.scoring.apply_tiered_caption_penalties', no_tiered_penalty), \
             patch('src.matching.scoring.apply_language_confidence_penalty', no_penalty), \
             patch('src.matching.scoring.apply_timing_penalty', no_penalty), \
             patch('src.matching.scoring.apply_current_project_boost', no_penalty), \
             patch('src.matching.scoring.apply_duration_ratio_calibration', no_penalty):

            result_conf, reason_str, breakdown = scoring.apply_all_adjustments(
                confidence=0.90,
                vo_segment=vo,
                video_segment=vid,
            )

        # Find the compounding breakdown entry
        compounding_entry = None
        for entry in breakdown:
            if entry['component'] == 'adjustment_compounding':
                compounding_entry = entry
                break

        assert compounding_entry is not None, "adjustment_compounding entry missing from breakdown"
        assert 'dampening_applied=False' in compounding_entry['reason'], \
            f"Dampening should NOT be active but got: {compounding_entry['reason']}"

        # Cumulative negative should be -0.20 (within -0.30 threshold)
        assert compounding_entry['adjustment'] == round(-0.20, 4), \
            f"Expected cumulative_negative=-0.20, got {compounding_entry['adjustment']}"

        # Result should be exactly 0.90 - 0.10 - 0.10 = 0.70 (no dampening)
        assert abs(result_conf - 0.70) < 0.01, \
            f"Expected ~0.70 without dampening, got {result_conf}"


class TestCompoundingGuardConfigurable:
    """Test that the threshold is configurable via MatchingScoringConfig."""

    def test_custom_threshold_triggers_earlier(self):
        """A tighter threshold (-0.15) triggers dampening sooner."""
        sc = MockScoringConfig(max_cumulative_negative_adjustment=-0.15)
        config = _make_config(scoring_config=sc)
        scoring = MatchScoring(config)
        vo = _make_segment(text="test")
        vid = _make_segment(text="test", source_file="v.mp4")

        # 3 penalties of -0.08 = -0.24, exceeds -0.15
        def penalty(conf, *args, **kwargs):
            return conf - 0.08, "penalty"

        def no_op(conf, *args, **kwargs):
            return conf, ""

        def no_tiered(conf, *args, **kwargs):
            return conf, []

        with patch('src.matching.scoring.apply_topic_penalty', penalty), \
             patch('src.matching.scoring.apply_broll_boost', penalty), \
             patch('src.matching.scoring.apply_caption_quality_adjustment', penalty), \
             patch('src.matching.scoring.apply_tiered_caption_penalties', no_tiered), \
             patch('src.matching.scoring.apply_language_confidence_penalty', no_op), \
             patch('src.matching.scoring.apply_timing_penalty', no_op), \
             patch('src.matching.scoring.apply_current_project_boost', no_op), \
             patch('src.matching.scoring.apply_duration_ratio_calibration', no_op):

            result_conf, _, breakdown = scoring.apply_all_adjustments(
                confidence=0.90,
                vo_segment=vo,
                video_segment=vid,
            )

        compounding_entry = next(
            e for e in breakdown if e['component'] == 'adjustment_compounding'
        )
        assert 'dampening_applied=True' in compounding_entry['reason']
        assert 'threshold=-0.15' in compounding_entry['reason']

    def test_custom_dampening_factor(self):
        """A different dampening factor (0.25) reduces penalties more aggressively."""
        sc = MockScoringConfig(
            max_cumulative_negative_adjustment=-0.10,
            compounding_dampening_factor=0.25,
        )
        config = _make_config(scoring_config=sc)
        scoring = MatchScoring(config)
        vo = _make_segment(text="test")
        vid = _make_segment(text="test", source_file="v.mp4")

        call_count = [0]

        def sequential_penalty(conf, *args, **kwargs):
            call_count[0] += 1
            return conf - 0.10, f"penalty_{call_count[0]}"

        def no_op(conf, *args, **kwargs):
            return conf, ""

        def no_tiered(conf, *args, **kwargs):
            return conf, []

        # First penalty: -0.10 (crosses -0.10 threshold)
        # Second penalty: -0.10 * 0.25 = -0.025 (dampened)
        with patch('src.matching.scoring.apply_topic_penalty', sequential_penalty), \
             patch('src.matching.scoring.apply_broll_boost', sequential_penalty), \
             patch('src.matching.scoring.apply_caption_quality_adjustment', no_op), \
             patch('src.matching.scoring.apply_tiered_caption_penalties', no_tiered), \
             patch('src.matching.scoring.apply_language_confidence_penalty', no_op), \
             patch('src.matching.scoring.apply_timing_penalty', no_op), \
             patch('src.matching.scoring.apply_current_project_boost', no_op), \
             patch('src.matching.scoring.apply_duration_ratio_calibration', no_op):

            result_conf, _, breakdown = scoring.apply_all_adjustments(
                confidence=0.90,
                vo_segment=vo,
                video_segment=vid,
            )

        # First penalty: full -0.10 -> 0.80 (cumulative = -0.10, triggers dampening)
        # Second penalty: -0.10 * 0.25 = -0.025 -> 0.775
        # So result should be ~0.775 (not 0.70 without dampening)
        assert result_conf > 0.70, \
            f"0.25 dampening factor should keep result > 0.70, got {result_conf}"


class TestCompoundingGuardBreakdown:
    """Test that the adjustment_compounding breakdown entry is correct."""

    def test_breakdown_entry_always_present(self):
        """The adjustment_compounding entry appears even with zero penalties."""
        config = _make_config()
        scoring = MatchScoring(config)
        vo = _make_segment(text="test")
        vid = _make_segment(text="test", source_file="v.mp4")

        def no_op(conf, *args, **kwargs):
            return conf, ""

        def no_tiered(conf, *args, **kwargs):
            return conf, []

        with patch('src.matching.scoring.apply_topic_penalty', no_op), \
             patch('src.matching.scoring.apply_broll_boost', no_op), \
             patch('src.matching.scoring.apply_caption_quality_adjustment', no_op), \
             patch('src.matching.scoring.apply_tiered_caption_penalties', no_tiered), \
             patch('src.matching.scoring.apply_language_confidence_penalty', no_op), \
             patch('src.matching.scoring.apply_timing_penalty', no_op), \
             patch('src.matching.scoring.apply_current_project_boost', no_op), \
             patch('src.matching.scoring.apply_duration_ratio_calibration', no_op):

            result_conf, _, breakdown = scoring.apply_all_adjustments(
                confidence=0.90,
                vo_segment=vo,
                video_segment=vid,
            )

        compounding_entry = next(
            (e for e in breakdown if e['component'] == 'adjustment_compounding'),
            None,
        )
        assert compounding_entry is not None
        assert compounding_entry['adjustment'] == 0.0
        assert 'dampening_applied=False' in compounding_entry['reason']
        assert 'dampened_count=0' in compounding_entry['reason']


class TestCompoundingGuardPositiveAdjustments:
    """Verify positive adjustments (boosts) don't affect the compounding guard."""

    def test_positive_adjustments_not_tracked(self):
        """Boosts should not count toward cumulative negative total."""
        config = _make_config()
        scoring = MatchScoring(config)
        vo = _make_segment(text="test")
        vid = _make_segment(text="test", source_file="v.mp4")

        def boost(conf, *args, **kwargs):
            return conf + 0.05, "boost"

        def penalty(conf, *args, **kwargs):
            return conf - 0.10, "penalty"

        def no_tiered(conf, *args, **kwargs):
            return conf, []

        # Mix of boosts and penalties: total negative = -0.20, within -0.30
        with patch('src.matching.scoring.apply_topic_penalty', penalty), \
             patch('src.matching.scoring.apply_broll_boost', boost), \
             patch('src.matching.scoring.apply_caption_quality_adjustment', penalty), \
             patch('src.matching.scoring.apply_tiered_caption_penalties', no_tiered), \
             patch('src.matching.scoring.apply_language_confidence_penalty', boost), \
             patch('src.matching.scoring.apply_timing_penalty', boost), \
             patch('src.matching.scoring.apply_current_project_boost', boost), \
             patch('src.matching.scoring.apply_duration_ratio_calibration', boost):

            result_conf, _, breakdown = scoring.apply_all_adjustments(
                confidence=0.90,
                vo_segment=vo,
                video_segment=vid,
            )

        compounding_entry = next(
            e for e in breakdown if e['component'] == 'adjustment_compounding'
        )
        # Only negatives counted: -0.10 + -0.10 = -0.20
        assert compounding_entry['adjustment'] == round(-0.20, 4), \
            f"Expected -0.20 cumulative negative (boosts excluded), got {compounding_entry['adjustment']}"
        assert 'dampening_applied=False' in compounding_entry['reason']
