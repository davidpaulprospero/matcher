"""
Tests for scoring adjustment audit summary (US-77-008).

Verifies:
- Summary includes all active adjustment names
- Unused adjustments are flagged
- Average magnitudes are computed correctly
- Top-3 most impactful adjustments are ranked by average absolute magnitude
- Edge cases: empty results, no breakdowns, single segment
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.matching.tiered_matcher import compute_scoring_audit_summary, log_scoring_audit_summary


# ---------------------------------------------------------------------------
# Lightweight stubs (avoid importing full MatchResult to keep test isolated)
# ---------------------------------------------------------------------------
@dataclass
class StubMatch:
    confidence: float = 0.8


@dataclass
class StubMatchResult:
    primary_match: Optional[StubMatch] = None
    confidence_breakdown: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestComputeScoringAuditSummary:
    """Unit tests for compute_scoring_audit_summary."""

    @pytest.mark.fast
    def test_summary_includes_all_active_adjustment_names(self):
        """Active adjustments appear in adjustment_counts."""
        results = [
            StubMatchResult(
                primary_match=StubMatch(confidence=0.85),
                confidence_breakdown=[
                    {'component': 'title_relevance', 'adjustment': 0.05, 'reason': 'keyword overlap'},
                    {'component': 'broll_boost', 'adjustment': 0.03, 'reason': 'broll detected'},
                    {'component': 'topic_penalty', 'adjustment': -0.10, 'reason': 'topic mismatch'},
                ],
            ),
            StubMatchResult(
                primary_match=StubMatch(confidence=0.72),
                confidence_breakdown=[
                    {'component': 'title_relevance', 'adjustment': 0.02, 'reason': 'partial overlap'},
                    {'component': 'chapter_topic_match', 'adjustment': 0.08, 'reason': 'chapter match'},
                ],
            ),
        ]

        summary = compute_scoring_audit_summary(results)

        assert summary['total_segments'] == 2
        counts = summary['adjustment_counts']
        assert 'title_relevance' in counts
        assert 'broll_boost' in counts
        assert 'topic_penalty' in counts
        assert 'chapter_topic_match' in counts
        # title_relevance fired twice (once per result)
        assert counts['title_relevance'] == 2
        assert counts['broll_boost'] == 1

    @pytest.mark.fast
    def test_unused_adjustments_are_flagged(self):
        """Adjustments that never fire appear in unused_adjustments."""
        results = [
            StubMatchResult(
                primary_match=StubMatch(confidence=0.90),
                confidence_breakdown=[
                    {'component': 'title_relevance', 'adjustment': 0.04, 'reason': 'ok'},
                ],
            ),
        ]

        summary = compute_scoring_audit_summary(results)

        unused = summary['unused_adjustments']
        # Only title_relevance fired, so all others should be unused
        assert 'broll_boost' in unused
        assert 'topic_penalty' in unused
        assert 'chapter_topic_match' in unused
        assert 'consecutive_source_penalty' in unused
        assert 'semantic_coherence' in unused
        assert 'temporal_coherence' in unused
        # title_relevance should NOT be in unused
        assert 'title_relevance' not in unused

    @pytest.mark.fast
    def test_average_magnitudes_computed_correctly(self):
        """avg_magnitudes is mean of absolute values per adjustment."""
        results = [
            StubMatchResult(
                primary_match=StubMatch(confidence=0.80),
                confidence_breakdown=[
                    {'component': 'topic_penalty', 'adjustment': -0.10, 'reason': 'a'},
                    {'component': 'broll_boost', 'adjustment': 0.06, 'reason': 'b'},
                ],
            ),
            StubMatchResult(
                primary_match=StubMatch(confidence=0.75),
                confidence_breakdown=[
                    {'component': 'topic_penalty', 'adjustment': -0.04, 'reason': 'c'},
                ],
            ),
        ]

        summary = compute_scoring_audit_summary(results)

        # topic_penalty: abs(-0.10) + abs(-0.04) / 2 = 0.07
        assert abs(summary['avg_magnitudes']['topic_penalty'] - 0.07) < 1e-6
        # broll_boost: abs(0.06) / 1 = 0.06
        assert abs(summary['avg_magnitudes']['broll_boost'] - 0.06) < 1e-6

    @pytest.mark.fast
    def test_top_3_most_impactful_ranked_by_avg_magnitude(self):
        """top_3_impactful sorted descending by average absolute magnitude."""
        results = [
            StubMatchResult(
                primary_match=StubMatch(confidence=0.80),
                confidence_breakdown=[
                    {'component': 'topic_penalty', 'adjustment': -0.15, 'reason': 'a'},
                    {'component': 'title_relevance', 'adjustment': 0.02, 'reason': 'b'},
                    {'component': 'broll_boost', 'adjustment': 0.08, 'reason': 'c'},
                    {'component': 'chapter_topic_match', 'adjustment': 0.05, 'reason': 'd'},
                ],
            ),
        ]

        summary = compute_scoring_audit_summary(results)

        top3 = summary['top_3_impactful']
        assert len(top3) == 3
        # Ranked: topic_penalty (0.15), broll_boost (0.08), chapter_topic_match (0.05)
        assert top3[0][0] == 'topic_penalty'
        assert top3[1][0] == 'broll_boost'
        assert top3[2][0] == 'chapter_topic_match'

    @pytest.mark.fast
    def test_empty_results(self):
        """Empty input produces zero totals and all adjustments are unused."""
        summary = compute_scoring_audit_summary([])

        assert summary['total_segments'] == 0
        assert summary['avg_confidence'] == 0.0
        assert summary['adjustment_counts'] == {}
        assert summary['avg_magnitudes'] == {}
        assert summary['top_3_impactful'] == []
        assert len(summary['unused_adjustments']) > 0  # All known are unused

    @pytest.mark.fast
    def test_no_breakdowns(self):
        """Results with no confidence_breakdown still count segments."""
        results = [
            StubMatchResult(
                primary_match=StubMatch(confidence=0.70),
                confidence_breakdown=[],
            ),
            StubMatchResult(
                primary_match=StubMatch(confidence=0.60),
                confidence_breakdown=[],
            ),
        ]

        summary = compute_scoring_audit_summary(results)

        assert summary['total_segments'] == 2
        assert abs(summary['avg_confidence'] - 0.65) < 1e-6
        assert summary['adjustment_counts'] == {}
        assert len(summary['unused_adjustments']) > 0

    @pytest.mark.fast
    def test_single_segment_with_all_adjustments(self):
        """Single segment that triggers many adjustments."""
        breakdown = [
            {'component': 'topic_penalty', 'adjustment': -0.05, 'reason': 'a'},
            {'component': 'broll_boost', 'adjustment': 0.03, 'reason': 'b'},
            {'component': 'title_relevance', 'adjustment': 0.04, 'reason': 'c'},
            {'component': 'description_relevance', 'adjustment': 0.02, 'reason': 'd'},
            {'component': 'tag_keyword_boost', 'adjustment': 0.06, 'reason': 'e'},
            {'component': 'chapter_topic_match', 'adjustment': 0.07, 'reason': 'f'},
            {'component': 'semantic_coherence', 'adjustment': 0.01, 'reason': 'g'},
        ]
        results = [
            StubMatchResult(
                primary_match=StubMatch(confidence=0.88),
                confidence_breakdown=breakdown,
            ),
        ]

        summary = compute_scoring_audit_summary(results)

        assert summary['total_segments'] == 1
        assert len(summary['adjustment_counts']) == 7
        # Unused should NOT contain any of the 7 that fired
        for entry in breakdown:
            assert entry['component'] not in summary['unused_adjustments']

    @pytest.mark.fast
    def test_avg_confidence_calculation(self):
        """Average confidence is computed from primary_match.confidence values."""
        results = [
            StubMatchResult(primary_match=StubMatch(confidence=0.90), confidence_breakdown=[]),
            StubMatchResult(primary_match=StubMatch(confidence=0.80), confidence_breakdown=[]),
            StubMatchResult(primary_match=StubMatch(confidence=0.70), confidence_breakdown=[]),
        ]

        summary = compute_scoring_audit_summary(results)

        assert summary['total_segments'] == 3
        assert abs(summary['avg_confidence'] - 0.80) < 1e-6

    @pytest.mark.fast
    def test_none_results_skipped(self):
        """None entries in results list are gracefully skipped."""
        results = [
            None,
            StubMatchResult(primary_match=StubMatch(confidence=0.85), confidence_breakdown=[
                {'component': 'title_relevance', 'adjustment': 0.03, 'reason': 'ok'},
            ]),
            None,
        ]

        summary = compute_scoring_audit_summary(results)

        assert summary['total_segments'] == 1
        assert summary['adjustment_counts']['title_relevance'] == 1


class TestLogScoringAuditSummary:
    """Tests for log_scoring_audit_summary (smoke test - just ensure it doesn't crash)."""

    @pytest.mark.fast
    def test_log_summary_does_not_crash(self, caplog):
        """Logging the summary with typical data doesn't raise."""
        summary = {
            'total_segments': 10,
            'avg_confidence': 0.82,
            'adjustment_counts': {'title_relevance': 8, 'topic_penalty': 5},
            'avg_magnitudes': {'title_relevance': 0.04, 'topic_penalty': 0.12},
            'top_3_impactful': [('topic_penalty', 0.12), ('title_relevance', 0.04)],
            'unused_adjustments': ['broll_boost', 'caption_quality'],
        }

        import logging
        with caplog.at_level(logging.INFO):
            log_scoring_audit_summary(summary)

        # Verify key parts of the log output
        log_text = caplog.text
        assert 'SCORING ADJUSTMENT AUDIT SUMMARY' in log_text
        assert 'Total segments matched: 10' in log_text
        assert 'title_relevance' in log_text
        assert 'topic_penalty' in log_text
        assert 'Unused adjustments' in log_text

    @pytest.mark.fast
    def test_log_summary_empty_data(self, caplog):
        """Logging an empty summary doesn't crash."""
        summary = {
            'total_segments': 0,
            'avg_confidence': 0.0,
            'adjustment_counts': {},
            'avg_magnitudes': {},
            'top_3_impactful': [],
            'unused_adjustments': ['topic_penalty', 'broll_boost'],
        }

        import logging
        with caplog.at_level(logging.INFO):
            log_scoring_audit_summary(summary)

        log_text = caplog.text
        assert 'Total segments matched: 0' in log_text

    @pytest.mark.fast
    def test_log_summary_all_adjustments_active(self, caplog):
        """When all adjustments fired, reports 'All known adjustments fired'."""
        summary = {
            'total_segments': 5,
            'avg_confidence': 0.75,
            'adjustment_counts': {'a': 5},
            'avg_magnitudes': {'a': 0.05},
            'top_3_impactful': [('a', 0.05)],
            'unused_adjustments': [],  # All fired
        }

        import logging
        with caplog.at_level(logging.INFO):
            log_scoring_audit_summary(summary)

        assert 'All known adjustments fired at least once' in caplog.text
