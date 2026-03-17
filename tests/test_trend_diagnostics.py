"""
Tests for match quality trend diagnostics.

US-84-010: Add actionable diagnostics to match quality trend degradation detection.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.metrics import (
    ChunkTrendMetrics,
    MatchQualityTrend,
    calculate_confidence_trend,
    populate_chunk_diagnostics,
    _build_degradation_diagnostics,
    log_trend_summary,
)

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


def _create_mock_match(confidence: float) -> MagicMock:
    """Helper to create a mock match with given confidence."""
    m = MagicMock()
    m.primary_match.confidence = confidence
    return m


class TestChunkTrendMetricsDiagnosticFields:
    """Tests for new diagnostic fields on ChunkTrendMetrics."""

    @pytest.mark.fast
    def test_chunk_has_avg_candidates_considered(self):
        """ChunkTrendMetrics should have avg_candidates_considered field."""
        cm = ChunkTrendMetrics(
            chunk_index=0, chunk_start_idx=0, chunk_end_idx=5,
            segment_count=5, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
        )
        assert hasattr(cm, 'avg_candidates_considered')
        assert cm.avg_candidates_considered == 0.0

    @pytest.mark.fast
    def test_chunk_has_dominant_negative_adjustment(self):
        """ChunkTrendMetrics should have dominant_negative_adjustment field."""
        cm = ChunkTrendMetrics(
            chunk_index=0, chunk_start_idx=0, chunk_end_idx=5,
            segment_count=5, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
        )
        assert hasattr(cm, 'dominant_negative_adjustment')
        assert cm.dominant_negative_adjustment == ""

    @pytest.mark.fast
    def test_chunk_has_negative_adjustment_counts(self):
        """ChunkTrendMetrics should have negative_adjustment_counts field."""
        cm = ChunkTrendMetrics(
            chunk_index=0, chunk_start_idx=0, chunk_end_idx=5,
            segment_count=5, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
        )
        assert hasattr(cm, 'negative_adjustment_counts')
        assert cm.negative_adjustment_counts == {}


class TestMatchQualityTrendDiagnosticFields:
    """Tests for new diagnostic fields on MatchQualityTrend."""

    @pytest.mark.fast
    def test_trend_has_cause_summary(self):
        """MatchQualityTrend should have cause_summary field."""
        trend = MatchQualityTrend()
        assert hasattr(trend, 'cause_summary')
        assert trend.cause_summary == ""

    @pytest.mark.fast
    def test_trend_has_trend_diagnostics(self):
        """MatchQualityTrend should have trend_diagnostics list."""
        trend = MatchQualityTrend()
        assert hasattr(trend, 'trend_diagnostics')
        assert trend.trend_diagnostics == []


class TestPopulateChunkDiagnostics:
    """Tests for populate_chunk_diagnostics function."""

    @pytest.mark.fast
    def test_populates_avg_candidates(self):
        """Should populate avg_candidates_considered for each chunk."""
        trend = MatchQualityTrend()
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=3,
                segment_count=3, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
            ),
            ChunkTrendMetrics(
                chunk_index=1, chunk_start_idx=3, chunk_end_idx=6,
                segment_count=3, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
            ),
        ]

        candidates = [10, 15, 20, 5, 8, 12]
        populate_chunk_diagnostics(trend, segment_candidates_considered=candidates)

        assert trend.chunk_metrics[0].avg_candidates_considered == pytest.approx(15.0)
        assert trend.chunk_metrics[1].avg_candidates_considered == pytest.approx(8.333, abs=0.01)

    @pytest.mark.fast
    def test_populates_dominant_negative_adjustment(self):
        """Should identify dominant negative adjustment per chunk."""
        trend = MatchQualityTrend()
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=3,
                segment_count=3, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
            ),
        ]

        adjustments = [
            {'chapter_coherence_penalty': -0.1, 'duration_ratio_penalty': -0.05},
            {'chapter_coherence_penalty': -0.15, 'stutter_penalty': -0.02},
            {'chapter_coherence_penalty': -0.08, 'duration_ratio_penalty': -0.03},
        ]
        populate_chunk_diagnostics(trend, segment_adjustments=adjustments)

        cm = trend.chunk_metrics[0]
        assert cm.dominant_negative_adjustment == 'chapter_coherence_penalty'
        assert cm.negative_adjustment_counts['chapter_coherence_penalty'] == 3
        assert cm.negative_adjustment_counts['duration_ratio_penalty'] == 2

    @pytest.mark.fast
    def test_ignores_positive_adjustments(self):
        """Should only count negative adjustments."""
        trend = MatchQualityTrend()
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=2,
                segment_count=2, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
            ),
        ]

        adjustments = [
            {'bonus_reward': 0.1, 'penalty_a': -0.05},
            {'bonus_reward': 0.2, 'penalty_a': -0.03},
        ]
        populate_chunk_diagnostics(trend, segment_adjustments=adjustments)

        cm = trend.chunk_metrics[0]
        assert 'bonus_reward' not in cm.negative_adjustment_counts
        assert cm.negative_adjustment_counts['penalty_a'] == 2
        assert cm.dominant_negative_adjustment == 'penalty_a'

    @pytest.mark.fast
    def test_no_adjustments_leaves_empty(self):
        """With no adjustments data, diagnostic fields stay at defaults."""
        trend = MatchQualityTrend()
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=3,
                segment_count=3, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
            ),
        ]

        populate_chunk_diagnostics(trend)

        cm = trend.chunk_metrics[0]
        assert cm.avg_candidates_considered == 0.0
        assert cm.dominant_negative_adjustment == ""
        assert cm.negative_adjustment_counts == {}

    @pytest.mark.fast
    def test_no_chunk_metrics_is_noop(self):
        """Empty chunk_metrics should not raise."""
        trend = MatchQualityTrend()
        populate_chunk_diagnostics(trend, segment_adjustments=[])
        assert trend.cause_summary == ""
        assert trend.trend_diagnostics == []


class TestCauseSummaryGeneration:
    """Tests for cause summary generation when trend is DEGRADING."""

    @pytest.mark.fast
    def test_cause_summary_identifies_dominant_penalty(self):
        """When trend is degrading, cause_summary should identify the top penalty type."""
        trend = MatchQualityTrend(
            trend_direction="degrading",
            overall_avg_confidence=0.65,
        )
        # Chunk 0 is above avg (not degrading), chunks 1-2 are below avg (degrading)
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=5,
                segment_count=5, rolling_avg_confidence=0.8, chunk_avg_confidence=0.8,
                negative_adjustment_counts={'chapter_coherence_penalty': 1},
                dominant_negative_adjustment='chapter_coherence_penalty',
            ),
            ChunkTrendMetrics(
                chunk_index=1, chunk_start_idx=5, chunk_end_idx=10,
                segment_count=5, rolling_avg_confidence=0.7, chunk_avg_confidence=0.55,
                negative_adjustment_counts={'chapter_coherence_penalty': 4, 'duration_ratio_penalty': 1},
                dominant_negative_adjustment='chapter_coherence_penalty',
            ),
            ChunkTrendMetrics(
                chunk_index=2, chunk_start_idx=10, chunk_end_idx=15,
                segment_count=5, rolling_avg_confidence=0.65, chunk_avg_confidence=0.50,
                negative_adjustment_counts={'chapter_coherence_penalty': 3, 'stutter_penalty': 2},
                dominant_negative_adjustment='chapter_coherence_penalty',
            ),
        ]

        populate_chunk_diagnostics(trend, segment_adjustments=None)

        # Chunks 1 and 2 are below overall avg (0.65), so degrading
        # chapter_coherence_penalty: 4+3=7, duration_ratio_penalty: 1, stutter_penalty: 2
        # Total negatives in degrading chunks: 10
        # chapter_coherence_penalty: 7/10 = 70%
        assert 'chapter_coherence_penalty' in trend.cause_summary
        assert '70%' in trend.cause_summary
        assert 'chunks 1-2' in trend.cause_summary

    @pytest.mark.fast
    def test_cause_summary_format_matches_spec(self):
        """Cause summary format: 'Degradation in chunks X-Y due to: N% of negative adjustments were Z'."""
        trend = MatchQualityTrend(
            trend_direction="degrading",
            overall_avg_confidence=0.7,
        )
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=8, chunk_start_idx=80, chunk_end_idx=90,
                segment_count=10, rolling_avg_confidence=0.6, chunk_avg_confidence=0.55,
                negative_adjustment_counts={'chapter_coherence_penalty': 6, 'other': 4},
                dominant_negative_adjustment='chapter_coherence_penalty',
            ),
            ChunkTrendMetrics(
                chunk_index=9, chunk_start_idx=90, chunk_end_idx=100,
                segment_count=10, rolling_avg_confidence=0.58, chunk_avg_confidence=0.50,
                negative_adjustment_counts={'chapter_coherence_penalty': 6, 'other': 4},
                dominant_negative_adjustment='chapter_coherence_penalty',
            ),
        ]

        populate_chunk_diagnostics(trend)

        expected_prefix = "Degradation in chunks 8-9 due to: "
        assert trend.cause_summary.startswith(expected_prefix)
        assert "60% of negative adjustments were chapter_coherence_penalty" in trend.cause_summary

    @pytest.mark.fast
    def test_no_cause_summary_when_stable(self):
        """No cause summary when trend is stable."""
        trend = MatchQualityTrend(
            trend_direction="stable",
            overall_avg_confidence=0.7,
        )
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=5,
                segment_count=5, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
                negative_adjustment_counts={'penalty_a': 2},
                dominant_negative_adjustment='penalty_a',
            ),
        ]

        populate_chunk_diagnostics(trend)
        assert trend.cause_summary == ""
        assert trend.trend_diagnostics == []

    @pytest.mark.fast
    def test_no_cause_summary_when_improving(self):
        """No cause summary when trend is improving."""
        trend = MatchQualityTrend(
            trend_direction="improving",
            overall_avg_confidence=0.7,
        )
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=5,
                segment_count=5, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
            ),
        ]

        populate_chunk_diagnostics(trend)
        assert trend.cause_summary == ""

    @pytest.mark.fast
    def test_degrading_no_adjustments_still_generates_summary(self):
        """Degrading trend with no adjustment data should still produce a summary."""
        trend = MatchQualityTrend(
            trend_direction="degrading",
            overall_avg_confidence=0.7,
        )
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=5,
                segment_count=5, rolling_avg_confidence=0.7, chunk_avg_confidence=0.5,
            ),
        ]

        populate_chunk_diagnostics(trend)
        assert 'Degradation' in trend.cause_summary
        assert 'no dominant adjustment identified' in trend.cause_summary

    @pytest.mark.fast
    def test_single_degrading_chunk_shows_single_index(self):
        """Single degrading chunk should show single chunk index, not range."""
        trend = MatchQualityTrend(
            trend_direction="degrading",
            overall_avg_confidence=0.7,
        )
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=5,
                segment_count=5, rolling_avg_confidence=0.8, chunk_avg_confidence=0.8,
            ),
            ChunkTrendMetrics(
                chunk_index=1, chunk_start_idx=5, chunk_end_idx=10,
                segment_count=5, rolling_avg_confidence=0.6, chunk_avg_confidence=0.5,
                negative_adjustment_counts={'penalty_x': 5},
                dominant_negative_adjustment='penalty_x',
            ),
        ]

        populate_chunk_diagnostics(trend)
        assert 'chunks 1' in trend.cause_summary
        # Should not be "chunks 1-1"
        assert '1-1' not in trend.cause_summary


class TestTrendDiagnosticsArray:
    """Tests for trend_diagnostics array in quality report."""

    @pytest.mark.fast
    def test_diagnostics_array_populated_when_degrading(self):
        """trend_diagnostics should be populated with per-chunk detail when degrading."""
        trend = MatchQualityTrend(
            trend_direction="degrading",
            overall_avg_confidence=0.7,
        )
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=5,
                segment_count=5, rolling_avg_confidence=0.7, chunk_avg_confidence=0.6,
                avg_candidates_considered=12.5,
                dominant_negative_adjustment='penalty_a',
            ),
        ]

        populate_chunk_diagnostics(trend)

        assert len(trend.trend_diagnostics) == 1
        diag = trend.trend_diagnostics[0]
        assert diag['chunk_index'] == 0
        assert diag['avg_confidence'] == pytest.approx(0.6)
        assert diag['avg_candidates_considered'] == pytest.approx(12.5)
        assert diag['dominant_negative_adjustment'] == 'penalty_a'

    @pytest.mark.fast
    def test_diagnostics_not_populated_when_stable(self):
        """trend_diagnostics should be empty when trend is stable."""
        trend = MatchQualityTrend(
            trend_direction="stable",
            overall_avg_confidence=0.7,
        )
        trend.chunk_metrics = [
            ChunkTrendMetrics(
                chunk_index=0, chunk_start_idx=0, chunk_end_idx=5,
                segment_count=5, rolling_avg_confidence=0.7, chunk_avg_confidence=0.7,
            ),
        ]

        populate_chunk_diagnostics(trend)
        assert trend.trend_diagnostics == []

    @pytest.mark.fast
    def test_diagnostics_serialized_in_to_dict(self):
        """trend_diagnostics should appear in to_dict() output when present."""
        trend = MatchQualityTrend(
            trend_direction="degrading",
            overall_avg_confidence=0.7,
            cause_summary="Degradation in chunks 2-3 due to: 75% of negative adjustments were penalty_x",
            trend_diagnostics=[
                {'chunk_index': 0, 'avg_confidence': 0.8, 'avg_candidates_considered': 10.0, 'dominant_negative_adjustment': ''},
                {'chunk_index': 1, 'avg_confidence': 0.5, 'avg_candidates_considered': 5.0, 'dominant_negative_adjustment': 'penalty_x'},
            ],
        )

        result = trend.to_dict()
        assert 'cause_summary' in result
        assert 'trend_diagnostics' in result
        assert len(result['trend_diagnostics']) == 2
        assert result['cause_summary'].startswith('Degradation')

    @pytest.mark.fast
    def test_diagnostics_roundtrip(self):
        """trend_diagnostics should survive to_dict -> from_dict roundtrip."""
        original = MatchQualityTrend(
            trend_direction="degrading",
            cause_summary="test cause",
            trend_diagnostics=[
                {'chunk_index': 0, 'avg_confidence': 0.6, 'avg_candidates_considered': 8.0, 'dominant_negative_adjustment': 'pen'},
            ],
        )

        restored = MatchQualityTrend.from_dict(original.to_dict())
        assert restored.cause_summary == "test cause"
        assert len(restored.trend_diagnostics) == 1
        assert restored.trend_diagnostics[0]['chunk_index'] == 0


class TestLogTrendSummaryWithDiagnostics:
    """Tests for log_trend_summary with US-84-010 diagnostics."""

    @pytest.mark.fast
    def test_logs_cause_summary_when_degrading(self, caplog):
        """Should log cause summary when trend is degrading."""
        trend = MatchQualityTrend(
            trend_direction="degrading",
            trend_slope=-0.05,
            cause_summary="Degradation in chunks 8-10 due to: 60% of negative adjustments were chapter_coherence_penalty",
            trend_diagnostics=[
                {'chunk_index': 8, 'avg_confidence': 0.55, 'avg_candidates_considered': 5.0, 'dominant_negative_adjustment': 'chapter_coherence_penalty'},
            ],
        )

        with caplog.at_level(logging.INFO, logger='src.matching.metrics'):
            log_trend_summary(trend)

        log_text = " ".join(r.message for r in caplog.records)
        assert "CAUSE:" in log_text
        assert "chapter_coherence_penalty" in log_text

    @pytest.mark.fast
    def test_logs_per_chunk_diagnostics(self, caplog):
        """Should log per-chunk diagnostics detail for degrading trends."""
        trend = MatchQualityTrend(
            trend_direction="degrading",
            trend_slope=-0.03,
            cause_summary="test cause",
            trend_diagnostics=[
                {'chunk_index': 0, 'avg_confidence': 0.8, 'avg_candidates_considered': 12.0, 'dominant_negative_adjustment': ''},
                {'chunk_index': 1, 'avg_confidence': 0.5, 'avg_candidates_considered': 4.0, 'dominant_negative_adjustment': 'penalty_x'},
            ],
        )

        with caplog.at_level(logging.INFO, logger='src.matching.metrics'):
            log_trend_summary(trend)

        log_text = " ".join(r.message for r in caplog.records)
        assert "Trend diagnostics" in log_text
        assert "penalty_x" in log_text


class TestEndToEndDiagnostics:
    """End-to-end test: calculate_confidence_trend + populate_chunk_diagnostics."""

    @pytest.mark.fast
    def test_correct_cause_for_single_dominant_penalty(self):
        """
        AC5: Unit test verifies correct cause identification when degradation
        is driven by a single dominant penalty type.
        """
        # Create a degrading match sequence: high confidence at start, low at end
        confidences = [0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.4, 0.3, 0.3, 0.3]
        matches = [_create_mock_match(c) for c in confidences]

        trend = calculate_confidence_trend(matches, chunk_count=5)
        assert trend.trend_direction == "degrading"

        # Build per-segment adjustment data:
        # First 5 segments have minor penalties, last 5 have dominant chapter_coherence_penalty
        segment_adjustments = [
            {'duration_ratio_penalty': -0.01},  # seg 0
            {'duration_ratio_penalty': -0.01},  # seg 1
            {},                                  # seg 2
            {},                                  # seg 3
            {'duration_ratio_penalty': -0.02},  # seg 4
            {'chapter_coherence_penalty': -0.2, 'duration_ratio_penalty': -0.05},  # seg 5
            {'chapter_coherence_penalty': -0.25},  # seg 6
            {'chapter_coherence_penalty': -0.3, 'stutter_penalty': -0.05},  # seg 7
            {'chapter_coherence_penalty': -0.35},  # seg 8
            {'chapter_coherence_penalty': -0.3, 'duration_ratio_penalty': -0.1},  # seg 9
        ]
        segment_candidates = [20, 18, 22, 19, 21, 8, 5, 4, 3, 3]

        populate_chunk_diagnostics(
            trend,
            segment_adjustments=segment_adjustments,
            segment_candidates_considered=segment_candidates,
        )

        # Verify cause summary identifies chapter_coherence_penalty as dominant
        assert 'chapter_coherence_penalty' in trend.cause_summary
        assert 'Degradation' in trend.cause_summary
        assert 'due to:' in trend.cause_summary

        # Verify trend_diagnostics array exists
        assert len(trend.trend_diagnostics) > 0

        # Verify per-chunk diagnostics have the required fields
        for diag in trend.trend_diagnostics:
            assert 'avg_confidence' in diag
            assert 'avg_candidates_considered' in diag
            assert 'dominant_negative_adjustment' in diag

        # Verify the later chunks show lower candidates
        later_diags = [d for d in trend.trend_diagnostics if d['chunk_index'] >= 3]
        if later_diags:
            assert later_diags[0]['avg_candidates_considered'] < 10
