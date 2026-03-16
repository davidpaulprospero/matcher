"""
Tests for match quality trend tracking and logging.

US-63-010: Add match quality trend logging across voiceover chunks.
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
    log_chunk_trend,
    log_trend_summary,
    _calculate_trend_slope,
)

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


class TestMatchQualityTrendDataclass:
    """Tests for MatchQualityTrend dataclass."""

    @pytest.mark.fast
    def test_dataclass_has_all_fields(self):
        """MatchQualityTrend should have all required fields."""
        trend = MatchQualityTrend()
        assert hasattr(trend, 'chunk_metrics')
        assert hasattr(trend, 'trend_direction')
        assert hasattr(trend, 'trend_slope')
        assert hasattr(trend, 'overall_avg_confidence')
        assert hasattr(trend, 'low_confidence_threshold')
        assert hasattr(trend, 'total_low_confidence_segments')

    @pytest.mark.fast
    def test_default_values(self):
        """Default values should be sensible defaults."""
        trend = MatchQualityTrend()
        assert trend.chunk_metrics == []
        assert trend.trend_direction == "stable"
        assert trend.trend_slope == 0.0
        assert trend.overall_avg_confidence == 0.0
        assert trend.low_confidence_threshold == 0.2
        assert trend.total_low_confidence_segments == 0


class TestMatchQualityTrendSerialization:
    """Tests for MatchQualityTrend serialization."""

    @pytest.mark.fast
    def test_to_dict_returns_dict(self):
        """to_dict() should return a dictionary."""
        trend = MatchQualityTrend()
        result = trend.to_dict()
        assert isinstance(result, dict)

    @pytest.mark.fast
    def test_to_dict_contains_all_fields(self):
        """to_dict() should include all fields."""
        trend = MatchQualityTrend(
            trend_direction="improving",
            trend_slope=0.05,
            overall_avg_confidence=0.75,
            low_confidence_threshold=0.2,
            total_low_confidence_segments=5,
        )
        trend.chunk_metrics.append(ChunkTrendMetrics(
            chunk_index=0,
            chunk_start_idx=0,
            chunk_end_idx=10,
            segment_count=10,
            rolling_avg_confidence=0.72,
            chunk_avg_confidence=0.70,
            low_confidence_segments=[2, 5],
        ))

        result = trend.to_dict()
        assert result['trend_direction'] == "improving"
        assert result['trend_slope'] == 0.05
        assert result['overall_avg_confidence'] == 0.75
        assert len(result['chunk_metrics']) == 1
        assert result['chunk_metrics'][0]['chunk_index'] == 0
        assert result['chunk_metrics'][0]['low_confidence_segments'] == [2, 5]

    @pytest.mark.fast
    def test_to_dict_json_serializable(self):
        """to_dict() result should be JSON serializable."""
        import json
        trend = MatchQualityTrend(
            trend_direction="degrading",
            trend_slope=-0.03,
            overall_avg_confidence=0.68,
        )
        result = trend.to_dict()
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    @pytest.mark.fast
    def test_from_dict_creates_instance(self):
        """from_dict() should create MatchQualityTrend instance."""
        data = {
            'trend_direction': 'improving',
            'trend_slope': 0.04,
            'overall_avg_confidence': 0.8,
            'chunk_metrics': [],
        }
        trend = MatchQualityTrend.from_dict(data)
        assert isinstance(trend, MatchQualityTrend)
        assert trend.trend_direction == 'improving'

    @pytest.mark.fast
    def test_roundtrip_serialization(self):
        """to_dict() -> from_dict() should preserve values."""
        original = MatchQualityTrend(
            trend_direction="stable",
            trend_slope=0.001,
            overall_avg_confidence=0.82,
            total_low_confidence_segments=3,
        )
        original.chunk_metrics.append(ChunkTrendMetrics(
            chunk_index=0,
            chunk_start_idx=0,
            chunk_end_idx=5,
            segment_count=5,
            rolling_avg_confidence=0.80,
            chunk_avg_confidence=0.78,
            low_confidence_segments=[1, 3],
        ))

        restored = MatchQualityTrend.from_dict(original.to_dict())
        assert restored.trend_direction == original.trend_direction
        assert restored.trend_slope == original.trend_slope
        assert len(restored.chunk_metrics) == len(original.chunk_metrics)


class TestCalculateConfidenceTrend:
    """Tests for calculate_confidence_trend function."""

    def _create_mock_match(self, confidence: float) -> MagicMock:
        """Helper to create a mock match with given confidence."""
        m = MagicMock()
        m.primary_match.confidence = confidence
        return m

    @pytest.mark.fast
    def test_empty_matches_returns_empty_trend(self):
        """Empty matches should return empty trend."""
        trend = calculate_confidence_trend([])
        assert trend.trend_direction == "stable"
        assert trend.chunk_metrics == []
        assert trend.total_low_confidence_segments == 0

    @pytest.mark.fast
    def test_single_match_handled(self):
        """Single match should work without error."""
        matches = [self._create_mock_match(0.8)]
        trend = calculate_confidence_trend(matches)
        assert trend.overall_avg_confidence == 0.8

    @pytest.mark.fast
    def test_calculates_overall_average(self):
        """Should calculate correct overall average."""
        matches = [self._create_mock_match(c) for c in [0.6, 0.7, 0.8, 0.9]]
        trend = calculate_confidence_trend(matches, chunk_count=2)
        assert trend.overall_avg_confidence == pytest.approx(0.75)

    @pytest.mark.fast
    def test_detects_improving_trend(self):
        """Should detect improving trend when confidence increases."""
        # Start low, end high
        confidences = [0.5, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9]
        matches = [self._create_mock_match(c) for c in confidences]
        trend = calculate_confidence_trend(matches, chunk_count=5)
        assert trend.trend_direction == "improving"
        assert trend.trend_slope > 0.02

    @pytest.mark.fast
    def test_detects_degrading_trend(self):
        """Should detect degrading trend when confidence decreases."""
        # Start high, end low
        confidences = [0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5]
        matches = [self._create_mock_match(c) for c in confidences]
        trend = calculate_confidence_trend(matches, chunk_count=5)
        assert trend.trend_direction == "degrading"
        assert trend.trend_slope < -0.02

    @pytest.mark.fast
    def test_detects_stable_trend(self):
        """Should detect stable trend when confidence is consistent."""
        # All around 0.7-0.8
        confidences = [0.7, 0.75, 0.72, 0.78, 0.73, 0.77, 0.74, 0.76, 0.72, 0.78]
        matches = [self._create_mock_match(c) for c in confidences]
        trend = calculate_confidence_trend(matches, chunk_count=5)
        assert trend.trend_direction == "stable"
        assert abs(trend.trend_slope) <= 0.02

    @pytest.mark.fast
    def test_identifies_low_confidence_segments(self):
        """Should identify segments significantly below rolling average."""
        # Most around 0.8, but a few very low
        confidences = [0.8, 0.8, 0.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.25, 0.8]
        matches = [self._create_mock_match(c) for c in confidences]
        trend = calculate_confidence_trend(matches, chunk_count=5, low_conf_threshold=0.2)

        # Segments at indices 2 and 8 should be flagged as low confidence
        all_low_conf_segments = []
        for cm in trend.chunk_metrics:
            all_low_conf_segments.extend(cm.low_confidence_segments)

        assert 2 in all_low_conf_segments
        assert 8 in all_low_conf_segments
        assert trend.total_low_confidence_segments >= 2

    @pytest.mark.fast
    def test_chunk_count_affects_granularity(self):
        """More chunks should create more chunk_metrics."""
        matches = [self._create_mock_match(0.7) for _ in range(100)]

        trend_5 = calculate_confidence_trend(matches, chunk_count=5)
        trend_10 = calculate_confidence_trend(matches, chunk_count=10)

        assert len(trend_5.chunk_metrics) == 5
        assert len(trend_10.chunk_metrics) == 10

    @pytest.mark.fast
    def test_handles_direct_match_objects(self):
        """Should handle Match objects without primary_match."""
        match = MagicMock(spec=['confidence'])
        match.confidence = 0.85
        del match.primary_match

        trend = calculate_confidence_trend([match])
        assert trend.overall_avg_confidence == 0.85

    @pytest.mark.fast
    def test_rolling_average_increases_over_chunks(self):
        """Rolling average should incorporate all previous chunks."""
        # All same confidence - rolling average should stay same
        matches = [self._create_mock_match(0.75) for _ in range(20)]
        trend = calculate_confidence_trend(matches, chunk_count=4)

        # All chunks should have same rolling avg since confidence is uniform
        for cm in trend.chunk_metrics:
            assert cm.rolling_avg_confidence == pytest.approx(0.75, abs=0.01)


class TestTrendSlope:
    """Tests for _calculate_trend_slope helper."""

    @pytest.mark.fast
    def test_empty_values_returns_zero(self):
        """Empty list should return zero slope."""
        assert _calculate_trend_slope([]) == 0.0

    @pytest.mark.fast
    def test_single_value_returns_zero(self):
        """Single value should return zero slope."""
        assert _calculate_trend_slope([0.5]) == 0.0

    @pytest.mark.fast
    def test_increasing_values_positive_slope(self):
        """Increasing values should have positive slope."""
        slope = _calculate_trend_slope([0.1, 0.2, 0.3, 0.4, 0.5])
        assert slope > 0

    @pytest.mark.fast
    def test_decreasing_values_negative_slope(self):
        """Decreasing values should have negative slope."""
        slope = _calculate_trend_slope([0.5, 0.4, 0.3, 0.2, 0.1])
        assert slope < 0

    @pytest.mark.fast
    def test_constant_values_zero_slope(self):
        """Constant values should have zero slope."""
        slope = _calculate_trend_slope([0.5, 0.5, 0.5, 0.5])
        assert slope == 0.0

    @pytest.mark.fast
    def test_linear_increase_correct_slope(self):
        """Linear increase should have correct slope."""
        # y = 0.1x, so slope should be 0.1
        slope = _calculate_trend_slope([0.0, 0.1, 0.2, 0.3, 0.4])
        assert slope == pytest.approx(0.1)


class TestLogChunkTrend:
    """Tests for log_chunk_trend function."""

    @pytest.mark.fast
    def test_logs_chunk_info(self, caplog):
        """Should log chunk information at INFO level."""
        chunk = ChunkTrendMetrics(
            chunk_index=2,
            chunk_start_idx=20,
            chunk_end_idx=30,
            segment_count=10,
            rolling_avg_confidence=0.75,
            chunk_avg_confidence=0.72,
            low_confidence_segments=[],
        )

        with caplog.at_level(logging.INFO, logger='src.matching.metrics'):
            log_chunk_trend(chunk, total_segments=100)

        assert any("Chunk 3" in r.message for r in caplog.records)
        assert any("30%" in r.message or "30.0%" in r.message for r in caplog.records)

    @pytest.mark.fast
    def test_logs_warning_for_low_confidence(self, caplog):
        """Should log warning when chunk has low confidence segments."""
        chunk = ChunkTrendMetrics(
            chunk_index=1,
            chunk_start_idx=10,
            chunk_end_idx=20,
            segment_count=10,
            rolling_avg_confidence=0.80,
            chunk_avg_confidence=0.65,
            low_confidence_segments=[12, 15, 18],
        )

        with caplog.at_level(logging.WARNING, logger='src.matching.metrics'):
            log_chunk_trend(chunk, total_segments=100)

        assert any("Low confidence segments" in r.message for r in caplog.records)


class TestLogTrendSummary:
    """Tests for log_trend_summary function."""

    @pytest.mark.fast
    def test_logs_summary_header(self, caplog):
        """Should log summary header."""
        trend = MatchQualityTrend()

        with caplog.at_level(logging.INFO, logger='src.matching.metrics'):
            log_trend_summary(trend)

        assert any("Match Quality Trend Summary" in r.message for r in caplog.records)

    @pytest.mark.fast
    def test_logs_trend_direction(self, caplog):
        """Should log trend direction."""
        trend = MatchQualityTrend(trend_direction="improving")

        with caplog.at_level(logging.INFO, logger='src.matching.metrics'):
            log_trend_summary(trend)

        assert any("IMPROVING" in r.message for r in caplog.records)

    @pytest.mark.fast
    def test_logs_warning_for_degrading_trend(self, caplog):
        """Should log warning when trend is degrading."""
        trend = MatchQualityTrend(trend_direction="degrading", trend_slope=-0.05)

        with caplog.at_level(logging.WARNING, logger='src.matching.metrics'):
            log_trend_summary(trend)

        assert any("TREND WARNING" in r.message for r in caplog.records)


class TestMatchStageIntegration:
    """Tests for integration with MatchStage."""

    @pytest.mark.fast
    def test_trend_functions_importable_from_metrics(self):
        """Trend functions should be importable from matching.metrics."""
        from src.matching.metrics import (
            MatchQualityTrend,
            ChunkTrendMetrics,
            calculate_confidence_trend,
            log_trend_summary,
        )
        assert MatchQualityTrend is not None
        assert ChunkTrendMetrics is not None
        assert calculate_confidence_trend is not None
        assert log_trend_summary is not None

    @pytest.mark.fast
    def test_match_stage_imports_trend_functions(self):
        """MatchStage should successfully import trend functions."""
        # This validates the import statement added to match.py
        from src.stages.match import MatchStage
        stage = MatchStage()
        assert stage.name == "MATCH"
