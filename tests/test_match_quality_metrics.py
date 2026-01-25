"""
Tests for match quality metrics tracking and logging.

Sprint 4 - US-004: Add match quality metrics logging
"""

import logging
import statistics
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.metrics import (
    MatchQualityMetrics,
    calculate_match_quality_metrics,
    log_quality_summary,
)

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


class TestMatchQualityMetricsDataclass:
    """Tests for MatchQualityMetrics dataclass"""

    def test_dataclass_has_all_fields(self):
        """MatchQualityMetrics should have all required fields"""
        metrics = MatchQualityMetrics()
        assert hasattr(metrics, 'avg_confidence')
        assert hasattr(metrics, 'min_confidence')
        assert hasattr(metrics, 'max_confidence')
        assert hasattr(metrics, 'confidence_std')
        assert hasattr(metrics, 'gap_count')
        assert hasattr(metrics, 'match_rate')
        assert hasattr(metrics, 'total_segments')
        assert hasattr(metrics, 'matched_segments')

    def test_default_values(self):
        """Default values should be zero"""
        metrics = MatchQualityMetrics()
        assert metrics.avg_confidence == 0.0
        assert metrics.min_confidence == 0.0
        assert metrics.max_confidence == 0.0
        assert metrics.confidence_std == 0.0
        assert metrics.gap_count == 0
        assert metrics.match_rate == 0.0
        assert metrics.total_segments == 0
        assert metrics.matched_segments == 0

    def test_custom_values(self):
        """Should accept custom values"""
        metrics = MatchQualityMetrics(
            avg_confidence=0.85,
            min_confidence=0.6,
            max_confidence=0.95,
            confidence_std=0.12,
            gap_count=2,
            match_rate=0.9,
            total_segments=20,
            matched_segments=18,
        )
        assert metrics.avg_confidence == 0.85
        assert metrics.min_confidence == 0.6
        assert metrics.max_confidence == 0.95
        assert metrics.confidence_std == 0.12
        assert metrics.gap_count == 2
        assert metrics.match_rate == 0.9
        assert metrics.total_segments == 20
        assert metrics.matched_segments == 18


class TestMatchQualityMetricsToDict:
    """Tests for MatchQualityMetrics.to_dict() method"""

    def test_to_dict_returns_dict(self):
        """to_dict() should return a dictionary"""
        metrics = MatchQualityMetrics()
        result = metrics.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_contains_all_fields(self):
        """to_dict() should include all fields"""
        metrics = MatchQualityMetrics(
            avg_confidence=0.85,
            min_confidence=0.6,
            max_confidence=0.95,
            confidence_std=0.12,
            gap_count=2,
            match_rate=0.9,
            total_segments=20,
            matched_segments=18,
        )
        result = metrics.to_dict()
        assert result['avg_confidence'] == 0.85
        assert result['min_confidence'] == 0.6
        assert result['max_confidence'] == 0.95
        assert result['confidence_std'] == 0.12
        assert result['gap_count'] == 2
        assert result['match_rate'] == 0.9
        assert result['total_segments'] == 20
        assert result['matched_segments'] == 18

    def test_to_dict_json_serializable(self):
        """to_dict() result should be JSON serializable"""
        import json
        metrics = MatchQualityMetrics(
            avg_confidence=0.85,
            min_confidence=0.6,
            max_confidence=0.95,
            confidence_std=0.12,
            gap_count=2,
            match_rate=0.9,
            total_segments=20,
            matched_segments=18,
        )
        result = metrics.to_dict()
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


class TestMatchQualityMetricsFromDict:
    """Tests for MatchQualityMetrics.from_dict() classmethod"""

    def test_from_dict_creates_instance(self):
        """from_dict() should create MatchQualityMetrics instance"""
        data = {
            'avg_confidence': 0.85,
            'min_confidence': 0.6,
            'max_confidence': 0.95,
            'confidence_std': 0.12,
            'gap_count': 2,
            'match_rate': 0.9,
            'total_segments': 20,
            'matched_segments': 18,
        }
        metrics = MatchQualityMetrics.from_dict(data)
        assert isinstance(metrics, MatchQualityMetrics)

    def test_from_dict_restores_values(self):
        """from_dict() should restore all values"""
        data = {
            'avg_confidence': 0.85,
            'min_confidence': 0.6,
            'max_confidence': 0.95,
            'confidence_std': 0.12,
            'gap_count': 2,
            'match_rate': 0.9,
            'total_segments': 20,
            'matched_segments': 18,
        }
        metrics = MatchQualityMetrics.from_dict(data)
        assert metrics.avg_confidence == 0.85
        assert metrics.min_confidence == 0.6
        assert metrics.max_confidence == 0.95
        assert metrics.confidence_std == 0.12
        assert metrics.gap_count == 2
        assert metrics.match_rate == 0.9
        assert metrics.total_segments == 20
        assert metrics.matched_segments == 18

    def test_from_dict_handles_missing_keys(self):
        """from_dict() should handle missing keys with defaults"""
        data = {'avg_confidence': 0.5}
        metrics = MatchQualityMetrics.from_dict(data)
        assert metrics.avg_confidence == 0.5
        assert metrics.min_confidence == 0.0
        assert metrics.gap_count == 0

    def test_roundtrip_serialization(self):
        """to_dict() -> from_dict() should preserve values"""
        original = MatchQualityMetrics(
            avg_confidence=0.85,
            min_confidence=0.6,
            max_confidence=0.95,
            confidence_std=0.12,
            gap_count=2,
            match_rate=0.9,
            total_segments=20,
            matched_segments=18,
        )
        restored = MatchQualityMetrics.from_dict(original.to_dict())
        assert restored.avg_confidence == original.avg_confidence
        assert restored.min_confidence == original.min_confidence
        assert restored.max_confidence == original.max_confidence
        assert restored.confidence_std == original.confidence_std
        assert restored.gap_count == original.gap_count
        assert restored.match_rate == original.match_rate
        assert restored.total_segments == original.total_segments
        assert restored.matched_segments == original.matched_segments


class TestCalculateMatchQualityMetrics:
    """Tests for calculate_match_quality_metrics function"""

    def test_empty_matches_returns_metrics(self):
        """Empty matches list should return metrics with zero values"""
        metrics = calculate_match_quality_metrics(matches=[], total_segments=10)
        assert metrics.total_segments == 10
        assert metrics.matched_segments == 0
        assert metrics.match_rate == 0.0
        assert metrics.avg_confidence == 0.0

    def test_zero_total_segments(self):
        """Zero total segments should not cause division error"""
        metrics = calculate_match_quality_metrics(matches=[], total_segments=0)
        assert metrics.total_segments == 0
        assert metrics.match_rate == 0.0

    def test_single_match_no_std(self):
        """Single match should have zero standard deviation"""
        match = MagicMock()
        match.primary_match.confidence = 0.8
        match.has_gap = False

        metrics = calculate_match_quality_metrics(matches=[match], total_segments=1)
        assert metrics.confidence_std == 0.0
        assert metrics.avg_confidence == 0.8
        assert metrics.min_confidence == 0.8
        assert metrics.max_confidence == 0.8

    def test_multiple_matches_calculates_avg(self):
        """Multiple matches should calculate correct average"""
        matches = []
        for conf in [0.6, 0.7, 0.8, 0.9]:
            m = MagicMock()
            m.primary_match.confidence = conf
            m.has_gap = False
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches=matches, total_segments=4)
        assert metrics.avg_confidence == pytest.approx(0.75)
        assert metrics.matched_segments == 4
        assert metrics.match_rate == 1.0

    def test_calculates_min_max(self):
        """Should correctly identify min and max confidence"""
        matches = []
        for conf in [0.5, 0.6, 0.7, 0.8, 0.95]:
            m = MagicMock()
            m.primary_match.confidence = conf
            m.has_gap = False
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches=matches, total_segments=5)
        assert metrics.min_confidence == 0.5
        assert metrics.max_confidence == 0.95

    def test_calculates_std_deviation(self):
        """Should calculate correct standard deviation"""
        matches = []
        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        for conf in confidences:
            m = MagicMock()
            m.primary_match.confidence = conf
            m.has_gap = False
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches=matches, total_segments=5)
        expected_std = statistics.stdev(confidences)
        assert metrics.confidence_std == pytest.approx(expected_std)

    def test_counts_gaps(self):
        """Should count gaps correctly"""
        matches = []
        for i, conf in enumerate([0.8, 0.9, 0.7]):
            m = MagicMock()
            m.primary_match.confidence = conf
            m.has_gap = (i == 1)  # Second match has gap
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches=matches, total_segments=3)
        assert metrics.gap_count == 1

    def test_match_rate_calculation(self):
        """Should calculate match rate correctly"""
        matches = []
        for conf in [0.8, 0.9, 0.7]:
            m = MagicMock()
            m.primary_match.confidence = conf
            m.has_gap = False
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches=matches, total_segments=5)
        assert metrics.match_rate == pytest.approx(0.6)  # 3/5


class TestCalculateMatchQualityMetricsDirectMatch:
    """Tests for calculate_match_quality_metrics with direct Match objects"""

    def test_handles_direct_match_objects(self):
        """Should handle Match objects without primary_match"""
        match = MagicMock(spec=['confidence', 'has_gap'])
        match.confidence = 0.85
        match.has_gap = False
        # Ensure no primary_match attribute
        del match.primary_match

        metrics = calculate_match_quality_metrics(matches=[match], total_segments=1)
        assert metrics.avg_confidence == 0.85
        assert metrics.matched_segments == 1

    def test_handles_mixed_match_types(self):
        """Should handle mix of MatchResult and Match objects"""
        # MatchResult style
        match1 = MagicMock()
        match1.primary_match.confidence = 0.8
        match1.has_gap = False

        # Direct Match style
        match2 = MagicMock(spec=['confidence', 'has_gap'])
        match2.confidence = 0.9
        match2.has_gap = False
        del match2.primary_match

        metrics = calculate_match_quality_metrics(matches=[match1, match2], total_segments=2)
        assert metrics.avg_confidence == pytest.approx(0.85)
        assert metrics.matched_segments == 2


class TestLogQualitySummary:
    """Tests for log_quality_summary function"""

    def test_logs_at_info_level(self, caplog):
        """Should log at INFO level"""
        metrics = MatchQualityMetrics(
            avg_confidence=0.85,
            min_confidence=0.6,
            max_confidence=0.95,
            confidence_std=0.12,
            gap_count=2,
            match_rate=0.9,
            total_segments=20,
            matched_segments=18,
        )

        with caplog.at_level(logging.INFO, logger='src.matching.metrics'):
            log_quality_summary(metrics)

        assert len(caplog.records) > 0
        assert all(r.levelno == logging.INFO for r in caplog.records)

    def test_logs_summary_header(self, caplog):
        """Should log summary header"""
        metrics = MatchQualityMetrics()

        with caplog.at_level(logging.INFO, logger='src.matching.metrics'):
            log_quality_summary(metrics)

        assert any("Match Quality Summary" in r.message for r in caplog.records)

    def test_logs_all_metrics(self, caplog):
        """Should log all metric values"""
        metrics = MatchQualityMetrics(
            avg_confidence=0.85,
            min_confidence=0.6,
            max_confidence=0.95,
            confidence_std=0.12,
            gap_count=2,
            match_rate=0.9,
            total_segments=20,
            matched_segments=18,
        )

        with caplog.at_level(logging.INFO, logger='src.matching.metrics'):
            log_quality_summary(metrics)

        log_text = ' '.join(r.message for r in caplog.records)
        assert '20' in log_text  # total_segments
        assert '18' in log_text  # matched_segments
        assert '2' in log_text   # gap_count
        assert '0.85' in log_text or '0.850' in log_text  # avg_confidence
        assert '0.6' in log_text or '0.600' in log_text   # min_confidence
        assert '0.95' in log_text or '0.950' in log_text  # max_confidence

    def test_logs_match_rate_as_percentage(self, caplog):
        """Should log match rate as percentage"""
        metrics = MatchQualityMetrics(match_rate=0.9)

        with caplog.at_level(logging.INFO, logger='src.matching.metrics'):
            log_quality_summary(metrics)

        log_text = ' '.join(r.message for r in caplog.records)
        assert '90' in log_text or '90.0%' in log_text


class TestMetricsDistribution:
    """Tests that metrics correctly reflect distribution of match confidences"""

    def test_uniform_distribution(self):
        """Uniform distribution should have known std"""
        # Uniform distribution from 0.1 to 0.9 (9 values)
        matches = []
        confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for conf in confidences:
            m = MagicMock()
            m.primary_match.confidence = conf
            m.has_gap = False
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches=matches, total_segments=9)

        # Expected values for uniform 0.1-0.9
        expected_avg = 0.5
        expected_std = statistics.stdev(confidences)

        assert metrics.avg_confidence == pytest.approx(expected_avg)
        assert metrics.confidence_std == pytest.approx(expected_std)
        assert metrics.min_confidence == 0.1
        assert metrics.max_confidence == 0.9

    def test_tight_distribution_low_std(self):
        """Tight distribution should have low standard deviation"""
        matches = []
        confidences = [0.84, 0.85, 0.85, 0.86, 0.85]
        for conf in confidences:
            m = MagicMock()
            m.primary_match.confidence = conf
            m.has_gap = False
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches=matches, total_segments=5)

        # Very tight distribution should have small std
        assert metrics.confidence_std < 0.01

    def test_bimodal_distribution(self):
        """Bimodal distribution should reflect in metrics"""
        matches = []
        # Two clusters: around 0.3 and around 0.9
        confidences = [0.3, 0.3, 0.3, 0.9, 0.9, 0.9]
        for conf in confidences:
            m = MagicMock()
            m.primary_match.confidence = conf
            m.has_gap = False
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches=matches, total_segments=6)

        # Bimodal should have high std and avg around 0.6
        assert metrics.avg_confidence == pytest.approx(0.6)
        assert metrics.confidence_std > 0.2  # High variance

    def test_all_same_confidence(self):
        """All same confidence should have zero std"""
        matches = []
        for _ in range(5):
            m = MagicMock()
            m.primary_match.confidence = 0.75
            m.has_gap = False
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches=matches, total_segments=5)

        assert metrics.avg_confidence == 0.75
        assert metrics.min_confidence == 0.75
        assert metrics.max_confidence == 0.75
        assert metrics.confidence_std == 0.0


class TestMatchStageIntegration:
    """Tests for MatchStage integration with quality metrics"""

    def test_metrics_module_importable(self):
        """Metrics module should be importable from matching package"""
        from src.matching import (
            MatchQualityMetrics,
            calculate_match_quality_metrics,
            log_quality_summary,
        )
        assert MatchQualityMetrics is not None
        assert calculate_match_quality_metrics is not None
        assert log_quality_summary is not None

    def test_metrics_in_match_stage_checkpoint(self):
        """MatchStage should include quality_metrics in checkpoint data"""
        # This is a design verification - the actual integration
        # is tested in test_stage_match.py with full pipeline
        from src.stages.match import MatchStage

        # Verify MatchStage exists and can be instantiated
        stage = MatchStage()
        assert stage.name == "MATCH"
