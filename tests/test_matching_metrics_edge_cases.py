"""
Tests for matching metrics edge cases.

Sprint 47 - US-47-012: Edge cases for MatchQualityMetrics, calculate_match_quality_metrics,
log_confidence_histogram, and from_dict serialization.
"""

import json
import statistics
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.metrics import (
    MatchQualityMetrics,
    calculate_match_quality_metrics,
    log_confidence_histogram,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


# ---------------------------------------------------------------------------
# numpy.float32 confidence values (Rule 7 — embeddings truthiness)
# ---------------------------------------------------------------------------

class TestNumpyFloat32Confidences:
    """Verify calculate_match_quality_metrics handles numpy.float32 values."""

    def test_numpy_float32_confidence_converted(self):
        """numpy.float32 confidence values should be converted to Python float."""
        np = pytest.importorskip("numpy")

        match = MagicMock()
        match.primary_match.confidence = np.float32(0.85)
        match.has_gap = False

        metrics = calculate_match_quality_metrics([match], total_segments=1)

        # Must be native float, not numpy.float32
        assert type(metrics.avg_confidence) is float
        assert type(metrics.min_confidence) is float
        assert type(metrics.max_confidence) is float
        assert metrics.avg_confidence == pytest.approx(0.85, abs=1e-5)

    def test_numpy_float32_to_dict_serializable(self):
        """Metrics built from numpy.float32 should JSON-serialize via to_dict."""
        np = pytest.importorskip("numpy")

        matches = []
        for val in [np.float32(0.6), np.float32(0.8)]:
            m = MagicMock()
            m.primary_match.confidence = val
            m.has_gap = False
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches, total_segments=2)
        d = metrics.to_dict()

        # json.dumps must not raise TypeError
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_numpy_float32_std_calculation(self):
        """Standard deviation from numpy.float32 values should be correct."""
        np = pytest.importorskip("numpy")

        raw = [0.5, 0.6, 0.7, 0.8, 0.9]
        matches = []
        for v in raw:
            m = MagicMock()
            m.primary_match.confidence = np.float32(v)
            m.has_gap = False
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches, total_segments=5)

        expected_std = statistics.stdev([float(v) for v in raw])
        assert metrics.confidence_std == pytest.approx(expected_std, abs=1e-4)

    def test_numpy_float64_also_handled(self):
        """numpy.float64 values should also be converted without error."""
        np = pytest.importorskip("numpy")

        match = MagicMock()
        match.primary_match.confidence = np.float64(0.92)
        match.has_gap = False

        metrics = calculate_match_quality_metrics([match], total_segments=1)
        assert type(metrics.avg_confidence) is float
        assert metrics.avg_confidence == pytest.approx(0.92, abs=1e-10)


# ---------------------------------------------------------------------------
# Single match → std = 0.0 edge case
# ---------------------------------------------------------------------------

class TestSingleMatchStdZero:
    """Single match must produce confidence_std = 0.0 (len < 2 guard)."""

    def test_single_match_result_std_zero(self):
        """One MatchResult-style match should yield std = 0.0."""
        m = MagicMock()
        m.primary_match.confidence = 0.77
        m.has_gap = False

        metrics = calculate_match_quality_metrics([m], total_segments=1)
        assert metrics.confidence_std == 0.0

    def test_single_direct_match_std_zero(self):
        """One direct Match-style match should also yield std = 0.0."""
        m = MagicMock(spec=["confidence", "has_gap"])
        m.confidence = 0.65
        m.has_gap = False
        del m.primary_match

        metrics = calculate_match_quality_metrics([m], total_segments=1)
        assert metrics.confidence_std == 0.0

    def test_single_match_all_stats_equal(self):
        """With one match, avg == min == max."""
        m = MagicMock()
        m.primary_match.confidence = 0.42
        m.has_gap = False

        metrics = calculate_match_quality_metrics([m], total_segments=1)
        assert metrics.avg_confidence == metrics.min_confidence == metrics.max_confidence
        assert metrics.avg_confidence == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Empty match list
# ---------------------------------------------------------------------------

class TestEmptyMatchList:
    """calculate_match_quality_metrics with empty matches."""

    def test_empty_matches_positive_segments(self):
        """Empty list with positive total_segments returns zero metrics."""
        metrics = calculate_match_quality_metrics([], total_segments=10)
        assert metrics.matched_segments == 0
        assert metrics.match_rate == 0.0
        assert metrics.avg_confidence == 0.0
        assert metrics.total_segments == 10

    def test_empty_matches_zero_segments(self):
        """Empty list with 0 total_segments should not divide by zero."""
        metrics = calculate_match_quality_metrics([], total_segments=0)
        assert metrics.match_rate == 0.0
        assert metrics.total_segments == 0

    def test_empty_matches_negative_segments(self):
        """Negative total_segments should be treated gracefully."""
        metrics = calculate_match_quality_metrics([], total_segments=-1)
        assert metrics.match_rate == 0.0


# ---------------------------------------------------------------------------
# has_gap = True scenarios
# ---------------------------------------------------------------------------

class TestHasGapScenarios:
    """Gap matches should count as gaps and not inflate matched coverage."""

    def test_all_matches_have_gaps(self):
        """All matches with has_gap=True should all count as gaps."""
        matches = []
        for conf in [0.7, 0.8, 0.9]:
            m = MagicMock()
            m.primary_match.confidence = conf
            m.has_gap = True
            matches.append(m)

        metrics = calculate_match_quality_metrics(matches, total_segments=3)
        assert metrics.gap_count == 3
        assert metrics.matched_segments == 0
        assert metrics.match_rate == 0.0
        # Confidence stats still summarize the attempted scores.
        assert metrics.avg_confidence == pytest.approx(0.8)

    def test_mixed_gap_and_no_gap(self):
        """Mix of gap/no-gap should count gaps correctly."""
        m1 = MagicMock()
        m1.primary_match.confidence = 0.9
        m1.has_gap = False

        m2 = MagicMock()
        m2.primary_match.confidence = 0.5
        m2.has_gap = True

        metrics = calculate_match_quality_metrics([m1, m2], total_segments=2)
        assert metrics.gap_count == 1
        assert metrics.matched_segments == 1

    def test_empty_source_file_not_counted_as_matched(self):
        """A primary match with explicit empty source_file is not a usable match."""
        m = MagicMock()
        m.primary_match.confidence = 0.95
        m.primary_match.match_type = ""
        m.primary_match.video_segment.source_file = ""
        m.has_gap = False

        metrics = calculate_match_quality_metrics([m], total_segments=1)
        assert metrics.gap_count == 0
        assert metrics.matched_segments == 0
        assert metrics.match_rate == 0.0

    def test_no_primary_match_counts_as_gap(self):
        """Object without confidence or primary_match counts as gap."""
        m = MagicMock(spec=[])  # no confidence, no primary_match
        del m.primary_match
        del m.confidence

        metrics = calculate_match_quality_metrics([m], total_segments=1)
        assert metrics.gap_count == 1
        assert metrics.matched_segments == 0


# ---------------------------------------------------------------------------
# log_confidence_histogram edge cases
# ---------------------------------------------------------------------------

class TestHistogramEdgeCasesExtended:
    """Extended edge cases for log_confidence_histogram."""

    def test_empty_list_no_crash(self):
        """Empty confidences list should return valid histogram string."""
        result = log_confidence_histogram([])
        assert "Total segments: 0" in result
        # All buckets should show 0
        for label in ["0.0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]:
            line = [l for l in result.split("\n") if label in l][0]
            assert "0 (" in line or "   0" in line

    def test_single_value_histogram(self):
        """Single value should appear in exactly one bucket."""
        result = log_confidence_histogram([0.75])
        assert "Total segments: 1" in result

        lines = result.split("\n")
        bucket_0708 = [l for l in lines if "0.7-0.8" in l][0]
        assert "1 " in bucket_0708 or "   1" in bucket_0708

    def test_all_values_in_one_bucket(self):
        """All values in same bucket should produce one full bar, others empty."""
        confidences = [0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
        result = log_confidence_histogram(confidences)

        lines = result.split("\n")
        target_bucket = [l for l in lines if "0.7-0.8" in l][0]
        assert "100.0%" in target_bucket
        assert target_bucket.count("█") > 0

        # Other buckets should have 0 bars
        for label in ["0.0-0.5", "0.5-0.6", "0.6-0.7", "0.8-0.9", "0.9-1.0"]:
            other_bucket = [l for l in lines if label in l][0]
            assert other_bucket.count("█") == 0

    def test_all_values_zero(self):
        """All 0.0 values should go into first bucket."""
        result = log_confidence_histogram([0.0, 0.0, 0.0])
        lines = result.split("\n")
        first_bucket = [l for l in lines if "0.0-0.5" in l][0]
        assert "3 " in first_bucket or "   3" in first_bucket
        assert "100.0%" in first_bucket

    def test_all_values_one(self):
        """All 1.0 values should go into last bucket."""
        result = log_confidence_histogram([1.0, 1.0])
        lines = result.split("\n")
        last_bucket = [l for l in lines if "0.9-1.0" in l][0]
        assert "2 " in last_bucket or "   2" in last_bucket

    def test_histogram_returns_string(self):
        """Histogram should always return a string."""
        assert isinstance(log_confidence_histogram([]), str)
        assert isinstance(log_confidence_histogram([0.5]), str)
        assert isinstance(log_confidence_histogram([0.1] * 1000), str)


# ---------------------------------------------------------------------------
# MatchQualityMetrics.from_dict() edge cases
# ---------------------------------------------------------------------------

class TestFromDictEdgeCases:
    """from_dict with missing fields, extra fields, and type coercion."""

    def test_empty_dict_uses_defaults(self):
        """Empty dict should produce all-default metrics."""
        metrics = MatchQualityMetrics.from_dict({})
        assert metrics.avg_confidence == 0.0
        assert metrics.min_confidence == 0.0
        assert metrics.max_confidence == 0.0
        assert metrics.confidence_std == 0.0
        assert metrics.gap_count == 0
        assert metrics.match_rate == 0.0
        assert metrics.total_segments == 0
        assert metrics.matched_segments == 0

    def test_partial_dict_fills_defaults(self):
        """Dict with only some keys should default the rest."""
        metrics = MatchQualityMetrics.from_dict({"avg_confidence": 0.9, "gap_count": 3})
        assert metrics.avg_confidence == 0.9
        assert metrics.gap_count == 3
        assert metrics.min_confidence == 0.0
        assert metrics.total_segments == 0

    def test_extra_fields_ignored(self):
        """Extra fields in dict should be silently ignored."""
        data = {
            "avg_confidence": 0.8,
            "unknown_field": "garbage",
            "another_extra": 999,
        }
        metrics = MatchQualityMetrics.from_dict(data)
        assert metrics.avg_confidence == 0.8
        assert not hasattr(metrics, "unknown_field")

    def test_roundtrip_with_numpy_values(self):
        """to_dict → from_dict roundtrip should preserve values even from numpy sources."""
        np = pytest.importorskip("numpy")

        original = MatchQualityMetrics(
            avg_confidence=float(np.float32(0.85)),
            min_confidence=float(np.float32(0.4)),
            max_confidence=float(np.float32(0.99)),
            confidence_std=float(np.float32(0.15)),
            gap_count=2,
            match_rate=float(np.float32(0.9)),
            total_segments=10,
            matched_segments=9,
        )
        d = original.to_dict()
        restored = MatchQualityMetrics.from_dict(d)

        assert restored.avg_confidence == pytest.approx(original.avg_confidence, abs=1e-5)
        assert restored.gap_count == original.gap_count
        assert restored.total_segments == original.total_segments

    def test_from_dict_with_int_for_float_fields(self):
        """Integer values in float fields should work (Python auto-promotes)."""
        data = {
            "avg_confidence": 1,  # int instead of float
            "min_confidence": 0,
            "max_confidence": 1,
            "confidence_std": 0,
            "gap_count": 5,
            "match_rate": 1,
            "total_segments": 10,
            "matched_segments": 10,
        }
        metrics = MatchQualityMetrics.from_dict(data)
        assert metrics.avg_confidence == 1.0
        assert metrics.gap_count == 5
