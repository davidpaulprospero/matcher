"""
Tests for CaptionMetrics performance summary (US-59-009).

Verifies that:
- record_format_attempt() tracks per-format subprocess calls and timing
- get_performance_summary() accurately reflects subprocess calls and time spent
- Performance summary is included in summary() output
- Preflight and negative cache savings are tracked correctly
- Serialization round-trips through to_dict()/from_dict()
"""

import pytest

from src.caption_fetcher import CaptionMetrics


@pytest.fixture
def metrics():
    """Fresh CaptionMetrics instance."""
    return CaptionMetrics()


@pytest.mark.fast
class TestRecordFormatAttempt:
    """Test record_format_attempt() tracks per-format attempts."""

    def test_records_successful_attempt(self, metrics):
        """Successful attempt is recorded with correct fields."""
        metrics.record_format_attempt("vid1", "json3", True, 2.5)

        assert len(metrics.format_attempt_records) == 1
        record = metrics.format_attempt_records[0]
        assert record['video_id'] == "vid1"
        assert record['format_name'] == "json3"
        assert record['success'] is True
        assert record['elapsed_seconds'] == 2.5

    def test_records_failed_attempt(self, metrics):
        """Failed attempt is recorded correctly."""
        metrics.record_format_attempt("vid2", "vtt", False, 1.0)

        assert len(metrics.format_attempt_records) == 1
        record = metrics.format_attempt_records[0]
        assert record['success'] is False

    def test_records_multiple_attempts(self, metrics):
        """Multiple attempts accumulate in order."""
        metrics.record_format_attempt("vid1", "json3", False, 3.0)
        metrics.record_format_attempt("vid1", "vtt", True, 2.0)
        metrics.record_format_attempt("vid2", "json3", True, 1.5)

        assert len(metrics.format_attempt_records) == 3
        assert metrics.format_attempt_records[0]['format_name'] == "json3"
        assert metrics.format_attempt_records[1]['format_name'] == "vtt"
        assert metrics.format_attempt_records[2]['video_id'] == "vid2"


@pytest.mark.fast
class TestGetPerformanceSummary:
    """Test get_performance_summary() returns accurate metrics."""

    def test_empty_metrics(self, metrics):
        """Empty metrics returns zeroed summary."""
        summary = metrics.get_performance_summary()

        assert summary['total_subprocess_calls'] == 0
        assert summary['total_subprocess_seconds'] == 0.0
        assert summary['avg_seconds_per_call'] == 0.0
        assert summary['calls_saved_by_preflight'] == 0
        assert summary['calls_saved_by_negative_cache'] == 0

    def test_counts_subprocess_calls(self, metrics):
        """Total subprocess calls matches recorded attempts."""
        metrics.record_format_attempt("v1", "json3", True, 2.0)
        metrics.record_format_attempt("v1", "vtt", False, 1.0)
        metrics.record_format_attempt("v2", "json3", True, 3.0)

        summary = metrics.get_performance_summary()
        assert summary['total_subprocess_calls'] == 3

    def test_total_seconds(self, metrics):
        """Total seconds sums all elapsed times."""
        metrics.record_format_attempt("v1", "json3", True, 2.0)
        metrics.record_format_attempt("v2", "vtt", True, 3.0)

        summary = metrics.get_performance_summary()
        assert summary['total_subprocess_seconds'] == 5.0

    def test_average_seconds(self, metrics):
        """Average seconds correctly computed."""
        metrics.record_format_attempt("v1", "json3", True, 2.0)
        metrics.record_format_attempt("v2", "vtt", True, 4.0)

        summary = metrics.get_performance_summary()
        assert summary['avg_seconds_per_call'] == 3.0

    def test_preflight_savings(self, metrics):
        """Preflight savings are tracked."""
        metrics.record_preflight_saving("v1")
        metrics.record_preflight_saving("v2")

        summary = metrics.get_performance_summary()
        assert summary['calls_saved_by_preflight'] == 2

    def test_negative_cache_savings(self, metrics):
        """Negative cache savings are tracked."""
        metrics.record_negative_cache_saving("v1")

        summary = metrics.get_performance_summary()
        assert summary['calls_saved_by_negative_cache'] == 1

    def test_combined_performance_summary(self, metrics):
        """Full scenario: subprocess calls + savings from both optimizations."""
        # 5 videos: 3 had subprocess calls, 1 saved by preflight, 1 saved by neg-cache
        metrics.record_format_attempt("v1", "json3", True, 2.5)
        metrics.record_format_attempt("v2", "json3", False, 1.0)
        metrics.record_format_attempt("v2", "vtt", True, 1.5)
        metrics.record_format_attempt("v3", "json3", True, 3.0)
        metrics.record_preflight_saving("v4")
        metrics.record_negative_cache_saving("v5")

        summary = metrics.get_performance_summary()
        assert summary['total_subprocess_calls'] == 4
        assert summary['total_subprocess_seconds'] == 8.0
        assert summary['avg_seconds_per_call'] == 2.0
        assert summary['calls_saved_by_preflight'] == 1
        assert summary['calls_saved_by_negative_cache'] == 1


@pytest.mark.fast
class TestPerformanceSummaryInReport:
    """Test performance summary appears in summary() output."""

    def test_summary_includes_performance_line(self, metrics):
        """summary() output includes performance data when attempts recorded."""
        metrics.record_format_attempt("v1", "json3", True, 2.0)
        metrics.record_format_attempt("v2", "vtt", True, 3.0)

        output = metrics.summary()
        assert "Performance:" in output
        assert "2 subprocess calls" in output
        assert "5.0s total" in output

    def test_summary_includes_savings(self, metrics):
        """summary() includes preflight and negative cache savings."""
        metrics.record_format_attempt("v1", "json3", True, 1.0)
        metrics.record_preflight_saving("v2")
        metrics.record_negative_cache_saving("v3")

        output = metrics.summary()
        assert "by preflight" in output
        assert "by neg-cache" in output

    def test_summary_excludes_performance_when_empty(self, metrics):
        """summary() omits performance section when no data."""
        # Only record basic fetch data, no format attempts
        metrics.record_fetch_attempt("v1")

        output = metrics.summary()
        assert "Performance:" not in output


@pytest.mark.fast
class TestPerformanceSerialization:
    """Test performance data round-trips through to_dict/from_dict."""

    def test_to_dict_includes_performance_fields(self, metrics):
        """to_dict() includes new performance fields."""
        metrics.record_format_attempt("v1", "json3", True, 2.0)
        metrics.record_preflight_saving("v2")
        metrics.record_negative_cache_saving("v3")

        data = metrics.to_dict()
        assert len(data['format_attempt_records']) == 1
        assert data['calls_saved_by_preflight'] == 1
        assert data['calls_saved_by_negative_cache'] == 1

    def test_from_dict_restores_performance_fields(self, metrics):
        """from_dict() restores performance fields correctly."""
        metrics.record_format_attempt("v1", "json3", True, 2.5)
        metrics.record_format_attempt("v2", "vtt", False, 1.0)
        metrics.record_preflight_saving("v3")
        metrics.record_negative_cache_saving("v4")
        metrics.record_negative_cache_saving("v5")

        data = metrics.to_dict()
        restored = CaptionMetrics.from_dict(data)

        assert len(restored.format_attempt_records) == 2
        assert restored.calls_saved_by_preflight == 1
        assert restored.calls_saved_by_negative_cache == 2

        # Verify get_performance_summary matches
        original_summary = metrics.get_performance_summary()
        restored_summary = restored.get_performance_summary()
        assert original_summary == restored_summary

    def test_from_dict_handles_missing_fields(self):
        """from_dict() handles legacy data without performance fields."""
        legacy_data = {
            'fetch_attempts': 10,
            'successes': 8,
            'failures': 2,
        }

        restored = CaptionMetrics.from_dict(legacy_data)
        assert restored.format_attempt_records == []
        assert restored.calls_saved_by_preflight == 0
        assert restored.calls_saved_by_negative_cache == 0
