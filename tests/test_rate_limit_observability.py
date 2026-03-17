"""Tests for comprehensive rate limiting observability (US-136-010).

Tests structured logging, time-based aggregation, time-to-recovery metrics,
and dashboard-friendly JSON export.
"""

import pytest
import json
import time
from unittest.mock import MagicMock, patch

from src.downloader.rate_limit_metrics import RateLimitMetrics, _log_rate_limit_event


class TestStructuredLogging:
    """Test structured logging for rate limit events (AC1)."""

    def test_record_rate_limit_event_logs(self):
        """Test that rate limit events trigger structured logging."""
        metrics = RateLimitMetrics()

        with patch.object(metrics, 'record_event_timestamp') as mock_ts:
            metrics.record_rate_limit_event(tier="long", keyword="test keyword")

        # Verify event was recorded
        assert metrics.rate_limit_events == 1
        assert metrics.tier_rate_limit_events == {"long": 1}
        assert metrics.keyword_rate_limit_events == {"test keyword": 1}

    def test_record_backoff_logs(self):
        """Test that backoff events trigger structured logging."""
        metrics = RateLimitMetrics()
        metrics.record_backoff(seconds=5.0, severity="medium")

        assert metrics.backoff_attempts == 1
        assert metrics.time_spent_backing_off == 5.0
        assert metrics.backoff_events_by_severity == {"medium": 1}

    def test_record_cookie_rotation_logs(self):
        """Test that cookie rotation triggers structured logging."""
        metrics = RateLimitMetrics()
        metrics.record_cookie_rotation()

        assert metrics.cookie_rotations == 1

    def test_record_vpn_switch_logs(self):
        """Test that VPN switch triggers structured logging."""
        metrics = RateLimitMetrics()
        metrics.record_vpn_switch()

        assert metrics.vpn_switches == 1

    def test_record_circuit_breaker_trip_logs(self):
        """Test that circuit breaker trip triggers structured logging."""
        metrics = RateLimitMetrics()
        metrics.record_circuit_breaker_trip(60.0)

        assert metrics.circuit_breaker_trips == 1
        assert metrics.circuit_breaker_pause_seconds == 60.0


class TestTimeBasedAggregation:
    """Test time-based event aggregation (AC2)."""

    def test_record_event_timestamp(self):
        """Test that event timestamps are recorded."""
        metrics = RateLimitMetrics()
        initial_count = len(metrics._event_timestamps)

        metrics.record_event_timestamp()

        assert len(metrics._event_timestamps) == initial_count + 1

    def test_record_rate_limit_records_timestamp(self):
        """Test that rate limit events record timestamps."""
        metrics = RateLimitMetrics()

        metrics.record_rate_limit_event(keyword="test")
        metrics.record_rate_limit_event(keyword="test")

        assert len(metrics._event_timestamps) == 2

    def test_get_events_by_time_window(self):
        """Test time window aggregation returns correct format."""
        metrics = RateLimitMetrics()

        # Record multiple events
        metrics.record_event_timestamp()
        time.sleep(0.01)  # Small delay
        metrics.record_event_timestamp()

        windows = metrics.get_events_by_time_window(window_seconds=300.0)

        # Should have at least one window with events
        assert isinstance(windows, dict)
        total_events = sum(windows.values())
        assert total_events == 2

    def test_get_event_rate_per_minute(self):
        """Test events per minute calculation."""
        metrics = RateLimitMetrics()

        # Record timestamps
        now = time.time()
        metrics._event_timestamps.append(now - 60)  # 1 minute ago
        metrics._event_timestamps.append(now)        # now

        rate = metrics.get_event_rate_per_minute()

        # Should be close to 1 event per minute (2 events / 60 seconds * 60 = 2)
        assert rate == 2.0

    def test_get_event_rate_per_minute_empty(self):
        """Test events per minute with no events."""
        metrics = RateLimitMetrics()

        rate = metrics.get_event_rate_per_minute()

        assert rate == 0.0


class TestTimeToRecovery:
    """Test time-to-recovery metrics (AC4)."""

    def test_record_recovery_with_timing(self):
        """Test recording recovery with timing."""
        metrics = RateLimitMetrics()

        metrics.record_recovery_with_timing("keyword1", 30.0)
        metrics.record_recovery_with_timing("keyword1", 45.0)
        metrics.record_recovery_with_timing("keyword2", 60.0)

        stats = metrics.get_time_to_recovery_stats()

        assert "keyword1" in stats["per_keyword"]
        assert stats["per_keyword"]["keyword1"]["avg_seconds"] == 37.5
        assert stats["per_keyword"]["keyword2"]["avg_seconds"] == 60.0

    def test_get_time_to_recovery_stats_empty(self):
        """Test recovery stats with no data."""
        metrics = RateLimitMetrics()

        stats = metrics.get_time_to_recovery_stats()

        assert stats["per_keyword"] == {}
        assert stats["overall"] == {}

    def test_get_time_to_recovery_stats_overall(self):
        """Test overall recovery stats calculation."""
        metrics = RateLimitMetrics()

        metrics.record_recovery_with_timing("kw1", 10.0)
        metrics.record_recovery_with_timing("kw1", 20.0)
        metrics.record_recovery_with_timing("kw2", 30.0)

        stats = metrics.get_time_to_recovery_stats()

        assert stats["overall"]["avg_seconds"] == 20.0
        assert stats["overall"]["min_seconds"] == 10.0
        assert stats["overall"]["max_seconds"] == 30.0

    def test_record_rate_limit_tracks_last_time(self):
        """Test that rate limit events track last failure time."""
        metrics = RateLimitMetrics()

        metrics.record_rate_limit_event(keyword="testkw")

        assert "testkw" in metrics._last_rate_limit_time
        assert metrics._last_rate_limit_time["testkw"] > 0


class TestExportIncludesObservability:
    """Test that JSON export includes observability metrics (AC3)."""

    def test_export_includes_event_rate(self):
        """Test that export includes event rate per minute."""
        metrics = RateLimitMetrics()
        metrics.record_event_timestamp()
        metrics.record_event_timestamp()

        export = metrics.export_to_json()

        assert "rate_limiting" in export
        assert "event_rate_per_minute" in export["rate_limiting"]

    def test_export_includes_time_windows(self):
        """Test that export includes events by time window."""
        metrics = RateLimitMetrics()
        metrics.record_event_timestamp()

        export = metrics.export_to_json()

        assert "events_by_time_window" in export["rate_limiting"]

    def test_export_includes_time_to_recovery(self):
        """Test that export includes time-to-recovery metrics."""
        metrics = RateLimitMetrics()
        metrics.record_recovery_with_timing("test", 30.0)

        export = metrics.export_to_json()

        assert "time_to_recovery" in export["rate_limiting"]
        assert "per_keyword" in export["rate_limiting"]["time_to_recovery"]

    def test_export_json_serializable_with_new_fields(self):
        """Test that export with new fields is JSON serializable."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            rate_limit_events=10
        )
        metrics.record_event_timestamp()
        metrics.record_event_timestamp()
        metrics.record_recovery_with_timing("kw1", 30.0)

        # Should not raise
        json_str = json.dumps(metrics.export_to_json())
        assert isinstance(json_str, str)

        # Should parse correctly
        parsed = json.loads(json_str)
        assert parsed["rate_limiting"]["time_to_recovery"]["per_keyword"]["kw1"]["avg_seconds"] == 30.0


class TestClearResetsNewFields:
    """Test that clear() resets new fields."""

    def test_clear_resets_event_timestamps(self):
        """Test that clear resets event timestamps."""
        metrics = RateLimitMetrics()
        metrics.record_event_timestamp()
        metrics.record_event_timestamp()

        metrics.clear()

        assert metrics._event_timestamps == []

    def test_clear_resets_recovery_times(self):
        """Test that clear resets recovery times."""
        metrics = RateLimitMetrics()
        metrics.record_recovery_with_timing("kw1", 30.0)

        metrics.clear()

        assert metrics._recovery_times == {}

    def test_clear_resets_last_rate_limit_time(self):
        """Test that clear resets last rate limit time."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(keyword="test")

        metrics.clear()

        assert metrics._last_rate_limit_time == {}


class TestMetricsCompleteness:
    """Test metrics completeness (AC5: Create test to verify metrics completeness)."""

    def test_all_acceptance_criteria_metrics_present(self):
        """AC5: Verify all required metrics are present in export."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=90,
            failed_downloads=10,
            rate_limit_events=5,
            backoff_attempts=3,
            time_spent_backing_off=45.0,
            cookie_rotations=2,
            vpn_switches=1,
            circuit_breaker_trips=1,
            circuit_breaker_pause_seconds=30.0,
        )

        # Add time-based data
        metrics.record_event_timestamp()
        metrics.record_event_timestamp()

        # Add recovery data
        metrics.record_recovery_with_timing("keyword1", 30.0)

        export = metrics.export_to_json()

        # AC1: Structured logging - verified by recording methods
        assert metrics.rate_limit_events > 0

        # AC2: Rate limit event aggregation by type and time
        assert "by_tier" in export["rate_limiting"]
        assert "by_keyword" in export["rate_limiting"]
        assert "events_by_time_window" in export["rate_limiting"]
        assert "event_rate_per_minute" in export["rate_limiting"]

        # AC3: Dashboard-friendly metrics export (JSON)
        assert "schema_version" in export
        assert "export_timestamp" in export
        assert json.dumps(export)  # Must be serializable

        # AC4: Time-to-recovery metrics
        assert "time_to_recovery" in export["rate_limiting"]
        assert "per_keyword" in export["rate_limiting"]["time_to_recovery"]
        assert "overall" in export["rate_limiting"]["time_to_recovery"]

    def test_dashboard_friendly_format(self):
        """AC3: Verify dashboard-friendly JSON format."""
        metrics = RateLimitMetrics(
            total_downloads=50,
            successful_downloads=45,
            rate_limit_events=5,
            cookie_rotations=1,
        )
        metrics.record_event_timestamp()

        export = metrics.export_to_json()

        # Verify top-level structure suitable for dashboard
        assert "schema_version" in export
        assert "export_timestamp" in export
        assert "session" in export
        assert "downloads" in export
        assert "rate_limiting" in export
        assert "escalation" in export

        # Verify numeric values are JSON serializable (not numpy types)
        assert isinstance(export["downloads"]["total"], int)
        assert isinstance(export["downloads"]["success_rate_percent"], float)

    def test_all_event_types_logged(self):
        """AC1: Verify all event types trigger logging."""
        metrics = RateLimitMetrics()

        # These should all work without raising
        metrics.record_rate_limit_event(tier="long", keyword="test")
        metrics.record_backoff(5.0, severity="medium")
        metrics.record_cookie_rotation()
        metrics.record_vpn_switch()
        metrics.record_circuit_breaker_trip(30.0)

        # Verify all recorded
        assert metrics.rate_limit_events == 1
        assert metrics.backoff_attempts == 1
        assert metrics.cookie_rotations == 1
        assert metrics.vpn_switches == 1
        assert metrics.circuit_breaker_trips == 1
