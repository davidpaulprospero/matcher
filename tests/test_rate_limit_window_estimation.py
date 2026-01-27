"""
Tests for rate limit window duration estimation (US-002 Sprint 13).

Verifies:
- get_estimated_rate_limit_window() returns None with insufficient data
- Median calculation with 5 alternating failure/success timestamps
- Minimum threshold of 3 pairs returns valid result
- Only last 10 pairs used when more data exists
- Per-keyword window isolation
- Window estimate included in export_to_json() under rate_limiting.estimated_window_seconds
"""

import pytest
import time
from unittest.mock import patch

from src.downloader.rate_limit_metrics import RateLimitMetrics


@pytest.mark.fast
class TestEstimatedRateLimitWindowNone:
    """Test get_estimated_rate_limit_window() returns None when insufficient data."""

    def test_returns_none_with_no_events(self):
        """No events logged at all → None."""
        metrics = RateLimitMetrics()
        assert metrics.get_estimated_rate_limit_window("sunset") is None

    def test_returns_none_with_zero_pairs(self):
        """Only failures, no recoveries → no pairs → None."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_failure("sunset")
        metrics.record_rate_limit_failure("sunset")
        metrics.record_rate_limit_failure("sunset")
        assert metrics.get_estimated_rate_limit_window("sunset") is None

    def test_returns_none_with_one_pair(self):
        """Only 1 failure/recovery pair → below threshold → None."""
        metrics = RateLimitMetrics()
        metrics._rate_limit_events_log["sunset"] = [
            (100.0, 'failure'),
            (130.0, 'recovery'),
        ]
        assert metrics.get_estimated_rate_limit_window("sunset") is None

    def test_returns_none_with_two_pairs(self):
        """Only 2 failure/recovery pairs → below threshold → None."""
        metrics = RateLimitMetrics()
        metrics._rate_limit_events_log["sunset"] = [
            (100.0, 'failure'),
            (130.0, 'recovery'),
            (200.0, 'failure'),
            (260.0, 'recovery'),
        ]
        assert metrics.get_estimated_rate_limit_window("sunset") is None

    def test_returns_none_for_unknown_keyword(self):
        """Querying a keyword with no events → None."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_failure("ocean")
        assert metrics.get_estimated_rate_limit_window("mountain") is None


@pytest.mark.fast
class TestEstimatedRateLimitWindowMedian:
    """Test median calculation with known timestamp patterns."""

    def test_median_with_5_pairs(self):
        """5 alternating failure/recovery pairs with gaps 30, 60, 45, 90, 120.

        Sorted gaps: [30, 45, 60, 90, 120] → median = 60.
        """
        metrics = RateLimitMetrics()
        # Build events: failure at t, recovery at t+gap
        events = []
        base = 1000.0
        gaps = [30.0, 60.0, 45.0, 90.0, 120.0]
        t = base
        for gap in gaps:
            events.append((t, 'failure'))
            events.append((t + gap, 'recovery'))
            t += gap + 50.0  # space between pairs

        metrics._rate_limit_events_log["sunset"] = events
        result = metrics.get_estimated_rate_limit_window("sunset")
        assert result == 60.0

    def test_median_with_even_pairs(self):
        """4 pairs with gaps 20, 40, 60, 80 → median = (40+60)/2 = 50."""
        metrics = RateLimitMetrics()
        events = []
        t = 1000.0
        gaps = [20.0, 40.0, 60.0, 80.0]
        for gap in gaps:
            events.append((t, 'failure'))
            events.append((t + gap, 'recovery'))
            t += gap + 100.0

        metrics._rate_limit_events_log["ocean"] = events
        result = metrics.get_estimated_rate_limit_window("ocean")
        assert result == 50.0


@pytest.mark.fast
class TestEstimatedRateLimitWindowMinThreshold:
    """Test window estimation with exactly 3 pairs at minimum threshold."""

    def test_exactly_3_pairs_returns_valid_result(self):
        """3 pairs with gaps 30, 60, 90 → median = 60."""
        metrics = RateLimitMetrics()
        events = [
            (100.0, 'failure'), (130.0, 'recovery'),   # gap = 30
            (200.0, 'failure'), (260.0, 'recovery'),   # gap = 60
            (400.0, 'failure'), (490.0, 'recovery'),   # gap = 90
        ]
        metrics._rate_limit_events_log["sunset"] = events
        result = metrics.get_estimated_rate_limit_window("sunset")
        assert result == 60.0

    def test_3_pairs_with_equal_gaps(self):
        """3 pairs with identical gaps → median = that gap."""
        metrics = RateLimitMetrics()
        events = [
            (100.0, 'failure'), (145.0, 'recovery'),   # gap = 45
            (200.0, 'failure'), (245.0, 'recovery'),   # gap = 45
            (300.0, 'failure'), (345.0, 'recovery'),   # gap = 45
        ]
        metrics._rate_limit_events_log["mountain"] = events
        result = metrics.get_estimated_rate_limit_window("mountain")
        assert result == 45.0


@pytest.mark.fast
class TestEstimatedRateLimitWindowLastTenPairs:
    """Test that only last 10 pairs are used when more data exists."""

    def test_oldest_pairs_discarded_when_more_than_10(self):
        """Record 15 pairs; only last 10 should be used for median."""
        metrics = RateLimitMetrics()
        events = []
        t = 1000.0

        # First 5 pairs with large gaps (1000s each) — should be discarded
        for _ in range(5):
            events.append((t, 'failure'))
            events.append((t + 1000.0, 'recovery'))
            t += 1500.0

        # Next 10 pairs with small gaps (10s each) — should be kept
        for _ in range(10):
            events.append((t, 'failure'))
            events.append((t + 10.0, 'recovery'))
            t += 50.0

        metrics._rate_limit_events_log["sunset"] = events

        result = metrics.get_estimated_rate_limit_window("sunset")
        # If last 10 pairs are used, median should be 10.0 (all gaps = 10)
        # If ALL 15 pairs were used, median would be higher
        assert result == 10.0

    def test_exactly_10_pairs_uses_all(self):
        """Record exactly 10 pairs; all should be used."""
        metrics = RateLimitMetrics()
        events = []
        t = 1000.0
        gaps = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        for gap in gaps:
            events.append((t, 'failure'))
            events.append((t + gap, 'recovery'))
            t += gap + 50.0

        metrics._rate_limit_events_log["ocean"] = events
        result = metrics.get_estimated_rate_limit_window("ocean")
        # Sorted: [10,20,30,40,50,60,70,80,90,100] → median = (50+60)/2 = 55
        assert result == 55.0


@pytest.mark.fast
class TestPerKeywordWindowIsolation:
    """Test per-keyword window isolation — one keyword doesn't affect another."""

    def test_keyword_a_independent_of_keyword_b(self):
        """Keyword A's pairs don't affect keyword B's estimate."""
        metrics = RateLimitMetrics()

        # Keyword A: 3 pairs with gaps 30, 60, 90 → median = 60
        metrics._rate_limit_events_log["ocean"] = [
            (100.0, 'failure'), (130.0, 'recovery'),
            (200.0, 'failure'), (260.0, 'recovery'),
            (300.0, 'failure'), (390.0, 'recovery'),
        ]

        # Keyword B: 3 pairs with gaps 10, 20, 30 → median = 20
        metrics._rate_limit_events_log["sunset"] = [
            (100.0, 'failure'), (110.0, 'recovery'),
            (200.0, 'failure'), (220.0, 'recovery'),
            (300.0, 'failure'), (330.0, 'recovery'),
        ]

        assert metrics.get_estimated_rate_limit_window("ocean") == 60.0
        assert metrics.get_estimated_rate_limit_window("sunset") == 20.0

    def test_keyword_with_no_data_returns_none(self):
        """Keyword with no events returns None even when other keywords have data."""
        metrics = RateLimitMetrics()
        metrics._rate_limit_events_log["ocean"] = [
            (100.0, 'failure'), (130.0, 'recovery'),
            (200.0, 'failure'), (260.0, 'recovery'),
            (300.0, 'failure'), (390.0, 'recovery'),
        ]

        assert metrics.get_estimated_rate_limit_window("ocean") == 60.0
        assert metrics.get_estimated_rate_limit_window("mountain") is None


@pytest.mark.fast
class TestWindowEstimateInExportJson:
    """Test window estimate included in export_to_json() output."""

    def test_estimated_window_seconds_in_rate_limiting_section(self):
        """Verify estimated_window_seconds appears under rate_limiting key."""
        metrics = RateLimitMetrics()

        # Add enough pairs for 2 keywords
        metrics._rate_limit_events_log["ocean"] = [
            (100.0, 'failure'), (130.0, 'recovery'),
            (200.0, 'failure'), (260.0, 'recovery'),
            (300.0, 'failure'), (390.0, 'recovery'),
        ]
        metrics._rate_limit_events_log["sunset"] = [
            (100.0, 'failure'), (110.0, 'recovery'),
            (200.0, 'failure'), (220.0, 'recovery'),
            (300.0, 'failure'), (330.0, 'recovery'),
        ]

        export = metrics.export_to_json()
        assert "estimated_window_seconds" in export["rate_limiting"]
        windows = export["rate_limiting"]["estimated_window_seconds"]
        assert windows["ocean"] == 60.0
        assert windows["sunset"] == 20.0

    def test_estimated_window_empty_when_insufficient_data(self):
        """Verify estimated_window_seconds is empty dict with insufficient data."""
        metrics = RateLimitMetrics()
        # Only 1 pair (below threshold)
        metrics._rate_limit_events_log["ocean"] = [
            (100.0, 'failure'), (130.0, 'recovery'),
        ]

        export = metrics.export_to_json()
        assert "estimated_window_seconds" in export["rate_limiting"]
        assert export["rate_limiting"]["estimated_window_seconds"] == {}

    def test_estimated_window_only_includes_valid_keywords(self):
        """Keywords with insufficient data excluded from export."""
        metrics = RateLimitMetrics()

        # Ocean: 3 pairs → included
        metrics._rate_limit_events_log["ocean"] = [
            (100.0, 'failure'), (130.0, 'recovery'),
            (200.0, 'failure'), (260.0, 'recovery'),
            (300.0, 'failure'), (390.0, 'recovery'),
        ]

        # Sunset: only 2 pairs → excluded
        metrics._rate_limit_events_log["sunset"] = [
            (100.0, 'failure'), (110.0, 'recovery'),
            (200.0, 'failure'), (220.0, 'recovery'),
        ]

        export = metrics.export_to_json()
        windows = export["rate_limiting"]["estimated_window_seconds"]
        assert "ocean" in windows
        assert "sunset" not in windows


@pytest.mark.fast
class TestRecordMethods:
    """Test record_rate_limit_failure() and record_rate_limit_recovery()."""

    def test_record_failure_creates_keyword_entry(self):
        """First failure for a keyword creates the events log entry."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_failure("sunset")
        assert "sunset" in metrics._rate_limit_events_log
        assert len(metrics._rate_limit_events_log["sunset"]) == 1
        assert metrics._rate_limit_events_log["sunset"][0][1] == 'failure'

    def test_record_recovery_creates_keyword_entry(self):
        """First recovery for a keyword creates the events log entry."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_recovery("sunset")
        assert "sunset" in metrics._rate_limit_events_log
        assert len(metrics._rate_limit_events_log["sunset"]) == 1
        assert metrics._rate_limit_events_log["sunset"][0][1] == 'recovery'

    def test_record_failure_and_recovery_sequence(self):
        """Sequence of failure/recovery events recorded correctly."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_failure("sunset")
        metrics.record_rate_limit_recovery("sunset")
        metrics.record_rate_limit_failure("sunset")
        metrics.record_rate_limit_recovery("sunset")

        events = metrics._rate_limit_events_log["sunset"]
        assert len(events) == 4
        assert events[0][1] == 'failure'
        assert events[1][1] == 'recovery'
        assert events[2][1] == 'failure'
        assert events[3][1] == 'recovery'

    def test_timestamps_are_monotonically_increasing(self):
        """Timestamps from record methods should be increasing."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_failure("sunset")
        metrics.record_rate_limit_recovery("sunset")

        events = metrics._rate_limit_events_log["sunset"]
        assert events[0][0] <= events[1][0]

    def test_clear_resets_events_log(self):
        """clear() should reset the events log."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_failure("sunset")
        metrics.record_rate_limit_recovery("sunset")
        assert len(metrics._rate_limit_events_log) > 0

        metrics.clear()
        assert metrics._rate_limit_events_log == {}
