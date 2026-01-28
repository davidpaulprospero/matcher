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


# ============================================================================
# US-009 Sprint 21: Rate Limit Window Estimation Edge Cases
# ============================================================================


@pytest.mark.fast
class TestSparseTimingDataUS009:
    """Test AC2: median gap calculation handles sparse timing data correctly."""

    def test_single_large_gap_in_sequence(self):
        """Test median with one outlier large gap among normal gaps.

        Gaps: [30, 30, 1000, 30, 30] → sorted: [30, 30, 30, 30, 1000] → median = 30
        """
        metrics = RateLimitMetrics()
        events = []
        t = 1000.0
        # Normal gaps with one large outlier
        gaps = [30.0, 30.0, 1000.0, 30.0, 30.0]
        for gap in gaps:
            events.append((t, 'failure'))
            events.append((t + gap, 'recovery'))
            t += gap + 100.0

        metrics._rate_limit_events_log["sparse"] = events
        result = metrics.get_estimated_rate_limit_window("sparse")
        assert result == 30.0  # Median is robust to outlier

    def test_very_small_gaps_near_zero(self):
        """Test with gaps close to zero (immediate recoveries).

        Gaps: [0.1, 0.2, 0.1] → median = 0.1
        """
        metrics = RateLimitMetrics()
        events = [
            (100.0, 'failure'), (100.1, 'recovery'),   # 0.1s gap
            (200.0, 'failure'), (200.2, 'recovery'),   # 0.2s gap
            (300.0, 'failure'), (300.1, 'recovery'),   # 0.1s gap
        ]
        metrics._rate_limit_events_log["quick"] = events
        result = metrics.get_estimated_rate_limit_window("quick")
        assert abs(result - 0.1) < 0.001

    def test_large_time_gaps_between_pairs(self):
        """Test with large time separation between failure/recovery pairs.

        The gap within a pair matters, not the gap between pairs.
        """
        metrics = RateLimitMetrics()
        events = [
            (0.0, 'failure'), (60.0, 'recovery'),        # pair 1: gap = 60
            (100000.0, 'failure'), (100030.0, 'recovery'),  # pair 2: gap = 30 (1 day later)
            (200000.0, 'failure'), (200045.0, 'recovery'),  # pair 3: gap = 45 (2 days later)
        ]
        metrics._rate_limit_events_log["sparse"] = events
        result = metrics.get_estimated_rate_limit_window("sparse")
        assert result == 45.0  # median of [60, 30, 45] sorted = [30, 45, 60] → 45

    def test_alternating_small_and_large_gaps(self):
        """Test with alternating small/large gaps.

        Gaps: [10, 1000, 10, 1000, 10] → sorted: [10, 10, 10, 1000, 1000] → median = 10
        """
        metrics = RateLimitMetrics()
        events = []
        t = 1000.0
        gaps = [10.0, 1000.0, 10.0, 1000.0, 10.0]
        for gap in gaps:
            events.append((t, 'failure'))
            events.append((t + gap, 'recovery'))
            t += gap + 50.0

        metrics._rate_limit_events_log["alternating"] = events
        result = metrics.get_estimated_rate_limit_window("alternating")
        assert result == 10.0  # Median of odd number, center value

    def test_identical_timestamps_failure_recovery(self):
        """Test with failure and recovery at identical timestamps (0 gap).

        Gaps: [0.0, 30.0, 60.0] → median = 30.0
        """
        metrics = RateLimitMetrics()
        events = [
            (100.0, 'failure'), (100.0, 'recovery'),   # 0s gap
            (200.0, 'failure'), (230.0, 'recovery'),   # 30s gap
            (300.0, 'failure'), (360.0, 'recovery'),   # 60s gap
        ]
        metrics._rate_limit_events_log["zero_gap"] = events
        result = metrics.get_estimated_rate_limit_window("zero_gap")
        assert result == 30.0


@pytest.mark.fast
class TestSessionCountPersistenceUS009:
    """Test AC3: session count tracking persists across pipeline resumptions."""

    def test_events_log_survives_to_dict_from_dict_roundtrip(self):
        """Test that events log is NOT persisted in to_dict (expected behavior).

        The events log is intentionally not persisted since it's runtime-only.
        Session count tracks how many sessions contributed to metrics.
        """
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_failure("sunset")
        metrics.record_rate_limit_recovery("sunset")
        metrics.record_rate_limit_failure("sunset")
        metrics.record_rate_limit_recovery("sunset")
        metrics.record_rate_limit_failure("sunset")
        metrics.record_rate_limit_recovery("sunset")

        # Events log should have data
        assert len(metrics._rate_limit_events_log.get("sunset", [])) == 6

        # to_dict() should include _rate_limit_events_log from asdict()
        data = metrics.to_dict()
        # The dataclass asdict() includes the field
        assert "_rate_limit_events_log" in data

        # But from_dict() doesn't restore it (intentional)
        restored = RateLimitMetrics.from_dict(data)
        assert restored._rate_limit_events_log == {}

    def test_session_count_increments_on_checkpoint_restore(self):
        """Test session_count increments when restoring from checkpoint."""
        metrics = RateLimitMetrics(session_count=1)
        data = metrics.to_dict()

        restored = RateLimitMetrics.from_checkpoint(data)
        assert restored.session_count == 2

    def test_session_count_tracks_multiple_resumptions(self):
        """Test session_count accumulates across multiple checkpoint restores."""
        # Session 1
        metrics = RateLimitMetrics(session_count=1)
        data1 = metrics.to_dict()

        # Session 2
        metrics2 = RateLimitMetrics.from_checkpoint(data1)
        assert metrics2.session_count == 2
        data2 = metrics2.to_dict()

        # Session 3
        metrics3 = RateLimitMetrics.from_checkpoint(data2)
        assert metrics3.session_count == 3

    def test_window_estimation_requires_fresh_data_each_session(self):
        """Test that window estimation needs new data after restore.

        Since events log isn't persisted, each session starts fresh.
        This is expected behavior - window estimates are per-session.
        """
        metrics = RateLimitMetrics()
        # Build up 3 pairs
        metrics._rate_limit_events_log["sunset"] = [
            (100.0, 'failure'), (130.0, 'recovery'),
            (200.0, 'failure'), (260.0, 'recovery'),
            (300.0, 'failure'), (390.0, 'recovery'),
        ]
        assert metrics.get_estimated_rate_limit_window("sunset") == 60.0

        # Simulate checkpoint save/restore
        data = metrics.to_dict()
        restored = RateLimitMetrics.from_checkpoint(data)

        # No events in restored (expected)
        assert restored.get_estimated_rate_limit_window("sunset") is None

    def test_keyword_rate_limit_events_persist_across_sessions(self):
        """Test keyword_rate_limit_events (counts) DO persist across sessions."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(keyword="sunset")
        metrics.record_rate_limit_event(keyword="sunset")
        metrics.record_rate_limit_event(keyword="ocean")

        data = metrics.to_dict()
        restored = RateLimitMetrics.from_checkpoint(data)

        # Counts should persist
        assert restored.keyword_rate_limit_events == {"sunset": 2, "ocean": 1}
        assert restored.session_count == 2


@pytest.mark.fast
class TestPerKeywordBreakdownAggregationUS009:
    """Test AC4: per-keyword breakdown aggregates correctly in export JSON."""

    def test_export_by_keyword_matches_recorded_events(self):
        """Test that export by_keyword matches recorded events exactly."""
        metrics = RateLimitMetrics()
        keywords = {"sunset": 5, "ocean": 3, "mountain": 2}
        for kw, count in keywords.items():
            for _ in range(count):
                metrics.record_rate_limit_event(keyword=kw)

        export = metrics.export_to_json()
        assert export["rate_limiting"]["by_keyword"] == keywords

    def test_export_by_keyword_empty_when_no_keywords(self):
        """Test export has empty by_keyword when no keywords recorded."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event()  # No keyword
        metrics.record_rate_limit_event(tier="short")  # No keyword

        export = metrics.export_to_json()
        assert export["rate_limiting"]["by_keyword"] == {}

    def test_export_by_keyword_accumulates_across_multiple_records(self):
        """Test that by_keyword accumulates from multiple recording calls."""
        metrics = RateLimitMetrics()
        # Simulate multiple batches of events
        for _ in range(3):
            metrics.record_rate_limit_event(keyword="sunset")
        for _ in range(2):
            metrics.record_rate_limit_event(keyword="ocean")
        # Later batch
        for _ in range(4):
            metrics.record_rate_limit_event(keyword="sunset")

        export = metrics.export_to_json()
        assert export["rate_limiting"]["by_keyword"]["sunset"] == 7
        assert export["rate_limiting"]["by_keyword"]["ocean"] == 2

    def test_export_by_keyword_with_special_characters(self):
        """Test keywords with spaces, unicode, and special chars."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(keyword="sunset beach")
        metrics.record_rate_limit_event(keyword="日本語")
        metrics.record_rate_limit_event(keyword="key-with-dashes")

        export = metrics.export_to_json()
        by_keyword = export["rate_limiting"]["by_keyword"]
        assert by_keyword["sunset beach"] == 1
        assert by_keyword["日本語"] == 1
        assert by_keyword["key-with-dashes"] == 1

    def test_export_estimated_window_matches_keyword_data(self):
        """Test that estimated_window_seconds aligns with by_keyword keys."""
        metrics = RateLimitMetrics()
        # Keywords with enough data for window estimation
        for _ in range(10):
            metrics.record_rate_limit_event(keyword="sunset")

        # Create 3 pairs for sunset
        metrics._rate_limit_events_log["sunset"] = [
            (100.0, 'failure'), (130.0, 'recovery'),
            (200.0, 'failure'), (260.0, 'recovery'),
            (300.0, 'failure'), (390.0, 'recovery'),
        ]

        export = metrics.export_to_json()
        # Both should have "sunset"
        assert "sunset" in export["rate_limiting"]["by_keyword"]
        assert "sunset" in export["rate_limiting"]["estimated_window_seconds"]

    def test_export_by_keyword_total_matches_rate_limit_events(self):
        """Test sum of by_keyword equals rate_limit_events when all have keywords."""
        metrics = RateLimitMetrics()
        keywords = {"a": 5, "b": 3, "c": 7}
        for kw, count in keywords.items():
            for _ in range(count):
                metrics.record_rate_limit_event(keyword=kw)

        export = metrics.export_to_json()
        by_keyword_sum = sum(export["rate_limiting"]["by_keyword"].values())
        assert by_keyword_sum == export["rate_limiting"]["total_events"]
        assert by_keyword_sum == 15


@pytest.mark.fast
class TestOverlappingFailureWindowsUS009:
    """Test AC5: window estimation handles overlapping failure windows."""

    def test_multiple_failures_before_recovery(self):
        """Test with multiple failures before a single recovery.

        Pattern: F, F, F, R — only first F paired with R.
        """
        metrics = RateLimitMetrics()
        events = [
            (100.0, 'failure'),   # F1
            (110.0, 'failure'),   # F2 (ignored, no recovery before next failure)
            (120.0, 'failure'),   # F3 (ignored)
            (200.0, 'recovery'),  # R pairs with F1 → gap = 100
            # Need 2 more pairs
            (300.0, 'failure'),
            (360.0, 'recovery'),  # gap = 60
            (400.0, 'failure'),
            (480.0, 'recovery'),  # gap = 80
        ]
        metrics._rate_limit_events_log["overlap"] = events
        result = metrics.get_estimated_rate_limit_window("overlap")
        # gaps = [100, 60, 80] → median = 80
        assert result == 80.0

    def test_interleaved_failures_and_recoveries(self):
        """Test with interleaved F/R pattern that isn't strictly alternating.

        Pattern: F, R, F, F, R, F, R
        Pairs: (F1,R1), (F2,R2), (F3,R3) where F2 skips one F
        """
        metrics = RateLimitMetrics()
        events = [
            (100.0, 'failure'),   # F1
            (130.0, 'recovery'),  # R1 → pair (F1,R1) gap = 30
            (200.0, 'failure'),   # F2
            (210.0, 'failure'),   # F3 (will be skipped - F2 already matched)
            (260.0, 'recovery'),  # R2 → pair (F2,R2) gap = 60
            (300.0, 'failure'),   # F4
            (345.0, 'recovery'),  # R3 → pair (F4,R3) gap = 45
        ]
        metrics._rate_limit_events_log["interleaved"] = events
        result = metrics.get_estimated_rate_limit_window("interleaved")
        # gaps = [30, 60, 45] → median = 45
        assert result == 45.0

    def test_recovery_without_prior_failure(self):
        """Test with recovery events that have no matching prior failure.

        Pattern: R, F, R, F, R, F, R
        The first R is ignored, then normal pairs form.
        """
        metrics = RateLimitMetrics()
        events = [
            (50.0, 'recovery'),    # Orphan R - no prior failure
            (100.0, 'failure'),
            (130.0, 'recovery'),   # gap = 30
            (200.0, 'failure'),
            (260.0, 'recovery'),   # gap = 60
            (300.0, 'failure'),
            (390.0, 'recovery'),   # gap = 90
        ]
        metrics._rate_limit_events_log["orphan_r"] = events
        result = metrics.get_estimated_rate_limit_window("orphan_r")
        # gaps = [30, 60, 90] → median = 60
        assert result == 60.0

    def test_consecutive_recoveries(self):
        """Test with consecutive recovery events.

        Pattern: F, R, R, R, F, R
        Only first R after each F counts.
        """
        metrics = RateLimitMetrics()
        events = [
            (100.0, 'failure'),
            (130.0, 'recovery'),   # Pairs with F → gap = 30
            (140.0, 'recovery'),   # Orphan R
            (150.0, 'recovery'),   # Orphan R
            (200.0, 'failure'),
            (260.0, 'recovery'),   # gap = 60
            (300.0, 'failure'),
            (345.0, 'recovery'),   # gap = 45
        ]
        metrics._rate_limit_events_log["multi_r"] = events
        result = metrics.get_estimated_rate_limit_window("multi_r")
        # gaps = [30, 60, 45] → median = 45
        assert result == 45.0

    def test_all_failures_no_recoveries(self):
        """Test with only failures and no recoveries."""
        metrics = RateLimitMetrics()
        events = [
            (100.0, 'failure'),
            (200.0, 'failure'),
            (300.0, 'failure'),
            (400.0, 'failure'),
        ]
        metrics._rate_limit_events_log["all_fail"] = events
        result = metrics.get_estimated_rate_limit_window("all_fail")
        assert result is None

    def test_all_recoveries_no_failures(self):
        """Test with only recoveries and no failures."""
        metrics = RateLimitMetrics()
        events = [
            (100.0, 'recovery'),
            (200.0, 'recovery'),
            (300.0, 'recovery'),
        ]
        metrics._rate_limit_events_log["all_recover"] = events
        result = metrics.get_estimated_rate_limit_window("all_recover")
        assert result is None

    def test_overlapping_windows_different_keywords(self):
        """Test overlapping windows don't affect other keywords."""
        metrics = RateLimitMetrics()
        # Keyword A: messy overlapping pattern
        metrics._rate_limit_events_log["messy"] = [
            (100.0, 'failure'),
            (110.0, 'failure'),
            (200.0, 'recovery'),  # pairs with first F → 100
            (300.0, 'failure'),
            (400.0, 'recovery'),  # gap = 100
            (500.0, 'failure'),
            (600.0, 'recovery'),  # gap = 100
        ]
        # Keyword B: clean alternating pattern
        metrics._rate_limit_events_log["clean"] = [
            (100.0, 'failure'), (130.0, 'recovery'),  # gap = 30
            (200.0, 'failure'), (260.0, 'recovery'),  # gap = 60
            (300.0, 'failure'), (390.0, 'recovery'),  # gap = 90
        ]

        messy_result = metrics.get_estimated_rate_limit_window("messy")
        clean_result = metrics.get_estimated_rate_limit_window("clean")

        # Messy: gaps = [100, 100, 100] → median = 100
        assert messy_result == 100.0
        # Clean: gaps = [30, 60, 90] → median = 60
        assert clean_result == 60.0

    def test_trailing_failure_without_recovery(self):
        """Test with a trailing failure that never recovers.

        Pattern: F, R, F, R, F, R, F (last F has no R)
        """
        metrics = RateLimitMetrics()
        events = [
            (100.0, 'failure'), (130.0, 'recovery'),   # gap = 30
            (200.0, 'failure'), (260.0, 'recovery'),   # gap = 60
            (300.0, 'failure'), (390.0, 'recovery'),   # gap = 90
            (500.0, 'failure'),  # Trailing F - no R
        ]
        metrics._rate_limit_events_log["trailing_f"] = events
        result = metrics.get_estimated_rate_limit_window("trailing_f")
        # gaps = [30, 60, 90] → median = 60
        assert result == 60.0
