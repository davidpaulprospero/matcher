"""Tests for RetryQueueStats class (US-32-009)."""

import pytest
from src.downloader.retry_stats import RetryQueueStats


class TestRetryQueueStatsEmpty:
    """Tests for empty queue statistics."""

    def test_retry_stats_empty_queue_returns_zero_counts(self):
        """Empty RetryQueueStats returns zero counts."""
        stats = RetryQueueStats()

        assert stats.pending == 0
        assert stats.total_added == 0
        assert stats.total_retried == 0
        assert len(stats.completed_ids) == 0
        assert len(stats.failed_ids) == 0
        assert stats.current_pass == 0

        summary = stats.get_summary()
        assert summary['pending'] == 0
        assert summary['completed'] == 0
        assert summary['failed'] == 0
        assert summary['total_added'] == 0
        assert summary['total_retried'] == 0


class TestRetryQueueStatsFailureReasons:
    """Tests for failure reason tracking."""

    def test_retry_stats_tracks_failure_reasons_correctly(self):
        """RetryQueueStats correctly tracks and retrieves failure reasons."""
        stats = RetryQueueStats()

        # Record failures
        stats.record_failure("video1", "HTTP 403 Forbidden")
        stats.record_failure("video2", "Network timeout")
        stats.record_failure("video3", "Rate limited")

        # Verify tracking
        reasons = stats.get_failure_reasons()
        assert len(reasons) == 3
        assert reasons["video1"] == "HTTP 403 Forbidden"
        assert reasons["video2"] == "Network timeout"
        assert reasons["video3"] == "Rate limited"

        # Clear one failure (e.g., video succeeded on retry)
        stats.clear_failure("video2")
        reasons = stats.get_failure_reasons()
        assert len(reasons) == 2
        assert "video2" not in reasons

        # Update failure reason for same video
        stats.record_failure("video1", "New error message")
        reasons = stats.get_failure_reasons()
        assert reasons["video1"] == "New error message"


class TestRetryQueueStatsSummary:
    """Tests for summary format."""

    def test_retry_stats_get_summary_format_matches_expected_structure(self):
        """get_summary() returns dict with all expected keys and types."""
        stats = RetryQueueStats()

        # Set some values
        stats.pending = 5
        stats.completed_ids.add("vid1")
        stats.completed_ids.add("vid2")
        stats.failed_ids.add("vid3")
        stats.current_pass = 1
        stats.max_passes = 3
        stats.delay_seconds = 60.0
        stats.total_added = 10
        stats.total_retried = 7
        stats.respect_circuit_breaker = True
        stats.circuit_breaker_wait_time = 45.5
        stats.wait_for_cookie_cooldown = False
        stats.cookie_cooldown_wait_time = 30.123
        stats.max_combined_wait_seconds = 200.0
        stats.forced_retry = True
        stats.budget_state = {"remaining": 5}

        summary = stats.get_summary()

        # Verify all expected keys exist
        expected_keys = {
            'enabled', 'pending', 'completed', 'failed', 'current_pass',
            'max_passes', 'delay_seconds', 'total_added', 'total_retried',
            'respect_circuit_breaker', 'circuit_breaker_wait_time',
            'wait_for_cookie_cooldown', 'cookie_cooldown_wait_time',
            'max_combined_wait_seconds', 'forced_retry', 'budget_state'
        }
        assert set(summary.keys()) == expected_keys

        # Verify values match
        assert summary['enabled'] is True
        assert summary['pending'] == 5
        assert summary['completed'] == 2
        assert summary['failed'] == 1
        assert summary['current_pass'] == 1
        assert summary['max_passes'] == 3
        assert summary['delay_seconds'] == 60.0
        assert summary['total_added'] == 10
        assert summary['total_retried'] == 7
        assert summary['respect_circuit_breaker'] is True
        assert summary['circuit_breaker_wait_time'] == 45.5
        assert summary['wait_for_cookie_cooldown'] is False
        assert summary['cookie_cooldown_wait_time'] == 30.1  # rounded
        assert summary['max_combined_wait_seconds'] == 200.0
        assert summary['forced_retry'] is True
        assert summary['budget_state'] == {"remaining": 5}


class TestRetryQueueStatsMetrics:
    """Tests for metrics increment on add_item simulation."""

    def test_retry_stats_metrics_increment_on_add_item(self):
        """Metrics increment correctly when tracking add operations."""
        stats = RetryQueueStats()

        # Simulate adding items
        stats.total_added += 1
        stats.pending += 1

        assert stats.total_added == 1
        assert stats.pending == 1

        # Add more items
        stats.total_added += 2
        stats.pending += 2

        assert stats.total_added == 3
        assert stats.pending == 3

        # Simulate item completion
        stats.pending -= 1
        stats.completed_ids.add("video1")
        stats.total_retried += 1

        assert stats.pending == 2
        assert len(stats.completed_ids) == 1
        assert stats.total_retried == 1

        # Verify metrics calculation
        metrics = stats.get_retry_metrics()
        assert metrics['total_processed'] == 1  # 1 completed, 0 failed
        assert metrics['retry_efficiency'] == 1.0  # 1/1 = 100% success
        assert metrics['success_rate'] == pytest.approx(0.333, rel=0.01)  # 1/3

        # Add a failure
        stats.pending -= 1
        stats.failed_ids.add("video2")

        metrics = stats.get_retry_metrics()
        assert metrics['total_processed'] == 2  # 1 completed, 1 failed
        assert metrics['retry_efficiency'] == 0.5  # 1/2 = 50% success
