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


# =============================================================================
# US-144-009: Enhanced Retry Queue Metrics Tests
# =============================================================================

class TestKeywordRetryStats:
    """Tests for per-keyword retry success rate tracking."""

    def test_record_keyword_retry(self):
        """record_keyword_retry increments total_retries for keyword."""
        stats = RetryQueueStats()

        stats.record_keyword_retry("python tutorial")
        stats.record_keyword_retry("python tutorial")
        stats.record_keyword_retry("javascript")

        keyword_stats = stats.get_keyword_retry_stats()
        assert keyword_stats["python tutorial"]["total_retries"] == 2
        assert keyword_stats["javascript"]["total_retries"] == 1

    def test_record_keyword_retry_success(self):
        """record_keyword_retry_success increments successes."""
        stats = RetryQueueStats()

        stats.record_keyword_retry_success("python tutorial")
        stats.record_keyword_retry_success("python tutorial")
        stats.record_keyword_retry_success("javascript")

        keyword_stats = stats.get_keyword_retry_stats()
        assert keyword_stats["python tutorial"]["successes"] == 2
        assert keyword_stats["javascript"]["successes"] == 1

    def test_record_keyword_retry_failure(self):
        """record_keyword_retry_failure increments failures."""
        stats = RetryQueueStats()

        stats.record_keyword_retry_failure("python tutorial")
        stats.record_keyword_retry_failure("python tutorial")

        keyword_stats = stats.get_keyword_retry_stats()
        assert keyword_stats["python tutorial"]["failures"] == 2

    def test_keyword_success_rate_calculation(self):
        """get_keyword_retry_stats calculates success rate correctly."""
        stats = RetryQueueStats()

        # 3 successes, 1 failure = 75% success rate
        for _ in range(3):
            stats.record_keyword_retry_success("python tutorial")
        stats.record_keyword_retry_failure("python tutorial")

        keyword_stats = stats.get_keyword_retry_stats()
        assert keyword_stats["python tutorial"]["success_rate"] == 0.75

    def test_keyword_zero_success_rate(self):
        """get_keyword_retry_stats returns 0.0 when all retries failed."""
        stats = RetryQueueStats()

        stats.record_keyword_retry_failure("python tutorial")
        stats.record_keyword_retry_failure("python tutorial")

        keyword_stats = stats.get_keyword_retry_stats()
        assert keyword_stats["python tutorial"]["success_rate"] == 0.0


class TestRetryLatencyHistogram:
    """Tests for retry latency histogram tracking."""

    def test_record_first_failure(self):
        """record_first_failure stores timestamp for video."""
        stats = RetryQueueStats()

        stats.record_first_failure("video1")
        assert "video1" in stats._first_failure_time

    def test_record_retry_success_latency_calculates_time(self):
        """record_retry_success_latency calculates latency from first failure."""
        stats = RetryQueueStats()

        # Record first failure
        stats.record_first_failure("video1")

        # Simulate time passing (in practice this would be real time)
        import time
        time.sleep(0.01)  # Small delay to ensure measurable latency

        # Record success - should calculate latency
        latency = stats.record_retry_success_latency("video1")

        assert latency is not None
        assert latency >= 0.01  # At least 10ms
        assert latency < 1.0  # Less than 1 second

    def test_record_retry_success_latency_returns_none_if_no_first_failure(self):
        """record_retry_success_latency returns None if no first failure recorded."""
        stats = RetryQueueStats()

        latency = stats.record_retry_success_latency("video_unknown")
        assert latency is None

    def test_retry_latency_stats_empty(self):
        """get_retry_latency_stats returns zeros for empty data."""
        stats = RetryQueueStats()

        latency_stats = stats.get_retry_latency_stats()
        assert latency_stats["min"] == 0.0
        assert latency_stats["max"] == 0.0
        assert latency_stats["avg"] == 0.0
        assert latency_stats["median"] == 0.0
        assert latency_stats["sample_count"] == 0


class TestCategoryDistribution:
    """Tests for category distribution metrics."""

    def test_record_retry_by_category(self):
        """record_retry_by_category increments category count."""
        stats = RetryQueueStats()

        stats.record_retry_by_category("rate_limit")
        stats.record_retry_by_category("rate_limit")
        stats.record_retry_by_category("network")

        assert stats.retries_by_category["rate_limit"] == 2
        assert stats.retries_by_category["network"] == 1

    def test_category_distribution_percentages(self):
        """get_category_distribution returns correct percentages."""
        stats = RetryQueueStats()

        stats.record_retry_by_category("rate_limit")
        stats.record_retry_by_category("rate_limit")
        stats.record_retry_by_category("rate_limit")
        stats.record_retry_by_category("network")
        # Total: 4 retries, 3 rate_limit (75%), 1 network (25%)

        distribution = stats.get_category_distribution()
        assert distribution["rate_limit"] == 0.75
        assert distribution["network"] == 0.25

    def test_category_distribution_empty(self):
        """get_category_distribution returns empty dict when no data."""
        stats = RetryQueueStats()

        distribution = stats.get_category_distribution()
        assert distribution == {}


class TestRetryEfficiencyScore:
    """Tests for retry efficiency score."""

    def test_retry_efficiency_score_with_data(self):
        """get_retry_efficiency_score calculates correctly."""
        stats = RetryQueueStats()

        # Simulate: 10 retries for 5 successful downloads = 2.0 efficiency
        stats.completed_ids.add("v1")
        stats.completed_ids.add("v2")
        stats.completed_ids.add("v3")
        stats.completed_ids.add("v4")
        stats.completed_ids.add("v5")
        stats.total_retried = 10

        efficiency = stats.get_retry_efficiency_score()
        assert efficiency == 2.0

    def test_retry_efficiency_score_zero_successes(self):
        """get_retry_efficiency_score returns 0.0 when no successes."""
        stats = RetryQueueStats()

        stats.total_retried = 10

        efficiency = stats.get_retry_efficiency_score()
        assert efficiency == 0.0

    def test_retry_efficiency_score_zero_retries(self):
        """get_retry_efficiency_score returns 0.0 when no retries."""
        stats = RetryQueueStats()

        stats.completed_ids.add("v1")

        efficiency = stats.get_retry_efficiency_score()
        assert efficiency == 0.0


class TestEnhancedMetrics:
    """Tests for combined enhanced metrics."""

    def test_get_enhanced_metrics_returns_all_metrics(self):
        """get_enhanced_metrics returns all new metric groups."""
        stats = RetryQueueStats()

        # Add some data
        stats.record_keyword_retry("python")
        stats.record_keyword_retry_success("python")
        stats.record_retry_by_category("rate_limit")
        stats.record_first_failure("v1")
        stats.completed_ids.add("v1")
        stats.total_retried = 1

        enhanced = stats.get_enhanced_metrics()

        assert "keyword_retry_stats" in enhanced
        assert "retry_latency" in enhanced
        assert "category_distribution" in enhanced
        assert "retry_efficiency_score" in enhanced

    def test_get_retry_metrics_includes_enhanced_metrics(self):
        """get_retry_metrics includes enhanced metrics in output."""
        stats = RetryQueueStats()

        stats.record_keyword_retry("python")
        stats.record_retry_by_category("rate_limit")

        metrics = stats.get_retry_metrics()

        assert "keyword_retry_stats" in metrics
        assert "retry_latency" in metrics
        assert "category_distribution" in metrics
        assert "retry_efficiency_score" in metrics


class TestResetClearsEnhancedMetrics:
    """Tests that reset clears enhanced metrics."""

    def test_reset_clears_keyword_retry_stats(self):
        """reset clears keyword retry stats."""
        stats = RetryQueueStats()

        stats.record_keyword_retry("python")
        stats.reset()

        assert stats._keyword_retry_stats == {}

    def test_reset_clears_retry_latencies(self):
        """reset clears retry latencies."""
        stats = RetryQueueStats()

        stats.record_first_failure("v1")
        stats.reset()

        assert stats._first_failure_time == {}
        assert stats.retry_latencies == []

    def test_reset_clears_category_distribution(self):
        """reset clears category distribution."""
        stats = RetryQueueStats()

        stats.record_retry_by_category("rate_limit")
        stats.reset()

        assert stats.retries_by_category == {}
