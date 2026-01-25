"""Tests for RateLimitMetrics class and integration.

Tests US-010: Add rate limiting metrics to pipeline report.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.downloader.rate_limit_metrics import RateLimitMetrics


class TestRateLimitMetricsInit:
    """Test RateLimitMetrics initialization."""

    def test_default_values(self):
        """Test default metric values are zero."""
        metrics = RateLimitMetrics()
        assert metrics.total_downloads == 0
        assert metrics.successful_downloads == 0
        assert metrics.failed_downloads == 0
        assert metrics.retry_attempts == 0
        assert metrics.rate_limit_events == 0
        assert metrics.backoff_attempts == 0
        assert metrics.time_spent_backing_off == 0.0
        assert metrics.cookie_rotations == 0
        assert metrics.vpn_switches == 0

    def test_custom_values(self):
        """Test creating metrics with custom values."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=90,
            failed_downloads=10,
            rate_limit_events=5
        )
        assert metrics.total_downloads == 100
        assert metrics.successful_downloads == 90
        assert metrics.failed_downloads == 10
        assert metrics.rate_limit_events == 5


class TestDownloadRecording:
    """Test download metrics recording."""

    def test_record_download_attempt(self):
        """Test recording a download attempt."""
        metrics = RateLimitMetrics()
        metrics.record_download_attempt()
        assert metrics.total_downloads == 1

    def test_record_download_success(self):
        """Test recording a successful download."""
        metrics = RateLimitMetrics()
        metrics.record_download_attempt()
        metrics.record_download_success()
        assert metrics.total_downloads == 1
        assert metrics.successful_downloads == 1
        assert metrics.failed_downloads == 0

    def test_record_download_failure(self):
        """Test recording a failed download."""
        metrics = RateLimitMetrics()
        metrics.record_download_attempt()
        metrics.record_download_failure()
        assert metrics.total_downloads == 1
        assert metrics.successful_downloads == 0
        assert metrics.failed_downloads == 1

    def test_multiple_downloads(self):
        """Test recording multiple downloads."""
        metrics = RateLimitMetrics()
        # 3 downloads: 2 success, 1 failure
        for _ in range(3):
            metrics.record_download_attempt()
        metrics.record_download_success()
        metrics.record_download_success()
        metrics.record_download_failure()

        assert metrics.total_downloads == 3
        assert metrics.successful_downloads == 2
        assert metrics.failed_downloads == 1


class TestRetryRecording:
    """Test retry metrics recording."""

    def test_record_retry(self):
        """Test recording a retry attempt."""
        metrics = RateLimitMetrics()
        metrics.record_retry('timeout')
        assert metrics.retry_attempts == 1
        assert metrics.retries_by_error_type == {'timeout': 1}

    def test_record_multiple_retries_same_type(self):
        """Test recording multiple retries of same type."""
        metrics = RateLimitMetrics()
        metrics.record_retry('transient')
        metrics.record_retry('transient')
        metrics.record_retry('transient')
        assert metrics.retry_attempts == 3
        assert metrics.retries_by_error_type == {'transient': 3}

    def test_record_retries_different_types(self):
        """Test recording retries of different types."""
        metrics = RateLimitMetrics()
        metrics.record_retry('timeout')
        metrics.record_retry('transient')
        metrics.record_retry('network')
        metrics.record_retry('transient')

        assert metrics.retry_attempts == 4
        assert metrics.retries_by_error_type == {
            'timeout': 1,
            'transient': 2,
            'network': 1
        }

    def test_record_retries_exhausted(self):
        """Test recording retries exhaustion."""
        metrics = RateLimitMetrics()
        metrics.record_retries_exhausted()
        assert metrics.max_retry_count_reached == 1


class TestRateLimitRecording:
    """Test rate limit metrics recording."""

    def test_record_rate_limit_event(self):
        """Test recording a rate limit event."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event()
        assert metrics.rate_limit_events == 1

    def test_record_backoff(self):
        """Test recording a backoff delay."""
        metrics = RateLimitMetrics()
        metrics.record_backoff(5.0)
        assert metrics.backoff_attempts == 1
        assert metrics.time_spent_backing_off == 5.0

    def test_record_multiple_backoffs(self):
        """Test recording multiple backoff delays."""
        metrics = RateLimitMetrics()
        metrics.record_backoff(5.0)
        metrics.record_backoff(10.0)
        metrics.record_backoff(20.0)
        assert metrics.backoff_attempts == 3
        assert metrics.time_spent_backing_off == 35.0

    def test_record_cookie_rotation(self):
        """Test recording a cookie rotation."""
        metrics = RateLimitMetrics()
        metrics.record_cookie_rotation()
        assert metrics.cookie_rotations == 1

    def test_record_vpn_switch(self):
        """Test recording a VPN switch."""
        metrics = RateLimitMetrics()
        metrics.record_vpn_switch()
        assert metrics.vpn_switches == 1


class TestCircuitBreakerRecording:
    """Test circuit breaker metrics recording."""

    def test_record_circuit_breaker_trip(self):
        """Test recording a circuit breaker trip."""
        metrics = RateLimitMetrics()
        metrics.record_circuit_breaker_trip(60.0)
        assert metrics.circuit_breaker_trips == 1
        assert metrics.circuit_breaker_pause_seconds == 60.0

    def test_record_multiple_trips(self):
        """Test recording multiple circuit breaker trips."""
        metrics = RateLimitMetrics()
        metrics.record_circuit_breaker_trip(60.0)
        metrics.record_circuit_breaker_trip(60.0)
        assert metrics.circuit_breaker_trips == 2
        assert metrics.circuit_breaker_pause_seconds == 120.0


class TestBatchRetryRecording:
    """Test batch retry metrics recording."""

    def test_record_batch_retry_pass(self):
        """Test recording a batch retry pass."""
        metrics = RateLimitMetrics()
        metrics.record_batch_retry_pass(successes=5, failures=2)
        assert metrics.batch_retry_passes == 1
        assert metrics.batch_retry_successes == 5
        assert metrics.batch_retry_failures == 2

    def test_record_multiple_passes(self):
        """Test recording multiple batch retry passes."""
        metrics = RateLimitMetrics()
        metrics.record_batch_retry_pass(successes=5, failures=3)
        metrics.record_batch_retry_pass(successes=2, failures=1)
        assert metrics.batch_retry_passes == 2
        assert metrics.batch_retry_successes == 7
        assert metrics.batch_retry_failures == 4


class TestSpeedRecording:
    """Test speed metrics recording."""

    def test_record_speed_sample(self):
        """Test recording a speed sample."""
        metrics = RateLimitMetrics()
        metrics.record_speed_sample(5.0)
        assert metrics.speed_samples == 1
        assert metrics.avg_speed_mbps == 5.0

    def test_record_multiple_speed_samples(self):
        """Test recording multiple speed samples uses running average."""
        metrics = RateLimitMetrics()
        metrics.record_speed_sample(10.0)  # avg = 10
        metrics.record_speed_sample(20.0)  # avg = (10 + 20) / 2 = 15
        metrics.record_speed_sample(15.0)  # avg = (15 + (15-15)/3) = 15
        assert metrics.speed_samples == 3
        assert metrics.avg_speed_mbps == 15.0

    def test_record_timeout_extension(self):
        """Test recording a timeout extension."""
        metrics = RateLimitMetrics()
        metrics.record_timeout_extension()
        assert metrics.timeout_extensions == 1


class TestCalculatedProperties:
    """Test calculated metric properties."""

    def test_avg_retry_count_no_downloads(self):
        """Test average retry count with no downloads."""
        metrics = RateLimitMetrics()
        assert metrics.avg_retry_count == 0.0

    def test_avg_retry_count_with_downloads(self):
        """Test average retry count with downloads."""
        metrics = RateLimitMetrics(total_downloads=10, retry_attempts=20)
        assert metrics.avg_retry_count == 2.0

    def test_rate_limit_percentage_no_downloads(self):
        """Test rate limit percentage with no downloads."""
        metrics = RateLimitMetrics()
        assert metrics.rate_limit_percentage == 0.0

    def test_rate_limit_percentage_with_downloads(self):
        """Test rate limit percentage with downloads."""
        metrics = RateLimitMetrics(total_downloads=100, rate_limit_events=15)
        assert metrics.rate_limit_percentage == 15.0

    def test_success_rate_no_downloads(self):
        """Test success rate with no downloads."""
        metrics = RateLimitMetrics()
        assert metrics.success_rate == 0.0

    def test_success_rate_with_downloads(self):
        """Test success rate with downloads."""
        metrics = RateLimitMetrics(total_downloads=100, successful_downloads=85)
        assert metrics.success_rate == 85.0


class TestSubsystemIntegration:
    """Test integration with subsystem stats."""

    def test_update_from_circuit_breaker_enabled(self):
        """Test updating from circuit breaker stats when enabled."""
        metrics = RateLimitMetrics()
        stats = {
            'enabled': True,
            'total_trips': 3,
            'total_paused_seconds': 180.0
        }
        metrics.update_from_circuit_breaker(stats)
        assert metrics.circuit_breaker_trips == 3
        assert metrics.circuit_breaker_pause_seconds == 180.0

    def test_update_from_circuit_breaker_disabled(self):
        """Test updating from circuit breaker stats when disabled."""
        metrics = RateLimitMetrics()
        stats = {'enabled': False}
        metrics.update_from_circuit_breaker(stats)
        assert metrics.circuit_breaker_trips == 0

    def test_update_from_retry_queue_enabled(self):
        """Test updating from retry queue stats when enabled."""
        metrics = RateLimitMetrics()
        stats = {
            'enabled': True,
            'current_pass': 2,
            'total_retried': 10,
            'failed': 3
        }
        metrics.update_from_retry_queue(stats)
        assert metrics.batch_retry_passes == 2
        assert metrics.batch_retry_successes == 10
        assert metrics.batch_retry_failures == 3

    def test_update_from_retry_queue_disabled(self):
        """Test updating from retry queue stats when disabled."""
        metrics = RateLimitMetrics()
        stats = {'enabled': False}
        metrics.update_from_retry_queue(stats)
        assert metrics.batch_retry_passes == 0

    def test_update_from_speed_tracker(self):
        """Test updating from speed tracker stats."""
        metrics = RateLimitMetrics()
        stats = {
            'samples': 25,
            'avg_speed_mbps': 3.5
        }
        metrics.update_from_speed_tracker(stats)
        assert metrics.speed_samples == 25
        assert metrics.avg_speed_mbps == 3.5


class TestConfigRecommendations:
    """Test config recommendation generation."""

    def test_no_recommendations_when_no_issues(self):
        """Test no recommendations for healthy metrics."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=95,
            rate_limit_events=5,  # 5% < 10% threshold
            cookie_rotations=1  # Has cookie rotation enabled, so no recommendation
        )
        recommendations = metrics.get_config_recommendations()
        assert len(recommendations) == 0

    def test_recommendation_high_rate_limiting(self):
        """Test recommendation for high rate limiting (>10%)."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            rate_limit_events=15  # 15% > 10% threshold
        )
        recommendations = metrics.get_config_recommendations()
        assert len(recommendations) >= 1
        assert 'rate limiting' in recommendations[0].lower()

    def test_recommendation_high_retry_exhaustion(self):
        """Test recommendation for high retry exhaustion rate."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            max_retry_count_reached=25  # 25% > 20% threshold
        )
        recommendations = metrics.get_config_recommendations()
        assert len(recommendations) >= 1
        assert any('retry' in r.lower() for r in recommendations)

    def test_recommendation_many_circuit_breaker_trips(self):
        """Test recommendation for frequent circuit breaker trips."""
        metrics = RateLimitMetrics(
            circuit_breaker_trips=5  # > 3 trips
        )
        recommendations = metrics.get_config_recommendations()
        assert len(recommendations) >= 1
        assert any('circuit' in r.lower() for r in recommendations)

    def test_recommendation_no_cookie_rotation(self):
        """Test recommendation when rate limits but no cookie rotation."""
        metrics = RateLimitMetrics(
            rate_limit_events=10,
            cookie_rotations=0
        )
        recommendations = metrics.get_config_recommendations()
        assert len(recommendations) >= 1
        assert any('cookie' in r.lower() for r in recommendations)

    def test_recommendation_many_timeout_extensions(self):
        """Test recommendation for many timeout extensions."""
        metrics = RateLimitMetrics(
            timeout_extensions=10  # > 5
        )
        recommendations = metrics.get_config_recommendations()
        assert len(recommendations) >= 1
        assert any('timeout' in r.lower() for r in recommendations)

    def test_recommendation_low_batch_retry_recovery(self):
        """Test recommendation for low batch retry recovery rate."""
        metrics = RateLimitMetrics(
            batch_retry_passes=2,
            batch_retry_successes=2,
            batch_retry_failures=8  # 20% recovery rate < 30%
        )
        recommendations = metrics.get_config_recommendations()
        assert len(recommendations) >= 1
        assert any('batch' in r.lower() for r in recommendations)


class TestSummary:
    """Test summary generation."""

    def test_summary_empty_metrics(self):
        """Test summary with no metrics recorded."""
        metrics = RateLimitMetrics()
        summary = metrics.summary()
        assert 'Downloads: 0 total' in summary

    def test_summary_with_downloads(self):
        """Test summary with downloads."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=90,
            failed_downloads=10
        )
        summary = metrics.summary()
        assert 'Downloads: 100 total' in summary
        assert '90 successful' in summary
        assert '10 failed' in summary
        assert 'Success rate: 90.0%' in summary

    def test_summary_with_retries(self):
        """Test summary with retries."""
        metrics = RateLimitMetrics(
            total_downloads=50,
            retry_attempts=25,
            retries_by_error_type={'timeout': 15, 'transient': 10}
        )
        summary = metrics.summary()
        assert 'Retries: 25 total' in summary
        assert 'timeout: 15' in summary
        assert 'transient: 10' in summary

    def test_summary_with_rate_limiting(self):
        """Test summary with rate limiting metrics."""
        metrics = RateLimitMetrics(
            rate_limit_events=5,
            backoff_attempts=3,
            time_spent_backing_off=45.0
        )
        summary = metrics.summary()
        assert 'Rate limiting: 5 events' in summary
        assert '3 backoffs' in summary
        assert '45.0s total' in summary

    def test_summary_with_escalations(self):
        """Test summary with cookie/VPN escalations."""
        metrics = RateLimitMetrics(
            cookie_rotations=3,
            vpn_switches=2
        )
        summary = metrics.summary()
        assert 'Escalations:' in summary
        assert 'cookie rotations: 3' in summary
        assert 'VPN switches: 2' in summary

    def test_summary_with_circuit_breaker(self):
        """Test summary with circuit breaker activity."""
        metrics = RateLimitMetrics(
            circuit_breaker_trips=2,
            circuit_breaker_pause_seconds=120.0
        )
        summary = metrics.summary()
        assert 'Circuit breaker: 2 trips' in summary
        assert '120.0s paused' in summary

    def test_summary_with_batch_retry(self):
        """Test summary with batch retry activity."""
        metrics = RateLimitMetrics(
            batch_retry_passes=2,
            batch_retry_successes=5,
            batch_retry_failures=3
        )
        summary = metrics.summary()
        assert 'Batch retry: 2 passes' in summary
        assert '5 recovered' in summary
        assert '3 failed' in summary

    def test_summary_with_speed_data(self):
        """Test summary with speed data."""
        metrics = RateLimitMetrics(
            speed_samples=10,
            avg_speed_mbps=5.5,
            timeout_extensions=3
        )
        summary = metrics.summary()
        assert 'Network: 5.50 MB/s avg' in summary
        assert '10 samples' in summary
        assert '3 timeout extensions' in summary


class TestPersistence:
    """Test checkpoint persistence."""

    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=90,
            rate_limit_events=10,
            retries_by_error_type={'timeout': 5}
        )
        data = metrics.to_dict()
        assert data['total_downloads'] == 100
        assert data['successful_downloads'] == 90
        assert data['rate_limit_events'] == 10
        assert data['retries_by_error_type'] == {'timeout': 5}

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            'total_downloads': 100,
            'successful_downloads': 90,
            'failed_downloads': 10,
            'rate_limit_events': 5
        }
        metrics = RateLimitMetrics.from_dict(data)
        assert metrics.total_downloads == 100
        assert metrics.successful_downloads == 90
        assert metrics.failed_downloads == 10
        assert metrics.rate_limit_events == 5

    def test_from_dict_empty(self):
        """Test deserialization from empty dict."""
        metrics = RateLimitMetrics.from_dict({})
        assert metrics.total_downloads == 0
        assert metrics.rate_limit_events == 0

    def test_from_dict_none(self):
        """Test deserialization from None."""
        metrics = RateLimitMetrics.from_dict(None)
        assert metrics.total_downloads == 0

    def test_from_dict_missing_fields(self):
        """Test deserialization handles missing fields."""
        data = {'total_downloads': 50}  # Most fields missing
        metrics = RateLimitMetrics.from_dict(data)
        assert metrics.total_downloads == 50
        assert metrics.successful_downloads == 0  # Default
        assert metrics.rate_limit_events == 0  # Default

    def test_roundtrip(self):
        """Test serialization roundtrip preserves data."""
        original = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=90,
            failed_downloads=10,
            retry_attempts=25,
            retries_by_error_type={'timeout': 15, 'transient': 10},
            rate_limit_events=8,
            backoff_attempts=5,
            time_spent_backing_off=75.0,
            cookie_rotations=2,
            vpn_switches=1,
            circuit_breaker_trips=1,
            circuit_breaker_pause_seconds=60.0,
            batch_retry_passes=2,
            batch_retry_successes=5,
            batch_retry_failures=3,
            speed_samples=50,
            avg_speed_mbps=3.5,
            timeout_extensions=2
        )
        data = original.to_dict()
        restored = RateLimitMetrics.from_dict(data)

        assert restored.total_downloads == original.total_downloads
        assert restored.successful_downloads == original.successful_downloads
        assert restored.rate_limit_events == original.rate_limit_events
        assert restored.retries_by_error_type == original.retries_by_error_type
        assert restored.time_spent_backing_off == original.time_spent_backing_off
        assert restored.cookie_rotations == original.cookie_rotations


class TestClear:
    """Test metrics clearing."""

    def test_clear_resets_all_metrics(self):
        """Test clear() resets all metrics to defaults."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=90,
            rate_limit_events=10
        )
        metrics.clear()

        assert metrics.total_downloads == 0
        assert metrics.successful_downloads == 0
        assert metrics.rate_limit_events == 0
        assert metrics.retries_by_error_type == {}


class TestVideoDownloaderIntegration:
    """Test integration with VideoDownloader."""

    def test_metrics_available_from_downloader(self):
        """Test that RateLimitMetrics can be imported from downloader."""
        from src.downloader import RateLimitMetrics
        metrics = RateLimitMetrics()
        assert metrics.total_downloads == 0

    def test_metrics_in_downloader_all(self):
        """Test that RateLimitMetrics is in __all__."""
        from src.downloader import __all__
        assert 'RateLimitMetrics' in __all__


class TestHealingOrchestratorIntegration:
    """Test integration with HealingOrchestrator."""

    def test_set_rate_limit_metrics(self):
        """Test setting rate limit metrics on orchestrator."""
        from src.agents.orchestrator import HealingOrchestrator
        from unittest.mock import MagicMock, patch

        # Create minimal mock config
        mock_config = MagicMock()
        mock_config.healing = None

        with patch('src.agents.orchestrator.HEALER_REGISTRY', []):
            orchestrator = HealingOrchestrator(
                config=mock_config,
                project_dir='/tmp/test'
            )

        metrics = RateLimitMetrics(total_downloads=100, rate_limit_events=10)
        orchestrator.set_rate_limit_metrics(metrics)

        assert orchestrator._rate_limit_metrics == metrics

    def test_print_report_includes_rate_limiting(self, capsys):
        """Test that print_report includes rate limiting section."""
        from src.agents.orchestrator import HealingOrchestrator
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.healing = None

        with patch('src.agents.orchestrator.HEALER_REGISTRY', []):
            orchestrator = HealingOrchestrator(
                config=mock_config,
                project_dir='/tmp/test'
            )

        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=80,
            rate_limit_events=15
        )
        orchestrator.set_rate_limit_metrics(metrics)
        orchestrator.print_report()

        captured = capsys.readouterr()
        assert 'RATE LIMITING' in captured.out
        assert 'Downloads: 100 total' in captured.out
        assert 'Success rate: 80.0%' in captured.out

    def test_print_report_includes_recommendations(self, capsys):
        """Test that print_report includes recommendations when warranted."""
        from src.agents.orchestrator import HealingOrchestrator
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.healing = None

        with patch('src.agents.orchestrator.HEALER_REGISTRY', []):
            orchestrator = HealingOrchestrator(
                config=mock_config,
                project_dir='/tmp/test'
            )

        # Create metrics that trigger recommendations
        metrics = RateLimitMetrics(
            total_downloads=100,
            rate_limit_events=20,  # 20% > 10% threshold
            cookie_rotations=0
        )
        orchestrator.set_rate_limit_metrics(metrics)
        orchestrator.print_report()

        captured = capsys.readouterr()
        assert 'Recommendations:' in captured.out

    def test_reset_clears_rate_limit_metrics(self):
        """Test that reset clears rate limit metrics."""
        from src.agents.orchestrator import HealingOrchestrator
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.healing = None

        with patch('src.agents.orchestrator.HEALER_REGISTRY', []):
            orchestrator = HealingOrchestrator(
                config=mock_config,
                project_dir='/tmp/test'
            )

        metrics = RateLimitMetrics(total_downloads=100)
        orchestrator.set_rate_limit_metrics(metrics)
        assert orchestrator._rate_limit_metrics is not None

        orchestrator.reset()
        assert orchestrator._rate_limit_metrics is None


class TestKeywordRateLimitTracking:
    """Test per-keyword rate limit event tracking (US-009)."""

    def test_record_rate_limit_event_with_keyword(self):
        """Test recording rate limit event with keyword parameter."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(keyword="sunset beach")
        assert metrics.rate_limit_events == 1
        assert metrics.keyword_rate_limit_events == {"sunset beach": 1}

    def test_record_rate_limit_event_with_tier_and_keyword(self):
        """Test recording rate limit event with both tier and keyword."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(tier="long", keyword="drone footage")
        assert metrics.rate_limit_events == 1
        assert metrics.tier_rate_limit_events == {"long": 1}
        assert metrics.keyword_rate_limit_events == {"drone footage": 1}

    def test_record_multiple_events_same_keyword(self):
        """Test recording multiple events for the same keyword."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(keyword="nature documentary")
        metrics.record_rate_limit_event(keyword="nature documentary")
        metrics.record_rate_limit_event(keyword="nature documentary")
        assert metrics.rate_limit_events == 3
        assert metrics.keyword_rate_limit_events == {"nature documentary": 3}

    def test_record_events_different_keywords(self):
        """Test recording events for different keywords."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(keyword="sunset")
        metrics.record_rate_limit_event(keyword="ocean waves")
        metrics.record_rate_limit_event(keyword="sunset")
        metrics.record_rate_limit_event(keyword="mountain landscape")
        assert metrics.rate_limit_events == 4
        assert metrics.keyword_rate_limit_events == {
            "sunset": 2,
            "ocean waves": 1,
            "mountain landscape": 1
        }

    def test_record_event_without_keyword_does_not_add_to_dict(self):
        """Test that events without keyword don't add to keyword dict."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event()
        metrics.record_rate_limit_event(tier="short")
        assert metrics.rate_limit_events == 2
        assert metrics.keyword_rate_limit_events == {}

    def test_keyword_events_in_to_dict(self):
        """Test that keyword events are serialized to dict."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(keyword="test keyword")
        data = metrics.to_dict()
        assert data["keyword_rate_limit_events"] == {"test keyword": 1}

    def test_keyword_events_from_dict(self):
        """Test that keyword events are restored from dict."""
        data = {
            "rate_limit_events": 5,
            "keyword_rate_limit_events": {"sunset": 3, "ocean": 2}
        }
        metrics = RateLimitMetrics.from_dict(data)
        assert metrics.rate_limit_events == 5
        assert metrics.keyword_rate_limit_events == {"sunset": 3, "ocean": 2}

    def test_keyword_events_from_dict_missing(self):
        """Test backward compatibility when keyword_rate_limit_events is missing."""
        data = {"rate_limit_events": 5}
        metrics = RateLimitMetrics.from_dict(data)
        assert metrics.rate_limit_events == 5
        assert metrics.keyword_rate_limit_events == {}

    def test_keyword_events_cleared(self):
        """Test that clear() resets keyword events."""
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(keyword="test")
        metrics.clear()
        assert metrics.keyword_rate_limit_events == {}

    def test_keyword_events_roundtrip(self):
        """Test keyword events survive checkpoint roundtrip."""
        original = RateLimitMetrics()
        original.record_rate_limit_event(keyword="keyword1")
        original.record_rate_limit_event(keyword="keyword1")
        original.record_rate_limit_event(keyword="keyword2")

        data = original.to_dict()
        restored = RateLimitMetrics.from_dict(data)

        assert restored.keyword_rate_limit_events == {"keyword1": 2, "keyword2": 1}


class TestKeywordSummaryDisplay:
    """Test keyword display in summary when > 10 total events (US-009)."""

    def test_summary_no_keywords_shown_under_10_events(self):
        """Test that keywords are not shown when <= 10 total events."""
        metrics = RateLimitMetrics()
        for i in range(10):
            metrics.record_rate_limit_event(keyword=f"keyword{i}")
        summary = metrics.summary()
        assert "Top keywords:" not in summary

    def test_summary_shows_keywords_over_10_events(self):
        """Test that top keywords are shown when > 10 total events."""
        metrics = RateLimitMetrics()
        for _ in range(8):
            metrics.record_rate_limit_event(keyword="problematic keyword")
        for _ in range(4):
            metrics.record_rate_limit_event(keyword="other keyword")
        summary = metrics.summary()
        assert "Top keywords:" in summary
        assert '"problematic keyword": 8' in summary

    def test_summary_shows_top_5_keywords_only(self):
        """Test that only top 5 keywords are shown (sorted by count)."""
        metrics = RateLimitMetrics()
        # Create 7 keywords with different counts
        keywords = [
            ("kw1", 10), ("kw2", 8), ("kw3", 6), ("kw4", 5),
            ("kw5", 4), ("kw6", 3), ("kw7", 2)
        ]
        for kw, count in keywords:
            for _ in range(count):
                metrics.record_rate_limit_event(keyword=kw)

        summary = metrics.summary()
        assert "Top keywords:" in summary
        # Top 5 should be shown
        assert '"kw1": 10' in summary
        assert '"kw2": 8' in summary
        assert '"kw3": 6' in summary
        assert '"kw4": 5' in summary
        assert '"kw5": 4' in summary
        # Bottom 2 should not be shown
        assert '"kw6": 3' not in summary
        assert '"kw7": 2' not in summary

    def test_summary_keywords_sorted_by_count_descending(self):
        """Test that keywords are sorted by count in descending order."""
        metrics = RateLimitMetrics()
        # Add in random order
        for _ in range(3):
            metrics.record_rate_limit_event(keyword="low")
        for _ in range(7):
            metrics.record_rate_limit_event(keyword="high")
        for _ in range(5):
            metrics.record_rate_limit_event(keyword="medium")

        summary = metrics.summary()
        lines = summary.split('\n')
        keyword_line = [l for l in lines if "Top keywords:" in l][0]
        # "high" (7) should appear before "medium" (5) before "low" (3)
        high_pos = keyword_line.find('"high"')
        medium_pos = keyword_line.find('"medium"')
        low_pos = keyword_line.find('"low"')
        assert high_pos < medium_pos < low_pos


class TestKeywordConfigRecommendations:
    """Test config recommendations for high-rate-limit keywords (US-009)."""

    def test_no_recommendation_under_10_events(self):
        """Test no keyword recommendation when < 10 total events."""
        metrics = RateLimitMetrics()
        for _ in range(9):
            metrics.record_rate_limit_event(keyword="problematic")
        recommendations = metrics.get_config_recommendations()
        assert not any("keyword" in r.lower() and "rate limit" in r.lower()
                      for r in recommendations)

    def test_no_recommendation_when_no_high_rate_keyword(self):
        """Test no recommendation when no keyword exceeds 30% threshold."""
        metrics = RateLimitMetrics()
        # Distribute events evenly - no single keyword > 30%
        for i in range(12):
            metrics.record_rate_limit_event(keyword=f"kw{i % 6}")
        # Each keyword has 2 events, total 12, so each is 16.6% < 30%
        recommendations = metrics.get_config_recommendations()
        keyword_recs = [r for r in recommendations
                       if "keyword" in r.lower() and "rate limit" in r.lower()]
        assert len(keyword_recs) == 0

    def test_recommendation_for_high_rate_limit_keyword(self):
        """Test recommendation when a keyword exceeds 30% threshold."""
        metrics = RateLimitMetrics()
        # 8 events for "problematic", 4 for others = 8/12 = 66% > 30%
        for _ in range(8):
            metrics.record_rate_limit_event(keyword="problematic keyword")
        for _ in range(4):
            metrics.record_rate_limit_event(keyword="other")

        recommendations = metrics.get_config_recommendations()
        keyword_recs = [r for r in recommendations
                       if "problematic keyword" in r.lower()]
        assert len(keyword_recs) == 1
        assert "skip_keywords" in keyword_recs[0]

    def test_recommendation_shows_multiple_high_rate_keywords(self):
        """Test recommendation shows up to 3 high-rate-limit keywords."""
        metrics = RateLimitMetrics()
        # 10 events for "bad1", 8 for "bad2", 6 for "bad3", 2 for "ok"
        # Total = 26, threshold = 7.8 (30%)
        for _ in range(10):
            metrics.record_rate_limit_event(keyword="bad1")
        for _ in range(8):
            metrics.record_rate_limit_event(keyword="bad2")
        for _ in range(6):
            metrics.record_rate_limit_event(keyword="bad3")
        for _ in range(2):
            metrics.record_rate_limit_event(keyword="ok")

        recommendations = metrics.get_config_recommendations()
        keyword_recs = [r for r in recommendations
                       if "bad1" in r or "bad2" in r]
        assert len(keyword_recs) >= 1
        # Should mention bad1 and bad2 (both > 30%)
        rec_text = keyword_recs[0]
        assert '"bad1"' in rec_text
        assert '"bad2"' in rec_text

    def test_recommendation_suggests_skip_keywords_config(self):
        """Test that recommendation suggests keyword.skip_keywords config."""
        metrics = RateLimitMetrics()
        for _ in range(15):
            metrics.record_rate_limit_event(keyword="problematic")

        recommendations = metrics.get_config_recommendations()
        keyword_recs = [r for r in recommendations if "problematic" in r.lower()]
        assert len(keyword_recs) == 1
        assert "skip_keywords" in keyword_recs[0]


class TestKeywordCrossSessionPersistence:
    """Test keyword events are preserved across sessions (US-009)."""

    def test_from_checkpoint_preserves_keyword_events(self):
        """Test that from_checkpoint() preserves keyword events."""
        data = {
            "rate_limit_events": 15,
            "keyword_rate_limit_events": {"sunset": 10, "ocean": 5},
            "session_count": 1
        }
        metrics = RateLimitMetrics.from_checkpoint(data)

        assert metrics.keyword_rate_limit_events == {"sunset": 10, "ocean": 5}
        assert metrics.session_count == 2  # Incremented

    def test_keyword_events_accumulate_across_sessions(self):
        """Test that keyword events accumulate across multiple sessions."""
        # Session 1: Record some events
        session1 = RateLimitMetrics()
        session1.record_rate_limit_event(keyword="sunrise")
        session1.record_rate_limit_event(keyword="sunrise")
        session1.record_rate_limit_event(keyword="sunset")

        # Save to checkpoint
        checkpoint_data = session1.to_dict()

        # Session 2: Restore and continue
        session2 = RateLimitMetrics.from_checkpoint(checkpoint_data)
        session2.record_rate_limit_event(keyword="sunrise")
        session2.record_rate_limit_event(keyword="mountain")

        # Verify accumulation
        assert session2.keyword_rate_limit_events == {
            "sunrise": 3,  # 2 from session1 + 1 from session2
            "sunset": 1,   # 1 from session1
            "mountain": 1  # 1 from session2
        }
        assert session2.rate_limit_events == 5
        assert session2.session_count == 2


class TestExportToJson:
    """Test JSON export functionality (US-012)."""

    def test_export_to_json_returns_dict(self):
        """Test that export_to_json returns a dict."""
        metrics = RateLimitMetrics()
        export = metrics.export_to_json()
        assert isinstance(export, dict)

    def test_export_schema_version(self):
        """Test that export includes schema version."""
        metrics = RateLimitMetrics()
        export = metrics.export_to_json()
        assert export["schema_version"] == "1.0"

    def test_export_timestamp(self):
        """Test that export includes export timestamp."""
        metrics = RateLimitMetrics()
        export = metrics.export_to_json()
        assert "export_timestamp" in export
        # Should be ISO format
        assert "T" in export["export_timestamp"]

    def test_export_session_info(self):
        """Test that export includes session information."""
        metrics = RateLimitMetrics(session_count=3)
        metrics.session_start_time = "2026-01-25T10:00:00Z"
        metrics.session_end_time = "2026-01-25T12:00:00Z"

        export = metrics.export_to_json()
        assert "session" in export
        assert export["session"]["session_count"] == 3
        assert export["session"]["session_start_time"] == "2026-01-25T10:00:00Z"
        assert export["session"]["session_end_time"] == "2026-01-25T12:00:00Z"

    def test_export_downloads_section(self):
        """Test that export includes download statistics."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=90,
            failed_downloads=10
        )
        export = metrics.export_to_json()

        assert "downloads" in export
        assert export["downloads"]["total"] == 100
        assert export["downloads"]["successful"] == 90
        assert export["downloads"]["failed"] == 10
        assert export["downloads"]["success_rate_percent"] == 90.0

    def test_export_retries_section(self):
        """Test that export includes retry statistics."""
        metrics = RateLimitMetrics(
            retry_attempts=25,
            total_downloads=50,
            max_retry_count_reached=5,
            retries_by_error_type={"timeout": 15, "transient": 10}
        )
        export = metrics.export_to_json()

        assert "retries" in export
        assert export["retries"]["total_attempts"] == 25
        assert export["retries"]["avg_per_download"] == 0.5
        assert export["retries"]["max_reached_count"] == 5
        assert export["retries"]["by_error_type"] == {"timeout": 15, "transient": 10}

    def test_export_rate_limiting_section(self):
        """Test that export includes rate limiting statistics."""
        metrics = RateLimitMetrics(
            rate_limit_events=15,
            total_downloads=100,
            tier_rate_limit_events={"short": 5, "long": 10},
            keyword_rate_limit_events={"sunset": 8, "ocean": 7},
            backoff_attempts=10,
            time_spent_backing_off=120.567,
            backoff_events_by_severity={"low": 3, "medium": 5, "high": 2}
        )
        export = metrics.export_to_json()

        assert "rate_limiting" in export
        rl = export["rate_limiting"]
        assert rl["total_events"] == 15
        assert rl["percentage_of_downloads"] == 15.0
        assert rl["by_tier"] == {"short": 5, "long": 10}
        assert rl["by_keyword"] == {"sunset": 8, "ocean": 7}
        assert rl["backoff"]["total_attempts"] == 10
        assert rl["backoff"]["total_seconds"] == 120.57  # Rounded to 2 decimals
        assert rl["backoff"]["by_severity"] == {"low": 3, "medium": 5, "high": 2}

    def test_export_escalation_section(self):
        """Test that export includes escalation statistics."""
        metrics = RateLimitMetrics(
            cookie_rotations=3,
            vpn_switches=2
        )
        export = metrics.export_to_json()

        assert "escalation" in export
        assert export["escalation"]["cookie_rotations"] == 3
        assert export["escalation"]["vpn_switches"] == 2

    def test_export_circuit_breaker_section(self):
        """Test that export includes circuit breaker statistics."""
        metrics = RateLimitMetrics(
            circuit_breaker_trips=2,
            circuit_breaker_pause_seconds=120.5
        )
        export = metrics.export_to_json()

        assert "circuit_breaker" in export
        assert export["circuit_breaker"]["total_trips"] == 2
        assert export["circuit_breaker"]["total_pause_seconds"] == 120.5

    def test_export_batch_retry_section(self):
        """Test that export includes batch retry statistics."""
        metrics = RateLimitMetrics(
            batch_retry_passes=3,
            batch_retry_successes=10,
            batch_retry_failures=2
        )
        export = metrics.export_to_json()

        assert "batch_retry" in export
        assert export["batch_retry"]["total_passes"] == 3
        assert export["batch_retry"]["total_recovered"] == 10
        assert export["batch_retry"]["total_failed"] == 2

    def test_export_network_section(self):
        """Test that export includes network statistics."""
        metrics = RateLimitMetrics(
            speed_samples=50,
            avg_speed_mbps=5.5678,
            timeout_extensions=3
        )
        export = metrics.export_to_json()

        assert "network" in export
        assert export["network"]["speed_samples"] == 50
        assert export["network"]["avg_speed_mbps"] == 5.568  # Rounded to 3 decimals
        assert export["network"]["timeout_extensions"] == 3

    def test_export_includes_recommendations(self):
        """Test that export includes recommendations."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            rate_limit_events=20,  # > 10% threshold
            cookie_rotations=0
        )
        export = metrics.export_to_json()

        assert "recommendations" in export
        assert isinstance(export["recommendations"], list)
        # Should have recommendation about high rate limiting
        assert len(export["recommendations"]) > 0

    def test_export_without_config(self):
        """Test that export works without config."""
        metrics = RateLimitMetrics()
        export = metrics.export_to_json(config=None)

        assert "config_snapshot" not in export

    def test_export_with_config_includes_snapshot(self):
        """Test that export includes config snapshot when config provided."""
        metrics = RateLimitMetrics()

        # Create mock config
        mock_config = MagicMock()
        mock_download = MagicMock()
        mock_config.download = mock_download

        mock_rate_limit = MagicMock()
        mock_rate_limit.initial_backoff_seconds = 5.0
        mock_rate_limit.max_backoff_before_rotate = 60.0
        mock_rate_limit.backoff_multiplier = 2.0
        mock_rate_limit.per_tier_isolation = True
        mock_rate_limit.share_budget_across_keywords = True
        mock_rate_limit.max_backoff_budget = 300.0
        mock_rate_limit.adaptive_multiplier = True
        mock_download.rate_limit = mock_rate_limit

        mock_circuit_breaker = MagicMock()
        mock_circuit_breaker.enabled = True
        mock_circuit_breaker.consecutive_failures_threshold = 5
        mock_circuit_breaker.pause_seconds = 60.0
        mock_circuit_breaker.block_download_retries = True
        mock_download.circuit_breaker = mock_circuit_breaker

        mock_batch_retry = MagicMock()
        mock_batch_retry.enabled = True
        mock_batch_retry.delay_seconds = 120.0
        mock_batch_retry.max_passes = 2
        mock_batch_retry.respect_circuit_breaker = True
        mock_batch_retry.wait_for_cookie_cooldown = True
        mock_download.batch_retry = mock_batch_retry

        mock_speed_tracking = MagicMock()
        mock_speed_tracking.enabled = True
        mock_speed_tracking.window_size = 5
        mock_speed_tracking.min_speed_mbps = 1.0
        mock_speed_tracking.max_timeout_multiplier = 2.0
        mock_speed_tracking.rate_limit_signal_threshold = 0.1
        mock_speed_tracking.consecutive_slow_samples = 3
        mock_download.speed_tracking = mock_speed_tracking

        mock_cookie_rotation = MagicMock()
        mock_cookie_rotation.enabled = False
        mock_cookie_rotation.rotation_strategy = "on_error"
        mock_cookie_rotation.cooldown_seconds = 300
        mock_cookie_rotation.max_rotations_per_session = 0
        mock_download.cookie_rotation = mock_cookie_rotation

        mock_vpn = MagicMock()
        mock_vpn.enabled = False
        mock_vpn.rotate_on_rate_limit = True
        mock_vpn.switch_delay_seconds = 10
        mock_vpn.max_switches_per_session = 10
        mock_vpn.verify_connection = True
        mock_download.vpn = mock_vpn

        export = metrics.export_to_json(config=mock_config)

        assert "config_snapshot" in export
        cs = export["config_snapshot"]
        assert "rate_limit" in cs
        assert cs["rate_limit"]["initial_backoff_seconds"] == 5.0
        assert cs["rate_limit"]["per_tier_isolation"] == True
        assert "circuit_breaker" in cs
        assert cs["circuit_breaker"]["enabled"] == True
        assert "batch_retry" in cs
        assert "speed_tracking" in cs
        assert "cookie_rotation" in cs
        assert "vpn" in cs

    def test_export_json_serializable(self):
        """Test that export is JSON serializable."""
        import json
        metrics = RateLimitMetrics(
            total_downloads=100,
            retries_by_error_type={"timeout": 5},
            tier_rate_limit_events={"short": 3}
        )
        export = metrics.export_to_json()

        # Should not raise
        json_str = json.dumps(export)
        assert isinstance(json_str, str)

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["downloads"]["total"] == 100


class TestExportToJsonFile:
    """Test JSON file export functionality (US-012)."""

    def test_export_to_json_file(self, tmp_path):
        """Test that export_to_json_file writes valid JSON file."""
        import json
        metrics = RateLimitMetrics(total_downloads=50)
        output_path = tmp_path / "metrics.json"

        metrics.export_to_json_file(str(output_path))

        assert output_path.exists()
        with open(output_path, 'r') as f:
            data = json.load(f)
        assert data["downloads"]["total"] == 50

    def test_export_to_json_file_creates_parent_dirs(self, tmp_path):
        """Test that export_to_json_file creates parent directories."""
        import json
        metrics = RateLimitMetrics()
        output_path = tmp_path / "nested" / "dir" / "metrics.json"

        metrics.export_to_json_file(str(output_path))

        assert output_path.exists()
        with open(output_path, 'r') as f:
            data = json.load(f)
        assert data["schema_version"] == "1.0"

    def test_export_to_json_file_with_config(self, tmp_path):
        """Test that export_to_json_file includes config when provided."""
        import json
        metrics = RateLimitMetrics()

        # Create mock config with spec to prevent auto-creation of attributes
        mock_config = MagicMock()
        mock_download = MagicMock()
        mock_config.download = mock_download

        mock_rate_limit = MagicMock()
        mock_rate_limit.initial_backoff_seconds = 10.0
        mock_rate_limit.max_backoff_before_rotate = 60.0
        mock_rate_limit.backoff_multiplier = 2.0
        mock_rate_limit.per_tier_isolation = True
        mock_rate_limit.share_budget_across_keywords = True
        mock_rate_limit.max_backoff_budget = 300.0
        mock_rate_limit.adaptive_multiplier = True
        mock_download.rate_limit = mock_rate_limit

        # Set other configs to proper None (using configure_mock to override MagicMock behavior)
        mock_download.configure_mock(
            circuit_breaker=None,
            batch_retry=None,
            speed_tracking=None,
            cookie_rotation=None,
            vpn=None
        )

        output_path = tmp_path / "metrics.json"
        metrics.export_to_json_file(str(output_path), config=mock_config)

        with open(output_path, 'r') as f:
            data = json.load(f)
        assert "config_snapshot" in data
        assert data["config_snapshot"]["rate_limit"]["initial_backoff_seconds"] == 10.0

    def test_export_to_json_file_utf8_encoding(self, tmp_path):
        """Test that export uses UTF-8 encoding."""
        metrics = RateLimitMetrics()
        # Add a keyword with unicode characters
        metrics.record_rate_limit_event(keyword="日本語キーワード")

        output_path = tmp_path / "metrics.json"
        metrics.export_to_json_file(str(output_path))

        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "日本語キーワード" in content

    def test_export_to_json_file_pretty_printed(self, tmp_path):
        """Test that export is pretty-printed with indentation."""
        metrics = RateLimitMetrics()
        output_path = tmp_path / "metrics.json"
        metrics.export_to_json_file(str(output_path))

        with open(output_path, 'r') as f:
            content = f.read()
        # Pretty-printed JSON should have newlines and indentation
        assert "\n" in content
        assert "  " in content  # Indentation


class TestExportCompleteSchema:
    """Test complete export schema matches documentation (US-012)."""

    def test_complete_export_structure(self):
        """Test that export matches documented schema structure."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=95,
            failed_downloads=5,
            retry_attempts=50,
            retries_by_error_type={"timeout": 20, "rate_limit": 15, "network": 10, "transient": 5},
            max_retry_count_reached=3,
            rate_limit_events=15,
            tier_rate_limit_events={"short": 5, "medium": 5, "long": 5},
            keyword_rate_limit_events={"sunset": 10, "ocean": 5},
            backoff_attempts=10,
            time_spent_backing_off=120.5,
            backoff_events_by_severity={"low": 3, "medium": 5, "high": 2},
            cookie_rotations=3,
            vpn_switches=1,
            circuit_breaker_trips=2,
            circuit_breaker_pause_seconds=120.0,
            batch_retry_passes=2,
            batch_retry_successes=10,
            batch_retry_failures=2,
            speed_samples=50,
            avg_speed_mbps=5.5,
            timeout_extensions=3,
            session_count=1,
            session_start_time="2026-01-25T10:00:00Z",
            session_end_time="2026-01-25T12:34:56Z"
        )

        export = metrics.export_to_json()

        # Verify top-level keys
        required_keys = [
            "schema_version", "export_timestamp", "session",
            "downloads", "retries", "rate_limiting", "escalation",
            "circuit_breaker", "batch_retry", "network", "recommendations"
        ]
        for key in required_keys:
            assert key in export, f"Missing key: {key}"

        # Verify nested structures
        assert "session_count" in export["session"]
        assert "total" in export["downloads"]
        assert "success_rate_percent" in export["downloads"]
        assert "by_error_type" in export["retries"]
        assert "by_tier" in export["rate_limiting"]
        assert "by_keyword" in export["rate_limiting"]
        assert "backoff" in export["rate_limiting"]
        assert "by_severity" in export["rate_limiting"]["backoff"]
        assert "cookie_rotations" in export["escalation"]
        assert "vpn_switches" in export["escalation"]
        assert "total_trips" in export["circuit_breaker"]
        assert "total_recovered" in export["batch_retry"]
        assert "avg_speed_mbps" in export["network"]
