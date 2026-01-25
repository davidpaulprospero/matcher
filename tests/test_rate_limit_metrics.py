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
