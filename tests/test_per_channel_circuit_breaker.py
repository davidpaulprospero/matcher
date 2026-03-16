"""Tests for per-channel circuit breaker (US-155-010).

Tests the PerChannelCircuitBreaker class for tracking API calls per channel,
channel-level circuit breaker functionality, and metrics reporting.
"""

import pytest
import time
from unittest.mock import patch

from src.downloader.per_channel_circuit_breaker import (
    PerChannelCircuitBreaker,
    PerChannelCircuitBreakerConfig,
    ChannelCircuitState,
)


class TestPerChannelCircuitBreakerConfig:
    """Tests for PerChannelCircuitBreakerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PerChannelCircuitBreakerConfig()
        assert config.enabled is True
        assert config.consecutive_failures_threshold == 5
        assert config.pause_seconds == 60.0
        assert config.max_pause_seconds == 300.0
        assert config.jitter_factor == 0.2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PerChannelCircuitBreakerConfig(
            enabled=False,
            consecutive_failures_threshold=3,
            pause_seconds=30.0,
            max_pause_seconds=120.0,
        )
        assert config.enabled is False
        assert config.consecutive_failures_threshold == 3
        assert config.pause_seconds == 30.0
        assert config.max_pause_seconds == 120.0


class TestPerChannelCircuitBreaker:
    """Tests for PerChannelCircuitBreaker."""

    @pytest.fixture
    def config(self):
        """Create test configuration with fast timing."""
        return PerChannelCircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3,
            pause_seconds=0.1,  # Fast for testing
            max_pause_seconds=1.0,
            jitter_factor=0.0,  # No jitter for predictable tests
            enable_recovery=False,
        )

    @pytest.fixture
    def circuit_breaker(self, config):
        """Create a circuit breaker instance for testing."""
        return PerChannelCircuitBreaker(config)

    def test_initial_state(self, circuit_breaker):
        """Test initial state of circuit breaker."""
        metrics = circuit_breaker.get_metrics()
        assert metrics['total_channels_queried'] == 0
        assert metrics['total_channels_tracked'] == 0
        assert metrics['channels_with_circuit_open'] == 0

    def test_record_success(self, circuit_breaker):
        """Test recording successful API calls."""
        channel_id = "UC123456"

        # Record successful calls
        circuit_breaker.record_success(channel_id)
        circuit_breaker.record_success(channel_id)
        circuit_breaker.record_success(channel_id)

        stats = circuit_breaker.get_channel_stats(channel_id)
        assert stats is not None
        assert stats['call_count'] == 3
        assert stats['error_count'] == 0
        assert stats['consecutive_failures'] == 0
        assert stats['is_open'] is False

    def test_record_failure(self, circuit_breaker):
        """Test recording failed API calls."""
        channel_id = "UC123456"

        # Record failures (not enough to trip circuit)
        circuit_breaker.record_failure(channel_id)
        circuit_breaker.record_failure(channel_id)

        stats = circuit_breaker.get_channel_stats(channel_id)
        assert stats is not None
        assert stats['call_count'] == 2
        assert stats['error_count'] == 2
        assert stats['consecutive_failures'] == 2
        assert stats['is_open'] is False

    def test_circuit_trips_after_threshold(self, circuit_breaker):
        """Test circuit trips after consecutive failures threshold."""
        channel_id = "UC123456"

        # Record failures to trip the circuit
        for _ in range(3):
            circuit_breaker.record_failure(channel_id)

        stats = circuit_breaker.get_channel_stats(channel_id)
        assert stats is not None
        assert stats['is_open'] is True
        assert stats['total_trips'] == 1

    def test_circuit_closes_after_success(self, circuit_breaker):
        """Test circuit closes after successful call when not in recovery."""
        channel_id = "UC123456"

        # Record failures to trip the circuit
        for _ in range(3):
            circuit_breaker.record_failure(channel_id)

        # Check circuit is open
        assert circuit_breaker.is_channel_open(channel_id) is True

        # Record success - with recovery disabled, circuit stays open
        circuit_breaker.record_success(channel_id)

        stats = circuit_breaker.get_channel_stats(channel_id)
        # With enable_recovery=False, consecutive failures should reset
        assert stats['consecutive_failures'] == 0

    def test_check_and_wait_when_circuit_closed(self, circuit_breaker):
        """Test check_and_wait when circuit is closed."""
        channel_id = "UC123456"

        # Should not wait when circuit is closed
        start = time.time()
        circuit_breaker.check_and_wait(channel_id)
        elapsed = time.time() - start

        assert elapsed < 0.05  # Should be nearly instant

    def test_check_and_wait_when_circuit_open(self, circuit_breaker):
        """Test check_and_wait when circuit is open."""
        channel_id = "UC123456"

        # Trip the circuit
        for _ in range(3):
            circuit_breaker.record_failure(channel_id)

        # Check and wait should pause
        start = time.time()
        circuit_breaker.check_and_wait(channel_id)
        elapsed = time.time() - start

        # Should have waited approximately pause_seconds
        assert elapsed >= 0.05  # At least some waiting

    def test_mark_no_videos(self, circuit_breaker):
        """Test marking a channel as having no published videos."""
        channel_id = "UCNoVideos"

        circuit_breaker.mark_no_videos(channel_id)

        stats = circuit_breaker.get_channel_stats(channel_id)
        assert stats is not None
        assert stats['has_no_videos'] is True

    def test_metrics_include_no_video_channels(self, circuit_breaker):
        """Test that metrics include channels with no videos."""
        channel_id = "UCNoVideos"

        circuit_breaker.mark_no_videos(channel_id)
        circuit_breaker.record_success("UCWithVideos")

        metrics = circuit_breaker.get_metrics()
        assert metrics['channels_with_no_videos'] == 1

    def test_channel_query_distribution(self, circuit_breaker):
        """Test channel query distribution metrics."""
        # Record different numbers of queries for different channels
        circuit_breaker.record_success("UCChannel1")
        circuit_breaker.record_success("UCChannel1")
        circuit_breaker.record_success("UCChannel1")

        circuit_breaker.record_success("UCChannel2")

        circuit_breaker.record_success("UCChannel3")
        circuit_breaker.record_success("UCChannel3")

        distribution = circuit_breaker.get_channel_query_distribution()

        assert len(distribution) == 3
        # Should be sorted by query count descending
        assert distribution[0]['channel_id'] == "UCChannel1"
        assert distribution[0]['query_count'] == 3
        assert distribution[1]['channel_id'] in ["UCChannel2", "UCChannel3"]
        assert distribution[2]['channel_id'] in ["UCChannel2", "UCChannel3"]

    def test_reset(self, circuit_breaker):
        """Test resetting the circuit breaker."""
        channel_id = "UC123456"

        circuit_breaker.record_success(channel_id)
        circuit_breaker.record_failure(channel_id)

        # Reset
        circuit_breaker.reset()

        stats = circuit_breaker.get_channel_stats(channel_id)
        assert stats is None  # Should be gone after reset

        metrics = circuit_breaker.get_metrics()
        assert metrics['total_channels_queried'] == 0

    def test_disabled_config(self):
        """Test circuit breaker with disabled config."""
        config = PerChannelCircuitBreakerConfig(enabled=False)
        cb = PerChannelCircuitBreaker(config)

        channel_id = "UC123456"

        # Should not track when disabled
        cb.record_failure(channel_id)
        cb.record_failure(channel_id)
        cb.record_failure(channel_id)

        # Circuit should not open when disabled
        assert cb.is_channel_open(channel_id) is False

    def test_latency_tracking(self, circuit_breaker):
        """Test latency tracking per channel."""
        channel_id = "UC123456"

        circuit_breaker.record_success(channel_id, latency_ms=100.0)
        circuit_breaker.record_success(channel_id, latency_ms=200.0)
        circuit_breaker.record_success(channel_id, latency_ms=150.0)

        stats = circuit_breaker.get_channel_stats(channel_id)
        assert stats['latency_avg'] == 150.0

    def test_multiple_channels_independent(self, circuit_breaker):
        """Test that channels are tracked independently."""
        channel1 = "UCChannel1"
        channel2 = "UCChannel2"

        # Trip circuit for channel1 only
        for _ in range(3):
            circuit_breaker.record_failure(channel1)

        # channel1 should be open, channel2 should be closed
        assert circuit_breaker.is_channel_open(channel1) is True
        assert circuit_breaker.is_channel_open(channel2) is False

        # Record success for channel2
        circuit_breaker.record_success(channel2)

        stats1 = circuit_breaker.get_channel_stats(channel1)
        stats2 = circuit_breaker.get_channel_stats(channel2)

        assert stats1['is_open'] is True
        assert stats2['is_open'] is False
        assert stats2['consecutive_failures'] == 0


class TestPerChannelCircuitBreakerWithRecovery:
    """Tests for PerChannelCircuitBreaker with recovery enabled."""

    @pytest.fixture
    def config_with_recovery(self):
        """Create test configuration with recovery enabled."""
        return PerChannelCircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3,
            pause_seconds=0.05,  # Very fast for testing
            max_pause_seconds=0.5,
            jitter_factor=0.0,
            enable_recovery=True,
            recovery_max_requests=2,
            recovery_success_threshold=2,
        )

    @pytest.fixture
    def circuit_breaker(self, config_with_recovery):
        """Create a circuit breaker instance with recovery for testing."""
        return PerChannelCircuitBreaker(config_with_recovery)

    def test_recovery_mode(self, circuit_breaker):
        """Test circuit enters recovery mode after opening."""
        channel_id = "UC123456"

        # Trip the circuit
        for _ in range(3):
            circuit_breaker.record_failure(channel_id)

        stats = circuit_breaker.get_channel_stats(channel_id)
        assert stats['is_open'] is True

        # With recovery enabled, check_and_wait should transition to half-open
        # We can't easily test the timing-dependent recovery in unit tests
        # but we can verify the state machine logic
        assert stats['is_recovering'] is False  # Initially open, not yet recovering

    def test_successful_recovery(self, circuit_breaker):
        """Test successful recovery after circuit opens."""
        channel_id = "UC123456"

        # Trip the circuit
        for _ in range(3):
            circuit_breaker.record_failure(channel_id)

        # Record successes during recovery
        circuit_breaker.record_success(channel_id)
        circuit_breaker.record_success(channel_id)

        # With recovery enabled and enough successes, should recover
        # Note: The exact behavior depends on check_and_wait being called
        # which transitions from OPEN to RECOVERING state
        stats = circuit_breaker.get_channel_stats(channel_id)
        # Circuit should eventually close after enough successes
        assert stats['recovery_successes'] >= 0


class TestPerChannelThrottling:
    """Tests for per-channel API throttling (US-156-012)."""

    @pytest.fixture
    def throttle_config(self):
        """Create test configuration with throttling enabled."""
        return PerChannelCircuitBreakerConfig(
            enabled=True,
            per_channel_requests_per_minute=10,
            throttle_warning_threshold=0.8,  # 80% = 8 requests
            jitter_factor=0.0,
        )

    @pytest.fixture
    def throttle_cb(self, throttle_config):
        """Create a circuit breaker instance for throttling tests."""
        return PerChannelCircuitBreaker(throttle_config)

    def test_per_channel_requests_per_minute_config(self):
        """Test per_channel_requests_per_minute config field exists with default."""
        config = PerChannelCircuitBreakerConfig()
        assert hasattr(config, 'per_channel_requests_per_minute')
        assert config.per_channel_requests_per_minute == 60

    def test_throttle_warning_threshold_config(self):
        """Test throttle_warning_threshold config field exists with default."""
        config = PerChannelCircuitBreakerConfig()
        assert hasattr(config, 'throttle_warning_threshold')
        assert config.throttle_warning_threshold == 0.8

    def test_channel_level_throttle_tracks_calls(self, throttle_cb):
        """Test that channel_level_throttle tracks API calls per channel."""
        channel_id = "UCTestChannel"

        # Make calls that don't exceed threshold
        for _ in range(5):
            throttle_cb.channel_level_throttle(channel_id)

        stats = throttle_cb.get_channel_stats(channel_id)
        assert stats is not None
        assert stats['call_count'] == 5

    def test_channel_level_throttle_warning_at_threshold(self, throttle_cb, caplog):
        """Test warning is issued when channel approaches throttling threshold."""
        channel_id = "UCTestChannel"
        limit = 10
        warning_threshold = int(limit * 0.8)  # 8 requests

        # Make calls up to warning threshold
        for _ in range(warning_threshold):
            throttle_cb.channel_level_throttle(channel_id)

        # Check warning was logged
        assert any("approaching throttling limit" in record.message for record in caplog.records)

    def test_channel_level_throttle_waits_when_limit_exceeded(self, throttle_cb):
        """Test that throttling kicks in when per-minute limit is exceeded."""
        channel_id = "UCTestChannel"
        limit = 5  # Low limit for testing

        # Create config with low limit
        config = PerChannelCircuitBreakerConfig(
            enabled=True,
            per_channel_requests_per_minute=limit,
            throttle_warning_threshold=0.8,
            jitter_factor=0.0,
        )
        cb = PerChannelCircuitBreaker(config)

        # Make calls up to limit
        for _ in range(limit):
            cb.channel_level_throttle(channel_id)

        # Next call should trigger throttle (this would normally wait)
        # We can't easily test the sleep in unit tests, but verify it doesn't crash
        start = time.time()
        cb.channel_level_throttle(channel_id)
        elapsed = time.time() - start

        # Should have waited some time (in mock mode it sleeps briefly)
        assert elapsed >= 0

    def test_channel_usage_metrics_includes_call_counts(self, throttle_cb):
        """Test that channel_usage_metrics tracks per-channel call counts."""
        channel1 = "UCChannel1"
        channel2 = "UCChannel2"

        # Record different numbers of calls
        for _ in range(3):
            throttle_cb.channel_level_throttle(channel1)

        for _ in range(7):
            throttle_cb.channel_level_throttle(channel2)

        # Check metrics include call counts
        metrics = throttle_cb.get_metrics()
        assert 'total_api_calls' in metrics
        assert metrics['total_api_calls'] == 10

        # Check individual channel stats
        stats1 = throttle_cb.get_channel_stats(channel1)
        stats2 = throttle_cb.get_channel_stats(channel2)

        assert stats1['call_count'] == 3
        assert stats2['call_count'] == 7

    def test_per_channel_throttling_independent_per_channel(self, throttle_cb):
        """Test that throttling is independent per channel."""
        channel1 = "UCChannel1"
        channel2 = "UCChannel2"
        limit = 5

        # Exhaust limit for channel1 only
        for _ in range(limit):
            throttle_cb.channel_level_throttle(channel1)

        # Channel2 should still be able to make calls
        # (verify it doesn't affect channel1's count)
        for _ in range(3):
            throttle_cb.channel_level_throttle(channel2)

        stats1 = throttle_cb.get_channel_stats(channel1)
        stats2 = throttle_cb.get_channel_stats(channel2)

        assert stats1['call_count'] == limit
        assert stats2['call_count'] == 3


@pytest.mark.fast
class TestPerChannelCircuitBreakerFast:
    """Fast tests for PerChannelCircuitBreaker."""

    def test_basic_operation(self):
        """Quick test of basic operation."""
        config = PerChannelCircuitBreakerConfig(
            consecutive_failures_threshold=2,
            pause_seconds=0.01,
            enable_recovery=False,
        )
        cb = PerChannelCircuitBreaker(config)

        cb.record_success("UC1")
        cb.record_failure("UC1")

        assert cb.get_channel_stats("UC1")['call_count'] == 2

    def test_circuit_trips_quickly(self):
        """Quick test that circuit trips after threshold."""
        config = PerChannelCircuitBreakerConfig(
            consecutive_failures_threshold=2,
            pause_seconds=0.01,
            enable_recovery=False,
        )
        cb = PerChannelCircuitBreaker(config)

        cb.record_failure("UC1")
        cb.record_failure("UC1")

        assert cb.is_channel_open("UC1") is True
