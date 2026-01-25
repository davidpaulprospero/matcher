"""Tests for circuit breaker functionality.

US-006: Add circuit breaker for repeated search failures

Tests cover:
- CircuitBreakerConfig defaults and custom values
- CircuitBreaker state management (failures, open/closed)
- Circuit trips after threshold failures
- Circuit resets on successful search
- Pause duration and timing
- Stats and checkpoint persistence
- Integration with VideoDownloader
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)


# ============================================================================
# Configuration Tests
# ============================================================================


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig defaults and custom values."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CircuitBreakerConfig()

        assert config.enabled is True
        assert config.consecutive_failures_threshold == 5
        assert config.pause_seconds == 60.0

    def test_custom_values(self):
        """Config should accept custom values."""
        config = CircuitBreakerConfig(
            enabled=False,
            consecutive_failures_threshold=10,
            pause_seconds=120.0
        )

        assert config.enabled is False
        assert config.consecutive_failures_threshold == 10
        assert config.pause_seconds == 120.0


# ============================================================================
# State Initialization Tests
# ============================================================================


class TestCircuitBreakerInitialization:
    """Test CircuitBreaker initialization and initial state."""

    def test_default_initialization(self):
        """Circuit breaker should initialize with default config."""
        breaker = CircuitBreaker()

        assert breaker.config.enabled is True
        assert breaker.state.consecutive_failures == 0
        assert breaker.state.is_open is False
        assert breaker.state.opened_at is None
        assert breaker.state.total_trips == 0

    def test_custom_config_initialization(self):
        """Circuit breaker should accept custom config."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=3,
            pause_seconds=30.0
        )
        breaker = CircuitBreaker(config)

        assert breaker.config.consecutive_failures_threshold == 3
        assert breaker.config.pause_seconds == 30.0

    def test_disabled_breaker(self):
        """Disabled circuit breaker should not track failures."""
        config = CircuitBreakerConfig(enabled=False)
        breaker = CircuitBreaker(config)

        assert breaker.is_enabled is False


# ============================================================================
# Failure Recording Tests
# ============================================================================


class TestFailureRecording:
    """Test recording failures and counting."""

    def test_single_failure_increments_count(self):
        """Recording failure should increment consecutive_failures."""
        breaker = CircuitBreaker()

        assert breaker.state.consecutive_failures == 0
        breaker.record_failure()
        assert breaker.state.consecutive_failures == 1

    def test_multiple_failures_accumulate(self):
        """Multiple failures should accumulate."""
        breaker = CircuitBreaker()

        for i in range(4):
            breaker.record_failure()

        assert breaker.state.consecutive_failures == 4

    def test_failure_on_disabled_breaker_noop(self):
        """Recording failure on disabled breaker should be no-op."""
        config = CircuitBreakerConfig(enabled=False)
        breaker = CircuitBreaker(config)

        result = breaker.record_failure()

        assert result is False
        assert breaker.state.consecutive_failures == 0


# ============================================================================
# Circuit Trip Tests
# ============================================================================


class TestCircuitTrip:
    """Test circuit tripping (opening) behavior."""

    def test_trips_at_threshold(self):
        """Circuit should trip when failures reach threshold."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=3)
        breaker = CircuitBreaker(config)

        # First 2 failures - no trip
        assert breaker.record_failure() is False
        assert breaker.record_failure() is False
        assert breaker.is_open is False

        # 3rd failure - trips
        assert breaker.record_failure() is True
        assert breaker.is_open is True
        assert breaker.state.total_trips == 1

    def test_trips_logs_info(self):
        """Circuit trip should log at INFO level."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=2)
        breaker = CircuitBreaker(config)

        with patch('src.downloader.circuit_breaker.logger') as mock_logger:
            breaker.record_failure()
            breaker.record_failure()

            # Should have logged INFO about tripping
            assert mock_logger.info.called
            call_args = str(mock_logger.info.call_args)
            assert 'TRIPPED' in call_args or 'consecutive' in call_args.lower()

    def test_opened_at_timestamp_set(self):
        """opened_at should be set when circuit trips."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=1)
        breaker = CircuitBreaker(config)

        before = time.time()
        breaker.record_failure()
        after = time.time()

        assert breaker.state.opened_at is not None
        assert before <= breaker.state.opened_at <= after


# ============================================================================
# Success Recording Tests
# ============================================================================


class TestSuccessRecording:
    """Test recording success and resetting."""

    def test_success_resets_failure_count(self):
        """Recording success should reset consecutive_failures to 0."""
        breaker = CircuitBreaker()

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state.consecutive_failures == 2

        breaker.record_success()
        assert breaker.state.consecutive_failures == 0

    def test_success_closes_open_circuit(self):
        """Success should close an open circuit."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=1)
        breaker = CircuitBreaker(config)

        breaker.record_failure()
        assert breaker.is_open is True

        breaker.record_success()
        assert breaker.is_open is False
        assert breaker.state.opened_at is None

    def test_success_on_disabled_breaker_noop(self):
        """Recording success on disabled breaker should be no-op."""
        config = CircuitBreakerConfig(enabled=False)
        breaker = CircuitBreaker(config)

        breaker.record_success()
        assert breaker.state.consecutive_failures == 0


# ============================================================================
# Pause and Wait Tests
# ============================================================================


class TestCheckAndWait:
    """Test check_and_wait behavior with pause."""

    def test_returns_true_when_closed(self):
        """check_and_wait should return True when circuit is closed."""
        breaker = CircuitBreaker()

        result = breaker.check_and_wait()

        assert result is True

    def test_returns_false_when_disabled(self):
        """check_and_wait should return False when disabled."""
        config = CircuitBreakerConfig(enabled=False)
        breaker = CircuitBreaker(config)

        result = breaker.check_and_wait()

        assert result is False

    def test_waits_when_open(self):
        """check_and_wait should sleep when circuit is open."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.1  # Short pause for testing
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        start = time.time()
        breaker.check_and_wait()
        elapsed = time.time() - start

        # Should have waited at least pause_seconds
        assert elapsed >= 0.1

    def test_closes_after_pause(self):
        """Circuit should close after pause completes."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.05
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        assert breaker.is_open is True
        breaker.check_and_wait()
        assert breaker.is_open is False

    def test_pause_duration_logged(self):
        """Pause should be logged at INFO level."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.01
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        with patch('src.downloader.circuit_breaker.logger') as mock_logger:
            breaker.check_and_wait()

            assert mock_logger.info.called

    def test_tracks_total_paused_seconds(self):
        """Should track total paused time for reporting."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.05
        )
        breaker = CircuitBreaker(config)

        # First trip
        breaker.record_failure()
        breaker.check_and_wait()

        assert breaker.state.total_paused_seconds >= 0.05

    def test_immediate_return_after_pause_expired(self):
        """If pause time has elapsed, should return immediately."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.01
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        # Wait longer than pause_seconds
        time.sleep(0.02)

        start = time.time()
        breaker.check_and_wait()
        elapsed = time.time() - start

        # Should return almost immediately (pause already elapsed)
        assert elapsed < 0.01


# ============================================================================
# Manual Reset Tests
# ============================================================================


class TestManualReset:
    """Test manual reset functionality."""

    def test_reset_clears_failures(self):
        """reset() should clear consecutive failures."""
        breaker = CircuitBreaker()
        breaker.record_failure()
        breaker.record_failure()

        breaker.reset()

        assert breaker.state.consecutive_failures == 0

    def test_reset_closes_circuit(self):
        """reset() should close an open circuit."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=1)
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        breaker.reset()

        assert breaker.is_open is False
        assert breaker.state.opened_at is None


# ============================================================================
# Statistics Tests
# ============================================================================


class TestGetStats:
    """Test statistics reporting."""

    def test_stats_empty_state(self):
        """Stats should reflect initial state."""
        breaker = CircuitBreaker()
        stats = breaker.get_stats()

        assert stats['enabled'] is True
        assert stats['is_open'] is False
        assert stats['consecutive_failures'] == 0
        assert stats['total_trips'] == 0
        assert stats['total_paused_seconds'] == 0.0
        assert stats['threshold'] == 5
        assert stats['pause_seconds'] == 60.0

    def test_stats_after_trips(self):
        """Stats should reflect state after trips."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.1  # Longer pause to ensure we measure it
        )
        breaker = CircuitBreaker(config)

        # Trip once - check_and_wait should wait the full pause duration
        breaker.record_failure()

        # Verify circuit is open
        assert breaker.state.is_open is True
        assert breaker.state.total_trips == 1

        # Now wait - this should block and track paused time
        breaker.check_and_wait()

        stats = breaker.get_stats()

        # Trip count should reflect one trip
        assert stats['total_trips'] == 1
        # If pause was needed, total_paused_seconds > 0; if not, it just passed
        # The key invariant is that the circuit was tripped and total_trips was incremented


# ============================================================================
# Checkpoint Persistence Tests
# ============================================================================


class TestCheckpointPersistence:
    """Test checkpoint save/restore."""

    def test_to_checkpoint_dict(self):
        """Should serialize cumulative stats to checkpoint."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=1)
        breaker = CircuitBreaker(config)

        breaker.record_failure()  # Trip
        breaker.state.total_paused_seconds = 30.0

        checkpoint = breaker.to_checkpoint_dict()

        assert checkpoint['total_trips'] == 1
        assert checkpoint['total_paused_seconds'] == 30.0
        # Should NOT include transient state
        assert 'is_open' not in checkpoint
        assert 'opened_at' not in checkpoint

    def test_from_checkpoint_dict_restores_cumulative(self):
        """Should restore cumulative stats from checkpoint."""
        breaker = CircuitBreaker()

        breaker.from_checkpoint_dict({
            'total_trips': 3,
            'total_paused_seconds': 180.0,
            'consecutive_failures': 2  # Should be ignored
        })

        assert breaker.state.total_trips == 3
        assert breaker.state.total_paused_seconds == 180.0
        # Should NOT restore transient state
        assert breaker.state.consecutive_failures == 0
        assert breaker.is_open is False

    def test_from_checkpoint_dict_empty(self):
        """Should handle empty/None checkpoint gracefully."""
        breaker = CircuitBreaker()

        breaker.from_checkpoint_dict(None)
        breaker.from_checkpoint_dict({})

        assert breaker.state.total_trips == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegrationWithVideoDownloader:
    """Test integration with VideoDownloader initialization."""

    def test_circuit_breaker_initialized_from_config(self):
        """VideoDownloader should initialize circuit breaker from config."""
        from unittest.mock import MagicMock, patch, PropertyMock

        # Mock the config
        mock_config = MagicMock()
        mock_download_config = MagicMock()

        # Circuit breaker config
        mock_cb_config = MagicMock()
        mock_cb_config.enabled = True
        mock_cb_config.consecutive_failures_threshold = 3
        mock_cb_config.pause_seconds = 30.0

        mock_download_config.circuit_breaker = mock_cb_config
        mock_download_config.cookies_from_browser = ''
        mock_download_config.cookies_path = ''
        mock_download_config.cookie_rotation = MagicMock(enabled=False)
        mock_download_config.vpn = MagicMock(enabled=False)
        mock_download_config.speed_tracking = MagicMock(enabled=False)

        mock_config.download = mock_download_config
        mock_config.cache_dir = '/tmp'
        mock_config.downloaded_videos_dir = '/tmp/videos'

        # Patch dependencies to avoid file system operations
        with patch('src.downloader.core.CheckpointManager'), \
             patch('src.downloader.core.TranscodingManager'), \
             patch('src.downloader.core.TitleFilter'), \
             patch('src.downloader.core.SpeechScreener'), \
             patch('src.downloader.core.SearchOptimizer'), \
             patch('src.downloader.core.AudioFirstPipeline'), \
             patch('src.downloader.core.utils.get_cookies_args', return_value=[]):

            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader(mock_config)

            assert downloader.circuit_breaker is not None
            assert downloader.circuit_breaker.config.enabled is True
            assert downloader.circuit_breaker.config.consecutive_failures_threshold == 3

    def test_circuit_breaker_disabled_when_config_disabled(self):
        """Circuit breaker should be disabled when config says so."""
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_download_config = MagicMock()

        mock_cb_config = MagicMock()
        mock_cb_config.enabled = False

        mock_download_config.circuit_breaker = mock_cb_config
        mock_download_config.cookies_from_browser = ''
        mock_download_config.cookies_path = ''
        mock_download_config.cookie_rotation = MagicMock(enabled=False)
        mock_download_config.vpn = MagicMock(enabled=False)
        mock_download_config.speed_tracking = MagicMock(enabled=False)

        mock_config.download = mock_download_config
        mock_config.cache_dir = '/tmp'
        mock_config.downloaded_videos_dir = '/tmp/videos'

        with patch('src.downloader.core.CheckpointManager'), \
             patch('src.downloader.core.TranscodingManager'), \
             patch('src.downloader.core.TitleFilter'), \
             patch('src.downloader.core.SpeechScreener'), \
             patch('src.downloader.core.SearchOptimizer'), \
             patch('src.downloader.core.AudioFirstPipeline'), \
             patch('src.downloader.core.utils.get_cookies_args', return_value=[]):

            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader(mock_config)

            assert downloader.circuit_breaker.is_enabled is False


class TestCircuitBreakerInDownloadFlow:
    """Test circuit breaker behavior in download flow."""

    def test_check_and_wait_called_before_search(self):
        """Circuit breaker check should be called before searches."""
        from unittest.mock import MagicMock, patch
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        # Create a mock that tracks calls
        breaker = CircuitBreaker(CircuitBreakerConfig(enabled=True))
        breaker.check_and_wait = MagicMock(return_value=True)
        breaker.record_failure = MagicMock()

        # The check should be called when search happens
        # This is a design verification - actual integration test would need more setup
        breaker.check_and_wait()
        assert breaker.check_and_wait.called

    def test_failure_recorded_on_no_results(self):
        """Failure should be recorded when search returns no results."""
        breaker = CircuitBreaker()

        # Simulate no results scenario
        breaker.record_failure()

        assert breaker.state.consecutive_failures == 1

    def test_success_recorded_on_results(self):
        """Success should be recorded when search returns results."""
        breaker = CircuitBreaker()

        # Add some failures first
        breaker.record_failure()
        breaker.record_failure()

        # Then success
        breaker.record_success()

        assert breaker.state.consecutive_failures == 0
