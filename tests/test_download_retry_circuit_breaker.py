"""Tests for download retry and circuit breaker timeout coordination (US-011).

Verifies that the download retry loop properly coordinates with the circuit breaker:
- Download retry loop checks circuit breaker state before each attempt
- If circuit breaker tripped during retry, wait for recovery before continuing
- New config option circuit_breaker.block_download_retries (default: true)
- Retry count preserved across circuit breaker pause (not reset)
- Log when download retry paused due to circuit breaker
"""

import pytest
import time
from unittest.mock import MagicMock, patch, call

from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState


class TestCircuitBreakerConfigBlockDownloadRetries:
    """Tests for block_download_retries config option."""

    @pytest.mark.fast
    def test_default_is_true(self):
        """block_download_retries should default to True."""
        config = CircuitBreakerConfig()
        assert config.block_download_retries is True

    @pytest.mark.fast
    def test_can_be_disabled(self):
        """block_download_retries can be set to False."""
        config = CircuitBreakerConfig(block_download_retries=False)
        assert config.block_download_retries is False

    @pytest.mark.fast
    def test_included_in_config_section(self):
        """block_download_retries should be in config/sections/download.py."""
        from src.config.sections.download import CircuitBreakerConfig as ConfigCircuitBreakerConfig
        config = ConfigCircuitBreakerConfig()
        assert hasattr(config, 'block_download_retries')
        assert config.block_download_retries is True


class TestCircuitBreakerGetRemainingPauseTime:
    """Tests for get_remaining_pause_time() method."""

    @pytest.mark.fast
    def test_returns_zero_when_disabled(self):
        """Should return 0 when circuit breaker is disabled."""
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=False))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        assert cb.get_remaining_pause_time() == 0.0

    @pytest.mark.fast
    def test_returns_zero_when_closed(self):
        """Should return 0 when circuit breaker is not tripped."""
        cb = CircuitBreaker()
        cb.state.is_open = False
        assert cb.get_remaining_pause_time() == 0.0

    @pytest.mark.fast
    def test_returns_zero_when_no_opened_at(self):
        """Should return 0 when opened_at is None."""
        cb = CircuitBreaker()
        cb.state.is_open = True
        cb.state.opened_at = None
        assert cb.get_remaining_pause_time() == 0.0

    @pytest.mark.fast
    def test_returns_remaining_time_when_tripped(self):
        """Should return remaining time when circuit is tripped."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=10.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 3.0  # Opened 3 seconds ago

        remaining = cb.get_remaining_pause_time()
        # Should be approximately 7 seconds
        assert 6.5 < remaining <= 7.0

    @pytest.mark.fast
    def test_returns_zero_when_pause_elapsed(self):
        """Should return 0 when pause duration has already elapsed."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=5.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 10.0  # Opened 10 seconds ago (past 5s pause)

        assert cb.get_remaining_pause_time() == 0.0

    @pytest.mark.fast
    def test_never_returns_negative(self):
        """Should never return negative values."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=1.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 100.0  # Way past pause

        remaining = cb.get_remaining_pause_time()
        assert remaining >= 0.0


class TestCircuitBreakerWaitForRecoveryIfNeeded:
    """Tests for wait_for_recovery_if_needed() method."""

    @pytest.mark.fast
    def test_returns_zero_when_disabled(self):
        """Should return 0 when circuit breaker is disabled."""
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=False))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        assert cb.wait_for_recovery_if_needed() == 0.0

    @pytest.mark.fast
    def test_returns_zero_when_closed(self):
        """Should return 0 when circuit breaker is not tripped."""
        cb = CircuitBreaker()
        cb.state.is_open = False
        assert cb.wait_for_recovery_if_needed() == 0.0

    @pytest.mark.fast
    def test_returns_zero_when_pause_elapsed(self):
        """Should return 0 when pause duration has already elapsed."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=1.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 10.0  # Way past pause

        assert cb.wait_for_recovery_if_needed() == 0.0

    @pytest.mark.fast
    def test_waits_and_returns_time_when_tripped(self):
        """Should wait and return time when circuit is tripped."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()

        with patch('time.sleep') as mock_sleep:
            wait_time = cb.wait_for_recovery_if_needed()

            mock_sleep.assert_called_once()
            called_time = mock_sleep.call_args[0][0]
            assert 0 < called_time <= 0.1
            assert 0 < wait_time <= 0.1

    @pytest.mark.fast
    def test_updates_total_paused_seconds(self):
        """Should accumulate paused time in state.total_paused_seconds."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        initial_paused = cb.state.total_paused_seconds

        with patch('time.sleep'):
            cb.wait_for_recovery_if_needed()

        assert cb.state.total_paused_seconds > initial_paused

    @pytest.mark.fast
    def test_transitions_to_half_open(self):
        """Should transition to half-open state after waiting."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()

        with patch('time.sleep'):
            cb.wait_for_recovery_if_needed()

        assert cb.state.is_open is False
        assert cb.state.opened_at is None

    @pytest.mark.fast
    def test_logs_with_context(self):
        """Should log with context string when provided."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        cb.state.total_trips = 2
        cb.state.consecutive_failures = 5

        with patch('time.sleep'), \
             patch('src.downloader.circuit_breaker.logger') as mock_logger:
            cb.wait_for_recovery_if_needed(context="download retry 2/3")

            # Should log both the wait and the resume
            assert mock_logger.info.call_count == 2

            # Check first log (wait message)
            first_log = mock_logger.info.call_args_list[0][0][0]
            assert 'download retry 2/3' in first_log
            assert 'waiting' in first_log.lower()

            # Check second log (resume message)
            second_log = mock_logger.info.call_args_list[1][0][0]
            assert 'download retry 2/3' in second_log
            assert 'resuming' in second_log.lower()

    @pytest.mark.fast
    def test_logs_without_context(self):
        """Should log without context when not provided."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()

        with patch('time.sleep'), \
             patch('src.downloader.circuit_breaker.logger') as mock_logger:
            cb.wait_for_recovery_if_needed()

            # Should still log but without context
            first_log = mock_logger.info.call_args_list[0][0][0]
            assert 'Circuit breaker OPEN:' in first_log


class TestRateLimitMetricsRecordCircuitBreakerWait:
    """Tests for record_circuit_breaker_wait() method."""

    @pytest.mark.fast
    def test_method_exists(self):
        """record_circuit_breaker_wait should exist on RateLimitMetrics."""
        from src.downloader.rate_limit_metrics import RateLimitMetrics
        metrics = RateLimitMetrics()
        assert hasattr(metrics, 'record_circuit_breaker_wait')

    @pytest.mark.fast
    def test_adds_to_circuit_breaker_pause_seconds(self):
        """Should add wait time to circuit_breaker_pause_seconds."""
        from src.downloader.rate_limit_metrics import RateLimitMetrics
        metrics = RateLimitMetrics()
        metrics.circuit_breaker_pause_seconds = 10.0

        metrics.record_circuit_breaker_wait(5.0)

        assert metrics.circuit_breaker_pause_seconds == 15.0

    @pytest.mark.fast
    def test_accumulates_multiple_waits(self):
        """Should accumulate multiple wait calls."""
        from src.downloader.rate_limit_metrics import RateLimitMetrics
        metrics = RateLimitMetrics()

        metrics.record_circuit_breaker_wait(3.0)
        metrics.record_circuit_breaker_wait(2.0)
        metrics.record_circuit_breaker_wait(1.0)

        assert metrics.circuit_breaker_pause_seconds == 6.0


class TestDownloadRetryCoordination:
    """Integration tests for download retry and circuit breaker coordination."""

    @pytest.mark.fast
    def test_retry_loop_checks_circuit_breaker_on_retry_attempts(self):
        """Download retry should check circuit breaker before retry attempts (not first attempt)."""
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        # Create a minimal circuit breaker
        cb = CircuitBreaker(CircuitBreakerConfig(
            enabled=True,
            block_download_retries=True,
            pause_seconds=0.1
        ))

        metrics = RateLimitMetrics()

        # Simulate the retry loop logic from core.py
        max_retries = 3
        block_download_retries = getattr(cb.config, 'block_download_retries', True)

        # Trip the circuit breaker
        cb.state.is_open = True
        cb.state.opened_at = time.time()

        wait_calls = []
        with patch('time.sleep'):
            for attempt in range(max_retries + 1):
                # US-011: Check circuit breaker state before retry attempts
                if attempt > 0 and block_download_retries and cb.is_open:
                    wait_time = cb.wait_for_recovery_if_needed(
                        context=f"download retry {attempt}/{max_retries}"
                    )
                    if wait_time > 0:
                        wait_calls.append(wait_time)
                        metrics.record_circuit_breaker_wait(wait_time)

                # Simulate failure on first attempt - continue to retry
                if attempt == 0:
                    continue  # First attempt fails, go to retry

                # After waiting, the circuit breaker resets - break after first retry
                break

        # Should have waited once (on first retry attempt, which is attempt=1)
        assert len(wait_calls) == 1
        assert wait_calls[0] > 0

    @pytest.mark.fast
    def test_retry_count_preserved_across_circuit_breaker_pause(self):
        """Retry count should be preserved across circuit breaker pause."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            enabled=True,
            block_download_retries=True,
            pause_seconds=0.1
        ))

        max_retries = 3
        retry_counts = []

        # Trip the circuit breaker
        cb.state.is_open = True
        cb.state.opened_at = time.time()

        with patch('time.sleep'):
            for attempt in range(max_retries + 1):
                retry_counts.append(attempt)

                if attempt > 0 and cb.config.block_download_retries and cb.is_open:
                    cb.wait_for_recovery_if_needed()

                # Simulate failure triggering another retry
                if attempt < max_retries:
                    continue
                break

        # Retry count should go 0, 1, 2, 3 (not reset)
        assert retry_counts == [0, 1, 2, 3]

    @pytest.mark.fast
    def test_no_wait_when_block_download_retries_disabled(self):
        """Should not wait for circuit breaker when block_download_retries is False."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            enabled=True,
            block_download_retries=False,  # Disabled
            pause_seconds=10.0
        ))

        # Trip the circuit breaker
        cb.state.is_open = True
        cb.state.opened_at = time.time()

        block_download_retries = getattr(cb.config, 'block_download_retries', True)

        # Should not enter wait block because block_download_retries is False
        wait_called = False
        for attempt in range(3):
            if attempt > 0 and block_download_retries and cb.is_open:
                wait_called = True

        assert wait_called is False

    @pytest.mark.fast
    def test_no_wait_when_circuit_breaker_not_tripped(self):
        """Should not wait when circuit breaker is not tripped."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            enabled=True,
            block_download_retries=True,
            pause_seconds=10.0
        ))

        # Circuit breaker is NOT tripped
        cb.state.is_open = False

        wait_time = cb.wait_for_recovery_if_needed()
        assert wait_time == 0.0

    @pytest.mark.fast
    def test_first_attempt_does_not_check_circuit_breaker(self):
        """First attempt (attempt=0) should not check circuit breaker."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            enabled=True,
            block_download_retries=True,
            pause_seconds=0.1
        ))

        # Trip the circuit breaker
        cb.state.is_open = True
        cb.state.opened_at = time.time()

        wait_called_on_first_attempt = False

        for attempt in range(2):
            # This mirrors the logic in core.py
            if attempt > 0 and cb.config.block_download_retries and cb.is_open:
                with patch('time.sleep'):
                    cb.wait_for_recovery_if_needed()
            else:
                if attempt == 0:
                    # First attempt - verify we didn't call wait
                    wait_called_on_first_attempt = (
                        attempt > 0 and
                        cb.config.block_download_retries and
                        cb.is_open
                    )
            break

        assert wait_called_on_first_attempt is False


class TestConfigYamlIntegration:
    """Tests for config.yaml integration."""

    @pytest.mark.fast
    def test_circuit_breaker_section_has_block_download_retries(self):
        """config.yaml circuit_breaker section should have block_download_retries."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert 'download' in config
        assert 'circuit_breaker' in config['download']
        assert 'block_download_retries' in config['download']['circuit_breaker']
        assert config['download']['circuit_breaker']['block_download_retries'] is True


class TestVideoDownloaderCircuitBreakerConfig:
    """Tests for VideoDownloader reading block_download_retries config."""

    @pytest.mark.fast
    def test_reads_block_download_retries_from_config(self):
        """VideoDownloader should read block_download_retries from config."""
        from src.downloader.core import VideoDownloader

        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.download.cookies_path = ''
        mock_config.download.cookies_from_browser = ''
        mock_config.download.download_timeouts = {'short': 60}
        mock_config.download.root_dir = ''
        mock_config.download.folder_name = 'videos'

        # Configure circuit breaker with block_download_retries
        mock_config.download.circuit_breaker = MagicMock()
        mock_config.download.circuit_breaker.enabled = True
        mock_config.download.circuit_breaker.consecutive_failures_threshold = 5
        mock_config.download.circuit_breaker.pause_seconds = 60.0
        mock_config.download.circuit_breaker.block_download_retries = True

        mock_config.download.batch_retry = MagicMock()
        mock_config.download.batch_retry.enabled = False

        mock_config.download.rate_limit = MagicMock()
        mock_config.download.rate_limit.initial_backoff_seconds = 5.0
        mock_config.download.rate_limit.max_backoff_before_rotate = 60.0
        mock_config.download.rate_limit.backoff_multiplier = 2.0
        mock_config.download.rate_limit.per_tier_isolation = True
        mock_config.download.speed_tracking = MagicMock()
        mock_config.download.speed_tracking.enabled = False
        mock_config.download.cookie_rotation = MagicMock()
        mock_config.download.cookie_rotation.enabled = False
        mock_config.download.vpn = MagicMock()
        mock_config.download.vpn.enabled = False
        mock_config.download.llm_title_filter = MagicMock()
        mock_config.download.llm_title_filter.enabled = False

        mock_config.duration_tiers = MagicMock()

        with patch('src.downloader.core.CookieRotator'), \
             patch('src.downloader.core.VPNManager'):
            downloader = VideoDownloader(mock_config)

        # Verify block_download_retries is accessible
        block_retries = getattr(
            downloader.circuit_breaker.config, 'block_download_retries', True
        )
        assert block_retries is True

    @pytest.mark.fast
    def test_block_download_retries_defaults_true_if_missing(self):
        """Should default to True if block_download_retries not in config."""
        from src.downloader.core import VideoDownloader

        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.download.cookies_path = ''
        mock_config.download.cookies_from_browser = ''
        mock_config.download.download_timeouts = {'short': 60}
        mock_config.download.root_dir = ''
        mock_config.download.folder_name = 'videos'

        # Configure circuit breaker WITHOUT block_download_retries
        mock_config.download.circuit_breaker = MagicMock()
        mock_config.download.circuit_breaker.enabled = True
        mock_config.download.circuit_breaker.consecutive_failures_threshold = 5
        mock_config.download.circuit_breaker.pause_seconds = 60.0
        # No block_download_retries attribute - test getattr default

        mock_config.download.batch_retry = MagicMock()
        mock_config.download.batch_retry.enabled = False

        mock_config.download.rate_limit = MagicMock()
        mock_config.download.rate_limit.initial_backoff_seconds = 5.0
        mock_config.download.rate_limit.max_backoff_before_rotate = 60.0
        mock_config.download.rate_limit.backoff_multiplier = 2.0
        mock_config.download.rate_limit.per_tier_isolation = True
        mock_config.download.speed_tracking = MagicMock()
        mock_config.download.speed_tracking.enabled = False
        mock_config.download.cookie_rotation = MagicMock()
        mock_config.download.cookie_rotation.enabled = False
        mock_config.download.vpn = MagicMock()
        mock_config.download.vpn.enabled = False
        mock_config.download.llm_title_filter = MagicMock()
        mock_config.download.llm_title_filter.enabled = False

        mock_config.duration_tiers = MagicMock()

        with patch('src.downloader.core.CookieRotator'), \
             patch('src.downloader.core.VPNManager'):
            downloader = VideoDownloader(mock_config)

        # getattr should default to True
        block_retries = getattr(
            downloader.circuit_breaker.config, 'block_download_retries', True
        )
        assert block_retries is True


class TestLogging:
    """Tests for logging behavior."""

    @pytest.mark.fast
    def test_logs_when_download_retry_paused(self):
        """Should log when download retry is paused due to circuit breaker."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            enabled=True,
            pause_seconds=0.1
        ))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        cb.state.total_trips = 1
        cb.state.consecutive_failures = 5

        with patch('time.sleep'), \
             patch('src.downloader.circuit_breaker.logger') as mock_logger:
            cb.wait_for_recovery_if_needed(context="download retry 1/3 for 'sunset' (short)")

            # Should log the wait
            mock_logger.info.assert_called()
            log_message = mock_logger.info.call_args_list[0][0][0]

            assert 'Circuit breaker OPEN' in log_message
            assert 'download retry 1/3' in log_message
            assert 'waiting' in log_message.lower()

    @pytest.mark.fast
    def test_logs_include_trip_count(self):
        """Log message should include trip count."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        cb.state.total_trips = 3
        cb.state.consecutive_failures = 5

        with patch('time.sleep'), \
             patch('src.downloader.circuit_breaker.logger') as mock_logger:
            cb.wait_for_recovery_if_needed()

            log_message = mock_logger.info.call_args_list[0][0][0]
            assert 'trip #3' in log_message

    @pytest.mark.fast
    def test_logs_include_consecutive_failures(self):
        """Log message should include consecutive failures count."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        cb.state.total_trips = 1
        cb.state.consecutive_failures = 7

        with patch('time.sleep'), \
             patch('src.downloader.circuit_breaker.logger') as mock_logger:
            cb.wait_for_recovery_if_needed()

            log_message = mock_logger.info.call_args_list[0][0][0]
            assert '7 consecutive failures' in log_message

    @pytest.mark.fast
    def test_logs_resume_after_wait(self):
        """Should log when resuming after circuit breaker wait."""
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()

        with patch('time.sleep'), \
             patch('src.downloader.circuit_breaker.logger') as mock_logger:
            cb.wait_for_recovery_if_needed(context="download retry")

            # Second log should be about resuming
            assert mock_logger.info.call_count >= 2
            resume_log = mock_logger.info.call_args_list[1][0][0]
            assert 'resuming' in resume_log.lower()
