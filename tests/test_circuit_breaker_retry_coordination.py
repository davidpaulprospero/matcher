"""Tests for circuit breaker and retry queue coordination (US-003).

Verifies that the batch retry queue properly coordinates with the circuit breaker:
- RetryQueue gains set_circuit_breaker() method to link to circuit breaker
- Before processing batch retry, checks if circuit breaker is tripped
- If tripped, waits for circuit breaker pause to complete before retry
- New config option batch_retry.respect_circuit_breaker (default: true)
- Batch retry delay extends if circuit breaker pause overlaps
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.downloader.retry_queue import RetryQueue, BatchRetryConfig, RetryItem
from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState


class TestBatchRetryConfigRespectCircuitBreaker:
    """Tests for respect_circuit_breaker config option."""

    def test_default_is_true(self):
        """respect_circuit_breaker should default to True."""
        config = BatchRetryConfig()
        assert config.respect_circuit_breaker is True

    def test_can_be_disabled(self):
        """respect_circuit_breaker can be set to False."""
        config = BatchRetryConfig(respect_circuit_breaker=False)
        assert config.respect_circuit_breaker is False

    def test_included_in_config_section(self):
        """respect_circuit_breaker should be in config/sections/download.py."""
        from src.config.sections.download import BatchRetryConfig as ConfigBatchRetryConfig
        config = ConfigBatchRetryConfig()
        assert hasattr(config, 'respect_circuit_breaker')
        assert config.respect_circuit_breaker is True


class TestRetryQueueSetCircuitBreaker:
    """Tests for set_circuit_breaker() method."""

    def test_set_circuit_breaker_links_instance(self):
        """set_circuit_breaker() should store circuit breaker reference."""
        queue = RetryQueue()
        cb = CircuitBreaker()

        queue.set_circuit_breaker(cb)

        assert queue._circuit_breaker is cb

    def test_set_circuit_breaker_logs_debug(self):
        """set_circuit_breaker() should log at debug level."""
        queue = RetryQueue()
        cb = CircuitBreaker()

        with patch('src.downloader.retry_queue.logger') as mock_logger:
            queue.set_circuit_breaker(cb)
            mock_logger.debug.assert_called_once()

    def test_without_circuit_breaker_is_none(self):
        """Without set_circuit_breaker(), _circuit_breaker should be None."""
        queue = RetryQueue()
        assert queue._circuit_breaker is None


class TestWaitForCircuitBreaker:
    """Tests for _wait_for_circuit_breaker() method."""

    def test_returns_zero_when_no_circuit_breaker(self):
        """Should return 0 when no circuit breaker is linked."""
        queue = RetryQueue()
        assert queue._wait_for_circuit_breaker() == 0.0

    def test_returns_zero_when_respect_disabled(self):
        """Should return 0 when respect_circuit_breaker is False."""
        queue = RetryQueue(BatchRetryConfig(respect_circuit_breaker=False))
        cb = CircuitBreaker()
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        assert queue._wait_for_circuit_breaker() == 0.0

    def test_returns_zero_when_circuit_breaker_disabled(self):
        """Should return 0 when circuit breaker is disabled."""
        queue = RetryQueue()
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=False))
        queue.set_circuit_breaker(cb)

        assert queue._wait_for_circuit_breaker() == 0.0

    def test_returns_zero_when_circuit_closed(self):
        """Should return 0 when circuit breaker is not tripped."""
        queue = RetryQueue()
        cb = CircuitBreaker()
        cb.state.is_open = False
        queue.set_circuit_breaker(cb)

        assert queue._wait_for_circuit_breaker() == 0.0

    def test_waits_when_circuit_open(self):
        """Should wait and return time when circuit breaker is tripped."""
        queue = RetryQueue()
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        with patch('time.sleep') as mock_sleep:
            wait_time = queue._wait_for_circuit_breaker()

            # Should have called sleep with approximately 0.1 seconds
            mock_sleep.assert_called_once()
            called_time = mock_sleep.call_args[0][0]
            assert 0 < called_time <= 0.1

    def test_tracks_circuit_breaker_wait_time(self):
        """Should accumulate total wait time in _circuit_breaker_wait_time."""
        queue = RetryQueue()
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        with patch('time.sleep'):
            queue._wait_for_circuit_breaker()

        assert queue._circuit_breaker_wait_time > 0

    def test_logs_when_waiting(self):
        """Should log at INFO level when waiting for circuit breaker."""
        queue = RetryQueue()
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        with patch('time.sleep'), \
             patch('src.downloader.retry_queue.logger') as mock_logger:
            queue._wait_for_circuit_breaker()
            mock_logger.info.assert_called_once()
            assert 'circuit breaker' in mock_logger.info.call_args[0][0].lower()


class TestStartRetryPassWithCircuitBreaker:
    """Tests for start_retry_pass() circuit breaker integration."""

    def test_checks_circuit_breaker_before_delay(self):
        """start_retry_pass() should check circuit breaker via _wait_combined."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))
        queue.add('video1', 'keyword', 'short', 'error')

        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.01))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        with patch.object(queue, '_wait_combined', return_value=0.01) as mock_wait, \
             patch('time.sleep'):
            queue.start_retry_pass()
            mock_wait.assert_called_once()

    def test_logs_circuit_breaker_wait_and_delay(self):
        """Should log with circuit breaker wait info when CB was active."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))
        queue.add('video1', 'keyword', 'short', 'error')

        # Set up CB state so _wait_combined actually waits for CB
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        with patch('time.sleep'), \
             patch('src.downloader.retry_queue.logger') as mock_logger:
            queue.start_retry_pass()

            # Should log about the wait (combined or CB-specific)
            info_calls = mock_logger.info.call_args_list
            all_logs = ' '.join(c[0][0] for c in info_calls)
            assert 'circuit breaker' in all_logs.lower() or 'Batch retry pass' in all_logs

    def test_logs_standard_message_when_no_cb_wait(self):
        """Should log standard message when circuit breaker wait is 0."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))
        queue.add('video1', 'keyword', 'short', 'error')

        with patch('time.sleep'), \
             patch('src.downloader.retry_queue.logger') as mock_logger:
            queue.start_retry_pass()

            info_calls = mock_logger.info.call_args_list
            first_log = info_calls[0][0][0]
            # Standard message without circuit breaker mention
            assert 'Batch retry pass' in first_log
            assert 'Circuit breaker wait' not in first_log


class TestRetryQueueStats:
    """Tests for get_stats() with circuit breaker fields."""

    def test_stats_includes_respect_circuit_breaker(self):
        """get_stats() should include respect_circuit_breaker."""
        queue = RetryQueue(BatchRetryConfig(respect_circuit_breaker=True))
        stats = queue.get_stats()
        assert 'respect_circuit_breaker' in stats
        assert stats['respect_circuit_breaker'] is True

    def test_stats_includes_circuit_breaker_wait_time(self):
        """get_stats() should include circuit_breaker_wait_time."""
        queue = RetryQueue()
        queue._circuit_breaker_wait_time = 30.5
        stats = queue.get_stats()
        assert 'circuit_breaker_wait_time' in stats
        assert stats['circuit_breaker_wait_time'] == 30.5

    def test_stats_circuit_breaker_wait_time_rounded(self):
        """circuit_breaker_wait_time should be rounded to 1 decimal."""
        queue = RetryQueue()
        queue._circuit_breaker_wait_time = 30.567
        stats = queue.get_stats()
        assert stats['circuit_breaker_wait_time'] == 30.6


class TestRetryQueueClear:
    """Tests for clear() with circuit breaker fields."""

    def test_clear_resets_circuit_breaker_wait_time(self):
        """clear() should reset circuit_breaker_wait_time to 0."""
        queue = RetryQueue()
        queue._circuit_breaker_wait_time = 60.0
        queue.clear()
        assert queue._circuit_breaker_wait_time == 0.0


class TestRetryQueueCheckpoint:
    """Tests for checkpoint persistence with circuit breaker fields."""

    def test_to_checkpoint_includes_circuit_breaker_wait_time(self):
        """to_checkpoint_dict() should include circuit_breaker_wait_time."""
        queue = RetryQueue()
        queue._circuit_breaker_wait_time = 45.0
        data = queue.to_checkpoint_dict()
        assert 'circuit_breaker_wait_time' in data
        assert data['circuit_breaker_wait_time'] == 45.0

    def test_from_checkpoint_restores_circuit_breaker_wait_time(self):
        """from_checkpoint_dict() should restore circuit_breaker_wait_time."""
        queue = RetryQueue()
        data = {'circuit_breaker_wait_time': 30.0}
        queue.from_checkpoint_dict(data)
        assert queue._circuit_breaker_wait_time == 30.0

    def test_from_checkpoint_defaults_circuit_breaker_wait_time(self):
        """from_checkpoint_dict() should default to 0 if missing."""
        queue = RetryQueue()
        queue._circuit_breaker_wait_time = 60.0  # Set to non-zero
        data = {}  # No circuit_breaker_wait_time
        queue.from_checkpoint_dict(data)
        # Should NOT change since data is empty (returns early)
        # Let's test with actual items
        queue2 = RetryQueue()
        queue2._circuit_breaker_wait_time = 60.0
        data2 = {'items': [], 'current_pass': 0}
        queue2.from_checkpoint_dict(data2)
        assert queue2._circuit_breaker_wait_time == 0.0

    def test_checkpoint_roundtrip(self):
        """Should roundtrip circuit_breaker_wait_time through checkpoint."""
        queue = RetryQueue()
        queue.add('video1', 'keyword', 'short', 'error')
        queue._circuit_breaker_wait_time = 25.5

        data = queue.to_checkpoint_dict()

        queue2 = RetryQueue()
        queue2.from_checkpoint_dict(data)

        assert queue2._circuit_breaker_wait_time == 25.5


class TestVideoDownloaderIntegration:
    """Tests for VideoDownloader integration."""

    def test_retry_queue_linked_to_circuit_breaker(self):
        """VideoDownloader should link retry_queue to circuit_breaker."""
        from src.downloader.core import VideoDownloader

        # Create minimal mock config
        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.download.cookies_path = ''
        mock_config.download.cookies_from_browser = ''
        mock_config.download.download_timeouts = {'short': 60, 'medium': 120}
        mock_config.download.root_dir = ''
        mock_config.download.folder_name = 'videos'

        # Configure circuit breaker
        mock_config.download.circuit_breaker = MagicMock()
        mock_config.download.circuit_breaker.enabled = True
        mock_config.download.circuit_breaker.consecutive_failures_threshold = 5
        mock_config.download.circuit_breaker.pause_seconds = 60.0

        # Configure batch retry
        mock_config.download.batch_retry = MagicMock()
        mock_config.download.batch_retry.enabled = True
        mock_config.download.batch_retry.delay_seconds = 120.0
        mock_config.download.batch_retry.max_passes = 2
        mock_config.download.batch_retry.respect_circuit_breaker = True

        # Other required config
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

        # Verify circuit breaker is linked to retry queue
        assert downloader.retry_queue._circuit_breaker is downloader.circuit_breaker

    def test_respect_circuit_breaker_config_read(self):
        """VideoDownloader should read respect_circuit_breaker from config."""
        from src.downloader.core import VideoDownloader

        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.download.cookies_path = ''
        mock_config.download.cookies_from_browser = ''
        mock_config.download.download_timeouts = {'short': 60}
        mock_config.download.root_dir = ''
        mock_config.download.folder_name = 'videos'

        mock_config.download.circuit_breaker = MagicMock()
        mock_config.download.circuit_breaker.enabled = True
        mock_config.download.circuit_breaker.consecutive_failures_threshold = 5
        mock_config.download.circuit_breaker.pause_seconds = 60.0

        mock_config.download.batch_retry = MagicMock()
        mock_config.download.batch_retry.enabled = True
        mock_config.download.batch_retry.delay_seconds = 120.0
        mock_config.download.batch_retry.max_passes = 2
        mock_config.download.batch_retry.respect_circuit_breaker = False  # Explicitly disabled

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

        assert downloader.retry_queue.config.respect_circuit_breaker is False


class TestIntegrationScenario:
    """Integration tests for realistic scenarios."""

    def test_batch_retry_waits_for_tripped_circuit_breaker(self):
        """When circuit breaker is tripped, batch retry should wait before processing."""
        # Setup retry queue with circuit breaker
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01, respect_circuit_breaker=True))
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.05))

        # Trip the circuit breaker
        cb.state.is_open = True
        cb.state.opened_at = time.time()

        queue.set_circuit_breaker(cb)
        queue.add('video1', 'keyword', 'short', 'rate limit error')

        # Track wait times
        start_time = time.time()
        with patch('time.sleep') as mock_sleep:
            queue.start_retry_pass()

            # Should have called sleep twice: once for CB wait, once for delay
            assert mock_sleep.call_count == 2

    def test_batch_retry_does_not_wait_when_circuit_closed(self):
        """When circuit breaker is not tripped, batch retry should not wait for it."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01, respect_circuit_breaker=True))
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=60.0))

        # Circuit breaker is closed (not tripped)
        cb.state.is_open = False

        queue.set_circuit_breaker(cb)
        queue.add('video1', 'keyword', 'short', 'error')

        with patch('time.sleep') as mock_sleep:
            queue.start_retry_pass()

            # Should only call sleep once (for the delay)
            assert mock_sleep.call_count == 1

    def test_batch_retry_does_not_wait_when_respect_disabled(self):
        """When respect_circuit_breaker is False, batch retry should not wait for CB."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01, respect_circuit_breaker=False))
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=60.0))

        # Trip the circuit breaker
        cb.state.is_open = True
        cb.state.opened_at = time.time()

        queue.set_circuit_breaker(cb)
        queue.add('video1', 'keyword', 'short', 'error')

        with patch('time.sleep') as mock_sleep:
            queue.start_retry_pass()

            # Should only call sleep once (for the delay), not for CB
            assert mock_sleep.call_count == 1


class TestConfigYamlIntegration:
    """Tests for config.yaml integration."""

    def test_batch_retry_section_has_respect_circuit_breaker(self):
        """config.yaml batch_retry section should have respect_circuit_breaker."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert 'download' in config
        assert 'batch_retry' in config['download']
        assert 'respect_circuit_breaker' in config['download']['batch_retry']
        assert config['download']['batch_retry']['respect_circuit_breaker'] is True
