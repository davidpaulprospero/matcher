"""Tests for cookie cooldown synchronization with batch retry queue.

Tests US-007 acceptance criteria:
1. RetryQueue gains set_cookie_rotator() method to access cooldown info
2. Before batch retry, check if any required cookies are in cooldown
3. If cookies in cooldown, extend batch retry delay to wait for cooldown expiry
4. New config option batch_retry.wait_for_cookie_cooldown (default: true)
5. Log when batch retry delay extended due to cookie cooldown
6. Tests verify batch retry waits for cookie cooldown expiry
"""

import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from src.downloader.retry_queue import RetryQueue, BatchRetryConfig


@dataclass
class MockCookieConfig:
    """Mock cookie rotation config for testing."""
    enabled: bool = True
    cookie_files: list = None
    cooldown_seconds: int = 300
    rotation_strategy: str = "on_error"
    max_rotations_per_session: int = 0
    rotate_on_errors: list = None

    def __post_init__(self):
        if self.cookie_files is None:
            self.cookie_files = []
        if self.rotate_on_errors is None:
            self.rotate_on_errors = ["429"]


class MockCookieRotator:
    """Mock CookieRotator for testing RetryQueue integration."""

    def __init__(self, config=None, available=1):
        self.config = config or MockCookieConfig(enabled=True, cooldown_seconds=300)
        self._failed_cookies = {}
        self._available_count = available
        self._is_enabled = True

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    @property
    def available_cookies(self) -> int:
        return self._available_count


class TestCookieCooldownConfig:
    """Test BatchRetryConfig wait_for_cookie_cooldown option."""

    def test_default_value_true(self):
        """Test wait_for_cookie_cooldown defaults to True."""
        config = BatchRetryConfig()
        assert config.wait_for_cookie_cooldown is True

    def test_custom_value_false(self):
        """Test wait_for_cookie_cooldown can be set to False."""
        config = BatchRetryConfig(wait_for_cookie_cooldown=False)
        assert config.wait_for_cookie_cooldown is False

    def test_config_in_stats(self):
        """Test wait_for_cookie_cooldown appears in stats."""
        queue = RetryQueue(BatchRetryConfig(wait_for_cookie_cooldown=True))
        stats = queue.get_stats()
        assert 'wait_for_cookie_cooldown' in stats
        assert stats['wait_for_cookie_cooldown'] is True


class TestSetCookieRotator:
    """Test RetryQueue.set_cookie_rotator() method."""

    def test_set_cookie_rotator(self):
        """Test linking cookie rotator to retry queue."""
        queue = RetryQueue()
        rotator = MockCookieRotator()

        queue.set_cookie_rotator(rotator)

        assert queue._cookie_rotator is rotator

    def test_set_cookie_rotator_logs(self):
        """Test set_cookie_rotator logs debug message."""
        queue = RetryQueue()
        rotator = MockCookieRotator()

        with patch('src.downloader.retry_queue.logger') as mock_logger:
            queue.set_cookie_rotator(rotator)
            mock_logger.debug.assert_called_once()
            assert 'cookie rotator' in mock_logger.debug.call_args[0][0].lower()


class TestWaitForCookieCooldown:
    """Test _wait_for_cookie_cooldown method."""

    def test_no_rotator_returns_zero(self):
        """Test returns 0 when no cookie rotator is linked."""
        queue = RetryQueue()

        wait_time = queue._wait_for_cookie_cooldown()

        assert wait_time == 0.0

    def test_disabled_config_returns_zero(self):
        """Test returns 0 when wait_for_cookie_cooldown is disabled."""
        queue = RetryQueue(BatchRetryConfig(wait_for_cookie_cooldown=False))
        rotator = MockCookieRotator(available=0)  # All in cooldown
        queue.set_cookie_rotator(rotator)

        wait_time = queue._wait_for_cookie_cooldown()

        assert wait_time == 0.0

    def test_rotator_disabled_returns_zero(self):
        """Test returns 0 when cookie rotator is disabled."""
        queue = RetryQueue()
        rotator = MockCookieRotator()
        rotator._is_enabled = False
        queue.set_cookie_rotator(rotator)

        wait_time = queue._wait_for_cookie_cooldown()

        assert wait_time == 0.0

    def test_cookies_available_returns_zero(self):
        """Test returns 0 when cookies are available."""
        queue = RetryQueue()
        rotator = MockCookieRotator(available=2)
        queue.set_cookie_rotator(rotator)

        wait_time = queue._wait_for_cookie_cooldown()

        assert wait_time == 0.0

    def test_no_failed_cookies_returns_zero(self):
        """Test returns 0 when no cookies are in failed state."""
        queue = RetryQueue()
        rotator = MockCookieRotator(available=0)  # No available
        rotator._failed_cookies = {}  # But no failed cookies either (empty dict)
        queue.set_cookie_rotator(rotator)

        wait_time = queue._wait_for_cookie_cooldown()

        assert wait_time == 0.0

    def test_waits_for_shortest_cooldown(self):
        """Test waits for shortest cooldown when all cookies unavailable."""
        queue = RetryQueue()
        config = MockCookieConfig(cooldown_seconds=300)  # 5 minute cooldown
        rotator = MockCookieRotator(config=config, available=0)

        # Cookie 1 failed 250s ago (50s remaining)
        # Cookie 2 failed 100s ago (200s remaining)
        now = time.time()
        rotator._failed_cookies = {
            '/path/cookie1.txt': now - 299.9,  # 0.1s remaining
            '/path/cookie2.txt': now - 100,    # 200s remaining
        }
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep') as mock_sleep:
            wait_time = queue._wait_for_cookie_cooldown()

        # Should wait for shortest cooldown (~0.1s)
        assert wait_time > 0
        assert wait_time < 1.0  # Should be close to 0.1s
        mock_sleep.assert_called_once()

    def test_logs_waiting_message(self):
        """Test logs INFO message when waiting for cooldown."""
        queue = RetryQueue()
        config = MockCookieConfig(cooldown_seconds=10)
        rotator = MockCookieRotator(config=config, available=0)

        now = time.time()
        rotator._failed_cookies = {'/path/cookie.txt': now - 5}  # 5s remaining
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep'):
            with patch('src.downloader.retry_queue.logger') as mock_logger:
                queue._wait_for_cookie_cooldown()

                # Should log INFO about waiting
                info_calls = [c for c in mock_logger.info.call_args_list]
                assert len(info_calls) >= 1
                assert 'cookie cooldown' in info_calls[0][0][0].lower()

    def test_accumulates_wait_time(self):
        """Test cookie cooldown wait time is accumulated."""
        queue = RetryQueue()
        config = MockCookieConfig(cooldown_seconds=10)
        rotator = MockCookieRotator(config=config, available=0)

        now = time.time()
        rotator._failed_cookies = {'/path/cookie.txt': now - 8}  # 2s remaining
        queue.set_cookie_rotator(rotator)

        assert queue._cookie_cooldown_wait_time == 0.0

        with patch('time.sleep'):
            queue._wait_for_cookie_cooldown()

        assert queue._cookie_cooldown_wait_time > 0


class TestStartRetryPassCookieCooldown:
    """Test start_retry_pass includes cookie cooldown check."""

    def test_checks_cookie_cooldown(self):
        """Test start_retry_pass calls _wait_for_cookie_cooldown."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))
        queue.add("video1", "keyword1", "short", "Error")

        with patch.object(queue, '_wait_for_cookie_cooldown', return_value=5.0) as mock_wait:
            with patch('time.sleep'):  # Skip the actual delay
                queue.start_retry_pass()

        mock_wait.assert_called_once()

    def test_logs_both_waits(self):
        """Test logs both circuit breaker and cookie cooldown waits."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))
        queue.add("video1", "keyword1", "short", "Error")

        # Mock both wait methods to return non-zero
        with patch.object(queue, '_wait_for_circuit_breaker', return_value=10.0):
            with patch.object(queue, '_wait_for_cookie_cooldown', return_value=5.0):
                with patch('time.sleep'):
                    with patch('src.downloader.retry_queue.logger') as mock_logger:
                        queue.start_retry_pass()

        # Check that log includes both wait types
        info_calls = mock_logger.info.call_args_list
        log_message = str(info_calls)
        assert 'circuit breaker' in log_message
        assert 'cookie cooldown' in log_message

    def test_logs_only_cookie_wait(self):
        """Test logs only cookie cooldown when circuit breaker didn't wait."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))
        queue.add("video1", "keyword1", "short", "Error")

        with patch.object(queue, '_wait_for_circuit_breaker', return_value=0.0):
            with patch.object(queue, '_wait_for_cookie_cooldown', return_value=5.0):
                with patch('time.sleep'):
                    with patch('src.downloader.retry_queue.logger') as mock_logger:
                        queue.start_retry_pass()

        info_calls = mock_logger.info.call_args_list
        log_message = str(info_calls)
        assert 'cookie cooldown' in log_message


class TestCookieCooldownStats:
    """Test cookie cooldown tracking in stats."""

    def test_stats_include_cookie_cooldown_wait_time(self):
        """Test stats include cookie_cooldown_wait_time."""
        queue = RetryQueue()
        queue._cookie_cooldown_wait_time = 15.5

        stats = queue.get_stats()

        assert 'cookie_cooldown_wait_time' in stats
        assert stats['cookie_cooldown_wait_time'] == 15.5

    def test_stats_rounds_wait_time(self):
        """Test cookie_cooldown_wait_time is rounded to 1 decimal."""
        queue = RetryQueue()
        queue._cookie_cooldown_wait_time = 15.5678

        stats = queue.get_stats()

        assert stats['cookie_cooldown_wait_time'] == 15.6


class TestCookieCooldownCheckpoint:
    """Test cookie cooldown persistence in checkpoint."""

    def test_checkpoint_includes_cookie_cooldown(self):
        """Test checkpoint dict includes cookie_cooldown_wait_time."""
        queue = RetryQueue()
        queue._cookie_cooldown_wait_time = 30.0

        data = queue.to_checkpoint_dict()

        assert 'cookie_cooldown_wait_time' in data
        assert data['cookie_cooldown_wait_time'] == 30.0

    def test_restore_from_checkpoint(self):
        """Test cookie_cooldown_wait_time is restored from checkpoint."""
        queue = RetryQueue()
        data = {
            'items': [],
            'current_pass': 0,
            'completed_ids': [],
            'failed_ids': [],
            'total_added': 0,
            'total_retried': 0,
            'circuit_breaker_wait_time': 10.0,
            'cookie_cooldown_wait_time': 25.0,
        }

        queue.from_checkpoint_dict(data)

        assert queue._cookie_cooldown_wait_time == 25.0

    def test_restore_missing_field_defaults_zero(self):
        """Test missing cookie_cooldown_wait_time defaults to 0."""
        queue = RetryQueue()
        data = {
            'items': [],
            'current_pass': 0,
            'completed_ids': [],
            'failed_ids': [],
            'total_added': 0,
            'total_retried': 0,
            # No cookie_cooldown_wait_time
        }

        queue.from_checkpoint_dict(data)

        assert queue._cookie_cooldown_wait_time == 0.0

    def test_checkpoint_roundtrip(self):
        """Test save and restore roundtrip preserves cookie cooldown."""
        queue1 = RetryQueue()
        queue1._cookie_cooldown_wait_time = 42.5

        data = queue1.to_checkpoint_dict()

        queue2 = RetryQueue()
        queue2.from_checkpoint_dict(data)

        assert queue2._cookie_cooldown_wait_time == 42.5


class TestCookieCooldownClear:
    """Test clear() resets cookie cooldown state."""

    def test_clear_resets_cookie_cooldown(self):
        """Test clear() resets cookie_cooldown_wait_time to 0."""
        queue = RetryQueue()
        queue._cookie_cooldown_wait_time = 100.0

        queue.clear()

        assert queue._cookie_cooldown_wait_time == 0.0


class TestVideoDownloaderCookieRotatorIntegration:
    """Integration tests for cookie rotator linking in VideoDownloader."""

    def test_cookie_rotator_linked_to_retry_queue(self):
        """Test cookie rotator is linked to retry queue when enabled."""
        from src.downloader.core import VideoDownloader
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.download.batch_retry = MagicMock()
        mock_config.download.batch_retry.enabled = True
        mock_config.download.batch_retry.delay_seconds = 60.0
        mock_config.download.batch_retry.max_passes = 2
        mock_config.download.batch_retry.respect_circuit_breaker = True
        mock_config.download.batch_retry.wait_for_cookie_cooldown = True
        mock_config.download.circuit_breaker = MagicMock()
        mock_config.download.circuit_breaker.enabled = False
        mock_config.download.speed_tracking = MagicMock()
        mock_config.download.speed_tracking.enabled = False
        mock_config.download.rate_limit = None
        mock_config.download.vpn = None
        mock_config.download.cookies_from_browser = ''
        mock_config.download.cookies_path = ''
        mock_config.cache_dir = '/tmp/test_cache'
        mock_config.downloaded_videos_dir = '/tmp/test_videos'

        # Enable cookie rotation
        mock_config.download.cookie_rotation = MagicMock()
        mock_config.download.cookie_rotation.enabled = True
        mock_config.download.cookie_rotation.cookie_files = ['/path/to/cookie.txt']
        mock_config.download.cookie_rotation.cooldown_seconds = 300
        mock_config.download.cookie_rotation.rotation_strategy = 'on_error'
        mock_config.download.cookie_rotation.max_rotations_per_session = 0
        mock_config.download.cookie_rotation.rotate_on_errors = ['429']

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.CookieRotator') as MockRotator:
                                    mock_rotator_instance = MagicMock()
                                    mock_rotator_instance.is_enabled = True
                                    MockRotator.return_value = mock_rotator_instance

                                    downloader = VideoDownloader(mock_config)

        # Verify cookie rotator was linked to retry queue
        assert downloader.retry_queue._cookie_rotator is mock_rotator_instance

    def test_cookie_rotator_not_linked_when_disabled(self):
        """Test cookie rotator is not linked when cookie rotation disabled."""
        from src.downloader.core import VideoDownloader
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.download.batch_retry = MagicMock()
        mock_config.download.batch_retry.enabled = True
        mock_config.download.batch_retry.delay_seconds = 60.0
        mock_config.download.batch_retry.max_passes = 2
        mock_config.download.batch_retry.respect_circuit_breaker = True
        mock_config.download.batch_retry.wait_for_cookie_cooldown = True
        mock_config.download.circuit_breaker = MagicMock()
        mock_config.download.circuit_breaker.enabled = False
        mock_config.download.speed_tracking = MagicMock()
        mock_config.download.speed_tracking.enabled = False
        mock_config.download.rate_limit = None
        mock_config.download.cookie_rotation = None  # Disabled
        mock_config.download.vpn = None
        mock_config.download.cookies_from_browser = ''
        mock_config.download.cookies_path = ''
        mock_config.cache_dir = '/tmp/test_cache'
        mock_config.downloaded_videos_dir = '/tmp/test_videos'

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                downloader = VideoDownloader(mock_config)

        # Verify cookie rotator was NOT linked
        assert downloader.retry_queue._cookie_rotator is None


class TestConfigSectionIntegration:
    """Test config section includes wait_for_cookie_cooldown."""

    def test_config_section_has_option(self):
        """Test BatchRetryConfig in download config has option."""
        from src.config.sections.download import BatchRetryConfig

        config = BatchRetryConfig()
        assert hasattr(config, 'wait_for_cookie_cooldown')
        assert config.wait_for_cookie_cooldown is True

    def test_config_dict_conversion(self):
        """Test dict is converted properly with new option."""
        from src.config.sections.download import DownloadConfig

        config_dict = {
            'batch_retry': {
                'enabled': True,
                'delay_seconds': 90.0,
                'max_passes': 3,
                'respect_circuit_breaker': True,
                'wait_for_cookie_cooldown': False  # Explicitly set
            }
        }

        config = DownloadConfig(batch_retry=config_dict['batch_retry'])

        assert config.batch_retry.wait_for_cookie_cooldown is False
