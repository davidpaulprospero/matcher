"""Tests for retry queue and circuit breaker deadlock fix (US-006).

Verifies that when both circuit breaker AND cookie cooldown are active simultaneously,
the retry queue:
- Uses min(cb_remaining, cooldown_remaining) + buffer instead of sequential waits
- Respects max_combined_wait_seconds cap
- Forces retry with best-available cookie method when cap exceeded
- Logs deadlock detection
"""

import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from src.downloader.retry_queue import RetryQueue, BatchRetryConfig
from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


# --- Mock helpers ---

@dataclass
class MockCookieConfig:
    """Mock cookie rotation config."""
    enabled: bool = True
    cooldown_seconds: int = 300
    cookie_files: list = None
    rotation_strategy: str = "on_error"
    max_rotations_per_session: int = 0
    rotate_on_errors: list = None

    def __post_init__(self):
        if self.cookie_files is None:
            self.cookie_files = []
        if self.rotate_on_errors is None:
            self.rotate_on_errors = ["429"]


class MockCookieRotator:
    """Mock CookieRotator for testing."""

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


# --- Config Tests ---

class TestMaxCombinedWaitConfig:
    """Tests for max_combined_wait_seconds config option."""

    @pytest.mark.fast
    def test_default_value(self):
        """max_combined_wait_seconds defaults to 300."""
        config = BatchRetryConfig()
        assert config.max_combined_wait_seconds == 300.0

    @pytest.mark.fast
    def test_custom_value(self):
        """max_combined_wait_seconds can be customized."""
        config = BatchRetryConfig(max_combined_wait_seconds=600.0)
        assert config.max_combined_wait_seconds == 600.0

    @pytest.mark.fast
    def test_in_config_section(self):
        """max_combined_wait_seconds in config/sections/download.py."""
        from src.config.sections.download import BatchRetryConfig as ConfigBRC
        config = ConfigBRC()
        assert hasattr(config, 'max_combined_wait_seconds')
        assert config.max_combined_wait_seconds == 300.0

    @pytest.mark.fast
    def test_in_stats(self):
        """max_combined_wait_seconds appears in get_stats()."""
        queue = RetryQueue(BatchRetryConfig(max_combined_wait_seconds=200.0))
        stats = queue.get_stats()
        assert 'max_combined_wait_seconds' in stats
        assert stats['max_combined_wait_seconds'] == 200.0

    @pytest.mark.fast
    def test_forced_retry_in_stats(self):
        """forced_retry flag appears in get_stats()."""
        queue = RetryQueue()
        stats = queue.get_stats()
        assert 'forced_retry' in stats
        assert stats['forced_retry'] is False

    @pytest.mark.fast
    def test_in_config_yaml(self):
        """max_combined_wait_seconds exists in config.yaml."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert 'download' in config
        assert 'batch_retry' in config['download']
        assert 'max_combined_wait_seconds' in config['download']['batch_retry']
        assert config['download']['batch_retry']['max_combined_wait_seconds'] == 300.0


# --- Non-blocking remaining time helpers ---

class TestGetCbRemaining:
    """Tests for _get_cb_remaining() non-blocking check."""

    @pytest.mark.fast
    def test_no_circuit_breaker_returns_zero(self):
        """Returns 0 when no CB linked."""
        queue = RetryQueue()
        assert queue._get_cb_remaining() == 0.0

    @pytest.mark.fast
    def test_respect_disabled_returns_zero(self):
        """Returns 0 when respect_circuit_breaker is False."""
        queue = RetryQueue(BatchRetryConfig(respect_circuit_breaker=False))
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=60))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)
        assert queue._get_cb_remaining() == 0.0

    @pytest.mark.fast
    def test_closed_returns_zero(self):
        """Returns 0 when CB not tripped."""
        queue = RetryQueue()
        cb = CircuitBreaker()
        cb.state.is_open = False
        queue.set_circuit_breaker(cb)
        assert queue._get_cb_remaining() == 0.0

    @pytest.mark.fast
    def test_open_returns_remaining(self):
        """Returns positive remaining time when CB is open."""
        queue = RetryQueue()
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=60.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 10.0  # opened 10s ago
        queue.set_circuit_breaker(cb)

        remaining = queue._get_cb_remaining()
        assert 49.0 < remaining <= 50.0

    @pytest.mark.fast
    def test_elapsed_returns_zero(self):
        """Returns 0 when CB pause already elapsed."""
        queue = RetryQueue()
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=5.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 100.0  # way past
        queue.set_circuit_breaker(cb)

        assert queue._get_cb_remaining() == 0.0


class TestGetCookieCooldownRemaining:
    """Tests for _get_cookie_cooldown_remaining() non-blocking check."""

    @pytest.mark.fast
    def test_no_rotator_returns_zero(self):
        """Returns 0 when no rotator linked."""
        queue = RetryQueue()
        assert queue._get_cookie_cooldown_remaining() == 0.0

    @pytest.mark.fast
    def test_disabled_returns_zero(self):
        """Returns 0 when wait_for_cookie_cooldown disabled."""
        queue = RetryQueue(BatchRetryConfig(wait_for_cookie_cooldown=False))
        rotator = MockCookieRotator(available=0)
        queue.set_cookie_rotator(rotator)
        assert queue._get_cookie_cooldown_remaining() == 0.0

    @pytest.mark.fast
    def test_cookies_available_returns_zero(self):
        """Returns 0 when cookies available."""
        queue = RetryQueue()
        rotator = MockCookieRotator(available=2)
        queue.set_cookie_rotator(rotator)
        assert queue._get_cookie_cooldown_remaining() == 0.0

    @pytest.mark.fast
    def test_cooldown_active_returns_remaining(self):
        """Returns remaining seconds when all cookies in cooldown."""
        queue = RetryQueue()
        config = MockCookieConfig(cooldown_seconds=300)
        rotator = MockCookieRotator(config=config, available=0)
        now = time.time()
        rotator._failed_cookies = {
            '/path/cookie1.txt': now - 250,  # 50s remaining
            '/path/cookie2.txt': now - 100,  # 200s remaining
        }
        queue.set_cookie_rotator(rotator)

        remaining = queue._get_cookie_cooldown_remaining()
        # Should be close to 50s (shortest cooldown)
        assert 49.0 < remaining <= 50.5


# --- Combined Wait Tests ---

class TestWaitCombined:
    """Tests for _wait_combined() method."""

    @pytest.mark.fast
    def test_neither_blocking_returns_zero(self):
        """Returns 0 when neither CB nor cooldown active."""
        queue = RetryQueue()
        with patch('time.sleep'):
            wait = queue._wait_combined()
        assert wait == 0.0
        assert queue._forced_retry is False

    @pytest.mark.fast
    def test_only_cb_blocking_uses_cb_wait(self):
        """When only CB blocks, delegates to _wait_for_circuit_breaker."""
        queue = RetryQueue()
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        with patch('time.sleep'):
            wait = queue._wait_combined()

        assert wait > 0
        assert queue._forced_retry is False
        assert queue._circuit_breaker_wait_time > 0

    @pytest.mark.fast
    def test_only_cooldown_blocking_uses_cooldown_wait(self):
        """When only cooldown blocks, delegates to _wait_for_cookie_cooldown."""
        queue = RetryQueue()
        config = MockCookieConfig(cooldown_seconds=10)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time() - 5}  # 5s remaining
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep'):
            wait = queue._wait_combined()

        assert wait > 0
        assert queue._forced_retry is False
        assert queue._cookie_cooldown_wait_time > 0

    @pytest.mark.fast
    def test_both_blocking_uses_min_plus_buffer(self):
        """When both block, waits min(cb, cooldown) + 5s buffer."""
        queue = RetryQueue(BatchRetryConfig(max_combined_wait_seconds=500.0))
        # CB: 30s remaining
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=40.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 10.0  # 30s remaining
        queue.set_circuit_breaker(cb)

        # Cooldown: 50s remaining
        config = MockCookieConfig(cooldown_seconds=100)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time() - 50}  # 50s remaining
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep') as mock_sleep:
            wait = queue._wait_combined()

        # Should wait min(30, 50) + 5 = 35s
        assert 34.0 < wait <= 36.0
        mock_sleep.assert_called_once()
        called_time = mock_sleep.call_args[0][0]
        assert 34.0 < called_time <= 36.0
        assert queue._forced_retry is False

    @pytest.mark.fast
    def test_combined_exceeds_max_forces_retry(self):
        """When combined estimate exceeds max_combined_wait_seconds, forces retry."""
        queue = RetryQueue(BatchRetryConfig(max_combined_wait_seconds=20.0))

        # CB: 100s remaining
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=110.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 10.0
        queue.set_circuit_breaker(cb)

        # Cooldown: 200s remaining
        config = MockCookieConfig(cooldown_seconds=300)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time() - 100}
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep') as mock_sleep:
            wait = queue._wait_combined()

        # min(100, 200) + 5 = 105 > 20 => forced
        assert wait == 0.0
        assert queue._forced_retry is True
        mock_sleep.assert_not_called()

    @pytest.mark.fast
    def test_forced_retry_logs_deadlock_detection(self):
        """Forced retry logs a warning with deadlock details."""
        queue = RetryQueue(BatchRetryConfig(max_combined_wait_seconds=10.0))

        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=60.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        config = MockCookieConfig(cooldown_seconds=300)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time() - 100}
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep'), \
             patch('src.downloader.retry_queue.logger') as mock_logger:
            queue._wait_combined()

        # Should log warning with deadlock message
        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args[0][0]
        assert 'forcing retry' in msg.lower()
        assert 'CB+cooldown' in msg
        assert 'best-available' in msg


class TestCombinedWaitRespectsCap:
    """Tests that combined wait respects max_combined_wait_seconds."""

    @pytest.mark.fast
    def test_cap_at_300_default(self):
        """Default 300s cap prevents excessive waiting."""
        queue = RetryQueue()  # default max_combined_wait_seconds=300

        # CB: 200s remaining
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=200.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        # Cooldown: 400s remaining
        config = MockCookieConfig(cooldown_seconds=500)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time() - 100}
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep') as mock_sleep:
            wait = queue._wait_combined()

        # min(200, 400) + 5 = 205 < 300 => should wait normally
        assert wait > 0
        assert queue._forced_retry is False

    @pytest.mark.fast
    def test_cap_exceeded_forces_retry(self):
        """Cap exceeded forces immediate retry."""
        queue = RetryQueue(BatchRetryConfig(max_combined_wait_seconds=100.0))

        # CB: 150s remaining
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=160.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 10.0
        queue.set_circuit_breaker(cb)

        # Cooldown: 250s remaining
        config = MockCookieConfig(cooldown_seconds=300)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time() - 50}
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep') as mock_sleep:
            wait = queue._wait_combined()

        # min(150, 250) + 5 = 155 > 100 => forced
        assert wait == 0.0
        assert queue._forced_retry is True
        mock_sleep.assert_not_called()

    @pytest.mark.fast
    def test_forced_retry_uses_best_available_cookie(self):
        """When forced, should retry with whatever cookie is available (not waiting)."""
        queue = RetryQueue(BatchRetryConfig(
            max_combined_wait_seconds=10.0,
            delay_seconds=0.01
        ))
        queue.add('video1', 'keyword', 'short', 'rate limit')

        # CB: 200s remaining
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=200.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        # Cooldown: 300s remaining (all cookies exhausted)
        config = MockCookieConfig(cooldown_seconds=300)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time()}
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep'):
            queue.start_retry_pass()

        # forced_retry should be True - caller should use best-available cookie
        assert queue.forced_retry is True
        # Pass should have been started despite deadlock
        assert queue.current_pass == 1


# --- Start Retry Pass Integration ---

class TestStartRetryPassWithDeadlock:
    """Tests for start_retry_pass() with combined wait logic."""

    @pytest.mark.fast
    def test_normal_pass_no_blocking(self):
        """Normal pass with no CB or cooldown blocking."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))
        queue.add('video1', 'keyword', 'short', 'error')

        with patch('time.sleep'):
            pass_num = queue.start_retry_pass()

        assert pass_num == 1
        assert queue._forced_retry is False

    @pytest.mark.fast
    def test_forced_pass_logs_correctly(self):
        """Forced pass logs FORCED message."""
        queue = RetryQueue(BatchRetryConfig(
            max_combined_wait_seconds=5.0,
            delay_seconds=0.01
        ))
        queue.add('video1', 'keyword', 'short', 'error')

        # Force deadlock
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=200.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        config = MockCookieConfig(cooldown_seconds=300)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time()}
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep'), \
             patch('src.downloader.retry_queue.logger') as mock_logger:
            queue.start_retry_pass()

        # Look for FORCED in INFO logs
        info_messages = [c[0][0] for c in mock_logger.info.call_args_list]
        forced_logs = [m for m in info_messages if 'FORCED' in m]
        assert len(forced_logs) >= 1
        assert 'best-available cookie' in forced_logs[0].lower()

    @pytest.mark.fast
    def test_clear_resets_forced_retry(self):
        """clear() resets _forced_retry flag."""
        queue = RetryQueue()
        queue._forced_retry = True
        queue.clear()
        assert queue._forced_retry is False

    @pytest.mark.fast
    def test_forced_retry_property(self):
        """forced_retry property reflects _forced_retry."""
        queue = RetryQueue()
        assert queue.forced_retry is False
        queue._forced_retry = True
        assert queue.forced_retry is True


# --- Edge Cases ---

class TestEdgeCases:
    """Edge cases for deadlock detection."""

    @pytest.mark.fast
    def test_cb_only_at_exact_cap_allows_wait(self):
        """When only CB remaining equals cap, waits (no deadlock)."""
        queue = RetryQueue(BatchRetryConfig(max_combined_wait_seconds=50.0))
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=50.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        with patch('time.sleep'):
            wait = queue._wait_combined()

        # Only CB blocking - uses _wait_for_circuit_breaker, not combined path
        assert wait > 0
        assert queue._forced_retry is False

    @pytest.mark.fast
    def test_both_blocking_just_under_cap(self):
        """Combined wait just under cap proceeds normally."""
        queue = RetryQueue(BatchRetryConfig(max_combined_wait_seconds=50.0))

        # CB: 20s remaining
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=30.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 10.0
        queue.set_circuit_breaker(cb)

        # Cooldown: 40s remaining
        config = MockCookieConfig(cooldown_seconds=50)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time() - 10}
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep'):
            wait = queue._wait_combined()

        # min(20, 40) + 5 = 25 < 50 => should wait normally
        assert wait > 0
        assert queue._forced_retry is False

    @pytest.mark.fast
    def test_both_blocking_just_over_cap(self):
        """Combined wait just over cap forces retry."""
        queue = RetryQueue(BatchRetryConfig(max_combined_wait_seconds=20.0))

        # CB: 20s remaining
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=30.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time() - 10.0
        queue.set_circuit_breaker(cb)

        # Cooldown: 40s remaining
        config = MockCookieConfig(cooldown_seconds=50)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time() - 10}
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep') as mock_sleep:
            wait = queue._wait_combined()

        # min(20, 40) + 5 = 25 > 20 => forced
        assert wait == 0.0
        assert queue._forced_retry is True
        mock_sleep.assert_not_called()

    @pytest.mark.fast
    def test_zero_max_combined_wait_always_forces(self):
        """max_combined_wait_seconds=0 always forces when both blocking."""
        queue = RetryQueue(BatchRetryConfig(max_combined_wait_seconds=0.0))

        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=10.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        config = MockCookieConfig(cooldown_seconds=10)
        rotator = MockCookieRotator(config=config, available=0)
        rotator._failed_cookies = {'/path/c.txt': time.time() - 5}
        queue.set_cookie_rotator(rotator)

        with patch('time.sleep') as mock_sleep:
            wait = queue._wait_combined()

        assert wait == 0.0
        assert queue._forced_retry is True
        mock_sleep.assert_not_called()
