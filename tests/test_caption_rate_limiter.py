"""Tests for caption rate limiter with jitter.

Tests the UnifiedCaptionRateLimiter and IntegratedRateLimiter classes:
- Backoff calculation with exponential growth
- Jitter bounds verification
- Config integration
- Integration with RateLimitTracker
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.caption.rate_limiter import (
    RateLimitConfig,
    UnifiedCaptionRateLimiter,
    IntegratedRateLimiter,
    create_rate_limiter_from_config,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.enabled is True
        assert config.base_delay_seconds == 2.0
        assert config.max_delay_seconds == 120.0
        assert config.jitter_factor == 0.3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            enabled=False,
            base_delay_seconds=5.0,
            max_delay_seconds=60.0,
            jitter_factor=0.5,
        )
        assert config.enabled is False
        assert config.base_delay_seconds == 5.0
        assert config.max_delay_seconds == 60.0
        assert config.jitter_factor == 0.5

    def test_invalid_base_delay(self):
        """Test validation rejects non-positive base delay."""
        with pytest.raises(ValueError, match="base_delay_seconds must be positive"):
            RateLimitConfig(base_delay_seconds=0)

        with pytest.raises(ValueError, match="base_delay_seconds must be positive"):
            RateLimitConfig(base_delay_seconds=-1.0)

    def test_invalid_max_delay(self):
        """Test validation rejects max < base delay."""
        with pytest.raises(ValueError, match="max_delay_seconds must be >= base_delay_seconds"):
            RateLimitConfig(base_delay_seconds=10.0, max_delay_seconds=5.0)

    def test_invalid_jitter_factor(self):
        """Test validation rejects jitter factor outside 0-1 range."""
        with pytest.raises(ValueError, match="jitter_factor must be between 0.0 and 1.0"):
            RateLimitConfig(jitter_factor=-0.1)

        with pytest.raises(ValueError, match="jitter_factor must be between 0.0 and 1.0"):
            RateLimitConfig(jitter_factor=1.5)


class TestUnifiedCaptionRateLimiterBackoff:
    """Tests for exponential backoff calculation."""

    def test_first_attempt_no_delay(self):
        """Test first attempt (attempt=0) has no delay."""
        limiter = UnifiedCaptionRateLimiter()
        delay = limiter.calculate_backoff(0)
        assert delay == 0.0

    def test_exponential_backoff_sequence(self):
        """Test exponential backoff: base=2s, so 2, 4, 8, 16..."""
        config = RateLimitConfig(base_delay_seconds=2.0, max_delay_seconds=120.0)
        limiter = UnifiedCaptionRateLimiter(config)

        # attempt 1: 2 * 2^0 = 2s
        assert limiter.calculate_backoff(1) == 2.0
        # attempt 2: 2 * 2^1 = 4s
        assert limiter.calculate_backoff(2) == 4.0
        # attempt 3: 2 * 2^2 = 8s
        assert limiter.calculate_backoff(3) == 8.0
        # attempt 4: 2 * 2^3 = 16s
        assert limiter.calculate_backoff(4) == 16.0

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay_seconds."""
        config = RateLimitConfig(
            base_delay_seconds=2.0,
            max_delay_seconds=10.0,
        )
        limiter = UnifiedCaptionRateLimiter(config)

        # attempt 5: 2 * 2^4 = 32s, but capped at 10s
        assert limiter.calculate_backoff(5) == 10.0
        # attempt 10: would be 1024s, but capped at 10s
        assert limiter.calculate_backoff(10) == 10.0

    def test_disabled_returns_zero(self):
        """Test disabled limiter always returns 0 delay."""
        config = RateLimitConfig(enabled=False)
        limiter = UnifiedCaptionRateLimiter(config)

        assert limiter.calculate_backoff(1) == 0.0
        assert limiter.calculate_backoff(5) == 0.0
        assert limiter.calculate_backoff(10) == 0.0


class TestUnifiedCaptionRateLimiterJitter:
    """Tests for jitter bounds verification."""

    def test_jitter_within_bounds(self):
        """Test jitter stays within ±jitter_factor range."""
        config = RateLimitConfig(
            base_delay_seconds=10.0,
            max_delay_seconds=120.0,
            jitter_factor=0.3,
        )
        limiter = UnifiedCaptionRateLimiter(config)

        # Run multiple times to verify bounds
        for _ in range(100):
            delay = limiter.get_backoff_delay(1)  # Base should be 10s
            # With 0.3 jitter factor: 10 * (1-0.3) to 10 * (1+0.3) = 7 to 13
            assert 7.0 <= delay <= 13.0, f"Delay {delay} outside expected bounds"

    def test_zero_jitter_returns_exact(self):
        """Test zero jitter factor returns exact delay."""
        config = RateLimitConfig(
            base_delay_seconds=5.0,
            max_delay_seconds=120.0,
            jitter_factor=0.0,
        )
        limiter = UnifiedCaptionRateLimiter(config)

        # With 0 jitter, should always return exactly 5.0 for attempt 1
        for _ in range(10):
            assert limiter.get_backoff_delay(1) == 5.0

    def test_jitter_randomness(self):
        """Test jitter produces different values (not always the same)."""
        config = RateLimitConfig(jitter_factor=0.3)
        limiter = UnifiedCaptionRateLimiter(config)

        delays = [limiter.get_backoff_delay(1) for _ in range(20)]
        unique_delays = set(delays)

        # With randomness, we should get multiple unique values
        assert len(unique_delays) > 1, "Jitter should produce varied delays"

    def test_jitter_capped_at_max(self):
        """Test jittered delay doesn't exceed max_delay_seconds."""
        config = RateLimitConfig(
            base_delay_seconds=100.0,
            max_delay_seconds=110.0,
            jitter_factor=0.3,
        )
        limiter = UnifiedCaptionRateLimiter(config)

        # Base would be 100, with +30% jitter = 130, but max is 110
        for _ in range(50):
            delay = limiter.get_backoff_delay(1)
            assert delay <= 110.0, f"Delay {delay} exceeded max"


class TestUnifiedCaptionRateLimiterState:
    """Tests for rate limit state tracking."""

    def test_record_rate_limit_increments(self):
        """Test recording rate limit increments counters."""
        limiter = UnifiedCaptionRateLimiter()

        assert limiter.consecutive_rate_limits == 0
        assert limiter.total_rate_limits == 0

        limiter.record_rate_limit("video1")
        assert limiter.consecutive_rate_limits == 1
        assert limiter.total_rate_limits == 1

        limiter.record_rate_limit("video2")
        assert limiter.consecutive_rate_limits == 2
        assert limiter.total_rate_limits == 2

    def test_record_success_resets_consecutive(self):
        """Test success resets consecutive counter but not total."""
        limiter = UnifiedCaptionRateLimiter()

        limiter.record_rate_limit()
        limiter.record_rate_limit()
        assert limiter.consecutive_rate_limits == 2
        assert limiter.total_rate_limits == 2

        limiter.record_success()
        assert limiter.consecutive_rate_limits == 0
        assert limiter.total_rate_limits == 2

    def test_reset_clears_all(self):
        """Test reset clears all state."""
        limiter = UnifiedCaptionRateLimiter()

        limiter.record_rate_limit()
        limiter.record_rate_limit()
        limiter.reset()

        assert limiter.consecutive_rate_limits == 0
        assert limiter.total_rate_limits == 0

    def test_get_state(self):
        """Test get_state returns complete state dict."""
        config = RateLimitConfig(base_delay_seconds=5.0)
        limiter = UnifiedCaptionRateLimiter(config)
        limiter.record_rate_limit("test_video")

        state = limiter.get_state()

        assert state['enabled'] is True
        assert state['consecutive_rate_limits'] == 1
        assert state['total_rate_limits'] == 1
        assert state['base_delay_seconds'] == 5.0
        assert state['max_delay_seconds'] == 120.0
        assert state['jitter_factor'] == 0.3
        assert state['last_rate_limit_time'] > 0


class TestCreateRateLimiterFromConfig:
    """Tests for config integration factory function."""

    def test_empty_config_uses_defaults(self):
        """Test None/empty config uses default values."""
        limiter = create_rate_limiter_from_config(None)
        assert limiter.config.base_delay_seconds == 2.0
        assert limiter.config.max_delay_seconds == 120.0
        assert limiter.config.jitter_factor == 0.3

        limiter = create_rate_limiter_from_config({})
        assert limiter.config.base_delay_seconds == 2.0

    def test_config_dict_applied(self):
        """Test config dict values are applied correctly."""
        config_dict = {
            'rate_limit': {
                'enabled': False,
                'base_delay_seconds': 5.0,
                'max_delay_seconds': 60.0,
                'jitter_factor': 0.5,
            }
        }
        limiter = create_rate_limiter_from_config(config_dict)

        assert limiter.config.enabled is False
        assert limiter.config.base_delay_seconds == 5.0
        assert limiter.config.max_delay_seconds == 60.0
        assert limiter.config.jitter_factor == 0.5

    def test_partial_config_uses_defaults(self):
        """Test partial config uses defaults for missing values."""
        config_dict = {
            'rate_limit': {
                'base_delay_seconds': 10.0,
            }
        }
        limiter = create_rate_limiter_from_config(config_dict)

        assert limiter.config.base_delay_seconds == 10.0
        assert limiter.config.max_delay_seconds == 120.0  # default
        assert limiter.config.jitter_factor == 0.3  # default


class TestIntegratedRateLimiter:
    """Tests for integrated rate limiter with tracker."""

    def test_without_tracker(self):
        """Test works without tracker."""
        limiter = IntegratedRateLimiter()

        assert limiter.tracker is None
        assert limiter.should_pause() is False

        limiter.record_rate_limit("video1")
        assert limiter.should_pause() is True

    def test_with_tracker(self):
        """Test integration with mock tracker."""
        mock_tracker = MagicMock()
        mock_tracker.should_pause.return_value = True
        mock_tracker.get_recommended_delay.return_value = 5.0
        mock_tracker.get_state_summary.return_value = {'total': 1}

        limiter = IntegratedRateLimiter(tracker=mock_tracker)

        assert limiter.should_pause() is True
        mock_tracker.should_pause.assert_called()

    def test_record_rate_limit_updates_both(self):
        """Test rate limit recorded in both limiter and tracker."""
        mock_tracker = MagicMock()
        limiter = IntegratedRateLimiter(tracker=mock_tracker)

        limiter.record_rate_limit("video1")

        assert limiter.limiter.consecutive_rate_limits == 1
        mock_tracker.record_rate_limit.assert_called_once_with("video1")

    def test_record_success_updates_both(self):
        """Test success recorded in both limiter and tracker."""
        mock_tracker = MagicMock()
        limiter = IntegratedRateLimiter(tracker=mock_tracker)

        limiter.record_rate_limit()
        limiter.record_success()

        assert limiter.limiter.consecutive_rate_limits == 0
        mock_tracker.record_success.assert_called_once()

    def test_get_state_includes_tracker(self):
        """Test get_state includes tracker state when available."""
        mock_tracker = MagicMock()
        mock_tracker.get_state_summary.return_value = {'consecutive': 2, 'total': 5}

        limiter = IntegratedRateLimiter(tracker=mock_tracker)
        state = limiter.get_state()

        assert 'tracker' in state
        assert state['tracker'] == {'consecutive': 2, 'total': 5}

    def test_set_tracker(self):
        """Test tracker can be set after initialization."""
        limiter = IntegratedRateLimiter()
        assert limiter.tracker is None

        mock_tracker = MagicMock()
        limiter.set_tracker(mock_tracker)
        assert limiter.tracker is mock_tracker


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_rate_limits(self):
        """Test concurrent rate limit recording is thread-safe."""
        import threading

        limiter = UnifiedCaptionRateLimiter()
        threads = []
        num_threads = 10
        records_per_thread = 100

        def record_rate_limits():
            for _ in range(records_per_thread):
                limiter.record_rate_limit()

        for _ in range(num_threads):
            t = threading.Thread(target=record_rate_limits)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        expected_total = num_threads * records_per_thread
        assert limiter.total_rate_limits == expected_total


class TestWaitIfNeeded:
    """Tests for wait_if_needed functionality."""

    def test_no_wait_when_no_rate_limit(self):
        """Test no waiting when no rate limits recorded."""
        limiter = UnifiedCaptionRateLimiter()
        start = time.time()
        waited = limiter.wait_if_needed()
        elapsed = time.time() - start

        assert waited == 0.0
        assert elapsed < 0.1  # Should be nearly instant

    @patch('time.sleep')
    def test_waits_after_rate_limit(self, mock_sleep):
        """Test waits after rate limit recorded."""
        config = RateLimitConfig(jitter_factor=0.0)  # Disable jitter for predictable test
        limiter = UnifiedCaptionRateLimiter(config)

        limiter.record_rate_limit()
        limiter.wait_if_needed()

        # Should have called sleep with base delay (2s)
        mock_sleep.assert_called_once()
        call_arg = mock_sleep.call_args[0][0]
        assert call_arg == 2.0
