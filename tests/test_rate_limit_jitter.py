"""
Tests for jitter in exponential backoff (US-34-003).

Verifies:
- Jitter produces randomized but bounded delays
- Consecutive backoffs with jitter don't exceed max_backoff
- Jitter prevents thundering herd in parallel scenarios
- jitter_factor=0 produces deterministic delays
"""

import pytest
import random
from unittest.mock import patch

from src.caption_timeout_manager import RateLimitState


class TestJitterParameter:
    """Tests for jitter_factor parameter."""

    @pytest.mark.fast
    def test_default_jitter_factor(self):
        """Default jitter_factor should be 0.2 (20%)."""
        state = RateLimitState()
        # Call with default parameter
        with patch.object(random, 'uniform', return_value=0.1) as mock_uniform:
            state.record_rate_limit("video1")
            # Should have been called with (-0.2, 0.2)
            mock_uniform.assert_called_once_with(-0.2, 0.2)

    @pytest.mark.fast
    def test_custom_jitter_factor(self):
        """Custom jitter_factor should be used."""
        state = RateLimitState()
        with patch.object(random, 'uniform', return_value=0.25) as mock_uniform:
            state.record_rate_limit("video1", jitter_factor=0.5)
            # Should have been called with (-0.5, 0.5)
            mock_uniform.assert_called_once_with(-0.5, 0.5)

    @pytest.mark.fast
    def test_zero_jitter_deterministic(self):
        """jitter_factor=0 should produce deterministic base delay."""
        state = RateLimitState()
        state.base_backoff_seconds = 10.0

        # With no jitter, delay should be exactly the base
        delay = state.record_rate_limit("video1", jitter_factor=0.0)
        assert delay == 10.0

    @pytest.mark.fast
    def test_jitter_factor_clamped_negative(self):
        """Negative jitter_factor should be clamped to 0."""
        state = RateLimitState()
        state.base_backoff_seconds = 10.0

        # Negative should behave like 0 (no jitter)
        delay = state.record_rate_limit("video1", jitter_factor=-0.5)
        assert delay == 10.0

    @pytest.mark.fast
    def test_jitter_factor_clamped_above_one(self):
        """jitter_factor > 1.0 should be clamped to 1.0."""
        state = RateLimitState()
        state.base_backoff_seconds = 10.0

        with patch.object(random, 'uniform', return_value=0.5) as mock_uniform:
            state.record_rate_limit("video1", jitter_factor=2.0)
            # Should have been called with (-1.0, 1.0) not (-2.0, 2.0)
            mock_uniform.assert_called_once_with(-1.0, 1.0)


class TestJitterBounds:
    """Tests for jitter delay bounds."""

    @pytest.mark.fast
    def test_jitter_within_bounds_low(self):
        """Delay with negative jitter should be >= base * (1 - jitter_factor)."""
        state = RateLimitState()
        state.base_backoff_seconds = 10.0
        jitter_factor = 0.2

        # Simulate worst case negative jitter
        with patch.object(random, 'uniform', return_value=-jitter_factor):
            delay = state.record_rate_limit("video1", jitter_factor=jitter_factor)
            expected_min = 10.0 * (1 - jitter_factor)  # 8.0
            assert abs(delay - expected_min) < 0.01

    @pytest.mark.fast
    def test_jitter_within_bounds_high(self):
        """Delay with positive jitter should be <= base * (1 + jitter_factor)."""
        state = RateLimitState()
        state.base_backoff_seconds = 10.0
        jitter_factor = 0.2

        # Simulate worst case positive jitter
        with patch.object(random, 'uniform', return_value=jitter_factor):
            delay = state.record_rate_limit("video1", jitter_factor=jitter_factor)
            expected_max = 10.0 * (1 + jitter_factor)  # 12.0
            assert abs(delay - expected_max) < 0.01

    @pytest.mark.fast
    def test_jitter_never_exceeds_max_backoff(self):
        """Delay with jitter should never exceed max_backoff_seconds."""
        state = RateLimitState()
        state.base_backoff_seconds = 100.0
        state.max_backoff_seconds = 300.0

        # After 3 consecutive rate limits: 100 * 2^2 = 400 (exceeds max)
        # Even with positive jitter, should be capped at 300
        state.record_rate_limit("video1")  # 1st
        state.record_rate_limit("video2")  # 2nd

        # 3rd would be 400 base, jitter could push to 480
        with patch.object(random, 'uniform', return_value=0.2):
            delay = state.record_rate_limit("video3", jitter_factor=0.2)
            assert delay <= state.max_backoff_seconds

    @pytest.mark.fast
    def test_consecutive_backoffs_capped(self):
        """All consecutive backoffs with jitter should respect max_backoff."""
        state = RateLimitState()
        state.base_backoff_seconds = 10.0
        state.max_backoff_seconds = 50.0

        delays = []
        for i in range(10):
            delay = state.record_rate_limit(f"video{i}", jitter_factor=0.2)
            delays.append(delay)
            assert delay <= state.max_backoff_seconds, f"Delay {delay} exceeded max on attempt {i+1}"


class TestJitterRandomization:
    """Tests for jitter randomization properties."""

    @pytest.mark.fast
    def test_jitter_produces_varied_delays(self):
        """Multiple calls should produce different delays due to jitter."""
        delays = []

        for _ in range(20):
            state = RateLimitState()
            state.base_backoff_seconds = 10.0
            delay = state.record_rate_limit("video1", jitter_factor=0.2)
            delays.append(delay)

        # With 20 samples and 20% jitter, we should see variety
        unique_delays = len(set(round(d, 4) for d in delays))
        assert unique_delays > 1, "Jitter should produce varied delays"

    @pytest.mark.fast
    def test_jitter_distribution_centered(self):
        """Average delay with jitter should be close to base delay."""
        delays = []

        for _ in range(100):
            state = RateLimitState()
            state.base_backoff_seconds = 10.0
            delay = state.record_rate_limit("video1", jitter_factor=0.2)
            delays.append(delay)

        avg_delay = sum(delays) / len(delays)
        # With uniform distribution, average should be close to base
        assert 9.0 <= avg_delay <= 11.0, f"Average delay {avg_delay} not centered on 10.0"


class TestThunderingHerdPrevention:
    """Tests demonstrating jitter prevents thundering herd."""

    @pytest.mark.fast
    def test_parallel_workers_get_different_delays(self):
        """Simulated parallel workers should get staggered delays."""
        # Simulate 10 workers hitting rate limit at same time
        worker_delays = []

        for i in range(10):
            state = RateLimitState()
            state.base_backoff_seconds = 5.0
            delay = state.record_rate_limit(f"worker{i}", jitter_factor=0.2)
            worker_delays.append(delay)

        # Delays should be different (randomized)
        unique_delays = len(set(round(d, 4) for d in worker_delays))
        assert unique_delays > 1, "Workers should have staggered delays"

        # Check spread - min to max should be at least some range
        delay_range = max(worker_delays) - min(worker_delays)
        assert delay_range > 0.1, "Delays should have meaningful spread"

    @pytest.mark.fast
    def test_jitter_spreads_retry_times(self):
        """Jitter should spread out when workers retry."""
        # Without jitter, all would retry at exactly the same time
        # With jitter, retry times should be staggered

        base_time = 0.0
        retry_times = []

        for i in range(5):
            state = RateLimitState()
            state.base_backoff_seconds = 10.0
            delay = state.record_rate_limit(f"worker{i}", jitter_factor=0.3)
            retry_times.append(base_time + delay)

        # Retry times should vary
        unique_times = len(set(round(t, 4) for t in retry_times))
        assert unique_times > 1, "Retry times should be staggered"


class TestJitterIntegration:
    """Integration tests for jitter with other RateLimitState features."""

    @pytest.mark.fast
    def test_jitter_affects_global_backoff_until(self):
        """global_backoff_until should include jitter."""
        state = RateLimitState()
        state.base_backoff_seconds = 10.0

        # Record with known jitter
        with patch.object(random, 'uniform', return_value=0.1):
            with patch('time.time', return_value=1000.0):
                delay = state.record_rate_limit("video1", jitter_factor=0.2)

        # global_backoff_until should be now + delay with jitter
        # delay = 10.0 * 1.1 = 11.0
        expected_until = 1000.0 + delay
        assert state.global_backoff_until == expected_until

    @pytest.mark.fast
    def test_jitter_in_logging(self, caplog):
        """Log message should include jitter information."""
        import logging
        state = RateLimitState()
        state.base_backoff_seconds = 10.0

        with caplog.at_level(logging.WARNING):
            state.record_rate_limit("test_video", jitter_factor=0.2)

        log_text = caplog.text
        assert "jitter=20%" in log_text
        assert "base_delay=" in log_text
        assert "recommended_delay=" in log_text

    @pytest.mark.fast
    def test_success_resets_before_next_jitter(self):
        """record_success should reset consecutive count for next jitter calculation."""
        state = RateLimitState()
        state.base_backoff_seconds = 5.0

        # Build up consecutive rate limits
        state.record_rate_limit("video1", jitter_factor=0.0)  # 5s
        state.record_rate_limit("video2", jitter_factor=0.0)  # 10s (2^1)
        state.record_rate_limit("video3", jitter_factor=0.0)  # 20s (2^2)

        assert state.consecutive_rate_limits == 3

        # Reset
        state.record_success()
        assert state.consecutive_rate_limits == 0

        # Next rate limit should start fresh
        delay = state.record_rate_limit("video4", jitter_factor=0.0)
        assert delay == 5.0  # Back to base
