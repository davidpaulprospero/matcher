"""Tests for caption circuit breaker (US-33-009).

Tests the CaptionCircuitBreaker implementation which pauses caption fetching
when consecutive failures occur, protecting against rate limiting.
"""

import time
from unittest.mock import patch

import pytest

from src.caption.circuit_breaker import (
    CaptionCircuitBreaker,
    CaptionCircuitBreakerConfig,
    CaptionCircuitBreakerState,
)


class TestCaptionCircuitBreakerConfig:
    """Test CaptionCircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CaptionCircuitBreakerConfig()
        assert config.enabled is True
        assert config.threshold == 10
        assert config.pause_seconds == 120.0
        assert config.max_pause_seconds == 300.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CaptionCircuitBreakerConfig(
            enabled=False,
            threshold=5,
            pause_seconds=60.0,
            max_pause_seconds=180.0,
        )
        assert config.enabled is False
        assert config.threshold == 5
        assert config.pause_seconds == 60.0
        assert config.max_pause_seconds == 180.0


class TestCaptionCircuitBreakerState:
    """Test CaptionCircuitBreakerState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = CaptionCircuitBreakerState()
        assert state.consecutive_failures == 0
        assert state.is_open is False
        assert state.opened_at is None
        assert state.total_trips == 0
        assert state.total_paused_seconds == 0.0


class TestCaptionCircuitBreakerBasic:
    """Test basic CaptionCircuitBreaker functionality."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        breaker = CaptionCircuitBreaker()
        assert breaker.is_enabled is True
        assert breaker.is_open is False
        assert breaker.config.threshold == 10
        assert breaker.config.pause_seconds == 120.0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = CaptionCircuitBreakerConfig(
            enabled=True,
            threshold=5,
            pause_seconds=30.0,
        )
        breaker = CaptionCircuitBreaker(config)
        assert breaker.is_enabled is True
        assert breaker.config.threshold == 5
        assert breaker.config.pause_seconds == 30.0

    def test_disabled_breaker(self):
        """Test that disabled breaker allows all operations."""
        config = CaptionCircuitBreakerConfig(enabled=False)
        breaker = CaptionCircuitBreaker(config)

        # record_failure should return False when disabled
        assert breaker.record_failure() is False

        # Circuit should never open
        for _ in range(20):
            breaker.record_failure()
        assert breaker.is_open is False

    def test_record_success_resets_failures(self):
        """Test that success resets consecutive failure count."""
        breaker = CaptionCircuitBreaker()

        # Record some failures
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state.consecutive_failures == 3

        # Record success
        breaker.record_success()
        assert breaker.state.consecutive_failures == 0

    def test_record_failure_increments_count(self):
        """Test that failures increment the counter."""
        breaker = CaptionCircuitBreaker()

        for i in range(5):
            breaker.record_failure()
            assert breaker.state.consecutive_failures == i + 1


class TestCaptionCircuitBreakerTripping:
    """Test circuit breaker tripping behavior."""

    def test_trips_at_threshold(self):
        """Test that circuit trips when threshold is reached."""
        config = CaptionCircuitBreakerConfig(threshold=5, pause_seconds=10.0)
        breaker = CaptionCircuitBreaker(config)

        # Record failures up to threshold - 1 (should not trip)
        for i in range(4):
            tripped = breaker.record_failure()
            assert tripped is False
            assert breaker.is_open is False

        # The 5th failure should trip the circuit
        tripped = breaker.record_failure()
        assert tripped is True
        assert breaker.is_open is True
        assert breaker.state.total_trips == 1
        assert breaker.state.opened_at is not None

    def test_trip_logs_info(self):
        """Test that tripping logs an INFO message."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=10.0)
        breaker = CaptionCircuitBreaker(config)

        with patch('src.caption.circuit_breaker.logger') as mock_logger:
            breaker.record_failure()
            breaker.record_failure()

            # Verify info log was called with trip message
            info_calls = [call for call in mock_logger.info.call_args_list]
            assert len(info_calls) > 0
            # Check that one of the calls contains "TRIPPED"
            tripped_logs = [c for c in info_calls if 'TRIPPED' in str(c)]
            assert len(tripped_logs) > 0


class TestCaptionCircuitBreakerRecovery:
    """Test circuit breaker recovery behavior."""

    def test_check_and_wait_when_closed(self):
        """Test check_and_wait returns True when circuit is closed."""
        breaker = CaptionCircuitBreaker()
        assert breaker.check_and_wait() is True

    def test_check_and_wait_returns_false_when_disabled(self):
        """Test check_and_wait returns False when disabled."""
        config = CaptionCircuitBreakerConfig(enabled=False)
        breaker = CaptionCircuitBreaker(config)
        assert breaker.check_and_wait() is False

    def test_check_and_wait_after_pause(self):
        """Test that check_and_wait recovers after pause duration."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.1)
        breaker = CaptionCircuitBreaker(config)

        # Trip the circuit
        breaker.record_failure()
        assert breaker.is_open is True

        # Wait for pause to elapse
        time.sleep(0.15)

        # check_and_wait should close the circuit
        result = breaker.check_and_wait()
        assert result is True
        assert breaker.is_open is False

    def test_get_remaining_pause_time(self):
        """Test remaining pause time calculation."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=1.0)
        breaker = CaptionCircuitBreaker(config)

        # When not tripped, remaining is 0
        assert breaker.get_remaining_pause_time() == 0.0

        # Trip the circuit
        breaker.record_failure()

        # Should have close to 1.0 seconds remaining
        remaining = breaker.get_remaining_pause_time()
        assert 0.9 < remaining <= 1.0

        # Wait a bit
        time.sleep(0.2)
        remaining = breaker.get_remaining_pause_time()
        assert 0.7 < remaining < 0.9

    def test_manual_reset(self):
        """Test manual reset clears state."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=60.0)
        breaker = CaptionCircuitBreaker(config)

        # Record failures and trip
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is True
        assert breaker.state.consecutive_failures == 2

        # Reset
        breaker.reset()
        assert breaker.is_open is False
        assert breaker.state.consecutive_failures == 0
        assert breaker.state.opened_at is None


class TestCaptionCircuitBreakerPauseCapping:
    """Test pause duration capping."""

    def test_pause_capped_at_max(self):
        """Test that pause is capped at max_pause_seconds."""
        config = CaptionCircuitBreakerConfig(
            threshold=1,
            pause_seconds=500.0,  # Higher than max
            max_pause_seconds=100.0,
        )
        breaker = CaptionCircuitBreaker(config)

        effective_pause = breaker._get_effective_pause_seconds()
        assert effective_pause == 100.0

    def test_pause_not_capped_when_under_max(self):
        """Test that pause is not capped when under max."""
        config = CaptionCircuitBreakerConfig(
            threshold=1,
            pause_seconds=50.0,
            max_pause_seconds=100.0,
        )
        breaker = CaptionCircuitBreaker(config)

        effective_pause = breaker._get_effective_pause_seconds()
        assert effective_pause == 50.0


class TestCaptionCircuitBreakerStats:
    """Test circuit breaker statistics."""

    def test_get_stats(self):
        """Test get_stats returns correct values."""
        config = CaptionCircuitBreakerConfig(threshold=3, pause_seconds=30.0)
        breaker = CaptionCircuitBreaker(config)

        # Initial stats
        stats = breaker.get_stats()
        assert stats['enabled'] is True
        assert stats['is_open'] is False
        assert stats['consecutive_failures'] == 0
        assert stats['total_trips'] == 0
        assert stats['total_paused_seconds'] == 0.0
        assert stats['threshold'] == 3
        assert stats['pause_seconds'] == 30.0

        # After failures
        breaker.record_failure()
        breaker.record_failure()
        stats = breaker.get_stats()
        assert stats['consecutive_failures'] == 2
        assert stats['is_open'] is False

        # After trip
        breaker.record_failure()
        stats = breaker.get_stats()
        assert stats['consecutive_failures'] == 3
        assert stats['is_open'] is True
        assert stats['total_trips'] == 1


class TestCaptionCircuitBreakerCheckpoint:
    """Test checkpoint serialization/deserialization."""

    def test_to_checkpoint_dict(self):
        """Test serialization to checkpoint dict."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=10.0)
        breaker = CaptionCircuitBreaker(config)

        # Record some activity
        breaker.record_failure()
        breaker.record_failure()  # Trips
        breaker.state.total_paused_seconds = 15.5

        data = breaker.to_checkpoint_dict()
        assert data['consecutive_failures'] == 2
        assert data['total_trips'] == 1
        assert data['total_paused_seconds'] == 15.5
        # is_open should NOT be in checkpoint (transient state)
        assert 'is_open' not in data

    def test_from_checkpoint_dict(self):
        """Test restoring from checkpoint dict."""
        breaker = CaptionCircuitBreaker()

        # Restore from checkpoint
        data = {
            'consecutive_failures': 5,  # Should NOT be restored (start fresh)
            'total_trips': 3,
            'total_paused_seconds': 45.0,
        }
        breaker.from_checkpoint_dict(data)

        # Cumulative stats are restored
        assert breaker.state.total_trips == 3
        assert breaker.state.total_paused_seconds == 45.0

        # Transient state starts fresh
        assert breaker.state.consecutive_failures == 0
        assert breaker.state.is_open is False
        assert breaker.state.opened_at is None

    def test_from_empty_checkpoint(self):
        """Test restoring from empty checkpoint dict."""
        breaker = CaptionCircuitBreaker()
        breaker.record_failure()

        # Empty dict should not change state
        breaker.from_checkpoint_dict({})
        assert breaker.state.consecutive_failures == 1

        # None should also not crash
        breaker.from_checkpoint_dict(None)
        assert breaker.state.consecutive_failures == 1


class TestCaptionCircuitBreakerThreadSafety:
    """Test thread safety of circuit breaker (basic tests)."""

    def test_concurrent_failure_recording(self):
        """Test that concurrent failures are tracked correctly."""
        import threading

        config = CaptionCircuitBreakerConfig(threshold=100, pause_seconds=1.0)
        breaker = CaptionCircuitBreaker(config)

        def record_failures():
            for _ in range(10):
                breaker.record_failure()

        threads = [threading.Thread(target=record_failures) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded 100 failures total
        assert breaker.state.consecutive_failures == 100

    def test_concurrent_success_failure_mix(self):
        """Test mixed success/failure operations."""
        import threading

        config = CaptionCircuitBreakerConfig(threshold=1000, pause_seconds=1.0)
        breaker = CaptionCircuitBreaker(config)

        def alternate_operations():
            for i in range(50):
                if i % 2 == 0:
                    breaker.record_failure()
                else:
                    breaker.record_success()

        threads = [threading.Thread(target=alternate_operations) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # State should be consistent (not tripped since threshold is high)
        assert breaker.state.consecutive_failures >= 0
        assert breaker.is_open is False


class TestCaptionCircuitBreakerIntegration:
    """Integration tests for circuit breaker in caption fetching context."""

    def test_multiple_trip_cycles(self):
        """Test multiple trip and recovery cycles."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        for cycle in range(3):
            # Trip the circuit
            breaker.record_failure()
            breaker.record_failure()
            assert breaker.is_open is True
            assert breaker.state.total_trips == cycle + 1

            # Wait for recovery
            time.sleep(0.06)
            breaker.check_and_wait()
            assert breaker.is_open is False

            # Success resets failures
            breaker.record_success()
            assert breaker.state.consecutive_failures == 0

    def test_success_after_trip_resets(self):
        """Test that success after trip fully resets state."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Trip
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is True

        # Wait and recover
        time.sleep(0.06)
        breaker.check_and_wait()

        # Success after recovery
        breaker.record_success()
        assert breaker.state.consecutive_failures == 0
        assert breaker.is_open is False

        # Should need full threshold to trip again
        breaker.record_failure()
        assert breaker.is_open is False
        breaker.record_failure()
        assert breaker.is_open is True


# ============================================================================
# US-34-008: Circuit Breaker State Transition Tests
# ============================================================================


class TestStateTransitionClosedToOpen:
    """Test CLOSED -> OPEN transition after threshold failures (US-34-008)."""

    def test_closed_to_open_at_exact_threshold(self):
        """Test transition happens exactly at threshold."""
        config = CaptionCircuitBreakerConfig(threshold=3, pause_seconds=0.1)
        breaker = CaptionCircuitBreaker(config)

        # Verify starts CLOSED
        assert breaker.is_open is False
        assert breaker.state.consecutive_failures == 0

        # Failures 1 and 2 should remain CLOSED
        breaker.record_failure()
        assert breaker.is_open is False
        breaker.record_failure()
        assert breaker.is_open is False

        # Failure 3 (threshold) should transition to OPEN
        tripped = breaker.record_failure()
        assert tripped is True
        assert breaker.is_open is True
        assert breaker.state.total_trips == 1
        assert breaker.state.opened_at is not None

    def test_closed_to_open_sets_opened_at_timestamp(self):
        """Test that opened_at is set correctly on transition."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.1)
        breaker = CaptionCircuitBreaker(config)

        before = time.time()
        breaker.record_failure()
        after = time.time()

        assert breaker.state.opened_at is not None
        assert before <= breaker.state.opened_at <= after

    def test_closed_to_open_increments_total_trips(self):
        """Test that total_trips increments on each CLOSED -> OPEN transition."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        for expected_trips in range(1, 4):
            # Trip the circuit
            breaker.record_failure()
            assert breaker.state.total_trips == expected_trips

            # Recover
            time.sleep(0.06)
            breaker.check_and_wait()
            breaker.record_success()

    def test_success_resets_progress_toward_open(self):
        """Test that success resets failure count, preventing CLOSED -> OPEN."""
        config = CaptionCircuitBreakerConfig(threshold=3, pause_seconds=0.1)
        breaker = CaptionCircuitBreaker(config)

        # Build up failures
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state.consecutive_failures == 2

        # Success resets progress
        breaker.record_success()
        assert breaker.state.consecutive_failures == 0
        assert breaker.is_open is False

        # Need 3 more failures now
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is False  # Still CLOSED


class TestStateTransitionOpenToHalfOpen:
    """Test OPEN -> HALF_OPEN transition after pause_seconds elapsed (US-34-008)."""

    def test_open_to_half_open_after_pause_elapsed(self):
        """Test transition to HALF_OPEN after pause_seconds."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Trip to OPEN
        breaker.record_failure()
        assert breaker.is_open is True
        original_failures = breaker.state.consecutive_failures

        # Wait for pause to elapse
        time.sleep(0.06)

        # check_and_wait transitions to HALF_OPEN (is_open=False)
        result = breaker.check_and_wait()
        assert result is True
        assert breaker.is_open is False
        # Consecutive failures preserved in HALF_OPEN state
        assert breaker.state.consecutive_failures == original_failures

    def test_open_does_not_transition_before_pause(self):
        """Test that OPEN state persists before pause_seconds elapsed."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=1.0)
        breaker = CaptionCircuitBreaker(config)

        # Trip to OPEN
        breaker.record_failure()
        assert breaker.is_open is True

        # Check remaining time immediately
        remaining = breaker.get_remaining_pause_time()
        assert remaining > 0.9  # Should be close to 1.0s

        # is_open should still be True (not yet transitioned)
        assert breaker.is_open is True

    def test_half_open_clears_opened_at(self):
        """Test that opened_at is cleared on OPEN -> HALF_OPEN transition."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Trip to OPEN
        breaker.record_failure()
        assert breaker.state.opened_at is not None

        # Wait and transition to HALF_OPEN
        time.sleep(0.06)
        breaker.check_and_wait()

        assert breaker.state.opened_at is None

    def test_open_to_half_open_accumulates_paused_seconds(self):
        """Test that total_paused_seconds is updated on transition."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.1)
        breaker = CaptionCircuitBreaker(config)

        initial_paused = breaker.state.total_paused_seconds

        # Trip to OPEN
        breaker.record_failure()

        # Wait partial time then check (will wait remaining)
        time.sleep(0.05)
        breaker.check_and_wait()

        # Should have accumulated approximately 0.05s of pause
        assert breaker.state.total_paused_seconds > initial_paused


class TestStateTransitionHalfOpenToClosed:
    """Test HALF_OPEN -> CLOSED on success (US-34-008)."""

    def test_half_open_to_closed_on_success(self):
        """Test that success in HALF_OPEN state transitions to CLOSED."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Trip to OPEN
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is True

        # Wait and transition to HALF_OPEN
        time.sleep(0.06)
        breaker.check_and_wait()
        assert breaker.is_open is False
        # Still in HALF_OPEN - failures preserved
        assert breaker.state.consecutive_failures == 2

        # Success transitions to CLOSED
        breaker.record_success()
        assert breaker.is_open is False
        assert breaker.state.consecutive_failures == 0  # CLOSED state

    def test_half_open_to_closed_resets_failure_counter(self):
        """Test that failure counter is reset on HALF_OPEN -> CLOSED."""
        config = CaptionCircuitBreakerConfig(threshold=5, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Build up failures and trip
        for _ in range(5):
            breaker.record_failure()
        assert breaker.state.consecutive_failures == 5

        # Wait and transition to HALF_OPEN
        time.sleep(0.06)
        breaker.check_and_wait()

        # Success resets counter (CLOSED state)
        breaker.record_success()
        assert breaker.state.consecutive_failures == 0

    def test_closed_state_requires_full_threshold_again(self):
        """Test that after HALF_OPEN -> CLOSED, full threshold is required."""
        config = CaptionCircuitBreakerConfig(threshold=3, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Trip to OPEN
        for _ in range(3):
            breaker.record_failure()

        # Recover to HALF_OPEN then CLOSED
        time.sleep(0.06)
        breaker.check_and_wait()
        breaker.record_success()

        # Should need 3 failures again
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is False  # Not yet at threshold

        breaker.record_failure()
        assert breaker.is_open is True  # Now tripped again


class TestStateTransitionHalfOpenToOpen:
    """Test HALF_OPEN -> OPEN on failure with counter reset (US-34-008)."""

    def test_half_open_to_open_on_failure(self):
        """Test that failure in HALF_OPEN state transitions back to OPEN."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Trip to OPEN
        breaker.record_failure()
        breaker.record_failure()
        original_trips = breaker.state.total_trips

        # Wait and transition to HALF_OPEN
        time.sleep(0.06)
        breaker.check_and_wait()
        assert breaker.is_open is False  # HALF_OPEN

        # Failure in HALF_OPEN - should increment failure count
        # and if at threshold again, trip immediately
        breaker.record_failure()
        # Now at 3 consecutive failures, threshold is 2 -> trips
        assert breaker.is_open is True
        assert breaker.state.total_trips == original_trips + 1

    def test_half_open_to_open_increments_total_trips(self):
        """Test that total_trips increments on HALF_OPEN -> OPEN."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # First trip
        breaker.record_failure()
        assert breaker.state.total_trips == 1

        # Recover to HALF_OPEN
        time.sleep(0.06)
        breaker.check_and_wait()

        # Fail again in HALF_OPEN
        breaker.record_failure()
        assert breaker.state.total_trips == 2

    def test_half_open_preserves_failure_count_for_quick_retrip(self):
        """Test that HALF_OPEN preserves failures allowing quick re-trip."""
        config = CaptionCircuitBreakerConfig(threshold=3, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Trip with 3 failures
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state.consecutive_failures == 3

        # Recover to HALF_OPEN
        time.sleep(0.06)
        breaker.check_and_wait()

        # Failure count preserved - one more failure trips again
        assert breaker.state.consecutive_failures == 3
        breaker.record_failure()  # Now 4
        assert breaker.is_open is True  # Re-tripped

    def test_half_open_to_open_sets_new_opened_at(self):
        """Test that new opened_at timestamp is set on HALF_OPEN -> OPEN."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # First trip
        breaker.record_failure()
        first_opened_at = breaker.state.opened_at

        # Recover to HALF_OPEN
        time.sleep(0.06)
        breaker.check_and_wait()
        assert breaker.state.opened_at is None

        # Fail again in HALF_OPEN
        time.sleep(0.01)  # Small gap
        breaker.record_failure()

        assert breaker.state.opened_at is not None
        assert breaker.state.opened_at > first_opened_at


class TestStateTransitionSerialization:
    """Test state persistence through serialization/deserialization (US-34-008)."""

    def test_serialization_captures_cumulative_state(self):
        """Test that to_checkpoint_dict captures cumulative state."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=0.1)
        breaker = CaptionCircuitBreaker(config)

        # Build up state through multiple cycles
        for _ in range(3):
            breaker.record_failure()
            breaker.record_failure()  # Trip
            # Call check_and_wait immediately - it will wait 0.1s and accumulate pause
            breaker.check_and_wait()
            breaker.record_success()

        data = breaker.to_checkpoint_dict()
        assert data['total_trips'] == 3
        assert data['total_paused_seconds'] > 0  # Should have ~0.3s accumulated
        assert 'consecutive_failures' in data

    def test_deserialization_restores_cumulative_state(self):
        """Test that from_checkpoint_dict restores cumulative state."""
        breaker = CaptionCircuitBreaker()

        data = {
            'total_trips': 5,
            'total_paused_seconds': 123.4,
            'consecutive_failures': 7,  # Should be ignored
        }
        breaker.from_checkpoint_dict(data)

        assert breaker.state.total_trips == 5
        assert breaker.state.total_paused_seconds == 123.4
        # Transient state starts fresh
        assert breaker.state.consecutive_failures == 0
        assert breaker.state.is_open is False

    def test_deserialization_starts_in_closed_state(self):
        """Test that deserialization always starts in CLOSED state."""
        breaker = CaptionCircuitBreaker()

        # Even if is_open were somehow in the data, we start CLOSED
        data = {
            'total_trips': 10,
            'total_paused_seconds': 500.0,
            'is_open': True,  # Should be ignored
            'opened_at': time.time(),  # Should be ignored
        }
        breaker.from_checkpoint_dict(data)

        assert breaker.is_open is False
        assert breaker.state.opened_at is None
        assert breaker.state.consecutive_failures == 0

    def test_state_roundtrip_through_serialization(self):
        """Test that state survives serialization roundtrip."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=0.05)
        breaker1 = CaptionCircuitBreaker(config)

        # Build up state
        breaker1.record_failure()
        breaker1.record_failure()  # Trip
        time.sleep(0.06)
        breaker1.check_and_wait()
        breaker1.record_success()

        # Serialize
        data = breaker1.to_checkpoint_dict()

        # Deserialize to new breaker
        breaker2 = CaptionCircuitBreaker(config)
        breaker2.from_checkpoint_dict(data)

        assert breaker2.state.total_trips == breaker1.state.total_trips
        # total_paused_seconds may differ slightly due to timing

    def test_serialization_does_not_include_transient_state(self):
        """Test that is_open and opened_at are not serialized."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=60.0)
        breaker = CaptionCircuitBreaker(config)

        # Trip the breaker (long pause so it stays open)
        breaker.record_failure()
        assert breaker.is_open is True
        assert breaker.state.opened_at is not None

        data = breaker.to_checkpoint_dict()

        assert 'is_open' not in data
        assert 'opened_at' not in data

    def test_deserialization_with_none_or_empty(self):
        """Test that deserialization handles None/empty gracefully."""
        breaker = CaptionCircuitBreaker()
        breaker.record_failure()

        # None should not change state
        breaker.from_checkpoint_dict(None)
        assert breaker.state.consecutive_failures == 1

        # Empty dict should not change state
        breaker.from_checkpoint_dict({})
        assert breaker.state.consecutive_failures == 1


# ============================================================================
# US-62-006: Circuit Breaker Integration with Retry Budget
# ============================================================================


class TestCaptionCircuitBreakerRetryBudgetIntegration:
    """Test circuit breaker integration with CaptionRetryBudget (US-62-006)."""

    def test_retry_budget_circuit_breaker_reference_observable(self):
        """Test that circuit breaker can be referenced from retry budget for observability."""
        from src.caption.retry_budget import CaptionRetryBudget

        config = CaptionCircuitBreakerConfig(threshold=5, pause_seconds=10.0)
        breaker = CaptionCircuitBreaker(config)
        budget = CaptionRetryBudget()

        # Link circuit breaker to retry budget (US-40-011 / US-62-006)
        budget.circuit_breaker = breaker

        # Verify reference is set
        assert budget.circuit_breaker is breaker
        assert budget.circuit_breaker.config.threshold == 5

    def test_retry_budget_summary_includes_circuit_breaker_state(self):
        """Test that retry budget summary includes circuit breaker state when linked."""
        from src.caption.retry_budget import CaptionRetryBudget

        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=10.0)
        breaker = CaptionCircuitBreaker(config)
        budget = CaptionRetryBudget()
        budget.circuit_breaker = breaker

        # Initially closed
        summary = budget.get_summary()
        assert summary['circuit_breaker_state'] == 'closed'

        # Trip the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is True

        # Summary should now show open
        summary = budget.get_summary()
        assert summary['circuit_breaker_state'] == 'open'

    def test_retry_budget_formatted_summary_includes_circuit_breaker(self):
        """Test that formatted summary includes circuit breaker state."""
        from src.caption.retry_budget import CaptionRetryBudget

        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=10.0)
        breaker = CaptionCircuitBreaker(config)
        budget = CaptionRetryBudget()
        budget.circuit_breaker = breaker

        formatted = budget.get_formatted_summary()
        assert 'circuit_breaker=closed' in formatted

        # Trip the breaker
        breaker.record_failure()
        breaker.record_failure()

        formatted = budget.get_formatted_summary()
        assert 'circuit_breaker=open' in formatted

    def test_retry_budget_without_circuit_breaker_returns_none(self):
        """Test that retry budget without circuit breaker returns None for state."""
        from src.caption.retry_budget import CaptionRetryBudget

        budget = CaptionRetryBudget()
        # No circuit breaker linked

        summary = budget.get_summary()
        assert summary['circuit_breaker_state'] is None

        # Formatted summary should not include circuit_breaker suffix
        formatted = budget.get_formatted_summary()
        assert 'circuit_breaker=' not in formatted


class TestCaptionCircuitBreakerTripsAtConfiguredThreshold:
    """Test that circuit trips exactly at configured threshold (US-62-006 AC)."""

    def test_circuit_trips_at_5_failures_with_threshold_5(self):
        """Verify circuit trips after exactly 5 consecutive failures when threshold=5."""
        config = CaptionCircuitBreakerConfig(threshold=5, pause_seconds=60.0)
        breaker = CaptionCircuitBreaker(config)

        # 4 failures - should NOT trip
        for i in range(4):
            tripped = breaker.record_failure()
            assert tripped is False, f"Should not trip after {i+1} failures"
            assert breaker.is_open is False

        # 5th failure - SHOULD trip
        tripped = breaker.record_failure()
        assert tripped is True, "Should trip after 5 failures"
        assert breaker.is_open is True
        assert breaker.state.total_trips == 1

    def test_circuit_blocks_fetches_during_cooldown(self):
        """Verify circuit breaker blocks fetches during cooldown period."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.1)
        breaker = CaptionCircuitBreaker(config)

        # Trip the circuit
        breaker.record_failure()
        assert breaker.is_open is True

        # Remaining pause time should be > 0
        remaining = breaker.get_remaining_pause_time()
        assert remaining > 0

    def test_successful_probe_closes_circuit(self):
        """Verify successful probe after half-open closes circuit."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Trip the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is True

        # Wait for cooldown and transition to half-open
        time.sleep(0.06)
        breaker.check_and_wait()
        assert breaker.is_open is False  # Half-open state

        # Success should fully close
        breaker.record_success()
        assert breaker.state.consecutive_failures == 0
        assert breaker.is_open is False

    def test_failed_probe_reopens_circuit(self):
        """Verify failed probe in half-open state re-opens circuit."""
        config = CaptionCircuitBreakerConfig(threshold=2, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Trip the circuit
        breaker.record_failure()
        breaker.record_failure()
        initial_trips = breaker.state.total_trips

        # Wait for cooldown and transition to half-open
        time.sleep(0.06)
        breaker.check_and_wait()
        assert breaker.is_open is False  # Half-open

        # Failure in half-open should re-trip
        breaker.record_failure()
        # Now at 3 consecutive failures (2 from before + 1 new) > threshold of 2
        assert breaker.is_open is True
        assert breaker.state.total_trips == initial_trips + 1
