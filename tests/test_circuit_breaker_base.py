"""Tests for the abstract circuit breaker base class.

Verifies state transitions: closed -> open (on threshold) -> half-open (on timeout) -> closed (on success).
Also verifies that both download and caption circuit breakers inherit from the base class.
"""

import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from src.common.circuit_breaker_base import CircuitBreakerBase, CircuitBreakerStateBase


# --- Concrete test implementation ---

@dataclass
class StubConfig:
    """Minimal config for testing the base class."""
    enabled: bool = True
    pause_seconds: float = 1.0
    threshold: int = 3


class ConcreteCircuitBreaker(CircuitBreakerBase):
    """Concrete implementation of CircuitBreakerBase for testing."""

    def __init__(self, config: Optional[StubConfig] = None):
        self._config = config or StubConfig()
        self.state = self._create_state()
        self.trip_count = 0  # Track _on_trip calls
        self.failure_hook_count = 0  # Track _on_record_failure calls

    @property
    def config(self):
        return self._config

    def _create_state(self) -> CircuitBreakerStateBase:
        return CircuitBreakerStateBase()

    def _get_failure_threshold(self) -> int:
        return self._config.threshold

    def _get_effective_pause_seconds(self) -> float:
        return self._config.pause_seconds

    def _on_trip(self) -> None:
        self.trip_count += 1

    def _on_record_failure(self) -> None:
        self.failure_hook_count += 1

    def _get_domain_label(self) -> str:
        return "test"


# --- State transition tests ---

@pytest.mark.fast
class TestBaseClassStateTransitions:
    """Verify the core state machine: CLOSED -> OPEN -> HALF-OPEN -> CLOSED."""

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        cb = ConcreteCircuitBreaker()
        assert not cb.is_open
        assert cb.state.consecutive_failures == 0
        assert cb.state.total_trips == 0

    def test_closed_to_open_on_threshold(self):
        """Circuit transitions from CLOSED to OPEN when failure threshold reached."""
        cb = ConcreteCircuitBreaker(StubConfig(threshold=3))

        # First 2 failures don't trip
        assert cb.record_failure() is False
        assert not cb.is_open
        assert cb.record_failure() is False
        assert not cb.is_open

        # Third failure trips the circuit
        assert cb.record_failure() is True
        assert cb.is_open
        assert cb.state.total_trips == 1
        assert cb.state.opened_at is not None

    def test_open_to_half_open_on_timeout(self):
        """Circuit transitions to HALF-OPEN after pause duration elapses."""
        cb = ConcreteCircuitBreaker(StubConfig(threshold=2, pause_seconds=0.5))

        # Trip the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        # Mock time to simulate pause expiry
        opened_time = cb.state.opened_at
        with patch('src.common.circuit_breaker_base.time') as mock_time:
            # Simulate time after pause expired
            mock_time.time.return_value = opened_time + 1.0
            mock_time.sleep = time.sleep  # Don't actually sleep

            result = cb.check_and_wait()

        assert result is True
        assert not cb.is_open  # Now in half-open/closed
        assert cb.state.opened_at is None

    def test_half_open_to_closed_on_success(self):
        """Circuit transitions from HALF-OPEN to CLOSED on success."""
        cb = ConcreteCircuitBreaker(StubConfig(threshold=2, pause_seconds=0.01))

        # Trip, wait, then succeed
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        # Wait for recovery (short pause)
        time.sleep(0.02)
        cb.check_and_wait()

        # Now in half-open: record success to fully close
        cb.record_success()
        assert not cb.is_open
        assert cb.state.consecutive_failures == 0

    def test_half_open_to_open_on_failure(self):
        """Circuit returns to OPEN if failure occurs in half-open state."""
        cb = ConcreteCircuitBreaker(StubConfig(threshold=2, pause_seconds=0.01))

        # Trip and wait for half-open
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        cb.check_and_wait()

        # Failure count still at 2, so next 2 failures trip again (threshold=2)
        # Actually the failure count is preserved but the state is closed.
        # One more failure makes it 3, which >= threshold(2), so it trips again.
        tripped = cb.record_failure()
        assert tripped is True
        assert cb.is_open
        assert cb.state.total_trips == 2


@pytest.mark.fast
class TestBaseClassRecordFailure:
    """Test failure recording and hook invocation."""

    def test_failure_increments_counter(self):
        cb = ConcreteCircuitBreaker(StubConfig(threshold=5))
        cb.record_failure()
        assert cb.state.consecutive_failures == 1
        cb.record_failure()
        assert cb.state.consecutive_failures == 2

    def test_failure_calls_hook(self):
        cb = ConcreteCircuitBreaker(StubConfig(threshold=5))
        cb.record_failure()
        assert cb.failure_hook_count == 1
        cb.record_failure()
        assert cb.failure_hook_count == 2

    def test_trip_calls_hook(self):
        cb = ConcreteCircuitBreaker(StubConfig(threshold=2))
        cb.record_failure()
        assert cb.trip_count == 0
        cb.record_failure()
        assert cb.trip_count == 1

    def test_disabled_does_not_record(self):
        cb = ConcreteCircuitBreaker(StubConfig(enabled=False))
        result = cb.record_failure()
        assert result is False
        assert cb.state.consecutive_failures == 0


@pytest.mark.fast
class TestBaseClassRecordSuccess:
    """Test success recording."""

    def test_success_resets_failures(self):
        cb = ConcreteCircuitBreaker()
        cb.record_failure()
        cb.record_failure()
        assert cb.state.consecutive_failures == 2
        cb.record_success()
        assert cb.state.consecutive_failures == 0

    def test_success_closes_circuit(self):
        cb = ConcreteCircuitBreaker(StubConfig(threshold=2, pause_seconds=60.0))
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        # Directly close via success (simulating half-open probe)
        cb.record_success()
        assert not cb.is_open

    def test_disabled_does_nothing(self):
        cb = ConcreteCircuitBreaker(StubConfig(enabled=False))
        cb.state.consecutive_failures = 5
        cb.record_success()
        assert cb.state.consecutive_failures == 5  # Not reset


@pytest.mark.fast
class TestBaseClassReset:
    """Test manual reset."""

    def test_reset_clears_state(self):
        cb = ConcreteCircuitBreaker(StubConfig(threshold=2))
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        cb.reset()
        assert not cb.is_open
        assert cb.state.consecutive_failures == 0
        assert cb.state.opened_at is None


@pytest.mark.fast
class TestBaseClassCheckAndWait:
    """Test check_and_wait behavior."""

    def test_returns_false_when_disabled(self):
        cb = ConcreteCircuitBreaker(StubConfig(enabled=False))
        assert cb.check_and_wait() is False

    def test_returns_true_when_closed(self):
        cb = ConcreteCircuitBreaker()
        assert cb.check_and_wait() is True

    def test_waits_and_returns_true_when_open(self):
        cb = ConcreteCircuitBreaker(StubConfig(threshold=1, pause_seconds=0.01))
        cb.record_failure()
        assert cb.is_open

        time.sleep(0.02)
        result = cb.check_and_wait()
        assert result is True
        assert not cb.is_open


@pytest.mark.fast
class TestBaseClassStats:
    """Test statistics and serialization."""

    def test_get_stats(self):
        cb = ConcreteCircuitBreaker(StubConfig(threshold=3, pause_seconds=30.0))
        cb.record_failure()
        stats = cb.get_stats()
        assert stats['enabled'] is True
        assert stats['consecutive_failures'] == 1
        assert stats['threshold'] == 3
        assert stats['pause_seconds'] == 30.0

    def test_checkpoint_roundtrip(self):
        cb = ConcreteCircuitBreaker()
        cb.state.total_trips = 5
        cb.state.total_paused_seconds = 120.0
        cb.state.consecutive_failures = 3

        data = cb.to_checkpoint_dict()
        assert data['total_trips'] == 5
        assert data['total_paused_seconds'] == 120.0

        cb2 = ConcreteCircuitBreaker()
        cb2.from_checkpoint_dict(data)
        assert cb2.state.total_trips == 5
        assert cb2.state.total_paused_seconds == 120.0
        # Fresh start: failures and open state not restored
        assert cb2.state.consecutive_failures == 0
        assert not cb2.is_open


@pytest.mark.fast
class TestBaseClassProperties:
    """Test property accessors."""

    def test_is_enabled(self):
        cb = ConcreteCircuitBreaker(StubConfig(enabled=True))
        assert cb.is_enabled is True

        cb2 = ConcreteCircuitBreaker(StubConfig(enabled=False))
        assert cb2.is_enabled is False

    def test_is_open(self):
        cb = ConcreteCircuitBreaker(StubConfig(threshold=1))
        assert cb.is_open is False
        cb.record_failure()
        assert cb.is_open is True


@pytest.mark.fast
class TestInheritanceVerification:
    """Verify that both implementations inherit from CircuitBreakerBase."""

    def test_download_cb_inherits_from_base(self):
        from src.downloader.circuit_breaker import CircuitBreaker
        assert issubclass(CircuitBreaker, CircuitBreakerBase)

    def test_caption_cb_inherits_from_base(self):
        from src.caption.circuit_breaker import CaptionCircuitBreaker
        assert issubclass(CaptionCircuitBreaker, CircuitBreakerBase)

    def test_download_state_inherits_from_base(self):
        from src.downloader.circuit_breaker import CircuitBreakerState
        assert issubclass(CircuitBreakerState, CircuitBreakerStateBase)

    def test_caption_state_inherits_from_base(self):
        from src.caption.circuit_breaker import CaptionCircuitBreakerState
        assert issubclass(CaptionCircuitBreakerState, CircuitBreakerStateBase)

    def test_download_cb_instance_check(self):
        from src.downloader.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        assert isinstance(cb, CircuitBreakerBase)

    def test_caption_cb_instance_check(self):
        from src.caption.circuit_breaker import CaptionCircuitBreaker
        cb = CaptionCircuitBreaker()
        assert isinstance(cb, CircuitBreakerBase)
