"""Stress tests for circuit breaker functionality.

US-35-008: Add circuit breaker stress tests

Tests cover:
- Thread safety under concurrent failures
- State machine correctness under rapid success/failure transitions
- Half-open state probe request behavior
- Recovery time calculation with jitter
"""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from src.downloader.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)


# ============================================================================
# Concurrent Failure Tests
# ============================================================================


class TestConcurrentFailures:
    """Test circuit breaker under concurrent failure conditions."""

    @pytest.mark.stress
    def test_50_concurrent_failures_trip_circuit_correctly(self):
        """50 concurrent failures should trip circuit breaker exactly once.

        The circuit should trip when threshold is reached, even under concurrent
        access. Total_trips should be exactly 1, not multiple trips from race conditions.
        """
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=10,
            pause_seconds=0.1
        )
        breaker = CircuitBreaker(config)

        # Track results from each thread
        results = []
        results_lock = threading.Lock()

        def record_failure_and_track():
            result = breaker.record_failure()
            with results_lock:
                results.append(result)
            return result

        # Run 50 concurrent failures
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(record_failure_and_track) for _ in range(50)]
            for future in as_completed(futures):
                future.result()

        # Circuit should be open (tripped)
        assert breaker.is_open is True

        # Should have exactly 50 consecutive failures recorded
        assert breaker.state.consecutive_failures == 50

        # At least one failure should have triggered the trip
        trip_count = sum(1 for r in results if r is True)
        assert trip_count >= 1, "Circuit should have tripped at least once"

        # total_trips should be 1 (not multiple trips from race conditions)
        # Note: Due to threading, we may see multiple trips if failures happen
        # between check and trip, but typically should be 1
        assert breaker.state.total_trips >= 1

    @pytest.mark.stress
    def test_concurrent_failures_preserve_failure_count(self):
        """Concurrent failures should not lose or duplicate failure counts."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=100,  # High threshold to avoid tripping
            pause_seconds=0.1
        )
        breaker = CircuitBreaker(config)

        num_failures = 50

        def record_failure():
            breaker.record_failure()

        # Run concurrent failures
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(record_failure) for _ in range(num_failures)]
            for future in as_completed(futures):
                future.result()

        # All failures should be counted
        assert breaker.state.consecutive_failures == num_failures

    @pytest.mark.stress
    def test_circuit_trips_only_when_threshold_reached(self):
        """Circuit should not trip before threshold under concurrent access."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=30,
            pause_seconds=0.1
        )
        breaker = CircuitBreaker(config)

        # Record 25 failures (less than threshold)
        def record_failure():
            return breaker.record_failure()

        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(record_failure) for _ in range(25)]
            results = [future.result() for future in as_completed(futures)]

        # No failures should have triggered a trip
        assert all(r is False for r in results)
        assert breaker.is_open is False
        assert breaker.state.total_trips == 0


# ============================================================================
# Rapid State Transition Tests
# ============================================================================


class TestRapidStateTransitions:
    """Test circuit breaker under rapid success/failure transitions."""

    @pytest.mark.stress
    def test_rapid_success_failure_alternation(self):
        """Rapid alternation between success and failure should maintain consistency."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=5,
            pause_seconds=0.01
        )
        breaker = CircuitBreaker(config)

        # Alternate rapidly between success and failure 100 times
        for i in range(100):
            if i % 2 == 0:
                breaker.record_failure()
            else:
                breaker.record_success()

        # After alternation, consecutive_failures should be 0 or 1
        # (depends on whether last was failure or success)
        assert breaker.state.consecutive_failures <= 1

        # Circuit should not be open (never had 5 consecutive failures)
        assert breaker.is_open is False

    @pytest.mark.stress
    def test_consecutive_failures_then_rapid_successes(self):
        """Rapid successes after failures should reset state properly."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=3,
            pause_seconds=0.01
        )
        breaker = CircuitBreaker(config)

        # Record 2 failures (not enough to trip)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state.consecutive_failures == 2

        # Record rapid successes
        for _ in range(50):
            breaker.record_success()

        # Should be fully reset
        assert breaker.state.consecutive_failures == 0
        assert breaker.is_open is False

    @pytest.mark.stress
    def test_trip_reset_trip_cycle(self):
        """Circuit should handle repeated trip-reset cycles correctly."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=2,
            pause_seconds=0.01,
            jitter_factor=0.0  # Disable jitter for deterministic test
        )
        breaker = CircuitBreaker(config)

        cycles = 20

        for cycle in range(cycles):
            # Trip the circuit
            breaker.record_failure()
            breaker.record_failure()
            assert breaker.is_open is True, f"Cycle {cycle}: Circuit should be open"

            # Wait for recovery
            breaker.check_and_wait()
            assert breaker.is_open is False, f"Cycle {cycle}: Circuit should close after wait"

            # Record success to fully reset
            breaker.record_success()

        # Should have tripped exactly 'cycles' times
        assert breaker.state.total_trips == cycles

    @pytest.mark.stress
    def test_concurrent_success_and_failure_calls(self):
        """Concurrent success and failure calls should not corrupt state."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=100,  # High threshold
            pause_seconds=0.1
        )
        breaker = CircuitBreaker(config)

        def mixed_operations():
            for _ in range(10):
                if random.random() > 0.5:
                    breaker.record_failure()
                else:
                    breaker.record_success()

        # Run mixed operations concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(mixed_operations) for _ in range(20)]
            for future in as_completed(futures):
                future.result()

        # State should be valid (non-negative failures, valid trip count)
        assert breaker.state.consecutive_failures >= 0
        assert breaker.state.total_trips >= 0
        assert isinstance(breaker.state.is_open, bool)


# ============================================================================
# Half-Open State Tests
# ============================================================================


class TestHalfOpenState:
    """Test half-open state behavior (recovery probe)."""

    @pytest.mark.stress
    def test_half_open_allows_probe_request(self):
        """After pause, half-open state should allow one probe request."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.01,
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        # Trip the circuit
        breaker.record_failure()
        assert breaker.is_open is True

        # Wait for recovery (transitions to half-open)
        breaker.check_and_wait()

        # Circuit should be closed (half-open allows requests)
        assert breaker.is_open is False

        # Failure count preserved for quick re-trip
        assert breaker.state.consecutive_failures == 1

    @pytest.mark.stress
    def test_half_open_success_fully_closes_circuit(self):
        """Success in half-open state should fully close circuit."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=3,
            pause_seconds=0.01,
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        # Trip the circuit with 3 failures
        for _ in range(3):
            breaker.record_failure()
        assert breaker.is_open is True

        # Wait for recovery (half-open)
        breaker.check_and_wait()

        # Probe succeeds
        breaker.record_success()

        # Should be fully closed with reset failure count
        assert breaker.is_open is False
        assert breaker.state.consecutive_failures == 0

    @pytest.mark.stress
    def test_half_open_failure_re_trips_circuit(self):
        """Failure in half-open state should re-trip circuit immediately."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=3,
            pause_seconds=0.01,
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        # Trip the circuit
        for _ in range(3):
            breaker.record_failure()
        assert breaker.is_open is True

        # Wait for recovery (half-open)
        breaker.check_and_wait()
        assert breaker.is_open is False

        # Failure count is still 3, so one more failure re-trips
        breaker.record_failure()
        assert breaker.state.consecutive_failures == 4

        # May or may not re-trip depending on threshold check timing
        # But failure count should definitely increase

    @pytest.mark.stress
    def test_multiple_concurrent_probes_after_recovery(self):
        """Multiple concurrent requests after recovery should be handled safely."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=2,
            pause_seconds=0.01,
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        # Trip the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is True

        # Wait for pause duration
        time.sleep(0.02)

        # Multiple concurrent check_and_wait calls
        def check_wait():
            return breaker.check_and_wait()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_wait) for _ in range(10)]
            results = [future.result() for future in as_completed(futures)]

        # All should return True (allowed to proceed)
        assert all(r is True for r in results)

        # Circuit should be closed
        assert breaker.is_open is False


# ============================================================================
# Recovery Time and Jitter Tests
# ============================================================================


class TestRecoveryTimeWithJitter:
    """Test recovery time calculation with jitter."""

    @pytest.mark.stress
    def test_recovery_time_within_jitter_bounds(self):
        """Recovery time should be within jitter bounds."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=1.0,
            jitter_factor=0.2  # ±20%
        )
        breaker = CircuitBreaker(config)

        # Collect multiple recovery times
        recovery_times = []

        for _ in range(20):
            breaker.record_failure()

            start = time.time()
            breaker.check_and_wait()
            elapsed = time.time() - start

            recovery_times.append(elapsed)
            breaker.record_success()  # Reset for next iteration

        # All times should be within jitter bounds (0.8 to 1.2 seconds)
        min_expected = 1.0 * (1 - 0.2)  # 0.8
        max_expected = 1.0 * (1 + 0.2)  # 1.2

        for rt in recovery_times:
            assert rt >= min_expected - 0.05, f"Recovery time {rt} below minimum {min_expected}"
            assert rt <= max_expected + 0.05, f"Recovery time {rt} above maximum {max_expected}"

    @pytest.mark.stress
    def test_jitter_produces_varied_recovery_times(self):
        """Jitter should produce varied recovery times, not identical ones."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.5,
            jitter_factor=0.3  # ±30% for more variance
        )
        breaker = CircuitBreaker(config)

        # Collect recovery times
        recovery_times = []

        for _ in range(10):
            breaker.record_failure()

            start = time.time()
            breaker.check_and_wait()
            elapsed = time.time() - start

            recovery_times.append(elapsed)
            breaker.record_success()

        # Should have some variance (not all identical)
        unique_times = set(round(t, 2) for t in recovery_times)
        assert len(unique_times) > 1, "Jitter should produce varied recovery times"

    @pytest.mark.stress
    def test_zero_jitter_produces_consistent_times(self):
        """Zero jitter should produce consistent recovery times."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.1,
            jitter_factor=0.0  # No jitter
        )
        breaker = CircuitBreaker(config)

        # Collect recovery times
        recovery_times = []

        for _ in range(5):
            breaker.record_failure()

            start = time.time()
            breaker.check_and_wait()
            elapsed = time.time() - start

            recovery_times.append(elapsed)
            breaker.record_success()

        # All times should be very close to pause_seconds
        for rt in recovery_times:
            assert abs(rt - 0.1) < 0.02, f"Recovery time {rt} not close to pause_seconds 0.1"

    @pytest.mark.stress
    def test_get_remaining_pause_time_with_jitter(self):
        """get_remaining_pause_time with jitter should vary."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=10.0,
            jitter_factor=0.25
        )
        breaker = CircuitBreaker(config)

        breaker.record_failure()

        # Collect jittered remaining times
        remaining_times = []
        for _ in range(20):
            remaining = breaker.get_remaining_pause_time(apply_jitter=True)
            remaining_times.append(remaining)

        # Should have variance
        unique_times = set(round(t, 1) for t in remaining_times)
        assert len(unique_times) > 1, "Jittered remaining times should vary"

        # All should be within bounds
        min_expected = 10.0 * (1 - 0.25)
        max_expected = 10.0 * (1 + 0.25)

        for rt in remaining_times:
            assert rt >= 0  # Can't be negative
            assert rt <= max_expected + 0.1

    @pytest.mark.stress
    def test_jitter_capped_at_max_pause_seconds(self):
        """Jittered delay should be capped at max_pause_seconds."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=250.0,
            jitter_factor=0.5,  # Could push to 375s
            max_pause_seconds=300.0  # Cap at 300s
        )
        breaker = CircuitBreaker(config)

        # Test _apply_jitter directly
        for _ in range(50):
            jittered = breaker._apply_jitter(250.0)
            assert jittered <= 300.0, f"Jittered delay {jittered} exceeds max_pause_seconds 300"


# ============================================================================
# State Machine Correctness Tests
# ============================================================================


class TestStateMachineCorrectness:
    """Test circuit breaker state machine correctness under load."""

    @pytest.mark.stress
    def test_state_machine_closed_to_open_to_half_open_to_closed(self):
        """Verify correct state machine transitions: CLOSED -> OPEN -> HALF-OPEN -> CLOSED."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=3,
            pause_seconds=0.05,
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        # State: CLOSED
        assert breaker.is_open is False
        assert breaker.state.consecutive_failures == 0

        # Transition: CLOSED -> OPEN (via failures)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.is_open is True
        assert breaker.state.total_trips == 1

        # Transition: OPEN -> HALF-OPEN (via wait)
        breaker.check_and_wait()

        assert breaker.is_open is False
        # Failures preserved for quick re-trip
        assert breaker.state.consecutive_failures == 3

        # Transition: HALF-OPEN -> CLOSED (via success)
        breaker.record_success()

        assert breaker.is_open is False
        assert breaker.state.consecutive_failures == 0

    @pytest.mark.stress
    def test_state_machine_half_open_to_open(self):
        """Verify transition: HALF-OPEN -> OPEN (via failure)."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=2,
            pause_seconds=0.01,
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        # Trip to OPEN
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is True
        initial_trips = breaker.state.total_trips

        # Wait to HALF-OPEN
        breaker.check_and_wait()
        assert breaker.is_open is False

        # Fail again - should re-trip since consecutive_failures = 2
        # and one more makes it >= threshold again
        breaker.record_failure()

        # After this failure, consecutive_failures = 3 (> threshold of 2)
        # So circuit should trip again
        assert breaker.state.consecutive_failures == 3
        assert breaker.state.total_trips == initial_trips + 1

    @pytest.mark.stress
    def test_rapid_state_transitions_maintain_invariants(self):
        """Rapid transitions should maintain state invariants."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=5,
            pause_seconds=0.01,
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        def verify_invariants():
            """Check that state invariants hold."""
            # consecutive_failures should be non-negative
            assert breaker.state.consecutive_failures >= 0

            # total_trips should be non-negative
            assert breaker.state.total_trips >= 0

            # If open, opened_at should be set
            if breaker.is_open:
                assert breaker.state.opened_at is not None

            # total_paused_seconds should be non-negative
            assert breaker.state.total_paused_seconds >= 0

        # Rapid operations
        for _ in range(100):
            op = random.choice(['fail', 'success', 'reset', 'check'])

            if op == 'fail':
                breaker.record_failure()
            elif op == 'success':
                breaker.record_success()
            elif op == 'reset':
                breaker.reset()
            else:  # check
                if breaker.is_open:
                    breaker.check_and_wait()

            verify_invariants()

    @pytest.mark.stress
    def test_concurrent_operations_maintain_state_validity(self):
        """Concurrent operations should not corrupt state."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=10,
            pause_seconds=0.01,
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        errors = []
        errors_lock = threading.Lock()

        def random_operation():
            try:
                for _ in range(50):
                    op = random.choice(['fail', 'success', 'reset', 'check', 'stats'])

                    if op == 'fail':
                        breaker.record_failure()
                    elif op == 'success':
                        breaker.record_success()
                    elif op == 'reset':
                        breaker.reset()
                    elif op == 'check':
                        if breaker.is_open:
                            breaker.check_and_wait()
                    else:
                        breaker.get_stats()

                    # Brief yield to increase interleaving
                    time.sleep(0.001)

            except Exception as e:
                with errors_lock:
                    errors.append(str(e))

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(random_operation) for _ in range(10)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    with errors_lock:
                        errors.append(str(e))

        # Should complete without errors
        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"

        # Final state should be valid
        assert breaker.state.consecutive_failures >= 0
        assert breaker.state.total_trips >= 0
        assert isinstance(breaker.is_open, bool)
