"""
Stress tests for rate limiting under load.

Tests the rate limiting subsystem under high-volume conditions with 100+
simulated requests and 30% rate limit responses. Uses mocked delays to
ensure tests complete quickly (<30 seconds total).

User Story: US-33-011 - Add stress tests for rate limiting under load
Created: 2026-02-01
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.rate_limit_budget import RateLimitBudget
from src.downloader.rate_limit_metrics import RateLimitMetrics
from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def budget():
    """Create a RateLimitBudget with moderate limits for stress testing."""
    b = RateLimitBudget()
    b.max_rotations = 20
    b.max_vpn_switches = 5
    b.max_backoff_time = 120.0
    return b


@pytest.fixture
def metrics():
    """Create fresh RateLimitMetrics for stress testing."""
    return RateLimitMetrics()


@pytest.fixture
def circuit_breaker():
    """Create a CircuitBreaker with test-friendly config."""
    config = CircuitBreakerConfig(
        enabled=True,
        consecutive_failures_threshold=5,
        pause_seconds=10.0,  # Short pause for tests
        max_pause_seconds=30.0,
    )
    return CircuitBreaker(config)


# ============================================================================
# Helper Classes
# ============================================================================

@dataclass
class SimulatedRequest:
    """Represents a simulated download request."""
    request_id: int
    keyword: str
    is_rate_limited: bool
    backoff_delay: float = 0.0
    success: bool = False


@dataclass
class StressTestResults:
    """Aggregated results from a stress test run."""
    total_requests: int = 0
    rate_limited_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_backoff_time: float = 0.0
    circuit_breaker_trips: int = 0
    rotations_used: int = 0
    vpn_switches_used: int = 0
    requests: List[SimulatedRequest] = field(default_factory=list)

    @property
    def rate_limit_percentage(self) -> float:
        """Calculate percentage of requests that were rate limited."""
        if self.total_requests == 0:
            return 0.0
        return 100.0 * self.rate_limited_count / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return 100.0 * self.success_count / self.total_requests


def simulate_rate_limit_scenario(
    request_count: int,
    rate_limit_percentage: float,
    budget: RateLimitBudget,
    metrics: RateLimitMetrics,
    circuit_breaker: CircuitBreaker,
    max_backoff_per_request: float = 2.0,
    seed: int = 42,
) -> StressTestResults:
    """Simulate a batch of download requests with rate limiting.

    Args:
        request_count: Number of requests to simulate
        rate_limit_percentage: Percentage of requests that hit rate limits (0-100)
        budget: RateLimitBudget to track resources
        metrics: RateLimitMetrics to record events
        circuit_breaker: CircuitBreaker to manage failures
        max_backoff_per_request: Maximum backoff delay per rate-limited request
        seed: Random seed for reproducibility

    Returns:
        StressTestResults with aggregated statistics
    """
    random.seed(seed)
    results = StressTestResults()
    keywords = ["sunset", "ocean", "mountain", "forest", "city"]
    consecutive_failures = 0

    for i in range(request_count):
        request = SimulatedRequest(
            request_id=i,
            keyword=random.choice(keywords),
            is_rate_limited=random.random() * 100 < rate_limit_percentage,
        )
        results.requests.append(request)
        results.total_requests += 1

        # Record download attempt
        metrics.record_download_attempt()
        budget.record_attempt(keyword=request.keyword)

        if request.is_rate_limited:
            results.rate_limited_count += 1
            consecutive_failures += 1

            # Record rate limit event
            metrics.record_rate_limit_event(keyword=request.keyword)
            budget.record_failure(keyword=request.keyword)

            # Calculate backoff delay (simulated, not actually slept)
            backoff_delay = random.uniform(0.5, max_backoff_per_request)
            request.backoff_delay = backoff_delay

            if budget.can_backoff(backoff_delay):
                budget.record_backoff(backoff_delay, keyword=request.keyword)
                metrics.record_backoff(backoff_delay)
                results.total_backoff_time += backoff_delay
            else:
                # Backoff budget exhausted, try rotation
                if budget.can_rotate():
                    budget.record_rotation(keyword=request.keyword)
                    metrics.record_cookie_rotation()
                    results.rotations_used += 1
                elif budget.can_switch_vpn():
                    budget.record_vpn_switch(keyword=request.keyword)
                    metrics.record_vpn_switch()
                    results.vpn_switches_used += 1

            # Record failure in circuit breaker
            if circuit_breaker.record_failure():
                results.circuit_breaker_trips += 1
                metrics.record_circuit_breaker_trip(circuit_breaker.config.pause_seconds)

            metrics.record_download_failure()
            results.failure_count += 1
            request.success = False
        else:
            # Successful request
            consecutive_failures = 0
            circuit_breaker.record_success()
            metrics.record_download_success()
            budget.record_success(keyword=request.keyword)
            results.success_count += 1
            request.success = True

    return results


# ============================================================================
# Stress Tests
# ============================================================================

@pytest.mark.stress
class TestRateLimitStress:
    """Stress tests for rate limiting subsystem under load."""

    def test_100_requests_with_30_percent_rate_limiting(
        self, budget, metrics, circuit_breaker
    ):
        """Test 100 simulated requests with 30% rate limit responses.

        Acceptance Criteria:
        - Simulate 100 requests with 30% rate limit responses
        - Verify all metrics are correctly tracked
        """
        results = simulate_rate_limit_scenario(
            request_count=100,
            rate_limit_percentage=30.0,
            budget=budget,
            metrics=metrics,
            circuit_breaker=circuit_breaker,
        )

        # Verify request count
        assert results.total_requests == 100

        # Verify rate limiting occurred (should be around 30%, allow wider variance for randomness)
        assert 15 <= results.rate_limited_count <= 45, (
            f"Expected ~30% rate limits, got {results.rate_limit_percentage:.1f}%"
        )

        # Verify metrics recorded correctly
        assert metrics.total_downloads == 100
        assert metrics.successful_downloads == results.success_count
        assert metrics.failed_downloads == results.failure_count
        assert metrics.rate_limit_events == results.rate_limited_count

    def test_backoff_delays_within_configured_bounds(
        self, budget, metrics, circuit_breaker
    ):
        """Verify backoff delays stay within configured bounds.

        Acceptance Criteria:
        - Verify backoff delays stay within configured bounds
        """
        max_backoff = 3.0
        results = simulate_rate_limit_scenario(
            request_count=100,
            rate_limit_percentage=40.0,
            budget=budget,
            metrics=metrics,
            circuit_breaker=circuit_breaker,
            max_backoff_per_request=max_backoff,
        )

        # Check individual backoff delays are within bounds
        for request in results.requests:
            if request.is_rate_limited and request.backoff_delay > 0:
                assert request.backoff_delay <= max_backoff, (
                    f"Request {request.request_id} backoff {request.backoff_delay:.2f}s "
                    f"exceeds max {max_backoff}s"
                )

        # Verify total backoff tracked in budget doesn't exceed limit
        assert budget.backoff_time_spent <= budget.max_backoff_time + 0.01, (
            f"Budget backoff {budget.backoff_time_spent:.1f}s exceeds "
            f"max {budget.max_backoff_time:.1f}s"
        )

    def test_circuit_breaker_trips_at_correct_threshold(
        self, budget, metrics
    ):
        """Verify circuit breaker trips at correct threshold.

        Acceptance Criteria:
        - Verify circuit breaker trips at correct threshold (5 consecutive failures)
        """
        # Create circuit breaker with threshold of 5
        config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=5,
            pause_seconds=1.0,
        )
        cb = CircuitBreaker(config)

        # Record 4 failures - should not trip
        for i in range(4):
            tripped = cb.record_failure()
            assert not tripped, f"Circuit breaker tripped early at failure {i+1}"
            assert not cb.is_open, f"Circuit breaker open at failure {i+1}"

        # 5th failure should trip
        tripped = cb.record_failure()
        assert tripped, "Circuit breaker did not trip at threshold"
        assert cb.is_open, "Circuit breaker not open after trip"
        assert cb.state.total_trips == 1

        # Test with 100 requests, 100% failure rate
        cb2 = CircuitBreaker(config)
        trip_count = 0
        for i in range(100):
            if cb2.record_failure():
                trip_count += 1
                # Reset for next batch (simulating recovery)
                cb2.reset()

        # Should trip multiple times with 100 consecutive failures
        # 100 failures / 5 threshold = 20 trips
        assert trip_count == 20, f"Expected 20 trips, got {trip_count}"

    def test_budget_tracking_accurate_after_high_volume(
        self, budget, metrics, circuit_breaker
    ):
        """Verify budget tracking accurate after high-volume operations.

        Acceptance Criteria:
        - Verify budget tracking accurate after high-volume operations
        """
        results = simulate_rate_limit_scenario(
            request_count=200,
            rate_limit_percentage=50.0,  # High rate limit rate
            budget=budget,
            metrics=metrics,
            circuit_breaker=circuit_breaker,
            max_backoff_per_request=1.0,
        )

        # Verify budget tracking accuracy
        summary = budget.get_summary()

        # Rotations tracked correctly
        assert summary["rotations_used"] == results.rotations_used
        assert summary["rotations_remaining"] == budget.max_rotations - results.rotations_used

        # VPN switches tracked correctly
        assert summary["vpn_switches_used"] == results.vpn_switches_used

        # Backoff time tracked (should be close to results total)
        # Note: budget may cap earlier than results track
        assert summary["backoff_time_spent"] <= results.total_backoff_time + 0.01

        # Success/failure counts match
        assert budget.successes == results.success_count
        assert budget.failures == results.failure_count

    def test_metrics_consistency_after_stress(
        self, budget, metrics, circuit_breaker
    ):
        """Test that metrics remain internally consistent after stress testing."""
        results = simulate_rate_limit_scenario(
            request_count=150,
            rate_limit_percentage=35.0,
            budget=budget,
            metrics=metrics,
            circuit_breaker=circuit_breaker,
        )

        # Total downloads = successes + failures
        assert metrics.total_downloads == (
            metrics.successful_downloads + metrics.failed_downloads
        )

        # Rate limit events <= failed downloads
        assert metrics.rate_limit_events <= metrics.failed_downloads

        # Backoff attempts <= rate limit events
        assert metrics.backoff_attempts <= metrics.rate_limit_events

        # Circuit breaker trips tracked in metrics
        cb_stats = circuit_breaker.get_stats()
        assert cb_stats['total_trips'] == results.circuit_breaker_trips

    def test_stress_with_budget_exhaustion(self, metrics, circuit_breaker):
        """Test behavior when budget becomes exhausted under load."""
        # Create a tight budget that will exhaust quickly
        tight_budget = RateLimitBudget()
        tight_budget.max_rotations = 3
        tight_budget.max_vpn_switches = 1
        tight_budget.max_backoff_time = 10.0

        results = simulate_rate_limit_scenario(
            request_count=100,
            rate_limit_percentage=60.0,  # Very high rate limiting
            budget=tight_budget,
            metrics=metrics,
            circuit_breaker=circuit_breaker,
            max_backoff_per_request=2.0,
        )

        # Budget should be exhausted or nearly exhausted
        assert tight_budget.backoff_time_spent >= tight_budget.max_backoff_time * 0.8 or \
               tight_budget.rotations_used >= tight_budget.max_rotations * 0.8 or \
               tight_budget.vpn_switches_used >= tight_budget.max_vpn_switches * 0.8

        # Exhaustion detection works
        assert tight_budget.is_exhausted() or tight_budget.is_nearly_exhausted()

    @pytest.mark.parametrize("request_count,rate_limit_pct", [
        (100, 10),   # Low rate limiting
        (100, 30),   # Medium rate limiting
        (100, 50),   # High rate limiting
        (100, 70),   # Very high rate limiting
    ])
    def test_various_rate_limit_percentages(
        self, request_count, rate_limit_pct, budget, metrics, circuit_breaker
    ):
        """Test behavior across various rate limit percentages."""
        results = simulate_rate_limit_scenario(
            request_count=request_count,
            rate_limit_percentage=rate_limit_pct,
            budget=budget,
            metrics=metrics,
            circuit_breaker=circuit_breaker,
            seed=rate_limit_pct,  # Different seed per scenario
        )

        # Basic sanity checks
        assert results.total_requests == request_count
        assert metrics.total_downloads == request_count

        # Rate limit count should be roughly proportional
        expected_min = request_count * (rate_limit_pct / 100) * 0.5
        expected_max = request_count * (rate_limit_pct / 100) * 1.5
        assert expected_min <= results.rate_limited_count <= expected_max

    def test_circuit_breaker_recovery_after_success(self, budget, metrics):
        """Test that circuit breaker recovers after successful requests."""
        config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=5,
            pause_seconds=0.1,  # Very short for testing
        )
        cb = CircuitBreaker(config)

        # Trip the circuit breaker
        for _ in range(5):
            cb.record_failure()
        assert cb.is_open

        # Mock time.sleep to avoid actual delays
        with patch('time.sleep'):
            cb.check_and_wait()

        # Record success to reset
        cb.record_success()
        assert not cb.is_open
        assert cb.state.consecutive_failures == 0

    def test_keyword_distribution_under_load(self, budget, metrics, circuit_breaker):
        """Test that rate limiting is tracked per keyword under load."""
        results = simulate_rate_limit_scenario(
            request_count=100,
            rate_limit_percentage=40.0,
            budget=budget,
            metrics=metrics,
            circuit_breaker=circuit_breaker,
        )

        # Verify keywords are tracked in budget
        assert len(budget.keywords_rate_limited) > 0

        # Verify keyword distribution in metrics
        assert len(metrics.keyword_rate_limit_events) > 0

        # Total events across keywords should match total rate limit events
        total_keyword_events = sum(metrics.keyword_rate_limit_events.values())
        assert total_keyword_events == metrics.rate_limit_events


@pytest.mark.stress
class TestStressPerformance:
    """Performance tests to ensure stress tests complete quickly."""

    def test_stress_test_completes_under_30_seconds(
        self, budget, metrics, circuit_breaker
    ):
        """Ensure stress test simulation completes under 30 seconds.

        Acceptance Criteria:
        - Tests complete in under 30 seconds using mocked delays
        """
        start_time = time.time()

        # Run a comprehensive simulation
        results = simulate_rate_limit_scenario(
            request_count=1000,  # Even higher than AC requirement
            rate_limit_percentage=40.0,
            budget=budget,
            metrics=metrics,
            circuit_breaker=circuit_breaker,
        )

        elapsed = time.time() - start_time

        # Should complete well under 30 seconds (no actual sleeps)
        assert elapsed < 30.0, f"Stress test took {elapsed:.2f}s, exceeds 30s limit"

        # Should complete very quickly since delays are mocked
        assert elapsed < 1.0, f"Stress test took {elapsed:.2f}s, expected <1s with mocked delays"

        # Verify results are still valid
        assert results.total_requests == 1000
        assert metrics.total_downloads == 1000

    def test_no_actual_delays_in_simulation(self, budget, metrics, circuit_breaker):
        """Verify that simulation doesn't introduce actual time delays."""
        start_time = time.time()

        # Run with many rate limits that would normally cause delays
        simulate_rate_limit_scenario(
            request_count=500,
            rate_limit_percentage=80.0,  # Very high rate limiting
            budget=budget,
            metrics=metrics,
            circuit_breaker=circuit_breaker,
            max_backoff_per_request=10.0,  # Would be 5000s of delays if real
        )

        elapsed = time.time() - start_time

        # Even with simulated 4000s+ of backoff, should complete instantly
        assert elapsed < 2.0, f"Simulation took {elapsed:.2f}s, expected <2s"


@pytest.mark.stress
class TestConcurrentScenarios:
    """Test rate limiting behavior in concurrent-like scenarios."""

    def test_burst_of_rate_limits(self, budget, metrics):
        """Test handling a burst of consecutive rate limits."""
        # Create fresh circuit breaker with threshold 5
        config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=5,
            pause_seconds=0.1,
        )
        cb = CircuitBreaker(config)
        trip_count = 0

        # First 20 requests all rate limited (simulating a burst)
        for i in range(20):
            request = SimulatedRequest(
                request_id=i,
                keyword="burst_test",
                is_rate_limited=True,
            )
            metrics.record_download_attempt()
            metrics.record_rate_limit_event(keyword=request.keyword)
            budget.record_failure(keyword=request.keyword)

            if budget.can_backoff(1.0):
                budget.record_backoff(1.0, keyword=request.keyword)
                metrics.record_backoff(1.0)

            if cb.record_failure():
                trip_count += 1
                # Circuit trips, reset for next batch (simulating recovery wait)
                cb.reset()

            metrics.record_download_failure()

        # Circuit breaker should have tripped 4 times (20 failures / 5 threshold)
        assert trip_count == 4

        # All keywords should be tracked
        assert "burst_test" in budget.keywords_rate_limited

    def test_alternating_success_failure_pattern(
        self, budget, metrics, circuit_breaker
    ):
        """Test alternating success/failure pattern doesn't trip breaker."""
        for i in range(100):
            metrics.record_download_attempt()

            if i % 2 == 0:  # Even = success
                metrics.record_download_success()
                circuit_breaker.record_success()
                budget.record_success()
            else:  # Odd = failure
                metrics.record_download_failure()
                metrics.record_rate_limit_event()
                circuit_breaker.record_failure()
                budget.record_failure()

        # Circuit breaker should never trip (max 1 consecutive failure)
        assert circuit_breaker.state.total_trips == 0
        assert not circuit_breaker.is_open

        # But rate limits should still be tracked
        assert metrics.rate_limit_events == 50

    def test_recovery_after_total_budget_exhaustion(self, metrics, circuit_breaker):
        """Test that system handles complete budget exhaustion gracefully."""
        # Create an exhausted budget
        exhausted_budget = RateLimitBudget()
        exhausted_budget.max_rotations = 2
        exhausted_budget.max_vpn_switches = 1
        exhausted_budget.max_backoff_time = 5.0
        exhausted_budget.rotations_used = 2
        exhausted_budget.vpn_switches_used = 1
        exhausted_budget.backoff_time_spent = 5.0

        # Verify exhausted
        assert exhausted_budget.is_exhausted()

        # Can still track events even when exhausted
        for i in range(10):
            exhausted_budget.record_failure(keyword=f"kw_{i}")

        # All keywords tracked
        assert len(exhausted_budget.keywords_rate_limited) == 10

        # Failures counter incremented
        assert exhausted_budget.failures == 10
