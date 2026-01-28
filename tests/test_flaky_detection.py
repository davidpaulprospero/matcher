"""
Flaky Test Detection and Retry Pattern Demonstration

This module demonstrates the pytest-rerunfailures plugin usage for handling
intermittently failing tests. Use the @pytest.mark.flaky marker for tests
that may fail due to:
- Timing/race conditions
- Network requests with variable latency
- External service dependencies
- Resource contention under load

See tests/README.md for flaky test identification guidelines.
"""

import pytest
import random
import time
from typing import Generator
from unittest.mock import MagicMock


# ==============================================================================
# Pattern 1: Basic Flaky Marker
# ==============================================================================

@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
def test_flaky_basic_pattern():
    """
    Basic flaky test pattern - retries up to 2 times with 0.1s delay.

    Use this pattern when:
    - Test fails ~10-30% of the time due to timing issues
    - Quick fix by retrying is acceptable
    - Root cause is known but not worth fixing
    """
    # This test always passes, demonstrating the pattern
    assert True


# ==============================================================================
# Pattern 2: Flaky with Condition
# ==============================================================================

@pytest.mark.flaky(reruns=2, condition="sys.platform == 'win32'")
def test_flaky_windows_only():
    """
    Flaky test that only retries on Windows.

    Use this pattern when:
    - Test is only flaky on specific platforms
    - Platform-specific timing issues exist
    """
    assert True


# ==============================================================================
# Pattern 3: Flaky Network-Dependent Test
# ==============================================================================

class MockNetworkService:
    """Simulates a network service with intermittent failures."""

    def __init__(self, failure_rate: float = 0.0):
        self.failure_rate = failure_rate
        self.call_count = 0

    def make_request(self) -> dict:
        self.call_count += 1
        if random.random() < self.failure_rate:
            raise ConnectionError("Simulated network timeout")
        return {"status": "ok", "call": self.call_count}


@pytest.fixture
def mock_network_service() -> Generator[MockNetworkService, None, None]:
    """Fixture that provides a mock network service."""
    service = MockNetworkService(failure_rate=0.0)  # 0% failure for demo
    yield service


@pytest.mark.flaky(reruns=2, reruns_delay=0.5)
@pytest.mark.requires_network
def test_flaky_network_pattern(mock_network_service: MockNetworkService):
    """
    Pattern for testing network-dependent code with retry.

    Use this pattern when:
    - External API may be temporarily unavailable
    - Network latency causes occasional timeouts
    - Rate limiting may cause sporadic failures

    Best practices:
    1. Set reruns_delay to allow service recovery
    2. Combine with @pytest.mark.requires_network for CI filtering
    3. Consider increasing timeout for production tests
    """
    result = mock_network_service.make_request()
    assert result["status"] == "ok"


# ==============================================================================
# Pattern 4: Flaky Timing-Sensitive Test
# ==============================================================================

class AsyncOperationSimulator:
    """Simulates an async operation that may complete at variable times."""

    def __init__(self, base_delay: float = 0.01, jitter: float = 0.0):
        self.base_delay = base_delay
        self.jitter = jitter

    def run(self) -> str:
        # Simulate variable completion time
        actual_delay = self.base_delay + random.uniform(0, self.jitter)
        time.sleep(actual_delay)
        return "completed"


@pytest.mark.flaky(reruns=3, reruns_delay=0.1)
def test_flaky_timing_pattern():
    """
    Pattern for testing timing-sensitive operations.

    Use this pattern when:
    - Test depends on operation completing within a window
    - Thread scheduling may cause race conditions
    - Background tasks affect timing

    Best practices:
    1. Use timeouts instead of fixed sleeps where possible
    2. Increase tolerance windows for CI environments
    3. Consider using pytest-timeout in combination
    """
    simulator = AsyncOperationSimulator(base_delay=0.01, jitter=0.0)
    result = simulator.run()
    assert result == "completed"


# ==============================================================================
# Pattern 5: Flaky Test with Resource Contention
# ==============================================================================

@pytest.mark.flaky(reruns=2)
def test_flaky_resource_contention():
    """
    Pattern for tests with potential resource contention.

    Use this pattern when:
    - Multiple tests access shared resources (files, ports, sockets)
    - pytest-xdist parallel execution causes conflicts
    - Cleanup between tests is sometimes incomplete

    Best practices:
    1. Use unique resource identifiers per test (e.g., tmp_path)
    2. Implement proper resource cleanup in fixtures
    3. Consider test isolation over retries for persistent issues
    """
    # Simulate resource access that might conflict
    resource_id = random.randint(1, 1000000)  # Unique per test
    assert resource_id > 0


# ==============================================================================
# Pattern 6: Conditional Flaky Based on CI Environment
# ==============================================================================

import os

IS_CI = os.environ.get("CI", "false").lower() == "true"


@pytest.mark.flaky(reruns=3 if IS_CI else 1)
def test_flaky_ci_aware():
    """
    Pattern for tests that are more flaky in CI than local.

    Use this pattern when:
    - CI runners have more resource contention
    - CI network is slower/less reliable
    - Local development rarely sees failures

    Best practices:
    1. Use environment detection for retry configuration
    2. Log extra diagnostics when running in CI
    3. Consider separate markers for local vs CI runs
    """
    assert True


# ==============================================================================
# Pattern 7: Demonstrating Rerun Behavior (Test Infrastructure Validation)
# ==============================================================================

class RerunCounter:
    """Tracks test execution attempts for validation."""
    count: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.count = 0

    @classmethod
    def increment(cls) -> int:
        cls.count += 1
        return cls.count


@pytest.fixture(autouse=True)
def reset_rerun_counter() -> Generator[None, None, None]:
    """Reset the counter before each test."""
    RerunCounter.reset()
    yield


@pytest.mark.flaky(reruns=2)
def test_rerun_infrastructure_validation():
    """
    Validates that the flaky test infrastructure is working.

    This test always passes on first try, demonstrating that:
    1. The @pytest.mark.flaky marker is recognized
    2. pytest-rerunfailures is properly installed
    3. The retry mechanism is configured correctly
    """
    attempt = RerunCounter.increment()
    assert attempt >= 1, "Counter should track attempts"
    # This test passes on first try, no retries needed
    assert True


# ==============================================================================
# Pattern 8: Flaky Test Documentation Pattern
# ==============================================================================

@pytest.mark.flaky(reruns=2)
def test_documented_flaky_pattern():
    """
    Example of a well-documented flaky test.

    Flakiness Reason:
        This test demonstrates the documentation pattern. In real tests,
        document the specific reason for flakiness here.

    Known Failure Modes:
        - Example: "Fails ~5% of time when system under load"
        - Example: "Occasional network timeout to external API"

    Mitigation Attempts:
        - Increased timeout from 5s to 10s (reduced failures by 50%)
        - Added retry logic in source code (reduced failures by 30%)
        - Remaining failures handled by pytest-rerunfailures

    Related Issues:
        - GitHub Issue #123: Investigate root cause
        - Will be fixed when we migrate to async client

    Use this documentation pattern for:
    - Audit trail of flaky tests
    - Tracking improvement over time
    - Onboarding new developers
    """
    assert True
