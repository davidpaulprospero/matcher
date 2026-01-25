"""Circuit breaker for repeated search failures.

Implements the circuit breaker pattern to prevent hammering YouTube
when multiple consecutive searches fail. After threshold consecutive
failures, the circuit "trips" and pauses searching for a configured
duration before allowing new searches.

This protects against:
- Wasted API calls during widespread rate limiting
- Excessive retries that could worsen rate limit issues
- Unnecessarily slow pipeline execution during outages
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for search failure circuit breaker.

    When multiple consecutive searches fail (no results or errors),
    the circuit breaker trips and pauses all searches for a duration.
    This prevents hammering YouTube during rate limit windows.

    Example with defaults:
      - 5 searches fail in a row → circuit trips
      - Wait 60 seconds before allowing new searches
      - On next successful search → circuit resets to closed state

    Download retry coordination:
      When block_download_retries is enabled (default), the download retry loop
      in _run_download_cmd will check the circuit breaker state before each retry
      attempt. If the circuit breaker is tripped during a retry sequence, the
      retry will wait for the circuit breaker to recover before continuing.
      The retry count is preserved across circuit breaker pauses.
    """
    # Enable/disable circuit breaker
    enabled: bool = True

    # Number of consecutive failures before circuit trips (opens)
    consecutive_failures_threshold: int = 5

    # Duration to pause after circuit trips (seconds)
    pause_seconds: float = 60.0

    # Block download retries when circuit breaker is tripped
    # When true, download retry loop waits for circuit breaker recovery
    block_download_retries: bool = True


@dataclass
class CircuitBreakerState:
    """Internal state for circuit breaker."""
    consecutive_failures: int = 0
    is_open: bool = False  # True = circuit is tripped, searches paused
    opened_at: Optional[float] = None  # timestamp when circuit opened
    total_trips: int = 0  # Total times circuit has tripped this session
    total_paused_seconds: float = 0.0  # Total time spent paused this session


class CircuitBreaker:
    """Circuit breaker for YouTube search failures.

    Monitors consecutive search failures and temporarily pauses searches
    when a threshold is exceeded. This implements the circuit breaker
    pattern with three states:

    - CLOSED (normal): Searches allowed, failures tracked
    - OPEN (tripped): Searches paused, waiting for pause duration
    - HALF-OPEN (recovery): After pause, first search allowed as test

    Usage:
        breaker = CircuitBreaker(config)

        # Before each search
        breaker.check_and_wait()

        # After each search
        if search_successful:
            breaker.record_success()
        else:
            breaker.record_failure()

    Attributes:
        config: CircuitBreakerConfig with thresholds and timing
        state: Current state (failure count, open/closed, timing)
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker with configuration.

        Args:
            config: CircuitBreakerConfig. If None, uses defaults.
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()

    @property
    def is_enabled(self) -> bool:
        """Check if circuit breaker is enabled."""
        return self.config.enabled

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently open (tripped)."""
        return self.state.is_open

    def check_and_wait(self) -> bool:
        """Check circuit state and wait if necessary.

        If circuit is open (tripped), this method will:
        1. Log the pause clearly (INFO level, not hidden in debug)
        2. Sleep for the remaining pause duration
        3. Transition to half-open state (ready to test)

        Returns:
            True if search should proceed, False if circuit breaker is disabled.

        Note:
            This method blocks if circuit is open. Call before each search.
        """
        if not self.config.enabled:
            return False

        if not self.state.is_open:
            return True

        # Circuit is open - check if pause duration has elapsed
        elapsed = time.time() - self.state.opened_at
        remaining = self.config.pause_seconds - elapsed

        if remaining > 0:
            # Still in pause period - wait for remaining time
            logger.info(
                f"Circuit breaker OPEN: pausing {remaining:.1f}s "
                f"(trip #{self.state.total_trips}, "
                f"{self.state.consecutive_failures} consecutive failures)"
            )
            time.sleep(remaining)
            self.state.total_paused_seconds += remaining

        # Transition to half-open (closed but ready to trip quickly)
        logger.info("Circuit breaker: pause complete, allowing search (half-open)")
        self.state.is_open = False
        self.state.opened_at = None
        # Keep failure count - will reset on success or trip again on failure

        return True

    def record_success(self) -> None:
        """Record a successful search.

        Resets the consecutive failure counter and closes the circuit.
        Call after any search that returns results.
        """
        if not self.config.enabled:
            return

        if self.state.consecutive_failures > 0:
            logger.debug(
                f"Circuit breaker: search succeeded after "
                f"{self.state.consecutive_failures} failures, resetting counter"
            )

        self.state.consecutive_failures = 0
        self.state.is_open = False
        self.state.opened_at = None

    def record_failure(self) -> bool:
        """Record a search failure.

        Increments the consecutive failure counter. If threshold is reached,
        trips the circuit (opens it) and pauses future searches.

        Returns:
            True if circuit tripped (opened) as a result of this failure,
            False otherwise.
        """
        if not self.config.enabled:
            return False

        self.state.consecutive_failures += 1

        logger.debug(
            f"Circuit breaker: search failure "
            f"({self.state.consecutive_failures}/{self.config.consecutive_failures_threshold})"
        )

        # Check if we've hit the threshold
        if self.state.consecutive_failures >= self.config.consecutive_failures_threshold:
            self._trip()
            return True

        return False

    def _trip(self) -> None:
        """Trip the circuit breaker (open it).

        Called internally when consecutive failures reach threshold.
        Logs clearly at INFO level so users can see the pause happening.
        """
        self.state.is_open = True
        self.state.opened_at = time.time()
        self.state.total_trips += 1

        logger.info(
            f"Circuit breaker TRIPPED: {self.state.consecutive_failures} consecutive "
            f"search failures. Pausing for {self.config.pause_seconds:.0f}s before "
            f"allowing new searches. (trip #{self.state.total_trips})"
        )

    def reset(self) -> None:
        """Manually reset the circuit breaker.

        Clears all failure state and closes the circuit.
        Use when external conditions change (e.g., VPN switched, cookie rotated).
        """
        self.state.consecutive_failures = 0
        self.state.is_open = False
        self.state.opened_at = None
        logger.debug("Circuit breaker: manually reset")

    def get_remaining_pause_time(self) -> float:
        """Get remaining time until circuit breaker recovers.

        Returns:
            Remaining pause time in seconds, or 0.0 if not tripped.
        """
        if not self.config.enabled or not self.state.is_open:
            return 0.0

        if self.state.opened_at is None:
            return 0.0

        elapsed = time.time() - self.state.opened_at
        remaining = self.config.pause_seconds - elapsed
        return max(0.0, remaining)

    def wait_for_recovery_if_needed(self, context: str = "") -> float:
        """Wait for circuit breaker recovery if tripped.

        This is designed for download retry coordination (US-011). Unlike
        check_and_wait(), this method:
        - Returns the actual wait time for metrics tracking
        - Accepts a context string for more specific logging
        - Does not transition state (caller still needs check_and_wait for state transition)

        Args:
            context: Optional context string for logging (e.g., "download retry")

        Returns:
            The number of seconds waited, or 0.0 if no wait was needed.
        """
        if not self.config.enabled:
            return 0.0

        if not self.state.is_open:
            return 0.0

        remaining = self.get_remaining_pause_time()
        if remaining <= 0:
            return 0.0

        # Log the wait with context
        ctx_str = f" ({context})" if context else ""
        logger.info(
            f"Circuit breaker OPEN{ctx_str}: waiting {remaining:.1f}s for recovery "
            f"(trip #{self.state.total_trips}, "
            f"{self.state.consecutive_failures} consecutive failures)"
        )

        time.sleep(remaining)
        self.state.total_paused_seconds += remaining

        # Transition to half-open state
        logger.info(f"Circuit breaker: pause complete{ctx_str}, resuming")
        self.state.is_open = False
        self.state.opened_at = None

        return remaining

    def get_stats(self) -> dict:
        """Get circuit breaker statistics for reporting.

        Returns:
            Dict with stats including:
            - enabled: Whether circuit breaker is enabled
            - is_open: Current open/closed state
            - consecutive_failures: Current failure count
            - total_trips: Total times circuit has tripped this session
            - total_paused_seconds: Total time spent paused this session
            - threshold: Configured failure threshold
            - pause_seconds: Configured pause duration
        """
        return {
            'enabled': self.config.enabled,
            'is_open': self.state.is_open,
            'consecutive_failures': self.state.consecutive_failures,
            'total_trips': self.state.total_trips,
            'total_paused_seconds': round(self.state.total_paused_seconds, 1),
            'threshold': self.config.consecutive_failures_threshold,
            'pause_seconds': self.config.pause_seconds,
        }

    def to_checkpoint_dict(self) -> dict:
        """Serialize state to dictionary for checkpoint persistence.

        Returns:
            Dict that can be saved to checkpoint JSON.
        """
        return {
            'consecutive_failures': self.state.consecutive_failures,
            'total_trips': self.state.total_trips,
            'total_paused_seconds': self.state.total_paused_seconds,
            # Don't persist is_open/opened_at - start fresh on resume
        }

    def from_checkpoint_dict(self, data: dict) -> None:
        """Restore state from checkpoint dictionary.

        Args:
            data: Dict from checkpoint JSON.
        """
        if not data:
            return

        # Restore cumulative stats but not transient state
        self.state.total_trips = data.get('total_trips', 0)
        self.state.total_paused_seconds = data.get('total_paused_seconds', 0.0)
        # Don't restore consecutive_failures or is_open - start fresh on resume
        self.state.consecutive_failures = 0
        self.state.is_open = False
        self.state.opened_at = None
