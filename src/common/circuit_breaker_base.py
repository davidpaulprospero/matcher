"""Abstract base class for circuit breaker implementations.

Provides the common circuit breaker pattern shared by both the download
and caption circuit breaker implementations:
- State management: CLOSED (normal) -> OPEN (tripped) -> HALF-OPEN (recovery)
- Failure counting with configurable threshold
- Configurable pause duration with optional max cap
- Success recording to reset failure count and close circuit
- Statistics and checkpoint serialization

Domain-specific behavior (download cascade, jitter, escalation, budget
adjustment, caption batch abort) lives in the respective subclasses.

Extracted as part of US-67-010 to reduce code duplication and prevent
the two implementations from diverging.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerStateBase:
    """Internal state for circuit breaker.

    Shared by both download and caption circuit breakers.
    """
    consecutive_failures: int = 0
    is_open: bool = False  # True = circuit is tripped
    opened_at: Optional[float] = None  # timestamp when circuit opened
    total_trips: int = 0  # Total times circuit has tripped this session
    total_paused_seconds: float = 0.0  # Total time spent paused this session
    failure_history: list = None  # Recent failures with timestamps

    def __post_init__(self):
        if self.failure_history is None:
            self.failure_history = []


class CircuitBreakerBase(ABC):
    """Abstract base class for circuit breaker pattern.

    Implements the core state machine:
    - CLOSED (normal): Operations allowed, failures tracked
    - OPEN (tripped): Operations paused, waiting for pause duration
    - HALF-OPEN (recovery): After pause, first operation allowed as test

    Subclasses must implement:
    - _get_failure_threshold(): Return the failure count that triggers a trip
    - _get_effective_pause_seconds(): Return effective pause duration (may include
      jitter, escalation, budget adjustments)
    - _on_trip(): Called when circuit trips (for logging, cascade, etc.)
    - _on_record_failure(): Called on each failure (for cascade propagation)
    - _create_state(): Factory for the state dataclass
    - _get_domain_label(): Return domain label for logging (e.g., "search", "caption")
    """

    @property
    @abstractmethod
    def config(self):
        """Return the config object. Must have 'enabled' and 'pause_seconds' attrs."""
        ...

    @abstractmethod
    def _create_state(self) -> CircuitBreakerStateBase:
        """Create and return the initial state object."""
        ...

    @abstractmethod
    def _get_failure_threshold(self) -> int:
        """Return the number of consecutive failures that triggers a trip."""
        ...

    @abstractmethod
    def _get_effective_pause_seconds(self) -> float:
        """Return effective pause duration in seconds.

        May include jitter, escalation adjustments, budget adjustments, etc.
        """
        ...

    @abstractmethod
    def _on_trip(self) -> None:
        """Hook called after the circuit trips. For logging and cascade."""
        ...

    @abstractmethod
    def _on_record_failure(self) -> None:
        """Hook called after a failure is recorded. For cascade propagation."""
        ...

    @abstractmethod
    def _get_domain_label(self) -> str:
        """Return domain-specific label for log messages."""
        ...

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
        1. Log the pause clearly (INFO level)
        2. Sleep for the remaining pause duration
        3. Transition to half-open state (ready to test)

        Returns:
            True if operation should proceed, False if circuit breaker is disabled.
        """
        if not self.config.enabled:
            return False

        if not self.state.is_open:
            return True

        # Circuit is open - check if pause duration has elapsed
        effective_pause = self._get_effective_pause_seconds()
        elapsed = time.time() - self.state.opened_at
        remaining = effective_pause - elapsed

        if remaining > 0:
            self._log_pause_wait(remaining)
            time.sleep(remaining)
            self.state.total_paused_seconds += remaining

        # Transition to half-open
        label = self._get_domain_label()
        logger.info(f"Circuit breaker: pause complete, allowing {label} (half-open)")
        self.state.is_open = False
        self.state.opened_at = None

        return True

    def _log_pause_wait(self, remaining: float) -> None:
        """Log the pause wait. Subclasses can override for extra detail (e.g., jitter %)."""
        label = self._get_domain_label()
        logger.info(
            f"Circuit breaker OPEN: pausing {remaining:.1f}s "
            f"(trip #{self.state.total_trips}, "
            f"{self.state.consecutive_failures} consecutive failures)"
        )

    def record_success(self) -> None:
        """Record a successful operation.

        Resets the consecutive failure counter and closes the circuit.
        """
        if not self.config.enabled:
            return

        if self.state.consecutive_failures > 0:
            label = self._get_domain_label()
            logger.debug(
                f"Circuit breaker: {label} succeeded after "
                f"{self.state.consecutive_failures} failures, resetting counter"
            )

        self.state.consecutive_failures = 0
        self.state.is_open = False
        self.state.opened_at = None

    def record_failure(self) -> bool:
        """Record an operation failure.

        Increments the consecutive failure counter. If threshold is reached,
        trips the circuit (opens it) and pauses future operations.

        Returns:
            True if circuit tripped (opened) as a result of this failure,
            False otherwise.
        """
        if not self.config.enabled:
            return False

        self.state.consecutive_failures += 1

        # Track failure in history with timestamp
        failure_record = {'timestamp': time.time()}
        self.state.failure_history.append(failure_record)

        # Keep only last 100 failures in history
        if len(self.state.failure_history) > 100:
            self.state.failure_history = self.state.failure_history[-100:]

        threshold = self._get_failure_threshold()
        label = self._get_domain_label()
        logger.debug(
            f"Circuit breaker: {label} failure "
            f"({self.state.consecutive_failures}/{threshold})"
        )

        # Hook for cascade propagation
        self._on_record_failure()

        # Check if we've hit the threshold
        if self.state.consecutive_failures >= threshold:
            self._trip()
            return True

        return False

    def _trip(self) -> None:
        """Trip the circuit breaker (open it).

        Called internally when consecutive failures reach threshold.
        """
        self.state.is_open = True
        self.state.opened_at = time.time()
        self.state.total_trips += 1

        # Subclass hook for logging and cascade
        self._on_trip()

    def reset(self) -> None:
        """Manually reset the circuit breaker.

        Clears all failure state and closes the circuit.
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

        effective_pause = self._get_effective_pause_seconds()
        elapsed = time.time() - self.state.opened_at
        remaining = effective_pause - elapsed
        return max(0.0, remaining)

    def get_stats(self) -> dict:
        """Get circuit breaker statistics for reporting."""
        return {
            'enabled': self.config.enabled,
            'is_open': self.state.is_open,
            'consecutive_failures': self.state.consecutive_failures,
            'total_trips': self.state.total_trips,
            'total_paused_seconds': round(self.state.total_paused_seconds, 1),
            'threshold': self._get_failure_threshold(),
            'pause_seconds': self.config.pause_seconds,
        }

    def get_state(self) -> str:
        """Get the current circuit breaker state.

        Returns:
            'closed' - Normal operation, no failures or circuit reset
            'open' - Circuit is tripped, operations are paused
            'half_open' - Circuit was open but pause elapsed, testing recovery
        """
        if self.state.is_open:
            return 'open'
        elif self.state.consecutive_failures > 0:
            return 'half_open'
        else:
            return 'closed'

    def get_failure_history(self) -> list:
        """Get recent failure records with timestamps.

        Returns:
            List of dicts with 'timestamp' key (unix timestamp).
            Each record also has 'consecutive_failures' count at that point.
        """
        result = []
        for i, f in enumerate(self.state.failure_history):
            record = {'timestamp': f.get('timestamp', 0)}
            # Use stored value if available, otherwise calculate from index
            if 'consecutive_failures' in f:
                record['consecutive_failures'] = f['consecutive_failures']
            else:
                record['consecutive_failures'] = i + 1
            result.append(record)
        return result

    def to_checkpoint_dict(self) -> dict:
        """Serialize state to dictionary for checkpoint persistence."""
        return {
            'consecutive_failures': self.state.consecutive_failures,
            'is_open': self.state.is_open,
            'opened_at': self.state.opened_at,
            'total_trips': self.state.total_trips,
            'total_paused_seconds': self.state.total_paused_seconds,
            'failure_history': self.state.failure_history or [],
        }

    def from_checkpoint_dict(self, data: dict, checkpoint_age_seconds: float = 0.0) -> dict:
        """Restore state from checkpoint dictionary with stale state handling.

        Args:
            data: Dictionary with circuit breaker state from checkpoint
            checkpoint_age_seconds: Age of checkpoint in seconds. If > 3600 (1 hour),
                stale state handling applies - circuit is reset to half-open instead
                of fully closed.

        Returns:
            Dict with restoration info: {
                'restored': bool,
                'was_stale': bool,
                'restored_state': str (e.g., 'open', 'half_open', 'closed')
            }
        """
        result = {
            'restored': False,
            'was_stale': False,
            'restored_state': 'closed'
        }

        if not data:
            return result

        # Restore cumulative stats
        self.state.total_trips = data.get('total_trips', 0)
        self.state.total_paused_seconds = data.get('total_paused_seconds', 0.0)

        # Check if checkpoint is stale (> 1 hour old)
        is_stale = checkpoint_age_seconds > 3600.0

        if is_stale:
            # Stale checkpoint: reset to half-open state (conservative recovery)
            # This allows but the circuit to resume with caution
            result['was_stale'] = True
            self.state.consecutive_failures = data.get('consecutive_failures', 0)
            # Keep failure count but clear the open state
            self.state.is_open = False
            self.state.opened_at = None
            result['restored_state'] = 'half_open'
            logger.info(
                f"Circuit breaker: stale checkpoint detected (age: {checkpoint_age_seconds/3600:.1f}h), "
                f"restoring to half-open state with {self.state.consecutive_failures} failures"
            )
        else:
            # Fresh checkpoint: restore full state
            self.state.consecutive_failures = data.get('consecutive_failures', 0)
            self.state.is_open = data.get('is_open', False)

            if self.state.is_open:
                # Restore the timestamp so remaining pause can be calculated
                self.state.opened_at = data.get('opened_at')
                result['restored_state'] = 'open'
            else:
                self.state.opened_at = None
                result['restored_state'] = 'closed' if self.state.consecutive_failures == 0 else 'half_open'

        result['restored'] = True
        return result
