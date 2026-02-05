"""Circuit breaker for repeated caption fetch failures.

Implements the circuit breaker pattern to prevent hammering YouTube
when multiple consecutive caption fetches fail. After threshold consecutive
failures, the circuit "trips" and pauses fetching for a configured
duration before allowing new fetches.

This protects against:
- Wasted API calls during widespread rate limiting
- Excessive retries that could worsen rate limit issues
- Unnecessarily slow pipeline execution during outages

Adapted from src/downloader/circuit_breaker.py for caption-specific use.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.common.circuit_breaker_base import CircuitBreakerBase, CircuitBreakerStateBase

# Canonical config source (Rule 11): import from config/sections/download.py
from src.config.sections.download import CaptionCircuitBreakerConfig  # noqa: F401 - re-exported

if TYPE_CHECKING:
    from src.downloader.circuit_breaker import CircuitBreaker as DownloadCircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class CaptionCircuitBreakerState(CircuitBreakerStateBase):
    """Internal state for circuit breaker."""
    pass


class CaptionCircuitBreaker(CircuitBreakerBase):
    """Circuit breaker for YouTube caption fetch failures.

    Monitors consecutive fetch failures and temporarily pauses fetches
    when a threshold is exceeded. This implements the circuit breaker
    pattern with three states:

    - CLOSED (normal): Fetches allowed, failures tracked
    - OPEN (tripped): Fetches paused, waiting for pause duration
    - HALF-OPEN (recovery): After pause, first fetch allowed as test

    Usage:
        breaker = CaptionCircuitBreaker(config)

        # Before each fetch
        breaker.check_and_wait()

        # After each fetch
        if fetch_successful:
            breaker.record_success()
        else:
            breaker.record_failure()

    Attributes:
        config: CaptionCircuitBreakerConfig with thresholds and timing
        state: Current state (failure count, open/closed, timing)
    """

    def __init__(self, config: Optional[CaptionCircuitBreakerConfig] = None):
        """Initialize circuit breaker with configuration.

        Args:
            config: CaptionCircuitBreakerConfig. If None, uses defaults.
        """
        self._config = config or CaptionCircuitBreakerConfig()
        self.state = self._create_state()
        self._download_circuit_breaker: Optional['DownloadCircuitBreaker'] = None

    @property
    def config(self) -> CaptionCircuitBreakerConfig:
        return self._config

    def _create_state(self) -> CaptionCircuitBreakerState:
        return CaptionCircuitBreakerState()

    def _get_failure_threshold(self) -> int:
        return self.config.threshold

    def _get_domain_label(self) -> str:
        return "fetch"

    # --- Cascade linking ---

    def set_download_circuit_breaker(self, download_cb: 'DownloadCircuitBreaker') -> None:
        """Link the download circuit breaker for cascade coordination (US-61-003).

        When linked and circuit_breaker_cascade is enabled:
        - Caption CB failures increment the download CB's shared failure counter
        - When caption CB trips, it propagates the open state to download CB

        Args:
            download_cb: The download CircuitBreaker instance to coordinate with.
        """
        self._download_circuit_breaker = download_cb
        logger.debug("Caption circuit breaker linked to download circuit breaker for cascade")

    # --- Cascade logic ---

    def _cascade_failure_to_download(self) -> None:
        """Propagate failure to download circuit breaker (US-61-003).

        Called when caption CB records a failure. Increments the download CB's
        failure counter to speed up coordinated tripping when YouTube is
        broadly rate-limiting.
        """
        if not self._download_circuit_breaker:
            return

        cascade_enabled = getattr(self.config, 'circuit_breaker_cascade', True)
        if not cascade_enabled:
            return

        if not self._download_circuit_breaker.is_enabled:
            return

        # Increment download CB failure counter (but don't record full failure
        # which would apply its own threshold logic - just increment counter)
        self._download_circuit_breaker.state.consecutive_failures += 1
        logger.debug(
            f"Caption CB cascading failure to download CB "
            f"(download failures now: {self._download_circuit_breaker.state.consecutive_failures})"
        )

    def _cascade_trip_to_download(self) -> None:
        """Propagate trip (open) state to download circuit breaker (US-61-003).

        Called when caption CB trips. Propagates the pause state to download CB
        so both circuit breakers pause together.
        """
        if not self._download_circuit_breaker:
            return

        cascade_enabled = getattr(self.config, 'circuit_breaker_cascade', True)
        if not cascade_enabled:
            return

        if not self._download_circuit_breaker.is_enabled:
            return

        # If download CB is not already open, trip it
        if not self._download_circuit_breaker.state.is_open:
            logger.info(
                "Caption circuit breaker cascading trip to download circuit breaker"
            )
            self._download_circuit_breaker.state.is_open = True
            self._download_circuit_breaker.state.opened_at = self.state.opened_at
            self._download_circuit_breaker.state.total_trips += 1

    def _on_record_failure(self) -> None:
        """Called after each failure - cascade to download CB."""
        self._cascade_failure_to_download()

    def _on_trip(self) -> None:
        """Called after circuit trips - log and cascade to download CB."""
        effective_pause = self._get_effective_pause_seconds()
        logger.info(
            f"Caption circuit breaker TRIPPED: {self.state.consecutive_failures} consecutive "
            f"fetch failures. Pausing for {effective_pause:.0f}s before "
            f"allowing new fetches. (trip #{self.state.total_trips})"
        )
        self._cascade_trip_to_download()

    # --- Pause calculation ---

    def _get_effective_pause_seconds(self) -> float:
        """Get the effective pause duration, capped at max_pause_seconds.

        Returns:
            Effective pause duration in seconds.
        """
        pause = self.config.pause_seconds
        max_pause = getattr(self.config, 'max_pause_seconds', 300.0)
        if pause > max_pause:
            logger.debug(
                f"Caption circuit breaker pause capped: {pause:.0f}s -> {max_pause:.0f}s"
            )
            pause = max_pause
        return pause

    # --- Overrides to match original log messages exactly ---

    def check_and_wait(self) -> bool:
        """Check circuit state and wait if necessary.

        If circuit is open (tripped), this method will:
        1. Log the pause clearly (INFO level)
        2. Sleep for the remaining pause duration
        3. Transition to half-open state (ready to test)

        Returns:
            True if fetch should proceed, False if circuit breaker is disabled.

        Note:
            This method blocks if circuit is open. Call before each fetch.
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
            # Still in pause period - wait for remaining time
            logger.info(
                f"Caption circuit breaker OPEN: pausing {remaining:.1f}s "
                f"(trip #{self.state.total_trips}, "
                f"{self.state.consecutive_failures} consecutive failures)"
            )
            time.sleep(remaining)
            self.state.total_paused_seconds += remaining

        # Transition to half-open (closed but ready to trip quickly)
        logger.info("Caption circuit breaker: pause complete, allowing fetch (half-open)")
        self.state.is_open = False
        self.state.opened_at = None
        # Keep failure count - will reset on success or trip again on failure

        return True

    def record_success(self) -> None:
        """Record a successful fetch.

        Resets the consecutive failure counter and closes the circuit.
        Call after any fetch that returns captions successfully.
        """
        if not self.config.enabled:
            return

        if self.state.consecutive_failures > 0:
            logger.debug(
                f"Caption circuit breaker: fetch succeeded after "
                f"{self.state.consecutive_failures} failures, resetting counter"
            )

        self.state.consecutive_failures = 0
        self.state.is_open = False
        self.state.opened_at = None

    def record_failure(self) -> bool:
        """Record a fetch failure.

        Increments the consecutive failure counter. If threshold is reached,
        trips the circuit (opens it) and pauses future fetches.

        US-61-003: Also cascades failure to download circuit breaker if linked.

        Returns:
            True if circuit tripped (opened) as a result of this failure,
            False otherwise.
        """
        if not self.config.enabled:
            return False

        self.state.consecutive_failures += 1

        logger.debug(
            f"Caption circuit breaker: fetch failure "
            f"({self.state.consecutive_failures}/{self.config.threshold})"
        )

        # US-61-003: Cascade failure to download CB
        self._cascade_failure_to_download()

        # Check if we've hit the threshold
        if self.state.consecutive_failures >= self.config.threshold:
            self._trip()
            return True

        return False

    def _trip(self) -> None:
        """Trip the circuit breaker (open it).

        Called internally when consecutive failures reach threshold.
        Logs clearly at INFO level so users can see the pause happening.
        US-61-003: Also cascades trip to download circuit breaker if linked.
        """
        self.state.is_open = True
        self.state.opened_at = time.time()
        self.state.total_trips += 1

        effective_pause = self._get_effective_pause_seconds()
        logger.info(
            f"Caption circuit breaker TRIPPED: {self.state.consecutive_failures} consecutive "
            f"fetch failures. Pausing for {effective_pause:.0f}s before "
            f"allowing new fetches. (trip #{self.state.total_trips})"
        )

        # US-61-003: Cascade trip to download CB
        self._cascade_trip_to_download()

    def reset(self) -> None:
        """Manually reset the circuit breaker.

        Clears all failure state and closes the circuit.
        Use when external conditions change (e.g., VPN switched).
        """
        self.state.consecutive_failures = 0
        self.state.is_open = False
        self.state.opened_at = None
        logger.debug("Caption circuit breaker: manually reset")

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
            'threshold': self.config.threshold,
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
