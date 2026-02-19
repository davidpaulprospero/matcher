"""Retry queue processor for handling batch retry execution.

This module extracts the processing logic from RetryQueue into a dedicated class
following the Single Responsibility Principle:
- RetryQueue: Pure data structure (add/get/clear items)
- RetryQueueProcessor: Execution logic (delay, circuit breaker wait, cookie cooldown)

Usage:
    queue = RetryQueue(config)
    processor = RetryQueueProcessor(queue)
    processor.set_circuit_breaker(circuit_breaker)
    processor.set_cookie_rotator(cookie_rotator)

    # During download batch
    if rate_limit_error:
        queue.add(video_id, keyword, tier, error_message)

    # After batch completes
    if processor.has_pending():
        processor.start_retry_pass()
        for item in queue.get_pending_items():
            # Process item
            pass
        processor.finish_retry_pass()
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional, Dict, TYPE_CHECKING

from .core import SEVERITY_MULTIPLIERS

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreaker
    from .cookie_rotator import CookieRotator
    from .retry_queue import RetryQueue
    from .retry_queue import DownloadRetryBudget

logger = logging.getLogger(__name__)


class RetryQueueProcessor:
    """Processor for executing retry queue operations.

    Handles the execution aspects of batch retry:
    - Delay calculation with jitter
    - Circuit breaker coordination and waiting
    - Cookie cooldown coordination and waiting
    - Combined wait strategy to avoid deadlock

    The processor operates on a RetryQueue instance but doesn't modify
    its core data structure - only coordinates timing and execution.

    Attributes:
        queue: The RetryQueue instance to process
        _circuit_breaker: Optional linked circuit breaker
        _cookie_rotator: Optional linked cookie rotator
        _circuit_breaker_wait_time: Total time spent waiting for circuit breaker
        _cookie_cooldown_wait_time: Total time spent waiting for cookie cooldown
        _forced_retry: True when deadlock forced a retry without full wait
        _last_jitter_applied: Track last jitter for debugging/metrics
        _budget_state: Budget snapshot when items were queued
    """

    def __init__(self, queue: 'RetryQueue'):
        """Initialize processor with a RetryQueue instance.

        Args:
            queue: RetryQueue instance to process.
        """
        self._queue = queue
        self._circuit_breaker: Optional['CircuitBreaker'] = None
        self._cookie_rotator: Optional['CookieRotator'] = None
        self._circuit_breaker_wait_time: float = 0.0
        self._cookie_cooldown_wait_time: float = 0.0
        self._forced_retry: bool = False
        self._last_jitter_applied: float = 0.0
        self._budget_state: Optional[Dict] = None
        self._retry_budget: Optional['DownloadRetryBudget'] = None

        # US-136-011: Retry metrics tracking
        self._retry_attempts: int = 0  # Total retry attempts across all passes
        self._retry_successes: int = 0  # Successful retries (video downloaded)
        self._retry_failures: int = 0  # Failed retries (video still failing)
        self._pass_retry_counts: Dict[int, int] = {}  # retry count per pass

    @property
    def queue(self) -> 'RetryQueue':
        """Get the underlying RetryQueue instance."""
        return self._queue

    @property
    def config(self):
        """Get the queue's configuration."""
        return self._queue.config

    @property
    def forced_retry(self) -> bool:
        """True if the last retry pass was forced due to combined wait timeout."""
        return self._forced_retry

    @property
    def circuit_breaker_wait_time(self) -> float:
        """Total time spent waiting for circuit breaker."""
        return self._circuit_breaker_wait_time

    @property
    def cookie_cooldown_wait_time(self) -> float:
        """Total time spent waiting for cookie cooldown."""
        return self._cookie_cooldown_wait_time

    def set_circuit_breaker(self, circuit_breaker: 'CircuitBreaker') -> None:
        """Link a circuit breaker to coordinate retry timing.

        When a circuit breaker is linked and respect_circuit_breaker is enabled,
        the processor will check the circuit breaker state before processing.
        If tripped, it waits for the circuit breaker to recover before retrying.

        Args:
            circuit_breaker: CircuitBreaker instance to coordinate with.
        """
        self._circuit_breaker = circuit_breaker
        logger.debug("Retry processor: linked to circuit breaker")

    def set_cookie_rotator(self, cookie_rotator: 'CookieRotator') -> None:
        """Link a cookie rotator to coordinate retry timing with cookie cooldowns.

        When a cookie rotator is linked and wait_for_cookie_cooldown is enabled,
        the processor will check if any cookies are in cooldown before processing.
        If all cookies are in cooldown, it waits for the shortest cooldown to expire.

        Args:
            cookie_rotator: CookieRotator instance to coordinate with.
        """
        self._cookie_rotator = cookie_rotator
        logger.debug("Retry processor: linked to cookie rotator")

    def set_retry_budget(self, retry_budget: 'DownloadRetryBudget') -> None:
        """Link a retry budget tracker for per-video retry limits.

        When a retry budget is linked, the processor will check if a video
        has exhausted its retry budget before attempting to retry it.
        Videos with exhausted budgets are skipped instead of retried.

        Args:
            retry_budget: DownloadRetryBudget instance to coordinate with.
        """
        self._retry_budget = retry_budget
        logger.debug("Retry processor: linked to retry budget")

    def check_budget_exhausted(self, video_id: str) -> bool:
        """Check if a video has exhausted its retry budget.

        Args:
            video_id: YouTube video ID to check.

        Returns:
            True if budget exhausted, False otherwise.
        """
        if not self._retry_budget:
            return False
        return self._retry_budget.is_exhausted(video_id)

    def record_retry_attempt(self, video_id: str, backoff_seconds: float = None, error_category: str = "unknown") -> None:
        """Record a retry attempt for budget tracking.

        US-144-003: Now accepts error_category for category-aware backoff calculation.

        Args:
            video_id: YouTube video ID
            backoff_seconds: Backoff time used for this attempt. If None, calculates
                category-aware exponential backoff.
            error_category: Error category for category-aware backoff (e.g., "rate_limit",
                "network", "format", etc.)
        """
        if self._retry_budget:
            # US-144-003: Pass error_category for category-aware backoff
            self._retry_budget.record_attempt(video_id, backoff_seconds, error_category)

    def reset_budget(self, video_id: str) -> None:
        """Reset budget for a video after successful download.

        Args:
            video_id: YouTube video ID
        """
        if self._retry_budget:
            self._retry_budget.reset(video_id)

    def set_budget_state(self, budget_summary: Dict) -> None:
        """Store a snapshot of the rate limit budget state.

        Called before retry processing so the processor knows the
        remaining budget when deciding whether to retry items.

        Args:
            budget_summary: Dict from RateLimitBudget.get_summary()
        """
        self._budget_state = budget_summary
        logger.debug(
            f"Retry processor: budget state updated — "
            f"exhausted={budget_summary.get('is_exhausted', False)}, "
            f"backoff_remaining={budget_summary.get('backoff_time_remaining', 'N/A')}s"
        )

    def get_budget_state(self) -> Optional[Dict]:
        """Get the stored budget state snapshot.

        Returns:
            Budget summary dict, or None if not set.
        """
        return self._budget_state

    def get_max_severity(self) -> str:
        """Get the maximum severity level from all queued items.

        Used to scale retry delays based on the most severe error in the queue.
        Severity order: low < medium < high

        Returns:
            Maximum severity level ('low', 'medium', or 'high').
            Defaults to 'medium' if queue is empty.
        """
        if not self._queue.items:
            return 'medium'

        severity_order = {'low': 0, 'medium': 1, 'high': 2}
        max_severity = 'low'

        for item in self._queue.items.values():
            item_severity = getattr(item, 'severity', 'medium')
            if severity_order.get(item_severity, 1) > severity_order.get(max_severity, 0):
                max_severity = item_severity

        return max_severity

    def get_severity_multiplier(self) -> float:
        """Get the delay multiplier based on max severity in queue.

        Returns:
            Multiplier: 1.5 (low), 2.0 (medium), or 3.0 (high).
        """
        severity = self.get_max_severity()
        return SEVERITY_MULTIPLIERS.get(severity, 2.0)

    def _apply_jitter(self, delay: float) -> float:
        """Apply random jitter to a delay value.

        Jitter helps prevent thundering herd when multiple downloads
        resume simultaneously after retry delay.

        The jitter formula is: delay * (1 + random.uniform(-jitter, +jitter))

        For example, with jitter_factor=0.2 and delay=120s:
        - Minimum: 120 * (1 - 0.2) = 96s
        - Maximum: 120 * (1 + 0.2) = 144s

        The result is capped at max_combined_wait_seconds to prevent runaway delays.

        Args:
            delay: Base delay in seconds.

        Returns:
            Jittered delay, capped at max_combined_wait_seconds.
        """
        jitter_factor = getattr(self.config, 'jitter_factor', 0.2)

        # Clamp jitter_factor to valid range [0.0, 1.0]
        if jitter_factor < 0.0:
            jitter_factor = 0.0
        elif jitter_factor > 1.0:
            jitter_factor = 1.0

        # Apply jitter
        if jitter_factor > 0.0:
            jitter_multiplier = 1 + random.uniform(-jitter_factor, jitter_factor)
            jittered_delay = delay * jitter_multiplier
            self._last_jitter_applied = jitter_multiplier - 1.0
        else:
            jittered_delay = delay
            self._last_jitter_applied = 0.0

        # Cap at max_combined_wait_seconds
        max_wait = getattr(self.config, 'max_combined_wait_seconds', 300.0)
        if jittered_delay > max_wait:
            logger.debug(
                f"Retry processor jittered delay capped: {jittered_delay:.1f}s -> {max_wait:.0f}s"
            )
            jittered_delay = max_wait

        return jittered_delay

    def _get_cookie_cooldown_remaining(self) -> float:
        """Get remaining cookie cooldown time without waiting.

        Returns:
            Remaining seconds until shortest cookie cooldown expires,
            or 0.0 if cookies are available or cooldown check disabled.
        """
        if not self._cookie_rotator:
            return 0.0

        if not self.config.wait_for_cookie_cooldown:
            return 0.0

        if not self._cookie_rotator.is_enabled:
            return 0.0

        # Check if any cookies are available
        if self._cookie_rotator.available_cookies > 0:
            return 0.0

        # All cookies are in cooldown - find the shortest remaining cooldown
        if not self._cookie_rotator._failed_cookies:
            return 0.0

        cooldown_seconds = self._cookie_rotator.config.cooldown_seconds
        now = time.time()

        min_remaining = float('inf')
        for cookie_path, failed_time in self._cookie_rotator._failed_cookies.items():
            elapsed = now - failed_time
            remaining = cooldown_seconds - elapsed
            if remaining > 0 and remaining < min_remaining:
                min_remaining = remaining

        if min_remaining == float('inf') or min_remaining <= 0:
            return 0.0

        return min_remaining

    def _get_cb_remaining(self) -> float:
        """Get remaining circuit breaker pause time without waiting.

        Returns:
            Remaining seconds until circuit breaker recovers,
            or 0.0 if CB is not tripped or check disabled.
        """
        if not self._circuit_breaker:
            return 0.0

        if not self.config.respect_circuit_breaker:
            return 0.0

        if not self._circuit_breaker.is_enabled:
            return 0.0

        if not self._circuit_breaker.is_open:
            return 0.0

        elapsed = time.time() - self._circuit_breaker.state.opened_at
        remaining = self._circuit_breaker.config.pause_seconds - elapsed

        return max(0.0, remaining)

    def _wait_for_cookie_cooldown(self) -> float:
        """Wait for cookie cooldown to expire if all cookies are unavailable.

        Checks if all cookies are in cooldown. If so, calculates the shortest
        remaining cooldown time and waits for it to expire.

        Returns:
            The number of seconds waited (0 if cookies were available).
        """
        remaining = self._get_cookie_cooldown_remaining()
        if remaining <= 0:
            return 0.0

        # Log and wait for cookie cooldown to expire
        logger.info(
            f"Batch retry: waiting {remaining:.1f}s for cookie cooldown to expire "
            f"before processing retry queue"
        )
        time.sleep(remaining)
        self._cookie_cooldown_wait_time += remaining

        return remaining

    def _wait_for_circuit_breaker(self) -> float:
        """Wait for circuit breaker to recover if tripped.

        Checks if the circuit breaker is open (tripped) and if so, waits for
        the remaining pause duration before returning.

        Returns:
            The number of seconds waited (0 if circuit breaker was not tripped).
        """
        remaining = self._get_cb_remaining()
        if remaining <= 0:
            return 0.0

        # Log and wait for circuit breaker to recover
        logger.info(
            f"Batch retry: waiting {remaining:.1f}s for circuit breaker to recover "
            f"before processing retry queue"
        )
        time.sleep(remaining)
        self._circuit_breaker_wait_time += remaining

        return remaining

    def _wait_combined(self) -> float:
        """Wait for both circuit breaker and cookie cooldown using combined strategy.

        Instead of waiting for CB and cooldown sequentially (which can deadlock),
        uses min(cb_remaining, cooldown_remaining) + small buffer. If both are
        blocking and the total wait would exceed max_combined_wait_seconds,
        forces a retry with the best-available cookie method.

        Sets self._forced_retry = True if the max combined wait was exceeded.

        Returns:
            Total seconds waited.
        """
        cb_remaining = self._get_cb_remaining()
        cooldown_remaining = self._get_cookie_cooldown_remaining()

        # Neither blocking — no wait needed
        if cb_remaining <= 0 and cooldown_remaining <= 0:
            self._forced_retry = False
            return 0.0

        # Only one is blocking — wait for it normally
        if cb_remaining > 0 and cooldown_remaining <= 0:
            self._forced_retry = False
            return self._wait_for_circuit_breaker()

        if cooldown_remaining > 0 and cb_remaining <= 0:
            self._forced_retry = False
            return self._wait_for_cookie_cooldown()

        # Both are blocking — potential deadlock scenario
        # Use min(cb_remaining, cooldown_remaining) + 5s buffer
        combined_estimate = min(cb_remaining, cooldown_remaining) + 5.0
        max_wait = self.config.max_combined_wait_seconds

        if combined_estimate > max_wait:
            # Deadlock detected — force retry with best-available cookie
            logger.warning(
                f"Retry processor waited {combined_estimate:.0f}s for CB+cooldown "
                f"— forcing retry with best-available cookie method "
                f"(max_combined_wait={max_wait:.0f}s, "
                f"cb_remaining={cb_remaining:.1f}s, "
                f"cooldown_remaining={cooldown_remaining:.1f}s)"
            )
            self._forced_retry = True
            return 0.0

        # Wait for the shorter of the two blockers + buffer
        wait_time = combined_estimate
        logger.info(
            f"Batch retry: CB and cookie cooldown both active — "
            f"waiting {wait_time:.1f}s "
            f"(min of CB {cb_remaining:.1f}s / cooldown {cooldown_remaining:.1f}s + 5s buffer)"
        )
        time.sleep(wait_time)
        self._circuit_breaker_wait_time += min(cb_remaining, wait_time)
        self._cookie_cooldown_wait_time += max(0.0, wait_time - cb_remaining)
        self._forced_retry = False

        return wait_time

    def has_pending(self) -> bool:
        """Check if there are items waiting to be retried."""
        return self._queue.has_pending()

    def start_retry_pass(self) -> int:
        """Start a new retry pass.

        Increments the pass counter, checks circuit breaker state, checks cookie
        cooldowns, applies the delay, and logs the start. Should be called before
        processing items in the queue.

        When both circuit breaker AND cookie cooldown are active simultaneously,
        uses a combined wait strategy: wait for min(cb, cooldown) + buffer instead
        of waiting for both sequentially. If the combined wait would exceed
        max_combined_wait_seconds, forces retry with best-available cookie method.

        The circuit breaker and cookie cooldown wait times are additional to the
        retry delay (not subtracted from it).

        Returns:
            The new pass number (1-indexed).
        """
        if not self.config.enabled or not self._queue.items:
            return 0

        self._queue.current_pass += 1

        # Use combined wait strategy to avoid deadlock when both CB and
        # cookie cooldown are active simultaneously
        combined_wait = self._wait_combined()

        # Calculate effective delay with progressive/exponential backoff, severity scaling, and jitter
        # US-114-012: Progressive retry delay - exponential backoff with max cap
        # Formula: delay = min(initial_delay * (multiplier ^ (pass - 1)), max_delay)
        current_pass = self._queue.current_pass
        initial_delay = getattr(self.config, 'initial_delay_seconds', 1.0)
        max_delay = getattr(self.config, 'max_delay_seconds', 60.0)
        backoff_multiplier = getattr(self.config, 'backoff_multiplier', 2.0)

        # Calculate exponential backoff: initial * (multiplier ^ (pass-1))
        progressive_delay = initial_delay * (backoff_multiplier ** (current_pass - 1))
        # Cap at max_delay_seconds
        base_delay = min(progressive_delay, max_delay)

        # Apply severity multiplier: low=1.5x, medium=2.0x, high=3.0x
        severity_multiplier = self.get_severity_multiplier()
        max_severity = self.get_max_severity()
        scaled_delay = base_delay * severity_multiplier
        # Apply jitter
        effective_delay = self._apply_jitter(scaled_delay)
        jitter_pct = abs(self._last_jitter_applied) * 100

        # Build wait info message
        wait_parts = []
        if combined_wait > 0:
            cb_rem = self._get_cb_remaining()
            cookie_rem = self._get_cookie_cooldown_remaining()
            if cb_rem > 0 or self._circuit_breaker_wait_time > 0:
                wait_parts.append(f"circuit breaker: {self._circuit_breaker_wait_time:.1f}s")
            if cookie_rem > 0 or self._cookie_cooldown_wait_time > 0:
                wait_parts.append(f"cookie cooldown: {self._cookie_cooldown_wait_time:.1f}s")
            if not wait_parts:
                wait_parts.append(f"combined: {combined_wait:.1f}s")

        # Log at INFO level so users can see the retry happening
        if self._forced_retry:
            logger.info(
                f"Batch retry pass {self._queue.current_pass}/{self.config.max_passes}: "
                f"{len(self._queue.items)} videos queued. "
                f"FORCED — skipping wait (CB+cooldown deadlock exceeded "
                f"{self.config.max_combined_wait_seconds:.0f}s cap). "
                f"Retrying with best-available cookie method..."
            )
        elif wait_parts:
            logger.info(
                f"Batch retry pass {self._queue.current_pass}/{self.config.max_passes}: "
                f"{len(self._queue.items)} videos queued (severity={max_severity}, {severity_multiplier}x). "
                f"Waited for {', '.join(wait_parts)}, "
                f"additional delay: {effective_delay:.0f}s (jitter={jitter_pct:.0f}%)..."
            )
        else:
            logger.info(
                f"Batch retry pass {self._queue.current_pass}/{self.config.max_passes}: "
                f"{len(self._queue.items)} videos queued (severity={max_severity}, {severity_multiplier}x). "
                f"Waiting {effective_delay:.0f}s before retry (jitter={jitter_pct:.0f}%)..."
            )

        time.sleep(effective_delay)

        logger.info(
            f"Batch retry pass {self._queue.current_pass}: "
            f"starting retry of {len(self._queue.items)} videos"
        )

        return self._queue.current_pass

    def finish_retry_pass(self) -> None:
        """Finish the current retry pass.

        Moves videos that have exhausted all retry passes to permanently
        failed. Should be called after processing all items in a pass.
        """
        self._queue.finish_retry_pass()

    def clear(self) -> None:
        """Clear all processor state for a new session."""
        self._circuit_breaker_wait_time = 0.0
        self._cookie_cooldown_wait_time = 0.0
        self._forced_retry = False
        self._budget_state = None
        # US-136-011: Reset retry metrics
        self._retry_attempts = 0
        self._retry_successes = 0
        self._retry_failures = 0
        self._pass_retry_counts.clear()
        logger.debug("Retry processor: cleared for new session")

    def get_processor_stats(self) -> dict:
        """Get processor-specific statistics for reporting.

        Returns:
            Dict with processor stats including wait times, forced retry status, and retry metrics.
        """
        # Calculate success rate
        total_attempts = self._retry_attempts
        success_rate = 0.0
        if total_attempts > 0:
            success_rate = (self._retry_successes / total_attempts) * 100

        return {
            'circuit_breaker_wait_time': round(self._circuit_breaker_wait_time, 1),
            'cookie_cooldown_wait_time': round(self._cookie_cooldown_wait_time, 1),
            'forced_retry': self._forced_retry,
            'budget_state': self._budget_state,
            # US-136-011: Retry metrics
            'retry_attempts': self._retry_attempts,
            'retry_successes': self._retry_successes,
            'retry_failures': self._retry_failures,
            'retry_success_rate': round(success_rate, 1),
            'pass_retry_counts': dict(self._pass_retry_counts),
        }

    def record_retry_attempt(self) -> None:
        """Record a retry attempt (called when processing retry items).

        US-136-011: Tracks total retry attempts for metrics.
        """
        self._retry_attempts += 1
        pass_num = self._queue.current_pass
        self._pass_retry_counts[pass_num] = self._pass_retry_counts.get(pass_num, 0) + 1
        logger.debug(f"Retry metrics: recorded attempt (total={self._retry_attempts}, pass={pass_num})")

    def record_retry_success(self) -> None:
        """Record a successful retry (video downloaded successfully).

        US-136-011: Tracks successful retries for success rate metrics.
        """
        self._retry_successes += 1
        logger.debug(f"Retry metrics: recorded success (total={self._retry_successes})")

    def record_retry_failure(self) -> None:
        """Record a failed retry (video still failing after retry).

        US-136-011: Tracks failed retries for success rate metrics.
        """
        self._retry_failures += 1
        logger.debug(f"Retry metrics: recorded failure (total={self._retry_failures})")

    def to_checkpoint_dict(self) -> dict:
        """Serialize processor state to dictionary for checkpoint persistence.

        Returns:
            Dict that can be saved to checkpoint JSON.
        """
        result = {
            'circuit_breaker_wait_time': self._circuit_breaker_wait_time,
            'cookie_cooldown_wait_time': self._cookie_cooldown_wait_time,
        }

        # Include retry budget state if available
        if self._retry_budget:
            result['retry_budget'] = self._retry_budget.to_checkpoint_dict()

        return result

    def from_checkpoint_dict(self, data: dict) -> None:
        """Restore processor state from checkpoint dictionary.

        Args:
            data: Dict from checkpoint JSON.
        """
        if not data:
            return

        self._circuit_breaker_wait_time = data.get('circuit_breaker_wait_time', 0.0)
        self._cookie_cooldown_wait_time = data.get('cookie_cooldown_wait_time', 0.0)

        # Restore retry budget state if available
        if self._retry_budget and 'retry_budget' in data:
            self._retry_budget.from_checkpoint_dict(data['retry_budget'])

        if self._circuit_breaker_wait_time > 0 or self._cookie_cooldown_wait_time > 0:
            logger.debug(
                f"Retry processor: restored from checkpoint "
                f"(cb_wait={self._circuit_breaker_wait_time:.1f}s, "
                f"cookie_wait={self._cookie_cooldown_wait_time:.1f}s)"
            )
