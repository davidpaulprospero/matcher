"""Batch-level retry queue for rate-limited videos.

When rate limiting affects a batch, failed videos are collected and retried
together after a delay. This provides better recovery than individual retries
by allowing YouTube's rate limit window to pass before attempting the batch.

Key concepts:
- Failed videos are queued with their keyword and tier context
- After batch completes, the queue is processed with configurable delay
- Maximum 2 retry passes per download session (configurable)
- Queue is cleared on session start or when all retries complete

Circuit breaker coordination:
- When a circuit breaker is linked, the retry queue checks its state before processing
- If the circuit breaker is tripped, waits for it to recover before starting retries
- This prevents wasting retry attempts during active rate limit periods
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreaker
    from .cookie_rotator import CookieRotator
    from .rate_limit_budget import RateLimitBudget

logger = logging.getLogger(__name__)


@dataclass
class BatchRetryConfig:
    """Configuration for batch-level retry queue.

    When rate limiting affects multiple videos in a batch, collect them
    and retry the entire batch after a delay. This is more effective than
    individual retries because it allows the rate limit window to pass.

    Example with defaults:
      - Video fails due to rate limit → added to retry queue
      - After batch completes, wait 120s
      - Retry all queued videos together (pass 1)
      - If still failing, wait and retry again (pass 2)
      - After 2 passes, give up on remaining failures

    Circuit breaker coordination:
      When respect_circuit_breaker is enabled (default), the batch retry queue
      will check the circuit breaker state before processing. If the circuit
      breaker is tripped, the retry queue will wait for it to recover before
      retrying. This prevents retries from being wasted during active rate limits.
    """
    # Enable/disable batch retry queue
    enabled: bool = True

    # Delay before processing retry queue (seconds)
    # Should be long enough for rate limit window to pass
    delay_seconds: float = 120.0

    # Maximum retry passes per download session
    # After this many batch retries, give up on remaining failures
    max_passes: int = 2

    # Respect circuit breaker state when processing retries
    # If True, wait for circuit breaker to recover before retrying
    # If False, retry immediately after delay_seconds regardless of circuit breaker
    respect_circuit_breaker: bool = True

    # Wait for cookie cooldown before processing retries
    # If True, check if any cookies are in cooldown and extend delay if needed
    # If False, proceed with retry even if cookies are in cooldown
    wait_for_cookie_cooldown: bool = True


@dataclass
class RetryItem:
    """A single video waiting in the retry queue."""
    video_id: str
    keyword: str
    tier: str
    error_message: str
    retry_count: int = 0
    added_at: float = field(default_factory=time.time)


class RetryQueue:
    """Batch-level retry queue for rate-limited videos.

    Collects videos that fail due to rate limiting and retries them
    together after a delay. This provides better recovery than
    individual retries by:

    1. Batching failures together for efficient retry
    2. Applying a delay that lets rate limit windows pass
    3. Limiting total retry passes to avoid infinite loops

    Usage:
        queue = RetryQueue(config)

        # During download batch
        if rate_limit_error:
            queue.add(video_id, keyword, tier, error_message)

        # After batch completes
        if queue.has_pending():
            queue.process_retry_pass(download_func)

    Attributes:
        config: BatchRetryConfig with delay and pass limits
        items: Dict mapping video_id to RetryItem
        current_pass: Current retry pass number (1-indexed)
        stats: Statistics for reporting
    """

    def __init__(self, config: Optional[BatchRetryConfig] = None):
        """Initialize retry queue with configuration.

        Args:
            config: BatchRetryConfig. If None, uses defaults.
        """
        self.config = config or BatchRetryConfig()
        self.items: Dict[str, RetryItem] = {}
        self.current_pass: int = 0
        self._completed_ids: Set[str] = set()  # Successfully retried
        self._failed_ids: Set[str] = set()  # Permanently failed after max retries
        self._total_added: int = 0
        self._total_retried: int = 0
        self._circuit_breaker: Optional['CircuitBreaker'] = None
        self._circuit_breaker_wait_time: float = 0.0  # Total time spent waiting for circuit breaker
        self._cookie_rotator: Optional['CookieRotator'] = None
        self._cookie_cooldown_wait_time: float = 0.0  # Total time spent waiting for cookie cooldown
        self._budget_state: Optional[Dict] = None  # Budget snapshot when items were queued

    def set_circuit_breaker(self, circuit_breaker: 'CircuitBreaker') -> None:
        """Link a circuit breaker to coordinate retry timing.

        When a circuit breaker is linked and respect_circuit_breaker is enabled,
        the retry queue will check the circuit breaker state before processing.
        If tripped, it waits for the circuit breaker to recover before retrying.

        Args:
            circuit_breaker: CircuitBreaker instance to coordinate with.
        """
        self._circuit_breaker = circuit_breaker
        logger.debug("Retry queue: linked to circuit breaker")

    def set_cookie_rotator(self, cookie_rotator: 'CookieRotator') -> None:
        """Link a cookie rotator to coordinate retry timing with cookie cooldowns.

        When a cookie rotator is linked and wait_for_cookie_cooldown is enabled,
        the retry queue will check if any cookies are in cooldown before processing.
        If all cookies are in cooldown, it waits for the shortest cooldown to expire
        before retrying.

        Args:
            cookie_rotator: CookieRotator instance to coordinate with.
        """
        self._cookie_rotator = cookie_rotator
        logger.debug("Retry queue: linked to cookie rotator")

    def set_budget_state(self, budget_summary: Dict) -> None:
        """Store a snapshot of the rate limit budget state.

        Called before retry processing so the retry queue knows the
        remaining budget when deciding whether to retry items.

        Args:
            budget_summary: Dict from RateLimitBudget.get_summary()
        """
        self._budget_state = budget_summary
        logger.debug(
            f"Retry queue: budget state updated — "
            f"exhausted={budget_summary.get('is_exhausted', False)}, "
            f"backoff_remaining={budget_summary.get('backoff_time_remaining', 'N/A')}s"
        )

    def get_budget_state(self) -> Optional[Dict]:
        """Get the stored budget state snapshot.

        Returns:
            Budget summary dict, or None if not set.
        """
        return self._budget_state

    def _wait_for_cookie_cooldown(self) -> float:
        """Wait for cookie cooldown to expire if all cookies are unavailable.

        Checks if all cookies are in cooldown. If so, calculates the shortest
        remaining cooldown time and waits for it to expire.

        Returns:
            The number of seconds waited (0 if cookies were available).
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
        # Access the internal _failed_cookies dict to find cooldown times
        if not self._cookie_rotator._failed_cookies:
            return 0.0

        cooldown_seconds = self._cookie_rotator.config.cooldown_seconds
        now = time.time()

        # Find the cookie with the shortest remaining cooldown
        min_remaining = float('inf')
        for cookie_path, failed_time in self._cookie_rotator._failed_cookies.items():
            elapsed = now - failed_time
            remaining = cooldown_seconds - elapsed
            if remaining > 0 and remaining < min_remaining:
                min_remaining = remaining

        if min_remaining == float('inf') or min_remaining <= 0:
            return 0.0

        # Log and wait for cookie cooldown to expire
        logger.info(
            f"Batch retry: waiting {min_remaining:.1f}s for cookie cooldown to expire "
            f"before processing retry queue"
        )
        time.sleep(min_remaining)
        self._cookie_cooldown_wait_time += min_remaining

        return min_remaining

    def _wait_for_circuit_breaker(self) -> float:
        """Wait for circuit breaker to recover if tripped.

        Checks if the circuit breaker is open (tripped) and if so, waits for
        the remaining pause duration before returning.

        Returns:
            The number of seconds waited (0 if circuit breaker was not tripped).
        """
        if not self._circuit_breaker:
            return 0.0

        if not self.config.respect_circuit_breaker:
            return 0.0

        if not self._circuit_breaker.is_enabled:
            return 0.0

        if not self._circuit_breaker.is_open:
            return 0.0

        # Circuit breaker is tripped - calculate remaining wait time
        elapsed = time.time() - self._circuit_breaker.state.opened_at
        remaining = self._circuit_breaker.config.pause_seconds - elapsed

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

    @property
    def is_enabled(self) -> bool:
        """Check if batch retry is enabled."""
        return self.config.enabled

    @property
    def can_retry(self) -> bool:
        """Check if more retry passes are allowed."""
        return self.current_pass < self.config.max_passes

    def has_pending(self) -> bool:
        """Check if there are items waiting to be retried."""
        return len(self.items) > 0 and self.can_retry

    def add(
        self,
        video_id: str,
        keyword: str,
        tier: str,
        error_message: str
    ) -> bool:
        """Add a failed video to the retry queue.

        Args:
            video_id: YouTube video ID
            keyword: Search keyword that found this video
            tier: Duration tier (short, medium, long, longer)
            error_message: Error message from the failure

        Returns:
            True if added to queue, False if disabled or already in queue.
        """
        if not self.config.enabled:
            return False

        # Check if already in queue
        if video_id in self.items:
            # Update error message but don't re-add
            self.items[video_id].error_message = error_message
            logger.debug(f"Retry queue: {video_id} already queued, updated error")
            return False

        # Check if already completed or permanently failed
        if video_id in self._completed_ids or video_id in self._failed_ids:
            logger.debug(f"Retry queue: {video_id} already processed, skipping")
            return False

        self.items[video_id] = RetryItem(
            video_id=video_id,
            keyword=keyword,
            tier=tier,
            error_message=error_message,
            retry_count=0
        )
        self._total_added += 1

        logger.debug(
            f"Retry queue: added {video_id} ({keyword}/{tier}) - "
            f"queue size now {len(self.items)}"
        )
        return True

    def get_pending_items(self) -> List[RetryItem]:
        """Get all items waiting to be retried.

        Returns:
            List of RetryItem objects in the queue.
        """
        return list(self.items.values())

    def mark_success(self, video_id: str) -> None:
        """Mark a video as successfully retried.

        Removes from queue and adds to completed set.

        Args:
            video_id: YouTube video ID that succeeded.
        """
        if video_id in self.items:
            del self.items[video_id]
            self._completed_ids.add(video_id)
            self._total_retried += 1
            logger.debug(f"Retry queue: {video_id} succeeded, removed from queue")

    def mark_failed(self, video_id: str) -> None:
        """Mark a video as failed in the current retry pass.

        Increments retry count. If max passes reached after this pass,
        the video will be moved to permanently failed.

        Args:
            video_id: YouTube video ID that failed.
        """
        if video_id in self.items:
            self.items[video_id].retry_count += 1
            logger.debug(
                f"Retry queue: {video_id} failed again "
                f"(attempt {self.items[video_id].retry_count})"
            )

    def start_retry_pass(self) -> int:
        """Start a new retry pass.

        Increments the pass counter, checks circuit breaker state, checks cookie
        cooldowns, applies the delay, and logs the start. Should be called before
        processing items in the queue.

        If a circuit breaker is linked and respect_circuit_breaker is enabled,
        waits for the circuit breaker to recover before applying the retry delay.

        If a cookie rotator is linked and wait_for_cookie_cooldown is enabled,
        waits for cookie cooldowns to expire if all cookies are unavailable.

        The circuit breaker and cookie cooldown wait times are additional to the
        retry delay (not subtracted from it).

        Returns:
            The new pass number (1-indexed).
        """
        if not self.config.enabled or not self.items:
            return 0

        self.current_pass += 1

        # Check circuit breaker before starting retry pass
        # If circuit breaker is tripped, wait for it to recover first
        cb_wait = self._wait_for_circuit_breaker()

        # Check cookie cooldown before starting retry pass
        # If all cookies are in cooldown, wait for the shortest cooldown to expire
        cookie_wait = self._wait_for_cookie_cooldown()

        # Calculate effective delay
        # If circuit breaker was tripped or cookies were in cooldown, we already waited
        # The batch retry delay is additional time to let rate limits clear further
        effective_delay = self.config.delay_seconds

        # Build wait info message
        wait_parts = []
        if cb_wait > 0:
            wait_parts.append(f"circuit breaker: {cb_wait:.1f}s")
        if cookie_wait > 0:
            wait_parts.append(f"cookie cooldown: {cookie_wait:.1f}s")

        # Log at INFO level so users can see the retry happening
        if wait_parts:
            logger.info(
                f"Batch retry pass {self.current_pass}/{self.config.max_passes}: "
                f"{len(self.items)} videos queued. "
                f"Waited for {', '.join(wait_parts)}, "
                f"additional delay: {effective_delay:.0f}s..."
            )
        else:
            logger.info(
                f"Batch retry pass {self.current_pass}/{self.config.max_passes}: "
                f"{len(self.items)} videos queued. "
                f"Waiting {effective_delay:.0f}s before retry..."
            )

        time.sleep(effective_delay)

        logger.info(
            f"Batch retry pass {self.current_pass}: "
            f"starting retry of {len(self.items)} videos"
        )

        return self.current_pass

    def finish_retry_pass(self) -> None:
        """Finish the current retry pass.

        Moves videos that have exhausted all retry passes to permanently
        failed. Should be called after processing all items in a pass.
        """
        # Check if we've exhausted all passes
        if self.current_pass >= self.config.max_passes:
            # Move remaining items to permanently failed
            remaining = list(self.items.keys())
            for video_id in remaining:
                self._failed_ids.add(video_id)
                del self.items[video_id]

            if remaining:
                logger.warning(
                    f"Batch retry: {len(remaining)} videos failed after "
                    f"{self.config.max_passes} retry passes"
                )

    def clear(self) -> None:
        """Clear all state for a new session."""
        self.items.clear()
        self._completed_ids.clear()
        self._failed_ids.clear()
        self.current_pass = 0
        self._total_added = 0
        self._total_retried = 0
        self._circuit_breaker_wait_time = 0.0
        self._cookie_cooldown_wait_time = 0.0
        logger.debug("Retry queue: cleared for new session")

    def get_stats(self) -> dict:
        """Get retry queue statistics for reporting.

        Returns:
            Dict with stats including:
            - enabled: Whether batch retry is enabled
            - pending: Number of items currently queued
            - completed: Number of successfully retried videos
            - failed: Number of permanently failed videos
            - current_pass: Current retry pass number
            - max_passes: Maximum allowed passes
            - total_added: Total videos added to queue this session
            - total_retried: Total successful retries this session
            - respect_circuit_breaker: Whether circuit breaker is respected
            - circuit_breaker_wait_time: Total time spent waiting for circuit breaker
            - wait_for_cookie_cooldown: Whether cookie cooldown is respected
            - cookie_cooldown_wait_time: Total time spent waiting for cookie cooldown
        """
        return {
            'enabled': self.config.enabled,
            'pending': len(self.items),
            'completed': len(self._completed_ids),
            'failed': len(self._failed_ids),
            'current_pass': self.current_pass,
            'max_passes': self.config.max_passes,
            'delay_seconds': self.config.delay_seconds,
            'total_added': self._total_added,
            'total_retried': self._total_retried,
            'respect_circuit_breaker': self.config.respect_circuit_breaker,
            'circuit_breaker_wait_time': round(self._circuit_breaker_wait_time, 1),
            'wait_for_cookie_cooldown': self.config.wait_for_cookie_cooldown,
            'cookie_cooldown_wait_time': round(self._cookie_cooldown_wait_time, 1),
            'budget_state': self._budget_state,
        }

    def to_checkpoint_dict(self) -> dict:
        """Serialize state to dictionary for checkpoint persistence.

        Returns:
            Dict that can be saved to checkpoint JSON.
        """
        return {
            'items': [
                {
                    'video_id': item.video_id,
                    'keyword': item.keyword,
                    'tier': item.tier,
                    'error_message': item.error_message,
                    'retry_count': item.retry_count,
                }
                for item in self.items.values()
            ],
            'current_pass': self.current_pass,
            'completed_ids': list(self._completed_ids),
            'failed_ids': list(self._failed_ids),
            'total_added': self._total_added,
            'total_retried': self._total_retried,
            'circuit_breaker_wait_time': self._circuit_breaker_wait_time,
            'cookie_cooldown_wait_time': self._cookie_cooldown_wait_time,
        }

    def from_checkpoint_dict(self, data: dict) -> None:
        """Restore state from checkpoint dictionary.

        Args:
            data: Dict from checkpoint JSON.
        """
        if not data:
            return

        # Restore items
        self.items.clear()
        for item_data in data.get('items', []):
            video_id = item_data.get('video_id')
            if video_id:
                self.items[video_id] = RetryItem(
                    video_id=video_id,
                    keyword=item_data.get('keyword', ''),
                    tier=item_data.get('tier', 'short'),
                    error_message=item_data.get('error_message', ''),
                    retry_count=item_data.get('retry_count', 0),
                )

        # Restore state
        self.current_pass = data.get('current_pass', 0)
        self._completed_ids = set(data.get('completed_ids', []))
        self._failed_ids = set(data.get('failed_ids', []))
        self._total_added = data.get('total_added', 0)
        self._total_retried = data.get('total_retried', 0)
        self._circuit_breaker_wait_time = data.get('circuit_breaker_wait_time', 0.0)
        self._cookie_cooldown_wait_time = data.get('cookie_cooldown_wait_time', 0.0)

        if self.items:
            logger.debug(
                f"Retry queue: restored {len(self.items)} items from checkpoint "
                f"(pass {self.current_pass}/{self.config.max_passes})"
            )
