"""Batch-level retry queue for rate-limited videos.

When rate limiting affects a batch, failed videos are collected and retried
together after a delay. This provides better recovery than individual retries
by allowing YouTube's rate limit window to pass before attempting the batch.

Key concepts:
- Failed videos are queued with their keyword and tier context
- After batch completes, the queue is processed with configurable delay
- Maximum 2 retry passes per download session (configurable)
- Queue is cleared on session start or when all retries complete

Architecture (Single Responsibility Principle):
- RetryQueue: Pure data structure (add/get/clear items, track state)
- RetryQueueProcessor: Execution logic (delay, circuit breaker wait, cookie cooldown)

For processing, use RetryQueueProcessor which wraps a RetryQueue instance.
RetryQueue still exposes processing methods for backward compatibility, but
they delegate to an internal processor instance.

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

from .retry_stats import RetryQueueStats

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreaker
    from .cookie_rotator import CookieRotator
    from .rate_limit_budget import RateLimitBudget
    from .retry_processor import RetryQueueProcessor

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

    # Maximum combined wait time (seconds) when both circuit breaker and cookie
    # cooldown are blocking simultaneously. If exceeded, force-process the retry
    # queue with the current best-available cookie method instead of waiting
    # for both to clear. Prevents deadlock when CB and cooldown overlap.
    max_combined_wait_seconds: float = 300.0

    # Jitter factor for randomizing delay durations (0.0 to 1.0)
    # Delay is computed as: base_delay * (1 + random.uniform(-jitter, +jitter))
    # Default 0.2 means ±20% randomization to prevent thundering herd
    jitter_factor: float = 0.2


@dataclass
class RetryItem:
    """A single video waiting in the retry queue."""
    video_id: str
    keyword: str
    tier: str
    error_message: str
    retry_count: int = 0
    added_at: float = field(default_factory=time.time)
    severity: str = 'medium'  # low, medium, high - determines delay multiplier
    error_category: str = 'video_specific'  # 'network_systemic' or 'video_specific'


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

        # Processor handles execution logic (delay, CB wait, cookie cooldown)
        # Lazily created to avoid circular imports
        self._processor: Optional['RetryQueueProcessor'] = None

        # Initialize stats tracker with config values
        self._stats = RetryQueueStats(
            enabled=self.config.enabled,
            max_passes=self.config.max_passes,
            delay_seconds=self.config.delay_seconds,
            respect_circuit_breaker=self.config.respect_circuit_breaker,
            wait_for_cookie_cooldown=self.config.wait_for_cookie_cooldown,
            max_combined_wait_seconds=self.config.max_combined_wait_seconds,
        )

    @property
    def processor(self) -> 'RetryQueueProcessor':
        """Get the processor instance, creating it lazily if needed."""
        if self._processor is None:
            from .retry_processor import RetryQueueProcessor
            self._processor = RetryQueueProcessor(self)
        return self._processor

    # =========================================================================
    # Backward compatibility properties - delegate to processor
    # =========================================================================

    @property
    def _circuit_breaker(self) -> Optional['CircuitBreaker']:
        """Get circuit breaker from processor (backward compat)."""
        return self.processor._circuit_breaker

    @_circuit_breaker.setter
    def _circuit_breaker(self, value: Optional['CircuitBreaker']) -> None:
        """Set circuit breaker on processor (backward compat)."""
        self.processor._circuit_breaker = value

    @property
    def _cookie_rotator(self) -> Optional['CookieRotator']:
        """Get cookie rotator from processor (backward compat)."""
        return self.processor._cookie_rotator

    @_cookie_rotator.setter
    def _cookie_rotator(self, value: Optional['CookieRotator']) -> None:
        """Set cookie rotator on processor (backward compat)."""
        self.processor._cookie_rotator = value

    @property
    def _circuit_breaker_wait_time(self) -> float:
        """Get CB wait time from processor (backward compat)."""
        return self.processor._circuit_breaker_wait_time

    @_circuit_breaker_wait_time.setter
    def _circuit_breaker_wait_time(self, value: float) -> None:
        """Set CB wait time on processor (backward compat)."""
        self.processor._circuit_breaker_wait_time = value

    @property
    def _cookie_cooldown_wait_time(self) -> float:
        """Get cookie cooldown wait time from processor (backward compat)."""
        return self.processor._cookie_cooldown_wait_time

    @_cookie_cooldown_wait_time.setter
    def _cookie_cooldown_wait_time(self, value: float) -> None:
        """Set cookie cooldown wait time on processor (backward compat)."""
        self.processor._cookie_cooldown_wait_time = value

    @property
    def _budget_state(self) -> Optional[Dict]:
        """Get budget state from processor (backward compat)."""
        return self.processor._budget_state

    @_budget_state.setter
    def _budget_state(self, value: Optional[Dict]) -> None:
        """Set budget state on processor (backward compat)."""
        self.processor._budget_state = value

    @property
    def _forced_retry(self) -> bool:
        """Get forced retry flag from processor (backward compat)."""
        return self.processor._forced_retry

    @_forced_retry.setter
    def _forced_retry(self, value: bool) -> None:
        """Set forced retry flag on processor (backward compat)."""
        self.processor._forced_retry = value

    @property
    def _last_jitter_applied(self) -> float:
        """Get last jitter from processor (backward compat)."""
        return self.processor._last_jitter_applied

    @_last_jitter_applied.setter
    def _last_jitter_applied(self, value: float) -> None:
        """Set last jitter on processor (backward compat)."""
        self.processor._last_jitter_applied = value

    # =========================================================================
    # Processing methods - delegate to processor
    # =========================================================================

    def set_circuit_breaker(self, circuit_breaker: 'CircuitBreaker') -> None:
        """Link a circuit breaker to coordinate retry timing.

        When a circuit breaker is linked and respect_circuit_breaker is enabled,
        the retry queue will check the circuit breaker state before processing.
        If tripped, it waits for the circuit breaker to recover before retrying.

        Args:
            circuit_breaker: CircuitBreaker instance to coordinate with.
        """
        self.processor.set_circuit_breaker(circuit_breaker)
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
        self.processor.set_cookie_rotator(cookie_rotator)
        logger.debug("Retry queue: linked to cookie rotator")

    def set_budget_state(self, budget_summary: Dict) -> None:
        """Store a snapshot of the rate limit budget state.

        Called before retry processing so the retry queue knows the
        remaining budget when deciding whether to retry items.

        Args:
            budget_summary: Dict from RateLimitBudget.get_summary()
        """
        self.processor.set_budget_state(budget_summary)

    def get_budget_state(self) -> Optional[Dict]:
        """Get the stored budget state snapshot.

        Returns:
            Budget summary dict, or None if not set.
        """
        return self.processor.get_budget_state()

    def _apply_jitter(self, delay: float) -> float:
        """Apply random jitter to a delay value. Delegates to processor."""
        return self.processor._apply_jitter(delay)

    def _get_cookie_cooldown_remaining(self) -> float:
        """Get remaining cookie cooldown time. Delegates to processor."""
        return self.processor._get_cookie_cooldown_remaining()

    def _get_cb_remaining(self) -> float:
        """Get remaining circuit breaker pause time. Delegates to processor."""
        return self.processor._get_cb_remaining()

    def _wait_for_cookie_cooldown(self) -> float:
        """Wait for cookie cooldown to expire. Delegates to processor."""
        return self.processor._wait_for_cookie_cooldown()

    def _wait_for_circuit_breaker(self) -> float:
        """Wait for circuit breaker to recover. Delegates to processor."""
        return self.processor._wait_for_circuit_breaker()

    def _wait_combined(self) -> float:
        """Wait for both circuit breaker and cookie cooldown. Delegates to processor."""
        return self.processor._wait_combined()

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
        error_message: str,
        error_category: str = 'video_specific'
    ) -> bool:
        """Add a failed video to the retry queue.

        Args:
            video_id: YouTube video ID
            keyword: Search keyword that found this video
            tier: Duration tier (short, medium, long, longer)
            error_message: Error message from the failure
            error_category: 'network_systemic' or 'video_specific'.
                Network-systemic errors (DNS, no connectivity) affect all
                segments and should not be retried. Video-specific errors
                (403, unavailable) may succeed on retry with escalation.

        Returns:
            True if added to queue, False if disabled or already in queue.
        """
        if not self.config.enabled:
            return False

        # Import here to avoid circular import (core.py imports retry_queue.py)
        from .core import classify_error_severity

        # Check if already in queue
        if video_id in self.items:
            # Update error message, severity, and category but don't re-add
            self.items[video_id].error_message = error_message
            self.items[video_id].severity = classify_error_severity(error_message)
            self.items[video_id].error_category = error_category
            logger.debug(f"Retry queue: {video_id} already queued, updated error")
            return False

        # Check if already completed or permanently failed
        if video_id in self._completed_ids or video_id in self._failed_ids:
            logger.debug(f"Retry queue: {video_id} already processed, skipping")
            return False

        # Classify error severity for adaptive delay scaling
        severity = classify_error_severity(error_message)

        self.items[video_id] = RetryItem(
            video_id=video_id,
            keyword=keyword,
            tier=tier,
            error_message=error_message,
            retry_count=0,
            severity=severity,
            error_category=error_category,
        )
        self._total_added += 1
        self._stats.record_failure(video_id, error_message)

        logger.debug(
            f"Retry queue: added {video_id} ({keyword}/{tier}) severity={severity} "
            f"category={error_category} - queue size now {len(self.items)}"
        )
        return True

    def get_pending_items(self) -> List[RetryItem]:
        """Get all items waiting to be retried.

        Returns:
            List of RetryItem objects in the queue.
        """
        return list(self.items.values())

    def get_retryable_items(self) -> List[RetryItem]:
        """Get items that are worth retrying (excludes network_systemic errors).

        Network-systemic errors (DNS failure, no connectivity) affect all
        segments and won't resolve by retrying individual items. Only
        video-specific errors (403, removed) may succeed with escalation.

        Returns:
            List of RetryItem objects with error_category != 'network_systemic'.
        """
        return [
            item for item in self.items.values()
            if item.error_category != 'network_systemic'
        ]

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
            self._stats.clear_failure(video_id)
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

    @property
    def forced_retry(self) -> bool:
        """True if the last retry pass was forced due to combined wait timeout."""
        return self._forced_retry

    def start_retry_pass(self) -> int:
        """Start a new retry pass. Delegates to processor for execution logic.

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
        return self.processor.start_retry_pass()

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
        self.processor.clear()  # Clear processor state (wait times, forced_retry)
        self._stats.reset()
        logger.debug("Retry queue: cleared for new session")

    def get_stats(self) -> dict:
        """Get retry queue statistics for reporting.

        Delegates to RetryQueueStats.get_summary() after syncing current state.

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
        self._sync_stats()
        return self._stats.get_summary()

    def get_failure_reasons(self) -> Dict[str, str]:
        """Get mapping of failed video IDs to their error messages.

        Delegates to RetryQueueStats.get_failure_reasons().

        Returns:
            Dict mapping video_id to the last error message received.
        """
        self._sync_stats()
        return self._stats.get_failure_reasons()

    def get_retry_metrics(self) -> Dict:
        """Calculate retry-specific metrics for analysis.

        Delegates to RetryQueueStats.get_retry_metrics().

        Returns:
            Dict with metrics including success_rate, retry_efficiency, etc.
        """
        self._sync_stats()
        return self._stats.get_retry_metrics()

    def _sync_stats(self) -> None:
        """Sync internal state to RetryQueueStats for accurate reporting."""
        self._stats.pending = len(self.items)
        self._stats.completed_ids = self._completed_ids.copy()
        self._stats.failed_ids = self._failed_ids.copy()
        self._stats.current_pass = self.current_pass
        self._stats.total_added = self._total_added
        self._stats.total_retried = self._total_retried
        self._stats.circuit_breaker_wait_time = self._circuit_breaker_wait_time
        self._stats.cookie_cooldown_wait_time = self._cookie_cooldown_wait_time
        self._stats.forced_retry = self._forced_retry
        self._stats.budget_state = self._budget_state

    def to_checkpoint_dict(self) -> dict:
        """Serialize state to dictionary for checkpoint persistence.

        Returns:
            Dict that can be saved to checkpoint JSON.
        """
        checkpoint = {
            'items': [
                {
                    'video_id': item.video_id,
                    'keyword': item.keyword,
                    'tier': item.tier,
                    'error_message': item.error_message,
                    'retry_count': item.retry_count,
                    'error_category': item.error_category,
                }
                for item in self.items.values()
            ],
            'current_pass': self.current_pass,
            'completed_ids': list(self._completed_ids),
            'failed_ids': list(self._failed_ids),
            'total_added': self._total_added,
            'total_retried': self._total_retried,
        }
        # Merge processor checkpoint data
        checkpoint.update(self.processor.to_checkpoint_dict())
        return checkpoint

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
                    error_category=item_data.get('error_category', 'video_specific'),
                )

        # Restore state
        self.current_pass = data.get('current_pass', 0)
        self._completed_ids = set(data.get('completed_ids', []))
        self._failed_ids = set(data.get('failed_ids', []))
        self._total_added = data.get('total_added', 0)
        self._total_retried = data.get('total_retried', 0)

        # Restore processor state
        self.processor.from_checkpoint_dict(data)

        if self.items:
            logger.debug(
                f"Retry queue: restored {len(self.items)} items from checkpoint "
                f"(pass {self.current_pass}/{self.config.max_passes})"
            )
