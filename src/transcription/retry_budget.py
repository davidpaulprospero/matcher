"""
Batch retry budget management for transcription.

Provides TranscriptionRetryBudget to prevent infinite retry loops when
multiple videos fail transcription in a batch. Tracks total attempts,
failures, and cumulative backoff time across the batch.

Similar pattern to CaptionRetryBudget (src/caption/retry_budget.py) but
simpler since transcription retries are less complex (no format rotation,
VPN escalation, etc.).

Created for US-79-010.

US-137-002: Added per-error-category retry budget tracking.
US-137-011: Added jitter correlation and adaptive backoff strategies.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.logging_templates import log_error_with_context

logger = logging.getLogger(__name__)


class TranscriptionErrorCategory(Enum):
    """Error categories for transcription retry tracking (US-137-002).

    Categories:
        GPU_OOM: GPU out of memory errors
        TRANSIENT: Transient errors (network, temporary unavailability)
        TIMEOUT: Timeout errors
        LANGUAGE_DETECTION: Language detection failures
        PERMANENT: Permanent errors (file not found, invalid format)
        UNKNOWN: Unclassified errors
    """
    GPU_OOM = "gpu_oom"
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    LANGUAGE_DETECTION = "language_detection"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


@dataclass
class TranscriptionRetryBudget:
    """Tracks retry resources across a transcription batch (US-79-010).

    Prevents infinite retry loops by enforcing:
    - Maximum total attempts across all videos
    - Maximum cumulative backoff time

    When any limit is exceeded, is_exhausted() returns True, signaling
    transcribe_videos_parallel() to skip remaining retries.

    Thread Safety:
        All mutation methods are protected by a Lock for concurrent access.

    Attributes:
        attempts: Total transcription attempts across all videos.
        failures: Total failures across all videos.
        successes: Total successes across all videos.
        backoff_time_spent: Cumulative backoff delay (seconds).
        videos_skipped: Count of videos skipped due to budget exhaustion.
        skipped_video_ids: List of video IDs skipped due to budget exhaustion.
        max_attempts: Maximum allowed attempts (0 = unlimited).
        max_backoff_time: Maximum allowed backoff time in seconds (0 = unlimited).
    """

    # Usage counters
    attempts: int = 0
    failures: int = 0
    successes: int = 0
    backoff_time_spent: float = 0.0
    videos_skipped: int = 0
    skipped_video_ids: List[str] = field(default_factory=list)

    # Per-category tracking (US-137-002)
    attempts_by_category: Dict[TranscriptionErrorCategory, int] = field(default_factory=dict)
    failures_by_category: Dict[TranscriptionErrorCategory, int] = field(default_factory=dict)

    # Budget limits (set from config)
    max_attempts: int = 50
    max_backoff_time: float = 180.0  # 3 minutes total backoff budget

    # Thread-safety lock (RLock for reentrant calls from get_summary -> is_exhausted)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def record_attempt(
        self, video_id: str = "", error_category: Optional[TranscriptionErrorCategory] = None
    ) -> None:
        """Record a transcription attempt.

        Args:
            video_id: Optional video ID for logging context.
            error_category: Optional error category for per-category tracking (US-137-002).
        """
        with self._lock:
            self.attempts += 1
            if error_category is not None:
                self.attempts_by_category[error_category] = (
                    self.attempts_by_category.get(error_category, 0) + 1
                )
            # Log at INFO every 10 attempts and at important milestones
            if self.attempts % 10 == 0 or self.attempts == 1:
                pct_used = (self.attempts / self.max_attempts * 100) if self.max_attempts > 0 else 0
                logger.info(
                    f"[TRANSCRIBE] Retry budget consumed: {self.attempts}/{self.max_attempts} "
                    f"attempts ({pct_used:.0f}% used), {self.failures} failures, {self.successes} successes"
                )

    def record_failure(
        self, video_id: str = "", error_category: Optional[TranscriptionErrorCategory] = None
    ) -> None:
        """Record a failed transcription attempt.

        Args:
            video_id: Optional video ID for logging context.
            error_category: Optional error category for per-category tracking (US-137-002).
        """
        with self._lock:
            self.failures += 1
            if error_category is not None:
                self.failures_by_category[error_category] = (
                    self.failures_by_category.get(error_category, 0) + 1
                )
            # Log failure with context for tracking
            category_str = f" ({error_category.value})" if error_category else ""
            logger.info(f"[TRANSCRIBE] Transcription failed for {video_id or 'unknown'}{category_str}")
            logger.debug(f"TranscriptionRetryBudget: failure for {video_id}")

    def record_success(self, video_id: str = "") -> None:
        """Record a successful transcription.

        Args:
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.successes += 1

    def record_backoff(self, delay: float) -> None:
        """Record backoff time spent waiting.

        Args:
            delay: Backoff delay in seconds.
        """
        with self._lock:
            self.backoff_time_spent += delay
            # Log backoff consumption at milestones
            if self.max_backoff_time > 0:
                pct_used = (self.backoff_time_spent / self.max_backoff_time * 100)
                if pct_used >= 50 and (pct_used - (delay / self.max_backoff_time * 100)) < 50:
                    logger.info(
                        f"[TRANSCRIBE] Backoff budget: {self.backoff_time_spent:.1f}s/{self.max_backoff_time:.1f}s "
                        f"({pct_used:.0f}% used)"
                    )

    def record_skip(self, video_id: str) -> None:
        """Record a video skipped due to budget exhaustion.

        Args:
            video_id: ID of the skipped video.
        """
        with self._lock:
            self.videos_skipped += 1
            self.skipped_video_ids.append(video_id)
            # Log skip with TRANSCRIBE error code for tracking
            log_error_with_context(
                logger,
                "TRANSCRIBE-003",
                f"Video skipped due to retry budget exhaustion",
                video_id=video_id,
                videos_skipped=self.videos_skipped,
                attempts=self.attempts,
                failures=self.failures,
            )

    def is_exhausted(self) -> bool:
        """Check if the retry budget is exhausted.

        Returns:
            True if max_attempts or max_backoff_time exceeded.
        """
        with self._lock:
            was_exhausted = (
                self.max_attempts > 0 and self.attempts >= self.max_attempts
            ) or (self.max_backoff_time > 0 and self.backoff_time_spent >= self.max_backoff_time)

            # Log exhaustion with TRANSCRIBE error code
            if was_exhausted:
                reason = self.exhaustion_reason()
                log_error_with_context(
                    logger,
                    "TRANSCRIBE-003",
                    f"Transcription retry budget exhausted: {reason}",
                    attempts=self.attempts,
                    max_attempts=self.max_attempts,
                    backoff_time_spent=round(self.backoff_time_spent, 1),
                    max_backoff_time=self.max_backoff_time,
                )

            # Log category distribution when budget first exhausts (US-137-002)
            if was_exhausted and self.attempts_by_category:
                self._log_category_distribution()

            return was_exhausted

    def exhaustion_reason(self) -> Optional[str]:
        """Return the reason for budget exhaustion, or None if not exhausted.

        Returns:
            String describing which limit was hit, or None.
        """
        with self._lock:
            if self.max_attempts > 0 and self.attempts >= self.max_attempts:
                return f"max_attempts reached ({self.attempts}/{self.max_attempts})"
            if self.max_backoff_time > 0 and self.backoff_time_spent >= self.max_backoff_time:
                return (
                    f"max_backoff_time reached "
                    f"({self.backoff_time_spent:.1f}s/{self.max_backoff_time:.1f}s)"
                )
            return None

    def get_attempts_by_category(self) -> Dict[TranscriptionErrorCategory, int]:
        """Get breakdown of attempts by error category (US-137-002).

        Returns:
            Dict mapping error category to attempt count.
        """
        with self._lock:
            return dict(self.attempts_by_category)

    def get_failures_by_category(self) -> Dict[TranscriptionErrorCategory, int]:
        """Get breakdown of failures by error category (US-137-002).

        Returns:
            Dict mapping error category to failure count.
        """
        with self._lock:
            return dict(self.failures_by_category)

    def _log_category_distribution(self) -> None:
        """Log error category distribution when budget exhausts (US-137-002)."""
        with self._lock:
            if not self.attempts_by_category:
                return

            # Format category distribution for logging
            attempts_dist = ", ".join(
                f"{cat.value}: {count}"
                for cat, count in sorted(
                    self.attempts_by_category.items(), key=lambda x: x[1], reverse=True
                )
            )
            failures_dist = ", ".join(
                f"{cat.value}: {count}"
                for cat, count in sorted(
                    self.failures_by_category.items(), key=lambda x: x[1], reverse=True
                )
            )

            logger.info(
                f"TranscriptionRetryBudget: Category distribution at exhaustion - "
                f"attempts: [{attempts_dist}], failures: [{failures_dist}]"
            )

    def get_summary(self) -> Dict:
        """Get budget summary for metrics/logging.

        Returns:
            Dict with budget usage summary including per-category breakdown (US-137-002).
        """
        with self._lock:
            return {
                'total_attempts': self.attempts,
                'failed_attempts': self.failures,
                'successful_attempts': self.successes,
                'backoff_time_spent': round(self.backoff_time_spent, 2),
                'videos_skipped': self.videos_skipped,
                'skipped_video_ids': list(self.skipped_video_ids),
                'max_attempts': self.max_attempts,
                'max_backoff_time': self.max_backoff_time,
                'is_exhausted': self.is_exhausted(),
                'exhaustion_reason': self.exhaustion_reason(),
                # Per-category breakdown (US-137-002)
                'attempts_by_category': {
                    cat.value: count for cat, count in self.attempts_by_category.items()
                },
                'failures_by_category': {
                    cat.value: count for cat, count in self.failures_by_category.items()
                },
            }


class BackoffStrategy(Enum):
    """Backoff strategies for retry delays (US-137-011).

    Strategies:
        STANDARD: Basic exponential backoff (delay * 2^attempt)
        JITTER: Exponential backoff with uniform jitter to avoid thundering herd
        CORRELATED: Jitter with correlation factor to stagger retries across workers
        ADAPTIVE: Auto-select best strategy based on historical success rate
    """
    STANDARD = "standard"
    JITTER = "jitter"
    CORRELATED = "correlated"
    ADAPTIVE = "adaptive"


# Global state for correlated backoff (US-137-011)
# This allows workers to coordinate delays to avoid thundering herd
_correlated_backoff_state = {
    'last_adjustment_time': 0.0,
    'adjustment_count': 0,
    'base_offset': 0.0,
}
_correlated_lock = threading.Lock()


@dataclass
class TranscriptionBackoffManager:
    """Manages retry backoff with jitter correlation (US-137-011).

    Provides intelligent backoff strategies to avoid thundering herd problem
    when multiple workers retry simultaneously after failures.

    Attributes:
        base_delay: Base delay in seconds (default 1.0)
        jitter_factor: Jitter range as fraction of delay (0.0-1.0, default 0.3)
        correlation_factor: Correlation factor for staggering across workers (0.0-1.0, default 0.5)
        strategy: Current backoff strategy to use
        max_jitter_cap: Maximum jitter cap in seconds (default 10.0)
    """

    base_delay: float = 1.0
    jitter_factor: float = 0.3
    correlation_factor: float = 0.5
    strategy: BackoffStrategy = BackoffStrategy.STANDARD
    max_jitter_cap: float = 10.0

    # Historical tracking for adaptive strategy (US-137-011)
    _success_counts: Dict[BackoffStrategy, int] = field(default_factory=dict)
    _failure_counts: Dict[BackoffStrategy, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def calculate_delay(self, attempt: int, worker_id: Optional[int] = None) -> float:
        """Calculate backoff delay for given attempt number.

        Uses the configured strategy to calculate delay:
        - STANDARD: delay * 2^attempt
        - JITTER: delay * (1 + random.uniform(-jitter, jitter)) * 2^attempt
        - CORRELATED: JITTER + correlation offset based on worker_id
        - ADAPTIVE: Use best performing strategy from history

        Args:
            attempt: Retry attempt number (0-indexed)
            worker_id: Optional worker ID for correlated backoff

        Returns:
            Delay in seconds
        """
        with self._lock:
            # Get effective strategy (ADAPTIVE resolves to best performing)
            effective_strategy = self._get_effective_strategy()

            # Calculate base exponential delay
            base_delay = self.base_delay * (2 ** attempt)

            if effective_strategy == BackoffStrategy.STANDARD:
                return min(base_delay, self.max_jitter_cap)

            elif effective_strategy == BackoffStrategy.JITTER:
                # Jitter: delay * (1 + random.uniform(-jitter, jitter))
                jitter_range = base_delay * self.jitter_factor
                jittered_delay = base_delay + random.uniform(-jitter_range, jitter_range)
                return max(0.1, min(jittered_delay, self.max_jitter_cap))

            elif effective_strategy == BackoffStrategy.CORRELATED:
                # Jitter + correlation offset based on worker_id
                jitter_range = base_delay * self.jitter_factor
                jittered_delay = base_delay + random.uniform(-jitter_range, jitter_range)

                # Add correlation offset to stagger retries across workers
                if worker_id is not None:
                    offset = self._calculate_correlation_offset(worker_id, base_delay)
                    jittered_delay += offset

                return max(0.1, min(jittered_delay, self.max_jitter_cap))

            else:
                # Fallback to standard
                return min(base_delay, self.max_jitter_cap)

    def _get_effective_strategy(self) -> BackoffStrategy:
        """Get the effective strategy (ADAPTIVE resolves to best performing)."""
        if self.strategy != BackoffStrategy.ADAPTIVE:
            return self.strategy

        # Find strategy with best success rate
        best_strategy = BackoffStrategy.STANDARD
        best_rate = -1.0

        for strategy in [BackoffStrategy.STANDARD, BackoffStrategy.JITTER, BackoffStrategy.CORRELATED]:
            successes = self._success_counts.get(strategy, 0)
            failures = self._failure_counts.get(strategy, 0)
            total = successes + failures

            if total > 0:
                rate = successes / total
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strategy

        return best_strategy

    def _calculate_correlation_offset(self, worker_id: int, base_delay: float) -> float:
        """Calculate correlation offset for worker to avoid thundering herd.

        Uses a deterministic but distributed offset based on worker_id
        and adjusts over time to prevent synchronized retries.
        """
        global _correlated_backoff_state

        with _correlated_lock:
            current_time = time.time()

            # Recalculate base offset periodically to avoid synchronization
            if current_time - _correlated_backoff_state['last_adjustment_time'] > 60.0:
                _correlated_backoff_state['last_adjustment_time'] = current_time
                _correlated_backoff_state['adjustment_count'] += 1
                # New random base offset
                _correlated_backoff_state['base_offset'] = random.uniform(0, base_delay)

            # Calculate worker-specific offset using golden ratio for good distribution
            phi = 1.618033988749895
            worker_offset = (worker_id * phi) % 1.0  # 0-1 range
            worker_offset = worker_offset * base_delay * self.correlation_factor

            # Add base offset from global state
            total_offset = worker_offset + _correlated_backoff_state['base_offset']

            return total_offset

    def record_success(self, strategy_used: Optional[BackoffStrategy] = None) -> None:
        """Record a successful retry after backoff (for adaptive strategy).

        Args:
            strategy_used: The strategy that was used for this retry
        """
        with self._lock:
            effective = strategy_used or self._get_effective_strategy()
            self._success_counts[effective] = self._success_counts.get(effective, 0) + 1

    def record_failure(self, strategy_used: Optional[BackoffStrategy] = None) -> None:
        """Record a failed retry after backoff (for adaptive strategy).

        Args:
            strategy_used: The strategy that was used for this retry
        """
        with self._lock:
            effective = strategy_used or self._get_effective_strategy()
            self._failure_counts[effective] = self._failure_counts.get(effective, 0) + 1

    def get_success_rate(self, strategy: Optional[BackoffStrategy] = None) -> float:
        """Get success rate for a strategy or the effective strategy.

        Args:
            strategy: Optional strategy to get rate for, or None for effective

        Returns:
            Success rate as float between 0.0 and 1.0
        """
        with self._lock:
            effective = strategy or self._get_effective_strategy()
            successes = self._success_counts.get(effective, 0)
            failures = self._failure_counts.get(effective, 0)
            total = successes + failures

            if total == 0:
                return 0.5  # Default neutral rate

            return successes / total

    def get_strategy_stats(self) -> Dict:
        """Get statistics for all strategies.

        Returns:
            Dict with strategy stats including success rates
        """
        with self._lock:
            stats = {}
            for strategy in BackoffStrategy:
                if strategy == BackoffStrategy.ADAPTIVE:
                    effective = self._get_effective_strategy()
                    stats[strategy.value] = {
                        'effective_strategy': effective.value,
                        'successes': self._success_counts.get(effective, 0),
                        'failures': self._failure_counts.get(effective, 0),
                        'success_rate': self.get_success_rate(effective),
                    }
                else:
                    stats[strategy.value] = {
                        'successes': self._success_counts.get(strategy, 0),
                        'failures': self._failure_counts.get(strategy, 0),
                        'success_rate': self.get_success_rate(strategy),
                    }
            return stats

    def get_best_strategy(self) -> BackoffStrategy:
        """Get the best performing strategy based on historical success rate.

        Returns:
            Best strategy enum value
        """
        return self._get_effective_strategy()

    def reset_stats(self) -> None:
        """Reset historical statistics for all strategies."""
        with self._lock:
            self._success_counts.clear()
            self._failure_counts.clear()
