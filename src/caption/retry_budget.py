"""
Batch retry budget management for caption fetching.

Provides adaptive retry budget management that reduces retries when
batch-wide failure patterns are detected.

Includes:
- BatchRetryBudget: Per-category retry tracking with adaptive reduction
- CaptionRetryBudget: Cross-video budget tracking for attempts/backoff (US-33-010)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import CaptionErrorCategory, DEFAULT_RETRY_BUDGETS

logger = logging.getLogger(__name__)


@dataclass
class CaptionRetryBudgetConfig:
    """Configuration for CaptionRetryBudget (US-33-010).

    Controls when caption fetching should stop due to resource exhaustion.

    Configure in config.yaml under download.caption_first.retry_budget.
    """
    # Enable/disable retry budget tracking
    enabled: bool = True

    # Maximum total fetch attempts across all videos in batch
    # Set to 0 for unlimited attempts
    max_attempts: int = 100

    # Maximum cumulative backoff time (seconds) before exhaustion
    # Set to 0 for unlimited backoff
    max_backoff_time_seconds: float = 300.0

    # Automatic scaling settings (US-37-004)
    # When enabled, max_attempts scales up based on batch size
    auto_scale: bool = True

    # Attempts per video multiplier for auto-scaling
    # 2.0 = 1 attempt + 1 retry per video average
    # Only scales UP when batch > max_attempts / attempts_per_video
    attempts_per_video: float = 2.0

    # VPN rotation on rate limit exhaustion (US-37-008)
    # When budget exhausts with >50% RATE_LIMIT errors, trigger VPN rotation
    # This resets the budget and retries remaining videos with a new IP
    trigger_vpn_rotation_on_rate_limit: bool = True

    # Maximum VPN-triggered budget resets per session (US-37-008)
    # Prevents infinite loops if VPN rotation doesn't help
    max_vpn_resets_per_session: int = 2

    # Early termination settings (US-37-009)
    # When success rate drops below threshold, terminate early
    min_success_rate: float = 0.3  # 30% minimum success rate
    min_sample_for_early_termination: int = 20  # Check after 20 videos

    # Reset on scale-up settings (US-41-009)
    # When enabled, scaling up budget also resets usage counters
    # Useful when restoring from checkpoint with prior attempts and batch needs more budget
    reset_on_scale: bool = False


@dataclass
class CaptionRetryBudget:
    """Tracks retry resources used across all videos in a caption batch (US-33-010).

    Unlike BatchRetryBudget (which tracks per-category retry budgets and reduces them
    based on error patterns), CaptionRetryBudget tracks total resources used across
    the entire batch:

    - Total fetch attempts (success + failures)
    - Total failures
    - Total backoff time spent waiting

    When any limit is exceeded, budget_exhausted() returns True, signaling the
    CaptionStage to skip remaining videos and proceed with transcription fallback.

    Thread Safety:
        All mutation methods are protected by a Lock for concurrent access
        during parallel caption fetching.

    Attributes:
        attempts: Total fetch attempts across all videos.
        failures: Total failures across all videos.
        successes: Total successes across all videos.
        backoff_time_spent: Cumulative backoff delay (seconds).
        videos_skipped: Count of videos skipped due to budget exhaustion.
        max_attempts: Maximum allowed attempts (0 = unlimited).
        max_backoff_time: Maximum allowed backoff time (0 = unlimited).
    """
    # Usage counters
    attempts: int = 0
    failures: int = 0
    successes: int = 0
    backoff_time_spent: float = 0.0
    videos_skipped: int = 0

    # Batch size tracking (US-38-009)
    batch_size: Optional[int] = None

    # Error category tracking (US-37-006)
    error_counts: Dict[CaptionErrorCategory, int] = field(default_factory=dict)

    # Budget limits (set from config)
    max_attempts: int = 100
    max_backoff_time: float = 300.0  # 5 minutes total backoff budget

    # Auto-scaling settings (US-37-004)
    auto_scale: bool = True
    attempts_per_video: float = 2.0

    # VPN rotation settings (US-37-008)
    trigger_vpn_on_rate_limit: bool = True
    max_vpn_resets: int = 2
    vpn_resets_used: int = 0

    # Early termination settings (US-37-009)
    min_success_rate: float = 0.3  # 30% minimum
    min_sample_for_early_termination: int = 20  # Check after 20 videos
    early_terminated: bool = False
    early_termination_reason: Optional[str] = None

    # Reset on scale-up settings (US-41-009)
    reset_on_scale: bool = False

    # Thread-safety lock
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    # Circuit breaker reference for observability (US-40-011)
    # Optional - allows summary to include circuit breaker state when set
    circuit_breaker: Optional[Any] = field(default=None, repr=False, compare=False)

    # Per-video attempt tracking for diagnosing budget consumption (US-41-005)
    # Maps video_id -> number of attempts recorded for that video
    attempts_per_video_id: Dict[str, int] = field(default_factory=dict)

    # Circuit breaker trip tracking (US-41-006)
    # Counts how many times the circuit breaker tripped during this budget's lifetime
    circuit_breaker_trips: int = 0

    @classmethod
    def from_config(cls, config: Optional[CaptionRetryBudgetConfig]) -> "CaptionRetryBudget":
        """Create a CaptionRetryBudget from config.

        Args:
            config: CaptionRetryBudgetConfig instance or None for defaults.

        Returns:
            CaptionRetryBudget with limits set from config.
        """
        budget = cls()
        if config is None:
            return budget

        # Handle both dict and dataclass config
        if isinstance(config, dict):
            budget.max_attempts = int(config.get('max_attempts', 100))
            budget.max_backoff_time = float(config.get('max_backoff_time_seconds', 300.0))
            budget.auto_scale = bool(config.get('auto_scale', True))
            budget.attempts_per_video = float(config.get('attempts_per_video', 2.0))
            # US-37-008: VPN rotation on rate limit exhaustion
            budget.trigger_vpn_on_rate_limit = bool(config.get('trigger_vpn_rotation_on_rate_limit', True))
            budget.max_vpn_resets = int(config.get('max_vpn_resets_per_session', 2))
            # US-37-009: Early termination on low success rate
            budget.min_success_rate = float(config.get('min_success_rate', 0.3))
            budget.min_sample_for_early_termination = int(config.get('min_sample_for_early_termination', 20))
            # US-41-009: Reset on scale-up
            budget.reset_on_scale = bool(config.get('reset_on_scale', False))
        else:
            budget.max_attempts = int(getattr(config, 'max_attempts', 100))
            budget.max_backoff_time = float(getattr(config, 'max_backoff_time_seconds', 300.0))
            budget.auto_scale = bool(getattr(config, 'auto_scale', True))
            budget.attempts_per_video = float(getattr(config, 'attempts_per_video', 2.0))
            # US-37-008: VPN rotation on rate limit exhaustion
            budget.trigger_vpn_on_rate_limit = bool(getattr(config, 'trigger_vpn_rotation_on_rate_limit', True))
            budget.max_vpn_resets = int(getattr(config, 'max_vpn_resets_per_session', 2))
            # US-37-009: Early termination on low success rate
            budget.min_success_rate = float(getattr(config, 'min_success_rate', 0.3))
            budget.min_sample_for_early_termination = int(getattr(config, 'min_sample_for_early_termination', 20))
            # US-41-009: Reset on scale-up
            budget.reset_on_scale = bool(getattr(config, 'reset_on_scale', False))

        logger.debug(
            f"CaptionRetryBudget initialized: max_attempts={budget.max_attempts}, "
            f"max_backoff_time={budget.max_backoff_time}s, auto_scale={budget.auto_scale}, "
            f"attempts_per_video={budget.attempts_per_video}"
        )
        return budget

    def record_attempt(self, video_id: str = "") -> None:
        """Record a fetch attempt.

        Args:
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.attempts += 1
            attempts_count = self.attempts
            # Track per-video attempts (US-41-005)
            if video_id:
                self.attempts_per_video_id[video_id] = self.attempts_per_video_id.get(video_id, 0) + 1
                video_attempts = self.attempts_per_video_id[video_id]
                # Warn when single video consumes >3 attempts (indicates retry loop)
                if video_attempts == 4:  # Log only once when crossing threshold
                    logger.warning(
                        f"CaptionRetryBudget: video {video_id} has consumed {video_attempts} attempts "
                        f"(>3 indicates possible retry loop)"
                    )
        logger.debug(f"CaptionRetryBudget: attempt recorded for {video_id or 'unknown'} "
                     f"(total: {attempts_count})")
        # Check if we crossed a consumption threshold (US-37-005)
        self._check_and_log_threshold(video_id)

    def record_success(self, video_id: str = "") -> None:
        """Record a successful fetch.

        Args:
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.successes += 1
        logger.debug(f"CaptionRetryBudget: success for {video_id or 'unknown'} "
                     f"(total: {self.successes})")

    def record_failure(
        self,
        video_id: str = "",
        error_category: Optional[CaptionErrorCategory] = None
    ) -> None:
        """Record a failed fetch.

        Args:
            video_id: Optional video ID for logging context.
            error_category: Optional error category for tracking (US-37-006).
        """
        with self._lock:
            self.failures += 1
            failures_count = self.failures
            # Track error category if provided (US-37-006)
            if error_category is not None:
                self.error_counts[error_category] = self.error_counts.get(error_category, 0) + 1
        logger.debug(f"CaptionRetryBudget: failure for {video_id or 'unknown'} "
                     f"(total: {failures_count})"
                     f"{f' [{error_category.name}]' if error_category else ''}")
        # Check if we crossed a consumption threshold (US-37-005)
        self._check_and_log_threshold(video_id)

    def record_backoff(self, seconds: float, video_id: str = "") -> None:
        """Record backoff time spent.

        Args:
            seconds: Duration of backoff delay.
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.backoff_time_spent += seconds
            total_backoff = self.backoff_time_spent
        logger.debug(
            f"CaptionRetryBudget: backoff {seconds:.1f}s for {video_id or 'unknown'} "
            f"(total: {total_backoff:.1f}s)"
        )
        # Check if we crossed a consumption threshold (US-37-005)
        self._check_and_log_threshold(video_id)

    def record_skipped(self, video_id: str = "") -> None:
        """Record a video skipped due to budget exhaustion.

        Args:
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.videos_skipped += 1
        logger.debug(f"CaptionRetryBudget: skipped {video_id or 'unknown'} "
                     f"(total skipped: {self.videos_skipped})")

    def record_circuit_trip(self) -> None:
        """Record a circuit breaker trip event (US-41-006).

        Called when the circuit breaker transitions to the open state.
        This indicates transient failures that consumed budget, helping
        distinguish rate-limiting from other failure types.
        """
        with self._lock:
            self.circuit_breaker_trips += 1
            trips = self.circuit_breaker_trips
        logger.debug(f"CaptionRetryBudget: circuit breaker tripped (total trips: {trips})")

    def budget_exhausted(self) -> bool:
        """Check if the retry budget is exhausted.

        Returns True when:
        - max_attempts > 0 and attempts >= max_attempts, OR
        - max_backoff_time > 0 and backoff_time_spent >= max_backoff_time

        Returns:
            True if budget is exhausted and remaining videos should be skipped.
        """
        with self._lock:
            # US-41-010: Build progress suffix for EXHAUSTED messages
            progress_suffix = ""
            videos_processed = self.successes + self.failures
            if self.batch_size and self.batch_size > 0:
                progress_pct = round((videos_processed / self.batch_size) * 100, 0)
                progress_suffix = f" at {progress_pct:.0f}% progress ({videos_processed}/{self.batch_size} videos processed)"

            # Check attempt limit
            if self.max_attempts > 0 and self.attempts >= self.max_attempts:
                logger.info(
                    f"CaptionRetryBudget: EXHAUSTED (attempts: {self.attempts}/{self.max_attempts}){progress_suffix}"
                )
                # US-41-006: Log circuit breaker trips if any occurred
                if self.circuit_breaker_trips > 0:
                    logger.info(
                        f"CaptionRetryBudget: {self.circuit_breaker_trips} circuit breaker trip(s) "
                        f"contributed to budget exhaustion (indicates transient failures)"
                    )
                return True

            # Check backoff time limit
            if self.max_backoff_time > 0 and self.backoff_time_spent >= self.max_backoff_time:
                logger.info(
                    f"CaptionRetryBudget: EXHAUSTED (backoff: {self.backoff_time_spent:.1f}s/"
                    f"{self.max_backoff_time}s){progress_suffix}"
                )
                # US-41-006: Log circuit breaker trips if any occurred
                if self.circuit_breaker_trips > 0:
                    logger.info(
                        f"CaptionRetryBudget: {self.circuit_breaker_trips} circuit breaker trip(s) "
                        f"contributed to budget exhaustion (indicates transient failures)"
                    )
                return True

            return False

    def get_top_errors(self, limit: int = 5) -> List[tuple]:
        """Get the top error categories by count (US-37-006).

        Returns sorted list of (CaptionErrorCategory, count) tuples,
        ordered by count descending.

        Args:
            limit: Maximum number of categories to return. Default 5.

        Returns:
            List of (CaptionErrorCategory, int) tuples sorted by count descending.

        Example:
            >>> budget.get_top_errors()
            [(CaptionErrorCategory.NETWORK, 15), (CaptionErrorCategory.TIMEOUT, 5)]
        """
        with self._lock:
            sorted_errors = sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_errors[:limit]

    def get_high_attempt_videos(self, threshold: int = 3) -> List[str]:
        """Get video IDs that have consumed more than threshold attempts (US-41-005).

        Used to diagnose budget consumption issues - if one video consumes 10+
        attempts due to a retry loop, it starves other videos of budget.

        Args:
            threshold: Minimum attempts to be considered high. Default 3.
                       Returns videos with >threshold attempts (not >=).

        Returns:
            List of video IDs with more than threshold attempts, sorted by
            attempt count descending.

        Example:
            >>> budget.get_high_attempt_videos(3)
            ['video_abc', 'video_xyz']  # Videos with 4+ attempts
        """
        with self._lock:
            high_attempt = [
                (video_id, count)
                for video_id, count in self.attempts_per_video_id.items()
                if count > threshold
            ]
            # Sort by count descending
            high_attempt.sort(key=lambda x: x[1], reverse=True)
            return [video_id for video_id, _ in high_attempt]

    def get_rate_limit_error_percentage(self) -> float:
        """Get the percentage of failures that are RATE_LIMIT errors (US-37-008).

        Used to determine whether VPN rotation should be triggered when
        budget is exhausted. If >50% of failures are rate limit errors,
        VPN rotation may help get a fresh IP.

        Returns:
            Percentage (0.0 to 100.0) of failures that are RATE_LIMIT category.
            Returns 0.0 if no failures recorded.

        Example:
            >>> budget.get_rate_limit_error_percentage()
            65.5  # 65.5% of failures were rate limits
        """
        with self._lock:
            if self.failures == 0:
                return 0.0
            rate_limit_count = self.error_counts.get(CaptionErrorCategory.RATE_LIMIT, 0)
            return round((rate_limit_count / self.failures) * 100, 1)

    def should_trigger_vpn_rotation(self, rate_limit_threshold: float = 50.0) -> bool:
        """Check if VPN rotation should be triggered due to rate limit exhaustion (US-37-008).

        VPN rotation is triggered when:
        1. Budget is exhausted
        2. Rate limit errors account for >50% of failures
        3. VPN rotation is enabled in config
        4. VPN resets haven't been exhausted

        Args:
            rate_limit_threshold: Minimum percentage of rate limit errors to trigger.
                                  Default 50.0 (>50% rate limit errors triggers VPN).

        Returns:
            True if VPN rotation should be triggered, False otherwise.
        """
        with self._lock:
            # Check prerequisites
            if not self.budget_exhausted():
                return False
            if not self.trigger_vpn_on_rate_limit:
                return False
            if self.vpn_resets_used >= self.max_vpn_resets:
                logger.info(
                    f"VPN rotation limit reached ({self.vpn_resets_used}/{self.max_vpn_resets}), "
                    "cannot trigger more VPN rotations"
                )
                return False

            # Check rate limit percentage
            rate_limit_pct = self.get_rate_limit_error_percentage()
            if rate_limit_pct > rate_limit_threshold:
                logger.info(
                    f"Budget exhausted with {rate_limit_pct:.1f}% rate limit errors, "
                    f"rotating VPN ({self.vpn_resets_used + 1}/{self.max_vpn_resets})"
                )
                return True

            return False

    def record_vpn_reset(self) -> None:
        """Record a VPN-triggered budget reset (US-37-008).

        Called after successful VPN rotation to track reset count
        and prevent infinite loops.
        """
        with self._lock:
            self.vpn_resets_used += 1
            logger.info(
                f"CaptionRetryBudget: VPN reset recorded ({self.vpn_resets_used}/{self.max_vpn_resets})"
            )

    def can_vpn_reset(self) -> bool:
        """Check if more VPN-triggered resets are available (US-37-008).

        Returns:
            True if vpn_resets_used < max_vpn_resets.
        """
        with self._lock:
            return self.vpn_resets_used < self.max_vpn_resets

    def get_success_rate(self) -> float:
        """Get current success rate (US-37-009).

        Returns:
            Success rate as float (0.0 to 1.0).
            Returns 1.0 if no videos processed (no data to judge).

        Example:
            >>> budget.get_success_rate()
            0.25  # 25% success rate
        """
        with self._lock:
            total_processed = self.successes + self.failures
            if total_processed == 0:
                return 1.0  # No data = assume OK
            return self.successes / total_processed

    def should_terminate_early(self) -> bool:
        """Check if batch should terminate early due to low success rate (US-37-009).

        Early termination triggers when:
        1. At least min_sample_for_early_termination videos processed
        2. Success rate is below min_success_rate threshold
        3. Not already terminated

        Returns:
            True if early termination should occur.

        Example:
            >>> budget.should_terminate_early()
            True  # When 20+ videos processed and success rate < 30%
        """
        with self._lock:
            # Don't double-terminate
            if self.early_terminated:
                return False

            # Need sufficient sample
            total_processed = self.successes + self.failures
            if total_processed < self.min_sample_for_early_termination:
                return False

            # Check success rate
            success_rate = self.get_success_rate()
            if success_rate < self.min_success_rate:
                return True

            return False

    def check_and_terminate_early(self) -> bool:
        """Check if early termination should occur and mark as terminated (US-37-009).

        This method combines the check and action to ensure atomic operation.
        Call this after each video is processed to check if batch should stop.

        Returns:
            True if early termination was triggered (first time),
            False if already terminated or conditions not met.

        Side effects:
            Sets early_terminated = True and early_termination_reason if triggered.
        """
        with self._lock:
            if not self.should_terminate_early():
                return False

            # Calculate and store the termination reason
            success_rate = self.get_success_rate()
            total_processed = self.successes + self.failures
            self.early_terminated = True
            self.early_termination_reason = (
                f"Success rate {success_rate:.1%} below threshold {self.min_success_rate:.1%} "
                f"after {total_processed} videos ({self.successes} succeeded, {self.failures} failed)"
            )

            logger.warning(
                f"CaptionRetryBudget: EARLY TERMINATION - {self.early_termination_reason}"
            )
            return True

    def is_early_terminated(self) -> bool:
        """Check if budget was early-terminated (US-37-009).

        Returns:
            True if early_terminated flag is set.
        """
        with self._lock:
            return self.early_terminated

    def get_consumption_percentage(self) -> Dict[str, Optional[float]]:
        """Get percentage of budget consumed for each resource (US-37-005).

        Returns:
            Dict with keys 'attempts' and 'backoff_time', values are percentages
            (0.0 to 100.0+) or None if that limit is unlimited.

        Example:
            {'attempts': 45.0, 'backoff_time': 30.5}  # 45% attempts, 30.5% backoff used
        """
        with self._lock:
            result = {}

            if self.max_attempts > 0:
                result['attempts'] = round((self.attempts / self.max_attempts) * 100, 1)
            else:
                result['attempts'] = None

            if self.max_backoff_time > 0:
                result['backoff_time'] = round((self.backoff_time_spent / self.max_backoff_time) * 100, 1)
            else:
                result['backoff_time'] = None

            return result

    def _log_consumption_status(self, trigger: str, video_id: str = "") -> None:
        """Log detailed consumption status at key thresholds (US-37-005).

        Logs INFO when crossing 25%, 50%, 75%, 90% thresholds.
        Logs WARNING at 90%+ to alert impending exhaustion.

        Args:
            trigger: What caused this log ('attempt', 'failure', 'backoff', 'exhausted').
            video_id: Optional video ID for context.
        """
        consumption = self.get_consumption_percentage()
        attempts_pct = consumption.get('attempts')
        backoff_pct = consumption.get('backoff_time')

        # Determine the highest percentage for threshold checking
        max_pct = 0.0
        if attempts_pct is not None:
            max_pct = max(max_pct, attempts_pct)
        if backoff_pct is not None:
            max_pct = max(max_pct, backoff_pct)

        # Build status message
        status_parts = []
        if attempts_pct is not None:
            status_parts.append(f"attempts: {self.attempts}/{self.max_attempts} ({attempts_pct:.0f}%)")
        if backoff_pct is not None:
            status_parts.append(f"backoff: {self.backoff_time_spent:.1f}s/{self.max_backoff_time}s ({backoff_pct:.0f}%)")

        status_msg = ", ".join(status_parts)
        video_ctx = f" [{video_id}]" if video_id else ""

        # Check thresholds for INFO/WARNING logging
        # We log when crossing key thresholds: 25%, 50%, 75%, 90%
        if max_pct >= 90:
            logger.warning(
                f"CaptionRetryBudget: 90%+ consumed{video_ctx} - {status_msg} "
                f"(successes: {self.successes}, failures: {self.failures})"
            )
        elif max_pct >= 75:
            logger.info(
                f"CaptionRetryBudget: 75%+ consumed{video_ctx} - {status_msg}"
            )
        elif max_pct >= 50:
            logger.info(
                f"CaptionRetryBudget: 50%+ consumed{video_ctx} - {status_msg}"
            )
        elif max_pct >= 25:
            logger.info(
                f"CaptionRetryBudget: 25%+ consumed{video_ctx} - {status_msg}"
            )

    def _check_and_log_threshold(self, video_id: str = "") -> None:
        """Check if we just crossed a threshold and log if so (US-37-005).

        Called after mutations to log when crossing 25%, 50%, 75%, 90% thresholds.
        Uses a simple heuristic: log if current percentage is within 1% of a threshold.
        """
        consumption = self.get_consumption_percentage()
        attempts_pct = consumption.get('attempts')
        backoff_pct = consumption.get('backoff_time')

        # Get highest percentage
        max_pct = 0.0
        if attempts_pct is not None:
            max_pct = max(max_pct, attempts_pct)
        if backoff_pct is not None:
            max_pct = max(max_pct, backoff_pct)

        # Check if we just crossed a threshold (within small margin)
        thresholds = [25, 50, 75, 90]
        for threshold in thresholds:
            # Log if we're within 1 percentage point above threshold (just crossed it)
            if threshold <= max_pct < threshold + 2:
                self._log_consumption_status("threshold", video_id)
                break

    def get_progress_percentage(self) -> Optional[float]:
        """Get batch progress as percentage of videos processed (US-41-010).

        Returns:
            Percentage (0.0 to 100.0) of batch processed, or None if batch_size not set.
            Calculated as (successes + failures) / batch_size * 100.

        Example:
            >>> budget.batch_size = 175
            >>> budget.successes = 80
            >>> budget.failures = 20
            >>> budget.get_progress_percentage()
            57.14  # 100/175 = 57.14%
        """
        with self._lock:
            if self.batch_size is None or self.batch_size == 0:
                return None
            videos_processed = self.successes + self.failures
            return round((videos_processed / self.batch_size) * 100, 2)

    def get_videos_processed(self) -> int:
        """Get total videos processed (successes + failures).

        Returns:
            Total count of videos that have been processed.
        """
        with self._lock:
            return self.successes + self.failures

    def attempts_remaining(self) -> Optional[int]:
        """Get remaining attempts before exhaustion.

        Returns:
            Number of attempts remaining, or None if unlimited.
        """
        with self._lock:
            if self.max_attempts <= 0:
                return None
            return max(0, self.max_attempts - self.attempts)

    def backoff_time_remaining(self) -> Optional[float]:
        """Get remaining backoff time budget.

        Returns:
            Seconds of backoff remaining, or None if unlimited.
        """
        with self._lock:
            if self.max_backoff_time <= 0:
                return None
            return max(0.0, self.max_backoff_time - self.backoff_time_spent)

    def can_backoff(self, additional_seconds: float = 0.0) -> bool:
        """Check if backoff time is available within budget.

        Args:
            additional_seconds: Planned backoff duration to check.

        Returns:
            True if total backoff time would be under limit.
        """
        with self._lock:
            if self.max_backoff_time <= 0:
                return True  # Unlimited
            return (self.backoff_time_spent + additional_seconds) <= self.max_backoff_time

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of budget usage for reporting.

        Returns:
            Dict with budget usage statistics including error breakdown (US-37-006),
            early termination state (US-37-009), and high-attempt videos (US-41-005).
        """
        with self._lock:
            # Build error breakdown (category name -> count)
            error_breakdown = {cat.name: count for cat, count in self.error_counts.items()}

            # Get high-attempt videos (US-41-005)
            high_attempt_videos = self.get_high_attempt_videos(threshold=3)

            summary = {
                "attempts": self.attempts,
                "attempts_remaining": self.attempts_remaining(),
                "successes": self.successes,
                "failures": self.failures,
                "success_rate": round(self.get_success_rate(), 3),  # US-37-009
                "backoff_time_spent": round(self.backoff_time_spent, 1),
                "backoff_time_remaining": (
                    round(self.backoff_time_remaining(), 1)
                    if self.backoff_time_remaining() is not None
                    else None
                ),
                "videos_skipped": self.videos_skipped,
                "videos_processed": self.successes + self.failures,  # US-41-010
                "progress_percentage": self.get_progress_percentage(),  # US-41-010
                "is_exhausted": self.budget_exhausted(),
                "early_terminated": self.early_terminated,  # US-37-009
                "early_termination_reason": self.early_termination_reason,  # US-37-009
                "error_breakdown": error_breakdown,  # US-37-006
                "batch_size": self.batch_size,  # US-38-009
                "max_attempts": self.max_attempts,  # US-39-005: Include scaled max_attempts
                "circuit_breaker_state": self._get_circuit_breaker_state(),  # US-40-011
                "circuit_breaker_trips": self.circuit_breaker_trips,  # US-41-006
            }

            # Only include high_attempt_videos if there are any (US-41-005)
            if high_attempt_videos:
                summary["high_attempt_videos"] = high_attempt_videos

            return summary

    def _get_circuit_breaker_state(self) -> Optional[str]:
        """Get circuit breaker state for observability (US-40-011).

        Returns:
            'open' if circuit is tripped (blocking fetches),
            'closed' if circuit is normal,
            None if no circuit breaker is attached.
        """
        if self.circuit_breaker is None:
            return None
        # CaptionCircuitBreaker has is_open property
        if hasattr(self.circuit_breaker, 'is_open'):
            return 'open' if self.circuit_breaker.is_open else 'closed'
        return None

    def get_formatted_summary(self) -> str:
        """Get a formatted summary string for logging at stage completion (US-39-005).

        Returns a single-line summary with all key budget metrics for easy
        diagnosis of budget exhaustion issues.

        Returns:
            Formatted string: 'CaptionRetryBudget summary: {attempts}/{max_attempts} attempts,
            {successes} succeeded, {failures} failed, {skipped} skipped (batch_size={N}, circuit_breaker={state})'

        Example:
            >>> budget.get_formatted_summary()
            'CaptionRetryBudget summary: 150/175 attempts, 120 succeeded, 30 failed, 5 skipped (batch_size=150, circuit_breaker=closed)'
        """
        with self._lock:
            cb_state = self._get_circuit_breaker_state()
            cb_suffix = f", circuit_breaker={cb_state}" if cb_state else ""
            return (
                f"CaptionRetryBudget summary: {self.attempts}/{self.max_attempts} attempts, "
                f"{self.successes} succeeded, {self.failures} failed, "
                f"{self.videos_skipped} skipped (batch_size={self.batch_size or 0}{cb_suffix})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize budget state for checkpoint persistence.

        Returns:
            Dict with all budget state data including error_counts (US-37-006),
            vpn_resets_used (US-37-008), early termination state (US-37-009),
            and per-video attempt tracking (US-41-005).
        """
        with self._lock:
            return {
                "attempts": self.attempts,
                "failures": self.failures,
                "successes": self.successes,
                "backoff_time_spent": self.backoff_time_spent,
                "videos_skipped": self.videos_skipped,
                "max_attempts": self.max_attempts,
                "max_backoff_time": self.max_backoff_time,
                "error_counts": {cat.name: count for cat, count in self.error_counts.items()},  # US-37-006
                "vpn_resets_used": self.vpn_resets_used,  # US-37-008
                "max_vpn_resets": self.max_vpn_resets,  # US-37-008
                "early_terminated": self.early_terminated,  # US-37-009
                "early_termination_reason": self.early_termination_reason,  # US-37-009
                "batch_size": self.batch_size,  # US-38-009
                "attempts_per_video_id": dict(self.attempts_per_video_id),  # US-41-005
                "circuit_breaker_trips": self.circuit_breaker_trips,  # US-41-006
            }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "CaptionRetryBudget":
        """Create budget from checkpoint data.

        Args:
            data: Checkpoint data dict (may be None for new sessions).

        Returns:
            CaptionRetryBudget with restored state.
        """
        if not data:
            return cls()

        budget = cls(
            attempts=data.get("attempts", 0),
            failures=data.get("failures", 0),
            successes=data.get("successes", 0),
            backoff_time_spent=data.get("backoff_time_spent", 0.0),
            videos_skipped=data.get("videos_skipped", 0),
        )

        # Restore budget limits
        budget.max_attempts = data.get("max_attempts", 100)
        budget.max_backoff_time = data.get("max_backoff_time", 300.0)

        # Restore error counts (US-37-006)
        for cat_name, count in data.get("error_counts", {}).items():
            try:
                cat = CaptionErrorCategory[cat_name]
                budget.error_counts[cat] = count
            except KeyError:
                logger.warning(f"Unknown error category in checkpoint: {cat_name}")

        # Restore VPN reset tracking (US-37-008)
        budget.vpn_resets_used = data.get("vpn_resets_used", 0)
        budget.max_vpn_resets = data.get("max_vpn_resets", 2)

        # Restore early termination state (US-37-009)
        budget.early_terminated = data.get("early_terminated", False)
        budget.early_termination_reason = data.get("early_termination_reason")

        # Restore batch size (US-38-009)
        budget.batch_size = data.get("batch_size")

        # Restore per-video attempt tracking (US-41-005)
        budget.attempts_per_video_id = dict(data.get("attempts_per_video_id", {}))

        # Restore circuit breaker trip count (US-41-006)
        budget.circuit_breaker_trips = data.get("circuit_breaker_trips", 0)

        return budget

    def reset(self, preserve_vpn_count: bool = True) -> None:
        """Reset budget state for a new batch or after VPN rotation.

        Args:
            preserve_vpn_count: If True (default), preserve vpn_resets_used count.
                              Set to False only for full session reset.
        """
        with self._lock:
            self.attempts = 0
            self.failures = 0
            self.successes = 0
            self.backoff_time_spent = 0.0
            self.videos_skipped = 0
            self.error_counts.clear()  # US-37-006
            self.attempts_per_video_id.clear()  # US-41-005
            self.circuit_breaker_trips = 0  # US-41-006
            if not preserve_vpn_count:
                self.vpn_resets_used = 0  # US-37-008: Only reset for full session reset
            # US-37-009: Reset early termination state
            self.early_terminated = False
            self.early_termination_reason = None
        logger.debug(
            f"CaptionRetryBudget: reset for new batch "
            f"(vpn_resets preserved={preserve_vpn_count}, count={self.vpn_resets_used})"
        )

    def ensure_scaled(self, batch_size: int) -> bool:
        """Convenience method to ensure budget is scaled for batch (US-39-010).

        Makes it harder to forget to scale the budget by providing a single entry point
        that handles the auto_scale check internally.

        This method is idempotent - calling multiple times with the same batch_size
        has no effect after the first call.

        Args:
            batch_size: Number of videos in the batch.

        Returns:
            True if scaling occurred, False if already scaled or auto_scale disabled.

        Example:
            >>> budget = CaptionRetryBudget()
            >>> budget.ensure_scaled(200)  # Returns True, scales from 100 to 300
            True
            >>> budget.ensure_scaled(200)  # Returns False, already scaled
            False
            >>> budget.ensure_scaled(150)  # Returns False, already scaled to higher
            False
        """
        with self._lock:
            # Check if auto_scale is disabled
            if not self.auto_scale:
                logger.debug(
                    f"CaptionRetryBudget.ensure_scaled: auto_scale disabled, "
                    f"not scaling for batch_size={batch_size}"
                )
                return False

            # Check if already scaled for this batch size
            if self.batch_size == batch_size:
                logger.debug(
                    f"CaptionRetryBudget.ensure_scaled: already scaled for "
                    f"batch_size={batch_size}, skipping"
                )
                return False

            # Calculate required attempts for this batch
            required_attempts = int(batch_size * self.attempts_per_video + 0.5)

            # Only scale if required exceeds current max
            if required_attempts <= self.max_attempts:
                # Still record batch_size for tracking even if no scaling needed
                self.batch_size = batch_size
                logger.debug(
                    f"CaptionRetryBudget.ensure_scaled: max_attempts={self.max_attempts} "
                    f"sufficient for batch_size={batch_size} (required={required_attempts})"
                )
                return False

            # Perform scaling
            old_max = self.max_attempts
            self.max_attempts = required_attempts
            self.batch_size = batch_size

            # US-41-009: Reset counters on scale-up if configured
            if self.reset_on_scale:
                cleared_attempts = self.attempts
                self.attempts = 0
                self.failures = 0
                self.successes = 0
                self.backoff_time_spent = 0.0
                self.videos_skipped = 0
                self.error_counts.clear()
                self.attempts_per_video_id.clear()
                self.circuit_breaker_trips = 0
                self.early_terminated = False
                self.early_termination_reason = None
                logger.info(
                    f"CaptionRetryBudget.ensure_scaled: Budget scaled and reset: "
                    f"{cleared_attempts} attempts cleared (scaled from {old_max} to {self.max_attempts})"
                )
            else:
                logger.info(
                    f"CaptionRetryBudget.ensure_scaled: scaled max_attempts from {old_max} "
                    f"to {self.max_attempts} for batch of {batch_size} videos"
                )
            return True

    def verify_budget_sufficient(self, batch_size: int) -> None:
        """Verify budget is mathematically sufficient for batch (US-41-004).

        Fail-fast verification after ensure_scaled() to catch configuration errors
        before processing begins. Raises ValueError if budget is insufficient.

        Args:
            batch_size: Number of videos to process.

        Raises:
            ValueError: If max_attempts < batch_size * attempts_per_video
        """
        required_attempts = int(batch_size * self.attempts_per_video + 0.5)

        if self.max_attempts < required_attempts:
            raise ValueError(
                f"Retry budget insufficient: max_attempts={self.max_attempts} < required "
                f"{required_attempts} (batch_size={batch_size} × attempts_per_video="
                f"{self.attempts_per_video}). Either enable auto_scale=true in "
                f"config.yaml under download.caption_first.retry_budget, or increase "
                f"max_attempts to at least {required_attempts}."
            )

        # Warn if auto_scale is disabled and batch exceeds original max_attempts
        if not self.auto_scale and batch_size > self.max_attempts:
            logger.warning(
                f"[US-41-004] Retry budget auto_scale DISABLED: batch_size={batch_size} > "
                f"max_attempts={self.max_attempts}. Budget may exhaust before all videos "
                f"are processed. Enable auto_scale in config.yaml or increase max_attempts."
            )

    def scale_to_batch_size(self, batch_size: int, attempts_per_video: Optional[float] = None) -> int:
        """Scale max_attempts proportionally to batch size (US-37-003).

        The default max_attempts of 100 is insufficient for large batches (175+ videos).
        Budget exhausts at video 101, skipping remaining videos. This method scales
        the limit so each video has ~1.5 attempts on average (1 attempt + 0.5 retries).

        Args:
            batch_size: Number of videos in the batch.
            attempts_per_video: Average attempts per video. Defaults to self.attempts_per_video
                (from config, typically 1.5 = 1 + 0.5 retries).

        Returns:
            The new max_attempts value (for logging/testing convenience).

        Example:
            - 50 videos -> max_attempts stays at 100 (50 * 1.5 = 75 < 100)
            - 175 videos -> max_attempts becomes 263 (175 * 1.5 = 262.5, rounded up)
        """
        with self._lock:
            # Use instance config value if not overridden
            multiplier = attempts_per_video if attempts_per_video is not None else self.attempts_per_video

            # Calculate required attempts for this batch
            required_attempts = int(batch_size * multiplier + 0.5)  # Round up

            # Track batch size for summary logging (US-38-009)
            self.batch_size = batch_size

            # Only scale UP, never reduce below default
            if required_attempts > self.max_attempts:
                old_max = self.max_attempts
                self.max_attempts = required_attempts
                logger.info(
                    f"CaptionRetryBudget: scaled max_attempts from {old_max} to "
                    f"{self.max_attempts} for batch of {batch_size} videos "
                    f"({multiplier:.1f} attempts/video)"
                )
            else:
                logger.debug(
                    f"CaptionRetryBudget: max_attempts {self.max_attempts} sufficient for "
                    f"{batch_size} videos (required: {required_attempts})"
                )

            return self.max_attempts


@dataclass
class BatchRetryBudget:
    """Tracks cumulative errors across all videos in a batch (US-001 Sprint 8).

    Per-category retry budgets (US-003 Sprint 7) work per-video, but don't adapt
    based on batch-wide failure patterns. When 30% of videos fail with network
    errors, retrying remaining videos at full budget wastes time.

    This class tracks cumulative errors across the batch and reduces retry budgets
    based on detected patterns:
    - >30% network errors: reduce retry budget from 3 to 1
    - >50% network errors: disable retries entirely

    Thread Safety:
        All mutation methods are protected by a Lock for concurrent access
        during parallel caption fetching.

    Attributes:
        total_videos: Total videos in the batch.
        processed_videos: Number of videos processed so far.
        category_counts: Dict mapping CaptionErrorCategory -> count of failures.
        original_budgets: Copy of per-category retry budgets at batch start.
        reduced_budgets: Current (possibly reduced) per-category retry budgets.
        budget_reductions: List of (threshold, category, old_budget, new_budget) events.
    """
    total_videos: int = 0
    processed_videos: int = 0
    category_counts: Dict[CaptionErrorCategory, int] = field(default_factory=dict)
    original_budgets: Dict[CaptionErrorCategory, int] = field(default_factory=dict)
    reduced_budgets: Dict[CaptionErrorCategory, int] = field(default_factory=dict)
    budget_reductions: List[tuple] = field(default_factory=list)

    # Threshold configuration
    network_reduce_threshold: float = 0.30  # >30% network errors -> reduce retries
    network_disable_threshold: float = 0.50  # >50% network errors -> disable retries

    # Thread-safety lock (RLock for reentrant access from nested methods)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def __post_init__(self):
        """Initialize budgets from defaults."""
        if not self.original_budgets:
            self.original_budgets = dict(DEFAULT_RETRY_BUDGETS)
        if not self.reduced_budgets:
            self.reduced_budgets = dict(self.original_budgets)

    def record_success(self, video_id: str = "") -> None:
        """Record a successful caption fetch.

        Args:
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.processed_videos += 1
        logger.debug(f"BatchRetryBudget: success recorded for {video_id or 'unknown'}")

    def record_error(self, category: CaptionErrorCategory, video_id: str = "") -> None:
        """Record a failed caption fetch and update budgets if threshold crossed.

        Args:
            category: The error category (NETWORK, TIMEOUT, etc.).
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.processed_videos += 1
            self.category_counts[category] = self.category_counts.get(category, 0) + 1

            # Check if we need to reduce budgets
            self._check_thresholds_locked(category)

        logger.debug(
            f"BatchRetryBudget: {category.name} error recorded for {video_id or 'unknown'}, "
            f"count={self.category_counts.get(category, 0)}/{self.processed_videos}"
        )

    def _check_thresholds_locked(self, category: CaptionErrorCategory) -> None:
        """Check if error rate thresholds are crossed and reduce budgets.

        Must be called while holding the lock.
        """
        if self.processed_videos == 0:
            return

        error_rate = self.category_counts.get(category, 0) / self.processed_videos

        # Only apply threshold logic to NETWORK errors (most common transient issue)
        if category == CaptionErrorCategory.NETWORK:
            current_budget = self.reduced_budgets.get(category, DEFAULT_RETRY_BUDGETS.get(category, 3))
            original_budget = self.original_budgets.get(category, DEFAULT_RETRY_BUDGETS.get(category, 3))

            # >50% threshold: disable retries entirely
            if error_rate > self.network_disable_threshold and current_budget > 0:
                self.reduced_budgets[category] = 0
                self.budget_reductions.append((
                    error_rate,
                    category,
                    current_budget,
                    0,
                    f">50% network errors ({error_rate:.1%})"
                ))
                logger.warning(
                    f"BatchRetryBudget: >50% network errors ({error_rate:.1%}), "
                    f"disabling retries (was {current_budget})"
                )

            # >30% threshold: reduce to 1 retry (if not already at 0)
            elif error_rate > self.network_reduce_threshold and current_budget > 1:
                self.reduced_budgets[category] = 1
                self.budget_reductions.append((
                    error_rate,
                    category,
                    current_budget,
                    1,
                    f">30% network errors ({error_rate:.1%})"
                ))
                logger.warning(
                    f"BatchRetryBudget: >30% network errors ({error_rate:.1%}), "
                    f"reducing retries from {current_budget} to 1"
                )

    def budget_remaining(self, category: CaptionErrorCategory) -> int:
        """Get the current retry budget for a category.

        Returns the reduced budget if thresholds have been crossed,
        otherwise returns the original budget.
        """
        with self._lock:
            return self.reduced_budgets.get(
                category,
                self.original_budgets.get(category, DEFAULT_RETRY_BUDGETS.get(category, 0))
            )

    def get_error_rate(self, category: CaptionErrorCategory) -> float:
        """Get the current error rate for a category."""
        with self._lock:
            if self.processed_videos == 0:
                return 0.0
            return self.category_counts.get(category, 0) / self.processed_videos

    def get_total_retries_saved(self) -> int:
        """Calculate total retries saved by budget reduction."""
        with self._lock:
            if not self.budget_reductions:
                return 0

            remaining_videos = max(0, self.total_videos - self.processed_videos)
            total_saved = 0

            # For each reduction, calculate retries saved
            for error_rate, category, old_budget, new_budget, _ in self.budget_reductions:
                # Estimate: if X% of videos had this error type, X% of remaining will too
                expected_errors = int(remaining_videos * error_rate)
                retries_per_video_saved = old_budget - new_budget
                total_saved += expected_errors * retries_per_video_saved

            return total_saved

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of batch retry budget state."""
        with self._lock:
            error_rates = {}
            if self.processed_videos > 0:
                for cat, count in self.category_counts.items():
                    error_rates[cat.name] = round(count / self.processed_videos, 3)

            return {
                'total_videos': self.total_videos,
                'processed_videos': self.processed_videos,
                'error_rates': error_rates,
                'original_budgets': {k.name: v for k, v in self.original_budgets.items()},
                'reduced_budgets': {k.name: v for k, v in self.reduced_budgets.items()},
                'reductions_applied': len(self.budget_reductions),
                'estimated_retries_saved': self.get_total_retries_saved(),
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize budget state for checkpoint persistence."""
        with self._lock:
            return {
                'total_videos': self.total_videos,
                'processed_videos': self.processed_videos,
                'category_counts': {k.name: v for k, v in self.category_counts.items()},
                'original_budgets': {k.name: v for k, v in self.original_budgets.items()},
                'reduced_budgets': {k.name: v for k, v in self.reduced_budgets.items()},
                'budget_reductions': [
                    (rate, cat.name, old, new, reason)
                    for rate, cat, old, new, reason in self.budget_reductions
                ],
            }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "BatchRetryBudget":
        """Create budget from checkpoint data."""
        if not data:
            return cls()

        budget = cls(
            total_videos=data.get('total_videos', 0),
            processed_videos=data.get('processed_videos', 0),
        )

        # Restore category counts
        for name, count in data.get('category_counts', {}).items():
            try:
                cat = CaptionErrorCategory[name]
                budget.category_counts[cat] = count
            except KeyError:
                logger.warning(f"Unknown error category in checkpoint: {name}")

        # Restore budgets
        for name, value in data.get('original_budgets', {}).items():
            try:
                cat = CaptionErrorCategory[name]
                budget.original_budgets[cat] = value
            except KeyError:
                pass

        for name, value in data.get('reduced_budgets', {}).items():
            try:
                cat = CaptionErrorCategory[name]
                budget.reduced_budgets[cat] = value
            except KeyError:
                pass

        # Restore reduction history
        for rate, cat_name, old, new, reason in data.get('budget_reductions', []):
            try:
                cat = CaptionErrorCategory[cat_name]
                budget.budget_reductions.append((rate, cat, old, new, reason))
            except KeyError:
                pass

        return budget

    def reset(self) -> None:
        """Reset budget state for a new batch."""
        with self._lock:
            self.processed_videos = 0
            self.category_counts.clear()
            self.reduced_budgets = dict(self.original_budgets)
            self.budget_reductions.clear()
        logger.debug("BatchRetryBudget: reset for new batch")
