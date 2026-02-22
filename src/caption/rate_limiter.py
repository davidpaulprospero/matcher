"""
Unified rate limiter with jitter for caption fetching.

Provides exponential backoff with configurable jitter to improve caption
fetch reliability and avoid thundering herd problems when multiple workers
hit rate limits simultaneously.

Example:
    from src.caption.rate_limiter import UnifiedCaptionRateLimiter
    from src.caption_timeout_manager import RateLimitTracker

    limiter = UnifiedCaptionRateLimiter()
    tracker = RateLimitTracker()

    # Before each request
    delay = limiter.get_backoff_delay(attempt=1)
    if delay > 0:
        time.sleep(delay)

    # On rate limit - update both limiter and tracker
    limiter.record_rate_limit()
    tracker.record_rate_limit(video_id)

Integration with RateLimitTracker:
    The UnifiedCaptionRateLimiter handles delay calculation with jitter,
    while RateLimitTracker from caption_timeout_manager.py handles global
    state tracking across workers. Use both together for full functionality.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.caption_timeout_manager import RateLimitTracker

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting behavior.

    Attributes:
        enabled: Whether rate limiting is active
        base_delay_seconds: Base delay for exponential backoff
        max_delay_seconds: Maximum delay cap
        jitter_factor: Jitter range (0.0-1.0), 0.3 means ±30%
        format_backoff_multipliers: Per-format backoff multipliers (US-90-008)
            Format-specific multiplier applied when calculating backoff for that format.
            Example: {'json3': 1.5, 'srv3': 2.0, 'vtt': 2.5}
    """
    enabled: bool = True
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 120.0
    jitter_factor: float = 0.3
    format_backoff_multipliers: dict = None

    def __post_init__(self):
        """Validate configuration values."""
        if self.base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be positive")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if not 0.0 <= self.jitter_factor <= 1.0:
            raise ValueError("jitter_factor must be between 0.0 and 1.0")
        # Initialize format_backoff_multipliers if None
        if self.format_backoff_multipliers is None:
            self.format_backoff_multipliers = {}


class UnifiedCaptionRateLimiter:
    """Unified rate limiter with exponential backoff and jitter.

    Coordinates rate limiting for caption fetching with:
    - Exponential backoff (base 2s, max 120s by default)
    - Jitter to avoid thundering herd (±30% by default)
    - Thread-safe state management
    - Integration with RateLimitTracker

    Example:
        limiter = UnifiedCaptionRateLimiter()

        for attempt in range(max_retries):
            delay = limiter.get_backoff_delay(attempt)
            if delay > 0:
                time.sleep(delay)

            try:
                result = fetch_captions(video_id)
                limiter.record_success()
                break
            except RateLimitError:
                limiter.record_rate_limit()
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration (uses defaults if None)
        """
        self._config = config or RateLimitConfig()
        self._consecutive_rate_limits = 0
        self._total_rate_limits = 0
        self._last_rate_limit_time: float = 0.0
        self._lock = threading.Lock()
        # Per-format rate limit tracking (US-90-008)
        self._format_rate_limits: dict[str, int] = {}  # format -> consecutive count
        self._format_total_limits: dict[str, int] = {}  # format -> total count

    @property
    def config(self) -> RateLimitConfig:
        """Get current configuration."""
        return self._config

    @property
    def consecutive_rate_limits(self) -> int:
        """Get consecutive rate limit count (thread-safe)."""
        with self._lock:
            return self._consecutive_rate_limits

    @property
    def total_rate_limits(self) -> int:
        """Get total rate limit count (thread-safe)."""
        with self._lock:
            return self._total_rate_limits

    def _apply_jitter(self, delay: float) -> float:
        """Apply jitter to delay value.

        Args:
            delay: Base delay value

        Returns:
            Delay with jitter applied (delay * (1 ± jitter_factor))
        """
        if self._config.jitter_factor == 0.0:
            return delay

        # Random value between -jitter_factor and +jitter_factor
        jitter = random.uniform(-self._config.jitter_factor, self._config.jitter_factor)
        jittered = delay * (1.0 + jitter)

        # Ensure we don't go below 0 or above max
        return max(0.0, min(jittered, self._config.max_delay_seconds))

    def calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff delay for given attempt (no jitter).

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds before next attempt
        """
        if not self._config.enabled:
            return 0.0

        if attempt <= 0:
            return 0.0

        # Exponential backoff: base * 2^(attempt-1)
        delay = self._config.base_delay_seconds * (2 ** (attempt - 1))
        return min(delay, self._config.max_delay_seconds)

    def get_backoff_delay(self, attempt: int) -> float:
        """Get backoff delay with jitter for given attempt.

        Args:
            attempt: Attempt number (0-indexed, 0 = first attempt)

        Returns:
            Delay in seconds before next attempt (with jitter)
        """
        base_delay = self.calculate_backoff(attempt)
        return self._apply_jitter(base_delay)

    def get_rate_limit_delay(self) -> float:
        """Get delay based on consecutive rate limits.

        Uses internal consecutive count rather than attempt number.

        Returns:
            Delay in seconds (with jitter)
        """
        with self._lock:
            consecutive = self._consecutive_rate_limits

        base_delay = self.calculate_backoff(consecutive)
        return self._apply_jitter(base_delay)

    # Per-format rate limit methods (US-90-008)
    def calculate_format_backoff(self, attempt: int, format_name: str = "") -> float:
        """Calculate backoff delay with format-specific multiplier.

        Args:
            attempt: Attempt number (0-indexed)
            format_name: Caption format (e.g., 'json3', 'srv3', 'vtt')

        Returns:
            Delay in seconds with format-specific multiplier applied
        """
        if not self._config.enabled:
            return 0.0

        if attempt <= 0:
            return 0.0

        # Get format-specific multiplier or use default (2.0)
        multiplier = self._config.format_backoff_multipliers.get(format_name, 2.0)

        # Exponential backoff with format-specific multiplier
        delay = self._config.base_delay_seconds * (multiplier ** (attempt - 1))
        return min(delay, self._config.max_delay_seconds)

    def get_format_backoff_delay(self, format_name: str = "") -> float:
        """Get backoff delay for format with format-specific multiplier.

        Args:
            format_name: Caption format (e.g., 'json3', 'srv3', 'vtt')

        Returns:
            Delay in seconds with format-specific multiplier and jitter
        """
        with self._lock:
            consecutive = self._format_rate_limits.get(format_name, 0)

        base_delay = self.calculate_format_backoff(consecutive, format_name)
        return self._apply_jitter(base_delay)

    def record_format_rate_limit(self, format_name: str = "", video_id: str = "") -> float:
        """Record a rate limit event for a specific caption format.

        Args:
            format_name: Caption format (e.g., 'json3', 'srv3', 'vtt')
            video_id: Video ID that triggered rate limit (for logging)

        Returns:
            Recommended delay before next request
        """
        with self._lock:
            # Update global counter
            self._consecutive_rate_limits += 1
            self._total_rate_limits += 1
            self._last_rate_limit_time = time.time()

            # Update format-specific counter
            current_count = self._format_rate_limits.get(format_name, 0)
            self._format_rate_limits[format_name] = current_count + 1

            total_count = self._format_total_limits.get(format_name, 0)
            self._format_total_limits[format_name] = total_count + 1

            consecutive = self._format_rate_limits[format_name]

        delay = self.get_format_backoff_delay(format_name)

        logger.warning(
            f"Format rate limit recorded: format={format_name}, consecutive={consecutive}, "
            f"delay={delay:.2f}s, video={video_id or 'unknown'}"
        )

        return delay

    def record_rate_limit(self, video_id: str = "", format_name: str = "") -> float:
        """Record a rate limit event.

        Args:
            video_id: Video ID that triggered rate limit (for logging)
            format_name: Caption format (e.g., 'json3', 'srv3', 'vtt')

        Returns:
            Recommended delay before next request
        """
        # If format provided, use format-specific tracking
        if format_name:
            return self.record_format_rate_limit(format_name, video_id)

        with self._lock:
            self._consecutive_rate_limits += 1
            self._total_rate_limits += 1
            self._last_rate_limit_time = time.time()
            consecutive = self._consecutive_rate_limits

        delay = self.get_rate_limit_delay()

        logger.warning(
            f"Rate limit recorded: consecutive={consecutive}, "
            f"delay={delay:.2f}s, video={video_id or 'unknown'}"
        )

        return delay

    def record_success(self) -> None:
        """Record a successful request to reset consecutive counter."""
        with self._lock:
            if self._consecutive_rate_limits > 0:
                logger.debug(
                    f"Rate limit state reset (was {self._consecutive_rate_limits} consecutive)"
                )
            self._consecutive_rate_limits = 0

    def reset(self) -> None:
        """Reset all rate limit state."""
        with self._lock:
            self._consecutive_rate_limits = 0
            self._total_rate_limits = 0
            self._last_rate_limit_time = 0.0
            self._format_rate_limits = {}
            self._format_total_limits = {}

    def get_state(self) -> dict:
        """Get current state for metrics/logging.

        Returns:
            Dictionary with rate limit state information
        """
        with self._lock:
            return {
                'enabled': self._config.enabled,
                'consecutive_rate_limits': self._consecutive_rate_limits,
                'total_rate_limits': self._total_rate_limits,
                'last_rate_limit_time': self._last_rate_limit_time,
                'base_delay_seconds': self._config.base_delay_seconds,
                'max_delay_seconds': self._config.max_delay_seconds,
                'jitter_factor': self._config.jitter_factor,
                'format_rate_limits': dict(self._format_rate_limits),
                'format_total_limits': dict(self._format_total_limits),
            }

    def save_state(self) -> dict:
        """Get state for persistence across pipeline runs.

        Returns:
            Dictionary with rate limit state that can be saved to checkpoint/cache
        """
        with self._lock:
            return {
                'consecutive_rate_limits': self._consecutive_rate_limits,
                'total_rate_limits': self._total_rate_limits,
                'last_rate_limit_time': self._last_rate_limit_time,
                'format_rate_limits': dict(self._format_rate_limits),
                'format_total_limits': dict(self._format_total_limits),
            }

    def load_state(self, state: dict) -> None:
        """Load rate limit state from persistence.

        Args:
            state: Dictionary with rate limit state (from save_state)
        """
        with self._lock:
            self._consecutive_rate_limits = state.get('consecutive_rate_limits', 0)
            self._total_rate_limits = state.get('total_rate_limits', 0)
            self._last_rate_limit_time = state.get('last_rate_limit_time', 0.0)
            self._format_rate_limits = state.get('format_rate_limits', {})
            self._format_total_limits = state.get('format_total_limits', {})

        logger.info(
            f"Loaded rate limit state: consecutive={self._consecutive_rate_limits}, "
            f"total={self._total_rate_limits}, formats={list(self._format_rate_limits.keys())}"
        )

    def wait_if_needed(self) -> float:
        """Wait based on current rate limit state.

        Returns:
            Seconds waited
        """
        delay = self.get_rate_limit_delay()
        if delay > 0:
            logger.info(f"Rate limiter waiting {delay:.2f}s")
            time.sleep(delay)
        return delay


def create_rate_limiter_from_config(config_dict: Optional[dict] = None) -> UnifiedCaptionRateLimiter:
    """Factory function to create rate limiter from config dictionary.

    Args:
        config_dict: Configuration dictionary with rate_limit section

    Returns:
        Configured UnifiedCaptionRateLimiter instance
    """
    if not config_dict:
        return UnifiedCaptionRateLimiter()

    rate_limit_config = config_dict.get('rate_limit', {})

    config = RateLimitConfig(
        enabled=rate_limit_config.get('enabled', True),
        base_delay_seconds=rate_limit_config.get('base_delay_seconds', 2.0),
        max_delay_seconds=rate_limit_config.get('max_delay_seconds', 120.0),
        jitter_factor=rate_limit_config.get('jitter_factor', 0.3),
        format_backoff_multipliers=rate_limit_config.get('format_backoff_multipliers', {}),
    )

    return UnifiedCaptionRateLimiter(config)


class IntegratedRateLimiter:
    """Rate limiter that integrates with RateLimitTracker.

    Combines UnifiedCaptionRateLimiter (backoff calculation with jitter) with
    RateLimitTracker (global state across workers) for coordinated rate limiting.

    Example:
        from src.caption_timeout_manager import RateLimitTracker

        tracker = RateLimitTracker()
        limiter = IntegratedRateLimiter(tracker=tracker)

        # Before fetch
        limiter.wait_if_needed()

        try:
            result = fetch_captions(video_id)
            limiter.record_success()
        except RateLimitError:
            delay = limiter.record_rate_limit(video_id)
            time.sleep(delay)
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        tracker: Optional['RateLimitTracker'] = None,
    ):
        """Initialize integrated rate limiter.

        Args:
            config: Rate limit configuration
            tracker: Optional RateLimitTracker for global state coordination
        """
        self._limiter = UnifiedCaptionRateLimiter(config)
        self._tracker = tracker

    @property
    def limiter(self) -> UnifiedCaptionRateLimiter:
        """Get underlying rate limiter."""
        return self._limiter

    @property
    def tracker(self) -> Optional['RateLimitTracker']:
        """Get underlying tracker (if set)."""
        return self._tracker

    def set_tracker(self, tracker: 'RateLimitTracker') -> None:
        """Set or update the rate limit tracker."""
        self._tracker = tracker

    def should_pause(self) -> bool:
        """Check if requests should pause (from tracker or limiter state)."""
        if self._tracker and self._tracker.should_pause():
            return True
        return self._limiter.consecutive_rate_limits > 0

    def get_delay(self) -> float:
        """Get recommended delay with jitter.

        Uses tracker's recommended delay if available, otherwise calculates
        from limiter state with jitter applied.
        """
        if self._tracker:
            tracker_delay = self._tracker.get_recommended_delay()
            if tracker_delay > 0:
                return self._limiter._apply_jitter(tracker_delay)
        return self._limiter.get_rate_limit_delay()

    def wait_if_needed(self) -> float:
        """Wait if backoff is in effect.

        Returns:
            Seconds waited
        """
        delay = self.get_delay()
        if delay > 0:
            logger.info(f"Integrated rate limiter waiting {delay:.2f}s")
            time.sleep(delay)
        return delay

    def record_rate_limit(self, video_id: str = "", format_name: str = "") -> float:
        """Record a rate limit event in both limiter and tracker.

        Args:
            video_id: Video ID for logging/tracking
            format_name: Caption format (e.g., 'json3', 'srv3', 'vtt')

        Returns:
            Recommended delay before next request (with jitter)
        """
        # Record in limiter for jittered delay calculation
        delay = self._limiter.record_rate_limit(video_id, format_name)

        # Also record in tracker for global state
        if self._tracker:
            self._tracker.record_rate_limit(video_id)

        return delay

    def record_success(self) -> None:
        """Record success in both limiter and tracker."""
        self._limiter.record_success()
        if self._tracker:
            self._tracker.record_success()

    def reset(self) -> None:
        """Reset both limiter and tracker state."""
        self._limiter.reset()
        # Note: RateLimitTracker is a singleton, so we don't reset it here

    def get_state(self) -> dict:
        """Get combined state from limiter and tracker."""
        state = self._limiter.get_state()
        if self._tracker:
            state['tracker'] = self._tracker.get_state_summary()
        return state

    def save_state(self) -> dict:
        """Get state for persistence across pipeline runs."""
        return self._limiter.save_state()

    def load_state(self, state: dict) -> None:
        """Load rate limit state from persistence."""
        self._limiter.load_state(state)
