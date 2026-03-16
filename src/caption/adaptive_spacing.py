"""Adaptive request spacing for caption batch processing.

Adjusts delay between caption fetch requests based on error rate to avoid
triggering rate limits during batch processing.

Part of US-33-004: Add adaptive request spacing to caption batch processing.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSpacingConfig:
    """Configuration for adaptive request spacing.

    Attributes:
        min_interval_ms: Minimum interval between requests in milliseconds (default: 500)
        max_interval_ms: Maximum interval between requests in milliseconds (default: 5000)
        error_threshold_high: Error rate above which spacing is doubled (default: 0.2 = 20%)
        error_threshold_low: Error rate below which spacing is halved (default: 0.05 = 5%)
        window_size_error: Number of requests to track for error rate calculation (default: 10)
        window_size_success: Number of successful requests needed before halving (default: 20)
    """
    min_interval_ms: int = 500
    max_interval_ms: int = 5000
    error_threshold_high: float = 0.2  # 20% error rate triggers increase
    error_threshold_low: float = 0.05  # 5% error rate allows decrease
    window_size_error: int = 10  # Window for error rate calculation
    window_size_success: int = 20  # Window for success-based decrease


@dataclass
class RequestResult:
    """Result of a single caption fetch request.

    Attributes:
        success: True if request succeeded, False if it failed
        timestamp: Unix timestamp when request completed
        error_type: Type of error if failed (e.g., 'rate_limit', 'timeout')
    """
    success: bool
    timestamp: float
    error_type: Optional[str] = None


class AdaptiveSpacingController:
    """Controls request spacing based on error rate patterns.

    Implements adaptive delay between caption fetch requests:
    - When error rate >20% in last 10 requests, doubles the spacing (up to max)
    - When error rate <5% for 20 requests, halves the spacing (down to min)

    Example:
        controller = AdaptiveSpacingController(min_interval_ms=500)

        # Before each request
        delay = controller.get_delay_seconds()
        time.sleep(delay)

        # After each request
        controller.record_result(success=True)  # or False on error

        # Get current stats
        stats = controller.get_stats()

    Attributes:
        config: Adaptive spacing configuration
        _current_interval_ms: Current interval between requests in milliseconds
        _results: Deque of recent request results for tracking
        _last_request_time: Timestamp of last request
        _consecutive_success_count: Counter for consecutive successes
    """

    def __init__(
        self,
        min_interval_ms: int = 500,
        max_interval_ms: int = 5000,
        config: Optional[AdaptiveSpacingConfig] = None
    ):
        """Initialize adaptive spacing controller.

        Args:
            min_interval_ms: Minimum interval between requests in milliseconds
            max_interval_ms: Maximum interval between requests in milliseconds
            config: Full configuration object (overrides min/max if provided)
        """
        if config is not None:
            self.config = config
        else:
            self.config = AdaptiveSpacingConfig(
                min_interval_ms=min_interval_ms,
                max_interval_ms=max_interval_ms
            )

        self._current_interval_ms = self.config.min_interval_ms
        self._results: Deque[RequestResult] = deque(
            maxlen=max(self.config.window_size_error, self.config.window_size_success)
        )
        self._last_request_time: float = 0.0
        self._consecutive_success_count: int = 0

    @property
    def current_interval_ms(self) -> int:
        """Get current interval between requests in milliseconds."""
        return self._current_interval_ms

    def get_delay_seconds(self) -> float:
        """Get delay to wait before next request.

        Calculates delay based on current interval and time since last request.
        Returns 0 if enough time has already passed.

        Returns:
            Delay in seconds to wait before making next request.
        """
        if self._last_request_time == 0:
            return 0.0

        elapsed_ms = (time.time() - self._last_request_time) * 1000
        remaining_ms = self._current_interval_ms - elapsed_ms

        if remaining_ms <= 0:
            return 0.0

        return remaining_ms / 1000.0

    def record_request_start(self) -> None:
        """Record that a request is starting.

        Call this just before making a request to track timing.
        """
        self._last_request_time = time.time()

    def record_result(self, success: bool, error_type: Optional[str] = None) -> None:
        """Record the result of a caption fetch request.

        Tracks request results and adjusts spacing based on error patterns.

        Args:
            success: True if request succeeded, False if it failed
            error_type: Type of error if failed (e.g., 'rate_limit', 'timeout', '429')
        """
        result = RequestResult(
            success=success,
            timestamp=time.time(),
            error_type=error_type if not success else None
        )
        self._results.append(result)

        if success:
            self._consecutive_success_count += 1
        else:
            self._consecutive_success_count = 0

        # Adjust spacing based on patterns
        self._adjust_spacing()

    def _adjust_spacing(self) -> None:
        """Adjust request spacing based on error rate patterns."""
        # Check for high error rate (increase spacing)
        if len(self._results) >= self.config.window_size_error:
            recent_results = list(self._results)[-self.config.window_size_error:]
            error_count = sum(1 for r in recent_results if not r.success)
            error_rate = error_count / len(recent_results)

            if error_rate > self.config.error_threshold_high:
                old_interval = self._current_interval_ms
                new_interval = min(
                    self._current_interval_ms * 2,
                    self.config.max_interval_ms
                )
                if new_interval > old_interval:
                    self._current_interval_ms = new_interval
                    logger.warning(
                        f"High error rate ({error_rate:.1%}) detected - "
                        f"increasing request spacing: {old_interval}ms → {new_interval}ms"
                    )
                return  # Don't decrease in same cycle

        # Check for sustained success (decrease spacing)
        if len(self._results) >= self.config.window_size_success:
            recent_results = list(self._results)[-self.config.window_size_success:]
            error_count = sum(1 for r in recent_results if not r.success)
            error_rate = error_count / len(recent_results)

            if error_rate < self.config.error_threshold_low:
                old_interval = self._current_interval_ms
                new_interval = max(
                    self._current_interval_ms // 2,
                    self.config.min_interval_ms
                )
                if new_interval < old_interval:
                    self._current_interval_ms = new_interval
                    logger.info(
                        f"Low error rate ({error_rate:.1%}) - "
                        f"decreasing request spacing: {old_interval}ms → {new_interval}ms"
                    )

    def get_error_rate(self, window_size: Optional[int] = None) -> float:
        """Calculate error rate over recent requests.

        Args:
            window_size: Number of recent requests to consider (default: window_size_error)

        Returns:
            Error rate as a float between 0.0 and 1.0
        """
        if not self._results:
            return 0.0

        window = window_size or self.config.window_size_error
        recent_results = list(self._results)[-window:]

        if not recent_results:
            return 0.0

        error_count = sum(1 for r in recent_results if not r.success)
        return error_count / len(recent_results)

    def get_stats(self) -> dict:
        """Get current statistics for monitoring.

        Returns:
            Dictionary with:
            - current_interval_ms: Current request spacing
            - min_interval_ms: Minimum allowed spacing
            - max_interval_ms: Maximum allowed spacing
            - total_requests: Total requests tracked
            - error_rate: Error rate over error window
            - consecutive_successes: Number of consecutive successful requests
        """
        return {
            'current_interval_ms': self._current_interval_ms,
            'min_interval_ms': self.config.min_interval_ms,
            'max_interval_ms': self.config.max_interval_ms,
            'total_requests': len(self._results),
            'error_rate': self.get_error_rate(),
            'consecutive_successes': self._consecutive_success_count,
        }

    def reset(self) -> None:
        """Reset controller to initial state."""
        self._current_interval_ms = self.config.min_interval_ms
        self._results.clear()
        self._last_request_time = 0.0
        self._consecutive_success_count = 0
