"""Per-channel circuit breaker for YouTube API rate limit management.

Implements per-channel failure tracking to isolate rate-limited channels
while allowing other channels to continue. Similar to PerKeywordCircuitBreaker
but focused on channel-level rate limiting for video metadata queries.

US-155-010: Per-channel API usage tracking and limits.

Key features:
- Per-channel failure tracking: each channel has its own circuit breaker
- Per-channel pause duration: based on channel's own failure history
- Query distribution metrics: track which channels are queried most
- Graceful handling: channels with no published videos don't trigger circuit
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from src.common.circuit_breaker_base import CircuitBreakerBase, CircuitBreakerStateBase

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreakerCoordinator

logger = logging.getLogger(__name__)


def _mock_sleep(delay: float) -> None:
    """Apply mock delay for circuit breaker pauses in test mode.

    When MOCK_RATE_LIMITS=1 env var is set, sleeps for 0.01s instead of actual delay.

    Args:
        delay: The intended delay in seconds (used for logging)
    """
    env_value = os.environ.get('MOCK_RATE_LIMITS', '').lower()
    if env_value in ('1', 'true', 'yes'):
        mock_delay = float(os.environ.get('MOCK_DELAY_SECONDS', '0.01'))
        if mock_delay > 0:
            logger.debug(f"Mock circuit breaker: sleeping {mock_delay:.3f}s instead of {delay:.1f}s")
            time.sleep(mock_delay)
    else:
        time.sleep(delay)


@dataclass
class PerChannelCircuitBreakerConfig:
    """Configuration for per-channel circuit breaker.

    Attributes:
        enabled: Enable per-channel circuit breaker tracking
        consecutive_failures_threshold: Failures per channel before its circuit trips
        pause_seconds: Base pause duration per channel (can be scaled by history)
        max_pause_seconds: Maximum pause cap per channel
        jitter_factor: Random jitter factor (0.0 to 1.0)
        enable_recovery: Enable gradual traffic increase after circuit closes
        recovery_max_requests: Maximum requests allowed during recovery phase
        recovery_backoff_base: Base for exponential backoff during recovery
        recovery_success_threshold: Consecutive successes needed to consider recovery complete
        per_channel_requests_per_minute: Maximum API requests per channel per minute (default: 60)
        throttle_warning_threshold: Warning threshold as fraction of limit (default: 0.8 = 80%)
    """
    enabled: bool = True
    consecutive_failures_threshold: int = 5
    pause_seconds: float = 60.0
    max_pause_seconds: float = 300.0
    jitter_factor: float = 0.2
    enable_recovery: bool = True
    recovery_max_requests: int = 3
    recovery_backoff_base: float = 2.0
    recovery_max_backoff: float = 8.0
    recovery_success_threshold: int = 2
    per_channel_requests_per_minute: int = 60
    throttle_warning_threshold: float = 0.8


@dataclass
class ChannelCircuitState:
    """State for a single channel's circuit breaker."""
    consecutive_failures: int = 0
    is_open: bool = False
    opened_at: Optional[float] = None
    total_trips: int = 0
    total_paused_seconds: float = 0.0
    pause_history: List[float] = field(default_factory=list)
    # Recovery state
    is_recovering: bool = False
    recovery_attempts: int = 0
    recovery_requests_made: int = 0
    recovery_consecutive_successes: int = 0
    recovery_backoff_multiplier: float = 1.0
    recovery_successes: int = 0
    recovery_failures: int = 0


class PerChannelCircuitBreaker:
    """Per-channel circuit breaker for YouTube API.

    Tracks failures independently per channel and provides per-channel pause
    durations based on each channel's history. Prevents rate-limiting specific
    channels while allowing others to continue.

    Usage:
        cb = PerChannelCircuitBreaker(config)

        # Before querying a channel
        cb.check_and_wait("UC123456")

        # After query
        if query_failed:
            cb.record_failure("UC123456")
        else:
            cb.record_success("UC123456")

        # Check channel status
        is_open = cb.is_channel_open("UC123456")
        stats = cb.get_channel_stats("UC123456")

        # Get overall metrics
        metrics = cb.get_metrics()
    """

    def __init__(self, config: Optional[PerChannelCircuitBreakerConfig] = None):
        """Initialize the per-channel circuit breaker.

        Args:
            config: Configuration for the circuit breaker. Uses defaults if not provided.
        """
        self.config = config or PerChannelCircuitBreakerConfig()
        self._channel_states: Dict[str, ChannelCircuitState] = {}
        self._channel_call_counts: Dict[str, int] = {}  # Track API calls per channel
        self._channel_latencies: Dict[str, List[float]] = {}  # Track latency per channel
        self._channel_errors: Dict[str, int] = {}  # Track errors per channel
        self._queried_channels: Set[str] = set()  # Track which channels have been queried
        self._channels_with_no_videos: Set[str] = set()  # Track channels with no videos

        # Per-minute call tracking for rate limiting
        self._channel_call_timestamps: Dict[str, List[float]] = {}  # Timestamps of calls per channel
        self._throttle_warnings_issued: Dict[str, bool] = {}  # Track if warning was issued this minute

        # Global stats
        self._total_trips: int = 0
        self._total_paused_seconds: float = 0.0

        self._lock = None
        # Import threading lazily to avoid issues in some environments
        try:
            import threading
            self._lock = threading.RLock()
        except Exception:
            logger.warning("Could not initialize lock for PerChannelCircuitBreaker")

    def _get_lock(self):
        """Get lock for thread safety."""
        return self._lock or _DummyLock()

    def check_and_wait(self, channel_id: str) -> None:
        """Check channel circuit state and wait if circuit is open.

        Args:
            channel_id: The YouTube channel ID to check
        """
        if not self.config.enabled:
            return

        with self._get_lock():
            state = self._get_or_create_state(channel_id)

            if state.is_open:
                # Check if we should transition to half-open (recovery mode)
                if state.is_recovering:
                    if state.recovery_requests_made >= self.config.recovery_max_requests:
                        # Too many requests during recovery, reopen
                        logger.warning(
                            f"Channel {channel_id}: recovery requests exhausted, reopening circuit"
                        )
                        state.is_open = True
                        state.is_recovering = False
                        state.opened_at = time.time()
                        state.recovery_attempts += 1
                        state.recovery_failures += 1
                        state.recovery_requests_made = 0
                        state.recovery_consecutive_successes = 0
                        return

                    # Allow request during recovery
                    state.recovery_requests_made += 1
                    return

                # Circuit is open, calculate remaining pause time
                if state.opened_at is not None:
                    elapsed = time.time() - state.opened_at
                    if elapsed < self._calculate_pause_duration(state):
                        # Still in pause period
                        remaining = self._calculate_pause_duration(state) - elapsed
                        logger.info(
                            f"Channel {channel_id}: circuit open, waiting {remaining:.1f}s "
                            f"(opened {elapsed:.1f}s ago)"
                        )
                        _mock_sleep(remaining)
                    else:
                        # Transition to half-open (recovery mode) if enabled
                        if self.config.enable_recovery:
                            state.is_recovering = True
                            state.is_open = False
                            state.recovery_backoff_multiplier = 1.0
                            state.recovery_consecutive_successes = 0
                            logger.info(
                                f"Channel {channel_id}: circuit transitioning to recovery mode"
                            )
                        else:
                            # Close the circuit
                            state.is_open = False
                            state.consecutive_failures = 0
                            logger.info(f"Channel {channel_id}: circuit closed")

    def channel_level_throttle(self, channel_id: str) -> None:
        """Check and throttle channel if it exceeds per-minute request limit.

        This method should be called before making an API request to a channel.
        It tracks the current rate of requests and pauses if the channel has
        exceeded its per-minute quota.

        Args:
            channel_id: The YouTube channel ID to check
        """
        if not self.config.enabled:
            return

        with self._get_lock():
            # Clean up old timestamps (older than 1 minute)
            self._cleanup_old_timestamps(channel_id)

            # Add timestamp first so we count this request
            if channel_id not in self._channel_call_timestamps:
                self._channel_call_timestamps[channel_id] = []
            self._channel_call_timestamps[channel_id].append(time.time())

            # Track call count for channel usage metrics
            self._channel_call_counts[channel_id] = self._channel_call_counts.get(channel_id, 0) + 1
            self._queried_channels.add(channel_id)

            # Get count AFTER adding this request
            current_count = len(self._channel_call_timestamps[channel_id])
            limit = self.config.per_channel_requests_per_minute
            warning_threshold = int(limit * self.config.throttle_warning_threshold)

            # Check if warning threshold is reached and not yet warned
            if current_count >= warning_threshold and not self._throttle_warnings_issued.get(channel_id, False):
                logger.warning(
                    f"Channel {channel_id}: approaching throttling limit "
                    f"({current_count}/{limit} requests this minute, {self.config.throttle_warning_threshold*100:.0f}% threshold)"
                )
                self._throttle_warnings_issued[channel_id] = True

            # If we've exceeded the limit, throttle
            if current_count >= limit:
                # Calculate how long to wait until oldest call expires (60 seconds)
                timestamps = self._channel_call_timestamps.get(channel_id, [])
                if timestamps:
                    oldest_timestamp = min(timestamps)
                    wait_time = 60.0 - (time.time() - oldest_timestamp)
                    if wait_time > 0:
                        logger.info(
                            f"Channel {channel_id}: throttling at {current_count}/{limit} requests/min, "
                            f"waiting {wait_time:.1f}s"
                        )
                        _mock_sleep(wait_time)
                        # Clean up again after waiting
                        self._cleanup_old_timestamps(channel_id)

            # Ensure channel state exists for stats tracking
            self._get_or_create_state(channel_id)

    def _cleanup_old_timestamps(self, channel_id: str) -> None:
        """Remove timestamps older than 1 minute from channel's call history.

        Args:
            channel_id: The YouTube channel ID
        """
        if channel_id not in self._channel_call_timestamps:
            return

        current_time = time.time()
        cutoff = current_time - 60.0  # 1 minute ago

        # Filter out old timestamps
        self._channel_call_timestamps[channel_id] = [
            ts for ts in self._channel_call_timestamps[channel_id] if ts > cutoff
        ]

        # Reset warning flag if we're in a new minute (all timestamps are old)
        if not self._channel_call_timestamps[channel_id]:
            self._throttle_warnings_issued[channel_id] = False

    def _get_current_minute_count(self, channel_id: str) -> int:
        """Get the number of API calls in the current minute for a channel.

        Args:
            channel_id: The YouTube channel ID

        Returns:
            Number of calls in the current minute
        """
        self._cleanup_old_timestamps(channel_id)
        return len(self._channel_call_timestamps.get(channel_id, []))

    def record_success(self, channel_id: str, latency_ms: Optional[float] = None) -> None:
        """Record a successful API call for a channel.

        Args:
            channel_id: The YouTube channel ID
            latency_ms: Optional latency to record for metrics
        """
        if not self.config.enabled:
            return

        with self._get_lock():
            state = self._get_or_create_state(channel_id)

            # Track call count
            self._channel_call_counts[channel_id] = self._channel_call_counts.get(channel_id, 0) + 1

            # Track latency if provided
            if latency_ms is not None:
                if channel_id not in self._channel_latencies:
                    self._channel_latencies[channel_id] = []
                self._channel_latencies[channel_id].append(latency_ms)
                # Keep only last 100 latencies
                if len(self._channel_latencies[channel_id]) > 100:
                    self._channel_latencies[channel_id] = self._channel_latencies[channel_id][-100:]

            # Handle recovery mode
            if state.is_recovering:
                state.recovery_consecutive_successes += 1
                if state.recovery_consecutive_successes >= self.config.recovery_success_threshold:
                    # Recovery successful
                    state.is_recovering = False
                    state.recovery_successes += 1
                    state.recovery_attempts = 0
                    state.recovery_requests_made = 0
                    state.recovery_consecutive_successes = 0
                    state.consecutive_failures = 0
                    logger.info(f"Channel {channel_id}: circuit recovered successfully")
                return

            # Reset failure count on success
            if state.consecutive_failures > 0:
                logger.debug(
                    f"Channel {channel_id}: success after {state.consecutive_failures} failures"
                )
            state.consecutive_failures = 0

    def record_failure(self, channel_id: str) -> None:
        """Record a failed API call for a channel.

        Args:
            channel_id: The YouTube channel ID
        """
        if not self.config.enabled:
            return

        with self._get_lock():
            state = self._get_or_create_state(channel_id)

            # Track errors
            self._channel_errors[channel_id] = self._channel_errors.get(channel_id, 0) + 1

            # Track call count even for failures
            self._channel_call_counts[channel_id] = self._channel_call_counts.get(channel_id, 0) + 1

            # Handle recovery mode failure
            if state.is_recovering:
                state.recovery_consecutive_successes = 0
                state.recovery_backoff_multiplier = min(
                    state.recovery_backoff_multiplier * self.config.recovery_backoff_base,
                    self.config.recovery_max_backoff
                )
                logger.debug(
                    f"Channel {channel_id}: recovery request failed, backoff multiplier: "
                    f"{state.recovery_backoff_multiplier:.1f}x"
                )
                return

            state.consecutive_failures += 1

            if state.consecutive_failures >= self.config.consecutive_failures_threshold:
                # Trip the circuit
                state.is_open = True
                state.opened_at = time.time()
                state.total_trips += 1
                self._total_trips += 1

                pause_duration = self._calculate_pause_duration(state)
                state.total_paused_seconds += pause_duration
                self._total_paused_seconds += pause_duration

                state.pause_history.append(pause_duration)
                if len(state.pause_history) > self.config.recovery_max_requests * 2:
                    state.pause_history = state.pause_history[-self.config.recovery_max_requests * 2:]

                logger.warning(
                    f"Channel {channel_id}: circuit tripped after {state.consecutive_failures} "
                    f"failures (pause: {pause_duration:.1f}s)"
                )

    def mark_no_videos(self, channel_id: str) -> None:
        """Mark a channel as having no published videos.

        These channels won't trigger circuit breaker but are tracked for metrics.

        Args:
            channel_id: The YouTube channel ID
        """
        with self._get_lock():
            self._channels_with_no_videos.add(channel_id)
            self._queried_channels.add(channel_id)
            # Create channel state so stats are available
            self._get_or_create_state(channel_id)
            logger.debug(f"Channel {channel_id}: marked as having no published videos")

    def is_channel_open(self, channel_id: str) -> bool:
        """Check if a channel's circuit is currently open.

        Args:
            channel_id: The YouTube channel ID

        Returns:
            True if circuit is open, False otherwise
        """
        if not self.config.enabled:
            return False

        with self._get_lock():
            state = self._channel_states.get(channel_id)
            if state is None:
                return False
            return state.is_open

    def get_channel_stats(self, channel_id: str) -> Optional[Dict]:
        """Get statistics for a specific channel.

        Args:
            channel_id: The YouTube channel ID

        Returns:
            Dict with channel stats or None if not tracked
        """
        with self._get_lock():
            if channel_id not in self._channel_states:
                return None

            state = self._channel_states[channel_id]
            latencies = self._channel_latencies.get(channel_id, [])

            return {
                'channel_id': channel_id,
                'consecutive_failures': state.consecutive_failures,
                'is_open': state.is_open,
                'is_recovering': state.is_recovering,
                'total_trips': state.total_trips,
                'total_paused_seconds': state.total_paused_seconds,
                'call_count': self._channel_call_counts.get(channel_id, 0),
                'error_count': self._channel_errors.get(channel_id, 0),
                'latency_avg': sum(latencies) / len(latencies) if latencies else 0,
                'latency_p50': self._percentile(latencies, 0.5) if latencies else 0,
                'latency_p95': self._percentile(latencies, 0.95) if latencies else 0,
                'has_no_videos': channel_id in self._channels_with_no_videos,
                'requests_per_minute': self._get_current_minute_count(channel_id),
                'requests_per_minute_limit': self.config.per_channel_requests_per_minute,
                'throttle_warning_issued': self._throttle_warnings_issued.get(channel_id, False),
                'recovery_successes': state.recovery_successes,
            }

    def get_metrics(self) -> Dict:
        """Get overall circuit breaker metrics.

        Returns:
            Dict with aggregated metrics
        """
        with self._get_lock():
            all_latencies = []
            for lat_list in self._channel_latencies.values():
                all_latencies.extend(lat_list)

            return {
                'total_channels_queried': len(self._queried_channels),
                'total_channels_tracked': len(self._channel_states),
                'channels_with_no_videos': len(self._channels_with_no_videos),
                'channels_with_circuit_open': sum(
                    1 for s in self._channel_states.values() if s.is_open
                ),
                'channels_with_circuit_recovering': sum(
                    1 for s in self._channel_states.values() if s.is_recovering
                ),
                'total_trips': self._total_trips,
                'total_paused_seconds': self._total_paused_seconds,
                'total_api_calls': sum(self._channel_call_counts.values()),
                'total_errors': sum(self._channel_errors.values()),
                'avg_latency_ms': sum(all_latencies) / len(all_latencies) if all_latencies else 0,
                'top_queried_channels': self._get_top_channels(10),
                'channel_usage_metrics': self._get_channel_usage_metrics(),
                'per_channel_requests_per_minute': self.config.per_channel_requests_per_minute,
                'throttle_warning_threshold': self.config.throttle_warning_threshold,
            }

    def get_channel_query_distribution(self) -> List[Dict]:
        """Get distribution of queries across channels.

        Returns:
            List of dicts with channel_id and query count, sorted by count descending
        """
        with self._get_lock():
            sorted_channels = sorted(
                self._channel_call_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [
                {
                    'channel_id': ch_id,
                    'query_count': count,
                    'error_count': self._channel_errors.get(ch_id, 0),
                    'has_no_videos': ch_id in self._channels_with_no_videos,
                }
                for ch_id, count in sorted_channels
            ]

    def _get_or_create_state(self, channel_id: str) -> ChannelCircuitState:
        """Get or create circuit state for a channel.

        Args:
            channel_id: The YouTube channel ID

        Returns:
            The channel's circuit state
        """
        if channel_id not in self._channel_states:
            self._channel_states[channel_id] = ChannelCircuitState()
            self._queried_channels.add(channel_id)
        return self._channel_states[channel_id]

    def _calculate_pause_duration(self, state: ChannelCircuitState) -> float:
        """Calculate pause duration based on history.

        Args:
            state: The channel circuit state

        Returns:
            Pause duration in seconds
        """
        # Base pause from config
        pause = self.config.pause_seconds

        # Scale by history if available
        if state.pause_history:
            avg_pause = sum(state.pause_history) / len(state.pause_history)
            pause = max(pause, avg_pause)

        # Apply recovery backoff multiplier
        if state.is_recovering:
            pause = pause * state.recovery_backoff_multiplier

        # Cap at max pause
        pause = min(pause, self.config.max_pause_seconds)

        # Add jitter
        import random
        jitter = pause * self.config.jitter_factor * (random.random() * 2 - 1)
        pause = pause + jitter

        return max(0, pause)

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile of values.

        Args:
            values: List of values
            p: Percentile (0-1)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]

    def _get_top_channels(self, n: int) -> List[Dict]:
        """Get top N most queried channels.

        Args:
            n: Number of channels to return

        Returns:
            List of dicts with channel info
        """
        sorted_channels = sorted(
            self._channel_call_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        return [
            {
                'channel_id': ch_id,
                'query_count': count,
                'error_count': self._channel_errors.get(ch_id, 0),
            }
            for ch_id, count in sorted_channels
        ]

    def _get_channel_usage_metrics(self) -> List[Dict]:
        """Get per-channel usage metrics for the current minute.

        Returns:
            List of dicts with channel_id, requests_this_minute, limit, and utilization
        """
        metrics = []
        for channel_id in self._channel_call_timestamps:
            requests_this_minute = self._get_current_minute_count(channel_id)
            limit = self.config.per_channel_requests_per_minute
            utilization = requests_this_minute / limit if limit > 0 else 0

            metrics.append({
                'channel_id': channel_id,
                'requests_this_minute': requests_this_minute,
                'limit': limit,
                'utilization': utilization,
                'warning_issued': self._throttle_warnings_issued.get(channel_id, False),
            })

        # Sort by utilization descending
        metrics.sort(key=lambda x: x['utilization'], reverse=True)
        return metrics

    def reset(self) -> None:
        """Reset all circuit breaker state."""
        with self._get_lock():
            self._channel_states.clear()
            self._channel_call_counts.clear()
            self._channel_latencies.clear()
            self._channel_errors.clear()
            self._queried_channels.clear()
            self._channels_with_no_videos.clear()
            self._channel_call_timestamps.clear()
            self._throttle_warnings_issued.clear()
            self._total_trips = 0
            self._total_paused_seconds = 0.0
            logger.info("Per-channel circuit breaker reset")


class _DummyLock:
    """Dummy lock for environments where threading is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
