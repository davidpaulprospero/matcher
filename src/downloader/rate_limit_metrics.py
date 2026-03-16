"""
Rate limiting metrics collection and reporting.

Aggregates metrics from various rate-limiting subsystems:
- Core retries from _run_download_cmd
- Progressive backoff delays
- Cookie rotations
- VPN switches
- Circuit breaker trips
- Batch retry queue operations
- Download speed tracking

Implements US-010: Add rate limiting metrics to pipeline report.
Implements US-012: Add rate limit health dashboard data export.

JSON Export Schema (export_to_json):
    {
        "schema_version": "1.0",
        "export_timestamp": "2026-01-25T12:34:56.789Z",
        "session": {
            "session_count": 1,
            "session_start_time": "2026-01-25T10:00:00Z",
            "session_end_time": "2026-01-25T12:34:56Z"
        },
        "downloads": {
            "total": 100,
            "successful": 95,
            "failed": 5,
            "success_rate_percent": 95.0
        },
        "retries": {
            "total_attempts": 50,
            "avg_per_download": 0.5,
            "max_reached_count": 3,
            "by_error_type": {"timeout": 20, "rate_limit": 15, "network": 10, "transient": 5}
        },
        "rate_limiting": {
            "total_events": 15,
            "percentage_of_downloads": 15.0,
            "by_tier": {"short": 5, "medium": 5, "long": 5},
            "by_keyword": {"sunset": 10, "ocean": 5},
            "backoff": {
                "total_attempts": 10,
                "total_seconds": 120.5,
                "by_severity": {"low": 3, "medium": 5, "high": 2}
            }
        },
        "escalation": {
            "cookie_rotations": 3,
            "vpn_switches": 1
        },
        "circuit_breaker": {
            "total_trips": 2,
            "total_pause_seconds": 120.0
        },
        "batch_retry": {
            "total_passes": 2,
            "total_recovered": 10,
            "total_failed": 2
        },
        "network": {
            "speed_samples": 50,
            "avg_speed_mbps": 5.5,
            "timeout_extensions": 3
        },
        "config_snapshot": {
            "rate_limit": {...},
            "circuit_breaker": {...},
            "batch_retry": {...},
            "speed_tracking": {...},
            "cookie_rotation": {...},
            "vpn": {...}
        },
        "recommendations": ["Consider increasing initial_backoff_seconds...", ...]
    }
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


def _log_rate_limit_event(
    event_type: str,
    keyword: str = None,
    tier: str = None,
    details: dict = None
) -> None:
    """Structured logging helper for rate limit events.

    This provides dashboard-friendly structured logging that can be
    parsed by log aggregation tools (e.g., ELK, Splunk, Datadog).

    Args:
        event_type: Type of event (rate_limit, backoff, recovery, etc.)
        keyword: Optional keyword associated with the event
        tier: Optional duration tier
        details: Additional context dict
    """
    log_data = {"event_type": event_type}
    if keyword:
        log_data["keyword"] = keyword
    if tier:
        log_data["tier"] = tier
    if details:
        log_data.update(details)

    logger.info("rate_limit_event", extra=log_data)


@dataclass
class RateLimitMetrics:
    """
    Aggregate rate limiting metrics for reporting.

    Tracks all rate-limit related events during a download session to help
    users tune their configuration and understand pipeline behavior.

    Attributes:
        total_downloads: Total download attempts (successful + failed)
        successful_downloads: Downloads that completed successfully
        failed_downloads: Downloads that failed permanently

        retry_attempts: Total individual retry attempts within _run_download_cmd
        retries_by_error_type: Breakdown of retries by error category
        avg_retry_count: Average retries per download (0 if no downloads)

        rate_limit_events: Total rate limit errors encountered
        backoff_attempts: Number of progressive backoff delays applied
        time_spent_backing_off: Total seconds spent in backoff delays

        cookie_rotations: Number of cookie file rotations
        vpn_switches: Number of VPN server switches

        circuit_breaker_trips: Times circuit breaker tripped
        circuit_breaker_pause_seconds: Total pause time from circuit breaker

        batch_retry_passes: Number of batch retry passes executed
        batch_retry_successes: Videos successfully recovered via batch retry
        batch_retry_failures: Videos that failed after all retry passes

        speed_samples: Number of download speed measurements
        avg_speed_mbps: Average download speed (MB/s)
        timeout_extensions: Number of times timeout was extended due to slow speed
    """

    # Core download counts
    total_downloads: int = 0
    successful_downloads: int = 0
    failed_downloads: int = 0

    # Retry metrics
    retry_attempts: int = 0
    retries_by_error_type: Dict[str, int] = field(default_factory=dict)
    max_retry_count_reached: int = 0  # Times retries exhausted

    # Rate limit handling
    rate_limit_events: int = 0
    tier_rate_limit_events: Dict[str, int] = field(default_factory=dict)  # US-001: per-tier tracking
    keyword_rate_limit_events: Dict[str, int] = field(default_factory=dict)  # US-009: per-keyword tracking
    backoff_attempts: int = 0
    time_spent_backing_off: float = 0.0
    backoff_events_by_severity: Dict[str, int] = field(default_factory=dict)  # US-008: per-severity tracking

    # Cookie/VPN escalation
    cookie_rotations: int = 0
    vpn_switches: int = 0

    # Circuit breaker
    circuit_breaker_trips: int = 0
    circuit_breaker_pause_seconds: float = 0.0

    # Batch retry
    batch_retry_passes: int = 0
    batch_retry_successes: int = 0
    batch_retry_failures: int = 0

    # Speed tracking
    speed_samples: int = 0
    avg_speed_mbps: float = 0.0
    timeout_extensions: int = 0
    speed_escalations: int = 0  # US-001 Sprint 12: speed-triggered tier escalations

    # Global rate limit coordinator (US-35-002)
    slot_timeouts: int = 0  # Times acquire_download_slot timed out

    # Rate limit window estimation (US-002 Sprint 13)
    # Per-keyword list of (timestamp, event_type) where event_type is 'failure' or 'recovery'
    _rate_limit_events_log: Dict[str, List[Tuple[float, str]]] = field(
        default_factory=dict, repr=False
    )

    # Time-based aggregation for observability (US-136-010)
    # Buckets for aggregating events by time window (5-minute windows)
    _event_timestamps: List[float] = field(default_factory=list, repr=False)

    # Time-to-recovery tracking (US-136-010)
    # Maps keyword -> list of recovery times in seconds
    _recovery_times: Dict[str, List[float]] = field(default_factory=dict, repr=False)

    # Track last rate limit timestamp per keyword for time-to-recovery calculation
    _last_rate_limit_time: Dict[str, float] = field(default_factory=dict, repr=False)

    # Session metadata
    session_start_time: Optional[str] = None
    session_end_time: Optional[str] = None
    session_count: int = 1  # US-006: Track number of sessions contributing to metrics

    def record_download_attempt(self) -> None:
        """Record a download attempt."""
        self.total_downloads += 1

    def record_download_success(self) -> None:
        """Record a successful download."""
        self.successful_downloads += 1

    def record_download_failure(self) -> None:
        """Record a failed download."""
        self.failed_downloads += 1

    def record_retry(self, error_type: str = 'unknown') -> None:
        """Record a retry attempt.

        Args:
            error_type: Category of error that triggered retry
                        (transient, timeout, rate_limit, network)
        """
        self.retry_attempts += 1
        self.retries_by_error_type[error_type] = self.retries_by_error_type.get(error_type, 0) + 1

    def record_retries_exhausted(self) -> None:
        """Record that max retries were reached for a download."""
        self.max_retry_count_reached += 1

    def record_rate_limit_event(self, tier: str = None, keyword: str = None) -> None:
        """Record a rate limit error occurrence.

        Args:
            tier: Duration tier (short, medium, long, longer) for per-tier tracking
            keyword: Search keyword for per-keyword tracking (US-009)
        """
        self.rate_limit_events += 1
        if tier:
            self.tier_rate_limit_events[tier] = self.tier_rate_limit_events.get(tier, 0) + 1
        if keyword:
            self.keyword_rate_limit_events[keyword] = self.keyword_rate_limit_events.get(keyword, 0) + 1

        # Track timestamp for time-based aggregation (US-136-010 AC2)
        self.record_event_timestamp()

        # Track last rate limit time per keyword for time-to-recovery (US-136-010 AC4)
        if keyword:
            self._last_rate_limit_time[keyword] = time.time()

        # Structured logging for observability (AC1: structured logging for all rate limit events)
        _log_rate_limit_event(
            event_type="rate_limit",
            keyword=keyword,
            tier=tier,
            details={"total_events": self.rate_limit_events}
        )

    def record_backoff(self, seconds: float, severity: str = None) -> None:
        """Record a progressive backoff delay.

        Args:
            seconds: Duration of the backoff delay
            severity: Severity level (low, medium, high) for adaptive backoff tracking (US-008)
        """
        self.backoff_attempts += 1
        self.time_spent_backing_off += seconds
        if severity:
            self.backoff_events_by_severity[severity] = self.backoff_events_by_severity.get(severity, 0) + 1

        # Structured logging for observability (AC1: structured logging for all rate limit events)
        _log_rate_limit_event(
            event_type="backoff",
            tier=severity,
            details={"seconds": seconds, "total_backoff_time": self.time_spent_backing_off}
        )

    def record_cookie_rotation(self) -> None:
        """Record a cookie file rotation."""
        self.cookie_rotations += 1
        # Structured logging for observability
        _log_rate_limit_event(
            event_type="cookie_rotation",
            details={"total_rotations": self.cookie_rotations}
        )

    def record_vpn_switch(self) -> None:
        """Record a VPN server switch."""
        self.vpn_switches += 1
        # Structured logging for observability
        _log_rate_limit_event(
            event_type="vpn_switch",
            details={"total_switches": self.vpn_switches}
        )

    def record_slot_timeout(self) -> None:
        """Record a global rate limit slot acquisition timeout (US-35-002)."""
        self.slot_timeouts += 1

    def record_circuit_breaker_trip(self, pause_seconds: float) -> None:
        """Record a circuit breaker trip.

        Args:
            pause_seconds: Duration of the pause
        """
        self.circuit_breaker_trips += 1
        self.circuit_breaker_pause_seconds += pause_seconds
        # Structured logging for observability
        _log_rate_limit_event(
            event_type="circuit_breaker_trip",
            details={
                "pause_seconds": pause_seconds,
                "total_trips": self.circuit_breaker_trips,
                "total_pause_seconds": self.circuit_breaker_pause_seconds
            }
        )

    def record_circuit_breaker_wait(self, wait_seconds: float) -> None:
        """Record time spent waiting for circuit breaker recovery.

        This is separate from circuit_breaker_pause_seconds which tracks
        pauses from trips. This tracks additional waits from download retry
        coordination (US-011) where a retry is paused waiting for an
        already-tripped circuit breaker to recover.

        Args:
            wait_seconds: Duration of the wait
        """
        # Add to circuit breaker pause seconds since it's the same category
        # of time spent blocked by circuit breaker
        self.circuit_breaker_pause_seconds += wait_seconds

    def record_batch_retry_pass(self, successes: int, failures: int) -> None:
        """Record completion of a batch retry pass.

        Args:
            successes: Number of videos recovered this pass
            failures: Number of videos still failing this pass
        """
        self.batch_retry_passes += 1
        self.batch_retry_successes += successes
        self.batch_retry_failures += failures

    def record_speed_sample(self, speed_mbps: float) -> None:
        """Record a download speed measurement.

        Args:
            speed_mbps: Download speed in MB/s
        """
        # Update running average
        if self.speed_samples == 0:
            self.avg_speed_mbps = speed_mbps
        else:
            # Incremental average: new_avg = old_avg + (new_value - old_avg) / (n + 1)
            self.avg_speed_mbps = (
                self.avg_speed_mbps + (speed_mbps - self.avg_speed_mbps) / (self.speed_samples + 1)
            )
        self.speed_samples += 1

    def record_timeout_extension(self) -> None:
        """Record that a timeout was extended due to slow speed."""
        self.timeout_extensions += 1

    def record_speed_escalation(self) -> None:
        """Record a speed-triggered tier escalation."""
        self.speed_escalations += 1

    def record_rate_limit_failure(self, keyword: str) -> None:
        """Record a rate limit failure event with timestamp for window estimation.

        Args:
            keyword: The keyword that experienced the rate limit failure.
        """
        if keyword not in self._rate_limit_events_log:
            self._rate_limit_events_log[keyword] = []
        self._rate_limit_events_log[keyword].append((time.time(), 'failure'))

    def record_rate_limit_recovery(self, keyword: str) -> None:
        """Record a rate limit recovery event with timestamp for window estimation.

        Args:
            keyword: The keyword that recovered from rate limiting.
        """
        if keyword not in self._rate_limit_events_log:
            self._rate_limit_events_log[keyword] = []
        self._rate_limit_events_log[keyword].append((time.time(), 'recovery'))

    def get_estimated_rate_limit_window(self, keyword: str) -> Optional[float]:
        """Estimate the rate limit window duration for a keyword.

        Analyzes alternating failure/recovery timestamp pairs to compute
        the median time gap between a failure event and its subsequent
        recovery event. Requires at least 3 failure/recovery pairs.

        Only the last 10 pairs are used to keep the estimate current.

        Args:
            keyword: The keyword to estimate for.

        Returns:
            Median window duration in seconds, or None if fewer than 3 pairs exist.
        """
        events = self._rate_limit_events_log.get(keyword, [])
        if not events:
            return None

        # Extract failure/recovery pairs: find each failure followed by a recovery
        pairs: List[float] = []
        i = 0
        while i < len(events):
            if events[i][1] == 'failure':
                # Find the next recovery after this failure
                j = i + 1
                while j < len(events):
                    if events[j][1] == 'recovery':
                        gap = events[j][0] - events[i][0]
                        pairs.append(gap)
                        i = j + 1
                        break
                    j += 1
                else:
                    # No recovery found after this failure
                    i += 1
            else:
                i += 1

        if len(pairs) < 3:
            return None

        # Use only the last 10 pairs for current estimate
        recent_pairs = pairs[-10:]
        return statistics.median(recent_pairs)

    def _get_all_estimated_windows(self) -> Dict[str, Optional[float]]:
        """Get estimated rate limit windows for all tracked keywords.

        Returns:
            Dict mapping keyword to estimated window in seconds (or None if insufficient data).
        """
        result: Dict[str, Optional[float]] = {}
        for keyword in self._rate_limit_events_log:
            estimate = self.get_estimated_rate_limit_window(keyword)
            if estimate is not None:
                result[keyword] = round(estimate, 2)
        return result

    # ==================== Time-based Aggregation (US-136-010) ====================

    def record_event_timestamp(self) -> None:
        """Record current timestamp for time-based aggregation.

        This should be called whenever a rate limit event occurs to enable
        time-window-based aggregation for dashboard display.
        """
        self._event_timestamps.append(time.time())

    def get_events_by_time_window(self, window_seconds: float = 300.0) -> Dict[str, int]:
        """Get rate limit event counts aggregated by time windows.

        AC2: Implement rate limit event aggregation by type and time

        Args:
            window_seconds: Size of each time window in seconds (default: 5 minutes)

        Returns:
            Dict mapping time window (as ISO string) to event count
        """
        if not self._event_timestamps:
            return {}

        # Sort timestamps
        sorted_times = sorted(self._event_timestamps)
        if not sorted_times:
            return {}

        # Calculate window boundaries
        start_time = sorted_times[0]
        end_time = sorted_times[-1]

        # Create windows
        windows: Dict[str, int] = {}
        current = start_time
        while current <= end_time:
            window_key = datetime.fromtimestamp(current, timezone.utc).strftime("%Y-%m-%dT%H:%M")
            windows[window_key] = 0
            current += window_seconds

        # Count events in each window
        for ts in sorted_times:
            window_key = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M")
            if window_key in windows:
                windows[window_key] += 1

        return windows

    def get_event_rate_per_minute(self) -> float:
        """Calculate average events per minute over the session.

        Returns:
            Events per minute, or 0.0 if no events recorded.
        """
        if not self._event_timestamps or len(self._event_timestamps) < 2:
            return 0.0

        sorted_times = sorted(self._event_timestamps)
        duration_seconds = sorted_times[-1] - sorted_times[0]
        if duration_seconds <= 0:
            return 0.0

        event_count = len(sorted_times)
        return (event_count / duration_seconds) * 60.0

    # ==================== Time-to-Recovery Metrics (US-136-010) ====================

    def record_recovery_with_timing(self, keyword: str, recovery_time_seconds: float) -> None:
        """Record a recovery event with actual time to recovery.

        AC4: Include time-to-recovery metrics for rate limit events

        Args:
            keyword: The keyword that recovered
            recovery_time_seconds: How long it took to recover (seconds)
        """
        if keyword not in self._recovery_times:
            self._recovery_times[keyword] = []
        self._recovery_times[keyword].append(recovery_time_seconds)

        # Log the recovery event
        _log_rate_limit_event(
            event_type="recovery",
            keyword=keyword,
            details={
                "recovery_time_seconds": recovery_time_seconds,
                "keyword_avg_recovery_time": self._get_avg_recovery_time(keyword)
            }
        )

    def _get_avg_recovery_time(self, keyword: str) -> Optional[float]:
        """Get average recovery time for a keyword.

        Args:
            keyword: The keyword to get average for

        Returns:
            Average recovery time in seconds, or None if no data
        """
        times = self._recovery_times.get(keyword)
        if not times:
            return None
        return round(sum(times) / len(times), 2)

    def get_time_to_recovery_stats(self) -> Dict[str, Any]:
        """Get comprehensive time-to-recovery statistics.

        AC4: Include time-to-recovery metrics for rate limit events

        Returns:
            Dict with recovery statistics:
                - per_keyword: avg/median/min/max per keyword
                - overall: global averages
                - sample_counts: number of samples per keyword
        """
        if not self._recovery_times:
            return {
                "per_keyword": {},
                "overall": {},
                "sample_counts": {}
            }

        per_keyword = {}
        all_times = []

        for keyword, times in self._recovery_times.items():
            if times:
                all_times.extend(times)
                per_keyword[keyword] = {
                    "avg_seconds": round(sum(times) / len(times), 2),
                    "median_seconds": round(statistics.median(times), 2),
                    "min_seconds": round(min(times), 2),
                    "max_seconds": round(max(times), 2),
                }

        overall = {}
        if all_times:
            overall = {
                "avg_seconds": round(sum(all_times) / len(all_times), 2),
                "median_seconds": round(statistics.median(all_times), 2),
                "min_seconds": round(min(all_times), 2),
                "max_seconds": round(max(all_times), 2),
            }

        sample_counts = {kw: len(times) for kw, times in self._recovery_times.items() if times}

        return {
            "per_keyword": per_keyword,
            "overall": overall,
            "sample_counts": sample_counts
        }

    @property
    def avg_retry_count(self) -> float:
        """Calculate average retry count per download."""
        if self.total_downloads == 0:
            return 0.0
        return round(self.retry_attempts / self.total_downloads, 2)

    @property
    def rate_limit_percentage(self) -> float:
        """Calculate percentage of downloads affected by rate limiting."""
        if self.total_downloads == 0:
            return 0.0
        return round(100.0 * self.rate_limit_events / self.total_downloads, 1)

    @property
    def success_rate(self) -> float:
        """Calculate download success rate as percentage."""
        if self.total_downloads == 0:
            return 0.0
        return round(100.0 * self.successful_downloads / self.total_downloads, 1)

    def update_from_circuit_breaker(self, stats: dict) -> None:
        """Update metrics from circuit breaker stats.

        Args:
            stats: Dict from CircuitBreaker.get_stats()
        """
        if stats.get('enabled'):
            self.circuit_breaker_trips = stats.get('total_trips', 0)
            self.circuit_breaker_pause_seconds = stats.get('total_paused_seconds', 0.0)

    def update_from_retry_queue(self, stats: dict) -> None:
        """Update metrics from retry queue stats.

        Args:
            stats: Dict from RetryQueue.get_stats()
        """
        if stats.get('enabled'):
            self.batch_retry_passes = stats.get('current_pass', 0)
            self.batch_retry_successes = stats.get('total_retried', 0)
            failed = stats.get('failed', 0)
            if isinstance(failed, int):
                self.batch_retry_failures = failed
            else:
                self.batch_retry_failures = len(failed) if failed else 0

            # US-144-009: Update enhanced retry queue metrics
            self._update_retry_queue_enhanced(stats)

    def _update_retry_queue_enhanced(self, stats: dict) -> None:
        """US-144-009: Update enhanced retry queue metrics from stats.

        Args:
            stats: Dict from RetryQueue.get_stats() containing enhanced metrics
        """
        # Check if stats contains enhanced metrics from RetryQueueStats
        keyword_retry_stats = stats.get('keyword_retry_stats', {})
        retry_latency = stats.get('retry_latency', {})
        category_distribution = stats.get('category_distribution', {})
        retry_efficiency_score = stats.get('retry_efficiency_score', 0.0)

        # Store the enhanced metrics for export
        self._retry_queue_keyword_stats = keyword_retry_stats
        self._retry_queue_latency = retry_latency
        self._retry_queue_category_distribution = category_distribution
        self._retry_queue_efficiency = retry_efficiency_score

        # Log the enhanced metrics for visibility
        if keyword_retry_stats:
            logger.debug(f"Retry queue keyword stats: {keyword_retry_stats}")
        if retry_latency:
            logger.debug(f"Retry queue latency stats: {retry_latency}")
        if category_distribution:
            logger.debug(f"Retry queue category distribution: {category_distribution}")

    def update_from_speed_tracker(self, stats: dict) -> None:
        """Update metrics from speed tracker stats.

        Args:
            stats: Dict from DownloadSpeedTracker.get_speed_stats()
        """
        self.speed_samples = stats.get('samples', 0)
        self.avg_speed_mbps = stats.get('avg_speed_mbps', 0.0)

    def _get_youtube_api_metrics_for_export(self) -> Dict[str, Any]:
        """Get YouTube API metrics for export (US-150-010).

        Attempts to retrieve metrics from the YouTube API client if available.

        Returns:
            Dictionary with YouTube API metrics or placeholder if unavailable.
        """
        try:
            from .api_fallback_handler import get_youtube_api_client
            client = get_youtube_api_client()
            if client is not None:
                return client.get_api_metrics()
        except Exception:
            pass

        # Return placeholder if not available
        return {
            "enabled": False,
            "message": "YouTube API client not available"
        }

    def get_config_recommendations(self) -> List[str]:
        """
        Generate config recommendations based on collected metrics.

        Returns recommendations when:
        - Rate limit events > 10% of downloads
        - Many retries exhausted
        - Circuit breaker tripping frequently
        - Slow download speeds

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check if rate limiting is a significant issue (>10% threshold from AC)
        if self.total_downloads > 0 and self.rate_limit_percentage > 10:
            recommendations.append(
                f"High rate limiting detected ({self.rate_limit_percentage:.1f}% of downloads). "
                "Consider increasing download.rate_limit.initial_backoff_seconds or using "
                "more cookie files for rotation."
            )

        # Check retry exhaustion rate
        if self.total_downloads > 0:
            exhaustion_rate = 100.0 * self.max_retry_count_reached / self.total_downloads
            if exhaustion_rate > 20:
                recommendations.append(
                    f"High retry exhaustion rate ({exhaustion_rate:.1f}%). "
                    "Consider increasing download.max_retries or retry_delay."
                )

        # Check circuit breaker frequency
        if self.circuit_breaker_trips > 3:
            recommendations.append(
                f"Circuit breaker tripped {self.circuit_breaker_trips} times. "
                "YouTube may be heavily rate-limiting. Consider longer pauses between "
                "search batches or using VPN rotation."
            )

        # Check cookie rotations vs rate limit events
        if self.rate_limit_events > 0 and self.cookie_rotations == 0:
            recommendations.append(
                "Rate limit events detected but no cookie rotation configured. "
                "Enable download.cookie_rotation with multiple cookie files."
            )

        # Check slow speeds and timeout extensions
        if self.timeout_extensions > 5:
            recommendations.append(
                f"Timeout extended {self.timeout_extensions} times due to slow speeds. "
                "Network may be congested. Consider running during off-peak hours."
            )

        # Check if batch retry helped
        if self.batch_retry_passes > 0 and self.batch_retry_successes > 0:
            recovery_rate = 100.0 * self.batch_retry_successes / (
                self.batch_retry_successes + self.batch_retry_failures
            ) if (self.batch_retry_successes + self.batch_retry_failures) > 0 else 0
            if recovery_rate < 30:
                recommendations.append(
                    f"Batch retry recovery rate is low ({recovery_rate:.1f}%). "
                    "Consider increasing download.batch_retry.delay_seconds."
                )

        # Check for high-rate-limit keywords (US-009)
        # Recommend skipping keywords that cause > 30% of rate limit events
        if self.keyword_rate_limit_events and self.rate_limit_events > 10:
            # Find keywords with high event counts
            high_rate_limit_keywords = []
            threshold = self.rate_limit_events * 0.3  # 30% of total events
            for keyword, count in self.keyword_rate_limit_events.items():
                if count >= threshold:
                    high_rate_limit_keywords.append((keyword, count))

            if high_rate_limit_keywords:
                # Sort by count descending
                high_rate_limit_keywords.sort(key=lambda x: x[1], reverse=True)
                keyword_list = ", ".join(f'"{k}"' for k, _ in high_rate_limit_keywords[:3])
                recommendations.append(
                    f"Keywords causing high rate limiting: {keyword_list}. "
                    "Consider adding these to keyword.skip_keywords or using more "
                    "specific search terms."
                )

        return recommendations

    def summary(self) -> str:
        """
        Generate human-readable summary for the healing report.

        Returns:
            Multi-line string suitable for printing in the report
        """
        lines = []

        # Session indicator for cross-session metrics (US-006)
        if self.session_count > 1:
            lines.append(f"Total across {self.session_count} sessions:")

        # Download summary
        lines.append(
            f"Downloads: {self.total_downloads} total "
            f"({self.successful_downloads} successful, {self.failed_downloads} failed)"
        )

        if self.total_downloads > 0:
            lines.append(f"Success rate: {self.success_rate}%")

        # Retry summary
        if self.retry_attempts > 0:
            lines.append(f"Retries: {self.retry_attempts} total, avg {self.avg_retry_count}/download")
            if self.retries_by_error_type:
                type_str = ", ".join(f"{k}: {v}" for k, v in self.retries_by_error_type.items())
                lines.append(f"  By type: {type_str}")
            if self.max_retry_count_reached > 0:
                lines.append(f"  Retries exhausted: {self.max_retry_count_reached} downloads")

        # Rate limiting summary
        if self.rate_limit_events > 0 or self.backoff_attempts > 0:
            lines.append(
                f"Rate limiting: {self.rate_limit_events} events, "
                f"{self.backoff_attempts} backoffs ({self.time_spent_backing_off:.1f}s total)"
            )
            # Show per-tier breakdown if available
            if self.tier_rate_limit_events:
                tier_str = ", ".join(f"{k}: {v}" for k, v in sorted(self.tier_rate_limit_events.items()))
                lines.append(f"  By tier: {tier_str}")
            # Show per-severity breakdown if available (US-008)
            if self.backoff_events_by_severity:
                severity_str = ", ".join(f"{k}: {v}" for k, v in sorted(self.backoff_events_by_severity.items()))
                lines.append(f"  By severity: {severity_str}")
            # Show top 5 keywords when > 10 total rate limit events (US-009)
            if self.keyword_rate_limit_events and self.rate_limit_events > 10:
                # Sort by count descending, take top 5
                sorted_keywords = sorted(
                    self.keyword_rate_limit_events.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                keyword_str = ", ".join(f'"{k}": {v}' for k, v in sorted_keywords)
                lines.append(f"  Top keywords: {keyword_str}")

        # Escalation summary
        escalations = []
        if self.cookie_rotations > 0:
            escalations.append(f"cookie rotations: {self.cookie_rotations}")
        if self.vpn_switches > 0:
            escalations.append(f"VPN switches: {self.vpn_switches}")
        if self.speed_escalations > 0:
            escalations.append(f"speed-triggered: {self.speed_escalations}")
        if escalations:
            lines.append(f"Escalations: {', '.join(escalations)}")

        # Circuit breaker summary
        if self.circuit_breaker_trips > 0:
            lines.append(
                f"Circuit breaker: {self.circuit_breaker_trips} trips, "
                f"{self.circuit_breaker_pause_seconds:.1f}s paused"
            )

        # Batch retry summary
        if self.batch_retry_passes > 0:
            lines.append(
                f"Batch retry: {self.batch_retry_passes} passes, "
                f"{self.batch_retry_successes} recovered, {self.batch_retry_failures} failed"
            )

        # Speed summary
        if self.speed_samples > 0:
            speed_line = f"Network: {self.avg_speed_mbps:.2f} MB/s avg ({self.speed_samples} samples)"
            if self.timeout_extensions > 0:
                speed_line += f", {self.timeout_extensions} timeout extensions"
            lines.append(speed_line)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize metrics to dictionary for checkpoint/JSON.

        Returns:
            Dict representation of all metrics
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'RateLimitMetrics':
        """Create RateLimitMetrics from dictionary.

        Args:
            data: Dict from to_dict() or checkpoint

        Returns:
            New RateLimitMetrics instance
        """
        if not data:
            return cls()

        # Handle missing fields for backward compatibility
        return cls(
            total_downloads=data.get('total_downloads', 0),
            successful_downloads=data.get('successful_downloads', 0),
            failed_downloads=data.get('failed_downloads', 0),
            retry_attempts=data.get('retry_attempts', 0),
            retries_by_error_type=data.get('retries_by_error_type', {}),
            max_retry_count_reached=data.get('max_retry_count_reached', 0),
            rate_limit_events=data.get('rate_limit_events', 0),
            tier_rate_limit_events=data.get('tier_rate_limit_events', {}),
            keyword_rate_limit_events=data.get('keyword_rate_limit_events', {}),  # US-009
            backoff_attempts=data.get('backoff_attempts', 0),
            time_spent_backing_off=data.get('time_spent_backing_off', 0.0),
            backoff_events_by_severity=data.get('backoff_events_by_severity', {}),
            cookie_rotations=data.get('cookie_rotations', 0),
            vpn_switches=data.get('vpn_switches', 0),
            circuit_breaker_trips=data.get('circuit_breaker_trips', 0),
            circuit_breaker_pause_seconds=data.get('circuit_breaker_pause_seconds', 0.0),
            batch_retry_passes=data.get('batch_retry_passes', 0),
            batch_retry_successes=data.get('batch_retry_successes', 0),
            batch_retry_failures=data.get('batch_retry_failures', 0),
            speed_samples=data.get('speed_samples', 0),
            avg_speed_mbps=data.get('avg_speed_mbps', 0.0),
            timeout_extensions=data.get('timeout_extensions', 0),
            speed_escalations=data.get('speed_escalations', 0),
            session_start_time=data.get('session_start_time'),
            session_end_time=data.get('session_end_time'),
            session_count=data.get('session_count', 1),
        )

    @classmethod
    def from_checkpoint(cls, data: dict) -> 'RateLimitMetrics':
        """Create RateLimitMetrics from checkpoint and increment session count.

        Unlike from_dict(), this method is used specifically for resuming from
        a checkpoint. It:
        1. Loads previous metrics state
        2. Increments session_count to track cross-session aggregation
        3. Logs the restored state for visibility

        This enables cumulative metrics across multiple session boundaries for
        multi-day downloads.

        Args:
            data: Dict from checkpoint's rate_limit_metrics field

        Returns:
            New RateLimitMetrics instance with incremented session_count

        Note:
            The session_count reflects how many download sessions have
            contributed to the cumulative metrics. For example, if a user
            runs the pipeline 3 times (with interruptions), session_count=3.
        """
        if not data:
            return cls()

        # First create the metrics from the checkpoint data
        metrics = cls.from_dict(data)

        # Increment session count since we're starting a new session
        metrics.session_count += 1

        logger.info(
            f"Restored rate limit metrics from checkpoint (session {metrics.session_count}): "
            f"{metrics.total_downloads} downloads, {metrics.rate_limit_events} rate limits"
        )

        return metrics

    def clear(self) -> None:
        """Reset all metrics for a new session including session count."""
        self.total_downloads = 0
        self.successful_downloads = 0
        self.failed_downloads = 0
        self.retry_attempts = 0
        self.retries_by_error_type = {}
        self.max_retry_count_reached = 0
        self.rate_limit_events = 0
        self.tier_rate_limit_events = {}
        self.keyword_rate_limit_events = {}  # US-009
        self.backoff_attempts = 0
        self.time_spent_backing_off = 0.0
        self.backoff_events_by_severity = {}
        self.cookie_rotations = 0
        self.vpn_switches = 0
        self.circuit_breaker_trips = 0
        self.circuit_breaker_pause_seconds = 0.0
        self.batch_retry_passes = 0
        self.batch_retry_successes = 0
        self.batch_retry_failures = 0
        self.speed_samples = 0
        self.avg_speed_mbps = 0.0
        self.timeout_extensions = 0
        self.speed_escalations = 0
        self._rate_limit_events_log = {}  # US-002 Sprint 13
        # US-136-010: Clear new time-based aggregation and recovery tracking
        self._event_timestamps = []
        self._recovery_times = {}
        self._last_rate_limit_time = {}
        self.session_start_time = None
        self.session_end_time = None
        self.session_count = 1  # Reset to 1 for new session (US-006)

    # ==================== US-144-009: Enhanced Retry Queue Metrics ====================

    # These are set dynamically via _update_retry_queue_enhanced
    # No class-level defaults needed - they're set when metrics are received

    def _get_retry_queue_keyword_stats(self) -> Dict:
        """US-144-009: Get stored keyword retry stats."""
        return getattr(self, '_retry_queue_keyword_stats', {})

    def _get_retry_queue_latency(self) -> Dict:
        """US-144-009: Get stored retry latency stats."""
        return getattr(self, '_retry_queue_latency', {})

    def _get_retry_queue_category_distribution(self) -> Dict:
        """US-144-009: Get stored category distribution."""
        return getattr(self, '_retry_queue_category_distribution', {})

    def _get_retry_queue_efficiency(self) -> float:
        """US-144-009: Get stored retry efficiency score."""
        return getattr(self, '_retry_queue_efficiency', 0.0)

    def export_to_json(self, config: Any = None, escalation_manager: Any = None) -> dict:
        """Export metrics to structured JSON format for external monitoring tools.

        Creates a well-structured export with all metrics, timestamps, session info,
        active config values, and recommendations. The schema is documented in the
        module docstring.

        Args:
            config: Optional Config object to include rate-limiting config snapshot.
                    If provided, exports download.rate_limit, download.circuit_breaker,
                    download.batch_retry, download.speed_tracking, download.cookie_rotation,
                    and download.vpn config sections.
            escalation_manager: Optional EscalationManager to include per-keyword
                    escalation timelines and hot keywords in the export under
                    'escalation.keyword_timelines' and 'escalation.hot_keywords'.

        Returns:
            Dict with structured metrics data ready for JSON serialization.
            See module docstring for complete schema documentation.

        Example:
            >>> metrics = RateLimitMetrics()
            >>> metrics.record_download_attempt()
            >>> metrics.record_download_success()
            >>> data = metrics.export_to_json(config)
            >>> with open('metrics.json', 'w') as f:
            ...     json.dump(data, f, indent=2)
        """
        export_timestamp = datetime.now(timezone.utc).isoformat()

        # Build structured export
        export_data = {
            "schema_version": "1.0",
            "export_timestamp": export_timestamp,

            # Session metadata
            "session": {
                "session_count": self.session_count,
                "session_start_time": self.session_start_time,
                "session_end_time": self.session_end_time or export_timestamp,
            },

            # Download statistics
            "downloads": {
                "total": self.total_downloads,
                "successful": self.successful_downloads,
                "failed": self.failed_downloads,
                "success_rate_percent": self.success_rate,
            },

            # Retry statistics
            "retries": {
                "total_attempts": self.retry_attempts,
                "avg_per_download": self.avg_retry_count,
                "max_reached_count": self.max_retry_count_reached,
                "by_error_type": dict(self.retries_by_error_type),
            },

            # Rate limiting statistics
            "rate_limiting": {
                "total_events": self.rate_limit_events,
                "percentage_of_downloads": self.rate_limit_percentage,
                "by_tier": dict(self.tier_rate_limit_events),
                "by_keyword": dict(self.keyword_rate_limit_events),
                "estimated_window_seconds": self._get_all_estimated_windows(),
                "backoff": {
                    "total_attempts": self.backoff_attempts,
                    "total_seconds": round(self.time_spent_backing_off, 2),
                    "by_severity": dict(self.backoff_events_by_severity),
                },
                # US-136-010: Time-based aggregation for dashboard
                "event_rate_per_minute": round(self.get_event_rate_per_minute(), 2),
                "events_by_time_window": self.get_events_by_time_window(),
                # US-136-010: Time-to-recovery metrics
                "time_to_recovery": self.get_time_to_recovery_stats(),
            },

            # Escalation statistics
            "escalation": {
                "cookie_rotations": self.cookie_rotations,
                "vpn_switches": self.vpn_switches,
                "speed_escalations": self.speed_escalations,
            },

            # Circuit breaker statistics
            "circuit_breaker": {
                "total_trips": self.circuit_breaker_trips,
                "total_pause_seconds": round(self.circuit_breaker_pause_seconds, 2),
            },

            # Batch retry statistics
            "batch_retry": {
                "total_passes": self.batch_retry_passes,
                "total_recovered": self.batch_retry_successes,
                "total_failed": self.batch_retry_failures,
                # US-144-009: Enhanced retry queue metrics
                "keyword_stats": self._get_retry_queue_keyword_stats(),
                "retry_latency": self._get_retry_queue_latency(),
                "category_distribution": self._get_retry_queue_category_distribution(),
                "retry_efficiency": self._get_retry_queue_efficiency(),
            },

            # Network statistics
            "network": {
                "speed_samples": self.speed_samples,
                "avg_speed_mbps": round(self.avg_speed_mbps, 3),
                "timeout_extensions": self.timeout_extensions,
            },

            # YouTube API metrics (US-150-010)
            "youtube_api": self._get_youtube_api_metrics_for_export(),
        }

        # Include escalation timeline and hot keywords if escalation_manager provided
        if escalation_manager is not None:
            try:
                export_data["escalation"]["keyword_timelines"] = (
                    escalation_manager.get_keyword_escalation_timeline()
                )
                export_data["escalation"]["hot_keywords"] = (
                    escalation_manager.get_hot_keywords()
                )
                export_data["escalation"]["tier_effectiveness"] = (
                    escalation_manager.get_tier_effectiveness()
                )
                export_data["escalation"]["tier_recommendations"] = (
                    escalation_manager.get_tier_recommendations()
                )
            except Exception:
                logger.debug("Failed to include escalation timeline in export", exc_info=True)

        # Include config snapshot if provided
        if config is not None:
            config_snapshot = {}
            download_config = getattr(config, 'download', None)

            if download_config:
                # Rate limit config
                rate_limit = getattr(download_config, 'rate_limit', None)
                if rate_limit:
                    config_snapshot["rate_limit"] = {
                        "initial_backoff_seconds": getattr(rate_limit, 'initial_backoff_seconds', 5.0),
                        "max_backoff_before_rotate": getattr(rate_limit, 'max_backoff_before_rotate', 60.0),
                        "backoff_multiplier": getattr(rate_limit, 'backoff_multiplier', 2.0),
                        "per_tier_isolation": getattr(rate_limit, 'per_tier_isolation', True),
                        "share_budget_across_keywords": getattr(rate_limit, 'share_budget_across_keywords', True),
                        "max_backoff_budget": getattr(rate_limit, 'max_backoff_budget', 300.0),
                        "adaptive_multiplier": getattr(rate_limit, 'adaptive_multiplier', True),
                    }

                # Circuit breaker config
                circuit_breaker = getattr(download_config, 'circuit_breaker', None)
                if circuit_breaker:
                    config_snapshot["circuit_breaker"] = {
                        "enabled": getattr(circuit_breaker, 'enabled', True),
                        "consecutive_failures_threshold": getattr(circuit_breaker, 'consecutive_failures_threshold', 5),
                        "pause_seconds": getattr(circuit_breaker, 'pause_seconds', 60.0),
                        "block_download_retries": getattr(circuit_breaker, 'block_download_retries', True),
                    }

                # Batch retry config
                batch_retry = getattr(download_config, 'batch_retry', None)
                if batch_retry:
                    config_snapshot["batch_retry"] = {
                        "enabled": getattr(batch_retry, 'enabled', True),
                        "delay_seconds": getattr(batch_retry, 'delay_seconds', 120.0),
                        "max_passes": getattr(batch_retry, 'max_passes', 2),
                        "respect_circuit_breaker": getattr(batch_retry, 'respect_circuit_breaker', True),
                        "wait_for_cookie_cooldown": getattr(batch_retry, 'wait_for_cookie_cooldown', True),
                    }

                # Speed tracking config
                speed_tracking = getattr(download_config, 'speed_tracking', None)
                if speed_tracking:
                    config_snapshot["speed_tracking"] = {
                        "enabled": getattr(speed_tracking, 'enabled', True),
                        "window_size": getattr(speed_tracking, 'window_size', 5),
                        "min_speed_mbps": getattr(speed_tracking, 'min_speed_mbps', 1.0),
                        "max_timeout_multiplier": getattr(speed_tracking, 'max_timeout_multiplier', 2.0),
                        "rate_limit_signal_threshold": getattr(speed_tracking, 'rate_limit_signal_threshold', 0.1),
                        "consecutive_slow_samples": getattr(speed_tracking, 'consecutive_slow_samples', 3),
                    }

                # Cookie rotation config
                cookie_rotation = getattr(download_config, 'cookie_rotation', None)
                if cookie_rotation:
                    config_snapshot["cookie_rotation"] = {
                        "enabled": getattr(cookie_rotation, 'enabled', False),
                        "rotation_strategy": getattr(cookie_rotation, 'rotation_strategy', 'on_error'),
                        "cooldown_seconds": getattr(cookie_rotation, 'cooldown_seconds', 300),
                        "max_rotations_per_session": getattr(cookie_rotation, 'max_rotations_per_session', 0),
                    }

                # VPN config
                vpn = getattr(download_config, 'vpn', None)
                if vpn:
                    config_snapshot["vpn"] = {
                        "enabled": getattr(vpn, 'enabled', False),
                        "rotate_on_rate_limit": getattr(vpn, 'rotate_on_rate_limit', True),
                        "switch_delay_seconds": getattr(vpn, 'switch_delay_seconds', 10),
                        "max_switches_per_session": getattr(vpn, 'max_switches_per_session', 10),
                        "verify_connection": getattr(vpn, 'verify_connection', True),
                    }

            if config_snapshot:
                export_data["config_snapshot"] = config_snapshot

        return export_data

    def export_to_json_file(self, path: str, config: Any = None) -> None:
        """Export metrics to a JSON file.

        Convenience method that calls export_to_json() and writes to a file.

        Args:
            path: File path to write JSON to.
            config: Optional Config object to include in export.

        Raises:
            OSError: If file cannot be written.
        """
        from pathlib import Path as PathLib
        export_data = self.export_to_json(config)

        # Ensure parent directory exists
        PathLib(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported rate limit metrics to {path}")


class RateLimitMetricsAggregator:
    """Unified metrics aggregator collecting from all rate-limiting subsystems.

    Collects metrics from:
    - EscalationManager: tier escalation state and counts
    - CookieRotator: cookie rotation status
    - RateLimitBudget: budget consumption state
    - CircuitBreaker: circuit breaker trip stats

    Also uses classify_trigger() from escalation_manager to break down
    trigger categories for granular reporting.

    Implements US-004 Sprint 10: Unified rate-limit metrics aggregation.
    """

    def __init__(
        self,
        escalation_manager=None,
        cookie_rotator=None,
        rate_limit_budget=None,
        circuit_breaker=None,
    ):
        """Initialize aggregator with optional subsystem references.

        Args:
            escalation_manager: EscalationManager instance (has get_metrics())
            cookie_rotator: CookieRotator instance (has get_status())
            rate_limit_budget: RateLimitBudget instance (has to_dict())
            circuit_breaker: CircuitBreaker instance (has get_stats())
        """
        self._escalation_manager = escalation_manager
        self._cookie_rotator = cookie_rotator
        self._rate_limit_budget = rate_limit_budget
        self._circuit_breaker = circuit_breaker
        self._trigger_counts: Dict[str, int] = {}

    def record_trigger(self, stderr_output: str) -> None:
        """Record a trigger event by classifying its category.

        Uses classify_trigger() from escalation_manager module to
        categorize the stderr output and increment the corresponding counter.

        Args:
            stderr_output: Raw stderr text from a yt-dlp subprocess.
        """
        from .escalation_manager import classify_trigger

        category = classify_trigger(stderr_output)
        if category:
            self._trigger_counts[category] = self._trigger_counts.get(category, 0) + 1

    def aggregate(self) -> Dict[str, Any]:
        """Collect and return unified metrics from all subsystems.

        Returns:
            Dict with keys:
                escalation: Dict from EscalationManager.get_metrics() or empty
                cookies: Dict from CookieRotator.get_status() or empty
                budget: Dict from RateLimitBudget.to_dict() or empty
                circuit_breaker: Dict from CircuitBreaker.get_stats() or empty
                trigger_categories: Dict with category breakdown
                    e.g. {'403': 5, '429': 3, 'bot_detection': 1, ...}
        """
        result: Dict[str, Any] = {
            'escalation': {},
            'cookies': {},
            'budget': {},
            'circuit_breaker': {},
            'trigger_categories': dict(self._trigger_counts),
        }

        if self._escalation_manager is not None:
            try:
                result['escalation'] = self._escalation_manager.get_metrics()
            except Exception:
                logger.debug("Failed to collect escalation metrics", exc_info=True)

        if self._cookie_rotator is not None:
            try:
                result['cookies'] = self._cookie_rotator.get_status()
            except Exception:
                logger.debug("Failed to collect cookie metrics", exc_info=True)

        if self._rate_limit_budget is not None:
            try:
                result['budget'] = self._rate_limit_budget.to_dict()
            except Exception:
                logger.debug("Failed to collect budget metrics", exc_info=True)

        if self._circuit_breaker is not None:
            try:
                result['circuit_breaker'] = self._circuit_breaker.get_stats()
            except Exception:
                logger.debug("Failed to collect circuit breaker metrics", exc_info=True)

        return result

    def get_health_status(self) -> str:
        """Determine overall rate-limiting health status.

        Evaluates escalation tier distribution and budget remaining to
        return a simple health indicator.

        Returns:
            'healthy': Most keywords at Tier 1, budget mostly available
            'degraded': Significant escalation or budget partially consumed
            'critical': Majority of keywords at max tier or budget exhausted
        """
        # Start healthy, downgrade based on signals
        score = 0  # 0 = healthy, 1 = degraded, 2 = critical

        # Check escalation tier distribution
        if self._escalation_manager is not None:
            try:
                metrics = self._escalation_manager.get_metrics()
                kw_tiers = metrics.get('keywords_at_each_tier', {})
                total_keywords = sum(len(kws) for kws in kw_tiers.values())

                if total_keywords > 0:
                    # Count keywords at max tier (FULL_BYPASS / Tier 3)
                    max_tier_keywords = len(kw_tiers.get('FULL_BYPASS', []))
                    max_tier_pct = max_tier_keywords / total_keywords

                    avg_tier = metrics.get('average_tier', 1.0)

                    if max_tier_pct > 0.5 or avg_tier >= 2.5:
                        score = max(score, 2)  # critical
                    elif max_tier_pct > 0.2 or avg_tier >= 1.8:
                        score = max(score, 1)  # degraded
            except Exception:
                pass

        # Check budget remaining
        if self._rate_limit_budget is not None:
            try:
                budget_data = self._rate_limit_budget.to_dict()
                max_rotations = budget_data.get('max_rotations', 10)
                used_rotations = budget_data.get('rotations_used', 0)
                max_backoff = budget_data.get('max_backoff_time', 600)
                used_backoff = budget_data.get('backoff_time_spent', 0)

                if max_rotations > 0:
                    rotation_pct = used_rotations / max_rotations
                    if rotation_pct >= 0.9:
                        score = max(score, 2)
                    elif rotation_pct >= 0.5:
                        score = max(score, 1)

                if max_backoff > 0:
                    backoff_pct = used_backoff / max_backoff
                    if backoff_pct >= 0.9:
                        score = max(score, 2)
                    elif backoff_pct >= 0.5:
                        score = max(score, 1)
            except Exception:
                pass

        # Check circuit breaker state
        if self._circuit_breaker is not None:
            try:
                cb_stats = self._circuit_breaker.get_stats()
                if cb_stats.get('is_open', False):
                    score = max(score, 2)  # Open circuit breaker = critical
                elif cb_stats.get('total_trips', 0) > 3:
                    score = max(score, 1)  # Multiple trips = degraded
            except Exception:
                pass

        return ['healthy', 'degraded', 'critical'][score]
