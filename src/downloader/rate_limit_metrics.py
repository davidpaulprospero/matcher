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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


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

    def record_cookie_rotation(self) -> None:
        """Record a cookie file rotation."""
        self.cookie_rotations += 1

    def record_vpn_switch(self) -> None:
        """Record a VPN server switch."""
        self.vpn_switches += 1

    def record_circuit_breaker_trip(self, pause_seconds: float) -> None:
        """Record a circuit breaker trip.

        Args:
            pause_seconds: Duration of the pause
        """
        self.circuit_breaker_trips += 1
        self.circuit_breaker_pause_seconds += pause_seconds

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

    def update_from_speed_tracker(self, stats: dict) -> None:
        """Update metrics from speed tracker stats.

        Args:
            stats: Dict from DownloadSpeedTracker.get_speed_stats()
        """
        self.speed_samples = stats.get('samples', 0)
        self.avg_speed_mbps = stats.get('avg_speed_mbps', 0.0)

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
        self.session_start_time = None
        self.session_end_time = None
        self.session_count = 1  # Reset to 1 for new session (US-006)

    def export_to_json(self, config: Any = None) -> dict:
        """Export metrics to structured JSON format for external monitoring tools.

        Creates a well-structured export with all metrics, timestamps, session info,
        active config values, and recommendations. The schema is documented in the
        module docstring.

        Args:
            config: Optional Config object to include rate-limiting config snapshot.
                    If provided, exports download.rate_limit, download.circuit_breaker,
                    download.batch_retry, download.speed_tracking, download.cookie_rotation,
                    and download.vpn config sections.

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
                "backoff": {
                    "total_attempts": self.backoff_attempts,
                    "total_seconds": round(self.time_spent_backing_off, 2),
                    "by_severity": dict(self.backoff_events_by_severity),
                },
            },

            # Escalation statistics
            "escalation": {
                "cookie_rotations": self.cookie_rotations,
                "vpn_switches": self.vpn_switches,
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
            },

            # Network statistics
            "network": {
                "speed_samples": self.speed_samples,
                "avg_speed_mbps": round(self.avg_speed_mbps, 3),
                "timeout_extensions": self.timeout_extensions,
            },

            # Recommendations
            "recommendations": self.get_config_recommendations(),
        }

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
