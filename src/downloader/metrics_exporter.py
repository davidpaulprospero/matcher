"""Download metrics exporter for Prometheus and JSON formats.

Provides Prometheus-compatible metrics export for download operations,
integrating with RateLimitMetrics and GlobalRateLimitCoordinator.

Implements US-93-011: Add download metrics export.
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .rate_limit_metrics import RateLimitMetrics
    from .speed_tracker import DownloadSpeedTracker
    from ..rate_limit.coordinator import GlobalRateLimitCoordinator
    from .escalation_manager import EscalationManager

logger = logging.getLogger(__name__)


@dataclass
class DownloadMetricsConfig:
    """Configuration for download metrics export."""

    # Export settings
    enabled: bool = True
    export_interval_seconds: float = 60.0  # Configurable export interval
    output_dir: str = "output/download_metrics"
    export_prometheus: bool = True
    export_json: bool = True
    export_csv: bool = False

    # Export format: "json", "csv", or "both" (default: json)
    metrics_export_format: str = "json"

    # File settings
    include_timestamp: bool = True

    # Graceful failure handling
    continue_on_export_error: bool = True

    # US-143-009: HTTP server for /metrics endpoint
    # Enable HTTP server for Prometheus scraping
    enable_http_server: bool = False
    # HTTP port for metrics endpoint (default: 9090)
    http_port: int = 9090
    # HTTP host to bind to (default: "0.0.0.0")
    http_host: str = "0.0.0.0"


@dataclass
class DownloadMetrics:
    """Download-specific metrics for Prometheus export.

    Tracks cumulative metrics that can be exported at intervals.
    """

    # Cumulative counters (reset on export)
    download_total: int = 0
    download_successful: int = 0
    download_failed: int = 0
    download_duration_seconds: float = 0.0
    download_size_bytes: int = 0
    download_errors: int = 0
    retry_count: int = 0

    # Gauges (current state)
    active_downloads: int = 0

    # Per-keyword metrics (US-114-003)
    keyword_success: Dict[str, int] = field(default_factory=dict)
    keyword_failure: Dict[str, int] = field(default_factory=dict)

    # Per-duration-tier metrics (US-114-003)
    tier_success: Dict[str, int] = field(default_factory=dict)
    tier_failure: Dict[str, int] = field(default_factory=dict)

    # Per-region (VPN exit) metrics (US-114-003)
    region_success: Dict[str, int] = field(default_factory=dict)
    region_failure: Dict[str, int] = field(default_factory=dict)

    # Time-series for rate limit events (US-114-003)
    rate_limit_events_log: List[Dict[str, Any]] = field(default_factory=list)

    # US-129-012: Download-specific resource metrics
    # Bandwidth tracking
    bandwidth_bytes_total: int = 0
    bandwidth_mbps_samples: List[float] = field(default_factory=list)

    # Concurrent downloads tracking (for avg/max calculation)
    concurrent_downloads_samples: List[int] = field(default_factory=list)

    # Per-download resource tracking (optional CPU/memory per download)
    cpu_percent_samples: List[float] = field(default_factory=list)
    memory_mb_samples: List[float] = field(default_factory=list)

    # Duration samples for average calculation
    duration_samples: List[float] = field(default_factory=list)

    # US-143-009: Histogram for download duration buckets
    # Standard Prometheus histogram buckets: 0.1, 0.5, 1, 2, 5, 10, 30, 60 seconds
    duration_buckets: Dict[str, int] = field(default_factory=lambda: {
        "0.1": 0, "0.5": 0, "1": 0, "2": 0, "5": 0, "10": 0, "30": 0, "60": 0, "+Inf": 0
    })

    # US-143-009: Queue size gauge
    queue_size: int = 0

    # US-143-006: VPN server switch metrics
    vpn_server_switches: int = 0
    vpn_server_switch_log: List[Dict[str, Any]] = field(default_factory=list)

    # US-144-012: Per-tier duration tracking for average calculation
    tier_durations: Dict[str, List[float]] = field(default_factory=dict)

    # US-144-012: Session-level bytes tracking (total across all exports)
    session_bytes_total: int = 0

    # US-144-012: Retry effectiveness tracking
    videos_saved_by_retry: int = 0  # Downloads that succeeded after retry
    videos_failed_after_retries: int = 0  # Downloads that failed even after retries
    retry_attempts_before_success: List[int] = field(default_factory=list)

    def reset(self) -> None:
        """Reset cumulative counters after export."""
        self.download_total = 0
        self.download_successful = 0
        self.download_failed = 0
        self.download_duration_seconds = 0.0
        self.download_size_bytes = 0
        self.download_errors = 0
        self.retry_count = 0
        # Reset per-keyword metrics (US-114-003)
        self.keyword_success.clear()
        self.keyword_failure.clear()
        self.tier_success.clear()
        self.tier_failure.clear()
        self.region_success.clear()
        self.region_failure.clear()
        self.rate_limit_events_log.clear()
        # Reset US-129-012 resource metrics
        self.bandwidth_bytes_total = 0
        self.bandwidth_mbps_samples.clear()
        self.concurrent_downloads_samples.clear()
        self.cpu_percent_samples.clear()
        self.memory_mb_samples.clear()
        self.duration_samples.clear()
        # Reset histogram buckets (US-143-009)
        self.duration_buckets = {
            "0.1": 0, "0.5": 0, "1": 0, "2": 0, "5": 0, "10": 0, "30": 0, "60": 0, "+Inf": 0
        }
        self.queue_size = 0
        # Reset VPN server switch metrics (US-143-006)
        self.vpn_server_switches = 0
        self.vpn_server_switch_log.clear()
        # Reset US-144-012 tier duration tracking
        self.tier_durations.clear()
        # Note: session_bytes_total is NOT reset - it's cumulative across session
        # Reset US-144-012 retry effectiveness tracking
        self.videos_saved_by_retry = 0
        self.videos_failed_after_retries = 0
        self.retry_attempts_before_success.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        # Calculate bandwidth metrics
        bandwidth_mbps_avg = 0.0
        bandwidth_mbps_peak = 0.0
        if self.bandwidth_mbps_samples:
            bandwidth_mbps_avg = round(sum(self.bandwidth_mbps_samples) / len(self.bandwidth_mbps_samples), 2)
            bandwidth_mbps_peak = round(max(self.bandwidth_mbps_samples), 2)

        # Calculate concurrent downloads metrics
        concurrent_avg = 0
        concurrent_max = 0
        if self.concurrent_downloads_samples:
            concurrent_avg = round(sum(self.concurrent_downloads_samples) / len(self.concurrent_downloads_samples), 2)
            concurrent_max = max(self.concurrent_downloads_samples)

        # Calculate CPU per download average
        cpu_avg = 0.0
        if self.cpu_percent_samples:
            cpu_avg = round(sum(self.cpu_percent_samples) / len(self.cpu_percent_samples), 2)

        # Calculate memory per download average
        memory_avg = 0.0
        if self.memory_mb_samples:
            memory_avg = round(sum(self.memory_mb_samples) / len(self.memory_mb_samples), 2)

        # Calculate average duration
        duration_avg = 0.0
        if self.duration_samples:
            duration_avg = round(sum(self.duration_samples) / len(self.duration_samples), 2)

        # US-144-012: Calculate average duration per tier
        tier_duration_avg = {}
        for tier, durations in self.tier_durations.items():
            if durations:
                tier_duration_avg[tier] = round(sum(durations) / len(durations), 2)

        # US-144-012: Calculate success rate by tier
        tier_success_rate = {}
        for tier in set(self.tier_success.keys()) | set(self.tier_failure.keys()):
            success = self.tier_success.get(tier, 0)
            failure = self.tier_failure.get(tier, 0)
            total = success + failure
            if total > 0:
                tier_success_rate[tier] = round((success / total) * 100, 2)

        # US-144-012: Calculate retry effectiveness
        # Effectiveness = successes / (successes + failures)
        total_retry_attempts = self.videos_saved_by_retry + self.videos_failed_after_retries

        retry_effectiveness = 0.0
        if total_retry_attempts > 0:
            retry_effectiveness = round((self.videos_saved_by_retry / total_retry_attempts) * 100, 2)

        avg_attempts_before_success = 0.0
        if self.retry_attempts_before_success:
            avg_attempts_before_success = round(
                sum(self.retry_attempts_before_success) / len(self.retry_attempts_before_success), 2
            )

        return {
            "download_total": self.download_total,
            "download_successful": self.download_successful,
            "download_failed": self.download_failed,
            "download_duration_seconds": round(self.download_duration_seconds, 3),
            "download_duration_avg_seconds": duration_avg,
            "download_size_mb": round(self.download_size_bytes / (1024 * 1024), 2),
            "download_errors": self.download_errors,
            "retry_count": self.retry_count,
            "active_downloads": self.active_downloads,
            "queue_size": self.queue_size,
            # US-143-009: Histogram buckets
            "duration_buckets": dict(self.duration_buckets),
            # US-129-012: Resource metrics
            "bandwidth_bytes_total": self.bandwidth_bytes_total,
            "bandwidth_mbps_avg": bandwidth_mbps_avg,
            "bandwidth_mbps_peak": bandwidth_mbps_peak,
            "concurrent_downloads_avg": concurrent_avg,
            "concurrent_downloads_max": concurrent_max,
            "cpu_percent_per_download": cpu_avg,
            "memory_mb_per_download": memory_avg,
            "keyword_success": dict(self.keyword_success),
            "keyword_failure": dict(self.keyword_failure),
            "tier_success": dict(self.tier_success),
            "tier_failure": dict(self.tier_failure),
            "region_success": dict(self.region_success),
            "region_failure": dict(self.region_failure),
            "rate_limit_events": len(self.rate_limit_events_log),
            # US-143-006: VPN server switch metrics
            "vpn_server_switches": self.vpn_server_switches,
            "vpn_server_switch_log": list(self.vpn_server_switch_log),
            # US-144-012: Tier duration averages
            "tier_duration_avg_seconds": tier_duration_avg,
            # US-144-012: Success rate by tier
            "tier_success_rate_percent": tier_success_rate,
            # US-144-012: Session-level bytes
            "session_bytes_total": self.session_bytes_total,
            # US-144-012: Retry effectiveness
            "videos_saved_by_retry": self.videos_saved_by_retry,
            "videos_failed_after_retries": self.videos_failed_after_retries,
            "retry_effectiveness_percent": retry_effectiveness,
            "avg_attempts_before_success": avg_attempts_before_success,
        }


class DownloadMetricsExporter:
    """Export download metrics to Prometheus and JSON formats.

    Provides configurable interval-based export with graceful error handling.
    Integrates with RateLimitMetrics and GlobalRateLimitCoordinator.

    Example:
        exporter = DownloadMetricsExporter(config)
        exporter.start()

        # During downloads:
        exporter.record_download_start()
        exporter.record_download_complete(duration=5.2, size_bytes=1024000)

        # At shutdown:
        exporter.stop()
        exporter.export()  # Final export
    """

    def __init__(
        self,
        config: Optional[DownloadMetricsConfig] = None,
        rate_limit_metrics: Optional['RateLimitMetrics'] = None,
        coordinator: Optional['GlobalRateLimitCoordinator'] = None,
        speed_tracker: Optional['DownloadSpeedTracker'] = None,
        escalation_manager: Optional['EscalationManager'] = None,
    ):
        """Initialize metrics exporter.

        Args:
            config: Export configuration. If None, uses defaults.
            rate_limit_metrics: Optional RateLimitMetrics to include in exports.
            coordinator: Optional GlobalRateLimitCoordinator for slot metrics.
            speed_tracker: Optional DownloadSpeedTracker for speed analytics (US-129-006)
            escalation_manager: Optional EscalationManager for rate limit predictions (US-143-004)
        """
        self.config = config or DownloadMetricsConfig()
        self._rate_limit_metrics = rate_limit_metrics
        self._coordinator = coordinator
        self._speed_tracker = speed_tracker
        self._escalation_manager = escalation_manager

        self._metrics = DownloadMetrics()
        self._lock = threading.Lock()

        self._running = False
        self._export_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track export times for interval-based export
        self._last_export_time: Optional[float] = None

        # US-143-009: HTTP server for /metrics endpoint
        self._http_server: Optional[http.server.HTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

    def set_rate_limit_metrics(self, metrics: 'RateLimitMetrics') -> None:
        """Set the RateLimitMetrics instance to include in exports."""
        self._rate_limit_metrics = metrics

    def set_coordinator(self, coordinator: 'GlobalRateLimitCoordinator') -> None:
        """Set the GlobalRateLimitCoordinator instance to include in exports."""
        self._coordinator = coordinator

    def set_speed_tracker(self, speed_tracker: 'DownloadSpeedTracker') -> None:
        """Set the DownloadSpeedTracker instance to include in exports (US-129-006)."""
        self._speed_tracker = speed_tracker

    def set_escalation_manager(self, escalation_manager: 'EscalationManager') -> None:
        """Set the EscalationManager instance for rate limit predictions (US-143-004)."""
        self._escalation_manager = escalation_manager

    def get_speed_analytics(self) -> Dict[str, Any]:
        """Get speed analytics data for export (US-129-006).

        Returns:
            Dictionary with speed analytics including avg/median speed,
            speed trends, slow video detection, and rate limit warnings.
        """
        if self._speed_tracker is None:
            return {"enabled": False, "message": "Speed tracker not available"}

        try:
            stats = self._speed_tracker.get_speed_stats()
        except Exception as e:
            return {"enabled": True, "error": str(e)}

        if not stats or stats.get('samples', 0) == 0:
            return {
                "enabled": True,
                "samples": 0,
                "message": "No speed data available"
            }

        speeds = [r['speed_mbps'] for r in stats.get('records', [])]

        # Calculate median
        median_speed = round(statistics.median(speeds), 2) if speeds else 0.0

        # Determine speed trend over last N downloads
        trend = self._calculate_speed_trend(speeds)

        # Detect slow videos
        slow_threshold = 0.5  # MB/s - matches config.slow_download_warning_threshold
        slow_videos = [
            {
                "video_id": r['video_id'],
                "speed_mbps": r['speed_mbps'],
                "tier": r['tier']
            }
            for r in stats.get('records', [])
            if r['speed_mbps'] < slow_threshold
        ]

        # Check for sustained degradation (rate limit early warning)
        degradation_signal = None
        try:
            deg = self._speed_tracker.detect_sustained_degradation()
            if deg.detected:
                degradation_signal = {
                    "detected": True,
                    "degradation_percentage": deg.degradation_percentage,
                    "trend": deg.trend,
                    "message": deg.message,
                    "peak_speed_mbps": deg.peak_speed_mbps,
                    "current_speed_mbps": deg.current_speed_mbps
                }
        except Exception:
            pass

        # Check for rate limit signals
        rate_limit_signal = None
        try:
            rls = self._speed_tracker.detect_rate_limit_signals()
            if rls.detected:
                rate_limit_signal = {
                    "detected": True,
                    "consecutive_slow_count": rls.consecutive_slow_count,
                    "threshold": rls.threshold,
                    "message": rls.message
                }
        except Exception:
            pass

        return {
            "enabled": True,
            "samples": stats.get('samples', 0),
            "avg_speed_mbps": stats.get('avg_speed_mbps', 0.0),
            "median_speed_mbps": median_speed,
            "speed_std_dev": stats.get('std_dev_mbps', 0.0),
            "min_speed_mbps": stats.get('min_speed_mbps', 0.0),
            "max_speed_mbps": stats.get('max_speed_mbps', 0.0),
            "speed_trend": trend,
            "slow_videos": slow_videos,
            "slow_video_count": len(slow_videos),
            "sustained_degradation_warning": degradation_signal,
            "rate_limit_signal": rate_limit_signal,
            "variance_category": stats.get('variance_category', 'unknown'),
            "coefficient_of_variation": stats.get('coefficient_of_variation', 0.0),
            "per_video": [
                {
                    "video_id": r['video_id'],
                    "speed_mbps": r['speed_mbps'],
                    "bytes": r['bytes'],
                    "duration": r['duration'],
                    "tier": r['tier']
                }
                for r in stats.get('records', [])
            ]
        }

    def get_download_resource_metrics(self) -> Dict[str, Any]:
        """Get download-specific resource metrics compatible with resource_metrics schema (US-129-012).

        This method formats download resource metrics to be compatible with the
        existing pipeline resource_metrics export format.

        Returns:
            Dictionary with download-specific resource metrics:
            - bandwidth_bytes_total: Total bytes downloaded
            - bandwidth_mbps_avg: Average bandwidth in Mbps
            - bandwidth_mbps_peak: Peak bandwidth in Mbps
            - concurrent_downloads_avg: Average concurrent downloads
            - concurrent_downloads_max: Maximum concurrent downloads
            - cpu_percent_per_download: Average CPU usage per download
            - memory_mb_per_download: Average memory usage per download in MB
            - download_duration_avg_seconds: Average download duration
            - download_duration_total_seconds: Total download duration
        """
        with self._lock:
            metrics = self._metrics.to_dict()

        return {
            "stage_name": "DOWNLOAD_SEGMENTS",
            "format": "download_resource",
            "bandwidth_bytes_total": metrics.get("bandwidth_bytes_total", 0),
            "bandwidth_mbps_avg": metrics.get("bandwidth_mbps_avg", 0.0),
            "bandwidth_mbps_peak": metrics.get("bandwidth_mbps_peak", 0.0),
            "concurrent_downloads_avg": metrics.get("concurrent_downloads_avg", 0.0),
            "concurrent_downloads_max": metrics.get("concurrent_downloads_max", 0),
            "cpu_percent_per_download": metrics.get("cpu_percent_per_download", 0.0),
            "memory_mb_per_download": metrics.get("memory_mb_per_download", 0.0),
            "download_duration_avg_seconds": metrics.get("download_duration_avg_seconds", 0.0),
            "download_duration_total_seconds": metrics.get("download_duration_seconds", 0.0),
            "download_total": metrics.get("download_total", 0),
            "download_successful": metrics.get("download_successful", 0),
            "download_failed": metrics.get("download_failed", 0),
            "active_downloads": metrics.get("active_downloads", 0),
        }

    def _calculate_speed_trend(self, speeds: List[float]) -> str:
        """Calculate speed trend over recent downloads.

        Args:
            speeds: List of speeds in chronological order (oldest first)

        Returns:
            Trend string: 'increasing', 'decreasing', 'stable', or 'insufficient_data'
        """
        if len(speeds) < 3:
            return "insufficient_data"

        # Use first half vs second half comparison
        mid = len(speeds) // 2
        first_half = speeds[:mid]
        second_half = speeds[mid:]

        if not first_half or not second_half:
            return "stable"

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        # 10% threshold for trend detection
        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_half[0] * 0.9:
            # More strict check for decreasing - compare to first speed
            return "decreasing"
        else:
            return "stable"

    def record_download_start(self) -> None:
        """Record the start of a download (increment active downloads)."""
        with self._lock:
            self._metrics.active_downloads += 1

    def record_download_complete(
        self,
        duration: float,
        size_bytes: int,
        success: bool = True,
        keyword: Optional[str] = None,
        tier: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        """Record a completed download.

        Args:
            duration: Download duration in seconds.
            size_bytes: Downloaded file size in bytes.
            success: Whether the download was successful.
            keyword: Optional keyword for per-keyword tracking (US-114-003)
            tier: Optional duration tier for per-tier tracking (US-114-003)
            region: Optional VPN exit region for per-region tracking (US-114-003)
        """
        with self._lock:
            self._metrics.download_total += 1
            self._metrics.download_duration_seconds += duration
            self._metrics.download_size_bytes += size_bytes
            self._metrics.active_downloads = max(0, self._metrics.active_downloads - 1)

            # US-129-012: Track resource metrics
            # Track duration samples for average calculation
            self._metrics.duration_samples.append(duration)

            # Track bandwidth (bytes to mbps: bytes * 8 / 1,000,000 / duration)
            if duration > 0:
                mbps = (size_bytes * 8) / 1_000_000 / duration
                self._metrics.bandwidth_mbps_samples.append(mbps)

            # Track total bandwidth bytes
            self._metrics.bandwidth_bytes_total += size_bytes

            # US-143-009: Track duration in histogram buckets
            self._add_to_duration_buckets(duration)

            # US-144-012: Track per-tier durations for average calculation
            if tier:
                if tier not in self._metrics.tier_durations:
                    self._metrics.tier_durations[tier] = []
                self._metrics.tier_durations[tier].append(duration)

            # US-144-012: Track session-level bytes (cumulative across exports)
            self._metrics.session_bytes_total += size_bytes

            if success:
                self._metrics.download_successful += 1
                # Per-keyword success tracking
                if keyword:
                    self._metrics.keyword_success[keyword] = (
                        self._metrics.keyword_success.get(keyword, 0) + 1
                    )
                # Per-tier success tracking
                if tier:
                    self._metrics.tier_success[tier] = (
                        self._metrics.tier_success.get(tier, 0) + 1
                    )
                # Per-region success tracking
                if region:
                    self._metrics.region_success[region] = (
                        self._metrics.region_success.get(region, 0) + 1
                    )
            else:
                self._metrics.download_failed += 1
                self._metrics.download_errors += 1
                # Per-keyword failure tracking
                if keyword:
                    self._metrics.keyword_failure[keyword] = (
                        self._metrics.keyword_failure.get(keyword, 0) + 1
                    )
                # Per-tier failure tracking
                if tier:
                    self._metrics.tier_failure[tier] = (
                        self._metrics.tier_failure.get(tier, 0) + 1
                    )
                # Per-region failure tracking
                if region:
                    self._metrics.region_failure[region] = (
                        self._metrics.region_failure.get(region, 0) + 1
                    )

    def record_download_error(self) -> None:
        """Record a download error."""
        with self._lock:
            self._metrics.download_errors += 1
            self._metrics.active_downloads = max(0, self._metrics.active_downloads - 1)

    def _add_to_duration_buckets(self, duration: float) -> None:
        """Add duration to histogram buckets (US-143-009).

        Args:
            duration: Download duration in seconds.
        """
        # Standard histogram buckets: 0.1, 0.5, 1, 2, 5, 10, 30, 60, +Inf
        bucket_thresholds = [0.1, 0.5, 1, 2, 5, 10, 30, 60]

        for threshold in bucket_thresholds:
            if duration <= threshold:
                self._metrics.duration_buckets[str(threshold)] += 1
                break
        else:
            # Duration > 60 seconds
            self._metrics.duration_buckets["+Inf"] += 1

    def record_queue_size(self, size: int) -> None:
        """Record the current download queue size (US-143-009).

        Args:
            size: Current number of items in download queue.
        """
        with self._lock:
            self._metrics.queue_size = size

    def record_retry(self) -> None:
        """Record a download retry."""
        with self._lock:
            self._metrics.retry_count += 1

    def record_retry_success(self, attempts: int = 1) -> None:
        """Record a download that succeeded after retry (US-144-012).

        Args:
            attempts: Number of retry attempts before success.
        """
        with self._lock:
            self._metrics.videos_saved_by_retry += 1
            self._metrics.retry_attempts_before_success.append(attempts)

    def record_retry_failure(self) -> None:
        """Record a download that failed even after retries (US-144-012)."""
        with self._lock:
            self._metrics.videos_failed_after_retries += 1

    def record_download_duration(self, duration: float) -> None:
        """Record download duration without marking complete.

        Used for in-progress tracking.
        """
        with self._lock:
            self._metrics.download_duration_seconds += duration

    def record_concurrent_downloads(self, count: int) -> None:
        """Record concurrent download count for avg/max calculation (US-129-012).

        Call this periodically during download operations to track concurrency.

        Args:
            count: Number of currently active downloads.
        """
        with self._lock:
            self._metrics.concurrent_downloads_samples.append(count)
            # Also update the active_downloads gauge
            self._metrics.active_downloads = count

    def record_download_resources(
        self,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None,
    ) -> None:
        """Record CPU and memory usage per download (US-129-012).

        Call this during or after download to track per-download resource usage.

        Args:
            cpu_percent: CPU usage percentage for this download.
            memory_mb: Memory usage in MB for this download.
        """
        with self._lock:
            if cpu_percent is not None:
                self._metrics.cpu_percent_samples.append(cpu_percent)
            if memory_mb is not None:
                self._metrics.memory_mb_samples.append(memory_mb)

    def record_rate_limit_event(
        self,
        keyword: Optional[str] = None,
        tier: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        """Record a rate limit event with timestamp for time-series export.

        Args:
            keyword: Optional keyword associated with the event (US-114-003)
            tier: Optional duration tier associated with the event (US-114-003)
            region: Optional VPN exit region (US-114-003)
        """
        with self._lock:
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "keyword": keyword,
                "tier": tier,
                "region": region,
            }
            self._metrics.rate_limit_events_log.append(event)

    def get_rate_limit_timeseries(self) -> List[Dict[str, Any]]:
        """Get rate limit events as time-series data.

        Returns:
            List of rate limit event dicts with timestamps.
        """
        with self._lock:
            return list(self._metrics.rate_limit_events_log)

    def start(self) -> None:
        """Start the interval-based export background thread.

        Does nothing if not enabled or already running.
        """
        if not self.config.enabled or self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._last_export_time = time.time()

        self._export_thread = threading.Thread(
            target=self._export_loop,
            daemon=True,
            name="DownloadMetricsExporter",
        )
        self._export_thread.start()

        # US-143-009: Start HTTP server for /metrics endpoint
        self._start_http_server()

        logger.info("Download metrics exporter started")

    def stop(self) -> None:
        """Stop the interval-based export and perform final export."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._export_thread and self._export_thread.is_alive():
            self._export_thread.join(timeout=5.0)

        # Stop HTTP server (US-143-009)
        self._stop_http_server()

        # Final export
        try:
            self.export()
        except Exception as e:
            logger.warning(f"Final metrics export failed: {e}")

        logger.info("Download metrics exporter stopped")

    def _export_loop(self) -> None:
        """Background loop for interval-based export."""
        while self._running and not self._stop_event.is_set():
            try:
                # Calculate time since last export
                now = time.time()
                if self._last_export_time is None:
                    self._last_export_time = now
                    continue

                elapsed = now - self._last_export_time

                if elapsed >= self.config.export_interval_seconds:
                    self.export()
                    self._last_export_time = now

            except Exception as e:
                logger.warning(f"Metrics export loop error: {e}")

            # Small sleep to prevent CPU spinning
            self._stop_event.wait(1.0)

    def _start_http_server(self) -> None:
        """Start HTTP server for /metrics endpoint (US-143-009)."""
        if not getattr(self.config, 'enable_http_server', False):
            return

        http_host = getattr(self.config, 'http_host', '0.0.0.0')
        http_port = getattr(self.config, 'http_port', 9090)

        try:
            # Create handler that captures self reference
            metrics_exporter = self

            class MetricsHandler(http.server.BaseHTTPRequestHandler):
                """HTTP handler for /metrics endpoint."""

                def do_GET(self):
                    if self.path == "/metrics" or self.path == "/metrics/":
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain; version=0.0.4")
                        self.end_headers()

                        # Get current metrics in Prometheus format
                        try:
                            data = metrics_exporter._get_metrics_data()
                            # Generate Prometheus format directly
                            output = metrics_exporter._generate_prometheus_format(data)
                            self.wfile.write(output.encode("utf-8"))
                        except Exception as e:
                            self.wfile.write(f"# Error generating metrics: {e}".encode("utf-8"))
                    elif self.path == "/health" or self.path == "/health/":
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"OK")
                    else:
                        self.send_response(404)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"Not Found")

                def log_message(self, format, *args):
                    # Suppress HTTP server logs
                    pass

            self._http_server = http.server.HTTPServer((http_host, http_port), MetricsHandler)
            self._http_thread = threading.Thread(
                target=self._http_server.serve_forever,
                daemon=True,
                name="MetricsHTTPServer"
            )
            self._http_thread.start()
            logger.info(f"Metrics HTTP server started on {http_host}:{http_port}")

        except Exception as e:
            logger.warning(f"Failed to start metrics HTTP server: {e}")

    def _stop_http_server(self) -> None:
        """Stop HTTP server for /metrics endpoint (US-143-009)."""
        if self._http_server:
            try:
                self._http_server.shutdown()
                self._http_server = None
                logger.info("Metrics HTTP server stopped")
            except Exception as e:
                logger.warning(f"Error stopping metrics HTTP server: {e}")

        if self._http_thread and self._http_thread.is_alive():
            self._http_thread.join(timeout=2.0)
            self._http_thread = None

    def _generate_prometheus_format(self, data: Dict[str, Any]) -> str:
        """Generate Prometheus format string from metrics data.

        This is used by the HTTP endpoint to serve /metrics.

        Args:
            data: Metrics data dictionary.

        Returns:
            Prometheus format string.
        """
        # Reuse the _export_prometheus logic but return string instead of writing to file
        lines: List[str] = []

        downloads = data.get("downloads", {})

        lines.extend([
            "# HELP download_total Total number of download attempts",
            "# TYPE download_total counter",
            f'download_total {downloads.get("download_total", 0)}',
            "",
            "# HELP download_successful Total successful downloads",
            "# TYPE download_successful counter",
            f'download_successful {downloads.get("download_successful", 0)}',
            "",
            "# HELP download_failed Total failed downloads",
            "# TYPE download_failed counter",
            f'download_failed {downloads.get("download_failed", 0)}',
            "",
            "# HELP download_duration_seconds Total download duration in seconds",
            "# TYPE download_duration_seconds counter",
            f'download_duration_seconds {downloads.get("download_duration_seconds", 0)}',
            "",
            "# HELP download_size_bytes Total bytes downloaded",
            "# TYPE download_size_bytes counter",
            f'download_size_bytes {downloads.get("download_size_bytes", 0)}',
            "",
            "# HELP download_errors Total download errors",
            "# TYPE download_errors counter",
            f'download_errors {downloads.get("download_errors", 0)}',
            "",
            "# HELP download_active Current number of active downloads",
            "# TYPE download_active gauge",
            f'download_active {downloads.get("active_downloads", 0)}',
            "",
            "# HELP download_queue_size Current number of items in download queue",
            "# TYPE download_queue_size gauge",
            f'download_queue_size {downloads.get("queue_size", 0)}',
            "",
            "# HELP download_retry_count Total retry attempts",
            "# TYPE download_retry_count counter",
            f'download_retry_count {downloads.get("retry_count", 0)}',
            "",
        ])

        # Histogram
        duration_buckets = downloads.get("duration_buckets", {})
        if duration_buckets:
            cumulative = 0
            bucket_lines = []
            bucket_order = ["0.1", "0.5", "1", "2", "5", "10", "30", "60", "+Inf"]

            for bucket in bucket_order:
                count = duration_buckets.get(bucket, 0)
                cumulative += count
                bucket_lines.append(f'download_duration_seconds_bucket{{le="{bucket}"}} {cumulative}')

            lines.extend([
                "# HELP download_duration_seconds Download duration histogram",
                "# TYPE download_duration_seconds histogram",
            ])
            for bucket_line in bucket_lines:
                lines.append(bucket_line)
            lines.append(f'download_duration_seconds_sum {downloads.get("download_duration_seconds", 0)}')
            lines.append(f'download_duration_seconds_count {downloads.get("download_total", 0)}')
            lines.append("")

        # Rate limit metrics
        rate_limits = data.get("rate_limits", {})
        if rate_limits:
            lines.extend([
                "# HELP download_retry_attempts Total retry attempts from rate limiter",
                "# TYPE download_retry_attempts counter",
                f'download_retry_attempts {rate_limits.get("retry_attempts", 0)}',
                "",
                "# HELP download_rate_limit_events Total rate limit events",
                "# TYPE download_rate_limit_events counter",
                f'download_rate_limit_events {rate_limits.get("rate_limit_events", 0)}',
                "",
                "# HELP download_backoff_attempts Total backoff attempts",
                "# TYPE download_backoff_attempts counter",
                f'download_backoff_attempts {rate_limits.get("backoff_attempts", 0)}',
                "",
                "# HELP download_cookie_rotations Total cookie rotations",
                "# TYPE download_cookie_rotations counter",
                f'download_cookie_rotations {rate_limits.get("cookie_rotations", 0)}',
                "",
                "# HELP download_vpn_switches Total VPN switches",
                "# TYPE download_vpn_switches counter",
                f'download_vpn_switches {rate_limits.get("vpn_switches", 0)}',
                "",
            ])

        return "\n".join(lines)

    def export(self) -> Optional[Dict[str, Path]]:
        """Export metrics to configured formats.

        Returns:
            Dict mapping format name to output file path, or None if not enabled.
        """
        if not self.config.enabled:
            return None

        exported_files: Dict[str, Path] = {}

        try:
            # Get current metrics snapshot
            metrics_data = self._get_metrics_data()
        except Exception as e:
            if not self.config.continue_on_export_error:
                raise
            logger.warning(f"Failed to collect metrics data: {e}")
            return None

        # Determine export format based on config
        export_format = getattr(self.config, 'metrics_export_format', 'json')

        # Export to Prometheus format
        if self.config.export_prometheus:
            try:
                output_path = self._get_output_path("prometheus")
                self._export_prometheus(metrics_data, output_path)
                exported_files["prometheus"] = output_path
            except Exception as e:
                if not self.config.continue_on_export_error:
                    raise
                logger.warning(f"Prometheus export failed: {e}")

        # Export to JSON format (if format is json or both)
        if self.config.export_json and export_format in ("json", "both"):
            try:
                output_path = self._get_output_path("json")
                self._export_json(metrics_data, output_path)
                exported_files["json"] = output_path
            except Exception as e:
                if not self.config.continue_on_export_error:
                    raise
                logger.warning(f"JSON export failed: {e}")

        # Export to CSV format (if format is csv or both) (US-114-003)
        if self.config.export_csv and export_format in ("csv", "both"):
            try:
                output_path = self._get_output_path("csv")
                self._export_csv(metrics_data, output_path)
                exported_files["csv"] = output_path
            except Exception as e:
                if not self.config.continue_on_export_error:
                    raise
                logger.warning(f"CSV export failed: {e}")

        # Reset cumulative counters after successful export
        with self._lock:
            self._metrics.reset()

        return exported_files

    def _get_metrics_data(self) -> Dict[str, Any]:
        """Collect all metrics data for export."""
        with self._lock:
            metrics_snapshot = self._metrics.to_dict()

        # Build export data
        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "downloads": metrics_snapshot,
        }

        # Include rate limit metrics if available
        if self._rate_limit_metrics is not None:
            try:
                export_data["rate_limits"] = {
                    "retry_attempts": self._rate_limit_metrics.retry_attempts,
                    "max_retry_count_reached": self._rate_limit_metrics.max_retry_count_reached,
                    "rate_limit_events": self._rate_limit_metrics.rate_limit_events,
                    "backoff_attempts": self._rate_limit_metrics.backoff_attempts,
                    "time_spent_backing_off": self._rate_limit_metrics.time_spent_backing_off,
                    "cookie_rotations": self._rate_limit_metrics.cookie_rotations,
                    "vpn_switches": self._rate_limit_metrics.vpn_switches,
                    "circuit_breaker_trips": self._rate_limit_metrics.circuit_breaker_trips,
                    "success_rate": self._rate_limit_metrics.success_rate,
                }
            except Exception as e:
                logger.debug(f"Failed to include rate limit metrics: {e}")

        # Include coordinator metrics if available
        if self._coordinator is not None:
            try:
                coordinator_stats = self._coordinator.get_stats()
                export_data["rate_limit_coordinator"] = coordinator_stats
            except Exception as e:
                logger.debug(f"Failed to include coordinator metrics: {e}")

        # Include time-series data for rate limit events (US-114-003)
        with self._lock:
            export_data["timeseries"] = {
                "rate_limit_events": list(self._metrics.rate_limit_events_log),
            }

        # Include speed analytics (US-129-006)
        if self._speed_tracker is not None:
            try:
                export_data["speed_analytics"] = self.get_speed_analytics()
            except Exception as e:
                logger.debug(f"Failed to include speed analytics: {e}")

        # Include rate limit prediction (US-143-004)
        if self._escalation_manager is not None:
            try:
                prediction = self._escalation_manager.get_prediction_for_metrics()
                if prediction:
                    export_data["rate_limit_prediction"] = prediction
            except Exception as e:
                logger.debug(f"Failed to include rate limit prediction: {e}")

        return export_data

    def _get_output_path(self, format: str) -> Path:
        """Get output path for the given format."""
        base_dir = Path(self.config.output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = f"download_metrics.{format}"
        if self.config.include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"download_metrics_{timestamp}.{format}"

        return base_dir / filename

    def _export_prometheus(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export metrics to Prometheus text format.

        Args:
            data: Metrics data dictionary.
            output_path: File path to write Prometheus format.
        """
        lines: List[str] = []

        # Download metrics
        downloads = data.get("downloads", {})

        lines.extend([
            "# HELP download_total Total number of download attempts",
            "# TYPE download_total counter",
            f'download_total {downloads.get("download_total", 0)}',
            "",
            "# HELP download_successful Total successful downloads",
            "# TYPE download_successful counter",
            f'download_successful {downloads.get("download_successful", 0)}',
            "",
            "# HELP download_failed Total failed downloads",
            "# TYPE download_failed counter",
            f'download_failed {downloads.get("download_failed", 0)}',
            "",
            "# HELP download_duration_seconds Total download duration in seconds",
            "# TYPE download_duration_seconds counter",
            f'download_duration_seconds {downloads.get("download_duration_seconds", 0)}',
            "",
            "# HELP download_size_bytes Total bytes downloaded",
            "# TYPE download_size_bytes counter",
            f'download_size_bytes {downloads.get("download_size_bytes", 0)}',
            "",
            "# HELP download_errors Total download errors",
            "# TYPE download_errors counter",
            f'download_errors {downloads.get("download_errors", 0)}',
            "",
            "# HELP download_active Current number of active downloads",
            "# TYPE download_active gauge",
            f'download_active {downloads.get("active_downloads", 0)}',
            "",
            "# HELP download_retry_count Total retry attempts",
            "# TYPE download_retry_count counter",
            f'download_retry_count {downloads.get("retry_count", 0)}',
            "",
            # US-143-009: Queue size gauge
            "# HELP download_queue_size Current number of items in download queue",
            "# TYPE download_queue_size gauge",
            f'download_queue_size {downloads.get("queue_size", 0)}',
            "",
        ])

        # US-143-009: Histogram for download duration
        duration_buckets = downloads.get("duration_buckets", {})
        if duration_buckets:
            # Calculate cumulative bucket counts for Prometheus histogram format
            cumulative = 0
            bucket_lines = []
            # Prometheus histogram requires buckets in increasing order with +Inf last
            bucket_order = ["0.1", "0.5", "1", "2", "5", "10", "30", "60", "+Inf"]

            for bucket in bucket_order:
                count = duration_buckets.get(bucket, 0)
                cumulative += count
                bucket_lines.append(f'download_duration_seconds_bucket{{le="{bucket}"}} {cumulative}')

            lines.extend([
                "# HELP download_duration_seconds Download duration histogram",
                "# TYPE download_duration_seconds histogram",
            ])
            for bucket_line in bucket_lines:
                lines.append(bucket_line)
            # Add sum and count
            lines.append(f'download_duration_seconds_sum {downloads.get("download_duration_seconds", 0)}')
            lines.append(f'download_duration_seconds_count {downloads.get("download_total", 0)}')
            lines.append("")

        # Rate limit metrics (from RateLimitMetrics)
        rate_limits = data.get("rate_limits", {})
        if rate_limits:
            lines.extend([
                "# HELP download_retry_attempts Total retry attempts from rate limiter",
                "# TYPE download_retry_attempts counter",
                f'download_retry_attempts {rate_limits.get("retry_attempts", 0)}',
                "",
                "# HELP download_rate_limit_events Total rate limit events",
                "# TYPE download_rate_limit_events counter",
                f'download_rate_limit_events {rate_limits.get("rate_limit_events", 0)}',
                "",
                "# HELP download_backoff_attempts Total backoff attempts",
                "# TYPE download_backoff_attempts counter",
                f'download_backoff_attempts {rate_limits.get("backoff_attempts", 0)}',
                "",
                "# HELP download_backoff_seconds Total seconds spent in backoff",
                "# TYPE download_backoff_seconds counter",
                f'download_backoff_seconds {rate_limits.get("time_spent_backing_off", 0)}',
                "",
                "# HELP download_cookie_rotations Total cookie rotations",
                "# TYPE download_cookie_rotations counter",
                f'download_cookie_rotations {rate_limits.get("cookie_rotations", 0)}',
                "",
                "# HELP download_vpn_switches Total VPN switches",
                "# TYPE download_vpn_switches counter",
                f'download_vpn_switches {rate_limits.get("vpn_switches", 0)}',
                "",
                "# HELP download_circuit_breaker_trips Total circuit breaker trips",
                "# TYPE download_circuit_breaker_trips counter",
                f'download_circuit_breaker_trips {rate_limits.get("circuit_breaker_trips", 0)}',
                "",
                "# HELP download_success_rate_percent Download success rate percentage",
                "# TYPE download_success_rate_percent gauge",
                f'download_success_rate_percent {rate_limits.get("success_rate", 0)}',
                "",
            ])

        # Coordinator metrics (from GlobalRateLimitCoordinator)
        coordinator = data.get("rate_limit_coordinator", {})
        if coordinator:
            lines.extend([
                "# HELP rate_limit_coordinator_slots_acquired Total slots acquired",
                "# TYPE rate_limit_coordinator_slots_acquired counter",
                f'rate_limit_coordinator_slots_acquired {coordinator.get("total_acquired", 0)}',
                "",
                "# HELP rate_limit_coordinator_waits Total times had to wait for slot",
                "# TYPE rate_limit_coordinator_waits counter",
                f'rate_limit_coordinator_waits {coordinator.get("total_waits", 0)}',
                "",
                "# HELP rate_limit_coordinator_wait_seconds Total wait time in seconds",
                "# TYPE rate_limit_coordinator_wait_seconds counter",
                f'rate_limit_coordinator_wait_seconds {coordinator.get("total_wait_time", 0)}',
                "",
                "# HELP rate_limit_coordinator_rejections Total slot acquisition rejections",
                "# TYPE rate_limit_coordinator_rejections counter",
                f'rate_limit_coordinator_rejections {coordinator.get("rejections", 0)}',
                "",
            ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.debug(f"Exported download metrics to {output_path}")

    def _export_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export metrics to JSON file.

        Args:
            data: Metrics data dictionary.
            output_path: File path to write JSON.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"Exported download metrics JSON to {output_path}")

    def _export_csv(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export metrics to CSV file with per-keyword, per-tier, per-region data.

        Creates multiple CSV files:
        - Main metrics CSV with summary statistics
        - Per-keyword breakdown CSV
        - Per-tier breakdown CSV
        - Per-region breakdown CSV
        - Time-series CSV for rate limit events

        Args:
            data: Metrics data dictionary.
            output_path: Base file path for CSV (will create _keywords.csv, _tier.csv, etc.)
        """
        downloads = data.get("downloads", {})

        # Main metrics CSV
        lines = ["metric,value"]
        lines.append(f"download_total,{downloads.get('download_total', 0)}")
        lines.append(f"download_successful,{downloads.get('download_successful', 0)}")
        lines.append(f"download_failed,{downloads.get('download_failed', 0)}")
        lines.append(f"download_duration_seconds,{downloads.get('download_duration_seconds', 0)}")
        lines.append(f"download_size_mb,{downloads.get('download_size_mb', 0)}")
        lines.append(f"download_errors,{downloads.get('download_errors', 0)}")
        lines.append(f"retry_count,{downloads.get('retry_count', 0)}")
        lines.append(f"active_downloads,{downloads.get('active_downloads', 0)}")

        # Rate limit data
        rate_limits = data.get("rate_limits", {})
        if rate_limits:
            lines.append(f"rate_limit_events,{rate_limits.get('rate_limit_events', 0)}")
            lines.append(f"backoff_attempts,{rate_limits.get('backoff_attempts', 0)}")
            lines.append(f"cookie_rotations,{rate_limits.get('cookie_rotations', 0)}")
            lines.append(f"vpn_switches,{rate_limits.get('vpn_switches', 0)}")

        # Write main metrics CSV
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        base_path = str(output_path).replace(".csv", "")

        # Per-keyword CSV
        keyword_success = downloads.get("keyword_success", {})
        keyword_failure = downloads.get("keyword_failure", {})
        all_keywords = set(keyword_success.keys()) | set(keyword_failure.keys())
        if all_keywords:
            kw_lines = ["keyword,success,failure,total"]
            for kw in sorted(all_keywords):
                s = keyword_success.get(kw, 0)
                f = keyword_failure.get(kw, 0)
                kw_lines.append(f"{kw},{s},{f},{s + f}")
            with open(f"{base_path}_keywords.csv", "w", encoding="utf-8") as f:
                f.write("\n".join(kw_lines))

        # Per-tier CSV
        tier_success = downloads.get("tier_success", {})
        tier_failure = downloads.get("tier_failure", {})
        all_tiers = set(tier_success.keys()) | set(tier_failure.keys())
        if all_tiers:
            tier_lines = ["tier,success,failure,total"]
            for tier in sorted(all_tiers):
                s = tier_success.get(tier, 0)
                f = tier_failure.get(tier, 0)
                tier_lines.append(f"{tier},{s},{f},{s + f}")
            with open(f"{base_path}_tier.csv", "w", encoding="utf-8") as f:
                f.write("\n".join(tier_lines))

        # Per-region CSV
        region_success = downloads.get("region_success", {})
        region_failure = downloads.get("region_failure", {})
        all_regions = set(region_success.keys()) | set(region_failure.keys())
        if all_regions:
            region_lines = ["region,success,failure,total"]
            for region in sorted(all_regions):
                s = region_success.get(region, 0)
                f = region_failure.get(region, 0)
                region_lines.append(f"{region},{s},{f},{s + f}")
            with open(f"{base_path}_region.csv", "w", encoding="utf-8") as f:
                f.write("\n".join(region_lines))

        # Time-series CSV for rate limit events
        timeseries = data.get("timeseries", {}).get("rate_limit_events", [])
        if timeseries:
            ts_lines = ["timestamp,keyword,tier,region"]
            for event in timeseries:
                ts_lines.append(
                    f"{event.get('timestamp', '')},{event.get('keyword', '')},"
                    f"{event.get('tier', '')},{event.get('region', '')}"
                )
            with open(f"{base_path}_timeseries.csv", "w", encoding="utf-8") as f:
                f.write("\n".join(ts_lines))

        logger.debug(f"Exported download metrics CSV to {output_path}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot (for external querying).

        Returns:
            Dictionary with current metrics values.
        """
        with self._lock:
            return self._metrics.to_dict()


def create_metrics_exporter(
    config: Optional[Dict[str, Any]] = None,
    rate_limit_metrics: Optional['RateLimitMetrics'] = None,
    coordinator: Optional['GlobalRateLimitCoordinator'] = None,
    speed_tracker: Optional['DownloadSpeedTracker'] = None,
) -> DownloadMetricsExporter:
    """Factory function to create a configured DownloadMetricsExporter.

    Args:
        config: Optional config dict. If None, uses defaults.
        rate_limit_metrics: Optional RateLimitMetrics instance.
        coordinator: Optional GlobalRateLimitCoordinator instance.
        speed_tracker: Optional DownloadSpeedTracker for speed analytics (US-129-006)

    Returns:
        Configured DownloadMetricsExporter instance.
    """
    if config:
        cfg = DownloadMetricsConfig(
            enabled=config.get("enabled", True),
            export_interval_seconds=config.get("export_interval_seconds", 60.0),
            output_dir=config.get("output_dir", "output/download_metrics"),
            export_prometheus=config.get("export_prometheus", True),
            export_json=config.get("export_json", True),
            export_csv=config.get("export_csv", False),
            metrics_export_format=config.get("metrics_export_format", "json"),
            include_timestamp=config.get("include_timestamp", True),
            continue_on_export_error=config.get("continue_on_export_error", True),
            # US-143-009: HTTP server options
            enable_http_server=config.get("enable_http_server", False),
            http_port=config.get("http_port", 9090),
            http_host=config.get("http_host", "0.0.0.0"),
        )
    else:
        cfg = DownloadMetricsConfig()

    exporter = DownloadMetricsExporter(
        config=cfg,
        rate_limit_metrics=rate_limit_metrics,
        coordinator=coordinator,
        speed_tracker=speed_tracker,
    )

    return exporter
