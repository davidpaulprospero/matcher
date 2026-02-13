"""Download metrics exporter for Prometheus and JSON formats.

Provides Prometheus-compatible metrics export for download operations,
integrating with RateLimitMetrics and GlobalRateLimitCoordinator.

Implements US-93-011: Add download metrics export.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .rate_limit_metrics import RateLimitMetrics
    from ..rate_limit.coordinator import GlobalRateLimitCoordinator

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

    # File settings
    include_timestamp: bool = True

    # Graceful failure handling
    continue_on_export_error: bool = True


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

    def reset(self) -> None:
        """Reset cumulative counters after export."""
        self.download_total = 0
        self.download_successful = 0
        self.download_failed = 0
        self.download_duration_seconds = 0.0
        self.download_size_bytes = 0
        self.download_errors = 0
        self.retry_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "download_total": self.download_total,
            "download_successful": self.download_successful,
            "download_failed": self.download_failed,
            "download_duration_seconds": round(self.download_duration_seconds, 3),
            "download_size_mb": round(self.download_size_bytes / (1024 * 1024), 2),
            "download_errors": self.download_errors,
            "retry_count": self.retry_count,
            "active_downloads": self.active_downloads,
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
    ):
        """Initialize metrics exporter.

        Args:
            config: Export configuration. If None, uses defaults.
            rate_limit_metrics: Optional RateLimitMetrics to include in exports.
            coordinator: Optional GlobalRateLimitCoordinator for slot metrics.
        """
        self.config = config or DownloadMetricsConfig()
        self._rate_limit_metrics = rate_limit_metrics
        self._coordinator = coordinator

        self._metrics = DownloadMetrics()
        self._lock = threading.Lock()

        self._running = False
        self._export_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track export times for interval-based export
        self._last_export_time: Optional[float] = None

    def set_rate_limit_metrics(self, metrics: 'RateLimitMetrics') -> None:
        """Set the RateLimitMetrics instance to include in exports."""
        self._rate_limit_metrics = metrics

    def set_coordinator(self, coordinator: 'GlobalRateLimitCoordinator') -> None:
        """Set the GlobalRateLimitCoordinator instance to include in exports."""
        self._coordinator = coordinator

    def record_download_start(self) -> None:
        """Record the start of a download (increment active downloads)."""
        with self._lock:
            self._metrics.active_downloads += 1

    def record_download_complete(
        self,
        duration: float,
        size_bytes: int,
        success: bool = True,
    ) -> None:
        """Record a completed download.

        Args:
            duration: Download duration in seconds.
            size_bytes: Downloaded file size in bytes.
            success: Whether the download was successful.
        """
        with self._lock:
            self._metrics.download_total += 1
            self._metrics.download_duration_seconds += duration
            self._metrics.download_size_bytes += size_bytes
            self._metrics.active_downloads = max(0, self._metrics.active_downloads - 1)

            if success:
                self._metrics.download_successful += 1
            else:
                self._metrics.download_failed += 1
                self._metrics.download_errors += 1

    def record_download_error(self) -> None:
        """Record a download error."""
        with self._lock:
            self._metrics.download_errors += 1
            self._metrics.active_downloads = max(0, self._metrics.active_downloads - 1)

    def record_retry(self) -> None:
        """Record a download retry."""
        with self._lock:
            self._metrics.retry_count += 1

    def record_download_duration(self, duration: float) -> None:
        """Record download duration without marking complete.

        Used for in-progress tracking.
        """
        with self._lock:
            self._metrics.download_duration_seconds += duration

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
        logger.info("Download metrics exporter started")

    def stop(self) -> None:
        """Stop the interval-based export and perform final export."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._export_thread and self._export_thread.is_alive():
            self._export_thread.join(timeout=5.0)

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

        # Export to JSON format
        if self.config.export_json:
            try:
                output_path = self._get_output_path("json")
                self._export_json(metrics_data, output_path)
                exported_files["json"] = output_path
            except Exception as e:
                if not self.config.continue_on_export_error:
                    raise
                logger.warning(f"JSON export failed: {e}")

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
        ])

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
) -> DownloadMetricsExporter:
    """Factory function to create a configured DownloadMetricsExporter.

    Args:
        config: Optional config dict. If None, uses defaults.
        rate_limit_metrics: Optional RateLimitMetrics instance.
        coordinator: Optional GlobalRateLimitCoordinator instance.

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
            include_timestamp=config.get("include_timestamp", True),
            continue_on_export_error=config.get("continue_on_export_error", True),
        )
    else:
        cfg = DownloadMetricsConfig()

    exporter = DownloadMetricsExporter(
        config=cfg,
        rate_limit_metrics=rate_limit_metrics,
        coordinator=coordinator,
    )

    return exporter
