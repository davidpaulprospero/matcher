"""Download speed monitoring for adaptive timeouts.

Tracks download speeds and adjusts timeouts dynamically based on network conditions.
Part of US-005: Implement download speed monitoring for adaptive timeouts.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class DownloadSpeedConfig:
    """Configuration for download speed monitoring.

    Attributes:
        enabled: Enable speed tracking (default: True)
        window_size: Number of downloads to track in sliding window (default: 5)
        min_speed_mbps: Minimum expected speed in MB/s (default: 1.0)
        max_timeout_multiplier: Maximum timeout extension (default: 2.0)
        enable_adaptive_timeout: Use speed data to extend timeouts (default: True)
    """
    enabled: bool = True
    window_size: int = 5
    min_speed_mbps: float = 1.0  # MB/s below which timeout gets extended
    max_timeout_multiplier: float = 2.0  # Maximum timeout extension (2x)
    enable_adaptive_timeout: bool = True


@dataclass
class DownloadRecord:
    """Record of a single download for speed tracking.

    Attributes:
        video_id: YouTube video ID
        bytes_downloaded: File size in bytes
        duration_seconds: Time taken to download in seconds
        timestamp: Unix timestamp when download completed
        tier: Download tier (short, medium, long, longer)
    """
    video_id: str
    bytes_downloaded: int
    duration_seconds: float
    timestamp: float
    tier: str

    @property
    def speed_mbps(self) -> float:
        """Calculate download speed in MB/s."""
        if self.duration_seconds <= 0:
            return 0.0
        return (self.bytes_downloaded / (1024 * 1024)) / self.duration_seconds


class DownloadSpeedTracker:
    """Track download speeds using a sliding window for adaptive timeouts.

    This class monitors actual download speeds and uses that data to dynamically
    adjust timeouts. On slow networks, timeouts can be extended up to 2x to prevent
    unnecessary timeout failures.

    Example:
        tracker = DownloadSpeedTracker(config)

        # Record a download
        start_time = time.time()
        # ... download happens ...
        tracker.record_download(video_id, file_size, time.time() - start_time, tier)

        # Get adjusted timeout
        base_timeout = 120
        adjusted = tracker.get_adjusted_timeout(base_timeout)  # May be up to 240

    Attributes:
        config: SpeedTracker configuration
        _records: Deque of recent download records (sliding window)
    """

    def __init__(self, config: DownloadSpeedConfig = None):
        """Initialize speed tracker.

        Args:
            config: Speed tracking configuration. Uses defaults if None.
        """
        self.config = config or DownloadSpeedConfig()
        self._records: deque[DownloadRecord] = deque(maxlen=self.config.window_size)

    def record_download(
        self,
        video_id: str,
        bytes_downloaded: int,
        duration_seconds: float,
        tier: str = "unknown"
    ) -> None:
        """Record a completed download for speed tracking.

        Args:
            video_id: YouTube video ID
            bytes_downloaded: File size in bytes
            duration_seconds: Time taken to download
            tier: Download tier (short, medium, long, longer)
        """
        if not self.config.enabled:
            return

        if duration_seconds <= 0 or bytes_downloaded <= 0:
            logger.debug(f"Skipping speed record: invalid values (bytes={bytes_downloaded}, duration={duration_seconds})")
            return

        record = DownloadRecord(
            video_id=video_id,
            bytes_downloaded=bytes_downloaded,
            duration_seconds=duration_seconds,
            timestamp=time.time(),
            tier=tier
        )

        self._records.append(record)

        speed_mbps = record.speed_mbps
        logger.debug(
            f"Speed tracker: {video_id} downloaded at {speed_mbps:.2f} MB/s "
            f"({bytes_downloaded / (1024*1024):.1f} MB in {duration_seconds:.1f}s)"
        )

    def get_average_speed_mbps(self) -> float:
        """Get average download speed from recent downloads.

        Returns:
            Average speed in MB/s, or 0.0 if no records.
        """
        if not self._records:
            return 0.0

        total_bytes = sum(r.bytes_downloaded for r in self._records)
        total_duration = sum(r.duration_seconds for r in self._records)

        if total_duration <= 0:
            return 0.0

        return (total_bytes / (1024 * 1024)) / total_duration

    def get_adjusted_timeout(self, base_timeout: int) -> int:
        """Get timeout adjusted based on current network speed.

        If the average download speed is below the minimum expected speed,
        the timeout is extended proportionally, up to max_timeout_multiplier.

        Args:
            base_timeout: Original timeout in seconds

        Returns:
            Adjusted timeout (may be same or up to 2x base_timeout)
        """
        if not self.config.enable_adaptive_timeout:
            return base_timeout

        if len(self._records) < 2:
            # Not enough data to adjust - need at least 2 samples
            return base_timeout

        avg_speed = self.get_average_speed_mbps()

        if avg_speed <= 0:
            return base_timeout

        # If speed is above minimum, no adjustment needed
        if avg_speed >= self.config.min_speed_mbps:
            return base_timeout

        # Calculate multiplier based on how slow the connection is
        # speed_ratio < 1 means connection is slower than expected
        speed_ratio = avg_speed / self.config.min_speed_mbps

        # Inverse ratio gives timeout multiplier (slower = longer timeout)
        # Cap at max_timeout_multiplier
        multiplier = min(1.0 / speed_ratio, self.config.max_timeout_multiplier)

        adjusted_timeout = int(base_timeout * multiplier)

        if adjusted_timeout > base_timeout:
            logger.info(
                f"Slow network detected ({avg_speed:.2f} MB/s < {self.config.min_speed_mbps} MB/s) - "
                f"extending timeout {base_timeout}s → {adjusted_timeout}s ({multiplier:.1f}x)"
            )

        return adjusted_timeout

    def get_speed_stats(self) -> Dict[str, Any]:
        """Get speed statistics for debugging.

        Returns:
            Dictionary with speed stats:
            - samples: Number of downloads tracked
            - avg_speed_mbps: Average speed in MB/s
            - min_speed_mbps: Minimum speed in MB/s
            - max_speed_mbps: Maximum speed in MB/s
            - total_bytes: Total bytes downloaded
            - total_duration: Total download duration
            - records: List of recent download records
        """
        if not self._records:
            return {
                'samples': 0,
                'avg_speed_mbps': 0.0,
                'min_speed_mbps': 0.0,
                'max_speed_mbps': 0.0,
                'total_bytes': 0,
                'total_duration': 0.0,
                'records': []
            }

        speeds = [r.speed_mbps for r in self._records]

        return {
            'samples': len(self._records),
            'avg_speed_mbps': round(self.get_average_speed_mbps(), 2),
            'min_speed_mbps': round(min(speeds), 2),
            'max_speed_mbps': round(max(speeds), 2),
            'total_bytes': sum(r.bytes_downloaded for r in self._records),
            'total_duration': round(sum(r.duration_seconds for r in self._records), 2),
            'records': [
                {
                    'video_id': r.video_id,
                    'bytes': r.bytes_downloaded,
                    'duration': round(r.duration_seconds, 2),
                    'speed_mbps': round(r.speed_mbps, 2),
                    'tier': r.tier
                }
                for r in self._records
            ]
        }

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize tracker state for checkpoint persistence.

        Returns:
            Dictionary that can be saved to checkpoint.json
        """
        return {
            'records': [
                {
                    'video_id': r.video_id,
                    'bytes_downloaded': r.bytes_downloaded,
                    'duration_seconds': r.duration_seconds,
                    'timestamp': r.timestamp,
                    'tier': r.tier
                }
                for r in self._records
            ]
        }

    def from_checkpoint_dict(self, data: Dict[str, Any]) -> None:
        """Restore tracker state from checkpoint.

        Args:
            data: Dictionary from checkpoint.json
        """
        self._records.clear()

        records = data.get('records', [])
        for r in records:
            try:
                record = DownloadRecord(
                    video_id=r['video_id'],
                    bytes_downloaded=r['bytes_downloaded'],
                    duration_seconds=r['duration_seconds'],
                    timestamp=r['timestamp'],
                    tier=r.get('tier', 'unknown')
                )
                self._records.append(record)
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping invalid speed record: {e}")

        if self._records:
            logger.debug(f"Restored {len(self._records)} speed records from checkpoint")

    def clear(self) -> None:
        """Clear all speed records."""
        self._records.clear()
