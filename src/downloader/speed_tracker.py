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
        window_size: Number of downloads to track in sliding window (default: 10)
        min_speed_mbps: Minimum expected speed in MB/s (default: 1.0)
        max_timeout_multiplier: Maximum timeout extension (default: 2.0)
        enable_adaptive_timeout: Use speed data to extend timeouts (default: True)
        safety_factor: Multiplier for adaptive timeout calculation (default: 1.5)
        rate_limit_signal_threshold: Speed in MB/s below which rate limiting is suspected
        consecutive_slow_samples: Number of slow samples before emitting rate limit signal
        variance_threshold: Coefficient of variation above which network is considered flaky (default: 0.5)
        slow_download_warning_threshold: Speed in MB/s below which to warn about slow download (default: 0.5)
        enable_variance_detection: Enable variance detection for network issue identification (default: True)
    """
    enabled: bool = True
    window_size: int = 10  # Default 10 for running average
    min_speed_mbps: float = 1.0  # MB/s below which timeout gets extended
    max_timeout_multiplier: float = 2.0  # Maximum timeout extension (2x)
    enable_adaptive_timeout: bool = True
    safety_factor: float = 1.5  # Multiplier for estimated_size / avg_speed timeout calc
    rate_limit_signal_threshold: float = 0.1  # MB/s (100 KB/s) - near-stalled threshold
    consecutive_slow_samples: int = 3  # Samples below threshold before signal
    # Variance detection (US-93-012)
    variance_threshold: float = 0.5  # CV above 0.5 (50%) = flaky network
    slow_download_warning_threshold: float = 0.5  # MB/s - warn below this
    enable_variance_detection: bool = True  # Enable variance/flakiness detection


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


@dataclass
class RateLimitSignal:
    """Signal indicating potential rate limiting detected from speed patterns.

    Attributes:
        detected: True if rate limit signal is active
        consecutive_slow_count: Number of consecutive slow samples
        recent_speeds: List of recent speeds in MB/s (for context)
        threshold: Speed threshold used for detection
        message: Human-readable description of the signal
    """
    detected: bool
    consecutive_slow_count: int
    recent_speeds: List[float]
    threshold: float
    message: str


@dataclass
class SpeedVarianceSignal:
    """Signal indicating network flakiness detected from speed variance.

    High variance (coefficient of variation) suggests:
    - Intermittent network issues
    - WiFi congestion or instability
    - Potential for connection drops

    Attributes:
        is_flaky: True if network is considered flaky (high variance)
        coefficient_of_variation: CV = std_dev / mean (0.5 = 50% variation)
        std_dev_mbps: Standard deviation of speeds in MB/s
        mean_speed_mbps: Mean speed in MB/s
        variance_category: Classification: 'stable', 'moderate', 'high', 'severe'
        message: Human-readable description
    """
    is_flaky: bool
    coefficient_of_variation: float
    std_dev_mbps: float
    mean_speed_mbps: float
    variance_category: str
    message: str


@dataclass
class SlowDownloadWarning:
    """Warning for individual slow downloads.

    Attributes:
        video_id: The video ID
        speed_mbps: Actual download speed
        threshold: Warning threshold used
        is_slow: True if speed is below threshold
        message: Human-readable description
    """
    video_id: str
    speed_mbps: float
    threshold: float
    is_slow: bool
    message: str


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

    def calculate_adaptive_timeout(
        self,
        estimated_size_bytes: int,
        base_timeout: int,
        safety_factor: Optional[float] = None
    ) -> int:
        """Calculate adaptive timeout based on estimated file size and observed speed.

        Uses the formula: estimated_size / avg_speed * safety_factor
        This provides a more accurate timeout based on actual network conditions.

        Args:
            estimated_size_bytes: Estimated file size in bytes
            base_timeout: Fallback timeout when no speed data available
            safety_factor: Multiplier for the calculated timeout (default from config: 1.5)
                          Higher values give more buffer for variable network conditions

        Returns:
            Calculated timeout in seconds, or base_timeout if insufficient data

        Example:
            # With avg speed of 2 MB/s, 100 MB file, safety_factor 1.5:
            # timeout = (100 MB / 2 MB/s) * 1.5 = 75 seconds
            timeout = tracker.calculate_adaptive_timeout(100 * 1024 * 1024, 300)
        """
        if not self.config.enable_adaptive_timeout:
            return base_timeout

        if len(self._records) < 2:
            # Not enough data to calculate - need at least 2 samples
            return base_timeout

        avg_speed = self.get_average_speed_mbps()

        if avg_speed <= 0:
            return base_timeout

        # Use config safety_factor if not provided
        if safety_factor is None:
            safety_factor = self.config.safety_factor

        # Convert estimated size to MB
        estimated_size_mb = estimated_size_bytes / (1024 * 1024)

        # Calculate timeout: size_MB / speed_MB_per_s * safety_factor
        calculated_timeout = int((estimated_size_mb / avg_speed) * safety_factor)

        # Apply minimum of base_timeout for small files
        calculated_timeout = max(calculated_timeout, 30)  # At least 30 seconds

        # Apply maximum of max_timeout_multiplier * base_timeout
        max_timeout = int(base_timeout * self.config.max_timeout_multiplier)
        calculated_timeout = min(calculated_timeout, max_timeout)

        if calculated_timeout != base_timeout:
            logger.debug(
                f"Adaptive timeout: {estimated_size_mb:.1f} MB at {avg_speed:.2f} MB/s "
                f"→ {calculated_timeout}s (safety_factor={safety_factor})"
            )

        return calculated_timeout

    def detect_rate_limit_signals(self) -> RateLimitSignal:
        """Detect potential rate limiting based on speed anomalies.

        Analyzes recent download speeds to detect patterns that often precede
        hard rate limit failures:
        - Near-zero speeds (< rate_limit_signal_threshold) for consecutive samples
        - Stalled progress indicating throttling

        This allows preemptive backoff before receiving actual rate limit errors,
        reducing the chance of getting blocked.

        Returns:
            RateLimitSignal with detection status and context.
            - detected: True if rate limit signal is active
            - consecutive_slow_count: Number of slow samples
            - recent_speeds: Last N speeds for debugging
            - message: Human-readable description

        Example:
            signal = tracker.detect_rate_limit_signals()
            if signal.detected:
                circuit_breaker.record_failure()  # Trigger preemptive backoff
        """
        threshold = self.config.rate_limit_signal_threshold
        required_samples = self.config.consecutive_slow_samples

        # Need enough samples to detect pattern
        if len(self._records) < required_samples:
            return RateLimitSignal(
                detected=False,
                consecutive_slow_count=0,
                recent_speeds=[],
                threshold=threshold,
                message=f"Insufficient samples ({len(self._records)}/{required_samples})"
            )

        # Get speeds from most recent records (newest last in deque)
        recent_speeds = [r.speed_mbps for r in self._records]

        # Count consecutive slow samples from the end (most recent)
        consecutive_slow = 0
        for speed in reversed(recent_speeds):
            if speed < threshold:
                consecutive_slow += 1
            else:
                break  # Stop at first non-slow sample

        detected = consecutive_slow >= required_samples

        if detected:
            # Build descriptive message with speed history
            speed_history = ", ".join(f"{s:.2f}" for s in recent_speeds[-required_samples:])
            message = (
                f"Rate limit signal: {consecutive_slow} consecutive downloads below "
                f"{threshold} MB/s threshold. Recent speeds (MB/s): [{speed_history}]"
            )
            # Log at WARNING level for visibility
            logger.warning(message)
        else:
            message = f"No rate limit signal ({consecutive_slow}/{required_samples} slow samples)"

        return RateLimitSignal(
            detected=detected,
            consecutive_slow_count=consecutive_slow,
            recent_speeds=recent_speeds[-self.config.window_size:],
            threshold=threshold,
            message=message
        )

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
            ],
            # Variance metrics (US-93-012)
            'std_dev_mbps': round(self._calculate_std_dev(), 2),
            'coefficient_of_variation': round(self._calculate_cv(), 2),
            'variance_category': self._get_variance_category()
        }

    def _calculate_std_dev(self) -> float:
        """Calculate standard deviation of download speeds.

        Returns:
            Standard deviation in MB/s, or 0.0 if insufficient data.
        """
        if len(self._records) < 2:
            return 0.0

        speeds = [r.speed_mbps for r in self._records]
        mean = sum(speeds) / len(speeds)

        # Population standard deviation
        variance = sum((s - mean) ** 2 for s in speeds) / len(speeds)
        return variance ** 0.5

    def _calculate_cv(self) -> float:
        """Calculate coefficient of variation (CV = std_dev / mean).

        CV is a normalized measure of dispersion. Higher values indicate
        more variability in download speeds:
        - CV < 0.25: Stable network
        - 0.25 <= CV < 0.5: Moderate variation
        - 0.5 <= CV < 1.0: High variation (flaky)
        - CV >= 1.0: Severe variation (unstable)

        Returns:
            Coefficient of variation, or 0.0 if insufficient data.
        """
        if len(self._records) < 2:
            return 0.0

        mean = self.get_average_speed_mbps()
        if mean <= 0:
            return 0.0

        std_dev = self._calculate_std_dev()
        return std_dev / mean

    def _get_variance_category(self) -> str:
        """Get category classification for current variance level.

        Returns:
            Category: 'stable', 'moderate', 'high', or 'severe'
        """
        cv = self._calculate_cv()
        if cv < 0.25:
            return 'stable'
        elif cv < 0.5:
            return 'moderate'
        elif cv < 1.0:
            return 'high'
        else:
            return 'severe'

    def detect_variance_signals(self) -> SpeedVarianceSignal:
        """Detect network flakiness based on speed variance.

        High coefficient of variation (CV) indicates unstable network:
        - Consistent slow downloads (low CV): Network is slow but stable
        - Variable downloads (high CV): Network has intermittent issues

        This helps distinguish between:
        - "Just slow" -> Extend timeout
        - "Flaky" -> May need to abort or implement backoff

        Returns:
            SpeedVarianceSignal with flakiness detection results.

        Example:
            signal = tracker.detect_variance_signals()
            if signal.is_flaky:
                logger.warning(f"Flaky network detected: {signal.message}")
                # Consider aborting or implementing aggressive backoff
        """
        if not self.config.enable_variance_detection:
            return SpeedVarianceSignal(
                is_flaky=False,
                coefficient_of_variation=0.0,
                std_dev_mbps=0.0,
                mean_speed_mbps=0.0,
                variance_category='stable',
                message="Variance detection disabled"
            )

        if len(self._records) < 2:
            return SpeedVarianceSignal(
                is_flaky=False,
                coefficient_of_variation=0.0,
                std_dev_mbps=0.0,
                mean_speed_mbps=0.0,
                variance_category='stable',
                message="Insufficient samples for variance analysis"
            )

        std_dev = self._calculate_std_dev()
        cv = self._calculate_cv()
        mean = self.get_average_speed_mbps()
        category = self._get_variance_category()

        is_flaky = cv >= self.config.variance_threshold

        if is_flaky:
            message = (
                f"Network flakiness detected: CV={cv:.2f} (threshold={self.config.variance_threshold}) - "
                f"std_dev={std_dev:.2f} MB/s, mean={mean:.2f} MB/s, category={category}"
            )
            logger.warning(message)
        else:
            message = f"Network stable: CV={cv:.2f}, category={category}"

        return SpeedVarianceSignal(
            is_flaky=is_flaky,
            coefficient_of_variation=round(cv, 3),
            std_dev_mbps=round(std_dev, 2),
            mean_speed_mbps=round(mean, 2),
            variance_category=category,
            message=message
        )

    def check_slow_download_warning(self, video_id: str) -> SlowDownloadWarning:
        """Check if the most recent download was slow and should warn.

        This checks the LAST download (most recent) against the slow threshold
        to provide per-download warnings for analysis.

        Args:
            video_id: The video ID to check

        Returns:
            SlowDownloadWarning with warning details.
        """
        threshold = self.config.slow_download_warning_threshold

        if not self._records:
            return SlowDownloadWarning(
                video_id=video_id,
                speed_mbps=0.0,
                threshold=threshold,
                is_slow=False,
                message="No download records available"
            )

        # Get the most recent record
        recent_record = self._records[-1]
        speed = recent_record.speed_mbps
        is_slow = speed < threshold

        if is_slow:
            message = (
                f"Slow download warning: {video_id} at {speed:.2f} MB/s "
                f"(below {threshold} MB/s threshold)"
            )
            logger.warning(message)
        else:
            message = f"Download speed OK: {video_id} at {speed:.2f} MB/s"

        return SlowDownloadWarning(
            video_id=video_id,
            speed_mbps=round(speed, 2),
            threshold=threshold,
            is_slow=is_slow,
            message=message
        )

    def should_abort_due_to_speed(self) -> bool:
        """Determine if downloads should be aborted due to severe network issues.

        Abort decision considers:
        1. Very high variance (severe network instability)
        2. Combined with low average speed (not just slow, but unreliable)

        Returns:
            True if abort is recommended due to network conditions.
        """
        if not self.config.enable_variance_detection:
            return False

        if len(self._records) < 3:
            return False  # Need more data to decide

        cv = self._calculate_cv()
        mean = self.get_average_speed_mbps()

        # Abort if: severe variance AND low average speed
        # This means network is not just slow, it's unreliable
        severe_variance = cv >= 1.0  # 100%+ variation
        low_speed = mean < self.config.min_speed_mbps * 0.5  # Below half of min expected

        if severe_variance and low_speed:
            logger.error(
                f"Abort recommended: severe variance (CV={cv:.2f}) with low speed "
                f"({mean:.2f} MB/s) - network is unstable"
            )
            return True

        return False

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


class PerKeywordSpeedTracker:
    """Manages per-keyword DownloadSpeedTracker instances for isolation.

    Each keyword gets its own independent speed tracker so that slow downloads
    for one keyword don't affect rate limit signal detection for another.

    Example:
        tracker = PerKeywordSpeedTracker(config)
        tracker.record_download("cats", "video1", 10*1024*1024, 5.0, "short")
        tracker.record_download("dogs", "video2", 500*1024, 10.0, "short")

        # "dogs" being slow doesn't affect "cats"
        signal_cats = tracker.detect_rate_limit_signals("cats")
        signal_dogs = tracker.detect_rate_limit_signals("dogs")
    """

    def __init__(self, config: DownloadSpeedConfig = None):
        self.config = config or DownloadSpeedConfig()
        self._trackers: Dict[str, DownloadSpeedTracker] = {}

    def _get_tracker(self, keyword: str) -> DownloadSpeedTracker:
        """Get or create a tracker for the given keyword."""
        if keyword not in self._trackers:
            self._trackers[keyword] = DownloadSpeedTracker(self.config)
        return self._trackers[keyword]

    def record_download(
        self,
        keyword: str,
        video_id: str,
        bytes_downloaded: int,
        duration_seconds: float,
        tier: str = "unknown"
    ) -> None:
        """Record a download for a specific keyword."""
        self._get_tracker(keyword).record_download(
            video_id, bytes_downloaded, duration_seconds, tier
        )

    def detect_rate_limit_signals(self, keyword: str) -> RateLimitSignal:
        """Detect rate limit signals for a specific keyword."""
        if keyword not in self._trackers:
            return RateLimitSignal(
                detected=False,
                consecutive_slow_count=0,
                recent_speeds=[],
                threshold=self.config.rate_limit_signal_threshold,
                message=f"No data for keyword '{keyword}'"
            )
        return self._get_tracker(keyword).detect_rate_limit_signals()

    def get_average_speed_mbps(self, keyword: str) -> float:
        """Get average speed for a specific keyword."""
        if keyword not in self._trackers:
            return 0.0
        return self._get_tracker(keyword).get_average_speed_mbps()

    def get_keywords(self) -> List[str]:
        """Get list of tracked keywords."""
        return list(self._trackers.keys())

    def clear(self, keyword: str = None) -> None:
        """Clear records for a keyword, or all keywords if None."""
        if keyword is not None:
            if keyword in self._trackers:
                self._trackers[keyword].clear()
        else:
            self._trackers.clear()
