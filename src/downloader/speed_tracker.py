"""Download speed monitoring for adaptive timeouts.

Tracks download speeds and adjusts timeouts dynamically based on network conditions.
Part of US-005: Implement download speed monitoring for adaptive timeouts.
US-123-010: Add download throttling self-regulation to prevent rate limits.
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
        # US-109-006: Multi-signal correlation configuration
        enable_signal_correlation: Enable correlation of speed signals with error patterns (default: True)
        enable_sustained_degradation: Enable detection of gradual speed degradation (default: True)
        degradation_samples_required: Number of samples to detect sustained degradation (default: 3)
        degradation_threshold: Speed drop percentage to trigger degradation warning (default: 0.5 = 50% drop)
        speed_signal_weight: Weight for speed signals in correlation (default: 0.6)
        error_signal_weight: Weight for error patterns in correlation (default: 0.4)
        correlation_threshold: Combined score threshold to trigger escalation (default: 0.7)
        # US-136-006: Early warning and adaptive thresholds
        enable_early_warning: Enable early warning before circuit breaker trips (default: True)
        early_warning_threshold: Speed drop percentage to trigger early warning (default: 0.3 = 30% drop)
        early_warning_samples: Minimum samples before early warning can trigger (default: 2)
        enable_adaptive_thresholds: Enable time-of-day adaptive speed thresholds (default: True)
        enable_anomaly_detection: Enable sudden drop anomaly detection (default: True)
        anomaly_threshold: Z-score above which speed is considered anomalous (default: 2.0)
        anomaly_window: Number of samples for anomaly baseline (default: 5)
        # US-144-008: Adaptive speed threshold based on historical average
        min_samples_for_adaptive_threshold: Minimum samples before adaptive threshold activates (default: 5)
        adaptive_threshold_multiplier: Multiplier for historical avg to calculate threshold (default: 0.5)
        enable_keyword_adaptive_threshold: Enable per-keyword adaptive threshold based on historical average (default: True)
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
    # US-109-006: Signal correlation configuration
    enable_signal_correlation: bool = True  # Enable speed + error correlation
    enable_sustained_degradation: bool = True  # Enable gradual degradation detection
    degradation_samples_required: int = 3  # Minimum samples for degradation detection
    degradation_threshold: float = 0.5  # 50% speed drop = degradation
    speed_signal_weight: float = 0.6  # Weight for speed signal in correlation
    error_signal_weight: float = 0.4  # Weight for error pattern signal in correlation
    correlation_threshold: float = 0.7  # Combined score threshold (0.0-1.0)
    # US-136-006: Early warning and adaptive thresholds
    enable_early_warning: bool = True  # Enable early warning system
    early_warning_threshold: float = 0.3  # 30% drop triggers warning
    early_warning_samples: int = 2  # Minimum samples before early warning
    enable_adaptive_thresholds: bool = True  # Enable time-of-day adaptive thresholds
    enable_anomaly_detection: bool = True  # Enable sudden drop detection
    anomaly_threshold: float = 2.0  # Z-score threshold for anomaly
    anomaly_window: int = 5  # Samples for anomaly baseline
    # US-144-008: Adaptive speed threshold based on historical average
    min_samples_for_adaptive_threshold: int = 5  # Minimum samples before adaptive threshold activates
    adaptive_threshold_multiplier: float = 0.5  # Threshold = historical_avg * this_multiplier
    enable_keyword_adaptive_threshold: bool = True  # Enable per-keyword adaptive threshold


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


@dataclass
class SustainedDegradationSignal:
    """Signal indicating sustained degradation (gradual speed drop) over multiple samples.

    This detects gradual network degradation rather than sudden stalls. A pattern of
    decreasing speeds over 3+ samples indicates a developing rate limit before it
    becomes severe.

    Attributes:
        detected: True if sustained degradation is detected
        degradation_percentage: Percentage of speed drop from peak (0.0 = no drop, 1.0 = complete drop)
        samples_analyzed: Number of samples used in analysis
        peak_speed_mbps: Highest speed in the analysis window
        current_speed_mbps: Current (most recent) speed in the analysis window
        trend: Direction of speed trend: 'improving', 'stable', 'degrading', 'severe'
        message: Human-readable description
    """
    detected: bool
    degradation_percentage: float
    samples_analyzed: int
    peak_speed_mbps: float
    current_speed_mbps: float
    trend: str
    message: str


@dataclass
class CorrelatedSignal:
    """Combined signal from speed and error pattern correlation.

    When both speed signals and error patterns occur together, the combined signal
    is stronger evidence of rate limiting than either signal alone.

    Attributes:
        detected: True if correlated signal indicates rate limiting
        correlation_score: Combined score (0.0-1.0) from weighted speed + error signals
        speed_signal_score: Normalized speed signal strength (0.0-1.0)
        error_signal_score: Normalized error pattern signal strength (0.0-1.0)
        speed_signal_detected: Whether speed signal alone was detected
        error_signal_detected: Whether error pattern signal was detected
        recommended_action: 'escalate', 'watch', or 'none'
        message: Human-readable description
    """
    detected: bool
    correlation_score: float
    speed_signal_score: float
    error_signal_score: float
    speed_signal_detected: bool
    error_signal_detected: bool
    recommended_action: str
    message: str


@dataclass
class EarlyWarningSignal:
    """Signal indicating early warning of speed degradation before circuit breaker trips.

    This provides a heads-up warning before the circuit breaker needs to trip,
    allowing for preemptive action like reducing concurrency or backing off.

    Attributes:
        detected: True if early warning is active
        warning_level: 'none', 'minor', 'moderate', 'severe'
        degradation_percentage: Percentage of speed drop from baseline
        baseline_speed_mbps: Speed baseline for comparison
        current_speed_mbps: Current (most recent) speed
        recommended_action: 'none', 'reduce_concurrency', 'backoff', 'escalate'
        message: Human-readable description
    """
    detected: bool
    warning_level: str
    degradation_percentage: float
    baseline_speed_mbps: float
    current_speed_mbps: float
    recommended_action: str
    message: str


@dataclass
class AnomalyRecord:
    """Record of a detected anomaly for history tracking.

    Attributes:
        timestamp: Unix timestamp when anomaly was detected
        z_score: Z-score of the anomaly
        current_speed_mbps: Speed that triggered the anomaly
        baseline_mean_mbps: Mean of baseline window
        baseline_std_mbps: Standard deviation of baseline
        anomaly_type: Type of anomaly: 'sudden_drop', 'spike'
    """
    timestamp: float
    z_score: float
    current_speed_mbps: float
    baseline_mean_mbps: float
    baseline_std_mbps: float
    anomaly_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'z_score': self.z_score,
            'current_speed_mbps': self.current_speed_mbps,
            'baseline_mean_mbps': self.baseline_mean_mbps,
            'baseline_std_mbps': self.baseline_std_mbps,
            'anomaly_type': self.anomaly_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnomalyRecord":
        """Create from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            z_score=data['z_score'],
            current_speed_mbps=data['current_speed_mbps'],
            baseline_mean_mbps=data['baseline_mean_mbps'],
            baseline_std_mbps=data['baseline_std_mbps'],
            anomaly_type=data['anomaly_type'],
        )


@dataclass
class AnomalyMetrics:
    """Metrics for anomaly detection.

    Attributes:
        total_anomalies_detected: Total number of anomalies detected
        sudden_drops: Number of sudden drop anomalies
        spikes: Number of speed spike anomalies
        anomaly_history: List of recent anomaly records
    """
    total_anomalies_detected: int
    sudden_drops: int
    spikes: int
    anomaly_history: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_anomalies_detected': self.total_anomalies_detected,
            'sudden_drops': self.sudden_drops,
            'spikes': self.spikes,
            'anomaly_history': self.anomaly_history,
        }


@dataclass
class SpeedAnomalySignal:
    """Signal indicating sudden speed anomaly (drop) detected.

    Detects sudden, sharp drops in download speed that may indicate transient
    network issues or early-stage rate limiting.

    Attributes:
        is_anomalous: True if current speed is anomalous (sudden drop)
        z_score: Z-score of current speed relative to baseline
        current_speed_mbps: Current speed
        baseline_mean_mbps: Mean speed in baseline window
        baseline_std_mbps: Standard deviation in baseline window
        anomaly_type: 'sudden_drop', 'spike', 'none'
        message: Human-readable description
    """
    is_anomalous: bool
    z_score: float
    current_speed_mbps: float
    baseline_mean_mbps: float
    baseline_std_mbps: float
    anomaly_type: str
    message: str


@dataclass
class AdaptiveThresholdResult:
    """Result of adaptive threshold calculation based on time-of-day.

    Different times of day have different typical network conditions,
    so thresholds should adapt accordingly.

    Attributes:
        threshold_mbps: Adapted speed threshold in MB/s
        time_period: Time period identifier: 'morning', 'afternoon', 'evening', 'overnight'
        multiplier: Multiplier applied to base threshold
        reason: Human-readable reason for the adaptation
    """
    threshold_mbps: float
    time_period: str
    multiplier: float
    reason: str


@dataclass
class KeywordAdaptiveThresholdResult:
    """Result of adaptive threshold calculation based on historical keyword average.

    US-144-008: Dynamic speed threshold based on historical average per keyword.
    The threshold is calculated as: historical_avg * multiplier

    Attributes:
        threshold_mbps: Adapted speed threshold in MB/s based on historical average
        historical_avg_mbps: Historical average speed for this keyword
        samples_used: Number of samples used for historical average
        multiplier: Multiplier applied to historical average
        is_adaptive: True if adaptive threshold is being used (enough samples)
        fallback_threshold_mbps: The fallback static threshold when not enough samples
        reason: Human-readable reason for the adaptation
    """
    threshold_mbps: float
    historical_avg_mbps: float
    samples_used: int
    multiplier: float
    is_adaptive: bool
    fallback_threshold_mbps: float
    reason: str


@dataclass
class SpeedWarningMetrics:
    """Metrics for speed-based early warnings.

    Tracks warning statistics for observability and analysis.

    Attributes:
        early_warnings_issued: Number of early warnings issued
        warnings_by_level: Dict mapping warning level to count
        anomaly_count: Number of anomalies detected
        adaptive_threshold_adjustments: Number of times threshold was adapted
        total_speed_checks: Number of speed checks performed
    """
    early_warnings_issued: int
    warnings_by_level: Dict[str, int]
    anomaly_count: int
    adaptive_threshold_adjustments: int
    total_speed_checks: int


@dataclass
class ThrottleState:
    """State of the self-regulation throttling mechanism.

    Attributes:
        current_concurrency: Current concurrency level after throttling
        original_concurrency: Original max_concurrency before any throttling
        throttle_level: Number of times throttled (0 = not throttled)
        is_throttled: Whether currently in throttled state
        last_throttle_time: Timestamp of last throttle action
        last_error_time: Timestamp of most recent error
        error_count_in_window: Number of errors in the current window
        is_recovering: Whether currently in recovery mode
        recovery_step: Number of recovery steps taken
    """
    current_concurrency: int
    original_concurrency: int
    throttle_level: int = 0
    is_throttled: bool = False
    last_throttle_time: float = 0.0
    last_error_time: float = 0.0
    error_count_in_window: int = 0
    is_recovering: bool = False
    recovery_step: int = 0


@dataclass
class ThrottleSignal:
    """Signal indicating throttling state and recommended action.

    Attributes:
        should_throttle: True if throttling should be applied
        should_recover: True if recovery should be attempted
        new_concurrency: Recommended new concurrency level
        reason: Human-readable reason for the signal
        throttle_level: Current throttle level (times throttled)
    """
    should_throttle: bool
    should_recover: bool
    new_concurrency: int
    reason: str
    throttle_level: int


@dataclass
class ErrorPatternSignal:
    """Signal from error pattern analysis.

    Tracks recent error patterns that may indicate rate limiting.

    Attributes:
        error_count: Number of errors in the recent window
        error_types: List of error type categories detected
        rate_limit_error_count: Number of rate-limit specific errors (429, quota, etc.)
        severity: Overall severity: 'none', 'low', 'medium', 'high'
        message: Human-readable description
    """
    error_count: int
    error_types: List[str]
    rate_limit_error_count: int
    severity: str
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
        # US-123-010: Throttling state
        self._throttle_state: Optional[ThrottleState] = None
        self._error_timestamps: deque = deque(maxlen=100)  # Track error timestamps
        # US-143-003: Anomaly detection history
        self._anomaly_history: deque = deque(maxlen=50)  # Track recent anomalies
        self._anomaly_count: int = 0
        self._sudden_drop_count: int = 0
        self._spike_count: int = 0

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

    def detect_sustained_degradation(self) -> SustainedDegradationSignal:
        """Detect sustained degradation (gradual speed drop) over multiple samples.

        This method analyzes the speed trend to detect gradual degradation rather than
        sudden stalls. A pattern of decreasing speeds over 3+ samples indicates a
        developing rate limit before it becomes severe.

        Detection criteria:
        - Need at least degradation_samples_required (default 3) samples
        - Calculate speed drop from peak to current
        - If drop exceeds degradation_threshold (default 50%), signal detected

        Returns:
            SustainedDegradationSignal with detection status and trend analysis.

        Example:
            signal = tracker.detect_sustained_degradation()
            if signal.detected:
                logger.warning(f"Degradation detected: {signal.message}")
                # Could trigger preemptive backoff
        """
        if not self.config.enable_sustained_degradation:
            return SustainedDegradationSignal(
                detected=False,
                degradation_percentage=0.0,
                samples_analyzed=0,
                peak_speed_mbps=0.0,
                current_speed_mbps=0.0,
                trend='stable',
                message="Sustained degradation detection disabled"
            )

        required_samples = self.config.degradation_samples_required

        if len(self._records) < required_samples:
            return SustainedDegradationSignal(
                detected=False,
                degradation_percentage=0.0,
                samples_analyzed=len(self._records),
                peak_speed_mbps=0.0,
                current_speed_mbps=0.0,
                trend='insufficient_data',
                message=f"Insufficient samples ({len(self._records)}/{required_samples})"
            )

        # Get speeds in chronological order (oldest to newest)
        speeds = [r.speed_mbps for r in self._records]
        samples_analyzed = len(speeds)

        peak_speed = max(speeds)
        current_speed = speeds[-1]

        # Calculate degradation as percentage drop from peak
        if peak_speed > 0:
            degradation_percentage = (peak_speed - current_speed) / peak_speed
        else:
            degradation_percentage = 0.0

        # Determine trend based on speed changes
        if len(speeds) >= 3:
            # Calculate simple trend using first half vs second half average
            mid = len(speeds) // 2
            first_half_avg = sum(speeds[:mid]) / mid
            second_half_avg = sum(speeds[mid:]) / (len(speeds) - mid)

            if second_half_avg > first_half_avg * 1.1:
                trend = 'improving'
            elif second_half_avg < first_half_avg * 0.9:
                trend = 'degrading'
            else:
                trend = 'stable'
        else:
            # For small sample sizes, use direct comparison
            if current_speed > peak_speed * 0.9:
                trend = 'stable'
            elif current_speed > peak_speed * 0.5:
                trend = 'degrading'
            else:
                trend = 'severe'

        # Check if degradation threshold is exceeded
        detected = degradation_percentage >= self.config.degradation_threshold

        if detected:
            message = (
                f"Sustained degradation detected: {degradation_percentage*100:.1f}% speed drop "
                f"from {peak_speed:.2f} MB/s to {current_speed:.2f} MB/s "
                f"over {samples_analyzed} samples (threshold: {self.config.degradation_threshold*100:.0f}%)"
            )
            logger.warning(message)
        else:
            message = (
                f"No sustained degradation: {degradation_percentage*100:.1f}% drop "
                f"(peak: {peak_speed:.2f}, current: {current_speed:.2f})"
            )

        return SustainedDegradationSignal(
            detected=detected,
            degradation_percentage=round(degradation_percentage, 3),
            samples_analyzed=samples_analyzed,
            peak_speed_mbps=round(peak_speed, 2),
            current_speed_mbps=round(current_speed, 2),
            trend=trend,
            message=message
        )

    def get_speed_signal_score(self) -> float:
        """Calculate normalized speed signal score (0.0-1.0).

        This converts speed signal strength to a normalized score for correlation:
        - 0.0: No speed issues (all fast)
        - 0.5: Moderate degradation
        - 1.0: Severe rate limiting (all slow)

        Returns:
            Normalized score from 0.0 to 1.0.
        """
        if len(self._records) < 2:
            return 0.0  # No data = no signal

        threshold = self.config.rate_limit_signal_threshold

        # Count slow samples
        slow_count = sum(1 for r in self._records if r.speed_mbps < threshold)
        slow_ratio = slow_count / len(self._records)

        # Also factor in average speed degradation from expected
        avg_speed = self.get_average_speed_mbps()
        if avg_speed > 0:
            # How far below expected speed are we?
            expected = self.config.min_speed_mbps
            speed_ratio = min(avg_speed / expected, 1.0)  # Cap at 1.0

            # Combine: slow ratio has more weight (70%), speed ratio (30%)
            signal_score = (slow_ratio * 0.7) + ((1.0 - speed_ratio) * 0.3)
            return min(signal_score, 1.0)
        else:
            return slow_ratio

    def detect_correlated_signals(
        self,
        error_signal: Optional[ErrorPatternSignal] = None
    ) -> CorrelatedSignal:
        """Detect correlated signals: combine speed and error pattern signals.

        When both speed signals and error patterns occur together, the combined
        signal is stronger evidence of rate limiting than either signal alone.
        This provides more accurate preemptive detection.

        Args:
            error_signal: Optional error pattern signal from error_aggregator.
                         If None, uses only speed signal.

        Returns:
            CorrelatedSignal with combined analysis and recommended action.

        Example:
            # With error signal from error aggregator
            error_signal = ErrorPatternSignal(
                error_count=3,
                error_types=['rate_limit', '429'],
                rate_limit_error_count=2,
                severity='medium',
                message="Rate limit errors detected"
            )
            correlated = tracker.detect_correlated_signals(error_signal)

            # Or without error signal (speed only)
            correlated = tracker.detect_correlated_signals()

            if correlated.detected and correlated.recommended_action == 'escalate':
                escalation_manager.escalate()
        """
        if not self.config.enable_signal_correlation:
            return CorrelatedSignal(
                detected=False,
                correlation_score=0.0,
                speed_signal_score=0.0,
                error_signal_score=0.0,
                speed_signal_detected=False,
                error_signal_detected=False,
                recommended_action='none',
                message="Signal correlation disabled"
            )

        # Calculate speed signal score
        speed_signal_score = self.get_speed_signal_score()
        speed_signal_detected = speed_signal_score >= 0.5  # 50%+ = positive signal

        # Calculate error signal score
        if error_signal is not None:
            error_signal_detected = error_signal.severity in ('medium', 'high')
            # Normalize error score: 0=none, 0.25=low, 0.5=medium, 1.0=high
            error_score_map = {'none': 0.0, 'low': 0.25, 'medium': 0.5, 'high': 1.0}
            error_signal_score = error_score_map.get(error_signal.severity, 0.0)
        else:
            error_signal_detected = False
            error_signal_score = 0.0

        # Calculate weighted correlation score
        speed_weight = self.config.speed_signal_weight
        error_weight = self.config.error_signal_weight

        # Normalize weights to ensure they sum to 1.0
        total_weight = speed_weight + error_weight
        if total_weight > 0:
            speed_weight = speed_weight / total_weight
            error_weight = error_weight / total_weight

        correlation_score = (
            (speed_signal_score * speed_weight) +
            (error_signal_score * error_weight)
        )

        # Determine if correlated signal is detected
        detected = correlation_score >= self.config.correlation_threshold

        # Determine recommended action
        if detected:
            if correlation_score >= 0.85:
                recommended_action = 'escalate'
            else:
                recommended_action = 'watch'
        else:
            recommended_action = 'none'

        # Build message
        if detected:
            message = (
                f"Correlated signal detected: score={correlation_score:.2f} "
                f"(speed={speed_signal_score:.2f}, error={error_signal_score:.2f})"
            )
            if recommended_action == 'escalate':
                logger.warning(message)
            else:
                logger.info(message)
        else:
            message = (
                f"No correlated signal: score={correlation_score:.2f} "
                f"(threshold={self.config.correlation_threshold:.2f})"
            )

        return CorrelatedSignal(
            detected=detected,
            correlation_score=round(correlation_score, 3),
            speed_signal_score=round(speed_signal_score, 3),
            error_signal_score=round(error_signal_score, 3),
            speed_signal_detected=speed_signal_detected,
            error_signal_detected=error_signal_detected,
            recommended_action=recommended_action,
            message=message
        )

    # =========================================================================
    # US-136-006: EARLY WARNING SYSTEM
    # =========================================================================

    def detect_early_warning(self) -> EarlyWarningSignal:
        """Detect early warning of speed degradation before circuit breaker trips.

        This provides a heads-up warning before the circuit breaker needs to trip,
        allowing for preemptive action. Uses a less sensitive threshold than the
        circuit breaker to give early notice.

        Detection criteria:
        - Need at least early_warning_samples (default 2) samples
        - Calculate speed drop from baseline (first half average) to current
        - If drop exceeds early_warning_threshold (default 30%), signal warning

        Returns:
            EarlyWarningSignal with warning level and recommended action.

        Example:
            signal = tracker.detect_early_warning()
            if signal.detected:
                logger.warning(f"Early warning: {signal.message}")
                # Could reduce concurrency or take preemptive action
        """
        if not self.config.enable_early_warning:
            return EarlyWarningSignal(
                detected=False,
                warning_level='none',
                degradation_percentage=0.0,
                baseline_speed_mbps=0.0,
                current_speed_mbps=0.0,
                recommended_action='none',
                message="Early warning disabled"
            )

        required_samples = self.config.early_warning_samples + 1

        if len(self._records) < required_samples:
            return EarlyWarningSignal(
                detected=False,
                warning_level='none',
                degradation_percentage=0.0,
                baseline_speed_mbps=0.0,
                current_speed_mbps=0.0,
                recommended_action='none',
                message=f"Insufficient samples ({len(self._records)}/{required_samples})"
            )

        # Calculate baseline from first half of samples
        speeds = [r.speed_mbps for r in self._records]
        mid = len(speeds) // 2
        baseline = sum(speeds[:mid]) / mid
        current = speeds[-1]

        # Calculate degradation percentage
        if baseline > 0:
            degradation = (baseline - current) / baseline
        else:
            degradation = 0.0

        # Determine warning level based on degradation
        warning_level = 'none'
        recommended_action = 'none'

        if degradation >= self.config.early_warning_threshold:
            # Classify severity
            if degradation >= 0.6:
                warning_level = 'severe'
                recommended_action = 'escalate'
            elif degradation >= 0.45:
                warning_level = 'moderate'
                recommended_action = 'backoff'
            else:
                warning_level = 'minor'
                recommended_action = 'reduce_concurrency'

            message = (
                f"EARLY WARNING: {degradation*100:.1f}% speed degradation detected. "
                f"Baseline: {baseline:.2f} MB/s, Current: {current:.2f} MB/s. "
                f"Level: {warning_level}. Action: {recommended_action}"
            )
            logger.warning(message)
        else:
            message = f"No early warning: {degradation*100:.1f}% degradation (threshold: {self.config.early_warning_threshold*100:.0f}%)"

        return EarlyWarningSignal(
            detected=degradation >= self.config.early_warning_threshold,
            warning_level=warning_level,
            degradation_percentage=round(degradation, 3),
            baseline_speed_mbps=round(baseline, 2),
            current_speed_mbps=round(current, 2),
            recommended_action=recommended_action,
            message=message
        )

    def detect_anomaly(self) -> SpeedAnomalySignal:
        """Detect sudden speed anomalies using statistical analysis.

        Uses z-score to detect sudden drops or spikes in download speed
        relative to a baseline window.

        Returns:
            SpeedAnomalySignal with anomaly detection results.

        Example:
            signal = tracker.detect_anomaly()
            if signal.is_anomalous:
                logger.warning(f"Anomaly detected: {signal.message}")
        """
        if not self.config.enable_anomaly_detection:
            return SpeedAnomalySignal(
                is_anomalous=False,
                z_score=0.0,
                current_speed_mbps=0.0,
                baseline_mean_mbps=0.0,
                baseline_std_mbps=0.0,
                anomaly_type='none',
                message="Anomaly detection disabled"
            )

        window_size = min(self.config.anomaly_window, len(self._records))

        if window_size < 3:
            return SpeedAnomalySignal(
                is_anomalous=False,
                z_score=0.0,
                current_speed_mbps=0.0,
                baseline_mean_mbps=0.0,
                baseline_std_mbps=0.0,
                anomaly_type='none',
                message=f"Insufficient samples for anomaly detection ({len(self._records)})"
            )

        # Use all but last record as baseline, last record as test
        speeds = [r.speed_mbps for r in self._records]
        baseline_speeds = speeds[:-1][-window_size:]
        current_speed = speeds[-1]

        # Calculate baseline statistics
        mean = sum(baseline_speeds) / len(baseline_speeds)

        # Calculate standard deviation
        variance = sum((s - mean) ** 2 for s in baseline_speeds) / len(baseline_speeds)
        std = variance ** 0.5

        if std <= 0:
            return SpeedAnomalySignal(
                is_anomalous=False,
                z_score=0.0,
                current_speed_mbps=round(current_speed, 2),
                baseline_mean_mbps=round(mean, 2),
                baseline_std_mbps=0.0,
                anomaly_type='none',
                message="Baseline has no variance - cannot detect anomaly"
            )

        # Calculate z-score
        z_score = (current_speed - mean) / std
        abs_z = abs(z_score)

        # Determine if anomalous
        is_anomalous = abs_z >= self.config.anomaly_threshold

        # Classify anomaly type
        if is_anomalous:
            if z_score < 0:
                anomaly_type = 'sudden_drop'
                message = (
                    f"SUDDEN DROP detected: z-score={z_score:.2f}, "
                    f"current={current_speed:.2f} MB/s vs baseline={mean:.2f}±{std:.2f} MB/s"
                )
                logger.warning(message)
                # Record anomaly in history
                self._record_anomaly(z_score, current_speed, mean, std, 'sudden_drop')
            else:
                anomaly_type = 'spike'
                message = f"SPEED SPIKE detected: z-score={z_score:.2f}, current={current_speed:.2f} MB/s vs baseline={mean:.2f} MB/s"
                logger.info(message)
                # Record anomaly in history
                self._record_anomaly(z_score, current_speed, mean, std, 'spike')
        else:
            anomaly_type = 'none'
            message = f"No anomaly: z-score={z_score:.2f} (threshold={self.config.anomaly_threshold})"

        return SpeedAnomalySignal(
            is_anomalous=is_anomalous,
            z_score=round(z_score, 3),
            current_speed_mbps=round(current_speed, 2),
            baseline_mean_mbps=round(mean, 2),
            baseline_std_mbps=round(std, 2),
            anomaly_type=anomaly_type,
            message=message
        )

    def _record_anomaly(
        self,
        z_score: float,
        current_speed: float,
        baseline_mean: float,
        baseline_std: float,
        anomaly_type: str
    ) -> None:
        """Record an anomaly in history for debugging.

        Args:
            z_score: Z-score of the anomaly
            current_speed: Current speed in MB/s
            baseline_mean: Mean of baseline window
            baseline_std: Standard deviation of baseline
            anomaly_type: Type of anomaly ('sudden_drop' or 'spike')
        """
        import time as time_module
        record = AnomalyRecord(
            timestamp=time_module.time(),
            z_score=z_score,
            current_speed_mbps=current_speed,
            baseline_mean_mbps=baseline_mean,
            baseline_std_mbps=baseline_std,
            anomaly_type=anomaly_type
        )
        self._anomaly_history.append(record)
        self._anomaly_count += 1

        if anomaly_type == 'sudden_drop':
            self._sudden_drop_count += 1
        elif anomaly_type == 'spike':
            self._spike_count += 1

        logger.debug(
            f"Anomaly recorded: {anomaly_type} (z={z_score:.2f}), "
            f"total={self._anomaly_count}, drops={self._sudden_drop_count}, spikes={self._spike_count}"
        )

    def get_anomaly_metrics(self) -> AnomalyMetrics:
        """Get metrics for anomaly detection.

        Returns:
            AnomalyMetrics with anomaly detection statistics.
        """
        history_dicts = [record.to_dict() for record in self._anomaly_history]

        return AnomalyMetrics(
            total_anomalies_detected=self._anomaly_count,
            sudden_drops=self._sudden_drop_count,
            spikes=self._spike_count,
            anomaly_history=history_dicts
        )

    def get_anomaly_history(self) -> List[Dict[str, Any]]:
        """Get anomaly history for debugging.

        Returns:
            List of anomaly records as dictionaries.
        """
        return [record.to_dict() for record in self._anomaly_history]

    def clear_anomaly_history(self) -> None:
        """Clear anomaly history."""
        self._anomaly_history.clear()
        self._anomaly_count = 0
        self._sudden_drop_count = 0
        self._spike_count = 0

    def get_anomaly_signal_for_escalation(self) -> Dict[str, Any]:
        """Get anomaly signal formatted for integration with rate limit predictor.

        This method returns a signal that can be passed to the rate limit predictor
        to record anomaly events for pattern analysis.

        Returns:
            Dict with anomaly signal details for escalation integration.

        Example:
            signal = tracker.get_anomaly_signal_for_escalation()
            if signal['should_escalate']:
                # Pass to rate limit predictor
                predictor.record_rate_limit_event(
                    trigger_category='speed_anomaly',
                    keyword=signal.get('anomaly_type')
                )
        """
        # Get latest anomaly if any
        if not self._anomaly_history:
            return {
                'has_anomaly': False,
                'should_escalate': False,
                'anomaly_type': None,
                'z_score': 0.0,
                'message': 'No anomalies recorded'
            }

        latest = self._anomaly_history[-1]

        # Determine if should escalate based on anomaly type and severity
        # Sudden drops are more concerning than spikes
        should_escalate = (
            latest.anomaly_type == 'sudden_drop' and
            abs(latest.z_score) >= self.config.anomaly_threshold * 1.5  # More severe
        )

        return {
            'has_anomaly': True,
            'should_escalate': should_escalate,
            'anomaly_type': latest.anomaly_type,
            'z_score': latest.z_score,
            'current_speed_mbps': latest.current_speed_mbps,
            'baseline_mean_mbps': latest.baseline_mean_mbps,
            'baseline_std_mbps': latest.baseline_std_mbps,
            'timestamp': latest.timestamp,
            'total_anomalies': self._anomaly_count,
            'recent_drops': self._sudden_drop_count,
            'recent_spikes': self._spike_count,
            'message': (
                f"Anomaly signal: {latest.anomaly_type} (z={latest.z_score:.2f}), "
                f"should_escalate={should_escalate}"
            )
        }

    def record_anomaly_to_predictor(
        self,
        predictor: "RateLimitPredictor",
        keyword: Optional[str] = None
    ) -> None:
        """Record anomaly events to rate limit predictor for pattern analysis.

        This integrates anomaly detection with the predictive rate limit system,
        allowing historical anomaly patterns to inform future rate limit predictions.

        Args:
            predictor: RateLimitPredictor instance to record to
            keyword: Optional keyword associated with the downloads

        Example:
            predictor = RateLimitPredictor()
            tracker.record_anomaly_to_predictor(predictor, keyword="tutorial")
        """
        signal = self.get_anomaly_signal_for_escalation()

        if signal['has_anomaly'] and signal['should_escalate']:
            # Record as a potential rate limit event
            predictor.record_rate_limit_event(
                timestamp=signal['timestamp'],
                trigger_category=f"speed_anomaly_{signal['anomaly_type']}",
                tier="tier1",
                keyword=keyword
            )
            logger.info(
                f"Anomaly recorded to predictor: {signal['anomaly_type']} "
                f"(z={signal['z_score']:.2f}) for keyword={keyword}"
            )

    def get_adaptive_threshold(self, base_threshold: float = None) -> AdaptiveThresholdResult:
        """Get adaptive speed threshold based on time-of-day.

        Different times of day have different typical network conditions:
        - Morning (6-12): Moderate usage, standard thresholds
        - Afternoon (12-18): Higher usage, slightly more lenient
        - Evening (18-22): Peak usage, most lenient thresholds
        - Overnight (22-6): Lowest usage, strictest thresholds

        Args:
            base_threshold: Base threshold to adapt. If None, uses config rate_limit_signal_threshold.

        Returns:
            AdaptiveThresholdResult with adapted threshold and context.

        Example:
            result = tracker.get_adaptive_threshold(0.5)
            if result.threshold_mbps < base_threshold:
                logger.info(f"Adapted threshold for {result.time_period}: {result.threshold_mbps}")
        """
        if not self.config.enable_adaptive_thresholds:
            threshold = base_threshold or self.config.rate_limit_signal_threshold
            return AdaptiveThresholdResult(
                threshold_mbps=threshold,
                time_period='unknown',
                multiplier=1.0,
                reason="Adaptive thresholds disabled"
            )

        # Get current hour
        current_hour = time.localtime().tm_hour

        # Determine time period and set multiplier
        if 6 <= current_hour < 12:
            time_period = 'morning'
            multiplier = 1.0
            reason = "Morning hours - standard network conditions"
        elif 12 <= current_hour < 18:
            time_period = 'afternoon'
            multiplier = 1.1
            reason = "Afternoon hours - slightly higher usage expected"
        elif 18 <= current_hour < 22:
            time_period = 'evening'
            multiplier = 1.3
            reason = "Evening peak hours - higher usage expected"
        else:  # 22-6
            time_period = 'overnight'
            multiplier = 0.8
            reason = "Overnight hours - lower usage, stricter thresholds"

        base = base_threshold or self.config.rate_limit_signal_threshold
        threshold = base * multiplier

        logger.debug(
            f"Adaptive threshold: {time_period} (hour {current_hour}) - "
            f"multiplier={multiplier}, threshold={threshold:.3f} MB/s"
        )

        return AdaptiveThresholdResult(
            threshold_mbps=round(threshold, 3),
            time_period=time_period,
            multiplier=round(multiplier, 2),
            reason=reason
        )

    def get_keyword_adaptive_threshold(
        self,
        base_threshold: float = None
    ) -> KeywordAdaptiveThresholdResult:
        """Get adaptive speed threshold based on historical average for this keyword.

        US-144-008: Dynamic speed threshold based on historical average per keyword.
        The threshold is calculated as: historical_avg * adaptive_threshold_multiplier

        This provides a personalized threshold based on the actual download speeds
        observed for this specific keyword, rather than using a global static threshold.

        Args:
            base_threshold: Base/fallback threshold to use when not enough samples.
                          If None, uses config.rate_limit_signal_threshold.

        Returns:
            KeywordAdaptiveThresholdResult with adapted threshold and context.

        Example:
            result = tracker.get_keyword_adaptive_threshold(0.1)
            if result.is_adaptive:
                logger.info(f"Using adaptive threshold {result.threshold_mbps} MB/s "
                           f"based on {result.samples_used} samples")
            else:
                logger.info(f"Using fallback threshold {result.threshold_mbps} MB/s "
                           f"(need {config.min_samples_for_adaptive_threshold} samples)")
        """
        if not self.config.enable_keyword_adaptive_threshold:
            fallback = base_threshold or self.config.rate_limit_signal_threshold
            return KeywordAdaptiveThresholdResult(
                threshold_mbps=fallback,
                historical_avg_mbps=0.0,
                samples_used=len(self._records),
                multiplier=1.0,
                is_adaptive=False,
                fallback_threshold_mbps=fallback,
                reason="Keyword adaptive threshold disabled"
            )

        min_samples = self.config.min_samples_for_adaptive_threshold
        fallback = base_threshold or self.config.rate_limit_signal_threshold

        # Check if we have enough samples for adaptive threshold
        if len(self._records) < min_samples:
            return KeywordAdaptiveThresholdResult(
                threshold_mbps=fallback,
                historical_avg_mbps=self.get_average_speed_mbps(),
                samples_used=len(self._records),
                multiplier=self.config.adaptive_threshold_multiplier,
                is_adaptive=False,
                fallback_threshold_mbps=fallback,
                reason=f"Insufficient samples ({len(self._records)}/{min_samples})"
            )

        # Calculate historical average
        historical_avg = self.get_average_speed_mbps()

        # Calculate adaptive threshold as historical_avg * multiplier
        multiplier = self.config.adaptive_threshold_multiplier
        adaptive_threshold = historical_avg * multiplier

        # Ensure minimum threshold is at least the fallback value
        # (don't allow threshold to go below fallback)
        threshold = max(adaptive_threshold, fallback * 0.1)  # At least 10% of fallback

        logger.debug(
            f"Keyword adaptive threshold: {threshold:.3f} MB/s "
            f"(historical_avg={historical_avg:.2f} MB/s, multiplier={multiplier}, "
            f"samples={len(self._records)})"
        )

        return KeywordAdaptiveThresholdResult(
            threshold_mbps=round(threshold, 3),
            historical_avg_mbps=round(historical_avg, 3),
            samples_used=len(self._records),
            multiplier=round(multiplier, 2),
            is_adaptive=True,
            fallback_threshold_mbps=fallback,
            reason=f"Adaptive threshold based on {len(self._records)} historical samples"
        )

    def get_warning_metrics(self) -> SpeedWarningMetrics:
        """Get metrics for speed-based early warnings.

        Returns:
            SpeedWarningMetrics with warning statistics.
        """
        # For now, return basic metrics - could be expanded with instance tracking
        return SpeedWarningMetrics(
            early_warnings_issued=0,
            warnings_by_level={'minor': 0, 'moderate': 0, 'severe': 0},
            anomaly_count=0,
            adaptive_threshold_adjustments=0,
            total_speed_checks=len(self._records)
        )

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
            ],
            # US-143-003: Anomaly detection history
            'anomaly_history': [
                record.to_dict() for record in self._anomaly_history
            ],
            'anomaly_count': self._anomaly_count,
            'sudden_drop_count': self._sudden_drop_count,
            'spike_count': self._spike_count,
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

        # US-143-003: Restore anomaly history
        self._anomaly_history.clear()
        anomaly_records = data.get('anomaly_history', [])
        for ar in anomaly_records:
            try:
                record = AnomalyRecord.from_dict(ar)
                self._anomaly_history.append(record)
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping invalid anomaly record: {e}")

        self._anomaly_count = data.get('anomaly_count', 0)
        self._sudden_drop_count = data.get('sudden_drop_count', 0)
        self._spike_count = data.get('spike_count', 0)

        if self._anomaly_history:
            logger.debug(f"Restored {len(self._anomaly_history)} anomaly records from checkpoint")

    def clear(self) -> None:
        """Clear all speed records."""
        self._records.clear()

    # =========================================================================
    # SELF-REGULATION THROTTLING (US-123-010)
    # =========================================================================

    def record_error(self, error_type: str = "rate_limit") -> None:
        """Record an error for throttling analysis.

        Args:
            error_type: Type of error (e.g., 'rate_limit', '429', '403')
        """
        current_time = time.time()
        self._error_timestamps.append(current_time)
        logger.debug(f"Recorded {error_type} error at {current_time}")

    def _get_error_count_in_window(self, window_seconds: int) -> int:
        """Get number of errors within the specified time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Number of errors in the window
        """
        if not self._error_timestamps:
            return 0

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Count errors within window
        count = sum(1 for ts in self._error_timestamps if ts >= cutoff_time)
        return count

    def should_throttle(
        self,
        threshold: int = 3,
        window_seconds: int = 600,
        current_concurrency: int = 4,
        throttle_factor: float = 0.5,
        min_concurrent: int = 1
    ) -> ThrottleSignal:
        """Determine if throttling should be applied based on error patterns.

        Args:
            threshold: Number of errors in window
            window_seconds: Time window to to trigger throttling track errors
            current_concurrency: Current concurrency level
            throttle_factor: Factor to reduce concurrency by
            min_concurrent: Minimum concurrency to maintain

        Returns:
            ThrottleSignal with recommendation
        """
        error_count = self._get_error_count_in_window(window_seconds)

        # Initialize throttle state if needed
        if self._throttle_state is None:
            self._throttle_state = ThrottleState(
                current_concurrency=current_concurrency,
                original_concurrency=current_concurrency
            )

        # Check if we should throttle
        if error_count >= threshold and not self._throttle_state.is_throttled:
            # Calculate new concurrency
            new_concurrency = max(
                int(current_concurrency * (1 - throttle_factor)),
                min_concurrent
            )

            self._throttle_state.is_throttled = True
            self._throttle_state.throttle_level += 1
            self._throttle_state.current_concurrency = new_concurrency
            self._throttle_state.last_throttle_time = time.time()

            reason = (
                f"Throttling: {error_count} errors in {window_seconds}s "
                f"(threshold={threshold}). Reducing concurrency {current_concurrency} → {new_concurrency}"
            )
            logger.warning(reason)

            return ThrottleSignal(
                should_throttle=True,
                should_recover=False,
                new_concurrency=new_concurrency,
                reason=reason,
                throttle_level=self._throttle_state.throttle_level
            )

        # Update throttle state even if not throttling yet
        if self._error_timestamps:
            self._throttle_state.last_error_time = self._error_timestamps[-1]
        self._throttle_state.error_count_in_window = error_count

        return ThrottleSignal(
            should_throttle=False,
            should_recover=False,
            new_concurrency=current_concurrency,
            reason=f"No throttling needed: {error_count}/{threshold} errors in window",
            throttle_level=self._throttle_state.throttle_level
        )

    def should_recover(
        self,
        recovery_window_seconds: int = 300,
        current_concurrency: int = 4,
        recovery_factor: float = 0.25,
        max_recovery_concurrency: int = 0
    ) -> ThrottleSignal:
        """Determine if recovery (speed increase) should be attempted.

        Recovery is attempted when no errors have occurred within the recovery window.

        Args:
            recovery_window_seconds: Time without errors before recovery
            current_concurrency: Current concurrency level
            recovery_factor: Factor to increase concurrency by
            max_recovery_concurrency: Max concurrency to recover to (0 = no limit)

        Returns:
            ThrottleSignal with recovery recommendation
        """
        if self._throttle_state is None or not self._throttle_state.is_throttled:
            return ThrottleSignal(
                should_throttle=False,
                should_recover=False,
                new_concurrency=current_concurrency,
                reason="Not currently throttled",
                throttle_level=0
            )

        current_time = time.time()
        last_error_time = self._throttle_state.last_error_time

        # Check if enough time has passed without errors
        if last_error_time > 0 and (current_time - last_error_time) < recovery_window_seconds:
            return ThrottleSignal(
                should_throttle=False,
                should_recover=False,
                new_concurrency=current_concurrency,
                reason=f"Recovery waiting: {current_time - last_error_time:.0f}s since last error (need {recovery_window_seconds}s)",
                throttle_level=self._throttle_state.throttle_level
            )

        # Determine max recovery concurrency
        max_conc = max_recovery_concurrency if max_recovery_concurrency > 0 else self._throttle_state.original_concurrency

        # Calculate new concurrency (gradual recovery)
        new_concurrency = min(
            int(current_concurrency * (1 + recovery_factor)),
            max_conc
        )

        # Only recover if we can actually increase
        if new_concurrency <= current_concurrency:
            return ThrottleSignal(
                should_throttle=False,
                should_recover=False,
                new_concurrency=current_concurrency,
                reason=f"Already at max recovery level ({current_concurrency})",
                throttle_level=self._throttle_state.throttle_level
            )

        # Apply recovery
        self._throttle_state.current_concurrency = new_concurrency
        self._throttle_state.recovery_step += 1
        self._throttle_state.is_recovering = True

        reason = (
            f"Recovering: {recovery_window_seconds}s without errors. "
            f"Increasing concurrency {current_concurrency} → {new_concurrency}"
        )
        logger.info(reason)

        return ThrottleSignal(
            should_throttle=False,
            should_recover=True,
            new_concurrency=new_concurrency,
            reason=reason,
            throttle_level=self._throttle_state.throttle_level
        )

    def reset_throttle(self) -> None:
        """Reset throttle state to allow fresh start."""
        self._throttle_state = None
        self._error_timestamps.clear()
        logger.info("Throttle state reset")

    def get_throttle_state(self) -> Optional[ThrottleState]:
        """Get current throttle state for reporting.

        Returns:
            Current ThrottleState or None if not initialized
        """
        return self._throttle_state

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize tracker state for checkpoint persistence.

        Returns:
            Dictionary that can be saved to checkpoint.json
        """
        data = {
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

        # Include throttle state
        if self._throttle_state is not None:
            data['throttle_state'] = {
                'current_concurrency': self._throttle_state.current_concurrency,
                'original_concurrency': self._throttle_state.original_concurrency,
                'throttle_level': self._throttle_state.throttle_level,
                'is_throttled': self._throttle_state.is_throttled,
                'last_throttle_time': self._throttle_state.last_throttle_time,
                'last_error_time': self._throttle_state.last_error_time,
                'is_recovering': self._throttle_state.is_recovering,
                'recovery_step': self._throttle_state.recovery_step
            }

        # Include error timestamps
        data['error_timestamps'] = list(self._error_timestamps)

        # US-143-003: Anomaly detection history
        data['anomaly_history'] = [
            record.to_dict() for record in self._anomaly_history
        ]
        data['anomaly_count'] = self._anomaly_count
        data['sudden_drop_count'] = self._sudden_drop_count
        data['spike_count'] = self._spike_count

        return data

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

        # Restore throttle state
        throttle_data = data.get('throttle_state')
        if throttle_data:
            self._throttle_state = ThrottleState(
                current_concurrency=throttle_data.get('current_concurrency', 4),
                original_concurrency=throttle_data.get('original_concurrency', 4),
                throttle_level=throttle_data.get('throttle_level', 0),
                is_throttled=throttle_data.get('is_throttled', False),
                last_throttle_time=throttle_data.get('last_throttle_time', 0.0),
                last_error_time=throttle_data.get('last_error_time', 0.0),
                is_recovering=throttle_data.get('is_recovering', False),
                recovery_step=throttle_data.get('recovery_step', 0)
            )
            logger.debug(f"Restored throttle state: level={self._throttle_state.throttle_level}, is_throttled={self._throttle_state.is_throttled}")

        # Restore error timestamps
        self._error_timestamps = deque(data.get('error_timestamps', []), maxlen=100)

        # US-143-003: Restore anomaly history
        self._anomaly_history.clear()
        anomaly_records = data.get('anomaly_history', [])
        for ar in anomaly_records:
            try:
                record = AnomalyRecord.from_dict(ar)
                self._anomaly_history.append(record)
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping invalid anomaly record: {e}")

        self._anomaly_count = data.get('anomaly_count', 0)
        self._sudden_drop_count = data.get('sudden_drop_count', 0)
        self._spike_count = data.get('spike_count', 0)

        if self._anomaly_history:
            logger.debug(f"Restored {len(self._anomaly_history)} anomaly records from checkpoint")


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

    def detect_sustained_degradation(self, keyword: str) -> SustainedDegradationSignal:
        """Detect sustained degradation for a specific keyword."""
        if keyword not in self._trackers:
            return SustainedDegradationSignal(
                detected=False,
                degradation_percentage=0.0,
                samples_analyzed=0,
                peak_speed_mbps=0.0,
                current_speed_mbps=0.0,
                trend='insufficient_data',
                message=f"No data for keyword '{keyword}'"
            )
        return self._get_tracker(keyword).detect_sustained_degradation()

    def detect_correlated_signals(
        self,
        keyword: str,
        error_signal: Optional[ErrorPatternSignal] = None
    ) -> CorrelatedSignal:
        """Detect correlated signals for a specific keyword."""
        if keyword not in self._trackers:
            return CorrelatedSignal(
                detected=False,
                correlation_score=0.0,
                speed_signal_score=0.0,
                error_signal_score=0.0,
                speed_signal_detected=False,
                error_signal_detected=False,
                recommended_action='none',
                message=f"No data for keyword '{keyword}'"
            )
        return self._get_tracker(keyword).detect_correlated_signals(error_signal)

    def detect_early_warning(self, keyword: str) -> EarlyWarningSignal:
        """Detect early warning for a specific keyword."""
        if keyword not in self._trackers:
            return EarlyWarningSignal(
                detected=False,
                warning_level='none',
                degradation_percentage=0.0,
                baseline_speed_mbps=0.0,
                current_speed_mbps=0.0,
                recommended_action='none',
                message=f"No data for keyword '{keyword}'"
            )
        return self._get_tracker(keyword).detect_early_warning()

    def detect_anomaly(self, keyword: str) -> SpeedAnomalySignal:
        """Detect speed anomaly for a specific keyword."""
        if keyword not in self._trackers:
            return SpeedAnomalySignal(
                is_anomalous=False,
                z_score=0.0,
                current_speed_mbps=0.0,
                baseline_mean_mbps=0.0,
                baseline_std_mbps=0.0,
                anomaly_type='none',
                message=f"No data for keyword '{keyword}'"
            )
        return self._get_tracker(keyword).detect_anomaly()

    def get_adaptive_threshold(
        self,
        keyword: str,
        base_threshold: float = None
    ) -> AdaptiveThresholdResult:
        """Get adaptive threshold for a specific keyword."""
        if keyword not in self._trackers:
            threshold = base_threshold or self.config.rate_limit_signal_threshold
            return AdaptiveThresholdResult(
                threshold_mbps=threshold,
                time_period='unknown',
                multiplier=1.0,
                reason=f"No data for keyword '{keyword}'"
            )
        return self._get_tracker(keyword).get_adaptive_threshold(base_threshold)

    def get_keyword_adaptive_threshold(
        self,
        keyword: str,
        base_threshold: float = None
    ) -> KeywordAdaptiveThresholdResult:
        """Get adaptive threshold based on historical average for a specific keyword.

        US-144-008: Dynamic speed threshold based on historical average per keyword.
        Each keyword gets its own personalized threshold based on observed speeds.

        Args:
            keyword: The keyword to get adaptive threshold for
            base_threshold: Base/fallback threshold to use when not enough samples

        Returns:
            KeywordAdaptiveThresholdResult with adapted threshold for this keyword.

        Example:
            tracker = PerKeywordSpeedTracker(config)
            # Record downloads for "python tutorial" keyword
            tracker.record_download("python tutorial", "vid1", 50*1024*1024, 10.0, "short")
            ...

            # Get adaptive threshold for this keyword
            result = tracker.get_keyword_adaptive_threshold("python tutorial", 0.1)
            if result.is_adaptive:
                # Use personalized threshold
                threshold = result.threshold_mbps
        """
        if keyword not in self._trackers:
            fallback = base_threshold or self.config.rate_limit_signal_threshold
            return KeywordAdaptiveThresholdResult(
                threshold_mbps=fallback,
                historical_avg_mbps=0.0,
                samples_used=0,
                multiplier=self.config.adaptive_threshold_multiplier,
                is_adaptive=False,
                fallback_threshold_mbps=fallback,
                reason=f"No data for keyword '{keyword}'"
            )
        return self._get_tracker(keyword).get_keyword_adaptive_threshold(base_threshold)

    def get_warning_metrics(self, keyword: str) -> SpeedWarningMetrics:
        """Get warning metrics for a specific keyword."""
        if keyword not in self._trackers:
            return SpeedWarningMetrics(
                early_warnings_issued=0,
                warnings_by_level={'minor': 0, 'moderate': 0, 'severe': 0},
                anomaly_count=0,
                adaptive_threshold_adjustments=0,
                total_speed_checks=0
            )
        return self._get_tracker(keyword).get_warning_metrics()

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
