"""
Download Segments Stage - Download matched video segments

Stage 6 of the simplified 7-stage pipeline:
- Downloads only the matched video segments (not full videos)
- Runs after MATCH and ITERATIVE_MATCH stages
- Efficient: only downloads portions of videos that are actually used
- US-48-005: Integrates 4-tier escalation (impersonation, extractor_args,
  cookie rotation) via EscalationManager for resilient downloading
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import random
import shutil
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from . import Stage, StageMetrics, StageResult, register_stage, validate_required_state_attrs
from .error_aggregator import ErrorAggregator
from ..logging_templates import (
    log_stage_start,
    log_stage_complete,
    log_stage_skip,
    log_progress,
    log_error_with_context,
)


@runtime_checkable
class DownloadProgressCallback(Protocol):
    """Protocol for download progress callbacks.

    US-89-010: Enables real-time progress UI and external monitoring integrations.
    Implement this protocol to receive download progress events.

    Methods:
        on_start: Called when a download starts
        on_progress: Called periodically during download with bytes/speed/ETA
        on_complete: Called when download completes successfully
        on_error: Called when download fails
    """

    def on_start(self, video_id: str, segment_info: Dict[str, Any]) -> None:
        """Called when a download segment starts.

        Args:
            video_id: YouTube video ID
            segment_info: Dict with 'start', 'end', 'output_file' keys
        """
        ...

    def on_progress(
        self,
        video_id: str,
        bytes_downloaded: int,
        total_bytes: int,
        speed: Optional[float],
        eta: Optional[float],
    ) -> None:
        """Called periodically during download with progress data.

        Args:
            video_id: YouTube video ID
            bytes_downloaded: Bytes downloaded so far
            total_bytes: Total bytes to download (may be estimate)
            speed: Download speed in bytes per second (None if unknown)
            eta: Estimated seconds remaining (None if unknown)
        """
        ...

    def on_complete(self, video_id: str, file_path: str, file_bytes: int, duration: float) -> None:
        """Called when download completes successfully.

        Args:
            video_id: YouTube video ID
            file_path: Path to downloaded file
            file_bytes: Size of downloaded file in bytes
            duration: Time taken to download in seconds
        """
        ...

    def on_error(self, video_id: str, error_msg: str) -> None:
        """Called when download fails.

        Args:
            video_id: YouTube video ID
            error_msg: Error message describing the failure
        """
        ...


class MultiCallback:
    """Delegate pattern for multiple callbacks.

    US-89-010: Supports multiple progress callbacks for UI integrations.
    """

    def __init__(self, callbacks: Optional[List[DownloadProgressCallback]] = None):
        self._callbacks = list(callbacks) if callbacks else []

    def add(self, callback: DownloadProgressCallback) -> None:
        """Add a callback to the delegate."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove(self, callback: DownloadProgressCallback) -> None:
        """Remove a callback from the delegate."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def on_start(self, video_id: str, segment_info: Dict[str, Any]) -> None:
        """Forward on_start to all callbacks."""
        for cb in self._callbacks:
            try:
                cb.on_start(video_id, segment_info)
            except Exception:
                pass  # Don't let callback errors break downloads

    def on_progress(
        self,
        video_id: str,
        bytes_downloaded: int,
        total_bytes: int,
        speed: Optional[float],
        eta: Optional[float],
    ) -> None:
        """Forward on_progress to all callbacks."""
        for cb in self._callbacks:
            try:
                cb.on_progress(video_id, bytes_downloaded, total_bytes, speed, eta)
            except Exception:
                pass

    def on_complete(self, video_id: str, file_path: str, file_bytes: int, duration: float) -> None:
        """Forward on_complete to all callbacks."""
        for cb in self._callbacks:
            try:
                cb.on_complete(video_id, file_path, file_bytes, duration)
            except Exception:
                pass

    def on_error(self, video_id: str, error_msg: str) -> None:
        """Forward on_error to all callbacks."""
        for cb in self._callbacks:
            try:
                cb.on_error(video_id, error_msg)
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._callbacks)


# US-113-010: Download progress utilities for ETA and bandwidth calculation

# Default assumed maximum bandwidth in MB/s (used for utilization calculation)
DEFAULT_MAX_BANDWIDTH_MBPS = 100.0


def calculate_eta_seconds(downloaded_bytes: int, total_bytes: int, speed_bytes_per_sec: Optional[float]) -> Optional[float]:
    """Calculate ETA in seconds based on current download speed and remaining size.

    Args:
        downloaded_bytes: Number of bytes already downloaded
        total_bytes: Total size of the file in bytes
        speed_bytes_per_sec: Current download speed in bytes per second

    Returns:
        Estimated seconds remaining, or None if calculation not possible
    """
    if speed_bytes_per_sec is None or speed_bytes_per_sec <= 0:
        return None

    remaining_bytes = total_bytes - downloaded_bytes
    if remaining_bytes <= 0:
        return 0.0

    return remaining_bytes / speed_bytes_per_sec


def calculate_bandwidth_utilization(speed_bytes_per_sec: Optional[float], max_bandwidth_mbps: float = DEFAULT_MAX_BANDWIDTH_MBPS) -> Optional[float]:
    """Calculate bandwidth utilization percentage.

    Args:
        speed_bytes_per_sec: Current download speed in bytes per second
        max_bandwidth_mbps: Assumed maximum bandwidth in MB/s (default: 100 MB/s)

    Returns:
        Bandwidth utilization as percentage (0-100), or None if speed unavailable
    """
    if speed_bytes_per_sec is None or speed_bytes_per_sec <= 0:
        return None

    # Convert max bandwidth to bytes/sec
    max_bytes_per_sec = max_bandwidth_mbps * 1024 * 1024

    # Calculate utilization percentage
    utilization = (speed_bytes_per_sec / max_bytes_per_sec) * 100.0

    # Cap at 100% (could exceed due to measurement variance)
    return min(utilization, 100.0)


def format_eta_display(eta_seconds: Optional[float]) -> str:
    """Format ETA seconds as human-readable string.

    Args:
        eta_seconds: ETA in seconds

    Returns:
        Formatted string like "5m 30s" or "< 1s" or "unknown"
    """
    if eta_seconds is None:
        return "unknown"

    if eta_seconds <= 0:
        return "< 1s"

    if eta_seconds < 1:
        return "< 1s"

    if eta_seconds < 60:
        return f"{int(eta_seconds)}s"

    minutes = int(eta_seconds // 60)
    remaining_seconds = int(eta_seconds % 60)

    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        return f"{minutes}m"

    hours = minutes // 60
    remaining_minutes = minutes % 60
    if remaining_minutes > 0:
        return f"{hours}h {remaining_minutes}m"
    return f"{hours}h"


def format_progress_line(
    video_id: str,
    downloaded_mb: float,
    total_mb: float,
    speed_kbps: float,
    eta_seconds: Optional[float],
    bandwidth_util: Optional[float],
    tier: Optional[int],
    elapsed: float,
) -> str:
    """Format a consistent progress output line.

    Args:
        video_id: YouTube video ID
        downloaded_mb: Downloaded size in MB
        total_mb: Total size in MB
        speed_kbps: Download speed in KB/s
        eta_seconds: Estimated seconds remaining
        bandwidth_util: Bandwidth utilization percentage
        tier: Current escalation tier (1-4)
        elapsed: Elapsed time in seconds

    Returns:
        Formatted progress string consistent with pipeline logging
    """
    eta_str = format_eta_display(eta_seconds)
    util_str = f"{bandwidth_util:.0f}%" if bandwidth_util is not None else "N/A"
    tier_str = f"T{tier}" if tier is not None else "T1"

    return (
        f"[DOWNLOAD] {video_id}: {downloaded_mb:.1f}MB / {total_mb:.1f}MB | "
        f"{speed_kbps:.0f}KB/s | ETA: {eta_str} | BW: {util_str} | {tier_str} | {elapsed:.0f}s"
    )


def calculate_network_congestion_factor(speeds_bytes_per_sec: List[float]) -> float:
    """Calculate network congestion factor based on speed variance.

    Higher variance indicates network congestion or instability, which affects
    ETA accuracy. Returns a factor between 0.5 (very stable) and 2.0 (very congested).

    Args:
        speeds_bytes_per_sec: List of recent download speeds in bytes/second

    Returns:
        Congestion factor: 1.0 = normal, >1.0 = congested (less accurate ETA),
        <1.0 = better than expected
    """
    if not speeds_bytes_per_sec or len(speeds_bytes_per_sec) < 2:
        return 1.0  # No variance data, assume normal

    # Calculate coefficient of variation (CV)
    mean_speed = statistics.mean(speeds_bytes_per_sec)
    if mean_speed <= 0:
        return 1.0

    try:
        stdev = statistics.stdev(speeds_bytes_per_sec)
    except statistics.StatisticsError:
        return 1.0

    cv = stdev / mean_speed  # Coefficient of variation

    # Map CV to congestion factor:
    # CV < 0.1 (very stable): factor = 0.9 (optimistic)
    # CV = 0.3 (normal): factor = 1.0
    # CV > 0.5 (very variable): factor = 1.5+
    if cv < 0.1:
        return 0.9
    elif cv < 0.2:
        return 0.95
    elif cv < 0.3:
        return 1.0
    elif cv < 0.5:
        return 1.25
    elif cv < 0.75:
        return 1.5
    else:
        return 2.0  # High congestion, ETA may be very inaccurate


def calculate_eta_confidence_interval(
    eta_seconds: Optional[float],
    speeds_bytes_per_sec: List[float],
    confidence_level: float = 0.8
) -> tuple[Optional[float], Optional[float]]:
    """Calculate confidence interval for ETA based on speed variance.

    Args:
        eta_seconds: Base ETA calculation in seconds
        speeds_bytes_per_sec: List of recent download speeds for variance
        confidence_level: Confidence level (0.8 = 80%, 0.9 = 90%, etc.)

    Returns:
        Tuple of (lower_bound, upper_bound) seconds, or (None, None) if unable to calculate
    """
    if eta_seconds is None or eta_seconds <= 0:
        return None, None

    if not speeds_bytes_per_sec or len(speeds_bytes_per_sec) < 3:
        return eta_seconds * 0.8, eta_seconds * 1.2  # Default 20% margin

    # Calculate coefficient of variation
    mean_speed = statistics.mean(speeds_bytes_per_sec)
    if mean_speed <= 0:
        return eta_seconds * 0.8, eta_seconds * 1.2

    try:
        stdev = statistics.stdev(speeds_bytes_per_sec)
    except statistics.StatisticsError:
        return eta_seconds * 0.8, eta_seconds * 1.2

    cv = stdev / mean_speed

    # Map CV to margin percentage (higher variance = wider interval)
    if cv < 0.1:
        margin = 0.1  # 10% margin
    elif cv < 0.2:
        margin = 0.15
    elif cv < 0.3:
        margin = 0.2
    elif cv < 0.5:
        margin = 0.3
    else:
        margin = 0.5  # 50% margin for high variance

    # Adjust margin based on confidence level
    # Higher confidence = wider interval
    if confidence_level >= 0.9:
        margin *= 1.3
    elif confidence_level >= 0.95:
        margin *= 1.5

    lower_bound = eta_seconds * (1 - margin)
    upper_bound = eta_seconds * (1 + margin)

    return lower_bound, upper_bound


def format_eta_confidence_display(
    eta_seconds: Optional[float],
    lower_bound: Optional[float],
    upper_bound: Optional[float]
) -> str:
    """Format ETA with confidence interval for display.

    Args:
        eta_seconds: Base ETA in seconds
        lower_bound: Lower bound of confidence interval
        upper_bound: Upper bound of confidence interval

    Returns:
        Formatted string like "5m 30s (±1m)" or "unknown"
    """
    if eta_seconds is None:
        return "unknown"

    eta_str = format_eta_display(eta_seconds)

    if lower_bound is None or upper_bound is None:
        return eta_str

    # Format the range
    lower_str = format_eta_display(lower_bound)
    upper_str = format_eta_display(upper_bound)

    return f"{eta_str} (±{lower_str}-{upper_str})"


class ETAHistory:
    """Track ETA predictions vs actuals for accuracy analysis.

    Stores historical predictions and their outcomes to calculate
    actual ETA accuracy over time.
    """

    def __init__(self, max_history: int = 100):
        """Initialize ETA history tracker.

        Args:
            max_history: Maximum number of entries to retain
        """
        from collections import deque
        self._history: deque = deque(maxlen=max_history)
        self._predictions: deque = deque(maxlen=max_history)

    def record_prediction(
        self,
        predicted_eta: float,
        remaining_bytes: int,
        current_speed: float,
        timestamp: float
    ) -> None:
        """Record an ETA prediction.

        Args:
            predicted_eta: Predicted seconds remaining
            remaining_bytes: Estimated remaining bytes
            current_speed: Current speed in bytes/sec
            timestamp: Unix timestamp
        """
        self._predictions.append({
            'predicted_eta': predicted_eta,
            'remaining_bytes': remaining_bytes,
            'current_speed': current_speed,
            'timestamp': timestamp,
        })

    def record_actual(self, actual_duration: float) -> None:
        """Record actual time taken after completion.

        Args:
            actual_duration: Actual seconds taken
        """
        if not self._predictions:
            return

        prediction = self._predictions.popleft()
        self._history.append({
            'predicted_eta': prediction['predicted_eta'],
            'actual_duration': actual_duration,
            'remaining_bytes': prediction['remaining_bytes'],
            'current_speed': prediction['current_speed'],
        })

    def get_accuracy_stats(self) -> Dict[str, float]:
        """Calculate accuracy statistics from history.

        Returns:
            Dict with accuracy metrics:
            - mean_error_pct: Mean percentage error
            - max_error_pct: Maximum percentage error
            - accuracy_score: 100 - mean_error_pct
            - sample_count: Number of samples
        """
        # Calculate errors for all samples
        errors = []
        for entry in self._history:
            predicted = entry['predicted_eta']
            actual = entry['actual_duration']

            if predicted > 0 and actual > 0:
                error_pct = abs(predicted - actual) / actual * 100
                errors.append(error_pct)

        if not errors:
            return {
                'mean_error_pct': 0.0,
                'max_error_pct': 0.0,
                'accuracy_score': 100.0,
                'sample_count': len(self._history),
            }

        # With fewer than 3 samples, still report the error but note it may be unreliable
        return {
            'mean_error_pct': statistics.mean(errors),
            'max_error_pct': max(errors),
            'accuracy_score': 100.0 - statistics.mean(errors),
            'sample_count': len(self._history),
        }

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()
        self._predictions.clear()


if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, DownloadedVideo

logger = logging.getLogger(__name__)

# US-52-006: Error classification delegated to shared module
from ..downloader.error_classification import (
    is_network_failure as _is_network_failure,
    is_escalation_error as _is_escalation_error,
    classify_error_category,
    NETWORK_FAILURE_THRESHOLD,
    BOT_DETECTION_ABORT_THRESHOLD,
)


@dataclass
class _DownloadLoopContext:
    """Internal shared state for the download loop sub-methods.

    US-57-007: Holds references that the decomposed loop methods need,
    avoiding long parameter lists while keeping each method focused.
    """

    stats: 'SegmentDownloadStats'
    escalation_mgr: Any = None
    cookie_rotator: Any = None
    circuit_breaker: Any = None
    download_config: Any = None
    bot_floor_threshold: int = 5
    bot_abort_threshold: int = 0
    network_failure_threshold: int = 0
    consecutive_network_failures: int = 0
    consecutive_bot_detections: int = 0


@dataclass
class SegmentDownloadStats:
    """Typed container for segment download statistics.

    Replaces the previously untyped ``stats`` dict in ``_download_segments()``,
    providing IDE autocompletion, typo-safe attribute access, and explicit
    defaults for every field.
    """

    succeeded: int = 0
    failed: int = 0
    cached: int = 0
    attempted: int = 0
    total: int = 0
    retry_count: int = 0
    segment_durations: List[float] = field(default_factory=list)
    total_bytes: int = 0
    error_categories: Dict[str, int] = field(default_factory=dict)
    error_aggregator: ErrorAggregator = field(default_factory=ErrorAggregator)
    progress_hooks_data: Dict[str, int] = field(default_factory=lambda: {
        'total_downloaded_bytes': 0,
        'segments_with_progress': 0,
        'segments_finished': 0,
    })
    download_speeds: List[float] = field(default_factory=list)
    # US-81-002: Per-item error tracking for batch error isolation
    failed_items: List[Dict[str, Any]] = field(default_factory=list)

    # -- convenience mutators --------------------------------------------------

    def increment_success(self, duration: float = 0.0, file_bytes: int = 0) -> None:
        """Record a successful segment download."""
        self.succeeded += 1
        self.attempted += 1
        if duration:
            self.segment_durations.append(duration)
        if file_bytes:
            self.total_bytes += file_bytes
        if duration > 0 and file_bytes > 0:
            self.download_speeds.append(file_bytes / duration)

    def increment_failure(
        self,
        category: str | None = None,
        error_msg: str | None = None,
        video_id: str | None = None,
    ) -> None:
        """Record a failed segment download, optionally categorised.

        US-81-002: Also appends structured error info to failed_items list.
        """
        self.failed += 1
        self.attempted += 1
        if category:
            self.error_categories[category] = self.error_categories.get(category, 0) + 1
        if error_msg and category:
            self.error_aggregator.record(error_msg, category)
        # US-81-002: Track per-item failure details
        if video_id:
            self.failed_items.append({
                'video_id': video_id,
                'error_type': category or 'unknown',
                'message': error_msg or 'unknown error',
            })

    def increment_cached(self, file_bytes: int = 0) -> None:
        """Record a segment served from cache."""
        self.cached += 1
        self.attempted += 1
        if file_bytes:
            self.total_bytes += file_bytes

    def average_speed(self) -> float:
        """Return mean download speed in bytes/second across all segments.

        Returns 0.0 if no speed measurements have been recorded.
        """
        if not self.download_speeds:
            return 0.0
        return sum(self.download_speeds) / len(self.download_speeds)

    def is_speed_degrading(self, window: int = 5, threshold: float = 0.5) -> bool:
        """Detect sustained speed degradation over recent downloads.

        Returns True if the last *window* speeds are all below *threshold*
        of the overall average speed.  This indicates gradual throttling
        rather than a single slow download.

        Args:
            window: Number of most-recent speeds to inspect.
            threshold: Fraction of overall average below which a speed is
                       considered degraded (0.5 = 50 %).

        Returns:
            True if the tail window is consistently slow.
        """
        if len(self.download_speeds) < window:
            return False
        overall_avg = self.average_speed()
        if overall_avg <= 0:
            return False
        cutoff = overall_avg * threshold
        tail = self.download_speeds[-window:]
        return all(s < cutoff for s in tail)


def _apply_escalation_to_ydl_opts(ydl_opts: Dict[str, Any], escalation_result) -> None:
    """Translate EscalationResult CLI args to yt-dlp Python API ydl_opts.

    The EscalationManager returns CLI args (e.g., ['--impersonate', 'X',
    '--extractor-args', 'youtube:player_client=a,b']). This function
    translates them to ydl_opts dict keys for the Python API.

    Translation:
        --impersonate X           → ydl_opts['impersonate'] = 'X'
        --extractor-args youtube:player_client=X  → ydl_opts['extractor_args'] = {'youtube': {'player_client': 'X'}}

    Cookie rotation is handled separately via cookiefile, not via CLI args.

    Args:
        ydl_opts: The yt-dlp options dict to modify in-place.
        escalation_result: EscalationResult from EscalationManager.get_escalation_args().
    """
    if not escalation_result or not escalation_result.args:
        return

    args = escalation_result.args
    i = 0
    while i < len(args):
        if args[i] == '--impersonate' and i + 1 < len(args):
            try:
                from yt_dlp.networking.impersonate import ImpersonateTarget
            except ImportError:
                logger.warning(
                    "ImpersonateTarget not available in this yt-dlp version; "
                    "skipping impersonation for '%s'", args[i + 1]
                )
                i += 2
                continue
            try:
                target = ImpersonateTarget.from_str(args[i + 1])
                # Lowercase client/os fields - available targets are lowercase
                # but from_str() preserves original case
                target = ImpersonateTarget(
                    client=target.client.lower() if target.client else None,
                    version=target.version,
                    os=target.os.lower() if target.os else None,
                    os_version=target.os_version,
                )
                ydl_opts['impersonate'] = target
            except Exception:
                logger.warning(
                    "Failed to parse impersonate target '%s'; skipping impersonation",
                    args[i + 1]
                )
                # Fall back to no impersonation rather than crashing
            i += 2
        elif args[i] == '--extractor-args' and i + 1 < len(args):
            # Skip escalation's extractor_args — alternative player_clients
            # (web_safari, tv_downgraded, web) can trigger YouTube SABR
            # (Server-side Adaptive Bitrate) which returns no downloadable
            # format URLs, causing "No video formats found!".
            # We override with 'tv' client below instead.
            logger.debug(
                "Skipping escalation extractor_args for segment download: %s",
                args[i + 1],
            )
            i += 2
        else:
            i += 1


@register_stage
class DownloadVideoSegmentsStage(Stage):
    """
    Downloads video segments after matching.

    In the simplified 7-stage pipeline, this stage downloads only the
    portions of YouTube videos that were matched to voiceover segments.

    Inputs:
        - state.matches: List of Match objects with video IDs and time ranges
        - state.video_ids: List of video IDs from VIDEO_SEARCH stage

    Outputs:
        - state.downloaded_segments: List of DownloadedVideo with local file paths
        - Updated state.matches with local file paths
    """

    name = "DOWNLOAD_SEGMENTS"
    description = "Download matched video segments"
    DEPENDS_ON = ['MATCH']
    PRODUCES = ['downloaded_segments']

    # Browsers that yt-dlp supports for cookie extraction
    _KNOWN_BROWSERS = ('firefox', 'chrome', 'edge', 'safari', 'opera', 'brave')

    def __init__(
        self,
        orchestrator=None,
        circuit_breaker=None,
        escalation_manager=None,
        rate_limit_budget=None,
    ):
        self.downloader = None
        self._orchestrator = orchestrator
        # US-82-010: Constructor-injected dependencies for testability.
        # When None, defaults are created from self.downloader at runtime.
        self._injected_circuit_breaker = circuit_breaker
        self._injected_escalation_manager = escalation_manager
        self._injected_rate_limit_budget = rate_limit_budget

    def _get_orchestrator(self):
        """Return the orchestrator, creating a lightweight wrapper if needed.

        US-82-007: When tests set self.downloader directly (bypassing run()),
        wrap it in a SegmentDownloadOrchestrator so _execute_download works.
        """
        if self._orchestrator is not None:
            return self._orchestrator
        if self.downloader is not None:
            from ..downloader.orchestrator import SegmentDownloadOrchestrator
            orch = SegmentDownloadOrchestrator.__new__(SegmentDownloadOrchestrator)
            orch._config = None
            orch._downloader = self.downloader
            self._orchestrator = orch
        return self._orchestrator

    @staticmethod
    def _validate_cookie_config(download_config) -> None:
        """US-50-012 / US-52-009: Validate cookie configuration at stage init.

        Emits warnings when:
        - No cookie source is configured (neither cookies_from_browser,
          cookies_path, nor cookie_rotation with files)
        - cookies_from_browser is set to a browser that isn't installed

        These warnings appear once at stage startup, not per-download.
        """
        cookies_from_browser = getattr(download_config, 'cookies_from_browser', '')
        cookies_path = getattr(download_config, 'cookies_path', '')

        # US-52-009: Also check cookie_rotation as a valid cookie source
        cookie_rotation = getattr(download_config, 'cookie_rotation', None)
        has_cookie_rotation = False
        if cookie_rotation is not None:
            rotation_enabled = getattr(cookie_rotation, 'enabled', False)
            rotation_files = getattr(cookie_rotation, 'cookie_files', [])
            has_cookie_rotation = rotation_enabled and bool(rotation_files)

        if not cookies_from_browser and not cookies_path and not has_cookie_rotation:
            logger.warning(
                "No cookie source configured. YouTube will likely block all download "
                "requests with 403/bot-detection errors. "
                "Set download.cookies_from_browser to your browser name "
                "(firefox, chrome, edge, safari, opera, brave) in config.yaml"
            )
            return

        if cookies_from_browser:
            # Best-effort check: see if the browser executable is on PATH
            browser_exe = cookies_from_browser.lower()
            # Map browser names to common executable names
            _exe_map = {
                'firefox': 'firefox',
                'chrome': 'google-chrome' if shutil.which('google-chrome') else 'chrome',
                'edge': 'msedge',
                'safari': 'safari',
                'opera': 'opera',
                'brave': 'brave',
            }
            exe_name = _exe_map.get(browser_exe, browser_exe)
            if not shutil.which(exe_name) and not shutil.which(browser_exe):
                logger.warning(
                    f"cookies_from_browser is set to '{cookies_from_browser}' but "
                    f"'{cookies_from_browser}' does not appear to be installed "
                    f"(not found on PATH). Cookie extraction may fail. "
                    f"Verify the browser is installed or use a cookies_path file instead."
                )

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the video segment download stage.

        US-44-002: Validates required state attributes exist.
        """
        # US-167-009: Track stage timing
        stage_start_time = time.time()

        # US-44-002: Validate required attributes exist
        validate_required_state_attrs(state, ['matches'], self.name)

        warnings = []

        try:
            if not state.matches:
                log_stage_skip(logger, "DOWNLOAD_SEGMENTS", reason="no_matches")
                return StageResult.ok({'skipped': True, 'reason': 'no_matches'}, warnings)

            # US-59-011: Check caption_batch_low_yield flag from CAPTION stage
            # If set, many videos will need transcription after download
            caption_batch_low_yield = getattr(state, 'caption_batch_low_yield', False)
            if caption_batch_low_yield:
                logger.warning(
                    "US-59-011: caption_batch_low_yield=True - many videos in the batch had no "
                    "captions available. These videos will require transcription after download."
                )
                logger.warning("Many videos have no captions - transcription will be needed")

            # Get download settings from config
            download_config = config.download
            buffer_seconds = download_config.segment_buffer

            # US-50-012: Validate cookie configuration early (before download loop)
            self._validate_cookie_config(download_config)

            logger.info(f"Downloading matched segments (buffer: {buffer_seconds}s)")

            # Collect segments to download (US-48-008: merge overlapping/adjacent)
            # US-129-010: Pass download_config for segment validation
            segments_to_download = self._collect_matched_segments(state, buffer_seconds, download_config)

            if not segments_to_download:
                logger.warning("No valid segments to download")
                log_stage_skip(logger, "DOWNLOAD_SEGMENTS", reason="no_valid_segments")
                return StageResult.ok({'skipped': True, 'reason': 'no_valid_segments'}, warnings)

            logger.info(f"Total segments to download: {len(segments_to_download)}")

            # Estimate size: assume average 5MB per segment if duration unknown
            estimated_size_mb = len(segments_to_download) * 5
            log_stage_start(
                logger, "DOWNLOAD_SEGMENTS",
                total_segments=len(segments_to_download),
                estimated_size_mb=estimated_size_mb
            )

            # Apply test mode download limit if enabled
            test_mode_max_downloads = getattr(config, '_test_mode_max_downloads', None)
            if test_mode_max_downloads is not None and isinstance(test_mode_max_downloads, int) and len(segments_to_download) > test_mode_max_downloads:
                original_count = len(segments_to_download)
                segments_to_download = segments_to_download[:test_mode_max_downloads]
                logger.info(f"Test mode: limited segments_to_download from {original_count} to {test_mode_max_downloads}")

            # Initialize downloader via orchestrator (US-82-007)
            if self._orchestrator is None:
                from ..downloader.orchestrator import SegmentDownloadOrchestrator
                self._orchestrator = SegmentDownloadOrchestrator(config=config)
            self.downloader = self._orchestrator.downloader
            output_dir = Path(config.downloaded_videos_dir)

            # Restore impersonation state from checkpoint for cross-run learning
            if (self.downloader and getattr(self.downloader, 'impersonation_manager', None)
                    and hasattr(checkpoint, 'data') and checkpoint.data is not None):
                imp_state = getattr(checkpoint.data, 'impersonation_state', None)
                if imp_state:
                    self.downloader.impersonation_manager.restore_state(imp_state)
                    logger.info(
                        f"Restored impersonation state: "
                        f"{imp_state.get('stats', {}).get('calls_made', 0)} prior calls"
                    )

            # US-51-010: Restore retry queue from checkpoint on resume
            restored_rq_data = getattr(state, '_restored_retry_queue', None)
            if restored_rq_data and self.downloader.retry_queue:
                self.downloader.retry_queue.from_checkpoint_dict(restored_rq_data)
                pending_count = len(self.downloader.retry_queue.items)
                failed_count = len(self.downloader.retry_queue._failed_ids)
                if pending_count > 0:
                    logger.info(
                        f"Restored retry queue: {pending_count} videos to retry, "
                        f"{failed_count} permanently skipped"
                    )
                # Clean up temporary state attribute
                delattr(state, '_restored_retry_queue')

            # US-81-003: Shared mutable container for partial_progress tracking
            # Updated by _download_segments and read by checkpoint_progress
            _partial_progress = {
                'completed_ids': [],
                'failed_ids': [],
                'total_count': len(segments_to_download),
            }

            # US-81-003: Configurable checkpoint I/O frequency
            _checkpoint_every_n = int(getattr(
                download_config, 'segment_checkpoint_every_n', 10
            ))
            if _checkpoint_every_n <= 0:
                _checkpoint_every_n = 1
            _checkpoint_counter = [0]  # mutable for closure

            # US-81-004: Get progress reporter from state (set by pipeline)
            _progress_reporter = getattr(state, '_progress_reporter', None)
            if _progress_reporter:
                _progress_reporter.update(total=len(segments_to_download))

            # Download segments with progress callback for checkpointing
            def checkpoint_progress(current: int, total: int, downloaded: list):
                """Save progress checkpoint during download.

                US-81-003: Includes partial_progress dict with completed_ids,
                failed_ids, total_count for partial resume on interruption.
                Only writes to disk every N calls to reduce I/O overhead.
                Always writes on abort (current < total check in caller).
                """
                # US-81-004: Update centralized progress reporter
                if _progress_reporter:
                    prev_completed = getattr(checkpoint_progress, '_prev', 0)
                    delta = current - prev_completed
                    if delta > 0:
                        _progress_reporter.update(completed=delta)
                    checkpoint_progress._prev = current

                _checkpoint_counter[0] += 1
                # Write every N items, or on final item, or when aborting
                is_final = (current >= total)
                if _checkpoint_counter[0] < _checkpoint_every_n and not is_final:
                    return
                _checkpoint_counter[0] = 0

                checkpoint_data = {
                    'segment_count': len(downloaded),
                    'segments_completed': current,
                    'segments_total': total,
                    'in_progress': current < total,
                    # US-81-003: Partial progress for resume (deep copy lists)
                    'partial_progress': {
                        'completed_ids': list(_partial_progress['completed_ids']),
                        'failed_ids': list(_partial_progress['failed_ids']),
                        'total_count': _partial_progress['total_count'],
                    },
                }
                # US-51-010: Include retry queue state in intermediate checkpoints
                if self.downloader and self.downloader.retry_queue:
                    rq = self.downloader.retry_queue
                    if rq.items or rq._failed_ids:
                        checkpoint_data['retry_queue'] = rq.to_checkpoint_dict()
                # Save impersonation state for cross-run learning (top-level checkpoint field)
                if self.downloader and getattr(self.downloader, 'impersonation_manager', None):
                    if hasattr(checkpoint, 'data') and checkpoint.data is not None:
                        checkpoint.data.impersonation_state = self.downloader.impersonation_manager.to_dict()
                checkpoint.save_intermediate('DOWNLOAD_SEGMENTS', checkpoint_data)

            stage_start_time = time.time()

            # US-51-010: Process restored retry queue BEFORE new segments
            pre_retry_downloaded = []
            if (restored_rq_data and self.downloader and self.downloader.retry_queue
                    and self.downloader.retry_queue.has_pending()):
                logger.info("Processing restored retry queue before new segments")
                self._process_retry_queue(
                    output_dir, buffer_seconds, pre_retry_downloaded,
                    len(segments_to_download), checkpoint_progress
                )
                if pre_retry_downloaded:
                    logger.info(
                        f"Restored retry queue: {len(pre_retry_downloaded)} "
                        f"videos recovered from previous session"
                    )

            # US-81-009: Get batch failure threshold from config
            _batch_failure_threshold = getattr(
                config.pipeline, 'batch_failure_threshold', 0.5
            )
            _batch_failure_min_sample = getattr(
                config.pipeline, 'batch_failure_min_sample', 5
            )

            downloaded_segments, download_stats, throughput_samples = self._download_segments(
                segments_to_download,
                output_dir,
                buffer_seconds,
                checkpoint_progress,
                partial_progress=_partial_progress,
                batch_failure_threshold=_batch_failure_threshold,
                batch_failure_min_sample=_batch_failure_min_sample,
            )

            # US-51-010: Merge pre-retry downloads into main list
            if pre_retry_downloaded:
                downloaded_segments.extend(pre_retry_downloaded)

            elapsed = time.time() - stage_start_time

            # Print end-of-stage summary
            self._print_summary(download_stats, elapsed)

            # Update matches to reference local files
            self._update_matches_with_local_paths(state, downloaded_segments)

            # US-84-011: Propagate video_chapters/video_tags from VSR to DownloadedVideo
            self._propagate_vsr_metadata(state, downloaded_segments)

            # Store downloaded segments in state
            state.downloaded_segments = downloaded_segments

            checkpoint_data = {
                'segment_count': len(downloaded_segments),
                'total_matches': len(state.matches),
                'retry_count': download_stats.retry_count,
                # US-81-002: Per-item error details for batch error isolation
                'failed_items': download_stats.failed_items,
                # US-81-003: Clear partial_progress on full completion
                'partial_progress': None,
            }

            # US-50-008: Include circuit breaker metrics in checkpoint
            if self.downloader:
                _cb = getattr(self.downloader, 'circuit_breaker', None)
                if _cb:
                    checkpoint_data['circuit_breaker'] = {
                        'total_trips': _cb.state.total_trips,
                        'total_paused_seconds': round(_cb.state.total_paused_seconds, 1),
                    }

            # US-49-012: Collect escalation summary from escalation manager
            escalation_summary = {}
            if self.downloader:
                _esc_mgr = getattr(self.downloader, 'escalation_manager', None)
                if _esc_mgr and hasattr(_esc_mgr, 'get_metrics'):
                    try:
                        esc_metrics = _esc_mgr.get_metrics()
                        # Build serializable summary: videos per tier, totals, effectiveness
                        escalation_summary = {
                            'total_escalations': esc_metrics.get('total_escalations', 0),
                            'videos_per_tier': {
                                tier: len(keywords)
                                for tier, keywords in esc_metrics.get('keywords_at_each_tier', {}).items()
                            },
                            'escalations_per_tier': esc_metrics.get('escalations_per_tier', {}),
                            'total_403s': esc_metrics.get('total_403s', 0),
                            'total_successes': esc_metrics.get('total_successes', 0),
                            'average_tier': esc_metrics.get('average_tier', 1.0),
                        }
                        # Include tier effectiveness if available
                        if hasattr(_esc_mgr, 'get_tier_effectiveness'):
                            escalation_summary['tier_effectiveness'] = _esc_mgr.get_tier_effectiveness()
                    except Exception as esc_err:
                        logger.debug(f"Could not collect escalation summary: {esc_err}")

            # US-52-011: Add tier_distribution from escalation_mgr.keyword_states
            # Shows count of downloads at each escalation tier for diagnosing
            # whether YouTube is broadly blocking (all Tier 3+) vs isolated failures.
            if self.downloader:
                _esc_mgr_td = getattr(self.downloader, 'escalation_manager', None)
                if _esc_mgr_td and hasattr(_esc_mgr_td, 'keyword_states'):
                    try:
                        tier_distribution: Dict[str, int] = {}
                        for _kw, _state in _esc_mgr_td.keyword_states.items():
                            tier_name = _state.current_tier.name
                            tier_distribution[tier_name] = tier_distribution.get(tier_name, 0) + 1
                        escalation_summary['tier_distribution'] = tier_distribution
                    except Exception as td_err:
                        logger.debug(f"Could not collect tier distribution: {td_err}")

            # US-50-009: Add bot_detection_count and network_failure_count from error categories
            escalation_summary['bot_detection_count'] = download_stats.error_categories.get('bot_detection', 0)
            escalation_summary['network_failure_count'] = download_stats.error_categories.get('network', 0)

            # US-50-009: Log structured escalation summary at stage completion
            self._log_escalation_summary(escalation_summary)

            # US-51-006: Include progress hook data in escalation_summary for checkpoint
            if download_stats.progress_hooks_data:
                escalation_summary['download_progress'] = download_stats.progress_hooks_data

            # US-51-010: Persist retry queue to checkpoint for resume
            if self.downloader and self.downloader.retry_queue:
                rq = self.downloader.retry_queue
                if rq.items or rq._failed_ids:
                    checkpoint_data['retry_queue'] = rq.to_checkpoint_dict()

            # Stage metrics for pipeline observability (US-49-009 + US-49-012 + US-81-007)
            metrics = StageMetrics(
                items_processed=download_stats.succeeded + download_stats.cached,
                items_failed=download_stats.failed,
                duration_seconds=elapsed,
                error_categories=download_stats.error_categories,
                escalation_summary=escalation_summary,
                throughput_samples=throughput_samples,
            )
            metrics.compute_throughput()

            # Fail the stage if we had segments to download but got none.
            # This prevents the pipeline from proceeding to OUTPUT with an
            # empty V1 track — previously this silently succeeded and produced
            # timelines with only stock/entity fallback tracks.
            if (len(downloaded_segments) == 0
                    and len(segments_to_download) > 0):
                fail_msg = (
                    f"All segment downloads failed: 0/{len(segments_to_download)} "
                    f"segments downloaded, {download_stats.failed} failures. "
                    f"Check yt-dlp/ffmpeg errors in logs."
                )
                logger.error(f"[DOWNLOAD_SEGMENTS] {fail_msg}")
                return StageResult.fail(fail_msg, warnings, metrics)

            return StageResult.ok(checkpoint_data, warnings, metrics)

        except Exception as e:
            # Add context about what was being processed when error occurred
            total_matches = len(state.matches) if state and getattr(state, 'matches', None) else 0
            segments_count = len(segments_to_download) if 'segments_to_download' in dir() and segments_to_download else 0
            log_error_with_context(
                logger, "DL-001", f"Video segment download failed: {e}",
                total_matches=total_matches, segments_to_download=segments_count
            )
            return StageResult.fail(str(e), warnings)

    def _collect_matched_segments(
        self, state: 'PipelineState', buffer_seconds: float = 5.0,
        download_config: Any = None
    ) -> List[Dict[str, Any]]:
        """Collect segment info from matches for downloading.

        US-48-008: Uses exact float values for dedup keys (not round()) to
        preserve precision for segments differing by <0.5s. Also merges
        overlapping/adjacent segments from the same video to reduce downloads.

        US-129-008: Includes segment_index to enable sorting by voiceover
        segment timeline for priority-based downloading.

        US-129-010: Validates segments before adding to queue:
        - Duration >= min_duration_seconds (configurable)
        - Duration <= max_duration_seconds (if configured)
        - Segment times within video duration bounds
        - Filter duplicate segments
        """
        # Check if download_all_tracks is enabled
        download_all_tracks = False
        if download_config:
            download_all_tracks = getattr(download_config, 'download_all_tracks', False)

        # US-129-010: Get segment validation config
        seg_validation = None
        if download_config:
            seg_validation = getattr(download_config, 'segment_validation', None)

        # Build video duration lookup from video_search_results
        video_durations: Dict[str, float] = {}
        if hasattr(state, 'video_search_results') and state.video_search_results:
            for vsr in state.video_search_results:
                if hasattr(vsr, 'video_id') and hasattr(vsr, 'duration'):
                    if vsr.video_id and vsr.duration > 0:
                        video_durations[vsr.video_id] = vsr.duration

        # US-129-010: Track validation stats
        validation_stats = {
            'too_short': 0,
            'too_long': 0,
            'out_of_bounds': 0,
            'duplicate': 0,
        }

        raw_segments = []

        for match in state.matches:
            # US-129-008: Get segment_index for voiceover timeline ordering
            segment_index = getattr(match, 'segment_index', 0)

            # Handle MatchResult structure (has primary_match)
            if hasattr(match, 'primary_match') and match.primary_match:
                pm = match.primary_match
                if hasattr(pm, 'video_segment') and pm.video_segment:
                    video_id = getattr(pm.video_segment, 'source_file', '')
                    start_time = getattr(pm.video_segment, 'start_time', 0.0)
                    end_time = getattr(pm.video_segment, 'end_time', start_time + 10.0)
                    # US-114-002: Get confidence from primary_match
                    confidence = getattr(pm, 'confidence', 0.5)
                else:
                    continue
            # Handle plain Match structure
            elif hasattr(match, 'video_file'):
                video_id = match.video_file
                start_time = getattr(match, 'video_start', 0.0)
                end_time = getattr(match, 'video_end', start_time + 10.0)
                # US-114-002: Get confidence from match
                confidence = getattr(match, 'confidence', 0.5)
            else:
                continue

            # Skip if no video ID
            if not video_id:
                continue

            # US-114-002: Calculate duration tier from segment duration
            duration = end_time - start_time
            if duration <= 30:
                duration_tier = "short"
            elif duration <= 90:
                duration_tier = "medium"
            elif duration <= 300:
                duration_tier = "long"
            else:
                duration_tier = "longer"

            # US-129-010: Validate segment before adding to queue
            if seg_validation:
                # Validate duration
                min_dur = getattr(seg_validation, 'min_duration_seconds', 1.0)
                max_dur = getattr(seg_validation, 'max_duration_seconds', 0.0)
                validate_bounds = getattr(seg_validation, 'validate_bounds', True)
                log_skipped = getattr(seg_validation, 'log_skipped', True)
                strictness = getattr(seg_validation, 'strictness', 'lenient')

                # Check min duration
                if min_dur > 0 and duration < min_dur:
                    validation_stats['too_short'] += 1
                    if log_skipped:
                        logger.debug(f"Segment {video_id}[{start_time:.1f}-{end_time:.1f}] skipped: too_short ({duration:.1f}s < {min_dur}s)")
                    if strictness == 'strict':
                        raise ValueError(f"Segment validation failed: duration {duration:.1f}s < min {min_dur}s")
                    continue

                # Check max duration
                if max_dur > 0 and duration > max_dur:
                    validation_stats['too_long'] += 1
                    if log_skipped:
                        logger.debug(f"Segment {video_id}[{start_time:.1f}-{end_time:.1f}] skipped: too_long ({duration:.1f}s > {max_dur}s)")
                    if strictness == 'strict':
                        raise ValueError(f"Segment validation failed: duration {duration:.1f}s > max {max_dur}s")
                    continue

                # Check bounds (start/end within video duration)
                if validate_bounds and video_id in video_durations:
                    video_duration = video_durations[video_id]
                    if start_time < 0 or end_time > video_duration:
                        validation_stats['out_of_bounds'] += 1
                        if log_skipped:
                            logger.debug(f"Segment {video_id}[{start_time:.1f}-{end_time:.1f}] skipped: out_of_bounds (video duration: {video_duration:.1f}s)")
                        if strictness == 'strict':
                            raise ValueError(f"Segment validation failed: segment [{start_time:.1f}-{end_time:.1f}] exceeds video duration {video_duration:.1f}s")
                        continue

            raw_segments.append({
                'video_id': video_id,
                'start': start_time,
                'end': end_time,
                'confidence': confidence,
                'duration_tier': duration_tier,
                'segment_index': segment_index,  # US-129-008: For voiceover timeline ordering
            })

            # US-XXX: Collect alternatives, secondary, and strategy matches when download_all_tracks is enabled
            if download_all_tracks:
                # Helper: extract segment info from either live AlternativeMatch/StrategyMatch
                # objects or restored checkpoint dicts (from Match._*_data fields)
                def _extract_track_segments(items):
                    for item in (items or []):
                        if hasattr(item, 'video_segment') and item.video_segment:
                            # Live object (AlternativeMatch/StrategyMatch)
                            vid = getattr(item.video_segment, 'source_file', '')
                            s = getattr(item.video_segment, 'start_time', 0.0)
                            e = getattr(item.video_segment, 'end_time', s + 10.0)
                            c = getattr(item, 'confidence', 0.5)
                        elif isinstance(item, dict):
                            # Restored from checkpoint dict
                            vs = item.get('video_segment', {})
                            vid = vs.get('source_file', '') if isinstance(vs, dict) else ''
                            s = float(vs.get('start_time', 0.0)) if isinstance(vs, dict) else 0.0
                            e = float(vs.get('end_time', s + 10.0)) if isinstance(vs, dict) else s + 10.0
                            c = float(item.get('confidence', 0.5))
                        else:
                            continue
                        if vid:
                            raw_segments.append({
                                'video_id': vid,
                                'start': s,
                                'end': e,
                                'confidence': c,
                                'duration_tier': duration_tier,
                                'segment_index': segment_index,
                            })

                # Collect from live MatchResult fields or restored Match._*_data fields
                _extract_track_segments(
                    getattr(match, 'alternatives', None) or getattr(match, '_alternatives_data', [])
                )
                _extract_track_segments(
                    getattr(match, 'secondary_matches', None) or getattr(match, '_secondary_matches_data', [])
                )
                _extract_track_segments(
                    getattr(match, 'strategy_matches', None) or getattr(match, '_strategy_matches_data', [])
                )

        # Deduplicate exact matches using (video_id, start, end) tuple
        # US-129-010: Track duplicates if validation enabled
        seen = set()
        deduped = []
        filter_dups = False
        if seg_validation:
            filter_dups = getattr(seg_validation, 'filter_duplicates', True)

        for seg in raw_segments:
            key = (seg['video_id'], seg['start'], seg['end'])
            if key not in seen:
                seen.add(key)
                deduped.append(seg)
            elif filter_dups:
                # This is a duplicate
                validation_stats['duplicate'] += 1
                log_skipped = getattr(seg_validation, 'log_skipped', True)
                if log_skipped:
                    logger.debug(f"Segment {seg['video_id']}[{seg['start']:.1f}-{seg['end']:.1f}] skipped: duplicate")

        # US-129-010: Log validation summary
        if seg_validation and getattr(seg_validation, 'log_skipped', True):
            total_skipped = sum(validation_stats.values())
            if total_skipped > 0:
                logger.info(f"Segment validation: {total_skipped} segments skipped "
                           f"(too_short={validation_stats['too_short']}, "
                           f"too_long={validation_stats['too_long']}, "
                           f"out_of_bounds={validation_stats['out_of_bounds']}, "
                           f"duplicate={validation_stats['duplicate']})")

        # Merge overlapping/adjacent segments from the same video
        return self._merge_segments(deduped, buffer_seconds)

    @staticmethod
    def _merge_segments(
        segments: List[Dict[str, Any]], buffer_seconds: float
    ) -> List[Dict[str, Any]]:
        """Merge overlapping or adjacent segments from the same video.

        Two segments from the same video are merged if they overlap or
        the gap between them is less than 2 * buffer_seconds (since both
        would have buffer applied, their downloaded ranges would overlap).

        Args:
            segments: Deduplicated segment list.
            buffer_seconds: Per-segment buffer (used to compute merge threshold).

        Returns:
            Merged segment list.
        """
        if not segments:
            return []

        # Group by video_id
        by_video: Dict[str, List[Dict[str, Any]]] = {}
        for seg in segments:
            by_video.setdefault(seg['video_id'], []).append(seg)

        merged = []
        merge_gap = 2 * buffer_seconds

        for video_id, segs in by_video.items():
            # Sort by start time
            segs.sort(key=lambda s: s['start'])

            current = dict(segs[0])  # copy first segment
            for seg in segs[1:]:
                # Merge if overlapping or gap < 2*buffer
                if seg['start'] <= current['end'] + merge_gap:
                    current['end'] = max(current['end'], seg['end'])
                else:
                    merged.append(current)
                    current = dict(seg)
            merged.append(current)

        return merged

    def _download_segments(
        self,
        segments: List[Dict[str, Any]],
        output_dir: Path,
        buffer_seconds: float,
        progress_callback,
        download_progress_callback: Optional[Any] = None,
        partial_progress: Optional[Dict[str, Any]] = None,
        batch_failure_threshold: float = 1.0,
        batch_failure_min_sample: int = 5,
    ):
        """Download video segments via sub-methods: prepare, check, execute, handle.

        US-57-007: Returns (downloaded_segments list, SegmentDownloadStats).
        US-81-003: Tracks partial_progress (completed_ids, failed_ids, total_count)
        and saves incremental checkpoint every N downloads for partial resume.
        US-89-010: Added download_progress_callback for detailed progress (bytes/speed/ETA).

        Args:
            download_progress_callback: Optional callback(s) for detailed download progress.
                Can be a single DownloadProgressCallback, a list of callbacks, or a MultiCallback.
                Receives on_start, on_progress, on_complete, on_error events.
            partial_progress: Mutable dict shared with checkpoint callback.
                Updated in-place with completed_ids, failed_ids, total_count.
            batch_failure_threshold: US-81-009: Max failure rate before abort (1.0 = disabled).
            batch_failure_min_sample: Minimum items processed before threshold check activates.
        """
        from ..state import DownloadedVideo

        # US-89-010: Convert download_progress_callback to MultiCallback
        progress_callbacks: Optional[MultiCallback] = None
        if download_progress_callback is not None:
            if isinstance(download_progress_callback, MultiCallback):
                progress_callbacks = download_progress_callback
            elif isinstance(download_progress_callback, list):
                # List of callbacks - filter to valid ones and create MultiCallback
                valid_cbs = [cb for cb in download_progress_callback if cb]
                if valid_cbs:
                    progress_callbacks = MultiCallback(valid_cbs)
            else:
                # Single callback - wrap in list then MultiCallback
                progress_callbacks = MultiCallback([download_progress_callback])

        # Create download context early (needed for priority_boost config access)
        total = len(segments)
        stats = SegmentDownloadStats(total=total)
        ctx = self._prepare_download_context(stats)

        # US-129-008: Sort segments by voiceover segment timeline for priority downloading
        # Earlier voiceover segments get higher priority to enable faster iterative match feedback
        dl_cfg = ctx.download_config
        priority_boost = float(getattr(dl_cfg, 'priority_boost_for_early_segments', 1.5)) if dl_cfg else 1.5
        if priority_boost > 1.0 and segments:
            # Sort by segment_index (voiceover timeline order)
            segments = sorted(segments, key=lambda s: s.get('segment_index', 0))
            # Calculate priority for each segment (lower index = higher priority)
            max_index = max(s.get('segment_index', 0) for s in segments) or 1
            for seg in segments:
                idx = seg.get('segment_index', 0)
                # Priority boost: earlier segments get boosted priority
                # priority = 1.0 + boost * (1 - idx/max_index)
                seg['priority_score'] = 1.0 + priority_boost * (1.0 - idx / max_index)
            logger.debug(f"US-129-008: Sorted {len(segments)} segments by voiceover timeline (priority_boost={priority_boost})")

        downloaded = []
        # US-81-007: Track per-item throughput samples
        _throughput_samples: List[float] = []

        # US-81-003: Track completed/failed IDs via shared partial_progress dict
        if partial_progress is None:
            partial_progress = {'completed_ids': [], 'failed_ids': [], 'total_count': total}

        # Concurrent segment downloads: dispatch to threaded implementation
        num_workers = int(getattr(dl_cfg, 'segment_concurrent_workers', 1)) if dl_cfg else 1
        cookie_files = []
        if dl_cfg:
            _cookie_rotation = getattr(dl_cfg, 'cookie_rotation', None)
            if _cookie_rotation:
                cookie_files = list(getattr(_cookie_rotation, 'cookie_files', []) or [])
        if num_workers > 1 and len(cookie_files) >= num_workers:
            logger.info(
                f"Concurrent segment download: {num_workers} workers, "
                f"{len(cookie_files)} cookie files"
            )
            return self._download_segments_concurrent(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=buffer_seconds,
                progress_callback=progress_callback,
                progress_callbacks=progress_callbacks,
                partial_progress=partial_progress,
                batch_failure_threshold=batch_failure_threshold,
                batch_failure_min_sample=batch_failure_min_sample,
                ctx=ctx,
                stats=stats,
                downloaded=downloaded,
                _throughput_samples=_throughput_samples,
                num_workers=num_workers,
                cookie_files=cookie_files,
            )
        elif num_workers > 1:
            logger.warning(
                f"Concurrent workers={num_workers} requested but only "
                f"{len(cookie_files)} cookie files available. Falling back to sequential."
            )

        # Adaptive request delay to avoid YouTube rate-limiting
        dl_cfg = ctx.download_config
        base_delay = float(getattr(dl_cfg, 'segment_request_delay', 1.0)) if dl_cfg else 1.0
        max_delay = float(getattr(dl_cfg, 'segment_request_delay_max', 30.0)) if dl_cfg else 30.0
        # US-85-008: Jitter factor to prevent thundering herd
        jitter_factor = float(getattr(dl_cfg, 'segment_request_delay_jitter', 0.25)) if dl_cfg else 0.25
        current_delay = base_delay
        _did_network_request = False

        for idx, seg in enumerate(segments, 1):
            _item_start = time.monotonic()  # US-81-007: per-item timing
            # Sleep between network requests (skip before first, skip after cache hits)
            # US-85-008: Apply random jitter to prevent synchronized request bursts
            if _did_network_request and current_delay > 0:
                if jitter_factor > 0:
                    jittered_delay = current_delay * random.uniform(1 - jitter_factor, 1 + jitter_factor)
                else:
                    jittered_delay = current_delay
                time.sleep(jittered_delay)
            _did_network_request = False

            video_id = seg['video_id']
            start = max(0, seg['start'] - buffer_seconds)
            end = seg['end'] + buffer_seconds
            seg_key = f"{video_id}_{int(start)}_{int(end)}.mp4"
            output_file = output_dir / seg_key

            # Cache hit — no network needed
            if output_file.exists():
                logger.info(f"Segment already exists: {output_file}")
                downloaded.append(DownloadedVideo(
                    file=str(output_file),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    source='segment_cache'
                ))
                try:
                    _cached_bytes = output_file.stat().st_size
                except OSError:
                    _cached_bytes = 0
                stats.increment_cached(file_bytes=_cached_bytes)
                partial_progress['completed_ids'].append(seg_key)
                ctx.consecutive_network_failures = 0
                if ctx.consecutive_bot_detections > 0:
                    ctx.consecutive_bot_detections = 0
                    if ctx.escalation_mgr:
                        ctx.escalation_mgr.clear_tier_floor()
                # US-81-007: Record throughput for cache hit
                _cache_elapsed = time.monotonic() - _item_start
                if _cache_elapsed > 0:
                    _throughput_samples.append(1.0 / _cache_elapsed)
                self._print_progress(idx, total, stats)
                if progress_callback:
                    progress_callback(idx, total, downloaded)
                continue

            # Check preconditions (circuit breaker, etc.)
            skip_reason = self._check_preconditions(ctx, video_id, start, end, output_file)
            if skip_reason:
                stats.increment_failure(
                    category='precondition',
                    error_msg=skip_reason,
                    video_id=video_id,
                )
                partial_progress['failed_ids'].append(seg_key)
                self._print_progress(idx, total, stats)
                if progress_callback:
                    progress_callback(idx, total, downloaded)
                continue

            # US-129-008: Log priority info for early segments
            priority_score = seg.get('priority_score', 1.0)
            segment_index = seg.get('segment_index', 0)
            if priority_score > 1.0:
                logger.debug(
                    f"Downloading segment {idx}/{total}: {video_id} "
                    f"(voiceover_idx={segment_index}, priority={priority_score:.2f})"
                )

            # US-164-008: Log segment download start with video_id and time_range
            logger.info(
                f"[DOWNLOAD_SEGMENTS] Starting download: video_id={video_id}, "
                f"time_range=({start:.1f}, {end:.1f}), segment={idx}/{total}"
            )

            # Execute the download
            _did_network_request = True
            _checksum_retries = 0
            _max_checksum_retries = 2

            # Get checksum config for retry settings
            dl_cfg = ctx.download_config
            if dl_cfg:
                checksum_cfg = getattr(dl_cfg, 'checksum_validation', None)
                if checksum_cfg:
                    _max_checksum_retries = getattr(checksum_cfg, 'max_retries', 2)
            _download_successful = False

            while not _download_successful:
                result = self._execute_download(
                    ctx, video_id, start, end, output_file, progress_callbacks
                )

                # US-143-012: Validate checksum after successful download
                if result.get('success'):
                    validation_result = self._validate_checksum(
                        output_file,
                        expected_checksum=None,  # yt-dlp doesn't provide expected checksum
                        expected_size=None,       # Could get from result if available
                    )

                    # Log validation results
                    dl_cfg = ctx.download_config
                    if dl_cfg:
                        checksum_cfg = getattr(dl_cfg, 'checksum_validation', None)
                        log_to_metrics = getattr(checksum_cfg, 'log_to_metrics', True) if checksum_cfg else True
                        if log_to_metrics:
                            # Log to stats
                            stats.error_aggregator.record(
                                f"checksum_valid={validation_result['valid']}",
                                'checksum_validation'
                            )

                    if not validation_result['valid']:
                        # Checksum validation failed - retry if enabled
                        retry_on_failure = True
                        if dl_cfg:
                            checksum_cfg = getattr(dl_cfg, 'checksum_validation', None)
                            if checksum_cfg:
                                retry_on_failure = getattr(checksum_cfg, 'retry_on_failure', True)

                        if retry_on_failure and _checksum_retries < _max_checksum_retries:
                            _checksum_retries += 1
                            logger.warning(
                                f"Checksum validation failed for {output_file}, "
                                f"retrying ({_checksum_retries}/{_max_checksum_retries}): "
                                f"{validation_result.get('error_msg')}"
                            )
                            # Delete corrupted file
                            if output_file.exists():
                                try:
                                    output_file.unlink()
                                except OSError:
                                    pass
                            continue  # Retry the download
                        else:
                            # Mark as failed due to checksum
                            result['success'] = False
                            result['error_msg'] = f"Checksum validation failed: {validation_result.get('error_msg')}"
                            log_error_with_context(logger, "DL-003", f"Checksum validation failed for {output_file}: {validation_result.get('error_msg')}", video_id=video_id, file_path=output_file)

                _download_successful = True

            # US-114-002: Add segment value data to result for retry prioritization
            result['duration_tier'] = seg.get('duration_tier', '')
            result['match_confidence'] = seg.get('confidence', 0.0)

            # Handle the result (success, failure, abort signals)
            abort = self._handle_result(
                ctx, result, video_id, start, end, output_file,
                downloaded, idx, total, progress_callback,
            )
            # US-81-003: Track outcome
            if result.get('success'):
                partial_progress['completed_ids'].append(seg_key)
            else:
                partial_progress['failed_ids'].append(seg_key)
            self._print_progress(idx, total, stats)

            # US-81-007: Record per-item throughput sample
            _item_elapsed = time.monotonic() - _item_start
            if _item_elapsed > 0:
                _throughput_samples.append(1.0 / _item_elapsed)

            # Adaptive delay: back off on failure, reset on success
            if result.get('success'):
                current_delay = base_delay
            else:
                current_delay = min(current_delay * 2, max_delay)

            if abort:
                break

            # US-81-009: Check batch failure threshold after each item
            items_done = stats.succeeded + stats.cached + stats.failed
            if items_done > 0 and batch_failure_threshold < 1.0:
                from . import check_batch_failure_threshold, BatchFailureThresholdExceeded
                try:
                    check_batch_failure_threshold(
                        items_processed=items_done,
                        items_failed=stats.failed,
                        threshold=batch_failure_threshold,
                        failed_items=partial_progress.get('failed_ids', []),
                        min_sample_size=batch_failure_min_sample,
                    )
                except BatchFailureThresholdExceeded as e:
                    failed_ids = partial_progress.get('failed_ids', [])
                    log_error_with_context(
                        logger, "DL-001", f"Batch failure threshold exceeded: {e}",
                        items_processed=items_done, items_failed=stats.failed,
                        threshold=batch_failure_threshold, failed_ids=failed_ids
                    )
                    break

            # Checkpoint progress
            if progress_callback:
                progress_callback(idx, total, downloaded)

        # Process retry queue if there are pending items
        self._process_retry_queue(output_dir, buffer_seconds, downloaded, total, progress_callback, stats)

        # US-49-009: Log structured error summary with actionable diagnostics
        self._log_error_summary(stats)

        return downloaded, stats, _throughput_samples

    def _download_segments_concurrent(
        self,
        segments: List[Dict[str, Any]],
        output_dir: Path,
        buffer_seconds: float,
        progress_callback,
        progress_callbacks: Optional[Any],
        partial_progress: Dict[str, Any],
        batch_failure_threshold: float,
        batch_failure_min_sample: int,
        ctx: _DownloadLoopContext,
        stats: SegmentDownloadStats,
        downloaded: list,
        _throughput_samples: List[float],
        num_workers: int,
        cookie_files: List[str],
    ):
        """Download segments concurrently using a thread pool.

        Each worker is assigned a dedicated cookie file (round-robin) so that
        YouTube sees distinct sessions, allowing full bandwidth utilisation
        across multiple streams.
        """
        from ..state import DownloadedVideo

        total = len(segments)
        dl_cfg = ctx.download_config
        base_delay = float(getattr(dl_cfg, 'segment_request_delay', 1.0)) if dl_cfg else 1.0
        max_delay = float(getattr(dl_cfg, 'segment_request_delay_max', 30.0)) if dl_cfg else 30.0
        jitter_factor = float(getattr(dl_cfg, 'segment_request_delay_jitter', 0.25)) if dl_cfg else 0.25

        # Thread-safety: protect shared mutable state
        _lock = threading.Lock()
        _abort = threading.Event()
        _completed_count = [0]  # mutable counter for progress

        def _process_segment(seg: Dict[str, Any], worker_cookie: str) -> Optional[Dict[str, Any]]:
            """Download one segment using the worker's dedicated cookie.

            Returns a result dict or None if skipped (cache hit / precondition).
            """
            if _abort.is_set():
                return None

            _item_start = time.monotonic()
            video_id = seg['video_id']
            start = max(0, seg['start'] - buffer_seconds)
            end = seg['end'] + buffer_seconds
            seg_key = f"{video_id}_{int(start)}_{int(end)}.mp4"
            output_file = output_dir / seg_key

            # Cache hit — no network needed
            if output_file.exists():
                try:
                    _cached_bytes = output_file.stat().st_size
                except OSError:
                    _cached_bytes = 0
                with _lock:
                    downloaded.append(DownloadedVideo(
                        file=str(output_file),
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        source='segment_cache',
                    ))
                    stats.increment_cached(file_bytes=_cached_bytes)
                    partial_progress['completed_ids'].append(seg_key)
                    ctx.consecutive_network_failures = 0
                    _completed_count[0] += 1
                    idx = _completed_count[0]
                    _cache_elapsed = time.monotonic() - _item_start
                    if _cache_elapsed > 0:
                        _throughput_samples.append(1.0 / _cache_elapsed)
                    self._print_progress(idx, total, stats)
                    if progress_callback:
                        progress_callback(idx, total, downloaded)
                return None

            # Precondition check
            skip_reason = self._check_preconditions(ctx, video_id, start, end, output_file)
            if skip_reason:
                with _lock:
                    stats.increment_failure(
                        category='precondition', error_msg=skip_reason, video_id=video_id,
                    )
                    partial_progress['failed_ids'].append(seg_key)
                    _completed_count[0] += 1
                    idx = _completed_count[0]
                    self._print_progress(idx, total, stats)
                    if progress_callback:
                        progress_callback(idx, total, downloaded)
                return None

            logger.info(
                f"[DOWNLOAD_SEGMENTS] Starting download: video_id={video_id}, "
                f"time_range=({start:.1f}, {end:.1f}), cookie={worker_cookie}"
            )

            # Execute download with this worker's cookie
            _checksum_retries = 0
            _max_checksum_retries = 2
            if dl_cfg:
                checksum_cfg = getattr(dl_cfg, 'checksum_validation', None)
                if checksum_cfg:
                    _max_checksum_retries = getattr(checksum_cfg, 'max_retries', 2)
            _download_successful = False

            while not _download_successful and not _abort.is_set():
                result = self._execute_download(
                    ctx, video_id, start, end, output_file,
                    progress_callbacks, cookie_file_override=worker_cookie,
                )

                if result.get('success'):
                    validation_result = self._validate_checksum(
                        output_file, expected_checksum=None, expected_size=None,
                    )
                    if not validation_result['valid']:
                        retry_on_failure = True
                        if dl_cfg:
                            checksum_cfg = getattr(dl_cfg, 'checksum_validation', None)
                            if checksum_cfg:
                                retry_on_failure = getattr(checksum_cfg, 'retry_on_failure', True)
                        if retry_on_failure and _checksum_retries < _max_checksum_retries:
                            _checksum_retries += 1
                            logger.warning(
                                f"Checksum validation failed for {output_file}, "
                                f"retrying ({_checksum_retries}/{_max_checksum_retries})"
                            )
                            if output_file.exists():
                                try:
                                    output_file.unlink()
                                except OSError:
                                    pass
                            continue
                        else:
                            result['success'] = False
                            result['error_msg'] = f"Checksum validation failed: {validation_result.get('error_msg')}"

                _download_successful = True

            if _abort.is_set():
                return None

            result['duration_tier'] = seg.get('duration_tier', '')
            result['match_confidence'] = seg.get('confidence', 0.0)

            # Handle result with lock for shared state
            with _lock:
                _completed_count[0] += 1
                idx = _completed_count[0]
                abort = self._handle_result(
                    ctx, result, video_id, start, end, output_file,
                    downloaded, idx, total, progress_callback,
                )
                if result.get('success'):
                    partial_progress['completed_ids'].append(seg_key)
                else:
                    partial_progress['failed_ids'].append(seg_key)
                self._print_progress(idx, total, stats)

                _item_elapsed = time.monotonic() - _item_start
                if _item_elapsed > 0:
                    _throughput_samples.append(1.0 / _item_elapsed)

                # Check batch failure threshold
                items_done = stats.succeeded + stats.cached + stats.failed
                if items_done > 0 and batch_failure_threshold < 1.0:
                    from . import check_batch_failure_threshold, BatchFailureThresholdExceeded
                    try:
                        check_batch_failure_threshold(
                            items_processed=items_done,
                            items_failed=stats.failed,
                            threshold=batch_failure_threshold,
                            failed_items=partial_progress.get('failed_ids', []),
                            min_sample_size=batch_failure_min_sample,
                        )
                    except BatchFailureThresholdExceeded as e:
                        log_error_with_context(
                            logger, "DL-001", f"Batch failure threshold exceeded: {e}",
                            items_processed=items_done, items_failed=stats.failed,
                        )
                        _abort.set()

                if abort:
                    _abort.set()

                if progress_callback:
                    progress_callback(idx, total, downloaded)

            return result

        # Assign cookie files to workers round-robin and submit
        # Filter segments that aren't already cached for fair distribution
        work_items = []
        for seg in segments:
            video_id = seg['video_id']
            start = max(0, seg['start'] - buffer_seconds)
            end = seg['end'] + buffer_seconds
            seg_key = f"{video_id}_{int(start)}_{int(end)}.mp4"
            output_file = output_dir / seg_key
            work_items.append((seg, output_file.exists()))

        # Process cache hits first (fast, no network), then distribute network work
        cache_hits = [seg for seg, cached in work_items if cached]
        network_items = [seg for seg, cached in work_items if not cached]

        logger.info(
            f"Concurrent download plan: {len(cache_hits)} cached, "
            f"{len(network_items)} to download across {num_workers} workers"
        )

        # Process cache hits sequentially (fast, no contention)
        for seg in cache_hits:
            _process_segment(seg, cookie_files[0])

        # Download remaining segments concurrently with per-worker cookies
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix='seg_dl',
        ) as pool:
            futures = []
            for i, seg in enumerate(network_items):
                worker_cookie = cookie_files[i % num_workers]
                futures.append(pool.submit(_process_segment, seg, worker_cookie))

            # Wait for all to complete (results already handled inside _process_segment)
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"Segment download worker exception: {exc}")

        # Process retry queue
        self._process_retry_queue(output_dir, buffer_seconds, downloaded, total, progress_callback, stats)
        self._log_error_summary(stats)

        return downloaded, stats, _throughput_samples

    def _prepare_download_context(
        self, stats: SegmentDownloadStats
    ) -> _DownloadLoopContext:
        """Initialise shared loop state from downloader and config.

        US-57-007: Extracts the one-time setup that previously sat at the
        top of _download_segments into its own method.

        Returns:
            A populated _DownloadLoopContext.
        """
        # US-82-010: Prefer constructor-injected deps, fall back to downloader attrs
        escalation_mgr = self._injected_escalation_manager
        circuit_breaker = self._injected_circuit_breaker
        cookie_rotator = None
        if self.downloader:
            if escalation_mgr is None:
                escalation_mgr = getattr(self.downloader, 'escalation_manager', None)
            cookie_rotator = getattr(self.downloader, 'cookie_rotator', None)
            if circuit_breaker is None:
                circuit_breaker = getattr(self.downloader, 'circuit_breaker', None)

        # US-49-007: Wire circuit breaker into escalation manager
        if escalation_mgr and circuit_breaker:
            escalation_mgr.set_circuit_breaker(circuit_breaker)

        dl_cfg = getattr(self.downloader, 'download_config', None) if self.downloader else None

        bot_floor_threshold = 5
        bot_abort_threshold = BOT_DETECTION_ABORT_THRESHOLD
        network_failure_threshold = NETWORK_FAILURE_THRESHOLD
        if dl_cfg:
            bot_floor_threshold = int(getattr(dl_cfg, 'bot_detection_tier_floor_threshold', 5))
            bot_abort_threshold = int(getattr(
                dl_cfg, 'bot_detection_abort_threshold', BOT_DETECTION_ABORT_THRESHOLD
            ))
            network_failure_threshold = int(getattr(
                dl_cfg, 'network_failure_threshold', NETWORK_FAILURE_THRESHOLD
            ))

        return _DownloadLoopContext(
            stats=stats,
            escalation_mgr=escalation_mgr,
            cookie_rotator=cookie_rotator,
            circuit_breaker=circuit_breaker,
            download_config=dl_cfg,
            bot_floor_threshold=bot_floor_threshold,
            bot_abort_threshold=bot_abort_threshold,
            network_failure_threshold=network_failure_threshold,
        )

    def _check_preconditions(
        self,
        ctx: _DownloadLoopContext,
        video_id: str,
        start: float,
        end: float,
        output_file: Path,
    ) -> Optional[str]:
        """Check whether the download should be skipped or paused.

        US-57-007: Extracts circuit-breaker logic from the main loop.

        Returns:
            A skip reason string if the segment should be skipped,
            or ``None`` if the download may proceed.
        """
        cb = ctx.circuit_breaker
        if not cb or not cb.is_open:
            return None

        at_max_tier = False
        if ctx.escalation_mgr:
            from ..downloader.types import EscalationTier
            kw_state = ctx.escalation_mgr._get_state(video_id)
            at_max_tier = kw_state.current_tier >= EscalationTier.VPN_ROTATION

        if at_max_tier:
            logger.info(
                f"Circuit breaker open + max tier reached for {video_id} "
                f"— skipping to retry queue"
            )
            if self.downloader and self.downloader.retry_queue:
                self.downloader.retry_queue.add(
                    video_id=f"{video_id}_{int(start)}_{int(end)}",
                    keyword='segment',
                    tier='segment',
                    error_message='circuit_breaker_open_max_tier',
                    error_category='video_specific',
                    escalation_tier=int(kw_state.current_tier),
                )
            return 'circuit_breaker_open_max_tier'

        # Not at max tier — pause and wait for circuit recovery
        cb.check_and_wait()
        return None

    def _execute_download(
        self,
        ctx: _DownloadLoopContext,
        video_id: str,
        start: float,
        end: float,
        output_file: Path,
        progress_callback: Optional[MultiCallback] = None,
        cookie_file_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the actual yt-dlp download for a single segment.

        US-57-007: Extracts download execution from the main loop.
        US-82-007: Delegates to SegmentDownloadOrchestrator.download_segment().
        US-89-010: Added progress_callback for UI integrations.

        Returns:
            A dict with keys ``success`` (bool), and optionally ``duration``,
            ``error_msg``, or ``file_missing``.
        """
        # US-89-010: Notify callback of download start
        if progress_callback and len(progress_callback) > 0:
            try:
                progress_callback.on_start(
                    video_id,
                    {'start': start, 'end': end, 'output_file': str(output_file)},
                )
            except Exception:
                pass  # Don't let callback errors break downloads

        _progress_hook = self._make_progress_hook(video_id, ctx.stats, progress_callback, ctx.escalation_mgr)

        # US-49-004: Read stall timeout for process-level hang detection
        _stall_timeout = 120
        if ctx.download_config:
            _raw_stall = getattr(ctx.download_config, 'segment_stall_timeout', 120)
            try:
                _stall_timeout = int(_raw_stall)
            except (TypeError, ValueError):
                _stall_timeout = 120

        orch = self._get_orchestrator()
        # US-167-009: DEBUG-level sub-stage timing for segment download
        dl_start = time.time()
        result = orch.download_segment(
            video_id=video_id,
            start=start,
            end=end,
            output_file=output_file,
            progress_hooks=[_progress_hook],
            stall_timeout=_stall_timeout,
            cookie_file_override=cookie_file_override,
        )
        dl_elapsed = time.time() - dl_start
        logger.debug(
            f"[DOWNLOAD] Sub-stage timing: download_segment "
            f"for {video_id} [{start:.1f}-{end:.1f}] took {dl_elapsed:.2f}s"
        )

        # Convert SegmentDownloadResult to dict for backward-compat with _handle_result
        return {
            'success': result.success,
            'duration': result.duration,
            'error_msg': result.error_msg,
            'file_missing': result.file_missing,
            'impersonation_target': result.impersonation_target,
        }

    def _validate_checksum(
        self,
        file_path: Path,
        expected_checksum: Optional[str] = None,
        expected_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate downloaded segment integrity using checksum.

        US-143-012: Validates downloaded segment integrity using SHA256 checksums.

        Args:
            file_path: Path to the downloaded segment file
            expected_checksum: Optional expected checksum (hex string)
            expected_size: Optional expected file size in bytes

        Returns:
            Dict with keys:
                - valid (bool): True if validation passed
                - checksum (str): Calculated checksum (hex string)
                - size_match (bool): True if size matches expected
                - error_msg (str): Error message if validation failed
        """
        # Get checksum config from download config
        dl_cfg = getattr(self, 'download_config', None)
        if dl_cfg:
            checksum_cfg = getattr(dl_cfg, 'checksum_validation', None)
        else:
            checksum_cfg = None

        # Default values if config not available
        enabled = getattr(checksum_cfg, 'enabled', False) if checksum_cfg else False
        algorithm = getattr(checksum_cfg, 'algorithm', 'sha256') if checksum_cfg else 'sha256'
        min_file_size = getattr(checksum_cfg, 'min_file_size_bytes', 1024) if checksum_cfg else 1024
        verify_size = getattr(checksum_cfg, 'verify_file_size', True) if checksum_cfg else True

        if not enabled:
            return {'valid': True, 'checksum': None, 'size_match': True, 'error_msg': None}

        if not file_path.exists():
            return {'valid': False, 'checksum': None, 'size_match': False, 'error_msg': 'File not found'}

        try:
            file_size = file_path.stat().st_size
        except OSError as e:
            return {'valid': False, 'checksum': None, 'size_match': False, 'error_msg': f'Cannot stat file: {e}'}

        # Check minimum file size
        if file_size < min_file_size:
            logger.debug(f"Skipping checksum validation for small file: {file_path} ({file_size} bytes)")
            return {'valid': True, 'checksum': None, 'size_match': True, 'error_msg': 'File too small'}

        # Verify file size if expected size provided and enabled
        size_match = True
        if verify_size and expected_size is not None:
            size_match = file_size == expected_size
            if not size_match:
                logger.warning(f"File size mismatch for {file_path}: expected {expected_size}, got {file_size}")

        # Calculate checksum
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                # Read in chunks for memory efficiency
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
            calculated_checksum = hash_obj.hexdigest()
        except Exception as e:
            return {'valid': False, 'checksum': None, 'size_match': size_match, 'error_msg': f'Checksum error: {e}'}

        # Verify checksum if expected value provided
        checksum_valid = True
        if expected_checksum:
            checksum_valid = calculated_checksum.lower() == expected_checksum.lower()
            if not checksum_valid:
                logger.warning(
                    f"Checksum mismatch for {file_path}: "
                    f"expected {expected_checksum}, got {calculated_checksum}"
                )

        return {
            'valid': checksum_valid and size_match,
            'checksum': calculated_checksum,
            'size_match': size_match,
            'error_msg': None if (checksum_valid and size_match) else ('Size mismatch' if not size_match else 'Checksum mismatch'),
        }

    def _handle_result(
        self,
        ctx: _DownloadLoopContext,
        result: Dict[str, Any],
        video_id: str,
        start: float,
        end: float,
        output_file: Path,
        downloaded: list,
        idx: int,
        total: int,
        progress_callback,
    ) -> bool:
        """Process the outcome of a single segment download attempt.

        US-57-007: Dispatches to success or error handling. Returns
        ``True`` if the loop should abort (bot / network threshold hit).
        """
        from ..state import DownloadedVideo
        stats = ctx.stats

        # --- success path ---
        if result.get('success'):
            url = f"https://www.youtube.com/watch?v={video_id}"
            downloaded.append(DownloadedVideo(
                file=str(output_file), url=url, source='segment_download'
            ))
            try:
                _dl_bytes = output_file.stat().st_size
            except OSError:
                _dl_bytes = 0
            stats.increment_success(duration=result.get('duration', 0), file_bytes=_dl_bytes)

            # US-164-008: Log segment download completion with file size
            logger.info(
                f"[DOWNLOAD_SEGMENTS] Download complete: video_id={video_id}, "
                f"file_size={_dl_bytes} bytes, duration={result.get('duration', 0):.2f}s"
            )

            ctx.consecutive_network_failures = 0
            if ctx.consecutive_bot_detections > 0:
                ctx.consecutive_bot_detections = 0
                if ctx.escalation_mgr:
                    ctx.escalation_mgr.clear_tier_floor()
            if ctx.escalation_mgr:
                ctx.escalation_mgr.record_success(video_id)
            # Record impersonation success for success-rate tracking
            _imp_target = result.get('impersonation_target')
            if _imp_target and self.downloader and getattr(self.downloader, 'impersonation_manager', None):
                self.downloader.impersonation_manager.record_success(_imp_target)
            return False

        # --- file-missing edge case (download didn't error but file absent) ---
        if result.get('file_missing'):
            stats.failed += 1
            stats.attempted += 1
            stats.segment_durations.append(result.get('duration', 0))
            # US-81-002: Track as failed item
            stats.failed_items.append({
                'video_id': video_id,
                'error_type': 'file_missing',
                'message': f'Download succeeded but file not found: {output_file}',
            })
            log_error_with_context(logger, "DL-001", f"Download succeeded but file not found: {output_file}", video_id=video_id, file_path=output_file)
            return False

        # --- error path (delegated) ---
        return self._handle_download_error(
            ctx, result.get('error_msg', 'unknown error'),
            video_id, start, end, downloaded, idx, total, progress_callback,
            duration_tier=result.get('duration_tier', ''),
            match_confidence=result.get('match_confidence', 0.0),
            impersonation_target=result.get('impersonation_target'),
        )

    def _handle_download_error(
        self,
        ctx: _DownloadLoopContext,
        error_msg: str,
        video_id: str,
        start: float,
        end: float,
        downloaded: list,
        idx: int,
        total: int,
        progress_callback,
        duration_tier: str = "",
        match_confidence: float = 0.0,
        impersonation_target: Optional[str] = None,
    ) -> bool:
        """Handle a failed download: classify, escalate, and check abort thresholds.

        US-57-007: Extracted from _handle_result to keep each method under 80 lines.

        Returns:
            ``True`` if the loop should abort.
        """
        stats = ctx.stats
        _err_obj = classify_error_category(error_msg)
        stats.increment_failure(category=_err_obj.category, error_msg=error_msg, video_id=video_id)
        log_error_with_context(logger, "DL-001", f"Failed to download segment {video_id}: {error_msg}", video_id=video_id)

        is_bot_error = _is_escalation_error(_err_obj)
        if ctx.escalation_mgr and is_bot_error:
            ctx.escalation_mgr.record_failure(video_id, error_msg)
            if ctx.cookie_rotator and getattr(ctx.cookie_rotator, 'should_rotate', None):
                if ctx.cookie_rotator.should_rotate(error_msg):
                    ctx.cookie_rotator.rotate()
        # Record impersonation failure for success-rate tracking
        if impersonation_target and self.downloader and getattr(self.downloader, 'impersonation_manager', None):
            self.downloader.impersonation_manager.record_failure(impersonation_target)

        # Bot-detection tracking and abort
        if is_bot_error:
            ctx.consecutive_bot_detections += 1
            if (
                ctx.bot_floor_threshold > 0
                and ctx.consecutive_bot_detections >= ctx.bot_floor_threshold
                and ctx.escalation_mgr
            ):
                from ..downloader.types import EscalationTier
                ctx.escalation_mgr.set_tier_floor(EscalationTier.VPN_ROTATION)
                logger.warning(
                    f"Bot-detection tier floor activated: "
                    f"{ctx.consecutive_bot_detections} consecutive bot-detection "
                    f"errors across video IDs — new downloads start at max tier"
                )
            if self._should_abort_bot_detection(ctx, stats, idx, total, downloaded, progress_callback):
                return True

        # Network failure tracking and abort
        if _is_network_failure(_err_obj):
            ctx.consecutive_network_failures += 1
            logger.warning(
                f"Network failure detected ({ctx.consecutive_network_failures}/"
                f"{ctx.network_failure_threshold}): {error_msg}"
            )
            if ctx.consecutive_network_failures >= ctx.network_failure_threshold:
                remaining = total - idx
                log_error_with_context(
                    logger, "DL-006",
                    f"Aborting download loop: {ctx.consecutive_network_failures} consecutive "
                    f"network failures indicate systemic network issue. "
                    f"Skipping {remaining} remaining segment(s).",
                    video_id=video_id, consecutive_failures=ctx.consecutive_network_failures,
                    remaining=remaining
                )
                if progress_callback:
                    progress_callback(idx, total, downloaded)
                return True
        elif not is_bot_error:
            ctx.consecutive_network_failures = 0

        # Add to retry queue
        self._enqueue_retry(
            ctx, video_id, start, end, error_msg,
            duration_tier=duration_tier,
            match_confidence=match_confidence,
        )
        return False

    def _should_abort_bot_detection(
        self,
        ctx: _DownloadLoopContext,
        stats: SegmentDownloadStats,
        idx: int,
        total: int,
        downloaded: list,
        progress_callback,
    ) -> bool:
        """Check whether bot-detection count has exceeded the abort threshold.

        US-57-007: Extracted from _handle_download_error for line budget.
        """
        if ctx.bot_abort_threshold <= 0:
            return False
        if ctx.consecutive_bot_detections < ctx.bot_abort_threshold:
            return False

        remaining = total - idx
        stats.error_categories['bot_detection_abort'] = 1
        stats.error_aggregator.record(
            f"Bot-detection abort: {ctx.consecutive_bot_detections} "
            f"consecutive bot errors exceeded threshold "
            f"({ctx.bot_abort_threshold})",
            'bot_detection_abort',
        )
        log_error_with_context(
            logger, "DL-004",
            f"Aborting download loop: {ctx.consecutive_bot_detections} "
            f"consecutive bot-detection errors (threshold: "
            f"{ctx.bot_abort_threshold}). YouTube is broadly blocking "
            f"requests. Skipping {remaining} remaining segment(s). "
            f"Check cookie configuration.",
            video_id=video_id, consecutive_errors=ctx.consecutive_bot_detections,
            threshold=ctx.bot_abort_threshold,
            remaining=remaining
        )
        log_error_with_context(
            logger, "DL-004",
            "Suggested actions to resolve bot-detection:\n"
            "  1. Check/refresh your browser cookies "
            "(cookies_from_browser or cookies_path in config.yaml)\n"
            "  2. Enable Mullvad VPN rotation "
            "(download.mullvad.enabled: true)\n"
            "  3. Wait 15-30 minutes before retrying "
            "(YouTube rate limits are temporary)\n"
            "  4. Run with --resume to continue from this checkpoint",
            video_id=video_id
        )
        if progress_callback:
            progress_callback(idx, total, downloaded)
        return True

    def _enqueue_retry(
        self,
        ctx: _DownloadLoopContext,
        video_id: str,
        start: float,
        end: float,
        error_msg: str,
        duration_tier: str = "",
        match_confidence: float = 0.0,
    ) -> None:
        """Add a failed segment to the retry queue.

        US-57-007: Extracted from _handle_download_error for clarity.
        US-114-002: Added duration_tier and match_confidence for smart prioritization.
        """
        if not self.downloader or not self.downloader.retry_queue:
            return
        _err_obj = classify_error_category(error_msg)
        _esc_tier = 1
        if ctx.escalation_mgr:
            try:
                _esc_state = ctx.escalation_mgr._get_state(video_id)
                _esc_tier = int(_esc_state.current_tier)
            except Exception:
                pass
        self.downloader.retry_queue.add(
            video_id=f"{video_id}_{int(start)}_{int(end)}",
            keyword='segment',
            tier='segment',
            error_message=error_msg,
            error_category=_err_obj.category,
            escalation_tier=_esc_tier,
            duration_tier=duration_tier,
            match_confidence=match_confidence,
        )
        logger.debug(
            f"Added {video_id} to retry queue "
            f"(category={_err_obj.category}, escalation_tier={_esc_tier}, "
            f"duration_tier={duration_tier}, confidence={match_confidence})"
        )

    @staticmethod
    def _make_progress_hook(
        video_id: str,
        stats: SegmentDownloadStats,
        progress_callback: Optional[MultiCallback] = None,
        escalation_manager: Any = None,
    ) -> callable:
        """Create a yt-dlp progress_hooks callback for per-download observability.

        US-51-006: Logs download progress at INFO level for segments taking >30s,
        and logs final file size/time on completion. Accumulates totals into
        stats.progress_hooks_data for stage metrics.

        US-89-010: Now supports progress_callback for UI integrations. The callback
        receives on_progress calls with bytes_downloaded, total_bytes, speed, and eta.

        US-113-010: Enhanced progress reporting with ETA, bandwidth utilization, and tier display.

        Args:
            video_id: YouTube video ID being downloaded.
            stats: The stage stats dataclass; progress data is accumulated
                   under stats.progress_hooks_data.
            progress_callback: Optional MultiCallback for UI integrations.
            escalation_manager: Optional escalation manager for tier lookup.

        Returns:
            A callable suitable for ydl_opts['progress_hooks'].
        """
        hook_data = stats.progress_hooks_data
        _last_log_elapsed = [0.0]  # mutable container for closure
        _last_callback_elapsed = [0.0]  # Throttle callbacks to ~1 second

        def _get_current_tier() -> Optional[int]:
            """Get current escalation tier for this video."""
            if escalation_manager is None:
                return None
            try:
                state = escalation_manager._get_state(video_id)
                return int(state.current_tier)
            except Exception:
                return None

        def _hook(d: Dict[str, Any]) -> None:
            status = d.get('status', '')
            elapsed = d.get('elapsed', 0.0) or 0.0

            if status == 'downloading':
                downloaded = d.get('downloaded_bytes') or 0
                total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                speed = d.get('speed')  # bytes/sec, can be None
                eta = d.get('eta')  # seconds remaining, can be None

                # US-113-010: Calculate bandwidth utilization
                bandwidth_util = calculate_bandwidth_utilization(speed)
                current_tier = _get_current_tier()

                # US-89-010: Call progress callback ~1 per second (not on every event)
                if progress_callback and len(progress_callback) > 0:
                    if elapsed - _last_callback_elapsed[0] >= 1.0:
                        _last_callback_elapsed[0] = elapsed
                        try:
                            progress_callback.on_progress(
                                video_id,
                                downloaded,
                                total_bytes,
                                speed,
                                eta,
                            )
                        except Exception:
                            pass  # Don't let callback errors break downloads

                # Throttle: only log every 15s of elapsed time
                if elapsed - _last_log_elapsed[0] >= 15:
                    _last_log_elapsed[0] = elapsed

                    # US-113-010: Use enhanced format for consistent pipeline logging
                    downloaded_mb = downloaded / (1024 * 1024) if downloaded else 0
                    total_mb = total_bytes / (1024 * 1024) if total_bytes else 0
                    speed_kbps = (speed / 1024) if speed else 0

                    progress_line = format_progress_line(
                        video_id=video_id,
                        downloaded_mb=downloaded_mb,
                        total_mb=total_mb,
                        speed_kbps=speed_kbps,
                        eta_seconds=eta,
                        bandwidth_util=bandwidth_util,
                        tier=current_tier,
                        elapsed=elapsed,
                    )
                    logger.info(progress_line)
                    hook_data['segments_with_progress'] += 1

            elif status == 'finished':
                total_bytes = d.get('total_bytes') or d.get('downloaded_bytes') or 0
                if total_bytes:
                    hook_data['total_downloaded_bytes'] += total_bytes
                hook_data['segments_finished'] += 1

                # US-89-010: Notify callback of completion
                if progress_callback and len(progress_callback) > 0:
                    try:
                        progress_callback.on_complete(
                            video_id,
                            '',
                            total_bytes,
                            elapsed,
                        )
                    except Exception:
                        pass

                if elapsed and elapsed > 0:
                    # US-113-010: Use consistent format with tier
                    current_tier = _get_current_tier()
                    tier_str = f" [T{current_tier}]" if current_tier else ""
                    logger.info(
                        f"[DOWNLOAD] {video_id}: finished — "
                        f"{total_bytes / (1024 * 1024):.1f}MB in {elapsed:.1f}s{tier_str}"
                    )

        return _hook

    def _build_ydl_opts(
        self,
        *,
        video_id: str,
        start: float,
        end: float,
        output_file: Path,
        download_config,
        escalation_mgr=None,
        cookie_rotator=None,
        progress_hooks: Optional[List] = None,
    ) -> tuple:
        """Build ydl_opts dict for a yt-dlp Python API download call.

        US-52-005: Extracts the repeated ydl_opts construction from
        _download_segments and the retry loop into a single builder.

        Encapsulates:
        - Base options (format, output, ranges, timeouts, retries)
        - Cookie propagation (cookiesfrombrowser / cookiefile fallback)
        - Escalation application (impersonation, extractor_args, cookie rotation)

        Args:
            video_id: YouTube video ID.
            start: Segment start time in seconds.
            end: Segment end time in seconds.
            output_file: Path for the downloaded file.
            download_config: Download config object (or None).
            escalation_mgr: Optional EscalationManager instance.
            cookie_rotator: Optional CookieRotator instance.
            progress_hooks: Optional list of progress hook callables.

        Returns:
            (ydl_opts, escalation_result) tuple. escalation_result may be None.
        """
        # Read segment config from download config with fallback defaults
        _socket_timeout = 30
        _max_res = 1080
        _seg_format = 'best[height<={segment_max_resolution}]'
        if download_config:
            _seg_sock = getattr(download_config, 'segment_socket_timeout', 0)
            _socket_timeout = _seg_sock if _seg_sock else getattr(download_config, 'socket_timeout', 30)
            _max_res = getattr(download_config, 'segment_max_resolution', 1080)
            _seg_format = getattr(download_config, 'segment_format', _seg_format)

        # Build format string with fallback chain to handle Tier 2+ escalation
        # where alternative player_clients may not expose formats matching the
        # height filter (causes "Requested format is not available")
        _primary_format = _seg_format.format(segment_max_resolution=_max_res)
        # android client serves HLS (pre-merged video+audio streams only).
        # IMPORTANT: do NOT use bestvideo+bestaudio — android has no separate
        # video-only or audio-only formats, so the merge selector finds nothing.
        # Use 'best' (merged formats) with height cap instead.
        _format_with_fallback = f'best[height<=1080]/best[height<=720]/best'

        ydl_opts: Dict[str, Any] = {
            'format': _format_with_fallback,
            'outtmpl': str(output_file),
            'quiet': True,
            'no_warnings': True,
            # Skip gracefully when rate-limited/unavailable (no formats returned);
            # caller checks output_file.exists() to detect actual failure
            'ignore_no_formats_error': True,
            # Auto-update EJS challenge solver scripts from GitHub so yt-dlp can
            # solve YouTube's n-sig challenges (required for format extraction)
            'remote_components': {'ejs:github'},
            # Time-based download options
            'download_ranges': lambda info, ydl: [{'start_time': start, 'end_time': end}],
            'force_keyframes_at_cuts': True,
            # Network resilience (matches core.py subprocess args)
            'socket_timeout': _socket_timeout,
            'retries': 10,
            'fragment_retries': 10,
            # Re-extract video URL if download speed drops below threshold
            # (bypasses YouTube throttling by getting a fresh URL)
            'throttled_rate': 100_000,  # 100 KB/s minimum
            # Use concurrent fragment downloads to bypass single-stream throttling
            'concurrent_fragment_downloads': 4,
            # Force IPv4 — YouTube bot detection triggers on IPv6 addresses
            'source_address': '0.0.0.0',
            # Use android client — 360p primary
            'extractor_args': {'youtube': {'player_client': ['android']}},
        }

        if progress_hooks:
            ydl_opts['progress_hooks'] = progress_hooks

        # US-49-002: Propagate cookie auth to Python API
        if download_config:
            _browser = getattr(download_config, 'cookies_from_browser', '')
            if _browser:
                ydl_opts['cookiesfrombrowser'] = [_browser]
            else:
                _cookies_path = getattr(download_config, 'cookies_path', '')
                if not _cookies_path:
                    _cookie_rotation = getattr(download_config, 'cookie_rotation', None)
                    if _cookie_rotation:
                        _cookie_files = getattr(_cookie_rotation, 'cookie_files', [])
                        if _cookie_files:
                            _cookies_path = _cookie_files[0]
                if _cookies_path:
                    ydl_opts['cookiefile'] = _cookies_path

        # US-48-005: Apply escalation tiers (impersonation + extractor_args + cookies)
        escalation_result = None
        imp_target = None
        if escalation_mgr:
            try:
                escalation_result = escalation_mgr.get_escalation_args(video_id)
                _apply_escalation_to_ydl_opts(ydl_opts, escalation_result)
                imp_target = getattr(escalation_result, 'impersonation_target', None)

                # Tier 3: apply cookie rotation (overrides baseline cookies)
                if escalation_result.rotate_cookies and cookie_rotator:
                    cookie_path = cookie_rotator.get_current_cookie()
                    if cookie_path:
                        ydl_opts['cookiefile'] = cookie_path
                        # Remove browser cookies when using rotated cookie file
                        ydl_opts.pop('cookiesfrombrowser', None)
            except Exception as esc_err:
                logger.debug(f"Escalation lookup failed for {video_id}: {esc_err}")
        elif self.downloader and getattr(self.downloader, 'impersonation_manager', None):
            # Fallback: direct impersonation only (no escalation manager)
            try:
                imp_args, imp_target = self.downloader.impersonation_manager.get_impersonate_args_with_target()
                if len(imp_args) >= 2 and imp_args[0] == '--impersonate':
                    ydl_opts['impersonate'] = imp_args[1]
                    # When impersonation is applied, remove baseline cookies so
                    # impersonation is the sole auth method (no cookie interference)
                    ydl_opts.pop('cookiefile', None)
                    ydl_opts.pop('cookiesfrombrowser', None)
            except Exception:
                pass

        return ydl_opts, escalation_result, imp_target

    @staticmethod
    def _print_progress(current: int, total: int, stats: SegmentDownloadStats) -> None:
        """Print running progress line after each download attempt."""
        ok = stats.succeeded + stats.cached
        attempted = stats.attempted
        rate = (ok / attempted * 100) if attempted > 0 else 0.0
        progress_pct = int((current / total) * 100) if total > 0 else 0
        log_progress(
            logger, "DOWNLOAD_SEGMENTS",
            progress_pct=progress_pct,
            current=current,
            total=total,
            ok=ok,
            failed=stats.failed,
            cached=stats.cached,
            success_rate_pct=int(rate)
        )

    @staticmethod
    def _print_summary(stats: SegmentDownloadStats, elapsed: float) -> None:
        """Print end-of-stage summary."""
        ok = stats.succeeded + stats.cached
        attempted = stats.attempted
        rate = (ok / attempted * 100) if attempted > 0 else 0.0

        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            time_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        else:
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            time_str = f"{h}h {m}m"

        # US-167-009: Log stage completion with timing
        downloaded_count = ok
        failed_count = stats.failed
        total_size_mb = stats.total_bytes / (1024 * 1024) if stats.total_bytes > 0 else 0.0
        log_stage_complete(
            logger, "DOWNLOAD_SEGMENTS",
            elapsed_seconds=elapsed,
            attempted=attempted,
            total=stats.total,
            succeeded=stats.succeeded,
            cached=stats.cached,
            failed_count=failed_count,
            success_rate_pct=int(rate),
            elapsed_time=time_str,
            downloaded_count=downloaded_count,
            total_size_mb=round(total_size_mb, 2)
        )

        # Per-segment duration stats
        durations = stats.segment_durations
        if durations:
            avg_dur = statistics.mean(durations)
            median_dur = statistics.median(durations)
            logger.info(f"[DOWNLOAD_SEGMENTS] Avg segment time: {avg_dur:.1f}s, Median: {median_dur:.1f}s")

        # Total bytes downloaded
        if stats.total_bytes > 0:
            if stats.total_bytes < 1024 * 1024:
                size_str = f"{stats.total_bytes / 1024:.1f} KB"
            elif stats.total_bytes < 1024 * 1024 * 1024:
                size_str = f"{stats.total_bytes / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{stats.total_bytes / (1024 * 1024 * 1024):.2f} GB"
            logger.info(f"[DOWNLOAD_SEGMENTS] Total downloaded: {size_str}")

        # Retry count
        if stats.retry_count > 0:
            logger.info(f"[DOWNLOAD_SEGMENTS] Total retries: {stats.retry_count}")

    @staticmethod
    def _log_error_summary(stats: SegmentDownloadStats) -> None:
        """US-49-009 + US-51-011: Log structured error summary with per-category breakdown.

        Logs at INFO level with category counts and sample messages, and at
        WARNING level with actionable guidance when >50% of failures are
        bot-detection.
        """
        if not stats.failed:
            return  # No errors to summarize

        summary_parts = [f"{cat}={count}" for cat, count in sorted(stats.error_categories.items())]
        logger.info(
            f"Download error summary: total={stats.total} "
            f"succeeded={stats.succeeded} failed={stats.failed} "
            f"cached={stats.cached} skipped="
            f"{stats.total - stats.attempted} | "
            f"errors by category: {', '.join(summary_parts) if summary_parts else 'uncategorized'}"
        )

        # US-51-011: Log categorized table with sample messages via ErrorAggregator
        if stats.error_aggregator.total_errors > 0:
            stats.error_aggregator.log_summary(stage_name='DOWNLOAD_SEGMENTS')

        # Actionable guidance when >50% of failures are bot-detection
        bot_count = stats.error_categories.get('bot_detection', 0)
        if bot_count > 0 and (bot_count / stats.failed) > 0.5:
            logger.warning(
                "Most failures are bot-detection. "
                "Check cookie configuration (cookies_from_browser or cookies_path in config.yaml)."
            )

    @staticmethod
    def _log_escalation_summary(escalation_summary: Dict[str, Any]) -> None:
        """US-50-009: Log structured escalation tier effectiveness summary.

        Logs at INFO level with tier distribution, success/failure counts,
        and per-tier effectiveness rates.
        """
        if not escalation_summary or not escalation_summary.get('total_escalations', 0):
            # No escalation data to report (all downloads succeeded at tier 1)
            if escalation_summary:
                logger.info(
                    "Escalation summary: no escalations needed "
                    f"(bot_detection={escalation_summary.get('bot_detection_count', 0)} "
                    f"network_failures={escalation_summary.get('network_failure_count', 0)})"
                )
            return

        vpt = escalation_summary.get('videos_per_tier', {})
        tier_parts = [f"{tier}={count}" for tier, count in sorted(vpt.items())]

        logger.info(
            f"Escalation summary: total_escalations={escalation_summary.get('total_escalations', 0)} "
            f"average_tier={escalation_summary.get('average_tier', 1.0)} "
            f"videos_per_tier=[{', '.join(tier_parts)}] "
            f"bot_detection={escalation_summary.get('bot_detection_count', 0)} "
            f"network_failures={escalation_summary.get('network_failure_count', 0)}"
        )

        # Log tier effectiveness if available
        tier_eff = escalation_summary.get('tier_effectiveness', {})
        if tier_eff:
            for category, rates in tier_eff.items():
                rate_parts = [f"{t}={r:.1%}" for t, r in sorted(rates.items())]
                logger.info(
                    f"Tier effectiveness [{category}]: {', '.join(rate_parts)}"
                )

    def _process_retry_queue(
        self,
        output_dir: Path,
        buffer_seconds: float,
        downloaded: List['DownloadedVideo'],
        total: int,
        progress_callback,
        stats: Optional[SegmentDownloadStats] = None
    ) -> None:
        """Process any failed downloads in the retry queue.

        Attempts to retry failed segment downloads using the downloader's
        retry queue infrastructure. US-48-005: Uses escalation tiers for
        retries (items that originally failed at Tier 1 will retry at
        the escalated tier).
        """
        from ..state import DownloadedVideo

        if not self.downloader or not self.downloader.retry_queue:
            return

        retry_queue = self.downloader.retry_queue
        if not retry_queue.has_pending():
            return

        pending = retry_queue.get_pending_items()
        logger.info(f"Processing {len(pending)} items from retry queue")

        # Start retry pass (applies configured delay)
        retry_queue.start_retry_pass()

        # Get escalation manager and cookie rotator for retry pass
        escalation_mgr = getattr(self.downloader, 'escalation_manager', None)
        cookie_rotator = getattr(self.downloader, 'cookie_rotator', None)

        # Apply inter-request delay during retries too
        dl_cfg = getattr(self.downloader, 'download_config', None)
        _retry_delay = float(getattr(dl_cfg, 'segment_request_delay', 1.0)) if dl_cfg else 1.0
        # US-85-008: Jitter factor to prevent thundering herd
        _retry_jitter = float(getattr(dl_cfg, 'segment_request_delay_jitter', 0.25)) if dl_cfg else 0.25

        retry_attempts = 0
        for item in pending:
            # Parse video_id from the retry item (format: video_id_start_end)
            # US-48-008: Use rsplit to handle video IDs with underscores
            parts = item.video_id.rsplit('_', 2)
            if len(parts) < 3:
                logger.warning(f"Invalid retry item format: {item.video_id}")
                retry_queue.mark_failed(item.video_id)
                continue

            video_id = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                logger.warning(f"Invalid time range in retry item: {item.video_id}")
                retry_queue.mark_failed(item.video_id)
                continue

            output_file = output_dir / f"{video_id}_{start}_{end}.webm"

            if output_file.exists():
                retry_queue.mark_success(item.video_id)
                continue

            retry_attempts += 1

            url = f"https://www.youtube.com/watch?v={video_id}"

            # US-49-010: Apply stored escalation tier floor before getting args.
            if escalation_mgr and item.escalation_tier > 1:
                try:
                    from ..downloader.types import EscalationTier
                    stored_tier = EscalationTier(item.escalation_tier)
                    esc_state = escalation_mgr._get_state(video_id)
                    if esc_state.current_tier < stored_tier:
                        esc_state.current_tier = stored_tier
                        logger.debug(
                            f"Retry {video_id}: elevated escalation tier to "
                            f"{stored_tier.name} (from retry queue)"
                        )
                except (ValueError, Exception):
                    pass

            # US-82-007: Delegate retry download to orchestrator
            result = self._get_orchestrator().download_segment(
                video_id=video_id,
                start=start,
                end=end,
                output_file=output_file,
            )

            if result.success:
                downloaded.append(DownloadedVideo(
                    file=str(output_file),
                    url=url,
                    source='segment_retry'
                ))
                retry_queue.mark_success(item.video_id)
                if escalation_mgr:
                    escalation_mgr.record_success(video_id)
                if result.impersonation_target and self.downloader and getattr(self.downloader, 'impersonation_manager', None):
                    self.downloader.impersonation_manager.record_success(result.impersonation_target)
                logger.info(f"Retry succeeded for {video_id}")
            elif result.error_msg:
                logger.warning(f"Retry failed for {video_id}: {result.error_msg}")
                retry_queue.mark_failed(item.video_id)
                if escalation_mgr and _is_escalation_error(result.error_msg):
                    escalation_mgr.record_failure(video_id, result.error_msg)
                if result.impersonation_target and self.downloader and getattr(self.downloader, 'impersonation_manager', None):
                    self.downloader.impersonation_manager.record_failure(result.impersonation_target)
            else:
                retry_queue.mark_failed(item.video_id)

            # Delay between retry requests to avoid rate-limiting
            # US-85-008: Apply random jitter to prevent synchronized request bursts
            if _retry_delay > 0:
                if _retry_jitter > 0:
                    jittered_retry_delay = _retry_delay * random.uniform(1 - _retry_jitter, 1 + _retry_jitter)
                else:
                    jittered_retry_delay = _retry_delay
                time.sleep(jittered_retry_delay)

            # Update checkpoint with retry progress
            if progress_callback:
                progress_callback(len(downloaded), total, downloaded)

        # Update stats with retry count
        if stats is not None:
            stats.retry_count = retry_attempts

    def _update_matches_with_local_paths(
        self,
        state: 'PipelineState',
        downloaded_segments: List['DownloadedVideo']
    ):
        """Update match objects to reference local file paths.

        US-48-008: Uses rsplit('_', 2) to extract video_id from filename
        format '{video_id}_{start}_{end}.mp4', correctly handling video IDs
        that contain underscores (e.g., 'abc_def_0_15.mp4' → 'abc_def').
        """
        # Build mapping from video_id to local file
        file_map = {}
        for seg in downloaded_segments:
            # Extract video_id from filename: {video_id}_{start}_{end}.mp4
            filename = Path(seg.file).stem
            parts = filename.rsplit('_', 2)
            if len(parts) == 3:
                video_id = parts[0]
                file_map[video_id] = seg.file
            elif parts:
                # Fallback for unexpected format
                video_id = parts[0]
                file_map[video_id] = seg.file

        # Update matches
        updated_count = 0
        for match in state.matches:
            if hasattr(match, 'video_file') and match.video_file:
                video_id = match.video_file
                if video_id in file_map:
                    match.video_file = file_map[video_id]
                    updated_count += 1

        logger.info(f"Updated {updated_count} matches with local file paths")

    @staticmethod
    def _propagate_vsr_metadata(
        state: 'PipelineState',
        downloaded_segments: List['DownloadedVideo'],
    ) -> None:
        """US-84-011: Copy video_chapters/video_tags from VideoSearchResult to DownloadedVideo.

        After download, DownloadedVideo instances lack metadata from their
        source VideoSearchResult.  This method builds a video_id lookup from
        state.video_search_results and copies the two fields over.
        """
        if not state.video_search_results or not downloaded_segments:
            return

        # Build video_id -> VSR lookup
        vsr_lookup: Dict[str, Any] = {}
        for vsr in state.video_search_results:
            if isinstance(vsr, dict):
                vid = vsr.get('video_id', '')
            else:
                vid = getattr(vsr, 'video_id', '')
            if vid:
                vsr_lookup[vid] = vsr

        propagated = 0
        for dv in downloaded_segments:
            filename = Path(dv.file).stem
            parts = filename.rsplit('_', 2)
            video_id = parts[0] if parts else ''
            if video_id and video_id in vsr_lookup:
                vsr = vsr_lookup[video_id]
                if isinstance(vsr, dict):
                    chapters = vsr.get('video_chapters', [])
                    tags = vsr.get('video_tags', [])
                else:
                    chapters = getattr(vsr, 'video_chapters', [])
                    tags = getattr(vsr, 'video_tags', [])
                if chapters:
                    dv.video_chapters = chapters
                if tags:
                    dv.video_tags = tags
                if chapters or tags:
                    propagated += 1

        if propagated:
            logger.info(
                f"US-84-011: Propagated video_chapters/video_tags to "
                f"{propagated}/{len(downloaded_segments)} downloaded segments"
            )

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore from checkpoint.

        US-48-009: Validates restored segment files exist on disk and have
        non-zero size. Filters out deleted or empty files. Cross-references
        against state.matches to identify segments that still need downloading.

        US-51-010: Restores retry queue from checkpoint so previously-failed
        videos are retried before processing new segments on resume.
        """
        try:
            if config:
                output_dir = Path(config.downloaded_videos_dir)
                if output_dir.exists():
                    from ..state import DownloadedVideo
                    segments = []
                    skipped = 0
                    for f in output_dir.glob('*_*_*.mp4'):
                        if not f.exists() or f.stat().st_size == 0:
                            skipped += 1
                            continue
                        segments.append(DownloadedVideo(
                            file=str(f),
                            source='restored'
                        ))
                    state.downloaded_segments = segments
                    logger.info(
                        f"Restored DOWNLOAD_SEGMENTS: {len(segments)} valid, "
                        f"{skipped} invalid (missing or empty)"
                    )

                    # Cross-reference against matches to find segments needing download
                    if state.matches:
                        restored_ids = set()
                        for seg in segments:
                            fname = Path(seg.file).stem
                            parts = fname.rsplit('_', 2)
                            if len(parts) >= 3:
                                restored_ids.add(parts[0])

                        matched_ids = set()
                        for match in state.matches:
                            vid = getattr(match, 'video_file', '') or ''
                            if vid:
                                matched_ids.add(vid)

                        missing = matched_ids - restored_ids
                        if missing:
                            logger.info(
                                f"DOWNLOAD_SEGMENTS restore: {len(missing)} matched "
                                f"video(s) have no restored segments on disk"
                            )

            # US-51-010: Restore retry queue from checkpoint data
            cp_data = getattr(checkpoint, 'data', None)
            if cp_data:
                ds_data = getattr(cp_data, 'download_segments', {}) or {}
                retry_queue_data = ds_data.get('retry_queue')
                if retry_queue_data:
                    # Store on state for the run() method to pick up
                    if not hasattr(state, '_restored_retry_queue'):
                        state._restored_retry_queue = retry_queue_data
                    logger.info(
                        f"DOWNLOAD_SEGMENTS restore: found retry queue with "
                        f"{len(retry_queue_data.get('items', []))} pending items, "
                        f"{len(retry_queue_data.get('failed_ids', []))} permanently failed"
                    )

            return True
        except Exception as e:
            log_error_with_context(logger, "DL-001", f"Failed to restore DOWNLOAD_SEGMENTS: {e}")
            return True  # Non-critical, proceed anyway

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs"""
        if not state.matches:
            return "No matches available for segment download"
        return None

    def get_input_output_info(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get input/output info for dry-run preview"""
        # Count inputs (matches)
        input_count = len(state.matches) if state.matches else 0

        # Count outputs (downloaded segments)
        output_count = None
        if hasattr(state, 'downloaded_segments'):
            output_count = len(state.downloaded_segments)

        return {
            'inputs': 'matches',
            'outputs': 'video segments',
            'input_count': input_count,
            'output_count': output_count,
        }


@dataclass
class DownloadSegmentsConfig:
    """Typed configuration for programmatic segment downloads.

    US-89-002: Exposes configuration as typed dataclass for external callers
    and programmatic use cases.
    """

    buffer_seconds: float = 2.0
    """Buffer time in seconds to add before/after each segment."""

    max_concurrent: int = 1
    """Maximum concurrent downloads."""

    checkpoint_every_n: int = 10
    """Write checkpoint every N items."""

    batch_failure_threshold: float = 1.0
    """Fraction of failures that triggers abort (1.0 = disabled)."""

    retry_budget_max_attempts: int = 100
    """Maximum retry attempts per video."""

    base_delay: float = 1.0
    """Base delay between retries in seconds."""

    max_delay: float = 60.0
    """Maximum delay between retries in seconds."""

    @classmethod
    def from_config(cls, config: 'Config') -> 'DownloadSegmentsConfig':
        """Create config from pipeline Config object."""
        download_config = config.download
        return cls(
            buffer_seconds=download_config.segment_buffer,
            max_concurrent=getattr(download_config, 'max_concurrent', 1),
            checkpoint_every_n=int(getattr(download_config, 'segment_checkpoint_every_n', 10)),
            batch_failure_threshold=getattr(download_config, 'batch_failure_threshold', 1.0),
            retry_budget_max_attempts=int(getattr(download_config, 'segment_retry_budget_max_attempts', 100)),
            base_delay=getattr(download_config, 'segment_base_delay', 1.0),
            max_delay=getattr(download_config, 'segment_max_delay', 60.0),
        )


@dataclass
class DownloadSegmentsResult:
    """Typed result container for segment download operations.

    US-89-002: Provides structured return value with success count,
    failed list, and statistics for programmatic callers.
    """

    success_count: int = 0
    """Number of segments successfully downloaded."""

    failed_count: int = 0
    """Number of segments that failed to download."""

    cached_count: int = 0
    """Number of segments served from cache."""

    total_count: int = 0
    """Total number of segments processed."""

    failed_segments: List[Dict[str, Any]] = field(default_factory=list)
    """List of failed segment details with video_id, start, end, error."""

    stats: Optional[SegmentDownloadStats] = None
    """Detailed download statistics."""

    @property
    def succeeded(self) -> List[Dict[str, Any]]:
        """Returns empty list for compatibility - use success_count."""
        return []


def download_segments_from_matches(
    state: 'PipelineState',
    config: 'Config',
    progress_callback: Optional[callable] = None,
) -> DownloadSegmentsResult:
    """Programmatic API for downloading video segments from matches.

    US-89-002: Function-based API that accepts state and config objects,
    enabling programmatic access for external callers and better testability.

    Args:
        state: PipelineState object with matches attribute containing Match objects
        config: Config object with download settings
        progress_callback: Optional callback(current, total, downloaded) for UI integration

    Returns:
        DownloadSegmentsResult with success count, failed list, and stats
    """
    from pathlib import Path
    from ..downloader.orchestrator import SegmentDownloadOrchestrator

    # Validate matches exist
    if not state.matches:
        return DownloadSegmentsResult(
            success_count=0,
            failed_count=0,
            cached_count=0,
            total_count=0,
            failed_segments=[],
        )

    # Create stage instance and run
    stage = DownloadVideoSegmentsStage()

    # Use run() method to get StageResult
    from ..checkpoint import CheckpointManager
    from ..checkpoint import _CheckpointManager

    # Create a minimal checkpoint manager for API usage
    class NoopCheckpointManager:
        """Minimal checkpoint manager for API usage."""
        def __init__(self):
            self._project_path = None

        def save_intermediate(self, stage_name: str, data: dict):
            pass

        def save_final(self, stage_name: str, data: dict):
            pass

        def load_intermediate(self, stage_name: str) -> dict:
            return {}

        def load_final(self, stage_name: str) -> dict:
            return {}

        def has_checkpoint(self) -> bool:
            return False

        @property
        def project_path(self):
            return self._project_path

        @project_path.setter
        def project_path(self, value):
            self._project_path = value

    checkpoint_mgr = NoopCheckpointManager()

    # Run the stage
    result = stage.run(state, config, checkpoint_mgr)

    # Convert StageResult to DownloadSegmentsResult
    if result.success:
        data = result.data or {}
        stats = data.get('download_stats') if data else None

        # Extract failed items from checkpoint data if available
        failed_segments = []
        if hasattr(state, 'downloaded_segments'):
            # Calculate what failed by comparing to matches
            downloaded_ids = set()
            for dv in state.downloaded_segments:
                if hasattr(dv, 'url'):
                    # Extract video_id from URL
                    import re
                    match = re.search(r'v=([a-zA-Z0-9_-]{11})', dv.url)
                    if match:
                        downloaded_ids.add(match.group(1))

        return DownloadSegmentsResult(
            success_count=data.get('segment_count', 0),
            failed_count=data.get('failed_count', 0),
            cached_count=data.get('cached_count', 0),
            total_count=len(state.matches),
            failed_segments=failed_segments,
            stats=stats,
        )
    else:
        return DownloadSegmentsResult(
            success_count=0,
            failed_count=len(state.matches),
            cached_count=0,
            total_count=len(state.matches),
            failed_segments=[{'error': result.error}],
        )


def create_download_stage(
    *,
    config: 'Config' = None,
    circuit_breaker=None,
    escalation_manager=None,
    rate_limit_budget=None,
) -> DownloadVideoSegmentsStage:
    """Factory function that wires production defaults for DownloadVideoSegmentsStage.

    US-82-010: Provides a single entry-point for creating a fully-wired stage.
    When parameters are None the stage will create defaults at runtime via the
    SegmentDownloadOrchestrator (preserving current production behavior).

    Args:
        config: Optional Config used to pre-build the SegmentDownloadOrchestrator.
        circuit_breaker: Optional CircuitBreaker instance (default: created by VideoDownloader).
        escalation_manager: Optional EscalationManager instance (default: created by VideoDownloader).
        rate_limit_budget: Optional RateLimitBudget instance (default: created by VideoDownloader).

    Returns:
        A configured DownloadVideoSegmentsStage.
    """
    orchestrator = None
    if config is not None:
        from ..downloader.orchestrator import SegmentDownloadOrchestrator
        orchestrator = SegmentDownloadOrchestrator(config=config)

    return DownloadVideoSegmentsStage(
        orchestrator=orchestrator,
        circuit_breaker=circuit_breaker,
        escalation_manager=escalation_manager,
        rate_limit_budget=rate_limit_budget,
    )
