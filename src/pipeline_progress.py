"""
Pipeline Progress Reporter

This module provides centralized progress reporting with stage-level granularity
for the video matching pipeline. It tracks progress through pipeline stages,
calculates ETAs, monitors resource usage, and writes progress.json for external
consumption.

Classes:
    StageProgress: Dataclass representing progress within a single pipeline stage.
    ProgressReporter: Main class for tracking and reporting pipeline progress.

Integration with Pipeline Stages:
    The ProgressReporter integrates with pipeline stages via the event system.
    When a stage calls update(), it emits EVENT_STAGE_PROGRESS events that
    other components (like monitoring dashboards) can subscribe to. The reporter
    also calculates pipeline-wide ETA by combining:
    - Current stage throughput (items/second)
    - Historical stage durations from pipeline_history.py

Resource Monitoring:
    When psutil is available, the reporter tracks memory usage, CPU percent,
    disk I/O, and network I/O. This data is included in progress.json and
    emitted via events.

Usage:
    Basic stage tracking:
        reporter = ProgressReporter(project_dir)
        reporter.start_stage("DOWNLOAD_SEGMENTS", total_items=50)
        for item in items:
            process(item)
            reporter.update(completed=1)
        reporter.finish_stage()

    Custom stage tracking (using StageProgress directly):
        progress = StageProgress(
            stage_name="CUSTOM_STAGE",
            items_total=100,
            start_time=time.time()
        )
        # Update manually or integrate with your own loop
        progress.items_completed += 10
        # ... continue processing

    Reading progress from external tools:
        import json
        with open("project/progress.json") as f:
            status = json.load(f)
            print(f"Current stage: {status['current_stage']}")
            print(f"Progress: {status['stage_progress_percent']}%")
            print(f"ETA: {status['pipeline_eta_display']}")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pipeline_history import estimate_duration, estimate_duration_with_confidence, calculate_variance
from .pipeline_events import PipelineEventBus, PipelineEvent, EVENT_STAGE_PROGRESS

logger = logging.getLogger(__name__)

# Minimum interval between progress.json writes (seconds)
_MIN_WRITE_INTERVAL = 5.0

# Default timeout thresholds (in seconds)
_DEFAULT_STUCK_THRESHOLD_SECONDS = 300.0  # 5 minutes without progress = stuck
_DEFAULT_STAGE_DURATION_WARNING_THRESHOLD_SECONDS = 600.0  # 10 minutes = warning

# Global event bus for pipeline events
_event_bus: Optional[PipelineEventBus] = None

# Resource monitoring (lazy import to avoid issues on systems without psutil)
_psutil = None

# Track previous disk/network I/O for delta calculation
_prev_disk_io = None
_prev_net_io = None


def _get_resource_usage() -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Get current memory usage (MB), CPU percent, disk I/O (MB), and network (MB).

    Returns:
        Tuple of (memory_usage_mb, cpu_percent, disk_io_mb, network_mb).
        Returns (None, None, None, None) if unavailable.
    """
    global _psutil, _prev_disk_io, _prev_net_io
    if _psutil is False:
        return None, None, None, None

    if _psutil is None:
        try:
            import psutil
            _psutil = psutil
        except ImportError:
            _psutil = False
            return None, None, None, None

    try:
        process = _psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent(interval=0.1)

        # Get disk I/O (cumulative read/write bytes)
        disk_io_mb = None
        try:
            io_counters = process.io_counters()
            disk_io_mb = round((io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024), 1)
        except Exception:
            pass

        # Get network I/O (cumulative sent/received bytes)
        network_mb = None
        try:
            net_counters = process.net_io_counters()
            network_mb = round((net_counters.bytes_sent + net_counters.bytes_recv) / (1024 * 1024), 1)
        except Exception:
            pass

        return round(memory_mb, 1), round(cpu_percent, 1), disk_io_mb, network_mb
    except Exception:
        return None, None, None, None


def get_event_bus() -> PipelineEventBus:
    """Get or create the global pipeline event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = PipelineEventBus()
    return _event_bus


def set_event_bus(bus: PipelineEventBus) -> None:
    """Set a custom event bus (useful for testing)."""
    global _event_bus
    _event_bus = bus


def _format_duration(seconds: Optional[float]) -> Optional[str]:
    """
    Format duration in seconds as human-readable string.

    Args:
        seconds: Duration in seconds, or None.

    Returns:
        Formatted string like "5m 30s" or "1h 2m", or None if input is None.
    """
    if seconds is None:
        return None
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        return f"{minutes}m"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if remaining_seconds > 0:
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
    if remaining_minutes > 0:
        return f"{hours}h {remaining_minutes}m"
    return f"{hours}h"


@dataclass
class StageProgress:
    """Snapshot of progress within a single pipeline stage.

    This dataclass holds the state for tracking progress through a single
    pipeline stage. It records the stage name, item counts (total, completed,
    failed), and timing information.

    Attributes:
        stage_name: Name identifier for the stage (e.g., "DOWNLOAD_SEGMENTS").
        items_total: Total number of items to process in this stage.
        items_completed: Number of items successfully processed.
        items_failed: Number of items that failed processing.
        start_time: Unix timestamp when the stage started.

    Example:
        Creating a StageProgress for custom stage tracking:

        >>> from src.pipeline_progress import StageProgress
        >>> import time
        >>> progress = StageProgress(
        ...     stage_name="MY_CUSTOM_STAGE",
        ...     items_total=100,
        ...     start_time=time.time()
        ... )
        >>> progress.items_completed = 25
        >>> print(f"Processed {progress.items_completed}/{progress.items_total}")
        Processed 25/100
    """
    stage_name: str
    items_total: int = 0
    items_completed: int = 0
    items_failed: int = 0
    start_time: float = 0.0


class ProgressReporter:
    """Centralized pipeline progress reporter.

    Tracks current stage, item counts, elapsed time, and ETA for the entire
    pipeline. Writes progress.json to the project directory at most every
    5 seconds to avoid excessive I/O while keeping external tools updated.

    The reporter integrates with the pipeline event system, emitting
    EVENT_STAGE_PROGRESS events whenever progress is updated. This allows
    external tools (like monitoring dashboards) to react to progress changes
    in real-time.

    Attributes:
        project_dir: Path to the project directory.
        progress_path: Path to the progress.json file.

    Example:
        >>> from pathlib import Path
        >>> from src.pipeline_progress import ProgressReporter
        >>>
        >>> reporter = ProgressReporter(Path("E:/Projects/MyProject"))
        >>> reporter.set_remaining_stages(["VIDEO_SEARCH", "CAPTION", "MATCH"])
        >>> reporter.start_stage("VIDEO_SEARCH", total_items=100)
        >>> # ... process items ...
        >>> reporter.update(completed=10)
        >>> reporter.finish_stage()
        >>> # Continue with next stages...
        >>> reporter.finish_pipeline()
    """

    def __init__(
        self,
        project_dir: Path | str,
        remaining_stages: List[str] | None = None,
        stuck_threshold_seconds: float = _DEFAULT_STUCK_THRESHOLD_SECONDS,
        duration_warning_threshold_seconds: float = _DEFAULT_STAGE_DURATION_WARNING_THRESHOLD_SECONDS,
        confidence_level: float = 0.95,
    ) -> None:
        """Initialize the progress reporter.

        Args:
            project_dir: Path to the project directory where progress.json
                will be written.
            remaining_stages: Optional list of stage names in execution order.
                Used for pipeline-wide ETA calculation.
            stuck_threshold_seconds: Time in seconds without progress before
                considering a stage stuck (default: 300s = 5 minutes).
            duration_warning_threshold_seconds: Time in seconds before logging
                a warning that the stage is taking longer than expected
                (default: 600s = 10 minutes).
            confidence_level: Confidence level for ETA interval estimation
                (default: 0.95 for 95% confidence interval).
        """
        self.project_dir = Path(project_dir)
        self.progress_path = self.project_dir / "progress.json"
        self._pipeline_start: float = time.time()
        self._confidence_level = confidence_level

        # Current stage tracking
        self._current: Optional[StageProgress] = None
        self._completed_stages: list[str] = []

        # Remaining stages for pipeline-wide ETA (populated dynamically)
        self._remaining_stages: List[str] = remaining_stages or []

        # Write throttle
        self._last_write: float = 0.0

        # Timeout detection
        self._stuck_threshold_seconds = stuck_threshold_seconds
        self._duration_warning_threshold_seconds = duration_warning_threshold_seconds
        self._last_progress_time: float = 0.0
        self._warned_about_duration: bool = False
        self._warned_about_stuck: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_stage(self, stage_name: str, total_items: int = 0) -> None:
        """Begin tracking a new pipeline stage.

        This method initializes progress tracking for a new stage. If a stage
        is already in progress, it will be implicitly finished first.

        Args:
            stage_name: Name identifier for the stage (e.g., "DOWNLOAD_SEGMENTS",
                "MATCH", "CAPTION"). Should match stage names in STAGE_ORDER.
            total_items: Total number of items to process in this stage.
                Can be 0 if unknown initially (will be updated via update()).

        Example:
            >>> reporter.start_stage("CAPTION", total_items=50)
            >>> # Process captions...
            >>> reporter.finish_stage()
        """
        # Reset timeout warnings for new stage
        self.reset_timeout_warnings()
        self._current = StageProgress(
            stage_name=stage_name,
            items_total=total_items,
            start_time=time.time(),
        )
        self._last_progress_time = time.time()
        self._write_progress(force=True)

    def update(
        self,
        completed: int = 0,
        failed: int = 0,
        total: Optional[int] = None,
    ) -> None:
        """Report incremental progress within the current stage.

        This method updates the running totals for completed and failed items.
        It also emits an EVENT_STAGE_PROGRESS event for real-time monitoring
        and triggers a throttled write to progress.json.

        Args:
            completed: Number of newly completed items. This value is ADDED
                to the running total, not set directly. Use completed=1 for
                per-item updates or completed=5 for batch updates.
            failed: Number of newly failed items. Added to the running total
                of failed items.
            total: Optionally update the total item count. Use this when the
                total isn't known at stage start (e.g., when fetching video
                metadata reveals more items than expected).

        Example:
            >>> # Process items one at a time
            >>> for item in items:
            ...     process(item)
            ...     reporter.update(completed=1)
            >>>
            >>> # Process in batches
            >>> batch_size = 10
            >>> for batch in chunks(items, batch_size):
            ...     process_batch(batch)
            ...     reporter.update(completed=batch_size)
            >>>
            >>> # Update total mid-stage
            >>> videos = fetch_video_metadata()
            >>> reporter.update(total=len(videos))
        """
        if self._current is None:
            return
        self._current.items_completed += completed
        self._current.items_failed += failed
        if total is not None:
            self._current.items_total = total

        # Track last progress time (when items actually completed/failed)
        if completed > 0 or failed > 0:
            self._last_progress_time = time.time()
            # Reset stuck warning when progress is made
            self._warned_about_stuck = False

        # Emit stage progress event
        self._emit_progress_event()

        self._write_progress()

    def finish_stage(self) -> None:
        """Mark the current stage as finished and prepare for the next stage.

        This method records the completed stage name and resets internal state
        for tracking the next stage. The stage name is added to the
        completed_stages list in the progress snapshot.

        Note: This does NOT need to be called before start_stage() when moving
        directly to a new stage - start_stage() will automatically finish
        any in-progress stage.

        Example:
            >>> reporter.start_stage("MATCH", total_items=200)
            >>> # ... process matches ...
            >>> reporter.finish_stage()
            >>> reporter.start_stage("DOWNLOAD_SEGMENTS", total_items=150)
        """
        if self._current is not None:
            self._completed_stages.append(self._current.stage_name)
            self._current = None
        self._write_progress(force=True)

    def finish_pipeline(self) -> None:
        """Mark the entire pipeline as complete.

        This should be called when all pipeline stages have finished. It
        records any in-progress stage as completed and writes a final
        progress snapshot with pipeline_elapsed_seconds set.

        Example:
            >>> # After all stages are done
            >>> reporter.finish_pipeline()
            >>> # progress.json now shows pipeline complete
        """
        if self._current is not None:
            self._completed_stages.append(self._current.stage_name)
            self._current = None
        self._write_progress(force=True)

    def set_remaining_stages(self, stages: List[str]) -> None:
        """Set the ordered list of remaining stages for pipeline-wide ETA.

        This method configures the full pipeline stage order, enabling
        accurate ETA calculation that includes not just the current stage
        but all subsequent stages. This uses historical data from
        pipeline_history.py to estimate durations.

        Args:
            stages: Ordered list of stage names that remain to be executed.
                Should match the order in checkpoint.STAGE_ORDER.

        Example:
            >>> reporter.set_remaining_stages([
            ...     "VIDEO_SEARCH",
            ...     "CAPTION",
            ...     "MATCH",
            ...     "DOWNLOAD_SEGMENTS",
            ...     "OUTPUT"
            ... ])
        """
        self._remaining_stages = list(stages)

    def set_items_for_current_stage(self, items_count: int) -> None:
        """Set item count for current stage to improve ETA accuracy.

        Use this when the total item count isn't known at stage start but
        becomes available during stage execution. This allows the ETA
        calculation to be more accurate.

        Args:
            items_count: The total number of items to process in the
                current stage.

        Example:
            >>> reporter.start_stage("CAPTION")  # Unknown total initially
            >>> videos = fetch_all_videos()
            >>> reporter.set_items_for_current_stage(len(videos))
        """
        if self._current is not None:
            self._current.items_total = items_count

    def _emit_progress_event(self) -> None:
        """Emit EVENT_STAGE_PROGRESS event with current progress data."""
        if self._current is None:
            return

        try:
            bus = get_event_bus()
            # Calculate progress percentage
            pct = None
            if self._current.items_total > 0:
                pct = (self._current.items_completed / self._current.items_total) * 100

            # Get resource usage
            memory_mb, cpu_pct, disk_io_mb, network_mb = _get_resource_usage()

            event = PipelineEvent(
                event_type=EVENT_STAGE_PROGRESS,
                stage_name=self._current.stage_name,
                timestamp=time.time(),
                data={
                    "items_completed": self._current.items_completed,
                    "items_total": self._current.items_total,
                    "progress_percent": round(pct, 1) if pct is not None else None,
                    "items_failed": self._current.items_failed,
                    "memory_usage_mb": memory_mb,
                    "cpu_percent": cpu_pct,
                    "disk_io_mb": disk_io_mb,
                    "network_mb": network_mb,
                },
            )
            bus.emit(event)
        except Exception as e:
            logger.debug(f"Failed to emit progress event: {e}")

    # ------------------------------------------------------------------
    # Snapshot / ETA
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """
        Build a JSON-serialisable snapshot of current progress.

        Returns dict with keys: current_stage, items_total, items_completed, items_failed, elapsed_seconds,
        estimated_remaining_seconds, pipeline_eta_seconds, pipeline_eta_display, completed_stages,
        pipeline_elapsed_seconds, stage_progress_percent, memory_usage_mb, cpu_percent, disk_io_mb, network_mb.
        """
        now = time.time()

        # Get resource usage
        memory_mb, cpu_pct, disk_io_mb, network_mb = _get_resource_usage()

        snapshot: dict = {
            "current_stage": None,
            "items_total": 0,
            "items_completed": 0,
            "items_failed": 0,
            "stage_progress_percent": None,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "pipeline_eta_seconds": None,
            "pipeline_eta_lower": None,
            "pipeline_eta_upper": None,
            "pipeline_eta_confidence_level": None,
            "pipeline_eta_has_confidence_interval": False,
            "pipeline_eta_display": None,
            "pipeline_eta_by_stage": [],  # List of {stage, eta_seconds, eta_display}
            "completed_stages": list(self._completed_stages),
            "remaining_stages": list(self._remaining_stages),
            "pipeline_elapsed_seconds": round(now - self._pipeline_start, 1),
            "memory_usage_mb": memory_mb,
            "cpu_percent": cpu_pct,
            "disk_io_mb": disk_io_mb,
            "network_mb": network_mb,
            # Timeout detection fields
            "seconds_since_last_progress": None,
            "is_stage_stuck": False,
            "is_duration_warning": False,
        }

        if self._current is not None:
            elapsed = now - self._current.start_time
            snapshot["current_stage"] = self._current.stage_name
            snapshot["items_total"] = self._current.items_total
            snapshot["items_completed"] = self._current.items_completed
            snapshot["items_failed"] = self._current.items_failed
            snapshot["elapsed_seconds"] = round(elapsed, 1)

            # Calculate stage progress percentage
            if self._current.items_total > 0:
                pct = (self._current.items_completed / self._current.items_total) * 100
                snapshot["stage_progress_percent"] = round(pct, 1)

            snapshot["estimated_remaining_seconds"] = self._estimate_remaining(
                self._current.items_completed,
                self._current.items_total,
                elapsed,
            )

            # Calculate pipeline-wide ETA using historical data with confidence intervals
            pipeline_eta_ci = self._calculate_pipeline_eta_with_confidence()
            if pipeline_eta_ci is not None:
                snapshot["pipeline_eta_seconds"] = pipeline_eta_ci["point_estimate"]
                snapshot["pipeline_eta_lower"] = pipeline_eta_ci["lower_bound"]
                snapshot["pipeline_eta_upper"] = pipeline_eta_ci["upper_bound"]
                snapshot["pipeline_eta_confidence_level"] = pipeline_eta_ci["confidence_level"]
                snapshot["pipeline_eta_has_confidence_interval"] = pipeline_eta_ci["has_confidence_interval"]
                # Add stage-level ETA breakdown
                snapshot["pipeline_eta_by_stage"] = pipeline_eta_ci.get("by_stage", [])
                # Display as range if we have confidence interval
                if pipeline_eta_ci["has_confidence_interval"]:
                    lower_str = _format_duration(pipeline_eta_ci["lower_bound"])
                    upper_str = _format_duration(pipeline_eta_ci["upper_bound"])
                    snapshot["pipeline_eta_display"] = f"{lower_str} - {upper_str}"
                else:
                    snapshot["pipeline_eta_display"] = _format_duration(pipeline_eta_ci["point_estimate"])
            else:
                # Fallback to simple ETA
                pipeline_eta = self._calculate_pipeline_eta()
                snapshot["pipeline_eta_seconds"] = pipeline_eta
                snapshot["pipeline_eta_display"] = _format_duration(pipeline_eta)

            # Add timeout detection info
            if self._last_progress_time > 0:
                snapshot["seconds_since_last_progress"] = round(now - self._last_progress_time, 1)
            else:
                snapshot["seconds_since_last_progress"] = round(elapsed, 1)

            # Check if stage appears stuck (warning already logged)
            if not self._warned_about_stuck and snapshot["seconds_since_last_progress"] >= self._stuck_threshold_seconds:
                snapshot["is_stage_stuck"] = True

            # Check if duration warning threshold exceeded
            if not self._warned_about_duration and elapsed >= self._duration_warning_threshold_seconds:
                snapshot["is_duration_warning"] = True

        return snapshot

    def _calculate_pipeline_eta(self) -> Optional[float]:
        """
        Calculate pipeline-wide ETA based on historical stage durations.

        Uses historical data from pipeline_history.py to estimate remaining
        time for all stages (including current stage's remaining work and
        all subsequent stages).

        Returns:
            Estimated total seconds remaining for pipeline, or None if
            no historical data is available.
        """
        if not self._remaining_stages:
            return None

        total_estimated = 0.0
        has_any_history = False

        # Current stage: use throughput-based ETA if available, else historical
        if self._current is not None:
            current_stage = self._current.stage_name
            items_total = self._current.items_total
            items_completed = self._current.items_completed

            # First try throughput-based for current stage
            if items_completed > 0 and items_total > 0:
                elapsed = time.time() - self._current.start_time
                remaining_for_stage = self._estimate_remaining(
                    items_completed, items_total, elapsed
                )
                if remaining_for_stage is not None:
                    total_estimated += remaining_for_stage
                    has_any_history = True

            # If no throughput data, try historical estimate
            if total_estimated == 0.0:
                hist_est = self._get_historical_estimate(
                    current_stage, items_total
                )
                if hist_est is not None:
                    total_estimated += hist_est
                    has_any_history = True

        # Remaining stages after current: use historical data
        stages_after_current = False
        for stage in self._remaining_stages:
            if self._current is not None and stage == self._current.stage_name:
                stages_after_current = True
                continue
            if stages_after_current or self._current is None:
                hist_est = self._get_historical_estimate(stage, 0)
                if hist_est is not None:
                    total_estimated += hist_est
                    has_any_history = True

        if has_any_history:
            return round(total_estimated, 1)
        return None

    def _calculate_pipeline_eta_with_confidence(self) -> Optional[dict]:
        """
        Calculate pipeline-wide ETA with confidence interval.

        Uses historical data from pipeline_history.py to estimate remaining
        time for all stages with confidence intervals.

        Returns:
            Dict with point_estimate, lower_bound, upper_bound, confidence_level,
            margin_of_error, or None if no historical data is available.
        """
        if not self._remaining_stages:
            return None

        # Collect confidence intervals for all stages
        stage_estimates = []

        # Current stage: use throughput-based ETA if available, else historical
        if self._current is not None:
            current_stage = self._current.stage_name
            items_total = self._current.items_total
            items_completed = self._current.items_completed

            # First try throughput-based for current stage
            point_estimate = None
            lower_bound = None
            upper_bound = None
            has_ci = False

            if items_completed > 0 and items_total > 0:
                elapsed = time.time() - self._current.start_time
                remaining_for_stage = self._estimate_remaining(
                    items_completed, items_total, elapsed
                )
                if remaining_for_stage is not None:
                    point_estimate = remaining_for_stage
                    # For throughput-based, use 20% uncertainty as default
                    margin = point_estimate * 0.20
                    lower_bound = max(0, point_estimate - margin)
                    upper_bound = point_estimate + margin
                    has_ci = True

            # If no throughput data, try historical confidence interval
            if point_estimate is None:
                ci_result = estimate_duration_with_confidence(
                    self.project_dir,
                    current_stage,
                    items_total,
                    self._confidence_level,
                )
                if ci_result is not None:
                    point_estimate = ci_result["point_estimate"]
                    lower_bound = ci_result["lower_bound"]
                    upper_bound = ci_result["upper_bound"]
                    has_ci = ci_result["has_confidence_interval"]

            if point_estimate is not None:
                stage_estimates.append({
                    "stage": current_stage,
                    "point": point_estimate,
                    "lower": lower_bound,
                    "upper": upper_bound,
                    "has_ci": has_ci,
                })

        # Remaining stages after current: use historical data with confidence intervals
        stages_after_current = False
        for stage in self._remaining_stages:
            if self._current is not None and stage == self._current.stage_name:
                stages_after_current = True
                continue
            if stages_after_current or self._current is None:
                ci_result = estimate_duration_with_confidence(
                    self.project_dir, stage, 0, self._confidence_level
                )
                if ci_result is not None:
                    stage_estimates.append({
                        "stage": stage,
                        "point": ci_result["point_estimate"],
                        "lower": ci_result["lower_bound"],
                        "upper": ci_result["upper_bound"],
                        "has_ci": ci_result["has_confidence_interval"],
                    })

        if not stage_estimates:
            return None

        # Sum up point estimates and confidence interval bounds
        total_point = sum(s["point"] for s in stage_estimates)
        total_lower = sum(s["lower"] for s in stage_estimates)
        total_upper = sum(s["upper"] for s in stage_estimates)
        any_has_ci = any(s["has_ci"] for s in stage_estimates)

        # Build the breakdown list for stage-level ETA
        eta_by_stage = []
        for s in stage_estimates:
            stage_info = {
                "stage": s["stage"],
                "eta_seconds": s["point"],
                "eta_display": _format_duration(s["point"]),
            }
            if s["has_ci"]:
                stage_info["eta_lower"] = s["lower"]
                stage_info["eta_upper"] = s["upper"]
                stage_info["eta_display"] = f"{_format_duration(s['lower'])} - {_format_duration(s['upper'])}"
            eta_by_stage.append(stage_info)

        return {
            "point_estimate": round(total_point, 1),
            "lower_bound": round(total_lower, 1),
            "upper_bound": round(total_upper, 1),
            "confidence_level": self._confidence_level,
            "margin_of_error": round((total_upper - total_lower) / 2, 1),
            "has_confidence_interval": any_has_ci,
            "by_stage": eta_by_stage,
        }

    def _get_historical_estimate(
        self, stage_name: str, items_count: int
    ) -> Optional[float]:
        """
        Get historical duration estimate for a stage.

        Args:
            stage_name: Name of the stage.
            items_count: Number of items (for throughput-based estimation).

        Returns:
            Estimated duration in seconds, or None if no history available.
        """
        try:
            return estimate_duration(self.project_dir, stage_name, items_count)
        except Exception:
            return None

    @staticmethod
    def _estimate_remaining(
        completed: int, total: int, elapsed: float
    ) -> Optional[float]:
        """
        Estimate remaining seconds based on throughput.

        Returns None when estimation is not possible (no items
        completed yet or total is unknown/zero).
        """
        if completed <= 0 or total <= 0 or elapsed <= 0:
            return None
        remaining_items = total - completed
        if remaining_items <= 0:
            return 0.0
        throughput = completed / elapsed  # items per second
        return round(remaining_items / throughput, 1)

    # ------------------------------------------------------------------
    # Timeout Detection
    # ------------------------------------------------------------------

    def check_timeout(self) -> Optional[dict]:
        """Check for timeout conditions and return diagnostic info.

        This method should be called periodically (e.g., every 30 seconds)
        to check if the current stage is stuck or taking too long.

        Returns:
            Dict with timeout diagnostics or None if no timeout detected.
            Dict contains: is_stuck (bool), is_duration_warning (bool),
            seconds_since_last_progress (float), stage_elapsed_seconds (float).
        """
        if self._current is None:
            return None

        now = time.time()
        stage_elapsed = now - self._current.start_time
        seconds_since_progress = now - self._last_progress_time if self._last_progress_time > 0 else stage_elapsed

        result = {
            "is_stuck": False,
            "is_duration_warning": False,
            "seconds_since_last_progress": round(seconds_since_progress, 1),
            "stage_elapsed_seconds": round(stage_elapsed, 1),
            "stage_name": self._current.stage_name,
        }

        # Check if stuck (no progress for threshold)
        if not self._warned_about_stuck and seconds_since_progress >= self._stuck_threshold_seconds:
            result["is_stuck"] = True
            logger.warning(
                f"Stage '{self._current.stage_name}' appears stuck: "
                f"no progress for {seconds_since_progress:.0f}s "
                f"(threshold: {self._stuck_threshold_seconds}s). "
                f"Items completed: {self._current.items_completed}/{self._current.items_total}"
            )
            self._warned_about_stuck = True

        # Check if duration warning threshold exceeded
        if not self._warned_about_duration and stage_elapsed >= self._duration_warning_threshold_seconds:
            result["is_duration_warning"] = True
            logger.warning(
                f"Stage '{self._current.stage_name}' exceeded expected duration: "
                f"{stage_elapsed:.0f}s (warning threshold: {self._duration_warning_threshold_seconds}s). "
                f"Items: {self._current.items_completed}/{self._current.items_total}"
            )
            self._warned_about_duration = True

        return result

    def reset_timeout_warnings(self) -> None:
        """Reset timeout warning flags for a new stage."""
        self._warned_about_duration = False
        self._warned_about_stuck = False

    def get_timeout_status(self) -> dict:
        """Get current timeout detection status.

        Returns:
            Dict with stuck_threshold_seconds, duration_warning_threshold_seconds,
            last_progress_time, and warned flags.
        """
        return {
            "stuck_threshold_seconds": self._stuck_threshold_seconds,
            "duration_warning_threshold_seconds": self._duration_warning_threshold_seconds,
            "last_progress_time": self._last_progress_time,
            "warned_about_duration": self._warned_about_duration,
            "warned_about_stuck": self._warned_about_stuck,
        }

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _write_progress(self, force: bool = False) -> None:
        """Write progress.json, throttled to at most every _MIN_WRITE_INTERVAL."""
        now = time.time()
        if not force and (now - self._last_write) < _MIN_WRITE_INTERVAL:
            return

        self._last_write = now
        snapshot = self.get_snapshot()

        try:
            self.project_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self.progress_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            # Atomic-ish rename (safe on Windows for same-dir)
            if self.progress_path.exists():
                os.replace(str(tmp_path), str(self.progress_path))
            else:
                tmp_path.rename(self.progress_path)
        except OSError as exc:
            logger.debug(f"Failed to write progress.json: {exc}")
