"""
Pipeline Progress Reporter

Provides centralized progress reporting with stage-level granularity.
Writes a progress.json file to the project directory for external
consumption (e.g., /watch skill, monitoring dashboards).

Usage:
    reporter = ProgressReporter(project_dir)
    reporter.start_stage("DOWNLOAD_SEGMENTS", total_items=50)
    for item in items:
        process(item)
        reporter.update(completed=1)  # or completed=5 for batch
    reporter.finish_stage()
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum interval between progress.json writes (seconds)
_MIN_WRITE_INTERVAL = 5.0


@dataclass
class StageProgress:
    """Snapshot of progress within a single stage."""
    stage_name: str
    items_total: int = 0
    items_completed: int = 0
    items_failed: int = 0
    start_time: float = 0.0


class ProgressReporter:
    """
    Centralized pipeline progress reporter.

    Tracks current stage, item counts, elapsed time, and ETA.
    Writes progress.json to the project directory at most every
    ``_MIN_WRITE_INTERVAL`` seconds to avoid excessive I/O.
    """

    def __init__(self, project_dir: Path | str) -> None:
        self.project_dir = Path(project_dir)
        self.progress_path = self.project_dir / "progress.json"
        self._pipeline_start: float = time.time()

        # Current stage tracking
        self._current: Optional[StageProgress] = None
        self._completed_stages: list[str] = []

        # Write throttle
        self._last_write: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_stage(self, stage_name: str, total_items: int = 0) -> None:
        """Begin tracking a new stage."""
        self._current = StageProgress(
            stage_name=stage_name,
            items_total=total_items,
            start_time=time.time(),
        )
        self._write_progress(force=True)

    def update(
        self,
        completed: int = 0,
        failed: int = 0,
        total: Optional[int] = None,
    ) -> None:
        """
        Report incremental progress within the current stage.

        Args:
            completed: Number of newly completed items (added to running total).
            failed: Number of newly failed items (added to running total).
            total: Optionally update the total item count (e.g., when total
                   becomes known mid-stage).
        """
        if self._current is None:
            return
        self._current.items_completed += completed
        self._current.items_failed += failed
        if total is not None:
            self._current.items_total = total
        self._write_progress()

    def finish_stage(self) -> None:
        """Mark the current stage as finished and reset for the next."""
        if self._current is not None:
            self._completed_stages.append(self._current.stage_name)
            self._current = None
        self._write_progress(force=True)

    def finish_pipeline(self) -> None:
        """Mark the entire pipeline as complete."""
        if self._current is not None:
            self._completed_stages.append(self._current.stage_name)
            self._current = None
        self._write_progress(force=True)

    # ------------------------------------------------------------------
    # Snapshot / ETA
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """
        Build a JSON-serialisable snapshot of current progress.

        Returns dict with keys: current_stage, items_total,
        items_completed, items_failed, elapsed_seconds,
        estimated_remaining_seconds, completed_stages,
        pipeline_elapsed_seconds.
        """
        now = time.time()
        snapshot: dict = {
            "current_stage": None,
            "items_total": 0,
            "items_completed": 0,
            "items_failed": 0,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "completed_stages": list(self._completed_stages),
            "pipeline_elapsed_seconds": round(now - self._pipeline_start, 1),
        }

        if self._current is not None:
            elapsed = now - self._current.start_time
            snapshot["current_stage"] = self._current.stage_name
            snapshot["items_total"] = self._current.items_total
            snapshot["items_completed"] = self._current.items_completed
            snapshot["items_failed"] = self._current.items_failed
            snapshot["elapsed_seconds"] = round(elapsed, 1)
            snapshot["estimated_remaining_seconds"] = self._estimate_remaining(
                self._current.items_completed,
                self._current.items_total,
                elapsed,
            )

        return snapshot

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
