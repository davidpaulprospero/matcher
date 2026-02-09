"""
Pipeline Stage Timing History

Stores per-stage timing data after each pipeline run and provides
duration estimates for upcoming stages based on historical throughput.

History is stored in ``pipeline_history.json`` inside the project
directory so each project maintains its own history.  Only the last
``MAX_RUNS`` entries per stage are retained to keep the file small.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum number of historical runs stored per stage.
MAX_RUNS = 10

# Number of most recent runs used for estimation.
ESTIMATE_WINDOW = 3


def _history_path(project_dir: Path) -> Path:
    """Return the path to pipeline_history.json for a project."""
    return project_dir / "pipeline_history.json"


def load_history(project_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load timing history from disk.

    Returns:
        Mapping of stage_name -> list of run records, each containing
        ``duration``, ``items_processed``, and ``throughput``.
        Returns an empty dict if no history file exists or it is corrupt.
    """
    path = _history_path(project_dir)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug(f"Failed to load pipeline history: {exc}")
        return {}


def save_history(
    project_dir: Path,
    history: Dict[str, List[Dict[str, Any]]],
) -> None:
    """
    Persist timing history to disk.

    Trims each stage to at most ``MAX_RUNS`` entries before writing.
    Uses atomic-ish write (tmp + rename) to avoid corruption.
    """
    # Trim before saving
    trimmed: Dict[str, List[Dict[str, Any]]] = {}
    for stage, runs in history.items():
        trimmed[stage] = runs[-MAX_RUNS:]

    path = _history_path(project_dir)
    tmp_path = path.with_suffix(".json.tmp")
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2)
        if path.exists():
            os.replace(str(tmp_path), str(path))
        else:
            tmp_path.rename(path)
    except OSError as exc:
        logger.debug(f"Failed to save pipeline history: {exc}")


def append_stage_timing(
    project_dir: Path,
    stage_name: str,
    duration: float,
    items_processed: int,
    throughput: float,
) -> None:
    """
    Append a single stage run record to the history file.

    Args:
        project_dir: Project directory.
        stage_name: Name of the completed stage.
        duration: Wall-clock seconds for this stage run.
        items_processed: Number of items processed in this run.
        throughput: Items per second (items_processed / duration).
    """
    history = load_history(project_dir)
    if stage_name not in history:
        history[stage_name] = []
    history[stage_name].append({
        "duration": round(duration, 2),
        "items_processed": items_processed,
        "throughput": round(throughput, 4),
    })
    save_history(project_dir, history)


def estimate_duration(
    project_dir: Path,
    stage_name: str,
    items_count: int,
) -> Optional[float]:
    """
    Estimate duration for an upcoming stage run.

    Uses the average throughput of the last ``ESTIMATE_WINDOW`` runs for
    this stage.  If ``items_count`` is provided and historical throughput
    is available, scales the prediction:

        estimated = items_count / avg_throughput

    If no history exists for the stage, returns ``None`` (graceful
    degradation — caller should skip logging an estimate).

    Args:
        project_dir: Project directory.
        stage_name: Stage about to run.
        items_count: Number of items in the upcoming run.

    Returns:
        Estimated duration in seconds, or None if no history.
    """
    history = load_history(project_dir)
    runs = history.get(stage_name, [])
    if not runs:
        return None

    recent = runs[-ESTIMATE_WINDOW:]

    # Try throughput-based estimation when items_count is positive
    if items_count > 0:
        throughputs = [r["throughput"] for r in recent if r.get("throughput", 0) > 0]
        if throughputs:
            avg_throughput = sum(throughputs) / len(throughputs)
            return round(items_count / avg_throughput, 1)

    # Fall back to average duration when throughput is unavailable or items unknown
    durations = [r["duration"] for r in recent if r.get("duration", 0) > 0]
    if durations:
        return round(sum(durations) / len(durations), 1)

    return None
