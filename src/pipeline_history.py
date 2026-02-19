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
    trimmed: Dict[str, Any] = {}
    for stage, runs in history.items():
        if isinstance(runs, list):
            # Stage timing data (list of runs)
            trimmed[stage] = runs[-MAX_RUNS:]
        elif isinstance(runs, dict):
            # Resource usage data (dict of stage -> list of records)
            trimmed[stage] = {}
            for sub_stage, records in runs.items():
                if isinstance(records, list):
                    trimmed[stage][sub_stage] = records[-MAX_RESOURCE_HISTORY:]
                else:
                    trimmed[stage][sub_stage] = records
        else:
            trimmed[stage] = runs

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


# T-distribution critical values for common confidence levels (df = degrees of freedom)
# Approximation for common degrees of freedom (1-10, then infinity for large samples)
_T_DISTRIBUTION = {
    # df: {confidence_level: t_value}
    1: {0.80: 3.078, 0.90: 6.314, 0.95: 12.706, 0.99: 63.657},
    2: {0.80: 1.886, 0.90: 2.920, 0.95: 4.303, 0.99: 9.925},
    3: {0.80: 1.638, 0.90: 2.353, 0.95: 3.182, 0.99: 5.841},
    4: {0.80: 1.533, 0.90: 2.132, 0.95: 2.776, 0.99: 4.604},
    5: {0.80: 1.476, 0.90: 2.015, 0.95: 2.571, 0.99: 4.032},
    6: {0.80: 1.440, 0.90: 1.943, 0.95: 2.447, 0.99: 3.707},
    7: {0.80: 1.415, 0.90: 1.895, 0.95: 2.365, 0.99: 3.499},
    8: {0.80: 1.397, 0.90: 1.860, 0.95: 2.306, 0.99: 3.355},
    9: {0.80: 1.383, 0.90: 1.833, 0.95: 2.262, 0.99: 3.250},
    10: {0.80: 1.372, 0.90: 1.812, 0.95: 2.228, 0.99: 3.169},
    # For df > 10, use z-score approximation (close enough for our purposes)
    11: {0.80: 1.363, 0.90: 1.796, 0.95: 2.201, 0.99: 3.106},
}


def _get_t_critical(confidence_level: float, n_samples: int) -> float:
    """
    Get t-distribution critical value for given confidence level and sample size.

    Args:
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        n_samples: Number of samples (determines degrees of freedom)

    Returns:
        Critical t-value
    """
    df = min(max(n_samples - 1, 1), 10)  # Clamp to our lookup table

    # Get closest confidence level in our table
    levels = [0.80, 0.90, 0.95, 0.99]
    closest = min(levels, key=lambda x: abs(x - confidence_level))

    if df in _T_DISTRIBUTION:
        return _T_DISTRIBUTION[df].get(closest, 1.96)  # Default to z-score

    # Fallback to normal distribution z-score
    if confidence_level >= 0.99:
        return 2.576
    elif confidence_level >= 0.95:
        return 1.96
    elif confidence_level >= 0.90:
        return 1.645
    else:
        return 1.282


def calculate_variance(
    project_dir: Path,
    stage_name: str,
) -> Optional[dict]:
    """
    Calculate variance statistics for a stage's historical timing.

    Args:
        project_dir: Project directory.
        stage_name: Stage name to analyze.

    Returns:
        Dict with mean, std_dev, sample_count, or None if insufficient data.
    """
    history = load_history(project_dir)
    runs = history.get(stage_name, [])
    recent = runs[-ESTIMATE_WINDOW:]

    if len(recent) < 2:
        return None

    # Calculate variance on durations
    durations = [r["duration"] for r in recent if r.get("duration", 0) > 0]
    if len(durations) < 2:
        return None

    mean_duration = sum(durations) / len(durations)
    variance = sum((d - mean_duration) ** 2 for d in durations) / (len(durations) - 1)
    std_dev = variance ** 0.5

    # Also calculate throughput variance if available
    throughputs = [r["throughput"] for r in recent if r.get("throughput", 0) > 0]
    throughput_std_dev = None
    if len(throughputs) >= 2:
        mean_throughput = sum(throughputs) / len(throughputs)
        tp_variance = sum((t - mean_throughput) ** 2 for t in throughputs) / (len(throughputs) - 1)
        throughput_std_dev = tp_variance ** 0.5

    return {
        "mean_duration": round(mean_duration, 2),
        "std_dev": round(std_dev, 2),
        "sample_count": len(durations),
        "throughput_std_dev": round(throughput_std_dev, 4) if throughput_std_dev else None,
    }


def estimate_duration_with_confidence(
    project_dir: Path,
    stage_name: str,
    items_count: int,
    confidence_level: float = 0.95,
) -> Optional[dict]:
    """
    Estimate duration with confidence interval.

    Args:
        project_dir: Project directory.
        stage_name: Stage name.
        items_count: Number of items in the upcoming run.
        confidence_level: Confidence level (default 0.95 for 95% CI).

    Returns:
        Dict with 'point_estimate', 'lower_bound', 'upper_bound', 'confidence_level',
        'margin_of_error', or None if insufficient data for confidence interval.
    """
    history = load_history(project_dir)
    runs = history.get(stage_name, [])
    recent = runs[-ESTIMATE_WINDOW:]

    if len(recent) < 2:
        # Not enough data for confidence interval - return point estimate only
        point = estimate_duration(project_dir, stage_name, items_count)
        if point is None:
            return None
        return {
            "point_estimate": point,
            "lower_bound": point,
            "upper_bound": point,
            "confidence_level": confidence_level,
            "margin_of_error": 0.0,
            "has_confidence_interval": False,
        }

    # Calculate point estimate first
    point_estimate = estimate_duration(project_dir, stage_name, items_count)
    if point_estimate is None:
        return None

    # Calculate standard deviation
    variance_stats = calculate_variance(project_dir, stage_name)
    if variance_stats is None:
        return {
            "point_estimate": point_estimate,
            "lower_bound": point_estimate,
            "upper_bound": point_estimate,
            "confidence_level": confidence_level,
            "margin_of_error": 0.0,
            "has_confidence_interval": False,
        }

    # Calculate confidence interval using t-distribution
    std_dev = variance_stats["std_dev"]
    n = variance_stats["sample_count"]
    t_critical = _get_t_critical(confidence_level, n)

    # Standard error of the mean
    std_error = std_dev / (n ** 0.5)
    margin_of_error = t_critical * std_error

    lower_bound = max(0, point_estimate - margin_of_error)
    upper_bound = point_estimate + margin_of_error

    return {
        "point_estimate": point_estimate,
        "lower_bound": round(lower_bound, 1),
        "upper_bound": round(upper_bound, 1),
        "confidence_level": confidence_level,
        "margin_of_error": round(margin_of_error, 1),
        "has_confidence_interval": True,
    }


# US-138-009: Resource usage tracking and prediction functions

# Maximum number of historical resource records stored per project
MAX_RESOURCE_HISTORY = 20

# Number of most recent runs used for prediction averaging
RESOURCE_PREDICTION_WINDOW = 5


def append_resource_usage(
    project_dir: Path,
    stage_name: str,
    memory_mb: Optional[float],
    cpu_percent: Optional[float],
    voiceover_duration_seconds: Optional[float] = None,
    video_count: Optional[int] = None,
    segment_count: Optional[int] = None,
) -> None:
    """
    Append a resource usage record to history after a stage completes.

    Args:
        project_dir: Project directory.
        stage_name: Name of the completed stage.
        memory_mb: Peak memory usage in MB.
        cpu_percent: Average CPU usage percentage.
        voiceover_duration_seconds: Voiceover duration in seconds (for prediction).
        video_count: Number of videos processed (for prediction).
        segment_count: Number of segments processed (for prediction).
    """
    history = load_history(project_dir)

    # Get or create resource tracking section
    if "resource_usage" not in history:
        history["resource_usage"] = {}

    if stage_name not in history["resource_usage"]:
        history["resource_usage"][stage_name] = []

    record = {}
    if memory_mb is not None:
        record["memory_mb"] = round(memory_mb, 1)
    if cpu_percent is not None:
        record["cpu_percent"] = round(cpu_percent, 1)
    if voiceover_duration_seconds is not None:
        record["voiceover_duration_seconds"] = round(voiceover_duration_seconds, 1)
    if video_count is not None:
        record["video_count"] = video_count
    if segment_count is not None:
        record["segment_count"] = segment_count

    if record:
        history["resource_usage"][stage_name].append(record)

        # Trim to max history
        history["resource_usage"][stage_name] = history["resource_usage"][stage_name][-MAX_RESOURCE_HISTORY:]

        save_history(project_dir, history)


def get_resource_history(
    project_dir: Path,
    stage_name: Optional[str] = None,
    window: int = RESOURCE_PREDICTION_WINDOW,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get resource usage history from history file.

    Args:
        project_dir: Project directory.
        stage_name: Optional stage name to filter (returns all if None).
        window: Number of recent records to return (default 5).

    Returns:
        Dict mapping stage_name -> list of resource usage records.
    """
    history = load_history(project_dir)
    resource_history = history.get("resource_usage", {})

    if stage_name:
        records = resource_history.get(stage_name, [])
        return {stage_name: records[-window:]}

    # Return all stages with window applied
    result = {}
    for stage, records in resource_history.items():
        result[stage] = records[-window:]
    return result


def predict_memory_usage(
    project_dir: Path,
    voiceover_duration_seconds: float,
    video_count: int,
    segment_count: int,
    base_memory_per_vo_minute: float = 50.0,
    memory_per_video: float = 5.0,
    memory_per_segment: float = 2.0,
    history_window: int = RESOURCE_PREDICTION_WINDOW,
) -> Optional[Dict[str, Any]]:
    """
    Predict memory requirements for upcoming pipeline run.

    Uses a combination of:
    1. Historical average memory per stage from past runs
    2. Formula-based estimate using voiceover duration, video count, segment count

    Args:
        project_dir: Project directory.
        voiceover_duration_seconds: Total voiceover duration in seconds.
        video_count: Number of videos to process.
        segment_count: Number of voiceover segments.
        base_memory_per_vo_minute: Base memory per voiceover minute (MB).
        memory_per_video: Additional memory per video (MB).
        memory_per_segment: Additional memory per segment (MB).
        history_window: Number of historical runs to consider.

    Returns:
        Dict with prediction details:
        - predicted_memory_mb: Predicted peak memory in MB
        - historical_avg_mb: Average from historical data
        - formula_based_mb: Formula-based estimate
        - confidence: Prediction confidence (low/medium/high)
        - sample_count: Number of historical samples used
        - sources: List of data sources used
    """
    # Get historical data
    resource_history = get_resource_history(project_dir, window=history_window)

    # Calculate historical average memory usage
    all_memory_values = []
    for stage, records in resource_history.items():
        for record in records:
            if "memory_mb" in record:
                all_memory_values.append(record["memory_mb"])

    historical_avg_mb = None
    if all_memory_values:
        historical_avg_mb = sum(all_memory_values) / len(all_memory_values)

    # Calculate formula-based estimate
    voiceover_minutes = voiceover_duration_seconds / 60.0
    formula_based_mb = (
        (voiceover_minutes * base_memory_per_vo_minute) +
        (video_count * memory_per_video) +
        (segment_count * memory_per_segment)
    )

    # Determine predicted value - prefer historical if available
    predicted_memory_mb = None
    sources = []

    if historical_avg_mb is not None and len(all_memory_values) >= 3:
        # Use weighted average: 60% historical, 40% formula
        predicted_memory_mb = (historical_avg_mb * 0.6) + (formula_based_mb * 0.4)
        sources = ["historical", "formula"]
        confidence = "high" if len(all_memory_values) >= 5 else "medium"
    elif historical_avg_mb is not None:
        # Only have limited historical data
        predicted_memory_mb = (historical_avg_mb * 0.4) + (formula_based_mb * 0.6)
        sources = ["historical", "formula"]
        confidence = "low"
    else:
        # No historical data, use formula only
        predicted_memory_mb = formula_based_mb
        sources = ["formula"]
        confidence = "low"

    # Round predictions
    predicted_memory_mb = round(predicted_memory_mb, 1)
    if historical_avg_mb:
        historical_avg_mb = round(historical_avg_mb, 1)
    formula_based_mb = round(formula_based_mb, 1)

    return {
        "predicted_memory_mb": predicted_memory_mb,
        "historical_avg_mb": historical_avg_mb,
        "formula_based_mb": formula_based_mb,
        "confidence": confidence,
        "sample_count": len(all_memory_values),
        "sources": sources,
        "inputs": {
            "voiceover_duration_seconds": round(voiceover_duration_seconds, 1),
            "voiceover_minutes": round(voiceover_minutes, 2),
            "video_count": video_count,
            "segment_count": segment_count,
        },
    }


def get_stage_resource_stats(
    project_dir: Path,
    stage_name: str,
    window: int = RESOURCE_PREDICTION_WINDOW,
) -> Optional[Dict[str, Any]]:
    """
    Get resource usage statistics for a specific stage.

    Args:
        project_dir: Project directory.
        stage_name: Stage name to get stats for.
        window: Number of recent records to consider.

    Returns:
        Dict with avg_memory_mb, max_memory_mb, avg_cpu_percent, sample_count,
        or None if no data available.
    """
    resource_history = get_resource_history(project_dir, stage_name=stage_name, window=window)
    records = resource_history.get(stage_name, [])

    if not records:
        return None

    memory_values = [r["memory_mb"] for r in records if "memory_mb" in r]
    cpu_values = [r["cpu_percent"] for r in records if "cpu_percent" in r]

    if not memory_values:
        return None

    return {
        "avg_memory_mb": round(sum(memory_values) / len(memory_values), 1),
        "max_memory_mb": max(memory_values),
        "min_memory_mb": min(memory_values),
        "avg_cpu_percent": round(sum(cpu_values) / len(cpu_values), 1) if cpu_values else None,
        "max_cpu_percent": max(cpu_values) if cpu_values else None,
        "sample_count": len(records),
    }
