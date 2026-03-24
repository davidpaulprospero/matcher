#!/usr/bin/env python3
"""Pipeline watch script - monitors a running pipeline with duration-based auto-restart.

Cross-platform replacement for watch_auto.ps1.
Started as Claude Code background task.
Loops: check pipeline -> sleep interval -> repeat until duration expires.
Exits when duration expires OR pipeline completes OR error.
Claude Code will detect exit and auto-restart.
"""

from __future__ import annotations

import argparse
import gzip
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Standard scripts/ import pattern
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)

QUEUE_STOP_FILE = project_root / "Degold" / "queue_stop.txt"
CHECK_INTERVAL = 60  # seconds
STALL_THRESHOLD = 10  # minutes of inactivity before considering stalled
STAGE_STARTED_RE = re.compile(r"\[(\w+)\]\s+Stage started")


def read_checkpoint(path: str) -> dict | None:
    """Read a checkpoint file, handling both gzip and plain JSON."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            header = f.read(2)
            f.seek(0)
            raw = f.read()
        if header == b"\x1f\x8b":
            text = gzip.decompress(raw).decode("utf-8", errors="replace")
        else:
            text = raw.decode("utf-8", errors="replace")
        return json.loads(text)
    except Exception:
        return None


def find_latest_log(project_path: str) -> str | None:
    """Find the most recent run_*.log in the project's logs directory."""
    logs_dir = os.path.join(project_path, "logs")
    if not os.path.isdir(logs_dir):
        return None
    matches = glob.glob(os.path.join(logs_dir, "run_*.log"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def check_queue_stop() -> bool:
    """Check if the unified queue stop file signals a stop."""
    if not QUEUE_STOP_FILE.exists():
        return False
    try:
        content = QUEUE_STOP_FILE.read_text(encoding="utf-8", errors="replace").strip().lower()
        return content in ("true", "1", "stop")
    except Exception:
        return False


def count_pattern_in_file(filepath: str, pattern: str) -> int:
    """Count occurrences of a pattern in a file, line by line."""
    count = 0
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if pattern in line:
                    count += 1
    except Exception:
        pass
    return count


def read_tail_lines(filepath: str, n: int) -> list[str]:
    """Read the last n lines of a file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [line.rstrip("\n\r") for line in lines[-n:]]
    except Exception:
        return []


def get_current_stage_from_log(log_path: str) -> str:
    """Find the last 'Stage started' line in a log file to determine current stage."""
    stage = "UNKNOWN"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = STAGE_STARTED_RE.search(line)
                if m:
                    stage = m.group(1)
    except Exception:
        pass
    return stage


def write_status_md(
    project_path: str,
    iteration: int,
    stage: str,
    video_count: str,
    stream_states: int,
    captions_fetched: int,
    last_activity: str,
    time_since_write: str,
    elapsed_str: str,
    duration: int,
    remaining_str: str,
    log_content: str,
) -> None:
    """Write the PIPELINE_STATUS.md file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""# Pipeline Status

**Last Updated:** {timestamp}
**Iteration:** {iteration}
**Current Stage:** {stage}
**Elapsed:** {elapsed_str} / {duration}min (remaining: {remaining_str})

## Progress

| Metric | Count |
|--------|-------|
| Stage | {stage} |
| Video Candidates | {video_count} |
| Stream States Checked | {stream_states} |
| Captions Fetched | {captions_fetched} |
| Last Activity | {last_activity} |
| Time Since Last Write | {time_since_write} |

## Recent Activity (last 30 log lines)

```
{log_content}
```

## Status

Background watch active - checks every {CHECK_INTERVAL}s, duration {duration}min.
"""
    status_file = os.path.join(project_path, "PIPELINE_STATUS.md")
    try:
        with open(status_file, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline watch - monitors a running pipeline")
    parser.add_argument("--project-path", required=True, help="Path to project directory")
    parser.add_argument("--duration", type=int, default=5, help="Total duration in minutes before exit (default: 5)")
    args = parser.parse_args()

    project_path = args.project_path
    duration = args.duration

    # Write running marker
    running_marker = os.path.join(project_path, "watch_running.txt")
    status_marker = os.path.join(project_path, "watch_status.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(running_marker, "w", encoding="utf-8") as f:
            f.write(f"running|{timestamp}")
    except Exception:
        pass

    iteration = 0
    start_time = time.time()

    print(f"[Watch] Starting: project={project_path} duration={duration}min interval={CHECK_INTERVAL}s", flush=True)

    try:
        while True:
            iteration += 1

            # Check for queue stop file
            if check_queue_stop():
                print(f"WATCH_EXIT_QUEUE_STOP iteration={iteration}", flush=True)
                break

            # Check if duration has expired
            elapsed = time.time() - start_time
            elapsed_minutes = elapsed / 60.0
            if elapsed_minutes >= duration:
                print(
                    f"WATCH_EXIT_DURATION_EXPIRED iteration={iteration} "
                    f"elapsed={elapsed_minutes:.1f}min duration={duration}min",
                    flush=True,
                )
                break

            # Find latest log file
            log_file = find_latest_log(project_path)
            pipeline_running = True
            time_since_write_minutes = 0.0

            if log_file:
                try:
                    last_write = os.path.getmtime(log_file)
                    time_since_write_minutes = (time.time() - last_write) / 60.0
                except Exception:
                    time_since_write_minutes = 999.0

                # If no log activity for stall_threshold+ minutes, check if pipeline finished
                if time_since_write_minutes > STALL_THRESHOLD:
                    checkpoint_found = False
                    last_stage = "UNKNOWN"

                    for cp_file in [
                        os.path.join(project_path, "checkpoint.json"),
                        os.path.join(project_path, "checkpoint.backup.json"),
                        os.path.join(project_path, "checkpoint.backup.1.json"),
                    ]:
                        cp = read_checkpoint(cp_file)
                        if cp:
                            last_stage = cp.get("last_completed_stage", "UNKNOWN")
                            checkpoint_found = True
                            break

                    if checkpoint_found:
                        if last_stage == "OUTPUT":
                            print(f"WATCH_EXIT_PIPELINE_COMPLETE iteration={iteration} stage={last_stage}", flush=True)
                            pipeline_running = False
                        else:
                            print(
                                f"WATCH_EXIT_PIPELINE_STALLED iteration={iteration} "
                                f"lastStage={last_stage} timeSinceWrite={time_since_write_minutes:.1f}min",
                                flush=True,
                            )
                            pipeline_running = False
                    else:
                        print(
                            f"WATCH_EXIT_NO_LOG_ACTIVITY iteration={iteration} "
                            f"timeSinceWrite={time_since_write_minutes:.1f}min",
                            flush=True,
                        )
                        pipeline_running = False

                    if not pipeline_running:
                        break

            # Get current stage from latest log file
            stage = "UNKNOWN"
            if log_file:
                stage = get_current_stage_from_log(log_file)

            # Get video count from checkpoint
            video_count = "N/A"
            for cp_file in [
                os.path.join(project_path, "checkpoint.json"),
                os.path.join(project_path, "checkpoint.backup.json"),
                os.path.join(project_path, "checkpoint.backup.1.json"),
            ]:
                cp = read_checkpoint(cp_file)
                if cp:
                    stages = cp.get("stages", {})
                    vs = stages.get("VIDEO_SEARCH", {})
                    if isinstance(vs, dict) and "video_count" in vs:
                        video_count = str(vs["video_count"])
                    break

            # Count progress metrics
            stream_states = 0
            captions_fetched = 0
            if log_file:
                stream_states = count_pattern_in_file(log_file, "Stream state:")
                captions_fetched = count_pattern_in_file(log_file, "captions fetched")

            # Get last activity time
            last_activity = "N/A"
            time_since_write_str = "N/A"
            if log_file:
                try:
                    mtime = os.path.getmtime(log_file)
                    last_activity = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    time_since_write_str = f"{time_since_write_minutes:.1f} min"
                except Exception:
                    pass

            # Get recent log content
            log_content = ""
            if log_file:
                tail = read_tail_lines(log_file, 30)
                log_content = "\n".join(tail)

            # Calculate time strings
            elapsed_str = f"{elapsed_minutes:.1f} min"
            remaining = max(0, duration - elapsed_minutes)
            remaining_str = f"{remaining:.1f} min"

            # Write PIPELINE_STATUS.md
            write_status_md(
                project_path,
                iteration,
                stage,
                video_count,
                stream_states,
                captions_fetched,
                last_activity,
                time_since_write_str,
                elapsed_str,
                duration,
                remaining_str,
                log_content,
            )

            # Write status marker
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(status_marker, "w", encoding="utf-8") as f:
                    f.write(f"{iteration}|{timestamp}")
            except Exception:
                pass

            print(
                f"[Watch {iteration}] stage={stage} streamStates={stream_states} "
                f"captions={captions_fetched} lastActivity={last_activity}",
                flush=True,
            )

            # Sleep
            time.sleep(CHECK_INTERVAL)

    finally:
        # Cleanup
        try:
            os.remove(running_marker)
        except Exception:
            pass

    print(f"WATCH_LOOP_EXITED iteration={iteration}", flush=True)


if __name__ == "__main__":
    main()
