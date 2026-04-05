#!/usr/bin/env python3
"""Autorun watch script - monitors degold_autorun.py queue runner health.

Cross-platform replacement for autorun_watch.ps1.
Started as Claude Code background task.
Monitors autorun health, cycle progress, and reports status.
Exits on duration expiry, queue stop, or autorun failure.
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
from datetime import datetime, timezone
from pathlib import Path

# Standard scripts/ import pattern
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)

from degold_autorun import is_process_running  # noqa: E402

HEARTBEAT_STALE_SECONDS = 90.0

# Full stage order from the pipeline
STAGE_ORDER = [
    "ANALYZE",
    "ENTITY_IMAGES",
    "ENTITY_VIDEOS",
    "STOCK_FOOTAGE",
    "GENERATED_IMAGES",
    "VIDEO_SEARCH",
    "CAPTION",
    "MATCH",
    "ITERATIVE_MATCH",
    "DOWNLOAD_SEGMENTS",
    "OUTPUT",
]

# Progress regexes
DOWNLOAD_SEGMENTS_FULL_RE = re.compile(
    r"\[DOWNLOAD_SEGMENTS\] Progress: (\d+)% \((\d+)/(\d+)\).*ok=(\d+), failed=(\d+), cached=(\d+)"
)
DOWNLOAD_SEGMENTS_SHORT_RE = re.compile(
    r"\[DOWNLOAD_SEGMENTS\] Progress: (\d+)% \((\d+)/(\d+)\)"
)
VIDEO_SEARCH_RE = re.compile(r"\[(\d+)/(\d+)\] Searching: (.+)")
CAPTION_STARTED_RE = re.compile(r"\[CAPTION\] Stage started - video_count=(\d+)")
ERROR_PATTERNS = re.compile(
    r" - ERROR - | - CRITICAL - |Traceback|Exception:|HTTP 429|HTTP 403|HTTP 500"
)


def read_json_safe(path: str) -> dict | None:
    """Read a JSON file safely, returning None on any error."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None


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


def read_file_tail(path: str, tail_bytes: int = 32768) -> list[str]:
    """Read the tail of a file, returning lines. Uses shared read access."""
    try:
        file_size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if file_size > tail_bytes:
                f.seek(max(0, file_size - tail_bytes))
                f.readline()  # skip partial first line
            return [line.rstrip("\n\r") for line in f.readlines()]
    except Exception:
        return []


def derive_current_stage(last_completed: str) -> str:
    """Derive the current stage from the last completed stage."""
    try:
        idx = STAGE_ORDER.index(last_completed)
        if idx < len(STAGE_ORDER) - 1:
            return STAGE_ORDER[idx + 1]
        return "COMPLETED"
    except ValueError:
        return ""


def parse_uptime(started_at: str) -> str:
    """Parse started_at ISO timestamp and return uptime string like '2h30m'."""
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - start
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        return f"{hours}h{minutes}m"
    except Exception:
        return ""


def validate_pipeline_quality(proj_dir: str) -> list[str]:
    """Check completed pipeline for quality issues.

    Returns a list of warning strings (empty = healthy).
    """
    warnings: list[str] = []
    cp = read_checkpoint(os.path.join(proj_dir, "checkpoint.json"))
    if not cp:
        return warnings

    # Check download segments
    ds = cp.get("download_segments", {})
    if isinstance(ds, dict):
        seg_count = ds.get("segment_count", -1)
        total_matches = ds.get("total_matches", 0)
        failed_items = ds.get("failed_items", [])
        failed_count = len(failed_items) if isinstance(failed_items, list) else 0

        if total_matches > 0 and seg_count == 0:
            warnings.append(
                f"CRITICAL: 0/{total_matches} segments downloaded "
                f"({failed_count} failures) — V1 track will be EMPTY"
            )
        elif total_matches > 0 and seg_count > 0:
            success_rate = seg_count / total_matches
            if success_rate < 0.1:
                warnings.append(
                    f"LOW YIELD: only {seg_count}/{total_matches} segments "
                    f"downloaded ({success_rate:.0%})"
                )

    # Check match quality
    match_data = cp.get("match", {})
    if isinstance(match_data, dict):
        avg_conf = match_data.get("avg_confidence", 0)
        match_count = match_data.get("match_count", 0)
        if match_count > 0 and avg_conf < 0.10:
            warnings.append(
                f"LOW CONFIDENCE: avg match confidence {avg_conf:.0%} "
                f"across {match_count} matches"
            )

    return warnings


def extract_progress(lines: list[str], stage: str, log_file: str) -> str:
    """Extract stage-specific progress from log tail lines."""
    if stage == "DOWNLOAD_SEGMENTS":
        for line in reversed(lines):
            m = DOWNLOAD_SEGMENTS_FULL_RE.search(line)
            if m:
                return f"{m.group(1)}% ({m.group(2)}/{m.group(3)}) ok={m.group(4)} fail={m.group(5)} cached={m.group(6)}"
            m = DOWNLOAD_SEGMENTS_SHORT_RE.search(line)
            if m:
                return f"{m.group(1)}% ({m.group(2)}/{m.group(3)})"

    elif stage == "VIDEO_SEARCH":
        for line in reversed(lines):
            m = VIDEO_SEARCH_RE.search(line)
            if m:
                return f"{m.group(1)}/{m.group(2)} keywords"

    elif stage == "CAPTION":
        caption_total = 0
        caption_done = 0
        # Try non-verbose log first (smaller, has Stage started line)
        non_verbose = re.sub(r"_verbose\.log$", ".log", log_file)
        if non_verbose != log_file and os.path.exists(non_verbose):
            try:
                with open(non_verbose, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        m = CAPTION_STARTED_RE.search(line)
                        if m:
                            caption_total = int(m.group(1))
                        if "Stream state:" in line:
                            caption_done += 1
            except Exception:
                pass
        # Fallback: count from verbose tail
        if caption_total == 0:
            for line in lines:
                m = CAPTION_STARTED_RE.search(line)
                if m:
                    caption_total = int(m.group(1))
                if "Stream state:" in line:
                    caption_done += 1
        if caption_total > 0:
            pct = round((caption_done / caption_total) * 100)
            return f"{pct}% (~{caption_done}/{caption_total} checked)"
        elif caption_done > 0:
            return f"{caption_done} checked"

    elif stage == "MATCH":
        match_count = sum(1 for line in lines if "VOICEOVER:" in line)
        if match_count > 0:
            return f"{match_count} segments matched (tail)"

    elif stage == "OUTPUT":
        return "generating timeline"

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Autorun watch - monitors degold_autorun.py health")
    parser.add_argument("--duration", type=int, default=5, help="Total duration in minutes before exit (default: 5)")
    parser.add_argument("--check-interval", type=int, default=30, help="Check interval in seconds (default: 30)")
    parser.add_argument("--base-dir", default="", help="Project root directory (defaults to script's grandparent)")
    parser.add_argument("--log-file", default="", help="Override autorun log file path")
    parser.add_argument("--state-file", default="", help="Override autorun state file path")
    parser.add_argument("--lock-file", default="", help="Override autorun lock file path")
    parser.add_argument("--queue-stop-file", default="", help="Override queue stop file path")
    parser.add_argument("--queue-state-file", default="", help="Override queue state file path")
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Monitor unified autorunner (reads config/board_registry.yaml, uses project-root file paths)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else project_root

    if args.unified:
        log_file = args.log_file or str(base_dir / "logs" / "autorun.log")
        state_file = args.state_file or str(base_dir / "autorun_state.json")
        lock_file = args.lock_file or str(base_dir / "autorun.lock")
        queue_stop_file = args.queue_stop_file or str(base_dir / "autorun.stop")
        # For queue state, read all boards from registry
        queue_state_file = args.queue_state_file  # may be empty — handled below
    else:
        log_file = args.log_file or str(base_dir / "logs" / "degold_autorun.log")
        state_file = args.state_file or str(base_dir / "clients" / "degold" / "degold_autorun_state.json")
        lock_file = args.lock_file or str(base_dir / "clients" / "degold" / "degold_autorun.lock")
        queue_stop_file = args.queue_stop_file or str(base_dir / "clients" / "degold" / "queue_stop.txt")
        queue_state_file = args.queue_state_file or str(base_dir / "clients" / "degold" / "pipeline_queue_state.json")

    duration = args.duration
    check_interval = args.check_interval
    iteration = 0
    start_time = time.time()
    prev_running_card = ""
    prev_pipeline_log_file = ""

    print(f"[AutorunWatch] Starting: duration={duration}min interval={check_interval}s", flush=True)

    # Wait a bit for autorun to start up
    time.sleep(5)

    while True:
        iteration += 1

        # 1. Check for queue stop file
        if os.path.exists(queue_stop_file):
            try:
                with open(queue_stop_file, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip().lower()
                if content in ("true", "1", "stop"):
                    print(f"AUTORUN_WATCH_EXIT_QUEUE_STOP iteration={iteration}", flush=True)
                    break
            except Exception:
                pass

        # 2. Check if duration has expired
        elapsed = time.time() - start_time
        elapsed_minutes = elapsed / 60.0
        if elapsed_minutes >= duration:
            print(
                f"AUTORUN_WATCH_EXIT_DURATION_EXPIRED iteration={iteration} "
                f"elapsed={elapsed_minutes:.1f}min",
                flush=True,
            )
            break

        # 3. Check if autorun is running via lock file
        autorun_running = False
        status = "unknown"
        lock_pid = ""

        if os.path.exists(lock_file):
            lock_data = read_json_safe(lock_file)
            if lock_data:
                lock_pid = str(lock_data.get("pid", ""))
                heartbeat_at = lock_data.get("heartbeat_at", "")

                if lock_pid and is_process_running(lock_pid):
                    autorun_running = True
                    status = "running"
                else:
                    status = "dead_pid"

                # Check heartbeat staleness
                if heartbeat_at and autorun_running:
                    try:
                        hb_time = datetime.fromisoformat(heartbeat_at.replace("Z", "+00:00"))
                        now = datetime.now(timezone.utc)
                        since_hb = (now - hb_time).total_seconds()
                        if since_hb > HEARTBEAT_STALE_SECONDS:
                            status = "stale_heartbeat"
                            autorun_running = False
                    except Exception:
                        pass
            else:
                status = "lock_error"
        else:
            status = "no_lock"

        # 4. Check log file for recent activity
        log_activity = "none"
        if os.path.exists(log_file):
            try:
                time_since_log = (time.time() - os.path.getmtime(log_file)) / 60.0
                if time_since_log < 1:
                    log_activity = "active"
                elif time_since_log < 5:
                    log_activity = "recent"
                else:
                    log_activity = "stale"
            except Exception:
                pass

        # 5. Get state info
        last_launched = ""
        launch_outcome = ""
        pipeline_stage = ""
        suppressed_cards: list[str] = []
        consecutive_failures = 0
        uptime = ""
        state_data = read_json_safe(state_file)
        if state_data:
            runtime = state_data.get("runtime", {})
            if runtime:
                if runtime.get("started_at"):
                    uptime = parse_uptime(runtime["started_at"])

            last_cycle = state_data.get("last_cycle", {})
            if last_cycle:
                last_launched = last_cycle.get("launched_card_id", "")
                launch_outcome = last_cycle.get("launch_outcome", "")
                pipeline_stage = last_cycle.get("pipeline_stage", "")
                consecutive_failures = last_cycle.get("consecutive_failures", 0)

            suppressed_cards = state_data.get("suppressed_card_ids", []) or []

        # 6. Get queue counts (aggregate across boards in unified mode)
        ready_count = 0
        running_count = 0
        completed_count = 0
        running_card = ""
        board_counts_label = ""

        queue_state_files: list[tuple[str, str]] = []  # (board_key, path)
        if args.unified and not queue_state_file:
            # Read board registry to find all queue state files
            registry_path = base_dir / "config" / "board_registry.yaml"
            if registry_path.exists():
                try:
                    import yaml
                    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
                    for bkey, entry in (raw.get("boards") or {}).items():
                        if isinstance(entry, dict) and entry.get("enabled", True):
                            sf = entry.get("state_file")
                            if sf:
                                queue_state_files.append((bkey, str(base_dir / sf)))
                except Exception:
                    pass
        if not queue_state_files:
            queue_state_files = [("", queue_state_file or str(base_dir / "clients" / "degold" / "pipeline_queue_state.json"))]

        # Aggregate across all board queue state files
        all_queue_data: dict[str, dict | None] = {}
        board_parts: list[str] = []
        for bkey, qsf in queue_state_files:
            qd = read_json_safe(qsf)
            all_queue_data[bkey or "default"] = qd
        # Pick the first non-None for detailed pipeline inspection
        queue_data = next((qd for qd in all_queue_data.values() if qd), None)

        for bkey, qd in all_queue_data.items():
            if not qd:
                continue
            b_ready = 0
            b_running = 0
            b_completed = 0
            queue = qd.get("queue", {})
            if queue:
                ready_ids = queue.get("ready_card_ids", [])
                completed_ids = queue.get("completed_card_ids", [])
                b_ready = len(ready_ids) if isinstance(ready_ids, list) else 0
                b_completed = len(completed_ids) if isinstance(completed_ids, list) else 0

            # Check runtime_summary (authoritative)
            runtime_summary = qd.get("runtime_summary", {})
            if runtime_summary:
                b_running = int(runtime_summary.get("running_count", 0))
                running_ids = runtime_summary.get("running_card_ids", [])
                if running_ids and not running_card:
                    running_card = ",".join(str(x) for x in running_ids)

            # Fallback: check individual pipeline entries
            if b_running == 0:
                pipelines = qd.get("pipelines", {})
                if isinstance(pipelines, dict):
                    for key, pl in pipelines.items():
                        if isinstance(pl, dict) and pl.get("pipeline_runtime_state") == "running":
                            b_running += 1
                            if not running_card:
                                running_card = pl.get("card_id", key)

            ready_count += b_ready
            running_count += b_running
            completed_count += b_completed
            if bkey and args.unified:
                board_parts.append(f"{bkey}:{b_ready}rdy/{b_running}run")

        # Stale-queue fallback: if queue says 0 running but prev iteration had one with active log
        if running_count == 0 and prev_running_card and prev_pipeline_log_file:
            if os.path.exists(prev_pipeline_log_file):
                try:
                    age = time.time() - os.path.getmtime(prev_pipeline_log_file)
                    if age < 120:
                        running_card = prev_running_card
                        running_count = 1
                except Exception:
                    pass

        # 7. Read checkpoint stage and find pipeline log for the running pipeline
        checkpoint_stage = ""
        pipeline_log_file = ""
        pipeline_proj_dir = ""
        # In unified mode, search pipelines across all boards
        all_pipelines: dict = {}
        for _bkey, qd in all_queue_data.items():
            if qd and isinstance(qd.get("pipelines"), dict):
                all_pipelines.update(qd["pipelines"])
        if running_card and "," not in running_card and all_pipelines:
            pipelines = all_pipelines
            if isinstance(pipelines, dict):
                for key, pl in pipelines.items():
                    if not isinstance(pl, dict):
                        continue
                    if pl.get("card_id") != running_card:
                        continue
                    project_info = pl.get("project", {})
                    if not isinstance(project_info, dict):
                        continue
                    dirs = project_info.get("local_project_dirs", [])
                    if not dirs:
                        continue
                    pipeline_proj_dir = dirs[0]
                    # Read checkpoint
                    cp_file = os.path.join(pipeline_proj_dir, "checkpoint.json")
                    cp = read_checkpoint(cp_file)
                    if cp:
                        checkpoint_stage = cp.get("last_completed_stage", "")
                    # Find latest verbose log
                    logs_dir = os.path.join(pipeline_proj_dir, "logs")
                    if os.path.isdir(logs_dir):
                        verbose_logs = glob.glob(os.path.join(logs_dir, "run_*_verbose.log"))
                        if verbose_logs:
                            pipeline_log_file = max(verbose_logs, key=os.path.getmtime)
                    break

        # Derive current stage from last_completed_stage
        current_stage = derive_current_stage(checkpoint_stage) if checkpoint_stage else ""

        # 8. Extract progress from log tail
        stage_progress = ""
        if pipeline_log_file and os.path.exists(pipeline_log_file) and current_stage:
            progress_lines = read_file_tail(pipeline_log_file, 65536)
            stage_progress = extract_progress(progress_lines, current_stage, pipeline_log_file)

        # 8b. Validate pipeline quality when stage reaches OUTPUT/COMPLETED
        quality_warnings: list[str] = []
        if pipeline_proj_dir and current_stage in ("COMPLETED", "OUTPUT", ""):
            if checkpoint_stage == "OUTPUT":
                quality_warnings = validate_pipeline_quality(pipeline_proj_dir)

        # 9. Build and print output line
        details = f"[AutorunWatch] iteration={iteration} status={status} log={log_activity} pid={lock_pid}"
        details += f" uptime={uptime}"
        details += f" queue={ready_count}ready/{running_count}running/{completed_count}done"
        if board_parts:
            details += f" boards=[{' '.join(board_parts)}]"

        if running_card:
            details += f" active_pipeline={running_card}"
        if current_stage:
            details += f" stage={current_stage}"
        elif checkpoint_stage:
            details += f" stage={checkpoint_stage}"
        elif pipeline_stage:
            details += f" stage={pipeline_stage}"
        if stage_progress:
            details += f" progress=[{stage_progress}]"
        if last_launched and launch_outcome:
            details += f" last={last_launched}:{launch_outcome}"
        if suppressed_cards:
            details += f" suppressed={','.join(suppressed_cards)}"
        if consecutive_failures and consecutive_failures > 0:
            details += f" failures={consecutive_failures}"

        print(details, flush=True)

        # 9b. Print quality warnings (these are the important ones)
        if quality_warnings:
            print(f"  === PIPELINE QUALITY ISSUES ({len(quality_warnings)}) ===", flush=True)
            for qw in quality_warnings:
                print(f"  !! {qw}", flush=True)

        # 10. Show pipeline log tail and recent errors
        if pipeline_log_file and os.path.exists(pipeline_log_file):
            try:
                log_size = os.path.getsize(pipeline_log_file)
                log_age = time.time() - os.path.getmtime(pipeline_log_file)
                log_name = os.path.basename(pipeline_log_file)
                log_size_kb = round(log_size / 1024)
                print(f"  --- Pipeline log: {log_name} ({log_size_kb}KB, {log_age:.0f}s ago) ---", flush=True)

                tail_lines = read_file_tail(pipeline_log_file, 32768)

                # Show last 15 lines
                show_count = min(15, len(tail_lines))
                for line in tail_lines[-show_count:]:
                    trimmed = line[:180] + "..." if len(line) > 180 else line
                    print(f"  | {trimmed}", flush=True)

                # Scan for recent errors
                errors: list[str] = []
                for line in tail_lines:
                    if ERROR_PATTERNS.search(line):
                        trimmed = line[:160] + "..." if len(line) > 160 else line
                        if trimmed not in errors:
                            errors.append(trimmed)

                if errors:
                    print(f"  --- Recent errors ({len(errors)}) ---", flush=True)
                    for err in errors[:5]:
                        print(f"  ! {err}", flush=True)
                    if len(errors) > 5:
                        print(f"  ! ... and {len(errors) - 5} more", flush=True)
            except Exception as e:
                print(f"  [warn] Could not read pipeline log: {e}", flush=True)

        # 11. Check if autorun stopped unexpectedly
        if not autorun_running and status != "no_lock":
            print(f"AUTORUN_WATCH_EXIT_AUTORUN_STOPPED iteration={iteration} status={status}", flush=True)
            break

        # 12. Stale log warning
        if autorun_running and log_activity == "stale":
            print(
                f"AUTORUN_WATCH_LOG_STALE iteration={iteration} log={log_activity} failures={consecutive_failures}",
                flush=True,
            )

        # 13. Remember running pipeline for next iteration
        if running_card:
            prev_running_card = running_card
            if pipeline_log_file:
                prev_pipeline_log_file = pipeline_log_file

        # 14. Sleep
        time.sleep(check_interval)

    print(f"AUTORUN_WATCH_COMPLETE iteration={iteration}", flush=True)


if __name__ == "__main__":
    main()
