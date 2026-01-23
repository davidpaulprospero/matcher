#!/usr/bin/env python
"""
Pipeline Watch - Monitor and auto-start pipeline for a project.

Usage:
    python scripts/watch_pipeline.py <project_path>
    python scripts/watch_pipeline.py "E:/Edit Job/Project/Name__2026-01-15"
"""

import sys
import os

# Fix Windows console encoding for unicode output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import glob
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Stage wait times (seconds between checks)
STAGE_WAIT = {
    "ANALYZE": 15, "ENTITY_IMAGES": 20, "ENTITY_VIDEOS": 30,
    "DOWNLOAD": 60, "STOCK": 45, "BROLL_DOWNLOAD": 45, "REMIX": 10,
    "CAPTION": 20, "TRANSCRIBE": 45, "SCENE_DETECTION": 30,
    "MATCH": 30, "BROLL_MATCH": 20, "OUTPUT": 15,
}
DEFAULT_WAIT = 30


def is_pipeline_running(project_path: str, project_name: str) -> bool:
    """Check if pipeline process is running for this project."""
    try:
        result = subprocess.run(
            ['wmic', 'process', 'where', "name='python.exe'", 'get', 'commandline'],
            capture_output=True, text=True, timeout=5
        )
        # Check each command line for main.py with this project
        for line in result.stdout.split('\n'):
            line_lower = line.lower()
            # Must have main.py AND the project path/name
            if 'main.py' in line_lower:
                if (project_name.lower() in line_lower or
                    project_path.lower().replace('\\', '/') in line_lower or
                    project_path.lower() in line_lower):
                    return True
        return False
    except Exception:
        return False


def read_checkpoint(project_path: str) -> dict:
    """Read checkpoint.json and return status info."""
    cp_path = os.path.join(project_path, "checkpoint.json")
    if os.path.exists(cp_path):
        try:
            with open(cp_path, encoding='utf-8') as f:
                cp = json.load(f)
            # Handle both old and new checkpoint formats
            last_stage = cp.get("last_completed_stage") or cp.get("current_stage", "UNKNOWN")
            completed = cp.get("completed_stages", [])
            # If no completed_stages list, infer from last_completed_stage
            if not completed and last_stage and last_stage != "UNKNOWN":
                completed = [last_stage]  # At minimum, this stage is done
            return {
                "stage": last_stage,
                "completed": completed,
                "age": time.time() - os.path.getmtime(cp_path)
            }
        except Exception:
            pass
    return {"stage": "NOT_STARTED", "completed": [], "age": 9999}


def get_latest_log(project_path: str) -> tuple[str | None, float]:
    """Get latest log file and its age in seconds (excludes verbose logs)."""
    log_files = glob.glob(os.path.join(project_path, "logs", "run_*.log"))
    # Filter out verbose logs
    log_files = [f for f in log_files if '_verbose.log' not in f]
    if log_files:
        latest = max(log_files, key=os.path.getmtime)
        return latest, time.time() - os.path.getmtime(latest)
    return None, 9999


def get_log_tail(log_path: str, lines: int = 5) -> list[str]:
    """Get last N lines from log file."""
    if not log_path or not os.path.exists(log_path):
        return []
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            return [l.strip()[:100] for l in all_lines[-lines:]]
    except Exception:
        return []


def find_voiceover(project_path: str) -> str | None:
    """Find voiceover file in project."""
    vo_dir = os.path.join(project_path, 'voiceover')
    for ext in ['*.srt', '*.mp3', '*.wav', '*.mp4', '*.*']:
        files = glob.glob(os.path.join(vo_dir, ext))
        if files:
            return files[0]
    return None


def start_pipeline(project_path: str, voiceover: str) -> subprocess.Popen:
    """Start pipeline in background, return process."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / 'main.py'),
        '--project', project_path,
        '--voiceover', voiceover,
        '--non-interactive'
    ]
    # Start detached process
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )


def print_status(project_name: str, stage: str, completed: list, log_path: str | None, running: bool):
    """Print current status."""
    now = datetime.now().strftime('%H:%M:%S')
    n_completed = len(completed)
    total = 12

    status = "RUNNING" if running else "STOPPED"
    print(f"\n[{now}] {project_name}")
    print(f"  Stage: {stage} ({n_completed}/{total})")
    print(f"  Status: {status}")

    if log_path:
        for line in get_log_tail(log_path, 3):
            print(f"  > {line}")
    print("-" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/watch_pipeline.py <project_path> [--once]")
        sys.exit(1)

    project_path = os.path.abspath(sys.argv[1])
    project_name = os.path.basename(project_path)
    once_mode = "--once" in sys.argv  # Just print status and exit

    if not os.path.isdir(project_path):
        print(f"Error: Project not found: {project_path}")
        sys.exit(1)

    print(f"=" * 60)
    print(f"Watch: {project_name}")
    print(f"=" * 60)

    # Initial status
    cp = read_checkpoint(project_path)
    log_path, log_age = get_latest_log(project_path)
    process_running = is_pipeline_running(project_path, project_name)

    # Check if already finished
    is_finished = "OUTPUT" in cp["completed"] or cp["stage"] == "FINISHED"

    # Determine if running (any indicator) - but not if finished
    is_running = not is_finished and (process_running or log_age < 60 or cp["age"] < 120)

    print(f"Stage: {cp['stage']}")
    print(f"Completed: {len(cp['completed'])}/12")
    print(f"Process: {'Yes' if process_running else 'No'}")
    print(f"Log age: {log_age:.0f}s")

    if is_finished:
        print(f"\n*** Pipeline already FINISHED ***")
        print("Nothing to monitor.")
        sys.exit(0)

    # Start if not running
    if not is_running:
        voiceover = find_voiceover(project_path)
        if not voiceover:
            print("Error: No voiceover file found in voiceover/ folder")
            sys.exit(1)

        if once_mode:
            print(f"\nPipeline NOT running. Would start with: {os.path.basename(voiceover)}")
            print("\nDone (--once mode).")
            sys.exit(0)

        print(f"\nStarting pipeline...")
        print(f"Voiceover: {os.path.basename(voiceover)}")
        start_pipeline(project_path, voiceover)
        time.sleep(5)  # Wait for startup
        print("Pipeline started.")
    else:
        print(f"\nPipeline already running.")
        if once_mode:
            print_status(project_name, cp["stage"], cp["completed"], log_path, is_running)
            print("\nDone (--once mode).")
            sys.exit(0)

    # Monitor loop
    print(f"\nMonitoring (Ctrl+C to stop)...")
    print("-" * 60)

    last_stage = None
    consecutive_idle = 0

    try:
        while True:
            cp = read_checkpoint(project_path)
            log_path, log_age = get_latest_log(project_path)
            process_running = is_pipeline_running(project_path, project_name)

            stage = cp["stage"]
            is_running = process_running or log_age < 60 or cp["age"] < 120

            # Print status
            print_status(project_name, stage, cp["completed"], log_path, is_running)

            # Check for completion
            if stage == "FINISHED" or "OUTPUT" in cp["completed"]:
                print("\n*** Pipeline FINISHED ***")
                print(f"Completed stages: {len(cp['completed'])}")
                break

            # Check for stalled pipeline
            if not is_running:
                consecutive_idle += 1
                if consecutive_idle >= 3:
                    print("\n*** Pipeline appears STOPPED ***")
                    print("Run /logcheck to check for errors.")
                    break
            else:
                consecutive_idle = 0

            # Wait based on current stage
            wait_time = STAGE_WAIT.get(stage, DEFAULT_WAIT)
            if stage != last_stage:
                print(f"  (checking every {wait_time}s for {stage})")
                last_stage = stage

            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")

    print("\nDone.")


if __name__ == "__main__":
    main()
