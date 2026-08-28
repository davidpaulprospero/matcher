#!/usr/bin/env python3
"""Watch supervisor: keeps watch_auto.py (and main.py) alive for a project.

Loop (every --interval seconds, default 60s):
  1. If checkpoint.last_completed_stage == OUTPUT and main.py exited -> DONE (exit)
  2. If watch_auto.py is dead -> launch it (--duration 5)
  3. If main.py is dead and a checkpoint.json exists -> relaunch with --resume --pipeline-mode full

Spawns children with CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS so they survive
this supervisor's continued execution. Logs to --log.
"""

import argparse
import json
import gzip
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path("D:/_Projects/voiceover-matcher-dev")


def proc_alive_with(pattern: str) -> bool:
    """Return True if any python.exe process has pattern in its command line (Windows).

    Uses a small ps1 script written to %TEMP% to avoid quoting issues.
    CREATE_NO_WINDOW prevents the PowerShell console from flashing on screen.
    """
    try:
        ps_script = (
            f"Get-WmiObject Win32_Process -Filter \"Name='python.exe'\" | "
            f"Where-Object {{ $_.CommandLine -and $_.CommandLine -like '*{pattern}*' }} | "
            f"Select-Object -ExpandProperty ProcessId"
        )
        creationflags = (
            subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True, timeout=20,
            creationflags=creationflags,
        )
        return any(line.strip().isdigit() for line in out.stdout.splitlines())
    except Exception as e:
        print(f"[supervisor] proc check failed (pattern={pattern!r}): {e}", flush=True)
        return False


def read_last_completed_stage(project: Path) -> str | None:
    checkpoint = project / "checkpoint.json"
    if not checkpoint.exists():
        return None
    try:
        raw = checkpoint.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw).get("last_completed_stage")
    except Exception as e:
        print(f"[supervisor] checkpoint read failed: {e}", flush=True)
        return None


def spawn(cmd_list: list[str], log_path: Path) -> None:
    """Spawn detached child with stdout/stderr appended to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "ab", buffering=0)
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        cmd_list,
        stdout=f,
        stderr=f,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )


def launch_watch(project: str, watch_log: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[supervisor] {ts} launching watch", flush=True)
    spawn(
        [sys.executable, "scripts/watch_auto.py", "--project-path", project, "--duration", "5"],
        watch_log,
    )


def launch_pipeline(voiceover: str, project: str, log: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[supervisor] {ts} relaunching main pipeline with --resume", flush=True)
    spawn(
        [
            sys.executable, "main.py",
            "--voiceover", voiceover,
            "--project", project,
            "--pipeline-mode", "full",
            "--non-interactive",
            "--resume",
        ],
        log,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--voiceover", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--watch-log", required=True)
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    project = Path(args.project)
    main_log = Path(args.log)
    watch_log = Path(args.watch_log)
    last_relaunch = datetime.min

    print(f"[supervisor] project={project}", flush=True)
    print(f"[supervisor] voiceover={args.voiceover}", flush=True)
    print(f"[supervisor] main-log={main_log}", flush=True)
    print(f"[supervisor] watch-log={watch_log}", flush=True)
    print(f"[supervisor] interval={args.interval}s", flush=True)

    try:
        while True:
            stage = read_last_completed_stage(project)
            main_alive = proc_alive_with("main.py")
            watch_alive = proc_alive_with("watch_auto.py")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if stage == "OUTPUT" and not main_alive:
                print(f"[supervisor] {ts} OUTPUT complete + main exited. DONE.", flush=True)
                return 0

            if not watch_alive:
                print(f"[supervisor] {ts} watch dead -> relaunching", flush=True)
                launch_watch(args.project, watch_log)

            if not main_alive and (project / "checkpoint.json").exists():
                now = datetime.now()
                if (now - last_relaunch) > timedelta(minutes=3):
                    launch_pipeline(args.voiceover, args.project, main_log)
                    last_relaunch = now

            print(
                f"[supervisor] {ts} stage={stage} main={'Y' if main_alive else 'N'} "
                f"watch={'Y' if watch_alive else 'N'}",
                flush=True,
            )

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("[supervisor] interrupted", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
