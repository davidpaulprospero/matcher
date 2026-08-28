#!/usr/bin/env python3
"""
Sequential orchestrator for the 4 new Oscar voiceover pipelines.
Runs vo-5, vo-6, vo-7, vo-8 one at a time. Each inherits the global config
(download.segment_max_resolution=360, generated_images.enabled=false,
 llm.provider=ollama, matching.primary_provider=embedding_only).
"""
import os
import subprocess
import sys
from pathlib import Path

# Unset any paid Gemini keys so no paid calls sneak in
for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "MINIMAX_API_KEY"):
    os.environ.pop(k, None)

PROJECT_ROOT = Path("D:/_Projects/voiceover-matcher-dev")
BASE = Path("E:/Edit Job/Oscar")
DATE = "2026-08-03"

PIPELINES = [
    {"slug": "oscar-vo-5", "lang": "hr"},
    {"slug": "oscar-vo-6", "lang": "en"},
    {"slug": "oscar-vo-7", "lang": "es"},
    {"slug": "oscar-vo-8", "lang": "es"},
]


def run_one(slug: str) -> int:
    project_dir = BASE / f"{slug}__{DATE}"
    vo = project_dir / "voiceover" / f"{slug}.mp3"
    if not vo.exists():
        print(f"[ERROR] voiceover missing: {vo}", flush=True)
        return 1

    log_path = project_dir / "pipeline_run.log"
    print(f"\n========== Starting {slug} ==========", flush=True)
    print(f"project: {project_dir}", flush=True)
    print(f"voiceover: {vo}", flush=True)
    print(f"log: {log_path}", flush=True)

    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--project",
                str(project_dir),
                "--voiceover",
                str(vo),
                "--non-interactive",
                "--save-keywords",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
    print(f"[{slug}] exit code: {proc.returncode}", flush=True)
    return proc.returncode


def main() -> int:
    print(f"Running {len(PIPELINES)} pipelines sequentially...", flush=True)
    results = {}
    for p in PIPELINES:
        rc = run_one(p["slug"])
        results[p["slug"]] = rc
    print("\n========== Summary ==========", flush=True)
    for slug, rc in results.items():
        status = "OK" if rc == 0 else f"FAIL ({rc})"
        print(f"  {slug}: {status}", flush=True)
    return 0 if all(rc == 0 for rc in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())