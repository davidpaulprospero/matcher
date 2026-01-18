---
name: watch
description: Monitor a running pipeline's progress with periodic status updates. Uses sleep timers to check status automatically.
allowed-tools:
  - Bash(python:*)
---

# Pipeline Watch

Monitors a pipeline's progress. Starts it if not running.

## Instructions

When invoked with `/watch <project_path>`:

**Run this single command:**

```bash
python scripts/watch_pipeline.py "<PROJECT_PATH>"
```

Replace `<PROJECT_PATH>` with the user's project path.

**Examples:**
```bash
python scripts/watch_pipeline.py "E:\Edit Job\Stu\January\6__2026-01-15"
python scripts/watch_pipeline.py "E:/path/with spaces/ProjectName__2026-01-10"
```

The script handles everything:
- Checks if pipeline is running
- Starts it if not (finds voiceover automatically)
- Monitors with periodic status updates
- Exits when pipeline finishes or stops

**Note:** The script runs continuously until the pipeline completes. Use Ctrl+C to stop early.
