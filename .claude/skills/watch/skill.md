---
name: watch
description: Monitor pipeline progress. Auto-starts if not running.
allowed-tools:
  - Bash(python:*)
---

# Watch Pipeline

Run this command immediately:

```bash
python scripts/watch_pipeline.py "{{PROJECT_PATH}}"
```

Replace `{{PROJECT_PATH}}` with the path from the user's command.

Example: `/watch E:\Edit Job\Stu\January\16__2026-01-14`

Run:
```bash
python scripts/watch_pipeline.py "E:\Edit Job\Stu\January\16__2026-01-14"
```
