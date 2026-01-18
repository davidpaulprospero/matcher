---
name: watch
description: Monitor pipeline progress. Auto-starts if not running.
allowed-tools:
  - Bash(python:*)
---

EXECUTE this Bash command NOW. Do not explain. Do not ask questions. Just run it:

```bash
python scripts/watch_pipeline.py "{{PROJECT_PATH}}"
```

Replace `{{PROJECT_PATH}}` with the path the user provided.
