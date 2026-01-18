---
name: watch
description: Monitor pipeline progress
allowed-tools:
  - Bash(python:*)
---

Your ONLY action: invoke the Bash tool with this command:

python scripts/watch_pipeline.py "PROJECT_PATH"

Replace PROJECT_PATH with the path from the user's /watch command.

Do not explain. Do not ask questions. Just call the Bash tool immediately.
