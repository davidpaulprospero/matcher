---
name: watch
description: Monitor pipeline progress
allowed-tools:
  - Bash(python:*)
---

Your ONLY action: invoke the Bash tool with this command:

python scripts/watch_pipeline.py "PROJECT_PATH"

Replace PROJECT_PATH with the path from the user's /watch command.

IMPORTANT: Convert backslashes to forward slashes in the path.
Example: E:\Edit Job\Test becomes E:/Edit Job/Test

Do not explain. Do not ask questions. Just call the Bash tool immediately.
