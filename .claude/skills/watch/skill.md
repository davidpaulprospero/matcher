---
name: watch
description: Monitor pipeline progress
allowed-tools:
  - Bash(python:*)
---

Call Bash tool with command: python scripts/watch_pipeline.py "PATH"

Replace PATH with the project path from the user's message.

Example: If user says /watch E:\Test\Project then run: python scripts/watch_pipeline.py "E:\Test\Project"
