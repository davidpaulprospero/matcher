---
name: watch
description: Monitor pipeline progress with periodic updates
allowed-tools:
  - Bash(python:*)
  - Bash(sleep:*)
  - Bash(tail:*)
  - Bash(ls:*)
  - Bash(ps:*)
  - Bash(grep:*)
---

Monitor pipeline progress for a project. Use the Bash tool.

## Quick Check Mode
First, check current status:
```bash
python scripts/watch_pipeline.py "PROJECT_PATH" --once
```

If pipeline is FINISHED and user provided --fresh flag, restart:
```bash
cd "D:/_Projects/voiceover-matcher-subtitle" && python main.py --project "PROJECT_PATH" --voiceover "PROJECT_PATH/voiceover/voiceover.srt" --fresh --non-interactive 2>&1 &
```

## Continuous Monitoring (15-min intervals)
After starting pipeline, monitor every 15 minutes:
```bash
sleep 900 && echo "=== CHECK: $(date) ===" && tail -40 "$(ls -t PROJECT_PATH/logs/run_*.log | head -1)"
```

Check if process is still running:
```bash
ps aux | grep -i "python.*main" | grep -v grep || echo "Process not running"
```

## What to report at each check:
1. Current stage (from log entries)
2. Recent activity (last 10-20 log lines)
3. Any errors or warnings
4. Process status (running/stopped)

## When pipeline stops unexpectedly:
1. Check for errors in log
2. Offer to resume: `python main.py --project "PROJECT_PATH" --resume --non-interactive`
3. Or run /logcheck to diagnose

Replace PROJECT_PATH with the actual path from the user's /watch command. Use forward slashes.
