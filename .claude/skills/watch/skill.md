---
name: watch
description: Monitor pipeline progress with auto-loop. Runs in background, sleeps, exits, and will be restarted by the skill when it detects the exit.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
  - Task
  - TaskOutput
---

# Pipeline Watch Skill

Monitors a running pipeline with auto-loop. Runs in background, sleeps between checks, exits when duration expires or pipeline finishes - then automatically restarts.

## Usage

```
/watch <project_path>                   # Run for 5 minutes, auto-restart (default)
/watch <project_path> --duration 30    # Run for 30 minutes, auto-restart
/watch <project_path> --once            # One-time check only (no loop)
```

## Arguments

- `<project_path>`: Path to project directory (required)
- `--duration N`: Total duration in minutes before exiting and auto-restarting (default: 5 minutes)
- `--once`: One-time check only (no loop, no auto-restart)

## Instructions

### 1. Parse arguments

Extract project path and `--duration` value. Default duration is 5 minutes.

### 2. Validate project

Check that the path exists:

```bash
ls "<project_path>"
```

If path doesn't exist, ask user for correct path.

### 3. Run the watch script as Claude Code background task

**IMPORTANT: Never manually call Start-Sleep as a separate background task!** The watch_auto.ps1 script has its own built-in loop with sleep intervals. Just run the script once and let it handle the looping internally.

Use Claude Code's `run_in_background: true` parameter to start the PowerShell script:

```bash
powershell -ExecutionPolicy Bypass -File "D:\_Projects\voiceover-matcher-subtitle\scripts\watch_auto.ps1" -ProjectPath "<project_path>" -Duration <duration>
```

This runs the script as a background task within Claude Code. When it exits, Claude Code will activate with the output.

### 4. Report to user

Tell the user:
- Monitoring started for X minutes per cycle
- When duration expires (or pipeline completes/errors), it will auto-restart
- Check PIPELINE_STATUS.md for current status
- Say "Say 'stop' to halt the auto-restart loop" so user can stop if needed

### 5. Implement the skill-managed loop (critical!)

When the background task exits and Claude Code activates:

1. **Check for stop signal** - Look for a file `<project_path>/watch_stop.txt`. If it exists with "true" or "1", the user wants to stop. Remove the file and exit the loop.

2. Parse the output to determine exit reason:
   - `WATCH_EXIT_DURATION_EXPIRED` → Duration ran out, pipeline still running
   - `WATCH_EXIT_PIPELINE_COMPLETE` → Pipeline finished successfully
   - `WATCH_EXIT_PIPELINE_STALLED` → Pipeline stalled/failed
   - `WATCH_EXIT_ERROR` → Error occurred
   - `WATCH_EXIT_NO_LOG_ACTIVITY` → No log activity detected

3. **Report status** to user with current iteration, stage, and elapsed time

4. **Decide whether to continue:**
   - If `DURATION_EXPIRED` AND pipeline still running → continue (restart)
   - If `PIPELINE_COMPLETE` → continue (restart to monitor next run)
   - If `PIPELINE_STALLED` or `ERROR` → continue (restart to monitor)
   - If stop signal found → exit loop

5. **Auto-restart** - Run the watch script again (go to step 3) with same parameters

6. **Exit only when:**
   - User sends stop signal (watch_stop.txt file)
   - User explicitly interrupts

### 6. One-time check mode (--once)

If `--once` flag is passed:
- Run check once without background
- Write PIPELINE_STATUS.md
- Report status and exit (no loop, no auto-restart)

## Exit conditions

- Duration expires (auto-restart)
- Pipeline completes (auto-restart)
- Pipeline stalls/errors (auto-restart)
- User says "stop" (halt auto-restart)
