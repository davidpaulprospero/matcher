---
name: pipeline-watch
description: Monitor pipeline progress with auto-loop using Claude Code background agents. Runs a subagent in background that monitors the pipeline and auto-restarts on exit.
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

Monitors a running pipeline using Claude Code's background agent system. The subagent runs in background, monitors pipeline progress, and automatically restarts when it exits.

## Usage

```
/watch <project_path>                   # Run with background agent (default)
/watch <project_path> --duration 30   # Run for 30 minutes per cycle
/watcher <project_path>                # Alias for /watch
```

## Arguments

- `<project_path>`: Path to project directory (required)
- `--duration N`: Total duration in minutes before the background agent exits and auto-restarts (default: 30 minutes)

## Instructions

### 1. Parse arguments

Extract project path and `--duration` value. Default duration is 5 minutes.

### 2. Validate project

Check that the path exists:

```bash
ls "<project_path>"
```

If path doesn't exist, ask user for correct path.

### 3. Launch background agent for monitoring

Create a task that runs in the background using the `run_in_background: true` parameter:

```bash
python scripts/watch_auto.py --project-path "<project_path>" --duration <duration>
```

Use the Bash tool with `run_in_background: true`:

```
run_in_background: true
```

### 4. Report to user

Tell the user:
- Monitoring started for X minutes per cycle
- The background agent will auto-restart when it exits
- Check PIPELINE_STATUS.md for current status
- Say "Press Ctrl+F to stop all monitoring" so user can halt if needed

### 5. Handle background agent exit (when Claude Code reactivates)

When the background task exits and Claude Code activates:

1. **Check for stop signals** - Check both:
   - `<project_path>/watch_stop.txt` - project-specific stop
   - `Degold/queue_stop.txt` - unified queue stop (content: "true", "1", or "stop")

   If either exists with "true", "1", or "stop", remove it and exit the loop.

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

5. **Auto-restart** - Run the watch script again with `run_in_background: true`

6. **Exit only when:**
   - User presses Ctrl+F to kill background agents
   - User sends stop signal (watch_stop.txt file)
   - User explicitly interrupts

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+B | Background a running task |
| Ctrl+F | Kill all background agents (press twice to confirm) |
| Ctrl+T | Toggle task list |

## Exit conditions

- Duration expires (auto-restart via background agent)
- Pipeline completes (auto-restart)
- Pipeline stalls/errors (auto-restart)
- User presses Ctrl+F (halt all monitoring)
- User creates watch_stop.txt file (halt auto-restart)
