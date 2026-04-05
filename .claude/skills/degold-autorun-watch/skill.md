---
name: degold-autorun-watch
description: Monitor degold_autorun.py queue runner with auto-restart using Claude Code background agents. Tracks autorun health, cycle progress, and restarts on exit or failure.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
  - Task
  - TaskOutput
---

# Autorun Watch Skill

Monitors the degold_autorun.py queue runner using Claude Code's background agent system. The subagent runs in background, monitors autorun health, and automatically restarts when it exits or fails.

## Usage

```
/autorun-watch                    # Run with default 15 min duration
/autorun-watch --duration 30     # Run for 30 minutes per cycle
/autorun-watch -d 120            # Run for 2 hours per cycle
```

## Arguments

- `--duration N` or `-d N`: Total duration in minutes before the background agent exits and auto-restarts (default: 15 minutes)

## Instructions

### 1. Parse arguments

Extract `--duration` value. Default is 15 minutes.

### 2. Launch background agent for monitoring

Create a task that runs in the background:

```bash
# For unified autorun (preferred):
python scripts/autorun_watch.py --unified --duration <duration> --base-dir .

# For legacy single-board autorun:
python scripts/autorun_watch.py --duration <duration> --base-dir .
```

The `-BaseDir` parameter tells the watch script which project root to monitor. Always pass the current working directory so the script reads the correct lock/state/log files regardless of where the script is located.

Use the Bash tool with `run_in_background: true`:

```
run_in_background: true
```

### 3. Report to user

Tell the user:
- Monitoring started for X minutes per cycle
- The background agent will auto-restart when it exits
- Log file: `logs/degold_autorun.log`
- State file: `clients/degold/degold_autorun_state.json`
- Say "Press Ctrl+F to stop all monitoring" so user can halt if needed

**Discord notification:** If a Discord channel is active in the conversation (i.e., there are `<channel source="plugin:discord:discord">` messages), also reply via the Discord MCP `reply` tool to the active `chat_id` confirming that monitoring has started.

### Watch output format

Each iteration now shows key details in a single line:

```
[AutorunWatch] iteration=5 status=running log=active pid=12524 uptime=2h30m queue=6ready/1running/1done active_pipeline=o3gs4hZt stage=DOWNLOAD_SEGMENTS last=o3gs4hZt:running suppressed=jt9KjJxK failures=0
```

Fields:
- `status`: running, dead_pid, stale_heartbeat, no_lock, lock_error
- `log`: active (<1min), recent (<5min), stale (>5min), none
- `uptime`: autorun uptime from state file
- `queue`: ready/running/completed counts from queue state
- `active_pipeline`: card ID of currently running pipeline (if any)
- `stage`: current pipeline stage from checkpoint
- `last`: last launched card and outcome (running, startup_failed, not_launched)
- `suppressed`: cards being skipped after startup failures
- `failures`: consecutive cycle failure count

### 4. Handle background agent exit (when Claude Code reactivates)

When the background task exits and Claude Code activates:

1. **Check for stop signals** - Check:
   - `clients/degold/queue_stop.txt` - unified queue stop (content: "true", "1", or "stop")

   If it exists with "true", "1", or "stop", remove it and exit the loop.

2. Parse the output to determine exit reason:
   - `AUTORUN_WATCH_EXIT_DURATION_EXPIRED` → Duration ran out
   - `AUTORUN_WATCH_EXIT_QUEUE_STOP` → Queue stop signal received
   - `AUTORUN_WATCH_EXIT_AUTORUN_STOPPED` → Autorun stopped unexpectedly

3. **Report status** to user with current iteration and autorun state. **If a Discord channel is active**, also send a status update via the Discord MCP `reply` tool (include exit reason, cycle count, and whether restarting or stopping).

4. **Decide whether to continue:**
   - If `DURATION_EXPIRED` → continue (restart)
   - If `QUEUE_STOP` → exit loop
   - If `AUTORUN_STOPPED` → restart autorun, then continue monitoring
   - Any other exit → continue (restart)

5. **Auto-restart** - Run the watch script again with `run_in_background: true`

6. **Exit only when:**
   - User presses Ctrl+F to kill background agents
   - User sends stop signal (queue_stop.txt file)
   - User explicitly interrupts

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+B | Background a running task |
| Ctrl+F | Kill all background agents (press twice to confirm) |
| Ctrl+T | Toggle task list |

## Exit conditions

- Duration expires (auto-restart via background agent)
- Queue stop signal received (halt auto-restart)
- Autorun stops unexpectedly (restart autorun then continue)
- User presses Ctrl+F (halt all monitoring)
- User creates queue_stop.txt file (halt auto-restart)

## Relevant Commands

```bash
# Kill running pipeline, sync state, and wake autorun immediately
python scripts/pipeline_queue_state.py kill-pipeline
python scripts/pipeline_queue_state.py kill-pipeline --card-id 0tsPiUY5
python scripts/pipeline_queue_state.py kill-pipeline --dry-run

# Sync queue from Trello (fixes "circuit breaker" issues)
python scripts/pipeline_queue_state.py sync

# Check queue status
python scripts/pipeline_queue_state.py status

# Check autorun state
python scripts/degold_autorun.py --status

# Fix stale lock file
rm clients/degold/degold_autorun.lock

# Run prepare (download voiceovers for ready cards)
python scripts/pipeline_queue_state.py prepare

# Run autorun once
python scripts/degold_autorun.py --once

# Stop autorun
python scripts/degold_autorun.py --stop
# Or:
echo true > clients/degold/queue_stop.txt

# Force immediate cycle (skip sleep interval)
python scripts/degold_autorun.py --force-cycle
```

## Troubleshooting

### Problem: "Circuit breaker is open" or queue shows 0 ready despite cards in Trello
**Fix:** Run sync to refresh from Trello:
```bash
python scripts/pipeline_queue_state.py sync
```

### Problem: "Unknown URL format" when running prepare
**Fix:** Check `newproject` argument order - should be: `<project_name> <channel> <url>`

### Problem: Voiceover not downloading
**Fix:** Verify account has GWS token:
```bash
grep GOOGLE_WORKSPACE_CLI_TOKEN clients/degold/accounts/david.env
```

### Problem: Prepare shows "No missing projects" but voiceover folder is empty
**Fix:** Check the project directory exists but doesn't have voiceover.mp3 - may need manual recreation:
```bash
ls project/voiceover/
```

### Problem: Lock file shows stale PID
**Fix:** Remove stale lock:
```bash
rm clients/degold/degold_autorun.lock
```

## Key Files to Monitor

- `logs/degold_autorun.log` - Autorun cycle logs
- `clients/degold/degold_autorun_state.json` - Current state including last_cycle, runtime
- `clients/degold/degold_autorun.lock` - Lock file with heartbeat
- `clients/degold/pipeline_queue_state.json` - Queue state with ready cards
