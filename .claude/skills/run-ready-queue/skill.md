---
name: run-ready-queue
description: Queue multiple ready pipeline runs and execute them sequentially, one active pipeline at a time, using Claude Code's background running feature. Use when asked to run a "Ready to Run" list, start several Trello pipeline cards in order, keep the queue moving overnight, or turn a screenshot/pasted list of ready cards into filtered sequential execution without parallel launches.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
  - Task
  - TaskOutput
---

# Run Ready Queue Skill

Use Claude Code background tasks to keep the queue moving. Do not manually launch multiple `main.py` processes.

Source of truth:

- `scripts/pipeline_queue_state.py`
- `scripts/degold_autorun.py`
- `Degold/pipeline_queue_state.json`
- `Degold/degold_autorun_state.json`
- `logs/degold_autorun.log`

## Usage

```text
/run-ready-queue
/run-ready-queue UAxE4onN wid1ATcA ab12Cd34
/run-ready-queue https://trello.com/c/UAxE4onN https://trello.com/c/wid1ATcA
/run-ready-queue --once UAxE4onN wid1ATcA
```

Copy card IDs directly from queue output or Trello short URLs. Do not improvise ambiguous `O`/`0` characters.

## Instructions

### 0. Ask before launching

Before running anything, gather context and confirm intent with the user. Ask:

1. **Which board/channel?** — The queue may contain cards from multiple boards (RRU, DSR, STU). Ask which channel(s) to target, or if they want to run all channels. If the user says "Stu" or "NEW AMERICA", use `--channel STU`. If they say "Degold" or don't specify, run without a channel filter (defaults to all).
2. **Specific cards or full queue?** — If the user provided card IDs, URLs, or a screenshot, confirm the extracted IDs. If not, confirm they want the full ready queue.
3. **Continuous or one-shot?** — Default to continuous autorun. Only use `--once` if the user explicitly asks for a single cycle.

Skip these questions when:
- The user's intent is already clear from context (e.g., "run the STU queue" = `--channel STU`, no need to ask)
- The user provides explicit card IDs (use those directly)
- The user says "just run it" or similar

### 1. Resolve the target set

- If the user gives card IDs, use them.
- If the user gives Trello URLs, extract the short card IDs.
- If the user gives a screenshot or pasted "Ready to Run" list, extract the IDs/URLs from that list.
- If the user specifies a channel (STU, RRU, DSR), use `--channel` filtering instead of card IDs.
- Treat screenshots and pasted lists as a requested filter only. Queue state decides whether each card is still actionable and what order remains.
- Verify the targets against queue state before starting anything:

```bash
python scripts/pipeline_queue_state.py show --json
```

- Treat `Degold/pipeline_queue_state.json` as the source of truth, not the screenshot ordering.
- If a provided card is no longer `ready`, say that explicitly before launching.

### 2. Pick the command

A `/run-ready-queue` request means start or restart queue ownership now. Do not ask whether to start autorun unless the user explicitly asked for a dry run, inspection, or command preview.

For any request to keep multiple ready cards moving sequentially, prefer the persistent autorun loop, not `--once`.

If a pipeline is already running but the autorun loop is inactive, still start or restart `scripts/degold_autorun.py`. The runner should regain ownership, observe the active card, and continue the queue after that run finishes.

#### Channel filtering with `--channel`

Use `--channel` to restrict execution to cards from a specific board/channel. Can be repeated for multiple channels. Composes with `--card-id` as an intersection (both filters apply).

Run only STU/NEW AMERICA cards:

```bash
python scripts/degold_autorun.py --channel STU --interval-minutes 1 --skip-discord-prepare
```

Run only RRU cards, exit when drained:

```bash
python scripts/degold_autorun.py --channel RRU --interval-minutes 1 --skip-discord-prepare --stop-when-idle
```

Run multiple channels:

```bash
python scripts/degold_autorun.py --channel STU --channel RRU --interval-minutes 1 --skip-discord-prepare
```

#### Other command patterns

Run the full ready queue in background (all channels):

```bash
python scripts/degold_autorun.py --interval-minutes 1 --skip-discord-prepare
```

Run a filtered subset in background and exit when those targets are drained:

```bash
python scripts/degold_autorun.py --interval-minutes 1 --skip-discord-prepare --stop-when-idle --card-id UAxE4onN --card-id wid1ATcA --card-id ab12Cd34
```

Run one maintenance cycle in background:

```bash
python scripts/degold_autorun.py --once
```

Run one filtered maintenance cycle in background:

```bash
python scripts/degold_autorun.py --once --skip-discord-prepare --card-id UAxE4onN --card-id wid1ATcA
```

### 3. Launch it with Claude Code background running

Use a background task or Bash with `run_in_background: true`.

The command should be started as a background job, not as a blocking foreground shell call.
The queue runner must be `scripts/degold_autorun.py`; a per-project watch task is only a monitor and does not replace the queue runner.
If `Degold/degold_autorun.lock` exists but the owner PID is dead or the heartbeat is stale, let `scripts/degold_autorun.py` take over. A stale lock file is not a reason to pause the workflow.

```text
run_in_background: true
```

### 4. Report immediately after launch

After starting the background task, tell the user:

- whether the run targets the whole queue, a channel filter, or specific card IDs
- the exact target card IDs and/or channel filter
- the log file path: `logs/degold_autorun.log`
- the state file path: `Degold/degold_autorun_state.json`
- the stop command:

```bash
python scripts/degold_autorun.py --stop
```

- If relevant, mention `Ctrl+F` as the Claude Code shortcut to kill background agents.
- If a stale stop file was auto-cleared at startup, mention that briefly and continue.

### 5. Keep track of queue progress

For status updates, use this order:

1. Check the background task output via `TaskOutput`.
2. Read `Degold/degold_autorun_state.json` for the last cycle summary.
3. Refresh queue status if needed:

```bash
python scripts/pipeline_queue_state.py show --limit 20 --show-urls
```

If queue state looks stale relative to project logs or `degold_autorun_state.json`, refresh it with:

```bash
python scripts/pipeline_queue_state.py sync
python scripts/pipeline_queue_state.py show --limit 20 --show-urls
```

- If the task launched a pipeline, report the running card ID, project path, and remaining ready cards.
- If the task did not launch because one pipeline is already running, say that and keep the background queue runner alive.
- If queue state chooses a fresh launch because a `CAPTION` checkpoint has no usable caption text, treat that as expected auto-recovery and keep the queue moving.
- That fresh fallback may use a generic raw voiceover file when a strict card-prefixed voiceover file is missing.
- If `TaskOutput` is empty or incomplete, trust `Degold/degold_autorun_state.json`, `logs/degold_autorun.log`, and queue-state output over the background task transcript.
- If the user wants per-project log monitoring, use the existing `/watch` skill on the active project path.
- For queue execution, default to `--skip-discord-prepare` unless the user explicitly wants Discord pipeline-complete ingestion during the run.
- Use `last_cycle.launch_outcome` from `Degold/degold_autorun_state.json` as the primary launch result:
  - `running`: the launched card is active
  - `startup_failed`: the card wrote run logs but never became running
  - `launched_pending_confirmation`: process was started but queue state has not confirmed it yet
  - `not_launched`: nothing was launched in that cycle
- Use top-level `suppressed_card_ids` from `Degold/degold_autorun_state.json` to identify cards that autorun is temporarily skipping after a startup failure.

### 6. Handle background task exit

When Claude Code reactivates after the background task exits:

- If the output says `Target card set drained; exiting autorun loop.`, report completion and stop.
- If the user requested stop, report that the queue runner was stopped cleanly.
- If `last_cycle.launch_outcome == startup_failed`, do not ask the user whether to investigate.
  Automatically inspect `last_cycle.launched_latest_log` or the latest project log for that card, summarize the startup failure, and report whether it looks project-specific or systemic.
- If the failed launch looks project-specific and other ready cards remain, continue the queue with the remaining ready cards instead of the failed card.
  The runner now suppresses startup-failed cards automatically and fast-continues to the next unsuppressed ready target; report the suppressed card IDs and keep watching.
- If the failure is the known `caption_checkpoint_without_usable_text` case, do not stop for permission. Let the next queue-selected launch fall back to fresh automatically and continue watching.
- If the failed launch looks systemic, stop the queue runner and report the blocker clearly.
- If the background task exited unexpectedly and targeted cards are still actionable, restart the same background command.
- If the full-queue runner exits unexpectedly, restart it in background after checking `logs/degold_autorun.log` and `Degold/degold_autorun_state.json`.
- If an individual project watch task ends cleanly, do not treat that as queue completion by itself. Confirm the `degold_autorun.py` background task is still active; if it is not, restart the queue runner.
- If queue-state maintenance fails, fix the queue/state blocker first and then restart `scripts/degold_autorun.py`.
  Do not bypass `/run-ready-queue` by launching a single project with `main.py` just to keep work moving.

### 7. Recovery after queue/state failure

- Treat errors from `scripts/pipeline_queue_state.py` or `Degold/pipeline_queue_state.json` as queue infrastructure failures, not project-specific pipeline failures.
- Recover in this order:
  1. inspect `logs/degold_autorun.log` and `Degold/degold_autorun_state.json`
  2. if a healthy `main.py --project ...` run is already active and queue state shows it as running, leave it alone and restart autorun so queue ownership is restored
  3. if a duplicate or orphaned direct `main.py` run conflicts with queue state, stop the conflicting process
  4. fix the queue/state issue
  5. run `python scripts/pipeline_queue_state.py sync`
  6. confirm with `python scripts/pipeline_queue_state.py show --limit 20 --show-urls`
  7. restart the same `scripts/degold_autorun.py` background command
- Only inspect or launch a specific project directly after the queue runner is healthy again and only when the failure is clearly project-specific.

## Pipeline Kill and Restart

When the user asks to kill/restart a running pipeline (e.g. to pick up code changes), use:

```bash
# Kill all running pipelines, sync state, and signal autorun to restart immediately
python scripts/pipeline_queue_state.py kill-pipeline

# Kill specific card only
python scripts/pipeline_queue_state.py kill-pipeline --card-id 0tsPiUY5

# Preview without killing
python scripts/pipeline_queue_state.py kill-pipeline --dry-run
```

This is the preferred method over manual process killing because it:
1. Finds `main.py` processes by project path (handles multiple Python installs)
2. Terminates gracefully (SIGTERM, then SIGKILL after 5s)
3. Syncs queue state so autorun sees `running=0`
4. Creates `degold_autorun.force_cycle` so autorun wakes from sleep immediately

Do NOT manually kill PIDs — use `kill-pipeline` instead.
Do NOT manually launch `main.py` after killing — let autorun relaunch via force-cycle.

## Rules

- Before launching, ask the user which channel/board to target if it's not obvious from context. Don't assume — a brief question saves a wrong run.
- A `/run-ready-queue` request is authorization to start or restart autorun once the target is confirmed; do not ask for a second confirmation after the user answers the channel/card question.
- Use `--channel` when the user specifies a board or channel name (STU, RRU, DSR). Use `--card-id` when they give specific card IDs. Use both together for intersection filtering.
- Let `prepare --run-ready` choose the next launch. It already avoids parallel runs.
- Do not manually run `main.py` for multiple projects in parallel.
- Do not bypass queue continuation by manually launching the next card with `main.py` when `/run-ready-queue` was the requested workflow.
- If `main.py` is already running and autorun is inactive, restart autorun so it can monitor the active card and continue the queue afterward.
- Do not use `--once` when the user expects continuous sequential autorun across multiple cards.
- Filtered runs from screenshots/manual ready lists should use repeated `--card-id`, `--stop-when-idle`, and usually `--skip-discord-prepare`.
- Channel-filtered runs should use `--channel`, `--stop-when-idle`, and usually `--skip-discord-prepare`.
- Only include `discord-prepare` when the user explicitly wants new Discord pipeline-complete posts ingested into the queue.
- Do not treat `Degold/degold_autorun.lock` by itself as proof that autorun is healthy; stale-lock takeover is normal recovery.
- Keep user-facing summaries short: launched card, channel filter, remaining ready list, warnings, and how to stop.
- When killing and restarting pipelines, always use `python scripts/pipeline_queue_state.py kill-pipeline` instead of manual process killing.
