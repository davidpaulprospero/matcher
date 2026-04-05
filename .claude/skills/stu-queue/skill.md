---
name: stu-queue
description: Manage and run the STU/NEW AMERICA pipeline queue. Use for status checks, syncing, preparing projects, and running ready pipelines for the NEW AMERICA Trello board. Supports both queue maintenance and sequential pipeline execution.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
  - Task
  - TaskOutput
---

# Stu Queue Skill

Manage the STU/NEW AMERICA pipeline queue — sync, status, and prepare operations.

> **For running the STU queue**, use `/autorun stu` instead. The unified autorun handles all boards.

## Source of Truth

- `scripts/pipeline_queue_state.py --board stu` - Queue operations (shorthand for all STU flags)
- `Stu/pipeline_queue_state.json` - STU queue state
- `Stu/board_channel_map.yaml` - Board routing (NEW AMERICA -> STU)
- `Stu/accounts/david.env` - Account config
- `config/board_registry.yaml` - Board registry (used by `--board` shorthand)

## Usage

```text
/stu-queue                          # Default: sync + status
/stu-queue status                   # Show queue status only
/autorun stu                        # Run the STU queue (use unified autorun)
/autorun stu KxQ0SoGh ujKR12Dg     # Run specific STU cards
```

### Simplified Commands (using --board shorthand)

```bash
# Sync + status (replaces the old 4-flag pattern)
python scripts/pipeline_queue_state.py --board stu sync
python scripts/pipeline_queue_state.py --board stu show --limit 20 --show-urls

# Prepare a project
python scripts/pipeline_queue_state.py --board stu prepare --card-id KxQ0SoGh
```

## Instructions

### 0. Ask before running

When the user says `/stu-queue` without further context, determine their intent:

1. **Status only?** — If they just want to check the queue, run the default sync + status flow (no confirmation needed).
2. **Run the queue?** — If they want to actually execute pipelines, confirm:
   - Which cards? All ready, or specific IDs?
   - Continuous or one-shot?
   - Then proceed to the execution flow.

Skip questions when intent is clear (e.g., "run the stu queue", "stu-queue run").

### Default Behavior (status/maintenance)

When `/stu-queue` is invoked without "run" or card IDs:

1. Archive completed + sync:
```bash
python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" archive-completed --sync-first
```

2. Show status:
```bash
python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" status
```

### Execution Flow (running pipelines)

When the user wants to run STU pipelines, use the pipeline autorun with STU isolation flags.

The common STU autorun flags (used in all commands below):

```bash
STU_COMMON="--channel STU --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root E:/Edit\ Job/Stu --autorun-state-file Stu/stu_autorun_state.json --lock-file Stu/stu_autorun.lock --stop-file Stu/stu_autorun.stop --log-file logs/stu_autorun.log"
```

#### Run all ready STU cards (continuous):

```bash
python scripts/degold_autorun.py --channel STU --interval-minutes 1 --skip-discord-prepare --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" --autorun-state-file Stu/stu_autorun_state.json --lock-file Stu/stu_autorun.lock --stop-file Stu/stu_autorun.stop --log-file logs/stu_autorun.log
```

#### Run all ready STU cards, exit when drained:

```bash
python scripts/degold_autorun.py --channel STU --interval-minutes 1 --skip-discord-prepare --stop-when-idle --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" --autorun-state-file Stu/stu_autorun_state.json --lock-file Stu/stu_autorun.lock --stop-file Stu/stu_autorun.stop --log-file logs/stu_autorun.log
```

#### Run specific STU cards:

```bash
python scripts/degold_autorun.py --channel STU --card-id KxQ0SoGh --card-id ujKR12Dg --interval-minutes 1 --skip-discord-prepare --stop-when-idle --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" --autorun-state-file Stu/stu_autorun_state.json --lock-file Stu/stu_autorun.lock --stop-file Stu/stu_autorun.stop --log-file logs/stu_autorun.log
```

#### One maintenance cycle:

```bash
python scripts/degold_autorun.py --channel STU --once --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" --autorun-state-file Stu/stu_autorun_state.json --lock-file Stu/stu_autorun.lock --stop-file Stu/stu_autorun.stop --log-file logs/stu_autorun.log
```

### Launch as detached process

Launch the autorun as a detached process so it survives Claude Code session restarts.

**Option A — Claude Code background task** (use `run_in_background: true`):

```bash
python scripts/degold_autorun.py --channel STU --interval-minutes 1 --skip-discord-prepare --stop-when-idle --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" --autorun-state-file Stu/stu_autorun_state.json --lock-file Stu/stu_autorun.lock --stop-file Stu/stu_autorun.stop --log-file logs/stu_autorun.log
```

**Option B — Standalone detached process** (survives Claude Code crashes):

Linux:
```bash
nohup python scripts/degold_autorun.py --channel STU --interval-minutes 1 --skip-discord-prepare --stop-when-idle --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "/path/to/projects/Stu" --autorun-state-file Stu/stu_autorun_state.json --lock-file Stu/stu_autorun.lock --stop-file Stu/stu_autorun.stop --log-file logs/stu_autorun.log > /dev/null 2>&1 &
```

Windows:
```bash
powershell -Command "Start-Process -FilePath python -ArgumentList 'scripts/degold_autorun.py --channel STU --interval-minutes 1 --skip-discord-prepare --stop-when-idle --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root \"E:/Edit Job/Stu\" --autorun-state-file Stu/stu_autorun_state.json --lock-file Stu/stu_autorun.lock --stop-file Stu/stu_autorun.stop --log-file logs/stu_autorun.log' -WindowStyle Hidden"
```

After launching, report:

- Channel filter: STU
- Target card IDs (or "all ready STU cards")
- Log file: `logs/stu_autorun.log`
- Stop command: `python scripts/degold_autorun.py --stop --stop-file Stu/stu_autorun.stop`
- Verify running: `python -c "import json; d=json.load(open('Stu/stu_autorun.lock')); print(f'pid={d[\"pid\"]} status={d[\"status\"]} heartbeat={d[\"heartbeat_at\"]}')"` or `cat Stu/stu_autorun.lock | python -m json.tool`

### Check next ready STU card

```bash
python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" next --channel STU
```

### Prepare STU projects (dry run)

```bash
python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" prepare --channel STU --dry-run
```

## Voiceover Detection (NEW AMERICA Card Format)

NEW AMERICA cards use a specific description format with Google Doc links:

```
Script N - [Google Doc link]        <- text script (NOT a voiceover)
Script N - VO - [Google Doc link]   <- voiceover Google Doc (this IS the VO)
```

A Google Doc is only a voiceover if its label contains "VO" or "VOICE OVER". The first Google Doc (labeled just "Script N") is the text script and must NOT be counted as a voiceover.

After syncing, verify readiness by parsing the card description for lines matching `Script.*VO.*\[http` — only those indicate actual voiceovers.

## Troubleshooting

### Queue shows 0 ready despite cards in Trello
```bash
python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" sync
python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" status
```

### State file contaminated with non-STU cards
If the STU state file shows 42+ cards or cards with `channel=?`, the autorun was called without `--accounts-dir`/`--board-map-file`/`--projects-root`, causing it to sync all Degold boards into the STU state file. Fix by re-syncing with full STU flags:
```bash
python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" sync
```

### Cards routing to wrong channel
All cards on the NEW AMERICA board should route to STU. Ensure `Stu/accounts/david.env` has `TRELLO_BOARD_ID=6998b92db8fc2834b42b38dc`.

### False positive "ready" status
Cross-check using the VO detection rules above — only `Script N - VO - [link]` lines indicate actual voiceovers.

## Key Info

| Field | Value |
|-------|-------|
| Trello Board | NEW AMERICA (`6998b92db8fc2834b42b38dc`) |
| Board Short Link | `dsl7VdgG` |
| Account | David |
| Channel Code | STU |
| Local Projects Root | `E:\Edit Job\Stu` |
| State File | `Stu/pipeline_queue_state.json` |

## Rules

- Before running pipelines, ask which cards to target if not obvious from context.
- Always pass ALL STU override flags (queue, accounts, board map, projects root, autorun state, lock, stop, log) to `degold_autorun.py`. Never rely on defaults — they point to Degold.
- Always use `--state-file Stu/pipeline_queue_state.json` when targeting the STU queue.
- Do not manually launch `main.py` — let autorun handle it.
- For status-only requests, no confirmation needed — just sync and show.
- Keep user-facing summaries short: launched card, remaining ready list, and how to stop.
