---
name: autorun
description: Unified autorun for all Trello boards. Manages pipeline queue across Degold, STU, and all registered boards with a single command. Use when asked to run autorun, keep the queue moving, run pipelines overnight, or run the queue for any board.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
  - Task
  - TaskOutput
---

# Unified Autorun Skill

Manages pipeline execution across ALL registered Trello boards with a single autorun process. Reads `config/board_registry.yaml` to know which boards exist and their settings.

Replaces the old per-board approach where Degold and STU required separate autorun instances with different flag sets.

## Source of Truth

- `config/board_registry.yaml` - Board registry (all boards and their settings)
- `scripts/degold_autorun.py --unified` - The autorun script in unified mode
- `autorun_state.json` - Unified autorun state (project root)
- `autorun.lock` - Single lock file (project root)
- `autorun.stop` - Stop signal (project root)
- `logs/autorun.log` - Unified log
- Per-board queue state files stay in their board directories (e.g. `clients/degold/pipeline_queue_state.json`, `clients/stu/pipeline_queue_state.json`)

## Usage

```text
/autorun                          # Start for all enabled boards
/autorun degold                   # Only Degold board
/autorun stu                      # Only STU board
/autorun KxQ0SoGh ujKR12Dg       # Specific card IDs (auto-detects board)
/autorun status                   # Cross-board status
```

## Instructions

### 0. Ask before launching (when ambiguous)

Skip if the user's intent is clear. Otherwise ask:
1. **All boards or specific?** - Default is all enabled boards. If user says "Stu" or "NEW AMERICA", use `--board stu`.
2. **Specific cards or full queue?** - If user provides card IDs, use `--card-id`.
3. **Continuous or one-shot?** - Default continuous. Only use `--once` if explicitly requested.

### 1. Pick the command

#### All boards (default):

```bash
python scripts/degold_autorun.py --unified --interval-minutes 1
```

#### Single board:

```bash
python scripts/degold_autorun.py --unified --board stu --interval-minutes 1
python scripts/degold_autorun.py --unified --board degold --interval-minutes 1
```

#### Specific cards (exit when drained):

```bash
python scripts/degold_autorun.py --unified --card-id KxQ0SoGh --card-id ujKR12Dg --interval-minutes 1 --stop-when-idle
```

#### One maintenance cycle:

```bash
python scripts/degold_autorun.py --unified --once
```

#### Status:

```bash
python scripts/degold_autorun.py --unified --status
```

#### Stop:

```bash
python scripts/degold_autorun.py --unified --stop
```

### 2. Launch with background running

```text
run_in_background: true
```

### 3. Report after launch

Tell the user:
- Which boards are active
- Card IDs or channel filter if any
- Log: `logs/autorun.log`
- State: `autorun_state.json`
- Stop: `python scripts/degold_autorun.py --unified --stop`

### 4. Monitor progress

1. Check background task output via `TaskOutput`
2. Read `autorun_state.json` for last cycle summary (includes `board_key`)
3. Per-board queue status:

```bash
python scripts/pipeline_queue_state.py --board degold show --limit 20 --show-urls
python scripts/pipeline_queue_state.py --board stu show --limit 20 --show-urls
```

### 5. Handle exit

- `Target set drained` = all cards processed, report completion
- `startup_failed` = inspect the project log, report if project-specific or systemic
- Unexpected exit with remaining targets = restart the same command
- Autorun auto-suppresses failed cards and fast-continues to next ready card

## How It Works

Each cycle:
1. Syncs ALL registered boards (archive-completed + sync for each)
2. Collects ready cards across all boards
3. Picks the highest-priority board's first ready card
4. Runs prepare for that board with board-specific settings
5. Waits for the pipeline to complete
6. Repeats

Board priority is configured in `config/board_registry.yaml` (higher number = picked first when multiple boards have ready cards).

## Board Registry

```yaml
# config/board_registry.yaml
boards:
  degold:
    priority: 10
    state_file: "clients/degold/pipeline_queue_state.json"
    accounts_dir: "clients/degold/accounts"
    ...
  stu:
    priority: 5
    state_file: "clients/stu/pipeline_queue_state.json"
    ...
```

To add a new board: add its entry to the registry, create the board directory with `board_channel_map.yaml` and `accounts/`, and it's automatically included.

## Rules

- Always use `--unified` flag. Without it, the script falls back to legacy single-board mode.
- Let `prepare --run-ready` choose the next launch. Do not manually run `main.py`.
- For per-board queue operations (sync, status, prepare), use `python scripts/pipeline_queue_state.py --board <key> <command>`.
- Do not use the old flag-based approach (`--state-file clients/stu/... --accounts-dir clients/stu/...`). Use `--board` shorthand instead.
- Keep summaries short: boards, launched card, remaining ready, how to stop.
