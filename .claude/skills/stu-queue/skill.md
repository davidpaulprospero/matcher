# Stu Queue Skill

Build and operate a JSON-based pipeline queue for the **NEW AMERICA** Trello board (David's account), stored in `Stu/`.

## When to Use

Use this skill when:
- User asks about Stu pipeline queue status
- User wants to add cards to the Stu queue
- User mentions "Stu board" or "NEW AMERICA" pipeline status
- User asks to prepare or check Stu pipeline readiness

## Capabilities

- **Track Pipeline States** - All NEW AMERICA project pipeline states in `Stu/pipeline_queue_state.json`
- **Queue Pending Work** - Store full Trello card data for pending/not-started work
- **Auto-prepare Missing Projects** - Detect and prepare local projects under `E:\Edit Job\Stu\`
- **Ingest Discord Messages** - Parse pipeline-complete messages for STU channel
- **Readiness Checks** - Select next pipeline that can start only after checks pass

## Key Files

- `Stu/pipeline_queue_state.json` - Main queue state
- `Stu/board_channel_map.yaml` - Board routing (NEW AMERICA -> STU)
- `Stu/accounts/david.env` - Account config (David's Trello creds, NEW AMERICA board)
- `Degold/channel_routing.py` - Shared channel routing (includes STU)
- `scripts/pipeline_queue_state.py` - CLI for queue operations

## CLI Override Pattern

All commands use these flags to target the Stu queue instead of Degold:

```bash
STU_FLAGS="--state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root E:/Edit\ Job/Stu"
```

## Commands

```bash
# Default queue maintenance + status
python scripts/pipeline_queue_state.py $STU_FLAGS archive-completed --sync-first
python scripts/pipeline_queue_state.py $STU_FLAGS status

# Show queue status only
python scripts/pipeline_queue_state.py $STU_FLAGS status

# Sync queue state without archiving
python scripts/pipeline_queue_state.py $STU_FLAGS sync

# Prepare local projects from Discord pipeline-complete messages
python scripts/pipeline_queue_state.py $STU_FLAGS discord-prepare
```

## Default Behavior

When the user runs `/stu-queue` without a more specific request, do this by default:

1. Run `python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" archive-completed --sync-first`
2. Then run `python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" status`

This archives completed local project folders first, refreshes queue state from the NEW AMERICA Trello board, and then reports the current queue summary.

## Troubleshooting

### Queue shows 0 ready despite cards in Trello
Run sync to refresh:
```bash
python scripts/pipeline_queue_state.py $STU_FLAGS sync
python scripts/pipeline_queue_state.py $STU_FLAGS status
```

### Cards routing to wrong channel
Cards on the NEW AMERICA board with `RennReportsUS` labels will route to RRU via label overrides. This is expected. Cards without labels default to STU channel via the board map.

### Local project directory
STU projects live under `E:\Edit Job\Stu\`. The `CHANNEL_DIR_ALIASES` in `pipeline_queue_state.py` maps STU to `("Stu", "STU")` folder names.

## Key Info

| Field | Value |
|-------|-------|
| Trello Board | NEW AMERICA (`6998b92db8fc2834b42b38dc`) |
| Board Short Link | `dsl7VdgG` |
| Account | David |
| Channel Code | STU |
| Local Projects Root | `E:\Edit Job\Stu` |
| State File | `Stu/pipeline_queue_state.json` |
