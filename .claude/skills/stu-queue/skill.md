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

## Voiceover Detection (NEW AMERICA Card Format)

NEW AMERICA cards use a specific description format with Google Doc links for scripts and voiceovers. The pattern is:

```
Script N - [Google Doc link]        ← text script (NOT a voiceover)
Script N - VO - [Google Doc link]   ← voiceover Google Doc (this IS the VO)
```

### Readiness rules based on description parsing

| Description Pattern | Has Script | Has VO | Pipeline Ready |
|---|---|---|---|
| `Script N - [link]` + `Script N - VO - [link]` | Yes | **Yes** | Ready |
| `Script N - [link]` only (no VO line) | Yes | No | Not ready |
| `Script N -` + `Script N - VO -` (placeholders, no links) | No | No | Not ready |
| Empty description | No | No | Not ready |

**Important**: A Google Doc in the description is only a voiceover if its label contains "VO" or "VOICE OVER". The first Google Doc (labeled just "Script N") is the text script and must NOT be counted as a voiceover.

### When validating VO state

After syncing, verify readiness by parsing the card description text rather than relying solely on the queue's `has_raw_voiceover` flag, which may count any Google Doc as a VO candidate. The correct check is:
1. Parse the description for lines matching `Script.*VO.*\[http` (regex)
2. Only cards with a VO-labeled Google Doc link are truly ready

## Troubleshooting

### Queue shows 0 ready despite cards in Trello
Run sync to refresh:
```bash
python scripts/pipeline_queue_state.py $STU_FLAGS sync
python scripts/pipeline_queue_state.py $STU_FLAGS status
```

### Cards routing to wrong channel
All cards on the NEW AMERICA board should route to STU. The `board_channel_map.yaml` sets both `channel: STU` and `lipsync_channel: STU`. Ensure `Stu/accounts/david.env` has `TRELLO_BOARD_ID=6998b92db8fc2834b42b38dc` (the NEW AMERICA board, NOT the Military/War News board `699ddc7210f3d0fab35d2e5d`).

### False positive "ready" status
The queue may mark cards as ready if any Google Doc is detected in the description. Always cross-check using the VO detection rules above — only `Script N - VO - [link]` lines indicate actual voiceovers.

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
