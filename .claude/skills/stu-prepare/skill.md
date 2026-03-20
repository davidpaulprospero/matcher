# Stu Prepare Skill

Prepare local projects for **ready** cards on the NEW AMERICA Trello board.

## When to Use

Use this skill when:
- User asks to prepare Stu projects
- User wants to set up local project folders for ready NEW AMERICA cards
- User asks to download voiceovers for Stu queue cards

## What It Does

1. Syncs the Stu queue state from the NEW AMERICA Trello board
2. **Validates voiceover readiness** by parsing card descriptions (see VO detection below)
3. Finds cards that are truly ready but don't have a local project folder yet
4. For each unprepared card, runs `newproject` to:
   - Create the project folder under `E:\Edit Job\Stu\Stu\`
   - Download the voiceover from the VO Google Doc link on the card
   - Set up `project_config.yaml`

## Key Files

- `Stu/pipeline_queue_state.json` - Queue state (tracks which cards need projects)
- `Stu/accounts/david.env` - Trello + GWS credentials
- `src/cli/newproject.py` - Project creation logic (STU channel support)
- `scripts/pipeline_queue_state.py` - `prepare` subcommand

## CLI Override Pattern

```bash
STU_FLAGS="--state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root E:/Edit\ Job/Stu"
```

## CRITICAL: Do NOT start pipelines

**NEVER pass `--run-ready` to the prepare command.** This skill only creates project folders and downloads voiceovers. It must NOT auto-start the pipeline after preparation. The user will decide when to run pipelines separately.

## Commands

```bash
# Dry run: preview which cards need projects
python scripts/pipeline_queue_state.py $STU_FLAGS prepare --sync-first --dry-run

# Prepare all unprepared ready cards (NO --run-ready!)
python scripts/pipeline_queue_state.py $STU_FLAGS prepare --sync-first

# Prepare specific cards by ID (NO --run-ready!)
python scripts/pipeline_queue_state.py $STU_FLAGS prepare --sync-first --card-id CARD_ID_1 --card-id CARD_ID_2

# Prepare with limit
python scripts/pipeline_queue_state.py $STU_FLAGS prepare --sync-first --limit 2
```

## Default Behavior

When the user runs `/stu-prepare` without specifics:

1. First validate VO readiness (see detection rules below) to identify which ready cards actually have VOs
2. Show the user the list of cards that need projects and ask for confirmation
3. On confirmation, run prepare for only the cards with verified VOs, using `--card-id` for each:
   ```bash
   python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" prepare --card-id CARD1 --card-id CARD2
   ```
4. After prepare completes, run status to show updated state:
   ```bash
   python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" status
   ```

## Voiceover Detection (NEW AMERICA Card Format)

NEW AMERICA cards use a specific description format with Google Doc links:

```
Script N - [Google Doc link]        ← text script only (NOT a voiceover)
Script N - VO - [Google Doc link]   ← voiceover Google Doc (this IS the VO)
```

### Readiness rules

| Description Pattern | Has VO | Prepare? |
|---|---|---|
| `Script N - [link]` + `Script N - VO - [link]` | **Yes** | Yes |
| `Script N - [link]` only (no VO line) | No | Skip |
| `Script N -` + `Script N - VO -` (placeholders, no links) | No | Skip |
| Empty description | No | Skip |

**Important**: Only Google Docs labeled with "VO" or "VOICE OVER" are voiceovers. The first doc (labeled just "Script N") is the text script and must NOT be used as VO input.

### Validation step before preparing

Before running `prepare`, always validate VO state by parsing card descriptions from the state file:

```python
import re
# A card has a VO if its description contains a VO-labeled line with a Google Doc link
has_vo = bool(re.search(r'Script.*VO.*\[http', description))
```

Only prepare cards where this check passes. Report cards that the queue marks "ready" but lack a VO doc link as false positives.

## Troubleshooting

### "No missing projects detected"
All ready cards already have local project directories. Run `/stu-queue` to check current state.

### Project creation fails
- Check GWS token in `Stu/accounts/david.env` is not expired
- Verify the VO Google Doc link on the Trello card is accessible
- Check `E:\Edit Job\Stu\` directory exists and is writable

### False positive "ready" cards
The queue's `has_raw_voiceover` flag may count any Google Doc as a VO candidate. Always cross-check using the VO detection rules above before preparing.

## Key Info

| Field | Value |
|-------|-------|
| Trello Board | NEW AMERICA (`6998b92db8fc2834b42b38dc`) |
| Account | David |
| Channel Code | STU |
| Local Projects Root | `E:\Edit Job\Stu` |
| Project Subfolder | `E:\Edit Job\Stu\Stu\{cardid}-{project}__YYYY-MM-DD` |
