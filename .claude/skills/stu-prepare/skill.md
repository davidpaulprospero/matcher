# Stu Prepare Skill

Prepare local projects for **ready** cards on the NEW AMERICA Trello board.

## When to Use

Use this skill when:
- User asks to prepare Stu projects
- User wants to set up local project folders for ready NEW AMERICA cards
- User asks to download voiceovers for Stu queue cards

## What It Does

1. Syncs the Stu queue state from the NEW AMERICA Trello board
2. Finds cards that are **ready** but don't have a local project folder yet
3. For each unprepared card, runs `newproject` to:
   - Create the project folder under `E:\Edit Job\Stu\Stu\`
   - Download the voiceover from the Google Doc link on the card
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

## Commands

```bash
# Dry run: preview which cards need projects
python scripts/pipeline_queue_state.py $STU_FLAGS prepare --sync-first --dry-run

# Prepare all unprepared ready cards
python scripts/pipeline_queue_state.py $STU_FLAGS prepare --sync-first

# Prepare a specific card
python scripts/pipeline_queue_state.py $STU_FLAGS prepare --sync-first --card-id CARD_SHORT_ID

# Prepare with limit
python scripts/pipeline_queue_state.py $STU_FLAGS prepare --sync-first --limit 2
```

## Default Behavior

When the user runs `/stu-prepare` without specifics:

1. First do a **dry run** to show what would be prepared:
   ```bash
   python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" prepare --sync-first --dry-run
   ```
2. Show the user the list of cards that need projects and ask for confirmation
3. On confirmation, run the actual prepare:
   ```bash
   python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" prepare --sync-first
   ```
4. After prepare completes, run status to show updated state:
   ```bash
   python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" status
   ```

## Troubleshooting

### "No missing projects detected"
All ready cards already have local project directories. Run `/stu-queue` to check current state.

### Project creation fails
- Check GWS token in `Stu/accounts/david.env` is not expired
- Verify the Google Doc link on the Trello card is accessible
- Check `E:\Edit Job\Stu\` directory exists and is writable

### Voiceover download fails
The card's Google Doc link may not contain audio. Cards with Google Docs in their description are treated as voiceover sources. If the doc is a script (text only), the pipeline will handle text-to-speech conversion downstream.

## Key Info

| Field | Value |
|-------|-------|
| Trello Board | NEW AMERICA (`6998b92db8fc2834b42b38dc`) |
| Account | David |
| Channel Code | STU |
| Local Projects Root | `E:\Edit Job\Stu` |
| Project Subfolder | `E:\Edit Job\Stu\Stu\{project}__YYYY-MM-DD` |
