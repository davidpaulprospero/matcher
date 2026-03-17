# Pipeline Queue Skill

Build and operate a JSON-based pipeline queue from Trello cards and Discord pipeline-complete messages.

## When to Use

Use this skill when:
- User asks about pipeline queue status
- User wants to add cards to the queue
- User mentions "pipeline complete" from Discord
- User asks to prepare or check pipeline readiness

## Capabilities

- **Track Pipeline States** - All project pipeline states in `Degold/pipeline_queue_state.json`
- **Queue Pending Work** - Store full Trello card data for pending/not-started work
- **Auto-prepare Missing Projects** - Detect and prepare local projects that don't exist
- **Ingest Discord Messages** - Parse "Script / Voiceover / Description -- Pipeline Complete" messages
- **Readiness Checks** - Select next pipeline that can start only after checks pass

## Key Files

- `Degold/pipeline_queue_state.json` - Main queue state
- `Degold/discord_pipeline_projects.json` - Discord channel to project mapping
- `Degold/board_channel_map.yaml` - Trello board to Discord channel mapping
- `scripts/pipeline_queue_state.py` - CLI for queue operations
- `Degold/channel_routing.py` - Route cards to correct Discord channels

## Commands

```bash
# Default queue maintenance + status
python scripts/pipeline_queue_state.py archive-completed --sync-first
python scripts/pipeline_queue_state.py status

# Show queue status only
python scripts/pipeline_queue_state.py status

# Sync queue state without archiving
python scripts/pipeline_queue_state.py sync

# Prepare local projects from Discord pipeline-complete messages
python scripts/pipeline_queue_state.py discord-prepare
```

## Default Behavior

When the user runs `/pipeline-queue` without a more specific request, do this by default:

1. Run `python scripts/pipeline_queue_state.py archive-completed --sync-first`
2. Then run `python scripts/pipeline_queue_state.py status`

This default flow should archive completed local project folders first, refresh queue state, and then report the current queue summary.

## Troubleshooting

### Queue shows 0 ready despite cards in Trello
Run sync to refresh:
```bash
python scripts/pipeline_queue_state.py sync
python scripts/pipeline_queue_state.py status
```

### Prepare fails with "newproject_failed"
Check the error details - common causes:
- Missing voiceover folder on card (check Trello card attachments)
- Wrong account credentials for the card's board
- GWS token missing: `grep GOOGLE_WORKSPACE_CLI_TOKEN Degold/accounts/david.env`

### Voiceover not downloading
The system now uses GWS (Google Workspace CLI) for Drive downloads. Verify:
1. Account has GWS token in `Degold/accounts/{account}.env`
2. GWS is available: `python -c "import sys; sys.path.insert(0,'scripts'); import gws_drive; print(gws_drive.gws_is_available())"`

### Lock file issues
If autorun won't start due to stale lock:
```bash
rm Degold/degold_autorun.lock
```

## Key Account Mappings

| Channel | Account | Board |
|---------|---------|-------|
| RRU | David | Military / War News |
| DSR | Stuart | DeepSeaReports |
