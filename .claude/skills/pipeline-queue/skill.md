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
# Check queue status
python scripts/pipeline_queue_state.py status

# Add Trello card to queue
python scripts/pipeline_queue_state.py add <trello_url>

# Process Discord pipeline-complete
python scripts/pipeline_queue_state.py discord <message>
```
