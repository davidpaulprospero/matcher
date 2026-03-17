# Degold Autorun Skill

Monitor Trello-assigned Degold pipeline work on an hourly cadence and auto-queue one pipeline at a time.

## When to Use

Use this skill when:
- User asks to run autorun overnight
- User wants to keep the pipeline queue moving automatically
- User asks to monitor Degold pipeline work hourly

## Capabilities

- **Detect Newly Assigned Cards** - Scan configured Degold accounts for new cards
- **Hourly Cadence** - Run every 60 minutes without local LLM review
- **Auto-queue** - Queue one pipeline at a time for processing
- **Multi-account Support** - Work across multiple Degold Trello accounts

## Configuration

Accounts are configured in `Degold/accounts/*.env` files:
- `Degold/accounts/david.env` - David's account
- `Degold/accounts/stu.env` - Stu's account

## Key Files

- `Degold/degold_autorun_state.json` - Autorun state tracking
- `Degold/degold_autorun.lock` - Lock file to prevent concurrent runs
- `Degold/degold_autorun.stop` - Stop signal file
- `scripts/degold_autorun.py` - Main autorun script

## Usage

```bash
# Start autorun (runs in background)
python scripts/degold_autorun.py start

# Check status
python scripts/degold_autorun.py status

# Stop autorun
python scripts/degold_autorun.py stop
```

## Notes

- One pipeline at a time to avoid conflicts
- Skips cards that need LLM review
- Logs to console and optionally to file
