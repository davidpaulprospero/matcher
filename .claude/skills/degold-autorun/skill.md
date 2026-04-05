# Degold Autorun Skill (Deprecated)

> **Use `/autorun` instead.** The unified autorun manages all boards from a single process.
> For Degold only: `/autorun degold`. For all boards: `/autorun`.

Legacy skill for Degold-only autorun.

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

Accounts are configured in `clients/degold/accounts/*.env` files:
- `clients/degold/accounts/david.env` - David's account
- `clients/degold/accounts/stu.env` - Stu's account

## Key Files

- `clients/degold/degold_autorun_state.json` - Autorun state tracking
- `clients/degold/degold_autorun.lock` - Lock file to prevent concurrent runs
- `clients/degold/degold_autorun.stop` - Stop signal file
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
