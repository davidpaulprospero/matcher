# Lipsync Skill

Submit Degold lipsync jobs from Trello URLs using Playwright MCP.

## When to Use

Use this skill when:
- User asks to run `/lipsync`
- User wants to submit a lipsync job from a Trello card URL
- User mentions submitting a lipsync to Degold

## Workflow

1. **Get Trello URL** - Ask user for the Trello card URL
2. **Open Browser** - Use Playwright MCP to navigate to Trello
3. **Extract Info** - Get card title, description, attached audio/video
4. **Submit to Lipsync** - Fill Degold lipsync form and submit
5. **Confirm** - Verify submission succeeded and get job ID

## Notes

- Browser stays open in extension mode for background processing
- Store job ID in `Degold/lipsync_tracking.json` for tracking
- Use `Degold/queue_lipsync.py` for batch submissions
