---
name: dvids-download
description: Download DVIDS Hub videos matching a project's voiceover SRT topic, segmented to <=5MB chunks with >=60s total duration
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# DVIDS Download

Download DVIDS Hub videos related to a project's voiceover topic, segmented to small file sizes.

## When to Use

Use this skill when:
- User asks to download DVIDS videos for a project
- User mentions "DVIDS", "military footage", or "defense videos" for a project
- User wants topic-based video downloads from dvidshub.net

## What It Does

1. Locates the project's `voiceover.srt` file
2. Extracts topic keywords from the SRT content
3. Searches DVIDS Hub API for related videos (default: 10 per query)
4. Downloads each video and trims it to a single short clip (~10s, <=5MB)
5. Accumulates clips until total duration >= 60s
6. Saves 1 clip per video to `{project}/dvids/`

## Key Files

- `scripts/download_dvids.py` - Main download script
- `.env` - Must contain `DVIDS_API_KEY`

## Commands

```bash
# Download for a project directory
python scripts/download_dvids.py "E:/Edit Job/Degold/DeepSeaReports/PROJECT_DIR"

# Dry run: preview without downloading
python scripts/download_dvids.py "PROJECT_DIR" --dry-run

# Explicit search query (skip SRT parsing)
python scripts/download_dvids.py "PROJECT_DIR" --query "naval operations"

# Custom duration/size targets
python scripts/download_dvids.py "PROJECT_DIR" --min-total-dur 120 --max-segment-mb 3

# Explicit SRT file
python scripts/download_dvids.py "PROJECT_DIR" --srt "PROJECT_DIR/custom.srt"
```

## Default Behavior

When the user runs `/dvids-download <arg>`:

### If arg is a project directory path:

1. Verify `voiceover.srt` exists in the directory
2. Run dry-run first to show what will be downloaded:
   ```bash
   python scripts/download_dvids.py "PROJECT_PATH" --dry-run
   ```
3. On confirmation, run the actual download:
   ```bash
   python scripts/download_dvids.py "PROJECT_PATH"
   ```
4. Report the summary (videos downloaded, segments created, total duration, total size)

### If arg is a Trello card ID (8-char alphanumeric like ArxAPqCG):

1. Resolve the card to a project directory:
   ```bash
   # Check Degold queue state
   python -c "
   import json
   state = json.load(open('Degold/pipeline_queue_state.json'))
   card = state.get('pipelines', {}).get('CARD_ID', {})
   print(card.get('local_project_dir', 'NOT_FOUND'))
   print(card.get('name', ''))
   "
   ```
   ```bash
   # Also check STU queue state
   python -c "
   import json
   state = json.load(open('Stu/pipeline_queue_state.json'))
   card = state.get('pipelines', {}).get('CARD_ID', {})
   print(card.get('local_project_dir', 'NOT_FOUND'))
   print(card.get('name', ''))
   "
   ```
   ```bash
   # Scan filesystem for card ID in folder name
   ls -d E:/Edit\ Job/Degold/*/CARD_ID-* E:/Edit\ Job/Stu/CARD_ID-* 2>/dev/null
   ```
2. Once resolved, proceed as with a project directory (dry-run first, then download)

### If no arg or user wants queue sync:

1. Sync the queue first:
   ```bash
   python scripts/pipeline_queue_state.py --state-file Degold/pipeline_queue_state.json archive-completed --sync-first
   ```
2. Show status:
   ```bash
   python scripts/pipeline_queue_state.py --state-file Degold/pipeline_queue_state.json status
   ```
3. Ask the user which cards to download DVIDS videos for
4. For each selected card, resolve to project dir and run the download

## Output

One clip per video saved to `{project}/dvids/`:
- Files <= 5MB: saved as-is (e.g., `d999196.mp4`)
- Files > 5MB: trimmed to a single `d999196_clip.mp4` (~10s, <=5MB)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "DVIDS API key not found" | Set `DVIDS_API_KEY` in `.env` |
| "ffmpeg not found" | Install ffmpeg and add to PATH |
| "No segments found in SRT" | Use `--query` to specify search terms manually |
| Low total duration | Try `--query` with broader terms or `--max-results 50` |
