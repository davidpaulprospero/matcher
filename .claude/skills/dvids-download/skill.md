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
4. **Skips videos already downloaded for other projects** (via global tracking)
5. Downloads each video and trims it to a single short clip (~10s, <=5MB)
6. Accumulates clips until min-clips reached (default: 20)
7. Saves 1 clip per video to `{project}/dvids/`

## Key Files

- `scripts/download_dvids.py` - Main download script
- `.cache/dvids_downloaded.json` - Global tracking file (cross-project dedup)
- `.env` - Must contain `DVIDS_API_KEY`

## Global Tracking

A JSON file at `.cache/dvids_downloaded.json` tracks every downloaded video ID across all projects. When a video was already downloaded for another project, it is skipped with a `[SKIP]` message showing the original project name. This prevents duplicate downloads across projects with overlapping topics.

To bypass tracking (re-download everything): use `--no-tracking`.

## Commands

```bash
# Download for a project directory
python scripts/download_dvids.py "E:/Edit Job/Degold/DeepSeaReports/PROJECT_DIR"

# Dry run: preview without downloading (tracking still filters dupes)
python scripts/download_dvids.py "PROJECT_DIR" --dry-run

# Explicit search queries, comma-separated (skip SRT parsing)
python scripts/download_dvids.py "PROJECT_DIR" --query "submarine,torpedo,USS destroyer"

# Custom clip count / size targets
python scripts/download_dvids.py "PROJECT_DIR" --min-clips 30 --max-segment-mb 3

# Explicit SRT file
python scripts/download_dvids.py "PROJECT_DIR" --srt "PROJECT_DIR/custom.srt"

# Ignore tracking — download even if already downloaded elsewhere
python scripts/download_dvids.py "PROJECT_DIR" --no-tracking
```

## Default Behavior

When the user runs `/dvids-download <arg>`:

### If arg is a project directory path:

1. Verify `voiceover.srt` exists in the directory (check `voiceover/` subdirectory too)
2. Run dry-run first to show what will be downloaded:
   ```bash
   python scripts/download_dvids.py "PROJECT_PATH" --srt "SRT_PATH" --dry-run
   ```
3. On confirmation, run the actual download:
   ```bash
   python scripts/download_dvids.py "PROJECT_PATH" --srt "SRT_PATH"
   ```
4. Report the summary (videos downloaded, skipped, total duration, total size)

### If arg is a Trello card ID (8-char alphanumeric like ArxAPqCG):

1. Resolve the card to a project directory:
   ```bash
   # Check Degold queue state
   python -c "
   import json
   state = json.load(open('clients/degold/pipeline_queue_state.json'))
   card = state.get('pipelines', {}).get('CARD_ID', {})
   print(card.get('local_project_dir', 'NOT_FOUND'))
   print(card.get('name', ''))
   "
   ```
   ```bash
   # Also check STU queue state
   python -c "
   import json
   state = json.load(open('clients/stu/pipeline_queue_state.json'))
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
   python scripts/pipeline_queue_state.py --state-file clients/degold/pipeline_queue_state.json archive-completed --sync-first
   ```
2. Show status:
   ```bash
   python scripts/pipeline_queue_state.py --state-file clients/degold/pipeline_queue_state.json status
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
| Low clip count / many skipped | Topics overlap with prior projects. Use `--no-tracking` to force re-download, or try `--query` with different terms |
| Low total duration | Try `--query` with multiple comma-separated terms or `--max-results 50` |
