# Stu VO Download Skill

Download voiceover audio from NEW AMERICA Trello cards, combine parts, create project folders. No pipeline start.

## When to Use

Use this skill when:
- User asks to download voiceovers for Stu/NEW AMERICA cards
- User wants to prepare VO audio for STU projects
- User asks to combine audio parts from Trello card descriptions

## What It Does

For each card with a "Script N - VO" Google Doc link in its Trello description:
1. Fetches the VO Google Doc (contains Drive links to audio parts: Pt1, Pt2, ...)
2. Downloads all audio parts from Google Drive
3. Combines them in order into a single `voiceover.mp3`
4. Creates the project folder (via `setup_project.py`) if it doesn't exist
5. Validates the combined audio with ffprobe (duration, size)
6. **Does NOT start the pipeline**

## Commands

```bash
# Preview which cards have VO docs to download
python scripts/stu/download_vo.py --dry-run

# Download and combine all cards with VO docs
python scripts/stu/download_vo.py

# Specific card only
python scripts/stu/download_vo.py --card-id KxQ0SoGh

# Re-download even if voiceover already exists
python scripts/stu/download_vo.py --force
```

## Default Behavior

When the user runs `/stu-vo-download`:

1. First sync the queue state:
   ```bash
   python scripts/pipeline_queue_state.py --state-file Stu/pipeline_queue_state.json --accounts-dir Stu/accounts --board-map-file Stu/board_channel_map.yaml --projects-root "E:/Edit Job/Stu" sync
   ```
2. Then run the VO download:
   ```bash
   python scripts/stu/download_vo.py
   ```

## Requirements

- `gdown` - for Google Drive downloads (`pip install gdown`)
- `pydub` - for audio combining (`pip install pydub`)
- `ffprobe` - for validation (bundled with ffmpeg)

## Key Info

| Field | Value |
|-------|-------|
| State File | `Stu/pipeline_queue_state.json` |
| Projects Root | `E:\Edit Job\Stu` |
| Script | `scripts/stu/download_vo.py` |
