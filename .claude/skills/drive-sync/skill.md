# Drive Sync

Download missing project files from Google Drive to local project directories. Used when the Linux autorun machine has completed pipeline runs and uploaded results to the shared "Pipeline Sync" Drive folder.

## When to Use

- User says "sync from drive", "pull from drive", "download from drive", "get files from linux"
- User asks what's missing locally, or why OTIO media is broken
- User wants to check what's available on Drive after a Linux pipeline run

## What It Syncs

Pulls files the editor needs on Windows that are often missing after Linux→Windows transfer:

| Path | Synced? | Why |
|------|---------|-----|
| `output/` | Yes | OTIO, EDL, XML timelines for NLE import |
| `.cache/v/` | Yes | yt-dlp video segments (OTIO V1-V6 media refs) |
| `.cache/i/` | Yes | Entity images + entity stock videos (OTIO V9/V11 media refs) |
| `stock/` | Yes | Pexels/Shutterstock stock videos (OTIO V10 media refs) |
| `.cache/embeddings/` | No | Regenerated locally |
| `.cache/audio/` | No | Temporary processing |
| `voiceover/` | No | Already downloaded during project prep |

## Two Modes

### `scan` — Diagnose missing media (no Drive needed)

Parses OTIO files in each local project, checks which referenced media files are missing, and reports a summary. Use this first to see what's wrong.

```bash
# Scan all Stu projects
python scripts/drive_sync.py scan --state-file clients/stu/pipeline_queue_state.json

# Scan specific card
python scripts/drive_sync.py scan --state-file clients/stu/pipeline_queue_state.json --card-id 4aeiR2F2

# Scan Degold projects
python scripts/drive_sync.py scan --state-file clients/degold/pipeline_queue_state.json
```

### `sync` — Download missing files from Drive

Lists the Drive folder, matches subfolders to queue projects, downloads files that are missing or size-mismatched locally.

```bash
DRIVE_FOLDER_ID="1j1wJV64WjIBFmzDBaRferbJd2k0H8tJq"

# Dry run — preview what would download
python scripts/drive_sync.py sync \
  --state-file clients/stu/pipeline_queue_state.json \
  --accounts-dir clients/stu/accounts \
  --drive-folder-id $DRIVE_FOLDER_ID \
  --dry-run

# Sync all Stu projects
python scripts/drive_sync.py sync \
  --state-file clients/stu/pipeline_queue_state.json \
  --accounts-dir clients/stu/accounts \
  --drive-folder-id $DRIVE_FOLDER_ID

# Sync specific card
python scripts/drive_sync.py sync \
  --state-file clients/stu/pipeline_queue_state.json \
  --accounts-dir clients/stu/accounts \
  --drive-folder-id $DRIVE_FOLDER_ID \
  --card-id KxQ0SoGh

# Sync Degold projects
python scripts/drive_sync.py sync \
  --state-file clients/degold/pipeline_queue_state.json \
  --accounts-dir clients/degold/accounts \
  --drive-folder-id $DRIVE_FOLDER_ID
```

## Drive Folder

**Parent folder:** `Pipeline Sync` — ID: `1j1wJV64WjIBFmzDBaRferbJd2k0H8tJq`

Structure on Drive (Linux uploads here):
```
Pipeline Sync/
  <card_id>-<project_name>/
    output/
      20260401_120000/
        timeline_FULL.otio
        ...
    .cache/
      v/<video_id>/segment_*.mp4
      i/<images + sv/*.mp4>
    stock/
      sv/<pexels_id>.mp4
```

## Default Behavior

When the user runs `/drive-sync` without arguments:

1. Run `scan` on all queues (Stu + Degold) to show what's missing locally
2. If files are missing and Drive has them, ask user to confirm sync
3. Run `sync --dry-run` first, then actual sync on confirmation
4. Report results with integrity check

## Key Info

| Field | Value |
|-------|-------|
| Drive Folder ID | `1j1wJV64WjIBFmzDBaRferbJd2k0H8tJq` |
| Auth | `clients/stu/accounts/david.env` (GWS token) |
| Script | `scripts/drive_sync.py` |
| GWS wrapper | `scripts/gws_drive.py` |

## Known Missing Media Pattern

The Linux pipeline stores entity images/videos in a **global** `.cache/i/` directory (not inside the project folder). OTIO refs point to paths like:
```
/home/hpmint/Desktop/matcher/.cache/i/PROJECT_SHORT/filename.jpeg
/home/hpmint/Desktop/matcher/.cache/i/PROJECT_SHORT/sv/pexels_id.mp4
```

These are never transferred via USB because they're outside the project directory. The Drive sync solves this — the Linux upload script should include `.cache/i/PROJECT_SHORT/` contents in the project's Drive subfolder under `.cache/i/`.

After downloading, run `--output-only` to regenerate OTIO with correct Windows paths.

## Troubleshooting

### "No project folders found on Drive"
Linux hasn't uploaded yet. Check the Drive folder:
```bash
python scripts/gws_drive.py --env-file clients/stu/accounts/david.env list-folder --folder-id 1j1wJV64WjIBFmzDBaRferbJd2k0H8tJq
```

### Auth errors
GWS token expired. Run `gws auth login` and update `clients/stu/accounts/david.env`.

### OTIO media still broken after sync
The OTIO files contain Linux paths. Run `--output-only` on the project to regenerate OTIO with Windows paths pointing to the newly downloaded files.
