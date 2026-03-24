---
name: stu-download-attachments
description: Download Trello card attachments (images, videos, clips) for a STU/NEW AMERICA project directory
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Stu Download Attachments

Download all Trello card attachments for a NEW AMERICA project.

## When to Use

Use this skill when:
- User asks to download attachments from a Trello card for a Stu project
- User wants to pull images, videos, or clips from a NEW AMERICA card into a project folder
- User provides a project path and wants the card's attachments

## What It Does

1. Resolves the Trello card from the project directory (folder name card ID, `trello_card.json`, or title matching)
2. Fetches all attachments from the Trello card via API
3. Downloads Trello-hosted files to `{project}/attachments/`
4. Logs external URLs (Google Drive links, etc.) without downloading them
5. Skips already-downloaded files (same name + size) unless `--force`

## Key Files

- `scripts/stu/download_attachments.py` - Download script
- `Stu/pipeline_queue_state.json` - Queue state (used for card ID resolution)
- `Stu/accounts/david.env` - Trello credentials

## Commands

```bash
# Download attachments for a project (auto-resolve card)
python scripts/stu/download_attachments.py "E:/Edit Job/Stu/PROJECT_DIR"

# Dry run: preview attachments without downloading
python scripts/stu/download_attachments.py "E:/Edit Job/Stu/PROJECT_DIR" --dry-run

# Explicit card ID (for projects that don't embed card ID in folder name)
python scripts/stu/download_attachments.py "E:/Edit Job/Stu/PROJECT_DIR" --card-id KxQ0SoGh

# Force re-download (overwrite existing files)
python scripts/stu/download_attachments.py "E:/Edit Job/Stu/PROJECT_DIR" --force
```

## Default Behavior

When the user runs `/stu-download-attachments PROJECT_PATH`:

1. Run dry-run first to show what will be downloaded:
   ```bash
   python scripts/stu/download_attachments.py "PROJECT_PATH" --dry-run
   ```
2. If card ID cannot be resolved, ask the user for the card shortLink and use `--card-id`
3. On confirmation, run the actual download:
   ```bash
   python scripts/stu/download_attachments.py "PROJECT_PATH"
   ```
4. Report the summary (downloaded, skipped, external, failed)

If the user provides a card ID directly (e.g. `/stu-download-attachments E:\Edit Job\Stu\SAMPLE --card-id KxQ0SoGh`), skip the dry-run confirmation and download immediately.

## Card ID Resolution

The script auto-detects the card from the project directory:

| Method | Example |
|--------|---------|
| `trello_card.json` | File in project root with `card_id` or `shortLink` field |
| Folder name prefix | `KxQ0SoGh-THE $36 TRILLION BOMB__2026-03-19` |
| Title matching | Jaccard token similarity vs state file card titles |

If none match, the script shows all available cards and requires `--card-id`.

## Key Info

| Field | Value |
|-------|-------|
| Trello Board | NEW AMERICA (`6998b92db8fc2834b42b38dc`) |
| Account | David |
| Channel Code | STU |
| Local Projects Root | `E:\Edit Job\Stu` |
| Download Target | `{project}/attachments/` |
