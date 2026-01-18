# Scripts

Standalone scripts for common operations.

## Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `download_gdrive.py` | Download from public Google Drive links | No auth |
| `download_gdrive_auth.py` | Download with OAuth authentication | Google OAuth |
| `watch_pipeline.py` | Monitor/start pipeline for a project | `/watch` skill |

---

## Watch Pipeline

Monitors a running pipeline and auto-starts it if not running.

```bash
python scripts/watch_pipeline.py "<PROJECT_PATH>"
```

**Features:**
- Detects if pipeline is running (process check + log/checkpoint age)
- Auto-starts pipeline if not running (finds voiceover automatically)
- Shows periodic status updates based on current stage
- Exits when pipeline finishes or stalls

**Example:**
```bash
python scripts/watch_pipeline.py "E:\Edit Job\Stu\January\6__2026-01-15"
```

---

## Download Scripts

Standalone scripts for downloading media from Google Drive.

Both scripts are wrappers around the `src/gdrive/` module.

## Usage

### Anonymous Download (gdown)

```bash
python scripts/download_gdrive.py <url> [output_path]
```

**Examples:**
```bash
# Download by URL
python scripts/download_gdrive.py "https://drive.google.com/open?id=1ABC123xyz" voiceover.mp3

# Download by file ID
python scripts/download_gdrive.py "1ABC123xyz" output.mp4
```

**Best for:**
- Public files (shared with "Anyone with the link")
- Quick downloads without OAuth setup
- One-time downloads

**Fallback strategies:**
1. Standard gdown download
2. Fuzzy matching
3. Direct URL construction
4. Confirmation page handling (requests library)

### OAuth Download

```bash
python scripts/download_gdrive_auth.py <file_id_or_url_or_name> [output_path]
```

**Examples:**
```bash
# Download by file ID
python scripts/download_gdrive_auth.py "1ABC123xyz" voiceover.mp3

# Download by URL
python scripts/download_gdrive_auth.py "https://drive.google.com/file/d/1ABC123xyz" output.mp4

# Download by file name (searches your Drive)
python scripts/download_gdrive_auth.py "my_audio_file.mp3" voiceover.mp3
```

**Best for:**
- Private files you own or have access to
- Files shared only with you
- Searching by file name
- Recurring downloads

**First run:** Opens browser for Google authentication
**Later runs:** Uses cached credentials (stored in `~/.matcher_drive_auth/`)

## Setup for OAuth

1. First run will guide you through setup
2. Visit https://console.cloud.google.com/
3. Create OAuth 2.0 Desktop credentials
4. Download the credentials JSON file
5. Paste contents when prompted or save to `~/.matcher_drive_auth/credentials.json`

## Using in Python Code

```python
from src.gdrive import gdown_download, oauth_download, get_drive_service

# Anonymous download
success = gdown_download('https://drive.google.com/open?id=1ABC...', 'output.mp3')

# OAuth download
service = get_drive_service()
success = oauth_download(service, 'file_id', 'output.mp3')
```

## Using as a Skill

```bash
/gdrive-download "https://drive.google.com/open?id=1ABC..." voiceover.mp3
/gdrive-download "my_file.mp3" output.mp4 --oauth
```

## Troubleshooting

### gdown fails with "Access denied"
- Verify file is shared: "Anyone with the link can view"
- Try OAuth mode instead: `download_gdrive_auth.py`

### OAuth shows "credentials.json not found"
- Follow the interactive setup prompt
- Or manually save to `~/.matcher_drive_auth/credentials.json`

### Large file (>5GB) fails to download
- Try splitting into smaller files
- Use OAuth mode for better reliability
- Check internet connection stability

### File not found during search (OAuth mode)
- Verify file exists and you have access
- Check spelling of file name
- Use file ID directly instead of name

## Architecture

```
scripts/
├── download_gdrive.py          # CLI wrapper for anonymous download
├── download_gdrive_auth.py     # CLI wrapper for OAuth download
└── README.md                   # This file

src/gdrive/
├── __init__.py                 # Module exports
├── extract.py                  # URL/ID extraction utilities
├── fallback_downloader.py      # gdown + fallback strategies
├── oauth_downloader.py         # OAuth authentication + API
└── skill.py                    # Unified skill command
```

Both scripts import from `src/gdrive/` and provide CLI interfaces. The `skill.py` module provides a unified interface for both methods.
