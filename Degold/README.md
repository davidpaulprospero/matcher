# Degold Lipsync Tools

Automate submissions to the AI Lipsync Generator form at degoldmedia.duckdns.org.

## Quick Start

```bash
cd Degold

# Submit directly to lipsync
python lipsync_automator.py -t "EP42 - Test" -c RRU -a avatar.png audio1.mp3 audio2.mp3

# Fetch cards from Trello and submit
python trello_to_lipsync.py -l "Ready To Upload" -c RRU -a avatar.png audio.mp3
```

## Setup

1. Install dependencies:
   ```bash
   pip install requests python-dotenv
   ```

2. Configure Trello credentials in `accounts/` directory:
   ```bash
   cp accounts/example.env accounts/david.env
   # Edit david.env with your API key and token
   ```

3. Configure channels in `channels.py` (see CHANNELS.md for reference)

## Trello Integration

### Account Setup

1. Copy `accounts/example.env` to `accounts/<name>.env`
2. Fill in your Trello API key, token, and default board ID

### Trello to Lipsync Workflow

```bash
# List available accounts
python trello_to_lipsync.py --list-accounts

# List boards for an account
python trello_to_lipsync.py -A david --list-boards

# List lists on a board
python trello_to_lipsync.py -A david --list-lists

# Show cards in "Ready To Upload" (dry run)
python trello_to_lipsync.py -l "Ready To Upload" --dry-run

# Skip already processed cards
python trello_to_lipsync.py -l "Ready To Upload" --skip-processed --dry-run

# Submit cards to lipsync
python trello_to_lipsync.py -l "Ready To Upload" -c RRU -a avatar.png audio.mp3
```

### History Tracking

Track which cards have been submitted:

```bash
# Show submission history
python trello_to_lipsync.py --history

# Clear history
python trello_to_lipsync.py --clear-history

# Skip already processed cards
python trello_to_lipsync.py -l "Ready To Upload" --skip-processed
```

### Filter Options

| Option | Description |
|--------|-------------|
| `--list/-l` | Filter by list name |
| `--assigned-to-me/-m` | Only cards assigned to you |
| `--has-due` | Only cards with due dates |
| `--due-within N` | Cards due within N days |
| `--labels NAME` | Filter by label |
| `--limit N` | Max cards to process |
| `--skip-processed` | Skip cards already in history |

## Direct Lipsync Submission

### Command Line

```bash
# Basic usage (drive folder auto-filled from channels.py)
python lipsync_automator.py -t "Video Title" -c RRU -a avatar.png audio1.mp3 audio2.mp3

# With manual drive folder override
python lipsync_automator.py -t "Video Title" -c RRU -a avatar.png -d "folder_id" audio.mp3

# Show help
python lipsync_automator.py --help
```

### Python API

```python
from channels import get_channel
from lipsync_automator import submit_lipsync_job

# Get channel config
config = get_channel("RRU")

# Submit job
status, response = submit_lipsync_job(
    video_title="EP42 - Test",
    channel_code="RRU",
    avatar_path="avatar.png",
    audio_files=["01_hook.mp3", "02_segment.mp3"],
    drive_folder_id=config.drive_folder,
)

print(f"Status: {status}")
```

## Channel Configuration

Edit `channels.py` to add new channels:

```python
CHANNELS = {
    "RRU": ChannelConfig(
        code="RRU",
        drive_folder="1XJY8HUEWvFH0cI68tvyrPEyU7bTpeNLQ",
        avatar_folder="16cHw8fgefC89zelexv_OSQNoqzhKekwO",
        name="RennReports",
    ),
}
```

See `CHANNELS.md` for full channel documentation.

## Files

| File | Description |
|------|-------------|
| `lipsync_automator.py` | Direct submission script |
| `trello_to_lipsync.py` | Trello integration script |
| `history_tracker.py` | History tracking module |
| `channels.py` | Channel configuration module |
| `accounts/` | Trello account credentials |
| `CHANNELS.md` | Channel documentation |
| `README.md` | This file |

## Form Fields

| Field | Name | Type |
|-------|------|------|
| Video Title | `field-0` | text |
| Channel Code | `field-1` | select |
| Avatar Image | `field-2` | file |
| Audio Segments | `field-3` | file (multiple) |
| Drive Folder ID | `field-4` | text |
