# Degold Lipsync Tools

Automate submissions to the AI Lipsync Generator form at degoldmedia.duckdns.org.

## Quick Start

```bash
cd Degold
python lipsync_automator.py -t "EP42 - Test" -c RRU -a avatar.png audio1.mp3 audio2.mp3
```

## Setup

1. Install dependencies:
   ```bash
   pip install requests
   ```

2. Configure channels in `channels.py` (see CHANNELS.md for reference)

## Usage

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
| `lipsync_automator.py` | Main submission script |
| `channels.py` | Channel configuration module |
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
