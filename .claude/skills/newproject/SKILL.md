---
name: newproject
description: Create a new project, download voiceover from Google Drive links in a Google Doc, and start the pipeline.
allowed-tools:
  - Read
  - Bash(python:*)
  - Bash(mkdir:*)
---

# New Project Setup

Creates a new project folder, downloads voiceover files from Google Drive links in a Google Doc, combines them, and starts the pipeline.

## Usage

```
/newproject <project_name> <editing_for> <google_doc_url>
```

**Examples:**
```
/newproject 67 Stu https://docs.google.com/document/d/1abc123/edit
/newproject "Episode 5" Client https://docs.google.com/document/d/xyz789/edit?usp=sharing
```

## What It Does

1. **Creates project folder**: `E:\Edit Job\{editing_for}\{month}\{name}__{YYYY-MM-DD}`
   - Uses current month as subfolder (January, February, etc.)
   - Runs `setup_project.py` to create standard structure

2. **Downloads voiceover**:
   - Fetches Google Doc content from the URL
   - Extracts all Google Drive file links (MP3/audio files)
   - Downloads each file using gdown
   - Combines all audio into single `voiceover.mp3`
   - Places in project's `voiceover/` folder

3. **Starts pipeline**: Automatically runs the matcher pipeline

## Instructions

When this skill is invoked:

1. **Parse arguments**: Extract project_name, editing_for, google_doc_url from user input
   - Handle quoted names with spaces: `"Episode 5"`
   - Validate Google Doc URL format

2. **Run the newproject script**:
```bash
cd "D:/_Projects/voiceover-matcher" && python -m src.cli.newproject "<project_name>" "<editing_for>" "<google_doc_url>"
```

3. **Report results**:
   - Show created project path
   - Show number of audio files downloaded
   - Show pipeline start status

## Folder Structure Created

```
E:\Edit Job\{editing_for}\{month}\{name}__{YYYY-MM-DD}\
├── voiceover/
│   └── voiceover.mp3       # Combined audio from Google Drive
├── output/
├── logs/
├── .cache/
├── run.bat
├── convert.bat
├── analyze.bat
└── project_config.yaml
```

## Error Handling

- **Invalid Google Doc URL**: Must be `docs.google.com/document/d/...` format
- **No Drive links found**: Warns user, still creates project
- **Download failures**: Reports failed files, continues with successful ones
- **Empty audio**: Aborts if no audio files downloaded successfully

## Prerequisites

- `gdown` package installed: `pip install gdown`
- `pydub` package installed: `pip install pydub`
- FFmpeg installed at `C:\ffmpeg\bin\ffmpeg.exe`
