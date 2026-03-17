---
name: verify-video
description: Verify downloaded video files are valid, playable, and match expected duration
allowed-tools:
  - Glob
  - Read
  - Grep
  - Bash(python:*)
  - Bash(ffprobe:*)
  - Bash(ls:*)
  - Bash(find:*)
---

# Video Verify Skill

Verify that all video files referenced in OTIO timelines exist, are valid, and are playable.

## Invocation

```
/verify-video <project_path>
/verify-video E:\Edit Job\Degold\DeepSeaReports\ProjectName__2026-03-10
```

If no project path is provided, ask the user which project to verify.

## Workflow

### Step 1: Locate the project and latest output

Find the most recent output directory:
```bash
ls "<project_path>/output/"
```
Use the latest timestamped folder (e.g. `20260317_213951`).

### Step 2: Extract all video references from OTIO

Parse `target_url` fields from `timeline_FULL.otio` (it's JSON):

```python
import json, re, sys, os

otio_path = os.path.join(sys.argv[1], 'output', '<latest_dir>', 'timeline_FULL.otio')
with open(otio_path) as f:
    otio_str = f.read()

urls = set(re.findall(r'"target_url"\s*:\s*"([^"]+)"', otio_str))
print(f'Total unique video references: {len(urls)}')
```

### Step 3: Check file existence

For each `target_url`, check if the file exists and is non-zero:

```python
missing = []
zero_byte = []
valid = []

for url in sorted(urls):
    path = url
    if not os.path.exists(path):
        missing.append(url)
        continue
    size = os.path.getsize(path)
    if size == 0:
        zero_byte.append(url)
        continue
    valid.append((url, size))
```

Categorize missing files:
- Bare YouTube video IDs (11-char alphanumeric, no path separators beyond project root) are **entity/stock placeholders** — report them separately as non-critical
- The voiceover `.mp3` reference is expected — exclude from missing count
- Everything else is a genuinely missing video segment

### Step 4: ffprobe validation (sample)

Run ffprobe on a random sample of ~30 valid files to check for corruption:

```python
import subprocess, random

sample = random.sample(video_paths, min(30, len(video_paths)))
for path in sample:
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=codec_name,duration,width,height',
         '-of', 'default=noprint_wrappers=1', path],
        capture_output=True, text=True, timeout=10,
        encoding='utf-8', errors='replace'
    )
    # Check: returncode == 0, codec_name present, duration > 0.5s
```

### Step 5: Report results

Output a concise summary:

```
Video Verification Results
==========================
Total video references: 802
Valid (exist + non-zero): 793
Missing segments:           0
Entity/stock placeholders:  8  (non-critical)
Voiceover reference:        1  (expected)
Zero-byte:                  0
Corrupt (sample of 30):     0

All video segments verified OK.
```

If there are genuinely missing or corrupt segments, list them with filenames.

## Important Notes

- Videos are stored in `E:/v/matcher-alt/` as segment clips (format: `{videoId}_{start}_{end}.mp4`)
- The OTIO `target_url` is the authoritative source of truth for what files are needed
- Entity/stock tracks (V9-V11) may reference bare video IDs that were never downloaded — these are placeholders for NLE lookup, not errors
- Pass `encoding='utf-8', errors='replace'` to all subprocess calls (Windows requirement)
- Use `sys.argv[1]` to receive the project path (avoids special character issues in Python string literals)
