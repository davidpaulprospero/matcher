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
---

# Video Verify Skill

Verify downloaded video files are valid, playable, and have expected durations.

## Invocation

```
/video-verify <project_path>
/video-verify E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc-How_USS_Charlotte_SANK_an_Iranian_Warship__2026-03-10
```

## Workflow

### Phase 1: Find Downloaded Videos

- Find video files: `Glob **/*.mp4` and `Glob **/*.mkv` in project
- Look in `downloads/` or media directories

### Phase 2: Check Video Validity

For each video, verify:
1. File exists and is readable
2. Has valid video stream
3. Has valid audio stream (if expected)
4. Duration is reasonable

```python
import subprocess
import os

def verify_video(video_path):
    """Verify a video file using ffprobe."""
    if not os.path.exists(video_path):
        return {'valid': False, 'error': 'File not found'}

    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,duration,bit_rate,width,height',
        '-of', 'default=noprint_wrappers=1',
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {'valid': False, 'error': result.stderr}

    # Parse output
    info = {}
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            info[key] = value

    return {
        'valid': True,
        'codec': info.get('codec_name'),
        'duration': info.get('duration'),
        'width': info.get('width'),
        'height': info.get('height'),
        'bit_rate': info.get('bit_rate'),
    }
```

### Phase 3: Compare to Expected Duration

- Read checkpoint to get expected segment durations
- Compare actual video duration to expected

```python
import json

# Load checkpoint to get segment info
with open('checkpoint.json') as f:
    checkpoint = json.load(f)

# Get segment info
segments = checkpoint.get('match_results', {})

# For each segment, check if video exists and duration matches
for seg_id, seg_data in segments.items():
    expected_duration = seg_data.get('duration')
    video_path = seg_data.get('video_path')

    if video_path and os.path.exists(video_path):
        actual = verify_video(video_path)
        if actual['valid']:
            diff = abs(float(actual['duration']) - expected_duration)
            if diff > 1.0:  # 1 second tolerance
                print(f"⚠ {seg_id}: expected {expected_duration}s, got {actual['duration']}s")
```

### Phase 4: Report Results

Output summary:
```
Video Verification Results
==========================
Total videos: 391
Valid: 387 ✅
Missing: 2 ❌
Corrupt: 2 ❌

Missing files:
  - segment_045.mp4
  - segment_128.mp4

Corrupt files:
  - segment_023.mp4 (no video stream)
  - segment_156.mp4 (duration mismatch: expected 10s, got 0s)
```

## Common Issues

### Missing video files

- Check download stage completed successfully
- Verify file paths in checkpoint
- Re-run download for missing segments

### Duration mismatch

- Check if video was fully downloaded
- Re-download if truncated
- Check for encoding issues

### No video stream

- File may be audio-only
- Check if download format was correct
- Re-download with video option

## Quick Check Commands

```bash
# Count videos
ls -1 *.mp4 | wc -l

# Check for zero-byte files
find . -size 0 -name "*.mp4"

# Check video info
ffprobe -v error -show_entries format=duration:stream=codec_name -of default=noprint_wrappers1 video.mp4
```
