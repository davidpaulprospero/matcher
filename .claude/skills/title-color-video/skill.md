---
name: title-color-video
description: Generate a small FFmpeg video that alternates white/black backgrounds at each title timestamp found in a project SRT
allowed-tools:
  - Glob
  - Bash(python:*)
  - Read
  - Write
  - Edit
---

# Title Color Video Skill

Generate a small FFmpeg video that alternates white/black backgrounds at each title timestamp found in a project SRT.

## Invocation

```
/title-color-video <project_path> [--output <output.mp4>] [--phrases "phrase1|phrase2|..."] [--srt <path>] [--threshold <0-100>]
```

**Default output**: `title_colors.mp4` in the project root (not a temp directory).

Examples:
```
/title-color-video "E:\Edit Job\Kyteq\Disney\0k46X2wl-Disney_World_Buildings"
/title-color-video "E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc" --output title_colors.mp4
/title-color-video "E:\Edit Job\client\project" --phrases "First title|Second title|Third title"
```

## Workflow

### Phase 1: Find SRT

Search for SRT files in `project_dir/voiceover/` (primary) and `project_dir/` (fallback).

**Prefer trimmed variant**: if both `voiceover.srt` and `voiceover_trimmed.srt` exist, use the trimmed one. Trimmed SRTs are named like `voiceover_trimmed.srt` where the base is `voiceover.srt`.

If multiple SRTs exist without a clear winner, ask the user to specify `--voiceover`.

### Phase 2: Parse SRT

Parse the SRT file using `src.utils.parse_srt_file()` which returns `List[SRTSegment]`:
```python
from src.utils import parse_srt_file, SRTSegment
segments = parse_srt_file(srt_path)
# Each SRTSegment has: index, start_time, end_time, text
```

### Phase 3: Find Title Timestamps

Each title is introduced by a known transitional phrase. Default phrases:
```
Let's start with the most basic version
The building isn't just standing
That's the track
But it's not just vehicles sitting in buildings
The harder cases are the ones where something didn't just close and sit
The invisible park isn't just behind walls
Here is where it gets genuinely strange
Which raises a question that sounds simple
Here's where the story gets sharp
Which brings us to the final layer
Every data point in this video connects
Here is the practical piece
```

## Output Location

**Always export to project root**: `--output title_colors.mp4` (relative) or `<project_dir>/title_colors.mp4` (absolute). Never output to temp directories — the user expects the file in the project folder.

## Matching Strategy (Critical Order)

The 3-step matching must run in this exact order — Step 1 FIRST:

1. **Exact substring on single segment** (run FIRST — this is the primary match path)
2. **Exact substring across consecutive segments** (only if Step 1 finds nothing)
3. **Fuzzy fallback** only if neither exact strategy succeeds:
   - Use `rapidfuzz.partial_ratio` only (NOT `token_set_ratio` — gives false positives)
   - Skip segments where starting segment scores <60 individually
   - Skip segments where segment text is >1.5× longer than the phrase (filters continuation segments that coincidentally contain phrase substring)
4. **Deduplication**: if a phrase's best match lands within 5s of an already-accepted cut point, skip it

**Common bug**: If Step 1 code is missing or runs after Step 2, the algorithm will miss single-segment exact matches and incorrectly use fuzzy fallback, causing drift of 2-3 seconds on cuts that should be exact.

### Phase 4: Build Color Segments

Build a list of `(start_sec, end_sec, color)` tuples by alternating black/white:
- Segment 0: `0.0 → first_title_start` (black)
- Segment 1: `first_title_start → second_title_start` (white)
- Segment 2: `second_title_start → third_title_start` (black)
- ...continuing alternation

If the SRT has a total duration, the last segment runs to the end.

### Phase 5: Generate FFmpeg Command

Build a filter_complex using color sources (pattern from `scripts/edl_to_scene_video.py`):
```python
filter_parts = []
concat_inputs = []
for i, (start, end, color) in enumerate(segments):
    duration = end - start
    filter_parts.append(
        f"color=c={color}:s=640x360:r=24:d={duration:.3f}[v{i}]"
    )
    concat_inputs.append(f"[v{i}]")

filter_complex = "; ".join(filter_parts)
filter_complex += f"; {''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=0[out]"

cmd = [
    "ffmpeg", "-y",
    "-filter_complex", filter_complex,
    "-map", "[out]",
    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
    "-an",
    str(output_path)
]
```

### Phase 6: Run and Verify

Run the FFmpeg command. Verify output file exists and is playable.

## Size Optimization

- Resolution: 640x360
- Frame rate: 24fps
- Codec: libx264 with `ultrafast` preset and CRF 28
- No audio track

## Key Files

| File | Purpose |
|------|---------|
| `src/utils.py` | `parse_srt_file()`, `SRTSegment` dataclass |
| `scripts/edl_to_scene_video.py` | Reference for FFmpeg color filter_complex pattern |