---
name: verify-timing
description: Verify timeline timing matches SRT and report any drift
allowed-tools:
  - Glob
  - Read
  - Bash(python:*)
---

# Timing Verify Skill

Verify that timeline clip positions match the SRT voiceover timing exactly.

## Invocation

```
/timing-verify <project_path>
/timing-verify E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc-How_USS_Charlotte_SANK_an_Iranian_Warship__2026-03-10
```

## Workflow

### Phase 1: Find SRT and Timeline

- Find SRT file: `Glob **/*.srt` in project
- Find latest timeline: `Glob **/timeline_FULL.otio` in output directory

### Phase 2: Parse SRT

```python
def srt_time_to_frames(time_str, fps=30):
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return int((hours * 3600 + minutes * 60 + seconds) * fps)

# Parse all segments
segments = {}
with open(srt_path) as f:
    lines = f.readlines()
i = 0
while i < len(lines):
    if lines[i].strip().isdigit():
        idx = int(lines[i].strip())
        times = lines[i+1].strip().split(' --> ')
        segments[idx] = (srt_time_to_frames(times[0]), srt_time_to_frames(times[1]))
        i += 4
    else:
        i += 1
```

### Phase 3: Compare Timeline to SRT

```python
import opentimelineio as otio

timeline = otio.adapters.read_from_file(timeline_path)
v1 = timeline.tracks[0]

clips = []
pos = 0
for item in v1:
    if hasattr(item, 'source_range'):
        seg_idx = item.metadata.get('segment_index', -1)
        clips.append((pos, int(item.source_range.duration.value), seg_idx))
        pos += int(item.source_range.duration.value)
    else:
        pos += int(item.source_range.duration.value)

# Check intervals and report drift
for timeline_pos, dur, seg_idx in clips:
    if seg_idx >= 0 and seg_idx % 50 == 0:
        srt_seg_num = seg_idx + 1
        srt_start = segments[srt_seg_num][0]
        diff = timeline_pos - srt_start
        print(f"seg_idx={seg_idx}: timeline={timeline_pos}, SRT={srt_start}, diff={diff}")
```

### Phase 4: Report Results

Output a table:
```
| Position | Timeline | SRT | Diff | Status |
|----------|----------|-----|------|--------|
| seg 0    | 35       | 35  | +0    | ✅ PASS |
| seg 50   | 4301     | 4299| +2    | ✅ PASS |
| seg 100  | 8338     | 8336| +2    | ✅ PASS |
| seg 350  | 27662    | 27660| +2   | ✅ PASS |
```

### Pass Criteria

- ✅ PASS: 0-2 frame difference
- ⚠ WARNING: 3-5 frame difference
- ❌ FAIL: >5 frame difference

## Common Issues

### Timeline drift accumulates toward end

- Check position calculation in `src/otio/timeline.py`
- Ensure using direct SRT-based positioning, not cumulative

### First segment not at expected position

- Check leading gap calculation
- Verify SRT start time is being applied correctly
