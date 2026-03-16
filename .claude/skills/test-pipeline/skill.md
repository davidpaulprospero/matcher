---
name: test-pipeline
description: Run pipeline in test mode and verify OTIO timing, exports, and functionality
allowed-tools:
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - Bash(python:*)
  - Bash(python -m pytest:*)
  - Bash(python -m py_compile:*)
---

# Test Pipeline Skill

Run a pipeline in test mode and verify the outputs are correct.

## Invocation

```
/test-pipeline <project_path>
/test-pipeline E:\Edit Job\Degold\_test_queues\test_1
/test-pipeline E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc-How_USS_Charlotte_SANK_an_Iranian_Warship__2026-03-10
```

## Workflow

### Phase 1: Find Voiceover

- Find the SRT file in the project directory:
  ```
  Glob pattern: **/*.srt in project directory
  ```
- Determine the correct voiceover path

### Phase 2: Run Pipeline in Test Mode

- Run with output-only to regenerate OTIO:
  ```
  python main.py --voiceover <srt_path> --project <project_path> --output-only --non-interactive
  ```
- If checkpoint is incomplete, add `--fresh` flag

### Phase 3: Verify Timing

- Load the generated timeline and compare to SRT:
  ```python
  import opentimelineio as otio

  def srt_time_to_frames(time_str, fps=30):
      time_str = time_str.replace(',', '.')
      parts = time_str.split(':')
      hours = int(parts[0])
      minutes = int(parts[1])
      seconds = float(parts[2])
      return int((hours * 3600 + minutes * 60 + seconds) * fps)

  # Parse SRT
  segments = {}
  # ... parse SRT file ...

  # Load timeline
  timeline = otio.adapters.read_from_file(path)
  v1 = timeline.tracks[0]

  # Check clip positions vs SRT
  for item in v1:
      if hasattr(item, 'source_range'):
          seg_idx = item.metadata.get('segment_index', -1)
          # Compare timeline position to SRT start time
  ```

- Expected: 0-2 frame difference max throughout timeline

### Phase 4: Check Outputs

Verify these files exist in the output directory:
- `timeline_FULL.otio` - Full timeline
- `timeline_sequence.xml` - DaVinci Resolve XML
- `timeline.edl` - EDL file
- `timeline_A8_voiceover.otio` - Voiceover track

### Phase 5: Report Results

Output a summary table:
```
| Position | Timeline | SRT | Diff |
|----------|----------|-----|------|
| seg 0    | 35       | 35  | +0   |
| seg 100  | 8338     | 8336| +2   |
| ...      | ...      | ... | ...  |
```

## Common Issues

### Checkpoint incomplete

If you get "Checkpoint incomplete for output-only mode":
1. Restore from backup: Copy `checkpoint.backup.json` to `checkpoint.json`
2. Or use `--fresh` to ignore checkpoint

### Timing Drift

If timing drift exceeds 2 frames:
- Check the position calculation in `src/otio/timeline.py`
- Ensure using direct SRT-based positioning, not cumulative

### Missing entity tracks

If entity tracks (V9-V11) are empty:
- Check checkpoint has entity data
- Verify entity stage completed
