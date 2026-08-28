---
name: gap-fill
description: Find and fill gaps in an edited OTIO timeline with new targeted footage. Analyzes a DaVinci help.otio for missing clips, runs a focused pipeline to find better candidates, builds a multi-track OTIO aligned to the original, and optionally downloads full videos for manual use.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - Agent
  - CronCreate
  - CronDelete
  - Monitor
---

# Gap Fill Skill

Analyzes an edited OTIO timeline (exported from DaVinci Resolve) to find gaps where clips were removed, then runs a targeted pipeline to find better replacement footage and builds a multi-track OTIO that slots directly into the original timeline.

## When to Use

- User has an OTIO exported from DaVinci with gaps (removed clips)
- User says "fill the gaps", "find better clips", "gap fill", "missing clips"
- User wants to re-run matching for specific weak sections of a timeline
- User asks to download full videos related to specific gap themes

## Invocation

```
/gap-fill <project_path>
/gap-fill <project_path> --otio help.otio
/gap-fill <project_path> --download-full
/gap-fill <project_path> --otio help.otio --download-full
```

## Arguments

- `<project_path>` (required): Path to the project directory (e.g., `E:\Edit Job\Kyteq\Disney\0k46X2wl-Disney_World_Buildings`)
- `--otio <filename>`: Name of the gap OTIO file in the project directory (default: `help.otio`)
- `--download-full`: After pipeline completes, present a list of top-matched videos and let the user select which to download in full
- `--skip-pipeline`: Skip the pipeline run, only rebuild the OTIO from existing gap_fill checkpoint
- `--rebuild-otio`: Only rebuild the multi-track OTIO from existing gap_fill results

## Workflow

### Phase 1: Analyze Gaps

> **⚠️ CRITICAL: OTIO durations are in FRAMES, not seconds.**
> `child.duration().value` returns **frames** — you MUST divide by the timeline's frame rate (typically 30fps) to get seconds.
> A value like `35963` frames = `35963/30 ≈ 1198.8 seconds` (~20 min), NOT 35963 seconds.
> Always verify: `timeline.duration().value / rate` should equal the SRT duration if the timeline matches the voiceover.

1. Read the OTIO file (`<project_path>/<otio_file>`) — it's a JSON OTIO exported from DaVinci Resolve
2. Walk the first video track, accumulate timeline position for each child
3. **Convert frame counts to seconds**: `position_frames / rate` before comparing to SRT timestamps
4. Record all regions to fill — either:
   - Items where `OTIO_SCHEMA` starts with `Gap` and duration >= 0.5s, OR
   - **Disabled clips** (Clips with `metadata.enabled == False`) of any duration — DaVinci marks removed-but-preserved clips this way, and a disabled clip is functionally a gap in the rendered output.
   - **Adjacent tiny gaps** (< 0.5s) immediately before/after a fill region are absorbed into that region so we don't leave rim gaps around replacements.
   - Adjacent fill regions (back-to-back, no enabled content between) are merged into one region.
5. Parse the project's voiceover SRT from `<project_path>/voiceover/`
6. Map each fill region's timeline position (in seconds) to overlapping SRT segments
7. Report fill summary: count, total duration, thematic grouping (note how many regions came from disabled clips vs literal gaps)
8. Write `gap_analysis.txt` to project root for debugging reference

**Detailed per-gap report (print before running pipeline):**
```python
import opentimelineio as otio

def ts_to_seconds(ts):
    ts = ts.replace(',', '.')
    h, m, s = ts.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

RATE = timeline.duration().rate
video_track = next(t for t in timeline.tracks if t.kind == 'Video')

# Build item list with classification + duration so the region detection
# can look both forward and backward (to absorb tiny adjacent gaps).
def build_item_list(track):
    items = []
    for child in track:
        cls = child.__class__.__name__
        dur_s = child.duration().value / RATE
        enabled = True
        if cls == 'Clip':
            enabled = child.metadata.get('enabled', True) is not False
        items.append((cls, dur_s, enabled))
    return items

def find_fill_regions(track):
    """Find fill regions: literal gaps >= 0.5s OR any disabled clip.
    Adjacent tiny gaps (< 0.5s) immediately before/after a fill region are
    absorbed to avoid rim gaps. Overlapping/adjacent fill regions merge.
    """
    items = build_item_list(track)
    raw_indices = {i for i, (cls, dur_s, enabled) in enumerate(items)
                   if (cls == 'Gap' and dur_s >= 0.5) or (cls == 'Clip' and not enabled)}
    if not raw_indices:
        return []

    regions = []  # list of (start_idx, end_idx) inclusive
    sorted_idx = sorted(raw_indices)
    for idx in sorted_idx:
        if regions and idx <= regions[-1][1]:
            continue  # already inside previous region
        start, end = idx, idx
        # extend backwards over tiny gaps
        while start > 0 and items[start - 1] == ('Gap', items[start - 1][1], True) \
              and items[start - 1][1] < 0.5:
            start -= 1
        # extend forwards over tiny gaps
        while end + 1 < len(items) and items[end + 1][0] == 'Gap' and items[end + 1][1] < 0.5:
            end += 1
        # merge directly-adjacent (idx == last_end + 1) into previous region
        if regions and start == regions[-1][1] + 1:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))

    # Map item indices back to timeline positions
    pos_at = [0.0]
    for _cls, dur_s, _enabled in items:
        pos_at.append(pos_at[-1] + dur_s)

    fill_ranges = []
    for s, e in regions:
        gs, ge = pos_at[s], pos_at[e + 1]
        src = 'disabled' if any(items[j][0] == 'Clip' and not items[j][2]
                                for j in range(s, e + 1)) else 'gap'
        fill_ranges.append((gs, ge, src))
    return fill_ranges

fill_ranges = find_fill_regions(video_track)

# For diagnostics
gap_count = sum(1 for r in fill_ranges if r[2] == 'gap')
disabled_count = sum(1 for r in fill_ranges if r[2] == 'disabled')
print(f"Fill regions: {len(fill_ranges)} total "
      f"({gap_count} literal gaps, {disabled_count} disabled-clip regions)")

# For each fill region, show overlapping SRT text
for i, (gs, ge, source) in enumerate(fill_ranges):
    overlapping = [s for s in srt_segments
                   if gs <= (ts_to_seconds(s['start']) + ts_to_seconds(s['end'])) / 2 <= ge]
    text = ' | '.join(s['text'][:80] for s in overlapping[:3])
    src_label = '[gap]' if source == 'gap' else '[disabled clip]'
    print(f"FILL {i+1} {src_label}: {gs:.1f}s - {ge:.1f}s ({ge-gs:.1f}s)")
    print(f"  Text: {text}")
    print()
```

**Correct analysis code pattern:**
```python
import opentimelineio as otio

timeline = otio.adapters.read_from_file(otio_path)
rate = 30.0  # frames per second (from timeline or track)
position = 0.0  # accumulated position in frames

for child in video_track:
    dur_frames = child.duration().value
    dur_seconds = dur_frames / rate
    start_seconds = position / rate
    end_seconds = (position + dur_frames) / rate

    if child.__class__.__name__ == 'Gap' and dur_seconds >= 0.5:
        print(f"GAP at {start_seconds:.2f}s - {end_seconds:.2f}s ({dur_seconds:.2f}s)")

    position += dur_frames
```

**Key data to extract:**
- `gap_regions`: list of `(start_sec, end_sec)` for significant gaps
- `srt_segments`: SRT entries overlapping those gaps (with original timestamps preserved)
- `help_otio_items`: the full item list from the help OTIO with positions (needed for Phase 5)

### Phase 2: Create Gap-Only SRT

1. Create `<project_path>/gap_fill/voiceover/` directory
2. Write a gap-only SRT containing only SRT entries that overlap gap regions
3. **Preserve original timestamps** — do NOT renumber to start from 00:00:00
4. Copy the voiceover audio file (mp3/wav) to the gap_fill voiceover directory

### Phase 3: Run Targeted Pipeline

1. Create `<project_path>/gap_fill/project_config.yaml` to enforce 360p downloads:
   ```yaml
   download:
     segment_max_resolution: 360
   ```
   This is auto-loaded by `load_project_config()` from `src/cli/config_utils.py` and overrides `config.yaml` settings.
2. Create `<project_path>/gap_fill/saved_keywords.json` with a keyword preset (optional — the pipeline's LLM extraction from the gap-only SRT is often better than manual keywords)
3. Run the pipeline:
   ```bash
   cd D:/_Projects/voiceover-matcher-dev && python main.py \
     --voiceover "<project_path>/gap_fill/voiceover/gaps.srt" \
     --project "<project_path>/gap_fill" \
     --fresh --non-interactive
   ```
3. The pipeline will: ANALYZE → VIDEO_SEARCH → CAPTION → MATCH → DOWNLOAD_SEGMENTS → OUTPUT
4. **This is long-running** (30-90+ minutes depending on gap count). Run in background.
5. The pipeline checkpoints after each stage, so it can be resumed with `--resume` if interrupted.

**Pipeline monitoring:**
- Set up a cron job (every 5 min, not 30 min — pipeline can die mid-stage) to check health and auto-resume if the process dies
- Check progress via: `gzip.open(checkpoint.json)` → `last_completed_stage` and `download_segments.segments_completed`
- Kill duplicate processes if multiple get spawned (check with `wmic process`)
- **Note**: PyManager wrapper (`PyManager\python.exe`) exits with code 127 when the background task completes — the actual pipeline process continues independently. Do NOT restart based on PyManager exit; check for the raw `python.exe` process with the project path instead.

**Handling timeouts:**
- The bash timeout may kill the process during long stages (CAPTION, DOWNLOAD_SEGMENTS)
- Always use `run_in_background: true` with `timeout: 600000`
- The cron job will auto-resume from checkpoint

### Phase 4: Wait for Completion

Monitor until `last_completed_stage` == `OUTPUT`:
- CAPTION stage: ~30s per video × number of videos, 4 workers
- DOWNLOAD_SEGMENTS: ~10-15s per segment
- Check for `output/<timestamp>/` directory with `quality_report.json`

### Phase 5: Build Multi-Track OTIO

This is the critical alignment step. The output OTIO must **exactly mirror** the help.otio structure.

**Algorithm:**
1. Walk help.otio items sequentially, tracking timeline position
2. For each item:
   - If it's a **clip** (or gap < 0.5s): emit a **gap** of the same duration in our track
   - If it's a **significant gap** (>= 0.5s): fill with matched clips from gap_fill segments
3. For gap filling:
   - Find gap_fill segments overlapping this gap's time range
   - For each segment, look up the match for the current track (V1-V6)
   - Find the actual file in `.cache/v/matcher-alt/`
   - Create a clip with proper media reference
   - **Important**: Track per-track output position (`track_positions[track_key]`) during construction. When `clip_dur_f < dur_f` (segment doesn't fully cover gap), emit a gap for the uncovered portion — do NOT let gaps accumulate silently.

**Post-construction drift absorption:**
After the main loop, compare each track's total frames to `help_total_frames`. Absorb drift into the last emitted item:
```python
help_total = sum(c.duration().value for c in video_track)  # from help.otio
for track_key, vtrack, atrack in tracks:
    our_total = track_positions[track_key]
    drift = our_total - help_total
    if abs(drift) > 3:
        # Adjust last item's duration to absorb drift
        last_kind, last_item, item_dur = last_items_per_track[track_key]
        new_dur = max(1, int(item_dur - drift))
        sr = last_item.source_range
        last_item.source_range = otio_time.TimeRange(
            start_time=sr.start_time,
            duration=otio_time.RationalTime(value=float(new_dur), rate=RATE)
        )
```
**Target: 0 frame drift** for all 6 video tracks. Report actual drift after building.

**Build 6 tracks:** V1 (Primary), V2-V3 (Alternatives), V4-V6 (Secondary)

**Track name mapping (must match original pipeline output):**
```python
track_name_map = {
    'V1': 'V1 - Primary',
    'V2': 'V2 - Alternative 1',
    'V3': 'V3 - Alternative 2',
    'V4': 'V4 - Secondary Primary',
    'V5': 'V5 - Secondary 2',
    'V6': 'V6 - Secondary 3',
}
```

**DaVinci OTIO requirements (all are mandatory for import):**
- Timeline must have `global_start_time` (108000.0 frames = 1hr timecode)
- Timeline metadata must include `"Resolve_OTIO": {"Resolve OTIO Meta Version": "1.0"}`
- All items (tracks, clips, gaps) must have `"enabled": true, "color": null`
- Tracks must have `"Resolve_OTIO": {"Locked": false}` in metadata
- Clips must use `media_references` with `DEFAULT_MEDIA` key (NOT `media_reference`)
- Clips must have `"active_media_reference_key": "DEFAULT_MEDIA"`
- Stack must have `"enabled": true, "color": null`

**Media reference format for DaVinci:**
```json
{
  "media_references": {
    "DEFAULT_MEDIA": {
      "OTIO_SCHEMA": "ExternalReference.1",
      "metadata": {},
      "name": "filename.mp4",
      "target_url": "E:/Edit Job/.../filename.mp4",
      "available_range": {
        "OTIO_SCHEMA": "TimeRange.1",
        "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": "<file_duration_frames>"},
        "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 0.0}
      },
      "available_image_bounds": null
    }
  },
  "active_media_reference_key": "DEFAULT_MEDIA"
}
```

**File lookup for segments:**
- `v1_clip.file` is already a full filename (e.g., `j-UqbfH9M54_75_92.mp4`)
- `alternatives[].file` and `secondary[].file` are bare video IDs (e.g., `3p9fcgkohoo`)
- Build a file index from `.cache/v/matcher-alt/*.mp4` mapping `video_id → [(arc_start, arc_end, path, filename)]`
- Match by: `arc_start <= source_start <= arc_end`

**Duration verification:**
- Total duration of each output track MUST equal help.otio's total duration exactly
- Report drift — should be 0.000s

**Output:** `<project_path>/gap_fill_fixed.otio`

### Phase 6: Full Video Downloads (if --download-full)

After the pipeline completes:

1. Read the checkpoint's `video_search.search_results` to get all found videos with titles
2. Group by theme (Contemporary/Monorail, Club 33, Castle Suite, etc.) based on title keywords
3. Present the grouped list to the user with titles and durations
4. Let the user select which videos to download in full
5. Download selected videos:
   ```bash
   cd "<project_path>/gap_fill/.cache/v/matcher-alt" && \
   yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
     --merge-output-format mp4 -o "%(id)s_full.%(ext)s" \
     "https://www.youtube.com/watch?v=VIDEO_ID_1" \
     "https://www.youtube.com/watch?v=VIDEO_ID_2" ...
   ```
6. Report downloaded file sizes

## Key File Locations

| File | Location |
|------|----------|
| Input OTIO | `<project_path>/help.otio` (or specified with --otio) |
| Gap SRT | `<project_path>/gap_fill/voiceover/gaps.srt` |
| Pipeline checkpoint | `<project_path>/gap_fill/checkpoint.json` (gzipped) |
| Pipeline output | `<project_path>/gap_fill/output/<timestamp>/` |
| Segments JSON | `<project_path>/gap_fill/output/<timestamp>/timeline_segments.json` |
| Downloaded segments | `<project_path>/gap_fill/.cache/v/matcher-alt/` |
| Stock footage | `<project_path>/gap_fill/stock/sv/` |
| **Output OTIO** | `<project_path>/gap_fill_fixed.otio` |

## Reading the Checkpoint

The checkpoint is gzip-compressed JSON:
```python
import gzip, json
with gzip.open('checkpoint.json', 'rt', encoding='utf-8') as f:
    cp = json.load(f)
# Key fields:
cp['last_completed_stage']  # ANALYZE, VIDEO_SEARCH, CAPTION, MATCH, DOWNLOAD_SEGMENTS, OUTPUT
cp['download_segments']['segments_completed']  # download progress
cp['download_segments']['segments_total']
cp['video_search']['search_results']  # list of {video_id, title, duration, ...}
```

## Segment Match Lookup

```python
# V1: full filename
seg['v1_clip']['file']         # "j-UqbfH9M54_75_92.mp4"
seg['v1_clip']['confidence']   # 0.85
seg['v1_clip']['source_start'] # 338.99

# V2-V3 alternatives: bare video ID
seg['alternatives'][0]['track']        # "V2"
seg['alternatives'][0]['file']         # "hT97ouli1hA"
seg['alternatives'][0]['source_start'] # 150.48

# V4-V6 secondary: bare video ID
seg['secondary'][0]['track']        # "V4"
seg['secondary'][0]['file']         # "kCTRSeqVv6A"
seg['secondary'][0]['source_start'] # 251.2
```

## Troubleshooting

### Gaps appear at wrong timestamps or timeline seems impossibly long
- **Root cause**: OTIO frame values were interpreted as seconds directly.
- OTIO stores durations and positions as **frame counts**, not seconds.
- A timeline showing `35963 frames` is only `~20 minutes` at 30fps, not `10 hours`.
- Always divide by frame rate before comparing to SRT timestamps.

### "File not found in search directories" in DaVinci
- Media references must use forward-slash paths (e.g., `E:/Edit Job/...`), no `file:///` prefix
- Must use `media_references` with `DEFAULT_MEDIA` key (not `media_reference`)
- Must include `name` field on the ExternalReference
- Must include `available_range` with file duration and `available_image_bounds: null`

### Pipeline keeps timing out
- Expected: CAPTION and DOWNLOAD_SEGMENTS are long-running
- The cron job auto-resumes from checkpoint
- Check for duplicate processes with `wmic process where "name='python.exe'" get commandline | grep gap_fill`

### Duration mismatch between output and help.otio
- **Root cause**: Segments clip to gap region boundaries, creating `clip_dur_f < dur_f` per gap. Over 47 gaps, this accumulates to large drift (e.g., -284 frames).
- **Fix**: Use per-track position tracking (`track_positions[track_key]`) during construction. After the full loop, compare to `help_total_frames` and absorb drift into the last emitted item on each track:
  ```python
  for track_key, vtrack, atrack in tracks:
      drift = track_positions[track_key] - help_total_frames
      if abs(drift) > 3:
          adjust_last_item(last_items_per_track[track_key], -drift)
  ```
- Target: **0 frame drift** for all 6 video tracks. Report actual drift after building.

### 0 clips placed
- Check file locations: segments are in `.cache/v/matcher-alt/`, NOT `segments_archives/`
- V2-V6 use bare video IDs — need file index mapping, not direct filename lookup

### Segfaults during DOWNLOAD_SEGMENTS
- Memory pressure from concurrent downloads. Pipeline checkpoints frequently — just resume.

### Clips appear but source_range is wrong (Mismatch errors in DaVinci)
- **Root cause**: `source_range.start` was incorrectly calculated as `source_start * RATE` (raw video timestamp in frames).
- **Correct calculation**: `source_range.start = (source_start - arc_start) * RATE` (offset into arc, in frames).
  - Example: `source_start=129.36s`, `arc=108-139s`, so `offset = 129.36 - 108 = 21.36s` = `21.36 * 29.97 = 640.2 frames`.
- When `source_start` falls OUTSIDE arc bounds:
  - **V1**: Search other arcs from the same video to find one containing `source_start` (e.g., `n47ErRVSqSw_18_33.mp4` doesn't contain `source_start=53.04s`, but `n47ErRVSqSw_39_61.mp4` does).
  - **V2-V6**: Search arcs for the bare video ID to find one containing `source_start`.
  - If NO arc contains `source_start`, emit a GAP instead of a clip.
- **Clamp offset to arc bounds**: `offset_into_arc = max(0, min(source_start - arc_start, arc_dur))`.
- **Clamp duration**: Ensure `source_range.duration` doesn't exceed `available_range.duration`.

### source_range.duration doesn't match the gap region duration
- **Root cause**: Segment's total duration (end_frame - start_frame) may exceed the gap region boundary.
- When a segment starts before the gap region but ends within it (or vice versa), you must clip to the gap region.
- Example: segment at 454-643 (189 frames) but gap region is 641-870. The overlap is only 641-643 (2 frames).
- Use `clip_start = max(seg_start, gap_start)` and `clip_end = min(seg_end, gap_end)`.

### File index keys don't match video IDs from segments
- **Root cause**: A single video ID can have multiple arc segments (e.g., `MKhhTlDUwlg` has 4 entries: 8-24, 108-139, 139-153, 179-193).
- Build file index as `video_id → [(arc_start, arc_end, path, filename), ...]` (list, not single entry).
- When finding a file, match by `arc_start <= source_start <= arc_end` to find the correct arc segment.
- For V1 clips, also try exact filename match before falling back to arc matching.

### OTIO schema class names to use (avoid AttributeError)
- Use `otio.opentime.RationalTime` and `otio.opentime.TimeRange`, NOT `otio.schema.RationalTime`
- Use `otio.schema.Timeline`, `otio.schema.Stack`, `otio.schema.Track`, `otio.schema.Clip`, `otio.schema.Gap`, `otio.schema.ExternalReference`
- `otio.schema.VideoTrack` does NOT exist — use `otio.schema.Track`

### ExternalReference constructor signature
```python
# WRONG (raises TypeError):
media_ref = otio.schema.ExternalReference(target_url=path, name=fname)

# CORRECT:
media_ref = otio.schema.ExternalReference(
    target_url=path,
    available_range=TimeRange(
        duration=RationalTime(value=file_dur_frames, rate=RATE),
        start_time=RationalTime(value=0.0, rate=RATE)
    )
)
media_ref.name = fname  # Set name as attribute after construction
```

### Pipeline output structure vs our gap_fill_fixed structure
- Pipeline outputs like `timeline_FULL.otio` have `timeline.tracks` as a Stack with Track children directly
- Our original `gap_fill_fixed.otio` had `timeline.tracks` as a Stack containing a nested Stack("Video") containing Track children — WRONG
- DaVinci only reads the first level of tracks, so nested structures don't import properly
- Fix: append Track objects directly to `timeline.tracks`, don't wrap in intermediate Stack containers

### Media reference available_range.duration should match actual file duration from arc bounds
- When building `ExternalReference`, `available_range` should reflect the actual downloaded file's duration
- Use `arc_end - arc_start` from the cache filename (e.g., `MKhhTlDUwlg_108_139.mp4` → 139-108=31s → 930 frames at 30fps)
- Don't use the gap-fill timeline clip duration (which is the clip's position in the output timeline, not the file length)
- Pipeline outputs use actual file duration (928 frames for MKhhTlDUwlg_108_139.mp4 which is 30.93s)

### Track metadata must use item assignment, not direct assignment
- `vtrack.metadata = {'Resolve_OTIO': {'Locked': False}}` raises `AttributeError: can't set attribute`
- OTIO's `AnyDictionary` does not support direct assignment — use item assignment:
  ```python
  vtrack.metadata['Resolve_OTIO'] = {'Locked': False}  # CORRECT
  ```
- This applies to all track metadata mutations.

### Duration drift: why it happens and how to fix it
- **Symptom**: gap_fill_fixed.otio V1 total frames ≠ help.otio total frames (e.g., -284 frames drift)
- **Root cause**: When filling a significant gap, segments may not fully cover the gap region (seg starts after gap start, or ends before gap end). The algorithm emits clips of `clip_dur_f` but help.otio expects `dur_f` for that gap. Over 47 gaps, this accumulates.
- **Fix — per-track position tracking with final drift absorption**:
  1. Track `track_positions[track_key]` (cumulative output frames) for each track during construction
  2. After the full loop, compare `track_positions[track_key]` to `help_total_frames` (from help.otio)
  3. If drift exists, absorb it into the last emitted item:
     ```python
     drift = track_positions[track_key] - help_total_frames
     if abs(drift) > 3:
         last_item.source_range = otio_time.TimeRange(
             start_time=last_item.source_range.start_time,
             duration=otio_time.RationalTime(value=float(max(1, int(last_dur - drift))), rate=RATE)
         )
     ```
  4. Apply to all 6 video+audio track pairs — each should independently absorb drift
  5. The last item is typically a Gap, so adjust the gap's duration (not a clip's source_range)

### V2-V6 audio tracks must also be populated
- When placing V2-V6 clips, only `vtrack.append(vclip)` was happening — `atrack` was not getting items
- This caused audio tracks to have half the item count of their video counterparts (e.g., 77 vs 346)
- For every clip placed on V2-V6 video track, also append to `atrack`:
  ```python
  vtrack.append(vclip)
  aclip = make_clip(path, adj_src, clip_dur_f)
  if aclip:
      aclip.name = f'Audio ALT: {fname}'
      atrack.append(aclip)
  else:
      atrack.append(make_gap(clip_dur_f))
  ```

### yt-dlp fallback chain overrides segment_max_resolution
- In `src/downloader/orchestrator.py`, the format string was:
  ```python
  _format_with_fallback = f'{_primary_format}/best/bestvideo+bestaudio'
  ```
  where `_primary_format = "best[height<=360]"`. When no 360p format exists, yt-dlp falls back to 1080p.
- **Fix**: Remove the fallback chain — use strict height constraint with no fallback:
  ```python
  _format_with_fallback = _primary_format  # just "best[height<=360]", no fallback
  ```
- This forces all downloads to respect the height limit or fail gracefully.

### project_config.yaml is not auto-loaded by load_project_config
- `load_project_config()` loads global `config.yaml` but does NOT auto-load `project_config.yaml` from the project directory
- The `download.segment_max_resolution: 360` setting in `project_config.yaml` was being ignored
- **Fix**: Add auto-load logic to `load_project_config()` in `src/cli/config_utils.py`:
  ```python
  project_config_path = project_dir / "project_config.yaml"
  if project_config_path.exists():
      project_override = Config.from_yaml(str(project_config_path), skip_final_validation=True)
      for section_name in ('download', 'video_search', 'caption', 'match'):
          base_section = getattr(config, section_name, None)
          override_section = getattr(project_override, section_name, None)
          if base_section and override_section:
              for field_name in dir(override_section):
                  if not field_name.startswith('_') and hasattr(base_section, field_name):
                      override_val = getattr(override_section, field_name, None)
                      base_val = getattr(base_section, field_name, None)
                      if override_val is not None and override_val != base_val:
                          setattr(base_section, field_name, override_val)
  ```
- Use `skip_final_validation=True` to bypass API key validation during project config load.

### PyManager exit code 127 does not mean pipeline failure
- The PyManager wrapper (`C:\Program Files\PyManager\python.exe`) exits with code 127 when the background task completes
- The actual `python.exe` pipeline process continues running independently
- **Do NOT restart if PyManager wrapper dies** — check `wmic process where "name='python.exe'" get commandline | grep project_name` to confirm actual pipeline is still running
- Only restart if no python process with the project path is found

### Cron monitoring frequency
- Default 30-minute cron is too sparse for a pipeline that can die mid-stage
- Use 5-minute cron (`*/5 * * * *`) for active pipeline monitoring
- Cron job auto-expires after 7 days; recreate as needed
