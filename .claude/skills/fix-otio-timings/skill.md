# Skill: fix-otio-timings

Repairs OTIO timeline clip timings that are out of sync with their reference voiceover SRT.

## When to Use This Skill

Use `fix-otio-timings` when:
- You have a single OTIO file whose clips are misaligned with voiceover timing
- You want to fix timing drift based on an SRT file or checkpoint
- You need to inspect and verify timing alignment before deciding whether to fix

**Not the right tool?** See [Comparison with similar skills](#comparison-with-similar-skills) below.

## Invocation

```
/fix-otio-timings <path to OTIO file> [--srt path/to/reference.srt] [--output path/to/fixed.otio] [--use-checkpoint] [--dry-run]
```

Examples:
```bash
# Basic usage — auto-detect SRT from voiceover/ directory
fix-otio-timings "E:\Edit Job\Marcos\1\output\20260521_022003\timeline_V1_V1___Primary.otio"

# Use a specific SRT file as reference
fix-otio-timings "timeline.otio" --srt "voiceover_trimmed.srt"

# Use checkpoint trimmed_segments (recommended when available)
fix-otio-timings "timeline.otio" --use-checkpoint

# Output to new file instead of overwriting
fix-otio-timings "timeline.otio" --output "timeline_fixed.otio"

# Dry run — check only, no modifications
fix-otio-timings "timeline.otio" --dry-run

# Combine flags
fix-otio-timings "timeline.otio" --srt "voiceover.srt" --output "fixed.otio" --dry-run
```

## Problem This Solves

When a pipeline run uses silence-removed (trimmed) voiceover, the OTIO timeline may have incorrect clip timings because:

1. **`state.voiceover_path` empty on `--output-only`**: The restore logic checks `is_trimmed_voiceover(state.voiceover_path)` — if empty, it skips trimmed detection entirely and uses original timing instead of trimmed timing
2. **Checkpoint stores wrong `trimmed_segments`**: If `_last_trimmed_segments` was captured from `trimmed_segs` (post-compress/stretch) instead of raw Whisper output, the checkpoint stores wrong timestamps
3. **SRT file on disk has corrupted timings**: `voiceover_trimmed.srt` may have been written from compressed segments rather than raw Whisper output, causing drift

## How It Works (Step by Step)

### Step 1: Identify Timeline Timing System

The skill detects which timing system the timeline uses based on total duration:

| Timing System | Duration Range | Description |
|---------------|----------------|-------------|
| **ORIGINAL** | ~650-720s | Full voiceover with silences intact (e.g., 685.9s for Marcos) |
| **TRIMMED** | ~500-600s | Voiceover with silence removed (e.g., 556.1s) |
| **UNKNOWN** | Outside above ranges | Cannot determine — will abort |

**Concrete example:**
```
Timeline duration: 556.1s
Detected timing: TRIMMED (silence-removed voiceover)
```

### Step 2: Load Reference SRT

The skill parses the reference SRT file and builds a list of segments:
```python
# Each segment extracted as:
{
    'start': 12.345,   # seconds
    'end': 18.901,     # seconds
    'text': '...'
}
```

The skill auto-detects the SRT location:
- Looks for `voiceover_trimmed.srt` (preferred for TRIMMED timelines)
- Falls back to `voiceover.srt` (for ORIGINAL timelines)
- Or uses the explicit `--srt` path if provided

### Step 3: Check V1 Clip Alignment

For each clip in the V1 track, the skill computes:

- **Timeline position**: Cumulative position of clip start in the timeline
- **Expected position**: Start time from the corresponding SRT segment
- **Drift**: `timeline_position - expected_position`
- **Duration drift**: `clip_duration - srt_segment_duration`

### Step 4: Fix If Needed

**Option A — Fix from SRT** (when SRT has correct timing):
```
Applying fix from SRT...
```
Rebuilds V1 track with clip positions matching SRT segment timings.

**Option B — Fix from checkpoint** (preferred when available):
```
Using checkpoint: E:\Edit Job\Marcos\1\checkpoint.json
Applying fix from checkpoint.trimmed_segments...
```
Uses `checkpoint.json`'s `analyze.trimmed_segments` field, which contains clean timings from raw Whisper output (not affected by SRT parsing drift).

### Step 5: Validate Fix

Re-checks all clips after applying the fix and reports:
```
Re-checked: 158 | Drifted: 0 | Max drift: 0.015s
```

## The Two Timing Systems Explained

### ORIGINAL Timing

The timeline was built against the **full voiceover** with silences intact.

```
Original voiceover: |---speech---|---silence---|---speech---|---silence---|... (e.g., 685.9s)
Timeline clips:     |---clip 1---|---gap---|---clip 2---|---gap---|...
```

Use ORIGINAL timing when:
- The pipeline used `voiceover.mp3` (not trimmed)
- Clips appear to extend past the voiceover in DaVinci
- Timeline duration is longer than the trimmed voiceover

### TRIMMED Timing

The timeline was built against **silence-removed voiceover**.

```
Trimmed voiceover: |---speech---|---speech---|---speech---|... (e.g., 556.1s)
Timeline clips:    |---clip 1---|---clip 2---|---clip 3---|...
```

Use TRIMMED timing when:
- The pipeline used `voiceover_trimmed.mp3`
- Timeline duration closely matches the trimmed voiceover
- Clips align properly with the shortened voiceover in DaVinci

**Concrete example with real numbers:**

| System | Voiceover Duration | Timeline Duration | Clip Count |
|--------|-------------------|-------------------|------------|
| ORIGINAL (Marcos) | 685.9s | 685.9s | 158 clips |
| TRIMMED (Marcos) | 556.1s | 556.1s | 158 clips |

The clip count stays the same; only the timestamps differ.

## --use-checkpoint vs --srt: When to Use Each

| Scenario | Recommended Flag | Why |
|----------|-----------------|-----|
| Checkpoint has clean `trimmed_segments` | `--use-checkpoint` | Captured from raw Whisper output — no SRT parsing drift |
| SRT file is known to be correct | `--srt` | Direct fix from the reference file |
| Neither checkpoint nor SRT available | (neither) | Will error with "No reference SRT and no checkpoint" |
| Unsure which to use | `--use-checkpoint` first, then verify | Checkpoint is generally more reliable |
| Checkpoint has corrupted data | `--srt` | Fall back to SRT if checkpoint data is wrong |

**Example workflow:**
```bash
# First, check what the issue is
fix-otio-timings "timeline.otio" --dry-run

# If SRT is the source of truth, use it
fix-otio-timings "timeline.otio" --srt "voiceover_trimmed.srt"

# If checkpoint has clean data, prefer it
fix-otio-timings "timeline.otio" --use-checkpoint
```

## How to Interpret Results

### Drift Values Explained

| |drift| Value | Meaning | Action |
|---|------|--------|---------|--------|
| |drift| ≤ 0.033s | Acceptable — sub-frame at 30fps | No action needed |
| 0.033s < |drift| ≤ 0.5s | Minor rounding differences | Usually OK, monitor |
| |drift| > 0.5s | Error — clips misaligned | Fix recommended |
| |drift| > 5s | Severe — wrong timing system | Check ORIGINAL vs TRIMMED |

### Sample Output Interpretation

```
[fix-otio-timings] Checking: E:\Edit Job\Marcos\1\output\...\timeline.otio
[fix-otio-timings] Detected timing: TRIMMED (556.1s)
[fix-otio-timings] Reference SRT: E:\Edit Job\Marcos\1\voiceover\voiceover_trimmed.srt
[fix-otio-timings] V1 clips: 158
[fix-otio-timings] SRT segments: 158
[fix-otio-timings] Clips checked: 158 | Drifted: 0 | Max drift: 0.017s
[fix-otio-timings] Timeline OK — no fix needed
```

**Interpretation:** Timeline uses TRIMMED timing (556.1s), SRT has 158 segments, all clips are within tolerance (max drift 0.017s < 0.033s threshold).

### When Fix Is Applied

```
[fix-otio-timings] Clips checked: 158 | Drifted: 12 | Max drift: 8.3s
[fix-otio-timings] Applying fix from checkpoint.trimmed_segments...
[fix-otio-timings] Fix applied. Re-checking...
[fix-otio-timings] Re-checked: 158 | Drifted: 0 | Max drift: 0.015s
[fix-otio-timings] Fixed timeline saved: E:\Edit Job\Marcos\1\output\...\timeline.otio
```

**Interpretation:** 12 clips had drift > 0.5s (max 8.3s), fix was applied from checkpoint, re-check shows all clips now aligned (max drift 0.015s).

## Troubleshooting

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `OTIO file not found: <path>` | File path incorrect or missing | Verify path exists; use absolute path |
| `No reference SRT found` | SRT not in expected location and none specified | Use `--srt` to specify SRT path |
| `No checkpoint found` | No checkpoint.json in project directory | Ensure project has checkpoint or use `--srt` instead |
| `No reference SRT and no checkpoint — cannot fix` | Neither SRT nor checkpoint available | Provide `--srt` or ensure checkpoint exists |
| `Checkpoint has no trimmed_segments` | Checkpoint missing the field | Use `--srt` instead |
| `Timeline OK — no fix needed` | All clips within tolerance | No action needed |
| `Detected timing: UNKNOWN` | Duration doesn't match ORIGINAL or TRIMMED ranges | Verify you're using correct OTIO file |
| `SRT differs from checkpoint trimmed_segments` | SRT and checkpoint data conflict | Checkpoint data is used as authoritative |

## Multi-Track Support

Currently, `fix-otio-timings` operates on **V1 track only**. It does not fix V2-V6 or A1-A8 tracks.

For full multi-track regeneration (V1-V12, A1-A8), use **`fix-trimmed-otio`** instead, which regenerates the entire timeline from checkpoint.

## Project Compatibility

This skill works on **ANY project's OTIO**, not just Marcos projects. Requirements:

1. **OTIO file** with clip timing to fix
2. **Either:**
   - `checkpoint.json` (or `checkpoint.backup.json`) in the project directory containing `analyze.trimmed_segments`, OR
   - An SRT file (`voiceover.srt` or `voiceover_trimmed.srt`) alongside the voiceover

The skill auto-detects the project directory from the OTIO path:
```
OTIO path: .../output/<timestamp>/timeline_V1_*.otio
Project path: .../ (parent of output/)
```

## Comparison with Similar Skills

| Skill | What It Does | When to Use |
|-------|--------------|-------------|
| **fix-otio-timings** | Fixes V1 clip timings in a single OTIO file using SRT or checkpoint | Single OTIO file needs timing correction; V1 clips misaligned |
| **fix-trimmed-otio** | Regenerates ALL tracks (V1-V12, A1-A8) from checkpoint; fixes checkpoint/SRT first | Full project needs regeneration; multi-track timeline wrong |
| **verify-timing** | Reports timing drift per-segment; does NOT fix | Quick verification only; no modifications needed |

### Decision Guide

```
Do you need to fix timing?
├── NO → use verify-timing (read-only verification)
└── YES → What needs fixing?
    ├── Single OTIO file, V1 only → fix-otio-timings
    └── Full project, all tracks → fix-trimmed-otio
```

### Key Differences

**fix-otio-timings vs fix-trimmed-otio:**
- `fix-otio-timings`: Operates on one OTIO file; V1 only; uses existing OTIO structure
- `fix-trimmed-otio`: Regenerates entire project output; all tracks (V1-V12, A1-A8); full pipeline re-run

**fix-otio-timings vs verify-timing:**
- `fix-otio-timings`: Reads AND writes OTIO; can apply fixes
- `verify-timing`: Read-only; reports drift only; more detailed per-segment reporting

## Key Code References

- **Trimmed detection fix**: `src/stages/analyze.py` lines 239-249 — scans voiceover dir for `_trimmed.*` when `state.voiceover_path` is empty
- **Checkpoint storage fix**: `src/stages/analyze.py` lines 670-677 — captures from `result` (raw Whisper) not `trimmed_segs` (post-modify)
- **Remap function**: `src/transcription/utils.remap_segments_to_original_time()` — converts trimmed timestamps to original timeline using speech regions
- **Timeline creation**: `src/otio/timeline.create_timeline()` — places clips using `state.voiceover_segments[i]` timing

## Drift Thresholds

| Threshold | Value | Behavior |
|-----------|-------|----------|
| Acceptable | |drift| ≤ 0.033s | Sub-frame at 30fps; no action |
| Warning | 0.033s < |drift| ≤ 0.5s | Minor rounding; usually OK |
| Error | |drift| > 0.5s | Clips misaligned; fix recommended |
| Severe | |drift| > 5s | Wrong timing system; check ORIGINAL vs TRIMMED |
