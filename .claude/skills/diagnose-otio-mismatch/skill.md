# Skill: diagnose-otio-mismatch

Debug OTIO timeline timing issues by identifying WHERE the problem originates rather than fixing it directly.

## When to Use

Use `diagnose-otio-mismatch` when:
- A pipeline run produced a timeline with wrong clip positions, wrong durations, or wrong gap placement
- `--output-only` gives different results than a fresh pipeline run
- You want to understand WHY the timeline is wrong before touching code

**NOT for:** Actually fixing the timeline (use `fix-otio-timings` or `fix-trimmed-otio` for that)

## Invocation

```
/diagnose-otio-mismatch <path to timeline.otio> [--srt path/to/srt] [--project path/to/project]
```

Examples:
```bash
# With project — checks checkpoint, SRT on disk, and timeline
diagnose-otio-mismatch "E:\Edit Job\Marcos\1\output\20260522_034153\timeline_FULL.otio" --project "E:\Edit Job\Marcos\1"

# With SRT reference
diagnose-otio-mismatch "output/timeline_FULL.otio" --srt "voiceover/voiceover_trimmed.srt"

# Check which SRT is being used (trimmed vs original)
diagnose-otio-mismatch "output/timeline_V1_V1___Primary.otio" --project "E:\Edit Job\Marcos\1" --check-srt-source
```

---

## Quick Reference

| Symptom | Root Cause | Look Here |
|---------|-----------|-----------|
| Spurious gaps in V1 (clips separated by gaps that don't match SRT silences) | `timeline_frames` advances by `source_range.duration` (media duration), not SRT segment duration — causes drift | `timeline.py` lines ~1014-1019 |
| `duration_frames` too large or too small, causing clips to extend past audio | `expected_end_frames = _current_srt_end - _first_srt_start + leading_frames` uses match data (checkpoint), not actual SRT on disk | `timeline.py` lines ~1118-1124 |
| Timeline shows gaps but checkpoint SRT says no silence | `_resync_trimmed_segments()` was skipped because `state.voiceover_path` was empty on `--output-only` | `analyze.py` lines ~261-266 |
| Leading gap of ~0.5s when SRT starts at 0.000s | `first_segment_start` from match data has 0.500s offset instead of 0.000s from actual SRT | `timeline.py` line ~802 |
| `--output-only` produces different timing than fresh run | Checkpoint stored original-audio timings in `trimmed_segments`; SRT on disk has trimmed timings | `analyze.py` lines ~262-266 |
| Timeline clips extend past the voiceover in DaVinci | Used ORIGINAL timing but editor has TRIMMED audio | `analyze.py` lines ~239-260 |

---

## Diagnostic Output

The skill reports findings in six diagnostic areas. Each finding tells you WHAT to investigate and WHERE to look.

### Area 1: Timeline vs SRT Structure

Compares clip count and durations against the reference SRT. Contains four sub-checks:

**Sub-check 1a — Clip count vs SRT segment count**
- Compares `len(v1_clips)` to `len(srt_segments)`
- Mismatch triggers investigation of gap insertion logic

**Sub-check 1b — SRT vs Timeline total duration**
- Sums SRT segment durations and compares to `sum(clips) + sum(gaps)` in V1
- Flags SEVERE mismatch (> 1.0s) or minor (0.1s-1.0s)

**Sub-check 1c — Checkpoint vs Timeline clip count**
- Cross-validates `checkpoint.analyze.trimmed_segments` count against V1 clip count
- Detects clips dropped during OTIO export or duplicate clip insertion

**Sub-check 1d — Gap Location Analysis**
- When gaps exist, reports which SRT segments surround each gap
- Shows: gap duration, timeline position, before/after SRT segment text
- Flags gaps > 2s as likely incorrectly preserved silence

**Look here if mismatch:**
- `src/otio/timeline.py` lines ~1118-1124 — how `duration_frames` is calculated per clip
- `src/otio/timeline.py` lines ~1014-1019 — how gaps are inserted between clips

**Common causes:**
- `duration_frames = max(1, expected_end_frames - timeline_frames)` uses SRT end positions, but `timeline_frames` advances by `source_range.duration` (media duration), NOT SRT segment duration
- This causes `timeline_frames` to drift ahead/behind, triggering spurious gap insertions

### Area 2: Checkpoint vs Actual SRT

Checks whether the checkpoint's stored `trimmed_segments` match the SRT file on disk. Contains two sub-checks:

**Sub-check 2a — Timing vs SRT on disk**
- Compares `checkpoint.trimmed_segments[0].start` against `srt_segments[0].start`
- Flags mismatches indicating trimmed resync was skipped

**Sub-check 2b — Confidence Correlation**
- Scans `trimmed_segments` for confidence/score/match_confidence fields < 0.6
- Also flags segments where checkpoint duration differs from SRT duration by > 1.0s
- Reports low-confidence segments that may indicate timing issues
- Threshold for reporting: 10 segments (extras truncated)

**Look here if mismatch:**
- `src/stages/analyze.py` lines ~261-266 — the restore logic for trimmed voiceover
- `src/stages/analyze.py` lines ~282-344 — `_resync_trimmed_segments()` method

**Common causes:**
- On `--output-only`, if `state.voiceover_path` is empty, the restore logic at line ~261 checks `is_trimmed_voiceover(state.voiceover_path)` which returns False, skipping the trimmed resync entirely
- The checkpoint stores `_last_trimmed_segments` from post-transcription processing, not from the actual SRT file on disk
- Result: timeline uses wrong segment timings (checkpoint's instead of actual SRT's)

### Area 3: Leading Gap Investigation

Checks whether the timeline correctly handles the first SRT segment's start time.

**Look here if wrong:**
- `src/otio/timeline.py` lines ~938-963 — `leading_frames` calculation and leading gap insertion
- `src/otio/timeline.py` line ~802 — `voiceover_offset` application via `first_segment_start`

**Common causes:**
- If first SRT segment starts at 0.000s but checkpoint stored 0.500s, the leading gap will be 0.5s instead of 0s
- This happens because `first_segment_start = _seg_start(matches[0].primary_match.voiceover_segment)` uses checkpoint data, not the SRT on disk

### Area 4: SRT Source Resolution

For `--output-only` runs, determines which SRT file was actually used.

**Look here if wrong:**
- `src/stages/analyze.py` lines ~239-260 — how `vo_path` is reconstructed from checkpoint
- `src/stages/analyze.py` lines ~306-309 — how trimmed SRT companion file is located

**Common causes:**
- When `state.voiceover_path` is empty on `--output-only`, the code tries to reconstruct it from `checkpoint.data.voiceover_path`
- If the reconstructed path doesn't contain `_trimmed`, the `is_trimmed_voiceover()` check at line ~261 fails
- The trimmed resync is skipped even though the voiceover being used IS trimmed

### Area 5: Clip Duration vs SRT Duration

Compares each clip's duration against its corresponding SRT segment duration.

**Look here if mismatch:**
- `src/otio/timeline.py` lines ~1118-1124 — how `duration_frames` is calculated per clip

**Common causes:**
- `duration_frames = max(1, expected_end_frames - timeline_frames)` uses SRT end positions but `timeline_frames` advances by `source_range.duration` (media duration), not SRT segment duration
- When downloaded clip duration differs from SRT segment duration, this drift accumulates across all clips
- Per-clip rounding error: `round(duration * rate)` can lose/gain a frame per clip; over hundreds of clips this accumulates to seconds of misalignment

**Threshold:** Reports mismatches > 0.1s difference

### Area 6: Timeline Total Duration vs SRT Total Duration

Compares the sum of all SRT segment durations against the V1 track total duration.

**Look here if mismatch:**
- `src/otio/timeline.py` lines ~1118-1124 — how `duration_frames` is calculated
- `src/otio/timeline.py` lines ~1014-1019 — gap insertion logic

**Common causes:**
- Timeline is LONGER than SRT: clips have extra padding or media duration exceeds SRT segment duration
- Timeline is SHORTER than SRT: clips are trimmed or gaps are missing
- Accumulated drift from `timeline_frames` advancing by media duration instead of SRT segment duration
- Spurious gaps inserted due to `timeline_frames` drift causing `expected_start_frames > timeline_frames`

**Threshold:** Reports mismatches > 0.1s difference

---

## Diagnostic Checklist

Run through this checklist to identify the root cause:

| # | Check | How to Check | Expected | If Wrong |
|---|-------|--------------|----------|----------|
| 1 | Timeline clip count vs SRT segment count | Load OTIO, count clips in V1 | Equal to SRT segment count | Investigate `timeline.py` gap logic (lines ~1014-1019) |
| 2 | Timeline gap count vs SRT silence count | Load OTIO, count gaps in V1 | 0 for trimmed, matches SRT silences for original | `timeline.py` gap insertion: `if expected_start_frames > timeline_frames:` |
| 3 | Checkpoint `trimmed_segments[0].start` vs actual SRT | Load checkpoint, parse `voiceover_trimmed.srt` | Should match (within 0.01s) | `analyze.py` lines ~261-266 — `_resync_trimmed_segments()` was skipped |
| 4 | Actual SRT first segment start time | Parse `voiceover_trimmed.srt` first timestamp | Should be 0.000s for trimmed, original start for original | Check which SRT file is being used — see Area 4 |
| 5 | Checkpoint `segments[0].start` vs `trimmed_segments[0].start` | Load checkpoint, compare two arrays | May differ if checkpoint captured from post-transcription processing | `analyze.py` lines ~143-148 — how segments are captured |
| 6 | `state.voiceover_path` during `--output-only` restore | Log `state.voiceover_path` at restore time | Should point to `*_trimmed.mp3` for trimmed voiceover | `analyze.py` lines ~239-260 — vo_path reconstruction |
| 7 | Trimmed SRT companion file exists | Check `voiceover_trimmed.srt` alongside `voiceover_trimmed.mp3` | File must exist for `_resync_trimmed_segments()` to work | `analyze.py` lines ~306-309 |
| 8 | Checkpoint last segment end vs SRT last segment end | Compare `checkpoint.analyze.trimmed_segments[-1].end` vs SRT last | Should match (within 0.5s) — resync only triggered on >0.5s mismatch | `analyze.py` lines ~325-330 — mismatch detection threshold |
| 9 | `first_segment_start` source | Check `timeline.py` line ~802 | Uses `_seg_start(matches[0].primary_match.voiceover_segment)` from match data, not SRT file | `timeline.py` lines ~938-963 — leading_frames calculation |
| 10 | Leading gap size vs SRT first segment start | Compare timeline's first item duration vs SRT first segment start | For trimmed: leading gap should be 0 (SRT starts at 0) | `timeline.py` lines ~940-963 — leading_frames calculation |
| 11 | Duration calculation uses absolute positioning | Check `timeline.py` lines ~1118-1124 | `_first_srt_start` and `_current_srt_end` from match data (checkpoint), not recalculated from SRT | `timeline.py` lines ~995-1012 — loop uses `_seg_start` from matches |
| 12 | `gap_mode` configuration | Check `config.output.gap_mode` | Default is `scale`; `extend` mode extends previous clip to fill gap | `timeline.py` lines ~866-869 |

---

## Common Error Patterns

### Pattern 1: Spurious Gaps in V1

**Symptoms:** Timeline has gaps between clips that do not correspond to silence in the SRT.

**Root cause:** `timeline_frames` advances by `source_range.duration` (the downloaded video's actual duration), but gap detection uses `expected_start_frames` calculated from SRT timing. When `source_range.duration` differs from the SRT segment duration (which is common), `timeline_frames` drifts from `expected_start_frames`, triggering false gap insertions.

**Key code — `timeline.py` lines ~1010-1019:**
```python
first_srt_start = round(_seg_start(matches[0].primary_match.voiceover_segment) * time_scale_factor * frame_rate)
current_srt_start = round(_seg_start(vo_seg) * time_scale_factor * frame_rate)
expected_start_frames = current_srt_start - first_srt_start + leading_frames

if expected_start_frames > timeline_frames:   # <-- spurious gap when timeline_frames drifts
    gap_frames = expected_start_frames - timeline_frames
    gap_seconds = gap_frames / rate

    if gap_seconds >= min_gap_threshold:
        # Gap is significant — insert it
```

**Fix approach:** Use `fix-otio-timings` or `fix-trimmed-otio`.

---

### Pattern 2: Timeline Longer Than Voiceover in DaVinci

**Symptoms:** Clips extend past the end of the voiceover track. Editor sees timeline duration > audio duration.

**Root cause:** Checkpoint stores segments in **original-audio time** (full duration with silences), but editor uses **trimmed voiceover** (silence removed, shorter). The checkpoint's `trimmed_segments` field has original-audio timestamps (e.g., first segment starts at 0.500s), but the actual `voiceover_trimmed.srt` on disk has trimmed timestamps (first segment at 0.000s).

**Key code — `analyze.py` lines ~261-266:**
```python
if isinstance(vo_path, str) and vo_path and is_trimmed_voiceover(vo_path):
    # Always re-sync from the actual trimmed SRT on disk
    self._resync_trimmed_segments(state, segments)
elif trimmed_seg_dicts:
    # Fallback: re-parse trimmed SRT from disk (old checkpoints)
    self._resync_trimmed_segments(state, segments)
```

**Why it fails:** On `--output-only`, `state.voiceover_path` may be empty (`''`), so `is_trimmed_voiceover(vo_path)` returns `False`, and the resync is skipped entirely.

**Fix approach:** `fix-trimmed-otio` — regenerates all tracks from checkpoint.

---

### Pattern 3: Wrong First Segment Start (Leading Gap Mismatch)

**Symptoms:** Timeline has a leading gap of ~0.5s even though the SRT starts at 0.000s.

**Root cause:** `first_segment_start` at `timeline.py` line ~802 uses match data from checkpoint, not the actual SRT file:
```python
first_segment_start = _seg_start(matches[0].primary_match.voiceover_segment) if matches else 0.0
```

If the checkpoint's first segment was captured when voiceover started at 0.500s (e.g., after a intro silence), the leading gap calculation uses 0.500s:
```python
adjusted_first_segment_start = max(0.0, (first_segment_start * time_scale_factor) + voiceover_offset)
leading_frames = round(adjusted_first_segment_start * rate) if ... else 0
```

**Fix approach:** `_resync_trimmed_segments()` should correct this when called, but only if it runs.

---

### Pattern 4: Checkpoint/SRT Mismatch (Checkpoint vs Disk)

**Symptoms:** `checkpoint.analyze.trimmed_segments` has different timings than `voiceover_trimmed.srt` on disk.

**Root cause:** The checkpoint was captured from post-transcription processing where `trimmed_segments` may have been filled from `trimmed_segs` (post-compress/stretch), not from the raw SRT output. Later `--output-only` runs re-parse the SRT on disk and find a mismatch.

**Key code — `analyze.py` lines ~325-330:**
```python
cp_last_end = checkpoint_segments[-1].end if checkpoint_segments else 0
trimmed_last_end = trimmed_segments[-1].end if trimmed_segments else 0
if abs(cp_last_end - trimmed_last_end) < 0.5:
    # Timings already match — no resync needed
    return
```

**Fix approach:** `_resync_trimmed_segments()` remaps checkpoint segment timings to match the actual SRT on disk, preserving text from checkpoint but using timing from SRT.

---

## Example Sessions

### Example A: Good Timeline (All Checks Pass)

```
$ /diagnose-otio-mismatch "E:\Edit Job\Marcos\1\output\20260522_034153\timeline_FULL.otio" --project "E:\Edit Job\Marcos\1"

[diagnose-otio-mismatch] Analyzing: timeline_FULL.otio
[diagnose-otio-mismatch] Project: E:\Edit Job\Marcos\1
[diagnose-otio-mismatch] Reference SRT: E:\Edit Job\Marcos\1\voiceover\voiceover_trimmed.srt
[diagnose-otio-mismatch] SRT segments: 222

=== Area 1: Timeline vs SRT Structure ===
V1 items: 222 total, 222 clips, 0 gaps
SRT segments: 222
✓ Clip count matches SRT segment count
✓ No gaps (expected for trimmed voiceover)
✓ Clip positions match SRT (max drift 0.010s < 0.033s threshold)

=== Area 2: Checkpoint vs Actual SRT ===
Checkpoint analyze.segments: 222
Checkpoint trimmed_segments: 222
Checkpoint trimmed_segments[0].start: 0.000s
Actual voiceover_trimmed.srt[0].start: 0.000s
✓ Checkpoint matches actual SRT

=== Area 3: Leading Gap Investigation ===
Timeline has no leading gap (first item is a clip)
First SRT segment starts at: 0.000s
✓ Leading gap matches SRT first segment start

=== Area 4: SRT Source Resolution ===
Trimmed audio found: voiceover_trimmed.mp3
✓ Companion SRT exists: voiceover_trimmed.srt

=== DIAGNOSIS ===
✓ No obvious issues detected

Key investigation points:
  1. src/stages/analyze.py line ~261 — is trimmed resync being called? ✓ YES
  2. src/stages/analyze.py lines ~239-260 — is voiceover_path reconstructed correctly? ✓ YES
  3. src/otio/timeline.py lines ~1118-1124 — duration_frames calculation ✓ OK
  4. src/otio/timeline.py lines ~1014-1019 — gap insertion logic ✓ NO SPURIOUS GAPS
```

---

### Example B: Spurious Gaps (timeline_frames Drift)

```
$ /diagnose-otio-mismatch "E:\Edit Job\Client\1\output\timeline_FULL.otio" --project "E:\Edit Job\Client\1"

[diagnose-otio-mismatch] Analyzing: timeline_FULL.otio

=== Area 1: Timeline vs SRT Structure ===
V1 items: 158 total, 140 clips, 18 gaps
SRT segments: 158
⚠ MISMATCH: 140 clips vs 158 SRT segments
⚠ V1 has 18 gaps — investigate if these match SRT silences
  → Look at: src/otio/timeline.py lines ~1014-1019
  → Gap insertion uses: if expected_start_frames > timeline_frames:
  → timeline_frames advances by source_range.duration (media dur), not SRT segment dur
  → This causes drift → spurious gaps

=== DIAGNOSIS ===
⚠ Timeline has 18 gaps in V1
  → Problem: src/otio/timeline.py lines ~1014-1019
  → timeline_frames advances by source_range.duration, not SRT segment duration
  → This causes timeline_frames to drift, triggering spurious gaps
  → Fix: use fix-otio-timings or fix-trimmed-otio
```

**Interpretation:** 18 gaps in a 158-segment timeline means ~11% of segments triggered false gap insertions. The gap logic at line ~1014 compares `expected_start_frames` (from SRT) against `timeline_frames` (advancing by media duration). When a downloaded video is slightly shorter or longer than the SRT segment expects, drift accumulates.

---

### Example C: Checkpoint/SRT Mismatch (Trimmed Resync Skipped)

```
$ /diagnose-otio-mismatch "E:\Edit Job\Marcos\1\output\20260521_022003\timeline_FULL.otio" --project "E:\Edit Job\Marcos\1"

[diagnose-otio-mismatch] Analyzing: timeline_FULL.otio

=== Area 2: Checkpoint vs Actual SRT ===
Checkpoint analyze.segments: 158
Checkpoint trimmed_segments: 158
Checkpoint trimmed_segments[0].start: 0.500s
Actual voiceover_trimmed.srt[0].start: 0.000s
⚠ MISMATCH: checkpoint uses 0.500s, SRT has 0.000s
  → This causes wrong clip positions in timeline
  → Look at: src/stages/analyze.py lines ~261-266
  → The restore logic checks: is_trimmed_voiceover(state.voiceover_path)
  → If state.voiceover_path is empty or doesn't contain '_trimmed',
     the trimmed resync is skipped and wrong timings are used

=== DIAGNOSIS ===
⚠ Checkpoint uses different segment timing than SRT on disk
  → Problem: src/stages/analyze.py line ~261
  → Fix: always call _resync_trimmed_segments() for trimmed voiceover
  → Recommended: run fix-trimmed-otio to regenerate all tracks
```

**Interpretation:** Checkpoint stored first segment at 0.500s (original-audio time) but actual trimmed SRT starts at 0.000s. The 0.5s offset means `is_trimmed_voiceover()` returned False during restore, skipping `_resync_trimmed_segments()`. This causes all 158 clips to be positioned 0.5s too late.

---

### Example D: Severe Position Drift (duration_frames Accumulated Error)

```
$ /diagnose-otio-mismatch "E:\Edit Job\Old\1\output\timeline_FULL.otio" --project "E:\Edit Job\Old\1"

[diagnose-otio-mismatch] Analyzing: timeline_FULL.otio

=== Area 1: Timeline vs SRT Structure ===
V1 items: 222 total, 222 clips, 0 gaps
SRT segments: 222
✓ Clip count matches SRT segment count
⚠ SEVERE position drift: 12.4s (> 0.5s threshold)
  → Look at: src/otio/timeline.py lines ~1118-1124
  → duration_frames = max(1, expected_end_frames - timeline_frames)
  → timeline_frames advances by media duration, causing accumulated drift

=== DIAGNOSIS ===
⚠ Severe position drift: 12.4s
  → Problem: src/otio/timeline.py lines ~1118-1124
  → duration_frames calculation causes accumulated drift
  → Fix: use fix-otio-timings to rebuild V1 from SRT/checkpoint
```

**Interpretation:** 12.4s drift at clip 222 means each clip accumulates ~0.056s error on average. This is the classic `round(duration * rate)` per-clip floating-point error that the absolute positioning fix at line ~1118 was meant to address — but the checkpoint's match data may have been captured before that fix was applied.

---

## Key Code Locations

When investigating OTIO timing issues, check these in order:

### 1. Trimmed Voiceover Detection — `analyze.py` lines ~239-266

**What to check:** Is trimmed voiceover being detected correctly on `--output-only`?

```python
# Line ~239-260: vo_path reconstruction when state.voiceover_path is empty
if not vo_path and trimmed_seg_dicts:
    # Try to reconstruct from checkpoint's stored voiceover_path
    cp_vo_path = checkpoint.data.voiceover_path if hasattr(checkpoint, 'data') else ''
    if cp_vo_path:
        orig_vo = Path(cp_vo_path)
        trimmed_audio = orig_vo.parent / f'{orig_vo.stem}_trimmed{orig_vo.suffix}'
        if trimmed_audio.exists():
            vo_path = str(trimmed_audio)

# Line ~261-266: The resync call — ONLY runs if is_trimmed_voiceover returns True
if isinstance(vo_path, str) and vo_path and is_trimmed_voiceover(vo_path):
    self._resync_trimmed_segments(state, segments)
```

**Common failure:** `state.voiceover_path` is `''` on `--output-only`, `cp_vo_path` is also `''`, so the scan at lines ~255-258 looks for any `*_trimmed.*` file in the voiceover directory. If found, `vo_path` gets set and resync runs. But if the scan fails or no trimmed file exists, resync is skipped.

---

### 2. Duration Calculation — `timeline.py` lines ~1118-1124

**What to check:** Is `duration_frames` using absolute positioning to avoid accumulated drift?

```python
# Line ~1121-1123: Absolute end position prevents drift
_first_srt_start = round(_seg_start(matches[0].primary_match.voiceover_segment) * time_scale_factor * frame_rate)
_current_srt_end = round(_seg_end(vo_seg) * time_scale_factor * frame_rate)
expected_end_frames = _current_srt_end - _first_srt_start + leading_frames
duration_frames = max(1, expected_end_frames - timeline_frames)
```

**Why it matters:** Previous code used `round(source_range.duration.value)` per clip, which could lose/gain a frame per clip. Over 800 clips this accumulates to seconds of misalignment. The absolute positioning fix calculates `expected_end_frames` as an absolute position in the timeline, then subtracts `timeline_frames` (the running position) to get duration.

---

### 3. Gap Insertion — `timeline.py` lines ~1014-1019

**What to check:** Is `timeline_frames` advancing correctly relative to `expected_start_frames`?

```python
# Line ~1014: The gap trigger
if expected_start_frames > timeline_frames:
    gap_frames = expected_start_frames - timeline_frames
    gap_seconds = gap_frames / rate

    if gap_seconds >= min_gap_threshold:
        # Gap is significant
        ...
```

**Common failure:** `timeline_frames` advances by `source_range.duration` (the downloaded video's actual duration), but `expected_start_frames` is calculated from SRT segment start times. If the downloaded video is even 1 frame shorter per clip than the SRT expects, by clip 200 the drift is ~200 frames (~6.7s at 30fps).

---

### 4. Leading Gap Calculation — `timeline.py` lines ~938-963

**What to check:** Is `leading_frames` calculated from the actual SRT first segment start?

```python
# Line ~938: adjusted_first_segment_start includes offset and scaling
adjusted_first_segment_start = max(0.0, (first_segment_start * time_scale_factor) + voiceover_offset)

# Line ~941: leading_frames from adjusted_first_segment_start
leading_frames = round(adjusted_first_segment_start * rate) if matches and adjusted_first_segment_start > 0.1 else 0

# Line ~802: first_segment_start comes from match data, not SRT on disk
first_segment_start = _seg_start(matches[0].primary_match.voiceover_segment) if matches else 0.0
```

**Common failure:** `first_segment_start` uses `_seg_start(matches[0].primary_match.voiceover_segment)` — the first match's voiceover segment from checkpoint. If checkpoint was captured with original-audio timing (0.500s start), `leading_frames` will be calculated from 0.500s instead of 0.000s.

---

### 5. Trimmed SRT Resync — `analyze.py` lines ~282-344

**What to check:** Does `_resync_trimmed_segments()` correctly detect and fix the mismatch?

```python
# Lines ~306-309: Find companion trimmed SRT
trimmed_srt = vp.with_suffix('.srt')
if not trimmed_srt.exists():
    return  # Early exit — resync skipped if SRT missing

# Lines ~317-323: Segment count check
if len(trimmed_segments) != len(checkpoint_segments):
    logger.warning("Trimmed SRT segment count differs from checkpoint, skipping resync")
    return  # Early exit — can't resync if segment counts differ

# Lines ~325-330: Timing mismatch detection
cp_last_end = checkpoint_segments[-1].end if checkpoint_segments else 0
trimmed_last_end = trimmed_segments[-1].end if trimmed_segments else 0
if abs(cp_last_end - trimmed_last_end) < 0.5:
    return  # Already matches — no resync needed
```

**Common failure:** If segment counts differ, resync is skipped entirely. If the threshold `0.5s` is too permissive, large mismatches might go undetected.

---

## Related Skills

| Skill | What It Does | Use When |
|-------|--------------|----------|
| **fix-otio-timings** | Fixes V1 clip timings in a single OTIO file using SRT or checkpoint | Single OTIO file has timing drift; V1 clips misaligned |
| **fix-trimmed-otio** | Regenerates ALL tracks (V1-V12, A1-A8) from checkpoint; fixes checkpoint/SRT first | Full project needs regeneration; multi-track timeline wrong |
| **verify-timing** | Reports timing drift per-segment; does NOT fix | Quick verification only; no modifications needed |

### Decision Guide

```
Do you need to fix timing?
├── NO → use verify-timing (read-only verification)
└── YES → What needs fixing?
    ├── Single OTIO file, V1 only → fix-otio-timings
    └── Full project, all tracks → fix-trimmed-otio

Is the issue diagnosed but unclear which fix to use?
└── Use diagnose-otio-mismatch first to identify the root cause,
    then pick the appropriate fix skill based on the pattern.
```

---

## Key Code Locations to Remember

When investigating OTIO timing issues, check these in order:

1. **First stop — checkpoint vs SRT**: `src/stages/analyze.py` lines ~261-266
   - Is trimmed voiceover being detected correctly?
   - Is `_resync_trimmed_segments()` being called?

2. **Second stop — duration calculation**: `src/otio/timeline.py` lines ~1118-1124
   - `duration_frames = max(1, expected_end_frames - timeline_frames)`
   - Is `timeline_frames` advancing correctly?

3. **Third stop — gap insertion**: `src/otio/timeline.py` lines ~1014-1019
   - Gap logic: `if expected_start_frames > timeline_frames:`
   - Spurious gaps mean `timeline_frames` has drifted

4. **Fourth stop — segment source**: `src/otio/timeline.py` line ~802
   - `first_segment_start = _seg_start(matches[0].primary_match.voiceover_segment)`
   - Uses match data, not SRT on disk

5. **Fifth stop — clip duration**: `src/otio/timeline.py` lines ~1118-1124
   - `duration_frames = max(1, expected_end_frames - timeline_frames)`
   - `timeline_frames` advances by `source_range.duration` (media duration), not SRT segment duration
   - Per-clip rounding error accumulates over hundreds of clips

6. **Sixth stop — total duration**: `src/otio/timeline.py` lines ~1118-1124
   - Sum of all clip durations vs sum of all SRT segment durations
   - Drift indicates accumulated error from per-clip duration calculation