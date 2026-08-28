---
name: capcut-export
description: Export the top-most visible clips of an OTIO timeline as individual, exactly-cut MP4 files for drag-and-drop into CapCut. CapCut cannot import OTIO/EDL/XML, so this cuts each HQ source to the OTIO clip's exact source_range (offset + duration), frame-accurate, and writes ascending-numbered files (clip_0001.mp4 ...) plus a manifest. Use when the user says "export for capcut", "cut clips for capcut", "capcut import", or wants the top-most OTIO clips as standalone files.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - Agent
---

# CapCut Export Skill

CapCut has **no project import** (no OTIO/EDL/XML). To move a finished OTIO edit into
CapCut you must hand it individual media files. This skill cuts the **top-most visible
clip at every timeline moment** from its HQ source to the **exact** `source_range`
(offset into the source + duration) the OTIO specifies, re-encoding for frame accuracy,
and writes them as ascending-numbered files that drag straight into CapCut's media panel
in timeline order.

## When to Use

- User says "export for capcut", "cut clips for capcut", "make capcut files", "capcut import"
- User has an OTIO (often the HQ `_hq.otio` from the re-download-otio skill) and wants the
  visible clips as standalone MP4s cut to the exact edit points
- The downloaded HQ source files are **longer** than the OTIO clips (they contain the whole
  arc); this skill trims each to the precise in/out the timeline uses

## Invocation

```
/capcut-export <project_dir>
```

## Direct Script Invocation (recommended)

```bash
python scripts/capcut-export.py --project <project_dir>
```

Flags: `--otio <file>`, `--hq-cache <dir>`, `--out <dir>`, `--dry-run`, `--force`.

**Defaults:**
- `--otio`     → `<project>/downloadplease.otio` (or pass `<project>/_hq.otio` for
  HQ-quality cuts)
- `--hq-cache` → `<project>/v/matcher-hq`
- `--out`      → `<project>/capcut_import`

Encoding constants (in-script): `RATE=30.0`, `libx264 -preset fast -crf 18`, `aac -b:a 192k`,
`-movflags +faststart`.

**Resume after crash:** the script skips existing `clip_NNNN.mp4` whose probed
duration matches the manifest (Phase 5 status=`skip`). Just re-run the same
command — completed clips are preserved, interrupted ones get re-cut from scratch.
Use `--force` only if you want to re-cut everything (e.g., to pick up a new
hybrid-seek fix).

### Performance

Benchmarked on Fransisco (431 clips, 26.6 min timeline, 1080p sources):

| Approach | Time | Notes |
|----------|------|-------|
| Pure `-ss` after -i, naive keyframe probe | 25-30 min | Frame-accurate, slow |
| **Hybrid seek + packet-index KF cache** | **~13 min** | 2x faster, ±3 bad_start (re-cut those) |

Key optimisations:
- **Packet-index keyframe cache** (`-show_entries packet=pts_time,flags` with
  `K__` filter, ~0.2s/source) replaces per-cut probing. Cache one source once;
  re-use across all its cuts.
- **Hybrid seek** (`-ss prev_kf` BEFORE -i + `-ss fine_seek` AFTER -i) skips
  decoding from file start. See lesson #12.
- **Never use `-read_intervals "%${offset}"`** — `%` is percentage of duration,
  not seconds; a 335s source with offset=33 would scan 110s of file.

## Workflow

### Phase 1 — Parse OTIO, find top-most visible clips WITH source ranges

Identical top-most logic to the `re-download-otio` skill, **plus** the per-clip
`source_range`:

- For each track (`track_idx` = enumerate order), skip `not track.enabled`.
  `is_audio = track.kind == Audio`.
- **Position accumulator** sums **every** `clip.duration()` — gaps AND disabled
  clips both occupy timeline space. (Getting this wrong shifts every position
  downstream.)
- **Top-most selection** at each segment start: `covering` = clips where
  `start <= t < end` AND not Gap AND not Audio AND `enabled`. The top-most
  = `max(covering, key=track_idx)`. Note: gaps/disabled still advance the
  position accumulator above; they're just never the top-most visible clip.
- Resolve filename: `MissingReference` → `clip.name`; `ExternalReference` →
  `basename(target_url)` or `clip.name`. Parse
  `^(-?[A-Za-z0-9_-]{11})_(?:1080p_)?(\d+)_(\d+)\.mp4$` → `(vid, ss, ee)`. The
  `{11}` is YouTube's video ID length. Skip non-matches (images, stock, etc.).
- `offset_sec = source_range.start_time.value / RATE`,
  `duration_sec = source_range.duration.value / RATE`.

### Phase 2 — Deduplicate by SOURCE IDENTITY INCLUDING OFFSET

**CRITICAL:** dedup key is `(vid, ss, ee, round(offset_sec, 3))` — **NOT** `(vid, ss, ee)`.

The same HQ file is frequently referenced at **different offsets** on different tracks /
moments; those are genuinely different cuts and must each be exported. Deduping on
`(vid, ss, ee)` alone collapses them and silently drops most of the timeline (symptom: a
27-min timeline exports as ~9 min). Keep the first occurrence in `seg_start` order so
numbering follows the timeline.

`round(offset_sec, 3)` rounds to ~33 ms (one frame at 30 fps). Coarser than
that merges distinct cuts; finer is unnecessary — same physical frame will
produce the same export either way.

### Phase 3 — Pre-check HQ sources (ffprobe)

`precheck_sources()` classifies each distinct source file:

- **corrupt**: `probe_duration()` returns `None` (unreadable — e.g. missing moov atom). Drop
  all its clips.
- **truncated**: container duration < 95% of arc length `(ee - ss)`. **INFORMATIONAL ONLY —
  do NOT blanket-exclude.** A source shorter than its arc still yields valid cuts for every
  clip whose `offset + duration` lands inside the real footage.
- **video-short**: the video stream ends well before the audio stream
  (`probe_video_duration() < 0.95 * probe_duration()`). Clips whose `offset` is past the
  video end must be cut audio-only (ffmpeg would otherwise silently drop video).
- **out-of-bounds (per-clip)**: `offset + duration > actual_probed_duration + 0.05`. Drop
  just that clip (footage doesn't exist).

The only reasons a top-most clip is dropped: missing HQ file, `duration<=0`, corrupt source,
or per-clip out-of-bounds. Everything else is exported.

### Phase 4 — Cut each clip (frame-accurate re-encode)

The production `cut_clip()` uses **hybrid seek** (lesson #12) with **`-frames:v N`**
not `-t duration` (lesson #11):

```bash
# Production cut_clip() — Python pseudocode
n_frames = round(duration * src_fps)
cmd = [
  'ffmpeg', '-y',
  '-ss', f'{prev_kf}',         # fast keyframe seek BEFORE -i
  '-i', src,
  '-ss', f'{fine_seek}',       # frame-accurate over one GOP AFTER -i
  '-frames:v', str(n_frames),  # exact frame count, NOT -t duration
  '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
  '-c:a', 'aac', '-b:a', '192k',
  '-frames:a', str(round(duration * 44100)),
  '-movflags', '+faststart',
  '-loglevel', 'error', dst,
]
```

> ⚠️ **Don't copy the legacy template.** A naive `ffmpeg -ss <offset> -i <src>
> -t <duration>` exhibits BOTH bugs lessons #10 and #11 warn about — keyframe
> seek snap AND -t counting from the adjusted start. Always use the production
> cut_clip() flow.

Fallbacks (all handled in `main()`):
- **`force_silent`**: if `offset >= video_stream_duration`, cut with `-an` up front (no point
  attempting audio+video; video would be dropped anyway).
- **silent-video detection**: after a normal cut, `has_video_stream(dst)` is checked. If
  ffmpeg silently produced an audio-only file (source video too short), re-cut with `-an`.
- **AAC error 733**: some sources have mismatched A/V past the cut point; if stderr contains
  `733`, re-cut with `-an`.

Outputs `clip_0001.mp4 ... clip_NNNN.mp4` in timeline order, and `manifest.csv`.

### Phase 5 — Manifest

`manifest.csv` columns:
```
index, output, source_file, video_id, arc_start, arc_end,
offset_sec, duration_sec, actual_dur, status
```
`status` ∈ {`ok`, `skip`, `fail`}. Skip rows (file already exists with matching duration on
re-run) record the probed `actual_dur` too, so the manifest total stays meaningful.

## Recovering Bad Sources (when clips are being dropped)

**Decision tree:**

1. **Run `--dry-run` first.** It prints corrupt / truncated / video-short /
   out-of-bounds sources and skips actual cutting.
2. **Corrupt sources** → delete + re-download the exact section (commands below).
3. **Truncated sources** → leave alone UNLESS the per-clip out-of-bounds check is
   dropping too many (lesson #2). The clip-level check is the source of truth.
4. **After recovering / fixing**, re-run without `--dry-run` and proceed to
   **Black Replacement Workflow** to audit and clean any remaining bad_start /
   bad_end.

```bash
python scripts/capcut-export.py --project <project_dir> --dry-run   # first
python scripts/capcut-export.py --project <project_dir>             # then cut
```

For each **genuinely broken** source, delete it and re-download the exact section:

```bash
# HI first, fall back to LO — same format strings as re-download-otio
FMT_HI="bestvideo[ext=mp4][height>=720]+bestaudio[ext=m4a]"
FMT_LO="bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio[ext=m4a]/best"
yt-dlp -f "$FMT_HI" --download-sections "*${ss}-${ee}" --merge-output-format mp4 \
  --no-part -o "${vid}_1080p_${ss}_${ee}.mp4" --cookies <cookies> \
  "https://www.youtube.com/watch?v=${vid}"
```

**What re-download can and cannot fix:**
- ✅ **Corrupt / partial downloads** (tiny file, unreadable, truncated video stream) — re-fetch
  gets the real full file.
- ❌ **Source-truncated** (the YouTube video is genuinely shorter than the requested arc) —
  re-fetching returns the same short duration. That footage does not exist. The per-clip
  out-of-bounds check correctly drops only the cuts that fall past the real end.

**After recovering sources, do a FRESH export (clear the out dir first).** Recovered sources
put previously-filtered clips back into the ready list **in timeline order**, which shifts the
positional `clip_NNNN` numbering of everything after them. A clean re-cut guarantees
contiguous numbering and a complete manifest:
```bash
rm -f <out>/clip_*.mp4 <out>/manifest.csv
python scripts/capcut-export.py --project <project_dir>
```

## Black Replacement Workflow

When the 5-agent audit flags bad clips (start/end PSNR below threshold), replace
them with exact-duration black video. The audit identifies *which* clips are
wrong; this workflow removes the corruption so the timeline is drag-safe.

### Step 1 — Run the audit

The audit script lives at `scripts/capcut-audit.py` (originally prototyped at
`C:\Users\daves\AppData\Local\Temp\capcut_audit\audit.py`):

```bash
python scripts/capcut-audit.py
```

Tunable in the script:
- `THRESHOLD_DB = 18.0` — below = mismatch
- `SWEEP_STARTS = [0.0, 0.5, -0.5, 1.0, -1.0]` — sample offsets to rule out near-miss

Writes `<out>/audit_bad_clips.csv` with columns `idx, src, offset, dur,
start_psnr, end_psnr, reason`. Only `bad_start` / `bad_end` rows are candidates
for blacking; other statuses (`no_source`, `no_export`, `*_unreadable`)
indicate a deeper problem to diagnose first.

### Step 2 — Diagnose before replacing

Spot-check 3-5 bad_end rows:
- **start_PSNR ≥ 40 dB, end_PSNR 6-12 dB** → **content mismatch** (scene change
  within the cut window). Black it.
- **start_PSNR < 18 dB** → **seek bug**, not content. Re-cut that single clip
  with pure `-ss` after -i (no hybrid). Re-audit. If it still fails, black.

### Step 3 — Replace with black

`scripts/capcut-replace-black.py` (originally at
`C:\Users\daves\AppData\Local\Temp\capcut_audit\replace_black.py`):

```bash
python scripts/capcut-replace-black.py
```

Per row in `audit_bad_clips.csv`:
- Read `duration_sec` from manifest (the OTIO contract; **never** the probed
  `actual_dur`, which is just the file's measured length)
- Generate black: `ffmpeg -f lavfi -i color=c=black:s=1920x1080:r=30 -t D
  -c:v libx264 -crf 18 -preset fast -c:a aac -b:a 192k -shortest
  -movflags +faststart clip_NNNN.mp4`
- Overwrite in place; update manifest `status=black`, `actual_dur=<probed>`

### Step 4 — Verify post-replacement (the manifest-overwrite trap)

**Gotcha:** re-running `capcut-replace-black.py` overwrites *every* row in
`audit_bad_clips.csv`, including clips that were already re-cut between audit
runs and now pass. After replacing, always:
1. Re-run the audit on the same out dir
2. Any clip now passing must be re-cut (back to `status=ok`, not black)
3. Update manifest accordingly

Fransisco: 52 black replacements overwrote 3 already-fixed clips (29, 41, 183),
requiring a re-cut to restore them to `status=ok`. Final state: 382 ok + 49
black (not 379 + 52).

## Validation (5-agent timing audit)

When the user asks to "validate the timings", launch **5 independent agents in parallel**
(single message, 5 `Agent` calls). Each must **re-parse the OTIO independently** — do not
trust the manifest — so systematic errors surface:

1. **Clip length** — ffprobe **all** 429-ish clips; compare actual vs OTIO `duration_sec`.
   Pass = within one frame (~50 ms).
2. **Start point** — sample ~30 clips; extract exported first frame and source frame at
   `offset`; compare with `ffmpeg -lavfi psnr`. Pass = PSNR ≥ 25 dB (genuine match
   usually ≥ 40 dB; 25-30 dB is "matches but motion-blurred").
3. **End point** — sample ~30; compare exported last frame vs source at `offset+duration`.
   Extract the last frame with **`-sseof -0.1 -update 1`** (fast `-ss D-0.04` can land past
   the last keyframe and yield "No filtered frames").
4. **Ordering & completeness** — re-derive the top-most list; verify files are contiguous
   `clip_0001..NNNN`, monotonic in timeline `seg_start`, no dups, and each `clip_NNNN` matches
   the i-th derived entry (source/offset/duration).
5. **End-to-end totals** — reconcile Σ(OTIO duration) vs Σ(manifest duration) vs
   Σ(manifest actual_dur); spot-probe ~12 files; confirm recovered sources present and not
   dropped; confirm dedup (0 identical `(source, offset)` pairs; N sources legitimately reused
   at multiple offsets).

**PSNR false-positive guard (agents 2 & 3):** fast-motion footage lowers PSNR and the exact
edge frame is ±1 frame sensitive. For any low score, re-sample the source at
`t ± {0.033, 0.066}` and take the best match; only flag a genuine error if **no** timestamp
within ±2 frames matches ≥ 25 dB.

> **Use `scripts/capcut-audit.py` for the bulk audit.** The 5-agent workflow above
> is for *re-parsing the OTIO independently* (sanity check on the manifest
> generation logic). For a wall-clock-fast batch audit of an existing export,
> run `capcut-audit.py` directly — it does the same PSNR + sweep checks in
> parallel against the manifest, ~5 min for 430 clips.
> See [Black Replacement Workflow](#black-replacement-workflow) for cleanup.

## Key File Locations

| File | Location |
|------|----------|
| Input OTIO | `<project>/downloadplease.otio` (or `_hq.otio`) |
| HQ sources | `<project>/v/matcher-hq/*_1080p_*.mp4` |
| Output clips | `<project>/capcut_import/clip_NNNN.mp4` |
| Manifest | `<project>/capcut_import/manifest.csv` |
| Export script | `scripts/capcut-export.py` |
| Audit script | `scripts/capcut-audit.py` |
| Replace-black script | `scripts/capcut-replace-black.py` |
| Bad-clip list | `<out>/audit_bad_clips.csv` (written by audit script) |

## Lessons Learned (Fransisco / Cantinflas — Jul 2026)

1. **Dedup MUST include offset.** `(vid, ss, ee)` collapsed distinct cuts of the same source →
   27-min timeline exported as 8.9 min. Fix: `(vid, ss, ee, round(offset_sec, 3))`.
2. **Truncation is not a reason to drop a whole source.** Blanket-excluding every clip from any
   source shorter than its arc threw away ~57 perfectly cuttable clips. Only the per-clip
   out-of-bounds check (against the **actual probed** duration) should drop cuts.
3. **ffmpeg silently drops video** when the cut offset is at/after the video stream's end (the
   audio stream keeps going). Always verify with `has_video_stream()` after cutting and re-cut
   `-an` when needed; pre-empt with `force_silent` when `offset >= video_dur`.
4. **AAC error 733** on A/V-mismatched sources → retry the cut with `-an`.
5. **Windows path normalization.** `os.path.normpath()` the `--hq-cache`/`--out` paths;
   mixed slashes from `os.path.join` made the glob find **0** HQ files.
6. **Regex must accept both forms**: `VIDEOID_ss_ee.mp4` and `VIDEOID_1080p_ss_ee.mp4`
   (`(?:1080p_)?`). Video IDs can start with `-`.
7. **Recovering sources renumbers the export** (filtered clips rejoin in timeline order). Always
   clear the out dir and re-cut fresh after re-downloading, so numbering + manifest are clean.
8. **Position accumulator includes gaps and disabled clips** — they occupy timeline space even
   though they're never the top-most visible clip.
9. Re-encode drift is negligible: ~2.6 s across 429 cuts / 26.6 min, every clip within ~40 ms
   of its exact OTIO duration.

10. **`-ss` BEFORE `-i` = keyframe seek (silent corruption).** ffmpeg's `-ss` placed before
    `-i` snaps to the nearest keyframe BEFORE the requested offset — can land many frames
    off, silently cutting from the wrong scene. In the Fransisco audit this produced 17
    clips whose first frame was from a different scene (PSNR 7-8 dB). **Always use `-ss`
    AFTER `-i` for frame-accurate seek.** Counter-intuitively slower (decodes from start
    or previous keyframe forward) but correct.
11. **`-t duration` misses the last frame.** When `-ss` is after `-i`, ffmpeg snaps the
    output start pts to the next frame boundary (~1 frame late), and `-t duration` counts
    from that adjusted start — so the actual output is ~1 frame short. **Use
    `-frames:v round(duration * src_fps)`** (with `-frames:a round(duration * 44100)` for
    audio) to get exactly the OTIO contract's number of frames. Drift becomes ±1/fps
    (33ms at 30fps, 40ms at 25fps) instead of missing the last frame entirely.
12. **Hybrid seek (fast + frame-accurate) speedup.** To make frame-accurate cuts faster
    on sources with large offsets, use the ffmpeg hybrid: `-ss prev_kf` BEFORE `-i` (fast
    keyframe seek) followed by `-ss fine_seek` AFTER `-i` (frame-accurate over one GOP).
    Compute `prev_kf` from the container's packet index (`ffprobe -show_entries
    packet=pts_time,flags`, filter `K__`), cached per source. This is **~2x faster**
    than pure `-ss` after `-i` on the Fransisco project (13 min vs 25-30 min for 431 clips).
    **Caveat**: the second `-ss` occasionally shifts the output start by 1 frame on
    certain keyframe positions, causing 1-3 clips to fail start-PSNR. Verify with the
    audit and re-cut any bad_start with pure `-ss` after `-i` as a targeted fix.
    **Never use `-read_intervals "%${offset}"` to find prev_kf** — `%` is a percentage of
    duration, not seconds, and a 335s file with offset=33 would scan 110s of file.
13. **Audit reference extraction must also use `-ss` after `-i`.** When checking whether an
    export's first/last frame matches the source, the SOURCE-side reference extraction has
    the same keyframe-seek bug. Use `-i src -ss offset -vframes 1` (not `-ss offset -i src
    -vframes 1`) for the audit's source frames too. Otherwise the audit reports false
    positives at the end (saw 75 → real 58 with the fix).
14. **Some bad_end clips are content, not timing.** When start PSNR is 40+ dB (cut starts
    correctly) but end PSNR is 6-12 dB, the source has a scene change within the cut
    window — the export's last frame is from a different scene than the source frame at
    `offset+duration`. These can't be fixed by seek accuracy; they reflect a real
    source/timeline mismatch at the cut point. Per user policy, replace with black video
    of exact duration (`ffmpeg -f lavfi -i color=c=black:s=1920x1080:r=30 -t D -c:v
    libx264 -crf 18 -c:a aac -b:a 192k`).
15. **`replace_black.py` is destructive — verify after every run.** See
    [Black Replacement Workflow → Step 4](#step-4--verify-post-replacement-the-manifest-overwrite-trap).
    Fransisco: 52 black replacements overwrote 3 already-fixed clips (29, 41, 183);
    re-cut restored them to `status=ok` (final 382 ok + 49 black, not 379 + 52).
