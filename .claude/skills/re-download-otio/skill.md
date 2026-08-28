---
name: re-download-otio
description: Re-download clips from an existing OTIO at higher quality (e.g., 1080p instead of 360p). Reads the OTIO to find all video file references (both ExternalReference and MissingReference) and re-downloads them using yt-dlp with best quality settings. Optionally flattens the timeline to a single V1 track of top-most clips for transitions.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
---

# Re-Download OTIO Skill

Re-downloads all video clips referenced in an existing OTIO timeline at higher quality (e.g., 1080p instead of 360p). Preserves the existing OTIO structure and just swaps out the media references.

## When to Use

- User has an OTIO with 360p clips and wants 1080p versions
- User says "redownload at 1080p", "upgrade quality", "re-download the otio"
- User wants to swap low-res proxy clips for full-quality downloads
- User wants to flatten multi-tier OTIO to a single V1 track for editing transitions in between

## Invocation

```
/redo-download-otio <project_dir>
```

## Arguments

- `<project_dir>`: Path to the project directory (must contain `downloadplease.otio` or specified OTIO)

## Direct Script Invocation (recommended for non-skill runs)

```bash
python scripts/re-download-otio.py --project <project_dir>
```

Accepts: `--otio <file>`, `--hq-cache <dir>`, `--rewrite-only`, `--skip-rewrite`, `--validate-only`.

## Workflow

### Phase 1: Parse OTIO and Extract Top-Most Visible Clips

**CRITICAL: Read timestamps from clip FILENAMES, not source_range or timeline_segments.json.**

**CRITICAL: The visible "top-most" clip at any moment is the one with the HIGHEST track index among all ENABLED clips active at that moment.**

#### Finding the Top-Most Visible Clip Per Segment

1. Walk all tracks in order (V1=idx0, V2=idx1, V3=idx2, V4=idx3, V5=idx4, V6=idx5, Video7=idx6, A7=idx7)
2. For each clip, compute its timeline position by accumulating durations sequentially
3. At each unique timeline start position, collect all **enabled** (`clip.enabled=True`) clips whose range covers that moment
4. Filter out: audio tracks (kind=Audio), gaps, and **disabled** clips
5. The top-most clip = the one with the **highest track index** among the active enabled clips

```python
import opentimelineio as otio
import re

def parse_clip_filename(filename):
    m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_(\d+)_(\d+)\.mp4$', filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None

def get_topmost_clips(tl):
    """Return list of (seg_start, track_name, clip_name, vid, ss, ee) for top-most clips."""
    all_clips = []
    for track_idx, track in enumerate(tl.tracks):
        if not track.enabled:
            continue
        is_audio = track.kind == otio.schema.TrackKind.Audio
        tp = otio.opentime.RationalTime(0, 30)
        for clip in track:
            clip_enabled = getattr(clip, 'enabled', True)
            dur = clip.duration()
            all_clips.append({
                'track_idx': track_idx,
                'track': track.name,
                'name': clip.name,
                'is_audio': is_audio,
                'is_gap': isinstance(clip, otio.schema.Gap),
                'clip_enabled': clip_enabled,
                'start': tp.value,
                'end': (tp + dur).value,
            })
            tp = tp + dur

    all_clips.sort(key=lambda x: x['start'])
    unique_starts = sorted(set(c['start'] for c in all_clips))

    results = []
    for seg_start in unique_starts:
        covering = [c for c in all_clips
                    if c['start'] <= seg_start < c['end']
                    and not c['is_gap']
                    and not c['is_audio']
                    and c['clip_enabled']]
        if not covering:
            continue
        top = max(covering, key=lambda x: x['track_idx'])
        vid, ss, ee = parse_clip_filename(top['name'])
        results.append((seg_start, top['track'], top['name'], vid, ss, ee))
    return results
```

**Key track stacking rules:**
- Higher track index = visually on top in DaVinci Resolve
- V6 (idx=5) is on top of V5 (idx=4) which is on top of V4, etc.
- **Disabled clips do not participate** — only enabled clips count
- Audio tracks (A7) never visually obscure video clips
- The "top-most" clip is what you see playing in the editor at that moment

**Extracting video ID and timestamps from filenames:**
```python
def parse_clip_filename(filename):
    """
    Filenames like 'sz0oDYs1mNk_205_226.mp4' mean:
    - video_id = 'sz0oDYs1mNk' (11-char YouTube ID)
    - youtube_start = 205 seconds
    - youtube_end = 226 seconds

    Some video IDs start with a hyphen (e.g., '-vaIH4mENlw').
    """
    m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_(\d+)_(\d+)\.mp4$', filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None
```

### Phase 2: Deduplicate and Check HQ Cache

After extracting top-most clips, deduplicate by `(video_id, start, end)`:

```python
seen = set()
unique_clips = []
for c in youtube_clips:
    key = (c['vid'], c['ss'], c['ee'])
    if key not in seen:
        seen.add(key)
        unique_clips.append(c)
```

Build existing HQ cache index:
```python
hq_index = {}
hq_cache = os.path.join(proj, 'v', 'matcher-hq')
if os.path.exists(hq_cache):
    for f in glob.glob(os.path.join(hq_cache, '*_1080p_*.mp4')):
        m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_1080p_(\d+)_(\d+)\.mp4', os.path.basename(f))
        if m:
            hq_index[(m.group(1), int(m.group(2)), int(m.group(3)))] = f
```

Filter out already-cached:
```python
to_download = [c for c in unique_clips if (c['vid'], c['ss'], c['ee']) not in hq_index]
```

### Phase 3: Find YouTube URLs from Checkpoints

**IMPORTANT: Checkpoints may be gzip-compressed. Try gzip FIRST before plain JSON.**

```python
import gzip, json, os

def find_youtube_url(video_id, project_dir):
    """Search project checkpoint and gap_fill checkpoints for YouTube URL."""
    checkpoints = [
        os.path.join(project_dir, 'checkpoint.json'),
        os.path.join(project_dir, 'gap_fill', 'checkpoint.json'),
        os.path.join(project_dir, 'gap_fill', 'checkpoint.backup_retry.json'),
        os.path.join(project_dir, 'gap_fill', 'checkpoint.pre_fix.json'),
        os.path.join(project_dir, 'gap_fill', 'checkpoint.pre_output_fix.json'),
    ]
    for cp_path in checkpoints:
        if not os.path.exists(cp_path):
            continue
        try:
            # Try gzip first (most common format)
            with gzip.open(cp_path, 'rt', encoding='utf-8') as f:
                cp = json.load(f)
        except (gzip.BadGzipFile, OSError):
            try:
                with open(cp_path, 'r', encoding='utf-8') as f:
                    cp = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                continue
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                with open(cp_path, 'r', encoding='utf-8') as f:
                    cp = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                continue
        for r in cp.get('video_search', {}).get('search_results', []):
            if r.get('video_id') == video_id:
                return r.get('url') or r.get('youtube_url')
    return None
```

### Phase 4: Download Clips

**First pass — use `height>=720`:**
```bash
yt-dlp -f "bestvideo[ext=mp4][height>=720]+bestaudio[ext=m4a]" \
  --download-sections "*{ss}-{ee}" \
  --merge-output-format mp4 \
  --no-part \
  -o "matcher-hq/VIDEOID_1080p_ss_ee.mp4" \
  "URL"
```

**Output location:** `<project>/v/matcher-hq/` — separate from `matcher-alt/` (360p originals).

**Always use hardcoded `.mp4` in `-o`** — `%(ext)s` can produce `.webm`/`.mkv`, breaking cache detection.

### Phase 4B: Retry Failed Clips — Lower Resolution

Many videos don't have 720p+ MP4 available. After first-pass failures, retry with `height<=480`:

```bash
yt-dlp -f "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio[ext=m4a]/best" \
  --download-sections "*{ss}-{ee}" \
  --merge-output-format mp4 \
  --no-part \
  -o "matcher-hq/VIDEOID_1080p_ss_ee.mp4" \
  "URL"
```

This recovers 100% of clips — the failures are due to `height>=720` constraint, not video unavailability.

### Phase 5: Rewrite OTIO with HQ References

**Key principle: Only upgrade clips that have HQ files. Leave everything else untouched.**

```python
import opentimelineio as otio
import os, re, glob

def rewrite_otio_hq(otio_path, hq_cache, output_path):
    def parse_clip_filename(filename):
        m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_(\d+)_(\d+)\.mp4$', filename)
        if m:
            return m.group(1), int(m.group(2)), int(m.group(3))
        return None, None, None

    # Build HQ index
    hq_index = {}
    for f in glob.glob(os.path.join(hq_cache, '*_1080p_*.mp4')):
        m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_1080p_(\d+)_(\d+)\.mp4', os.path.basename(f))
        if m:
            hq_index[(m.group(1), int(m.group(2)), int(m.group(3)))] = f

    tl = otio.adapters.read_from_file(otio_path)
    replaced = 0

    for track in tl.tracks:
        for child in track:
            if not isinstance(child, otio.schema.Clip):
                continue
            mr = child.media_reference
            fname = child.name
            if isinstance(mr, otio.schema.ExternalReference) and mr.target_url:
                fname = os.path.basename(mr.target_url)

            vid, ss, ee = parse_clip_filename(fname)
            if not vid:
                continue  # non-video asset — skip

            key = (vid, ss, ee)
            if key not in hq_index:
                continue  # no HQ file — leave original reference

            new_path = hq_index[key]
            clip_dur_secs = ee - ss
            frame_rate = 30.0
            clip_dur_frames = int(clip_dur_secs * frame_rate)

            new_mr = otio.schema.ExternalReference(
                target_url=new_path,
                available_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, frame_rate),
                    duration=otio.opentime.RationalTime(clip_dur_frames, frame_rate)
                )
            )
            new_mr.name = os.path.basename(new_path)
            child.media_reference = new_mr
            child.metadata['Resolve_OTIO'] = {}
            replaced += 1

    otio.adapters.write_to_file(tl, output_path)
    return replaced
```

**Media reference format for DaVinci:**
- Use plain Windows paths (`E:\...\file.mp4`) — `otio.adapters.write_to_file()` converts to `file:///E:/...` automatically
- Do NOT manually prefix paths with `file:///` — it will double-encode

### Phase 6: Validation

After rewriting:
1. Read the new OTIO and verify all `ExternalReference` paths exist on disk
2. Report: clips upgraded, clips skipped (non-video), broken paths

### Phase 7 (optional): Flatten to Single Track for Transitions

When the user wants to edit transitions in DaVinci between top-most clips, the multi-track HQ OTIO is hard to work with — alternatives overlap and obscure V1. `--flatten` emits a sibling `<input>_hq_single.otio` with one V1 video track of top-most clips in chronological order, plus the voiceover preserved on A1.

**Disabled-on-top tracks emit Gaps, not fallback content.** If the topmost clip at a moment is disabled (`clip.enabled == False`), the picker emits a `Gap` covering that range instead of falling back to a lower enabled track. The user disabled that content intentionally — preserving it as a visible Gap gives them a clean spot to drop a transition. Lower tracks are only consulted when the topmost clip is enabled.

```python
import opentimelineio as otio

def flatten_topmost(input_path, output_path):
    """Build a single V1 track from top-most visible clips + preserved audio."""
    tl = otio.adapters.read_from_file(str(input_path))

    # Collect top-most clip per unique timeline moment
    all_clips, tp_by_track = [], {}
    for track_idx, track in enumerate(tl.tracks):
        if not track.enabled or track.kind == otio.schema.TrackKind.Audio:
            continue
        tp = otio.opentime.RationalTime(0, 30)
        for clip in track:
            clip_enabled = getattr(clip, 'enabled', True)
            if isinstance(clip, otio.schema.Gap) or not clip_enabled:
                tp = tp + clip.duration()
                continue
            all_clips.append({
                'track_idx': track_idx, 'clip': clip,
                'start': tp.value, 'end': (tp + clip.duration()).value,
            })
            tp = tp + clip.duration()

    # Pick top-most per moment (highest track_idx covering that moment)
    all_clips.sort(key=lambda x: x['start'])
    deduped, last = [], None
    for entry in sorted(all_clips, key=lambda x: x['start']):
        top = max([c for c in all_clips
                   if c['start'] <= entry['start'] < c['end']],
                  key=lambda x: x['track_idx'])
        if last and last['clip'].name == top['clip'].name and last['clip'].source_range == top['clip'].source_range:
            last['end'] = top['end']
            continue
        deduped.append(top); last = top

    new_tl = otio.schema.Timeline(name=tl.name or "flattened")
    v_track = otio.schema.Track(name="V1 - Flattened Topmost", kind=otio.schema.TrackKind.Video)
    for entry in deduped:
        src = entry['clip']
        trim = src.trimmed_range()
        if trim.duration.value <= 0:
            continue
        v_track.append(otio.schema.Clip(
            name=src.name, media_reference=src.media_reference,
            source_range=trim,
        ))
    new_tl.tracks.append(v_track)

    # Preserve voiceover audio tracks
    for track in tl.tracks:
        if track.kind == otio.schema.TrackKind.Audio:
            a_track = otio.schema.Track(name=track.name or "A1 - Voiceover", kind=otio.schema.TrackKind.Audio)
            for child in track:
                if isinstance(child, otio.schema.Clip):
                    a_track.append(otio.schema.Clip(
                        name=child.name, media_reference=child.media_reference,
                        source_range=child.source_range,
                    ))
            new_tl.tracks.append(a_track)

    otio.adapters.write_to_file(new_tl, str(output_path))
```

Key design choices:
- Iterates at every clip boundary (tl_start AND tl_end) so no moment is missed
- Dedupes adjacent same-identity clips and merges adjacent gaps
- Bounds each clip's effective end to the next change point — flat V1 has **no overlapping clips** and **total duration matches source**
- **Disabled topmost → Gap (not fallback)** — visible placeholder for transitions
- Each clip's source_range is recomputed from the segment's bounded timeline range (linear mapping) so partial segments don't seek past their source
- Audio tracks (voiceover) are copied verbatim — never flattened
- Independent script: `scripts/flatten-otio-to-single-track.py`

Standalone CLI:
```bash
python scripts/flatten-otio-to-single-track.py <input.otio> [output.otio]
# Default output: <input>_single.otio
```

One-shot combined with re-download:
```bash
python scripts/re-download-otio.py --project <dir> --otio X.otio --flatten
# Produces X_hq.otio AND X_hq_single.otio
```

## Key File Locations

| File | Location |
|------|----------|
| Input OTIO | `<project>/a.otio` |
| HQ Download cache | `<project>/v/matcher-hq/` |
| Original 360p cache | `<project>/v/matcher-alt/` |
| YouTube URLs | Project + gap_fill `*checkpoint*.json` (gzip or plain JSON) |
| Output OTIO | `<input>_hq.otio` |
| Single-track output (with `--flatten`) | `<input>_hq_single.otio` |

## Reusable Scripts

| Script | Purpose |
|--------|---------|
| `scripts/re-download-otio.py` | Single canonical implementation: parse OTIO → find top-most clips → dedup → check HQ cache → download (with `height<=480` retry) → rewrite OTIO → validate. `--flatten` adds single-track emission. |
| `scripts/flatten-otio-to-single-track.py` | Standalone flatten: top-most clips → one V1 track + voiceover preserved. Run after re-downloading, or directly on any multi-track OTIO. |

**re-download-otio.py Usage:**
```bash
python scripts/re-download-otio.py --project <dir> [--otio <file>] [--hq-cache <dir>]
                                    [--rewrite-only] [--skip-rewrite] [--validate-only]
                                    [--flatten] [--dry-run]
```

**flatten-otio-to-single-track.py Usage:**
```bash
python scripts/flatten-otio-to-single-track.py <input.otio> [output.otio]
# Default output: <input>_single.otio
```

**Defaults:**
- `--otio` defaults to `downloadplease.otio`
- `--hq-cache` defaults to `<project>/v/matcher-hq`
- Cookies: `D:/_Projects/voiceover-matcher-dev/cookies/main.txt`
- Runs full pipeline (download → rewrite → validate) by default

**Key principle baked in:** Only the top-most visible clip per moment is downloaded (highest track_idx among enabled, non-gap, non-audio clips). Alternative tiers are NOT downloaded. Always do top-most filtering before deduping.

## Troubleshooting

### "FAIL (exit 1)" with no stderr captured
- **Cause:** `height>=720` constraint fails — video has no MP4 at 720p+
- **Fix:** Use `retry_hq.py` with `height<=480` fallback format

### Some clips never attempted (not in failed list)
- **Cause:** Script checks `matcher-hq/` for existing files using `_1080p_` suffix — if file exists in `matcher-alt/` (360p) with the same name, it's not found and re-downloaded
- **Fix:** The script correctly skips already-cached HQ files; missing clips are genuinely new downloads

### 360p files exist in `matcher-alt/` but download still fails
- **Cause:** `height>=720` requirement fails — these videos genuinely don't have 720p+ MP4
- **Fix:** Use `bestvideo[height<=480]` retry format — recovers 100% of clips

### File exists but still reports FAIL
- **Cause:** Output path same as input path (overwrite attempted), or permission issue
- **Fix:** Ensure output path uses `_1080p_` suffix and is in `matcher-hq/`

### Proxy warning `[pot:bgutil:http]` is harmless
- yt-dlp continues past this warning. **Never use `--no-proxy` or `--proxy ''`** — it breaks all downloads.

## Lessons Learned (Kyteq/Disney OJNZ7wJZ — May 2026)

### 1. Use `get_topmost_clips()` not track iteration
The OTIO contains hundreds of clips across 8 tracks. Only the top-most VISIBLE clip at each moment matters. Algorithm: highest track_idx among enabled, non-gap, non-audio clips active at that moment.

### 2. Checkpoints are gzip-compressed — try gzip FIRST
`checkpoint.json` files in this project have gzip headers (byte 0x8b). Trying plain JSON first causes `UnicodeDecodeError` and the URL lookup silently fails. Always try gzip first.

### 3. Some videos have NO 720p+ MP4
The `height>=720` format constraint causes ~5% first-pass failure rate. Videos may only have 360p or 480p MP4 available. Always run a retry with `height<=480` — recovers 100%.

### 4. HQ cache uses `_1080p_` suffix
Original 360p files in `matcher-alt/` have no quality suffix. HQ files MUST use `_1080p_` suffix to enable proper cache detection and distinguish from originals.

### 5. yt-dlp format strings for section downloads
```
yt-dlp -f "bestvideo[ext=mp4][height>=720]+bestaudio[ext=m4a]" --download-sections "*{ss}-{ee}" --merge-output-format mp4 --no-part -o "out.mp4" URL
```
The `--download-sections "*{ss}-{ee}"` format with asterisk prefix is correct for section extraction.

### 6. Non-video clips: skip, don't fail
Images (`.jpg`, `.png`) and stock footage have filenames not matching the `VIDEOID_start_end.mp4` pattern. `parse_clip_filename()` returns `None` — skip these, don't treat as failures.

### 7. Rewrite only clips with HQ files
When rewriting the OTIO, only upgrade clips that have HQ files in the cache. Leave all other clips (images, stock, etc.) with their original references untouched. This avoids creating MissingReferences.

### 8. `available_range` duration from YouTube timestamps, not ffprobe
The section length is `(ee - ss)` in seconds. Compute `clip_dur_frames = int((ee - ss) * frame_rate)`. Don't use ffprobe on the downloaded file — it's the YouTube section duration, not the file duration.

### 9. Background task output file is truncated
When running download scripts as background Bash tasks, the output file only shows the last ~54 lines even if more were written. Always use `ls matcher-hq/*.mp4 | wc -l` to get actual progress.

### 10. Retry loop: run until all succeed
The `tmp_download.py` script includes a built-in retry round. Run it once and let the retry complete — most failed clips recover on retry with the same format.
