#!/usr/bin/env python3
"""Verify enabled-flag consistency: flat picks should come from enabled source clips."""
import opentimelineio as otio
from pathlib import Path
from collections import defaultdict

BASE = Path("E:/Edit Job/Primise/21-8/2-china-driverless-cars")
SRC = BASE / "2-china-driverless-cars_hq.otio"
FLAT = BASE / "2-china-driverless-cars_hq_flat.otio"
VIS = BASE / "2-china-driverless-cars_hq_visible.otio"

src  = otio.adapters.read_from_file(str(SRC))
flat = otio.adapters.read_from_file(str(FLAT))
vis  = otio.adapters.read_from_file(str(VIS))

def url_name(c):
    mr = getattr(c, "media_reference", None)
    if mr is None or not hasattr(mr, "target_url"):
        return None
    return Path(mr.target_url).name if mr.target_url else None

# Build a per-tl index of (clip_name) for all source video tracks and which were enabled.
# Walk source tracks, mark disabled source clips.
src_clips_by_tl = []  # list of (start, end, name, enabled, track_idx)
for ti, tr in enumerate(src.tracks):
    if tr.kind != "Video":
        continue
    cur = otio.opentime.RationalTime(0, 24)
    for c in tr:
        dur = c.trimmed_range().duration
        s, e = cur.value, (cur + dur).value
        name = url_name(c)
        if not isinstance(c, otio.schema.Gap):
            src_clips_by_tl.append((s, e, name, c.enabled, ti))
        cur = cur + dur

# Build flat clips by tl
flat_clips_by_tl = []
v1_flat = flat.tracks[1]
cur = otio.opentime.RationalTime(0, 24)
for c in v1_flat:
    dur = c.trimmed_range().duration
    s, e = cur.value, (cur + dur).value
    if not isinstance(c, otio.schema.Gap):
        flat_clips_by_tl.append((s, e, url_name(c), c.enabled))
    cur = cur + dur

# For each flat clip, look up matching source clips in the same tl window across all video tracks
violations = []
covered = 0
for fs, fe, fname, fenabled in flat_clips_by_tl:
    if fname is None:
        continue
    candidates = [c for c in src_clips_by_tl if c[0] < fe and c[1] > fs and c[2] == fname]
    if not candidates:
        continue
    covered += 1
    # If ALL matching candidates in the source are disabled, that's a violation.
    # The flat exporter should prefer enabled clips and only fall back to disabled if no enabled alternative exists.
    if all(not c[3] for c in candidates):
        violations.append((fs, fe, fname, [c[4] for c in candidates]))

print(f"Flat clips matched to source: {covered}/{len(flat_clips_by_tl)}")
print(f"Flat clips sourced from disabled-only: {len(violations)}")
for v in violations[:20]:
    print(f"  {v}")

# Also: missing-reference counts
def count_missing(tl):
    n = 0
    for tr in tl.tracks:
        for c in tr:
            if isinstance(c, otio.schema.Clip):
                mr = getattr(c, "media_reference", None)
                if mr is None:
                    n += 1
                elif mr.__class__.__name__ == "MissingReference":
                    n += 1
    return n

print(f"\nMissingReference counts: source={count_missing(src)} flat={count_missing(flat)} visible={count_missing(vis)}")

# Source-range sanity: are start_times ever > 1e6 (absurd)?
absurd = 0
neg = 0
for tr in src.tracks:
    for c in tr:
        if isinstance(c, otio.schema.Clip):
            sr = c.source_range
            if sr is None: continue
            st = sr.start_time.value
            du = sr.duration.value
            if st < 0:
                neg += 1
            if st > 1e6:
                absurd += 1
print(f"Source: negative start_time count={neg}, absurdly large start_time count={absurd}")
