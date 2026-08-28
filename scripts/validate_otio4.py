#!/usr/bin/env python3
"""Find source of the flat clip at tl=13696..13743 and verify V3 origin."""
import opentimelineio as otio
from pathlib import Path

BASE = Path("E:/Edit Job/Primise/21-8/2-china-driverless-cars")
SRC = BASE / "2-china-driverless-cars_hq.otio"
FLAT = BASE / "2-china-driverless-cars_hq_flat.otio"

src = otio.adapters.read_from_file(str(SRC))
flat = otio.adapters.read_from_file(str(FLAT))
v1_flat = flat.tracks[1]

# Reconstruct flat V1 timeline positions and look at each clip's source
cursor = otio.opentime.RationalTime(0, 24)
flat_clips_at_region = []
for j, c in enumerate(v1_flat):
    dur = c.trimmed_range().duration
    if (cursor + dur).value > 13696 and cursor.value < 13744:
        flat_clips_at_region.append((j, cursor.value, (cursor + dur).value, c))
    cursor = cursor + dur

print("Flat clips overlapping tl=13696..13744:")
for (j, s, e, c) in flat_clips_at_region:
    url = c.media_reference.target_url if hasattr(c.media_reference, "target_url") else None
    name = Path(url).name if url else "(no url)"
    print(f"  [{j}] tl={s}..{e} url={name}")
    print(f"      metadata = {c.metadata}")
    print(f"      source_range = {c.source_range}")

# Now find this clip in source tracks. Walk all source video tracks and find any
# child whose trimmed_range overlaps tl=13696..13744 AND whose media URL matches.
print("\nSource tracks - children overlapping tl=13696..13744:")
TARGET_NAME = "7uxQh4vg1Rc_1080p_114_121.mp4"
TARGET_NAME_SHORT = "7uxQh4vg1Rc_114_121.mp4"

for ti, track in enumerate(src.tracks):
    if track.kind != "Video":
        continue
    cur = otio.opentime.RationalTime(0, 24)
    for ci, c in enumerate(track):
        dur = c.trimmed_range().duration
        s, e = cur.value, (cur + dur).value
        if e > 13696 and s < 13744:
            url = c.media_reference.target_url if hasattr(c.media_reference, "target_url") else None
            name = Path(url).name if url else "(no url)"
            is_gap = isinstance(c, otio.schema.Gap)
            print(f"  track {ti} [{ci}] tl={s}..{e} ({'GAP' if is_gap else 'CLIP'}) name={name}")
        cur = cur + dur
