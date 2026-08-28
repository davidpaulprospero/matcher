#!/usr/bin/env python3
"""Inspect flat V1 around tl=13696..13744."""
import opentimelineio as otio
from pathlib import Path

BASE = Path("E:/Edit Job/Primise/21-8/2-china-driverless-cars")
FLAT = BASE / "2-china-driverless-cars_hq_flat.otio"
flat = otio.adapters.read_from_file(str(FLAT))
v1_flat = flat.tracks[1]

cursor = otio.opentime.RationalTime(0, 24)
print("Walking flat V1:")
for j, c in enumerate(v1_flat):
    dur = c.trimmed_range().duration
    start = cursor.value
    end = (cursor + dur).value
    # Print entries that are around or before/after the target region
    if end > 13400 and start < 14000:
        url = c.media_reference.target_url if (c.media_reference and hasattr(c.media_reference, "target_url")) else None
        name = Path(url).name if url else "(no url)"
        is_gap = isinstance(c, otio.schema.Gap)
        print(f"  [{j}] tl={start}..{end} ({'GAP' if is_gap else 'CLIP'}) name={name}")
    cursor = cursor + dur
print(f"\nTotal tl = {cursor.value}")
