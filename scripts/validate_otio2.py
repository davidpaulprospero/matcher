#!/usr/bin/env python3
"""Targeted OTIO checks."""
import opentimelineio as otio
from pathlib import Path

BASE = Path("E:/Edit Job/Primise/21-8/2-china-driverless-cars")
SRC = BASE / "2-china-driverless-cars_hq.otio"
FLAT = BASE / "2-china-driverless-cars_hq_flat.otio"
VIS = BASE / "2-china-driverless-cars_hq_visible.otio"

def file_url(c):
    mr = c.media_reference
    if mr is None or not hasattr(mr, "target_url"):
        return None
    return mr.target_url

src = otio.adapters.read_from_file(str(SRC))
flat = otio.adapters.read_from_file(str(FLAT))
vis  = otio.adapters.read_from_file(str(VIS))

# --- Check 1: flat V1 is only video, audio separate ---
print("=== FLAT structure ===")
for t in flat.tracks:
    print(f"  {t.name} kind={t.kind} enabled={t.enabled}")

# --- Check 2: tl=13696..13743 region in flat V1 ---
print("\n=== Flat V1 around tl=13696..13743 ===")
v1_flat = flat.tracks[1]  # V1 - Flattened
tl_in = otio.opentime.RationalTime(13696, 24)
tl_out = otio.opentime.RationalTime(13744, 24)
cursor = otio.opentime.RationalTime(0, 24)
found = []
for c in v1_flat:
    dur = c.trimmed_range().duration
    if (cursor + dur) > tl_in and cursor < tl_out:
        url = file_url(c)
        name = Path(url).name if url else "(no url)"
        is_gap = isinstance(c, otio.schema.Gap)
        print(f"  tl={cursor.value}..{(cursor+dur).value} kind={'GAP' if is_gap else 'CLIP'} url={name}")
    cursor = cursor + dur
    if cursor.value > tl_out.value + 100:
        break

# --- Check 3: Visible - track names preserved, disabled consistency ---
print("\n=== VISIBLE track names ===")
for t in vis.tracks:
    print(f"  {t.name} kind={t.kind}")

# Compare V1 visible choices vs flat choices for disabled-clip consistency
print("\n=== DISABLED CLIP CHECK ===")
# Walk source timeline at the same tl positions as flat and visible.
# For each flat clip, find its source tidx (look at metadata) and verify it matches visible V1.
src_v_tracks = [t for t in src.tracks if t.kind == "Video"]
flat_v1 = flat.tracks[1]

# Inspect metadata of flat clips
print("First 5 flat V1 clips metadata:")
for i, c in enumerate(list(flat_v1)[:5]):
    print(f"  [{i}] metadata={c.metadata}")

# Check for 'tidx' metadata field commonly used by OTIO export
sample = list(flat_v1)[0] if len(list(flat_v1)) else None
if sample is not None:
    print(f"\nFlat V1 first clip metadata keys: {list(sample.metadata.keys())}")
    print(f"  metadata: {sample.metadata}")
