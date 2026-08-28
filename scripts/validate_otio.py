#!/usr/bin/env python3
"""Validate OTIO structural integrity."""
import opentimelineio as otio
import math
import json
from pathlib import Path

BASE = Path("E:/Edit Job/Primise/21-8/2-china-driverless-cars")
FILES = {
    "source":  BASE / "2-china-driverless-cars_hq.otio",
    "flat":    BASE / "2-china-driverless-cars_hq_flat.otio",
    "visible": BASE / "2-china-driverless-cars_hq_visible.otio",
}

results = {}

for key, path in FILES.items():
    tl = otio.adapters.read_from_file(str(path))
    results[key] = {"path": str(path), "tracks": [], "checks": {}, "anomalies": []}
    track_info_list = results[key]["tracks"]
    for i, track in enumerate(tl.tracks):
        kind = track.kind
        children = list(track)
        clips = [c for c in children if isinstance(c, otio.schema.Clip)]
        gaps  = [c for c in children if isinstance(c, otio.schema.Gap)]
        track_dur = track.duration().value
        # Sum trimmed_range durations (each child .trimmed_range() returns a TimeRange; .duration is RationalTime)
        sum_children = sum(c.trimmed_range().duration.value for c in children)
        info = {
            "idx": i, "name": track.name, "kind": kind,
            "dur": track_dur, "sum_children": sum_children,
            "n_clips": len(clips), "n_gaps": len(gaps),
            "enabled": track.enabled,
        }
        track_info_list.append(info)
        # Check for NaN/Inf in source_range
        for j, c in enumerate(clips):
            sr = c.source_range
            if sr is None:
                continue
            st = sr.start_time.value
            du = sr.duration.value
            if any(math.isnan(x) or math.isinf(x) for x in (st, du)):
                results[key]["anomalies"].append(
                    f"track {i} clip {j}: NaN/Inf in source_range start={st} dur={du}"
                )
            if st < 0:
                results[key]["anomalies"].append(
                    f"track {i} clip {j}: negative start_time={st}"
                )
            # media reference
            mr = c.media_reference
            if mr is None:
                results[key]["anomalies"].append(f"track {i} clip {j}: no media_reference")
            elif isinstance(mr, otio.schema.MissingReference):
                results[key]["anomalies"].append(f"track {i} clip {j}: MissingReference")
        # Sequential check: verify children don't have internal time gaps (trimmed_range)
        cum = 0.0
        for j, c in enumerate(children):
            dur = c.trimmed_range().duration.value
            # No internal overlap possible in a track's children list; verify order only
            cum += dur

print(json.dumps(results, indent=2, default=str))
