#!/usr/bin/env python3
"""Rewrite OTIO to point to HQ 1080p files."""
import os, re, opentimelineio as otio

PROJ = r'e:\Edit Job\Kyteq\cruise\oZi3XyYl-The Cruise Ship Casino SECRETS They Hide From Pass'
OTIO_PATH = os.path.join(PROJ, 'redownload.otio')
HQ_CACHE = os.path.join(PROJ, 'cache', 'v', 'matcher-hq')
OUTPUT_PATH = os.path.join(PROJ, 'redownload_hq.otio')

def parse_clip_filename(filename):
    m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_(\d+)_(\d+)\.mp4$', filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None

# Build HQ index
hq_index = {}
for f in os.listdir(HQ_CACHE):
    m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_1080p_(\d+)_(\d+)\.mp4$', f)
    if m:
        hq_index[(m.group(1), int(m.group(2)), int(m.group(3)))] = os.path.join(HQ_CACHE, f)

print(f'HQ index: {len(hq_index)} files')

# Read OTIO
timeline = otio.adapters.read_from_file(OTIO_PATH)

converted = 0
skipped_non_yt = 0
missing_hq = 0

for track in timeline.tracks:
    for child in track:
        if not isinstance(child, otio.schema.Clip):
            continue

        mr = child.media_reference
        if isinstance(mr, otio.schema.MissingReference):
            fname = child.name
        elif isinstance(mr, otio.schema.ExternalReference):
            fname = os.path.basename(mr.target_url) if mr.target_url else child.name
        else:
            skipped_non_yt += 1
            continue

        vid, ss, ee = parse_clip_filename(fname)
        if not vid:
            skipped_non_yt += 1
            continue

        key = (vid, ss, ee)
        if key not in hq_index:
            missing_hq += 1
            print(f'  Missing HQ: {fname}')
            continue

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
        converted += 1

print(f'Converted: {converted}')
print(f'Skipped (non-YouTube): {skipped_non_yt}')
print(f'Missing HQ: {missing_hq}')

# Write
otio.adapters.write_to_file(timeline, OUTPUT_PATH)
print(f'Written: {OUTPUT_PATH}')

# Verify
timeline2 = otio.adapters.read_from_file(OUTPUT_PATH)
verify = sum(1 for t in timeline2.tracks for c in t if isinstance(c, otio.schema.Clip) and isinstance(c.media_reference, otio.schema.ExternalReference))
print(f'Verified: {verify} clips with ExternalReference')