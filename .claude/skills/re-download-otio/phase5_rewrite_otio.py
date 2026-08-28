import opentimelineio as otio
import os, re, glob

OTIO_PATH = r"E:\Edit Job\Kyteq\Disney\OJNZ7wJZ-How Big is Disney World's Bus Fleet__2026-05-05\a.otio"
PROJECT_DIR = r"E:\Edit Job\Kyteq\Disney\OJNZ7wJZ-How Big is Disney World's Bus Fleet__2026-05-05"
HQ_CACHE = os.path.join(PROJECT_DIR, "v", "matcher-hq")
OUTPUT_OTIO = OTIO_PATH.replace(".otio", "_hq.otio")

def parse_clip_filename(filename):
    m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_(\d+)_(\d+)\.mp4$', filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None

# Build HQ index
hq_index = {}
for f in glob.glob(os.path.join(HQ_CACHE, '*_1080p_*.mp4')):
    m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_1080p_(\d+)_(\d+)\.mp4', os.path.basename(f))
    if m:
        hq_index[(m.group(1), int(m.group(2)), int(m.group(3)))] = f

print(f"HQ index: {len(hq_index)} files")

# Load OTIO
tl = otio.adapters.read_from_file(OTIO_PATH)
print(f"Loaded timeline with {len(tl.tracks)} tracks")

# Rewrite only clips that have HQ files — leave everything else untouched
replaced = 0
skipped = 0

for track_idx, track in enumerate(tl.tracks):
    for child in track:
        if not isinstance(child, otio.schema.Clip):
            continue

        mr = child.media_reference
        fname = child.name

        # ExternalReference: use target_url basename
        if isinstance(mr, otio.schema.ExternalReference) and mr.target_url:
            fname = os.path.basename(mr.target_url)

        vid, ss, ee = parse_clip_filename(fname)
        if not vid:
            # Non-YouTube clip (image, stock footage, etc.) — skip, leave as-is
            skipped += 1
            continue

        key = (vid, ss, ee)
        if key not in hq_index:
            # This clip wasn't downloaded — leave original reference as-is
            continue

        new_path = hq_index[key]

        # Compute duration from timestamps
        clip_dur_secs = ee - ss
        frame_rate = 30.0
        clip_dur_frames = int(clip_dur_secs * frame_rate)

        # Create new ExternalReference with HQ file
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

print(f"\nRewritten: {replaced} clips upgraded to HQ, {skipped} skipped (non-video, left as-is)")

# Write output
otio.adapters.write_to_file(tl, OUTPUT_OTIO)
print(f"Saved to: {OUTPUT_OTIO}")

# Validate
print("\nValidating...")
tl2 = otio.adapters.read_from_file(OUTPUT_OTIO)
ok = 0
broken = []
for track in tl2.tracks:
    for child in track:
        if isinstance(child, otio.schema.Clip):
            mr = child.media_reference
            if isinstance(mr, otio.schema.ExternalReference):
                if os.path.exists(mr.target_url):
                    ok += 1
                else:
                    broken.append(mr.target_url)
print(f"Validation: {ok} ExternalReferences valid, {len(broken)} broken")
if broken:
    for p in broken[:10]:
        print(f"  BROKEN: {p}")
