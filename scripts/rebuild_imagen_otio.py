"""Rebuild imagen OTIO from summary.json — DaVinci-compatible structure."""
import json, opentimelineio as otio
from opentimelineio.opentime import RationalTime, TimeRange
from pathlib import Path

project = Path(r'E:\Edit Job\Stu\6fHg4mGc-1. WHY CHINA IS QUIETLY BUYING AMERICA')
summary_path = project / 'generated_images' / 'summary.json'

with open(summary_path) as f:
    summary = json.load(f)

images = summary['images']
print(f'Rebuilding OTIO with {len(images)} images (DaVinci-compatible)')

RATE = 30.0

# ── Timeline ──────────────────────────────────────────────────────────────────
timeline = otio.schema.Timeline(name='generated_images_timeline')
timeline.metadata['Resolve_OTIO'] = {'Resolve OTIO Meta Version': '1.0'}
timeline.global_start_time = RationalTime(value=0, rate=RATE)

# ── Tracks Stack — name="" (matching working OTIOs)
timeline.tracks.name = ''

# ── V12 Track — direct child of timeline.tracks
v12 = otio.schema.Track(name='V12 - Generated Images', kind=otio.schema.TrackKind.Video)
v12.enabled = True
v12.color = None
v12.metadata['Resolve_OTIO'] = {'Locked': False}

current_time = 0.0  # seconds

for img in images:
    file_path = img['file']
    if not Path(file_path).exists():
        print(f'SKIP (not found): {file_path}')
        continue

    start = img['start_time']
    end = img['end_time']
    duration = max(0.001, end - start)
    batch_id = img['batch_id']

    # Gap from current position to this clip's start
    gap_dur = start - current_time
    if gap_dur > 0.001:
        gap = otio.schema.Gap()
        gap.source_range = TimeRange(
            start_time=RationalTime(0, RATE),
            duration=RationalTime(int(gap_dur * RATE), RATE)
        )
        gap.enabled = True
        gap.color = None
        v12.append(gap)

    dur_frames = int(duration * RATE)

    # ExternalReference — forward-slash Windows path, available_range
    safe_path = file_path.replace("\\", "/")
    file_name = Path(file_path).name

    media_ref = otio.schema.ExternalReference(
        target_url=safe_path,
        available_range=TimeRange(
            start_time=RationalTime(0, RATE),
            duration=RationalTime(dur_frames, RATE)
        )
    )
    media_ref.name = file_name

    # Clip — use DIRECT media_reference (NOT media_references dict)
    # This produces ExternalReference in JSON, not MissingReference
    clip = otio.schema.Clip(name=f'IMG:{batch_id}')
    clip.media_reference = media_ref
    clip.active_media_reference_key = 'DEFAULT_MEDIA'

    clip.source_range = TimeRange(
        start_time=RationalTime(0, RATE),
        duration=RationalTime(dur_frames, RATE)
    )
    clip.enabled = True
    clip.color = None

    # FreezeFrame — hold frame 0 for entire clip duration
    clip.effects.append(otio.schema.FreezeFrame())

    # Metadata matching pipeline convention
    clip.metadata['Resolve_OTIO'] = {}
    clip.metadata['batch_id'] = batch_id
    clip.metadata['start_time'] = start
    clip.metadata['end_time'] = end

    v12.append(clip)
    current_time = end

# V12 is direct child of timeline.tracks — no inner Stack wrapper
timeline.tracks.append(v12)

out_path = project / 'generated_images.otio'
otio.adapters.write_to_file(timeline, str(out_path))

clips = [c for c in v12 if c.__class__.__name__ == 'Clip']
gaps = [c for c in v12 if c.__class__.__name__ == 'Gap']
print(f'Written: {out_path}')
print(f'  V12 clips: {len(clips)}, gaps: {len(gaps)}')
print(f'  Timeline duration: {timeline.duration().value / RATE:.2f}s')
print(f'  timeline.tracks.name: "{timeline.tracks.name}"')

# Verify JSON output
json_str = timeline.to_json_string()
# Check first clip in JSON
import json as _json
obj = _json.loads(json_str)
track = obj['tracks']['children'][0]
clip_data = track['children'][0]
print(f'  First clip schema: {clip_data["OTIO_SCHEMA"]}')
print(f'  First clip media_ref: {clip_data.get("media_reference", clip_data.get("media_references", {}))}')