"""Build gap_fill_fixed.otio using pipeline OTIOs as source of truth."""
import json, os, opentimelineio as otio

RATE = 30.0
proj = r'E:\Edit Job\Kyteq\cruise\GIgnVPvf-The 12 NEW Things That Are Banned On Cruises in 20'
output_dir = os.path.join(proj, 'gap_fill', 'output', '20260421_075630')
cache_dir = os.path.join(proj, 'gap_fill', '.cache', 'v', 'matcher-alt')
out_path = os.path.join(proj, 'gap_fill_fixed.otio')

# 1. Load help.otio structure
with open(os.path.join(proj, 'help.otio'), 'r', encoding='utf-8') as f:
    help_otio = json.load(f)
v_track = [t for t in help_otio['tracks']['children'] if t.get('kind') == 'Video'][0]
items = v_track['children']

# Build position map for help.otio
item_data = []
pos = 0.0
for idx, item in enumerate(items):
    schema = item.get('OTIO_SCHEMA', '')
    sr = item.get('source_range', {})
    dur_val = sr.get('duration', {}).get('value', 0)
    dur_rate = sr.get('duration', {}).get('rate', RATE)
    dur_sec = dur_val / dur_rate
    is_gap = schema.startswith('Gap')
    item_data.append({
        'index': idx, 'start_sec': pos, 'end_sec': pos + dur_sec,
        'dur_sec': dur_sec, 'dur_frames': dur_val, 'rate': dur_rate,
        'is_gap': is_gap, 'is_significant': is_gap and dur_sec >= 0.5
    })
    pos += dur_sec
total_duration_sec = pos
total_duration_frames = sum(i['dur_frames'] for i in item_data)
print(f'Help OTIO: {len(items)} items, {total_duration_sec:.4f}s, {total_duration_frames:.0f} frames')

# 2. Load clips from each pipeline track OTIO
track_files = {
    0: 'timeline_V1_V1___Primary.otio',
    1: 'timeline_V2_V2___Alternativ.otio',
    2: 'timeline_V3_V3___Alternativ.otio',
    3: 'timeline_V4_V4___Secondary_.otio',
    4: 'timeline_V5_V5___Secondary_.otio',
    5: 'timeline_V6_V6___Secondary_.otio'
}
track_names = ['V1 - Primary', 'V2 - Alternative A', 'V3 - Alternative B',
               'V4 - Secondary A', 'V5 - Secondary B', 'V6 - Secondary C']

def load_pipe_clips(track_idx):
    path = os.path.join(output_dir, track_files[track_idx])
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    clips = []
    for t in data['tracks']['children']:
        if t.get('kind') == 'Video':
            pos = 0.0
            for c in t['children']:
                schema = c.get('OTIO_SCHEMA', '')
                sr = c.get('source_range', {})
                dur_val = sr.get('duration', {}).get('value', 0)
                dur_sec = dur_val / RATE
                if schema == 'Clip.2':
                    refs = c.get('media_references', {})
                    key = c.get('active_media_reference_key', '')
                    mr = refs.get(key, {}) if key else {}
                    fname = mr.get('name', '')
                    base = fname.replace('matcher-alt_', '').replace('.mp4', '')
                    clips.append({'start': pos, 'end': pos + dur_sec, 'dur': dur_sec, 'name': base})
                pos += dur_sec
            break
    return clips

pipe_clips = {t: load_pipe_clips(t) for t in range(6)}
for t in range(6):
    print(f'Track {t}: {len(pipe_clips[t])} clips from pipeline')

# 3. Helper functions
def make_gap(dur_frames, r=RATE):
    return {
        'OTIO_SCHEMA': 'Gap.1', 'metadata': {}, 'name': '',
        'source_range': {
            'OTIO_SCHEMA': 'TimeRange.1',
            'duration': {'OTIO_SCHEMA': 'RationalTime.1', 'rate': r, 'value': dur_frames},
            'start_time': {'OTIO_SCHEMA': 'RationalTime.1', 'rate': r, 'value': 0.0}
        },
        'effects': [], 'markers': [], 'enabled': True, 'color': None
    }

def make_clip(name, dur_frames, file_dur_frames, r=RATE):
    # Find file path from name (e.g. Z5UCeohVZCY_215_228.mp4)
    filepath = os.path.join(cache_dir, name)
    target_url = filepath.replace('\\', '/')
    return {
        'OTIO_SCHEMA': 'Clip.2',
        'metadata': {'Resolve_OTIO': {}},
        'name': name,
        'source_range': {
            'OTIO_SCHEMA': 'TimeRange.1',
            'duration': {'OTIO_SCHEMA': 'RationalTime.1', 'rate': r, 'value': dur_frames},
            'start_time': {'OTIO_SCHEMA': 'RationalTime.1', 'rate': r, 'value': 0.0}
        },
        'effects': [], 'markers': [], 'enabled': True, 'color': None,
        'media_references': {
            'DEFAULT_MEDIA': {
                'OTIO_SCHEMA': 'ExternalReference.1', 'metadata': {}, 'name': name,
                'available_range': {
                    'OTIO_SCHEMA': 'TimeRange.1',
                    'duration': {'OTIO_SCHEMA': 'RationalTime.1', 'rate': r, 'value': float(file_dur_frames)},
                    'start_time': {'OTIO_SCHEMA': 'RationalTime.1', 'rate': r, 'value': 0.0}
                },
                'available_image_bounds': None, 'target_url': target_url
            }
        },
        'active_media_reference_key': 'DEFAULT_MEDIA'
    }

def get_file_dur(name):
    filepath = os.path.join(cache_dir, name)
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', filepath],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        info = json.loads(r.stdout)
        dur_sec = float(info['format']['duration'])
        return round(dur_sec * RATE)
    except Exception:
        return round(15.0 * RATE)

import subprocess

# 4. Build 6 tracks
output_tracks = []
for track_idx in range(6):
    children = []
    total_track_frames = 0.0
    clips_used = 0

    for idata in item_data:
        gap_start = idata['start_sec']
        gap_end = idata['end_sec']
        gap_frames = idata['dur_frames']
        gap_rate = idata['rate']
        is_gap = idata['is_gap']
        is_sig = idata['is_significant']

        if is_gap:
            if is_sig:
                # Significant gap in help.otio: fill with pipeline clips
                overlapping = [c for c in pipe_clips[track_idx]
                              if c['end'] > gap_start and c['start'] < gap_end]
                overlapping.sort(key=lambda c: c['start'])

                if not overlapping:
                    children.append(make_gap(gap_frames, gap_rate))
                    total_track_frames += gap_frames
                    continue

                filled = 0.0
                for clip in overlapping:
                    clip_start_in_gap = max(0, clip['start'] - gap_start)
                    clip_end_in_gap = min(gap_end - gap_start, clip['end'] - gap_start)
                    clip_dur = round((clip_end_in_gap - clip_start_in_gap) * gap_rate)

                    if clip_dur <= 0:
                        continue
                    if filled + clip_dur > gap_frames:
                        clip_dur = gap_frames - filled

                    if clip_dur > 0:
                        file_dur = get_file_dur(clip['name'])
                        c = make_clip(clip['name'], clip_dur, file_dur, gap_rate)
                        children.append(c)
                        filled += clip_dur
                        clips_used += 1

                # Absorb rounding drift
                drift = gap_frames - filled
                if drift != 0 and children:
                    last = children[-1]
                    old_dur = last['source_range']['duration']['value']
                    last['source_range']['duration']['value'] = old_dur + drift
                    filled += drift
                total_track_frames += filled
            else:
                # Small gap: emit gap of same duration
                children.append(make_gap(gap_frames, gap_rate))
                total_track_frames += gap_frames
        else:
            # Original clip in help.otio: emit gap
            children.append(make_gap(gap_frames, gap_rate))
            total_track_frames += gap_frames

    expected = total_duration_frames
    drift = total_track_frames - expected
    dur_sec = total_track_frames / RATE
    print(f'Track {track_idx} ({track_names[track_idx]}): {clips_used} clips, '
          f'{total_track_frames:.0f}/{expected:.0f} frames, drift={drift:.0f}')

    track_obj = {
        'OTIO_SCHEMA': 'Track.1',
        'metadata': {'Resolve_OTIO': {'Locked': False}},
        'name': track_names[track_idx],
        'source_range': None, 'effects': [], 'markers': [], 'enabled': True, 'color': None,
        'kind': 'Video', 'children': children
    }
    output_tracks.append(track_obj)

# 5. Write output OTIO
output_otio = {
    'OTIO_SCHEMA': 'Timeline.1',
    'metadata': {'Resolve_OTIO': {'Resolve OTIO Meta Version': '1.0'}, 'gap_fill': True},
    'name': 'Gap Fill',
    'global_start_time': {
        'OTIO_SCHEMA': 'RationalTime.1', 'rate': RATE, 'value': 108000.0
    },
    'tracks': {
        'OTIO_SCHEMA': 'Stack.1', 'metadata': {}, 'name': 'tracks', 'source_range': None,
        'effects': [], 'markers': [], 'enabled': True, 'color': None,
        'children': output_tracks
    }
}

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output_otio, f, indent=2)

file_size = os.path.getsize(out_path)
print(f'\nOutput: {out_path}')
print(f'Help OTIO total: {total_duration_sec:.4f}s ({total_duration_frames:.0f} frames)')
print(f'File size: {file_size / 1024:.1f} KB')
print('Done.')
