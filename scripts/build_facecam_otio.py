"""Build a separate facecam OTIO timeline synced to the main timeline."""
import glob
import json
import os
import subprocess
import sys


def get_file_duration(fpath):
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', fpath],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    info = json.loads(result.stdout)
    return float(info['format']['duration'])


def make_time(value, rate):
    return {'OTIO_SCHEMA': 'RationalTime.1', 'value': value, 'rate': rate}


def make_range(start_val, dur_val, rate):
    return {
        'OTIO_SCHEMA': 'TimeRange.1',
        'start_time': make_time(start_val, rate),
        'duration': make_time(dur_val, rate),
    }


def make_gap(dur_val, rate):
    return {
        'OTIO_SCHEMA': 'Gap.1',
        'metadata': {},
        'name': '',
        'source_range': make_range(0, dur_val, rate),
        'effects': [],
        'markers': [],
        'enabled': True,
        'color': None,
    }


def main():
    proj_dirs = glob.glob('E:/Edit Job/Stu/3.*TRILLION*')
    if not proj_dirs:
        print('ERROR: Project directory not found')
        sys.exit(1)
    proj_dir = proj_dirs[0]

    output_dirs = glob.glob(os.path.join(proj_dir, 'output', '*'))
    output_dir = output_dirs[0]
    facecam_dir = os.path.join(proj_dir, 'facecam')
    main_otio_path = os.path.join(output_dir, 'timeline_FULL.otio')

    # Read main OTIO
    with open(main_otio_path, 'r', encoding='utf-8') as f:
        main_otio = json.load(f)

    # Read generation log
    with open(os.path.join(facecam_dir, 'generation_log.json'), 'r', encoding='utf-8') as f:
        gen_log = json.load(f)

    # Calculate total timeline duration from V1
    v1 = main_otio['tracks']['children'][0]
    total_dur = 0.0
    for clip in v1['children']:
        sr = clip.get('source_range', {})
        if sr:
            dur = sr.get('duration', {})
            total_dur += dur.get('value', 0) / dur.get('rate', 1)

    rate = 30.0
    segments = sorted(gen_log['segments'], key=lambda s: s['start_s'])

    # Build track children
    children = []
    current_pos = 0.0

    for seg in segments:
        start_s = seg['start_s']
        facecam_path = os.path.join(facecam_dir, seg['facecam_file']).replace(os.sep, '/')
        # DaVinci expects bare paths (no file:/// prefix), matching main OTIO format
        file_url = facecam_path
        actual_dur = get_file_duration(facecam_path)

        # Gap before clip
        gap_dur = start_s - current_pos
        if gap_dur > 0.001:
            children.append(make_gap(round(gap_dur * rate), rate))

        # Clip
        seg_label = ','.join(str(i) for i in seg['segment_indices'])
        clip_frames = round(actual_dur * rate)

        children.append({
            'OTIO_SCHEMA': 'Clip.2',
            'name': f'[Facecam S{seg_label}] {seg["facecam_file"]}',
            'source_range': make_range(0, clip_frames, rate),
            'media_references': {
                'DEFAULT_MEDIA': {
                    'OTIO_SCHEMA': 'ExternalReference.1',
                    'metadata': {},
                    'name': seg['facecam_file'],
                    'available_image_bounds': None,
                    'target_url': file_url,
                    'available_range': make_range(0, clip_frames, rate),
                },
            },
            'active_media_reference_key': 'DEFAULT_MEDIA',
            'effects': [],
            'markers': [],
            'enabled': True,
            'color': None,
            'metadata': {
                'Resolve_OTIO': {},
                'facecam': {
                    'segment_indices': seg['segment_indices'],
                    'text': seg['text'],
                    'confidence': seg.get('confidence', 0),
                }
            },
        })

        current_pos = start_s + actual_dur

    # Trailing gap — clamp so total matches main timeline exactly
    remaining = total_dur - current_pos
    if remaining > 0.001:
        children.append(make_gap(round(remaining * rate), rate))
    elif remaining < -0.001:
        # Facecam clips extend slightly past main timeline; trim trailing gap if present
        # or just let it be slightly longer (NLE will handle it)
        pass

    # Build timeline
    facecam_otio = {
        'OTIO_SCHEMA': 'Timeline.1',
        'name': 'Facecam Timeline',
        'metadata': {'source': 'facecam_generation', 'synced_to': 'timeline_FULL.otio'},
        'tracks': {
            'OTIO_SCHEMA': 'Stack.1',
            'name': 'tracks',
            'children': [{
                'OTIO_SCHEMA': 'Track.1',
                'metadata': {},
                'name': 'V1 - Facecam',
                'source_range': None,
                'effects': [],
                'markers': [],
                'enabled': True,
                'color': None,
                'children': children,
                'kind': 'Video',
            }],
            'source_range': None,
            'effects': [], 'markers': [], 'metadata': {},
        },
        'global_start_time': make_time(0, rate),
    }

    out_path = os.path.join(output_dir, 'timeline_facecam.otio')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(facecam_otio, f, indent=2)

    # Report
    num_clips = sum(1 for c in children if 'Clip' in c['OTIO_SCHEMA'])
    num_gaps = sum(1 for c in children if 'Gap' in c['OTIO_SCHEMA'])
    track_dur = sum(
        c['source_range']['duration']['value'] / c['source_range']['duration']['rate']
        for c in children
    )

    print(f'Written: {out_path}')
    print(f'Track: {num_clips} clips, {num_gaps} gaps')
    print(f'Facecam track duration: {track_dur:.2f}s | Main timeline: {total_dur:.2f}s')
    print()
    print('Placement:')
    pos = 0.0
    for c in children:
        dur = c['source_range']['duration']['value'] / c['source_range']['duration']['rate']
        if 'Clip' in c['OTIO_SCHEMA']:
            print(f'  {c["name"]:55s} @ {pos:8.2f}s  dur={dur:.2f}s')
        pos += dur


if __name__ == '__main__':
    main()
