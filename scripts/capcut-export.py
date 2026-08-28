#!/usr/bin/env python3
"""Export top-most OTIO clips as individual files for CapCut import.

CapCut doesn't support OTIO/EDL/XML import, so we cut the HQ video files
to the exact OTIO clip durations and save them in ascending-numbered
files that can be dragged into CapCut's media panel.

Per-clip cut uses the OTIO clip's source_range.start_time (offset into the
source HQ file) and source_range.duration (length). Re-encodes for
frame-accurate cuts so the output duration matches the OTIO exactly.

Usage:
    python scripts/capcut-export.py --project <dir> [--otio <file>]
                                    [--hq-cache <dir>] [--out <dir>]
                                    [--dry-run] [--force]

Defaults:
    OTIO:      <project>/downloadplease.otio
    HQ cache:  <project>/v/matcher-hq
    Output:    <project>/capcut_import
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys

import opentimelineio as otio


RATE = 30.0
VIDEO_CRF = '18'
VIDEO_PRESET = 'fast'
AUDIO_BITRATE = '192k'


def parse_clip_filename(fname):
    """Returns (video_id, start_sec, end_sec) or (None, None, None).

    Accepts both standard ('vid_ss_ee.mp4') and HQ ('vid_1080p_ss_ee.mp4') forms.
    """
    m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_(?:1080p_)?(\d+)_(\d+)\.mp4$', fname)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None


def find_top_most_clips_with_range(otio_path):
    """Walk OTIO, find top-most visible clip per moment, return source_range info.

    Top-most = highest track_idx among enabled, non-gap, non-audio clips at
    that moment (matches the re-download-otio skill semantics).

    Returns: (timeline, [(name, vid, ss, ee, offset_sec, duration_sec), ...])
    """
    tl = otio.adapters.read_from_file(otio_path)
    all_clips = []
    for track_idx, track in enumerate(tl.tracks):
        if not track.enabled:
            continue
        is_audio = track.kind == otio.schema.TrackKind.Audio
        tp = otio.opentime.RationalTime(0, RATE)
        for clip in track:
            all_clips.append({
                'track_idx': track_idx,
                'is_audio': is_audio,
                'is_gap': isinstance(clip, otio.schema.Gap),
                'clip_enabled': getattr(clip, 'enabled', True),
                'start': tp.value,
                'end': (tp + clip.duration()).value,
                'name': clip.name,
                'mr': getattr(clip, 'media_reference', None),
                'source_range': getattr(clip, 'source_range', None),
            })
            tp = tp + clip.duration()

    unique_starts = sorted(set(c['start'] for c in all_clips))
    seen = set()
    results = []
    for seg_start in unique_starts:
        covering = [c for c in all_clips
                    if c['start'] <= seg_start < c['end']
                    and not c['is_gap'] and not c['is_audio'] and c['clip_enabled']]
        if not covering:
            continue
        top = max(covering, key=lambda x: x['track_idx'])
        mr = top['mr']
        if isinstance(mr, otio.schema.MissingReference):
            fname = top['name']
        elif isinstance(mr, otio.schema.ExternalReference):
            fname = os.path.basename(mr.target_url) if mr.target_url else top['name']
        else:
            fname = top['name']
        vid, ss, ee = parse_clip_filename(fname)
        if not vid:
            continue
        sr = top['source_range']
        offset_sec = sr.start_time.value / RATE if sr else 0.0
        duration_sec = sr.duration.value / RATE if sr else 0.0
        # Dedup by full source identity: same HQ file at the same offset = same cut.
        # Different clips on different tracks can reference the same HQ file at
        # different offsets — those must be exported as separate cuts.
        key = (vid, ss, ee, round(offset_sec, 3))
        if key in seen:
            continue
        seen.add(key)

        results.append({
            'name': top['name'], 'vid': vid, 'ss': ss, 'ee': ee,
            'offset_sec': offset_sec, 'duration_sec': duration_sec,
        })
    return tl, results


def build_hq_index(hq_cache):
    """Map (vid, ss, ee) -> HQ file path. Always prefers the widest available arc.

    For each (vid, ss) group, the file with the largest `ee` is used for every
    cut — the wider file is a strict superset of the narrower one, so cuts at
    the same offset/duration yield the same frames. This avoids OOB drops when
    the OTIO clip's source_range extends slightly past the original arc end
    (e.g., a 38.5s cut against a `0_38` source would fail, but a `0_40` source
    contains those extra frames).
    """
    by_key = {}
    for f in glob.glob(os.path.join(hq_cache, '*_1080p_*.mp4')):
        vid, ss, ee = parse_clip_filename(os.path.basename(f))
        if vid:
            by_key.setdefault((vid, ss), []).append((ee, f))
    index = {}
    for (vid, ss), entries in by_key.items():
        entries.sort()  # ascending by ee
        widest_ee, widest_path = entries[-1]
        # Register each exact-arc key against the widest file.
        for ee, _path in entries:
            index[(vid, ss, ee)] = widest_path
    return index


_KF_CACHE = {}


def _keyframe_pts_list(path):
    """Return sorted list of keyframe timestamps in `path` (cached per path).

    Uses the container's packet index (flags=K__) — reads metadata only,
    ~0.2s for a 335s file. Much faster than `-skip_frame nokey` (which decodes).
    """
    if path in _KF_CACHE:
        return _KF_CACHE[path]
    try:
        out = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'packet=pts_time,flags',
             '-of', 'csv=p=0', path],
            text=True, encoding='utf-8', errors='replace',
            stderr=subprocess.DEVNULL,
        )
        kfs = []
        for line in out.split('\n'):
            parts = line.strip().split(',')
            if len(parts) >= 2 and 'K' in parts[1]:
                try: kfs.append(float(parts[0]))
                except ValueError: pass
        kfs.sort()
    except Exception:
        kfs = []
    _KF_CACHE[path] = kfs
    return kfs


def _prev_keyframe_pts(path, offset):
    """Return pts of the latest keyframe <= offset (0.0 if none).

    Uses cached packet-index keyframe list — no frame decode, ~0s per query
    after the first call to a given source.
    """
    kfs = _keyframe_pts_list(path)
    prev = 0.0
    for t in kfs:
        if t > offset:
            break
        prev = t
    return prev


def cut_clip(src, dst, offset, duration, with_audio=True):
    """Cut a clip with ffmpeg (re-encodes for frame-accurate output).

    Hybrid seek (fast + frame-accurate):
      `-ss` BEFORE `-i` does a fast keyframe-only seek to the previous keyframe
      (cached per source via packet index — no per-clip decode). `-ss` AFTER
      `-i` then frame-accurately advances to the requested offset, but only
      over a small range (one GOP). This is O(GOP) to decode instead of
      O(offset) — significant speedup on sources with large offsets.

    Frame count: `-frames:v N` (probed from src_fps) instead of `-t duration`.
    ffmpeg's `-ss` after `-i` snaps the start pts to the next frame boundary,
    so `-t duration` would miss the last frame. `-frames:v` is exact.

    If with_audio=False, drops the audio stream entirely (use as fallback when
    the source has mismatched audio/video durations past the cut point).
    """
    src_fps = _probe_fps(src) or RATE
    n_frames = max(1, round(duration * src_fps))
    prev_kf = _prev_keyframe_pts(src, offset)
    fine_seek = max(0.0, offset - prev_kf)
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{prev_kf:.3f}',     # fast keyframe seek
        '-i', src,
        '-ss', f'{fine_seek:.3f}',   # frame-accurate advance (one GOP)
        '-frames:v', str(n_frames),
        '-c:v', 'libx264',
        '-preset', VIDEO_PRESET,
        '-crf', VIDEO_CRF,
    ]
    if with_audio:
        cmd += ['-c:a', 'aac', '-b:a', AUDIO_BITRATE, '-frames:a', str(round(duration * 44100))]
    else:
        cmd += ['-an']
    cmd += ['-movflags', '+faststart', '-loglevel', 'error', dst]
    return subprocess.run(cmd, capture_output=True, text=True,
                         encoding='utf-8', errors='replace')


def _probe_fps(path):
    """Return the source's video frame rate as a float, or None on failure."""
    import json
    try:
        out = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate',
             '-of', 'default=nk=1:nw=1', path],
            text=True, encoding='utf-8', errors='replace',
            stderr=subprocess.DEVNULL,
        ).strip()
        if '/' in out:
            num, den = out.split('/')
            num, den = float(num), float(den)
            if den > 0:
                return num / den
        return float(out) if out else None
    except Exception:
        return None


def probe_duration(path):
    """Return media duration in seconds via ffprobe, or None on failure.

    Returns None if the file is unreadable (e.g., missing moov atom = corrupt).
    """
    try:
        out = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=nk=1:nw=1', path],
            text=True, encoding='utf-8', errors='replace',
            stderr=subprocess.DEVNULL,
        ).strip()
        return float(out) if out else None
    except Exception:
        return None


def has_video_stream(path):
    """Return True if the file has at least one video stream."""
    try:
        out = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v',
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', path],
            text=True, encoding='utf-8', errors='replace',
            stderr=subprocess.DEVNULL,
        ).strip()
        return bool(out)
    except Exception:
        return False


def probe_video_duration(path):
    """Return video stream duration in seconds, or None if no/malformed video."""
    try:
        out = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=duration', '-of', 'default=nk=1:nw=1', path],
            text=True, encoding='utf-8', errors='replace',
            stderr=subprocess.DEVNULL,
        ).strip()
        return float(out) if out else None
    except Exception:
        return None


def precheck_sources(hq_index, clips):
    """Probe each HQ source for readability, truncation, and per-stream bounds.

    Returns: (corrupt_set, truncated_set, video_short_set, video_dur_map)
        corrupt_set:     set of (vid, ss, ee) keys whose source file is unreadable
        truncated_set:   set of keys where container duration < 95% of arc length
                         (download didn't complete)
        video_short_set: set of keys where the video stream ends before the audio
                         stream — clips whose offset is past the video end will be
                         exported as audio-only (ffmpeg silently drops video)
        video_dur_map:   dict mapping (vid, ss, ee) -> video stream duration in seconds
    """
    duration_map = {}
    video_dur_map = {}
    corrupt_set = set()
    truncated_set = set()
    video_short_set = set()
    seen_keys = set()
    for c in clips:
        key = (c['vid'], c['ss'], c['ee'])
        if key in seen_keys or key not in hq_index:
            continue
        seen_keys.add(key)
        path = hq_index[key]
        dur = probe_duration(path)
        if dur is None:
            corrupt_set.add(key)
            continue
        duration_map[key] = dur
        vdur = probe_video_duration(path)
        if vdur is not None:
            video_dur_map[key] = vdur
            if vdur < dur * 0.95:
                video_short_set.add(key)
        expected = key[2] - key[1]
        if dur < expected * 0.95:
            truncated_set.add(key)
    return corrupt_set, truncated_set, video_short_set, video_dur_map


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--project', required=True, help='Project directory (must contain OTIO)')
    p.add_argument('--otio', help='OTIO filename inside project (default: downloadplease.otio)')
    p.add_argument('--hq-cache', help='HQ download directory (default: <project>/v/matcher-hq)')
    p.add_argument('--out', help='Output directory (default: <project>/capcut_import)')
    p.add_argument('--dry-run', action='store_true', help='Plan only; do not run ffmpeg')
    p.add_argument('--force', action='store_true', help='Re-cut even if output file exists')
    args = p.parse_args()

    project_dir = args.project
    otio_name = args.otio or 'downloadplease.otio'
    otio_path = os.path.join(project_dir, otio_name)
    hq_cache = os.path.normpath(args.hq_cache or os.path.join(project_dir, 'v', 'matcher-hq'))
    out_dir = os.path.normpath(args.out or os.path.join(project_dir, 'capcut_import'))

    if not os.path.exists(otio_path):
        print(f'ERROR: OTIO not found: {otio_path}', file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(hq_cache):
        print(f'ERROR: HQ cache not found: {hq_cache}', file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    print(f'Project: {project_dir}')
    print(f'OTIO:    {otio_path}')
    print(f'HQ dir:  {hq_cache}')
    print(f'Output:  {out_dir}')

    timeline, clips = find_top_most_clips_with_range(otio_path)
    hq_index = build_hq_index(hq_cache)
    print(f'Top-most visible clips: {len(clips)}')
    print(f'HQ files available:     {len(hq_index)}')

    ready, missing, zero_dur, corrupt, oob, video_short = [], [], [], [], [], []
    for c in clips:
        key = (c['vid'], c['ss'], c['ee'])
        if key not in hq_index:
            missing.append(c)
            continue
        if c['duration_sec'] <= 0:
            zero_dur.append(c)
            continue
        ready.append(c)

    print('Pre-checking HQ sources via ffprobe...', flush=True)
    corrupt_keys, truncated_keys, video_short_keys, video_durations = precheck_sources(hq_index, ready)
    hq_durations = video_durations  # alias for backwards compat below
    if corrupt_keys:
        print(f'Corrupt HQ files:      {len(corrupt_keys)} source(s)', flush=True)
        for k in sorted(corrupt_keys):
            print(f'  {os.path.basename(hq_index[k])}')
    if truncated_keys:
        print(f'Truncated HQ files:    {len(truncated_keys)} source(s) (actual < 95% of arc)')
        for k in sorted(truncated_keys):
            path = hq_index[k]
            actual = probe_duration(path) or 0
            expected = k[2] - k[1]
            print(f'  {os.path.basename(path)}: {actual:.1f}s actual, expected {expected}s')
    if video_short_keys:
        print(f'Video-stream short:    {len(video_short_keys)} source(s) (video < audio)')
        for k in sorted(video_short_keys):
            vdur = video_durations.get(k, 0)
            adur = probe_duration(hq_index[k]) or 0
            print(f'  {os.path.basename(hq_index[k])}: video {vdur:.1f}s, audio {adur:.1f}s')
    for c in ready:
        key = (c['vid'], c['ss'], c['ee'])
        if key in corrupt_keys:
            corrupt.append(c)
            continue
        if key in video_short_keys and c['offset_sec'] > video_durations.get(key, 0):
            video_short.append(c)
            continue
        # Truncated sources are NOT blanket-excluded: a source shorter than its
        # arc still yields valid cuts for clips whose offset+duration fall inside
        # the real footage. Only clips that cut past the actual end are dropped
        # (handled by the per-clip out-of-bounds check below, which uses the
        # actual probed duration, not the arc length).
        hq_dur = probe_duration(hq_index[key]) or 0.0
        if c['offset_sec'] + c['duration_sec'] > hq_dur + 0.05:
            oob.append((c, hq_dur))
    if oob:
        print(f'Out-of-bounds cuts:    {len(oob)} clip(s) (offset+duration > HQ duration)')
        for c, hd in oob[:5]:
            key = (c['vid'], c['ss'], c['ee'])
            print(f'  {os.path.basename(hq_index[key])}: '
                  f'{c["offset_sec"]:.2f}s + {c["duration_sec"]:.2f}s > {hd:.2f}s')

    ready = [c for c in ready if (c['vid'], c['ss'], c['ee']) not in corrupt_keys
             and not any(c is x[0] for x in oob)
             and not any(c is x for x in video_short)]

    print(f'Ready to export: {len(ready)} clips')
    if missing:
        print(f'Missing HQ file: {len(missing)} clips (will skip)')
    if zero_dur:
        print(f'Zero-duration:   {len(zero_dur)} clips (will skip)')
    if corrupt:
        print(f'Corrupt source:  {len(corrupt)} clips (will skip)')

    if args.dry_run:
        total_sec = sum(c['duration_sec'] for c in ready)
        print(f'\n=== Dry-run Plan ===')
        print(f'Would export:  {len(ready)} clips')
        print(f'Would skip:    {len(missing)} (no HQ file) + {len(zero_dur)} (zero duration) + '
              f'{len(corrupt)} (corrupt/oob) + {len(video_short)} (video past end)')
        print(f'Total duration: {total_sec:.1f}s ({total_sec/60:.1f} min)')
        if ready:
            print(f'\nFirst 5 clips:')
            for i, c in enumerate(ready[:5], 1):
                src_name = os.path.basename(hq_index[(c['vid'], c['ss'], c['ee'])])
                print(f'  clip_{i:04d}.mp4 <- {src_name} '
                      f'@ {c["offset_sec"]:.2f}s for {c["duration_sec"]:.2f}s')
        return

    manifest_path = os.path.join(out_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='', encoding='utf-8') as mf:
        writer = csv.writer(mf)
        writer.writerow(['index', 'output', 'source_file', 'video_id', 'arc_start',
                         'arc_end', 'offset_sec', 'duration_sec', 'actual_dur', 'status'])

    ok = fail = skip = 0
    for i, c in enumerate(ready, 1):
        key = (c['vid'], c['ss'], c['ee'])
        src = hq_index[key]
        dst = os.path.join(out_dir, f'clip_{i:04d}.mp4')
        src_name = os.path.basename(src)
        log_prefix = f'[{i}/{len(ready)}] clip_{i:04d}.mp4 <- {src_name}'

        if os.path.exists(dst) and not args.force:
            existing = probe_duration(dst)
            if existing is not None and abs(existing - c['duration_sec']) < 0.1:
                print(f'{log_prefix} ... SKIP (already exists, {existing:.2f}s)', flush=True)
                skip += 1
                _write_manifest_row(manifest_path, i, dst, src_name, c, 'skip', existing)
                continue

        print(f'{log_prefix} @ {c["offset_sec"]:.2f}s for {c["duration_sec"]:.2f}s ...',
              flush=True, end='')
        key = (c['vid'], c['ss'], c['ee'])
        vdur = video_durations.get(key)
        force_silent = vdur is not None and c['offset_sec'] >= vdur
        if force_silent:
            r = cut_clip(src, dst, c['offset_sec'], c['duration_sec'], with_audio=False)
        else:
            r = cut_clip(src, dst, c['offset_sec'], c['duration_sec'])
            if r.returncode == 0 and not has_video_stream(dst):
                # ffmpeg silently dropped the video stream (source has short video).
                r = cut_clip(src, dst, c['offset_sec'], c['duration_sec'], with_audio=False)
                if r.returncode == 0:
                    print(' OK (video-only, source video too short) ', end='')
            elif r.returncode != 0 and '733' in (r.stderr or ''):
                r = cut_clip(src, dst, c['offset_sec'], c['duration_sec'], with_audio=False)
                if r.returncode == 0:
                    print(' OK (video-only, audio dropped) ', end='')
        audio_ok = r.returncode == 0
        if audio_ok and os.path.exists(dst) and os.path.getsize(dst) > 1000:
            actual = probe_duration(dst) or 0.0
            drift = actual - c['duration_sec']
            print(f' OK ({actual:.2f}s, drift {drift:+.2f}s)')
            ok += 1
            _write_manifest_row(manifest_path, i, dst, src_name, c, 'ok', actual)
        else:
            err = (r.stderr or '').strip()[-160:]
            print(f' FAIL: {err}')
            fail += 1
            _write_manifest_row(manifest_path, i, dst, src_name, c, 'fail')

    print(f'\n=== Export Summary ===')
    print(f'OK:   {ok}')
    print(f'FAIL: {fail}')
    print(f'SKIP: {skip} (already exported)')
    print(f'Output dir: {out_dir}')
    print(f'Manifest:   {manifest_path}')


def _write_manifest_row(path, idx, dst, src_name, clip, status, actual_dur=None):
    with open(path, 'a', newline='', encoding='utf-8') as mf:
        writer = csv.writer(mf)
        writer.writerow([
            idx, os.path.basename(dst), src_name, clip['vid'], clip['ss'], clip['ee'],
            f'{clip["offset_sec"]:.3f}', f'{clip["duration_sec"]:.3f}',
            f'{actual_dur:.3f}' if actual_dur is not None else '',
            status,
        ])


if __name__ == '__main__':
    main()