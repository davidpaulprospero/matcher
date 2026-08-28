#!/usr/bin/env python3
"""Re-download OTIO clips at HQ (1080p) and rewrite references.

Single canonical implementation of the re-download-otio skill.

Workflow (per skill):
  1. Parse OTIO. Find TOP-MOST visible clip at each timeline moment
     (highest track_idx among enabled, non-gap, non-audio clips).
  2. Dedup by (vid, ss, ee). Skip already-cached HQ files.
  3. Find YouTube URLs from project + gap_fill checkpoints (gzip-first).
  4. Download via yt-dlp with --download-sections.
     First pass: height>=720. Retry on fail: height<=480.
  5. Rewrite OTIO media_references to point to HQ files.
  6. Validate: paths exist, available_range matches section length.
  7. Optional --flatten: also emit <input>_hq_single.otio with one V1 track of
     top-most clips + voiceover preserved on A1, for transitions in DaVinci.

Usage:
    python scripts/re-download-otio.py --project <dir> [--otio <file>]
                                        [--hq-cache <dir>]
                                        [--rewrite-only] [--skip-rewrite]
                                        [--validate-only] [--flatten]

Defaults:
    OTIO:      <project>/downloadplease.otio
    HQ cache:  <project>/v/matcher-hq
    Cookies:   D:/_Projects/voiceover-matcher-dev/cookies/main.txt
"""
import argparse
import gzip
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import opentimelineio as otio


COOKIES_DIR = r'D:\_Projects\voiceover-matcher-dev\cookies'
COOKIES_DEFAULT = os.path.join(COOKIES_DIR, 'main.txt')
# Player client rotation: web_embedded works reliably on nightly (2026.08.18+),
# mweb is the fallback. web_embedded bypasses the per-IP CDN throttle that 403s
# mweb-format URLs on residential PH IPs.
EXTRACTOR_ARGS = ['--extractor-args', 'youtube:player_client=web_embedded,mweb']
FMT_HI = 'bv*[ext=mp4][height>=720]+ba[ext=m4a]/bv*+ba/best'
FMT_LO = ('bv*[ext=mp4][height<=480]+ba[ext=m4a]/'
          'bv*[height<=480]+ba[ext=m4a]/bv*+ba/best')
RATE = 30.0

# Default cookie rotation order. backup1 has been the most-reliable as of
# 2026-08-19; main.txt currently 403s on this IP. Order is overridable via
# --cookies (single file) or --cookies-list (comma-separated, in order).
COOKIES_ROTATION_DEFAULT = [
    os.path.join(COOKIES_DIR, 'backup1.txt'),
    os.path.join(COOKIES_DIR, 'backup2.txt'),
    os.path.join(COOKIES_DIR, 'backup3.txt'),
    os.path.join(COOKIES_DIR, 'main.txt'),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project', required=True, help='Project directory (must contain OTIO)')
    p.add_argument('--otio', help='OTIO filename inside project (default: downloadplease.otio)')
    p.add_argument('--hq-cache', help='HQ download directory (default: <project>/v/matcher-hq)')
    p.add_argument('--rewrite-only', action='store_true', help='Skip download, only rewrite OTIO')
    p.add_argument('--skip-rewrite', action='store_true', help='Download only, do not rewrite OTIO')
    p.add_argument('--validate-only', action='store_true', help='Only validate existing HQ OTIO')
    p.add_argument('--dry-run', action='store_true', help='Parse and report plan; do not download or rewrite')
    p.add_argument('--flatten', action='store_true',
                   help='After rewrite, emit <otio>_single.otio with top-most clips on one V1 track + voiceover audio')
    p.add_argument('--workers', type=int, default=1,
                   help='Parallel yt-dlp workers (default 1). '
                        '3-4 is safe; >6 risks YouTube rate limiting.')
    p.add_argument('--cookies', default=None,
                   help='Single cookies file (overrides rotation). Default: rotation through all .txt in cookies dir.')
    p.add_argument('--cookies-list', default=None,
                   help='Comma-separated cookies files to try in order. '
                        'First successful wins per clip.')
    return p.parse_args()


def load_cp(path):
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except UnicodeDecodeError:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None


def extract_videos(cp):
    if not cp or not isinstance(cp, dict):
        return {}
    vs = cp.get('video_search', {})
    if isinstance(vs, list):
        results = vs
    elif isinstance(vs, dict):
        results = vs.get('search_results', [])
    else:
        results = []
    out = {}
    for r in results:
        if isinstance(r, dict) and r.get('video_id'):
            url = r.get('url') or r.get('youtube_url')
            if url:
                out[r['video_id']] = url
    return out


def parse_clip_filename(fname):
    """Returns (video_id, start_sec, end_sec) or (None, None, None).

    Accepts both standard ('vid_ss_ee.mp4') and HQ ('vid_1080p_ss_ee.mp4') forms.
    """
    m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_(?:1080p_)?(\d+)_(\d+)\.mp4$', fname)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None


def collect_urls(project_dir):
    """Search project + gap_fill checkpoints (gzip-first) for YouTube URLs."""
    cp_files = (
        [os.path.join(project_dir, 'checkpoint.json'),
         os.path.join(project_dir, 'gap_fill', 'checkpoint.json')] +
        glob.glob(os.path.join(project_dir, 'gap_fill', 'checkpoint*.json')) +
        glob.glob(os.path.join(project_dir, 'gap_fill', 'checkpoint*.json.gz'))
    )
    url_map = {}
    for cp_path in cp_files:
        url_map.update(extract_videos(load_cp(cp_path)))
    return url_map, len(cp_files)


def find_top_most_clips(otio_path):
    """Per skill: only the highest-track enabled clip per moment is visible."""
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
            })
            tp = tp + clip.duration()

    unique_starts = sorted(set(c['start'] for c in all_clips))
    seen = set()
    results = []
    for seg_start in unique_starts:
        covering = [c for c in all_clips
                    if c['start'] <= seg_start < c['end']
                    and not c['is_gap'] and not c['is_audio'] and c['clip_enabled']
                    and isinstance(c['mr'], otio.schema.ExternalReference)]
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
        key = (vid, ss, ee)
        if key in seen:
            continue
        seen.add(key)
        results.append({'name': top['name'], 'vid': vid, 'ss': ss, 'ee': ee})
    return tl, results


def build_hq_index(hq_cache):
    """Map (vid, ss, ee) -> absolute path for HQ files with _1080p_ suffix."""
    index = {}
    for f in glob.glob(os.path.join(hq_cache, '*_1080p_*.mp4')):
        vid, ss, ee = parse_clip_filename(os.path.basename(f))
        if vid:
            index[(vid, ss, ee)] = f
    return index


def run_ytdlp(url, out_path, fmt, ss, ee, cookies, timeout=120):
    cmd = [
        'yt-dlp', '-f', fmt,
        '--download-sections', f'*{ss}-{ee}',
        '--merge-output-format', 'mp4',
        '--no-part',
        '-o', out_path,
        '--cookies', cookies,
        *EXTRACTOR_ARGS,
        url,
    ]
    # Use DEVNULL for stdout to avoid pipe-buffer deadlocks on long outputs;
    # capture stderr to a temp file we can read for the last error line.
    import tempfile
    stderr_fd, stderr_path = tempfile.mkstemp(prefix='ytdlp_err_', suffix='.log')
    os.close(stderr_fd)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=open(stderr_path, 'w', encoding='utf-8', errors='replace'),
            timeout=timeout,
        )
        with open(stderr_path, 'r', encoding='utf-8', errors='replace') as f:
            stderr = f.read()
        return subprocess.CompletedProcess(cmd, returncode=proc.returncode,
                                          stdout='', stderr=stderr)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, returncode=-1,
                                          stdout='', stderr='TIMEOUT after 120s')
    finally:
        try:
            os.unlink(stderr_path)
        except OSError:
            pass


_ROTATION_COUNTER = [0]  # mutable; advances after each clip attempt


def _download_one(clip, url_map, hq_cache, cookies_list):
    """Download a single clip with strict round-robin cookie rotation.

    Per attempt: start with cookies_list[counter % N]. If that cookie fails
    (both HI and LO formats), try the next cookie in the list as a fallback
    before declaring failure. The counter always advances, so every clip
    uses a different starting cookie — rotation happens on success and on
    retry alike.

    Returns (out_name, ok, err_msg, cookie_used).
    """
    vid, ss, ee = clip['vid'], clip['ss'], clip['ee']
    out_name = f'{vid}_1080p_{ss}_{ee}.mp4'
    out_path = os.path.join(hq_cache, out_name)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        _ROTATION_COUNTER[0] += 1
        return out_name, True, '', None

    n = len(cookies_list)
    start = _ROTATION_COUNTER[0] % n
    _ROTATION_COUNTER[0] += 1

    last_err = ''
    for offset in range(n):
        cookies = cookies_list[(start + offset) % n]
        r = run_ytdlp(url_map[vid], out_path, FMT_HI, ss, ee, cookies)
        if r.returncode != 0:
            r = run_ytdlp(url_map[vid], out_path, FMT_LO, ss, ee, cookies)
        if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            return out_name, True, '', os.path.basename(cookies)
        last_err = (r.stderr or '').strip()[-120:]
    return out_name, False, last_err, None


def download_missing(clips, url_map, hq_cache, hq_index, workers=1, cookies_list=None):
    """Download clips not yet in HQ cache. Returns (ok, fail, fail_names)."""
    cookies_list = cookies_list or [COOKIES_DEFAULT]
    to_dl = [c for c in clips
             if (c['vid'], c['ss'], c['ee']) not in hq_index
             and c['vid'] in url_map]
    no_url = [c for c in clips
              if (c['vid'], c['ss'], c['ee']) not in hq_index
              and c['vid'] not in url_map]
    print('BEFORE DOWNLOAD LOOP', flush=True); import sys; sys.stdout.flush()
    print(f'Need to download: {len(to_dl)} clips; {len(no_url)} have no URL (skipping)',
          flush=True)
    print(f'Cookies ({len(cookies_list)}, try in order):', flush=True)
    for c in cookies_list:
        print(f'  - {os.path.basename(c)}', flush=True)

    if workers > 1:
        return _download_parallel(to_dl, url_map, hq_cache, workers, cookies_list)

    ok = fail = 0
    fail_names = []
    for i, clip in enumerate(to_dl, 1):
        vid, ss, ee = clip['vid'], clip['ss'], clip['ee']
        out_name = f'{vid}_1080p_{ss}_{ee}.mp4'

        if os.path.exists(os.path.join(hq_cache, out_name)) and os.path.getsize(os.path.join(hq_cache, out_name)) > 1000:
            ok += 1
            continue

        print(f'[{i}/{len(to_dl)}] {out_name} ...', flush=True, end='')
        _, ok_dl, err, ck = _download_one(clip, url_map, hq_cache, cookies_list)
        if ok_dl:
            print(f' OK (cookie={ck})', flush=True)
            ok += 1
        else:
            print(f' FAIL: {err}', flush=True)
            fail += 1
            fail_names.append(out_name)

    print(f'\n=== Download Summary ===')
    print(f'OK: {ok}  FAIL: {fail}  NO-URL: {len(no_url)}')
    if fail_names:
        print('Failed files:')
        for f in fail_names:
            print(f'  {f}')
    return ok, fail, fail_names


def _download_parallel(to_dl, url_map, hq_cache, workers, cookies_list):
    """Download clips in parallel. Each worker is pinned to a unique primary
    cookie (worker i -> cookies_list[i % N]), so concurrent downloads never
    compete on the same auth token. If the primary fails, the worker falls
    back through the remaining cookies in order.
    """
    import threading
    counter = [0, 0, 0]  # [done, ok, fail]
    lock = threading.Lock()
    fail_names = []
    n_cookies = len(cookies_list)

    def task(worker_idx, clip):
        # Rotate primary cookie per worker to spread load across auth tokens.
        primary = cookies_list[worker_idx % n_cookies]
        ordered = [primary] + [c for c in cookies_list if c != primary]
        out_name, ok, err, ck = _download_one_ordered(clip, url_map, hq_cache, ordered)
        with lock:
            counter[0] += 1
            n = counter[0]
            ck_label = ck or '-'
            if ok:
                counter[1] += 1
                print(f'[{n}/{len(to_dl)}] W{worker_idx} {out_name} ... OK (cookie={ck_label})', flush=True)
            else:
                counter[2] += 1
                fail_names.append(out_name)
                print(f'[{n}/{len(to_dl)}] W{worker_idx} {out_name} ... FAIL: {err}', flush=True)

    print(f'Running {workers} workers in parallel (each pinned to a unique primary cookie):', flush=True)
    for i in range(workers):
        print(f'  W{i} primary: {os.path.basename(cookies_list[i % n_cookies])}', flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(task, i % workers, c) for i, c in enumerate(to_dl)]
        for f in as_completed(futures):
            f.result()  # propagate exceptions

    ok, fail = counter[1], counter[2]
    print(f'\n=== Download Summary ===')
    print(f'OK: {ok}  FAIL: {fail}')
    if fail_names:
        print('Failed files:')
        for f in fail_names:
            print(f'  {f}')
    return ok, fail, fail_names


def _download_one_ordered(clip, url_map, hq_cache, cookies_list):
    """Download a single clip with a fixed cookie order (no rotation).
    Used by parallel workers to keep their pinned primary cookie. Returns
    (out_name, ok, err_msg, cookie_used).
    """
    vid, ss, ee = clip['vid'], clip['ss'], clip['ee']
    out_name = f'{vid}_1080p_{ss}_{ee}.mp4'
    out_path = os.path.join(hq_cache, out_name)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return out_name, True, '', None

    last_err = ''
    for cookies in cookies_list:
        r = run_ytdlp(url_map[vid], out_path, FMT_HI, ss, ee, cookies)
        if r.returncode != 0:
            r = run_ytdlp(url_map[vid], out_path, FMT_LO, ss, ee, cookies)
        if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            return out_name, True, '', os.path.basename(cookies)
        last_err = (r.stderr or '').strip()[-120:]
    return out_name, False, last_err, None


def rewrite_otio(timeline, hq_index, out_path, topmost_keys=None):
    """Swap media_references for clips that have HQ files.

    Only swaps clips that are enabled, on an enabled track, and whose
    (vid, ss, ee) appears as a topmost-visible clip somewhere in the
    timeline. Disabled clips and never-visible alt-tier clips are left
    untouched.
    """
    replaced = 0
    for track in timeline.tracks:
        if not track.enabled:
            continue
        for child in track:
            if not isinstance(child, otio.schema.Clip):
                continue
            if not getattr(child, 'enabled', True):
                continue
            mr = child.media_reference
            if isinstance(mr, otio.schema.ExternalReference) and mr.target_url:
                fname = os.path.basename(mr.target_url)
            else:
                fname = child.name
            vid, ss, ee = (None,)*3
            m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_(?:1080p_)?(\d+)_(\d+)\.mp4$', fname)
            if m:
                vid, ss, ee = m.group(1), int(m.group(2)), int(m.group(3))
            if not vid or (vid, ss, ee) not in hq_index:
                continue
            if topmost_keys is not None and (vid, ss, ee) not in topmost_keys:
                continue
            new_path = hq_index[(vid, ss, ee)]
            clip_dur_frames = int((ee - ss) * RATE)
            new_mr = otio.schema.ExternalReference(
                target_url=new_path,
                available_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, RATE),
                    duration=otio.opentime.RationalTime(clip_dur_frames, RATE),
                )
            )
            new_mr.name = os.path.basename(new_path)
            child.media_reference = new_mr
            child.metadata['Resolve_OTIO'] = {}
            replaced += 1

    otio.adapters.write_to_file(timeline, out_path)
    return replaced


def validate_otio(otio_path, hq_cache_dir):
    """Verify all HQ paths exist and available_range matches section length."""
    tl = otio.adapters.read_from_file(otio_path)
    total = 0
    hq_refs = 0
    missing = 0
    duration_mismatch = 0
    for track in tl.tracks:
        for clip in track:
            if not isinstance(clip, otio.schema.Clip):
                continue
            mr = clip.media_reference
            total += 1
            if not isinstance(mr, otio.schema.ExternalReference) or not mr.target_url:
                continue
            target = mr.target_url.replace('file:///', '').replace('file:', '')
            target = target.replace('\\', '/')
            if hq_cache_dir.replace('\\', '/') not in target:
                continue
            hq_refs += 1
            if not os.path.exists(target):
                missing += 1
                continue
            fname = os.path.basename(target)
            vid, ss, ee = parse_clip_filename(fname)
            if not vid:
                continue
            expected_frames = int((ee - ss) * RATE)
            actual_frames = int(mr.available_range.duration.value) if mr.available_range else 0
            if abs(expected_frames - actual_frames) > 0:
                duration_mismatch += 1

    print(f'Validation: {total} clips total, {hq_refs} HQ refs, '
          f'{missing} missing paths, {duration_mismatch} duration mismatches')
    return missing == 0 and duration_mismatch == 0


def main():
    args = parse_args()
    project_dir = args.project
    otio_name = args.otio or 'downloadplease.otio'
    otio_path = os.path.join(project_dir, otio_name)
    hq_cache = args.hq_cache or os.path.join(project_dir, 'v', 'matcher-hq')
    out_otio = otio_path.replace('.otio', '_hq.otio')

    if not os.path.exists(otio_path):
        print(f'ERROR: OTIO not found: {otio_path}', file=sys.stderr)
        sys.exit(1)

    hq_cache = os.path.normpath(hq_cache)
    os.makedirs(hq_cache, exist_ok=True)

    print(f'Project: {project_dir}')
    print(f'OTIO:    {otio_path}')
    print(f'HQ dir:  {hq_cache}')

    if args.validate_only:
        validate_otio(out_otio, hq_cache)
        return

    url_map, n_cps = collect_urls(project_dir)
    print(f'Loaded {len(url_map)} video URLs from {n_cps} checkpoints', flush=True)

    timeline, clips = find_top_most_clips(otio_path)
    print(f'Top-most visible clips: {len(clips)}', flush=True)

    hq_index = build_hq_index(hq_cache)
    print(f'Existing HQ cache: {len(hq_index)} files', flush=True)

    if args.dry_run:
        to_dl = [c for c in clips
                 if (c['vid'], c['ss'], c['ee']) not in hq_index and c['vid'] in url_map]
        cached = [c for c in clips if (c['vid'], c['ss'], c['ee']) in hq_index]
        no_url = [c for c in clips
                  if (c['vid'], c['ss'], c['ee']) not in hq_index and c['vid'] not in url_map]
        print(f'\n=== Dry-run Plan ===')
        print(f'Would download: {len(to_dl)} clips')
        print(f'Already cached: {len(cached)} clips')
        print(f'No URL (skip):  {len(no_url)} clips')
        print(f'Would rewrite:  {len(cached)} cached clips have HQ refs available')
        if to_dl[:5]:
            print(f'\nFirst 5 to download:')
            for c in to_dl[:5]:
                print(f'  {c["vid"]}_{c["ss"]}_{c["ee"]}')
        if no_url[:5]:
            print(f'\nFirst 5 with no URL:')
            for c in no_url[:5]:
                print(f'  {c["vid"]}_{c["ss"]}_{c["ee"]}')
        return

    if args.cookies:
        cookies_list = [args.cookies]
    elif args.cookies_list:
        cookies_list = [c.strip() for c in args.cookies_list.split(',') if c.strip()]
    else:
        cookies_list = COOKIES_ROTATION_DEFAULT

    if not args.rewrite_only:
        download_missing(clips, url_map, hq_cache, hq_index,
                         workers=args.workers, cookies_list=cookies_list)
        hq_index = build_hq_index(hq_cache)  # refresh

    if not args.skip_rewrite:
        topmost_keys = set((c['vid'], c['ss'], c['ee']) for c in clips)
        replaced = rewrite_otio(timeline, hq_index, out_otio,
                                topmost_keys=topmost_keys)
        print(f'\nRewrote {replaced} clip refs -> {out_otio}')
        print(f'Size: {os.path.getsize(out_otio)/1024:.0f} KB')
        validate_otio(out_otio, hq_cache)

        if args.flatten:
            import importlib.util
            _spec = importlib.util.spec_from_file_location(
                'flatten_otio_to_single_track',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flatten-otio-to-single-track.py'),
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            do_flatten = _mod.flatten
            single_out = out_otio.replace('.otio', '_single.otio')
            do_flatten(out_otio, single_out)
            print(f'\nFlattened single-track OTIO -> {single_out}')


if __name__ == '__main__':
    main()