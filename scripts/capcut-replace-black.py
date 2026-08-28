"""Replace flagged bad clips with a black video of the exact OTIO duration.

For each row in audit_bad_clips.csv:
  - Read duration_sec from the manifest (the OTIO contract; never the probed actual_dur)
  - Generate black: ffmpeg -f lavfi -i color=c=black:s=1920x1080:r=30 -t D -c:v libx264 -crf 18 -preset fast
                       -c:a aac -b:a 192k -movflags +faststart clip_NNNN.mp4
  - Overwrite in place
  - Update manifest: status=black, actual_dur=<new probed dur>
"""
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = r'E:\Edit Job\Samples Projects\Fransisco_2026-07-09\Fransisco'
MANIFEST = os.path.join(PROJECT, 'capcut_import', 'manifest.csv')
BADLIST = os.path.join(PROJECT, 'capcut_import', 'audit_bad_clips.csv')
OUT_DIR = os.path.join(PROJECT, 'capcut_import')

BLACK_CRF = '18'
BLACK_PRESET = 'fast'
BLACK_ABR = '192k'
W, H, FPS = 1920, 1080, 30


def make_black(dst, duration):
    """Generate a black MP4 of `duration` seconds at 1920x1080 30fps."""
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'lavfi', '-i', f'color=c=black:s={W}x{H}:r={FPS}',
        '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
        '-t', f'{duration:.3f}',
        '-c:v', 'libx264', '-preset', BLACK_PRESET, '-crf', BLACK_CRF,
        '-c:a', 'aac', '-b:a', BLACK_ABR,
        '-shortest', '-movflags', '+faststart',
        dst,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    return r.returncode == 0, (r.stderr or '').strip()[-200:]


def probe_duration(path):
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


def replace_one(idx, duration):
    dst = os.path.join(OUT_DIR, f'clip_{idx:04d}.mp4')
    ok, err = make_black(dst, duration)
    if not ok:
        return idx, False, f'ffmpeg fail: {err}'
    actual = probe_duration(dst)
    drift = (actual - duration) if actual is not None else None
    if actual is None:
        return idx, False, 'unreadable after gen'
    if abs(drift) > 0.1:
        return idx, False, f'drift {drift:+.3f}s exceeds 0.1s'
    return idx, True, f'{actual:.3f}s (drift {drift:+.3f}s)'


def main():
    if not os.path.exists(BADLIST):
        print(f'ERROR: {BADLIST} not found', file=sys.stderr)
        sys.exit(1)

    with open(BADLIST, newline='', encoding='utf-8') as f:
        bad = list(csv.DictReader(f))
    print(f'Replacing {len(bad)} bad clips with black video...')
    t0 = time.time()

    ok_results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(replace_one, int(b['idx']), float(b['dur'])): b for b in bad}
        done = 0
        for fut in as_completed(futures):
            done += 1
            idx, ok, msg = fut.result()
            ok_results[idx] = (ok, msg)
            if done % 10 == 0 or done == len(bad):
                ok_count = sum(1 for v in ok_results.values() if v[0])
                print(f'  [{done}/{len(bad)}] ok={ok_count}  elapsed={time.time()-t0:.0f}s', flush=True)

    # Update manifest
    rows = []
    with open(MANIFEST, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    updated = 0
    for r in rows:
        idx = int(r['index'])
        if idx in ok_results:
            ok, msg = ok_results[idx]
            if ok:
                r['status'] = 'black'
                r['actual_dur'] = msg.split('s')[0]
                updated += 1

    with open(MANIFEST, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    elapsed = time.time() - t0
    n_ok = sum(1 for v in ok_results.values() if v[0])
    n_fail = len(bad) - n_ok
    print(f'\nDone in {elapsed:.0f}s')
    print(f'Replaced OK:   {n_ok}')
    print(f'Failed:        {n_fail}')
    print(f'Manifest updated: {updated} rows -> status=black')

    if n_fail:
        print('\nFailures:')
        for idx, (ok, msg) in ok_results.items():
            if not ok:
                print(f'  clip_{idx:04d}: {msg}')


if __name__ == '__main__':
    main()