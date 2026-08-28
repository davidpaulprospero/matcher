"""Batch audit: for every clip_*.mp4, check whether its first/last frame matches
the corresponding HQ source frame at the manifest-specified offset/duration.

Method: decode both sides to PNG (lossless, codec-independent), then compare
with PSNR. Low PSNR at the expected timestamp means the cut came from the
wrong source or wrong offset — flag it for black-video replacement.

Concurrency: ThreadPoolExecutor, max 8 workers (CPU + ffmpeg I/O).
"""
import csv
import os
import sys
import subprocess
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np

PROJECT = r'E:\Edit Job\Samples Projects\Fransisco_2026-07-09\Fransisco'
MANIFEST = os.path.join(PROJECT, 'capcut_import', 'manifest.csv')
HQ_DIR = os.path.join(PROJECT, 'v', 'matcher-hq')
OUT_DIR = os.path.join(PROJECT, 'capcut_import')
TMP = r'C:\Users\daves\AppData\Local\Temp\capcut_audit_frames'
os.makedirs(TMP, exist_ok=True)

# Threshold (dB) below which we declare a mismatch
THRESHOLD_DB = 18.0
# Re-probe tolerance (seconds) for source-side sweep before declaring a miss
SWEEP_STARTS = [0.0, 0.5, -0.5, 1.0, -1.0]


def ffmpeg_frame(src, offset, out_png):
    """Extract one frame at offset as PNG. Returns True on success.

    NOTE: -ss MUST come AFTER -i (lesson #13 / skill lesson #10). Putting -ss
    before -i triggers ffmpeg's keyframe-seek, which snaps to the nearest
    keyframe BEFORE the offset and yields the wrong frame. This is used for
    SWEEP_STARTS verification — false sweep results would let real bad_start
    clips slip through as "near-miss."
    """
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-i', src,
           '-ss', f'{offset:.3f}',
           '-vframes', '1', out_png]
    try:
        subprocess.run(cmd, timeout=30, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(out_png) and os.path.getsize(out_png) > 1000
    except Exception:
        return False


def psnr(a_path, b_path):
    """Compute PSNR between two PNGs. Returns (psnr_db, mse) or (None, None)."""
    try:
        a = np.asarray(Image.open(a_path).convert('RGB'), dtype=np.float64)
        b = np.asarray(Image.open(b_path).convert('RGB'), dtype=np.float64)
        if a.shape != b.shape:
            return None, None
        mse = float(np.mean((a - b) ** 2))
        if mse < 1e-3:
            return 99.0, mse
        return 10 * math.log10(255 * 255 / mse), mse
    except Exception:
        return None, None


def check_clip(row):
    """Returns (index, status, info_dict) for one clip."""
    idx = int(row['index'])
    src_name = row['source_file']
    offset = float(row['offset_sec'])
    duration = float(row['duration_sec'])
    src_path = os.path.join(HQ_DIR, src_name)
    clip_path = os.path.join(OUT_DIR, row['output'])

    info = {'idx': idx, 'src': src_name, 'offset': offset, 'dur': duration,
            'src_exists': os.path.exists(src_path),
            'clip_exists': os.path.exists(clip_path)}

    # Source exists check
    if not info['src_exists']:
        info['reason'] = 'no_source'
        return idx, 'no_source', info

    if not info['clip_exists']:
        info['reason'] = 'no_export'
        return idx, 'no_export', info

    # Extract first/last frames from export
    exp_first = os.path.join(TMP, f'exp_{idx:04d}_first.png')
    exp_last = os.path.join(TMP, f'exp_{idx:04d}_last.png')
    if not ffmpeg_frame(clip_path, 0.0, exp_first):
        info['reason'] = 'export_unreadable'
        return idx, 'export_unreadable', info
    # Last frame: try -sseof first (most reliable)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-sseof', '-0.1',
           '-i', clip_path, '-update', '1', '-vframes', '1', exp_last]
    try:
        subprocess.run(cmd, timeout=30, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # Fallback: duration-relative
        ffmpeg_frame(clip_path, max(0, duration - 0.1), exp_last)

    # Source-side first frame at expected offset (frame-accurate: -ss AFTER -i).
    src_first = os.path.join(TMP, f'src_{idx:04d}_first.png')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-i', src_path,
           '-ss', f'{offset:.3f}',
           '-vframes', '1', src_first]
    try:
        subprocess.run(cmd, timeout=30, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    if not (os.path.exists(src_first) and os.path.getsize(src_first) > 1000):
        info['reason'] = 'src_unreadable'
        return idx, 'src_unreadable', info

    start_psnr, start_mse = psnr(exp_first, src_first)

    # Source-side last frame at offset+duration (frame-accurate: -ss AFTER -i).
    # Avoid `-ss BEFORE -i` (keyframe seek) — same bug that broke the export cut.
    src_last = os.path.join(TMP, f'src_{idx:04d}_last.png')
    src_end = offset + duration
    # Probe one frame BEFORE the requested end (last frame of the cut window).
    # Going exactly to the boundary can land on the next GOP / black frame.
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-i', src_path,
           '-ss', f'{max(0.0, src_end - 0.033):.3f}',
           '-vframes', '1', src_last]
    try:
        subprocess.run(cmd, timeout=30, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    if not (os.path.exists(src_last) and os.path.getsize(src_last) > 1000):
        # Fallback: actual last frame of source (in case cut window reaches EOF)
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-sseof', '-0.1',
               '-i', src_path, '-update', '1', '-vframes', '1', src_last]
        try:
            subprocess.run(cmd, timeout=30, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    end_psnr, end_mse = psnr(exp_last, src_last) if os.path.exists(src_last) else (None, None)

    info['start_psnr'] = start_psnr
    info['end_psnr'] = end_psnr

    start_ok = start_psnr is not None and start_psnr >= THRESHOLD_DB
    end_ok = end_psnr is not None and end_psnr >= THRESHOLD_DB

    if start_ok and end_ok:
        info['reason'] = 'ok'
        return idx, 'ok', info

    # Mismatch: try sweep on start to rule out near-miss
    if not start_ok:
        best_sweep = start_psnr if start_psnr is not None else -1.0
        for delta in SWEEP_STARTS:
            sweep_off = offset + delta
            if sweep_off < 0:
                continue
            sweep_png = os.path.join(TMP, f'src_{idx:04d}_sweep.png')
            if ffmpeg_frame(src_path, sweep_off, sweep_png):
                p, _ = psnr(exp_first, sweep_png)
                if p is not None and p > best_sweep:
                    best_sweep = p
        info['start_best_psnr'] = best_sweep if best_sweep >= 0 else None
        # If best sweep also fails, hard mismatch
        if best_sweep < THRESHOLD_DB:
            psnr_str = f'{best_sweep:.1f}' if best_sweep >= 0 else 'N/A'
            info['reason'] = f'start_mismatch(PSNR={psnr_str})'
            return idx, 'bad_start', info
        # Otherwise it was a near-miss (different exact offset), still flag
        info['reason'] = f'start_near_miss(PSNR={best_sweep:.1f})'
        return idx, 'bad_start', info

    psnr_str = f'{end_psnr:.1f}' if end_psnr is not None else 'N/A'
    info['reason'] = f'end_mismatch(PSNR={psnr_str})'
    return idx, 'bad_end', info


def main():
    with open(MANIFEST, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['status'] in ('ok', 'skip')]

    print(f'Auditing {len(rows)} clips (threshold PSNR {THRESHOLD_DB} dB) ...')
    t0 = time.time()

    bad = []
    counts = {'ok': 0, 'bad_start': 0, 'bad_end': 0,
              'no_source': 0, 'no_export': 0,
              'export_unreadable': 0, 'src_unreadable': 0}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(check_clip, r): r for r in rows}
        done = 0
        for fut in as_completed(futures):
            done += 1
            idx, status, info = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status in ('bad_start', 'bad_end'):
                bad.append(info)
            if done % 25 == 0 or done == len(rows):
                elapsed = time.time() - t0
                print(f'  [{done}/{len(rows)}] {elapsed:.0f}s  bad={len(bad)}', flush=True)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.0f}s')
    print(f'OK:              {counts["ok"]}')
    print(f'Bad start:       {counts["bad_start"]}')
    print(f'Bad end:         {counts["bad_end"]}')
    print(f'No source:       {counts["no_source"]}')
    print(f'No export:       {counts["no_export"]}')
    print(f'Unreadable:      {counts["export_unreadable"]} exp / {counts["src_unreadable"]} src')

    out_csv = os.path.join(OUT_DIR, 'audit_bad_clips.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['idx', 'src', 'offset', 'dur',
                                          'start_psnr', 'end_psnr', 'reason'])
        w.writeheader()
        for b in sorted(bad, key=lambda x: x['idx']):
            w.writerow({k: b.get(k, '') for k in w.fieldnames})
    print(f'\nBad-clip list: {out_csv}')


if __name__ == '__main__':
    main()