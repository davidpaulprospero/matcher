import opentimelineio as otio
import re
import os
import gzip
import json
import glob
import subprocess
import time
import pickle

OTIO_PATH = r"E:\Edit Job\Kyteq\Disney\OJNZ7wJZ-How Big is Disney World's Bus Fleet__2026-05-05\a.otio"
PROJECT_DIR = r"E:\Edit Job\Kyteq\Disney\OJNZ7wJZ-How Big is Disney World's Bus Fleet__2026-05-05"
HQ_CACHE = os.path.join(PROJECT_DIR, "v", "matcher-hq")
os.makedirs(HQ_CACHE, exist_ok=True)

def parse_clip_filename(filename):
    m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_(\d+)_(\d+)\.mp4$', filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None

def find_youtube_url(video_id, project_dir):
    checkpoints = [
        os.path.join(project_dir, 'checkpoint.json'),
        os.path.join(project_dir, 'gap_fill', 'checkpoint.json'),
        os.path.join(project_dir, 'gap_fill', 'checkpoint.backup_retry.json'),
        os.path.join(project_dir, 'gap_fill', 'checkpoint.pre_fix.json'),
        os.path.join(project_dir, 'gap_fill', 'checkpoint.pre_output_fix.json'),
    ]
    for cp_path in checkpoints:
        if not os.path.exists(cp_path):
            continue
        try:
            with gzip.open(cp_path, 'rt', encoding='utf-8') as f:
                cp = json.load(f)
        except (gzip.BadGzipFile, OSError):
            try:
                with open(cp_path, 'r', encoding='utf-8') as f:
                    cp = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                continue
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                with open(cp_path, 'r', encoding='utf-8') as f:
                    cp = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                continue
        for r in cp.get('video_search', {}).get('search_results', []):
            if r.get('video_id') == video_id:
                return r.get('url') or r.get('youtube_url')
    return None

def get_topmost_clips(tl):
    all_clips = []
    for track_idx, track in enumerate(tl.tracks):
        if not track.enabled:
            continue
        is_audio = track.kind == otio.schema.TrackKind.Audio
        tp = otio.opentime.RationalTime(0, 30)
        for clip in track:
            clip_enabled = getattr(clip, 'enabled', True)
            dur = clip.duration()
            all_clips.append({
                'track_idx': track_idx,
                'track': track.name,
                'name': clip.name,
                'is_audio': is_audio,
                'is_gap': isinstance(clip, otio.schema.Gap),
                'clip_enabled': clip_enabled,
                'start': tp.value,
                'end': (tp + dur).value,
            })
            tp = tp + dur
    all_clips.sort(key=lambda x: x['start'])
    unique_starts = sorted(set(c['start'] for c in all_clips))
    results = []
    for seg_start in unique_starts:
        covering = [c for c in all_clips
                    if c['start'] <= seg_start < c['end']
                    and not c['is_gap']
                    and not c['is_audio']
                    and c['clip_enabled']]
        if not covering:
            continue
        top = max(covering, key=lambda x: x['track_idx'])
        vid, ss, ee = parse_clip_filename(top['name'])
        results.append({**top, 'vid': vid, 'ss': ss, 'ee': ee})
    return results

print("Loading OTIO...")
tl = otio.adapters.read_from_file(OTIO_PATH)
topmost = get_topmost_clips(tl)
youtube_clips = [c for c in topmost if c['vid'] is not None]

# Deduplicate
seen = set()
unique_clips = []
for c in youtube_clips:
    key = (c['vid'], c['ss'], c['ee'])
    if key not in seen:
        seen.add(key)
        unique_clips.append(c)

print(f"Unique clips to download: {len(unique_clips)}")

# Build existing HQ index
hq_index = {}
if os.path.exists(HQ_CACHE):
    for f in glob.glob(os.path.join(HQ_CACHE, '*_1080p_*.mp4')) + glob.glob(os.path.join(HQ_CACHE, '*_hq_*.mp4')):
        m = re.match(r'^(-?[a-zA-Z0-9_-]{11})_(1080p|hq)_(\d+)_(\d+)\.mp4', os.path.basename(f))
        if m:
            hq_index[(m.group(1), int(m.group(3)), int(m.group(4)))] = f

# Filter out already-cached
to_download = [c for c in unique_clips if (c['vid'], c['ss'], c['ee']) not in hq_index]
print(f"Already cached: {len(unique_clips) - len(to_download)}")
print(f"Need to download: {len(to_download)}")

# Save for reference
with open(os.path.join(PROJECT_DIR, 'download_manifest.pkl'), 'wb') as f:
    pickle.dump(to_download, f)
print(f"Saved manifest with {len(to_download)} clips")
print(f"Starting downloads to {HQ_CACHE}")
print("=" * 60)

# Download loop
success = 0
failed = []
for i, clip in enumerate(to_download):
    vid, ss, ee = clip['vid'], clip['ss'], clip['ee']
    url = find_youtube_url(vid, PROJECT_DIR)
    if not url:
        print(f"[{i+1}/{len(to_download)}] SKIP {vid} ss={ss} ee={ee} — URL not found")
        failed.append({**clip, 'reason': 'no_url'})
        continue

    output = os.path.join(HQ_CACHE, f"{vid}_1080p_{ss}_{ee}.mp4")

    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4][height>=720]+bestaudio[ext=m4a]',
        '--download-sections', f'*{ss}-{ee}',
        '--merge-output-format', 'mp4',
        '--no-part',
        '-o', output,
        url
    ]

    print(f"[{i+1}/{len(to_download)}] {vid} ss={ss} ee={ee}...", end=' ', flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    if result.returncode == 0 and os.path.exists(output):
        print(f"OK ({os.path.getsize(output) // 1024}KB)")
        success += 1
    else:
        print(f"FAIL (exit {result.returncode})")
        failed.append({**clip, 'url': url, 'reason': 'download_failed', 'stderr': result.stderr[-200:] if result.stderr else ''})

print("=" * 60)
print(f"First pass complete: {success}/{len(to_download)} succeeded, {len(failed)} failed")

# Retry failed clips
if failed:
    print(f"\nRetrying {len(failed)} failed clips...")
    retry_success = 0
    retry_failed = []
    for i, clip in enumerate(failed):
        vid, ss, ee = clip['vid'], clip['ss'], clip['ee']
        url = clip.get('url') or find_youtube_url(vid, PROJECT_DIR)
        if not url:
            retry_failed.append({**clip, 'reason': 'no_url_retry'})
            continue

        output = os.path.join(HQ_CACHE, f"{vid}_1080p_{ss}_{ee}.mp4")

        cmd = [
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4][height>=720]+bestaudio[ext=m4a]',
            '--download-sections', f'*{ss}-{ee}',
            '--merge-output-format', 'mp4',
            '--no-part',
            '-o', output,
            url
        ]

        print(f"[retry {i+1}/{len(failed)}] {vid} ss={ss} ee={ee}...", end=' ', flush=True)

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

        if result.returncode == 0 and os.path.exists(output):
            print(f"OK ({os.path.getsize(output) // 1024}KB)")
            retry_success += 1
        else:
            print(f"FAIL")
            retry_failed.append({**clip, 'reason': 'retry_failed'})

    print(f"\nRetry complete: {retry_success}/{len(failed)} recovered, {len(retry_failed)} still failing")
    final_success = success + retry_success
    final_failed = retry_failed
else:
    final_success = success
    final_failed = failed

print(f"\n=== FINAL: {final_success}/{len(to_download)} downloaded ({len(final_failed)} failed) ===")
if final_failed:
    print("Failed clips:")
    for c in final_failed:
        print(f"  {c['vid']} ss={c['ss']} ee={c['ee']} — {c.get('reason', 'unknown')}")