#!/usr/bin/env python3
"""
Fast channel_id fix with larger batches and periodic saves.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import shutil
import time
# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)


# Import standardized output functions
from script_utils import (
    print_header, print_ok, print_warn, print_error, print_info,
    set_verbosity
)

def fetch_video_metadata_batch(video_ids: List[str]) -> Dict[str, dict]:
    """Fetch metadata for multiple videos using yt-dlp."""
    if not video_ids:
        return {}
    
    urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
    
    cmd = [
        'yt-dlp',
        '--ignore-config',
        '--dump-json',
        '--no-download',
        '--quiet',
    ] + urls
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 min timeout per batch
            encoding='utf-8',
            errors='ignore'
        )
        
        metadata = {}
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                data = json.loads(line)
                vid = data.get('id', '')
                if vid:
                    metadata[vid] = {
                        'channel_id': data.get('channel_id', data.get('uploader_id', '')),
                        'channel': data.get('channel', data.get('uploader', '')),
                    }
            except json.JSONDecodeError:
                continue
        
        return metadata
        
    except Exception as e:
        print(f"    Warning: Batch fetch failed: {e}")
        return {}

def fix_checkpoint(checkpoint_path: str, json_mode: bool = False):
    """Fix missing channel_id with periodic saves."""
    path = Path(checkpoint_path)
    if not path.exists():
        msg = f"ERROR: Checkpoint not found: {path}"
        if json_mode:
            print(json.dumps({"error": msg, "success": False}))
            return False
        print(msg)
        return False

    json_output = {"success": False, "videos_fixed": 0, "videos_failed": 0, "total_processed": 0, "duration": 0}

    if json_mode:
        print(json.dumps({"status": "loading", "checkpoint": str(path)}))
    else:
        print(f"\nLoading checkpoint: {path}")

    with open(path, 'r') as f:
        data = json.load(f)

    download = data.get('download', {})
    downloaded_videos = download.get('downloaded_videos', [])

    if not json_mode:
        print(f"Total videos: {len(downloaded_videos)}")

    # Create backup if not exists
    backup_path = path.with_suffix('.json.backup')
    if not backup_path.exists():
        if json_mode:
            print(json.dumps({"status": "backup_created", "path": str(backup_path)}))
        else:
            print(f"Creating backup: {backup_path}")
        shutil.copy2(path, backup_path)

    # Find videos without channel_id
    videos_to_fix = []
    failed_videos = []
    for i, v in enumerate(downloaded_videos):
        if not v.get('channel_id'):
            url = v.get('url', '')
            if 'v=' in url:
                vid = url.split('v=')[-1].split('&')[0]
                if len(vid) == 11:
                    videos_to_fix.append((i, vid))

    if not videos_to_fix:
        msg = "✅ All videos already have channel_id!"
        if json_mode:
            print(json.dumps({"success": True, "message": msg, "videos_fixed": 0, "videos_failed": 0, "total_processed": 0, "duration": 0}))
        else:
            print(msg)
        return True

    if not json_mode:
        print(f"Need to fix: {len(videos_to_fix)} videos")
        estimated_batches = (len(videos_to_fix) + 19) // 20
        print(f"Estimated: {estimated_batches} batches (~{estimated_batches * 5 // 60 + 1} minutes)\n")

    # Process in batches of 20 (smaller for reliability)
    batch_size = 20
    total_fixed = 0
    total_failed = 0
    start_time = time.time()

    for batch_num, batch_start in enumerate(range(0, len(videos_to_fix), batch_size), 1):
        batch = videos_to_fix[batch_start:batch_start + batch_size]
        video_ids = [vid for _, vid in batch]

        if json_mode:
            print(json.dumps({"status": "processing_batch", "batch": batch_num, "total_batches": (len(videos_to_fix)-1)//batch_size + 1, "videos": video_ids}))
        else:
            print(f"Batch {batch_num}/{(len(videos_to_fix)-1)//batch_size + 1}: Fetching {len(batch)} videos...", end=' ', flush=True)

        metadata = fetch_video_metadata_batch(video_ids)

        # Update checkpoint data
        fixed_in_batch = 0
        for idx, vid in batch:
            if vid in metadata:
                downloaded_videos[idx]['channel_id'] = metadata[vid]['channel_id']
                if metadata[vid]['channel']:
                    downloaded_videos[idx]['channel'] = metadata[vid]['channel']
                fixed_in_batch += 1
                total_fixed += 1
            else:
                total_failed += 1
                failed_videos.append(vid)

        if json_mode:
            print(json.dumps({"status": "batch_complete", "batch": batch_num, "fixed": fixed_in_batch}))
        else:
            print(f"✓ Fixed {fixed_in_batch}")

        # Save every 25 batches (500 videos)
        if batch_num % 25 == 0:
            if json_mode:
                print(json.dumps({"status": "saving_checkpoint", "progress": f"{total_fixed}/{len(videos_to_fix)}"}))
            else:
                print(f"  Saving checkpoint (progress: {total_fixed}/{len(videos_to_fix)})...")
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            elapsed = time.time() - start_time
            if not json_mode:
                print(f"  Elapsed: {elapsed//60:.0f}m {elapsed%60:.0f}s")

    # Final save
    if json_mode:
        print(json.dumps({"status": "saving_checkpoint", "progress": "final"}))
    else:
        print(f"\nSaving final checkpoint...")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    elapsed = time.time() - start_time

    if json_mode:
        json_output = {
            "success": True,
            "videos_fixed": total_fixed,
            "videos_failed": total_failed,
            "total_processed": len(videos_to_fix),
            "duration": round(elapsed, 2),
            "checkpoint_path": str(path)
        }
        print(json.dumps(json_output))
    else:
        print(f"\n{'='*60}")
        print(f"✅ FIXED {total_fixed}/{len(videos_to_fix)} videos")
        print(f"⏱️  Total time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
        print(f"{'='*60}")
        print("\nYou can now resume your pipeline with:")
        print(f"  cd {path.parent}")
        print("  python main.py --resume")

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix missing channel_id in checkpoint files')
    parser.add_argument('checkpoint', help='Path to checkpoint.json file')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    args = parser.parse_args()

    fix_checkpoint(args.checkpoint, json_mode=args.json)
