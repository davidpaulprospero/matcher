"""
Fast channel_id fix with larger batches and periodic saves.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import shutil
import time

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

def fix_checkpoint(checkpoint_path: str):
    """Fix missing channel_id with periodic saves."""
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"ERROR: Checkpoint not found: {path}")
        return False
    
    print(f"\nLoading checkpoint: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    download = data.get('download', {})
    downloaded_videos = download.get('downloaded_videos', [])
    
    print(f"Total videos: {len(downloaded_videos)}")
    
    # Create backup if not exists
    backup_path = path.with_suffix('.json.backup')
    if not backup_path.exists():
        print(f"Creating backup: {backup_path}")
        shutil.copy2(path, backup_path)
    
    # Find videos without channel_id
    videos_to_fix = []
    for i, v in enumerate(downloaded_videos):
        if not v.get('channel_id'):
            url = v.get('url', '')
            if 'v=' in url:
                vid = url.split('v=')[-1].split('&')[0]
                if len(vid) == 11:
                    videos_to_fix.append((i, vid))
    
    if not videos_to_fix:
        print("✅ All videos already have channel_id!")
        return True
    
    print(f"Need to fix: {len(videos_to_fix)} videos")
    estimated_batches = (len(videos_to_fix) + 19) // 20
    print(f"Estimated: {estimated_batches} batches (~{estimated_batches * 5 // 60 + 1} minutes)\n")
    
    # Process in batches of 20 (smaller for reliability)
    batch_size = 20
    total_fixed = 0
    start_time = time.time()
    
    for batch_num, batch_start in enumerate(range(0, len(videos_to_fix), batch_size), 1):
        batch = videos_to_fix[batch_start:batch_start + batch_size]
        video_ids = [vid for _, vid in batch]
        
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
        
        print(f"✓ Fixed {fixed_in_batch}")
        
        # Save every 25 batches (500 videos)
        if batch_num % 25 == 0:
            print(f"  Saving checkpoint (progress: {total_fixed}/{len(videos_to_fix)})...")
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            elapsed = time.time() - start_time
            print(f"  Elapsed: {elapsed//60:.0f}m {elapsed%60:.0f}s")
    
    # Final save
    print(f"\nSaving final checkpoint...")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ FIXED {total_fixed}/{len(videos_to_fix)} videos")
    print(f"⏱️  Total time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"{'='*60}")
    print("\nYou can now resume your pipeline with:")
    print(f"  cd {path.parent}")
    print("  python main.py --resume")
    
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_channel_ids_fast.py <checkpoint.json>")
        sys.exit(1)
    
    fix_checkpoint(sys.argv[1])
