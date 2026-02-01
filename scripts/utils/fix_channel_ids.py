"""
Fix missing channel_id in checkpoint metadata.

This script re-fetches video metadata to populate channel_id for videos
that don't have it, using yt-dlp bulk info fetching.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_video_metadata_batch(video_ids: List[str]) -> Dict[str, dict]:
    """Fetch metadata for multiple videos using yt-dlp."""
    if not video_ids:
        return {}
    
    # Build URLs
    urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
    
    cmd = [
        'yt-dlp',
        '--ignore-config',
        '--dump-json',
        '--no-download',
        '--quiet',
    ] + urls
    
    logger.info(f"Fetching metadata for {len(video_ids)} videos...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
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
                        'title': data.get('title', ''),
                        'duration': data.get('duration', 0),
                    }
            except json.JSONDecodeError:
                continue
        
        logger.info(f"  Fetched metadata for {len(metadata)} videos")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {e}")
        return {}

def fix_checkpoint_channel_ids(checkpoint_path: str, dry_run: bool = False):
    """Fix missing channel_id entries in checkpoint."""
    
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"ERROR: Checkpoint not found: {path}")
        return False
    
    print(f"\nLoading checkpoint: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    download = data.get('download', {})
    downloaded_videos = download.get('downloaded_videos', [])
    
    print(f"Total videos in checkpoint: {len(downloaded_videos)}")
    
    # Find videos without channel_id
    videos_needing_fix = []
    for i, v in enumerate(downloaded_videos):
        if not v.get('channel_id'):
            # Extract video ID from URL
            url = v.get('url', '')
            if 'v=' in url:
                vid = url.split('v=')[-1].split('&')[0]
                if len(vid) == 11:
                    videos_needing_fix.append((i, vid, v))
    
    if not videos_needing_fix:
        print("✅ All videos already have channel_id! No fix needed.")
        return True
    
    print(f"Found {len(videos_needing_fix)} videos missing channel_id")
    
    if dry_run:
        print("\nDRY RUN - would fix these videos:")
        for idx, vid, v in videos_needing_fix[:10]:
            print(f"  - {vid}: {v.get('title', 'N/A')[:40]}...")
        if len(videos_needing_fix) > 10:
            print(f"  ... and {len(videos_needing_fix) - 10} more")
        return True
    
    # Fetch metadata in batches of 50
    batch_size = 50
    total_fixed = 0
    
    for batch_start in range(0, len(videos_needing_fix), batch_size):
        batch = videos_needing_fix[batch_start:batch_start + batch_size]
        video_ids = [vid for _, vid, _ in batch]
        
        metadata = fetch_video_metadata_batch(video_ids)
        
        # Update checkpoint data
        for idx, vid, v in batch:
            if vid in metadata:
                v['channel_id'] = metadata[vid]['channel_id']
                v['channel'] = metadata[vid]['channel']
                total_fixed += 1
    
    print(f"\n✅ Fixed {total_fixed}/{len(videos_needing_fix)} videos")
    
    # Save updated checkpoint
    backup_path = path.with_suffix('.json.backup')
    print(f"Creating backup: {backup_path}")
    import shutil
    shutil.copy2(path, backup_path)
    
    print(f"Saving updated checkpoint: {path}")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Done! You can now resume your pipeline with: python main.py --resume")
    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix missing channel_id in checkpoint')
    parser.add_argument('checkpoint', nargs='?', default='checkpoint.json', help='Path to checkpoint.json')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without making changes')
    args = parser.parse_args()
    
    fix_checkpoint_channel_ids(args.checkpoint, args.dry_run)
