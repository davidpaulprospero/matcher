#!/usr/bin/env python3
"""
Find checkpoint.json files and fix missing channel_id metadata.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_checkpoints(start_path: Path = None):
    """Find all checkpoint.json files."""
    if start_path is None:
        start_path = Path('.')
    
    checkpoints = []
    print(f"Searching for checkpoint.json files starting from: {start_path.absolute()}")
    print("This may take a few minutes for large drives...")
    
    for cp in start_path.rglob('checkpoint.json'):
        try:
            with open(cp, 'r') as f:
                data = json.load(f)
            
            download = data.get('download', {})
            videos = download.get('downloaded_videos', [])
            
            if len(videos) > 1000:  # Only show checkpoints with many videos
                with_channel = sum(1 for v in videos if v.get('channel_id'))
                checkpoints.append({
                    'path': cp,
                    'total_videos': len(videos),
                    'with_channel_id': with_channel,
                    'last_stage': data.get('last_completed_stage', 'unknown')
                })
        except Exception as e:
            pass
    
    return checkpoints

def analyze_checkpoint(cp_info: dict):
    """Analyze a checkpoint file."""
    path = cp_info['path']
    print(f"\n{'='*70}")
    print(f"CHECKPOINT: {path}")
    print(f"{'='*70}")
    print(f"Total videos: {cp_info['total_videos']}")
    print(f"Videos WITH channel_id: {cp_info['with_channel_id']}")
    print(f"Videos WITHOUT channel_id: {cp_info['total_videos'] - cp_info['with_channel_id']}")
    print(f"Last completed stage: {cp_info['last_stage']}")
    
    missing_ratio = (cp_info['total_videos'] - cp_info['with_channel_id']) / cp_info['total_videos']
    print(f"Missing channel_id: {missing_ratio:.1%}")
    
    if missing_ratio > 0.5:
        print("\n⚠️  ISSUE: Most videos are missing channel_id!")
        print("   This causes slow batch pre-checking.")
    
    return missing_ratio > 0.1  # Return True if fix needed

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
    
    print(f"    Fetching metadata for {len(video_ids)} videos...")
    
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
        
        print(f"    ✓ Fetched metadata for {len(metadata)} videos")
        return metadata
        
    except Exception as e:
        print(f"    ✗ Failed to fetch metadata: {e}")
        return {}

def fix_checkpoint(checkpoint_path: Path):
    """Fix missing channel_id in checkpoint."""
    print(f"\n{'='*70}")
    print(f"FIXING: {checkpoint_path}")
    print(f"{'='*70}")
    
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    
    download = data.get('download', {})
    downloaded_videos = download.get('downloaded_videos', [])
    
    # Find videos without channel_id
    videos_needing_fix = []
    for i, v in enumerate(downloaded_videos):
        if not v.get('channel_id'):
            url = v.get('url', '')
            if 'v=' in url:
                vid = url.split('v=')[-1].split('&')[0]
                if len(vid) == 11:
                    videos_needing_fix.append((i, vid, v))
    
    if not videos_needing_fix:
        print("✅ All videos already have channel_id! No fix needed.")
        return True
    
    print(f"Found {len(videos_needing_fix)} videos missing channel_id")
    
    # Fetch metadata in batches
    batch_size = 50
    total_fixed = 0
    
    for batch_start in range(0, len(videos_needing_fix), batch_size):
        batch = videos_needing_fix[batch_start:batch_start + batch_size]
        video_ids = [vid for _, vid, _ in batch]
        
        print(f"\n  Batch {batch_start//batch_size + 1}/{(len(videos_needing_fix)-1)//batch_size + 1}")
        metadata = fetch_video_metadata_batch(video_ids)
        
        # Update checkpoint data
        for idx, vid, v in batch:
            if vid in metadata:
                v['channel_id'] = metadata[vid]['channel_id']
                v['channel'] = metadata[vid]['channel']
                total_fixed += 1
    
    print(f"\n✅ Fixed {total_fixed}/{len(videos_needing_fix)} videos")
    
    # Save updated checkpoint
    backup_path = checkpoint_path.with_suffix('.json.backup')
    print(f"\nCreating backup: {backup_path}")
    shutil.copy2(checkpoint_path, backup_path)
    
    print(f"Saving updated checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n{'='*70}")
    print("DONE! You can now resume your pipeline with:")
    print(f"  cd {checkpoint_path.parent}")
    print("  python main.py --resume")
    print(f"{'='*70}")
    return True

def main():
    import argparse
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

    parser = argparse.ArgumentParser(description='Find and fix checkpoint metadata')
    parser.add_argument('--path', type=Path, default=Path('.'), help='Starting path for search')
    parser.add_argument('--fix', type=Path, help='Fix a specific checkpoint file')
    parser.add_argument('--auto', action='store_true', help='Auto-fix all checkpoints with issues')
    args = parser.parse_args()
    
    if args.fix:
        fix_checkpoint(args.fix)
        return
    
    print("Searching for checkpoint.json files with many videos...")
    checkpoints = find_checkpoints(args.path)
    
    if not checkpoints:
        print("\nNo checkpoint.json files with >1000 videos found.")
        print("\nSearching for ANY checkpoint.json files...")
        
        # Broader search
        for cp in args.path.rglob('checkpoint.json'):
            try:
                with open(cp, 'r') as f:
                    data = json.load(f)
                download = data.get('download', {})
                videos = download.get('downloaded_videos', [])
                if videos:
                    print(f"\nFound: {cp}")
                    print(f"  Videos: {len(videos)}")
                    response = input("Analyze this checkpoint? (y/n): ")
                    if response.lower() == 'y':
                        checkpoints.append({
                            'path': cp,
                            'total_videos': len(videos),
                            'with_channel_id': sum(1 for v in videos if v.get('channel_id')),
                            'last_stage': data.get('last_completed_stage', 'unknown')
                        })
            except:
                pass
    
    if not checkpoints:
        print("\nNo checkpoint files found.")
        print("\nPossible locations to check:")
        print("  - E:\\Edit Job\\<client>\\<project>")
        print("  - D:\\Projects\\<project>")
        print("  - Your project directory from the last run")
        return
    
    print(f"\n\nFound {len(checkpoints)} checkpoint(s) with >1000 videos:")
    for i, cp in enumerate(checkpoints, 1):
        print(f"\n{i}. {cp['path']}")
        print(f"   Videos: {cp['total_videos']} (with channel_id: {cp['with_channel_id']})")
        print(f"   Last stage: {cp['last_stage']}")
    
    if args.auto:
        for cp in checkpoints:
            if cp['with_channel_id'] < cp['total_videos']:
                fix_checkpoint(cp['path'])
        return
    
    # Interactive mode
    if len(checkpoints) == 1:
        cp = checkpoints[0]
        needs_fix = analyze_checkpoint(cp)
        if needs_fix:
            response = input("\nFix this checkpoint? (y/n): ")
            if response.lower() == 'y':
                fix_checkpoint(cp['path'])
    else:
        choice = input("\nEnter number of checkpoint to analyze/fix (or 'all' to fix all): ")
        if choice.lower() == 'all':
            for cp in checkpoints:
                if cp['with_channel_id'] < cp['total_videos']:
                    fix_checkpoint(cp['path'])
        else:
            try:
                idx = int(choice) - 1
                cp = checkpoints[idx]
                needs_fix = analyze_checkpoint(cp)
                if needs_fix:
                    response = input("\nFix this checkpoint? (y/n): ")
                    if response.lower() == 'y':
                        fix_checkpoint(cp['path'])
            except (ValueError, IndexError):
                print("Invalid choice")

if __name__ == '__main__':
    main()
