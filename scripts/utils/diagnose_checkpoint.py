"""
Diagnose checkpoint metadata issue.
Run this in your project directory to see why channel_id isn't being used.
"""
import json
import sys
from pathlib import Path

def diagnose_checkpoint(checkpoint_path: str):
    """Analyze checkpoint for channel_id metadata issues."""
    
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"ERROR: Checkpoint not found: {path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing checkpoint: {path}")
    print(f"{'='*60}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"\n1. STAGE STATUS:")
    print(f"   Last completed stage: {data.get('last_completed_stage', 'UNKNOWN')}")
    
    # DOWNLOAD stage data
    download = data.get('download', {})
    downloaded_videos = download.get('downloaded_videos', [])
    
    print(f"\n2. DOWNLOAD STAGE:")
    print(f"   Total videos in checkpoint: {len(downloaded_videos)}")
    
    # Check for channel_id
    with_channel_id = sum(1 for v in downloaded_videos if v.get('channel_id'))
    without_channel_id = len(downloaded_videos) - with_channel_id
    
    print(f"   Videos WITH channel_id: {with_channel_id}")
    print(f"   Videos WITHOUT channel_id: {without_channel_id}")
    
    if downloaded_videos:
        print(f"\n3. SAMPLE VIDEO DATA (first 3):")
        for i, v in enumerate(downloaded_videos[:3], 1):
            print(f"   Video {i}:")
            print(f"     - title: {v.get('title', 'N/A')[:50]}...")
            print(f"     - url: {v.get('url', 'N/A')}")
            print(f"     - channel_id: '{v.get('channel_id', '')}'")
            print(f"     - channel: '{v.get('channel', '')}'")
            print(f"     - Has channel_id? {bool(v.get('channel_id'))}")
    
    # CAPTION stage data
    caption = data.get('caption', {})
    caption_results = caption.get('caption_results', {})
    
    print(f"\n4. CAPTION STAGE:")
    print(f"   Caption results count: {len(caption_results)}")
    print(f"   Success count: {caption.get('success_count', 0)}")
    print(f"   Skip count (cached): {caption.get('skip_count', 0)}")
    
    # Analyze mismatch
    print(f"\n5. ANALYSIS:")
    if with_channel_id == 0 and len(downloaded_videos) > 0:
        print("   ❌ PROBLEM: No videos have channel_id!")
        print("   Possible causes:")
        print("   - Old checkpoint from before channel_id was added")
        print("   - search_video_metadata() didn't extract channel_id")
        print("   - DownloadedVideo was created without channel_id")
    elif with_channel_id < len(downloaded_videos):
        print(f"   ⚠️  PARTIAL: Only {with_channel_id}/{len(downloaded_videos)} have channel_id")
        print("   Some videos will need individual pre-checks")
    else:
        print(f"   ✅ OK: All {with_channel_id} videos have channel_id")
    
    if data.get('last_completed_stage') == 'DOWNLOAD' and with_channel_id == 0:
        print(f"\n6. RECOMMENDATION:")
        print("   The DOWNLOAD stage completed but no channel_id was saved.")
        print("   To fix, delete the checkpoint and re-run:")
        print(f"   rm '{checkpoint_path}'")
        print("   Then run: python main.py --resume")
    
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        diagnose_checkpoint(sys.argv[1])
    else:
        # Auto-find checkpoint in current directory
        checkpoint = Path('checkpoint.json')
        if checkpoint.exists():
            diagnose_checkpoint('checkpoint.json')
        else:
            # Search in subdirectories
            found = False
            for cp in Path('.').rglob('checkpoint.json'):
                diagnose_checkpoint(cp)
                found = True
            if not found:
                print("No checkpoint.json found in current directory or subdirectories")
