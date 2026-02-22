#!/usr/bin/env python3
"""
import argparse
Monitor the channel_id fix progress.
"""
import json
import time
from pathlib import Path
import sys
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

def check_progress(checkpoint_path: str):
    """Check how many videos now have channel_id."""
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        download = data.get('download', {})
        videos = download.get('downloaded_videos', [])
        
        with_channel = sum(1 for v in videos if v.get('channel_id'))
        return {
            'total': len(videos),
            'with_channel': with_channel,
            'without': len(videos) - with_channel
        }
    except:
        return None

if __name__ == '__main__':
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else 'checkpoint.json'
    
    print(f"Monitoring: {checkpoint}")
    print("-" * 50)
    
    last_with = 0
    while True:
        result = check_progress(checkpoint)
        if result:
            pct = result['with_channel'] / result['total'] * 100
            print(f"\rProgress: {result['with_channel']}/{result['total']} ({pct:.1f}%) - {result['without']} remaining", end='')
            
            if result['without'] == 0:
                print("\n\n✅ All videos fixed!")
                break
            
            if result['with_channel'] == last_with:
                # No change, maybe process stopped
                time.sleep(10)
            else:
                last_with = result['with_channel']
                time.sleep(5)
        else:
            print("\nCould not read checkpoint")
            break
