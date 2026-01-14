#!/usr/bin/env python3
"""
Fix Special Characters in Video Filenames

DaVinci Resolve crashes on files with special characters like %, &, #, etc.
This script:
1. Finds files with problematic characters
2. Renames them to safe names
3. Updates cache files to reference the new names

Usage:
    python fix_special_chars.py "E:/v/2"
    python fix_special_chars.py "E:/v/2" --dry-run
"""

import os
import sys
import re
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Characters that cause issues in DaVinci Resolve
UNSAFE_CHARS = re.compile(r'[%&$#@!^*()+=\[\]{}|\\:;"\'<>,?`~]')

def is_safe_filename(name: str) -> bool:
    """Check if filename is safe for DaVinci Resolve"""
    return not UNSAFE_CHARS.search(name)

def make_safe_filename(name: str) -> str:
    """Convert filename to safe version"""
    # Replace unsafe characters with underscore
    safe = UNSAFE_CHARS.sub('_', name)
    # Remove multiple consecutive underscores
    safe = re.sub(r'_+', '_', safe)
    # Remove leading/trailing underscores
    safe = safe.strip('_')
    return safe

def find_unsafe_files(directory: str) -> List[Tuple[Path, Path]]:
    """Find files with unsafe characters and generate new safe names"""
    directory = Path(directory)
    renames = []
    
    for root, dirs, files in os.walk(directory):
        root_path = Path(root)
        for filename in files:
            if not is_safe_filename(filename):
                old_path = root_path / filename
                
                # Generate safe name
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                safe_stem = make_safe_filename(stem)
                new_filename = safe_stem + suffix
                
                # Handle collision
                new_path = root_path / new_filename
                counter = 1
                while new_path.exists() and new_path != old_path:
                    new_filename = f"{safe_stem}_{counter}{suffix}"
                    new_path = root_path / new_filename
                    counter += 1
                
                if new_path != old_path:
                    renames.append((old_path, new_path))
    
    return renames

def update_json_file(json_path: Path, old_to_new: Dict[str, str]) -> bool:
    """Update a JSON file, replacing old paths with new paths"""
    if not json_path.exists():
        return False
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        for old_path, new_path in old_to_new.items():
            # Replace various path formats
            content = content.replace(old_path.replace('\\', '/'), new_path.replace('\\', '/'))
            content = content.replace(old_path.replace('/', '\\'), new_path.replace('/', '\\'))
            content = content.replace(old_path, new_path)
            
            # Also replace just the filename
            old_name = Path(old_path).name
            new_name = Path(new_path).name
            content = content.replace(f'"{old_name}"', f'"{new_name}"')
        
        if content != original:
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"  Warning: Could not update {json_path}: {e}")
    
    return False

def update_cache_directory(cache_dir: Path, old_to_new: Dict[str, str]) -> int:
    """Update all cache files in directory"""
    updated = 0
    
    if not cache_dir.exists():
        return 0
    
    # Find all JSON files in cache
    for json_file in cache_dir.rglob('*.json'):
        if update_json_file(json_file, old_to_new):
            print(f"  Updated: {json_file.name}")
            updated += 1
    
    return updated

def main():
    parser = argparse.ArgumentParser(description='Fix special characters in video filenames')
    parser.add_argument('video_dir', help='Directory containing videos')
    parser.add_argument('--cache-dir', help='Cache directory to update (default: video_dir/../.cache)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--project-dir', help='Project directory (for finding cache)')
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: Directory not found: {video_dir}")
        sys.exit(1)
    
    # Find cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    elif args.project_dir:
        cache_dir = Path(args.project_dir) / '.cache'
    else:
        # Try common locations
        cache_dir = video_dir.parent / '.cache'
        if not cache_dir.exists():
            cache_dir = video_dir / '.cache'
    
    print(f"\n{'='*60}")
    print("  FIX SPECIAL CHARACTERS IN FILENAMES")
    print(f"{'='*60}")
    print(f"  Video directory: {video_dir}")
    print(f"  Cache directory: {cache_dir}")
    if args.dry_run:
        print(f"  Mode: DRY RUN (no changes will be made)")
    print()
    
    # Find files to rename
    renames = find_unsafe_files(video_dir)
    
    if not renames:
        print("  ✓ No files with unsafe characters found!")
        return
    
    print(f"  Found {len(renames)} file(s) with unsafe characters:\n")
    
    for old_path, new_path in renames:
        old_name = old_path.name
        new_name = new_path.name
        print(f"    {old_name}")
        print(f"    → {new_name}\n")
    
    if args.dry_run:
        print("  [DRY RUN] No changes made.")
        return
    
    # Confirm
    print(f"  This will rename {len(renames)} file(s) and update cache.")
    response = input("  Continue? [y/N]: ").strip().lower()
    if response != 'y':
        print("  Cancelled.")
        return
    
    # Build mapping
    old_to_new = {}
    
    # Rename files
    print("\n  Renaming files...")
    for old_path, new_path in renames:
        try:
            old_path.rename(new_path)
            old_to_new[str(old_path)] = str(new_path)
            print(f"    ✓ {old_path.name} → {new_path.name}")
        except Exception as e:
            print(f"    ✗ Failed to rename {old_path.name}: {e}")
    
    # Update cache
    if cache_dir.exists() and old_to_new:
        print("\n  Updating cache files...")
        updated = update_cache_directory(cache_dir, old_to_new)
        print(f"    Updated {updated} cache file(s)")
    
    print(f"\n  {'='*60}")
    print(f"  DONE! Renamed {len(old_to_new)} file(s)")
    print(f"  {'='*60}")
    print(f"\n  Next step: Re-run the pipeline with --skip-download --skip-transcription")
    print(f"  Or just regenerate output with your existing checkpoint.")

if __name__ == '__main__':
    main()
