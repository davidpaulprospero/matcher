#!/usr/bin/env python
"""
Relink Offline Media in DaVinci Resolve.
Attempts to find and relink offline media using known locations.

Search locations:
1. E:/v (short video path)
2. E:/i (short image path)
3. Original path with forward slashes
4. Project .cache directories
5. Global cache (~/.matcher_global_cache)

Usage:
    python relink_media.py                    # Auto-relink all offline media
    python relink_media.py --dry-run          # Show what would be relinked
    python relink_media.py --folder "E:/v"    # Add custom search folder
    python relink_media.py --interactive      # Ask before each relink

Run from DaVinci Resolve: Workspace > Scripts > Edit > relink_media
"""

import sys
import os
import argparse
from pathlib import Path
from resolve_utils import get_resolve, get_current_project


# Default search locations
DEFAULT_SEARCH_PATHS = [
    "E:/v",
    "E:/i",
    os.path.expanduser("~/.matcher_global_cache"),
    os.path.expanduser("~/.matcher_entity_cache"),
]


def find_file_in_paths(filename, search_paths):
    """Search for a file in multiple paths."""
    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue

        # Direct match
        direct = os.path.join(base_path, filename)
        if os.path.exists(direct):
            return direct

        # Search subdirectories (1 level deep for performance)
        try:
            for subdir in os.listdir(base_path):
                subdir_path = os.path.join(base_path, subdir)
                if os.path.isdir(subdir_path):
                    candidate = os.path.join(subdir_path, filename)
                    if os.path.exists(candidate):
                        return candidate

                    # Also check common subfolders
                    for subfolder in ['videos', 'images', 'stock', 'broll', 'audio']:
                        candidate = os.path.join(subdir_path, subfolder, filename)
                        if os.path.exists(candidate):
                            return candidate
        except PermissionError:
            continue

    return None


def get_all_media_pool_items(media_pool):
    """Recursively get all MediaPoolItems from the media pool."""
    items = []

    def traverse_folder(folder):
        # Get clips in this folder
        clips = folder.GetClipList()
        if clips:
            items.extend(clips)

        # Recurse into subfolders
        subfolders = folder.GetSubFolderList()
        if subfolders:
            for subfolder in subfolders:
                traverse_folder(subfolder)

    root = media_pool.GetRootFolder()
    traverse_folder(root)
    return items


def check_if_offline(mpi):
    """Check if a MediaPoolItem is offline."""
    props = mpi.GetClipProperty()
    if not props:
        return True

    file_path = props.get('File Path', '')
    if not file_path:
        return True

    # Normalize path
    file_path = file_path.replace('\\', '/')
    return not os.path.exists(file_path)


def relink_media(project, search_paths, dry_run=False, interactive=False):
    """Find and relink offline media."""
    media_pool = project.GetMediaPool()
    all_items = get_all_media_pool_items(media_pool)

    print(f"\nScanning {len(all_items)} media pool items...")
    print(f"Search paths: {search_paths}")
    print("-" * 50)

    offline_items = []
    relinked = 0
    failed = 0

    for mpi in all_items:
        props = mpi.GetClipProperty()
        if not props:
            continue

        original_path = props.get('File Path', '')
        if not original_path:
            continue

        # Normalize path
        original_path = original_path.replace('\\', '/')

        # Check if offline
        if os.path.exists(original_path):
            continue

        clip_name = mpi.GetName()
        filename = os.path.basename(original_path)

        offline_items.append({
            'item': mpi,
            'name': clip_name,
            'original_path': original_path,
            'filename': filename,
        })

    if not offline_items:
        print("No offline media found!")
        return 0

    print(f"\nFound {len(offline_items)} offline items")
    print("-" * 50)

    for item_info in offline_items:
        mpi = item_info['item']
        filename = item_info['filename']
        original_path = item_info['original_path']

        print(f"\n  {item_info['name']}")
        print(f"    Original: {original_path}")

        # Try to find the file
        new_path = find_file_in_paths(filename, search_paths)

        if new_path:
            new_path = new_path.replace('\\', '/')
            print(f"    Found at: {new_path}")

            if interactive:
                response = input("    Relink? [Y/n]: ").strip().lower()
                if response and response != 'y':
                    print("    Skipped")
                    continue

            if not dry_run:
                # Use RelinkClips for the relink
                success = media_pool.RelinkClips([mpi], os.path.dirname(new_path))
                if success:
                    print("    RELINKED")
                    relinked += 1
                else:
                    # Try ReplaceClip as fallback
                    success = mpi.ReplaceClip(new_path)
                    if success:
                        print("    RELINKED (via ReplaceClip)")
                        relinked += 1
                    else:
                        print("    FAILED to relink")
                        failed += 1
            else:
                print("    Would relink (dry-run)")
                relinked += 1
        else:
            print("    NOT FOUND in search paths")
            failed += 1

    print("\n" + "=" * 50)
    print(f"SUMMARY:")
    print(f"  Total offline: {len(offline_items)}")
    print(f"  Relinked: {relinked}")
    print(f"  Failed: {failed}")

    if dry_run:
        print("\n  (This was a dry run - no changes were made)")

    return failed


def main():
    parser = argparse.ArgumentParser(description="Relink offline media in DaVinci Resolve")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be relinked without making changes")
    parser.add_argument('--interactive', action='store_true', help="Ask before each relink")
    parser.add_argument('--folder', type=str, action='append', help="Add custom search folder (can be used multiple times)")

    args = parser.parse_args()

    resolve = get_resolve()
    if not resolve:
        return 1

    project = resolve.GetProjectManager().GetCurrentProject()
    if not project:
        print("Error: No project is open")
        return 1

    print(f"Project: {project.GetName()}")

    # Build search paths
    search_paths = list(DEFAULT_SEARCH_PATHS)
    if args.folder:
        search_paths = args.folder + search_paths

    # Filter to existing paths
    search_paths = [p for p in search_paths if os.path.exists(p)]

    if not search_paths:
        print("Error: No valid search paths found")
        print(f"Checked: {DEFAULT_SEARCH_PATHS}")
        return 1

    return relink_media(project, search_paths, dry_run=args.dry_run, interactive=args.interactive)


if __name__ == "__main__":
    sys.exit(main())
