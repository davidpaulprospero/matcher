#!/usr/bin/env python
"""
Import Media Folder into DaVinci Resolve Media Pool.
Bulk imports media from our short path locations (E:/v, E:/i).

Usage:
    python import_media_folder.py "E:/v/MyProject"     # Import from specific folder
    python import_media_folder.py --all-v              # Import all projects from E:/v
    python import_media_folder.py --recent 5           # Import 5 most recent projects
    python import_media_folder.py --subfolder videos   # Create subfolder in media pool

Run from DaVinci Resolve: Workspace > Scripts > Edit > import_media_folder
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from resolve_utils import get_resolve, get_current_project


# Media extensions to import
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mxf', '.dv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.exr', '.dpx'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aac', '.m4a', '.flac', '.aiff'}

ALL_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS


def get_media_files(folder_path, recursive=True, include_audio=True):
    """Get all media files from a folder."""
    folder = Path(folder_path)
    media_files = []

    extensions = ALL_MEDIA_EXTENSIONS if include_audio else (VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)

    if recursive:
        for ext in extensions:
            media_files.extend(folder.rglob(f"*{ext}"))
            media_files.extend(folder.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            media_files.extend(folder.glob(f"*{ext}"))
            media_files.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(set(str(f) for f in media_files))


def get_recent_projects(base_path, count=5):
    """Get the most recently modified project folders from a base path."""
    base = Path(base_path)
    if not base.exists():
        return []

    # Get all subdirectories
    subdirs = [d for d in base.iterdir() if d.is_dir()]

    # Sort by modification time (most recent first)
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    return [str(d) for d in subdirs[:count]]


def import_folder(media_pool, folder_path, subfolder_name=None, include_audio=True):
    """Import media from a folder into the media pool."""
    folder_path = Path(folder_path)
    print(f"\nImporting from: {folder_path}")

    # Get media files
    media_files = get_media_files(folder_path, recursive=True, include_audio=include_audio)

    if not media_files:
        print("  No media files found")
        return 0

    print(f"  Found {len(media_files)} media files")

    # Create subfolder if requested
    if subfolder_name:
        root = media_pool.GetRootFolder()
        subfolder = media_pool.AddSubFolder(root, subfolder_name)
        if subfolder:
            media_pool.SetCurrentFolder(subfolder)
            print(f"  Created subfolder: {subfolder_name}")
        else:
            print(f"  Warning: Could not create subfolder '{subfolder_name}'")

    # Import in batches (DaVinci can be slow with large imports)
    batch_size = 50
    imported_count = 0

    for i in range(0, len(media_files), batch_size):
        batch = media_files[i:i+batch_size]
        result = media_pool.ImportMedia(batch)

        if result:
            imported_count += len(result)
            progress = min(i + batch_size, len(media_files))
            print(f"  Progress: {progress}/{len(media_files)} ({imported_count} imported)")
        else:
            print(f"  Warning: Batch {i//batch_size + 1} failed to import")

    # Reset to root folder
    media_pool.SetCurrentFolder(media_pool.GetRootFolder())

    print(f"  Imported {imported_count} files")
    return imported_count


def main():
    parser = argparse.ArgumentParser(description="Import media folder into DaVinci Resolve")
    parser.add_argument('folder', nargs='?', help="Path to folder to import")
    parser.add_argument('--all-v', action='store_true', help="Import all projects from E:/v")
    parser.add_argument('--all-i', action='store_true', help="Import all projects from E:/i")
    parser.add_argument('--recent', type=int, metavar='N', help="Import N most recent projects from E:/v")
    parser.add_argument('--subfolder', type=str, help="Create subfolder in media pool with this name")
    parser.add_argument('--no-audio', action='store_true', help="Skip audio files (they can cause import issues)")
    parser.add_argument('--list-only', action='store_true', help="List files without importing")

    args = parser.parse_args()

    # Determine folders to import
    folders_to_import = []

    if args.folder:
        if os.path.exists(args.folder):
            folders_to_import.append(args.folder)
        else:
            print(f"Error: Folder not found: {args.folder}")
            return 1
    elif args.all_v:
        if os.path.exists("E:/v"):
            folders_to_import = [str(d) for d in Path("E:/v").iterdir() if d.is_dir()]
        else:
            print("Error: E:/v does not exist")
            return 1
    elif args.all_i:
        if os.path.exists("E:/i"):
            folders_to_import = [str(d) for d in Path("E:/i").iterdir() if d.is_dir()]
        else:
            print("Error: E:/i does not exist")
            return 1
    elif args.recent:
        folders_to_import = get_recent_projects("E:/v", args.recent)
        if not folders_to_import:
            print("Error: No projects found in E:/v")
            return 1
    else:
        print("Usage: import_media_folder.py <folder>")
        print("       import_media_folder.py --all-v")
        print("       import_media_folder.py --recent 5")
        return 1

    print(f"Folders to import: {len(folders_to_import)}")
    for f in folders_to_import:
        print(f"  - {f}")

    if args.list_only:
        print("\nListing files (--list-only mode):")
        for folder in folders_to_import:
            files = get_media_files(folder, include_audio=not args.no_audio)
            print(f"\n{folder}: {len(files)} files")
            for f in files[:10]:
                print(f"    {os.path.basename(f)}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 10} more")
        return 0

    # Connect to DaVinci
    resolve = get_resolve()
    if not resolve:
        return 1

    project = get_current_project(resolve)
    if not project:
        return 1

    media_pool = project.GetMediaPool()
    print(f"\nProject: {project.GetName()}")

    # Import each folder
    total_imported = 0
    for folder in folders_to_import:
        # Use folder name as subfolder name if not specified and multiple folders
        subfolder_name = args.subfolder
        if not subfolder_name and len(folders_to_import) > 1:
            subfolder_name = os.path.basename(folder)

        count = import_folder(
            media_pool,
            folder,
            subfolder_name=subfolder_name,
            include_audio=not args.no_audio
        )
        total_imported += count

    print("\n" + "=" * 50)
    print(f"Total imported: {total_imported} files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
