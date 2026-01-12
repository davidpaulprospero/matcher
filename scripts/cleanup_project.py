#!/usr/bin/env python3
"""
Project Cleanup Script for Voiceover-Matcher

Archives essential files and removes large temporary data after project completion.
Preserves global caches (~/.matcher_*) while cleaning project-specific files.

Usage:
    python cleanup_project.py --project "E:\\Projects\\MyDoc__2026-01-03"
    python cleanup_project.py --project "E:\\Projects\\MyDoc" --level full --zip
    python cleanup_project.py --project "E:\\Projects\\MyDoc" --dry-run
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yaml


# Cleanup levels
LEVEL_CACHE = "cache"      # Delete caches only, keep videos/images
LEVEL_MEDIA = "media"      # Delete caches + videos/images
LEVEL_FULL = "full"        # Delete everything except output

# Files/folders to ALWAYS keep in archive
ARCHIVE_ESSENTIALS = [
    "output",               # OTIO, EDL, XML files (new name)
    "otio_output",          # OTIO, EDL, XML files (legacy name)
    "voiceover",            # Original script
    "project_config.yaml",  # Project settings
    "run.bat",              # Launcher (Windows)
    "run.sh",               # Launcher (Unix)
]

# Files/folders to DELETE at each level
DELETE_AT_CACHE = [
    ".cache",               # All cache subdirectories
    "checkpoint.json",
    "checkpoint.backup.json",
    "saved_keywords.json",
    "logs",
]

DELETE_AT_MEDIA = DELETE_AT_CACHE + [
    "videos",               # Downloaded YouTube videos (new name)
    "downloaded_videos",    # Downloaded YouTube videos (legacy name)
    "images",               # Downloaded entity images (local)
]

DELETE_AT_FULL = DELETE_AT_MEDIA + [
    "voiceover",            # Original script (user should have backup)
]

# Export file patterns (rendered videos in root directory)
EXPORT_PATTERNS = ["*.mov", "*.mp4", "*.avi", "*.mkv"]


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes"""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def load_project_config(project_dir: Path) -> dict:
    """Load project_config.yaml to find custom paths"""
    config_path = project_dir / "project_config.yaml"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def find_short_path_dirs(project_dir: Path, config: dict) -> Dict[str, Optional[Path]]:
    """
    Find video/image directories from short path config.

    Returns dict with 'videos' and 'images' paths (or None if not configured).
    """
    result = {'videos': None, 'images': None}
    project_name = project_dir.name[:15]  # Short path uses truncated name

    # Check for download.root_dir (videos)
    download_config = config.get('download', {})
    if isinstance(download_config, dict):
        root_dir = download_config.get('root_dir')
        if root_dir:
            videos_path = Path(root_dir) / project_name
            if videos_path.exists():
                result['videos'] = videos_path

    # Check for image_search.root_dir (images)
    image_config = config.get('image_search', {})
    if isinstance(image_config, dict):
        root_dir = image_config.get('root_dir')
        if root_dir:
            images_path = Path(root_dir) / project_name
            if images_path.exists():
                result['images'] = images_path

    return result


def load_checkpoint_stats(project_dir: Path) -> dict:
    """Extract stats from checkpoint.json for manifest"""
    checkpoint_path = project_dir / "checkpoint.json"
    stats = {
        'last_stage': None,
        'keywords': [],
        'topic': '',
        'video_count': 0,
        'match_count': 0,
        'avg_confidence': 0,
        'created_at': None,
        'completed_at': None,
    }

    if not checkpoint_path.exists():
        return stats

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        stats['last_stage'] = data.get('last_completed_stage')
        stats['created_at'] = data.get('created_at')
        stats['completed_at'] = data.get('updated_at')

        # Analyze stage data
        analyze = data.get('analyze', {})
        stats['keywords'] = analyze.get('keywords', [])
        stats['topic'] = analyze.get('topic_context', '')
        stats['segment_count'] = analyze.get('segment_count', 0)

        # Download stage data
        download = data.get('download', {})
        stats['video_count'] = len(download.get('video_paths', []))

        # Match stage data
        match = data.get('match', {})
        stats['match_count'] = match.get('match_count', 0)
        stats['avg_confidence'] = match.get('avg_confidence', 0)

    except Exception:
        pass

    return stats


def generate_manifest(project_dir: Path, level: str, sizes: dict,
                     stats: dict, archived_files: List[str]) -> dict:
    """Generate manifest.json with project metadata"""
    manifest = {
        'project_name': project_dir.name,
        'cleanup_date': datetime.now().isoformat(),
        'cleanup_level': level,
        'sizes': {
            'before_bytes': sizes.get('before', 0),
            'before_human': format_size(sizes.get('before', 0)),
            'after_bytes': sizes.get('after', 0),
            'after_human': format_size(sizes.get('after', 0)),
            'freed_bytes': sizes.get('before', 0) - sizes.get('after', 0),
            'freed_human': format_size(sizes.get('before', 0) - sizes.get('after', 0)),
        },
        'pipeline_stats': {
            'last_stage': stats.get('last_stage'),
            'keywords': stats.get('keywords', []),
            'topic': stats.get('topic', ''),
            'segment_count': stats.get('segment_count', 0),
            'video_count': stats.get('video_count', 0),
            'match_count': stats.get('match_count', 0),
            'avg_confidence': stats.get('avg_confidence', 0),
            'pipeline_started': stats.get('created_at'),
            'pipeline_completed': stats.get('completed_at'),
        },
        'archived_files': archived_files,
        'preserved_locations': {
            'global_video_cache': str(Path.home() / '.matcher_global_cache'),
            'global_entity_cache': str(Path.home() / '.matcher_entity_cache'),
        }
    }

    return manifest


def collect_export_files(project_dir: Path) -> List[Tuple[Path, str]]:
    """
    Collect export/render files from project root directory.

    Returns list of (path, description) tuples.
    """
    exports = []
    for pattern in EXPORT_PATTERNS:
        for file in project_dir.glob(pattern):
            if file.is_file():
                exports.append((file, f"export/{file.name}"))
    return exports


def collect_deletable_paths(project_dir: Path, level: str,
                           short_paths: dict,
                           include_exports: bool = False) -> List[Tuple[Path, str]]:
    """
    Collect paths to delete based on cleanup level.

    Returns list of (path, description) tuples.
    """
    paths = []

    # Determine which items to delete
    if level == LEVEL_CACHE:
        delete_list = DELETE_AT_CACHE
    elif level == LEVEL_MEDIA:
        delete_list = DELETE_AT_MEDIA
    else:  # LEVEL_FULL
        delete_list = DELETE_AT_FULL

    # Add project-local paths
    for item in delete_list:
        path = project_dir / item
        if path.exists():
            paths.append((path, f"project/{item}"))

    # Add short path directories (if at media or full level)
    if level in (LEVEL_MEDIA, LEVEL_FULL):
        if short_paths.get('videos'):
            paths.append((short_paths['videos'], f"short-path/videos"))
        if short_paths.get('images'):
            paths.append((short_paths['images'], f"short-path/images"))

    # Add export files if requested
    if include_exports:
        paths.extend(collect_export_files(project_dir))

    return paths


def create_archive(project_dir: Path, level: str) -> Tuple[Path, List[str]]:
    """
    Create archive folder with essential files.

    Returns (archive_path, list of archived files).
    """
    archive_dir = project_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    archived = []

    # Determine what to archive based on level
    archive_list = list(ARCHIVE_ESSENTIALS)
    if level == LEVEL_FULL:
        # At full level, voiceover is deleted, but we still want to archive it first
        pass  # voiceover already in ARCHIVE_ESSENTIALS

    for item in archive_list:
        src = project_dir / item
        dst = archive_dir / item

        if src.exists():
            if src.is_dir():
                # Copy directory
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                # Count files
                file_count = sum(1 for _ in dst.rglob("*") if _.is_file())
                archived.append(f"{item}/ ({file_count} files)")
            else:
                # Copy file
                shutil.copy2(src, dst)
                archived.append(item)

    return archive_dir, archived


def create_zip_archive(archive_dir: Path, project_dir: Path) -> Path:
    """Create ZIP file of archive folder"""
    zip_name = f"{project_dir.name}__ARCHIVE.zip"
    zip_path = project_dir / zip_name

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in archive_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(archive_dir)
                zf.write(file, arcname)

    return zip_path


def delete_paths(paths: List[Tuple[Path, str]], dry_run: bool = False) -> int:
    """
    Delete collected paths.

    Returns total bytes freed.
    """
    total_freed = 0

    for path, description in paths:
        if not path.exists():
            continue

        size = get_dir_size(path) if path.is_dir() else path.stat().st_size
        total_freed += size

        if dry_run:
            print(f"  [DRY-RUN] Would delete: {description} ({format_size(size)})")
        else:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"  Deleted: {description} ({format_size(size)})")
            except Exception as e:
                print(f"  Error deleting {description}: {e}")

    return total_freed


def cleanup_project(project_dir: Path, level: str = LEVEL_CACHE,
                   create_zip: bool = False, dry_run: bool = False,
                   yes: bool = False, include_exports: bool = False) -> bool:
    """
    Main cleanup function.

    Args:
        project_dir: Path to project directory
        level: Cleanup level (cache, media, full)
        create_zip: Whether to create ZIP archive
        dry_run: If True, only show what would be deleted
        yes: Skip confirmation prompt
        include_exports: Also delete export/render files (*.mov, *.mp4) from root

    Returns:
        True if cleanup succeeded, False otherwise
    """
    project_dir = Path(project_dir).resolve()

    # Validate project directory
    if not project_dir.exists():
        print(f"  Error: Project directory not found: {project_dir}")
        return False

    # Check if it's a valid project (has run.bat or project_config.yaml)
    is_project = (
        (project_dir / "run.bat").exists() or
        (project_dir / "run.sh").exists() or
        (project_dir / "project_config.yaml").exists()
    )
    if not is_project:
        print(f"  Error: Not a valid project directory: {project_dir}")
        print("  (Missing run.bat, run.sh, or project_config.yaml)")
        return False

    # Check if already archived
    if (project_dir / ".archived").exists():
        print(f"  Warning: Project already archived. Skipping.")
        return True

    # Load config to find short paths
    config = load_project_config(project_dir)
    short_paths = find_short_path_dirs(project_dir, config)

    # Load checkpoint stats
    stats = load_checkpoint_stats(project_dir)

    # Calculate current size
    print()
    print("=" * 60)
    print(f"  PROJECT CLEANUP: {project_dir.name}")
    print("=" * 60)
    print()
    print(f"  Project: {project_dir}")
    print(f"  Level:   {level}")
    print()

    # Collect paths to delete
    delete_paths_list = collect_deletable_paths(project_dir, level, short_paths, include_exports)

    # Calculate sizes
    size_before = get_dir_size(project_dir)

    # Add short path sizes
    if short_paths.get('videos'):
        size_before += get_dir_size(short_paths['videos'])
    if short_paths.get('images'):
        size_before += get_dir_size(short_paths['images'])

    delete_size = sum(
        get_dir_size(p) if p.is_dir() else p.stat().st_size
        for p, _ in delete_paths_list if p.exists()
    )

    print(f"  Current size:    {format_size(size_before)}")
    print(f"  Will delete:     {format_size(delete_size)}")
    print(f"  Estimated after: {format_size(size_before - delete_size)}")
    print()

    # Show what will be deleted
    print("  Items to delete:")
    for path, desc in delete_paths_list:
        if path.exists():
            size = get_dir_size(path) if path.is_dir() else path.stat().st_size
            print(f"    - {desc}: {format_size(size)}")
    print()

    # Show short path info if applicable
    if short_paths.get('videos') or short_paths.get('images'):
        print("  Short path locations:")
        if short_paths.get('videos'):
            print(f"    - Videos: {short_paths['videos']}")
        if short_paths.get('images'):
            print(f"    - Images: {short_paths['images']}")
        print()

    # Confirmation
    if not dry_run and not yes:
        print("  WARNING: This will permanently delete the above items.")
        print("  Global caches (~/.matcher_*) will NOT be affected.")
        print()

        try:
            confirm = input("  Continue? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return False

        if confirm not in ('y', 'yes'):
            print("  Cancelled.")
            return False

    # Create archive
    print()
    print("-" * 60)
    print("  CREATING ARCHIVE")
    print("-" * 60)

    if dry_run:
        print("  [DRY-RUN] Would create archive/ folder with:")
        for item in ARCHIVE_ESSENTIALS:
            if (project_dir / item).exists():
                print(f"    - {item}")
        archived_files = [item for item in ARCHIVE_ESSENTIALS if (project_dir / item).exists()]
    else:
        archive_dir, archived_files = create_archive(project_dir, level)
        print(f"  Created archive at: {archive_dir}")
        for item in archived_files:
            print(f"    - {item}")

    # Create manifest
    print()
    sizes = {'before': size_before, 'after': size_before - delete_size}
    manifest = generate_manifest(project_dir, level, sizes, stats, archived_files)

    manifest_path = project_dir / "archive" / "manifest.json"
    if not dry_run:
        manifest_path.parent.mkdir(exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Created manifest: {manifest_path.name}")
    else:
        print("  [DRY-RUN] Would create manifest.json")

    # Create ZIP if requested
    if create_zip:
        print()
        if dry_run:
            print(f"  [DRY-RUN] Would create ZIP: {project_dir.name}__ARCHIVE.zip")
        else:
            zip_path = create_zip_archive(project_dir / "archive", project_dir)
            print(f"  Created ZIP: {zip_path.name} ({format_size(zip_path.stat().st_size)})")

    # Delete files
    print()
    print("-" * 60)
    print("  DELETING FILES")
    print("-" * 60)

    freed = delete_paths(delete_paths_list, dry_run)

    # Create .archived marker
    if not dry_run:
        marker = project_dir / ".archived"
        marker.write_text(json.dumps({
            'archived_at': datetime.now().isoformat(),
            'level': level,
            'freed_bytes': freed,
        }, indent=2))

    # Summary
    print()
    print("=" * 60)
    print("  CLEANUP COMPLETE" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 60)
    print()
    print(f"  Space freed: {format_size(freed)}")
    print()
    print("  Preserved:")
    print(f"    - archive/ (essential files)")
    print(f"    - ~/.matcher_global_cache/ (cross-project videos)")
    print(f"    - ~/.matcher_entity_cache/ (cross-project images)")
    print()

    if not dry_run:
        print("  To restore or re-run:")
        print(f"    1. Copy files from archive/ back to project root")
        print(f"    2. Run: python main.py --project \"{project_dir}\" --fresh")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Clean up completed Voiceover-Matcher projects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Cleanup levels:
  cache   Delete caches only, keep videos/images (30-40% space saved)
  media   Delete caches + videos/images (90% space saved)
  full    Delete everything except output (95%+ space saved)

Examples:
  python cleanup_project.py --project "E:\\Projects\\MyDoc"
  python cleanup_project.py --project "E:\\Projects\\MyDoc" --level media
  python cleanup_project.py --project "E:\\Projects\\MyDoc" --level full --zip
  python cleanup_project.py --project "E:\\Projects\\MyDoc" --dry-run

Global caches (~/.matcher_*) are NEVER deleted by this script.
        '''
    )

    parser.add_argument(
        '--project', '-p',
        required=True,
        help='Path to project directory'
    )

    parser.add_argument(
        '--level', '-l',
        choices=[LEVEL_CACHE, LEVEL_MEDIA, LEVEL_FULL],
        default=LEVEL_CACHE,
        help='Cleanup level (default: cache)'
    )

    parser.add_argument(
        '--zip', '-z',
        action='store_true',
        help='Create ZIP archive of essential files'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )

    parser.add_argument(
        '--include-exports', '-e',
        action='store_true',
        help='Also delete export/render files (*.mov, *.mp4) from project root'
    )

    args = parser.parse_args()

    success = cleanup_project(
        project_dir=args.project,
        level=args.level,
        create_zip=args.zip,
        dry_run=args.dry_run,
        yes=args.yes,
        include_exports=args.include_exports
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
