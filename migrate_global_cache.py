#!/usr/bin/env python3
"""
Global Cache Migration Script

Moves the global cache to a new location and optionally updates config.yaml.

Usage:
    python migrate_global_cache.py <destination_path>
    python migrate_global_cache.py <destination_path> --source <source_path>
    python migrate_global_cache.py <destination_path> --update-config
    python migrate_global_cache.py <destination_path> --copy  # Copy instead of move

Examples:
    # Move from default location (~/.matcher_global_cache) to D: drive
    python migrate_global_cache.py D:/matcher_cache

    # Move from custom source to new location
    python migrate_global_cache.py E:/new_cache --source C:/old_cache

    # Move and update config.yaml automatically
    python migrate_global_cache.py D:/matcher_cache --update-config

    # Copy instead of move (keeps original)
    python migrate_global_cache.py D:/backup_cache --copy
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime


# Default source location
DEFAULT_GLOBAL_CACHE = Path.home() / ".matcher_global_cache"

# Expected subdirectories in a valid global cache
CACHE_SUBDIRS = [
    "video_registry",
    "transcripts",
    "embeddings",
    "scenes",
    "topics",
    "keywords"
]


def get_cache_stats(cache_dir: Path) -> dict:
    """Get statistics about a cache directory."""
    stats = {
        "exists": cache_dir.exists(),
        "total_size_mb": 0,
        "video_count": 0,
        "transcript_count": 0,
        "scene_count": 0,
        "subdirs_found": []
    }

    if not cache_dir.exists():
        return stats

    # Check subdirectories
    for subdir in CACHE_SUBDIRS:
        subdir_path = cache_dir / subdir
        if subdir_path.exists():
            stats["subdirs_found"].append(subdir)

    # Count files and size
    total_size = 0
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            file_path = Path(root) / file
            try:
                total_size += file_path.stat().st_size
            except OSError:
                pass

            # Count specific file types
            if file.endswith(".json"):
                parent = Path(root).name
                if parent == "video_registry":
                    stats["video_count"] += 1
                elif parent == "transcripts":
                    stats["transcript_count"] += 1
                elif parent == "scenes":
                    stats["scene_count"] += 1

    stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)

    # Subtract index files from video count
    if stats["video_count"] > 0:
        stats["video_count"] -= 1  # Subtract index.json

    return stats


def validate_source(source_path: Path) -> bool:
    """Validate that source is a valid global cache directory."""
    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}")
        return False

    if not source_path.is_dir():
        print(f"Error: Source path is not a directory: {source_path}")
        return False

    # Check for expected subdirectories
    found_subdirs = []
    for subdir in CACHE_SUBDIRS:
        if (source_path / subdir).exists():
            found_subdirs.append(subdir)

    if not found_subdirs:
        print(f"Warning: No cache subdirectories found in {source_path}")
        print(f"Expected subdirectories: {', '.join(CACHE_SUBDIRS)}")
        response = input("Continue anyway? (y/N): ").strip().lower()
        return response == 'y'

    return True


def validate_destination(dest_path: Path, allow_existing: bool = False) -> bool:
    """Validate destination path is suitable."""
    if dest_path.exists():
        if not allow_existing:
            stats = get_cache_stats(dest_path)
            if stats["subdirs_found"]:
                print(f"Error: Destination already contains a cache: {dest_path}")
                print(f"  Found: {', '.join(stats['subdirs_found'])}")
                print("Use --merge to merge caches (not yet implemented) or choose a different path.")
                return False
            else:
                print(f"Warning: Destination exists but doesn't look like a cache: {dest_path}")
                response = input("Overwrite contents? (y/N): ").strip().lower()
                return response == 'y'

    # Check parent directory exists and is writable
    parent = dest_path.parent
    if not parent.exists():
        print(f"Creating parent directory: {parent}")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error: Cannot create parent directory: {e}")
            return False

    # Check write permission
    try:
        test_file = parent / ".write_test"
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        print(f"Error: Cannot write to destination: {e}")
        return False

    return True


def update_config_yaml(new_cache_path: Path, config_path: Path = None) -> bool:
    """Update config.yaml with new cache path."""
    if config_path is None:
        # Look for config.yaml in common locations
        possible_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent / "config.yaml",
        ]
        for p in possible_paths:
            if p.exists():
                config_path = p
                break

    if config_path is None or not config_path.exists():
        print("Warning: Could not find config.yaml to update")
        return False

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find and replace cache_dir line
        import re

        # Pattern to match cache_dir in global_cache section
        # Handles both quoted and unquoted paths
        pattern = r'(global_cache:.*?cache_dir:\s*)(["\']?)([^"\'\n]+)(["\']?)'

        # Normalize path for config (use forward slashes for cross-platform)
        new_path_str = str(new_cache_path).replace('\\', '/')

        # Use a simpler approach - line by line
        lines = content.split('\n')
        in_global_cache = False
        updated = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('global_cache:'):
                in_global_cache = True
            elif in_global_cache and stripped.startswith('cache_dir:'):
                # Found the line to update
                indent = len(line) - len(line.lstrip())
                lines[i] = ' ' * indent + f'cache_dir: "{new_path_str}"  # Updated by migrate_global_cache.py'
                updated = True
                break
            elif in_global_cache and stripped and not stripped.startswith('#') and not stripped.startswith('-'):
                # Check if we've left global_cache section (new top-level key)
                if not line.startswith(' ') and not line.startswith('\t'):
                    in_global_cache = False

        if updated:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"Updated config.yaml: cache_dir = \"{new_path_str}\"")
            return True
        else:
            print("Warning: Could not find cache_dir setting in config.yaml")
            return False

    except Exception as e:
        print(f"Error updating config.yaml: {e}")
        return False


def migrate_cache(source: Path, dest: Path, copy_mode: bool = False) -> bool:
    """Move or copy cache from source to destination."""
    operation = "Copying" if copy_mode else "Moving"
    print(f"\n{operation} global cache:")
    print(f"  From: {source}")
    print(f"  To:   {dest}")

    source_stats = get_cache_stats(source)
    print(f"\nSource cache statistics:")
    print(f"  Size: {source_stats['total_size_mb']} MB")
    print(f"  Videos: {source_stats['video_count']}")
    print(f"  Transcripts: {source_stats['transcript_count']}")
    print(f"  Scenes: {source_stats['scene_count']}")

    # Confirm before proceeding
    response = input(f"\nProceed with {operation.lower()}? (y/N): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return False

    try:
        if copy_mode:
            print(f"\nCopying cache...")
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            print(f"\nMoving cache...")
            shutil.move(str(source), str(dest))

        # Verify destination
        dest_stats = get_cache_stats(dest)

        print(f"\nMigration complete!")
        print(f"  New location: {dest}")
        print(f"  Size: {dest_stats['total_size_mb']} MB")
        print(f"  Videos: {dest_stats['video_count']}")

        if not copy_mode:
            print(f"\nOriginal location {source} has been removed.")

        return True

    except shutil.Error as e:
        print(f"Error during migration: {e}")
        return False
    except OSError as e:
        print(f"OS error during migration: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate global cache to a new location",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s D:/matcher_cache
      Move cache from ~/.matcher_global_cache to D:/matcher_cache

  %(prog)s E:/cache --source C:/old_cache
      Move cache from custom source location

  %(prog)s D:/cache --update-config
      Move cache and update config.yaml automatically

  %(prog)s D:/backup --copy
      Copy cache instead of moving (preserves original)

  %(prog)s --show-current
      Show current cache location and statistics
        """
    )

    parser.add_argument(
        "destination",
        nargs="?",
        help="Destination path for the global cache"
    )
    parser.add_argument(
        "--source", "-s",
        help=f"Source cache path (default: {DEFAULT_GLOBAL_CACHE})",
        default=str(DEFAULT_GLOBAL_CACHE)
    )
    parser.add_argument(
        "--update-config", "-u",
        action="store_true",
        help="Update config.yaml with new cache path"
    )
    parser.add_argument(
        "--config-path", "-c",
        help="Path to config.yaml (auto-detected if not specified)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move (preserves original)"
    )
    parser.add_argument(
        "--show-current",
        action="store_true",
        help="Show current cache location and statistics"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()

    # Show current cache info
    if args.show_current:
        print(f"Default cache location: {DEFAULT_GLOBAL_CACHE}")
        print(f"Checking: {source}")

        stats = get_cache_stats(source)
        if stats["exists"]:
            print(f"\nCache found:")
            print(f"  Size: {stats['total_size_mb']} MB")
            print(f"  Videos: {stats['video_count']}")
            print(f"  Transcripts: {stats['transcript_count']}")
            print(f"  Scenes: {stats['scene_count']}")
            print(f"  Subdirectories: {', '.join(stats['subdirs_found'])}")
        else:
            print(f"\nNo cache found at {source}")
        return 0

    # Require destination for migration
    if not args.destination:
        parser.print_help()
        print("\nError: destination path is required for migration")
        return 1

    dest = Path(args.destination).expanduser().resolve()

    # Validate paths
    if not validate_source(source):
        return 1

    if not validate_destination(dest):
        return 1

    # Dry run mode
    if args.dry_run:
        operation = "copy" if args.copy else "move"
        print(f"\nDry run - would {operation}:")
        print(f"  From: {source}")
        print(f"  To:   {dest}")

        stats = get_cache_stats(source)
        print(f"\nCache size: {stats['total_size_mb']} MB")

        if args.update_config:
            print(f"\nWould update config.yaml with: cache_dir: \"{dest}\"")
        return 0

    # Perform migration
    if not migrate_cache(source, dest, copy_mode=args.copy):
        return 1

    # Update config if requested
    if args.update_config:
        config_path = Path(args.config_path) if args.config_path else None
        update_config_yaml(dest, config_path)
    else:
        print(f"\nRemember to update your config.yaml:")
        print(f"  global_cache:")
        print(f"    cache_dir: \"{str(dest).replace(chr(92), '/')}\"")

    return 0


if __name__ == "__main__":
    sys.exit(main())
