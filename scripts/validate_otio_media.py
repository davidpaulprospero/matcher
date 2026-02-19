#!/usr/bin/env python3
"""
Validate OTIO timeline media references against files on disk.

Usage:
    python scripts/validate_otio_media.py <otio_file>
    python scripts/validate_otio_media.py <otio_file> --fix  # Remove clips with missing media
    python scripts/validate_otio_media.py <otio_file> --rebase E:/new/path  # Rebase paths

Diagnoses:
- Missing media files referenced in timeline
- Audio-only files that cause DaVinci import issues
- Problematic unicode paths
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

# Add scripts directory to path for imports
_script_path = os.path.abspath(__file__)
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(scripts_dir))

import opentimelineio as otio

# Import standardized output functions
from script_utils import print_error, print_warn, print_ok, print_info

# Audio-only extensions that cause DaVinci to hang
AUDIO_ONLY_EXTS = {'.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg'}

# Image extensions (no audio stream - DaVinci will warn but it's benign)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif'}


def extract_file_path(target_url: str) -> Optional[str]:
    """Extract file path from OTIO target_url."""
    if not target_url:
        return None

    # Handle file:// URLs
    if target_url.startswith('file://'):
        parsed = urlparse(target_url)
        path = unquote(parsed.path)
        # On Windows, file:///C:/... becomes /C:/... - remove leading /
        if path.startswith('/') and len(path) > 2 and path[2] == ':':
            path = path[1:]
        return path

    # Plain path
    return target_url


def check_file_exists(file_path: str) -> Tuple[bool, str]:
    """
    Check if file exists and categorize it.

    Returns:
        (exists, category) where category is 'video', 'audio', 'image', or 'unknown'
    """
    if not file_path:
        return False, 'unknown'

    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in AUDIO_ONLY_EXTS:
        category = 'audio'
    elif ext in IMAGE_EXTS:
        category = 'image'
    elif ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}:
        category = 'video'
    else:
        category = 'unknown'

    return path.exists(), category


def has_problematic_path(file_path: str) -> bool:
    """Check if path has characters that cause DaVinci import issues."""
    try:
        if '\ufffd' in file_path or '�' in file_path:
            return True
        for char in file_path:
            if ord(char) > 127:
                return True
        return False
    except Exception:
        return True


def collect_media_references(timeline: otio.schema.Timeline) -> List[Dict]:
    """
    Collect all media references from timeline.

    Returns list of dicts with:
        - track_name: Track containing the clip
        - clip_name: Clip name
        - target_url: Media file path/URL
        - file_path: Extracted file path
    """
    refs = []

    for track in timeline.tracks:
        track_name = track.name or f"Track ({track.kind})"

        for item in track:
            if isinstance(item, otio.schema.Clip):
                media_ref = item.media_reference

                if isinstance(media_ref, otio.schema.ExternalReference):
                    file_path = extract_file_path(media_ref.target_url)
                    refs.append({
                        'track_name': track_name,
                        'clip_name': item.name,
                        'target_url': media_ref.target_url,
                        'file_path': file_path,
                    })

    return refs


def validate_otio(otio_path: str) -> Dict:
    """
    Validate OTIO file and report issues.

    Returns dict with:
        - total_clips: Total clip count
        - missing_files: List of missing file paths
        - audio_only_files: Audio files that may cause issues
        - problematic_paths: Paths with unicode issues
        - image_files: Image files (will warn about audio in DaVinci)
        - valid_files: Files that exist and are valid
    """
    timeline = otio.adapters.read_from_file(otio_path)
    refs = collect_media_references(timeline)

    result = {
        'total_clips': len(refs),
        'missing_files': [],
        'audio_only_files': [],
        'problematic_paths': [],
        'image_files': [],
        'valid_files': [],
    }

    # Track unique paths (avoid duplicate checks)
    checked_paths: Set[str] = set()

    for ref in refs:
        file_path = ref['file_path']
        if not file_path or file_path in checked_paths:
            continue
        checked_paths.add(file_path)

        # Check for problematic paths
        if has_problematic_path(file_path):
            result['problematic_paths'].append({
                'path': file_path,
                'track': ref['track_name'],
                'clip': ref['clip_name'],
            })
            continue

        # Check if file exists and categorize
        exists, category = check_file_exists(file_path)

        if not exists:
            result['missing_files'].append({
                'path': file_path,
                'track': ref['track_name'],
                'clip': ref['clip_name'],
                'category': category,
            })
        elif category == 'audio':
            result['audio_only_files'].append({
                'path': file_path,
                'track': ref['track_name'],
            })
        elif category == 'image':
            result['image_files'].append({
                'path': file_path,
                'track': ref['track_name'],
            })
        else:
            result['valid_files'].append(file_path)

    result['unique_files_checked'] = len(checked_paths)
    return result


def fix_otio(otio_path: str, output_path: Optional[str] = None) -> str:
    """
    Fix OTIO by replacing clips with missing media with gaps.

    Returns path to fixed OTIO file.
    """
    timeline = otio.adapters.read_from_file(otio_path)

    fixed_count = 0

    for track in timeline.tracks:
        items_to_replace = []

        for idx, item in enumerate(list(track)):
            if isinstance(item, otio.schema.Clip):
                media_ref = item.media_reference

                if isinstance(media_ref, otio.schema.ExternalReference):
                    file_path = extract_file_path(media_ref.target_url)

                    if file_path:
                        exists, category = check_file_exists(file_path)

                        # Replace if missing or audio-only
                        if not exists or category == 'audio':
                            items_to_replace.append((idx, item))

        # Replace clips with gaps (reverse order to preserve indices)
        for idx, clip in reversed(items_to_replace):
            gap = otio.schema.Gap(
                source_range=clip.source_range,
                name=f"[FIXED] {clip.name}"
            )
            gap.metadata['original_clip'] = clip.name
            gap.metadata['missing_file'] = True

            track.remove(clip)
            track.insert(idx, gap)
            fixed_count += 1

    # Save fixed timeline
    if output_path is None:
        stem = Path(otio_path).stem
        output_path = str(Path(otio_path).parent / f"{stem}_fixed.otio")

    otio.adapters.write_to_file(timeline, output_path)

    print_ok(f"Fixed {fixed_count} clips with missing/invalid media")
    print_info(f"Saved to: {output_path}")

    return output_path


def print_report(result: Dict, verbose: bool = False):
    """Print validation report."""
    print("\n" + "=" * 60)
    print("  OTIO MEDIA VALIDATION REPORT")
    print("=" * 60)

    print(f"\n  Total clips: {result['total_clips']}")
    print(f"  Unique files checked: {result.get('unique_files_checked', 'N/A')}")
    print(f"  Valid files: {len(result['valid_files'])}")
    print(f"  Missing files: {len(result['missing_files'])}")
    print(f"  Audio-only files: {len(result['audio_only_files'])}")
    print(f"  Image files: {len(result['image_files'])}")
    print(f"  Problematic paths: {len(result['problematic_paths'])}")

    # Missing files (critical - causes random fallback)
    if result['missing_files']:
        print(f"\n  [ERROR] MISSING FILES ({len(result['missing_files'])} - causes random media fallback):")
        # Group by track
        by_track = {}
        for item in result['missing_files']:
            track = item['track']
            if track not in by_track:
                by_track[track] = []
            by_track[track].append(item)

        for track, items in by_track.items():
            print(f"\n    {track}: {len(items)} missing")
            if verbose:
                for item in items[:5]:  # Show first 5
                    print(f"      - {item['path']}")
                if len(items) > 5:
                    print(f"      ... and {len(items) - 5} more")

    # Audio-only files (warning - causes DaVinci import issues)
    if result['audio_only_files']:
        print(f"\n  [WARN] AUDIO-ONLY FILES ({len(result['audio_only_files'])} - may cause DaVinci issues):")
        if verbose:
            for item in result['audio_only_files'][:5]:
                print(f"      - {item['path']}")

    # Image files (info - DaVinci will warn about no audio, but it's normal)
    if result['image_files']:
        print(f"\n  [INFO] IMAGE FILES ({len(result['image_files'])} - DaVinci audio warning is normal):")
        if verbose:
            for item in result['image_files'][:3]:
                print(f"      - {item['path']}")

    # Problematic paths
    if result['problematic_paths']:
        print(f"\n  [WARN] PROBLEMATIC PATHS ({len(result['problematic_paths'])} - unicode issues):")
        if verbose:
            for item in result['problematic_paths'][:5]:
                print(f"      - {item['path']}")

    # Summary and recommendations
    print("\n" + "-" * 60)
    if result['missing_files']:
        print("  RECOMMENDATIONS:")
        print("    1. Re-run pipeline with --output-only to regenerate OTIO")
        print("    2. Or use DaVinci's 'Relink Selected Clips' to fix paths")
        print("    3. Or run with --fix flag to replace missing clips with gaps")
    elif result['audio_only_files']:
        print("  RECOMMENDATIONS:")
        print("    - Audio-only files should not be in video tracks")
        print("    - Re-run pipeline to download video segments")
    else:
        print("  [OK] All media files valid!")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Validate OTIO media references against files on disk'
    )
    parser.add_argument('otio_file', help='Path to OTIO file')
    parser.add_argument('--fix', action='store_true',
                       help='Replace clips with missing media with gaps')
    parser.add_argument('--output', '-o', help='Output path for fixed OTIO')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed file paths')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')

    args = parser.parse_args()

    if not Path(args.otio_file).exists():
        print_error(f"OTIO file not found: {args.otio_file}", exit_code=1)

    # Validate
    result = validate_otio(args.otio_file)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result, verbose=args.verbose)

    # Fix if requested
    if args.fix:
        print("\n  Fixing OTIO...")
        fix_otio(args.otio_file, args.output)

    # Exit with error code if issues found
    if result['missing_files'] or result['audio_only_files']:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
