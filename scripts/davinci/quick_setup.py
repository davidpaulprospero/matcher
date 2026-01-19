#!/usr/bin/env python
"""
Quick Setup for DaVinci Resolve.
One-click import of pipeline output: XML bins -> OTIO -> Diagnose -> Track setup.

This is the recommended way to import pipeline output into DaVinci Resolve.
It handles the proper import order to avoid offline media issues.

Usage:
    python quick_setup.py /path/to/project/output/20260115_123456
    python quick_setup.py /path/to/project  # Auto-finds latest output
    python quick_setup.py --auto            # Uses most recent E:/v project

Run from DaVinci Resolve: Workspace > Scripts > Edit > quick_setup
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

from resolve_utils import get_resolve, get_current_project, TRACK_LAYOUT
from import_workflow import find_output_files, import_xml_media_bins, import_otio_timeline
from diagnose_timeline import diagnose_timeline, print_diagnosis


def find_latest_output(project_path):
    """Find the latest output folder in a project directory."""
    project = Path(project_path)
    output_folder = project / "output"

    if not output_folder.exists():
        # Maybe it IS the output folder
        if any(project.glob("*.otio")):
            return str(project)
        return None

    # Find most recent subfolder (they're named with timestamps like 20260115_123456)
    subfolders = [d for d in output_folder.iterdir() if d.is_dir()]
    if not subfolders:
        # Check if output files are directly in output/
        if any(output_folder.glob("*.otio")):
            return str(output_folder)
        return None

    # Sort by name (timestamp format sorts correctly)
    subfolders.sort(key=lambda d: d.name, reverse=True)
    return str(subfolders[0])


def find_most_recent_project():
    """Find the most recently modified project in E:/v."""
    base = Path("E:/v")
    if not base.exists():
        return None

    # Get all project folders
    projects = [d for d in base.iterdir() if d.is_dir()]
    if not projects:
        return None

    # Sort by modification time
    projects.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return str(projects[0])


def setup_tracks(timeline, preset='primary'):
    """Configure track visibility according to preset."""
    track_count = timeline.GetTrackCount("video")

    presets = {
        'primary': [1, 2, 3],           # V1-V3 only
        'all': list(range(1, 11)),      # All tracks
        'broll': [1, 8],                # Primary + B-roll
    }

    enabled = presets.get(preset, presets['primary'])

    print(f"\n  Configuring tracks (preset: {preset}):")
    for i in range(1, min(track_count + 1, 11)):
        should_enable = i in enabled
        timeline.SetTrackEnable("video", i, should_enable)
        state = "[ON] " if should_enable else "[OFF]"
        desc = TRACK_LAYOUT.get(i, f"V{i}")
        print(f"    V{i}: {state} {desc}")


def run_quick_setup(project, output_path, track_preset='primary'):
    """Run the full quick setup workflow."""
    media_pool = project.GetMediaPool()

    print(f"\n{'='*60}")
    print(f"QUICK SETUP: {os.path.basename(output_path)}")
    print(f"{'='*60}")

    # Step 1: Find files
    print("\n[1/4] Analyzing output folder...")
    files = find_output_files(output_path)

    print(f"  Found: {len(files['xml_bins'])} XML bins, "
          f"{len(files['otio_parts'])} OTIO parts, "
          f"{len(files['otio_full'])} OTIO full")

    if not files['otio_parts'] and not files['otio_full']:
        print("  ERROR: No OTIO files found!")
        return 1

    # Step 2: Import XML media bins
    if files['xml_bins']:
        print("\n[2/4] Importing XML media bins...")
        for xml_file in files['xml_bins']:
            import_xml_media_bins(media_pool, str(xml_file))
    else:
        print("\n[2/4] No XML media bins found (will try to import sources from OTIO)")

    # Step 3: Import OTIO timeline
    print("\n[3/4] Importing OTIO timeline...")
    timelines_to_import = files['otio_parts'] if files['otio_parts'] else files['otio_full']

    created_timelines = []
    for otio_file in timelines_to_import:
        timeline = import_otio_timeline(
            media_pool,
            str(otio_file),
            import_source_clips=(len(files['xml_bins']) == 0)
        )
        if timeline:
            created_timelines.append(timeline)

    if not created_timelines:
        print("  ERROR: No timelines were created!")
        return 1

    # Set first timeline as current
    current_timeline = created_timelines[0]
    project.SetCurrentTimeline(current_timeline)
    print(f"  Set current timeline: {current_timeline.GetName()}")

    # Step 4: Diagnose and configure
    print("\n[4/4] Running diagnostics...")
    results = diagnose_timeline(current_timeline, verbose=False)

    print(f"  Total clips: {results['total_video_clips']}")
    print(f"  Offline: {len(results['offline_clips'])}")
    print(f"  Issues: {len(results['problematic_clips'])}")

    # Configure tracks
    setup_tracks(current_timeline, track_preset)

    # Summary
    print(f"\n{'='*60}")
    print("SETUP COMPLETE!")
    print(f"{'='*60}")

    if created_timelines:
        print(f"\nCreated {len(created_timelines)} timeline(s):")
        for tl in created_timelines:
            print(f"  - {tl.GetName()}")

    if results['offline_clips']:
        print(f"\n⚠ WARNING: {len(results['offline_clips'])} clips are offline!")
        print("  Run relink_media.py to attempt automatic relinking")

    print(f"\nNext steps:")
    print("  1. Switch to Edit page and review timeline")
    print("  2. If media is offline: run relink_media.py")
    print("  3. Use track_manager.py to toggle track sets")
    print("  4. Export when ready: export_all_formats.py")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Quick setup for DaVinci Resolve")
    parser.add_argument('path', nargs='?', help="Path to output folder or project")
    parser.add_argument('--auto', action='store_true', help="Use most recent E:/v project")
    parser.add_argument('--tracks', choices=['primary', 'all', 'broll'], default='primary',
                       help="Track preset to apply (default: primary)")

    args = parser.parse_args()

    # Determine path
    if args.auto:
        project_path = find_most_recent_project()
        if not project_path:
            print("Error: No projects found in E:/v")
            return 1
        output_path = find_latest_output(project_path)
        if not output_path:
            print(f"Error: No output folder found in {project_path}")
            return 1
    elif args.path:
        if os.path.isfile(args.path) and args.path.endswith('.otio'):
            output_path = os.path.dirname(args.path)
        elif os.path.isdir(args.path):
            # Check if it's an output folder or project folder
            if any(Path(args.path).glob("*.otio")):
                output_path = args.path
            else:
                output_path = find_latest_output(args.path)
                if not output_path:
                    print(f"Error: No output folder found in {args.path}")
                    return 1
        else:
            print(f"Error: Path not found: {args.path}")
            return 1
    else:
        print("Usage: quick_setup.py /path/to/output/folder")
        print("       quick_setup.py /path/to/project  # Auto-finds latest output")
        print("       quick_setup.py --auto            # Uses most recent E:/v project")
        return 1

    print(f"Output folder: {output_path}")

    # Connect to DaVinci
    resolve = get_resolve()
    if not resolve:
        return 1

    project = resolve.GetProjectManager().GetCurrentProject()
    if not project:
        print("Error: No project is open in DaVinci Resolve")
        return 1

    print(f"DaVinci project: {project.GetName()}")

    return run_quick_setup(project, output_path, track_preset=args.tracks)


if __name__ == "__main__":
    sys.exit(main())
