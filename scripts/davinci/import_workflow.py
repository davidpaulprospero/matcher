#!/usr/bin/env python
"""
Import Workflow for DaVinci Resolve.
Handles the recommended import order: XML media bins first, then OTIO timeline.

This solves Rule 17 (OTIO offline media) from CLAUDE.md:
- OTIO: DaVinci extracts filename only, media often appears offline
- XML <bin>: Full paths, media links correctly
- Solution: Import XML bins first to populate Media Pool, then OTIO auto-links

Usage:
    python import_workflow.py /path/to/output/folder
    python import_workflow.py /path/to/timeline.otio
    python import_workflow.py --xml-only /path/to/media_bins.xml
    python import_workflow.py --otio-only /path/to/timeline.otio

Run from DaVinci Resolve: Workspace > Scripts > Edit > import_workflow
"""

import sys
import os
import argparse
from pathlib import Path
from resolve_utils import get_resolve, get_current_project


def find_output_files(folder_path):
    """Find OTIO and XML files in an output folder."""
    folder = Path(folder_path)

    # Find XML media bin files
    xml_files = list(folder.glob("*_media_bins.xml")) + list(folder.glob("*_media.xml"))

    # Find OTIO files (prefer split parts if they exist)
    otio_parts = sorted(folder.glob("*_PART*.otio"))
    otio_full = list(folder.glob("timeline.otio")) + list(folder.glob("*.otio"))

    # Remove parts from full list
    otio_full = [f for f in otio_full if '_PART' not in f.name]

    return {
        'xml_bins': xml_files,
        'otio_parts': otio_parts,
        'otio_full': otio_full,
    }


def import_xml_media_bins(media_pool, xml_path):
    """Import media from XML bin file into media pool."""
    print(f"\n  Importing XML media bins: {xml_path}")

    # Create a subfolder for imported media
    root = media_pool.GetRootFolder()
    folder_name = Path(xml_path).stem.replace('_media_bins', '').replace('_media', '')

    # Try to import - this uses the special media bin XML format
    # Note: DaVinci's ImportFolderFromFile is for DRB files
    # For XML media bins, we need to use a different approach

    # Read the XML and extract file paths
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root_elem = tree.getroot()

        # Find all file paths in the XML
        file_paths = []
        for pathurl in root_elem.iter('pathurl'):
            if pathurl.text:
                # Convert file:/// URL to path
                path = pathurl.text
                if path.startswith('file:///'):
                    path = path[8:]  # Remove file:///
                path = path.replace('/', os.sep)  # Normalize separators
                if os.path.exists(path):
                    file_paths.append(path)

        if file_paths:
            print(f"    Found {len(file_paths)} media files")

            # Import in batches
            batch_size = 50
            imported = 0
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i+batch_size]
                result = media_pool.ImportMedia(batch)
                if result:
                    imported += len(result)
                    print(f"    Imported batch: {len(result)} files")

            print(f"    Total imported: {imported} files")
            return imported > 0
        else:
            print("    No media files found in XML")
            return False

    except Exception as e:
        print(f"    Error parsing XML: {e}")
        return False


def import_otio_timeline(media_pool, otio_path, import_source_clips=False):
    """Import OTIO timeline."""
    print(f"\n  Importing OTIO timeline: {otio_path}")

    options = {
        'timelineName': Path(otio_path).stem,
        'importSourceClips': import_source_clips,  # False since we imported via XML
    }

    # If we have short paths configured, add them as source clips path
    for search_path in ['E:/v', 'E:/i']:
        if os.path.exists(search_path):
            options['sourceClipsPath'] = search_path
            break

    timeline = media_pool.ImportTimelineFromFile(str(otio_path), options)

    if timeline:
        print(f"    Success: Created timeline '{timeline.GetName()}'")
        return timeline
    else:
        print(f"    Failed to import timeline")
        return None


def run_import_workflow(project, folder_path, xml_only=False, otio_only=False):
    """Run the full import workflow."""
    media_pool = project.GetMediaPool()

    if os.path.isfile(folder_path):
        # Single file specified
        if folder_path.endswith('.xml'):
            xml_files = [Path(folder_path)]
            otio_parts = []
            otio_full = []
        elif folder_path.endswith('.otio'):
            xml_files = []
            # Check if there are parts
            folder = Path(folder_path).parent
            base_name = Path(folder_path).stem.replace('_PART1', '').replace('_PART2', '').replace('_PART3', '').replace('_PART4', '')
            otio_parts = sorted(folder.glob(f"{base_name}_PART*.otio"))
            if otio_parts:
                otio_full = []
            else:
                otio_full = [Path(folder_path)]
        else:
            print(f"Unknown file type: {folder_path}")
            return 1
    else:
        # Folder specified
        files = find_output_files(folder_path)
        xml_files = files['xml_bins']
        otio_parts = files['otio_parts']
        otio_full = files['otio_full']

    print(f"\nFound files:")
    print(f"  XML media bins: {len(xml_files)}")
    print(f"  OTIO parts: {len(otio_parts)}")
    print(f"  OTIO full: {len(otio_full)}")

    # Step 1: Import XML media bins (unless otio_only)
    if xml_files and not otio_only:
        print("\n" + "=" * 50)
        print("STEP 1: Importing XML Media Bins")
        print("=" * 50)

        for xml_file in xml_files:
            import_xml_media_bins(media_pool, str(xml_file))

    if xml_only:
        print("\nXML-only mode: Skipping OTIO import")
        return 0

    # Step 2: Import OTIO timeline(s)
    if otio_parts or otio_full:
        print("\n" + "=" * 50)
        print("STEP 2: Importing OTIO Timeline(s)")
        print("=" * 50)

        # Prefer parts if they exist (Rule 17: large timelines are split)
        timelines_to_import = otio_parts if otio_parts else otio_full

        created_timelines = []
        for otio_file in timelines_to_import:
            timeline = import_otio_timeline(
                media_pool,
                str(otio_file),
                import_source_clips=(len(xml_files) == 0)  # Only import sources if no XML
            )
            if timeline:
                created_timelines.append(timeline)

        if created_timelines:
            print(f"\n  Created {len(created_timelines)} timeline(s)")
            # Set the first timeline as current
            project.SetCurrentTimeline(created_timelines[0])
            print(f"  Set current timeline: {created_timelines[0].GetName()}")
    else:
        print("\nNo OTIO files found to import")

    print("\n" + "=" * 50)
    print("Import workflow complete!")
    print("=" * 50)

    # Suggest next steps
    print("\nNext steps:")
    print("  1. Run diagnose_timeline.py to check for issues")
    print("  2. Run relink_media.py if media is offline")
    print("  3. Run track_manager.py --primary to enable main tracks")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Import workflow for DaVinci Resolve")
    parser.add_argument('path', nargs='?', help="Path to output folder or specific file")
    parser.add_argument('--xml-only', action='store_true', help="Only import XML media bins")
    parser.add_argument('--otio-only', action='store_true', help="Only import OTIO timeline")

    args = parser.parse_args()

    if not args.path:
        print("Usage: import_workflow.py /path/to/output/folder")
        print("       import_workflow.py /path/to/timeline.otio")
        print("       import_workflow.py --xml-only /path/to/media_bins.xml")
        return 1

    if not os.path.exists(args.path):
        print(f"Error: Path not found: {args.path}")
        return 1

    resolve = get_resolve()
    if not resolve:
        return 1

    project = resolve.GetProjectManager().GetCurrentProject()
    if not project:
        print("Error: No project is open")
        return 1

    print(f"Project: {project.GetName()}")

    return run_import_workflow(project, args.path, xml_only=args.xml_only, otio_only=args.otio_only)


if __name__ == "__main__":
    sys.exit(main())
