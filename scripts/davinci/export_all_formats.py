#!/usr/bin/env python
"""
Export Timeline to Multiple Formats in DaVinci Resolve.
Exports to AAF, EDL, FCPXML, OTIO, and CSV in one operation.

Usage:
    python export_all_formats.py                    # Export to ~/exports/
    python export_all_formats.py /path/to/output   # Export to specific folder
    python export_all_formats.py --formats edl,xml # Export only specific formats

Available formats: aaf, edl, xml, fcpxml, otio, csv, tab

Run from DaVinci Resolve: Workspace > Scripts > Edit > export_all_formats
"""

import sys
import os
import argparse
from datetime import datetime
from resolve_utils import get_resolve, get_current_timeline


def export_timeline(resolve, timeline, output_folder, formats=None):
    """Export timeline to multiple formats."""
    timeline_name = timeline.GetName()

    # Sanitize timeline name for filename
    safe_name = "".join(c for c in timeline_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{safe_name}_{timestamp}"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define available exports
    all_exports = {
        'aaf': {
            'ext': '.aaf',
            'type': resolve.EXPORT_AAF,
            'subtype': resolve.EXPORT_AAF_NEW,
            'name': 'AAF',
        },
        'edl': {
            'ext': '.edl',
            'type': resolve.EXPORT_EDL,
            'subtype': resolve.EXPORT_NONE,
            'name': 'EDL (CMX3600)',
        },
        'xml': {
            'ext': '.xml',
            'type': resolve.EXPORT_FCP_7_XML,
            'subtype': None,
            'name': 'FCP 7 XML',
        },
        'fcpxml': {
            'ext': '.fcpxml',
            'type': resolve.EXPORT_FCPXML_1_10,
            'subtype': None,
            'name': 'FCPXML 1.10',
        },
        'otio': {
            'ext': '.otio',
            'type': resolve.EXPORT_OTIO,
            'subtype': None,
            'name': 'OpenTimelineIO',
        },
        'csv': {
            'ext': '.csv',
            'type': resolve.EXPORT_TEXT_CSV,
            'subtype': None,
            'name': 'CSV',
        },
        'tab': {
            'ext': '.txt',
            'type': resolve.EXPORT_TEXT_TAB,
            'subtype': None,
            'name': 'Tab-delimited',
        },
    }

    # Determine which formats to export
    if formats:
        export_formats = {k: v for k, v in all_exports.items() if k in formats}
    else:
        # Default: export all except tab (redundant with CSV)
        export_formats = {k: v for k, v in all_exports.items() if k != 'tab'}

    print(f"\nExporting timeline: {timeline_name}")
    print(f"Output folder: {output_folder}")
    print("-" * 50)

    results = {}
    for fmt_key, fmt_info in export_formats.items():
        filepath = os.path.join(output_folder, base_filename + fmt_info['ext'])

        print(f"  {fmt_info['name']}...", end=' ')

        try:
            if fmt_info['subtype'] is not None:
                success = timeline.Export(filepath, fmt_info['type'], fmt_info['subtype'])
            else:
                success = timeline.Export(filepath, fmt_info['type'])

            if success:
                print(f"OK -> {os.path.basename(filepath)}")
                results[fmt_key] = {'success': True, 'path': filepath}
            else:
                print("FAILED")
                results[fmt_key] = {'success': False, 'error': 'Export returned False'}
        except Exception as e:
            print(f"ERROR: {e}")
            results[fmt_key] = {'success': False, 'error': str(e)}

    # Summary
    print("-" * 50)
    success_count = sum(1 for r in results.values() if r['success'])
    print(f"Exported {success_count}/{len(results)} formats successfully")

    if success_count > 0:
        print(f"\nFiles saved to: {output_folder}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Export DaVinci Resolve timeline to multiple formats")
    parser.add_argument('output', nargs='?', help="Output folder path")
    parser.add_argument('--formats', type=str, help="Comma-separated list of formats (aaf,edl,xml,fcpxml,otio,csv,tab)")

    args = parser.parse_args()

    resolve = get_resolve()
    if not resolve:
        return 1

    timeline = get_current_timeline(resolve)
    if not timeline:
        return 1

    # Determine output folder
    if args.output:
        output_folder = args.output
    else:
        # Default to ~/exports/ProjectName/
        project = resolve.GetProjectManager().GetCurrentProject()
        project_name = project.GetName() if project else "Unknown"
        output_folder = os.path.join(os.path.expanduser("~"), "exports", project_name)

    # Parse formats
    formats = None
    if args.formats:
        formats = [f.strip().lower() for f in args.formats.split(',')]

    results = export_timeline(resolve, timeline, output_folder, formats)

    # Return code: 0 if all succeeded, 1 if any failed
    all_success = all(r['success'] for r in results.values())
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
