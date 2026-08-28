#!/usr/bin/env python3
"""Fix OTIO file paths for Windows compatibility.

Converts Linux-style paths (E:/v/Degold/..., /home/hpmint/...) to correct Windows
paths by extracting the filename and pointing to the project's segments_archives folder.

Usage:
    python fix_otio_paths.py <otio_file_or_folder> [project_base]

Examples:
    python fix_otio_paths.py "timeline_FULL.otio"
    python fix_otio_paths.py "E:/Edit Job/Client/Project/output" "E:/Edit Job/Client/Project"
"""

import json
import os
import re
import sys
import shutil
from pathlib import Path


def convert_linux_to_windows_path(path: str, project_base: str) -> str:
    """Convert Linux-style path to Windows path using project_base/segments_archives."""
    if not path or not isinstance(path, str):
        return path

    # Skip URLs
    if path.startswith(('http://', 'https://', 'file://')):
        return path

    # Check if it's a Linux path that needs conversion
    needs_conversion = False

    if path.startswith('E:/v/') or path.startswith('e:/v/'):
        needs_conversion = True
    elif path.startswith('/home/hpmint/'):
        needs_conversion = True

    if not needs_conversion:
        return path

    # Extract filename from path
    filename = os.path.basename(path)
    if not filename or '.' not in filename:
        return path  # No clear file extension, skip

    # Build new path to segments_archives
    new_path = os.path.join(project_base, 'segments_archives', filename)
    return new_path


def fix_otio_file(otio_path: str, project_base: str, dry_run: bool = False) -> dict:
    """Fix all target_url paths in a single OTIO file."""
    results = {
        'file': otio_path,
        'changed': 0,
        'samples': [],
        'errors': []
    }

    try:
        with open(otio_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        results['errors'].append(f'Failed to read: {e}')
        return results

    def fix_urls(obj, depth=0):
        nonlocal results
        if depth > 30:
            return
        if isinstance(obj, dict):
            if 'target_url' in obj and obj['target_url']:
                old_url = obj['target_url']
                new_url = convert_linux_to_windows_path(old_url, project_base)
                if old_url != new_url:
                    obj['target_url'] = new_url
                    results['changed'] += 1
                    if len(results['samples']) < 5:
                        results['samples'].append(f'{os.path.basename(old_url)} -> {os.path.basename(new_url)}')
            for v in obj.values():
                fix_urls(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                fix_urls(item, depth + 1)

    fix_urls(data)

    if results['changed'] > 0 and not dry_run:
        # Create backup
        backup_path = otio_path + '.bak'
        shutil.copy(otio_path, backup_path)
        # Write fixed version
        with open(otio_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    return results


def find_project_base(otio_path: str) -> str:
    """Attempt to find project base directory from OTIO file location."""
    path = Path(otio_path)
    if path.is_file():
        path = path.parent

    # Look for segments_archives in parent directories
    current = path
    for _ in range(5):
        if (current / 'segments_archives').exists():
            return str(current)
        current = current.parent

    # Fall back to output directory -> project
    if 'output' in str(path):
        project = Path(path) / '..'
        return str(project.resolve())

    return str(path)


def process_path(input_path: str, project_base: str = None, dry_run: bool = False) -> list:
    """Process a file or folder of OTIO files."""
    results = []
    input_p = Path(input_path)

    # Determine project base if not provided
    if not project_base:
        project_base = find_project_base(input_path)
        print(f'Project base: {project_base}')

    if input_p.is_file() and input_p.suffix == '.otio':
        results.append(fix_otio_file(str(input_p), project_base, dry_run))
    elif input_p.is_dir():
        for otio_file in input_p.glob('*.otio'):
            results.append(fix_otio_file(str(otio_file), project_base, dry_run))
    else:
        print(f'Not a valid OTIO file or directory: {input_path}')
        return results

    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print(f'Args: {sys.argv}')
        sys.exit(1)

    input_path = sys.argv[1]
    project_base = sys.argv[2] if len(sys.argv) > 2 else None
    dry_run = '--dry-run' in sys.argv

    if not os.path.exists(input_path):
        print(f'Path not found: {input_path}')
        sys.exit(1)

    results = process_path(input_path, project_base, dry_run)

    total_changed = 0
    for r in results:
        if r['changed'] > 0:
            print(f"Fixed {r['changed']} paths in {r['file']}")
            if r['samples']:
                for s in r['samples'][:3]:
                    print(f"  {s}")
            total_changed += r['changed']
        if r['errors']:
            print(f"Errors in {r['file']}: {r['errors']}")

    if dry_run:
        print(f'\n[DRY RUN] Would fix {total_changed} paths total')
    else:
        print(f'\nFixed {total_changed} paths total')
        if total_changed > 0:
            print('Backup files created with .bak extension')


if __name__ == '__main__':
    main()