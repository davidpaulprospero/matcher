#!/usr/bin/env python3
"""
Bulk Test Marker Assignment Script

Analyzes test files using categorize_tests.py output and adds appropriate
pytest markers to unmarked tests. Supports dry-run mode for previewing
changes before applying.

Usage:
    # Preview proposed marker additions (dry-run)
    python scripts/assign_test_markers.py --dry-run

    # Apply markers to test files
    python scripts/assign_test_markers.py --apply

    # Apply only 'fast' markers
    python scripts/assign_test_markers.py --apply --marker fast

    # Apply markers to specific directory
    python scripts/assign_test_markers.py --apply --path tests/test_agents

    # Show verbose output with file diffs
    python scripts/assign_test_markers.py --dry-run --verbose

    # Export report as JSON
    python scripts/assign_test_markers.py --dry-run --output report.json
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

from script_utils import print_ok, print_warn, print_error, print_info, print_header, set_verbosity

# Change to project root so relative paths work correctly
os.chdir(project_root)

# Import from categorize_tests (after sys.path setup)
from categorize_tests import (
    TestInfo,
    analyze_test_directory,
    KNOWN_MARKERS,
)


@dataclass
class MarkerEdit:
    """A proposed marker edit to a test file."""
    file_path: str
    test_name: str
    line_number: int
    marker_to_add: str
    reason: str
    applied: bool = False
    error: str = ""


def collect_edits(tests: list[TestInfo], marker_filter: Optional[str] = None) -> list[MarkerEdit]:
    """Collect all proposed marker edits from test analysis."""
    edits = []

    for test in tests:
        if not test.suggested_marker:
            continue

        # Skip if marker filter specified and doesn't match
        if marker_filter and test.suggested_marker != marker_filter:
            continue

        # Skip if test already has any known marker
        has_marker = any(m in KNOWN_MARKERS for m in test.markers)
        if has_marker:
            continue

        edit = MarkerEdit(
            file_path=test.file_path,
            test_name=test.test_name,
            line_number=test.line_number,
            marker_to_add=test.suggested_marker,
            reason=test.suggestion_reason,
        )
        edits.append(edit)

    return edits


def check_existing_markers(file_path: Path) -> set[str]:
    """Check what markers are already in use in a file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return set()

    markers = set()
    # Find @pytest.mark.X patterns
    for match in re.finditer(r"@pytest\.mark\.(\w+)", content):
        markers.add(match.group(1))

    return markers


def apply_marker_to_file(file_path: Path, edits: list[MarkerEdit], verbose: bool = False) -> tuple[int, int]:
    """Apply marker edits to a single file.

    Returns:
        Tuple of (applied_count, error_count)
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")
    except Exception as e:
        for edit in edits:
            edit.error = str(e)
        return 0, len(edits)

    # Sort edits by line number descending so we can modify from bottom up
    # This prevents line number shifts from affecting earlier edits
    sorted_edits = sorted(edits, key=lambda e: e.line_number, reverse=True)

    applied = 0
    errors = 0

    for edit in sorted_edits:
        line_idx = edit.line_number - 1  # Convert to 0-indexed

        if line_idx < 0 or line_idx >= len(lines):
            edit.error = f"Line {edit.line_number} out of range"
            errors += 1
            continue

        line = lines[line_idx]

        # Check if this line starts the test function or method
        if not line.strip().startswith("def test_"):
            # Look for the def line (might be on next line after decorators)
            found = False
            for i in range(line_idx, min(line_idx + 5, len(lines))):
                if lines[i].strip().startswith("def test_"):
                    line_idx = i
                    line = lines[line_idx]
                    found = True
                    break
            if not found:
                edit.error = f"Could not find test function at line {edit.line_number}"
                errors += 1
                continue

        # Get indentation of the def line
        indent = len(line) - len(line.lstrip())
        indent_str = " " * indent

        # Check if marker already exists above
        if line_idx > 0:
            prev_lines = "\n".join(lines[max(0, line_idx - 5):line_idx])
            if f"@pytest.mark.{edit.marker_to_add}" in prev_lines:
                edit.error = f"Marker @pytest.mark.{edit.marker_to_add} already present"
                errors += 1
                continue

        # Insert the marker decorator above the def line
        marker_line = f"{indent_str}@pytest.mark.{edit.marker_to_add}"
        lines.insert(line_idx, marker_line)

        edit.applied = True
        applied += 1

        if verbose:
            rel_path = file_path.relative_to(Path.cwd()) if file_path.is_absolute() else file_path
            print_info(f"  + {rel_path}:{edit.line_number}: @pytest.mark.{edit.marker_to_add}")

    # Write back to file if any edits applied
    if applied > 0:
        new_content = "\n".join(lines)

        # Ensure pytest import exists
        if "@pytest.mark." in new_content and "import pytest" not in new_content:
            # Add import at top after any existing imports
            import_lines = []
            other_lines = []
            in_imports = True
            for line in lines:
                if in_imports and (line.startswith("import ") or line.startswith("from ") or not line.strip()):
                    import_lines.append(line)
                else:
                    in_imports = False
                    other_lines.append(line)

            # Add pytest import
            import_lines.append("import pytest")
            new_content = "\n".join(import_lines + other_lines)

        file_path.write_text(new_content, encoding="utf-8")

    return applied, errors


def print_dry_run_report(edits: list[MarkerEdit], verbose: bool = False):
    """Print dry-run report of proposed changes."""
    print_header("DRY-RUN: PROPOSED MARKER ADDITIONS")

    if not edits:
        print_info("No marker additions needed!")
        return

    # Group by marker type
    by_marker = defaultdict(list)
    for edit in edits:
        by_marker[edit.marker_to_add].append(edit)

    # Summary
    print_info(f"Total proposed additions: {len(edits)}")
    for marker, marker_edits in sorted(by_marker.items(), key=lambda x: -len(x[1])):
        print_info(f"  @pytest.mark.{marker}: {len(marker_edits)} tests")

    if verbose:
        print_info("Detailed Changes:")
        # Group by file
        by_file = defaultdict(list)
        for edit in edits:
            by_file[edit.file_path].append(edit)

        for file_path, file_edits in sorted(by_file.items()):
            try:
                rel_path = Path(file_path).relative_to(Path.cwd())
            except ValueError:
                rel_path = file_path
            print_info(f"\n{rel_path}:")
            for edit in sorted(file_edits, key=lambda e: e.line_number):
                print_info(f"  Line {edit.line_number}: {edit.test_name}")
                print_info(f"    + @pytest.mark.{edit.marker_to_add}")
                print_info(f"    Reason: {edit.reason}")

    print_info("To apply these changes, run:")
    print_info("  python scripts/assign_test_markers.py --apply")


def print_apply_report(edits: list[MarkerEdit]):
    """Print report after applying markers."""
    print_header("MARKER APPLICATION RESULTS")

    applied = [e for e in edits if e.applied]
    errors = [e for e in edits if e.error]

    print_info(f"Total edits attempted: {len(edits)}")
    print_info(f"Successfully applied: {len(applied)}")
    print_info(f"Errors: {len(errors)}")

    if applied:
        # Group by marker
        by_marker = defaultdict(list)
        for edit in applied:
            by_marker[edit.marker_to_add].append(edit)

        print_info("Applied Markers:")
        for marker, marker_edits in sorted(by_marker.items(), key=lambda x: -len(x[1])):
            print_info(f"  @pytest.mark.{marker}: {len(marker_edits)} tests")

    if errors:
        print_warn("Errors:")
        for edit in errors[:10]:  # Show first 10 errors
            try:
                rel_path = Path(edit.file_path).relative_to(Path.cwd())
            except ValueError:
                rel_path = edit.file_path
            print_warn(f"  {rel_path}:{edit.line_number}: {edit.error}")
        if len(errors) > 10:
            print_warn(f"  ... and {len(errors) - 10} more errors")


def write_json_report(edits: list[MarkerEdit], output_path: Path):
    """Write edit report as JSON."""
    # Group by marker
    by_marker = defaultdict(list)
    for edit in edits:
        by_marker[edit.marker_to_add].append({
            "file": str(Path(edit.file_path).relative_to(Path.cwd()) if Path(edit.file_path).is_absolute() else edit.file_path),
            "test": edit.test_name,
            "line": edit.line_number,
            "reason": edit.reason,
            "applied": edit.applied,
            "error": edit.error,
        })

    report = {
        "total_edits": len(edits),
        "applied_count": sum(1 for e in edits if e.applied),
        "error_count": sum(1 for e in edits if e.error),
        "edits_by_marker": dict(by_marker),
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_ok(f"JSON report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk assign pytest markers to unmarked tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("tests"),
        help="Path to test directory or file (default: tests/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show proposed changes without applying (default mode)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply marker additions to test files",
    )
    parser.add_argument(
        "--marker",
        choices=list(KNOWN_MARKERS.keys()),
        help="Only add this specific marker type",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON report to this path",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    # Default to dry-run if neither specified
    if not args.apply and not args.dry_run:
        args.dry_run = True

    # Validate args
    if args.apply and args.dry_run:
        parser.error("Cannot use both --apply and --dry-run")

    # Change to project root if running from scripts/
    if not args.path.exists():
        project_root = Path(__file__).parent.parent
        args.path = project_root / args.path
        if not args.path.exists():
            print_error(f"Test path not found: {args.path}", exit_code=1)

    print_info(f"Analyzing tests in: {args.path}")
    tests = analyze_test_directory(args.path)

    if not tests:
        print_warn("No tests found!")
        sys.exit(1)

    # Collect proposed edits
    edits = collect_edits(tests, marker_filter=args.marker)

    if args.dry_run:
        print_dry_run_report(edits, verbose=args.verbose)
    else:
        # Apply mode
        if not edits:
            print_info("No marker additions needed!")
            return

        print_info(f"Applying {len(edits)} marker additions...")

        # Group edits by file
        by_file = defaultdict(list)
        for edit in edits:
            by_file[edit.file_path].append(edit)

        total_applied = 0
        total_errors = 0

        for file_path, file_edits in by_file.items():
            applied, errors = apply_marker_to_file(
                Path(file_path),
                file_edits,
                verbose=args.verbose,
            )
            total_applied += applied
            total_errors += errors

        print_apply_report(edits)

    # Write JSON output if specified
    if args.output:
        write_json_report(edits, args.output)


if __name__ == "__main__":
    main()
