#!/usr/bin/env python3
"""
Fixture Adoption Verification Script (US-010)

Verifies that factory fixtures from tests/conftest.py and tests/fixtures/
are adopted in at least the required minimum number of test files.

This script can be run as part of CI to ensure fixture usage doesn't regress.

Usage:
    python scripts/verify_fixture_adoption.py

Exit codes:
    0: All adoption requirements met
    1: One or more requirements not met

Requirements:
    - Factory fixtures should be used in at least 10 test files
    - Fixture definition files (conftest.py, fixtures/__init__.py) are excluded
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

# Minimum required adoption levels
MIN_FIXTURE_ADOPTION_FILES = 10

# Factory fixtures to track
FACTORY_FIXTURES = [
    # From tests/conftest.py (pytest fixtures)
    "srt_segment_factory",
    "config_factory",
    "pipeline_state_factory",
    "match_result_factory",
    "match_factory",
    # From tests/fixtures/__init__.py (helper functions)
    "create_mock_config",
    "create_mock_state",
    "create_test_checkpoint",
    "create_checkpoint_with_populated_stages",
    "create_concurrent_escalation_fixture",
    "create_otio_timeline_fixture",
    "create_broll_propagation_chain_state",
]

# Files to exclude (definition files)
EXCLUDED_FILES = {
    "conftest.py",
    "__init__.py",
    "README.md",
}


def find_test_files(tests_dir: Path) -> List[Path]:
    """Find all Python test files, excluding definition files."""
    test_files = []

    for root, dirs, files in os.walk(tests_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != "__pycache__"]

        for file in files:
            if file.endswith(".py") and file not in EXCLUDED_FILES:
                test_files.append(Path(root) / file)

    return test_files


def check_fixture_usage(file_path: Path, fixtures: List[str]) -> Set[str]:
    """Check which fixtures are used in a file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return set()

    used = set()
    for fixture in fixtures:
        # Check for fixture usage (as parameter or direct call)
        if re.search(rf"\b{fixture}\b", content):
            used.add(fixture)

    return used


def get_fixture_adoption_report(tests_dir: Path) -> Tuple[int, dict, dict]:
    """
    Generate fixture adoption report.

    Returns:
        Tuple of (files_using_fixtures, per_fixture_usage, files_with_usage)
    """
    test_files = find_test_files(tests_dir)

    files_using_fixtures: Set[Path] = set()
    per_fixture_usage: dict = {f: [] for f in FACTORY_FIXTURES}
    files_with_usage: dict = {}

    for file_path in test_files:
        used_fixtures = check_fixture_usage(file_path, FACTORY_FIXTURES)

        if used_fixtures:
            files_using_fixtures.add(file_path)
            files_with_usage[file_path] = used_fixtures

            for fixture in used_fixtures:
                per_fixture_usage[fixture].append(file_path)

    return len(files_using_fixtures), per_fixture_usage, files_with_usage


def main():
    """Main entry point."""
    # Find tests directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    tests_dir = repo_root / "tests"

    if not tests_dir.exists():
        print(f"ERROR: Tests directory not found: {tests_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Fixture Adoption Verification Report")
    print("=" * 60)
    print()

    # Generate report
    total_files, per_fixture, files_with_usage = get_fixture_adoption_report(tests_dir)

    # Print summary
    print(f"Fixtures tracked: {len(FACTORY_FIXTURES)}")
    print(f"Test files using fixtures: {total_files}")
    print(f"Minimum required: {MIN_FIXTURE_ADOPTION_FILES}")
    print()

    # Print per-fixture usage
    print("Per-fixture usage:")
    print("-" * 40)
    for fixture in FACTORY_FIXTURES:
        count = len(per_fixture[fixture])
        status = "[x]" if count > 0 else "[ ]"
        print(f"  {status} {fixture}: {count} files")
    print()

    # Print files using fixtures (sorted by path)
    print("Files using factory fixtures:")
    print("-" * 40)
    for file_path in sorted(files_with_usage.keys()):
        rel_path = file_path.relative_to(tests_dir)
        fixtures = ", ".join(sorted(files_with_usage[file_path]))
        print(f"  {rel_path}")
        print(f"    -> {fixtures}")
    print()

    # Check requirements
    success = True

    if total_files < MIN_FIXTURE_ADOPTION_FILES:
        print(f"FAIL: Only {total_files} files use fixtures (minimum: {MIN_FIXTURE_ADOPTION_FILES})")
        success = False
    else:
        print(f"PASS: {total_files} files use fixtures (minimum: {MIN_FIXTURE_ADOPTION_FILES})")

    print()
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
