#!/usr/bin/env python3
"""
Fixture Adoption Verification Script (US-010, extended US-003)

Verifies that factory fixtures from tests/conftest.py and tests/fixtures/
are adopted in at least the required minimum number of test files.

Also detects anti-patterns: manual object construction that should use factories.

Usage:
    python scripts/verify_fixture_adoption.py           # Basic report
    python scripts/verify_fixture_adoption.py --strict  # Fail if <80% adoption in new files
    python scripts/verify_fixture_adoption.py --suggest # Show fixture replacements
    python scripts/verify_fixture_adoption.py --ci      # CI-friendly output (exit code only)

Exit codes:
    0: All adoption requirements met
    1: One or more requirements not met (or strict mode threshold failed)

Requirements:
    - Factory fixtures should be used in at least 10 test files
    - Fixture definition files (conftest.py, fixtures/__init__.py) are excluded
    - In --strict mode: 80% of new test files must use factories
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Import standardized output functions
from script_utils import print_ok, print_warn, print_error, print_info, print_header

# Change to project root so relative paths work correctly
os.chdir(project_root)

# Minimum required adoption levels
MIN_FIXTURE_ADOPTION_FILES = 10
STRICT_ADOPTION_THRESHOLD = 0.80  # 80% of test files should use factories

# Factory fixtures to track (what we WANT tests to use)
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

# Manual construction patterns to detect (anti-patterns)
# Format: (pattern_regex, description, suggested_fixture)
MANUAL_CONSTRUCTION_PATTERNS = [
    # Direct Config instantiation
    (r"\bConfig\s*\(", "Direct Config() instantiation", "config_factory or create_mock_config"),
    # Direct PipelineState instantiation
    (r"\bPipelineState\s*\(", "Direct PipelineState() instantiation", "pipeline_state_factory or create_mock_state"),
    # Direct SRTSegment instantiation
    (r"\bSRTSegment\s*\(", "Direct SRTSegment() instantiation", "srt_segment_factory"),
    # Direct Match instantiation
    (r"\bMatch\s*\(", "Direct Match() instantiation", "match_factory"),
    # Direct MatchResult instantiation
    (r"\bMatchResult\s*\(", "Direct MatchResult() instantiation", "match_result_factory"),
    # Direct DownloadedVideo instantiation
    (r"\bDownloadedVideo\s*\(", "Direct DownloadedVideo() instantiation", "tests/fixtures/downloader_fixtures.py"),
    # Checkpoint dict creation (manual)
    (r'["\'](last_completed_stage|checkpoint_version)["\']', "Manual checkpoint dict creation", "create_test_checkpoint"),
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


def check_manual_construction(file_path: Path) -> List[Dict]:
    """
    Check for manual object construction patterns that should use fixtures.

    Returns:
        List of dicts with pattern info: {pattern, description, suggestion, line_numbers}
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")
    except Exception:
        return []

    findings = []

    for pattern_regex, description, suggestion in MANUAL_CONSTRUCTION_PATTERNS:
        line_numbers = []
        for line_num, line in enumerate(lines, start=1):
            # Skip comments and import statements
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("from ") or stripped.startswith("import "):
                continue

            if re.search(pattern_regex, line):
                line_numbers.append(line_num)

        if line_numbers:
            findings.append({
                "pattern": pattern_regex,
                "description": description,
                "suggestion": suggestion,
                "line_numbers": line_numbers,
                "count": len(line_numbers)
            })

    return findings


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


def get_manual_construction_report(tests_dir: Path) -> Dict[Path, List[Dict]]:
    """
    Generate report of manual object construction (anti-patterns).

    Returns:
        Dict mapping file paths to list of findings
    """
    test_files = find_test_files(tests_dir)
    report = {}

    for file_path in test_files:
        findings = check_manual_construction(file_path)
        if findings:
            report[file_path] = findings

    return report


def calculate_adoption_rate(tests_dir: Path) -> Tuple[float, int, int]:
    """
    Calculate fixture adoption rate.

    Returns:
        Tuple of (adoption_rate, files_using_factories, total_test_files)
    """
    test_files = find_test_files(tests_dir)
    total_files = len(test_files)

    if total_files == 0:
        return 0.0, 0, 0

    files_using_fixtures = 0
    for file_path in test_files:
        used_fixtures = check_fixture_usage(file_path, FACTORY_FIXTURES)
        if used_fixtures:
            files_using_fixtures += 1

    adoption_rate = files_using_fixtures / total_files
    return adoption_rate, files_using_fixtures, total_files


def print_suggestions(tests_dir: Path, manual_report: Dict[Path, List[Dict]]) -> None:
    """Print fixture replacement suggestions for files with manual construction."""
    print("\n" + "=" * 60)
    print("  Fixture Replacement Suggestions")
    print("=" * 60)
    print()

    if not manual_report:
        print("  No manual construction patterns found. Great job!")
        return

    for file_path in sorted(manual_report.keys()):
        rel_path = file_path.relative_to(tests_dir)
        print(f"  {rel_path}:")
        for finding in manual_report[file_path]:
            lines_str = ", ".join(str(ln) for ln in finding["line_numbers"][:5])
            if len(finding["line_numbers"]) > 5:
                lines_str += f" (+{len(finding['line_numbers']) - 5} more)"
            print(f"    - {finding['description']}")
            print(f"      Lines: {lines_str}")
            print(f"      Suggestion: Use {finding['suggestion']}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify fixture factory adoption in test files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_fixture_adoption.py           # Basic report
  python scripts/verify_fixture_adoption.py --strict  # Fail if <80% adoption
  python scripts/verify_fixture_adoption.py --suggest # Show replacement suggestions
  python scripts/verify_fixture_adoption.py --ci      # CI-friendly (minimal output)

CI Integration:
  Add to .github/workflows/tests.yml:
    - name: Check fixture adoption
      run: python scripts/verify_fixture_adoption.py --strict

  Or as pre-commit hook in .pre-commit-config.yaml:
    - repo: local
      hooks:
        - id: fixture-adoption
          name: Check fixture adoption
          entry: python scripts/verify_fixture_adoption.py --strict
          language: python
          pass_filenames: false
        """
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=f"Fail if adoption rate is below {STRICT_ADOPTION_THRESHOLD * 100:.0f}%%"
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Show suggested fixture replacements for manual construction"
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI-friendly output (minimal, exit code only)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    # Find tests directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    tests_dir = repo_root / "tests"

    if not tests_dir.exists():
        if not args.ci:
            print(f"ERROR: Tests directory not found: {tests_dir}")
        sys.exit(1)

    # Generate reports
    total_files, per_fixture, files_with_usage = get_fixture_adoption_report(tests_dir)
    manual_report = get_manual_construction_report(tests_dir)
    adoption_rate, adopting_files, all_files = calculate_adoption_rate(tests_dir)

    # Count total manual construction issues
    total_manual_issues = sum(
        sum(f["count"] for f in findings)
        for findings in manual_report.values()
    )
    files_with_manual = len(manual_report)

    # JSON output mode
    if args.json:
        import json
        result = {
            "timestamp": datetime.now().isoformat(),
            "adoption_rate": round(adoption_rate * 100, 1),
            "threshold": STRICT_ADOPTION_THRESHOLD * 100,
            "files_using_fixtures": total_files,
            "total_test_files": all_files,
            "files_with_manual_construction": files_with_manual,
            "total_manual_issues": total_manual_issues,
            "pass": (
                total_files >= MIN_FIXTURE_ADOPTION_FILES
                and (not args.strict or adoption_rate >= STRICT_ADOPTION_THRESHOLD)
            ),
            "per_fixture_usage": {f: len(paths) for f, paths in per_fixture.items()},
        }
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["pass"] else 1)

    # CI mode - minimal output
    if args.ci:
        pass_basic = total_files >= MIN_FIXTURE_ADOPTION_FILES
        pass_strict = adoption_rate >= STRICT_ADOPTION_THRESHOLD

        if args.strict:
            success = pass_basic and pass_strict
            status = "PASS" if success else "FAIL"
            print(f"{status}: Adoption {adoption_rate * 100:.1f}% (threshold: {STRICT_ADOPTION_THRESHOLD * 100:.0f}%)")
        else:
            success = pass_basic
            status = "PASS" if success else "FAIL"
            print(f"{status}: {total_files} files use fixtures (min: {MIN_FIXTURE_ADOPTION_FILES})")

        sys.exit(0 if success else 1)

    # Full report mode
    print("=" * 60)
    print("  Fixture Adoption Verification Report")
    print("=" * 60)
    print()

    # Print summary
    print(f"Fixtures tracked: {len(FACTORY_FIXTURES)}")
    print(f"Test files using fixtures: {total_files}/{all_files}")
    print(f"Adoption rate: {adoption_rate * 100:.1f}%")
    print(f"Minimum required: {MIN_FIXTURE_ADOPTION_FILES}")
    if args.strict:
        print(f"Strict threshold: {STRICT_ADOPTION_THRESHOLD * 100:.0f}%")
    print()

    # Print per-fixture usage
    print("Per-fixture usage:")
    print("-" * 40)
    for fixture in FACTORY_FIXTURES:
        count = len(per_fixture[fixture])
        status = "[x]" if count > 0 else "[ ]"
        print(f"  {status} {fixture}: {count} files")
    print()

    # Print manual construction summary
    print("Manual construction detection (anti-patterns):")
    print("-" * 40)
    print(f"  Files with manual construction: {files_with_manual}")
    print(f"  Total manual construction calls: {total_manual_issues}")
    if files_with_manual > 0:
        # Show top offenders
        sorted_manual = sorted(
            manual_report.items(),
            key=lambda x: sum(f["count"] for f in x[1]),
            reverse=True
        )[:5]
        print("  Top offenders:")
        for file_path, findings in sorted_manual:
            rel_path = file_path.relative_to(tests_dir)
            total = sum(f["count"] for f in findings)
            print(f"    - {rel_path}: {total} instances")
    print()

    # Print files using fixtures (sorted by path, limited)
    print("Files using factory fixtures (showing first 20):")
    print("-" * 40)
    sorted_files = sorted(files_with_usage.keys())[:20]
    for file_path in sorted_files:
        rel_path = file_path.relative_to(tests_dir)
        fixtures = ", ".join(sorted(files_with_usage[file_path]))
        print(f"  {rel_path}")
        print(f"    -> {fixtures}")
    if len(files_with_usage) > 20:
        print(f"  ... and {len(files_with_usage) - 20} more files")
    print()

    # Check requirements
    success = True

    if total_files < MIN_FIXTURE_ADOPTION_FILES:
        print(f"FAIL: Only {total_files} files use fixtures (minimum: {MIN_FIXTURE_ADOPTION_FILES})")
        success = False
    else:
        print(f"PASS: {total_files} files use fixtures (minimum: {MIN_FIXTURE_ADOPTION_FILES})")

    # Strict mode check
    if args.strict:
        if adoption_rate < STRICT_ADOPTION_THRESHOLD:
            print(f"FAIL (strict): Adoption rate {adoption_rate * 100:.1f}% < {STRICT_ADOPTION_THRESHOLD * 100:.0f}%")
            success = False
        else:
            print(f"PASS (strict): Adoption rate {adoption_rate * 100:.1f}% >= {STRICT_ADOPTION_THRESHOLD * 100:.0f}%")

    print()
    print("=" * 60)

    # Print suggestions if requested
    if args.suggest:
        print_suggestions(tests_dir, manual_report)

    # Print CI integration hint
    if not args.ci and not args.json:
        print()
        print("CI Integration:")
        print("  Add to .github/workflows/tests.yml:")
        print("    - name: Check fixture adoption")
        print("      run: python scripts/verify_fixture_adoption.py --strict")
        print()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
