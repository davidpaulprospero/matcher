#!/usr/bin/env python3
"""
Test Categorization Script

Analyzes test marker distribution and suggests markers for unmarked tests
based on I/O patterns (file operations, network requests, subprocess calls).

Usage:
    # Generate marker distribution report
    python scripts/categorize_tests.py

    # Output detailed CSV with suggestions
    python scripts/categorize_tests.py --output report.csv

    # Auto-suggest markers with reasons
    python scripts/categorize_tests.py --auto-suggest

    # Analyze specific directory
    python scripts/categorize_tests.py --path tests/test_agents

    # JSON output for CI integration
    python scripts/categorize_tests.py --json --output report.json
"""

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Known markers from pytest.ini
KNOWN_MARKERS = {
    "fast": "Pure unit tests (no I/O, no network, mocks only)",
    "slow": "Slow tests (>5 seconds)",
    "integration": "Integration tests requiring external resources",
    "stress": "Stress/chaos tests",
    "simulation": "Healing simulation tests",
    "requires_api": "Tests requiring API keys",
    "requires_network": "Tests making HTTP requests",
    "flaky": "Intermittent failures (timing, network, race conditions)",
}

# I/O patterns that suggest markers
IO_PATTERNS = {
    "requires_network": [
        r"\brequests\.",
        r"\bhttpx\.",
        r"\baiohttp\.",
        r"\burllib\.",
        r"\.get\s*\(\s*['\"]https?://",
        r"\.post\s*\(\s*['\"]https?://",
        r"socket\.",
        r"websocket",
    ],
    "slow": [
        r"time\.sleep\s*\(\s*(\d+)",  # sleep > 2 seconds
        r"@pytest\.mark\.timeout\s*\(\s*(\d+)",  # timeout > 30 seconds
    ],
    "integration": [
        r"subprocess\.(run|Popen|call)",
        r"os\.system\s*\(",
        r"shutil\.(rmtree|copytree|move)",
        r"tempfile\.(mkdtemp|NamedTemporaryFile|TemporaryDirectory)",
        r"sqlite3\.connect",
        r"redis\.",
        r"pymongo\.",
    ],
    "requires_api": [
        r"GEMINI_API_KEY",
        r"ANTHROPIC_API_KEY",
        r"OPENAI_API_KEY",
        r"PEXELS_API_KEY",
        r"PIXABAY_API_KEY",
        r"os\.environ\.get\s*\(\s*['\"][A-Z_]+_API_KEY",
    ],
    "fast": [
        # Patterns indicating pure unit tests (used for suggestion when no I/O found)
        r"@pytest\.mark\.fast",
        r"Mock\(",
        r"MagicMock\(",
        r"patch\(",
    ],
}


@dataclass
class TestInfo:
    """Information about a test function."""
    file_path: str
    test_name: str
    markers: list = field(default_factory=list)
    io_patterns: list = field(default_factory=list)
    suggested_marker: str = ""
    suggestion_reason: str = ""
    line_number: int = 0


class TestAnalyzer(ast.NodeVisitor):
    """AST visitor to extract test functions and their markers."""

    def __init__(self, file_path: str, source_code: str):
        self.file_path = file_path
        self.source_code = source_code
        self.source_lines = source_code.split("\n")
        self.tests: list[TestInfo] = []
        self.current_class = None

    def visit_ClassDef(self, node: ast.ClassDef):
        """Track current class for test method names."""
        old_class = self.current_class
        if node.name.startswith("Test"):
            self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract test functions and their markers."""
        if not node.name.startswith("test_"):
            return

        # Get full test name
        if self.current_class:
            test_name = f"{self.current_class}::{node.name}"
        else:
            test_name = node.name

        # Extract markers
        markers = []
        for decorator in node.decorator_list:
            marker = self._extract_marker(decorator)
            if marker:
                markers.append(marker)

        # Get test body as string for I/O pattern analysis
        start_line = node.lineno - 1
        end_line = node.end_lineno if node.end_lineno else start_line + 1
        test_body = "\n".join(self.source_lines[start_line:end_line])

        # Detect I/O patterns
        io_patterns = self._detect_io_patterns(test_body)

        # Suggest marker if unmarked
        suggested, reason = self._suggest_marker(markers, io_patterns, test_body)

        test_info = TestInfo(
            file_path=self.file_path,
            test_name=test_name,
            markers=markers,
            io_patterns=io_patterns,
            suggested_marker=suggested,
            suggestion_reason=reason,
            line_number=node.lineno,
        )
        self.tests.append(test_info)

    def _extract_marker(self, decorator: ast.expr) -> Optional[str]:
        """Extract pytest marker name from decorator."""
        # @pytest.mark.fast
        if isinstance(decorator, ast.Attribute):
            if (
                isinstance(decorator.value, ast.Attribute)
                and isinstance(decorator.value.value, ast.Name)
                and decorator.value.value.id == "pytest"
                and decorator.value.attr == "mark"
            ):
                return decorator.attr

        # @pytest.mark.fast()
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                if (
                    isinstance(decorator.func.value, ast.Attribute)
                    and isinstance(decorator.func.value.value, ast.Name)
                    and decorator.func.value.value.id == "pytest"
                    and decorator.func.value.attr == "mark"
                ):
                    return decorator.func.attr

        return None

    def _detect_io_patterns(self, test_body: str) -> list[str]:
        """Detect I/O patterns in test body."""
        found_patterns = []
        for marker, patterns in IO_PATTERNS.items():
            if marker == "fast":
                continue  # Skip fast patterns for detection
            for pattern in patterns:
                if re.search(pattern, test_body):
                    found_patterns.append(f"{marker}:{pattern[:30]}")
        return found_patterns

    def _suggest_marker(
        self, markers: list, io_patterns: list, test_body: str
    ) -> tuple[str, str]:
        """Suggest a marker based on patterns."""
        # Already has a known marker
        for m in markers:
            if m in KNOWN_MARKERS:
                return "", ""

        # Check for network patterns
        for pattern in io_patterns:
            if pattern.startswith("requires_network"):
                return "requires_network", f"Found network I/O pattern: {pattern.split(':')[1]}"

        # Check for integration patterns (subprocess, tempfile, etc.)
        for pattern in io_patterns:
            if pattern.startswith("integration"):
                return "integration", f"Found I/O pattern: {pattern.split(':')[1]}"

        # Check for API key patterns
        for pattern in io_patterns:
            if pattern.startswith("requires_api"):
                return "requires_api", f"Found API key pattern: {pattern.split(':')[1]}"

        # Check for slow patterns (sleep > 2s)
        sleep_match = re.search(r"time\.sleep\s*\(\s*(\d+(?:\.\d+)?)", test_body)
        if sleep_match:
            sleep_time = float(sleep_match.group(1))
            if sleep_time > 2:
                return "slow", f"Found time.sleep({sleep_time}) > 2 seconds"

        # If uses only mocks and no I/O, suggest fast
        has_mocks = bool(re.search(r"(Mock|MagicMock|patch)\(", test_body))
        has_no_io = not io_patterns

        if has_mocks and has_no_io:
            return "fast", "Uses mocks with no detected I/O patterns"

        # No I/O patterns but not clearly fast
        if has_no_io:
            return "fast", "No I/O patterns detected (likely pure unit test)"

        return "", ""


def analyze_test_file(file_path: Path) -> list[TestInfo]:
    """Analyze a single test file and return test info."""
    try:
        source_code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source_code)
        analyzer = TestAnalyzer(str(file_path), source_code)
        analyzer.visit(tree)
        return analyzer.tests
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
        return []


def analyze_test_directory(test_path: Path) -> list[TestInfo]:
    """Analyze all test files in directory."""
    all_tests = []

    if test_path.is_file():
        return analyze_test_file(test_path)

    # Recursively find all test files
    for test_file in sorted(test_path.rglob("test_*.py")):
        # Skip broken benchmarks
        if "benchmarks_broken" in str(test_file):
            continue
        tests = analyze_test_file(test_file)
        all_tests.extend(tests)

    return all_tests


def generate_marker_report(tests: list[TestInfo]) -> dict:
    """Generate marker distribution report."""
    # Count by marker
    marker_counts = defaultdict(int)
    unmarked_count = 0
    tests_with_suggestions = 0

    for test in tests:
        if test.markers:
            for marker in test.markers:
                if marker in KNOWN_MARKERS:
                    marker_counts[marker] += 1
        else:
            unmarked_count += 1

        if test.suggested_marker:
            tests_with_suggestions += 1

    # Count suggestions by type
    suggestion_counts = defaultdict(int)
    for test in tests:
        if test.suggested_marker:
            suggestion_counts[test.suggested_marker] += 1

    return {
        "total_tests": len(tests),
        "marker_distribution": dict(marker_counts),
        "unmarked_count": unmarked_count,
        "tests_with_suggestions": tests_with_suggestions,
        "suggestion_distribution": dict(suggestion_counts),
    }


def print_report(report: dict, tests: list[TestInfo], verbose: bool = False):
    """Print marker distribution report to stdout."""
    print("\n" + "=" * 60)
    print("TEST MARKER DISTRIBUTION REPORT")
    print("=" * 60)

    print(f"\nTotal tests analyzed: {report['total_tests']}")
    print(f"Unmarked tests: {report['unmarked_count']}")
    print(f"Tests with suggestions: {report['tests_with_suggestions']}")

    print("\n--- Marker Distribution ---")
    for marker, count in sorted(report["marker_distribution"].items(), key=lambda x: -x[1]):
        desc = KNOWN_MARKERS.get(marker, "")
        print(f"  {marker:20} {count:5} tests  ({desc})")

    if report["suggestion_distribution"]:
        print("\n--- Suggested Markers ---")
        for marker, count in sorted(report["suggestion_distribution"].items(), key=lambda x: -x[1]):
            print(f"  {marker:20} {count:5} tests would benefit from this marker")

    if verbose and report["tests_with_suggestions"] > 0:
        print("\n--- Tests Needing Markers (sample) ---")
        shown = 0
        for test in tests:
            if test.suggested_marker and shown < 20:
                rel_path = Path(test.file_path).relative_to(Path.cwd()) if Path(test.file_path).is_absolute() else test.file_path
                print(f"  {rel_path}:{test.line_number}")
                print(f"    Test: {test.test_name}")
                print(f"    Suggested: @pytest.mark.{test.suggested_marker}")
                print(f"    Reason: {test.suggestion_reason}")
                print()
                shown += 1
        if report["tests_with_suggestions"] > 20:
            print(f"  ... and {report['tests_with_suggestions'] - 20} more (use --output for full list)")


def write_csv(tests: list[TestInfo], output_path: Path):
    """Write test analysis to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_file",
            "test_name",
            "line_number",
            "current_markers",
            "suggested_marker",
            "reason",
            "io_patterns",
        ])
        for test in tests:
            # Make path relative for cleaner output
            try:
                rel_path = Path(test.file_path).relative_to(Path.cwd())
            except ValueError:
                rel_path = test.file_path

            writer.writerow([
                str(rel_path),
                test.test_name,
                test.line_number,
                ", ".join(test.markers) if test.markers else "(none)",
                test.suggested_marker or "(none)",
                test.suggestion_reason or "",
                "; ".join(test.io_patterns) if test.io_patterns else "",
            ])

    print(f"\nCSV report written to: {output_path}")


def write_json(report: dict, tests: list[TestInfo], output_path: Path):
    """Write test analysis to JSON file."""
    # Add test details to report
    output = {
        **report,
        "tests_needing_markers": [
            {
                "file": str(Path(t.file_path).relative_to(Path.cwd()) if Path(t.file_path).is_absolute() else t.file_path),
                "test": t.test_name,
                "line": t.line_number,
                "suggested": t.suggested_marker,
                "reason": t.suggestion_reason,
            }
            for t in tests
            if t.suggested_marker
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nJSON report written to: {output_path}")


def print_auto_suggestions(tests: list[TestInfo]):
    """Print auto-suggest output with marker addition recommendations."""
    print("\n" + "=" * 60)
    print("AUTO-SUGGEST: RECOMMENDED MARKER ADDITIONS")
    print("=" * 60)

    # Group by suggested marker
    by_marker = defaultdict(list)
    for test in tests:
        if test.suggested_marker:
            by_marker[test.suggested_marker].append(test)

    for marker, marker_tests in sorted(by_marker.items()):
        print(f"\n### @pytest.mark.{marker} ({len(marker_tests)} tests)")
        print(f"# {KNOWN_MARKERS.get(marker, 'No description')}")
        print()

        # Group by file
        by_file = defaultdict(list)
        for test in marker_tests:
            by_file[test.file_path].append(test)

        for file_path, file_tests in sorted(by_file.items()):
            try:
                rel_path = Path(file_path).relative_to(Path.cwd())
            except ValueError:
                rel_path = file_path
            print(f"# {rel_path}")
            for test in file_tests:
                print(f"#   Line {test.line_number}: {test.test_name}")
                print(f"#   Reason: {test.suggestion_reason}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze test marker distribution and suggest markers",
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
        "--output",
        "-o",
        type=Path,
        help="Output file path (CSV or JSON based on extension)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format (requires --output)",
    )
    parser.add_argument(
        "--auto-suggest",
        action="store_true",
        help="Output recommended marker additions grouped by marker type",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output including sample tests needing markers",
    )

    args = parser.parse_args()

    # Validate args
    if args.json and not args.output:
        parser.error("--json requires --output")

    # Change to project root if running from scripts/
    if not args.path.exists():
        project_root = Path(__file__).parent.parent
        args.path = project_root / args.path
        if not args.path.exists():
            print(f"Error: Test path not found: {args.path}", file=sys.stderr)
            sys.exit(1)

    print(f"Analyzing tests in: {args.path}")
    tests = analyze_test_directory(args.path)

    if not tests:
        print("No tests found!")
        sys.exit(1)

    report = generate_marker_report(tests)

    # Output based on flags
    if args.auto_suggest:
        print_auto_suggestions(tests)
    else:
        print_report(report, tests, verbose=args.verbose)

    # Write output file if specified
    if args.output:
        if args.json or args.output.suffix == ".json":
            write_json(report, tests, args.output)
        else:
            write_csv(tests, args.output)

    # Return non-zero if there are many unmarked tests (CI integration)
    # Only warn if >50% unmarked
    if report["unmarked_count"] > report["total_tests"] * 0.5:
        print(f"\nWarning: {report['unmarked_count']}/{report['total_tests']} tests are unmarked")
        # Don't fail, just warn


if __name__ == "__main__":
    main()
