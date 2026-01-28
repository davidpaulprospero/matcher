#!/usr/bin/env python
"""
Generate coverage badge for README display (US-010, Sprint 19).

Reads coverage.xml and generates a markdown badge URL for shields.io.

Usage:
    python scripts/generate_coverage_badge.py

Output:
    Prints markdown badge URL to stdout.
    Updates README.md if --update flag is provided.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def get_coverage_percent(coverage_xml: Path = Path("coverage.xml")) -> float:
    """Parse coverage.xml and return overall line coverage percentage."""
    if not coverage_xml.exists():
        print(f"ERROR: {coverage_xml} not found. Run pytest with --cov first.", file=sys.stderr)
        return 0.0

    try:
        tree = ET.parse(coverage_xml)
        root = tree.getroot()

        # Get line-rate from root coverage element
        line_rate = float(root.attrib.get('line-rate', 0))
        return round(line_rate * 100, 1)
    except Exception as e:
        print(f"ERROR: Failed to parse {coverage_xml}: {e}", file=sys.stderr)
        return 0.0


def get_badge_color(percent: float) -> str:
    """Return badge color based on coverage percentage."""
    if percent >= 90:
        return "brightgreen"
    elif percent >= 80:
        return "green"
    elif percent >= 70:
        return "yellow"
    elif percent >= 60:
        return "orange"
    else:
        return "red"


def generate_badge_url(percent: float) -> str:
    """Generate shields.io badge URL."""
    color = get_badge_color(percent)
    return f"https://img.shields.io/badge/coverage-{percent}%25-{color}"


def generate_markdown(percent: float) -> str:
    """Generate markdown badge link."""
    url = generate_badge_url(percent)
    return f"![Coverage](url)"


def main():
    """Main entry point."""
    percent = get_coverage_percent()

    if percent == 0.0:
        print("Coverage: 0% (run pytest --cov=src first)")
        sys.exit(1)

    badge_url = generate_badge_url(percent)
    markdown = generate_markdown(percent)

    print(f"Coverage: {percent}%")
    print(f"Badge URL: {badge_url}")
    print(f"Markdown: {markdown}")

    # Per-module breakdown (if verbose)
    if "--verbose" in sys.argv:
        print("\nPer-module coverage:")
        print("  Run: pytest --cov=src/matching --cov=src/agents --cov=src/compilation --cov-report=term-missing")


if __name__ == "__main__":
    main()
