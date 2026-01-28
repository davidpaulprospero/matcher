#!/usr/bin/env python
"""
Per-module coverage breakdown script (US-010, Sprint 24).

Generates per-module coverage reports with support for:
- Critical module highlighting (matching, agents, compilation, stages)
- Warning for modules below 85% target
- JSON output for CI integration
- Baseline comparison for regression detection

Usage:
    python scripts/coverage_by_module.py                    # Basic report
    python scripts/coverage_by_module.py --json             # JSON output
    python scripts/coverage_by_module.py --output report.json  # Save to file
    python scripts/coverage_by_module.py --compare-baseline baseline.json
    python scripts/coverage_by_module.py --save-baseline    # Save current as baseline

Requires: coverage.xml (run pytest --cov=src first)
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# Critical modules that must maintain high coverage
# Package names as they appear in coverage.xml (without src. prefix)
CRITICAL_MODULES = {
    "matching": "Video-to-voiceover matching",
    "agents": "Self-healing pipeline agents",
    "compilation": "Keyword compilation pipeline",
    "stages": "Pipeline stage implementations",
}

# Target coverage threshold
TARGET_COVERAGE = 85.0

# Warning thresholds
WARNING_THRESHOLDS = {
    "critical": 80.0,  # Critical modules must be above this
    "standard": 70.0,  # Standard modules warning threshold
}


@dataclass
class ModuleCoverage:
    """Coverage data for a single module."""
    name: str
    lines_covered: int
    lines_total: int
    coverage_percent: float
    is_critical: bool
    below_target: bool
    description: str = ""


@dataclass
class CoverageReport:
    """Full coverage report."""
    timestamp: str
    overall_coverage: float
    target: float
    modules: list
    critical_modules: list
    warnings: list
    summary: dict


def parse_coverage_xml(coverage_file: Path) -> tuple[float, dict[str, ModuleCoverage]]:
    """Parse coverage.xml and extract per-module coverage data."""
    if not coverage_file.exists():
        print(f"ERROR: {coverage_file} not found. Run pytest --cov=src first.", file=sys.stderr)
        sys.exit(1)

    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
    except Exception as e:
        print(f"ERROR: Failed to parse {coverage_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # Get overall coverage
    overall_rate = float(root.attrib.get('line-rate', 0))
    overall_percent = round(overall_rate * 100, 2)

    # Parse per-package coverage
    modules: dict[str, ModuleCoverage] = {}

    for package in root.findall('.//package'):
        pkg_name = package.attrib.get('name', 'unknown')

        # Aggregate lines from all classes in package
        lines_covered = 0
        lines_total = 0

        for cls in package.findall('.//class'):
            for line in cls.findall('.//line'):
                lines_total += 1
                if int(line.attrib.get('hits', 0)) > 0:
                    lines_covered += 1

        if lines_total == 0:
            continue

        coverage_percent = round((lines_covered / lines_total) * 100, 2)

        # Determine if this is a critical module
        is_critical = False
        description = ""
        for critical_path, desc in CRITICAL_MODULES.items():
            # Check if package name starts with critical module name
            if pkg_name == critical_path or pkg_name.startswith(f"{critical_path}."):
                is_critical = True
                description = desc
                break

        modules[pkg_name] = ModuleCoverage(
            name=pkg_name,
            lines_covered=lines_covered,
            lines_total=lines_total,
            coverage_percent=coverage_percent,
            is_critical=is_critical,
            below_target=coverage_percent < TARGET_COVERAGE,
            description=description,
        )

    return overall_percent, modules


def aggregate_by_top_level(modules: dict[str, ModuleCoverage]) -> dict[str, ModuleCoverage]:
    """Aggregate coverage by top-level module (e.g., src.matching, src.agents)."""
    aggregated: dict[str, dict] = {}

    for name, mod in modules.items():
        # Get top-level module (e.g., src.matching from src.matching.core)
        parts = name.split(".")
        if len(parts) >= 2:
            top_level = ".".join(parts[:2])
        else:
            top_level = name

        if top_level not in aggregated:
            aggregated[top_level] = {
                "lines_covered": 0,
                "lines_total": 0,
                "is_critical": False,
                "description": "",
            }

        aggregated[top_level]["lines_covered"] += mod.lines_covered
        aggregated[top_level]["lines_total"] += mod.lines_total

        # Mark as critical if any sub-module is critical
        if mod.is_critical:
            aggregated[top_level]["is_critical"] = True
            aggregated[top_level]["description"] = mod.description

    # Convert to ModuleCoverage objects
    result: dict[str, ModuleCoverage] = {}
    for name, data in aggregated.items():
        if data["lines_total"] == 0:
            continue
        coverage_percent = round((data["lines_covered"] / data["lines_total"]) * 100, 2)
        result[name] = ModuleCoverage(
            name=name,
            lines_covered=data["lines_covered"],
            lines_total=data["lines_total"],
            coverage_percent=coverage_percent,
            is_critical=data["is_critical"],
            below_target=coverage_percent < TARGET_COVERAGE,
            description=data["description"],
        )

    return result


def generate_warnings(modules: dict[str, ModuleCoverage]) -> list[str]:
    """Generate warning messages for modules below target."""
    warnings = []

    for name, mod in sorted(modules.items(), key=lambda x: x[1].coverage_percent):
        if mod.is_critical and mod.coverage_percent < WARNING_THRESHOLDS["critical"]:
            warnings.append(
                f"CRITICAL: {name} at {mod.coverage_percent}% (target: {TARGET_COVERAGE}%, "
                f"minimum: {WARNING_THRESHOLDS['critical']}%)"
            )
        elif mod.below_target:
            warnings.append(f"WARNING: {name} at {mod.coverage_percent}% (target: {TARGET_COVERAGE}%)")

    return warnings


def compare_to_baseline(
    current: dict[str, ModuleCoverage],
    baseline_file: Path,
    threshold: float = 0.0
) -> tuple[list[str], list[str]]:
    """Compare current coverage to baseline, return regressions and improvements."""
    if not baseline_file.exists():
        return [f"Baseline file not found: {baseline_file}"], []

    try:
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
    except Exception as e:
        return [f"Failed to load baseline: {e}"], []

    baseline_modules = {m["name"]: m["coverage_percent"] for m in baseline_data.get("modules", [])}

    regressions = []
    improvements = []

    for name, mod in current.items():
        if name in baseline_modules:
            baseline_pct = baseline_modules[name]
            diff = mod.coverage_percent - baseline_pct

            if diff < -threshold:
                regressions.append(
                    f"REGRESSION: {name}: {baseline_pct}% -> {mod.coverage_percent}% ({diff:+.2f}%)"
                )
            elif diff > threshold:
                improvements.append(
                    f"IMPROVED: {name}: {baseline_pct}% -> {mod.coverage_percent}% ({diff:+.2f}%)"
                )

    return regressions, improvements


def print_text_report(
    overall: float,
    modules: dict[str, ModuleCoverage],
    warnings: list[str],
    regressions: list[str] = None,
    improvements: list[str] = None,
    verbose: bool = False,
):
    """Print human-readable coverage report."""
    print("=" * 70)
    print("COVERAGE REPORT BY MODULE")
    print("=" * 70)
    print(f"\nOverall Coverage: {overall}% (target: {TARGET_COVERAGE}%)")

    if overall >= TARGET_COVERAGE:
        print("[OK] Coverage target met")
    else:
        print(f"[WARN] Below target by {TARGET_COVERAGE - overall:.2f}%")

    # Critical modules section
    print("\n" + "-" * 70)
    print("CRITICAL MODULES")
    print("-" * 70)
    print(f"{'Module':<35} {'Coverage':>10} {'Lines':>15} {'Status':>10}")
    print("-" * 70)

    critical_mods = {k: v for k, v in modules.items() if v.is_critical}
    for name, mod in sorted(critical_mods.items(), key=lambda x: x[1].coverage_percent, reverse=True):
        status = "[OK]" if not mod.below_target else "[WARN]"
        lines = f"{mod.lines_covered}/{mod.lines_total}"
        print(f"{name:<35} {mod.coverage_percent:>9.2f}% {lines:>15} {status:>10}")

    # All modules section (if verbose)
    if verbose:
        print("\n" + "-" * 70)
        print("ALL MODULES")
        print("-" * 70)
        print(f"{'Module':<35} {'Coverage':>10} {'Lines':>15} {'Critical':>10}")
        print("-" * 70)

        for name, mod in sorted(modules.items(), key=lambda x: x[1].coverage_percent, reverse=True):
            critical = "Yes" if mod.is_critical else ""
            lines = f"{mod.lines_covered}/{mod.lines_total}"
            print(f"{name:<35} {mod.coverage_percent:>9.2f}% {lines:>15} {critical:>10}")

    # Warnings section
    if warnings:
        print("\n" + "-" * 70)
        print("WARNINGS")
        print("-" * 70)
        for warn in warnings:
            print(f"  {warn}")

    # Baseline comparison
    if regressions:
        print("\n" + "-" * 70)
        print("REGRESSIONS (vs baseline)")
        print("-" * 70)
        for reg in regressions:
            print(f"  {reg}")

    if improvements:
        print("\n" + "-" * 70)
        print("IMPROVEMENTS (vs baseline)")
        print("-" * 70)
        for imp in improvements:
            print(f"  {imp}")

    print("\n" + "=" * 70)


def generate_json_report(
    overall: float,
    modules: dict[str, ModuleCoverage],
    warnings: list[str],
    regressions: list[str] = None,
    improvements: list[str] = None,
) -> dict:
    """Generate JSON-formatted coverage report."""
    # Separate critical and non-critical
    critical_mods = [asdict(m) for m in modules.values() if m.is_critical]
    all_mods = [asdict(m) for m in modules.values()]

    # Sort by coverage
    critical_mods.sort(key=lambda x: x["coverage_percent"], reverse=True)
    all_mods.sort(key=lambda x: x["coverage_percent"], reverse=True)

    # Summary statistics
    below_target_count = sum(1 for m in modules.values() if m.below_target)
    critical_below_target = sum(1 for m in modules.values() if m.is_critical and m.below_target)

    report = CoverageReport(
        timestamp=datetime.now().isoformat(),
        overall_coverage=overall,
        target=TARGET_COVERAGE,
        modules=all_mods,
        critical_modules=critical_mods,
        warnings=warnings,
        summary={
            "total_modules": len(modules),
            "below_target_count": below_target_count,
            "critical_modules_count": len(critical_mods),
            "critical_below_target": critical_below_target,
            "meets_target": overall >= TARGET_COVERAGE,
        },
    )

    result = asdict(report)

    if regressions:
        result["regressions"] = regressions
    if improvements:
        result["improvements"] = improvements

    return result


def save_baseline(modules: dict[str, ModuleCoverage], overall: float, output_path: Path):
    """Save current coverage as baseline for future comparison."""
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "overall_coverage": overall,
        "target": TARGET_COVERAGE,
        "modules": [asdict(m) for m in modules.values()],
    }

    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"Baseline saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-module coverage breakdown report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/coverage_by_module.py                        # Basic report
  python scripts/coverage_by_module.py --verbose              # Show all modules
  python scripts/coverage_by_module.py --json                 # JSON to stdout
  python scripts/coverage_by_module.py --output report.json   # JSON to file
  python scripts/coverage_by_module.py --compare-baseline baseline.json
  python scripts/coverage_by_module.py --save-baseline        # Save current as baseline
        """
    )

    parser.add_argument(
        "--coverage-file", "-c",
        type=Path,
        default=Path("coverage.xml"),
        help="Path to coverage.xml file (default: coverage.xml)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output JSON format"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Write output to file (for --json or --save-baseline)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all modules, not just critical ones"
    )
    parser.add_argument(
        "--compare-baseline", "-b",
        type=Path,
        metavar="BASELINE",
        help="Compare to baseline JSON file"
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.0,
        help="Minimum change to report as regression/improvement (default: 0.0)"
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current coverage as baseline (use --output for path)"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if regressions detected"
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit with code 1 if overall coverage is below this threshold"
    )

    args = parser.parse_args()

    # Parse coverage data
    overall, raw_modules = parse_coverage_xml(args.coverage_file)

    # Aggregate by top-level module
    modules = aggregate_by_top_level(raw_modules)

    # Generate warnings
    warnings = generate_warnings(modules)

    # Compare to baseline if provided
    regressions = []
    improvements = []
    if args.compare_baseline:
        regressions, improvements = compare_to_baseline(
            modules, args.compare_baseline, args.regression_threshold
        )

    # Save baseline if requested
    if args.save_baseline:
        output_path = args.output or Path("coverage_baseline.json")
        save_baseline(modules, overall, output_path)
        if not args.json:
            return

    # Output report
    if args.json:
        report = generate_json_report(overall, modules, warnings, regressions, improvements)
        json_output = json.dumps(report, indent=2)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"Report saved to: {args.output}")
        else:
            print(json_output)
    else:
        print_text_report(overall, modules, warnings, regressions, improvements, args.verbose)

    # Exit codes
    exit_code = 0

    if args.fail_on_regression and regressions:
        print(f"\nFailed: {len(regressions)} regression(s) detected", file=sys.stderr)
        exit_code = 1

    if args.fail_under is not None and overall < args.fail_under:
        print(f"\nFailed: Coverage {overall}% below threshold {args.fail_under}%", file=sys.stderr)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
