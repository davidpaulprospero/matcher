#!/usr/bin/env python3
"""
Mutation Testing Runner

Executes mutmut mutation testing on targeted source modules and reports results.
Designed for CI integration with configurable thresholds.

Mutation testing creates "mutants" (small changes to source code) and verifies
that the test suite catches them. A killed mutant = good tests, a surviving
mutant = potential test gap.

Usage:
    # Run mutation testing on default target (src/matching/scoring.py)
    python scripts/run_mutation_tests.py

    # Run on specific module
    python scripts/run_mutation_tests.py --module src/matching/similarity.py

    # Run with custom threshold
    python scripts/run_mutation_tests.py --threshold 0.70

    # Generate HTML report
    python scripts/run_mutation_tests.py --html

    # Run against all matching modules
    python scripts/run_mutation_tests.py --all-matching

    # Show surviving mutants
    python scripts/run_mutation_tests.py --show-survivors

    # CI mode (fail if below threshold)
    python scripts/run_mutation_tests.py --ci --threshold 0.70
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default configuration
DEFAULT_TARGET = "src/matching/scoring.py"
DEFAULT_THRESHOLD = 0.70  # 70% mutation score threshold
DEFAULT_RUNNER = "pytest"

# Module groups for batch testing
MODULE_GROUPS = {
    "matching": [
        "src/matching/scoring.py",
        "src/matching/similarity.py",
        "src/matching/strategies/",
    ],
    "core": [
        "src/matching/scoring.py",
    ],
}

# Output paths
RESULTS_DIR = Path(__file__).parent.parent / "tests" / "mutation_results"
HTML_REPORT_DIR = RESULTS_DIR / "html"


def check_mutmut_installed() -> bool:
    """Check if mutmut is available."""
    try:
        # mutmut 2.5.x uses 'version' subcommand instead of --version flag
        result = subprocess.run(
            [sys.executable, "-m", "mutmut", "version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def run_mutation_tests(
    target: str,
    runner: str = DEFAULT_RUNNER,
    tests_dir: str = "tests/test_matching*.py",
    verbose: bool = False,
) -> dict:
    """Run mutmut on a target module.

    Args:
        target: Path to source file or directory to mutate
        runner: Test runner command (pytest)
        tests_dir: Tests to run for killing mutants
        verbose: Print verbose output

    Returns:
        Dictionary with mutation testing results
    """
    print(f"\n{'='*60}")
    print(f"Running mutation tests on: {target}")
    print(f"{'='*60}")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build mutmut command
    cmd = [
        sys.executable,
        "-m",
        "mutmut",
        "run",
        f"--paths-to-mutate={target}",
        f"--runner={runner} {tests_dir} -x -q --tb=no",
    ]

    if verbose:
        print(f"Command: {' '.join(cmd)}")

    # Run mutation testing
    print("\nStarting mutation testing (this may take several minutes)...")
    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=Path(__file__).parent.parent,
    )

    # Get results using mutmut results command
    results = get_mutation_results(verbose=verbose)

    return results


def get_mutation_results(verbose: bool = False) -> dict:
    """Parse mutmut results from the cache.

    Returns:
        Dictionary with mutation testing statistics
    """
    cwd = Path(__file__).parent.parent

    # Run mutmut results to get summary
    cmd = [sys.executable, "-m", "mutmut", "results"]

    # Use encoding='utf-8' and errors='replace' for Windows compatibility (CLAUDE.md Rule 27)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=cwd,
    )

    # Also get killed count using result-ids command (more reliable for 2.5.x)
    killed_result = subprocess.run(
        [sys.executable, "-m", "mutmut", "result-ids", "killed"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=cwd,
    )
    killed_ids = killed_result.stdout.strip().split() if killed_result.stdout else []
    killed_count = len(killed_ids)

    if verbose:
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    # Parse results from output
    raw_output = result.stdout or ""
    results = {
        "timestamp": datetime.now().isoformat(),
        "killed": killed_count,  # Use count from result-ids command
        "survived": 0,
        "timeout": 0,
        "suspicious": 0,
        "skipped": 0,
        "total": 0,
        "score": 0.0,
        "raw_output": raw_output,
    }

    # Parse the output for statistics
    # mutmut 2.5.x output format: "Survived 🙁 (17)" or "Killed mutants (14)"
    import re
    for line in raw_output.split("\n"):
        line_lower = line.lower()

        # Try mutmut 2.5.x format: "Status (count)" with optional emoji
        # Examples: "Survived 🙁 (17)", "Killed mutants (14)", "Timeout (5)"
        paren_match = re.search(r'\((\d+)\)\s*$', line)

        if "killed" in line_lower:
            try:
                if paren_match:
                    results["killed"] = int(paren_match.group(1))
                else:
                    # Try legacy format: "Killed: 42"
                    parts = line.split(":")
                    if len(parts) >= 2:
                        results["killed"] = int(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "survived" in line_lower:
            try:
                if paren_match:
                    results["survived"] = int(paren_match.group(1))
                else:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        results["survived"] = int(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "timeout" in line_lower:
            try:
                if paren_match:
                    results["timeout"] = int(paren_match.group(1))
                else:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        results["timeout"] = int(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "untested" in line_lower or "skipped" in line_lower:
            try:
                if paren_match:
                    results["skipped"] = int(paren_match.group(1))
            except (ValueError, IndexError):
                pass

    # Calculate total and score
    results["total"] = (
        results["killed"]
        + results["survived"]
        + results["timeout"]
        + results["suspicious"]
    )

    if results["total"] > 0:
        # Killed + timeout are considered "caught"
        caught = results["killed"] + results["timeout"]
        results["score"] = caught / results["total"]

    return results


def show_surviving_mutants(verbose: bool = False) -> list:
    """Get list of surviving mutants for investigation.

    Returns:
        List of surviving mutant descriptions
    """
    cmd = [sys.executable, "-m", "mutmut", "show", "survived"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=Path(__file__).parent.parent,
    )

    survivors = []
    if result.stdout:
        # Each mutant is typically shown as a diff
        print("\n" + "="*60)
        print("SURVIVING MUTANTS (test gaps)")
        print("="*60)
        print(result.stdout)

        # Count mutants from output
        for line in result.stdout.split("\n"):
            if line.startswith("Mutant"):
                survivors.append(line)

    return survivors


def generate_html_report() -> Optional[Path]:
    """Generate HTML report of mutation testing results.

    Returns:
        Path to HTML report or None if failed
    """
    HTML_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = HTML_REPORT_DIR / "index.html"

    cmd = [
        sys.executable,
        "-m",
        "mutmut",
        "html",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=Path(__file__).parent.parent,
    )

    # mutmut creates html/ directory in current dir
    source_html = Path(__file__).parent.parent / "html"
    if source_html.exists():
        # Move to our results directory
        import shutil
        if HTML_REPORT_DIR.exists():
            shutil.rmtree(HTML_REPORT_DIR)
        shutil.move(str(source_html), str(HTML_REPORT_DIR))
        print(f"\nHTML report generated: {HTML_REPORT_DIR / 'index.html'}")
        return output_path

    return None


def save_results(results: dict, target: str) -> Path:
    """Save mutation testing results to JSON.

    Args:
        results: Results dictionary
        target: Target module path

    Returns:
        Path to saved results file
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create filename from target
    target_name = Path(target).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"mutation_{target_name}_{timestamp}.json"

    # Add metadata
    results["target"] = target
    results["saved_at"] = datetime.now().isoformat()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


def print_summary(results: dict, threshold: float):
    """Print formatted summary of mutation testing results."""
    print("\n" + "="*60)
    print("MUTATION TESTING SUMMARY")
    print("="*60)
    print(f"  Total mutants:     {results['total']}")
    print(f"  Killed:            {results['killed']} (good - tests caught the mutant)")
    print(f"  Survived:          {results['survived']} (bad - potential test gap)")
    print(f"  Timeout:           {results['timeout']} (caught via timeout)")
    print(f"  Suspicious:        {results['suspicious']}")
    print()
    print(f"  Mutation Score:    {results['score']*100:.1f}%")
    print(f"  Threshold:         {threshold*100:.1f}%")
    print()

    if results['score'] >= threshold:
        print(f"  Status:            PASSED ✓")
    else:
        print(f"  Status:            FAILED ✗")
        print(f"  Gap:               {(threshold - results['score'])*100:.1f}% below threshold")


def main():
    parser = argparse.ArgumentParser(
        description="Mutation Testing Runner for mutmut",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--module", "-m",
        type=str,
        default=DEFAULT_TARGET,
        help=f"Path to source module to mutate (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Mutation score threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--tests", "-T",
        type=str,
        default="tests/test_matching*.py",
        help="Test pattern for killing mutants (default: tests/test_matching*.py)",
    )
    parser.add_argument(
        "--all-matching",
        action="store_true",
        help="Run mutation testing on all matching modules",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report after testing",
    )
    parser.add_argument(
        "--show-survivors",
        action="store_true",
        help="Show surviving mutants (test gaps)",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Show results from last run without re-running tests",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit with code 1 if below threshold",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Check mutmut is installed
    if not check_mutmut_installed():
        print("ERROR: mutmut is not installed")
        print("Install with: pip install mutmut")
        return 1

    # Handle --results-only mode
    if args.results_only:
        results = get_mutation_results(verbose=args.verbose)
        print_summary(results, args.threshold)
        if args.show_survivors:
            show_surviving_mutants(args.verbose)
        return 0 if results['score'] >= args.threshold else 1

    # Determine targets
    targets = []
    if args.all_matching:
        targets = MODULE_GROUPS["matching"]
    else:
        targets = [args.module]

    # Run mutation testing on each target
    all_results = []
    for target in targets:
        results = run_mutation_tests(
            target=target,
            tests_dir=args.tests,
            verbose=args.verbose,
        )
        all_results.append((target, results))

        if args.save:
            save_results(results, target)

        print_summary(results, args.threshold)

    # Show survivors if requested
    if args.show_survivors:
        show_surviving_mutants(args.verbose)

    # Generate HTML report if requested
    if args.html:
        generate_html_report()

    # CI mode exit code
    if args.ci:
        # Check if any target failed threshold
        for target, results in all_results:
            if results['score'] < args.threshold:
                print(f"\nCI FAILURE: {target} score {results['score']*100:.1f}% < {args.threshold*100:.1f}%")
                return 1
        print("\nCI SUCCESS: All targets meet threshold")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
