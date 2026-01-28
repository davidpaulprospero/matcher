#!/usr/bin/env python3
"""
CI Benchmark Runner

Executes pytest benchmarks and compares results against baselines.
Designed for CI integration with configurable regression thresholds.

Usage:
    # Run all benchmarks and save results
    python scripts/benchmark_runner.py --output results.json

    # Compare results to baseline with 20% threshold
    python scripts/benchmark_runner.py --compare baseline.json --threshold 0.20

    # Run specific suites
    python scripts/benchmark_runner.py --suite pipeline embedding

    # Generate new baseline
    python scripts/benchmark_runner.py --output baseline.json --save-baseline
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Benchmark suites and their test files
BENCHMARK_SUITES = {
    "pipeline": "test_pipeline_performance.py",
    "keyword": "test_keyword_performance_simple.py",
    "embedding": "test_embedding_performance.py",
    "transcription": "test_transcription_performance.py",
}

# Default paths
BENCHMARK_DIR = Path(__file__).parent.parent / "tests" / "benchmarks"
DEFAULT_BASELINE = BENCHMARK_DIR / "baseline.json"


def run_benchmarks(
    suites: Optional[list] = None,
    output_path: Optional[Path] = None,
    min_rounds: int = 5,
    warmup: bool = True,
    verbose: bool = False,
) -> dict:
    """Run pytest benchmarks and return results.

    Args:
        suites: List of suite names to run (None = all)
        output_path: Path to save JSON results
        min_rounds: Minimum benchmark rounds
        warmup: Enable warmup
        verbose: Print verbose output

    Returns:
        Dictionary with benchmark results
    """
    # Build test file list
    if suites:
        test_files = [BENCHMARK_DIR / BENCHMARK_SUITES[s] for s in suites if s in BENCHMARK_SUITES]
    else:
        test_files = list(BENCHMARK_DIR.glob("test_*.py"))

    if not test_files:
        print("No benchmark files found")
        return {}

    # Build pytest command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--benchmark-only",
        f"--benchmark-min-rounds={min_rounds}",
        "--benchmark-json=.benchmark_temp.json",
    ]

    if warmup:
        cmd.append("--benchmark-warmup=on")

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    cmd.extend(str(f) for f in test_files)

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    # Run benchmarks
    result = subprocess.run(cmd, capture_output=not verbose, text=True)

    if result.returncode != 0 and not verbose:
        print(f"Benchmark run failed (exit code {result.returncode})")
        print(result.stderr)
        return {}

    # Parse results
    temp_json = Path(".benchmark_temp.json")
    if not temp_json.exists():
        print("No benchmark results generated")
        return {}

    with open(temp_json) as f:
        raw_results = json.load(f)

    # Clean up temp file
    temp_json.unlink()

    # Convert to our format
    results = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "runner": "local",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "benchmarks": {},
    }

    for benchmark in raw_results.get("benchmarks", []):
        name = benchmark["name"]
        stats = benchmark["stats"]
        results["benchmarks"][name] = {
            "mean": stats["mean"],
            "stddev": stats["stddev"],
            "min": stats["min"],
            "max": stats["max"],
            "rounds": stats["rounds"],
        }

    # Save if output path specified
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


def compare_benchmarks(
    current: dict,
    baseline: dict,
    threshold: float = 0.20,
    verbose: bool = False,
) -> tuple[bool, list]:
    """Compare current benchmarks against baseline.

    Args:
        current: Current benchmark results
        baseline: Baseline benchmark results
        threshold: Regression threshold (0.20 = 20%)
        verbose: Print detailed output

    Returns:
        Tuple of (passed: bool, regressions: list)
    """
    regressions = []
    improvements = []
    ok_tests = []
    new_tests = []

    current_benchmarks = current.get("benchmarks", {})
    baseline_benchmarks = baseline.get("benchmarks", {})

    print("\nBenchmark Comparison Results")
    print("=" * 40)

    for name, current_stats in current_benchmarks.items():
        if name not in baseline_benchmarks:
            new_tests.append(name)
            if verbose:
                print(f"[NEW] {name}: NEW (no baseline)")
            continue

        baseline_stats = baseline_benchmarks[name]
        baseline_mean = baseline_stats["mean"]
        current_mean = current_stats["mean"]

        if baseline_mean == 0:
            change_pct = 0
        else:
            change_pct = (current_mean - baseline_mean) / baseline_mean

        status_char = "[OK]"
        status = "OK"

        if change_pct > threshold:
            status_char = "[WARN]"
            status = "REGRESSION"
            regressions.append({
                "name": name,
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "change_pct": change_pct,
            })
        elif change_pct < -threshold:
            status_char = "[OK]"
            status = "IMPROVED"
            improvements.append(name)
        else:
            ok_tests.append(name)

        sign = "+" if change_pct >= 0 else ""
        print(f"{status_char} {name}: {baseline_mean:.4f}s -> {current_mean:.4f}s ({sign}{change_pct*100:.1f}%) [{status}]")

    # Print summary
    print()
    print(f"Summary: {len(ok_tests)} OK, {len(regressions)} REGRESSION, {len(improvements)} IMPROVED, {len(new_tests)} NEW")
    print(f"Threshold: {threshold*100:.0f}%")

    if regressions:
        print()
        print("Regressions detected:")
        for reg in regressions:
            print(f"  - {reg['name']}: +{reg['change_pct']*100:.1f}%")
        return False, regressions

    return True, []


def load_baseline(path: Path) -> dict:
    """Load baseline from JSON file."""
    if not path.exists():
        print(f"Baseline not found: {path}")
        return {}

    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="CI Benchmark Runner for pytest-benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Path to save benchmark results JSON",
    )
    parser.add_argument(
        "--compare", "-c",
        type=Path,
        help="Path to baseline JSON for comparison",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.20,
        help="Regression threshold (default: 0.20 = 20%%)",
    )
    parser.add_argument(
        "--suite", "-s",
        nargs="+",
        choices=list(BENCHMARK_SUITES.keys()),
        help="Specific benchmark suites to run",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Mark output as a new baseline (adds metadata)",
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=5,
        help="Minimum benchmark rounds (default: 5)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable warmup",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List available benchmark suites and exit",
    )

    args = parser.parse_args()

    if args.list_suites:
        print("Available benchmark suites:")
        for name, file in BENCHMARK_SUITES.items():
            print(f"  {name}: {file}")
        return 0

    # Run benchmarks
    results = run_benchmarks(
        suites=args.suite,
        output_path=args.output,
        min_rounds=args.min_rounds,
        warmup=not args.no_warmup,
        verbose=args.verbose,
    )

    if not results.get("benchmarks"):
        print("No benchmark results to process")
        return 1

    # Add baseline marker if saving as baseline
    if args.save_baseline:
        results["is_baseline"] = True
        results["created_by"] = "benchmark_runner.py"
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Baseline saved to {args.output}")

    # Compare against baseline if specified
    if args.compare:
        baseline = load_baseline(args.compare)
        if not baseline:
            return 1

        passed, regressions = compare_benchmarks(
            results,
            baseline,
            threshold=args.threshold,
            verbose=args.verbose,
        )

        if not passed:
            print(f"\nExit code: 1 (regressions detected)")
            return 1

        print(f"\nExit code: 0 (all benchmarks within threshold)")
        return 0

    # Just print summary if no comparison
    print(f"\nBenchmarks run: {len(results.get('benchmarks', {}))}")
    for name, stats in results.get("benchmarks", {}).items():
        print(f"  {name}: {stats['mean']:.4f}s (±{stats['stddev']:.4f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
