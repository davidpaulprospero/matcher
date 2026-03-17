#!/usr/bin/env python3
"""
Pipeline Benchmark Script

Runs standardized pipeline benchmarks to measure performance across different
configurations and track performance trends over time.

Usage:
    # Run benchmark on a test project
    python scripts/benchmark.py run --project "E:/Projects/TestProject" --output results.json

    # Compare two benchmark runs
    python scripts/benchmark.py compare --baseline results_old.json --current results_new.json

    # Generate performance trend report
    python scripts/benchmark.py trends --project "E:/Projects/TestProject"

    # Run with config comparison
    python scripts/benchmark.py run --project "E:/Projects/TestProject" --config-a base.yaml --config-b improved.yaml --compare-configs

Features:
    - Measures timing per pipeline stage (ANALYZE, VIDEO_SEARCH, CAPTION, MATCH, etc.)
    - Stores results in a benchmark database (SQLite by default)
    - Compares results between config versions
    - Generates performance trend reports over time
    - Supports --save-results and --compare flags per acceptance criteria
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Import standardized output functions
from script_utils import print_ok, print_warn, print_error, print_info, print_header
from script_utils import progress_bar, track_progress, set_verbosity, get_verbosity
from utils.cli_helpers import json_output

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Change to project root so relative paths work correctly
os.chdir(project_root)

# Default paths
DEFAULT_BENCHMARK_DB = project_root / ".benchmarks" / "benchmark.db"
BENCHMARK_DIR = project_root / ".benchmarks"


def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize benchmark database with schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Benchmark runs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            project_name TEXT,
            config_path TEXT,
            timestamp TEXT NOT NULL,
            total_duration REAL,
            pipeline_version TEXT,
            python_version TEXT,
            notes TEXT,
            tags TEXT
        )
    """)

    # Stage timings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_timings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            duration REAL NOT NULL,
            items_processed INTEGER,
            throughput REAL,
            FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
        )
    """)

    # Config snapshots table (for config comparison)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS config_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            config_key TEXT NOT NULL,
            config_value TEXT,
            FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
        )
    """)

    conn.commit()
    return conn


def get_run_id() -> str:
    """Generate unique run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_pipeline_benchmark(
    project_path: str,
    config_path: Optional[str] = None,
    stages: Optional[list[str]] = None,
    fresh: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run the pipeline and capture per-stage timing.

    Args:
        project_path: Path to the project directory
        config_path: Optional path to config file
        stages: Optional list of stages to run (None = all)
        fresh: Whether to start fresh (--fresh flag)
        verbose: Print verbose output

    Returns:
        Dictionary with benchmark results including stage timings
    """
    # Build command
    cmd = [
        sys.executable,
        str(project_root / "main.py"),
        "--project", project_path,
    ]

    if config_path:
        cmd.extend(["--config", config_path])

    if fresh:
        cmd.append("--fresh")

    if stages:
        # Note: Pipeline doesn't support running specific stages directly,
        # but we can note which stages we want to focus analysis on
        pass

    if verbose:
        print_info(f"Running: {' '.join(cmd)}")

    # Run pipeline and time it
    start_time = time.time()

    # Capture stage timings from checkpoint if available
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )

    total_duration = time.time() - start_time

    # Parse stage timings from output
    stage_timings = {}

    # Look for stage timing info in stdout/stderr
    output = result.stdout + result.stderr

    # Try to find timing information in logs
    # The pipeline outputs timing summary at the end
    for line in output.split('\n'):
        if '[TIMING]' in line or 'Stage timing' in line.lower():
            # Try to parse stage timings from log line
            # Format: "StageName: X.XXs" or similar
            import re
            matches = re.findall(r'(\w+):\s*([\d.]+)s?', line)
            for stage, duration in matches:
                try:
                    stage_timings[stage.upper()] = float(duration)
                except ValueError:
                    pass

    # If no timings found in logs, try to load from checkpoint
    if not stage_timings:
        checkpoint_path = Path(project_path) / "checkpoint.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    cp = json.load(f)
                    stage_metrics = cp.get('stage_metrics', {})
                    stage_timings = {
                        name: data.get('elapsed', data.get('duration', 0))
                        for name, data in stage_metrics.items()
                    }
            except Exception as e:
                if verbose:
                    print_warn(f"Could not parse checkpoint: {e}")

    # If still no timings, estimate from total duration
    if not stage_timings:
        # This is a fallback - ideally we'd get real timings
        stage_timings['ESTIMATED_TOTAL'] = total_duration

    return {
        "run_id": get_run_id(),
        "project_path": project_path,
        "config_path": config_path,
        "timestamp": datetime.now().isoformat(),
        "total_duration": total_duration,
        "stage_timings": stage_timings,
        "return_code": result.returncode,
        "output_preview": output[-2000:] if output else "",
    }


def save_benchmark_result(
    db_path: Path,
    result: dict[str, Any],
    notes: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
    """Save benchmark result to database."""
    conn = init_database(db_path)
    cursor = conn.cursor()

    run_id = result["run_id"]

    # Insert benchmark run
    cursor.execute("""
        INSERT INTO benchmark_runs (run_id, project_name, config_path, timestamp,
                                    total_duration, notes, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        Path(result["project_path"]).name,
        result.get("config_path"),
        result["timestamp"],
        result["total_duration"],
        notes,
        json.dumps(tags) if tags else None,
    ))

    # Insert stage timings
    for stage_name, duration in result.get("stage_timings", {}).items():
        cursor.execute("""
            INSERT INTO stage_timings (run_id, stage_name, duration)
            VALUES (?, ?, ?)
        """, (run_id, stage_name, duration))

    conn.commit()
    conn.close()

    return run_id


def compare_benchmarks(
    baseline_result: dict[str, Any],
    current_result: dict[str, Any],
    verbose: bool = False,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Compare two benchmark results.

    Args:
        baseline_result: Baseline benchmark results
        current_result: Current benchmark results
        verbose: Print detailed output

    Returns:
        Tuple of (passed: bool, regressions: list)
    """
    regressions: list[dict[str, Any]] = []

    baseline_timings = baseline_result.get("stage_timings", {})
    current_timings = current_result.get("stage_timings", {})

    print_header("Benchmark Comparison Results")

    # Compare total duration
    baseline_total = baseline_result.get("total_duration", 0)
    current_total = current_result.get("total_duration", 0)

    if baseline_total > 0:
        total_change = (current_total - baseline_total) / baseline_total
    else:
        total_change = 0

    total_status = "OK" if total_change <= 0.1 else "WARN"
    if total_change <= 0.1:
        print_ok(f"Total Duration: {baseline_total:.2f}s -> {current_total:.2f}s "
                f"({'+' if total_change >= 0 else ''}{total_change*100:.1f}%)")
    else:
        print_warn(f"Total Duration: {baseline_total:.2f}s -> {current_total:.2f}s "
                  f"({'+' if total_change >= 0 else ''}{total_change*100:.1f}%)")

    # Compare each stage
    all_stages = set(baseline_timings.keys()) | set(current_timings.keys())

    # Use progress bar for iterating over stages
    verbose = get_verbosity() >= 1
    for stage in progress_bar(sorted(all_stages), desc="Comparing stages", disable=not verbose):
        baseline_dur = baseline_timings.get(stage, 0)
        current_dur = current_timings.get(stage, 0)

        if baseline_dur > 0:
            change = (current_dur - baseline_dur) / baseline_dur
        elif current_dur > 0:
            change = 1.0  # 100% increase if baseline was 0
        else:
            change = 0

        if change > 0.2:  # 20% regression threshold
            status = "REGRESSION"
            regressions.append({
                "stage": stage,
                "baseline": baseline_dur,
                "current": current_dur,
                "change_pct": change * 100,
            })
            print_error(f"{stage}: {baseline_dur:.2f}s -> {current_dur:.2f}s "
                       f"({'+' if change >= 0 else ''}{change*100:.1f}%)")
        elif change < -0.1:
            print_ok(f"{stage}: {baseline_dur:.2f}s -> {current_dur:.2f}s "
                    f"({'+' if change >= 0 else ''}{change*100:.1f}%) (IMPROVED)")
        else:
            print_ok(f"{stage}: {baseline_dur:.2f}s -> {current_dur:.2f}s "
                    f"({'+' if change >= 0 else ''}{change*100:.1f}%)")

    # Summary
    print()
    if regressions:
        print_error(f"Summary: {len(regressions)} REGRESSION, {len(all_stages) - len(regressions)} OK")
        print_warn("Regressions detected:")
        for reg in regressions:
            print_info(f"  - {reg['stage']}: +{reg['change_pct']:.1f}%")
        return False, regressions

    print_ok("All stages OK (within 20% threshold)")
    return True, []


def generate_trend_report(
    db_path: Path,
    project_path: Optional[str] = None,
    stage_name: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Generate performance trend report.

    Args:
        db_path: Path to benchmark database
        project_path: Optional project path to filter by
        stage_name: Optional stage to focus on
        output_path: Optional path to save report

    Returns:
        Dictionary with trend data
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Query benchmark runs
    if project_path:
        cursor.execute("""
            SELECT run_id, project_name, config_path, timestamp, total_duration
            FROM benchmark_runs
            WHERE project_name = ?
            ORDER BY timestamp DESC
            LIMIT 20
        """, (Path(project_path).name,))
    else:
        cursor.execute("""
            SELECT run_id, project_name, config_path, timestamp, total_duration
            FROM benchmark_runs
            ORDER BY timestamp DESC
            LIMIT 20
        """)

    runs = [dict(row) for row in cursor.fetchall()]

    # Query stage timings for trends
    trends: dict[str, list[dict[str, Any]]] = {}

    # Use progress bar for iterating over runs
    verbose = get_verbosity() >= 1
    for run in progress_bar(runs, desc="Processing runs", disable=not verbose):
        cursor.execute("""
            SELECT stage_name, duration
            FROM stage_timings
            WHERE run_id = ?
        """, (run["run_id"],))

        for row in cursor.fetchall():
            stage = row["stage_name"]
            if stage_name and stage != stage_name:
                continue

            if stage not in trends:
                trends[stage] = []

            trends[stage].append({
                "timestamp": run["timestamp"],
                "duration": row["duration"],
                "run_id": run["run_id"],
            })

    conn.close()

    # Calculate trend statistics
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "project_filter": project_path,
        "stage_filter": stage_name,
        "total_runs": len(runs),
        "trends": {},
    }

    for stage, data in trends.items():
        if len(data) < 2:
            continue

        durations = [d["duration"] for d in data]
        avg_duration = sum(durations) / len(durations)

        # Calculate trend (positive = getting slower)
        first_half = durations[:len(durations)//2]
        second_half = durations[len(durations)//2:]

        if first_half and second_half:
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            if first_avg > 0:
                trend_pct = ((second_avg - first_avg) / first_avg) * 100
            else:
                trend_pct = 0
        else:
            trend_pct = 0

        report["trends"][stage] = {
            "runs": len(data),
            "avg_duration": round(avg_duration, 2),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "trend_pct": round(trend_pct, 1),
            "trend_direction": "slower" if trend_pct > 5 else ("faster" if trend_pct < -5 else "stable"),
            "data": data[:10],  # Last 10 runs
        }

    # Print summary
    print_header("Performance Trend Report")
    print_info(f"Total benchmark runs: {len(runs)}")

    if project_path:
        print_info(f"Project: {project_path}")
    if stage_name:
        print_info(f"Stage: {stage_name}")

    print()

    trends_dict: dict[str, Any] = report["trends"]
    for stage, data in sorted(trends_dict.items()):
        trend_icon = "↓" if data["trend_direction"] == "faster" else ("↑" if data["trend_direction"] == "slower" else "→")
        print_info(f"{trend_icon} {stage}:")
        print_info(f"    Avg: {data['avg_duration']:.2f}s | Min: {data['min_duration']:.2f}s | Max: {data['max_duration']:.2f}s")
        print_info(f"    Trend: {data['trend_pct']:+.1f}% ({data['trend_direction']})")
        print_info(f"    Runs tracked: {data['runs']}")
        print()

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print_ok(f"Report saved to: {output_path}")

    return report


def run_config_comparison(
    project_path: str,
    config_a: str,
    config_b: str,
    db_path: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run pipeline with two different configs and compare results."""

    print_info(f"Running benchmark with config A: {config_a}")
    result_a = run_pipeline_benchmark(project_path, config_a, verbose=verbose)
    save_benchmark_result(db_path, result_a, notes=f"Config A: {config_a}", tags=["config_a"])

    print_info(f"Running benchmark with config B: {config_b}")
    result_b = run_pipeline_benchmark(project_path, config_b, verbose=verbose)
    save_benchmark_result(db_path, result_b, notes=f"Config B: {config_b}", tags=["config_b"])

    # Compare results
    passed, regressions = compare_benchmarks(result_a, result_b, verbose=verbose)

    return {
        "config_a": config_a,
        "config_b": config_b,
        "result_a": result_a,
        "result_b": result_b,
        "passed": passed,
        "regressions": regressions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Benchmark Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline benchmark")
    run_parser.add_argument("--project", "-p", required=True, help="Project path")
    run_parser.add_argument("--config", "-c", help="Config file path")
    run_parser.add_argument("--output", "-o", type=Path, help="Output JSON path")
    run_parser.add_argument("--save-results", action="store_true",
                           help="Save results to benchmark database")
    run_parser.add_argument("--db", type=Path, default=DEFAULT_BENCHMARK_DB,
                           help="Benchmark database path")
    run_parser.add_argument("--fresh", action="store_true", default=True,
                           help="Start fresh (default: True)")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")
    run_parser.add_argument("--notes", help="Notes for this run")
    run_parser.add_argument("--tags", nargs="*", help="Tags for this run")
    run_parser.add_argument("--json", "-j", action="store_true",
                           help="Output results as JSON")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark runs")
    compare_parser.add_argument("--baseline", "-b", required=True, type=Path,
                               help="Baseline benchmark JSON")
    compare_parser.add_argument("--current", "-c", required=True, type=Path,
                               help="Current benchmark JSON")
    compare_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    compare_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")
    compare_parser.add_argument("--json", "-j", action="store_true",
                               help="Output results as JSON")

    # Trends command
    trends_parser = subparsers.add_parser("trends", help="Generate performance trends")
    trends_parser.add_argument("--project", "-p", help="Project path to filter by")
    trends_parser.add_argument("--stage", "-s", help="Focus on specific stage")
    trends_parser.add_argument("--db", type=Path, default=DEFAULT_BENCHMARK_DB,
                              help="Benchmark database path")
    trends_parser.add_argument("--output", "-o", type=Path, help="Output JSON path")
    trends_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    trends_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")
    trends_parser.add_argument("--json", "-j", action="store_true",
                               help="Output results as JSON")

    # Config comparison command
    config_parser = subparsers.add_parser("compare-configs", help="Compare two configs")
    config_parser.add_argument("--project", "-p", required=True, help="Project path")
    config_parser.add_argument("--config-a", required=True, help="Config file A")
    config_parser.add_argument("--config-b", required=True, help="Config file B")
    config_parser.add_argument("--db", type=Path, default=DEFAULT_BENCHMARK_DB,
                              help="Benchmark database path")
    config_parser.add_argument("--output", "-o", type=Path, help="Output JSON path")
    config_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    config_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")
    config_parser.add_argument("--json", "-j", action="store_true",
                               help="Output results as JSON")

    args = parser.parse_args()

    # Set verbosity level based on flags
    if hasattr(args, 'quiet') and args.quiet:
        set_verbosity(0)
    elif hasattr(args, 'verbose') and args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "run":
        result = run_pipeline_benchmark(
            args.project,
            config_path=args.config,
            fresh=args.fresh,
            verbose=args.verbose,
        )

        # JSON output mode
        if args.json:
            json_output(
                success=(result.get("return_code", 1) == 0),
                script_name="benchmark",
                data={
                    "project_path": args.project,
                    "total_duration": result.get("total_duration", 0),
                    "stage_timings": result.get("stage_timings", {}),
                    "return_code": result.get("return_code", 1),
                }
            )
        else:
            # Print summary
            print_header("Benchmark Complete")
            print_ok(f"Completed in {result['total_duration']:.2f}s")
            print_info("Stage Timings:")
            for stage, duration in result.get("stage_timings", {}).items():
                print_info(f"  {stage}: {duration:.2f}s")

        if args.save_results:
            run_id = save_benchmark_result(
                args.db,
                result,
                notes=args.notes,
                tags=args.tags,
            )
            if not args.json:
                print_ok(f"Saved to database: {args.db}")
                print_info(f"Run ID: {run_id}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            if not args.json:
                print_ok(f"Results saved to: {args.output}")

        return 0

    elif args.command == "compare":
        with open(args.baseline) as f:
            baseline = json.load(f)
        with open(args.current) as f:
            current = json.load(f)

        passed, regressions = compare_benchmarks(baseline, current, verbose=args.verbose)

        if args.json:
            json_output(
                success=passed,
                script_name="benchmark",
                data={
                    "comparison": "baseline_vs_current",
                    "baseline": str(args.baseline),
                    "current": str(args.current),
                    "passed": passed,
                    "regressions": regressions,
                }
            )

        return 0 if passed else 1

    elif args.command == "trends":
        report = generate_trend_report(
            args.db,
            project_path=args.project,
            stage_name=args.stage,
            output_path=args.output,
        )

        if args.json:
            json_output(
                success=True,
                script_name="benchmark",
                data={
                    "command": "trends",
                    "report": report,
                }
            )

        return 0

    elif args.command == "compare-configs":
        result = run_config_comparison(
            args.project,
            args.config_a,
            args.config_b,
            args.db,
            verbose=args.verbose,
        )

        if args.json:
            json_output(
                success=result["passed"],
                script_name="benchmark",
                data={
                    "command": "compare-configs",
                    "config_a": args.config_a,
                    "config_b": args.config_b,
                    "passed": result["passed"],
                    "regressions": result.get("regressions", []),
                }
            )
        elif args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print_ok(f"Results saved to: {args.output}")

        return 0 if result["passed"] else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
