#!/usr/bin/env python3
"""
Integration Test Runner with Isolated Environment

US-010: Add integration test runner with isolated environment

This script provides an isolated execution environment for integration tests
with the following features:
- Temporary directory setup for each test run
- Automatic cleanup of test artifacts
- Parallel test execution with --parallel flag
- Timeout protection (default: 5 minutes per test)
- Comprehensive test reporting

Usage:
    python scripts/run_integration_tests.py
    python scripts/run_integration_tests.py --parallel 4
    python scripts/run_integration_tests.py --timeout 300 --verbose
    python scripts/run_integration_tests.py --keep-artifacts
    python scripts/run_integration_tests.py --pattern "test_caption*"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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


# =============================================================================
# Constants
# =============================================================================

DEFAULT_TIMEOUT = 300  # 5 minutes per test
DEFAULT_PARALLEL_WORKERS = 1
INTEGRATION_MARKERS = ["integration", "requires_network", "requires_api"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test execution."""
    name: str
    status: str  # passed, failed, skipped, error, timeout
    duration: float
    output: str = ""
    error: str = ""


@dataclass
class RunSummary:
    """Summary of an integration test run."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    timeouts: int = 0
    duration: float = 0.0
    start_time: str = ""
    end_time: str = ""
    temp_dir: str = ""
    results: List[TestResult] = field(default_factory=list)


# =============================================================================
# Environment Setup
# =============================================================================

class IsolatedTestEnvironment:
    """
    Manages an isolated test environment with temporary directories.

    Creates a clean temporary directory structure for each test run,
    sets up environment variables, and handles cleanup.
    """

    def __init__(self, base_dir: Optional[Path] = None, keep_artifacts: bool = False):
        self.keep_artifacts = keep_artifacts
        self.base_dir = base_dir
        self.temp_dir: Optional[Path] = None
        self.original_env: Dict[str, str] = {}

    def setup(self) -> Path:
        """Create isolated test environment and return temp directory path."""
        # Create temp directory
        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir = Path(tempfile.mkdtemp(
                prefix="integration_test_",
                dir=str(self.base_dir)
            ))
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="integration_test_"))

        # Create subdirectories for test isolation
        (self.temp_dir / "cache").mkdir()
        (self.temp_dir / "output").mkdir()
        (self.temp_dir / "downloads").mkdir()
        (self.temp_dir / "projects").mkdir()

        # Save and set environment variables for isolation
        self.original_env = dict(os.environ)
        os.environ["INTEGRATION_TEST_TEMP_DIR"] = str(self.temp_dir)
        os.environ["INTEGRATION_TEST_CACHE_DIR"] = str(self.temp_dir / "cache")
        os.environ["INTEGRATION_TEST_OUTPUT_DIR"] = str(self.temp_dir / "output")
        os.environ["INTEGRATION_TEST_DOWNLOADS_DIR"] = str(self.temp_dir / "downloads")

        # Disable caching for integration tests to ensure clean runs
        os.environ["INTEGRATION_TEST_NO_CACHE"] = "1"

        return self.temp_dir

    def cleanup(self):
        """Clean up test environment and artifacts."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

        # Remove temp directory if not keeping artifacts
        if self.temp_dir and self.temp_dir.exists() and not self.keep_artifacts:
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temp directory {self.temp_dir}: {e}")

    def __enter__(self) -> Path:
        return self.setup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# =============================================================================
# Test Runner
# =============================================================================

class IntegrationTestRunner:
    """
    Runs integration tests with isolation, timeouts, and parallel execution.
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        parallel: int = DEFAULT_PARALLEL_WORKERS,
        verbose: bool = False,
        pattern: Optional[str] = None,
        markers: Optional[List[str]] = None,
        keep_artifacts: bool = False,
        base_dir: Optional[Path] = None,
    ):
        self.timeout = timeout
        self.parallel = parallel
        self.verbose = verbose
        self.pattern = pattern
        self.markers = markers or INTEGRATION_MARKERS
        self.keep_artifacts = keep_artifacts
        self.base_dir = base_dir
        self.project_root = Path(__file__).parent.parent

    def discover_tests(self) -> List[str]:
        """Discover integration tests matching criteria."""
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests"),
            "--collect-only", "-q",
            "-m", " or ".join(self.markers),
        ]

        if self.pattern:
            cmd.extend(["-k", self.pattern])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=str(self.project_root),
                timeout=60,
            )

            # Parse test names from output
            tests = []
            for line in result.stdout.splitlines():
                line = line.strip()
                # Filter out summary lines and empty lines
                if line and "::" in line and not line.startswith(("=", "-", "<")):
                    tests.append(line)

            return tests

        except subprocess.TimeoutExpired:
            print("Warning: Test discovery timed out")
            return []
        except Exception as e:
            print(f"Error discovering tests: {e}")
            return []

    def run_single_test(
        self,
        test_name: str,
        temp_dir: Path,
    ) -> TestResult:
        """Run a single test with timeout protection."""
        start_time = time.time()

        cmd = [
            sys.executable, "-m", "pytest",
            test_name,
            "-v",
            "--tb=short",
            "--no-header",
            f"--basetemp={temp_dir / 'pytest'}",
            "--non-interactive",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=str(self.project_root),
                timeout=self.timeout,
                env={
                    **os.environ,
                    "INTEGRATION_TEST_TEMP_DIR": str(temp_dir),
                },
            )

            duration = time.time() - start_time

            # Determine status from return code
            if result.returncode == 0:
                status = "passed"
            elif result.returncode == 5:
                status = "skipped"  # pytest exit code for no tests collected
            else:
                status = "failed"

            return TestResult(
                name=test_name,
                status=status,
                duration=duration,
                output=result.stdout,
                error=result.stderr,
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                status="timeout",
                duration=duration,
                error=f"Test timed out after {self.timeout} seconds",
                output=e.stdout.decode('utf-8', errors='replace') if e.stdout else "",
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                status="error",
                duration=duration,
                error=str(e),
            )

    def run_parallel(
        self,
        tests: List[str],
        temp_dir: Path,
    ) -> List[TestResult]:
        """Run tests in parallel using pytest-xdist if available, otherwise sequential."""
        if self.parallel <= 1:
            return self._run_sequential(tests, temp_dir)

        # Try to use pytest-xdist for parallel execution
        start_time = time.time()

        cmd = [
            sys.executable, "-m", "pytest",
            *tests,
            "-v",
            "--tb=short",
            f"--basetemp={temp_dir / 'pytest'}",
            f"-n={self.parallel}",
            "--non-interactive",
            "--dist=loadfile",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=str(self.project_root),
                timeout=self.timeout * len(tests),  # Total timeout
                env={
                    **os.environ,
                    "INTEGRATION_TEST_TEMP_DIR": str(temp_dir),
                },
            )

            # Parse results from output
            return self._parse_pytest_output(result.stdout, result.stderr, tests, time.time() - start_time)

        except subprocess.TimeoutExpired:
            print("Warning: Parallel execution timed out, falling back to sequential")
            return self._run_sequential(tests, temp_dir)
        except Exception as e:
            if "pytest-xdist" in str(e) or "xdist" in str(e) or "-n" in str(e):
                print("Note: pytest-xdist not installed, running tests sequentially")
                return self._run_sequential(tests, temp_dir)
            raise

    def _run_sequential(
        self,
        tests: List[str],
        temp_dir: Path,
    ) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        for i, test in enumerate(tests, 1):
            if self.verbose:
                print(f"\n[{i}/{len(tests)}] Running: {test}")

            result = self.run_single_test(test, temp_dir)
            results.append(result)

            if self.verbose:
                status_symbol = {
                    "passed": "✓",
                    "failed": "✗",
                    "skipped": "○",
                    "error": "!",
                    "timeout": "⏱",
                }.get(result.status, "?")
                print(f"  {status_symbol} {result.status.upper()} ({result.duration:.2f}s)")

        return results

    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
        tests: List[str],
        total_duration: float,
    ) -> List[TestResult]:
        """Parse pytest output to extract individual test results."""
        results = []
        avg_duration = total_duration / max(len(tests), 1)

        for test in tests:
            # Try to find result in output
            status = "passed"
            if f"{test} FAILED" in stdout or f"{test} FAILED" in stderr:
                status = "failed"
            elif f"{test} SKIPPED" in stdout or f"{test} SKIPPED" in stderr:
                status = "skipped"
            elif f"{test} ERROR" in stdout or f"{test} ERROR" in stderr:
                status = "error"

            results.append(TestResult(
                name=test,
                status=status,
                duration=avg_duration,  # Approximate
                output="",
                error="",
            ))

        return results

    def run(self) -> RunSummary:
        """Run all integration tests and return summary."""
        start_time = datetime.now()

        print("=" * 70)
        print("INTEGRATION TEST RUNNER")
        print("=" * 70)
        print(f"Start time: {start_time.isoformat()}")
        print(f"Timeout per test: {self.timeout}s")
        print(f"Parallel workers: {self.parallel}")
        print(f"Markers: {', '.join(self.markers)}")
        if self.pattern:
            print(f"Pattern: {self.pattern}")
        print()

        # Discover tests
        print("Discovering tests...")
        tests = self.discover_tests()

        if not tests:
            print("No integration tests found matching criteria.")
            return RunSummary(
                total=0,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
            )

        print(f"Found {len(tests)} test(s)")
        print()

        # Set up isolated environment and run tests
        env = IsolatedTestEnvironment(
            base_dir=self.base_dir,
            keep_artifacts=self.keep_artifacts,
        )

        with env as temp_dir:
            print(f"Temp directory: {temp_dir}")
            print("-" * 70)

            if self.parallel > 1:
                results = self.run_parallel(tests, temp_dir)
            else:
                results = self._run_sequential(tests, temp_dir)

            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()

            # Build summary
            summary = RunSummary(
                total=len(results),
                passed=sum(1 for r in results if r.status == "passed"),
                failed=sum(1 for r in results if r.status == "failed"),
                skipped=sum(1 for r in results if r.status == "skipped"),
                errors=sum(1 for r in results if r.status == "error"),
                timeouts=sum(1 for r in results if r.status == "timeout"),
                duration=total_duration,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                temp_dir=str(temp_dir) if self.keep_artifacts else "",
                results=results,
            )

            # Print summary
            self._print_summary(summary)

            return summary

    def _print_summary(self, summary: RunSummary):
        """Print run summary to console."""
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total:    {summary.total}")
        print(f"Passed:   {summary.passed}")
        print(f"Failed:   {summary.failed}")
        print(f"Skipped:  {summary.skipped}")
        print(f"Errors:   {summary.errors}")
        print(f"Timeouts: {summary.timeouts}")
        print(f"Duration: {summary.duration:.2f}s")
        print()

        # List failed/error tests
        failures = [r for r in summary.results if r.status in ("failed", "error", "timeout")]
        if failures:
            print("FAILURES:")
            print("-" * 70)
            for result in failures:
                print(f"  [{result.status.upper()}] {result.name}")
                if result.error:
                    # Indent error message
                    for line in result.error.splitlines()[:5]:
                        print(f"    {line}")
            print()

        # Final status
        if summary.failed == 0 and summary.errors == 0 and summary.timeouts == 0:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed")

        if summary.temp_dir:
            print(f"\nArtifacts preserved at: {summary.temp_dir}")


# =============================================================================
# JSON Report
# =============================================================================

def summary_to_json(summary: RunSummary) -> Dict:
    """Convert RunSummary to JSON-serializable dictionary."""
    return {
        "timestamp": summary.start_time,
        "end_time": summary.end_time,
        "duration_seconds": summary.duration,
        "totals": {
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "skipped": summary.skipped,
            "errors": summary.errors,
            "timeouts": summary.timeouts,
        },
        "success_rate": summary.passed / max(summary.total, 1) * 100,
        "results": [
            {
                "name": r.name,
                "status": r.status,
                "duration": r.duration,
                "error": r.error[:500] if r.error else "",  # Truncate long errors
            }
            for r in summary.results
        ],
    }


def export_report(summary: RunSummary, output_path: Path):
    """Export run summary as JSON report."""
    report = {
        "timestamp": summary.start_time,
        "duration_seconds": summary.duration,
        "totals": {
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "skipped": summary.skipped,
            "errors": summary.errors,
            "timeouts": summary.timeouts,
        },
        "success_rate": summary.passed / max(summary.total, 1) * 100,
        "results": [
            {
                "name": r.name,
                "status": r.status,
                "duration": r.duration,
            }
            for r in summary.results
        ],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"Report exported to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run integration tests with isolated environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_integration_tests.py
  python scripts/run_integration_tests.py --parallel 4
  python scripts/run_integration_tests.py --timeout 600 --verbose
  python scripts/run_integration_tests.py --keep-artifacts
  python scripts/run_integration_tests.py --pattern "test_caption*"
  python scripts/run_integration_tests.py --markers integration requires_network
  python scripts/run_integration_tests.py --json-report report.json
        """,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per test in seconds (default: {DEFAULT_TIMEOUT})",
    )

    parser.add_argument(
        "--parallel", "-n",
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_PARALLEL_WORKERS})",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--pattern", "-k",
        type=str,
        help="Test name pattern to match (like pytest -k)",
    )

    parser.add_argument(
        "--markers", "-m",
        nargs="+",
        default=INTEGRATION_MARKERS,
        help=f"Markers to select (default: {' '.join(INTEGRATION_MARKERS)})",
    )

    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep test artifacts after completion",
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory for temp files (default: system temp)",
    )

    parser.add_argument(
        "--json-report",
        type=Path,
        help="Export JSON report to file",
    )

    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON to stdout",
    )

    args = parser.parse_args()

    runner = IntegrationTestRunner(
        timeout=args.timeout,
        parallel=args.parallel,
        verbose=args.verbose,
        pattern=args.pattern,
        markers=args.markers,
        keep_artifacts=args.keep_artifacts,
        base_dir=args.base_dir,
    )

    summary = runner.run()

    # Handle JSON output to stdout
    if args.json:
        result = summary_to_json(summary)
        print(json.dumps(result, indent=2))
        # Validate JSON is parseable
        try:
            json.loads(json.dumps(result))
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON output is invalid: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0 if summary.failed == 0 and summary.errors == 0 and summary.timeouts == 0 else 1)

    if args.json_report:
        export_report(summary, args.json_report)

    # Exit with appropriate code
    if summary.failed > 0 or summary.errors > 0 or summary.timeouts > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
