#!/usr/bin/env python3
"""
Analyze test files for parallelization safety.

Detects patterns that may prevent safe parallel execution with pytest-xdist:
- Global mutable state
- Singleton patterns
- File system operations without isolation
- os.chdir() usage
- Session-scoped fixtures with side effects

Usage:
    python scripts/analyze_test_parallelization.py [--verbose] [--suggest-serial]
    python scripts/analyze_test_parallelization.py --fix  # Auto-add @pytest.mark.serial

Exit codes:
    0 - All tests are parallelization-safe
    1 - Unsafe tests found (check output for details)
"""

import argparse
import ast
import os
import re
import sys
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


@dataclass
class ParallelizationIssue:
    """Represents a potential parallelization issue in a test file."""

    file: Path
    line: int
    category: str  # 'global_state', 'singleton', 'chdir', 'file_lock', 'session_fixture'
    description: str
    severity: str = "warning"  # 'warning', 'error', 'info'
    function_name: Optional[str] = None


@dataclass
class FileAnalysis:
    """Analysis results for a single test file."""

    file: Path
    issues: list[ParallelizationIssue] = field(default_factory=list)
    test_count: int = 0
    fixture_count: int = 0
    has_session_scope: bool = False
    uses_tmp_path: bool = False
    uses_chdir: bool = False
    global_variables: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return len([i for i in self.issues if i.severity == "error"]) == 0


class ParallelizationAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze test files for parallelization issues."""

    GLOBAL_STATE_PATTERNS = [
        r"global\s+(\w+)",
        r"(\w+)\s*=\s*\[\](?:\s*#.*)?$",  # module-level list assignment
        r"(\w+)\s*=\s*\{\}(?:\s*#.*)?$",  # module-level dict assignment
    ]

    UNSAFE_PATTERNS = [
        (r"os\.chdir\s*\(", "chdir", "Directory change affects all workers"),
        (r"fcntl\.flock\s*\(", "file_lock", "File locking can cause deadlocks"),
        (r"fcntl\.lockf\s*\(", "file_lock", "File locking can cause deadlocks"),
        (r"portalocker\.", "file_lock", "File locking can cause deadlocks"),
        (r"sqlite3\.connect\s*\([^)]*:memory:", "sqlite_memory", "In-memory SQLite is not shared"),
        (r"@singleton", "singleton", "Singleton pattern may conflict across workers"),
        (r"Singleton\s*\(", "singleton", "Singleton pattern may conflict across workers"),
    ]

    def __init__(self, file_path: Path, source_lines: list[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.issues: list[ParallelizationIssue] = []
        self.test_count = 0
        self.fixture_count = 0
        self.has_session_scope = False
        self.uses_tmp_path = False
        self.uses_chdir = False
        self.global_variables: list[str] = []
        self.current_function: Optional[str] = None

    def analyze(self) -> FileAnalysis:
        """Run full analysis on the file."""
        source_text = "\n".join(self.source_lines)

        # Pattern-based analysis
        self._analyze_patterns(source_text)

        # AST-based analysis
        try:
            tree = ast.parse(source_text)
            self.visit(tree)
        except SyntaxError as e:
            self.issues.append(
                ParallelizationIssue(
                    file=self.file_path,
                    line=e.lineno or 0,
                    category="syntax_error",
                    description=f"Could not parse: {e.msg}",
                    severity="warning",
                )
            )

        return FileAnalysis(
            file=self.file_path,
            issues=self.issues,
            test_count=self.test_count,
            fixture_count=self.fixture_count,
            has_session_scope=self.has_session_scope,
            uses_tmp_path=self.uses_tmp_path,
            uses_chdir=self.uses_chdir,
            global_variables=self.global_variables,
        )

    def _analyze_patterns(self, source_text: str):
        """Analyze source using regex patterns."""
        for line_num, line in enumerate(self.source_lines, 1):
            for pattern, category, desc in self.UNSAFE_PATTERNS:
                if re.search(pattern, line):
                    self.issues.append(
                        ParallelizationIssue(
                            file=self.file_path,
                            line=line_num,
                            category=category,
                            description=desc,
                            severity="error" if category in ("chdir", "file_lock") else "warning",
                        )
                    )
                    if category == "chdir":
                        self.uses_chdir = True

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Analyze function definitions."""
        self.current_function = node.name

        # Check for test functions
        if node.name.startswith("test_"):
            self.test_count += 1

        # Check for fixtures
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if decorator.func.attr == "fixture":
                        self.fixture_count += 1
                        # Check for session scope
                        for keyword in decorator.keywords:
                            if keyword.arg == "scope":
                                if isinstance(keyword.value, ast.Constant):
                                    if keyword.value.value == "session":
                                        self.has_session_scope = True
                                        self.issues.append(
                                            ParallelizationIssue(
                                                file=self.file_path,
                                                line=node.lineno,
                                                category="session_fixture",
                                                description=f"Session-scoped fixture '{node.name}' may have shared state",
                                                severity="warning",
                                                function_name=node.name,
                                            )
                                        )

            elif isinstance(decorator, ast.Attribute):
                if decorator.attr == "fixture":
                    self.fixture_count += 1

        # Check for tmp_path parameter
        for arg in node.args.args:
            if arg.arg == "tmp_path":
                self.uses_tmp_path = True

        self.generic_visit(node)
        self.current_function = None

    def visit_Global(self, node: ast.Global):
        """Detect global statement usage."""
        for name in node.names:
            self.global_variables.append(name)
            self.issues.append(
                ParallelizationIssue(
                    file=self.file_path,
                    line=node.lineno,
                    category="global_state",
                    description=f"Global variable '{name}' modified in function",
                    severity="error",
                    function_name=self.current_function,
                )
            )

    def visit_Assign(self, node: ast.Assign):
        """Detect module-level mutable assignments."""
        # Only care about module-level (col_offset 0)
        if node.col_offset == 0 and self.current_function is None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Check for mutable default values
                    if isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
                        self.issues.append(
                            ParallelizationIssue(
                                file=self.file_path,
                                line=node.lineno,
                                category="global_state",
                                description=f"Module-level mutable '{target.id}' - could cause race conditions",
                                severity="info",
                            )
                        )

        self.generic_visit(node)


def analyze_test_directory(tests_dir: Path, verbose: bool = False) -> list[FileAnalysis]:
    """Analyze all test files in a directory."""
    results = []

    test_files = sorted(tests_dir.rglob("test_*.py"))

    for test_file in test_files:
        try:
            source_lines = test_file.read_text(encoding="utf-8").splitlines()
            analyzer = ParallelizationAnalyzer(test_file, source_lines)
            analysis = analyzer.analyze()
            results.append(analysis)

            if verbose and analysis.issues:
                print_info(f"{test_file.relative_to(tests_dir)}: {len(analysis.issues)} issues")
                for issue in analysis.issues:
                    print_info(f"  L{issue.line}: [{issue.severity}] {issue.category} - {issue.description}")

        except Exception as e:
            print_error(f"Error analyzing {test_file}: {e}", exit_code=1)

    return results


def suggest_serial_tests(analyses: list[FileAnalysis]) -> list[Path]:
    """Identify tests that should be marked with @pytest.mark.serial."""
    serial_candidates = []

    for analysis in analyses:
        # Tests with error-level issues should be serial
        error_issues = [i for i in analysis.issues if i.severity == "error"]
        if error_issues:
            serial_candidates.append(analysis.file)
        # Tests with session fixtures that have side effects
        elif analysis.has_session_scope and analysis.uses_chdir:
            serial_candidates.append(analysis.file)

    return serial_candidates


def print_summary(analyses: list[FileAnalysis]):
    """Print analysis summary."""
    total_files = len(analyses)
    total_tests = sum(a.test_count for a in analyses)
    total_issues = sum(len(a.issues) for a in analyses)
    error_count = sum(len([i for i in a.issues if i.severity == "error"]) for a in analyses)
    warning_count = sum(len([i for i in a.issues if i.severity == "warning"]) for a in analyses)
    info_count = sum(len([i for i in a.issues if i.severity == "info"]) for a in analyses)

    safe_files = len([a for a in analyses if a.is_safe])
    files_with_tmp_path = len([a for a in analyses if a.uses_tmp_path])
    files_with_session = len([a for a in analyses if a.has_session_scope])
    files_with_chdir = len([a for a in analyses if a.uses_chdir])

    print_header("TEST PARALLELIZATION ANALYSIS SUMMARY")

    print_info(f"Files analyzed: {total_files}")
    print_info(f"Total tests:    {total_tests}")
    print_info(f"Safe for parallel:  {safe_files}/{total_files} ({100*safe_files/total_files:.1f}%)")

    if error_count > 0:
        print_error(f"Issues found: Errors: {error_count}, Warnings: {warning_count}, Info: {info_count}")
    elif warning_count > 0:
        print_warn(f"Issues found: Errors: {error_count}, Warnings: {warning_count}, Info: {info_count}")
    else:
        print_info(f"Issues found: Errors: {error_count}, Warnings: {warning_count}, Info: {info_count}")

    print_info(f"Patterns detected:")
    print_info(f"  Files using tmp_path:       {files_with_tmp_path}")
    print_info(f"  Files with session scope:   {files_with_session}")
    print_info(f"  Files using os.chdir():     {files_with_chdir}")

    # List files with errors
    error_files = [a for a in analyses if not a.is_safe]
    if error_files:
        print_error(f"Files needing @pytest.mark.serial ({len(error_files)}):")
        for analysis in error_files:
            print_error(f"  - {analysis.file.name}")
            for issue in analysis.issues:
                if issue.severity == "error":
                    print_error(f"      L{issue.line}: {issue.description}")


def generate_report(analyses: list[FileAnalysis], output_path: Optional[Path] = None):
    """Generate a detailed report in Markdown format."""
    serial_candidates = suggest_serial_tests(analyses)

    lines = [
        "# Test Parallelization Analysis Report",
        "",
        "Generated by `scripts/analyze_test_parallelization.py`",
        "",
        "## Summary",
        "",
        f"- **Total files analyzed:** {len(analyses)}",
        f"- **Total tests:** {sum(a.test_count for a in analyses)}",
        f"- **Files safe for parallel:** {len([a for a in analyses if a.is_safe])}/{len(analyses)}",
        f"- **Files needing @pytest.mark.serial:** {len(serial_candidates)}",
        "",
        "## Files Requiring Serial Execution",
        "",
        "These files should be marked with `@pytest.mark.serial` or have their tests refactored:",
        "",
    ]

    if serial_candidates:
        for path in serial_candidates:
            analysis = next(a for a in analyses if a.file == path)
            lines.append(f"### {path.name}")
            lines.append("")
            lines.append(f"- **Tests:** {analysis.test_count}")
            lines.append(f"- **Issues:**")
            for issue in analysis.issues:
                if issue.severity == "error":
                    lines.append(f"  - L{issue.line}: {issue.description}")
            lines.append("")
    else:
        lines.append("None - all tests are parallelization-safe!")
        lines.append("")

    lines.extend([
        "## Parallelization Guidelines",
        "",
        "### Safe Patterns",
        "- Using `tmp_path` fixture for file operations",
        "- Using `monkeypatch` for environment variables",
        "- Using function-scoped fixtures",
        "- Using mock objects instead of real I/O",
        "",
        "### Unsafe Patterns",
        "- `os.chdir()` - affects all workers",
        "- Global mutable state modified in tests",
        "- Session-scoped fixtures with side effects",
        "- File locks without worker isolation",
        "- Singleton patterns without test isolation",
        "",
        "### Running Tests",
        "",
        "```bash",
        "# Run all tests in parallel (excluding serial tests)",
        "pytest tests/ -n auto -m 'not serial'",
        "",
        "# Run serial tests separately",
        "pytest tests/ -m serial",
        "",
        "# Run with both",
        "pytest tests/ -n auto --dist loadgroup",
        "```",
        "",
    ])

    content = "\n".join(lines)

    if output_path:
        output_path.write_text(content, encoding="utf-8")
        print_ok(f"Report written to: {output_path}")
    else:
        print(content)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze test files for parallelization safety",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--suggest-serial", action="store_true", help="List files needing @pytest.mark.serial")
    parser.add_argument("--report", type=Path, help="Generate Markdown report to file")
    parser.add_argument("--tests-dir", type=Path, default=Path("tests"), help="Tests directory (default: tests)")

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    tests_dir = project_root / args.tests_dir

    if not tests_dir.exists():
        print_error(f"Tests directory not found: {tests_dir}", exit_code=1)

    print_info(f"Analyzing tests in: {tests_dir}")
    analyses = analyze_test_directory(tests_dir, verbose=args.verbose)

    print_summary(analyses)

    if args.suggest_serial:
        serial = suggest_serial_tests(analyses)
        if serial:
            print_info("Suggested files for @pytest.mark.serial:")
            for path in serial:
                print_info(f"  - {path.relative_to(tests_dir)}")

    if args.report:
        generate_report(analyses, args.report)

    # Exit with error if unsafe tests found
    unsafe_count = len([a for a in analyses if not a.is_safe])
    sys.exit(1 if unsafe_count > 0 else 0)


if __name__ == "__main__":
    main()
