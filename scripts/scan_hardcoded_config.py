#!/usr/bin/env python3
"""
Scanner for hardcoded config values in stages.

Scans src/stages/*.py files for hardcoded numeric/string literals that should
use config getattr() pattern instead. Helps identify values that should be
configurable via config.yaml.

Usage:
    python scan_hardcoded_config.py
    python scan_hardcoded_config.py --verbose
    python scan_hardcoded_config.py --json
    python scan_hardcoded_config.py --format json
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Import standardized output functions
from script_utils import print_ok, print_warn, print_error, set_verbosity


# Patterns that indicate hardcoded config values (candidate patterns)
CANDIDATE_PATTERNS = [
    # min_duration = 30, threshold = 0.5, timeout = 60, max_xxx = 100
    r'\b(min|max|threshold|timeout|limit|count|size|duration)_?\w*\s*=\s*\d+\.?\d*\b',
    # _variable = 30 (private variables)
    r'\b_\w+\s*=\s*\d+\.?\d*\b',
    # Some specific hardcoded values that are commonly config-related
    r'=\s*30\b',   # timeout 30
    r'=\s*60\b',   # timeout 60
    r'=\s*120\b',  # timeout 120
    r'=\s*1080\b', # resolution
    r'=\s*720\b',  # resolution
    r'=\s*5\b',    # threshold/count (but be careful)
    r'=\s*10\b',   # threshold/count
]

# False positive patterns to exclude
EXCLUDE_PATTERNS = [
    # Constants/enums
    r'^\s*#\s*constant',
    r'^\s*#\s*enum',
    r'^\s*#\s*default',
    # Test fixtures
    r'def test_',
    r'@pytest',
    r'fixture',
    # Already using getattr pattern (safe)
    r'getattr\(',
    r'get\(',
    r'\.get\(',
    # Docstrings and comments
    r'^\s*"""',
    r"^\s*'''",
    r'^\s*#',
    # Import statements
    r'^import\s',
    r'^from\s',
    # Class/function definitions
    r'^\s*def\s',
    r'^\s*class\s',
    # Type annotations
    r':\s*int\s*=',
    r':\s*float\s*=',
    r':\s*str\s*=',
    r':\s*bool\s*=',
    r':\s*Optional\[',
    r'->\s*\w+',
    # Lambda functions
    r'lambda\s',
    # Dictionary comprehensions
    r'\[\s*\w+\s*for',
    # Range/len calls
    r'range\(',
    r'len\(',
    # Mathematical operations (not config)
    r'\+\s*\d+',
    r'-\s*\d+',
    r'\*\s*\d+',
    r'/\s*\d+',
    r'%\s*\d+',
    # List indices
    r'\[\d+\]',
    # Version numbers
    r'version\s*=',
    r'__version__',
    # URL/port numbers (not config)
    r'https?://',
    r'port\s*=',
    # Enum values
    r'Tier\.\w+',
    r'Stage\.\w+',
    r'Status\.\w+',
    r'LogLevel\.\w+',
    # Logging levels
    r'logging\.\w+',
    r'level\s*=\s*logging',
    # String literals that are not config
    r'=\s*["\'][^"\']*["\']',
    # True/False/None
    r'=\s*(True|False|None)\s*$',
    # Self references
    r'self\.\w+\s*=',
    r'self\.\w+\s*=\s*self',
    # _state/_config suffixes (already handled)
    r'_state\s*=',
    r'_config\s*=',


]


def is_false_positive(line: str, pattern_match: str) -> bool:
    """Check if the match is a false positive."""
    # Skip if line contains getattr pattern (safe)
    if 'getattr' in line or '.get(' in line:
        return True

    # Skip if line is in exclude patterns
    for exclude in EXCLUDE_PATTERNS:
        if re.search(exclude, line):
            return True

    # Skip if it's a simple counter increment/decrement
    if re.match(r'^\s*\w+\s*[+-]=\s*\d+', line):
        return True

    # Skip if it's a range or index access
    if re.search(r'\[\s*\d+\s*\]', line):
        return True

    # Skip if it's in a test function
    if 'def test_' in line or '@pytest' in line:
        return True

    # Skip logging configuration
    if 'logging.' in line and ('level' in line or 'format' in line):
        return True

    # Skip type annotations
    if re.search(r':\s*(int|float|str|bool|list|dict|Optional)\b', line):
        return True

    # Skip lambda functions
    if 'lambda' in line:
        return True

    # Skip URL/port definitions
    if 'http' in line.lower() or 'port' in line:
        return True

    # Skip version strings
    if 'version' in line.lower() or '__version__' in line:
        return True

    # Skip self assignments (except config-related)
    if 'self.' in line and not any(x in line for x in ['config', 'threshold', 'timeout', 'duration', 'max_', 'min_']):
        return True

    return False


def scan_file(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    """Scan a single file for hardcoded config values."""
    findings = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        if verbose:
            print_warn(f"Could not read {file_path}: {e}")
        return findings

    for line_num, line in enumerate(lines, 1):
        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith('#'):
            continue

        # Check each candidate pattern
        for pattern in CANDIDATE_PATTERNS:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                match_text = match.group(0)

                # Check if it's a false positive
                if is_false_positive(line, match_text):
                    continue

                # Check if line already uses config
                if 'config' in line.lower() and ('getattr' in line or 'get(' in line):
                    continue

                findings.append({
                    'file': str(file_path.relative_to(project_root)),
                    'line': line_num,
                    'pattern': match_text,
                    'content': line.strip(),
                })

                if verbose:
                    print_warn(f"  Found: {file_path.name}:{line_num} -> {match_text}")

    return findings


def scan_stages_dir(stages_dir: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    """Scan all Python files in stages directory."""
    all_findings = []

    if not stages_dir.exists():
        print_error(f"Stages directory not found: {stages_dir}")
        return all_findings

    # Get all Python files, excluding __init__.py and test files
    py_files = [
        f for f in stages_dir.glob('*.py')
        if f.name != '__init__.py' and not f.name.startswith('test_')
    ]

    if verbose:
        print_ok(f"Scanning {len(py_files)} stage files...")

    for py_file in py_files:
        findings = scan_file(py_file, verbose)
        all_findings.extend(findings)

    return all_findings


def format_findings_json(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format findings as JSON report."""
    return {
        'scanner': 'scan_hardcoded_config',
        'version': '1.0.0',
        'total_findings': len(findings),
        'findings': findings,
    }


def format_findings_text(findings: List[Dict[str, Any]]) -> str:
    """Format findings as human-readable text."""
    if not findings:
        return "[OK] No hardcoded config values found"

    lines = []
    lines.append(f"\nFound {len(findings)} potential hardcoded config values:\n")

    # Group by file
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for f in findings:
        by_file.setdefault(f['file'], []).append(f)

    for file_path, file_findings in sorted(by_file.items()):
        lines.append(f"  {file_path}:")
        for finding in sorted(file_findings, key=lambda x: x['line']):
            lines.append(f"    Line {finding['line']}: {finding['pattern']}")
            lines.append(f"      {finding['content'][:80]}")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scan src/stages/ for hardcoded config values"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--stages-dir',
        type=str,
        help='Custom stages directory path'
    )

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    # Determine stages directory
    if args.stages_dir:
        stages_dir = Path(args.stages_dir)
    else:
        stages_dir = project_root / 'src' / 'stages'

    # Scan for hardcoded values
    findings = scan_stages_dir(stages_dir, verbose=args.verbose)

    # Output results
    if args.json or args.format == 'json':
        report = format_findings_json(findings)
        print(json.dumps(report, indent=2))
    else:
        print(format_findings_text(findings))

    # Return exit code based on findings
    if findings:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
