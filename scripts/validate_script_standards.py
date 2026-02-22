#!/usr/bin/env python3
"""
Script Standards Validation Tool

Validates that scripts in the scripts/ directory follow the project's
standards for consistent script structure.

This tool can be used as:
- Pre-commit hook
- CI validation step
- Local development check

Usage:
    python scripts/validate_script_standards.py
    python scripts/validate_script_standards.py --verbose
    python scripts/validate_script_standards.py --fix
    python scripts/validate_script_standards.py --pre-commit
    python scripts/validate_script_standards.py --json
"""

import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)

# Import standardized output functions
from script_utils import (
    print_header, print_ok, print_warn, print_error, print_info,
    set_verbosity
)


@dataclass
class ValidationResult:
    """Result for a single script validation."""
    script_path: str
    is_utils: bool = False
    passed: bool = False
    checks: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)


def check_shebang(content: str) -> Tuple[bool, str]:
    """Check if script has proper shebang."""
    lines = content.split('\n')
    if not lines:
        return False, "Empty file"
    first_line = lines[0].strip()
    if first_line == '#!/usr/bin/env python3':
        return True, ""
    elif first_line.startswith('#!'):
        return False, f"Invalid shebang: {first_line}"
    else:
        return False, "Missing shebang (expected: #!/usr/bin/env python3)"


def fix_shebang(content: str) -> str:
    """Add shebang line if missing."""
    lines = content.split('\n')
    if lines and not lines[0].startswith('#!'):
        lines.insert(0, '#!/usr/bin/env python3')
    return '\n'.join(lines)


def fix_sys_path_setup(content: str) -> str:
    """Add proper sys.path setup if missing."""
    lines = content.split('\n')

    # Find insertion point (after imports, ideally after docstring)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('#!') or line.strip().startswith('"""') or line.strip().startswith("'''"):
            continue
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_idx = i + 1

    path_setup = '''# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)
'''

    # Check if already has path setup
    if 'sys.path.insert' in content:
        return content

    # Insert at found position
    if insert_idx < len(lines):
        lines.insert(insert_idx, path_setup)
    else:
        lines.append(path_setup)

    return '\n'.join(lines)


def fix_script_utils_import(content: str) -> str:
    """Add script_utils import if missing."""
    # Check if already has script_utils import
    if 'from script_utils import' in content:
        return content

    # Find where to insert - after path setup block or after imports
    lines = content.split('\n')

    # Find insertion point: after os.chdir(project_root) line, or after last import
    insert_idx = 0
    for i, line in enumerate(lines):
        if 'os.chdir(project_root)' in line:
            # Insert after this line (the path setup block)
            insert_idx = i + 1
            # Skip any empty lines
            while insert_idx < len(lines) and not lines[insert_idx].strip():
                insert_idx += 1
            break

    if insert_idx == 0:
        # No path setup found, insert after existing imports
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1

    import_section = '''# Import standardized output functions
from script_utils import (
    print_header, print_ok, print_warn, print_error, print_info,
    set_verbosity
)
'''

    if insert_idx < len(lines):
        lines.insert(insert_idx, import_section)
    else:
        lines.append(import_section)

    return '\n'.join(lines)


def fix_chdir(content: str) -> str:
    """Add os.chdir(project_root) if missing."""
    if 'os.chdir(project_root)' in content:
        return content

    # Already handled by fix_sys_path_setup
    return content


def fix_argparse_import(content: str) -> str:
    """Add argparse import if missing."""
    # Check if already has argparse import
    if re.search(r'import\s+argparse', content) or re.search(r'from\s+argparse\s+import', content):
        return content

    lines = content.split('\n')

    # Find insertion point - after existing standard imports, before script_utils import
    insert_idx = 0
    found_script_utils = False
    for i, line in enumerate(lines):
        if 'from script_utils import' in line:
            found_script_utils = True
            break
        if line.startswith('import ') or line.startswith('from '):
            insert_idx = i + 1

    # If we found script_utils import first, insert before it
    if found_script_utils:
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#') and not line.startswith('\"\"\"'):
                insert_idx = i
                break

    import_line = 'import argparse'

    if insert_idx < len(lines):
        lines.insert(insert_idx, import_line)
    else:
        lines.append(import_line)

    return '\n'.join(lines)


def check_docstring(content: str) -> Tuple[bool, str]:
    """Check if script has module-level docstring."""
    pattern = r'(?:^#!.*\n)?\s*("""|\'\'\')[\s\S]*?\1'
    match = re.search(pattern, content, re.MULTILINE)
    if match:
        return True, ""
    return False, "Missing module-level docstring"


def fix_docstring(content: str) -> str:
    """Add basic docstring template if missing."""
    # Find where to insert (after shebang if present)
    lines = content.split('\n')
    insert_idx = 0
    if lines and lines[0].startswith('#!'):
        insert_idx = 1
        # Skip empty lines
        while insert_idx < len(lines) and not lines[insert_idx].strip():
            insert_idx += 1

    docstring = '"""Script description.\n"""'

    # Insert docstring
    if insert_idx < len(lines) and lines[insert_idx].strip():
        lines.insert(insert_idx, docstring)
    elif insert_idx < len(lines):
        lines[insert_idx] = docstring
    else:
        lines.append(docstring)

    return '\n'.join(lines)


def check_script_utils_import(content: str) -> Tuple[bool, str]:
    """Check if script imports from script_utils."""
    # Check for proper import pattern
    patterns = [
        r'from\s+script_utils\s+import\s+',
        r'from\s+script_utils\s+import\s*\(',
    ]
    for pattern in patterns:
        if re.search(pattern, content):
            return True, ""
    return False, "Missing import from script_utils"


def check_sys_path_setup(content: str) -> Tuple[bool, str]:
    """Check if script has proper sys.path setup."""
    # Look for standard path setup pattern
    patterns = [
        r'_script_path\s*=\s*os\.path\.abspath\(__file__\)',
        r'project_root\s*=\s*Path\(_script_path\)\.parent\.parent',
        r'sys\.path\.insert\(0,\s*str\(project_root\)\)',
    ]
    found = 0
    for pattern in patterns:
        if re.search(pattern, content):
            found += 1

    if found >= 2:
        return True, ""
    elif found > 0:
        return False, f"Incomplete sys.path setup (found {found}/3 patterns)"
    return False, "Missing sys.path setup"


def check_chdir(content: str) -> Tuple[bool, str]:
    """Check if script changes to project root directory."""
    patterns = [
        r'os\.chdir\(project_root\)',
        r'os\.chdir\(_script_path\.parent\.parent\)',
    ]
    for pattern in patterns:
        if re.search(pattern, content):
            return True, ""
    return False, "Missing os.chdir(project_root)"


def check_argparse_import(content: str) -> Tuple[bool, str]:
    """Check if script has argparse import."""
    patterns = [
        r'import\s+argparse',
        r'from\s+argparse\s+import',
    ]
    for pattern in patterns:
        if re.search(pattern, content):
            return True, ""
    return False, "Missing argparse import"


def validate_script(script_path: Path, content: str) -> ValidationResult:
    """Validate a single script against standards."""
    result = ValidationResult(script_path=str(script_path))

    # Skip utils modules
    if script_path.name in ('script_utils.py', 'cli_helpers.py'):
        result.is_utils = True
        result.passed = True
        result.checks = {'utils_module': True}
        return result

    # Run checks
    checks = {
        'shebang': check_shebang(content),
        'docstring': check_docstring(content),
        'script_utils_import': check_script_utils_import(content),
        'sys_path_setup': check_sys_path_setup(content),
        'chdir': check_chdir(content),
        'argparse_import': check_argparse_import(content),
    }

    result.checks = {k: v[0] for k, v in checks.items()}

    for check_name, (passed, error) in checks.items():
        if not passed:
            result.errors.append(f"{check_name}: {error}")

    result.passed = len(result.errors) == 0

    return result


def fix_script(script_path: Path, dry_run: bool = False) -> Tuple[ValidationResult, str]:
    """Apply fixes to a script that doesn't meet standards.

    Returns tuple of (ValidationResult, modified_content).
    """
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    result = validate_script(script_path, content)
    if result.is_utils:
        return result, content

    original_content = content
    fixes = []

    # Apply fixes in order - order matters for dependencies
    # 1. Shebang first
    if not result.checks.get('shebang', True):
        content = fix_shebang(content)
        fixes.append('shebang')

    # 2. Docstring
    if not result.checks.get('docstring', True):
        content = fix_docstring(content)
        fixes.append('docstring')

    # 3. sys.path setup (must come before script_utils import)
    if not result.checks.get('sys_path_setup', True):
        content = fix_sys_path_setup(content)
        fixes.append('sys_path_setup')

    # 4. script_utils import (needs chdir in place)
    if not result.checks.get('script_utils_import', True):
        content = fix_script_utils_import(content)
        fixes.append('script_utils_import')

    # 5. chdir is handled by sys_path_setup

    # 6. argparse import
    if not result.checks.get('argparse_import', True):
        content = fix_argparse_import(content)
        fixes.append('argparse_import')

    result.fixes_applied = fixes

    # Write fixed content if changes were made and not dry run
    if content != original_content and not dry_run:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # Re-validate after fixes (preserve fixes_applied)
    new_result = validate_script(script_path, content)
    new_result.fixes_applied = fixes
    new_result.is_utils = result.is_utils

    return new_result, content


def get_diff(script_path: Path) -> Tuple[str, str]:
    """Get diff of proposed changes without applying them.

    Returns tuple of (original_content, modified_content).
    """
    with open(script_path, 'r', encoding='utf-8') as f:
        original_content = f.read()

    result, modified_content = fix_script(script_path, dry_run=True)

    return original_content, modified_content


def get_scripts_to_validate() -> List[Path]:
    """Get list of Python scripts to validate."""
    scripts = []

    # Main scripts directory
    for path in scripts_dir.glob('*.py'):
        if path.name != '__init__.py':
            scripts.append(path)

    # Utils scripts
    utils_dir = scripts_dir / 'utils'
    if utils_dir.exists():
        for path in utils_dir.glob('*.py'):
            if path.name != '__init__.py':
                scripts.append(path)

    return sorted(scripts)


def run_validation(
    scripts: List[Path],
    verbose: bool = False,
    fix: bool = False,
    diff: bool = False,
    pre_commit: bool = False
) -> Dict[str, Any]:
    """Run validation on all scripts."""
    results = []
    passed_count = 0
    failed_count = 0
    fixed_count = 0
    diffs: Dict[str, Dict[str, str]] = {}

    for script_path in scripts:
        if verbose:
            print_info(f"Checking {script_path.name}...")

        if diff:
            # Show diff without applying
            original, modified = get_diff(script_path)
            if original != modified:
                diffs[str(script_path)] = {
                    'original': original,
                    'modified': modified
                }
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            result = validate_script(script_path, content)
        elif fix:
            result, _ = fix_script(script_path, dry_run=False)
            if result.fixes_applied:
                fixed_count += 1
                print_ok(f"Fixed {script_path.name}: {', '.join(result.fixes_applied)}")
        else:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            result = validate_script(script_path, content)

        results.append(asdict(result))

        if result.passed:
            passed_count += 1
            if verbose:
                print_ok(f"{script_path.name}: PASSED")
        else:
            failed_count += 1
            if verbose:
                print_warn(f"{script_path.name}: FAILED")
                for error in result.errors:
                    print_info(f"  - {error}")

    # Summary
    summary = {
        'total': len(scripts),
        'passed': passed_count,
        'failed': failed_count,
        'fixed': fixed_count,
        'results': results
    }

    if diffs:
        summary['diffs'] = diffs

    return summary


def print_summary(summary: Dict[str, Any], verbose: bool = False):
    """Print validation summary."""
    print_header("Script Standards Validation")

    print_ok(f"Total scripts: {summary['total']}")
    print_ok(f"Passed: {summary['passed']}")

    if summary['failed'] > 0:
        print_warn(f"Failed: {summary['failed']}")

    if summary['fixed'] > 0:
        print_ok(f"Fixed: {summary['fixed']}")

    if summary['failed'] > 0:
        print_header("Failed Scripts")
        for result in summary['results']:
            if not result['passed'] and not result['is_utils']:
                print_warn(f"  {Path(result['script_path']).name}")
                for error in result['errors']:
                    print_info(f"    - {error}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate script standards compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_script_standards.py
  python scripts/validate_script_standards.py --verbose
  python scripts/validate_script_standards.py --fix
  python scripts/validate_script_standards.py --pre-commit
  python scripts/validate_script_standards.py --json
        """
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--fix', '-f',
        action='store_true',
        help='Automatically fix issues where possible'
    )

    parser.add_argument(
        '--diff', '-d',
        action='store_true',
        help='Show diff of proposed changes without applying them'
    )

    parser.add_argument(
        '--pre-commit',
        action='store_true',
        help='Run in pre-commit mode (exit code only, quiet)'
    )

    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )

    parser.add_argument(
        '--scripts',
        help='Comma-separated list of specific scripts to validate (default: all)'
    )

    args = parser.parse_args()

    # Set verbosity
    if args.pre_commit:
        set_verbosity(0)
    elif args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    # Get scripts to validate
    if args.scripts:
        scripts = [scripts_dir / s for s in args.scripts.split(',')]
    else:
        scripts = get_scripts_to_validate()

    # Run validation
    summary = run_validation(
        scripts,
        verbose=args.verbose or not args.pre_commit,
        fix=args.fix,
        diff=args.diff,
        pre_commit=args.pre_commit
    )

    # Output
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary, verbose=args.verbose)

    # Exit code
    if summary['failed'] > 0 and not args.fix:
        sys.exit(1)
    elif summary['fixed'] > 0:
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
