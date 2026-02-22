#!/usr/bin/env python3
"""
Script Health Validation Tool

Validates that all scripts in the scripts/ directory follow the project's
coding standards including:
- Proper shebang line
- Docstring with usage instructions
- Uses script_utils for output (not custom logging)
- Has argparse with --help support

Usage:
    python scripts/check_script_health.py
    python scripts/check_script_health.py --verbose
    python scripts/check_script_health.py --fix
"""

import os
import re
import sys
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Import standardized output functions
from script_utils import print_header, print_ok, print_warn, print_error, print_info, set_verbosity


@dataclass
class ScriptHealthResult:
    """Results for a single script's health check."""
    script_path: Path
    is_utils: bool = False
    has_shebang: bool = False
    has_docstring: bool = False
    has_usage_in_docstring: bool = False
    uses_script_utils: bool = False
    has_argparse: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        """Calculate health score (0-100)."""
        score = 0
        if self.has_shebang or self.is_utils:
            score += 20
        if self.has_docstring:
            score += 20
        if self.has_usage_in_docstring:
            score += 20
        if self.uses_script_utils or self.is_utils:
            score += 25
        if self.has_argparse or self.is_utils:
            score += 15
        return score

    @property
    def is_healthy(self) -> bool:
        """Check if script passes all health checks."""
        if self.is_utils:
            # Utility modules only require docstring
            return self.has_docstring and self.has_usage_in_docstring
        return (self.has_shebang and self.has_docstring and
                self.has_usage_in_docstring and self.uses_script_utils and
                self.has_argparse)


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


def check_docstring(content: str) -> Tuple[bool, bool, str]:
    """Check if script has docstring with usage instructions."""
    # Look for module-level docstring (after any shebang)
    # Match first triple-quoted string anywhere in file
    pattern = r'(?:^#!.*\n)?\s*("""|\'\'\')[\s\S]*?\1'
    match = re.search(pattern, content, re.MULTILINE)

    if not match:
        return False, False, "Missing module-level docstring"

    docstring = match.group(0)

    # Check for usage instructions
    has_usage = 'Usage:' in docstring or 'usage:' in docstring

    return True, has_usage, ""


def check_script_utils(content: str) -> Tuple[bool, str]:
    """Check if script uses script_utils for output."""
    # Check for imports from script_utils
    import_pattern = r'from\s+script_utils\s+import\s+'
    if re.search(import_pattern, content):
        return True, ""

    # Check for direct script_utils imports
    direct_import = r'import\s+script_utils'
    if re.search(direct_import, content):
        return True, ""

    # Check for common custom logging patterns that should be avoided
    custom_logging_patterns = [
        r'logging\.(info|debug|warning|error)',
        r'logger\.(info|debug|warning|error)',
    ]

    has_custom_logging = any(re.search(p, content) for p in custom_logging_patterns)

    if has_custom_logging:
        return False, "Uses custom logging instead of script_utils"
    # Plain print is acceptable - not considered custom logging
    return False, "Does not import from script_utils"


def check_argparse(content: str) -> Tuple[bool, str]:
    """Check if script has argparse with --help support."""
    # Check for argparse import
    has_argparse_import = re.search(r'import\s+argparse', content)
    if not has_argparse_import:
        return False, "Missing argparse import"

    # Check for ArgumentParser usage
    has_parser = re.search(r'ArgumentParser', content)
    if not has_parser:
        return False, "Missing ArgumentParser"

    # Check for add_argument
    has_args = re.search(r'add_argument', content)
    if not has_args:
        return False, "Missing add_argument calls"

    return True, ""


def is_utils_module(script_path: Path) -> bool:
    """Check if a script is in the utils subdirectory."""
    return 'utils' in script_path.parts


def fix_shebang(content: str) -> Tuple[bool, str, str]:
    """Add shebang line if missing. Returns (fixed, new_content, message)."""
    lines = content.split('\n')
    if not lines:
        return False, content, "Empty file"

    first_line = lines[0].strip()
    if first_line.startswith('#!'):
        # Shebang already exists
        return False, content, ""

    # Add shebang at the beginning
    new_content = "#!/usr/bin/env python3\n\n" + content
    return True, new_content, "Added shebang: #!/usr/bin/env python3"


def fix_docstring(content: str, script_name: str) -> Tuple[bool, str, str]:
    """Add basic docstring template if missing. Returns (fixed, new_content, message)."""
    # Check if docstring already exists
    pattern = r'(?:^#!.*\n)?\s*("""|\'\'\')[\s\S]*?\1'
    if re.search(pattern, content, re.MULTILINE):
        return False, content, "Docstring already exists"

    # Create a basic docstring template
    docstring = f'''"""
{script_name}

Description of what this script does.

Usage:
    python {script_name} [options]
"""

'''

    # Add docstring after shebang if present, otherwise at the beginning
    lines = content.split('\n')
    if lines and lines[0].strip().startswith('#!'):
        # Insert after shebang line (and any blank lines after it)
        insert_idx = 1
        while insert_idx < len(lines) and not lines[insert_idx].strip():
            insert_idx += 1
        new_lines = lines[:insert_idx] + [docstring] + lines[insert_idx:]
        new_content = '\n'.join(new_lines)
    else:
        new_content = docstring + content

    return True, new_content, "Added basic docstring template"


def fix_argparse(content: str) -> Tuple[bool, str, str]:
    """Add argparse import if missing. Returns (fixed, new_content, message)."""
    # Check if argparse is already imported
    if re.search(r'import\s+argparse', content):
        return False, content, "argparse already imported"

    # Find a good place to add the import
    # Look for other imports to maintain ordering
    import_lines = []
    other_lines = []
    lines = content.split('\n')
    in_import_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            in_import_block = True
            import_lines.append(line)
        elif in_import_block and not stripped:
            # Skip blank lines in import block
            import_lines.append(line)
        elif in_import_block and not (stripped.startswith('import ') or stripped.startswith('from ')):
            in_import_block = False
            other_lines.append(line)
        else:
            other_lines.append(line)

    # Insert argparse import with other imports
    new_import_lines = import_lines + ['', 'import argparse']
    new_content = '\n'.join(new_import_lines + other_lines)

    return True, new_content, "Added argparse import"


def can_auto_fix(result: ScriptHealthResult) -> List[str]:
    """Determine which issues can be auto-fixed."""
    fixes = []
    if not result.has_shebang and not result.is_utils:
        fixes.append("shebang")
    if not result.has_docstring:
        fixes.append("docstring")
    if not result.has_argparse and not result.is_utils:
        fixes.append("argparse_import")
    return fixes


def apply_fixes(script_path: Path, dry_run: bool = False) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Apply auto-fixes to a script. Returns (success, fixed_items, failed_items, warnings).
    """
    fixed_items = []
    failed_items = []
    warnings = []

    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [], [f"Failed to read file: {e}"], []

    original_content = content

    # Check what can be fixed
    is_utils = is_utils_module(script_path)

    # Build new content with fixes in correct order
    new_lines = []

    # 1. Add shebang first (if needed, not for utils)
    lines = content.split('\n')
    first_line = lines[0].strip() if lines else ""

    if not is_utils and not first_line.startswith('#!'):
        new_lines.append("#!/usr/bin/env python3")
        new_lines.append("")
        fixed_items.append("Added shebang: #!/usr/bin/env python3")

    # 2. Add docstring after shebang (or at start)
    has_docstring = bool(re.search(r'(?:^#!.*\n)?\s*("""|\'\'\')[\s\S]*?\1', content, re.MULTILINE))

    if not has_docstring:
        docstring = f'''"""
{script_path.name}

Description of what this script does.

Usage:
    python {script_path.name} [options]
"""
'''
        new_lines.append(docstring)
        fixed_items.append("Added basic docstring template")

    # 3. Add argparse import with other imports
    has_argparse = bool(re.search(r'import\s+argparse', content))

    if not is_utils and not has_argparse:
        # Collect existing imports
        import_section = []
        code_section = []
        in_imports = True

        for line in lines:
            stripped = line.strip()
            if in_imports and (stripped.startswith('import ') or stripped.startswith('from ')):
                import_section.append(line)
            elif in_imports and stripped and not stripped.startswith('#'):
                in_imports = False
                if import_section:
                    import_section.append('')
                import_section.append('import argparse')
                import_section.append('')
                code_section.append(line)
            else:
                code_section.append(line)

        if import_section and 'import argparse' not in '\n'.join(import_section):
            # Add argparse to import section
            import_section.append('import argparse')
            fixed_items.append("Added argparse import")

        # Rebuild content
        result = '\n'.join(new_lines) if new_lines else ''
        if import_section:
            result += '\n'.join(import_section) + '\n'
        if code_section:
            result += '\n'.join(code_section)
        content = result
    else:
        # Just add to new_lines if we have any
        if new_lines:
            content = '\n'.join(new_lines) + '\n' + content

    # Write changes if not dry run
    if not dry_run and content != original_content:
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            return False, fixed_items, [f"Failed to write file: {e}"], warnings

    return True, fixed_items, failed_items, warnings


def check_script(script_path: Path, verbose: bool = False) -> ScriptHealthResult:
    """Check a single script's health."""
    is_utils = is_utils_module(script_path)
    result = ScriptHealthResult(script_path=script_path, is_utils=is_utils)

    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        result.errors.append(f"Failed to read file: {e}")
        return result

    # Check shebang (optional for utility modules)
    result.has_shebang, shebang_msg = check_shebang(content)
    if not result.has_shebang:
        if is_utils:
            result.warnings.append("Shebang recommended for utility modules")
        else:
            result.errors.append(shebang_msg)

    # Check docstring - required for all
    result.has_docstring, result.has_usage_in_docstring, docstring_msg = check_docstring(content)
    if not result.has_docstring:
        result.errors.append(docstring_msg)
    elif not result.has_usage_in_docstring:
        result.warnings.append("Docstring missing 'Usage:' instructions")

    # Check script_utils (not required for utility modules - they're libraries)
    result.uses_script_utils, utils_msg = check_script_utils(content)
    if not result.uses_script_utils:
        if is_utils:
            result.warnings.append("script_utils not required for utility modules")
        else:
            result.errors.append(utils_msg)

    # Check argparse (not required for utility modules - they're libraries)
    result.has_argparse, argparse_msg = check_argparse(content)
    if not result.has_argparse:
        if is_utils:
            result.warnings.append("argparse not required for utility modules")
        else:
            result.errors.append(argparse_msg)

    if verbose:
        for warning in result.warnings:
            print_warn(f"  {script_path.name}: {warning}")

    return result


def get_scripts_list(scripts_dir: Path, include_utils: bool = False) -> List[Path]:
    """
    Get list of Python scripts to check (excluding __init__.py and this script).

    Args:
        scripts_dir: Directory containing scripts to check
        include_utils: If True, also include scripts/utils/*.py files
    """
    scripts = []
    for f in scripts_dir.glob('*.py'):
        # Skip __init__.py and this script
        if f.name == '__init__.py':
            continue
        if f.name == 'check_script_health.py':
            continue
        # Skip utility subdirectories
        if f.name == 'script_utils.py':
            continue
        if f.name.startswith('utils_'):
            continue
        scripts.append(f)

    # Include utils directory if requested
    if include_utils:
        utils_dir = scripts_dir / 'utils'
        if utils_dir.exists():
            for f in utils_dir.glob('*.py'):
                # Skip __init__.py but include other utility modules
                if f.name == '__init__.py':
                    continue
                scripts.append(f)

    return sorted(scripts, key=lambda p: (str(p.parent), p.name))


def print_result(result: ScriptHealthResult, verbose: bool = False) -> None:
    """Print health check result for a script."""
    if result.is_healthy:
        print_ok(f"  {result.script_path.name}: {result.score}/100")
    else:
        print_warn(f"  {result.script_path.name}: {result.score}/100")

    if verbose:
        for error in result.errors:
            print_error(f"    - {error}", exit_code=None)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate script health and coding standards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/check_script_health.py
    python scripts/check_script_health.py --verbose
    python scripts/check_script_health.py --json
    python scripts/check_script_health.py --include-utils
    python scripts/check_script_health.py --fix
    python scripts/check_script_health.py --fix --dry-run
        """
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--scripts-dir', '-s', type=Path,
                        default=scripts_dir,
                        help='Directory containing scripts to check')
    parser.add_argument('--include-utils', '-u', action='store_true',
                        help='Also check scripts/utils/*.py utility modules')
    parser.add_argument('--fix', '-f', action='store_true',
                        help='Automatically apply fixes where possible')
    parser.add_argument('--dry-run', '-d', action='store_true',
                        help='Preview fixes without applying (used with --fix)')

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    print_header("Script Health Check")

    scripts = get_scripts_list(args.scripts_dir, include_utils=args.include_utils)

    if not scripts:
        print_warn("No scripts found to check")
        return 1

    print_info(f"Checking {len(scripts)} scripts...")

    results: List[ScriptHealthResult] = []
    for script in scripts:
        result = check_script(script, verbose=args.verbose)
        results.append(result)

    # Handle --fix mode
    if args.fix:
        print()
        if args.dry_run:
            print_header("Dry Run - Fixes to Apply")
            print_info("No changes will be made (--dry-run mode)")
        else:
            print_header("Applying Fixes")

        total_fixed = 0
        total_failed = 0

        for result in results:
            if result.is_healthy:
                continue  # Skip healthy scripts

            can_fix = can_auto_fix(result)
            if not can_fix:
                continue  # No auto-fixable issues

            print_info(f"Processing: {result.script_path.name}")

            success, fixed_items, failed_items, warnings = apply_fixes(
                result.script_path, dry_run=args.dry_run
            )

            if fixed_items:
                total_fixed += 1
                for item in fixed_items:
                    if args.dry_run:
                        print(f"    [WOULD FIX] {item}")
                    else:
                        print(f"    [FIXED] {item}")

            if warnings:
                for warning in warnings:
                    print_warn(f"    {warning}")

            if failed_items:
                total_failed += 1
                for item in failed_items:
                    print_error(f"    [FAILED] {item}")

        print()
        if args.dry_run:
            print(f"  Would fix: {total_fixed} script(s)")
            print(f"  Manual intervention needed: {total_failed} script(s)")
        else:
            print(f"  Fixed: {total_fixed} script(s)")
            print(f"  Failed: {total_failed} script(s)")

        # Re-check after fixes
        print()
        print_info("Re-checking scripts after fixes...")

        results = []
        for script in scripts:
            result = check_script(script, verbose=args.verbose)
            results.append(result)

    # Calculate overall stats
    total_score = sum(r.score for r in results)
    overall_score = int(total_score / len(results)) if results else 0
    healthy_count = sum(1 for r in results if r.is_healthy)
    unhealthy_count = len(results) - healthy_count

    if args.json:
        import json
        output = {
            "overall_score": overall_score,
            "total_scripts": len(scripts),
            "healthy_scripts": healthy_count,
            "unhealthy_scripts": unhealthy_count,
            "scripts": [
                {
                    "name": r.script_path.name,
                    "path": str(r.script_path),
                    "is_utils": r.is_utils,
                    "score": r.score,
                    "is_healthy": r.is_healthy,
                    "has_shebang": r.has_shebang,
                    "has_docstring": r.has_docstring,
                    "has_usage_in_docstring": r.has_usage_in_docstring,
                    "uses_script_utils": r.uses_script_utils,
                    "has_argparse": r.has_argparse,
                    "errors": r.errors,
                    "warnings": r.warnings,
                }
                for r in results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print()
        for result in results:
            print_result(result, verbose=args.verbose)

        print()
        print_header("Summary")
        print(f"  Overall Health Score: {overall_score}/100")
        print(f"  Healthy Scripts: {healthy_count}/{len(results)}")
        print(f"  Unhealthy Scripts: {unhealthy_count}/{len(results)}")

        if unhealthy_count > 0:
            print()
            print_warn(f"Issues found in {unhealthy_count} script(s):")
            for result in results:
                if not result.is_healthy:
                    print(f"  - {result.script_path.name}")
                    for error in result.errors:
                        print(f"      {error}")

    return 0 if unhealthy_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
