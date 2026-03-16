#!/usr/bin/env python3
"""
Tools CLI - Unified entry point for all project scripts.

Provides a git-like subcommand interface for managing and running
utility scripts in the project.

Usage:
    tools.py list                    # List all available scripts
    tools.py run <script>            # Run a script by name
    tools.py docs <script>           # Show script documentation
    tools.py validate                # Validate all scripts
    tools.py grep <pattern>          # Search in script files
    tools.py --help                  # Show this help

Examples:
    tools.py list
    tools.py run cleanup_project --project "E:\\Projects\\MyDoc"
    tools.py docs validate_config
    tools.py grep "def main" --context 3
    tools.py grep "import.*os" --ignore-case
"""

import argparse
import os
import re
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
_script_path = os.path.abspath(__file__)
scripts_dir = Path(_script_path).parent
project_root = scripts_dir.parent
sys.path.insert(0, str(project_root))

# Import standardized output functions
from script_utils import print_ok, print_warn, print_error, print_info, print_header


# Script metadata extraction
def get_script_metadata(script_path: Path) -> Dict:
    """
    Extract metadata from a script's docstring.

    Args:
        script_path: Path to the script file

    Returns:
        Dict with 'name', 'description', 'usage', 'category'
    """
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return {
            'name': script_path.stem,
            'description': 'Unable to read script',
            'usage': '',
            'category': 'unknown'
        }

    # Parse docstring
    lines = content.split('\n')
    description = ''
    usage = ''
    category = 'general'

    in_docstring = False
    docstring_lines = []

    for i, line in enumerate(lines):
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            if not in_docstring:
                in_docstring = True
                # Skip the opening quotes
                stripped = line.strip()[3:].strip()
                if stripped:
                    docstring_lines.append(stripped)
            else:
                in_docstring = False
                break
        elif in_docstring:
            docstring_lines.append(line)

    if docstring_lines:
        # First non-empty line is description
        for dl in docstring_lines:
            if dl.strip():
                description = dl.strip()
                break

        # Look for Usage: section
        for j, dl in enumerate(docstring_lines):
            if dl.strip().lower().startswith('usage:'):
                # Collect usage lines
                usage_lines = []
                for k in range(j + 1, len(docstring_lines)):
                    ul = docstring_lines[k].strip()
                    if ul and not ul.startswith('#'):
                        usage_lines.append(ul)
                    elif not ul:
                        break
                usage = '\n'.join(usage_lines[:3])  # Limit to 3 lines

    # Determine category from filename/path
    script_name = script_path.stem.lower()
    if 'test' in script_name or script_name.startswith('test_'):
        category = 'testing'
    elif 'config' in script_name:
        category = 'config'
    elif 'benchmark' in script_name or 'analyze' in script_name:
        category = 'analysis'
    elif 'coverage' in script_name:
        category = 'coverage'
    elif ' Ralph' in content or 'ralph' in script_name:
        category = 'ralph'
    else:
        category = 'general'

    return {
        'name': script_path.stem,
        'description': description,
        'usage': usage,
        'category': category,
        'path': str(script_path)
    }


def discover_scripts() -> Dict[str, Dict]:
    """
    Discover all scripts in the scripts directory.

    Returns:
        Dict mapping script names to their metadata
    """
    scripts = {}

    # Define which directories to scan
    scan_dirs = [
        scripts_dir,
        scripts_dir / 'utils'
    ]

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue

        for script_path in scan_dir.glob('*.py'):
            # Skip __init__ and this tool itself
            if script_path.stem.startswith('__') or script_path.name == 'tools.py':
                continue

            metadata = get_script_metadata(script_path)
            scripts[metadata['name']] = metadata

    return scripts


# Subcommand implementations
def cmd_list(scripts: Dict[str, Dict], args: argparse.Namespace) -> int:
    """List all available scripts."""
    # Filter by category if specified
    if args.category:
        filtered = {k: v for k, v in scripts.items() if v['category'] == args.category}
    else:
        filtered = scripts

    if args.json:
        import json
        print(json.dumps(filtered, indent=2))
        return 0

    # Group by category
    by_category: Dict[str, List[Dict]] = {}
    for name, meta in filtered.items():
        cat = meta['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(meta)

    # Print grouped list
    print("\nAvailable scripts:\n")
    for cat in sorted(by_category.keys()):
        print(f"  [{cat.upper()}]")
        for meta in sorted(by_category[cat], key=lambda x: x['name']):
            desc = meta['description'][:60] + '...' if len(meta['description']) > 60 else meta['description']
            print(f"    {meta['name']:<30} {desc}")
        print()

    return 0


def cmd_run(scripts: Dict[str, Dict], args: argparse.Namespace) -> int:
    """Run a script by name."""
    script_name = args.script

    # Find script (allow partial matches)
    matches = [k for k in scripts.keys() if k == script_name or k.startswith(script_name)]

    if not matches:
        print(f"Error: Script '{script_name}' not found", file=sys.stderr)
        print("\nRun 'tools.py list' to see available scripts.", file=sys.stderr)
        return 1

    if len(matches) > 1:
        print(f"Error: Ambiguous script name '{script_name}'. Did you mean:", file=sys.stderr)
        for m in matches:
            print(f"  - {m}", file=sys.stderr)
        return 1

    selected = matches[0]
    script_meta = scripts[selected]

    # Build command
    cmd = [sys.executable, script_meta['path']]

    # Pass through additional arguments
    if args.args:
        cmd.extend(args.args)

    # Run the script
    try:
        result = subprocess.run(cmd, cwd=str(project_root))
        return result.returncode
    except Exception as e:
        print(f"Error running script: {e}", file=sys.stderr)
        return 1


def cmd_docs(scripts: Dict[str, Dict], args: argparse.Namespace) -> int:
    """Show documentation for a script."""
    script_name = args.script

    # Find script
    matches = [k for k in scripts.keys() if k == script_name or k.startswith(script_name)]

    if not matches:
        print(f"Error: Script '{script_name}' not found", file=sys.stderr)
        return 1

    if len(matches) > 1:
        print(f"Error: Ambiguous script name. Matches: {', '.join(matches)}", file=sys.stderr)
        return 1

    selected = matches[0]
    meta = scripts[selected]

    # Read full docstring
    try:
        with open(meta['path'], 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading script: {e}", file=sys.stderr)
        return 1

    # Extract docstring
    docstring = ''
    in_docstring = False
    for line in content.split('\n'):
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            if not in_docstring:
                in_docstring = True
                stripped = line.strip()[3:].strip()
                if stripped:
                    docstring += stripped + '\n'
            else:
                break
        elif in_docstring:
            docstring += line + '\n'

    if args.json:
        import json
        print(json.dumps({
            'name': meta['name'],
            'description': meta['description'],
            'usage': meta['usage'],
            'category': meta['category'],
            'path': meta['path'],
            'docstring': docstring.strip()
        }, indent=2))
        return 0

    # Print formatted docs
    print(f"\n=== {meta['name']} ===\n")
    print(f"Category: {meta['category']}")
    print(f"Path: {meta['path']}\n")
    print(f"Description:\n  {meta['description']}\n")
    if meta['usage']:
        print(f"Usage:\n  {meta['usage']}\n")
    if docstring.strip():
        print(f"Full Documentation:\n{docstring.strip()}\n")

    return 0


def cmd_validate(scripts: Dict[str, Dict], args: argparse.Namespace) -> int:
    """Validate all scripts (syntax check)."""
    errors = []
    warnings = []

    for name, meta in scripts.items():
        path = Path(meta['path'])
        try:
            # Try to compile the script
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, str(path), 'exec')
        except SyntaxError as e:
            errors.append(f"{name}: Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            warnings.append(f"{name}: {type(e).__name__}: {e}")

    if args.json:
        import json
        print(json.dumps({
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'total_scripts': len(scripts)
        }, indent=2))
        return 0

    # Print results
    print(f"\nValidating {len(scripts)} scripts...\n")

    if errors:
        print("[ERRORS]")
        for e in errors:
            print(f"  {e}")
        print()

    if warnings:
        print("[WARNINGS]")
        for w in warnings:
            print(f"  {w}")
        print()

    if not errors and not warnings:
        print("[OK] All scripts validated successfully\n")

    return 1 if errors else 0


def cmd_grep(scripts: Dict[str, Dict], args: argparse.Namespace) -> int:
    """Search for a pattern in script files."""
    pattern = args.pattern
    context = args.context
    ignore_case = args.ignore_case
    regex = args.regex

    # Compile pattern
    flags = re.IGNORECASE if ignore_case else 0
    try:
        if regex:
            compiled_pattern = re.compile(pattern, flags)
        else:
            # Escape special regex chars for literal search
            compiled_pattern = re.compile(re.escape(pattern), flags)
    except re.error as e:
        print(f"Error: Invalid pattern: {e}", file=sys.stderr)
        return 1

    # Collect results
    matches: List[Tuple[str, Path, int, str, List[Tuple[int, str]]]] = []

    for name, meta in scripts.items():
        path = Path(meta['path'])
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            continue

        for line_num, line in enumerate(lines, start=1):
            if compiled_pattern.search(line):
                # Get context lines
                context_lines: List[Tuple[int, str]] = []
                if context > 0:
                    start = max(0, line_num - context - 1)
                    end = min(len(lines), line_num + context)
                    for ctx_line_num in range(start, end):
                        context_lines.append((ctx_line_num + 1, lines[ctx_line_num].rstrip('\n\r')))

                matches.append((name, path, line_num, line.rstrip('\n\r'), context_lines))

    if args.json:
        import json
        results = []
        for name, path, line_num, line, context_lines in matches:
            results.append({
                'script': name,
                'path': str(path),
                'line': line_num,
                'match': line,
                'context': [{'line': l, 'text': t} for l, t in context_lines] if context > 0 else []
            })
        print(json.dumps({
            'pattern': pattern,
            'ignore_case': ignore_case,
            'regex': regex,
            'matches': results,
            'total_matches': len(matches)
        }, indent=2))
        return 0

    # Print results
    if not matches:
        print(f"No matches found for pattern: {pattern}")
        return 1

    print(f"\nSearching for: {pattern}")
    if ignore_case:
        print("(case-insensitive)")
    print(f"Found {len(matches)} match(es) in {len(set(m[0] for m in matches))} file(s)\n")

    current_file = None
    for name, path, line_num, line, context_lines in matches:
        if current_file != name:
            current_file = name
            print(f"\n{name}:")

        if context > 0:
            for ctx_line_num, ctx_line in context_lines:
                prefix = ">" if ctx_line_num == line_num else " "
                print(f"  {prefix}{ctx_line_num:4d} │ {ctx_line}")
        else:
            print(f"  {line_num:4d} │ {line}")

    print()
    return 0


def compute_content_hash(content: str) -> str:
    """Compute a simple hash of script content for comparison."""
    import hashlib
    # Normalize whitespace for better comparison
    normalized = '\n'.join(line.rstrip() for line in content.split('\n'))
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:8]


def extract_code_signatures(scripts: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Extract code signatures (imports, function defs, class defs) from scripts.

    Returns:
        Dict mapping script name to list of code signatures
    """
    signatures: Dict[str, List[str]] = {}

    for name, meta in scripts.items():
        path = Path(meta['path'])
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        sigs = []

        # Extract imports
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                sigs.append(line.strip())

            # Extract function definitions
            if stripped.startswith('def ') or stripped.startswith('async def '):
                # Get function name
                match = re.match(r'(async def |def )(\w+)', stripped)
                if match:
                    sigs.append(f"def {match.group(2)}")

            # Extract class definitions
            if stripped.startswith('class '):
                match = re.match(r'class (\w+)', stripped)
                if match:
                    sigs.append(f"class {match.group(1)}")

        signatures[name] = sigs

    return signatures


def find_common_patterns(signatures: Dict[str, List[str]], min_occurrences: int = 2) -> Dict[str, List[str]]:
    """Find code patterns that appear in multiple scripts."""
    pattern_to_scripts: Dict[str, List[str]] = {}

    for script_name, sigs in signatures.items():
        for sig in sigs:
            if sig not in pattern_to_scripts:
                pattern_to_scripts[sig] = []
            pattern_to_scripts[sig].append(script_name)

    # Filter to only patterns appearing in multiple scripts
    common = {p: scripts for p, scripts in pattern_to_scripts.items() if len(scripts) >= min_occurrences}
    return common


def cmd_diff(scripts: Dict[str, Dict], args: argparse.Namespace) -> int:
    """Compare scripts to find similar code patterns."""
    import json
    import hashlib
    from collections import defaultdict

    # Filter by category if specified
    if args.category:
        filtered = {k: v for k, v in scripts.items() if v['category'] == args.category}
    else:
        filtered = scripts

    if len(filtered) < 2:
        print(f"Error: Need at least 2 scripts to compare (found {len(filtered)})", file=sys.stderr)
        if args.category:
            print(f"  Category '{args.category}' has {len(filtered)} script(s)", file=sys.stderr)
        return 1

    # Compute content hashes
    script_hashes: Dict[str, str] = {}
    for name, meta in filtered.items():
        path = Path(meta['path'])
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            script_hashes[name] = compute_content_hash(content)
        except Exception:
            continue

    # Group scripts by hash (exact duplicates)
    hash_groups: Dict[str, List[str]] = defaultdict(list)
    for name, h in script_hashes.items():
        hash_groups[h].append(name)

    # Extract code signatures for pattern comparison
    signatures = extract_code_signatures(filtered)
    common_patterns = find_common_patterns(signatures, min_occurrences=2)

    # Output as JSON if requested
    if args.json:
        output = {
            'by_hash': {
                h: scripts for h, scripts in hash_groups.items() if len(scripts) > 1
            },
            'common_patterns': {
                pattern: script_list
                for pattern, script_list in sorted(
                    common_patterns.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )[:50]  # Limit to top 50
            },
            'total_scripts_compared': len(filtered),
            'unique_hashes': len(hash_groups),
            'duplicate_groups': len([g for g in hash_groups.values() if len(g) > 1])
        }
        print(json.dumps(output, indent=2))
        return 0

    # Display results in table format
    print(f"\n=== Script Comparison ({len(filtered)} scripts) ===\n")

    # Show duplicate content
    duplicates = [g for g in hash_groups.values() if len(g) > 1]
    if duplicates:
        print("[DUPLICATE CONTENT]")
        print(f"{'Hash':<10} {'Scripts':<50}")
        print("-" * 60)
        for group in duplicates:
            h = script_hashes[group[0]]
            print(f"{h:<10} {', '.join(sorted(group))}")
        print()

    # Show common patterns
    if common_patterns:
        print(f"[COMMON PATTERNS] (showing top {min(20, len(common_patterns))})")
        print(f"{'Pattern':<40} {'Scripts':<30}")
        print("-" * 70)
        sorted_patterns = sorted(
            common_patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:20]

        for pattern, script_list in sorted_patterns:
            # Truncate long patterns
            display_pattern = pattern[:37] + "..." if len(pattern) > 40 else pattern
            display_scripts = ", ".join(sorted(script_list)[:3])
            if len(script_list) > 3:
                display_scripts += f" (+{len(script_list) - 3})"
            print(f"{display_pattern:<40} {display_scripts:<30}")
        print()

    # Show unique scripts summary
    unique_count = len([g for g in hash_groups.values() if len(g) == 1])
    print(f"[SUMMARY]")
    print(f"  Total scripts:     {len(filtered)}")
    print(f"  Unique content:    {unique_count}")
    print(f"  Duplicate groups:  {len(duplicates)}")
    print(f"  Common patterns:   {len(common_patterns)}")
    print()

    return 0


def generate_completions() -> str:
    """Generate shell completion script."""
    scripts = discover_scripts()
    script_names = sorted(scripts.keys())

    # Generate completion words
    words = ' '.join(script_names)

    return f'''# Completion for tools.py
# Source this file: source <(python tools.py --comp bash))

_{Path(__file__).stem}_cmds()
{{
    local cur prev
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"

    case $prev in
        run|docs)
            COMPREPLY=( $(compgen -W "{words}" -- "$cur") )
            return 0
            ;;
        --category)
            COMPREPLY=( $(compgen -W "general testing config analysis coverage ralph" -- "$cur") )
            return 0
            ;;
    esac

    # Main subcommands
    case $cur in
        list|run|docs|validate)
            COMPREPLY=( $(compgen -W "$cur" -- "") )
            return 0
            ;;
        -*)
            COMPREPLY=( $(compgen -W "--help --json --category --comp" -- "$cur") )
            return 0
            ;;
    esac

    COMPREPLY=( $(compgen -W "list run docs validate --help --json" -- "$cur") )
}}

complete -F _{Path(__file__).stem}_cmds python tools.py
'''


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add --comp flag for completion generation
    parser.add_argument(
        '--comp', '--completion',
        choices=['bash', 'zsh'],
        help='Generate shell completion script'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # list subcommand
    list_parser = subparsers.add_parser('list', help='List all available scripts')
    list_parser.add_argument(
        '--category', '-c',
        choices=['general', 'testing', 'config', 'analysis', 'coverage', 'ralph'],
        help='Filter by category'
    )
    list_parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output as JSON'
    )

    # run subcommand
    run_parser = subparsers.add_parser('run', help='Run a script by name')
    run_parser.add_argument(
        'script',
        help='Script name to run (partial match supported)'
    )
    run_parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Arguments to pass to the script'
    )

    # docs subcommand
    docs_parser = subparsers.add_parser('docs', help='Show script documentation')
    docs_parser.add_argument(
        'script',
        help='Script name (partial match supported)'
    )
    docs_parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output as JSON'
    )

    # validate subcommand
    validate_parser = subparsers.add_parser('validate', help='Validate all scripts')
    validate_parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output as JSON'
    )

    # grep subcommand
    grep_parser = subparsers.add_parser('grep', help='Search for a pattern in script files')
    grep_parser.add_argument(
        'pattern',
        help='Pattern to search for (literal or regex)'
    )
    grep_parser.add_argument(
        '--context', '-C',
        type=int,
        default=0,
        help='Show N lines of context around matches'
    )
    grep_parser.add_argument(
        '--ignore-case', '-i',
        action='store_true',
        help='Case-insensitive search'
    )
    grep_parser.add_argument(
        '--regex', '-E',
        action='store_true',
        help='Interpret pattern as regex (default: literal)'
    )
    grep_parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output as JSON'
    )

    # diff subcommand
    diff_parser = subparsers.add_parser('diff', help='Compare scripts to find similar code patterns')
    diff_parser.add_argument(
        '--category', '-c',
        choices=['general', 'testing', 'config', 'analysis', 'coverage', 'ralph'],
        help='Compare scripts within a specific category'
    )
    diff_parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output as JSON'
    )

    args = parser.parse_args()

    # Handle completion generation
    if args.comp:
        if args.comp == 'bash':
            print(generate_completions())
        elif args.comp == 'zsh':
            print(generate_completions().replace('bash', 'zsh'))
        return 0

    # Require subcommand
    if not args.command:
        parser.print_help()
        return 1

    # Discover scripts
    scripts = discover_scripts()

    # Route to subcommand handler
    if args.command == 'list':
        return cmd_list(scripts, args)
    elif args.command == 'run':
        return cmd_run(scripts, args)
    elif args.command == 'docs':
        return cmd_docs(scripts, args)
    elif args.command == 'validate':
        return cmd_validate(scripts, args)
    elif args.command == 'grep':
        return cmd_grep(scripts, args)
    elif args.command == 'diff':
        return cmd_diff(scripts, args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
