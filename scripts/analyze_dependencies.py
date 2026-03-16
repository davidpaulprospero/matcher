#!/usr/bin/env python3
"""
Script Dependency Analyzer

Analyzes Python scripts to build a dependency graph based on import statements.
Detects circular dependencies and outputs the graph in DOT format for visualization.

Usage:
    # Analyze all scripts in scripts/ directory
    python scripts/analyze_dependencies.py

    # Analyze specific directory
    python scripts/analyze_dependencies.py --path scripts

    # Output as DOT file for Graphviz
    python scripts/analyze_dependencies.py --output dependencies.dot

    # Show detailed output with circular dependencies highlighted
    python scripts/analyze_dependencies.py --verbose

    # JSON output for programmatic use
    python scripts/analyze_dependencies.py --json

    # Check for circular dependencies only
    python scripts/analyze_dependencies.py --check-cycles
"""

import argparse
import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def find_python_files(directory: str) -> list[str]:
    """Find all Python files in the given directory."""
    python_files = []
    dir_path = Path(directory)

    if dir_path.is_file():
        return [str(dir_path)]

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return sorted(python_files)


def extract_imports(file_path: str) -> list[str]:
    """Extract import statements from a Python file."""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError):
        return imports

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                if module:
                    imports.append(f"{module}.{alias.name}")
                else:
                    imports.append(alias.name)

    return imports


def filter_local_imports(imports: list[str], base_dir: str) -> list[str]:
    """Filter to only local imports (within the project)."""
    local_imports = []

    # Modules we consider as local (scripts directory)
    local_modules = {'script_utils', 'tools', 'batch_operations', 'script_utils.print_ok',
                     'script_utils.print_header', 'script_utils.print_warn', 'script_utils.print_error',
                     'script_utils.print_info', 'script_utils.set_verbosity', 'script_utils.get_verbosity',
                     'script_utils.progress_bar', 'script_utils.track_progress', 'script_utils.load_config_for_script',
                     'script_utils.add_config_argument'}

    for imp in imports:
        # Skip standard library and external packages
        if imp.startswith('.'):
            # Convert relative import to module path
            module_path = imp.lstrip('.')
            if module_path:
                local_imports.append(module_path)
        elif imp.startswith('scripts.'):
            # Absolute-style import from scripts package
            local_imports.append(imp[8:])  # Remove 'scripts.' prefix
        elif imp.startswith('src.'):
            # Import from src package
            local_imports.append(imp[4:])  # Remove 'src.' prefix
        elif imp in local_modules:
            # Direct imports from scripts directory (added to sys.path)
            local_imports.append(imp.split('.')[0])  # Get just the module name

    return local_imports


def build_dependency_graph(files: list[str], base_dir: str) -> dict[str, list[str]]:
    """Build a dependency graph from Python files."""
    graph = defaultdict(list)
    file_modules = {}

    # Create module name to file mapping
    for file_path in files:
        rel_path = os.path.relpath(file_path, base_dir)
        module_name = rel_path.replace('/', '.').replace('\\', '.').replace('.py', '')

        # Skip __init__ files for module naming
        if module_name.endswith('.__init__'):
            module_name = module_name[:-9]

        file_modules[file_path] = module_name

    # Build graph
    for file_path in files:
        imports = extract_imports(file_path)
        local_imports = filter_local_imports(imports, base_dir)

        # Map imports to file paths
        for imp in local_imports:
            # Try to find matching module
            for other_file, other_module in file_modules.items():
                if other_module == imp or other_module.endswith(f'.{imp}'):
                    if other_file != file_path:
                        module_name = file_modules[file_path]
                        if other_module not in graph[module_name]:
                            graph[module_name].append(other_module)

    return dict(graph)


def detect_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    """Detect circular dependencies using DFS."""
    cycles = []
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)

        path.pop()
        rec_stack.remove(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    return cycles


def generate_dot(graph: dict[str, list[str]], cycles: list[list[str]] = None) -> str:
    """Generate DOT format output for the dependency graph."""
    lines = [
        'digraph Dependencies {',
        '  rankdir=LR;',
        '  node [shape=box];',
        ''
    ]

    # Flatten cycles for highlighting
    cycle_nodes = set()
    if cycles:
        for cycle in cycles:
            for node in cycle[:-1]:  # Exclude duplicate end node
                cycle_nodes.add(node)

    # Add nodes
    all_nodes = set(graph.keys())
    for deps in graph.values():
        all_nodes.update(deps)

    for node in sorted(all_nodes):
        if node in cycle_nodes:
            lines.append(f'  "{node}" [style=filled, fillcolor=lightcoral];')
        else:
            lines.append(f'  "{node}";')

    lines.append('')

    # Add edges
    for node, deps in sorted(graph.items()):
        for dep in sorted(deps):
            lines.append(f'  "{node}" -> "{dep}";')

    lines.append('}')
    return '\n'.join(lines)


def format_cycles(cycles: list[list[str]]) -> str:
    """Format circular dependencies for display."""
    if not cycles:
        return "No circular dependencies detected."

    lines = ["Circular dependencies detected:"]
    for i, cycle in enumerate(cycles, 1):
        cycle_str = " -> ".join(cycle)
        lines.append(f"  {i}. {cycle_str}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Python script dependencies and detect cycles"
    )
    parser.add_argument(
        '--path', '-p',
        default='scripts',
        help='Directory to analyze (default: scripts)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--format',
        choices=['dot', 'json', 'text'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output as JSON (shorthand for --format json)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed dependency information'
    )
    parser.add_argument(
        '--check-cycles', '-c',
        action='store_true',
        help='Only check for circular dependencies'
    )

    args = parser.parse_args()

    # Handle --json shorthand
    if args.json:
        args.format = 'json'

    # Find Python files
    files = find_python_files(args.path)
    if not files:
        print(f"No Python files found in {args.path}", file=sys.stderr)
        sys.exit(1)

    # Build dependency graph
    base_dir = os.path.abspath(args.path)
    graph = build_dependency_graph(files, base_dir)

    # Detect cycles
    cycles = detect_cycles(graph)

    if args.check_cycles:
        print(format_cycles(cycles))
        sys.exit(1 if cycles else 0)

    # Build output based on format
    if args.format == 'dot':
        output = generate_dot(graph, cycles)
    elif args.format == 'json':
        output = json.dumps({
            'graph': graph,
            'cycles': cycles,
            'files': files
        }, indent=2)
    else:  # text
        lines = [f"Analyzed {len(files)} files"]
        lines.append(f"Found {len(graph)} dependencies")

        if cycles:
            lines.append("")
            lines.append(format_cycles(cycles))

        if args.verbose:
            lines.append("")
            lines.append("Dependency details:")
            for node, deps in sorted(graph.items()):
                if deps:
                    lines.append(f"  {node} -> {', '.join(deps)}")

        output = '\n'.join(lines)

    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Output written to {args.output}")
    else:
        print(output)

    # Exit with error if cycles found
    sys.exit(1 if cycles else 0)


if __name__ == '__main__':
    main()
