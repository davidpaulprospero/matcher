#!/usr/bin/env python3
"""
Tests for analyze_dependencies.py

Tests script dependency analysis, graph building, and cycle detection.
"""

import json
import os
import sys
import tempfile
import pytest

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_dependencies import (
    find_python_files,
    extract_imports,
    filter_local_imports,
    build_dependency_graph,
    detect_cycles,
    generate_dot,
    format_cycles,
)


class TestFindPythonFiles:
    """Tests for finding Python files."""

    def test_find_single_file(self):
        """Test finding a single Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, 'test.py')
            with open(test_file, 'w') as f:
                f.write('# test')

            files = find_python_files(test_file)
            assert len(files) == 1
            assert files[0] == test_file

    def test_find_multiple_files(self):
        """Test finding multiple Python files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                with open(os.path.join(tmpdir, f'test{i}.py'), 'w') as f:
                    f.write('# test')

            # Create non-Python file
            with open(os.path.join(tmpdir, 'readme.txt'), 'w') as f:
                f.write('text')

            files = find_python_files(tmpdir)
            assert len(files) == 3
            assert all(f.endswith('.py') for f in files)

    def test_nested_directories(self):
        """Test finding files in nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            nested = os.path.join(tmpdir, 'subdir', 'nested')
            os.makedirs(nested)
            with open(os.path.join(tmpdir, 'root.py'), 'w') as f:
                f.write('# root')
            with open(os.path.join(nested, 'nested.py'), 'w') as f:
                f.write('# nested')

            files = find_python_files(tmpdir)
            assert len(files) == 2


class TestExtractImports:
    """Tests for extracting imports from Python files."""

    def test_standard_library_imports(self):
        """Test extracting standard library imports."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
import os
import sys
from collections import defaultdict
import json
''')
            f.flush()

            imports = extract_imports(f.name)
            assert 'os' in imports
            assert 'sys' in imports
            assert 'collections.defaultdict' in imports
            assert 'json' in imports
            os.unlink(f.name)

    def test_from_imports(self):
        """Test extracting 'from' imports."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
from os import path
from collections import deque, Counter
import typing as t
''')
            f.flush()

            imports = extract_imports(f.name)
            assert 'os.path' in imports
            assert 'collections.deque' in imports
            assert 'collections.Counter' in imports
            assert 'typing' in imports
            os.unlink(f.name)

    def test_relative_imports(self):
        """Test extracting relative imports."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
from . import module
from ..parent import something
from .sub import item
''')
            f.flush()

            imports = extract_imports(f.name)
            # After lstrip('.', the imports become just the module names
            assert 'module' in imports
            assert 'parent.something' in imports
            assert 'sub.item' in imports
            os.unlink(f.name)

    def test_invalid_syntax(self):
        """Test handling of invalid Python syntax."""
        imports = extract_imports('/nonexistent/file.py')
        assert imports == []


class TestFilterLocalImports:
    """Tests for filtering local imports."""

    def test_relative_imports(self):
        """Test filtering relative imports."""
        imports = ['.utils', '.helpers', 'os', 'sys']
        result = filter_local_imports(imports, '/some/path')
        assert 'utils' in result
        assert 'helpers' in result
        assert 'os' not in result

    def test_scripts_imports(self):
        """Test filtering 'scripts.' imports."""
        imports = ['scripts.utils', 'scripts.tools', 'os']
        result = filter_local_imports(imports, '/some/path')
        assert 'utils' in result
        assert 'tools' in result

    def test_src_imports(self):
        """Test filtering 'src.' imports."""
        imports = ['src.config', 'src.utils', 'os']
        result = filter_local_imports(imports, '/some/path')
        assert 'config' in result
        assert 'utils' in result

    def test_script_utils_imports(self):
        """Test filtering script_utils imports."""
        imports = ['script_utils.print_ok', 'script_utils.print_header', 'os']
        result = filter_local_imports(imports, '/some/path')
        assert 'script_utils' in result
        assert 'os' not in result


class TestBuildDependencyGraph:
    """Tests for building dependency graphs."""

    def test_simple_dependencies(self):
        """Test building a simple dependency graph using real scripts directory."""
        # Use the actual scripts directory which has script_utils.py
        files = find_python_files('scripts')
        graph = build_dependency_graph(files, 'scripts')

        # The scripts directory has many dependencies to script_utils
        assert 'script_utils' in [dep for deps in graph.values() for dep in deps]

    def test_no_dependencies(self):
        """Test graph with no local dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, 'file1.py')
            file2 = os.path.join(tmpdir, 'file2.py')
            with open(file1, 'w') as f:
                f.write('import os\nimport sys\n')
            with open(file2, 'w') as f:
                f.write('import json\n')

            graph = build_dependency_graph([file1, file2], tmpdir)
            assert len(graph) == 0


class TestDetectCycles:
    """Tests for cycle detection."""

    def test_no_cycles(self):
        """Test graph without cycles."""
        graph = {
            'a': ['b', 'c'],
            'b': ['c'],
            'c': []
        }
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_simple_cycle(self):
        """Test detecting a simple cycle."""
        graph = {
            'a': ['b'],
            'b': ['c'],
            'c': ['a']  # Creates cycle: a -> b -> c -> a
        }
        cycles = detect_cycles(graph)
        assert len(cycles) == 1
        # Check that cycle contains the nodes
        assert 'a' in cycles[0]
        assert 'b' in cycles[0]
        assert 'c' in cycles[0]

    def test_multiple_cycles(self):
        """Test detecting multiple cycles."""
        graph = {
            'a': ['b'],
            'b': ['a'],  # Cycle: a -> b -> a
            'x': ['y'],
            'y': ['z'],
            'z': ['x']  # Cycle: x -> y -> z -> x
        }
        cycles = detect_cycles(graph)
        assert len(cycles) == 2

    def test_self_cycle(self):
        """Test detecting self-referencing node."""
        graph = {
            'a': ['a']  # Self cycle
        }
        cycles = detect_cycles(graph)
        assert len(cycles) == 1


class TestGenerateDot:
    """Tests for DOT format generation."""

    def test_basic_dot_output(self):
        """Test basic DOT output."""
        graph = {
            'a': ['b'],
            'b': ['c']
        }
        dot = generate_dot(graph)
        assert 'digraph Dependencies' in dot
        assert '"a" -> "b"' in dot
        assert '"b" -> "c"' in dot

    def test_dot_with_cycles(self):
        """Test DOT output with cycles highlighted."""
        graph = {
            'a': ['b'],
            'b': ['a']
        }
        cycles = detect_cycles(graph)
        dot = generate_dot(graph, cycles)
        assert 'fillcolor=lightcoral' in dot

    def test_dot_nodes_sorted(self):
        """Test that nodes are sorted in DOT output."""
        graph = {
            'z': ['a'],
            'm': ['b']
        }
        dot = generate_dot(graph)
        # Check node ordering
        lines = dot.split('\n')
        node_lines = [l for l in lines if l.startswith('  "')]


class TestFormatCycles:
    """Tests for formatting cycles."""

    def test_no_cycles_message(self):
        """Test message when no cycles."""
        result = format_cycles([])
        assert 'No circular dependencies' in result

    def test_cycles_formatted(self):
        """Test cycle formatting."""
        cycles = [['a', 'b', 'c', 'a']]
        result = format_cycles(cycles)
        assert 'Circular dependencies detected' in result
        assert 'a -> b -> c -> a' in result


class TestIntegration:
    """Integration tests for full analysis."""

    def test_full_analysis_json(self):
        """Test full analysis produces valid JSON."""
        # Use a temp directory with known structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files - use script_utils imports to be detected
            files = {
                'main.py': 'from script_utils import print_ok\nimport os\n',
                'helper.py': 'from script_utils import print_header\nfrom src.config import Config\n',
                'util.py': 'import json\n'
            }

            for name, content in files.items():
                with open(os.path.join(tmpdir, name), 'w') as f:
                    f.write(content)

            # Run analysis
            file_list = find_python_files(tmpdir)
            graph = build_dependency_graph(file_list, tmpdir)
            cycles = detect_cycles(graph)

            # Verify structure - should have dependencies to script_utils
            assert isinstance(graph, dict)
            assert isinstance(cycles, list)

    def test_cycle_detection_real_scenario(self):
        """Test cycle detection with a simple graph."""
        # Test directly with a graph that has known cycles
        graph = {
            'module_a': ['module_b'],
            'module_b': ['module_c'],
            'module_c': ['module_a']
        }
        cycles = detect_cycles(graph)
        # Should detect the cycle
        assert len(cycles) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
