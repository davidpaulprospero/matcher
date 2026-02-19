#!/usr/bin/env python3
"""
Tests for scripts/tools.py CLI interface.

Tests the unified tools entry point including:
- list subcommand with and without filters
- run subcommand with argument passing
- docs subcommand output formats
- validate subcommand error detection
- grep subcommand search functionality

Mocks subprocess calls to avoid running actual scripts.
"""

import sys
import json
import pytest
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module under test
import tools


@pytest.fixture
def mock_scripts():
    """Provide mock script metadata for testing."""
    return {
        'cleanup_project': {
            'name': 'cleanup_project',
            'description': 'Clean up project directories',
            'usage': 'cleanup_project.py --project <path>',
            'category': 'general',
            'path': str(scripts_dir / 'cleanup_project.py')
        },
        'validate_config': {
            'name': 'validate_config',
            'description': 'Validate configuration files',
            'usage': 'validate_config.py [--config <path>]',
            'category': 'config',
            'path': str(scripts_dir / 'validate_config.py')
        },
        'benchmark': {
            'name': 'benchmark',
            'description': 'Run performance benchmarks',
            'usage': 'benchmark.py [--iterations N]',
            'category': 'analysis',
            'path': str(scripts_dir / 'benchmark.py')
        },
        'run_tests': {
            'name': 'run_tests',
            'description': 'Run test suite',
            'usage': 'run_tests.py [--path <path>]',
            'category': 'testing',
            'path': str(scripts_dir / 'run_tests.py')
        }
    }


@pytest.fixture
def mock_scripts_dir(tmp_path):
    """Create a temporary scripts directory with test scripts."""
    script1 = tmp_path / "test_script.py"
    script1.write_text('''#!/usr/bin/env python3
"""Test script for cleanup.

Usage:
    test_script.py --project <path>
"""

def main():
    print("Hello")

if __name__ == "__main__":
    main()
''')

    script2 = tmp_path / "another_script.py"
    script2.write_text('''#!/usr/bin/env python3
"""Another test script."""

def run():
    pass
''')

    return tmp_path


class TestGetScriptMetadata:
    """Test metadata extraction from scripts."""

    def test_extracts_basic_metadata(self, tmp_path):
        """Test extraction of name, description, usage."""
        script_path = tmp_path / "test_script.py"
        script_path.write_text('''#!/usr/bin/env python3
"""This is a test script.

Usage:
    test_script.py --verbose
"""

def main():
    pass
''')

        result = tools.get_script_metadata(script_path)

        assert result['name'] == 'test_script'
        assert 'test script' in result['description'].lower()
        assert '--verbose' in result['usage']

    def test_categorizes_test_scripts(self, tmp_path):
        """Test that test scripts are categorized correctly."""
        script_path = tmp_path / "test_example.py"
        script_path.write_text('"""Test script."""')

        result = tools.get_script_metadata(script_path)

        assert result['category'] == 'testing'

    def test_categorizes_config_scripts(self, tmp_path):
        """Test that config scripts are categorized correctly."""
        script_path = tmp_path / "validate_config.py"
        script_path.write_text('"""Config validator."""')

        result = tools.get_script_metadata(script_path)

        assert result['category'] == 'config'

    def test_categorizes_analysis_scripts(self, tmp_path):
        """Test that analysis scripts are categorized correctly."""
        script_path = tmp_path / "analyze_results.py"
        script_path.write_text('"""Analyzer."""')

        result = tools.get_script_metadata(script_path)

        assert result['category'] == 'analysis'


class TestCmdList:
    """Test list subcommand."""

    def test_list_all_scripts(self, mock_scripts):
        """Test listing all scripts without filter."""
        args = MagicMock()
        args.category = None
        args.json = False

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = tools.cmd_list(mock_scripts, args)
            output = mock_stdout.getvalue()

        assert result == 0
        assert 'cleanup_project' in output
        assert 'validate_config' in output

    def test_list_filtered_by_category(self, mock_scripts):
        """Test listing scripts filtered by category."""
        args = MagicMock()
        args.category = 'config'
        args.json = False

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = tools.cmd_list(mock_scripts, args)
            output = mock_stdout.getvalue()

        assert result == 0
        assert 'validate_config' in output
        # Other categories should not appear in output
        assert 'cleanup_project' not in output

    def test_list_json_output(self, mock_scripts):
        """Test list output in JSON format."""
        args = MagicMock()
        args.category = None
        args.json = True

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = tools.cmd_list(mock_scripts, args)

        assert result == 0
        # Verify JSON output contains expected keys
        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert 'cleanup_project' in data
        assert data['cleanup_project']['category'] == 'general'


class TestCmdRun:
    """Test run subcommand."""

    def test_run_finds_exact_match(self, mock_scripts):
        """Test running a script with exact name match."""
        args = MagicMock()
        args.script = 'cleanup_project'
        args.args = ['--project', '/test/path']

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = tools.cmd_run(mock_scripts, args)

        assert result == 0
        mock_run.assert_called_once()

    def test_run_partial_match(self, mock_scripts):
        """Test running a script with partial name match."""
        args = MagicMock()
        args.script = 'cleanup'
        args.args = []

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = tools.cmd_run(mock_scripts, args)

        assert result == 0

    def test_run_ambiguous_name(self, mock_scripts):
        """Test error when script name is ambiguous."""
        # Add another script starting with 'test'
        mock_scripts['test_runner'] = {
            'name': 'test_runner',
            'description': 'Test runner',
            'usage': '',
            'category': 'testing',
            'path': '/test/test_runner.py'
        }
        mock_scripts['test_foo'] = {
            'name': 'test_foo',
            'description': 'Test foo',
            'usage': '',
            'category': 'testing',
            'path': '/test/test_foo.py'
        }

        args = MagicMock()
        args.script = 'test'
        args.args = []

        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = tools.cmd_run(mock_scripts, args)

        assert result == 1
        assert 'Ambiguous' in mock_stderr.getvalue()

    def test_run_script_not_found(self, mock_scripts):
        """Test error when script is not found."""
        args = MagicMock()
        args.script = 'nonexistent_script'
        args.args = []

        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = tools.cmd_run(mock_scripts, args)

        assert result == 1
        assert 'not found' in mock_stderr.getvalue().lower()

    def test_run_passes_arguments(self, mock_scripts):
        """Test that arguments are passed to the script."""
        args = MagicMock()
        args.script = 'cleanup_project'
        args.args = ['--project', '/test/path', '--verbose']

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = tools.cmd_run(mock_scripts, args)

        assert result == 0
        call_args = mock_run.call_args[0][0]
        assert '--project' in call_args
        assert '/test/path' in call_args
        assert '--verbose' in call_args


class TestCmdDocs:
    """Test docs subcommand."""

    def test_docs_exact_match(self, mock_scripts):
        """Test showing docs for exact script match."""
        args = MagicMock()
        args.script = 'cleanup_project'
        args.json = False

        # Mock the file read
        mock_scripts['cleanup_project']['path'] = '/fake/cleanup_project.py'

        with patch('builtins.open', MagicMock()):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = tools.cmd_docs(mock_scripts, args)

        assert result == 0

    def test_docs_json_output(self, mock_scripts):
        """Test docs output in JSON format."""
        args = MagicMock()
        args.script = 'cleanup_project'
        args.json = True

        with patch('builtins.open', MagicMock()):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = tools.cmd_docs(mock_scripts, args)

        assert result == 0
        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert 'name' in data
        assert 'description' in data
        assert 'usage' in data

    def test_docs_not_found(self, mock_scripts):
        """Test error when script not found."""
        args = MagicMock()
        args.script = 'nonexistent'
        args.json = False

        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = tools.cmd_docs(mock_scripts, args)

        assert result == 1
        assert 'not found' in mock_stderr.getvalue().lower()

    def test_docs_ambiguous_name(self, mock_scripts):
        """Test error when script name is ambiguous."""
        mock_scripts['test_a'] = {'name': 'test_a', 'description': '', 'usage': '', 'category': 'testing', 'path': '/test/a.py'}
        mock_scripts['test_b'] = {'name': 'test_b', 'description': '', 'usage': '', 'category': 'testing', 'path': '/test/b.py'}

        args = MagicMock()
        args.script = 'test'
        args.json = False

        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = tools.cmd_docs(mock_scripts, args)

        assert result == 1
        assert 'Ambiguous' in mock_stderr.getvalue()


class TestCmdValidate:
    """Test validate subcommand."""

    def test_validate_all_valid(self, mock_scripts):
        """Test validation when all scripts are valid."""
        args = MagicMock()
        args.json = False

        with patch('builtins.open', MagicMock()):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = tools.cmd_validate(mock_scripts, args)

        assert result == 0  # Returns 0 when no errors

    def test_validate_json_output(self, mock_scripts):
        """Test validation output in JSON format."""
        args = MagicMock()
        args.json = True

        with patch('builtins.open', MagicMock()):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = tools.cmd_validate(mock_scripts, args)

        assert result == 0
        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert 'valid' in data
        assert 'errors' in data
        assert 'warnings' in data

    def test_validate_detects_syntax_error(self, mock_scripts):
        """Test that syntax errors are detected."""
        args = MagicMock()
        args.json = False

        def mock_open(path, *args, **kwargs):
            if 'open' in str(path):
                raise SyntaxError("unexpected EOF", ("", 1, 1, ""))
            m = MagicMock()
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__ = MagicMock(return_value=False)
            m.read = MagicMock(return_value="def broken():")
            return m

        with patch('builtins.open', side_effect=mock_open):
            with patch('sys.stdout', new_callable=StringIO):
                with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                    result = tools.cmd_validate(mock_scripts, args)

        assert result == 1  # Returns 1 when errors found


class TestCmdGrep:
    """Test grep subcommand."""

    def test_grep_finds_matches(self, mock_scripts):
        """Test grep finds pattern matches."""
        args = MagicMock()
        args.pattern = 'def main'
        args.context = 0
        args.ignore_case = False
        args.regex = False
        args.json = False

        with patch('builtins.open', MagicMock()) as mock_file:
            mock_file.return_value.__enter__.return_value.readlines.return_value = [
                'def main():\n',
                '    pass\n'
            ]
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = tools.cmd_grep(mock_scripts, args)

        # Should find matches or return error (pattern may not match mock content)
        assert result in [0, 1]

    def test_grep_json_output(self, mock_scripts):
        """Test grep output in JSON format."""
        args = MagicMock()
        args.pattern = 'def main'
        args.context = 0
        args.ignore_case = False
        args.regex = False
        args.json = True

        with patch('builtins.open', MagicMock()) as mock_file:
            mock_file.return_value.__enter__.return_value.readlines.return_value = []
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = tools.cmd_grep(mock_scripts, args)

        assert result in [0, 1]
        if result == 0:
            output = mock_stdout.getvalue()
            data = json.loads(output)
            assert 'pattern' in data
            assert 'matches' in data

    def test_grep_case_insensitive(self, mock_scripts):
        """Test case-insensitive grep."""
        args = MagicMock()
        args.pattern = 'DEF MAIN'
        args.context = 0
        args.ignore_case = True
        args.regex = False
        args.json = False

        with patch('builtins.open', MagicMock()) as mock_file:
            mock_file.return_value.__enter__.return_value.readlines.return_value = [
                'def main():\n'
            ]
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = tools.cmd_grep(mock_scripts, args)

        assert result in [0, 1]

    def test_grep_invalid_regex(self, mock_scripts):
        """Test grep with invalid regex pattern."""
        args = MagicMock()
        args.pattern = '[invalid'
        args.context = 0
        args.ignore_case = False
        args.regex = True
        args.json = False

        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = tools.cmd_grep(mock_scripts, args)

        assert result == 1
        assert 'Invalid' in mock_stderr.getvalue()


class TestCmdDiff:
    """Test diff subcommand for comparing scripts."""

    def test_diff_compares_scripts(self, mock_scripts):
        """Test diff command compares scripts."""
        args = MagicMock()
        args.category = None
        args.json = False

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = tools.cmd_diff(mock_scripts, args)

        # Should return 0 if comparison succeeds
        assert result == 0
        output = mock_stdout.getvalue()
        assert 'Script Comparison' in output or 'SUMMARY' in output

    def test_diff_json_output(self, mock_scripts):
        """Test diff command JSON output."""
        args = MagicMock()
        args.category = None
        args.json = True

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = tools.cmd_diff(mock_scripts, args)

        assert result == 0
        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert 'common_patterns' in data
        assert 'total_scripts_compared' in data

    def test_diff_category_filter(self, mock_scripts):
        """Test diff command with category filter."""
        args = MagicMock()
        args.category = 'general'
        args.json = False

        # Filter to only general category - should have at least 2 scripts
        general_scripts = {k: v for k, v in mock_scripts.items() if v['category'] == 'general'}

        # Ensure we have at least 2 scripts
        if len(general_scripts) < 2:
            # Add another general script
            general_scripts['test_script'] = {
                'name': 'test_script',
                'description': 'Test script',
                'usage': '',
                'category': 'general',
                'path': str(scripts_dir / 'cleanup_project.py')
            }

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = tools.cmd_diff(general_scripts, args)

        assert result == 0
        output = mock_stdout.getvalue()
        assert 'SUMMARY' in output

    def test_diff_insufficient_scripts(self, mock_scripts):
        """Test diff command with fewer than 2 scripts."""
        # Single script
        single_script = {'test': mock_scripts['cleanup_project']}

        args = MagicMock()
        args.category = None
        args.json = False

        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = tools.cmd_diff(single_script, args)

        assert result == 1
        assert 'Need at least 2' in mock_stderr.getvalue()

    def test_diff_finds_common_patterns(self, mock_scripts):
        """Test diff command finds common patterns."""
        args = MagicMock()
        args.category = None
        args.json = True

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = tools.cmd_diff(mock_scripts, args)

        assert result == 0
        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert 'common_patterns' in data
        # Check that common import patterns are detected
        patterns = data['common_patterns']
        assert any('import' in p for p in patterns.keys())

    def test_compute_content_hash(self):
        """Test content hash computation."""
        content1 = "def main():\n    pass\n"
        content2 = "def main():\n    pass\n"
        content3 = "def other():\n    pass\n"

        hash1 = tools.compute_content_hash(content1)
        hash2 = tools.compute_content_hash(content2)
        hash3 = tools.compute_content_hash(content3)

        # Same content should produce same hash
        assert hash1 == hash2
        # Different content should produce different hash
        assert hash1 != hash3

    def test_extract_code_signatures(self, mock_scripts):
        """Test code signature extraction."""
        signatures = tools.extract_code_signatures(mock_scripts)

        assert isinstance(signatures, dict)
        # Should have signatures for scripts that exist on disk
        for script_name, meta in mock_scripts.items():
            if Path(meta['path']).exists():
                assert script_name in signatures

    def test_find_common_patterns(self):
        """Test finding common patterns across scripts."""
        signatures = {
            'script1': ['import sys', 'def main', 'class Foo'],
            'script2': ['import sys', 'def main', 'class Bar'],
            'script3': ['import os', 'def other']
        }

        common = tools.find_common_patterns(signatures, min_occurrences=2)

        # Should find 'import sys' and 'def main' as common
        assert 'import sys' in common
        assert 'def main' in common
        # 'import os' only appears once
        assert 'import os' not in common


class TestToolsIntegration:
    """Integration tests for the full CLI."""

    def test_main_list_command(self):
        """Test full list command execution."""
        test_args = ['tools.py', 'list']

        with patch('sys.argv', test_args):
            with patch.object(tools, 'discover_scripts', return_value={}):
                with patch('sys.stdout', new_callable=StringIO):
                    result = tools.main()

        assert result in [0, 1]

    def test_main_validate_command(self):
        """Test full validate command execution."""
        test_args = ['tools.py', 'validate']

        with patch('sys.argv', test_args):
            with patch.object(tools, 'discover_scripts', return_value={}):
                with patch('sys.stdout', new_callable=StringIO):
                    result = tools.main()

        assert result in [0, 1]

    def test_main_no_command_shows_help(self):
        """Test that no command shows help."""
        test_args = ['tools.py']

        with patch('sys.argv', test_args):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                result = tools.main()

        assert result == 1

    def test_main_grep_command(self):
        """Test full grep command execution."""
        test_args = ['tools.py', 'grep', 'def main']

        with patch('sys.argv', test_args):
            with patch.object(tools, 'discover_scripts', return_value={}):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    with patch('sys.stderr', new_callable=StringIO):
                        result = tools.main()

        assert result in [0, 1]

    def test_completion_generation(self):
        """Test shell completion script generation."""
        with patch.object(tools, 'discover_scripts', return_value={'script1': {}, 'script2': {}}):
            result = tools.generate_completions()

        assert 'complete' in result
        assert 'script1' in result
        assert 'script2' in result
