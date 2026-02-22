#!/usr/bin/env python3
"""
Tests for scripts/utils/cli_helpers.py

Tests the CLI utility functions including:
- confirm: User confirmation prompts
- format_size: Human-readable byte formatting
- validate_path: Path validation with various criteria
- parse_args: Argument parser with common options
- json_output: Standardized JSON output
- ensure_dir: Directory creation
- get_dir_size: Directory size calculation
- confirm_action: Action confirmation wrapper
"""

import sys
import json
import pytest
import tempfile
import argparse
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module under test
from utils.cli_helpers import (
    confirm,
    format_size,
    validate_path,
    parse_args,
    json_output,
    ensure_dir,
    get_dir_size,
    confirm_action,
    column_widths,
    table_from_list,
    table_print,
    validate_project_dir,
    validate_checkpoint,
    get_project_info,
    is_ralph_project,
    ScriptResult,
    format_duration,
    format_eta,
    ProgressTracker,
)


@pytest.mark.script
class TestConfirm:
    """Test confirm function"""

    def test_confirm_yes_input(self):
        """User entering 'y' returns True"""
        with patch('builtins.input', return_value='y'):
            result = confirm("Continue?")
            assert result is True

    def test_confirm_yes_full_input(self):
        """User entering 'yes' returns True"""
        with patch('builtins.input', return_value='yes'):
            result = confirm("Continue?")
            assert result is True

    def test_confirm_no_input(self):
        """User entering 'n' returns False"""
        with patch('builtins.input', return_value='n'):
            result = confirm("Continue?")
            assert result is False

    def test_confirm_no_full_input(self):
        """User entering 'no' returns False"""
        with patch('builtins.input', return_value='no'):
            result = confirm("Continue?")
            assert result is False

    def test_confirm_default_yes(self):
        """Empty input with default=True returns True"""
        with patch('builtins.input', return_value=''):
            result = confirm("Continue?", default=True)
            assert result is True

    def test_confirm_default_no(self):
        """Empty input with default=False returns False"""
        with patch('builtins.input', return_value=''):
            result = confirm("Continue?", default=False)
            assert result is False

    def test_confirm_invalid_input_reprompts(self):
        """Invalid input shows error and reprompts"""
        inputs = iter(['invalid', 'y'])
        with patch('builtins.input', side_effect=lambda x: next(inputs)):
            result = confirm("Continue?")
            assert result is True

    def test_confirm_keyboard_interrupt(self):
        """KeyboardInterrupt exits cleanly"""
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit) as exc_info:
                confirm("Continue?")
            assert exc_info.value.code == 0


@pytest.mark.script
class TestFormatSize:
    """Test format_size function"""

    def test_format_size_bytes(self):
        """Test bytes formatting"""
        assert format_size(0) == "0.0 B"
        assert format_size(1) == "1.0 B"
        assert format_size(512) == "512.0 B"

    def test_format_size_kilobytes(self):
        """Test kilobytes formatting"""
        assert format_size(1024) == "1.0 KB"
        assert format_size(1536) == "1.5 KB"
        assert format_size(10240) == "10.0 KB"

    def test_format_size_megabytes(self):
        """Test megabytes formatting"""
        assert format_size(1048576) == "1.0 MB"
        assert format_size(1572864) == "1.5 MB"
        assert format_size(10485760) == "10.0 MB"

    def test_format_size_gigabytes(self):
        """Test gigabytes formatting"""
        assert format_size(1073741824) == "1.0 GB"
        assert format_size(1610612736) == "1.5 GB"

    def test_format_size_terabytes(self):
        """Test terabytes formatting"""
        assert format_size(1099511627776) == "1.0 TB"

    def test_format_size_petabytes(self):
        """Test petabytes formatting (large values)"""
        assert format_size(1125899906842624) == "1.0 PB"


@pytest.mark.script
class TestValidatePath:
    """Test validate_path function"""

    def test_validate_path_exists(self):
        """Validates existing path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid, error = validate_path(tmpdir, must_exist=True)
            assert valid is True
            assert error is None

    def test_validate_path_not_exists(self):
        """Returns error for non-existing path when must_exist=True"""
        valid, error = validate_path("/nonexistent/path/12345", must_exist=True)
        assert valid is False
        assert "does not exist" in error

    def test_validate_path_not_exists_optional(self):
        """Allows non-existing path when must_exist=False"""
        valid, error = validate_path("/nonexistent/path/12345", must_exist=False)
        assert valid is True
        assert error is None

    def test_validate_path_is_directory(self):
        """Validates directory path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid, error = validate_path(tmpdir, must_be_dir=True)
            assert valid is True
            assert error is None

    def test_validate_path_is_file(self):
        """Validates file path"""
        with tempfile.NamedTemporaryFile() as tmpfile:
            valid, error = validate_path(tmpfile.name, must_be_file=True)
            assert valid is True
            assert error is None

    def test_validate_path_file_as_dir(self):
        """Returns error when path is file but directory required"""
        with tempfile.NamedTemporaryFile() as tmpfile:
            valid, error = validate_path(tmpfile.name, must_be_dir=True)
            assert valid is False
            assert "not a directory" in error

    def test_validate_path_dir_as_file(self):
        """Returns error when path is directory but file required"""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid, error = validate_path(tmpdir, must_be_file=True)
            assert valid is False
            assert "not a file" in error

    def test_validate_path_create_if_missing(self):
        """Creates path when create_if_missing=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "new_dir"
            valid, error = validate_path(new_path, must_exist=True, create_if_missing=True)
            assert valid is True
            assert error is None
            assert new_path.exists()

    def test_validate_path_create_fails(self):
        """Returns error when creation fails"""
        # Use a path that's definitely not creatable
        valid, error = validate_path("/proc/0/no_access", create_if_missing=True)
        # Either fails to create or path doesn't exist
        assert valid is False or error is not None

    def test_validate_path_string_input(self):
        """Accepts string path input"""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid, error = validate_path(tmpdir, must_exist=True)
            assert valid is True


@pytest.mark.script
class TestParseArgs:
    """Test parse_args function"""

    def test_parse_args_basic(self):
        """Creates parser with basic options"""
        # Save sys.argv
        original_argv = sys.argv
        try:
            sys.argv = ['test_script']

            args = parse_args("Test script", add_project_arg=False, add_verbose=False)

            # Should have json and yes flags by default
            assert hasattr(args, 'json')
            assert hasattr(args, 'yes')
            assert args.json is False
            assert args.yes is False
        finally:
            sys.argv = original_argv

    def test_parse_args_with_project_arg(self):
        """Adds --project/-p argument when requested"""
        original_argv = sys.argv
        try:
            sys.argv = ['test_script', '--project', '/some/path']

            args = parse_args("Test script", add_project_arg=True)

            assert hasattr(args, 'project')
            assert args.project == '/some/path'
        finally:
            sys.argv = original_argv

    def test_parse_args_with_verbose(self):
        """Adds --verbose/-v argument when requested"""
        original_argv = sys.argv
        try:
            sys.argv = ['test_script', '--verbose']

            args = parse_args("Test script", add_verbose=True)

            assert hasattr(args, 'verbose')
            assert args.verbose is True
        finally:
            sys.argv = original_argv

    def test_parse_args_with_json_flag(self):
        """Enables JSON output when --json passed"""
        original_argv = sys.argv
        try:
            sys.argv = ['test_script', '--json']

            args = parse_args("Test script", add_project_arg=False)

            assert args.json is True
        finally:
            sys.argv = original_argv

    def test_parse_args_with_yes_flag(self):
        """Enables yes flag when --yes passed"""
        original_argv = sys.argv
        try:
            sys.argv = ['test_script', '--yes']

            args = parse_args("Test script", add_project_arg=False)

            assert args.yes is True
        finally:
            sys.argv = original_argv

    def test_parse_args_with_short_flags(self):
        """Supports short flag versions -j and -y"""
        original_argv = sys.argv
        try:
            sys.argv = ['test_script', '-j', '-y']

            args = parse_args("Test script", add_project_arg=False)

            assert args.json is True
            assert args.yes is True
        finally:
            sys.argv = original_argv


@pytest.mark.script
class TestJsonOutput:
    """Test json_output function"""

    def test_json_output_success(self):
        """Outputs success JSON"""
        output = StringIO()
        with patch('sys.stdout', output):
            json_output(True, "test_script", {"key": "value"})

        result = json.loads(output.getvalue())
        assert result["success"] is True
        assert result["script"] == "test_script"
        assert result["data"] == {"key": "value"}
        assert result["errors"] == []
        assert result["warnings"] == []
        assert "timestamp" in result

    def test_json_output_error(self):
        """Outputs error JSON"""
        output = StringIO()
        with patch('sys.stdout', output):
            json_output(False, "test_script", errors=["Error 1", "Error 2"])

        result = json.loads(output.getvalue())
        assert result["success"] is False
        assert result["errors"] == ["Error 1", "Error 2"]

    def test_json_output_warnings(self):
        """Outputs warnings in JSON"""
        output = StringIO()
        with patch('sys.stdout', output):
            json_output(True, "test_script", warnings=["Warning 1"])

        result = json.loads(output.getvalue())
        assert result["warnings"] == ["Warning 1"]

    def test_json_output_custom_indent(self):
        """Uses custom indent when specified"""
        output = StringIO()
        with patch('sys.stdout', output):
            json_output(True, "test_script", indent=4)

        result_str = output.getvalue()
        # 4-space indent should produce more indentation
        assert "    " in result_str


@pytest.mark.script
class TestEnsureDir:
    """Test ensure_dir function"""

    def test_ensure_dir_existing(self):
        """Returns path for existing directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_dir(tmpdir)
            assert result == Path(tmpdir)

    def test_ensure_dir_creates_new(self):
        """Creates directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "subdir" / "nested"
            result = ensure_dir(str(new_dir))
            assert result.exists()
            assert result.is_dir()

    def test_ensure_dir_returns_path(self):
        """Returns Path object"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_dir(tmpdir)
            assert isinstance(result, Path)


@pytest.mark.script
class TestGetDirSize:
    """Test get_dir_size function"""

    def test_get_dir_size_empty(self):
        """Returns 0 for empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            size = get_dir_size(tmpdir)
            assert size == 0

    def test_get_dir_size_with_files(self):
        """Calculates size of directory with files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with known sizes
            file1 = Path(tmpdir) / "file1.txt"
            file2 = Path(tmpdir) / "file2.txt"
            file1.write_text("x" * 100)
            file2.write_text("y" * 200)

            size = get_dir_size(tmpdir)
            assert size >= 300

    def test_get_dir_size_nested(self):
        """Calculates size of nested directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("z" * 50)

            size = get_dir_size(tmpdir)
            assert size >= 50

    def test_get_dir_size_nonexistent(self):
        """Returns 0 for nonexistent directory"""
        size = get_dir_size("/nonexistent/directory")
        assert size == 0


@pytest.mark.script
class TestConfirmAction:
    """Test confirm_action function"""

    def test_confirm_action_with_yes_flag(self):
        """Returns True when yes_flag is True"""
        result = confirm_action("Delete", "file.txt", yes_flag=True)
        assert result is True

    def test_confirm_action_calls_confirm(self):
        """Calls confirm with formatted prompt"""
        with patch('utils.cli_helpers.confirm') as mock_confirm:
            mock_confirm.return_value = True
            result = confirm_action("Delete", "file.txt")
            mock_confirm.assert_called_once()
            call_args = mock_confirm.call_args[0]
            assert "Delete" in call_args[0]
            assert "file.txt" in call_args[0]

    def test_confirm_action_passes_default(self):
        """Passes default value to confirm"""
        with patch('utils.cli_helpers.confirm') as mock_confirm:
            confirm_action("Archive", "folder/", default=True)
            assert mock_confirm.call_args[1]["default"] is True


@pytest.mark.script
class TestNonInteractivePattern:
    """Test that scripts use --non-interactive pattern"""

    def test_confirm_works_without_interactive_input(self):
        """Confirm can work with mocked input (non-interactive)"""
        with patch('builtins.input', return_value='y'):
            result = confirm("Proceed?")
            assert result is True

    def test_parse_args_non_interactive(self):
        """parse_args works without user input"""
        original_argv = sys.argv
        try:
            sys.argv = ['script', '--yes']
            args = parse_args("Test", add_project_arg=False)
            assert args.yes is True
        finally:
            sys.argv = original_argv


@pytest.mark.script
class TestColumnWidths:
    """Test column_widths function"""

    def test_column_widths_basic(self):
        """Calculates widths from headers and rows"""
        headers = ["Name", "Status"]
        rows = [["Alice", "Active"], ["Bob", "Pending"]]
        widths = column_widths(headers, rows)
        # Default min_width=8 means headers determine min, values can exceed
        assert widths[0] >= 5  # "Alice"
        assert widths[1] >= 7  # "Active"

    def test_column_widths_with_longer_values(self):
        """Uses longer value when it exceeds header"""
        headers = ["Name", "Description"]
        rows = [["Alice", "A very long description here"]]
        widths = column_widths(headers, rows)
        assert widths[1] == len("A very long description here")

    def test_column_widths_min_width(self):
        """Applies minimum width constraint"""
        headers = ["A", "B"]
        rows = [["x", "y"]]
        widths = column_widths(headers, rows, min_width=10)
        assert widths[0] >= 10
        assert widths[1] >= 10

    def test_column_widths_max_width(self):
        """Applies maximum width constraint"""
        headers = ["LongHeader"]
        rows = [["A" * 100]]
        widths = column_widths(headers, rows, max_width=20)
        assert widths[0] == 20


@pytest.mark.script
class TestTableFromList:
    """Test table_from_list function"""

    def test_table_from_list_basic(self):
        """Converts list of dicts to table string"""
        data = [
            {"name": "Alice", "score": 95},
            {"name": "Bob", "score": 82}
        ]
        table = table_from_list(data)
        assert "name" in table
        assert "Alice" in table
        assert "Bob" in table

    def test_table_from_list_with_headers(self):
        """Uses custom headers when provided"""
        data = [{"name": "Alice", "score": 95}]
        table = table_from_list(data, headers={"name": "Name", "score": "Score"})
        assert "Name" in table
        assert "Score" in table
        assert "name" not in table

    def test_table_from_list_with_columns(self):
        """Filters to specified columns"""
        data = [{"name": "Alice", "score": 95, "extra": "value"}]
        table = table_from_list(data, columns=["name", "score"])
        assert "name" in table
        assert "Alice" in table

    def test_table_from_list_empty(self):
        """Returns empty string for empty data"""
        table = table_from_list([])
        assert table == ""

    def test_table_from_list_truncates(self):
        """Truncates long values when enabled"""
        data = [{"name": "A" * 100}]
        table = table_from_list(data, max_width=20, truncate=True)
        assert "..." in table
        assert len(max(table.split('\n'), key=len)) <= 20

    def test_table_from_list_no_truncate(self):
        """Does not truncate when disabled"""
        data = [{"name": "A" * 100}]
        table = table_from_list(data, truncate=False)
        assert "A" * 100 in table


@pytest.mark.script
class TestTablePrint:
    """Test table_print function"""

    def test_table_print_outputs(self):
        """Prints table to stream"""
        output = StringIO()
        data = [{"name": "Alice", "score": 95}]
        table_print(data, stream=output)
        result = output.getvalue()
        assert "Alice" in result

    def test_table_print_with_headers(self):
        """Uses custom headers"""
        output = StringIO()
        data = [{"name": "Alice", "score": 95}]
        table_print(data, headers={"name": "Name"}, stream=output)
        result = output.getvalue()
        assert "Name" in result
        assert "name" not in result

    def test_table_print_default_stream(self):
        """Uses sys.stdout by default"""
        output = StringIO()
        with patch('sys.stdout', output):
            data = [{"key": "value"}]
            table_print(data)
        assert "key" in output.getvalue()


@pytest.mark.script
class TestValidateProjectDir:
    """Test validate_project_dir function"""

    def test_validate_project_dir_valid(self):
        """Validates a proper project directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "TestProject"
            project_dir.mkdir()
            (project_dir / "voiceover").mkdir()
            (project_dir / "output").mkdir()
            (project_dir / "checkpoint.json").write_text('{"last_completed_stage": "MATCH"}')

            valid, error, info = validate_project_dir(project_dir)
            assert valid is True
            assert error is None
            assert info is not None
            assert info['name'] == "TestProject"

    def test_validate_project_dir_missing_voiceover(self):
        """Returns error when voiceover/ is missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "TestProject"
            project_dir.mkdir()
            (project_dir / "output").mkdir()
            (project_dir / "checkpoint.json").write_text('{"last_completed_stage": "MATCH"}')

            valid, error, info = validate_project_dir(project_dir)
            assert valid is False
            assert "voiceover" in error
            assert info is not None

    def test_validate_project_dir_missing_output(self):
        """Returns error when output/ is missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "TestProject"
            project_dir.mkdir()
            (project_dir / "voiceover").mkdir()
            (project_dir / "checkpoint.json").write_text('{"last_completed_stage": "MATCH"}')

            valid, error, info = validate_project_dir(project_dir)
            assert valid is False
            assert "output" in error

    def test_validate_project_dir_missing_checkpoint(self):
        """Returns error when checkpoint.json is missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "TestProject"
            project_dir.mkdir()
            (project_dir / "voiceover").mkdir()
            (project_dir / "output").mkdir()

            valid, error, info = validate_project_dir(project_dir)
            assert valid is False
            assert "checkpoint.json" in error

    def test_validate_project_dir_nonexistent(self):
        """Returns error for nonexistent directory"""
        valid, error, info = validate_project_dir("/nonexistent/project")
        assert valid is False
        assert "does not exist" in error

    def test_validate_project_dir_not_a_directory(self):
        """Returns error when path is a file, not directory"""
        with tempfile.NamedTemporaryFile() as tmpfile:
            valid, error, info = validate_project_dir(tmpfile.name)
            assert valid is False
            assert "not a directory" in error

    def test_validate_project_dir_no_subdirs_check(self):
        """Skips subdirectory check when disabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "TestProject"
            project_dir.mkdir()
            # No voiceover or output dirs
            (project_dir / "checkpoint.json").write_text('{"last_completed_stage": "MATCH"}')

            valid, error, info = validate_project_dir(project_dir, check_subdirs=False)
            assert valid is True

    def test_validate_project_dir_no_checkpoint_check(self):
        """Skips checkpoint check when disabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "TestProject"
            project_dir.mkdir()
            (project_dir / "voiceover").mkdir()
            (project_dir / "output").mkdir()
            # No checkpoint.json

            valid, error, info = validate_project_dir(project_dir, check_checkpoint=False)
            assert valid is True


@pytest.mark.script
class TestValidateCheckpoint:
    """Test validate_checkpoint function"""

    def test_validate_checkpoint_valid(self):
        """Validates a proper checkpoint file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'last_completed_stage': 'MATCH'}, f)
            f.flush()

            try:
                valid, error, data = validate_checkpoint(f.name, load_data=True)
                assert valid is True
                assert error is None
                assert data['last_completed_stage'] == 'MATCH'
            finally:
                Path(f.name).unlink()

    def test_validate_checkpoint_nonexistent(self):
        """Returns error for nonexistent file"""
        valid, error, data = validate_checkpoint("/nonexistent/checkpoint.json")
        assert valid is False
        assert "does not exist" in error

    def test_validate_checkpoint_invalid_json(self):
        """Returns error for invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            f.flush()

            try:
                valid, error, data = validate_checkpoint(f.name)
                assert valid is False
                assert "Invalid JSON" in error
            finally:
                Path(f.name).unlink()

    def test_validate_checkpoint_missing_stage(self):
        """Returns error when last_completed_stage is missing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'version': '4.0.0'}, f)
            f.flush()

            try:
                valid, error, data = validate_checkpoint(f.name)
                assert valid is False
                assert "last_completed_stage" in error
            finally:
                Path(f.name).unlink()

    def test_validate_checkpoint_unknown_stage(self):
        """Returns error for unknown stage"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'last_completed_stage': 'INVALID_STAGE'}, f)
            f.flush()

            try:
                valid, error, data = validate_checkpoint(f.name)
                assert valid is False
                assert "Unknown stage" in error
            finally:
                Path(f.name).unlink()

    def test_validate_checkpoint_from_directory(self):
        """Accepts directory path and appends checkpoint.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            checkpoint_path.write_text('{"last_completed_stage": "CAPTION"}')

            valid, error, data = validate_checkpoint(tmpdir, load_data=True)
            assert valid is True
            assert data['last_completed_stage'] == 'CAPTION'


@pytest.mark.script
class TestGetProjectInfo:
    """Test get_project_info function"""

    def test_get_project_info_basic(self):
        """Extracts basic project info"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "MyProject"
            project_dir.mkdir()
            (project_dir / "voiceover").mkdir()
            (project_dir / "output").mkdir()
            (project_dir / "checkpoint.json").write_text(
                '{"last_completed_stage": "VIDEO_SEARCH", "version": "4.0.0"}'
            )

            info = get_project_info(project_dir)
            assert info['name'] == "MyProject"
            assert info['has_voiceover'] is True
            assert info['has_output'] is True
            assert info['has_checkpoint'] is True
            assert info['last_stage'] == "VIDEO_SEARCH"
            assert info['checkpoint_version'] == "4.0.0"

    def test_get_project_info_minimal(self):
        """Works with minimal project structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "MinimalProject"
            project_dir.mkdir()

            info = get_project_info(project_dir)
            assert info['name'] == "MinimalProject"
            assert info['has_voiceover'] is False
            assert info['has_output'] is False
            assert info['has_checkpoint'] is False
            assert info['last_stage'] == ''


@pytest.mark.script
class TestIsRalphProject:
    """Test is_ralph_project function"""

    def test_is_ralph_project_by_name(self):
        """Detects Ralph in name"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "ralph_test_123"
            project_dir.mkdir()

            assert is_ralph_project(project_dir) is True

    def test_is_ralph_project_by_test_prefix(self):
        """Detects test_ prefix"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_something"
            project_dir.mkdir()

            assert is_ralph_project(project_dir) is True

    def test_is_ralph_project_by_sprint_prefix(self):
        """Detects sprint_ prefix"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "sprint_42"
            project_dir.mkdir()

            assert is_ralph_project(project_dir) is True

    def test_is_ralph_project_normal_project(self):
        """Returns False for normal project"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "MyDocumentary"
            project_dir.mkdir()

            assert is_ralph_project(project_dir) is False

    def test_is_ralph_project_by_subdir(self):
        """Detects Ralph subdirectory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "NormalProject"
            project_dir.mkdir()
            (project_dir / "ralph").mkdir()

            assert is_ralph_project(project_dir) is True

    def test_is_ralph_project_by_config_file(self):
        """Detects Ralph config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "NormalProject"
            project_dir.mkdir()
            (project_dir / "ralph-config.json").write_text("{}")

            assert is_ralph_project(project_dir) is True


@pytest.mark.script
class TestScriptResult:
    """Test ScriptResult dataclass"""

    def test_script_result_basic(self):
        """Creates basic ScriptResult"""
        result = ScriptResult(success=True, message="Test message")
        assert result.success is True
        assert result.message == "Test message"
        assert result.data == {}
        assert result.errors == []
        assert result.warnings == []
        assert result.timestamp is not None

    def test_script_result_success_factory(self):
        """Creates success result via factory method"""
        result = ScriptResult.success("Operation completed", {"count": 5})
        assert result.success is True
        assert result.message == "Operation completed"
        assert result.data == {"count": 5}
        assert result.errors == []
        assert result.warnings == []

    def test_script_result_success_with_warnings(self):
        """Creates success result with warnings"""
        result = ScriptResult.success("Completed with warnings", warnings=["Minor issue"])
        assert result.success is True
        assert result.warnings == ["Minor issue"]

    def test_script_result_failure_factory(self):
        """Creates failure result via factory method"""
        result = ScriptResult.failure("Operation failed", errors=["Error 1", "Error 2"])
        assert result.success is False
        assert result.message == "Operation failed"
        assert result.errors == ["Error 1", "Error 2"]
        assert result.warnings == []

    def test_script_result_failure_with_partial_data(self):
        """Creates failure result with partial data"""
        result = ScriptResult.failure("Partial failure", errors=["Error 1"], data={"completed": 3})
        assert result.success is False
        assert result.data == {"completed": 3}

    def test_script_result_to_json(self):
        """Serializes to JSON"""
        result = ScriptResult.success("Test message", {"key": "value"})
        json_str = result.to_json()
        assert "success" in json_str
        assert "Test message" in json_str
        assert "key" in json_str
        assert "value" in json_str

    def test_script_result_from_json(self):
        """Parses from JSON"""
        original = ScriptResult.success("Test message", {"key": "value"})
        json_str = original.to_json()
        parsed = ScriptResult.from_json(json_str)
        assert parsed.success == original.success
        assert parsed.message == original.message
        assert parsed.data == original.data

    def test_script_result_from_json_invalid(self):
        """Raises error for invalid JSON"""
        with pytest.raises(ValueError, match="Invalid JSON"):
            ScriptResult.from_json("not valid json")

    def test_script_result_from_json_missing_field(self):
        """Raises error for missing required field"""
        with pytest.raises(ValueError, match="Missing required field"):
            ScriptResult.from_json('{"success": true}')  # missing message

    def test_script_result_print_json(self):
        """Prints JSON to stdout"""
        result = ScriptResult.success("Test")
        output = StringIO()
        with patch('sys.stdout', output):
            result.print_json()
        assert "success" in output.getvalue()

    def test_script_result_timestamp_format(self):
        """Has ISO8601 timestamp"""
        result = ScriptResult.success("Test")
        # Timestamp should be parseable as ISO format
        parsed = datetime.fromisoformat(result.timestamp)
        assert parsed is not None


@pytest.mark.script
class TestFormatDuration:
    """Test format_duration function"""

    def test_format_duration_zero(self):
        """Formats zero seconds"""
        assert format_duration(0) == "0s"

    def test_format_duration_negative(self):
        """Returns 0s for negative values"""
        assert format_duration(-10) == "0s"

    def test_format_duration_seconds(self):
        """Formats seconds only"""
        assert format_duration(45) == "45s"

    def test_format_duration_minutes(self):
        """Formats minutes and seconds"""
        assert format_duration(90) == "1m 30s"
        assert format_duration(60) == "1m"

    def test_format_duration_hours(self):
        """Formats hours, minutes, seconds"""
        assert format_duration(3665) == "1h 1m 5s"
        assert format_duration(3600) == "1h"
        assert format_duration(7200) == "2h"

    def test_format_duration_sub_second(self):
        """Formats sub-second durations"""
        assert format_duration(0.5) == "500ms"
        assert format_duration(0.1) == "100ms"
        assert format_duration(0.01) == "10ms"

    def test_format_duration_very_small(self):
        """Formats very small sub-second durations"""
        assert format_duration(0.001) == "<1ms"


@pytest.mark.script
class TestFormatEta:
    """Test format_eta function"""

    def test_format_eta_basic(self):
        """Formats basic ETA"""
        result = format_eta(3665)
        assert "1h 1m 5s" in result
        assert "remaining" in result

    def test_format_eta_with_progress(self):
        """Formats ETA with progress"""
        result = format_eta(1800, completed=5, total=10)
        assert "30m" in result
        assert "5/10" in result
        assert "50%" in result

    def test_format_eta_show_elapsed(self):
        """Shows elapsed instead of remaining"""
        result = format_eta(300, show_remaining=False)
        assert "elapsed" in result

    def test_format_eta_zero_total(self):
        """Handles zero total gracefully"""
        result = format_eta(100, completed=5, total=0)
        assert "remaining" in result


@pytest.mark.script
class TestProgressTracker:
    """Test ProgressTracker class"""

    def test_progress_tracker_init(self):
        """Initializes with correct values"""
        tracker = ProgressTracker(total=10, description="Test")
        assert tracker.total == 10
        assert tracker.completed == 0
        assert tracker.description == "Test"

    def test_progress_tracker_update(self):
        """Updates completed count"""
        tracker = ProgressTracker(total=10, start_time=0)
        tracker.update(1)
        assert tracker.completed == 1

    def test_progress_tracker_multiple_updates(self):
        """Handles multiple updates"""
        tracker = ProgressTracker(total=10, start_time=0)
        tracker.update(3)
        tracker.update(2)
        assert tracker.completed == 5

    def test_progress_tracker_caps_at_total(self):
        """Does not exceed total"""
        tracker = ProgressTracker(total=5, start_time=0)
        tracker.update(10)
        assert tracker.completed == 5

    def test_progress_tracker_eta(self):
        """Calculates ETA"""
        import time
        tracker = ProgressTracker(total=10, start_time=time.time() - 5)
        tracker.update(5)  # 5 seconds for 5 items = 1s/item, 5 remaining
        eta = tracker.eta()
        assert eta >= 0

    def test_progress_tracker_progress_percent(self):
        """Calculates progress percentage"""
        tracker = ProgressTracker(total=10, start_time=0)
        tracker.update(2)
        assert tracker.progress_percent() == 20.0

    def test_progress_tracker_zero_total(self):
        """Handles zero total"""
        tracker = ProgressTracker(total=0, start_time=0)
        assert tracker.progress_percent() == 100.0

    def test_progress_tracker_is_complete(self):
        """Detects completion"""
        tracker = ProgressTracker(total=5, start_time=0)
        assert tracker.is_complete() is False
        tracker.update(5)
        assert tracker.is_complete() is True

    def test_progress_tracker_average_step_time(self):
        """Calculates average step time"""
        import time
        tracker = ProgressTracker(total=10, start_time=time.time() - 10)
        tracker.update(5)
        avg = tracker.average_step_time()
        assert avg > 0

    def test_progress_tracker_format_status(self):
        """Formats status string"""
        tracker = ProgressTracker(total=10, description="Test", start_time=0)
        tracker.update(2)
        status = tracker.format_status()
        assert "Test" in status
        assert "2/10" in status

    def test_progress_tracker_format_summary(self):
        """Formats summary string"""
        tracker = ProgressTracker(total=10, description="Test", start_time=0)
        tracker.update(10)
        summary = tracker.format_summary()
        assert "complete" in summary
        assert "10/10" in summary

    def test_progress_tracker_compare_estimate(self):
        """Compares estimated vs actual"""
        tracker = ProgressTracker(total=10, start_time=0)
        tracker.update(10)
        comparison = tracker.compare_estimate()
        assert "estimated_time" in comparison
        assert "actual_time" in comparison
        assert "difference" in comparison
        assert "accuracy" in comparison

    def test_progress_tracker_context_manager(self):
        """Works as context manager"""
        with ProgressTracker(total=5, description="Test") as tracker:
            tracker.update(2)
        assert tracker.completed == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
