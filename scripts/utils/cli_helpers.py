#!/usr/bin/env python3
"""
Common CLI utility functions for scripts.

Provides reusable CLI utilities for consistent behavior across all scripts
in the project, including user prompts, path validation, and formatting.

Usage:
    from scripts.utils.cli_helpers import confirm, format_size, validate_path, parse_args, json_output, table_print

Functions:
    - confirm(prompt)        - Ask for user confirmation (y/n)
    - format_size(bytes)    - Format bytes as human-readable string
    - validate_path(path, must_exist) - Validate path exists
    - parse_args()          - Standard argument parser helper
    - json_output()         - Print standardized JSON output
    - column_widths()       - Calculate optimal column widths for tables
    - table_from_list()     - Convert list of dicts to formatted table string
    - table_print()         - Print formatted table from list of dicts
    - validate_project_dir() - Validate project directory structure
    - validate_checkpoint()  - Validate checkpoint.json validity
    - get_project_info()    - Extract project metadata
    - is_ralph_project()    - Detect Ralph test projects

Table Functions Example:
    >>> data = [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 82}]
    >>> table_print(data, headers={"name": "Name", "score": "Score"})
    Name    Score
    -----   -----
    Alice   95
    Bob     82

Project Validation Example:
    >>> from scripts.utils.cli_helpers import validate_project_dir, get_project_info
    >>> valid, error = validate_project_dir("/path/to/project")
    >>> if valid:
    ...     info = get_project_info("/path/to/project")
    ...     print(f"Project: {info['name']}, Stage: {info['last_stage']}")
"""

import os
import re
import sys
import json
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ScriptResult:
    """
    Structured result type for script output.

    Provides a consistent schema for script results with support for
    success/failure states, data payloads, errors, and warnings.

    Attributes:
        success: Whether the operation succeeded
        message: Human-readable message describing the result
        data: Script-specific result data (default: empty dict)
        errors: List of error messages (default: empty list)
        warnings: List of warning messages (default: empty list)
        timestamp: ISO8601 timestamp of result creation

    Usage:
        # Success case
        result = ScriptResult.success("Operation completed", {"count": 5})

        # Failure case
        result = ScriptResult.failure("Operation failed", errors=["Error 1"])

        # JSON output
        print(result.to_json())
    """
    success: bool
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize result to JSON string.

        Args:
            indent: JSON indentation level (default: 2)

        Returns:
            JSON string representation
        """
        return json.dumps(asdict(self), indent=indent, default=str)

    def print_json(self, indent: int = 2) -> None:
        """
        Print result as JSON to stdout.

        Args:
            indent: JSON indentation level (default: 2)
        """
        print(self.to_json(indent=indent))


# Factory methods - added after class to avoid name conflict with 'success' field
def _success_factory(
    cls,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None
) -> ScriptResult:
    """
    Factory method for creating a success result.

    Args:
        message: Success message
        data: Optional result data
        warnings: Optional warnings to include

    Returns:
        ScriptResult with success=True
    """
    return ScriptResult(
        success=True,
        message=message,
        data=data or {},
        errors=[],
        warnings=warnings or [],
    )


def _failure_factory(
    cls,
    message: str,
    errors: Optional[List[str]] = None,
    data: Optional[Dict[str, Any]] = None
) -> ScriptResult:
    """
    Factory method for creating a failure result.

    Args:
        message: Failure message
        errors: List of error messages
        data: Optional partial data (e.g., partial results)

    Returns:
        ScriptResult with success=False
    """
    return ScriptResult(
        success=False,
        message=message,
        data=data or {},
        errors=errors or [],
        warnings=[],
    )


def _from_json_factory(cls, json_str: str) -> ScriptResult:
    """
    Parse result from JSON string.

    Args:
        json_str: JSON string to parse

    Returns:
        ScriptResult instance

    Raises:
        ValueError: If JSON is invalid or missing required fields
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # Validate required fields
    required_fields = ['success', 'message']
    for field_name in required_fields:
        if field_name not in data:
            raise ValueError(f"Missing required field: {field_name}")

    return ScriptResult(
        success=data.get('success', False),
        message=data.get('message', ''),
        data=data.get('data', {}),
        errors=data.get('errors', []),
        warnings=data.get('warnings', []),
        timestamp=data.get('timestamp', datetime.now().isoformat()),
    )


# Attach factory methods to class
ScriptResult.success = classmethod(_success_factory)
ScriptResult.failure = classmethod(_failure_factory)
ScriptResult.from_json = classmethod(_from_json_factory)


def confirm(prompt: str, default: bool = False) -> bool:
    """
    Ask for user confirmation with y/n prompt.

    Args:
        prompt: The question to ask the user
        default: Default value if user just presses Enter (True for yes, False for no)

    Returns:
        True if user confirmed (y/yes), False otherwise
    """
    choices = 'Y/n' if default else 'y/N'
    while True:
        try:
            response = input(f"  {prompt} [{choices}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            sys.exit(0)

        if not response:
            return default

        if response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        else:
            print("  Invalid input. Enter 'y' or 'n'.")


def format_size(size_bytes: Union[int, float]) -> str:
    """
    Format bytes as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable string (e.g., "1.5 GB")
    """
    size: float = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def validate_path(
    path: Union[str, Path],
    must_exist: bool = True,
    must_be_dir: bool = False,
    must_be_file: bool = False,
    create_if_missing: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate a path meets specified criteria.

    Args:
        path: Path to validate
        must_exist: If True, path must exist
        must_be_dir: If True, path must be a directory
        must_be_file: If True, path must be a file
        create_if_missing: If True and path doesn't exist, create it

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if path meets all criteria
        - error_message: None if valid, error description if invalid
    """
    path_obj = Path(path) if isinstance(path, str) else path

    # Check existence
    if not path_obj.exists():
        if create_if_missing:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                return True, None
            except Exception as e:
                return False, f"Could not create path: {e}"
        elif must_exist:
            return False, f"Path does not exist: {path_obj}"

    # Check type
    if must_be_dir:
        if not path_obj.is_dir():
            return False, f"Path is not a directory: {path_obj}"

    if must_be_file:
        if not path_obj.is_file():
            return False, f"Path is not a file: {path_obj}"

    return True, None


def parse_args(
    description: str,
    *args: Any,
    add_project_arg: bool = True,
    add_verbose: bool = False,
    add_quiet: bool = False,
    **kwargs: Any
) -> argparse.Namespace:
    """
    Create a standard argument parser with common options.

    This is a convenience wrapper around argparse.ArgumentParser that
    adds common options used across scripts.

    Args:
        description: Description for the argument parser
        *args: Additional positional args passed to ArgumentParser
        add_project_arg: If True, add --project/-p argument
        add_verbose: If True, add --verbose/-v argument
        add_quiet: If True, add --quiet/-q argument
        **kwargs: Additional keyword args passed to ArgumentParser

    Returns:
        Parsed arguments namespace
    """
    # Set up formatter for nice help text
    kwargs.setdefault('formatter_class', argparse.RawDescriptionHelpFormatter)

    parser = argparse.ArgumentParser(description, *args, **kwargs)

    # Standard --project argument
    if add_project_arg:
        parser.add_argument(
            '--project', '-p',
            required=False,
            help='Path to project directory'
        )

    # Standard --verbose argument
    if add_verbose:
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )

    # Standard --quiet argument
    if add_quiet:
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-essential output'
        )

    # Add --json for JSON output (common pattern)
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )

    # Add --yes/-y for skipping confirmations
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompts'
    )

    return parser.parse_args()


def ensure_dir(path: Union[str, Path], name: str = "directory") -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory
        name: Name of the directory for error messages

    Returns:
        Path object

    Raises:
        RuntimeError: If directory cannot be created
    """
    path_obj = Path(path) if isinstance(path, str) else path
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    except Exception as e:
        raise RuntimeError(f"Could not create {name}: {e}")


def get_dir_size(path: Union[str, Path], show_progress: bool = False) -> int:
    """
    Get total size of directory in bytes.

    Args:
        path: Directory path to calculate size of
        show_progress: If True, would show progress (for compatibility)

    Returns:
        Total size in bytes
    """
    path_obj = Path(path) if isinstance(path, str) else path
    total: int = 0

    try:
        for entry in path_obj.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass

    return total


def confirm_action(
    action: str,
    target: str,
    default: bool = False,
    yes_flag: bool = False
) -> bool:
    """
    Convenience function to confirm an action with standard format.

    Args:
        action: Action being confirmed (e.g., "Delete", "Archive")
        target: Target of the action (e.g., "these files", "this folder")
        default: Default value if user just presses Enter
        yes_flag: If True, skip confirmation and return True

    Returns:
        True if user confirmed, False otherwise
    """
    if yes_flag:
        return True

    prompt = f"{action} {target}?"
    return confirm(prompt, default=default)


def json_output(
    success: bool,
    script_name: str,
    data: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    indent: int = 2
) -> None:
    """
    Print standardized JSON output for script results.

    This provides a consistent JSON schema across all scripts for programmatic
    integration and CI/CD pipelines.

    Args:
        success: Whether the operation succeeded
        script_name: Name of the script (e.g., "download_list", "benchmark")
        data: Script-specific result data
        errors: List of error messages (if any)
        warnings: List of warning messages (if any)
        indent: JSON indentation level (default: 2)

    Output Schema:
        {
            "success": true|false,
            "script": "script_name",
            "timestamp": "ISO8601 timestamp",
            "data": { ... },        # script-specific data
            "errors": [],            # empty if no errors
            "warnings": []           # empty if no warnings
        }

    Usage:
        # Success case
        json_output(True, "download_list", {"downloaded": 5, "failed": 1})

        # Error case
        json_output(False, "download_list", errors=["Video not found: abc123"])
    """
    result = {
        "success": success,
        "script": script_name,
        "timestamp": datetime.now().isoformat(),
        "data": data or {},
        "errors": errors or [],
        "warnings": warnings or [],
    }
    print(json.dumps(result, indent=indent, default=str))


def column_widths(
    headers: List[str],
    rows: List[List[str]],
    min_width: int = 8,
    max_width: int = 50
) -> List[int]:
    """
    Calculate optimal column widths for table formatting.

    Analyzes all data to determine the minimum width needed for each column,
    with optional constraints.

    Args:
        headers: List of column header strings
        rows: List of data rows (each row is a list of cell values)
        min_width: Minimum column width (default: 8)
        max_width: Maximum column width (default: 50)

    Returns:
        List of widths, one per column

    Example:
        >>> headers = ["Name", "Status", "Count"]
        >>> rows = [["Alice", "Active", "42"], ["Bob", "Pending", "0"]]
        >>> column_widths(headers, rows)
        [8, 8, 5]  # "Active"/"Pending" -> 8, "Count" max -> 5
    """
    widths = [len(h) for h in headers]

    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                cell_str = str(cell) if cell is not None else ""
                widths[i] = max(widths[i], len(cell_str))

    # Apply constraints
    widths = [max(w, min_width) for w in widths]
    widths = [min(w, max_width) for w in widths]

    return widths


def table_from_list(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
    max_width: int = 50,
    truncate: bool = True
) -> str:
    """
    Convert a list of dictionaries to a formatted table string.

    Args:
        data: List of dictionaries representing rows
        columns: List of column keys to include (default: all keys from first row)
        headers: Optional mapping of column keys to display headers
        max_width: Maximum column width (default: 50)
        truncate: If True, truncate cells longer than max_width (default: True)

    Returns:
        Formatted table string with headers and rows

    Example:
        >>> data = [
        ...     {"name": "Alice", "status": "Active", "score": 95},
        ...     {"name": "Bob", "status": "Pending", "score": 82}
        ... ]
        >>> print(table_from_list(data, headers={"name": "Name", "score": "Score"}))
        Name    Score
        -----   -----
        Alice   95
        Bob     82
    """
    if not data:
        return ""

    # Determine columns to display
    if columns is None:
        columns = list(data[0].keys())

    # Get header strings
    header_row = [headers.get(col, col) if headers else col for col in columns]

    # Convert data to string rows
    rows = []
    for row in data:
        row_data = []
        for col in columns:
            value = row.get(col)
            cell = str(value) if value is not None else ""
            if truncate and len(cell) > max_width:
                cell = cell[:max_width - 3] + "..."
            row_data.append(cell)
        rows.append(row_data)

    # Calculate column widths
    widths = column_widths(header_row, rows, max_width=max_width)

    # Build table string
    lines = []

    # Header row
    header_cells = [h.ljust(widths[i]) for i, h in enumerate(header_row)]
    lines.append("  ".join(header_cells))

    # Separator
    sep_cells = ["-" * widths[i] for i in range(len(columns))]
    lines.append("  ".join(sep_cells))

    # Data rows
    for row in rows:
        row_cells = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append("  ".join(row_cells))

    return "\n".join(lines)


def table_print(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
    max_width: int = 50,
    truncate: bool = True,
    stream: Optional[Any] = None
) -> None:
    """
    Print a formatted table from a list of dictionaries.

    Args:
        data: List of dictionaries representing rows
        columns: List of column keys to include (default: all keys from first row)
        headers: Optional mapping of column keys to display headers
        max_width: Maximum column width (default: 50)
        truncate: If True, truncate cells longer than max_width (default: True)
        stream: Output stream (default: sys.stdout)

    Example:
        >>> table_print([
        ...     {"name": "Alice", "status": "Active"},
        ...     {"name": "Bob", "status": "Pending"}
        ... ], headers={"name": "Name", "status": "Status"})
        Name    Status
        -----   -----
        Alice   Active
        Bob     Pending
    """
    if stream is None:
        import sys
        stream = sys.stdout

    table = table_from_list(data, columns, headers, max_width, truncate)
    print(table, file=stream)


# Required subdirectories for a valid matcher project
PROJECT_REQUIRED_DIRS = ['voiceover', 'output']
# Required files for a valid matcher project
PROJECT_REQUIRED_FILES = ['checkpoint.json']
# Ralph test project indicators
RALPH_INDICATORS = ['ralph', 'test_', 'sprint_', '__Ralph__', ' Ralph ']


def _extract_project_info(path_obj: Path) -> Dict[str, Any]:
    """
    Extract basic project info from directory (internal helper).

    Args:
        path_obj: Path to project directory

    Returns:
        Dict with basic project info
    """
    return {
        'name': path_obj.name,
        'path': str(path_obj.resolve()),
        'has_voiceover': (path_obj / 'voiceover').exists(),
        'has_output': (path_obj / 'output').exists(),
        'has_checkpoint': (path_obj / 'checkpoint.json').exists(),
        'last_stage': '',
        'checkpoint_version': None,
        'is_ralph': False,
    }


def validate_project_dir(
    path: Union[str, Path],
    check_subdirs: bool = True,
    check_checkpoint: bool = True
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate a matcher project directory has required structure.

    Args:
        path: Path to the project directory
        check_subdirs: If True, verify voiceover/ and output/ exist
        check_checkpoint: If True, verify checkpoint.json exists

    Returns:
        Tuple of (is_valid, error_message, project_info)
        - is_valid: True if project structure is valid
        - error_message: None if valid, error description if invalid
        - project_info: Dict with project metadata if valid (or partial info on error)

    Example:
        >>> valid, error, info = validate_project_dir("/path/to/project")
        >>> if valid:
        ...     print(f"Project: {info['name']}, Last stage: {info['last_stage']}")
    """
    path_obj = Path(path) if isinstance(path, str) else path

    # Check directory exists
    if not path_obj.exists():
        return False, f"Project directory does not exist: {path_obj}", None

    if not path_obj.is_dir():
        return False, f"Path is not a directory: {path_obj}", None

    # Build project info (partial)
    project_info = _extract_project_info(path_obj)

    # Check required subdirectories
    if check_subdirs:
        for subdir in PROJECT_REQUIRED_DIRS:
            subdir_path = path_obj / subdir
            if not subdir_path.exists():
                return False, f"Required directory missing: {subdir}/", project_info

    # Check required files
    if check_checkpoint:
        checkpoint_path = path_obj / 'checkpoint.json'
        if not checkpoint_path.exists():
            return False, f"Required file missing: checkpoint.json", project_info

    # Validate checkpoint if it exists
    if check_checkpoint and checkpoint_path.exists():
        cp_valid, cp_error, _ = validate_checkpoint(checkpoint_path)
        if not cp_valid:
            return False, f"Invalid checkpoint: {cp_error}", project_info

    return True, None, project_info


def validate_checkpoint(
    path: Union[str, Path],
    load_data: bool = False
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate checkpoint.json file validity.

    Args:
        path: Path to checkpoint.json file or its parent directory
        load_data: If True, return the loaded checkpoint data

    Returns:
        Tuple of (is_valid, error_message, checkpoint_data)
        - is_valid: True if checkpoint is valid
        - error_message: None if valid, error description if invalid
        - checkpoint_data: Loaded checkpoint dict if load_data=True, else None

    Example:
        >>> valid, error, data = validate_checkpoint("/project/checkpoint.json", load_data=True)
        >>> if valid:
        ...     print(f"Last stage: {data.get('last_completed_stage')}")
    """
    path_obj = Path(path) if isinstance(path, str) else path

    # If given a directory, append checkpoint.json
    if path_obj.is_dir():
        path_obj = path_obj / 'checkpoint.json'

    # Check file exists
    if not path_obj.exists():
        return False, f"Checkpoint file does not exist: {path_obj}", None

    if not path_obj.is_file():
        return False, f"Checkpoint path is not a file: {path_obj}", None

    # Try to load JSON
    try:
        with open(path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None
    except Exception as e:
        return False, f"Error reading file: {e}", None

    # Validate required fields
    if not isinstance(data, dict):
        return False, "Checkpoint must be a JSON object", None

    # Check for essential fields
    if 'last_completed_stage' not in data:
        return False, "Missing required field: last_completed_stage", None

    # Validate last_completed_stage is a known stage
    valid_stages = ['', 'ANALYZE', 'VIDEO_SEARCH', 'CAPTION', 'MATCH',
                    'ITERATIVE_MATCH', 'DOWNLOAD_SEGMENTS', 'OUTPUT']
    stage = data.get('last_completed_stage', '')
    if stage and stage not in valid_stages:
        return False, f"Unknown stage: {stage}", None

    # Version is optional but if present should be valid format
    version = data.get('version')
    if version and not re.match(r'^\d+\.\d+(\.\d+)?$', str(version)):
        return False, f"Invalid version format: {version}", None

    if load_data:
        return True, None, data
    return True, None, None


def get_project_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract project metadata from project directory.

    Args:
        path: Path to the project directory

    Returns:
        Dict containing:
        - name: Project name (from directory name)
        - path: Full path to project
        - has_voiceover: Whether voiceover/ exists
        - has_output: Whether output/ exists
        - has_checkpoint: Whether checkpoint.json exists
        - last_stage: Last completed stage (from checkpoint)
        - checkpoint_version: Checkpoint version (if available)
        - is_ralph: Whether project appears to be a Ralph test project

    Example:
        >>> info = get_project_info("/path/to/project")
        >>> print(f"Project: {info['name']}, Stage: {info['last_stage']}")
    """
    path_obj = Path(path) if isinstance(path, str) else path
    info: Dict[str, Any] = {
        'name': path_obj.name,
        'path': str(path_obj.resolve()),
        'has_voiceover': False,
        'has_output': False,
        'has_checkpoint': False,
        'last_stage': '',
        'checkpoint_version': None,
        'is_ralph': False,
    }

    # Check directories
    info['has_voiceover'] = (path_obj / 'voiceover').exists()
    info['has_output'] = (path_obj / 'output').exists()

    # Check checkpoint
    checkpoint_path = path_obj / 'checkpoint.json'
    if checkpoint_path.exists():
        info['has_checkpoint'] = True
        valid, _, data = validate_checkpoint(checkpoint_path, load_data=True)
        if valid and data:
            info['last_stage'] = data.get('last_completed_stage', '')
            info['checkpoint_version'] = data.get('version')

    # Check if Ralph project
    info['is_ralph'] = is_ralph_project(path_obj)

    return info


def is_ralph_project(path: Union[str, Path]) -> bool:
    """
    Detect if a directory is a Ralph test project.

    Checks for common Ralph project naming patterns:
    - Contains 'ralph' (case-insensitive)
    - Starts with 'test_' or 'sprint_'
    - Contains '__Ralph__'
    - Contains ' Ralph ' (Ralph with spaces)

    Also checks for Ralph state files:
    - ralph/ subdirectory
    - ralph-config.json
    - sprint_history.json

    Args:
        path: Path to the project directory

    Returns:
        True if project appears to be a Ralph test project

    Example:
        >>> if is_ralph_project("/path/to/ralph_test_123"):
        ...     print("Ralph test project detected")
    """
    path_obj = Path(path) if isinstance(path, str) else path
    name_lower = path_obj.name.lower()

    # Check name patterns
    for indicator in RALPH_INDICATORS:
        if indicator.lower() in name_lower:
            return True

    # Check for Ralph subdirectory
    if (path_obj / 'ralph').exists() and (path_obj / 'ralph').is_dir():
        return True

    # Check for Ralph config files
    ralph_files = ['ralph-config.json', 'sprint_history.json', 'prd.json']
    for ralf_file in ralph_files:
        if (path_obj / ralf_file).exists():
            return True
        # Check in ralph/ subdirectory
        if (path_obj / 'ralph' / ralf_file).exists():
            return True

    return False


# =============================================================================
# PROGRESS TRACKING AND ETA UTILITIES (US-139-004)
# =============================================================================

def format_duration(seconds: Union[int, float]) -> str:
    """
    Format duration in seconds as human-readable string.

    Args:
        seconds: Duration in seconds (can be float for sub-second precision)

    Returns:
        Human-readable string (e.g., "1h 30m 45s", "45.2s", "500ms")

    Example:
        >>> format_duration(90)
        '1m 30s'
        >>> format_duration(3665)
        '1h 1m 5s'
        >>> format_duration(0.5)
        '500ms'
    """
    if seconds <= 0:
        return "0s"

    # Handle sub-second precision
    if seconds < 1:
        ms = seconds * 1000
        if ms <= 1:
            return "<1ms"
        return f"{ms:.0f}ms"

    # Handle seconds and above
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_eta(
    remaining: Union[int, float],
    completed: Optional[int] = None,
    total: Optional[int] = None,
    show_remaining: bool = True
) -> str:
    """
    Format ETA string with optional progress comparison.

    Args:
        remaining: Seconds remaining (estimated)
        completed: Number of items completed (optional)
        total: Total number of items (optional)
        show_remaining: If True, show remaining time; if False, show elapsed

    Returns:
        Formatted ETA string with optional progress

    Example:
        >>> format_eta(3665)
        '1h 1m 5s remaining'
        >>> format_eta(1800, completed=5, total=10)
        '30m remaining (5/10 - 50%)'
    """
    time_str = format_duration(remaining)

    if show_remaining:
        base = f"{time_str} remaining"
    else:
        base = f"{time_str} elapsed"

    # Add progress if completed and total are provided
    if completed is not None and total is not None and total > 0:
        percent = (completed / total) * 100
        base = f"{time_str} remaining ({completed}/{total} - {percent:.0f}%)"

    return base


class ProgressTracker:
    """
    Track progress for multi-step operations with ETA calculation.

    Provides estimated vs actual time comparison and can generate
    progress reports for long-running operations.

    Attributes:
        total: Total number of steps
        completed: Number of completed steps
        start_time: Start timestamp (time.time())
        step_times: List of times taken per step

    Example:
        >>> tracker = ProgressTracker(total=10, description="Processing items")
        >>> for item in items:
        ...     process(item)
        ...     tracker.update()
        ...     print(tracker.format_status())
        >>> print(tracker.format_summary())
    """

    def __init__(
        self,
        total: int,
        description: str = "Progress",
        start_time: Optional[float] = None
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of steps to track
            description: Description of the operation
            start_time: Optional start time (defaults to time.time())
        """
        import time
        self.total = max(0, total)
        self.completed = 0
        self.description = description
        self.start_time = start_time or time.time()
        self.step_times: List[float] = []
        self._last_update_time = self.start_time

    def update(self, steps: int = 1) -> None:
        """
        Update progress by completed steps.

        Args:
            steps: Number of steps completed (default: 1)
        """
        import time
        if steps <= 0:
            return

        now = time.time()
        elapsed_since_last = now - self._last_update_time

        # Record step time (average for smooth ETA)
        if self.completed > 0:
            self.step_times.append(elapsed_since_last)

        self.completed = min(self.completed + steps, self.total)
        self._last_update_time = now

    def eta(self) -> float:
        """
        Calculate estimated seconds remaining.

        Returns:
            Estimated seconds remaining, or 0 if complete
        """
        if self.completed == 0:
            return 0.0

        import time
        now = time.time()
        elapsed = now - self.start_time

        # Calculate average time per step
        if self.completed > 1:
            avg_time_per_step = elapsed / self.completed
        else:
            avg_time_per_step = elapsed

        remaining_steps = self.total - self.completed
        return max(0.0, avg_time_per_step * remaining_steps)

    def eta_str(self) -> str:
        """
        Get ETA as formatted string.

        Returns:
            Formatted ETA string (e.g., "5m 30s remaining")
        """
        return format_eta(self.eta(), self.completed, self.total)

    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.

        Returns:
            Elapsed seconds since start
        """
        import time
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        """
        Get elapsed time as formatted string.

        Returns:
            Formatted elapsed string
        """
        return format_duration(self.elapsed())

    def average_step_time(self) -> float:
        """
        Get average time per completed step.

        Returns:
            Average seconds per step, or 0 if no steps completed
        """
        if self.completed == 0:
            return 0.0
        return self.elapsed() / self.completed

    def progress_percent(self) -> float:
        """
        Get progress as percentage.

        Returns:
            Progress percentage (0-100)
        """
        if self.total == 0:
            return 100.0
        return (self.completed / self.total) * 100

    def is_complete(self) -> bool:
        """
        Check if all steps are complete.

        Returns:
            True if completed >= total
        """
        return self.completed >= self.total

    def format_status(self) -> str:
        """
        Format current status for display.

        Returns:
            Status string with progress, ETA, and timing
        """
        percent = self.progress_percent()
        eta_str = self.eta_str()
        elapsed_str = self.elapsed_str()

        return (
            f"{self.description}: {self.completed}/{self.total} "
            f"({percent:.0f}%) - {eta_str} - Elapsed: {elapsed_str}"
        )

    def format_summary(self) -> str:
        """
        Format summary after completion.

        Returns:
            Summary string with total time and averages
        """
        elapsed = self.elapsed()
        avg_time = self.average_step_time()

        return (
            f"{self.description} complete: "
            f"{self.completed}/{self.total} steps in {format_duration(elapsed)} "
            f"(avg {format_duration(avg_time)}/step)"
        )

    def compare_estimate(self) -> Dict[str, Any]:
        """
        Compare estimated vs actual time (useful after completion).

        Returns:
            Dict with estimated_time, actual_time, difference, and accuracy

        Example:
            >>> tracker.update(10)  # Complete all steps
            >>> comparison = tracker.compare_estimate()
            >>> print(f"Accuracy: {comparison['accuracy']:.1f}%")
        """
        estimated = self.eta()  # This gives remaining, so add completed
        actual = self.elapsed()

        # At completion, eta() returns 0, but we want total estimated
        if self.completed > 0 and self.total > 0:
            # Recalculate: what we expected at start
            if self.completed == self.total:
                # At completion, use average to get full estimate
                avg = actual / self.completed
                estimated_total = avg * self.total
            else:
                # During operation
                estimated_total = estimated + actual
        else:
            estimated_total = actual

        difference = actual - estimated_total
        accuracy = 100.0 - (abs(difference) / max(estimated_total, 0.1) * 100) if estimated_total > 0 else 100.0

        return {
            "estimated_time": estimated_total,
            "actual_time": actual,
            "difference": difference,
            "difference_str": f"{'+' if difference >= 0 else ''}{format_duration(difference)}",
            "accuracy": max(0.0, accuracy),
            "completed": self.completed,
            "total": self.total,
            "avg_step_time": self.average_step_time(),
        }

    def __enter__(self) -> 'ProgressTracker':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - optional summary output."""
        # Could auto-print summary here if desired
        pass
