#!/usr/bin/env python3
"""
Common utility functions for scripts.

Provides standardized output formatting for consistent developer experience
across all scripts in the project.

IMPORT GUIDELINES (Standard Pattern for scripts/)
================================================

All scripts in scripts/ directory should use the following import pattern:

    #!/usr/bin/env python3
    '''
    Script description.
    '''
    import os
    import sys
    from pathlib import Path

    # Add project root and scripts directory to path for imports
    _script_path = os.path.abspath(__file__)
    project_root = Path(_script_path).parent.parent
    scripts_dir = Path(_script_path).parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(scripts_dir))

    # Change to project root so relative paths (config.yaml) work correctly
    os.chdir(project_root)

    # Import standardized output functions
    from script_utils import print_error, print_warn

    # Import CLI helpers (if needed)
    from utils.cli_helpers import confirm

Key Rules:
- Use 'from script_utils import ...' (NO 's' - not 'scripts.script_utils')
- Use 'from utils.cli_helpers import ...' for utils modules
- Add BOTH project_root AND scripts_dir to sys.path
- Always os.chdir(project_root) for relative path consistency

Standard format:
    - [OK]   - Success messages
    - [WARN] - Warning messages
    - [ERROR] - Error messages
    - No emoji
    - 2-space indent for content

MIGRATION GUIDE (Converting from custom logging)
================================================

This guide explains how to migrate scripts from custom logging or print()
to using script_utils for consistent output.

1. REMOVE logging imports and setup:
   REMOVE:
       import logging
       logging.basicConfig(level=logging.INFO, format='...')
       logger = logging.getLogger(__name__)

   ADD:
       from script_utils import print_ok, print_warn, print_error, print_info, print_header

2. REPLACE logger calls with script_utils functions:

   | Old Pattern              | New Pattern                        |
   |--------------------------|-------------------------------------|
   | logger.info(msg)         | print_ok(msg) or print_info(msg)   |
   | logger.warning(msg)      | print_warn(msg)                    |
   | logger.error(msg)        | print_error(msg)                   |
   | logger.error(msg, 1)    | print_error(msg, exit_code=1)      |
   | logger.debug(msg)        | print_info(msg)                    |
   | print("Section")         | print_header("Section")            |
   | print(f"Result: {x}")    | print_ok(f"Result: {x}")             |

3. Choose the right function:
   - print_ok(msg): Success messages, positive results, completions
   - print_warn(msg): Non-fatal issues, expected edge cases
   - print_error(msg, exit_code): Fatal errors, failures (exit_code optional)
   - print_info(msg): Verbose-only information (only shown with -v)
   - print_header(title): Section headers for grouping output

4. VERBOSITY LEVELS:
   - print_ok, print_warn, print_error: Always shown (normal mode)
   - print_info: Only shown in verbose mode (-v flag, verbosity=2)
   - Use set_verbosity(level) to control: 0=quiet, 1=normal, 2=verbose

Example migration:

   BEFORE:
       import logging
       logging.basicConfig(level=logging.INFO, format='%(message)s')
       logger = logging.getLogger(__name__)

       logger.info("Starting download")
       logger.warning("No cookies found, continuing anyway")
       logger.error("Failed to download")
       sys.exit(1)

   AFTER:
       from script_utils import print_ok, print_warn, print_error

       print_ok("Starting download")
       print_warn("No cookies found, continuing anyway")
       print_error("Failed to download", exit_code=1)

PROGRESS BAR SUPPORT (Using tqdm)
=================================

script_utils provides three progress tracking utilities that integrate with tqdm:

1. progress_bar(iterable, ...) - Wrap an iterable with a progress bar
2. track_progress(total, ...)   - Context manager for manual iteration tracking
3. spinner(message)             - Context manager for indeterminate operations

Import:
    from script_utils import progress_bar, track_progress, spinner

Examples:

    # Method 1: Wrap iterable (simplest)
    for item in progress_bar(items, desc="Processing"):
        process(item)

    # Method 2: Manual tracking with context manager
    with track_progress(total=100, desc="Downloading") as pbar:
        for i in range(100):
            download(i)
            pbar.update(1)

    # Method 3: Indeterminate spinner (loading state)
    with spinner("Loading data") as sp:
        data = fetch_data()
    # Spinner auto-closes on exit

Note: All functions gracefully degrade if tqdm is not installed.
"""

import sys
import os
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Any, List, Optional, Union

# Verbosity levels: 0 = quiet, 1 = normal, 2 = verbose
_verbosity: int = 1


def set_verbosity(level: int) -> None:
    """Set the global verbosity level (0=quiet, 1=normal, 2=verbose)."""
    global _verbosity
    _verbosity = max(0, min(2, level))


def get_verbosity() -> int:
    """Get the current verbosity level."""
    return _verbosity


def print_header(title: str) -> None:
    """Print section header with title."""
    # Headers always shown in normal/verbose mode
    if _verbosity >= 1:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")


def print_ok(msg: str) -> None:
    """Print success message."""
    # Quiet mode suppresses OK messages
    if _verbosity >= 1:
        print(f"  [OK] {msg}")


def print_warn(msg: str) -> None:
    """Print warning message."""
    # Warnings always shown (normal/verbose)
    if _verbosity >= 1:
        print(f"  [WARN] {msg}")


def print_error(msg: str, exit_code: Optional[int] = None) -> None:
    """Print error message with optional exit."""
    # Errors always shown in all modes
    print(f"  [ERROR] {msg}")
    if exit_code is not None:
        sys.exit(exit_code)


def print_info(msg: str) -> None:
    """Print informational message."""
    # Info only shown in verbose mode
    if _verbosity >= 2:
        print(f"  [INFO] {msg}")


# Progress bar support using tqdm
# ============================================================

def _import_tqdm():
    """Lazy import of tqdm to avoid hard dependency."""
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        return None


def progress_bar(iterable, desc: str = "", total: Optional[int] = None,
                 unit: str = "it", leave: bool = True, disable: bool = False):
    """
    Create a progress bar wrapper around an iterable.

    Args:
        iterable: The iterable to wrap
        desc: Description shown before the progress bar
        total: Total number of items (if not inferrable from iterable)
        unit: Unit name for items (default: "it")
        leave: Whether to leave the progress bar after completion
        disable: Whether to disable the progress bar

    Returns:
        Wrapped iterable with progress bar, or plain iterable if tqdm unavailable

    Example:
        for item in progress_bar(items, desc="Processing"):
            process(item)
    """
    tqdm = _import_tqdm()
    if tqdm is None:
        # Fallback to plain iterable if tqdm not available
        print_warn("tqdm not available, progress bar disabled")
        return iterable

    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave, disable=disable)


def track_progress(total: Optional[int] = None, desc: str = "Progress",
                   unit: str = "it", leave: bool = True):
    """
    Context manager for iterative operations with progress tracking.

    Args:
        total: Total number of iterations
        desc: Description shown in progress bar
        unit: Unit name for items
        leave: Whether to leave progress bar after completion

    Yields:
        tqdm progress bar object

    Example:
        with track_progress(total=100, desc="Downloading") as pbar:
            for i in range(100):
                download(i)
                pbar.update(1)
    """
    tqdm = _import_tqdm()
    if tqdm is None:
        # Fallback to dummy context manager
        print_warn("tqdm not available, progress tracking disabled")
        yield None
        return

    with tqdm(total=total, desc=desc, unit=unit, leave=leave) as pbar:
        yield pbar


def spinner(message: str = "Working...", disable: bool = False):
    """
    Context manager for indeterminate operations (loading spinners).

    Args:
        message: Message to display next to spinner
        disable: Whether to disable the spinner

    Yields:
        Spinner object with .stop() method

    Example:
        with spinner("Loading data") as sp:
            data = load_data()
            sp.stop()  # Stop early if needed
        # Spinner auto-stops on context exit
    """
    tqdm = _import_tqdm()
    if tqdm is None:
        # Fallback to simple print
        print_info(message)
        class DummySpinner:
            def stop(self): pass
        yield DummySpinner()
        return

    # Use tqdm spinner (position=-1 puts it on its own line)
    pbar = tqdm(range(0), desc=message, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                position=-1, leave=False, disable=disable)

    class SpinnerWrapper:
        """Wrapper around tqdm spinner with additional methods."""
        def __init__(self, pbar):
            self._pbar = pbar

        def update(self, n: int = 1):
            """Update spinner (no-op for indeterminate)."""
            pass

        def set_message(self, msg: str):
            """Update spinner message."""
            self._pbar.set_description(msg)

        def stop(self):
            """Stop the spinner."""
            self._pbar.close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stop()

    yield SpinnerWrapper(pbar)


# =============================================================================
# SHARED CONFIG LOADING (US-132-006)
# =============================================================================

def load_config_for_script(
    config_path: str = "config.yaml",
    required: bool = True,
    skip_validation: bool = False
):
    """
    Load config for scripts with consistent error handling.

    This provides a unified way for scripts to load config.yaml with
    graceful handling of missing files and clear error messages.

    Args:
        config_path: Path to config file (default: "config.yaml")
        required: If True (default), error when config not found.
                  If False, returns None when config not found.
        skip_validation: If True, skip API key validation (for --dry-run etc.)

    Returns:
        Loaded Config object, or None if not required and not found

    Raises:
        FileNotFoundError: If config file not found and required=True
        ValueError: If config file is invalid

    Usage:
        # Simple usage (default config.yaml)
        config = load_config_for_script()

        # Custom config path
        config = load_config_for_script("custom.yaml")

        # Optional config (returns None if not found)
        config = load_config_for_script("optional.yaml", required=False)

        # Use with argparse (adds --config/-c flag)
        # See add_config_argument() below
    """
    from pathlib import Path as PathLib

    config_file = PathLib(config_path)

    # Check if config file exists
    if not config_file.exists():
        if required:
            print_error(
                f"Config file not found: {config_path}\n"
                f"  - Ensure config.yaml exists in the project root\n"
                f"  - Or use --config to specify a custom config path",
                exit_code=1
            )
            # Exit already happened, but return None for type safety
            return None
        else:
            print_warn(f"Config file not found: {config_path} (optional, skipping)")
            return None

    # Load the config
    try:
        from src.config import load_config
        config = load_config(config_path, skip_final_validation=skip_validation)
        print_info(f"Loaded config: {config_path}")
        return config
    except Exception as e:
        print_error(
            f"Failed to load config from {config_path}: {e}",
            exit_code=1
        )
        return None


def add_config_argument(parser, default: str = "config.yaml", help_text: Optional[str] = None):
    """
    Add --config/-c argument to an argparse parser.

    This provides consistent config path argument handling across all scripts.

    Args:
        parser: argparse.ArgumentParser instance
        default: Default config path (default: "config.yaml")
        help_text: Custom help text (optional)

    Returns:
        The argument group added (for reference if needed)

    Usage:
        parser = argparse.ArgumentParser()
        add_config_argument(parser)

        # Or with custom settings
        add_config_argument(parser, default="custom.yaml", help_text="Custom config")

        args = parser.parse_args()
        config = load_config_for_script(args.config)
    """
    if help_text is None:
        help_text = f'Path to config file (default: {default})'

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=default,
        help=help_text
    )

    return parser


def add_project_argument(parser, required: bool = False, help_text: Optional[str] = None):
    """
    Add --project/-p argument to an argparse parser.

    This provides consistent project path argument handling across all scripts.

    Args:
        parser: argparse.ArgumentParser instance
        required: If True, make the argument required (default: False)
        help_text: Custom help text (optional)

    Returns:
        The argument group added (for reference if needed)

    Usage:
        parser = argparse.ArgumentParser()
        add_project_argument(parser)

        # Or with required=True
        add_project_argument(parser, required=True)

        args = parser.parse_args()
    """
    if help_text is None:
        help_text = 'Path to project directory'

    parser.add_argument(
        '--project', '-p',
        type=str,
        required=required,
        help=help_text
    )

    return parser


def add_verbose_argument(parser, help_text: Optional[str] = None):
    """
    Add --verbose/-v argument to an argparse parser.

    This provides consistent verbose flag handling across all scripts.

    Args:
        parser: argparse.ArgumentParser instance
        help_text: Custom help text (optional)

    Returns:
        The argument added (for reference if needed)

    Usage:
        parser = argparse.ArgumentParser()
        add_verbose_argument(parser)

        args = parser.parse_args()
        if args.verbose:
            # Enable verbose logging
            set_verbosity(1)
    """
    if help_text is None:
        help_text = 'Enable verbose output'

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help=help_text
    )

    return parser


def add_quiet_argument(parser, help_text: Optional[str] = None):
    """
    Add --quiet/-q argument to an argparse parser.

    This provides consistent quiet flag handling across all scripts.

    Args:
        parser: argparse.ArgumentParser instance
        help_text: Custom help text (optional)

    Returns:
        The argument added (for reference if needed)

    Usage:
        parser = argparse.ArgumentParser()
        add_quiet_argument(parser)

        args = parser.parse_args()
        if args.quiet:
            # Suppress non-essential output
            set_verbosity(-1)
    """
    if help_text is None:
        help_text = 'Suppress non-essential output'

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help=help_text
    )

    return parser


def add_standard_arguments(
    parser,
    add_project: bool = True,
    add_verbose: bool = True,
    add_quiet: bool = True,
    project_required: bool = False
):
    """
    Add all standard arguments to an argparse parser.

    This is a convenience function that adds common arguments in one call:
    - --project/-p (if add_project=True)
    - --verbose/-v (if add_verbose=True)
    - --quiet/-q (if add_quiet=True)
    - --json/-j (always)
    - --yes/-y (always)

    Args:
        parser: argparse.ArgumentParser instance
        add_project: If True, add --project/-p argument
        add_verbose: If True, add --verbose/-v argument
        add_quiet: If True, add --quiet/-q argument
        project_required: If True, make project argument required

    Returns:
        The parser (for method chaining)

    Usage:
        parser = argparse.ArgumentParser(description="My script")
        add_standard_arguments(
            parser,
            add_project=True,
            add_verbose=True,
            add_quiet=True,
            project_required=False
        )
        args = parser.parse_args()
    """
    if add_project:
        add_project_argument(parser, required=project_required)

    if add_verbose:
        add_verbose_argument(parser)

    if add_quiet:
        add_quiet_argument(parser)

    # Always add these
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompts'
    )

    return parser


# =============================================================================
# FILE OPERATION UTILITIES
# =============================================================================

def safe_read_file(
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    errors: str = 'replace',
    binary: bool = False
) -> Union[str, bytes]:
    """
    Safely read a file with proper encoding handling.

    Args:
        file_path: Path to the file to read
        encoding: Text encoding (default: utf-8). Ignored if binary=True
        errors: How to handle encoding errors (default: 'replace')
        binary: If True, read as binary mode (default: False)

    Returns:
        File contents as str (text mode) or bytes (binary mode)

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        content = safe_read_file("config.yaml")
        binary_data = safe_read_file("image.png", binary=True)
    """
    path = Path(file_path)
    mode = 'rb' if binary else 'r'

    try:
        if binary:
            with open(path, mode) as f:
                return f.read()
        else:
            with open(path, mode, encoding=encoding, errors=errors) as f:
                return f.read()
    except FileNotFoundError:
        print_error(f"File not found: {file_path}", exit_code=1)
        return b'' if binary else ''  # Never reached due to exit


def safe_write_file(
    file_path: Union[str, Path],
    content: Union[str, bytes],
    encoding: str = 'utf-8',
    errors: str = 'replace',
    atomic: bool = True,
    create_dirs: bool = True
) -> bool:
    """
    Safely write a file with optional atomic writes.

    Args:
        file_path: Path to the file to write
        content: Content to write (str or bytes)
        encoding: Text encoding (default: utf-8). Ignored if content is bytes
        errors: How to handle encoding errors (default: 'replace')
        atomic: If True, write to temp file then rename (safer, default: True)
        create_dirs: If True, create parent directories if needed (default: True)

    Returns:
        True if write succeeded, False otherwise

    Example:
        safe_write_file("output.txt", "Hello world")
        safe_write_file("data.bin", binary_data, atomic=True)
    """
    path = Path(file_path)

    # Create parent directories if needed
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    is_binary = isinstance(content, bytes)

    try:
        if atomic:
            # Write to temporary file, then rename (atomic on POSIX, mostly atomic on Windows)
            dir_path = path.parent
            fd, temp_path = tempfile.mkstemp(dir=dir_path, prefix='.tmp_')
            try:
                if is_binary:
                    with os.fdopen(fd, 'wb') as f:
                        f.write(content)
                else:
                    with os.fdopen(fd, 'w', encoding=encoding, errors=errors) as f:
                        f.write(content)
                # Atomic rename
                shutil.move(temp_path, path)
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        else:
            # Direct write
            if is_binary:
                with open(path, 'wb') as f:
                    f.write(content)
            else:
                with open(path, 'w', encoding=encoding, errors=errors) as f:
                    f.write(content)

        print_info(f"Wrote: {file_path}")
        return True

    except Exception as e:
        print_error(f"Failed to write {file_path}: {e}")
        return False


def ensure_directory(
    dir_path: Union[str, Path],
    exist_ok: bool = True
) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        dir_path: Path to the directory
        exist_ok: If True, don't raise error if directory exists (default: True)

    Returns:
        Path object for the directory

    Example:
        ensure_directory("output/subdir")
        ensure_directory(project_dir / "cache" / "temp")
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=exist_ok)
    print_info(f"Directory: {path}")
    return path


def find_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True,
    include_hidden: bool = False
) -> List[Path]:
    """
    Find files matching a glob pattern.

    Args:
        directory: Directory to search in
        pattern: Glob pattern (default: "*" for all files)
        recursive: If True, search recursively (default: True)
        include_hidden: If True, include hidden files (default: False)

    Returns:
        List of matching file paths

    Example:
        # Find all Python files
        files = find_files("src", "*.py")

        # Find all YAML files in current directory only
        files = find_files(".", "*.yaml", recursive=False)

        # Find all test files
        files = find_files("tests", "*test*.py")
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        print_warn(f"Directory not found: {directory}")
        return []

    if recursive:
        # Use ** prefix for recursive search
        pattern = f"**/{pattern}" if not pattern.startswith("**/") else pattern
        matches = dir_path.glob(pattern)
    else:
        matches = dir_path.glob(pattern)

    result = []
    for path in matches:
        # Handle hidden files
        if not include_hidden and path.name.startswith('.'):
            continue
        if path.is_file():
            result.append(path)

    print_info(f"Found {len(result)} files matching '{pattern}' in {directory}")
    return sorted(result)


def get_file_hash(
    file_path: Union[str, Path],
    algorithm: str = 'sha256',
    chunk_size: int = 8192
) -> Optional[str]:
    """
    Calculate hash of a file for caching/integrity checks.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (default: sha256). Options: md5, sha1, sha256, sha512
        chunk_size: Read chunk size in bytes (default: 8192)

    Returns:
        Hex digest of file hash, or None if file doesn't exist

    Example:
        # Get SHA256 hash (default)
        hash = get_file_hash("large_file.mp4")

        # Get MD5 hash for quick comparison
        hash = get_file_hash("file.txt", algorithm="md5")
    """
    path = Path(file_path)

    if not path.exists():
        print_warn(f"File not found for hashing: {file_path}")
        return None

    try:
        hasher = hashlib.new(algorithm)

        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)

        return hasher.hexdigest()

    except Exception as e:
        print_error(f"Failed to hash {file_path}: {e}")
        return None


# Export the new functions
__all__ = [
    'print_header', 'print_ok', 'print_warn', 'print_error', 'print_info',
    'set_verbosity', 'get_verbosity',
    'progress_bar', 'track_progress', 'spinner',
    'load_config_for_script', 'add_config_argument',
    'add_project_argument', 'add_verbose_argument', 'add_quiet_argument',
    'add_standard_arguments',
    # File operation utilities
    'safe_read_file', 'safe_write_file', 'ensure_directory',
    'find_files', 'get_file_hash',
]
