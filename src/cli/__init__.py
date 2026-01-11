"""
CLI package - Command-line interface utilities for the matcher pipeline.

Extracted from main.py (Jan 2026) to improve maintainability.

Modules:
- args.py: Argument parsing
- environment.py: Environment and path setup
- config_utils.py: Configuration loading helpers
- logging_setup.py: Dual-file logging configuration
- interactive.py: Interactive prompts for voiceover selection
"""

from .args import parse_arguments
from .environment import (
    load_environment,
    strip_extended_path_prefix,
    INSTALL_DIR,
)
from .config_utils import (
    validate_root_directories,
    make_paths_project_relative,
    load_project_config,
    merge_config,
    validate_config_at_startup,
)
from .logging_setup import setup_logging
from .interactive import find_voiceover_interactive

__all__ = [
    # Argument parsing
    'parse_arguments',

    # Environment
    'load_environment',
    'strip_extended_path_prefix',
    'INSTALL_DIR',

    # Config utilities
    'validate_root_directories',
    'make_paths_project_relative',
    'load_project_config',
    'merge_config',
    'validate_config_at_startup',

    # Logging
    'setup_logging',

    # Interactive
    'find_voiceover_interactive',
]
