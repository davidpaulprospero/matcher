#!/usr/bin/env python3
"""
Config Diff CLI Tool

Compares two configuration files and shows differences in a human-readable format.

Usage:
    python scripts/config_diff.py config1.yaml config2.yaml
    python scripts/config_diff.py --current config.yaml     # Compare vs runtime config
    python scripts/config_diff.py config1.yaml config2.yaml --json  # JSON output

Exit codes:
    0 - Configs are identical
    1 - Configs are different
    2 - Error (file not found, parse error, etc.)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
sys.path.insert(0, str(project_root))

# Import standardized output functions
from script_utils import print_ok, print_warn, print_error, print_info, print_header

from src.config import Config
from src.config.utils import config_diff


def load_config_from_file(file_path: str) -> Config:
    """Load a config file and return Config object."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    return Config.from_yaml(str(path))


def get_runtime_config() -> Config:
    """Get the current runtime config (loads default config.yaml)."""
    return Config.from_yaml("config.yaml")


def format_diff_entry(path: str, values: Dict[str, Any]) -> str:
    """Format a single diff entry for display."""
    old_val = values.get("old")
    new_val = values.get("new")

    # Format None as "<not set>"
    old_str = "<not set>" if old_val is None else repr(old_val)
    new_str = "<not set>" if new_val is None else repr(new_val)

    return f"  {path}:\n    old: {old_str}\n    new: {new_str}"


def print_diff(diffs: Dict[str, Dict[str, Any]], output_format: str = "text") -> None:
    """Print the diff in the specified format."""
    if output_format == "json":
        print(json.dumps(diffs, indent=2, default=str))
        return

    if not diffs:
        print("Configs are identical - no differences found.")
        return

    print(f"\nFound {len(diffs)} difference(s):\n")

    # Sort by path for consistent output
    for path in sorted(diffs.keys()):
        print(format_diff_entry(path, diffs[path]))
        print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two config files and show differences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/config_diff.py config1.yaml config2.yaml
    python scripts/config_diff.py old.yaml new.yaml --json
    python scripts/config_diff.py config.yaml --current  # Compare vs runtime

Exit codes:
    0 - Configs are identical
    1 - Configs are different
    2 - Error
        """
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="Config files to compare (2 files or 1 with --current)"
    )

    parser.add_argument(
        "--current", "-c",
        action="store_true",
        help="Compare file(s) against current runtime config (config.yaml)"
    )

    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show additional details"
    )

    args = parser.parse_args()

    # Change to project root for config loading
    os.chdir(project_root)

    # Validate arguments
    if args.current:
        # Comparing against runtime config
        if len(args.files) != 1:
            parser.error("When using --current, provide exactly one config file")

        try:
            file_config = load_config_from_file(args.files[0])
            runtime_config = get_runtime_config()
            diffs = config_diff(file_config, runtime_config)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 2
    else:
        # Comparing two files
        if len(args.files) != 2:
            parser.error("Provide exactly two config files to compare")

        try:
            config_a = load_config_from_file(args.files[0])
            config_b = load_config_from_file(args.files[1])
            diffs = config_diff(config_a, config_b)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 2

    # Print diffs
    print_diff(diffs, output_format="json" if args.json else "text")

    if args.verbose and diffs:
        # Show summary
        print(f"\nSummary:")
        print(f"  Total differences: {len(diffs)}")

    # Return exit code based on whether configs are different
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
