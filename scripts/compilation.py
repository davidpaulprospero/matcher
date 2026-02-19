#!/usr/bin/env python3
"""
Compilation Video Pipeline

Generates compilation videos from topic/keywords without voiceover.
Creates N alternative tracks with unique clips filling target duration.

Usage:
    python compilation.py --topic "funny cats" --keywords "cat fails,cat jumps"
    python compilation.py --topic "nature documentary" --duration 300 --project "E:\\Edit"
    python compilation.py --topic "viral moments" --num-tracks 2 --duration 120
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from script_utils import print_ok, print_warn, print_error, print_info, print_header, set_verbosity

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
# Also add src for backwards compatibility
sys.path.insert(0, str(project_root / 'src'))

# Change to project root so relative paths work correctly
os.chdir(project_root)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate compilation videos from topic/keywords',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compilation.py --topic "funny cats" --keywords "cat fails,cat jumps"
  python compilation.py --topic "nature documentary" --duration 300
  python compilation.py --topic "viral moments" --num-tracks 2 --project "E:\\Edit"
        """
    )

    parser.add_argument(
        '--topic', '-t',
        required=True,
        help='Main topic for the compilation'
    )

    parser.add_argument(
        '--keywords', '-k',
        default='',
        help='Initial keywords (comma-separated)'
    )

    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=540,
        help='Target duration in seconds (default: 540 = 9 minutes)'
    )

    parser.add_argument(
        '--num-tracks', '-n',
        type=int,
        default=4,
        help='Number of alternative tracks (default: 4)'
    )

    parser.add_argument(
        '--project', '-p',
        default=None,
        help='Project/output directory (default: ./output/<topic>_<timestamp>)'
    )

    parser.add_argument(
        '--config', '-c',
        default=None,
        help='Path to compilation_config.yaml'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set verbosity level based on --verbose flag
    set_verbosity(2 if args.verbose else 1)

    # Parse keywords
    keywords = []
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(',') if k.strip()]

    # If no keywords provided, use topic as initial keyword
    if not keywords:
        keywords = [args.topic]

    # Set up project directory
    if args.project:
        project_dir = Path(args.project)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in args.topic)[:30]
        project_dir = Path.cwd() / 'output' / f"{safe_topic}_{timestamp}"

    project_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    from src.config import load_config, get_config
    from src.compilation.orchestrator import (
        CompilationOrchestrator,
        load_compilation_config,
    )
    from src.compilation.state import CompilationState

    # Load main config (for rate limiting, download settings, etc.)
    config_path = args.config if args.config else None
    if config_path:
        config = load_config(Path(config_path))
    else:
        config = get_config()

    # Load compilation-specific config
    compilation_config_path = Path(args.config) if args.config else None
    compilation_config = load_compilation_config(compilation_config_path)

    # Override with CLI args
    compilation_config['target_duration'] = args.duration
    compilation_config['num_tracks'] = args.num_tracks

    # Create initial state
    state = CompilationState(
        topic=args.topic,
        keywords=keywords,
        target_duration=args.duration,
        num_tracks=args.num_tracks,
        project_dir=str(project_dir),
        config_path=args.config,
    )

    # Run pipeline
    orchestrator = CompilationOrchestrator(config, compilation_config)

    try:
        success = orchestrator.run(state)

        if success:
            print_ok("")
            print_ok("Pipeline completed successfully!")
            print_ok(f"Output directory: {project_dir}")
            return 0
        else:
            print_error("Pipeline failed", exit_code=1)
            return 1

    except KeyboardInterrupt:
        print_warn("Interrupted by user")
        return 130

    except Exception as e:
        print_error(f"Pipeline error: {e}", exit_code=1)
        return 1


if __name__ == '__main__':
    sys.exit(main())
