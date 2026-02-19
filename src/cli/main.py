"""
Unified CLI entry point for matcher pipeline.

Provides subcommands: run, validate-config, cache-clean, project-create
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Import the main pipeline function
from main import main as run_pipeline


def create_parser() -> argparse.ArgumentParser:
    """Create the main parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Matcher - Voiceover-to-Footage Matching Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    matcher run --voiceover script.srt --project "MyProject"
    matcher validate-config
    matcher cache-clean --project "MyProject"
    matcher project-create "My New Project"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # === run subcommand ===
    run_parser = subparsers.add_parser(
        'run',
        help='Run the matching pipeline',
        description='Run the voiceover-to-footage matching pipeline'
    )
    _add_run_arguments(run_parser)

    # === validate-config subcommand ===
    validate_parser = subparsers.add_parser(
        'validate-config',
        help='Validate configuration file',
        description='Validate config.yaml and check for errors'
    )
    validate_parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    validate_parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    # === cache-clean subcommand ===
    cache_parser = subparsers.add_parser(
        'cache-clean',
        help='Clean cache directories',
        description='Clean various cache directories to free space'
    )
    cache_parser.add_argument(
        '--project',
        type=str,
        help='Project directory to clean caches for'
    )
    cache_parser.add_argument(
        '--all',
        action='store_true',
        help='Clean all caches (global and project-specific)'
    )
    cache_parser.add_argument(
        '--transcriptions',
        action='store_true',
        help='Clean transcription cache only'
    )
    cache_parser.add_argument(
        '--embeddings',
        action='store_true',
        help='Clean embeddings cache only'
    )
    cache_parser.add_argument(
        '--caption',
        action='store_true',
        help='Clean caption cache only'
    )
    cache_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )

    # === project-create subcommand ===
    create_parser = subparsers.add_parser(
        'project-create',
        help='Create a new project',
        description='Create a new project directory with standard structure'
    )
    create_parser.add_argument(
        'name',
        type=str,
        help='Project name'
    )
    create_parser.add_argument(
        '--client',
        type=str,
        help='Client name'
    )
    create_parser.add_argument(
        '--doc-url',
        type=str,
        help='Google Doc URL containing voiceover script'
    )
    create_parser.add_argument(
        '--voiceover',
        type=str,
        help='Path to voiceover file (SRT, MP3, WAV, MP4)'
    )
    create_parser.add_argument(
        '--template',
        type=str,
        choices=['fast', 'quality', 'debug'],
        help='Config template to use'
    )

    return parser


def _add_run_arguments(parser: argparse.ArgumentParser):
    """Add all the run arguments (mirrors main.py arguments)."""
    parser.add_argument(
        '--voiceover', '-v',
        type=str,
        help='Path to voiceover file (SRT, MP3, WAV, MP4)'
    )
    parser.add_argument(
        '--keywords', '-k',
        type=int,
        default=None,
        help='Number of keywords to extract'
    )
    parser.add_argument(
        '--project', '-p',
        type=str,
        default=None,
        help='Project directory'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--match-only',
        action='store_true',
        help='Skip download, only match existing footage'
    )
    parser.add_argument(
        '--output-only',
        action='store_true',
        help='Regenerate OTIO/EDL/XML only'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume interrupted pipeline run'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Force fresh start'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview pipeline execution plan'
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run in non-interactive mode'
    )


def main():
    """Main entry point for the matcher CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'run':
        _handle_run(args)
    elif args.command == 'validate-config':
        _handle_validate_config(args)
    elif args.command == 'cache-clean':
        _handle_cache_clean(args)
    elif args.command == 'project-create':
        _handle_project_create(args)


def _handle_run(args):
    """Handle the 'run' subcommand."""
    # Convert args namespace to list of strings for main.py
    sys.argv = ['matcher']

    if args.voiceover:
        sys.argv.extend(['--voiceover', args.voiceover])
    if args.keywords is not None:
        sys.argv.extend(['--keywords', str(args.keywords)])
    if args.project:
        sys.argv.extend(['--project', args.project])
    if args.config and args.config != 'config.yaml':
        sys.argv.extend(['--config', args.config])
    if args.match_only:
        sys.argv.append('--match-only')
    if args.output_only:
        sys.argv.append('--output-only')
    if args.resume:
        sys.argv.append('--resume')
    if args.fresh:
        sys.argv.append('--fresh')
    if args.dry_run:
        sys.argv.append('--dry-run')
    if args.non_interactive:
        sys.argv.append('--non-interactive')

    run_pipeline()


def _handle_validate_config(args):
    """Handle the 'validate-config' subcommand."""
    from src.cli.args import parse_arguments
    import sys

    # Save original argv
    original_argv = sys.argv

    # Set up argv for validation
    sys.argv = ['matcher', '--validate-config', '--config', args.config]

    if args.json:
        sys.argv.append('--validate-config-json')

    try:
        parse_arguments()
    except SystemExit:
        pass
    finally:
        sys.argv = original_argv


def _handle_cache_clean(args):
    """Handle the 'cache-clean' subcommand."""
    from scripts.cleanup_project import cleanup_project

    if not args.project and not args.all:
        print("Error: --project or --all is required")
        sys.exit(1)

    project_dir = Path(args.project) if args.project else None

    # Determine cleanup level
    if args.transcriptions:
        level = 'transcriptions'
    elif args.embeddings:
        level = 'embeddings'
    elif args.caption:
        level = 'caption'
    else:
        level = 'cache'

    try:
        cleanup_project(project_dir, level=level, dry_run=args.dry_run)
    except Exception as e:
        print(f"Error during cache cleanup: {e}")
        sys.exit(1)


def _handle_project_create(args):
    """Handle the 'project-create' subcommand."""
    import subprocess

    cmd = [sys.executable, 'scripts/setup_project.py', args.name]

    if args.client:
        cmd.extend(['--client', args.client])
    if args.doc_url:
        cmd.extend(['--doc-url', args.doc_url])
    if args.voiceover:
        cmd.extend(['--voiceover', args.voiceover])
    if args.template:
        cmd.extend(['--template', args.template])

    try:
        result = subprocess.run(cmd, text=True, encoding='utf-8', errors='replace')
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error creating project: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
