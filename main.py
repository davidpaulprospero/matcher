#!/usr/bin/env python3
"""
Voiceover-Matcher: End-to-End Documentary Footage Pipeline v3.0

One command to go from voiceover → matched timeline:
1. Extract keywords from voiceover (LLM)
2. Download footage from YouTube + Pexels + Pixabay
3. Transcribe & index videos (parallel + delta-aware)
4. Match to voiceover segments (with confidence enforcement)
5. Output DaVinci Resolve timeline (multiple styles)

Usage:
    python main.py --voiceover script.srt
    python main.py --voiceover script.srt --keywords 30
    python main.py --project "E:\\Projects\\MyDoc" --voiceover voiceover.srt
    python main.py --resume                    # Resume interrupted run
    python main.py --fresh                     # Force fresh start
    python main.py --use-keywords              # Use most recent saved keywords
    python main.py --use-keywords mypreset     # Use specific keyword preset
    python main.py --save-keywords             # Save keywords after extraction
    python main.py --list-keywords             # List saved keyword presets
    python main.py --match-only                # Skip download, just match existing
    python main.py --config custom_config.yaml # Use custom config file
"""
from __future__ import annotations

import os
import sys

# Suppress FFmpeg H.264 decoder warnings from OpenCV (must be set before cv2 import)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from typing import Optional

# =============================================================================
# PATH SETUP
# =============================================================================

# Determine install directory (where this script lives)
INSTALL_DIR = Path(__file__).parent.resolve()

# Add src to path
sys.path.insert(0, str(INSTALL_DIR / 'src'))

# Project directory (set later via --project argument)
PROJECT_DIR = None

# =============================================================================
# IMPORTS FROM CLI PACKAGE
# =============================================================================

from src.cli import (
    parse_arguments,
    load_environment,
    validate_root_directories,
    load_project_config,
    validate_config_at_startup,
    setup_logging,
    find_voiceover_interactive,
)
from src.config import (
    Config, load_config, get_config, set_config,
    ensure_dirs,
)
from src.checkpoint import KeywordManager

# Global config instance - loaded at startup
_config: Optional[Config] = None


def get_pipeline_config() -> Config:
    """Get the pipeline configuration (loads if not already loaded)"""
    global _config
    if _config is None:
        _config = get_config()
    return _config


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    global PROJECT_DIR, _config

    args = parse_arguments()

    # Determine project directory
    if args.project:
        PROJECT_DIR = Path(args.project).resolve()
    else:
        PROJECT_DIR = Path.cwd()

    # Load environment variables
    print("\n  Loading environment...")
    load_environment(PROJECT_DIR)

    # Load configuration
    print(f"  Loading configuration from {args.config}...")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try project dir first, then install dir
        if (PROJECT_DIR / args.config).exists():
            config_path = PROJECT_DIR / args.config
        elif (INSTALL_DIR / args.config).exists():
            config_path = INSTALL_DIR / args.config

    if args.project:
        config = load_project_config(PROJECT_DIR, config_path)
    else:
        config = load_config(str(config_path))

    _config = config
    set_config(config)

    # Validate root directories (E:/v, E:/i, etc.) and create them if needed
    validate_root_directories(config)

    # Apply --non-interactive flag
    if hasattr(args, 'non_interactive') and args.non_interactive:
        config.enhanced.non_interactive = True

    # Setup logging with dual log files (normal + verbose)
    if hasattr(args, 'project') and args.project:
        log_output_dir = Path(args.project)
    else:
        log_output_dir = Path(config.output.output_dir)

    logger = setup_logging(config, output_dir=log_output_dir)

    # Print log file locations
    if hasattr(logger, 'log_paths'):
        print(f"\n  📝 Log files:")
        print(f"    Normal:  {Path(logger.log_paths['normal']).name}")
        print(f"    Verbose: {Path(logger.log_paths['verbose']).name}")

    # Validate config
    if args.validate_config:
        print("\n  Configuration Validation:")
        errors = config.validate()
        if errors:
            for e in errors:
                print(f"    ⚠ {e}")
            sys.exit(1)
        else:
            print("    ✓ Configuration is valid")
            sys.exit(0)

    # Validate at startup
    if not validate_config_at_startup(config):
        print("\n  ⚠ Configuration has critical errors. Fix and retry.")
        sys.exit(1)

    # Ensure directories exist
    ensure_dirs(config)

    # Handle --list-keywords
    if getattr(args, 'list_keywords', False):
        keyword_manager = KeywordManager(PROJECT_DIR)
        if keyword_manager.has_presets():
            print("\n" + keyword_manager.get_summary())
        else:
            print("\n  No saved keyword presets found.")
            print(f"  Use --save-keywords to save keywords after extraction.")
        sys.exit(0)

    # Check for voiceover file
    if not args.voiceover:
        args.voiceover = find_voiceover_interactive(PROJECT_DIR)
        if not args.voiceover:
            sys.exit(1)

    # Resolve voiceover path
    vo_path = Path(args.voiceover)
    if not vo_path.is_absolute():
        vo_path = PROJECT_DIR / vo_path

    if not vo_path.exists():
        print(f"\n  Error: Voiceover file not found: {vo_path}")
        sys.exit(1)

    # Run pipeline using modular architecture
    from src.pipeline import create_default_pipeline, create_match_only_pipeline

    # Handle keyword presets/selection
    keyword_manager = KeywordManager(PROJECT_DIR)
    use_keywords = getattr(args, 'use_keywords', None)
    save_keywords = getattr(args, 'save_keywords', None)

    # Create pipeline
    audio_first = getattr(config.download.audio_first, 'enabled', False) if hasattr(config.download, 'audio_first') else False

    if args.match_only:
        pipeline = create_match_only_pipeline(config, PROJECT_DIR)
    else:
        pipeline = create_default_pipeline(config, PROJECT_DIR, audio_first_mode=audio_first)

    # Initialize state
    pipeline.state.voiceover_path = str(vo_path)
    pipeline.state.num_keywords = args.keywords
    pipeline.state.topic_context = ""

    # Handle keyword preset loading
    if use_keywords:
        loaded_keywords = keyword_manager.get_preset(use_keywords)
        if loaded_keywords:
            pipeline.state.keywords = loaded_keywords['keywords']
            pipeline.state.topic_context = loaded_keywords.get('topic', '')
            print(f"\n  ✓ Loaded keyword preset: {use_keywords}")
            print(f"    Keywords ({len(pipeline.state.keywords)}): {', '.join(pipeline.state.keywords[:5])}")
            if len(pipeline.state.keywords) > 5:
                print(f"    ... and {len(pipeline.state.keywords) - 5} more")

    # Handle fresh start
    if getattr(args, 'fresh', False):
        if pipeline.checkpoint.exists():
            import shutil
            checkpoint_file = pipeline.checkpoint.checkpoint_path
            backup_file = checkpoint_file.with_suffix('.backup.json')
            shutil.copy2(checkpoint_file, backup_file)
            checkpoint_file.unlink()
            print(f"\n  🔄 Fresh start: Deleted checkpoint (backup saved)")

    # Handle force rematch
    if getattr(args, 'force_rematch', False):
        pipeline.checkpoint.mark_stage_incomplete('MATCH')
        print(f"\n  🔄 Force rematch: Marked MATCH stage incomplete")

    # Handle refresh entities
    if getattr(args, 'refresh_entities', False):
        pipeline.checkpoint.refresh_entities = True
        print(f"\n  🔄 Refresh entities: Will re-download entity images")

    # Handle save matching fixtures
    if getattr(args, 'save_matching_fixtures', None):
        pipeline.state.save_matching_fixtures = args.save_matching_fixtures

    # Run the pipeline
    success = pipeline.run(resume=args.resume)

    # Save keywords if requested
    if save_keywords and pipeline.state.keywords:
        keyword_manager.save_keywords(
            name=save_keywords,
            keywords=pipeline.state.keywords,
            topic_context=pipeline.state.topic_context or "",
            entities=pipeline.state.entities or []
        )
        print(f"\n  ✓ Saved keyword preset: {save_keywords}")

    if not success:
        print("\n  ❌ Pipeline failed")
        sys.exit(1)
    else:
        print("\n  ✅ Pipeline completed successfully")


if __name__ == '__main__':
    main()
