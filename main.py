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


def _preload_cached_data_for_match_only(pipeline, config):
    """
    Preload cached data for match-only mode when checkpoint data is incomplete.

    This handles the case where the pipeline completed but the checkpoint wasn't
    updated properly. It loads embeddings and text_metadata from disk cache.
    """
    import logging
    import numpy as np
    logger = logging.getLogger(__name__)

    state = pipeline.state
    cache_dir = Path(config.cache.cache_dir)

    # 1. Load audio files from disk if not in state
    if not state.downloaded_audio and not state.downloaded_videos:
        from src.state import AudioDownload
        root_dir = getattr(config.download, 'root_dir', None)
        if root_dir:
            root_path = Path(root_dir)
            # Find project-specific subdirectory
            for subdir in root_path.iterdir():
                if subdir.is_dir():
                    audio_files = []
                    for audio_dir in subdir.rglob('*_audio'):
                        for mp3 in audio_dir.glob('*.mp3'):
                            audio_files.append(AudioDownload(
                                file=str(mp3),
                                url="",
                                video_id=mp3.stem,
                                title=mp3.stem,
                                duration=0.0,
                                keyword=audio_dir.name.replace('_audio', '').replace('_l', '').replace('_m', '')
                            ))
                    if audio_files:
                        state.downloaded_audio = audio_files
                        logger.info(f"Preloaded {len(audio_files)} audio files from disk")
                        break

    # 2. Load transcripts from cache
    transcripts_dir = cache_dir / 'transcriptions'
    if transcripts_dir.exists() and not state.transcripts:
        import json
        transcripts = {}
        for json_file in transcripts_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    segments = None
                    if isinstance(data, list):
                        segments = data
                    elif isinstance(data, dict) and 'segments' in data:
                        segments = data['segments']

                    if segments:
                        # Extract actual video/audio path from segment data
                        # (cache files are named by hash, but contain source_file)
                        video_path = None
                        for seg in segments:
                            if isinstance(seg, dict):
                                video_path = seg.get('source_file') or seg.get('video_path')
                                if video_path:
                                    break

                        # Use actual path as key (fall back to filename if not found)
                        key = video_path if video_path else json_file.stem
                        transcripts[key] = segments
            except Exception:
                pass
        if transcripts:
            state.transcripts = transcripts
            logger.info(f"Preloaded {len(transcripts)} transcripts from cache")

    # 3. Embeddings are loaded during TranscribeStage.restore() via _compute_embeddings
    # NOTE: We intentionally do NOT preload text_metadata or embeddings from the embedding cache
    # here because the embedding cache files don't contain actual video paths - they only
    # have hash-based filenames. Loading them here would set incorrect video_path values
    # and prevent proper B-roll flag propagation. The TranscribeStage._rebuild_text_metadata()
    # method correctly extracts video paths from transcript cache files.
    # See: https://github.com/project/issues/XXX - B-roll propagation fix


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

    # Apply caption-first mode CLI flags
    if getattr(args, 'caption_first', False):
        config.download.caption_first.enabled = True
        print("  Caption-first mode enabled via --caption-first")

    if getattr(args, 'caption_language', None):
        config.download.caption_first.preferred_language = args.caption_language
        print(f"  Caption language set to '{args.caption_language}' via --caption-language")

    if getattr(args, 'no_caption_fallback', False):
        config.download.caption_first.fallback_to_transcription = False
        print("  Caption fallback disabled via --no-caption-fallback")

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
    from src.agents import ResilientRunner, HealingOrchestrator, HealingStrategy

    # Handle keyword presets/selection
    keyword_manager = KeywordManager(PROJECT_DIR)
    use_keywords = getattr(args, 'use_keywords', None)
    save_keywords = getattr(args, 'save_keywords', None)

    # Create pipeline
    audio_first = getattr(config.download.audio_first, 'enabled', False) if hasattr(config.download, 'audio_first') else False

    if args.match_only:
        pipeline = create_match_only_pipeline(config, PROJECT_DIR)
        # Match-only requires checkpoint data - force resume mode
        if not args.resume:
            args.resume = True

        # Validate checkpoint exists
        if not pipeline.checkpoint.exists():
            print("\n  Error: --match-only requires existing checkpoint data.")
            print("  Run the full pipeline first to download and transcribe videos.")
            sys.exit(1)

        # Check checkpoint has sufficient data (at least SCENE_DETECTION completed)
        checkpoint_data = pipeline.checkpoint.load()
        if checkpoint_data:
            last_stage = checkpoint_data.last_completed_stage
            if last_stage:
                from src.checkpoint import STAGE_ORDER
                try:
                    last_idx = STAGE_ORDER.index(last_stage)
                    scene_idx = STAGE_ORDER.index('SCENE_DETECTION')
                    if last_idx < scene_idx:
                        print(f"\n  Error: Checkpoint incomplete for match-only mode.")
                        print(f"  Last completed: {last_stage}")
                        print(f"  Required: SCENE_DETECTION (run full pipeline first)")
                        sys.exit(1)

                    # If pipeline completed past SCENE_DETECTION, reset so
                    # MATCH, BROLL_MATCH, and OUTPUT can re-run
                    if last_stage in ('MATCH', 'BROLL_MATCH', 'DOWNLOAD_SEGMENTS', 'OUTPUT'):
                        pipeline.checkpoint.data.last_completed_stage = 'SCENE_DETECTION'
                        pipeline.checkpoint._atomic_save()  # Save without stage update
                        print(f"  Reset checkpoint from {last_stage} to SCENE_DETECTION for re-matching")
                except ValueError:
                    pass  # Unknown stage, let it proceed

        # Print confirmation
        print(f"\n  Match-only mode: Using checkpoint data from previous run")
        print(f"  Will skip: ANALYZE through SCENE_DETECTION")
        print(f"  Will run: MATCH, OUTPUT")

        # Preload cached data for match-only mode (fallback when checkpoint is incomplete)
        _preload_cached_data_for_match_only(pipeline, config)
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

    # Setup self-healing if enabled
    healing_config = getattr(config, 'healing', None)
    healing_enabled = healing_config and getattr(healing_config, 'enabled', True)
    orchestrator = None
    runner = None

    if healing_enabled:
        # Build strategy from config
        strategy_name = getattr(healing_config, 'strategy', 'conservative')
        strategy_map = {
            'aggressive': HealingStrategy.aggressive,
            'conservative': HealingStrategy.conservative,
            'interactive': HealingStrategy.interactive,
            'minimal': HealingStrategy.minimal,
        }
        strategy = strategy_map.get(strategy_name, HealingStrategy.conservative)()

        # Override strategy settings from config
        if hasattr(healing_config, 'max_attempts_per_stage'):
            strategy.max_attempts_per_stage = healing_config.max_attempts_per_stage
        if hasattr(healing_config, 'max_total_heals'):
            strategy.max_total_heals = healing_config.max_total_heals
        if hasattr(healing_config, 'heal_delay'):
            strategy.heal_delay = healing_config.heal_delay

        orchestrator = HealingOrchestrator(config, PROJECT_DIR, strategy)
        runner = ResilientRunner(config, PROJECT_DIR, orchestrator=orchestrator)

        print(f"\n  🛡️  Self-healing enabled: strategy={strategy_name}, max_attempts={strategy.max_attempts_per_stage}")

        # Run preflight checks if configured
        if getattr(healing_config, 'run_preflight', True):
            print(f"  Running preflight checks...")
            issues = orchestrator.run_preflight(pipeline.state)
            if issues:
                print(f"  Found {len(issues)} preflight issue(s)")
                if getattr(healing_config, 'auto_fix_preflight', True):
                    fixed_count, remaining_count = orchestrator.fix_preflight_issues(issues, pipeline.state)
                    if fixed_count:
                        print(f"  ✓ Auto-fixed {fixed_count} issue(s)")
                    if remaining_count:
                        print(f"  ⚠ {remaining_count} issue(s) could not be auto-fixed")
            else:
                print(f"  ✓ All preflight checks passed")

    # Run the pipeline
    if runner:
        success = runner.run_pipeline(pipeline, resume=args.resume)
    else:
        success = pipeline.run(resume=args.resume)

    # Print healing report if enabled
    if orchestrator and getattr(healing_config, 'print_report', True):
        orchestrator.print_report()

    # Save keywords if requested
    if save_keywords and pipeline.state.keywords:
        keyword_manager.save_keywords(
            name=save_keywords,
            keywords=pipeline.state.keywords,
            topic_context=pipeline.state.topic_context or "",
            entities=pipeline.state.extracted_entities or []
        )
        print(f"\n  ✓ Saved keyword preset: {save_keywords}")

    # Handle --export-metrics flag (export rate limit metrics to JSON)
    export_metrics_path = getattr(args, 'export_metrics', None)
    if export_metrics_path:
        _export_rate_limit_metrics(export_metrics_path, config, PROJECT_DIR)

    # Handle --export-caption-metrics flag (export caption fetch metrics to JSON)
    export_caption_metrics_path = getattr(args, 'export_caption_metrics', None)
    if export_caption_metrics_path:
        _export_caption_metrics(export_caption_metrics_path, config, PROJECT_DIR, checkpoint_manager)

    if not success:
        print("\n  ❌ Pipeline failed")
        sys.exit(1)
    else:
        print("\n  ✅ Pipeline completed successfully")


def _export_rate_limit_metrics(path: str, config, project_dir: Path):
    """Export rate limit metrics to JSON file.

    Loads metrics from the download checkpoint and exports to the specified path.

    Args:
        path: Output file path for JSON export
        config: Pipeline config
        project_dir: Project directory
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        # Load download checkpoint to get rate limit metrics
        cache_dir = Path(getattr(config.cache, 'cache_dir', '.cache'))
        if not cache_dir.is_absolute():
            cache_dir = project_dir / cache_dir

        checkpoint_file = cache_dir / "download_checkpoint.json"

        if not checkpoint_file.exists():
            print(f"\n  ⚠ No download checkpoint found at {checkpoint_file}")
            print("  Cannot export metrics - no download data available.")
            return

        import json
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        # Extract rate_limit_metrics from checkpoint
        rate_limit_metrics_data = checkpoint_data.get('rate_limit_metrics')
        if not rate_limit_metrics_data:
            print("\n  ⚠ No rate limit metrics in checkpoint.")
            print("  Run a download stage to collect metrics.")
            return

        # Reconstruct RateLimitMetrics from checkpoint data
        metrics = RateLimitMetrics.from_dict(rate_limit_metrics_data)

        # Resolve output path
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = project_dir / output_path

        # Export to JSON with config snapshot
        metrics.export_to_json_file(str(output_path), config)

        print(f"\n  📊 Exported rate limit metrics to {output_path}")
        print(f"     Downloads: {metrics.total_downloads} ({metrics.success_rate}% success)")
        print(f"     Rate limits: {metrics.rate_limit_events} events")

    except Exception as e:
        logger.warning(f"Failed to export rate limit metrics: {e}")
        print(f"\n  ⚠ Failed to export metrics: {e}")


def _export_caption_metrics(path: str, config, project_dir: Path, checkpoint_manager):
    """Export caption fetch metrics to JSON file (US-004 Sprint 7).

    Loads caption metrics from the pipeline checkpoint and exports to the specified path.

    Args:
        path: Output file path for JSON export
        config: Pipeline config
        project_dir: Project directory
        checkpoint_manager: CheckpointManager instance with loaded checkpoint
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        from src.caption_fetcher import CaptionMetrics

        # Try to get caption metrics from checkpoint manager's stages data
        caption_data = None
        if checkpoint_manager:
            stages = getattr(checkpoint_manager, '_stages', {})
            caption_data = stages.get('CAPTION', {})

        if not caption_data:
            # Fallback: try to load from checkpoint file directly
            checkpoint_file = project_dir / "checkpoint.json"
            if checkpoint_file.exists():
                import json
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                caption_data = checkpoint.get('stages', {}).get('CAPTION', {})

        if not caption_data:
            print("\n  ⚠ No caption stage data found in checkpoint.")
            print("  Run a CAPTION stage to collect metrics.")
            return

        # Extract caption_metrics from stage data
        caption_metrics_data = caption_data.get('caption_metrics')
        if not caption_metrics_data:
            print("\n  ⚠ No caption metrics in checkpoint.")
            print("  Run caption-first mode to collect metrics.")
            return

        # Reconstruct CaptionMetrics from checkpoint data
        metrics = CaptionMetrics.from_dict(caption_metrics_data)

        # Resolve output path
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = project_dir / output_path

        # Get video count from caption data
        video_count = caption_data.get('total_videos', len(caption_data.get('caption_results', {})))

        # Export to JSON with run metadata
        metrics.export_json(
            str(output_path),
            project_path=str(project_dir),
            config=config,
            video_count=video_count
        )

        print(f"\n  📊 Exported caption metrics to {output_path}")
        print(f"     Fetch attempts: {metrics.fetch_attempts}, Successes: {metrics.successes}")
        print(f"     Success rate: {metrics.success_rate}%, Cache hit rate: {metrics.cache_hit_rate}%")

    except Exception as e:
        logger.warning(f"Failed to export caption metrics: {e}")
        print(f"\n  ⚠ Failed to export caption metrics: {e}")


if __name__ == '__main__':
    main()
