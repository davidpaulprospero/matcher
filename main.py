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


def _auto_import_davinci_markers(config, project_dir: Path):
    """Auto-import DaVinci Resolve markers if configured.

    Checks for marker CSV files in:
    1. feedback.auto_import_path (explicit path)
    2. project_dir/markers.csv (convention)
    3. project_dir/output/*/markers.csv (exported from DaVinci)

    Args:
        config: Config object
        project_dir: Project directory path
    """
    feedback_config = getattr(config, 'feedback', None)
    if not feedback_config or not getattr(feedback_config, 'enabled', True):
        return

    if not getattr(feedback_config, 'import_davinci_markers', True):
        return

    # Check for explicit auto-import path
    auto_path = getattr(feedback_config, 'auto_import_path', '')
    csv_paths_to_try = []

    if auto_path:
        csv_paths_to_try.append(Path(auto_path))

    # Check convention locations
    csv_paths_to_try.extend([
        project_dir / 'markers.csv',
        project_dir / 'davinci_markers.csv',
        project_dir / 'feedback.csv',
    ])

    # Check output directories for exported markers
    output_dir = project_dir / 'output'
    if output_dir.exists():
        for subdir in sorted(output_dir.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True):
            if subdir.is_dir():
                csv_paths_to_try.append(subdir / 'markers.csv')

    # Find first existing CSV
    csv_path = None
    for path in csv_paths_to_try:
        if path.exists():
            csv_path = path
            break

    if not csv_path:
        return  # No marker file found

    print(f"\n  📥 Auto-importing DaVinci markers from: {csv_path.name}")

    try:
        from src.feedback import (
            load_rejection_database,
            import_davinci_markers_full,
            apply_approvals_to_database,
            generate_feedback_report,
        )

        # Load rejection database
        db = load_rejection_database(project_dir=project_dir, include_global=True)

        # Import markers with full feedback support
        result = import_davinci_markers_full(csv_path, project_dir)

        # Apply rejections
        if result.rejections:
            count = db.add_rejections(result.rejections)
            print(f"    ✓ Added {count} new rejections")

        # Apply approvals (boost channel scores)
        if result.approvals:
            approve_count = apply_approvals_to_database(result.approvals, db)
            print(f"    ✓ Recorded {approve_count} approvals for channel scoring")

        # Save database
        db.save()

        # Generate feedback report
        report_path = project_dir / 'feedback_report.md'
        generate_feedback_report(result, report_path)
        print(f"    ✓ Feedback report: {report_path.name}")

        # Print summary
        print(f"    Summary: {result.summary()}")

        if result.replacements:
            print(f"    ⚠ {len(result.replacements)} segments marked for replacement")

        if result.warnings:
            print(f"    ⚠ {len(result.warnings)} warnings to review")

    except Exception as e:
        print(f"    Error importing markers: {e}")


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

    # Handle --rejection-stats
    if getattr(args, 'rejection_stats', False):
        from src.feedback import load_rejection_database
        db = load_rejection_database(project_dir=PROJECT_DIR, include_global=True)
        print(f"\n  📊 Rejection Database Statistics:")
        print(f"    Total rejections: {db.get_rejection_count()}")
        print(f"    Blocked channels: {db.get_blocked_channel_count()}")
        reasons = db.get_rejections_by_reason()
        if reasons:
            print(f"    Rejections by reason:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"      {reason}: {count}")
        sys.exit(0)

    # Handle --import-rejections
    if getattr(args, 'import_rejections', None):
        from src.feedback import load_rejection_database
        from src.feedback.davinci_import import import_davinci_markers

        csv_path = Path(args.import_rejections)
        if not csv_path.exists():
            print(f"\n  Error: CSV file not found: {csv_path}")
            sys.exit(1)

        print(f"\n  📥 Importing rejections from: {csv_path}")
        db = load_rejection_database(project_dir=PROJECT_DIR, include_global=True)

        try:
            rejections = import_davinci_markers(csv_path, PROJECT_DIR)
            if rejections:
                count = db.add_rejections(rejections)
                db.save()
                print(f"    ✓ Imported {count} new rejections ({len(rejections)} total in CSV)")
            else:
                print(f"    No rejection markers found in CSV")
        except Exception as e:
            print(f"    Error importing markers: {e}")
            sys.exit(1)
        sys.exit(0)

    # Handle --list-clients
    if getattr(args, 'list_clients', False):
        from src.feedback.client_profiles import list_client_profiles, ClientProfile
        clients = list_client_profiles()
        if clients:
            print(f"\n  📋 Client Profiles ({len(clients)} total):")
            for client_id in clients:
                profile = ClientProfile.load(client_id)
                if profile:
                    print(f"    • {client_id}: {len(profile.projects)} projects, "
                          f"{profile.total_videos_accepted} accepted, "
                          f"{profile.total_videos_rejected} rejected")
        else:
            print("\n  No client profiles found.")
            print("  Use --client <name> to create a profile for a project.")
        sys.exit(0)

    # Handle --client-stats
    if getattr(args, 'client_stats', None):
        from src.feedback.client_profiles import ClientProfile, list_client_profiles
        client_id = args.client_stats

        if client_id == 'all':
            clients = list_client_profiles()
        else:
            clients = [client_id]

        for cid in clients:
            profile = ClientProfile.load(cid)
            if profile:
                print(f"\n  📊 Client: {profile.display_name} ({profile.client_id})")
                print(f"    Created: {profile.created[:10] if profile.created else 'unknown'}")
                print(f"    Projects: {len(profile.projects)}")
                print(f"    Videos: {profile.total_videos_accepted} accepted, {profile.total_videos_rejected} rejected")
                print(f"    Blacklist: {len(profile.blacklist_channels)} channels, {len(profile.blacklist_keywords)} keywords")
                if profile.evolved_preset:
                    print(f"    Evolved preset: {profile.evolved_preset.projects_analyzed} projects analyzed")
                    print(f"      Auto-blacklist: {len(profile.evolved_preset.auto_blacklist_channels)} channels")
                if profile.preferences:
                    print(f"    Preferences:")
                    print(f"      Style: {profile.preferences.content_style}")
                    print(f"      Duration: {profile.preferences.preferred_duration_min:.0f}-{profile.preferences.preferred_duration_max:.0f}s")
            else:
                print(f"\n  Client profile not found: {cid}")
        sys.exit(0)

    # Handle --evolve-preset
    if getattr(args, 'evolve_preset', False):
        from src.feedback.client_profiles import evolve_preset_from_history, ClientProfile
        client_id = getattr(args, 'client', None)
        if not client_id:
            print("\n  Error: --evolve-preset requires --client <name>")
            sys.exit(1)

        profile = ClientProfile.load(client_id)
        if not profile:
            print(f"\n  Error: No profile found for client '{client_id}'")
            print("  Run at least one project with --client to create a profile.")
            sys.exit(1)

        print(f"\n  🧬 Evolving preset for client: {client_id}")
        print(f"    Analyzing {len(profile.projects)} projects...")

        preset = evolve_preset_from_history(client_id)
        print(f"\n  ✓ Evolved preset generated:")
        print(f"    Auto-blacklist channels: {len(preset.auto_blacklist_channels)}")
        print(f"    Auto-blacklist keywords: {len(preset.auto_blacklist_keywords)}")
        print(f"    Preferred channels: {len(preset.preferred_channels)}")
        print(f"    Duration range: {preset.preferred_duration_range[0]:.0f}-{preset.preferred_duration_range[1]:.0f}s")
        if preset.rejection_patterns:
            print(f"    Rejection patterns:")
            for reason, count in sorted(preset.rejection_patterns.items(), key=lambda x: -x[1])[:5]:
                print(f"      {reason}: {count}")
        sys.exit(0)

    # Handle --apply-learnings
    if getattr(args, 'apply_learnings', None):
        from src.feedback.client_profiles import ClientProfile
        source_project = Path(args.apply_learnings)
        if not source_project.exists():
            print(f"\n  Error: Project not found: {source_project}")
            sys.exit(1)

        # Find client from source project
        client_id = getattr(args, 'client', None)
        if not client_id:
            # Try to infer from project path (e.g., E:/Edit Job/theresa/...)
            parts = source_project.parts
            for i, part in enumerate(parts):
                if part.lower() == 'edit job' and i + 1 < len(parts):
                    client_id = parts[i + 1].lower()
                    break

        if client_id:
            profile = ClientProfile.load(client_id)
            if profile:
                print(f"\n  📚 Applying learnings from client: {profile.display_name}")
                print(f"    Blacklist channels: {len(profile.blacklist_channels)}")
                print(f"    Blacklist keywords: {len(profile.blacklist_keywords)}")
                # Don't exit - continue with pipeline using these learnings
            else:
                print(f"\n  ⚠ No profile found for client '{client_id}', starting fresh")
        else:
            print(f"\n  ⚠ Could not determine client from path, skipping cross-project learning")

    # Store client ID for pipeline use
    client_id = getattr(args, 'client', None)
    if client_id:
        print(f"\n  👤 Client: {client_id}")
        # Ensure client profile exists
        from src.feedback.client_profiles import get_or_create_client_profile
        profile = get_or_create_client_profile(client_id)
        profile.add_project(str(PROJECT_DIR))
        profile.save()

    # ==========================================================================
    # KEYWORD MODE DETECTION
    # ==========================================================================
    # Keyword mode allows running without voiceover - keywords generate segments
    keyword_mode_active = False
    keyword_list = []

    # Check for --keyword-list CLI arg
    if hasattr(args, 'keyword_list') and args.keyword_list:
        keyword_list = [k.strip() for k in args.keyword_list.split(',') if k.strip()]
        if keyword_list:
            keyword_mode_active = True
            # Enable keyword mode in config
            config.keyword_mode.enabled = True
            # Override mode from CLI if specified
            if hasattr(args, 'keyword_mode') and args.keyword_mode:
                config.keyword_mode.mode = args.keyword_mode
            # Override duration if specified (handle dict/object config - Rule 6)
            if hasattr(args, 'duration') and args.duration:
                montage_cfg = config.keyword_mode.montage
                script_cfg = config.keyword_mode.script
                if isinstance(montage_cfg, dict):
                    montage_cfg['total_duration'] = args.duration
                else:
                    montage_cfg.total_duration = args.duration
                if isinstance(script_cfg, dict):
                    script_cfg['target_duration'] = args.duration
                else:
                    script_cfg.target_duration = args.duration
            # Override style/tone if specified (handle dict/object config - Rule 6)
            if hasattr(args, 'style') and args.style:
                script_cfg = config.keyword_mode.script
                if isinstance(script_cfg, dict):
                    script_cfg['style'] = args.style
                else:
                    script_cfg.style = args.style
            if hasattr(args, 'tone') and args.tone:
                script_cfg = config.keyword_mode.script
                if isinstance(script_cfg, dict):
                    script_cfg['tone'] = args.tone
                else:
                    script_cfg.tone = args.tone

            print(f"\n  🔑 Keyword Mode: {config.keyword_mode.mode}")
            print(f"    Keywords: {', '.join(keyword_list[:5])}")
            if len(keyword_list) > 5:
                print(f"    ... and {len(keyword_list) - 5} more")

    # Check if keyword mode is enabled in config (without CLI keywords)
    elif config.keyword_mode.enabled and config.keyword_mode.keywords:
        keyword_list = config.keyword_mode.keywords
        keyword_mode_active = True
        print(f"\n  🔑 Keyword Mode (from config): {config.keyword_mode.mode}")
        print(f"    Keywords: {', '.join(keyword_list[:5])}")
        if len(keyword_list) > 5:
            print(f"    ... and {len(keyword_list) - 5} more")

    # ==========================================================================
    # VOICEOVER FILE CHECK (only if not in keyword mode)
    # ==========================================================================
    vo_path = None
    if not keyword_mode_active:
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
    from src.pipeline import create_default_pipeline, create_match_only_pipeline, create_keyword_mode_pipeline
    from src.agents import ResilientRunner, HealingOrchestrator, HealingStrategy

    # Handle keyword presets/selection
    keyword_manager = KeywordManager(PROJECT_DIR)
    use_keywords = getattr(args, 'use_keywords', None)
    save_keywords = getattr(args, 'save_keywords', None)

    # Create pipeline
    audio_first = getattr(config.download.audio_first, 'enabled', False) if hasattr(config.download, 'audio_first') else False

    if keyword_mode_active:
        # Keyword mode pipeline - no voiceover required
        pipeline = create_keyword_mode_pipeline(config, PROJECT_DIR, mode=config.keyword_mode.mode)
        # Set keywords in state
        pipeline.state.keywords = keyword_list
        pipeline.state.project_dir = str(PROJECT_DIR)
        print(f"\n  Using keyword mode pipeline (mode={config.keyword_mode.mode})")
    elif args.match_only:
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

        # Auto-import DaVinci markers if configured
        _auto_import_davinci_markers(config, PROJECT_DIR)
    else:
        pipeline = create_default_pipeline(config, PROJECT_DIR, audio_first_mode=audio_first)

    # Initialize state
    if not keyword_mode_active:
        pipeline.state.voiceover_path = str(vo_path)
    pipeline.state.project_dir = str(PROJECT_DIR)
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

    # Handle high matches mode
    if getattr(args, 'high_matches', False):
        import logging
        hmm_logger = logging.getLogger('high_matches_mode')

        print(f"\n  [VALIDATION] High Matches Mode flag detected")
        hmm_logger.info("[high_matches] CLI flag --high-matches detected")

        # Enable high matches mode in config (handle dict/object config - Rule 6)
        hmm_config = config.matching.high_matches_mode
        hmm_logger.info(f"[high_matches] Config type: {type(hmm_config)}")
        hmm_logger.info(f"[high_matches] Config before: enabled={getattr(hmm_config, 'enabled', hmm_config.get('enabled', 'N/A') if isinstance(hmm_config, dict) else 'N/A')}")

        if isinstance(hmm_config, dict):
            hmm_config['enabled'] = True
        else:
            hmm_config.enabled = True

        # Override target confidence if specified
        if getattr(args, 'target_confidence', None):
            hmm_logger.info(f"[high_matches] Overriding target_confidence: {args.target_confidence}")
            if isinstance(hmm_config, dict):
                hmm_config['target_confidence'] = args.target_confidence
            else:
                hmm_config.target_confidence = args.target_confidence

        # Override coverage target if specified
        if getattr(args, 'coverage_target', None):
            hmm_logger.info(f"[high_matches] Overriding coverage_target: {args.coverage_target}")
            if isinstance(hmm_config, dict):
                hmm_config['coverage_target'] = args.coverage_target
            else:
                hmm_config.coverage_target = args.coverage_target

        # Print confirmation
        target = hmm_config.get('target_confidence', 0.90) if isinstance(hmm_config, dict) else hmm_config.target_confidence
        coverage = hmm_config.get('coverage_target', 0.85) if isinstance(hmm_config, dict) else hmm_config.coverage_target
        max_iter = hmm_config.get('max_iterations', 3) if isinstance(hmm_config, dict) else hmm_config.max_iterations
        enabled = hmm_config.get('enabled', False) if isinstance(hmm_config, dict) else hmm_config.enabled

        hmm_logger.info(f"[high_matches] Config after:")
        hmm_logger.info(f"[high_matches]   enabled: {enabled}")
        hmm_logger.info(f"[high_matches]   target_confidence: {target}")
        hmm_logger.info(f"[high_matches]   coverage_target: {coverage}")
        hmm_logger.info(f"[high_matches]   max_iterations: {max_iter}")

        print(f"\n  🎯 High Matches Mode: target={target:.0%} confidence, {coverage:.0%} coverage")
        print(f"     Max iterations: {max_iter}")
        print(f"     Dedicated log: output/logs/high_matches_*.log")

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

    # Ensure PO Token server is running (required for YouTube access)
    # Check if caption-first mode or audio-first mode is enabled (both need YouTube access)
    caption_first_enabled = getattr(config.download, 'caption_first', None)
    if caption_first_enabled:
        caption_first_enabled = getattr(caption_first_enabled, 'enabled', False)
    audio_first_enabled = getattr(config.download, 'audio_first', None)
    if audio_first_enabled:
        audio_first_enabled = getattr(audio_first_enabled, 'enabled', False)

    if caption_first_enabled or audio_first_enabled or not keyword_mode_active:
        try:
            from src.pot_utils.pot_server import ensure_pot_server, get_pot_server_status
            status = get_pot_server_status()
            if status['script_exists']:
                print(f"\n  🔑 Checking PO Token server...")
                if ensure_pot_server():
                    print(f"  ✓ PO Token server running on {status['host']}:{status['port']}")
                else:
                    print(f"  ⚠ PO Token server not available - YouTube may rate-limit requests")
                    print(f"    Start manually: node {status['script_path']}")
            else:
                logger.debug("PO Token server not installed, skipping check")
        except Exception as e:
            logger.debug(f"PO Token server check failed: {e}")

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

    if not success:
        print("\n  ❌ Pipeline failed")
        sys.exit(1)
    else:
        print("\n  ✅ Pipeline completed successfully")


if __name__ == '__main__':
    main()
