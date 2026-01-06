#!/usr/bin/env python3
"""
Voiceover-Matcher: End-to-End Documentary Footage Pipeline v3.0

CONFIGURATION CENTRALIZATION (v3.0):
- All settings loaded from config.yaml via centralized config.py
- No hardcoded values - everything configurable
- Hot-reload capability for config changes
- Performance metrics tracking

PERFORMANCE OPTIMIZATIONS (v2.4+):
- Parallel transcription with ThreadPoolExecutor
- Delta-aware indexing (only process NEW videos on retries)
- Batch embedding API calls (configurable batch size)
- Selective vision processing (skip if transcript covers content)

CHECKPOINT & RESUME (v3.1):
- Automatic checkpoints after each stage
- Resume interrupted runs with --resume
- Saved keyword presets for reproducible runs

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
# These "mmco: unref short failure" messages are harmless but noisy
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 mode
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # Fallback: replace stdout/stderr with UTF-8 writers
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

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
# CONFIGURATION LOADING (Centralized - Single Source of Truth)
# =============================================================================

from src.config import (
    Config, load_config, get_config, set_config, reload_config,
    ensure_dirs, get_api_key, get_config_metrics, log_hardcoded_warning
)
from src.checkpoint import (
    CheckpointManager, KeywordManager, SavedKeywords,
    format_resume_prompt, format_keyword_prompt, STAGE_ORDER
)
from src.match_index import MatchAwareIndex

# Global config instance - loaded at startup
_config: Optional[Config] = None


def get_pipeline_config() -> Config:
    """Get the pipeline configuration (loads if not already loaded)"""
    global _config
    if _config is None:
        _config = get_config()
    return _config


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def strip_extended_path_prefix(path: Path) -> Path:
    r"""
    Strip Windows extended-length path prefix (\\?\) from a Path.
    This prefix can cause issues with some applications.
    """
    path_str = str(path)
    
    prefixes = ['\\\\?\\', '\\\\.\\', '//?/', '//./']
    for prefix in prefixes:
        if path_str.startswith(prefix):
            path_str = path_str[len(prefix):]
            break
    
    return Path(path_str)


def load_environment(project_dir: Path = None):
    """Load .env file from install dir, then optionally from project dir"""
    try:
        from dotenv import load_dotenv
        
        # Load global .env from install directory
        global_env = INSTALL_DIR / '.env'
        if global_env.exists():
            load_dotenv(global_env)
            print(f"  ✓ Loaded global environment from {global_env}")
        
        # Load project-specific .env (overrides global)
        if project_dir:
            project_env = project_dir / '.env'
            if project_env.exists():
                load_dotenv(project_env, override=True)
                print(f"  ✓ Loaded project environment from {project_env}")
        
        # Fallback: try current working directory
        if not global_env.exists() and Path('.env').exists():
            load_dotenv()
            print(f"  ✓ Loaded environment from .env")
            
    except ImportError:
        pass  # dotenv not installed, rely on system env vars


def make_paths_project_relative(config: Config, project_dir: Path) -> Config:
    """Update config paths to be relative to project directory"""
    project_dir = project_dir.resolve()
    config.project_dir = str(project_dir)
    project_name = project_dir.name  # e.g., "zerokeyword__2025-12-24"
    
    # Shorten project name for folder (first 15 chars)
    short_project = project_name[:15].rstrip('_-')
    
    # Update downloading output dir - use new short path settings
    download_cfg = config.download if hasattr(config, 'download') else None
    
    if download_cfg and download_cfg.root_dir:
        # Use explicit root_dir (e.g., "E:/vids")
        folder_name = getattr(download_cfg, 'folder_name', 'videos')
        config.downloaded_videos_dir = str(Path(download_cfg.root_dir) / short_project)
    elif config.downloading.output_dir and not Path(config.downloading.output_dir).is_absolute():
        # Use project-relative path with short folder name
        folder_name = 'videos'
        if download_cfg:
            folder_name = getattr(download_cfg, 'folder_name', 'videos')
        config.downloaded_videos_dir = str(project_dir / folder_name)
    else:
        config.downloaded_videos_dir = str(project_dir / 'videos')
    
    # Update output dir
    if config.output.output_dir and not Path(config.output.output_dir).is_absolute():
        config.otio_output_dir = str(project_dir / config.output.output_dir)
    
    # Update cache dir (both nested and top-level for compatibility)
    if config.cache.cache_dir and not Path(config.cache.cache_dir).is_absolute():
        config.cache.cache_dir = str(project_dir / config.cache.cache_dir)
    # Also sync top-level cache_dir with nested cache.cache_dir
    config.cache_dir = config.cache.cache_dir
    
    # Update log dir
    if config.logging.log_dir and not Path(config.logging.log_dir).is_absolute():
        config.logging.log_dir = str(project_dir / config.logging.log_dir)
    
    return config


def load_project_config(project_dir: Path, config_path: Path = None) -> Config:
    """
    Load configuration with project overrides.
    
    Chain-of-thought: Project-specific config can override global settings
    Reasoning: Different projects may need different parameters
    Decision: Merge project_config.yaml into base config if it exists
    """
    import yaml
    
    # Load base config
    if config_path is None:
        config_path = INSTALL_DIR / 'config.yaml'
    
    config = load_config(str(config_path))
    
    # Load and merge project config if it exists
    project_config_path = project_dir / 'project_config.yaml'
    if project_config_path.exists():
        print(f"  ✓ Loading project config: {project_config_path}")
        try:
            with open(project_config_path, 'r', encoding='utf-8') as f:
                project_overrides = yaml.safe_load(f)
            
            if project_overrides:
                config = merge_config(config, project_overrides)
        except Exception as e:
            print(f"  ⚠ Could not load project config: {e}")
    
    # Make paths project-relative
    config = make_paths_project_relative(config, project_dir)
    
    return config


def merge_config(config, overrides: dict):
    """
    Deep merge overrides into config object.

    Handles nested dataclasses and None values properly.
    """
    if not overrides or config is None:
        return config

    for key, value in overrides.items():
        if hasattr(config, key):
            current = getattr(config, key)

            # If both are objects with attributes (but not None), recurse
            if isinstance(value, dict) and current is not None and hasattr(current, '__dict__'):
                merge_config(current, value)
            else:
                setattr(config, key, value)
        else:
            # Warn about unknown keys (likely typos in project_config.yaml)
            logger.warning(f"Unknown config key '{key}' in overrides - ignored")

    return config


def validate_config_at_startup(config: Config) -> bool:
    """
    Validate configuration at startup with clear error messages.
    
    Returns:
        True if config is valid, False otherwise
    """
    print("\n  Validating configuration...")
    
    errors = config.validate()
    
    if errors:
        print(f"  ⚠ Configuration warnings ({len(errors)}):")
        for error in errors:
            print(f"    • {error}")
        return len([e for e in errors if 'required' in e.lower()]) == 0
    else:
        print("  ✓ Configuration valid")
        return True


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: Config, output_dir: Path = None, run_timestamp: str = None) -> logging.Logger:
    """
    Setup logging with dual log files per run.

    Creates two log files:
    1. run_{timestamp}.log - Normal logging (INFO level, matches console)
    2. run_{timestamp}_verbose.log - Verbose logging (DEBUG level, everything)

    Args:
        config: Configuration object
        output_dir: Directory to save log files (default: project output dir)
        run_timestamp: Timestamp string for filenames (default: auto-generated)

    Returns:
        Logger instance
    """
    from datetime import datetime

    # Get log level from config
    log_level = getattr(logging, config.logging.log_level.upper(), logging.INFO)

    # Generate timestamp if not provided
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine output directory
    if output_dir is None:
        output_dir = Path(config.output.output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logs subdirectory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Log file paths
    normal_log_path = logs_dir / f"run_{run_timestamp}.log"
    verbose_log_path = logs_dir / f"run_{run_timestamp}_verbose.log"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler (matches config level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # Normal log file handler (INFO level)
    try:
        normal_handler = logging.FileHandler(normal_log_path, encoding='utf-8')
        normal_handler.setLevel(logging.INFO)
        normal_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        normal_handler.setFormatter(normal_format)
        root_logger.addHandler(normal_handler)
    except Exception as e:
        print(f"  Warning: Could not create normal log file: {e}")

    # Verbose log file handler (DEBUG level - everything)
    try:
        verbose_handler = logging.FileHandler(verbose_log_path, encoding='utf-8')
        verbose_handler.setLevel(logging.DEBUG)
        verbose_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        verbose_handler.setFormatter(verbose_format)
        root_logger.addHandler(verbose_handler)
    except Exception as e:
        print(f"  Warning: Could not create verbose log file: {e}")

    # Setup FFmpeg debug log (captures H.264 decoder warnings from OpenCV)
    try:
        from src.utils import setup_ffmpeg_debug_log
        ffmpeg_debug_path = setup_ffmpeg_debug_log(logs_dir)
    except Exception as e:
        ffmpeg_debug_path = None
        print(f"  Warning: Could not create FFmpeg debug log: {e}")

    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized")
    logger.debug(f"Normal log: {normal_log_path}")
    logger.debug(f"Verbose log: {verbose_log_path}")
    if ffmpeg_debug_path:
        logger.debug(f"FFmpeg debug log: {ffmpeg_debug_path}")

    # Store paths for reference
    logger.log_paths = {
        'normal': str(normal_log_path),
        'verbose': str(verbose_log_path),
        'ffmpeg_debug': str(ffmpeg_debug_path) if ffmpeg_debug_path else None
    }

    return logger


# =============================================================================
# IMPORTS (After config is available)
# =============================================================================

# Core imports (always available)
logger = logging.getLogger(__name__)

# Lazy imports for optional modules
def get_keyword_extractor(config: Config):
    """Lazy import of keyword extractor"""
    try:
        from src.keyword_extractor import LLMKeywordExtractor
        return LLMKeywordExtractor(config=config)
    except ImportError as e:
        logger.error(f"Could not import keyword_extractor: {e}")
        return None


def get_video_downloader(config: Config):
    """Lazy import of video downloader"""
    try:
        from src.downloader import VideoDownloader, DownloadCheckpoint
        return VideoDownloader(config=config), DownloadCheckpoint
    except ImportError as e:
        logger.error(f"Could not import downloader: {e}")
        return None, None


def check_module_availability() -> Dict[str, bool]:
    """Check which optional modules are available"""
    modules = {}
    
    # Enhanced features
    try:
        from src.keyword_remix import KeywordRemixer
        modules['keyword_remix'] = True
    except ImportError:
        modules['keyword_remix'] = False
    
    try:
        from src.pexels import download_pexels_footage
        modules['pexels'] = True
    except ImportError:
        modules['pexels'] = False
    
    try:
        from src.pixabay import download_pixabay_footage
        modules['pixabay'] = True
    except ImportError:
        modules['pixabay'] = False
    
    try:
        from src.multi_style import prompt_for_second_style, STYLE_DEFAULT, OTIOStyle
        modules['multi_style'] = True
    except ImportError:
        modules['multi_style'] = False
    
    # Performance optimizations
    try:
        from src.transcription import (
            transcribe_videos_parallel,
            DeltaAwareIndex,
            transcribe_voiceover_media
        )
        modules['optimized_transcription'] = True
    except ImportError:
        modules['optimized_transcription'] = False
    
    try:
        from src.embeddings import (
            compute_embeddings,
            get_embedding_provider,
            build_embedding_index,
            EmbeddingCache
        )
        modules['optimized_embeddings'] = True
    except ImportError:
        modules['optimized_embeddings'] = False
    
    try:
        from src.vision import process_video_vision
        modules['optimized_vision'] = True
    except ImportError:
        modules['optimized_vision'] = False
    
    return modules


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Voiceover-to-Footage Matching Pipeline v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --voiceover script.srt
    python main.py --voiceover script.srt --keywords 30
    python main.py --project "E:\\Projects\\MyDoc" --voiceover voiceover.srt

    # Checkpoint & Resume
    python main.py --resume                    # Resume interrupted run
    python main.py --fresh                     # Force fresh start

    # Saved Keywords (reproducible runs)
    python main.py --save-keywords             # Save keywords after extraction
    python main.py --save-keywords mypreset    # Save with custom name
    python main.py --use-keywords              # Use most recent saved keywords
    python main.py --use-keywords mypreset     # Use specific preset
    python main.py --list-keywords             # List all saved presets

    # Other options
    python main.py --match-only                # Skip download, match existing
    python main.py --config custom_config.yaml # Use custom config
        """
    )

    parser.add_argument(
        '--voiceover', '-v',
        type=str,
        help='Path to voiceover file (SRT, MP3, WAV, MP4)'
    )

    parser.add_argument(
        '--keywords', '-k',
        type=int,
        default=None,
        help='Number of keywords to extract (default: from config.yaml)'
    )

    parser.add_argument(
        '--project', '-p',
        type=str,
        default=None,
        help='Project directory (contains videos, outputs, project_config.yaml)'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    parser.add_argument(
        '--match-only',
        action='store_true',
        help='Skip download, only match existing footage'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume interrupted pipeline run'
    )

    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Force fresh start, ignore any existing checkpoint'
    )

    parser.add_argument(
        '--force-rematch',
        action='store_true',
        help='Force rematch all videos, ignoring cached matches (delta matching)'
    )

    parser.add_argument(
        '--use-keywords',
        type=str,
        nargs='?',
        const='latest',
        metavar='PRESET',
        help='Use saved keywords (specify preset name, or "latest" for most recent)'
    )

    parser.add_argument(
        '--save-keywords',
        type=str,
        nargs='?',
        const='auto',
        metavar='NAME',
        help='Save extracted keywords as a preset (auto-generates name if not specified)'
    )

    parser.add_argument(
        '--list-keywords',
        action='store_true',
        help='List saved keyword presets and exit'
    )

    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate config file and exit'
    )

    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run in non-interactive mode (skip all prompts, use defaults)'
    )

    parser.add_argument(
        '--save-matching-fixtures',
        type=str,
        metavar='PATH',
        help='Save matching inputs to fixture file for testing (e.g., fixtures/test.json)'
    )

    parser.add_argument(
        '--refresh-entities',
        action='store_true',
        help='Force re-download entity images (ignore local cache)'
    )

    return parser.parse_args()


def find_voiceover_interactive(project_dir: Path) -> Optional[str]:
    """
    Find voiceover files in the project directory and let user select.

    Searches:
    - project_dir/voiceover/ (primary location)
    - project_dir/*.srt, *.mp3, *.wav, *.mp4 (fallback)

    Returns:
        Path to selected voiceover file, or None if cancelled/not found
    """
    voiceover_extensions = {'.srt', '.mp3', '.wav', '.mp4', '.m4a', '.aac', '.flac', '.ogg'}
    candidates = []

    # Search in voiceover subdirectory first
    voiceover_dir = project_dir / 'voiceover'
    if voiceover_dir.exists():
        for f in voiceover_dir.iterdir():
            if f.is_file() and f.suffix.lower() in voiceover_extensions:
                candidates.append(f)

    # Also search project root
    for f in project_dir.iterdir():
        if f.is_file() and f.suffix.lower() in voiceover_extensions:
            # Avoid duplicates
            if f not in candidates:
                candidates.append(f)

    # Sort by name
    candidates.sort(key=lambda x: x.name.lower())

    if not candidates:
        print("\n  ─────────────────────────────────────────────────────────────")
        print("  NO VOICEOVER FILES FOUND")
        print("  ─────────────────────────────────────────────────────────────")
        print(f"\n  Searched in:")
        print(f"    • {voiceover_dir}")
        print(f"    • {project_dir}")
        print(f"\n  Supported formats: {', '.join(sorted(voiceover_extensions))}")
        print(f"\n  To continue:")
        print(f"    1. Place your voiceover file in the 'voiceover' folder")
        print(f"    2. Run again, or specify with: run.bat --voiceover <file>")
        return None

    if len(candidates) == 1:
        # Auto-select if only one file
        selected = candidates[0]
        print(f"\n  Auto-detected voiceover: {selected.name}")
        return str(selected)

    # Multiple files - let user choose
    print("\n  ─────────────────────────────────────────────────────────────")
    print("  SELECT VOICEOVER FILE")
    print("  ─────────────────────────────────────────────────────────────")
    print()

    for i, f in enumerate(candidates, 1):
        # Show relative path from project dir
        try:
            rel_path = f.relative_to(project_dir)
        except ValueError:
            rel_path = f.name

        # Show file size
        size_mb = f.stat().st_size / (1024 * 1024)

        # Show file type hint
        type_hint = {
            '.srt': 'subtitles',
            '.mp3': 'audio',
            '.wav': 'audio',
            '.mp4': 'video',
            '.m4a': 'audio',
            '.aac': 'audio',
            '.flac': 'audio',
            '.ogg': 'audio',
        }.get(f.suffix.lower(), 'file')

        print(f"    {i}. {rel_path}")
        print(f"       ({type_hint}, {size_mb:.1f} MB)")

    print(f"\n    0. Cancel")

    while True:
        try:
            choice = input(f"\n  Select [1-{len(candidates)}]: ").strip()

            if not choice:
                continue

            choice_num = int(choice)

            if choice_num == 0:
                print("  Cancelled.")
                return None
            elif 1 <= choice_num <= len(candidates):
                selected = candidates[choice_num - 1]
                print(f"\n  Selected: {selected.name}")
                return str(selected)
            else:
                print(f"  Invalid choice. Enter 1-{len(candidates)} or 0 to cancel.")

        except ValueError:
            # Maybe they typed a filename
            for f in candidates:
                if choice.lower() in f.name.lower():
                    print(f"\n  Selected: {f.name}")
                    return str(f)
            print("  Invalid input. Enter a number.")

        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return None


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
    
    # Apply --non-interactive flag
    if hasattr(args, 'non_interactive') and args.non_interactive:
        config.enhanced.non_interactive = True
    
    # Setup logging with dual log files (normal + verbose)
    # Determine output directory from config or project path
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
    
    # Run pipeline using new modular architecture
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
        keyword_manager.save_preset(
            name=save_keywords,
            keywords=pipeline.state.keywords,
            topic=pipeline.state.topic_context or ""
        )
        print(f"\n  ✓ Saved keyword preset: {save_keywords}")

    if not success:
        print("\n  ❌ Pipeline failed")
        sys.exit(1)
    else:
        print("\n  ✅ Pipeline completed successfully")


if __name__ == '__main__':
    main()