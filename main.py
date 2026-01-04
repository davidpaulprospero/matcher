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


def merge_config(config: Config, overrides: dict) -> Config:
    """Deep merge overrides into config object"""
    if not overrides:
        return config
    
    for key, value in overrides.items():
        if hasattr(config, key):
            current = getattr(config, key)
            
            # If both are objects with attributes, recurse
            if isinstance(value, dict) and hasattr(current, '__dict__'):
                merge_config(current, value)
            else:
                setattr(config, key, value)
        else:
            setattr(config, key, value)
    
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


# =============================================================================
# PIPELINE CLASS
# =============================================================================

class Pipeline:
    """
    End-to-end documentary footage pipeline.
    
    All settings are loaded from config.yaml - no hardcoded values.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.modules = check_module_availability()
        
        # Components (initialized lazily)
        self.downloader = None
        self.keyword_extractor = None
        self.scene_detector = None
        
        # State tracking
        self.keywords: List[str] = []
        self.downloaded_videos: List[dict] = []
        self.voiceover_segments: List[dict] = []
        
        # Processing state
        self.transcripts = {}
        self.embeddings = []
        self.text_metadata = []
        self.embedding_index = None
        self.matches = []
        self.cache = None
        
        # Enhanced features state
        self.topic_context = ""
        self.extracted_entities = []  # Named entities with context
        self.entity_images = {}  # Downloaded images for entities (name -> EntityImageResult)
        self.entity_videos = {}  # Downloaded videos for entities (name -> EntityVideoResult)
        self.keyword_remixer = None
        self.enhanced_enabled = config.enhanced.enabled
        self.second_style = None
        self.failed_keywords = []
        # Use config face_preference (for non-interactive mode)
        self.face_preference = getattr(config.enhanced, 'face_preference', 'neutral')
        
        # Performance tracking
        self.stage_timings = {}
        self.use_delta_indexing = True

        # Checkpoint manager for resume functionality
        self.checkpoint = None  # Initialized in run() with project dir
        self.resume_mode = False

        # Global cache for cross-project video reuse
        self.global_cache = None
        self.global_cache_videos = []  # Videos reused from global cache
        self._init_global_cache()

        # Run logger
        self.run_logger = None
        self._init_logger()
    
    def _init_logger(self):
        """Initialize run logger from config"""
        if self.config.logging.enabled:
            try:
                from src.logger import RunLogger
                self.run_logger = RunLogger(log_dir=self.config.logging.log_dir)
                self.run_logger.log_config(self.config)
                print(f"  ✓ Logging enabled: {self.run_logger.log_file}")
            except Exception as e:
                print(f"  ⚠ Could not initialize logger: {e}")

    def _init_global_cache(self):
        """Initialize global cache for cross-project video reuse"""
        gc_config = getattr(self.config, 'global_cache', None)
        if not gc_config or not getattr(gc_config, 'enabled', False):
            logger.debug("Global cache disabled")
            return

        try:
            from src.global_cache import GlobalCacheManager

            cache_dir = getattr(gc_config, 'cache_dir', '~/.matcher_global_cache')
            # Expand ~ to home directory
            cache_dir = str(Path(cache_dir).expanduser())

            self.global_cache = GlobalCacheManager(cache_dir=cache_dir, config=self.config)
            stats = self.global_cache.get_stats()
            logger.info(f"Global cache initialized: {stats['total_videos']} videos, "
                       f"{stats['total_topics']} topics, {stats['total_keywords']} keywords")
        except Exception as e:
            logger.warning(f"Could not initialize global cache: {e}")
            self.global_cache = None

    def _print_banner(self):
        """Print startup banner with config-driven values"""
        config = self.config
        
        print("=" * 70)
        print(f"  VOICEOVER-MATCHER: {config.project.name} v{config.project.version}")
        print("=" * 70)
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if PROJECT_DIR and PROJECT_DIR != Path.cwd():
            print(f"  Project: {PROJECT_DIR.name}")
            print(f"  Location: {PROJECT_DIR}")
        
        print(f"  Videos: {config.downloaded_videos_dir}")
        print(f"  Output: {config.otio_output_dir}")
        
        if self.run_logger:
            print(f"  Log: {self.run_logger.log_file}")
        
        # Show enhanced features status (from config)
        enhanced_status = []
        if self.modules['keyword_remix'] and config.enhanced.remix_enabled:
            enhanced_status.append(f"Confidence≥{config.enhanced.min_confidence:.0%}")
        if self.modules['pexels'] and config.enhanced.enable_pexels:
            enhanced_status.append("Pexels")
        if self.modules['pixabay'] and config.enhanced.enable_pixabay:
            enhanced_status.append("Pixabay")
        if self.modules['multi_style'] and config.multi_style.enabled:
            enhanced_status.append("Multi-Style")
        
        if enhanced_status:
            print(f"  Enhanced: {', '.join(enhanced_status)}")
        
        # Show optimization status (from config)
        opt_status = []
        if self.modules['optimized_transcription'] and config.pipeline.parallel_transcription:
            opt_status.append(f"Parallel({config.transcription.max_workers})")
        if self.modules['optimized_embeddings'] and config.pipeline.parallel_embedding:
            opt_status.append(f"Batch({config.embedding.batch_size})")
        if self.modules['optimized_vision'] and config.vision.enabled:
            opt_status.append("SelectiveVision")
        
        if opt_status:
            print(f"  Optimized: {', '.join(opt_status)}")
        
        # Show config metrics
        metrics = get_config_metrics()
        if metrics['load_time_total_ms'] > 0:
            print(f"  Config: Loaded in {metrics['load_time_total_ms']:.1f}ms")
        
        print("=" * 70)
    
    def _print_stage(self, stage, name: str):
        """Print stage header and log it"""
        print(f"\n{'─' * 70}")
        print(f"  STAGE {stage}: {name}")
        print(f"{'─' * 70}")

        if self.run_logger:
            self.run_logger.file_logger.info(f"STAGE {stage}: {name}")

    def _prompt_delta_matching(self):
        """
        Prompt user to choose delta matching mode at pipeline start.
        Delta matching only processes new/modified videos, using cached matches for unchanged ones.
        """
        config = self.config

        # Check if delta matching is available (match index exists with cached data)
        project_dir = PROJECT_DIR or Path.cwd()
        cache_dir = project_dir / ".cache"
        match_index_path = cache_dir / "match_index.json"
        cached_matches_path = cache_dir / "cached_matches.json"

        if not match_index_path.exists():
            # No previous matches - delta matching not applicable
            return

        try:
            import json

            # Load match index
            with open(match_index_path, 'r') as f:
                index_data = json.load(f)

            matched_videos = index_data.get('matched_videos', {})

            # Load cached matches count
            cached_matches_count = 0
            if cached_matches_path.exists():
                with open(cached_matches_path, 'r') as f:
                    cached_matches = json.load(f)
                    cached_matches_count = len(cached_matches) if isinstance(cached_matches, list) else 0

            if not matched_videos:
                return

            print(f"\n{'─' * 70}")
            print(f"  DELTA MATCHING")
            print(f"{'─' * 70}")
            print(f"  Found {len(matched_videos)} previously matched videos")
            if cached_matches_count > 0:
                print(f"  Cached matches: {cached_matches_count} segments")
            print()
            print(f"  [D] Delta match - Only process NEW videos (faster)")
            print(f"  [F] Full rematch - Rematch ALL videos (slower)")
            print(f"  [Q] Quit")
            print()

            while True:
                try:
                    choice = input("  Choice [D/F/Q]: ").strip().upper()
                    if choice == 'D':
                        # Use delta matching (default behavior)
                        self.force_rematch = False
                        print(f"  ✓ Using delta matching - will reuse cached matches for unchanged videos")
                        break
                    elif choice == 'F':
                        # Force full rematch
                        self.force_rematch = True
                        print(f"  ✓ Full rematch enabled - will rematch ALL videos")
                        break
                    elif choice == 'Q':
                        print("  Exiting.")
                        sys.exit(0)
                    else:
                        print("  Invalid choice. Please enter D, F, or Q.")
                except (EOFError, KeyboardInterrupt):
                    print("\n  Exiting.")
                    sys.exit(0)

        except Exception as e:
            # If we can't read the index, just continue without prompting
            logger.debug(f"Could not read match index for delta prompt: {e}")

    def _parse_srt(self, srt_path: str) -> List[dict]:
        """Parse SRT file into segments with smart splitting for long segments"""
        try:
            import srt

            # Try multiple encodings - SRT files often have different encodings
            encodings_to_try = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252']
            content = None

            for encoding in encodings_to_try:
                try:
                    with open(srt_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    # Check if content looks valid (has SRT timestamp arrow)
                    if content and '-->' in content:
                        break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if content is None:
                # Last resort: read as binary and detect BOM
                with open(srt_path, 'rb') as f:
                    raw = f.read()
                if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
                    content = raw.decode('utf-16', errors='ignore')
                else:
                    content = raw.decode('utf-8', errors='ignore')

            subtitles = list(srt.parse(content))
            
            segments = []
            for sub in subtitles:
                segments.append({
                    'index': sub.index,
                    'start_time': sub.start.total_seconds(),
                    'end_time': sub.end.total_seconds(),
                    'text': sub.content.strip(),
                    'duration': (sub.end - sub.start).total_seconds()
                })
            
            # Apply smart splitting for long segments
            segments = self._optimize_segments(segments)

            # Check for word-level timestamps file (generated by transcription)
            word_data = None
            words_path = Path(srt_path).with_suffix('.words.json')
            if words_path.exists():
                try:
                    import json
                    with open(words_path, 'r', encoding='utf-8') as f:
                        word_data = json.load(f)
                    logger.info(f"Loaded word timestamps from {words_path.name}")
                except Exception as e:
                    logger.debug(f"Could not load word timestamps: {e}")

            # Apply pause-based splitting for natural breaks
            segments = self._pause_split_segments(segments, word_data=word_data)

            return segments
        except ImportError:
            logger.error("srt package not installed. Install with: pip install srt")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to parse SRT: {e}")
            sys.exit(1)
    
    def _optimize_segments(self, segments: List[dict]) -> List[dict]:
        """
        Split segments that are significantly longer than the median.
        
        Logic:
        - Calculate median segment duration
        - If segment > multiplier * median, try to split at sentence boundaries
        - Distribute time proportionally based on character count
        """
        import re
        import statistics
        
        if len(segments) < 3:
            return segments
        
        # Calculate median duration
        durations = [s['duration'] for s in segments if s['duration'] > 0]
        if not durations:
            return segments
        
        # Get config values with defaults
        config = self.config
        split_multiplier = getattr(config.transcription, 'split_threshold_multiplier', 1.5)
        long_median = getattr(config.transcription, 'long_median_threshold', 8.0)
        absolute_threshold = getattr(config.transcription, 'absolute_split_threshold', 12.0)
        min_split = getattr(config.transcription, 'min_split_duration', 4.0)
        
        median_duration = statistics.median(durations)
        split_threshold = median_duration * split_multiplier  # Split if > multiplier * median
        
        # Don't split if median is already quite long
        if median_duration > long_median:
            split_threshold = absolute_threshold  # Only split very long segments
        
        # Minimum duration to consider splitting (avoid splitting short segments)
        min_split_duration = max(min_split, median_duration)
        
        optimized = []
        split_count = 0
        
        for seg in segments:
            duration = seg['duration']
            text = seg['text']
            
            # Check if segment should be split
            if duration > split_threshold and duration > min_split_duration:
                # Try to split at sentence boundaries
                sub_segments = self._split_segment_at_sentences(seg, median_duration)
                
                if len(sub_segments) > 1:
                    split_count += 1
                    optimized.extend(sub_segments)
                else:
                    optimized.append(seg)
            else:
                optimized.append(seg)
        
        # Re-index segments
        for i, seg in enumerate(optimized):
            seg['index'] = i + 1
        
        if split_count > 0:
            print(f"  ✓ Optimized: {split_count} long segments split ({len(segments)} → {len(optimized)})")
            logger.info(f"Segment optimization: {split_count} splits, {len(segments)} → {len(optimized)} segments")
        
        return optimized
    
    def _split_segment_at_sentences(self, segment: dict, target_duration: float) -> List[dict]:
        """
        Split a segment at sentence boundaries, distributing time proportionally.
        
        Args:
            segment: The segment to split
            target_duration: Target duration for each sub-segment
            
        Returns:
            List of sub-segments (may be just the original if can't split)
        """
        import re
        
        text = segment['text']
        start_time = segment['start_time']
        end_time = segment['end_time']
        total_duration = segment['duration']
        
        # Split at sentence boundaries: . ! ? (but not abbreviations like "Dr." or "U.S.")
        # Also split at semicolons and colons followed by space
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[;:])\s+'
        sentences = re.split(sentence_pattern, text)
        
        # If only one sentence, try splitting at commas for very long segments
        if len(sentences) == 1 and total_duration > target_duration * 2:
            # Split at major clause breaks (comma followed by conjunction or long phrase)
            comma_pattern = r',\s+(?=and |but |or |so |yet |while |when |if |because |although )'
            sentences = re.split(comma_pattern, text)
            
            # If still one piece and very long, force split at commas
            if len(sentences) == 1 and total_duration > target_duration * 3:
                sentences = [s.strip() for s in text.split(',') if s.strip()]
                # Rejoin with commas (they got removed)
                sentences = [s + (',' if i < len(sentences) - 1 else '') for i, s in enumerate(sentences)]
        
        # If still can't split or only one piece, return original
        if len(sentences) <= 1:
            return [segment]
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return [segment]
        
        # Calculate time distribution based on character length
        total_chars = sum(len(s) for s in sentences)
        if total_chars == 0:
            return [segment]
        
        sub_segments = []
        current_time = start_time
        
        for i, sentence in enumerate(sentences):
            # Proportional duration based on character count
            char_ratio = len(sentence) / total_chars
            seg_duration = total_duration * char_ratio
            
            # Ensure minimum duration of 0.5 seconds
            seg_duration = max(0.5, seg_duration)
            
            seg_end = current_time + seg_duration
            
            # Last segment should end exactly at original end time
            if i == len(sentences) - 1:
                seg_end = end_time
            
            sub_segments.append({
                'index': segment['index'],  # Will be re-indexed later
                'start_time': round(current_time, 3),
                'end_time': round(seg_end, 3),
                'text': sentence,
                'duration': round(seg_end - current_time, 3)
            })
            
            current_time = seg_end
        
        return sub_segments
    
    def _pause_split_segments(self, segments: List[dict], word_data: List[dict] = None) -> List[dict]:
        """
        Split segments at natural pauses - aggressive splitting for better matching.

        Split patterns (in order of priority):
        1. Sentence boundaries - every sentence becomes its own segment
        2. List item isolation - "Number X, Town" becomes its own segment
        3. Location patterns - "City, State," splits

        If word_data is provided (from Whisper word_timestamps), uses accurate
        word-level timing. Otherwise falls back to character-based estimation.
        """
        import re

        config = self.config
        pause_config = getattr(config.transcription, 'pause_split', None)

        # Check if pause splitting is enabled
        if not pause_config or not getattr(pause_config, 'enabled', False):
            return segments

        # Get config options
        split_sentences = getattr(pause_config, 'split_at_sentences', True)
        split_list_markers = getattr(pause_config, 'split_at_list_markers', True)
        split_locations = getattr(pause_config, 'split_at_locations', True)
        min_phrase_words = getattr(pause_config, 'min_phrase_words', 2)
        min_segment_duration = getattr(pause_config, 'min_segment_duration', 0.5)

        # Build word lookup if word_data available
        # word_data is list of segments, each with optional 'words' list
        word_lookup = {}  # Maps segment index -> list of word dicts with timing
        if word_data:
            for seg_idx, seg_data in enumerate(word_data):
                if 'words' in seg_data:
                    word_lookup[seg_idx] = seg_data['words']
            if word_lookup:
                print(f"  ✓ Using word-level timestamps for accurate pause-split timing")

        optimized = []
        split_count = 0

        for seg_idx, seg in enumerate(segments):
            text = seg['text']
            start_time = seg['start_time']
            end_time = seg['end_time']
            duration = seg['duration']

            # Skip very short segments
            if duration < min_segment_duration or len(text) < 10:
                optimized.append(seg)
                continue

            # PRIORITY 1: List item isolation
            if split_list_markers:
                list_pattern = re.compile(
                    r'([Nn]umber\s+\d{1,2}[,.\s]+[A-Z][a-zA-Z\s]+?)(?=[.!?]|$)'
                )
                numbered_pattern = re.compile(
                    r'(\d{1,2}\.\s+[A-Z][a-zA-Z\s,]+?)(?=[.!?]|$)'
                )
                has_list = list_pattern.search(text) or numbered_pattern.search(text)
                if has_list:
                    text = re.sub(r'(?<=[.!?])\s+(?=[Nn]umber\s+\d{1,2})', '|||SPLIT|||', text)
                    text = re.sub(r'(?<=[.!?])\s+(?=\d{1,2}\.\s+[A-Z])', '|||SPLIT|||', text)

            # PRIORITY 2: Sentence-level splitting
            if split_sentences:
                text = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '|||SPLIT|||', text)

            # PRIORITY 3: Location patterns
            if split_locations:
                text = re.sub(r'([A-Z][a-z]+,\s+[A-Z][a-z]+,)\s+', r'\1|||SPLIT|||', text)

            # Split the text
            parts = text.split('|||SPLIT|||')
            parts = [p.strip() for p in parts if p.strip()]

            # Filter out very short fragments
            if len(parts) > 1:
                valid_parts = []
                for part in parts:
                    word_count = len(part.split())
                    is_list_marker = re.match(r'^[Nn]umber\s+\d{1,2}', part) or re.match(r'^\d{1,2}\.', part)
                    ends_with_punct = part and part[-1] in '.!?'
                    if word_count >= min_phrase_words or is_list_marker or ends_with_punct:
                        valid_parts.append(part)
                    elif valid_parts:
                        valid_parts[-1] = valid_parts[-1] + ' ' + part
                    else:
                        valid_parts.append(part)
                parts = valid_parts

            # If no valid splits, keep original
            if len(parts) <= 1:
                optimized.append(seg)
                continue

            split_count += 1

            # Calculate timing for each part
            seg_words = word_lookup.get(seg_idx, [])

            if seg_words:
                # USE WORD-LEVEL TIMESTAMPS for accurate timing
                part_timings = self._calculate_part_timings_from_words(
                    parts, seg_words, start_time, end_time, min_segment_duration
                )
            else:
                # Fall back to character-based estimation
                part_timings = self._calculate_part_timings_from_chars(
                    parts, start_time, end_time, duration, min_segment_duration
                )

            for i, (part, (part_start, part_end)) in enumerate(zip(parts, part_timings)):
                optimized.append({
                    'index': seg['index'],
                    'start_time': round(part_start, 3),
                    'end_time': round(part_end, 3),
                    'text': part,
                    'duration': round(part_end - part_start, 3)
                })

        # Re-index all segments
        for i, seg in enumerate(optimized):
            seg['index'] = i + 1

        if split_count > 0:
            print(f"  ✓ Pause-split: {split_count} segments split ({len(segments)} → {len(optimized)})")
            logger.info(f"Pause-based splitting: {split_count} segments split, {len(segments)} → {len(optimized)} total")

        return optimized

    def _calculate_part_timings_from_words(
        self, parts: List[str], words: List[dict],
        start_time: float, end_time: float, min_duration: float
    ) -> List[tuple]:
        """Calculate accurate timing for each part using word-level timestamps."""
        duration = end_time - start_time

        # First pass: calculate natural timings from words
        natural_timings = []
        word_idx = 0

        for i, part in enumerate(parts):
            part_words = part.split()
            part_word_count = len(part_words)

            # Find the starting word for this part
            part_start = words[word_idx]['start'] if word_idx < len(words) else start_time

            # Advance word_idx by part_word_count
            words_consumed = 0
            part_end = part_start
            while words_consumed < part_word_count and word_idx < len(words):
                part_end = words[word_idx]['end']
                word_idx += 1
                words_consumed += 1

            natural_timings.append((part_start, part_end))

        # Calculate natural durations and total needed with min_duration
        natural_durations = [t[1] - t[0] for t in natural_timings]
        total_needed = sum(max(d, min_duration) for d in natural_durations)

        # If total needed exceeds available time, scale down
        if total_needed > duration:
            scale = duration / total_needed
            logger.debug(f"Pause-split (words): Time budget exceeded ({total_needed:.2f}s > {duration:.2f}s), scaling by {scale:.2f}")

            # Rebuild timings with scaled durations
            timings = []
            current_time = start_time
            for i, nat_dur in enumerate(natural_durations):
                part_duration = max(nat_dur, min_duration) * scale

                if i == len(parts) - 1:
                    part_end = end_time
                else:
                    part_end = current_time + part_duration

                # Safety: ensure positive duration
                if part_end <= current_time:
                    part_end = min(current_time + 0.001, end_time)

                timings.append((current_time, part_end))
                current_time = part_end

            return timings

        # No overflow - use natural timings with min_duration enforcement
        timings = []
        current_time = start_time

        for i, (part_start, part_end) in enumerate(natural_timings):
            # Ensure minimum duration
            if part_end - part_start < min_duration:
                part_end = part_start + min_duration

            # Clamp to end_time
            part_end = min(part_end, end_time)

            # Last part gets remaining time
            if i == len(parts) - 1:
                part_end = end_time

            # Safety: ensure positive duration
            if part_end <= current_time:
                part_end = min(current_time + 0.001, end_time)

            timings.append((current_time, part_end))
            current_time = part_end

        return timings

    def _calculate_part_timings_from_chars(
        self, parts: List[str], start_time: float,
        end_time: float, duration: float, min_duration: float
    ) -> List[tuple]:
        """Calculate timing for each part using character-based estimation (fallback)."""
        total_chars = sum(len(p) for p in parts)
        if total_chars == 0:
            return [(start_time, end_time)]

        # First pass: calculate natural durations
        natural_durations = []
        for part in parts:
            char_ratio = len(part) / total_chars
            natural_durations.append(duration * char_ratio)

        # Calculate total time needed if we enforce min_duration
        total_needed = sum(max(d, min_duration) for d in natural_durations)

        # If we need more time than available, scale down proportionally
        # This prevents overflow and zero-duration segments
        if total_needed > duration:
            # Scale factor to fit within available duration
            scale = duration / total_needed
            logger.debug(f"Pause-split: Time budget exceeded ({total_needed:.2f}s > {duration:.2f}s), scaling by {scale:.2f}")
            adjusted_durations = [max(d, min_duration) * scale for d in natural_durations]
        else:
            adjusted_durations = [max(d, min_duration) for d in natural_durations]

        # Build timings
        timings = []
        current_time = start_time

        for i, part_duration in enumerate(adjusted_durations):
            if i == len(parts) - 1:
                # Last part gets remaining time to avoid floating-point drift
                part_end = end_time
            else:
                part_end = current_time + part_duration

            # Ensure we don't exceed end_time and have at least some duration
            part_end = min(part_end, end_time)
            if part_end <= current_time:
                # Safety: ensure at least 1ms duration
                part_end = min(current_time + 0.001, end_time)

            timings.append((current_time, part_end))
            current_time = part_end

        return timings
    
    def _detect_list_items(self, segments: List[dict]) -> List[dict]:
        """
        Detect numbered list items from voiceover segments using LLM.
        
        Uses Gemini to analyze the transcript and identify countdown/countup lists
        like "Top 10 cities", "Number 5: Denver", etc.
        
        Returns list of dicts with:
        - number: The list number (int)
        - text: The item text (e.g., "Austin, Texas")
        - segment_index: Which segment it was found in (approximate)
        """
        import json
        
        config = self.config
        list_config = getattr(config.keyword, 'list_detection', None)
        
        if not list_config or not getattr(list_config, 'enabled', False):
            return []
        
        # Combine all segment text for analysis
        full_text = " ".join(seg.get('text', '') for seg in segments[:200])  # First 200 segments
        
        if len(full_text) < 100:
            return []
        
        # Truncate if too long (keep first ~8000 chars for API limits)
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "..."
        
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                logger.warning("No GEMINI_API_KEY for list detection")
                return []
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""Analyze this transcript and identify if it contains a NUMBERED LIST or COUNTDOWN.

Look for patterns like:
- "Number 10: Austin, Texas" / "Number 9: Miami"
- "Top 10 cities where..." then "First... Second... Third..."
- "#5 Denver" / "#4 Seattle"
- "10. Austin" / "9. Miami" (at start of sentences)

TRANSCRIPT:
{full_text}

If this transcript contains a numbered list (like a "Top 10" countdown), extract each item.

Respond with JSON only:
{{
  "has_list": true/false,
  "list_type": "countdown" or "countup" or null,
  "items": [
    {{"number": 10, "name": "Austin, Texas"}},
    {{"number": 9, "name": "Miami, Florida"}},
    ...
  ]
}}

If no numbered list is found, respond with:
{{"has_list": false, "list_type": null, "items": []}}

IMPORTANT: Only extract ACTUAL numbered list items (cities, places, things being ranked).
Do NOT extract random numbers or percentages from the text."""

            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response (remove markdown code blocks if present)
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            data = json.loads(response_text)
            
            if not data.get('has_list') or not data.get('items'):
                logger.debug("LLM detected no list in transcript")
                return []
            
            list_items = []
            for item in data['items']:
                num = item.get('number', 0)
                name = item.get('name', '')
                
                if num > 0 and name and len(name) >= 2:
                    list_items.append({
                        'number': num,
                        'text': name,
                        'segment_index': 0  # Approximate
                    })
            
            # Sort by number (descending for countdowns)
            list_items.sort(key=lambda x: x['number'], reverse=True)
            
            if list_items:
                logger.info(f"LLM detected {len(list_items)} list items: {[i['text'] for i in list_items[:5]]}")
            
            return list_items
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM list detection response: {e}")
            return []
        except Exception as e:
            logger.warning(f"LLM list detection failed: {e}")
            return []
    
    def _generate_list_keywords(self, list_items: List[dict], entity_texts: set, suffix: str = "footage") -> List[str]:
        """
        Generate download keywords from detected list items.
        
        Args:
            list_items: Detected list items from _detect_list_items()
            entity_texts: Set of entity texts already covered (lowercase)
            suffix: Suffix to add to keywords (e.g., "footage")
            
        Returns:
            List of keywords, skipping items already covered by entities
        """
        keywords = []
        
        for item in list_items:
            item_text = item['text']
            item_lower = item_text.lower()
            
            # Skip if this item is already covered by entity extraction
            # Check if any entity contains this item or vice versa
            is_covered = False
            for entity in entity_texts:
                if item_lower in entity or entity in item_lower:
                    is_covered = True
                    logger.debug(f"List item '{item_text}' already covered by entity")
                    break
            
            if is_covered:
                continue
            
            # Generate keyword
            # Clean up: "Austin, Texas" -> "Austin Texas"
            clean_text = item_text.replace(',', ' ').replace('  ', ' ').strip()
            keyword = f"{clean_text} {suffix}"
            keywords.append(keyword)
        
        return keywords
    
    def _generate_topic_from_segments(self) -> str:
        """Generate topic context from voiceover segments using LLM or heuristics"""
        if not self.voiceover_segments:
            return ""
        
        # Combine first few segments for context
        text_samples = []
        for seg in self.voiceover_segments[:5]:
            if isinstance(seg, dict):
                text_samples.append(seg.get('text', '')[:100])
            elif hasattr(seg, 'text'):
                text_samples.append(seg.text[:100])
        
        combined_text = " ".join(text_samples)
        
        if not combined_text:
            return ""
        
        # Try LLM-based topic detection
        config = self.config
        
        try:
            gemini_key = getattr(config, 'gemini_api_key', None) or os.getenv('GEMINI_API_KEY')
            if gemini_key:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                prompt = f"""Identify the main topic of this voiceover in 2-5 words.

Text: {combined_text[:1500]}

Respond with ONLY the topic (2-5 words), nothing else."""

                response = model.generate_content(prompt)
                topic = response.text.strip().strip('"').strip("'")
                if topic and len(topic) < 100:
                    return topic
        except Exception as e:
            logger.debug(f"LLM topic detection failed: {e}")
        
        # Fallback: extract key terms
        words = combined_text.split()
        key_words = [w for w in words if len(w) > 4 and w[0].isupper()][:3]
        if key_words:
            return " ".join(key_words) + " documentary"
        
        return "documentary content"
    
    def _detect_topic_from_keywords(self, keywords: List[str]) -> str:
        """Detect main topic from extracted keywords using LLM"""
        if not keywords:
            return "documentary content"
        
        config = self.config
        
        # Format keywords for analysis
        keywords_text = ", ".join(keywords[:15])  # Use top 15 keywords
        
        try:
            gemini_key = getattr(config, 'gemini_api_key', None) or os.getenv('GEMINI_API_KEY')
            if gemini_key:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                prompt = f"""Based on these video search keywords, identify the MAIN TOPIC in 3-6 words.

Keywords: {keywords_text}

Respond with ONLY the topic (3-6 words), nothing else. Be specific.
Examples: "Wichita homeless opioid crisis", "Amazon rainforest deforestation", "Silicon Valley startup culture"

Topic:"""

                response = model.generate_content(prompt)
                topic = response.text.strip().strip('"').strip("'").strip()
                if topic and len(topic) < 100:
                    logger.info(f"Detected topic from keywords: {topic}")
                    return topic
        except Exception as e:
            logger.debug(f"LLM topic detection from keywords failed: {e}")
        
        # Fallback: analyze keywords manually
        return self._extract_topic_from_keywords_heuristic(keywords)
    
    def _extract_topic_from_keywords_heuristic(self, keywords: List[str]) -> str:
        """Extract topic from keywords using word frequency analysis"""
        from collections import Counter
        
        # Split all keywords into words
        all_words = []
        for kw in keywords[:20]:
            # Remove common suffixes
            kw_clean = kw.replace(' footage', '').replace(' video', '').replace(' documentary', '')
            words = kw_clean.split()
            all_words.extend([w for w in words if len(w) > 3])
        
        # Count word frequency
        word_counts = Counter(all_words)
        
        # Get top words (excluding very common ones)
        common_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'USA', 'America'}
        top_words = [word for word, count in word_counts.most_common(10) 
                     if word not in common_words and count > 1]
        
        if top_words:
            # Take top 3-4 most frequent words
            topic_words = top_words[:4]
            return " ".join(topic_words) + " documentary"
        
        # Fallback: use first keyword
        if keywords:
            first_kw = keywords[0].replace(' footage', '').replace(' video', '')
            return first_kw[:50]
        
        return "documentary content"
    
    def _attach_entities_to_segments(self):
        """
        Attach extracted entities to their corresponding voiceover segments.
        This enables entity markers in OTIO/EDL output for text overlays.
        """
        if not self.extracted_entities or not self.voiceover_segments:
            return
        
        entities_attached = 0
        
        for seg in self.voiceover_segments:
            if isinstance(seg, dict):
                seg_text = seg.get('text', '').lower()
                seg_entities = []
                
                for entity in self.extracted_entities:
                    entity_text = entity.get('text', '')
                    if entity_text and entity_text.lower() in seg_text:
                        seg_entities.append(entity)
                
                if seg_entities:
                    seg['entities'] = seg_entities
                    entities_attached += len(seg_entities)
        
        logger.debug(f"Attached {entities_attached} entity references to voiceover segments")
    
    def _get_user_confirmation(self, prompt: str, default: bool = True) -> bool:
        """Get Y/N confirmation from user"""
        suffix = "[Y/n]" if default else "[y/N]"
        try:
            response = input(f"{prompt} {suffix}: ").strip().lower()
            if not response:
                return default
            return response in ('y', 'yes')
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
    
    def _get_user_input(self, prompt: str, default: str = "") -> str:
        """Get text input from user"""
        suffix = f"[{default}]" if default else ""
        try:
            response = input(f"{prompt} {suffix}: ").strip()
            return response if response else default
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
    
    # =========================================================================
    # ENHANCED FEATURES (Config-Driven)
    # =========================================================================
    
    def stage_download_stock(self, keywords: List[str]) -> dict:
        """Stage 2c: Download stock footage from Pexels/Pixabay (config-driven)"""
        config = self.config
        
        if not self.enhanced_enabled:
            return {'skipped': True}
        
        if not self.modules['pexels'] and not self.modules['pixabay']:
            return {'skipped': True, 'reason': 'no stock modules'}
        
        self._print_stage("2c", "DOWNLOAD STOCK FOOTAGE")
        
        stock_dir = Path(config.downloaded_videos_dir) / "stock"
        stock_dir.mkdir(parents=True, exist_ok=True)
        
        total = 0
        stock_keywords = keywords[:15]  # Limit for rate limits
        per_keyword = config.enhanced.stock_per_keyword
        
        # Pexels (config-driven)
        if self.modules['pexels'] and config.enhanced.enable_pexels:
            try:
                from src.pexels import download_pexels_footage
                print(f"  Pexels: Searching {len(stock_keywords)} keywords...")
                paths, counts = download_pexels_footage(
                    stock_keywords, 
                    str(stock_dir), 
                    per_keyword=per_keyword
                )
                total += len(paths)
                print(f"  ✓ Pexels: {len(paths)} videos")
                
                for kw, count in counts.items():
                    if count == 0 and kw not in self.failed_keywords:
                        self.failed_keywords.append(kw)
            except Exception as e:
                print(f"  ⚠ Pexels error: {e}")
        
        # Pixabay (config-driven)
        if self.modules['pixabay'] and config.enhanced.enable_pixabay:
            try:
                from src.pixabay import download_pixabay_footage
                print(f"  Pixabay: Searching {len(stock_keywords)} keywords...")
                paths, counts = download_pixabay_footage(
                    stock_keywords, 
                    str(stock_dir), 
                    per_keyword=per_keyword
                )
                total += len(paths)
                print(f"  ✓ Pixabay: {len(paths)} videos")
                
                for kw, count in counts.items():
                    if count == 0 and kw not in self.failed_keywords:
                        self.failed_keywords.append(kw)
            except Exception as e:
                print(f"  ⚠ Pixabay error: {e}")
        
        print(f"  ✓ Total stock footage: {total} videos")
        return {'downloaded': total}
    
    def stage_remix_zero_downloads(self, keywords: List[str]) -> List[str]:
        """Remix keywords that had zero downloads (config-driven)"""
        config = self.config
        
        if not self.enhanced_enabled or not self.keyword_remixer:
            return keywords
        
        if not self.failed_keywords:
            return keywords
        
        remix_count = config.enhanced.remix_keyword_count
        print(f"\n  Remixing {len(self.failed_keywords)} zero-download keywords...")
        
        new_keywords = []
        for kw in self.failed_keywords[:10]:
            try:
                remixed = self.keyword_remixer.remix_for_zero_downloads(kw, attempt=1)
                if remixed:
                    print(f"    '{kw}' → {remixed[:2]}")
                    new_keywords.extend(remixed[:remix_count])
            except Exception as e:
                logger.debug(f"Remix failed for '{kw}': {e}")
        
        return keywords + new_keywords
    
    # =========================================================================
    # MAIN PIPELINE STAGES (Config-Driven)
    # =========================================================================
    
    def _load_voiceover_segments(self, voiceover_path: str, quiet: bool = False) -> List[dict]:
        """
        Load voiceover segments from SRT or transcribe audio/video file.

        Args:
            voiceover_path: Path to voiceover file (.srt, .mp3, .wav, etc.)
            quiet: If True, suppress print statements

        Returns:
            List of segment dictionaries
        """
        config = self.config
        vo_path = Path(voiceover_path)

        if vo_path.suffix.lower() == '.srt':
            if not quiet:
                print(f"  Parsing SRT: {vo_path.name}")
            return self._parse_srt(str(vo_path))
        else:
            # Transcribe audio/video
            if self.modules['optimized_transcription']:
                from src.transcription import transcribe_voiceover_media
                if not quiet:
                    print(f"  Transcribing: {vo_path.name}")

                # Generate SRT path
                srt_path = vo_path.with_suffix('.srt')

                # Call transcription with correct arguments
                result_srt = transcribe_voiceover_media(
                    str(vo_path),
                    output_srt_path=str(srt_path),
                    model_name=config.transcription.model,
                    language=config.transcription.language if config.transcription.language != 'auto' else None,
                    compute_type=config.transcription.compute_type,
                    cache_dir=config.cache.cache_dir if config.cache.cache_transcriptions else None
                )

                if not quiet:
                    print(f"  ✓ Transcription saved to: {Path(result_srt).name}")
                return self._parse_srt(result_srt)
            else:
                logger.error("Transcription module not available for non-SRT files")
                sys.exit(1)

    def stage_analyze_voiceover(self, voiceover_path: str, num_keywords: int = None) -> List[str]:
        """
        Stage 1: Analyze voiceover and extract keywords.
        All parameters from config unless overridden.
        """
        config = self.config

        # Use config default if not specified
        if num_keywords is None:
            num_keywords = config.keyword.max_keywords

        self._print_stage("1", "ANALYZE VOICEOVER")

        # Load voiceover segments (handles both SRT and audio files)
        self.voiceover_segments = self._load_voiceover_segments(voiceover_path)

        print(f"  ✓ {len(self.voiceover_segments)} segments found")
        
        # Extract keywords
        print(f"\n  Extracting keywords (max {num_keywords})...")
        
        try:
            from src.keyword_extractor import LLMKeywordExtractor
            extractor = LLMKeywordExtractor(config=config)
            # Pass segments (list of dicts), get KeywordResult back
            result = extractor.extract_keywords(self.voiceover_segments, max_keywords=num_keywords)
            self.keywords = result.keywords
            
            # Store entities with context for display
            self.extracted_entities = result.entities if result.entities else []
            
            # Attach entities to voiceover segments for EDL/OTIO markers
            if self.extracted_entities:
                self._attach_entities_to_segments()
            
            # Detect topic from extracted keywords (not raw text)
            if self.keywords:
                self.topic_context = self._detect_topic_from_keywords(self.keywords)
                print(f"  Detected topic: {self.topic_context}")
            elif result.topic:
                self.topic_context = result.topic
                print(f"  Detected topic: {self.topic_context}")
            
            # Show entity count
            if self.extracted_entities:
                print(f"  Entities found: {len(self.extracted_entities)}")
                    
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            # TF-IDF fallback
            if config.keyword.use_tfidf_weights:
                print("  Falling back to TF-IDF extraction...")
                voiceover_text = " ".join([s.get('text', '') for s in self.voiceover_segments])
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
                try:
                    vectorizer.fit_transform([voiceover_text])
                    self.keywords = list(vectorizer.get_feature_names_out())
                except:
                    self.keywords = []
        
        print(f"  ✓ {len(self.keywords)} keywords extracted")

        # Detect chapters/topics in voiceover for chapter-based matching
        if config.matching.chapter_matching_enabled:
            self._detect_voiceover_chapters()

        return self.keywords

    def _detect_voiceover_chapters(self):
        """Detect chapters/topic sections in voiceover for chapter-based matching."""
        if not self.voiceover_segments:
            return

        try:
            from src.topic_extraction import ChapterDetector

            print(f"\n  Detecting chapters...")
            detector = ChapterDetector(self.config)

            # Detect chapters using LLM
            self.chapters = detector.detect_chapters(
                self.voiceover_segments,
                overall_topic=self.topic_context if hasattr(self, 'topic_context') else None
            )

            if self.chapters:
                print(f"  ✓ Found {len(self.chapters)} chapters:")
                for ch in self.chapters:
                    start_idx = ch.get('start_segment_idx', 0)
                    end_idx = ch.get('end_segment_idx', 0)
                    title = ch.get('title', 'Untitled')
                    topics = ch.get('topics', [])
                    topics_str = ', '.join(topics[:3]) if topics else 'no topics'
                    print(f"    [{start_idx}-{end_idx}] {title} ({topics_str})")

                # Assign chapter info to voiceover segments
                self._assign_chapters_to_segments()
            else:
                print(f"  ✓ No distinct chapters detected (treating as single topic)")

            # Detect location-focused chapters if enabled
            self._detect_location_chapters(detector)

        except Exception as e:
            logger.warning(f"Chapter detection failed: {e}")
            self.chapters = []

    def _detect_location_chapters(self, detector=None):
        """Detect location-focused chapters for location-aware matching."""
        # Check if location matching is enabled
        location_config = getattr(self.config.matching, 'location_matching', None)
        if not location_config:
            self.location_chapters = []
            return

        if isinstance(location_config, dict):
            enabled = location_config.get('enabled', False)
        else:
            enabled = getattr(location_config, 'enabled', False)

        if not enabled:
            self.location_chapters = []
            return

        try:
            from src.topic_extraction import ChapterDetector
            from src.location_service import create_location_service

            if detector is None:
                detector = ChapterDetector(self.config)

            # Initialize location service
            location_service = create_location_service(self.config)

            print(f"\n  Detecting location-focused chapters...")
            logger.info("Detecting location-focused chapters...")
            self.location_chapters = detector.detect_location_chapters(
                self.voiceover_segments,
                location_service=location_service,
                overall_topic=self.topic_context if hasattr(self, 'topic_context') else None
            )

            if self.location_chapters:
                print(f"  ✓ Found {len(self.location_chapters)} location chapters:")
                logger.info(f"Found {len(self.location_chapters)} location chapters")
                for lc in self.location_chapters:
                    loc_name = lc.location_name
                    loc_type = lc.location_type
                    country = ""
                    if lc.location_data:
                        country = lc.location_data.get('country_name', '')
                    print(f"    [{lc.start_segment_idx}-{lc.end_segment_idx}] {loc_name} ({loc_type}) {country}")

                # Store location service for video location extraction
                self._location_service = location_service
            else:
                print(f"  ✓ No location-focused chapters detected")
                logger.info("No location-focused chapters detected")
                self._location_service = None

        except Exception as e:
            logger.warning(f"Location chapter detection failed: {e}")
            self.location_chapters = []
            self._location_service = None

    def _extract_video_locations(self):
        """Extract and resolve locations from video metadata."""
        if not hasattr(self, '_location_service') or not self._location_service:
            self.video_locations = {}
            return

        if not self.location_chapters:
            self.video_locations = {}
            return

        try:
            from src.topic_extraction import extract_video_locations_batch

            # Build video metadata list
            videos = []
            if hasattr(self, 'downloaded_videos') and self.downloaded_videos:
                for dv in self.downloaded_videos:
                    if hasattr(dv, 'to_dict'):
                        videos.append(dv.to_dict())
                    elif isinstance(dv, dict):
                        videos.append(dv)

            if not videos:
                # Fallback: extract from text_metadata if available
                if hasattr(self, 'text_metadata') and self.text_metadata:
                    seen_paths = set()
                    for meta in self.text_metadata:
                        if isinstance(meta, dict):
                            path = meta.get('video_path', '')
                            title = meta.get('title', meta.get('video_title', ''))
                            keyword = meta.get('keyword', meta.get('source_keyword', ''))
                        else:
                            path = getattr(meta, 'source_file', '')
                            title = getattr(meta, 'title', '')
                            keyword = getattr(meta, 'keyword', '')

                        if path and path not in seen_paths:
                            seen_paths.add(path)
                            videos.append({
                                'file': path,
                                'title': title,
                                'keyword': keyword
                            })

            if videos:
                print(f"\n  Extracting video locations for {len(videos)} videos...")
                self.video_locations = extract_video_locations_batch(
                    videos=videos,
                    config=self.config,
                    location_service=self._location_service
                )
                if self.video_locations:
                    print(f"  ✓ Resolved locations for {len(self.video_locations)} videos")
            else:
                self.video_locations = {}

        except Exception as e:
            logger.warning(f"Video location extraction failed: {e}")
            self.video_locations = {}

    def _assign_chapters_to_segments(self):
        """Assign chapter_id and chapter_topics to each voiceover segment."""
        if not self.chapters:
            return

        for ch in self.chapters:
            start_idx = ch.get('start_segment_idx', 0)
            end_idx = ch.get('end_segment_idx', len(self.voiceover_segments) - 1)
            chapter_id = ch.get('chapter_id', 0)
            topics = ch.get('topics', [])

            for i in range(start_idx, min(end_idx + 1, len(self.voiceover_segments))):
                seg = self.voiceover_segments[i]
                if isinstance(seg, dict):
                    seg['chapter_id'] = chapter_id
                    seg['chapter_topics'] = topics
                else:
                    # Handle SRTSegment objects - store in topics field
                    if hasattr(seg, 'topics'):
                        seg.topics = topics

    def stage_image_search(self) -> Dict[str, any]:
        """
        Stage 1.5: Download images for entities.
        Searches Google/Bing/Stock APIs for images representing
        people, places, organizations mentioned in voiceover.
        """
        global PROJECT_DIR
        config = self.config
        
        if not config.image_search.enabled:
            logger.debug("Image search disabled in config")
            return {}
        
        if not hasattr(self, 'extracted_entities') or not self.extracted_entities:
            logger.debug("No entities available for image search")
            return {}
        
        self._print_stage("1.5", "ENTITY IMAGE SEARCH")
        
        try:
            from src.entity_images import download_entity_images, map_entities_to_segments
            
            # Filter entities by configured types
            allowed_types = config.image_search.entity_types
            entities_to_search = [
                e for e in self.extracted_entities
                if e.get('type', '') in allowed_types
            ]
            
            if not entities_to_search:
                print(f"  No entities of types {allowed_types} to search")
                return {}
            
            # Apply max_entities limit if configured
            max_entities = getattr(config.image_search, 'max_entities', 0)
            if max_entities > 0 and len(entities_to_search) > max_entities:
                print(f"  Limiting to {max_entities} entities (from {len(entities_to_search)})")
                entities_to_search = entities_to_search[:max_entities]
            
            print(f"  Searching images for {len(entities_to_search)} entities")
            print(f"  Entity types: {', '.join(allowed_types)}")
            print(f"  Images per entity: {config.image_search.images_per_entity}")
            print(f"  Minimum size: {config.image_search.min_size_mb}MB")
            
            # Get output directory - USE SHORT PATHS if configured
            image_cfg = config.image_search
            
            if getattr(image_cfg, 'root_dir', '') and image_cfg.root_dir:
                # Use explicit root_dir (e.g., "E:/i")
                project_name = PROJECT_DIR.name[:15] if PROJECT_DIR else "project"
                output_dir = Path(image_cfg.root_dir) / project_name
            elif PROJECT_DIR:
                # Use project-relative path
                folder_name = getattr(image_cfg, 'folder_name', 'images')
                output_dir = PROJECT_DIR / folder_name
            else:
                output_dir = Path(image_cfg.output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Output directory: {output_dir}")
            
            # Download images
            entity_results = download_entity_images(
                entities=entities_to_search,
                output_dir=str(output_dir),
                topic=self.topic_context or "",
                images_per_entity=config.image_search.images_per_entity,
                min_size_mb=config.image_search.min_size_mb,
                use_google=config.image_search.use_google,
                use_bing=getattr(config.image_search, 'use_bing', False),  # Disabled by default
                use_stock_apis=config.image_search.use_stock_apis,
                pexels_key=os.getenv("PEXELS_API_KEY"),
                pixabay_key=os.getenv("PIXABAY_API_KEY"),
                download_timeout=getattr(config.image_search, 'download_timeout', 10),
                max_search_time=getattr(config.image_search, 'max_search_time', 300),
                max_results_to_check=getattr(config.image_search, 'max_results_to_check', 500),
                search_until_found=getattr(config.image_search, 'search_until_found', True)
            )
            
            # Map entities to segments for timeline placement
            if entity_results:
                entity_segments = map_entities_to_segments(
                    entities_to_search,
                    self.voiceover_segments
                )
                
                # Update entity results with segment info
                for entity_name, result in entity_results.items():
                    result.segment_indices = entity_segments.get(entity_name, [])
                
                self.entity_images = entity_results
                
                # Summary
                total_images = sum(len(r.images) for r in entity_results.values())
                print(f"\n  ✓ Downloaded {total_images} images for {len(entity_results)} entities")
                
                # Show what was found
                for name, result in list(entity_results.items())[:5]:
                    segments_str = f"segments: {result.segment_indices[:3]}" if result.segment_indices else "no segment matches"
                    print(f"    • {name} ({result.entity_type}): {len(result.images)} images, {segments_str}")
                
                if len(entity_results) > 5:
                    print(f"    ... and {len(entity_results) - 5} more entities")
            else:
                print(f"  ⚠ No images downloaded")
                self.entity_images = {}
            
            return entity_results
            
        except ImportError as e:
            logger.error(f"Could not import imagedl: {e}")
            print(f"  ⚠ Image search module not available")
            return {}
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  ⚠ Image search failed: {e}")
            return {}
    
    def stage_stock_video(self) -> Dict[str, any]:
        """
        Stage 1.6: Download stock videos for entities.
        Searches Pexels/Pixabay APIs for stock footage representing
        people, places, organizations mentioned in voiceover.
        """
        global PROJECT_DIR
        config = self.config
        
        # Check if stock video search is enabled (use same config as image_search)
        if not config.image_search.enabled:
            logger.debug("Image/video search disabled in config")
            return {}
        
        if not config.image_search.use_stock_apis:
            logger.debug("Stock APIs disabled in config")
            return {}
        
        if not hasattr(self, 'extracted_entities') or not self.extracted_entities:
            logger.debug("No entities available for stock video search")
            return {}
        
        self._print_stage("1.6", "STOCK VIDEO SEARCH")
        
        try:
            from src.entity_images import download_entity_videos, map_entities_to_segments
            
            # Filter entities by configured types
            allowed_types = config.image_search.entity_types
            entities_to_search = [
                e for e in self.extracted_entities
                if e.get('type', '') in allowed_types
            ]
            
            if not entities_to_search:
                print(f"  No entities of types {allowed_types} to search")
                return {}
            
            # Apply max_entities limit if configured
            max_entities = getattr(config.image_search, 'max_entities', 0)
            if max_entities > 0 and len(entities_to_search) > max_entities:
                print(f"  Limiting to {max_entities} entities (from {len(entities_to_search)})")
                entities_to_search = entities_to_search[:max_entities]
            
            # Get videos_per_entity from config (default 3)
            videos_per_entity = getattr(config.image_search, 'videos_per_entity', 3)
            
            print(f"  Searching stock videos for {len(entities_to_search)} entities")
            print(f"  Entity types: {', '.join(allowed_types)}")
            print(f"  Videos per entity: {videos_per_entity}")
            
            # Get output directory - USE SHORT PATHS if configured
            image_cfg = config.image_search
            
            if getattr(image_cfg, 'root_dir', '') and image_cfg.root_dir:
                # Use explicit root_dir (e.g., "E:/i")
                project_name = PROJECT_DIR.name[:15] if PROJECT_DIR else "project"
                output_dir = Path(image_cfg.root_dir) / project_name
            elif PROJECT_DIR:
                # Use project-relative path
                folder_name = getattr(image_cfg, 'folder_name', 'images')
                output_dir = PROJECT_DIR / folder_name
            else:
                output_dir = Path(image_cfg.output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Output directory: {output_dir}")
            
            # Download stock videos
            entity_results = download_entity_videos(
                entities=entities_to_search,
                output_dir=str(output_dir),
                topic=self.topic_context or "",
                videos_per_entity=videos_per_entity,
                min_duration=3.0,
                max_duration=30.0,
                pexels_key=os.getenv("PEXELS_API_KEY"),
                pixabay_key=os.getenv("PIXABAY_API_KEY")
            )
            
            # Map entities to segments for timeline placement
            if entity_results:
                entity_segments = map_entities_to_segments(
                    entities_to_search,
                    self.voiceover_segments
                )
                
                # Update entity results with segment info
                for entity_name, result in entity_results.items():
                    result.segment_indices = entity_segments.get(entity_name, [])
                
                self.entity_videos = entity_results
                
                # Summary
                total_videos = sum(len(r.videos) for r in entity_results.values())
                print(f"\n  ✓ Downloaded {total_videos} stock videos for {len(entity_results)} entities")
                
                # Show what was found
                for name, result in list(entity_results.items())[:5]:
                    segments_str = f"segments: {result.segment_indices[:3]}" if result.segment_indices else "no segment matches"
                    print(f"    • {name} ({result.entity_type}): {len(result.videos)} videos, {segments_str}")
                
                if len(entity_results) > 5:
                    print(f"    ... and {len(entity_results) - 5} more entities")
            else:
                print(f"  ⚠ No stock videos downloaded")
                self.entity_videos = {}
            
            return entity_results
            
        except ImportError as e:
            logger.error(f"Could not import entity_images: {e}")
            print(f"  ⚠ Stock video module not available")
            return {}
        except Exception as e:
            logger.error(f"Stock video search failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  ⚠ Stock video search failed: {e}")
            return {}
    
    def _check_global_cache_for_videos(self, keywords: List[str]) -> Tuple[List[str], List[dict]]:
        """
        Check global cache for relevant videos before downloading.

        Returns:
            Tuple of (keywords_to_download, reusable_videos)
            - keywords_to_download: Keywords that need new downloads
            - reusable_videos: Videos from global cache that can be reused
        """
        if not self.global_cache:
            return keywords, []

        gc_config = getattr(self.config, 'global_cache', None)
        if not gc_config or not getattr(gc_config, 'check_before_download', True):
            return keywords, []

        print(f"\n  🔍 Checking global cache for existing videos...")

        # Get topics from voiceover for relevance filtering
        vo_topics = []
        if hasattr(self, 'voiceover_segments') and self.voiceover_segments:
            # Extract topics from voiceover text
            vo_text = ' '.join([
                s.text if hasattr(s, 'text') else s.get('text', '')
                for s in self.voiceover_segments[:10]  # First 10 segments
            ])
            vo_topics = [kw.lower() for kw in keywords[:5]]  # Use first 5 keywords as topics
            if self.topic_context:
                vo_topics.append(self.topic_context.lower())

        # Query global cache
        min_relevance = getattr(gc_config, 'min_topic_overlap', 0.3)
        max_reuse = getattr(gc_config, 'max_reuse_videos', 50)

        result = self.global_cache.find_videos_for_keywords(
            keywords=keywords,
            topics=vo_topics,
            min_relevance=min_relevance,
            max_results=max_reuse
        )

        logger.info(f"Global cache query: {result.total_cached_matches} matches, "
                   f"{result.files_exist_count} exist, {result.files_deleted_count} deleted")

        reusable_videos = []
        covered_keywords = set()

        # Process reusable videos
        for entry, relevance in result.reuse_videos:
            if entry.file_exists and entry.current_path:
                video_info = {
                    'path': entry.current_path,
                    'source': 'global_cache',
                    'relevance': relevance,
                    'topics': entry.topics,
                    'keywords': entry.keywords,
                    'has_transcript': entry.has_transcript,
                    'video_hash': entry.video_hash,
                    'face_score': entry.face_score  # Cached face score from global cache
                }
                reusable_videos.append(video_info)

                # Track which keywords are covered
                if entry.download_info and entry.download_info.keyword:
                    covered_keywords.add(entry.download_info.keyword.lower())
                for kw in entry.keywords:
                    covered_keywords.add(kw.lower())

        # Determine which keywords still need downloads
        keywords_to_download = []
        for kw in keywords:
            if kw.lower() not in covered_keywords:
                keywords_to_download.append(kw)

        if reusable_videos:
            print(f"  ✓ Found {len(reusable_videos)} reusable videos from global cache")
            for v in reusable_videos[:3]:
                print(f"    • {Path(v['path']).name} (relevance: {v['relevance']:.2f})")
            if len(reusable_videos) > 3:
                print(f"    ... and {len(reusable_videos) - 3} more")

            self.global_cache_videos = reusable_videos
            logger.info(f"Reusing {len(reusable_videos)} videos from global cache")

        if result.redownload_keywords:
            print(f"  📥 {len(result.redownload_keywords)} cached videos were deleted, adding to download queue")
            keywords_to_download.extend(result.redownload_keywords)

        if result.uncovered_keywords:
            logger.debug(f"Uncovered keywords: {result.uncovered_keywords}")

        print(f"  → {len(keywords_to_download)} keywords need new downloads")

        return keywords_to_download, reusable_videos

    def stage_download(self, keywords: List[str]) -> List[dict]:
        """
        Stage 2: Download footage from YouTube.
        Duration tiers and settings from config.
        Includes zero-download keyword remix for failed keywords.
        """
        config = self.config

        if config.pipeline.skip_download:
            print("  ⏭ Skipping download (config: skip_download=true)")
            logger.info("Skipping DOWNLOAD stage (config: skip_download=true)")
            return []

        self._print_stage("2", "DOWNLOAD FOOTAGE")

        # Check global cache first
        keywords_to_download, reusable_videos = self._check_global_cache_for_videos(keywords)

        try:
            from src.downloader import VideoDownloader, DownloadCheckpoint

            # Initialize downloader with config
            self.downloader = VideoDownloader(config=config)
            
            # Get duration tier settings from downloader (reflects merged config)
            print(f"  Duration tiers:")
            for tier_name in ['short', 'medium', 'long', 'longer']:
                min_s = self.downloader._get_tier_value(tier_name, 'min', 0)
                max_s = self.downloader._get_tier_value(tier_name, 'max', 120)
                per_kw = self.downloader._get_tier_value(tier_name, 'per_keyword', 5)
                print(f"    • {tier_name}: {min_s}-{max_s}s ({per_kw}/kw)")
            
            # Download using download_all() method (only for keywords not covered by cache)
            output_dir = Path(config.downloaded_videos_dir)

            if keywords_to_download:
                downloaded_videos, failed = self.downloader.download_all(
                    keywords=keywords_to_download,
                    output_dir=output_dir,
                    resume=True,
                    topic=self.topic_context or ""
                )
            else:
                downloaded_videos = []
                failed = []
                print(f"  ⏭ All keywords covered by global cache, skipping download")

            # Combine newly downloaded videos with reusable videos from global cache
            # Current project videos come first (higher priority)
            self.downloaded_videos = downloaded_videos

            # Add global cache videos (marked with source='global_cache')
            if reusable_videos:
                for gv in reusable_videos:
                    # Convert to format expected by rest of pipeline
                    self.downloaded_videos.append({
                        'path': gv['path'],
                        'file': gv['path'],
                        'source': 'global_cache',
                        'relevance': gv.get('relevance', 0.5),
                        'video_hash': gv.get('video_hash', ''),
                        'face_score': gv.get('face_score', 0.5)  # Preserve cached face score
                    })

            # Store failed keywords for potential remix
            self.failed_keywords = failed

            if failed:
                print(f"  ⚠ Failed keywords: {', '.join(failed[:5])}" +
                      (f" (+{len(failed)-5} more)" if len(failed) > 5 else ""))

            print(f"\n  ✓ Downloaded {len(downloaded_videos)} new videos")
            if reusable_videos:
                print(f"  ✓ Reusing {len(reusable_videos)} videos from global cache")
            
            # Zero-download keyword remix
            if failed and hasattr(config, 'zero_download_remix') and config.zero_download_remix.enabled:
                self._stage_zero_download_remix(failed, output_dir)
            
        except ImportError as e:
            logger.error(f"Could not import downloader: {e}")
            return []
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return []
        
        return self.downloaded_videos

    # =========================================================================
    # AUDIO-FIRST PIPELINE STAGES
    # =========================================================================

    def stage_download_audio(self, keywords: List[str]) -> List:
        """
        Stage 2A (Audio-First): Download audio only for transcription.

        Downloads lightweight MP3 files for fast transcription and matching,
        before deciding which video segments to download.
        """
        from src.downloader import AudioDownload

        config = self.config
        audio_config = getattr(config.download, 'audio_first', None)

        if not audio_config or not getattr(audio_config, 'enabled', False):
            logger.warning("Audio-first mode not enabled")
            return []

        self._print_stage("2A", "DOWNLOAD AUDIO (Audio-First Mode)")

        print(f"  📢 Audio-first mode: Downloading audio for transcription")
        print(f"    Buffer: {getattr(audio_config, 'buffer_seconds', 30)}s")
        print(f"    Merge gap: {getattr(audio_config, 'merge_gap_seconds', 15)}s")

        from src.downloader import VideoDownloader

        # Initialize downloader
        self.downloader = VideoDownloader(config=config)

        output_dir = Path(config.downloaded_videos_dir)
        all_audio_downloads = []

        # Download audio for each keyword and tier
        tiers = list(self.downloader.DURATION_TIERS.keys())
        total_keywords = len(keywords)
        failed_keywords = []

        for idx, keyword in enumerate(keywords, 1):
            print(f"\n  [{idx}/{total_keywords}] {keyword}")

            try:
                for tier in tiers:
                    per_kw = self.downloader._get_tier_value(tier, 'per_keyword', 5)
                    if per_kw <= 0:
                        continue

                    audio_downloads = self.downloader.download_audio_for_keyword(
                        keyword=keyword,
                        output_dir=output_dir,
                        tier=tier,
                        topic=self.topic_context or ""
                    )
                    all_audio_downloads.extend(audio_downloads)
            except Exception as e:
                logger.error(f"Audio download failed for keyword '{keyword}': {e}")
                import traceback
                traceback.print_exc()
                failed_keywords.append(keyword)
                # Continue with next keyword - don't lose partial downloads

        # Store for later use - even if some keywords failed
        self.audio_downloads = all_audio_downloads
        self.audio_downloads_by_id = {a.video_id: a for a in all_audio_downloads}

        if failed_keywords:
            print(f"\n  ⚠ {len(failed_keywords)} keywords failed: {failed_keywords}")
        print(f"\n  ✓ Downloaded {len(all_audio_downloads)} audio files")
        return all_audio_downloads

    def stage_download_video_segments(self) -> List:
        """
        Stage 2B (Audio-First): Download only matched video segments.

        Called after matching to download only the portions of videos
        that are actually used in the timeline.
        """
        from src.downloader import (
            collect_matched_segments,
            prepare_merged_segments,
            DownloadedSegment
        )

        config = self.config
        audio_config = getattr(config.download, 'audio_first', None)

        if not hasattr(self, 'matches') or not self.matches:
            logger.error("No match results - run matching first")
            return []

        if not hasattr(self, 'audio_downloads_by_id'):
            logger.error("No audio downloads - run audio download first")
            return []

        self._print_stage("2B", "DOWNLOAD VIDEO SEGMENTS")

        buffer_seconds = getattr(audio_config, 'buffer_seconds', 30.0)
        merge_gap = getattr(audio_config, 'merge_gap_seconds', 15.0)

        print(f"  🎬 Downloading matched video segments")
        print(f"    Buffer: {buffer_seconds}s, Merge gap: {merge_gap}s")

        try:
            # Collect matched segments from results
            segments_by_video = collect_matched_segments(
                self.matches,
                self.audio_downloads_by_id
            )

            total_matches = sum(len(segs) for segs in segments_by_video.values())
            print(f"    Matched segments: {total_matches} across {len(segments_by_video)} videos")

            # Merge segments with buffer
            merged_segments = prepare_merged_segments(
                segments_by_video,
                self.audio_downloads_by_id,
                buffer_seconds=buffer_seconds,
                merge_gap_seconds=merge_gap
            )

            print(f"    After merge: {len(merged_segments)} segments to download")

            # Download video segments
            output_dir = Path(config.downloaded_videos_dir)
            downloaded_segments = self.downloader.download_video_segments(
                merged_segments,
                output_dir
            )

            # Store for OTIO building
            self.downloaded_segments = downloaded_segments
            self.downloaded_segments_by_id = {}
            for seg in downloaded_segments:
                key = f"{seg.video_id}_{int(seg.original_start)}"
                self.downloaded_segments_by_id[key] = seg

            print(f"\n  ✓ Downloaded {len(downloaded_segments)} video segments")
            return downloaded_segments

        except Exception as e:
            logger.error(f"Video segment download failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _is_audio_first_enabled(self) -> bool:
        """Check if audio-first mode is enabled."""
        audio_config = getattr(self.config.download, 'audio_first', None)
        return audio_config and getattr(audio_config, 'enabled', False)

    def _rebuild_audio_downloads_from_disk(self):
        """Rebuild audio_downloads from audio files on disk when resuming."""
        from src.downloader import AudioDownload

        config = self.config
        output_dir = Path(config.downloaded_videos_dir)

        if not output_dir.exists():
            logger.warning(f"Audio directory not found: {output_dir}")
            return

        # Find all audio directories (ending with _audio)
        audio_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.endswith('_audio')]

        audio_downloads = []
        audio_extensions = {'.mp3', '.m4a', '.opus', '.ogg', '.wav', '.flac'}

        for audio_dir in audio_dirs:
            # Extract keyword from dir name (e.g., "Manitoba_s_audio" -> "Manitoba")
            keyword = audio_dir.name.rsplit('_', 2)[0] if '_' in audio_dir.name else audio_dir.name

            for audio_file in audio_dir.iterdir():
                if audio_file.suffix.lower() in audio_extensions:
                    video_id = audio_file.stem
                    audio_downloads.append(AudioDownload(
                        audio_file=str(audio_file),
                        video_id=video_id,
                        video_url=f"https://www.youtube.com/watch?v={video_id}",
                        title="",  # Unknown when rebuilding
                        channel="",
                        duration=0,  # Unknown when rebuilding
                        keyword=keyword,
                        duration_tier="short",  # Default
                        upload_date="",
                        license="Unknown"
                    ))

        self.audio_downloads = audio_downloads
        self.audio_downloads_by_id = {a.video_id: a for a in audio_downloads}
        print(f"  Rebuilt {len(audio_downloads)} audio downloads from disk")

    def _load_existing_videos(self):
        """
        Load existing videos from output directory when skipping downloads.
        Used when skip_download=true to use previously downloaded videos.
        
        Search order:
        1. pipeline.video_source_dir (if set in config - ABSOLUTE PATH)
        2. {project_dir}/downloaded_videos/
        3. {output_dir}/downloaded_videos/
        4. {cwd}/downloaded_videos/
        """
        global PROJECT_DIR
        config = self.config
        
        # Try multiple possible locations for downloaded videos
        possible_dirs = []
        seen_dirs = set()  # Track seen directories to avoid duplicates
        
        def add_dir(d):
            """Add directory if not already seen"""
            d = Path(d).resolve()
            if d not in seen_dirs:
                seen_dirs.add(d)
                possible_dirs.append(d)
        
        # 0. Explicit video_source_dir from config (FIRST PRIORITY)
        if hasattr(config.pipeline, 'video_source_dir') and config.pipeline.video_source_dir:
            explicit_dir = Path(config.pipeline.video_source_dir)
            add_dir(explicit_dir)
            add_dir(explicit_dir / "downloaded_videos")
        
        # 1. PROJECT_DIR (from --project argument) - most likely location
        if PROJECT_DIR:
            add_dir(PROJECT_DIR / "downloaded_videos")
            add_dir(PROJECT_DIR / "output" / "downloaded_videos")
        
        # 2. Output directory / downloaded_videos
        output_dir = Path(config.output.output_dir)
        if not output_dir.is_absolute() and PROJECT_DIR:
            output_dir = PROJECT_DIR / output_dir
        add_dir(output_dir / "downloaded_videos")
        add_dir(output_dir.parent / "downloaded_videos")
        
        # 3. Current working directory (last resort)
        add_dir(Path.cwd() / "downloaded_videos")
        
        # Find videos - search recursively in subfolders too
        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
        videos_dir = None
        video_files = []
        
        for test_dir in possible_dirs:
            if not test_dir.exists() or not test_dir.is_dir():
                continue
            
            # First check direct files
            for f in test_dir.iterdir():
                if f.is_file() and f.suffix.lower() in video_extensions:
                    video_files.append(f)
            
            # If no direct files, search subfolders (one level deep)
            if not video_files:
                for subdir in test_dir.iterdir():
                    if subdir.is_dir():
                        for f in subdir.iterdir():
                            if f.is_file() and f.suffix.lower() in video_extensions:
                                video_files.append(f)
            
            if video_files:
                videos_dir = test_dir
                break
            video_files = []  # Reset for next directory
        
        if video_files:
            self.downloaded_videos = [
                {'video_path': str(f), 'keyword': f.parent.name if f.parent != videos_dir else 'existing', 'source': 'local'}
                for f in video_files
            ]
            print(f"  ✓ Loaded {len(self.downloaded_videos)} existing videos from {videos_dir}")
            
            # Show breakdown by subfolder
            from collections import Counter
            folders = Counter(Path(v['video_path']).parent.name for v in self.downloaded_videos)
            if len(folders) > 1:
                print(f"    Sources: {dict(folders)}")
        else:
            print(f"  ⚠ No video files found (.mp4, .mkv, .webm, .avi, .mov)")
            print(f"    Searched locations:")
            for i, d in enumerate(possible_dirs[:5], 1):
                if d.exists():
                    try:
                        files = [f for f in d.iterdir() if f.is_file()]
                        dirs = [f for f in d.iterdir() if f.is_dir()]
                        # Check if subdirs have videos
                        subdir_videos = 0
                        for sd in dirs[:3]:
                            subdir_videos += len([f for f in sd.iterdir() if f.is_file() and f.suffix.lower() in video_extensions])
                        
                        status = f"{len(files)} files, {len(dirs)} folders"
                        if subdir_videos > 0:
                            status += f" ({subdir_videos} videos in subfolders)"
                        print(f"      {i}. [✓] {d}")
                        print(f"          {status}")
                    except Exception as e:
                        print(f"      {i}. [✓] {d} (error: {e})")
                else:
                    print(f"      {i}. [✗] {d}")
            print(f"    TIP: Set pipeline.video_source_dir in config.yaml to an absolute path")
            print(f"    Example: video_source_dir: \"E:/Edit Job/project/downloaded_videos\"")
            self.downloaded_videos = []
    
    def _stage_zero_download_remix(self, failed_keywords: List[str], output_dir: Path):
        """
        Sub-stage: Remix failed keywords and retry downloads.
        
        Args:
            failed_keywords: Keywords that had 0 downloads
            output_dir: Download output directory
        """
        config = self.config
        max_retries = getattr(config.zero_download_remix, 'max_retries', 2)
        
        print(f"\n  ─── ZERO-DOWNLOAD KEYWORD REMIX ───")
        print(f"  Remixing {len(failed_keywords)} failed keywords...")
        
        try:
            from src.keyword_remix import KeywordRemixer, remix_zero_download_keywords
            
            # Get topic context if available
            topic_context = getattr(self, 'topic_context', 'documentary video content')
            
            # Initialize remixer
            remixer = KeywordRemixer(
                config=config,
                topic_context=topic_context,
                cache_dir=str(config.cache.cache_dir) if hasattr(config, 'cache') else ".cache"
            )
            
            # Track retries
            still_failed = failed_keywords.copy()
            total_recovered = 0
            
            for attempt in range(1, max_retries + 1):
                if not still_failed:
                    break
                
                print(f"\n  Attempt {attempt}/{max_retries}: Remixing {len(still_failed)} keywords...")
                
                # Remix keywords
                batch_result = remixer.remix_keywords_batch(still_failed, attempt=attempt)
                
                # Collect remixed keywords
                remixed_keywords = []
                for r in batch_result.results:
                    if r.success and r.remixed_keywords:
                        print(f"    '{r.original_keyword}' → {r.remixed_keywords[:2]}")
                        remixed_keywords.extend(r.remixed_keywords)
                
                if not remixed_keywords:
                    print(f"  ⚠ No alternatives generated")
                    break
                
                # Remove duplicates and already-tried keywords
                remixed_keywords = list(set(remixed_keywords) - set(failed_keywords))
                
                if not remixed_keywords:
                    print(f"  ⚠ All alternatives already tried")
                    break
                
                print(f"  Downloading {len(remixed_keywords)} remixed keywords...")
                
                # Try downloading remixed keywords
                new_videos, new_failed = self.downloader.download_all(
                    keywords=remixed_keywords,
                    output_dir=output_dir,
                    resume=True,
                    topic=self.topic_context or ""
                )
                
                if new_videos:
                    self.downloaded_videos.extend(new_videos)
                    total_recovered += len(new_videos)
                    print(f"  ✓ Recovered {len(new_videos)} videos from remixed keywords")
                
                # Update still_failed (only keep originals that still have no coverage)
                # A keyword is "covered" if any of its remixes succeeded
                successful_originals = set()
                for r in batch_result.results:
                    if r.success:
                        # Check if any remixed keyword succeeded
                        for remixed in r.remixed_keywords:
                            if remixed not in new_failed:
                                successful_originals.add(r.original_keyword)
                                break
                
                still_failed = [k for k in still_failed if k not in successful_originals]
            
            # Final summary
            if total_recovered > 0:
                print(f"\n  ✓ Zero-download remix recovered {total_recovered} videos")
            else:
                print(f"\n  ⚠ Zero-download remix: No additional videos found")
            
            if still_failed:
                print(f"  ⚠ Still failed ({len(still_failed)}): {', '.join(still_failed[:3])}" +
                      (f" (+{len(still_failed)-3} more)" if len(still_failed) > 3 else ""))
        
        except ImportError as e:
            logger.warning(f"Could not import keyword_remix: {e}")
            print(f"  ⚠ Keyword remix module not available")
        except Exception as e:
            logger.error(f"Zero-download remix failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  ⚠ Remix failed: {e}")
    
    def stage_remix(self, keywords: List[str]) -> List[str]:
        """
        Stage 2.5: Zero-download keyword remix.
        Filters/remixes downloaded videos based on keyword relevance
        before transcription, without additional downloads.
        """
        config = self.config

        # Check if remix is enabled
        if not config.remix.enabled:
            logger.debug("Remix disabled in config")
            return []

        if not config.remix.trigger_after_download:
            logger.debug("Remix not set to trigger after download")
            return []

        self._print_stage("2.5", "KEYWORD REMIX (Zero-Download)")

        try:
            from src.keyword_remix import remix_downloaded_videos, remix_audio_files, RemixConfig, save_remix_report

            # Create RemixConfig from config
            remix_config = RemixConfig(
                enabled=config.remix.enabled,
                trigger_after_download=config.remix.trigger_after_download,
                min_relevance_score=config.remix.min_relevance_score,
                high_relevance_threshold=config.remix.high_relevance_threshold,
                max_files_to_process=config.remix.max_files_to_process,
                max_files_to_include=config.remix.max_files_to_include,
                fuzzy_match=config.remix.fuzzy_match,
                case_sensitive=config.remix.case_sensitive,
                interactive_curation=config.remix.interactive_curation,
                show_excluded=config.remix.show_excluded,
                auto_accept_filter=config.remix.auto_accept_filter,  # Pass upfront choice
                log_file_processing=config.remix.log_file_processing,
                parallel_scoring=config.remix.parallel_scoring,
                max_workers=config.remix.max_workers
            )

            print(f"  Remix settings (from config):")
            print(f"    • Min relevance score: {remix_config.min_relevance_score}")
            print(f"    • Max files to include: {remix_config.max_files_to_include}")
            print(f"    • Fuzzy matching: {remix_config.fuzzy_match}")
            print(f"    • Interactive curation: {remix_config.interactive_curation}")

            # Check if audio-first mode - use audio files instead of video directory
            if self._is_audio_first_enabled() and hasattr(self, 'audio_downloads') and self.audio_downloads:
                audio_files = [ad.audio_file for ad in self.audio_downloads]
                print(f"  Audio-first mode: scoring {len(audio_files)} audio files")

                included_paths, remix_result = remix_audio_files(
                    audio_files=audio_files,
                    keywords=keywords,
                    config=remix_config,
                    interactive=remix_config.interactive_curation,
                    show_progress=True
                )

                # Update audio_downloads to only include filtered files
                if included_paths:
                    included_set = set(included_paths)
                    self.audio_downloads = [ad for ad in self.audio_downloads if ad.audio_file in included_set]
                    self.audio_downloads_by_id = {a.video_id: a for a in self.audio_downloads}
            else:
                # Standard mode - scan video directory
                video_dir = Path(config.downloaded_videos_dir)

                if not video_dir.exists():
                    print(f"  ⚠ Video directory not found: {video_dir}")
                    return []

                included_paths, remix_result = remix_downloaded_videos(
                    video_dir=video_dir,
                    keywords=keywords,
                    config=remix_config,
                    interactive=remix_config.interactive_curation,
                    show_progress=True
                )
            
            # Store result for later stages
            self.remix_result = remix_result
            self.remixed_video_paths = included_paths
            
            # Save remix report
            if remix_result:
                output_dir = Path(config.output.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                report_path = output_dir / "remix_report.json"
                save_remix_report(remix_result, report_path)
                
                logger.info(json.dumps({
                    'event': 'remix_stage_complete',
                    'included_videos': len(included_paths),
                    'total_scanned': remix_result.total_files,
                    'avg_score': round(remix_result.avg_match_score, 3),
                    'processing_time': round(remix_result.processing_time_seconds, 2)
                }))
            
            print(f"\n  ✓ Remix complete: {len(included_paths)} videos selected")
            
            return included_paths
            
        except ImportError as e:
            logger.warning(f"Could not import keyword_remix module: {e}")
            print("  ⚠ Remix module not available, skipping")
            return []
        except Exception as e:
            logger.error(f"Remix failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  ⚠ Remix failed: {e}")
            return []

    def _load_transcripts_from_cache(self) -> dict:
        """
        Load transcripts and embeddings from cache when skipping transcription.
        This allows the pipeline to continue with cached data.
        """
        config = self.config
        self._print_stage("3", "LOAD CACHED TRANSCRIPTS")

        try:
            from src.transcription import TranscriptCache
            from src.embeddings import compute_embeddings, build_embedding_index, get_embedding_provider
            from src.utils import CacheManager

            # Get video files
            if hasattr(self, 'remixed_video_paths') and self.remixed_video_paths:
                video_files = [Path(p) for p in self.remixed_video_paths]
            else:
                videos_dir = Path(config.downloaded_videos_dir)
                video_files = list(videos_dir.rglob('*.mp4')) + list(videos_dir.rglob('*.webm'))

            if not video_files:
                print("  ⚠ No video files found")
                return {}

            print(f"  Found {len(video_files)} videos")

            # Load transcripts from cache
            cache = TranscriptCache(config.cache.cache_dir)
            self.transcripts = {}
            loaded = 0

            for vf in video_files:
                cached = cache.get(str(vf))
                if cached:
                    self.transcripts[str(vf)] = cached
                    loaded += 1

            print(f"  ✓ Loaded {loaded}/{len(video_files)} transcripts from cache")

            if not self.transcripts:
                print("  ⚠ No cached transcripts found - run without skip_transcription first")
                return {}

            # Build lookup of video info (source, face_score) by path
            video_info_lookup = {}
            if hasattr(self, 'downloaded_videos'):
                for v in self.downloaded_videos:
                    path = v.get('path') or v.get('file', '')
                    video_info_lookup[path] = {
                        'source': v.get('source'),
                        'face_score': v.get('face_score')
                    }

            # Build text metadata for matching
            self.text_metadata = []
            for video_path, segments in self.transcripts.items():
                # Get source and face_score for this video
                vinfo = video_info_lookup.get(video_path, {})
                source = vinfo.get('source')
                face_score = vinfo.get('face_score')

                for seg in segments:
                    if hasattr(seg, 'text'):
                        meta = {
                            'text': seg.text,
                            'video_path': video_path,
                            'start_time': seg.start_time,
                            'end_time': seg.end_time
                        }
                    else:
                        meta = {
                            'text': seg.get('text', ''),
                            'video_path': video_path,
                            'start_time': seg.get('start_time', 0),
                            'end_time': seg.get('end_time', 0)
                        }
                    # Add source and face_score if available (for global cache videos)
                    if source:
                        meta['source'] = source
                    if face_score is not None:
                        meta['face_score'] = face_score
                    self.text_metadata.append(meta)

            print(f"  ✓ Built {len(self.text_metadata)} text segments")

            # Load embeddings from cache (compute_embeddings checks batch cache first)
            provider = get_embedding_provider(config)
            cache_mgr = CacheManager(config.cache.cache_dir)
            text_strings = [t['text'] for t in self.text_metadata]

            # compute_embeddings uses EmbeddingCache.get_batch_cache() internally
            # This will return cached embeddings if they exist (very fast)
            # Only recomputes if the exact same text set isn't already cached
            self.embeddings = compute_embeddings(
                texts=text_strings,
                provider=provider,
                cache=cache_mgr,
                cache_key="video_segments",
                config=config
            )
            print(f"  ✓ Loaded/computed {len(self.embeddings)} embeddings")

            # Build index
            if self.embeddings is not None and len(self.embeddings) > 0:
                self.embedding_index = build_embedding_index(self.embeddings, config=config)
                print(f"  ✓ Built embedding index ({len(self.embeddings)} vectors)")

            return self.transcripts

        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def stage_transcribe(self) -> dict:
        """
        Stage 3: Transcribe and index videos.
        Parallel processing settings from config.
        Uses remixed video paths if available from stage_remix.
        """
        config = self.config
        
        if config.pipeline.skip_transcription:
            print("  ⏭ Skipping transcription - loading from cache...")
            logger.info("Skipping TRANSCRIBE stage (config: skip_transcription=true)")
            return self._load_transcripts_from_cache()
        
        self._print_stage("3", "TRANSCRIBE & INDEX")
        
        # Check if audio-first mode - use audio files
        if self._is_audio_first_enabled() and hasattr(self, 'audio_downloads') and self.audio_downloads:
            # Use audio files from audio-first download
            video_files = [Path(ad.audio_file) for ad in self.audio_downloads]
            print(f"  Using {len(video_files)} audio files from audio-first mode")
        # Check if we have remixed video paths from stage_remix
        elif hasattr(self, 'remixed_video_paths') and self.remixed_video_paths:
            # Use filtered videos from remix stage
            video_files = [Path(p) for p in self.remixed_video_paths]
            print(f"  Using {len(video_files)} videos from remix stage")
        else:
            # Fall back to scanning directory
            videos_dir = Path(config.downloaded_videos_dir)
            video_files = list(videos_dir.rglob('*.mp4')) + list(videos_dir.rglob('*.webm')) + list(videos_dir.rglob('*.mp3'))
        
        if not video_files:
            print("  ⚠ No video files found")
            return {}
        
        print(f"  Found {len(video_files)} videos to process")
        
        # Transcription (config-driven parallelism)
        if self.modules['optimized_transcription'] and config.pipeline.parallel_transcription:
            from src.transcription import transcribe_videos_parallel, DeltaAwareIndex
            
            print(f"  Using parallel transcription ({config.transcription.max_workers} workers)")
            
            # Delta-aware indexing
            if self.use_delta_indexing:
                delta_index = DeltaAwareIndex(config.cache.cache_dir)
                video_files_str = [str(v) for v in video_files]
                video_files_str = delta_index.get_new_videos(video_files_str)
                video_files = [Path(v) for v in video_files_str]
                print(f"  Delta indexing: {len(video_files)} new videos")
            
            if not video_files:
                print("  ✓ All videos already transcribed (cached)")
                self.transcripts = {}
            else:
                self.transcripts = transcribe_videos_parallel(
                    video_paths=[str(v) for v in video_files],
                    cache=config.cache.cache_dir,  # Pass cache directory path
                    config=config,
                    max_workers=config.transcription.max_workers,
                    show_progress=True
                )
        else:
            # Sequential fallback
            print("  Using sequential transcription")
            from src.transcription import transcribe_video, TranscriptCache
            
            # Create transcript cache
            cache = TranscriptCache(config.cache.cache_dir)
            
            self.transcripts = {}
            for vf in video_files:
                try:
                    segments = transcribe_video(
                        str(vf), 
                        cache=cache,
                        model_name=config.transcription.model,
                        language=config.transcription.language if config.transcription.language != 'auto' else None
                    )
                    self.transcripts[str(vf)] = segments
                except Exception as e:
                    logger.warning(f"Failed to transcribe {vf}: {e}")
        
        print(f"  ✓ Transcribed {len(self.transcripts)} videos")

        # Handle silent videos (B-roll) - generate descriptions for matching
        silent_video_config = getattr(config, 'silent_video', None)
        if silent_video_config and getattr(silent_video_config, 'enabled', True):
            self._handle_silent_videos(video_files)

        # Embeddings (config-driven batching)
        if self.modules['optimized_embeddings'] and config.pipeline.parallel_embedding:
            from src.embeddings import compute_embeddings, build_embedding_index, get_embedding_provider
            from src.utils import CacheManager
            
            print(f"  Computing embeddings (batch size: {config.embedding.batch_size})")
            
            # Collect texts - handle both dict and TranscriptSegment objects
            texts = []
            for video_path, segments in self.transcripts.items():
                for seg in segments:
                    # Handle both dict and dataclass objects
                    if hasattr(seg, 'text'):
                        # TranscriptSegment object
                        texts.append({
                            'text': seg.text,
                            'video_path': video_path,
                            'start_time': seg.start_time,
                            'end_time': seg.end_time
                        })
                    else:
                        # Dict
                        texts.append({
                            'text': seg.get('text', ''),
                            'video_path': video_path,
                            'start_time': seg.get('start_time', 0),
                            'end_time': seg.get('end_time', 0)
                        })
            
            self.text_metadata = texts
            text_strings = [t['text'] for t in texts]
            
            if not text_strings:
                print("  ⚠ No text segments to embed")
                self.embeddings = []
                return self.transcripts
            
            # Get embedding provider and cache
            provider = get_embedding_provider(config)
            cache = CacheManager(config.cache.cache_dir)
            
            # Compute embeddings
            self.embeddings = compute_embeddings(
                texts=text_strings,
                provider=provider,
                cache=cache,
                cache_key="video_segments"
            )
            
            # Build index
            if self.embeddings is not None and len(self.embeddings) > 0:
                self.embedding_index = build_embedding_index(self.embeddings, config=config)
                print(f"  ✓ Built embedding index ({len(self.embeddings)} vectors)")

        # Pre-detect faces if face preference is set (caches results for matching stage)
        # Skip for audio-first mode - face detection runs after video segments are downloaded
        face_pref = getattr(self, 'face_preference', 'neutral')
        if face_pref != 'neutral' and not self._is_audio_first_enabled():
            self._predetect_faces(video_files)
        elif face_pref != 'neutral' and self._is_audio_first_enabled():
            print(f"  ⏭ Face detection deferred (audio-first mode - will run after video download)")

        # Extract topics from video transcripts (for chapter-based matching)
        if config.matching.chapter_matching_enabled and config.matching.extract_video_topics:
            self._extract_video_topics()

        # Share transcripts to global cache (for cross-project reuse)
        self._share_to_global_cache(video_files)

        return self.transcripts

    def _share_to_global_cache(self, video_files: List[Path]):
        """
        Share processed video data to global cache for cross-project reuse.

        Registers videos and copies transcripts, topics, and face scores
        to the global cache so future projects can reuse this data.
        """
        if not self.global_cache:
            return

        gc_config = getattr(self.config, 'global_cache', None)
        if not gc_config:
            return

        share_transcripts = getattr(gc_config, 'share_transcripts', True)
        share_face = getattr(gc_config, 'share_face_detection', True)

        if not share_transcripts and not share_face:
            return

        print(f"\n  📤 Sharing to global cache...")
        registered = 0

        for vf in video_files:
            video_path = str(vf)

            # Skip videos from global cache (already registered)
            video_info = None
            for dv in getattr(self, 'downloaded_videos', []):
                if isinstance(dv, dict):
                    if dv.get('path') == video_path or dv.get('file') == video_path:
                        video_info = dv
                        break

            if video_info and video_info.get('source') == 'global_cache':
                continue

            try:
                # Get transcript for this video
                segments = self.transcripts.get(video_path, [])
                transcript_text = ' '.join([
                    s.text if hasattr(s, 'text') else s.get('text', '')
                    for s in segments[:5]  # First 5 segments as preview
                ])

                # Get topics if extracted
                topics = []
                if hasattr(self, 'video_topics') and video_path in self.video_topics:
                    vt = self.video_topics[video_path]
                    topics = vt.topics if hasattr(vt, 'topics') else []

                # Extract keyword from path
                keyword = self._extract_source_keyword(video_path)

                # Get face score if available
                face_score = 0.5
                if share_face:
                    try:
                        from src.face_detection import FaceDetector
                        detector = FaceDetector.get_instance()
                        if detector.is_available() and video_path in detector._cache:
                            face_score = detector._cache[video_path]
                    except:
                        pass

                # Get video duration
                duration = 0.0
                if segments:
                    last_seg = segments[-1]
                    duration = last_seg.end_time if hasattr(last_seg, 'end_time') else last_seg.get('end_time', 0)

                # Register in global cache
                entry = self.global_cache.register_video(
                    video_path=video_path,
                    download_keyword=keyword,
                    topics=topics,
                    project_id=str(PROJECT_DIR) if PROJECT_DIR else "",
                    duration=duration
                )

                if entry:
                    # Mark as processed
                    self.global_cache.mark_video_processed(
                        video_hash=entry.video_hash,
                        has_transcript=bool(segments),
                        has_embeddings=hasattr(self, 'embeddings') and self.embeddings is not None,
                        face_score=face_score
                    )

                    # Copy transcript to global cache
                    if share_transcripts and segments:
                        transcript_data = {
                            'segments': [
                                s.to_dict() if hasattr(s, 'to_dict') else s
                                for s in segments
                            ],
                            'video_path': video_path,
                            'topics': topics
                        }
                        self.global_cache.copy_transcript_to_global(
                            entry.video_hash, transcript_data
                        )

                    registered += 1

            except Exception as e:
                logger.debug(f"Failed to register {vf.name} in global cache: {e}")

        if registered > 0:
            print(f"  ✓ Registered {registered} videos in global cache")
            logger.info(f"Global cache: registered {registered} videos")

    def _predetect_faces(self, video_files: List[Path]):
        """Pre-detect faces in videos and cache results for matching stage."""
        try:
            from src.matching import FaceDetector
            
            detector = FaceDetector.get_instance()
            cache_dir = self.config.cache.cache_dir
            
            # Check how many need detection
            uncached = []
            for vf in video_files:
                video_path = str(vf)
                if video_path not in FaceDetector._cache:
                    uncached.append(vf)
            
            if not uncached:
                print(f"  ✓ Face detection: all {len(video_files)} videos cached")
                return
            
            print(f"  Detecting faces in {len(uncached)} videos...")

            faces_found = 0
            for i, vf in enumerate(uncached):
                video_path = str(vf)
                score = detector.get_face_score(video_path, cache_dir)
                if score > 0.2:  # At least 1 frame with faces
                    faces_found += 1

                # Save face score to global cache registry if available
                if hasattr(self, 'global_cache') and self.global_cache:
                    try:
                        video_hash = self.global_cache.get_video_hash(video_path)
                        if video_hash:
                            self.global_cache.mark_video_processed(
                                video_hash, face_score=score
                            )
                    except Exception:
                        pass  # Don't fail if global cache update fails

                # Progress every 10 videos
                if (i + 1) % 10 == 0:
                    print(f"    Processed {i + 1}/{len(uncached)}...", end='\r')

            print(f"  ✓ Face detection: {faces_found}/{len(uncached)} videos have faces")

        except Exception as e:
            logger.warning(f"Face pre-detection failed: {e}")

    def _map_segment_face_scores_to_audio(self):
        """
        Map face scores from video segments to original audio file paths.

        In audio-first mode, face detection runs on video segments (e.g., video_id_0035.mp4)
        but matching uses audio file paths (e.g., video_id.mp3). This method copies the
        face scores so they can be looked up by audio file path during second-pass matching.
        """
        if not hasattr(self, 'downloaded_segments') or not self.downloaded_segments:
            return

        if not hasattr(self, 'audio_downloads_by_id') or not self.audio_downloads_by_id:
            return

        try:
            from src.face_detection import FaceDetector

            detector = FaceDetector.get_instance()
            cache_dir = self.config.cache.cache_dir
            mapped_count = 0

            # For each downloaded segment, find the corresponding audio file
            for seg in self.downloaded_segments:
                video_id = seg.video_id
                segment_path = seg.file

                # Get face score for the video segment
                if segment_path in FaceDetector._cache:
                    face_score = FaceDetector._cache[segment_path]

                    # Find corresponding audio file
                    if video_id in self.audio_downloads_by_id:
                        audio_download = self.audio_downloads_by_id[video_id]
                        audio_path = audio_download.audio_file

                        # Map the face score to audio file path
                        if audio_path not in FaceDetector._cache:
                            FaceDetector._cache[audio_path] = face_score
                            mapped_count += 1

            if mapped_count > 0:
                logger.debug(f"Mapped {mapped_count} face scores from segments to audio files")

        except Exception as e:
            logger.warning(f"Failed to map segment face scores: {e}")

    def _extract_video_topics(self):
        """Extract topics from video transcripts for chapter-based matching."""
        if not self.transcripts:
            return

        try:
            from src.topic_extraction import TopicExtractor

            extractor = TopicExtractor(self.config, self.config.cache.cache_dir)

            # Prepare transcripts and metadata for batch extraction
            transcripts_dict = {}
            metadata_dict = {}

            for video_path, segments in self.transcripts.items():
                # Combine segment texts into full transcript
                if segments:
                    transcript_texts = []
                    for seg in segments:
                        if hasattr(seg, 'text'):
                            transcript_texts.append(seg.text)
                        elif isinstance(seg, dict):
                            transcript_texts.append(seg.get('text', ''))
                    full_transcript = ' '.join(transcript_texts)
                    transcripts_dict[video_path] = full_transcript

                    # Extract source keyword from video path
                    video_name = Path(video_path).stem
                    # Try to extract keyword from folder structure or filename
                    keyword = self._extract_source_keyword(video_path)
                    metadata_dict[video_path] = {
                        'title': video_name,
                        'keyword': keyword
                    }

            # Batch extract topics
            print(f"\n  Extracting video topics ({len(transcripts_dict)} videos)...")
            logger.info(f"Extracting topics from {len(transcripts_dict)} videos")
            self.video_topics = extractor.extract_batch(transcripts_dict, metadata_dict)

            # Count topics extracted
            topics_count = sum(1 for vt in self.video_topics.values() if vt.topics)
            print(f"  ✓ Extracted topics for {topics_count}/{len(self.video_topics)} videos")

            # Show sample topics
            sample_videos = list(self.video_topics.items())[:3]
            for video_path, vt in sample_videos:
                if vt.topics:
                    video_name = Path(video_path).name[:30]
                    topics_str = ', '.join(vt.topics[:3])
                    print(f"    • {video_name}: [{topics_str}]")

        except Exception as e:
            logger.warning(f"Video topic extraction failed: {e}")
            self.video_topics = {}

    def _handle_silent_videos(self, video_files: List[Path]):
        """
        Handle videos with no speech (B-roll) by generating descriptions.

        Silent videos are valuable B-roll footage. This method:
        1. Detects videos with no/sparse transcripts
        2. Generates descriptions using Vision API or LLM from title/keyword
        3. Creates synthetic segments for embedding and matching
        4. Caches descriptions to avoid repeated API calls

        B-roll videos get a confidence boost during matching since they're
        versatile visual content without talking heads.
        """
        config = self.config
        silent_config = getattr(config, 'silent_video', None)

        # Default settings if no config
        min_words = getattr(silent_config, 'min_words_threshold', 10) if silent_config else 10
        use_vision = getattr(silent_config, 'use_vision_api', True) if silent_config else True
        use_llm = getattr(silent_config, 'use_llm_fallback', True) if silent_config else True

        # B-roll description cache
        import hashlib
        import json
        import time
        broll_cache_file = Path(config.cache.cache_dir) / "broll_descriptions.json"
        broll_cache = {}
        if broll_cache_file.exists():
            try:
                with open(broll_cache_file, 'r', encoding='utf-8') as f:
                    broll_cache = json.load(f)
                logger.info(f"Loaded B-roll cache: {len(broll_cache)} entries from {broll_cache_file}")
            except Exception as e:
                logger.debug(f"Could not load B-roll cache: {e}")
        else:
            logger.info(f"No B-roll cache found at {broll_cache_file}")

        def get_cache_key(video_path: str) -> str:
            """Generate cache key from video path (normalized for consistency)"""
            # Normalize path: forward slashes, lowercase for consistent matching
            normalized = str(video_path).replace('\\', '/').lower()
            return hashlib.md5(normalized.encode()).hexdigest()[:16]

        def save_broll_cache():
            """Save B-roll cache to disk"""
            try:
                with open(broll_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(broll_cache, f, indent=2)
            except Exception as e:
                logger.debug(f"Could not save B-roll cache: {e}")

        # Find silent videos
        silent_videos = []
        for vf in video_files:
            video_path = str(vf)
            segments = self.transcripts.get(video_path, [])

            # Count words in transcript
            total_words = 0
            for seg in segments:
                text = seg.text if hasattr(seg, 'text') else seg.get('text', '')
                total_words += len(text.split())

            if total_words < min_words:
                silent_videos.append(video_path)

        if not silent_videos:
            return

        print(f"\n  📹 Found {len(silent_videos)} silent/B-roll videos")
        logger.info(f"Silent video handling: Found {len(silent_videos)} videos with <{min_words} words")
        for sv in silent_videos:
            logger.debug(f"  Silent video: {Path(sv).name}")

        # Track which videos we successfully described
        described = 0
        cached_count = 0
        total_silent = len(silent_videos)

        for idx, video_path in enumerate(silent_videos, 1):
            description = None
            source = None
            duration = 30.0

            # Check cache first
            cache_key = get_cache_key(video_path)
            from_cache = False
            if cache_key in broll_cache:
                cached = broll_cache[cache_key]
                description = cached.get('description')
                source = cached.get('source', 'cached')
                duration = cached.get('duration', 30.0)
                cached_count += 1
                from_cache = True
                logger.debug(f"({idx}/{total_silent}) Using cached B-roll description for {Path(video_path).name}")

            # Try Vision API first (if enabled and available)
            if use_vision and description is None:
                try:
                    from src.vision import VisionProcessor

                    processor = VisionProcessor(config)
                    if processor.is_available():
                        # Get video duration for sampling
                        import subprocess
                        result = subprocess.run(
                            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                            capture_output=True, text=True, timeout=10
                        )
                        duration = float(result.stdout.strip()) if result.stdout.strip() else 30.0

                        # Sample 3 frames and describe
                        descriptions = []
                        sample_times = [duration * 0.25, duration * 0.5, duration * 0.75]
                        for t in sample_times:
                            scene = {'start_time': max(0, t - 1), 'end_time': t + 1}
                            desc = processor.describe_scene(
                                video_path, scene,
                                cache_dir=config.cache.cache_dir
                            )
                            if desc:
                                descriptions.append(desc)

                        if descriptions:
                            description = ' '.join(descriptions)
                            source = 'vision'
                            logger.debug(f"({idx}/{total_silent}) Vision API described {Path(video_path).name}: {description[:100]}...")
                except Exception as e:
                    logger.debug(f"({idx}/{total_silent}) Vision API failed for {Path(video_path).name}: {e}")

            # Fallback to LLM from title/keyword
            if use_llm and description is None:
                try:
                    keyword = self._extract_source_keyword(video_path)
                    video_name = Path(video_path).stem

                    # Clean up video name for description
                    clean_name = video_name.replace('_', ' ').replace('-', ' ')
                    # Remove video IDs (11-char YouTube IDs)
                    import re
                    clean_name = re.sub(r'\b[a-zA-Z0-9_-]{11}\b', '', clean_name).strip()

                    # Generate description from keyword + filename
                    if self.config.gemini_api_key:
                        import google.generativeai as genai
                        genai.configure(api_key=self.config.gemini_api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')

                        prompt = f"""Generate a short visual description (2-3 sentences) for a stock video based on:
- Search keyword: {keyword}
- Filename: {clean_name}

Describe what visual content this video likely contains. Focus on subjects, actions, and setting.
Be specific and descriptive for semantic matching purposes."""

                        response = model.generate_content(prompt)
                        description = response.text.strip()
                        source = 'llm'
                        logger.debug(f"({idx}/{total_silent}) LLM described {Path(video_path).name}: {description[:100]}...")
                    else:
                        # Simple fallback - use keyword as description
                        description = f"Video footage of {keyword}. Visual content showing {clean_name}."
                        source = 'keyword'
                        logger.debug(f"({idx}/{total_silent}) Keyword fallback for {Path(video_path).name}: {description}")

                except Exception as e:
                    logger.debug(f"({idx}/{total_silent}) LLM fallback failed for {Path(video_path).name}: {e}")

            # Create synthetic segments if we have a description
            if description:
                # Get video duration (skip if already from cache)
                if cache_key not in broll_cache:
                    try:
                        import subprocess
                        result = subprocess.run(
                            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                            capture_output=True, text=True, timeout=10
                        )
                        duration = float(result.stdout.strip()) if result.stdout.strip() else 30.0
                    except:
                        duration = 30.0

                    # Save to cache
                    broll_cache[cache_key] = {
                        'description': description,
                        'source': source,
                        'duration': duration,
                        'video_name': Path(video_path).name,
                        'timestamp': time.time()
                    }

                # Create synthetic transcript segment
                from src.transcription import TranscriptSegment

                synthetic_segment = TranscriptSegment(
                    index=0,
                    text=description,
                    start_time=0.0,
                    end_time=duration,
                    source_file=video_path
                )

                # Mark as B-roll for potential boost
                synthetic_segment.is_broll = True
                synthetic_segment.description_source = source

                # Add/replace in transcripts
                self.transcripts[video_path] = [synthetic_segment]
                described += 1

                # Only print for newly generated descriptions (not cached)
                if not from_cache:
                    print(f"    ({idx}/{total_silent}) ✓ {Path(video_path).name}: {source} ({len(description)} chars)")
                    logger.info(f"({idx}/{total_silent}) B-roll description: {Path(video_path).name} via {source} ({len(description)} chars)")

        # Save cache after processing
        save_broll_cache()

        if described > 0:
            new_count = described - cached_count
            if cached_count > 0:
                print(f"  ✓ B-roll descriptions: {cached_count} cached, {new_count} new ({described} total)")
            else:
                print(f"  ✓ Generated descriptions for {described}/{len(silent_videos)} silent videos")
            logger.info(f"Silent video summary: {described}/{len(silent_videos)} described ({cached_count} cached, {new_count} new)")

    def _extract_source_keyword(self, video_path: str) -> str:
        """Try to extract the source search keyword from video path structure."""
        path = Path(video_path)
        parts = path.parts

        # Look for keyword folder patterns like 'downloads/keyword_name/video.mp4'
        for i, part in enumerate(parts):
            if part in ('downloads', 'downloaded_videos', 'videos'):
                if i + 1 < len(parts) - 1:  # There's a subfolder after downloads
                    keyword_folder = parts[i + 1]
                    # Clean up folder name
                    return keyword_folder.replace('_', ' ').replace('-', ' ')

        # Fallback: use filename without extension
        return path.stem.replace('_', ' ').replace('-', ' ')

    def stage_match(self) -> List[dict]:
        """
        Stage 4: Match voiceover to footage.
        All matching parameters from config.

        Supports delta matching: only processes new videos if enabled.
        """
        config = self.config

        if config.pipeline.skip_matching:
            print("  ⏭ Skipping matching (config: skip_matching=true)")
            logger.info("Skipping MATCH stage (config: skip_matching=true)")
            return []

        self._print_stage("4", "MATCH FOOTAGE")

        # Check if we have data to match
        if not self.voiceover_segments:
            print("  ⚠ No voiceover segments to match")
            return []

        if not self.text_metadata or self.embeddings is None or len(self.embeddings) == 0:
            print("  ⚠ No video data to match against")
            return []

        try:
            from src.matching import match_all_segments
            from src.utils import SRTSegment, CacheManager, MatchResult, Match
            from src.embeddings import compute_embeddings, get_embedding_provider

            # Get project directory for match index
            project_dir = PROJECT_DIR or Path(getattr(self, 'voiceover_path', '.')).parent

            # Initialize match index for delta matching
            match_index = MatchAwareIndex(str(project_dir))

            # Check delta matching settings
            delta_enabled = getattr(config.matching, 'delta_matching_enabled', True)
            force_rematch = getattr(self, 'force_rematch', False) or getattr(config.matching, 'force_rematch', False)

            print(f"  Matching settings (from config):")
            print(f"    • Min confidence: {config.matching.min_confidence}")
            print(f"    • High confidence threshold: {config.matching.high_confidence_threshold}")
            print(f"    • Embedding candidates: {config.matching.embedding_candidates}")
            print(f"    • LLM rerank candidates: {config.matching.llm_rerank_candidates}")
            print(f"    • Max clip reuse: {config.matching.max_clip_reuse}")
            print(f"    • Delta matching: {'enabled' if delta_enabled and not force_rematch else 'disabled'}")

            # Convert voiceover dict segments to SRTSegment objects
            print(f"\n  Preparing voiceover segments...")
            vo_segments = []
            for i, seg in enumerate(self.voiceover_segments):
                if isinstance(seg, dict):
                    vo_segment = SRTSegment(
                        index=seg.get('index', i),
                        start_time=seg.get('start_time', seg.get('start', 0)),
                        end_time=seg.get('end_time', seg.get('end', 0)),
                        text=seg.get('text', ''),
                        source_file=seg.get('source_file', ''),
                        keywords=seg.get('keywords', []),
                        entities=seg.get('entities', []),
                        topics=seg.get('chapter_topics', [])  # Chapter topics for topic-based matching
                    )
                else:
                    vo_segment = seg
                    # Copy chapter topics if available
                    if hasattr(seg, 'chapter_topics') and not getattr(seg, 'topics', None):
                        vo_segment.topics = seg.chapter_topics
                vo_segments.append(vo_segment)

            # Convert video metadata to SRTSegment objects
            print(f"  Preparing video segments...")
            video_segments = []
            video_paths_set = set()
            for i, meta in enumerate(self.text_metadata):
                if isinstance(meta, dict):
                    vid_segment = SRTSegment(
                        index=i,
                        start_time=meta.get('start_time', 0),
                        end_time=meta.get('end_time', 0),
                        text=meta.get('text', ''),
                        source_file=meta.get('video_path', ''),
                    )
                    # Add source and face_score for global cache videos
                    if meta.get('source'):
                        vid_segment.source = meta['source']
                    if meta.get('face_score') is not None:
                        vid_segment.face_score = meta['face_score']
                    video_paths_set.add(meta.get('video_path', ''))
                else:
                    vid_segment = meta
                    video_paths_set.add(getattr(meta, 'source_file', ''))
                video_segments.append(vid_segment)

            all_video_paths = list(video_paths_set)

            # Get embedding provider and cache
            cache_dir = config.cache.cache_dir if hasattr(config.cache, 'cache_dir') else ".cache"
            provider = get_embedding_provider(config)
            cache = CacheManager(cache_dir)

            # ================================================================
            # DELTA MATCHING LOGIC
            # ================================================================
            use_cached_matches = False
            cached_matches = None

            if delta_enabled and not force_rematch:
                # Check for voiceover changes
                voiceover_path = getattr(self, 'voiceover_path', None)
                if voiceover_path and match_index.is_voiceover_changed(voiceover_path):
                    print(f"  ⚠ Voiceover changed - will rematch all videos")
                    match_index.clear()
                else:
                    # Check for new/modified/deleted videos
                    new_videos = match_index.get_new_videos(all_video_paths)
                    modified_videos = match_index.get_modified_videos(all_video_paths)
                    deleted_videos = match_index.get_deleted_videos(all_video_paths)

                    if deleted_videos:
                        print(f"  ⚠ {len(deleted_videos)} videos deleted - removing from index")
                        match_index.remove_videos(deleted_videos)

                    if modified_videos:
                        print(f"  ⚠ {len(modified_videos)} videos modified - will rematch")
                        match_index.remove_videos(modified_videos)
                        new_videos = list(set(new_videos) | set(modified_videos))

                    if not new_videos and not deleted_videos:
                        # No changes - try to use cached matches
                        cached_matches = match_index.get_cached_matches()
                        if cached_matches:
                            print(f"  ✓ No new videos - using {len(cached_matches)} cached matches")
                            use_cached_matches = True
                    elif new_videos:
                        print(f"  → {len(new_videos)} new videos to match (out of {len(all_video_paths)} total)")

            if force_rematch:
                print(f"  ⚠ Force rematch enabled - clearing match cache")
                match_index.clear()

            # ================================================================
            # MATCHING
            # ================================================================

            if use_cached_matches and cached_matches:
                # Reconstruct match results from cache
                self.matches = []
                for m_dict in cached_matches:
                    try:
                        # Reconstruct MatchResult from dict
                        primary_data = m_dict.get('primary_match', {})
                        primary_match = Match(
                            voiceover_segment=SRTSegment.from_dict(primary_data.get('voiceover_segment', {})),
                            video_segment=SRTSegment.from_dict(primary_data.get('video_segment', {})),
                            video_scene=None,
                            confidence=primary_data.get('confidence', 0.0),
                            reasoning=primary_data.get('reasoning', ''),
                            is_keyword_match=primary_data.get('is_keyword_match', False),
                            is_visual_match=primary_data.get('is_visual_match', False),
                            embedding_similarity=primary_data.get('embedding_similarity', 0.0),
                            clip_reuse_count=primary_data.get('clip_reuse_count', 0)
                        )
                        match_result = MatchResult(
                            primary_match=primary_match,
                            has_gap=m_dict.get('has_gap', False),
                            gap_reason=m_dict.get('gap_reason', '')
                        )
                        self.matches.append(match_result)
                    except Exception as e:
                        logger.warning(f"Could not reconstruct cached match: {e}")
            else:
                # Compute voiceover embeddings
                print(f"  Computing voiceover embeddings...")
                vo_texts = [seg.text for seg in vo_segments]

                vo_embeddings = compute_embeddings(
                    texts=vo_texts,
                    provider=provider,
                    cache=cache,
                    cache_key="voiceover"
                )

                if vo_embeddings is None or len(vo_embeddings) == 0:
                    print("  ⚠ Failed to compute voiceover embeddings")
                    return []

                # Match all segments using the high-level function
                print(f"  Running two-stage matching...")

                # Get video topics for chapter-based matching
                video_topics = getattr(self, 'video_topics', None)
                if video_topics and config.matching.chapter_matching_enabled:
                    print(f"  Using chapter-based matching with {len(video_topics)} video topics")

                # Get location chapters and video locations for location-aware matching
                location_chapters = getattr(self, 'location_chapters', None)
                video_locations = getattr(self, 'video_locations', None)

                # Extract video locations if we have location chapters
                if location_chapters and not video_locations:
                    self._extract_video_locations()
                    video_locations = getattr(self, 'video_locations', None)

                if location_chapters:
                    print(f"  Using location-aware matching with {len(location_chapters)} location chapters")

                self.matches = match_all_segments(
                    voiceover_segments=vo_segments,
                    video_segments=video_segments,
                    voiceover_embeddings=vo_embeddings,
                    video_embeddings=self.embeddings,
                    scenes=getattr(self, 'scenes', None),
                    config=config,
                    cache=cache,
                    embedding_index=self.embedding_index,
                    face_preference=getattr(self, 'face_preference', 'neutral'),
                    video_topics=video_topics,
                    location_chapters=location_chapters,
                    video_locations=video_locations
                )

                # Save matches to cache for delta matching
                if delta_enabled:
                    try:
                        matches_as_dicts = []
                        for m in self.matches:
                            if m and hasattr(m, 'primary_match') and m.primary_match:
                                matches_as_dicts.append({
                                    'primary_match': m.primary_match.to_dict(),
                                    'has_gap': getattr(m, 'has_gap', False),
                                    'gap_reason': getattr(m, 'gap_reason', '')
                                })

                        voiceover_path = getattr(self, 'voiceover_path', None)
                        match_index.save_matches(matches_as_dicts, voiceover_path)

                        # Update voiceover hash
                        if voiceover_path:
                            match_index.set_voiceover_hash(voiceover_path)

                        # Mark all videos as matched
                        match_index.mark_matched_batch(all_video_paths)

                        print(f"  ✓ Saved matches to cache for future delta matching")
                    except Exception as e:
                        logger.warning(f"Could not save match cache: {e}")

            # Calculate confidence stats
            confidences = [m.primary_match.confidence for m in self.matches if m and m.primary_match]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            print(f"\n  ✓ Matched {len(self.matches)} segments")
            print(f"  Average confidence: {avg_conf:.1%}")

            # Check if remix needed (config-driven threshold)
            if self.enhanced_enabled and avg_conf < config.enhanced.min_confidence:
                print(f"  ⚠ Below {config.enhanced.min_confidence:.0%} threshold - consider keyword remix")

        except ImportError as e:
            logger.error(f"Could not import matching: {e}")
            import traceback
            traceback.print_exc()
            return []
        except Exception as e:
            logger.error(f"Matching failed: {e}")
            import traceback
            traceback.print_exc()
            return []

        return self.matches
    
    def stage_output(self) -> dict:
        """
        Stage 5: Generate output files.
        Output formats from config.

        Each run creates timestamped output files to preserve history.
        """
        from datetime import datetime

        config = self.config

        self._print_stage("5", "GENERATE OUTPUT")

        # Generate timestamp for this run's outputs
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create timestamped subdirectory for this run's outputs
        base_output_dir = Path(config.otio_output_dir)
        output_dir = base_output_dir / run_timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Output directory: {output_dir}")

        outputs = {}

        if not self.matches:
            print("  ⚠ No matches to export")
            return outputs

        try:
            from src.otio_builder import create_timeline, save_timeline, save_timeline_split, save_timeline_as_edl, generate_resolve_xml_with_bins

            # Generate timeline
            print(f"  Creating timeline...")
            timeline = create_timeline(
                matches=self.matches,
                config=config,
                voiceover_path=getattr(self, 'voiceover_path', None),
                frame_rate=getattr(config.output, 'frame_rate', 30.0),
                entity_images=getattr(self, 'entity_images', None),  # V9 entity stills
                entity_videos=getattr(self, 'entity_videos', None),  # V10 stock videos
                downloaded_segments=getattr(self, 'downloaded_segments', None)  # Audio-first video segments
            )

            # Output formats (config-driven)
            if config.output.generate_otio:
                otio_base_path = output_dir / "timeline"
                
                # Check if we should split the OTIO
                split_otio = getattr(config.output, 'split_otio', True)
                
                if split_otio:
                    # Save split OTIO files (tracks + full)
                    otio_paths = save_timeline_split(timeline, str(otio_base_path))
                    outputs['otio'] = otio_paths
                    
                    # Categorize paths for display
                    full = [p for p in otio_paths if '_FULL' in p]
                    tracks = [p for p in otio_paths if '_V' in Path(p).name and '_FULL' not in p]
                    audio = [p for p in otio_paths if '_A8_' in p]
                    
                    print(f"  ✓ OTIO files generated ({len(otio_paths)} total):")
                    
                    # Show track files
                    if tracks:
                        print(f"    Individual tracks:")
                        for p in tracks:
                            print(f"      - {Path(p).name}")
                    
                    # Show audio
                    if audio:
                        for p in audio:
                            print(f"      - {Path(p).name}")
                    
                    # Show full timeline (last)
                    if full:
                        print(f"    Full timeline:")
                        for p in full:
                            print(f"      - {Path(p).name}")
                else:
                    # Save single OTIO file
                    otio_path = str(otio_base_path) + ".otio"
                    save_timeline(timeline, otio_path)
                    outputs['otio'] = otio_path
                    print(f"  ✓ OTIO: {otio_path}")
            
            if config.output.generate_edl:
                edl_path = output_dir / "timeline.edl"
                save_timeline_as_edl(
                    self.matches,
                    str(edl_path),
                    frame_rate=getattr(config.output, 'frame_rate', 30.0),
                    timeline_start_tc=getattr(config.output, 'timeline_start_tc', "01:00:00:00"),
                    entities=getattr(self, 'extracted_entities', [])  # Pass entities for text overlay markers
                )
                outputs['edl'] = str(edl_path)
                print(f"  ✓ EDL: {edl_path}")

            # Generate DaVinci Resolve XML with media bin AND timeline (FALLBACK)
            if getattr(config.output, 'generate_xml', True):
                xml_base_path = output_dir / "timeline"
                num_parts = getattr(config.output, 'xml_parts', 2)
                xml_paths = generate_resolve_xml_with_bins(
                    matches=self.matches,
                    output_path=str(xml_base_path),
                    voiceover_path=getattr(self, 'voiceover_path', None),
                    frame_rate=getattr(config.output, 'frame_rate', 30.0),
                    entity_images=getattr(self, 'entity_images', None),
                    entity_videos=getattr(self, 'entity_videos', None),
                    config=config,
                    num_parts=num_parts
                )
                outputs['xml'] = xml_paths
                print(f"  ✓ XML (fallback): {Path(xml_paths[0]).name}")

            # Generate report if enabled
            if config.output.generate_report:
                report_path = output_dir / "match_report.md"
                self._generate_report(self.matches, str(report_path))
                outputs['report'] = str(report_path)
                print(f"  ✓ Report: {report_path}")
            
        except ImportError as e:
            logger.error(f"Could not import otio_builder: {e}")
        except Exception as e:
            logger.error(f"Output generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return outputs
    
    def _generate_report(self, matches, output_path: str):
        """Generate a markdown report of matches"""
        lines = [
            "# Match Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total segments: {len(matches)}",
            "",
            "## Matches",
            ""
        ]
        
        for i, match_result in enumerate(matches, 1):
            if not match_result or not hasattr(match_result, 'primary_match') or not match_result.primary_match:
                continue
            
            m = match_result.primary_match
            lines.append(f"### Segment {i}")
            lines.append(f"**Voiceover:** {m.voiceover_segment.text[:100]}...")
            lines.append(f"**Video:** {Path(m.video_segment.source_file).name}")
            lines.append(f"**Confidence:** {m.confidence:.1%}")
            lines.append(f"**Reasoning:** {m.reasoning}")
            lines.append("")
        
        Path(output_path).write_text("\n".join(lines), encoding='utf-8')
    
    def run(self, voiceover_path: str, num_keywords: int = None, match_only: bool = False,
            resume: bool = False, fresh: bool = False,
            use_keywords: str = None, save_keywords: str = None,
            force_rematch: bool = False):
        """
        Run the complete pipeline.
        All settings from config.yaml.

        Args:
            voiceover_path: Path to voiceover file
            num_keywords: Number of keywords to extract (None = use config)
            match_only: Skip download, only match existing footage
            resume: Resume from checkpoint if available
            fresh: Force fresh start, ignore checkpoint
            use_keywords: Use saved keywords preset (name or 'latest')
            save_keywords: Save extracted keywords with this name ('auto' for auto-name)
            force_rematch: Force rematch all videos, ignoring cached matches
        """
        # Store force_rematch for use in stage_match
        self.force_rematch = force_rematch
        start_time = time.time()
        
        # Initialize checkpoint and keyword managers
        project_dir = Path(voiceover_path).parent
        if PROJECT_DIR:
            project_dir = PROJECT_DIR
        
        config_hash = getattr(self.config, '_hash', '')
        self.checkpoint = CheckpointManager(project_dir, config_hash)
        self.keyword_manager = KeywordManager(project_dir)
        
        # Handle checkpoint logic
        if fresh and self.checkpoint.exists():
            print("\n  🗑️ Clearing checkpoint (--fresh mode)")
            self.checkpoint.clear()
        elif self.checkpoint.exists():
            self.checkpoint.load()

            # Auto-clear stale checkpoints (default: 24 hours)
            stale_hours = getattr(self.config.pipeline, 'checkpoint_stale_hours', 24.0)
            if self.checkpoint.is_stale(stale_hours):
                age_hours = self.checkpoint.get_age_hours()
                print(f"\n  🗑️ Auto-clearing stale checkpoint ({age_hours:.1f} hours old, limit: {stale_hours}h)")
                self.checkpoint.clear()
                # Don't show interactive prompt - just continue fresh
            elif not resume and not self.config.enhanced.non_interactive and not match_only:
                # Interactive mode: ask user about valid checkpoint
                validation = self.checkpoint.validate(voiceover_path)
                print(format_resume_prompt(self.checkpoint))
                while True:
                    try:
                        choice = input("  Choice [R/F/Q]: ").strip().upper()
                        if choice == 'R':
                            resume = True
                            break
                        elif choice == 'F':
                            self.checkpoint.clear()
                            break
                        elif choice == 'Q':
                            print("  Exiting.")
                            return
                    except (EOFError, KeyboardInterrupt):
                        print("\n  Exiting.")
                        return
            elif match_only:
                # Match-only mode: skip checkpoint prompt, just clear and proceed
                print(f"\n  ⏭ Match-only mode: ignoring checkpoint, skipping to matching stage")
                self.checkpoint.clear()
            elif resume:
                # --resume flag: automatically resume
                validation = self.checkpoint.validate(voiceover_path)
                print(f"\n  📂 Resuming from checkpoint: {self.checkpoint.data.last_completed_stage}")
                for warning in validation.get('warnings', []):
                    print(f"  ⚠ {warning}")
            else:
                # Non-interactive mode without --resume: warn and continue fresh
                print(f"\n  ⚠ Checkpoint exists but --resume not specified. Starting fresh.")
                self.checkpoint.clear()
        
        self.resume_mode = resume and self.checkpoint.data is not None
        
        self._print_banner()

        # =====================================================================
        # UPFRONT CONFIGURATION - All prompts happen here, then pipeline runs
        # =====================================================================

        # Delta matching prompt (if not already specified via CLI)
        if not force_rematch and not self.config.enhanced.non_interactive:
            self._prompt_delta_matching()

        # Stage 1: Analyze voiceover (extracts keywords + detects topic)
        keywords = None
        use_saved = False
        
        # Check if resuming from checkpoint
        if self.resume_mode and self.checkpoint.should_skip_stage("ANALYZE"):
            print(f"\n  ⏭ Skipping ANALYZE (completed in previous run)")
            logger.info("Skipping ANALYZE stage (checkpoint resume)")
            # Restore state from checkpoint
            analyze_data = self.checkpoint.get_stage_data("ANALYZE")
            keywords = analyze_data.get('keywords', [])
            self.keywords = keywords
            self.voiceover_segments = analyze_data.get('segments', [])
            self.topic_context = analyze_data.get('topic_context', '')
            self.extracted_entities = analyze_data.get('entities', [])

            # Detect location chapters (needed for location-aware matching)
            self._detect_location_chapters()

            use_saved = True
            
        # Check if using saved keywords from command line
        elif use_keywords:
            preset = None
            if use_keywords == 'latest':
                preset = self.keyword_manager.get_latest()
            else:
                preset = self.keyword_manager.get_preset(use_keywords)
            
            if preset:
                print(f"\n  📂 Using saved keywords: [{preset.name}]")
                print(f"     Created: {preset.created_at[:19] if preset.created_at else 'unknown'}")
                print(f"     Keywords: {len(preset.keywords)}")
                print(f"     Topic: {preset.topic_context[:50]}..." if preset.topic_context else "")
                
                keywords = preset.keywords
                self.keywords = keywords
                self.topic_context = preset.topic_context
                self.extracted_entities = preset.entities
                
                # Still need to parse voiceover segments
                self.voiceover_segments = self._load_voiceover_segments(voiceover_path)
                print(f"  ✓ Parsed {len(self.voiceover_segments)} voiceover segments")

                # Detect location chapters (needed for location-aware matching)
                self._detect_location_chapters()

                use_saved = True
            else:
                print(f"\n  ⚠ Saved keywords '{use_keywords}' not found, extracting new keywords...")
        
        # Check for saved keywords (interactive mode)
        elif not self.config.enhanced.non_interactive and self.keyword_manager.has_presets():
            print(format_keyword_prompt(self.keyword_manager))
            while True:
                try:
                    choice = input("  Choice [U/L/N]: ").strip().upper()
                    if choice == 'U':
                        preset = self.keyword_manager.get_latest()
                        if preset:
                            print(f"\n  📂 Using saved keywords: [{preset.name}]")
                            keywords = preset.keywords
                            self.keywords = keywords
                            self.topic_context = preset.topic_context
                            self.extracted_entities = preset.entities
                            self.voiceover_segments = self._load_voiceover_segments(voiceover_path)
                            print(f"  ✓ Parsed {len(self.voiceover_segments)} voiceover segments")
                            self._detect_location_chapters()
                            use_saved = True
                        break
                    elif choice == 'L':
                        # List all presets with selection
                        presets = self.keyword_manager.list_presets()
                        print(f"\n  Saved presets:")
                        for i, p in enumerate(presets, 1):
                            kw_preview = ", ".join(p.keywords[:2])
                            if len(p.keywords) > 2:
                                kw_preview += f"... (+{len(p.keywords)-2})"
                            print(f"    {i}. [{p.name}] {len(p.keywords)} keywords: {kw_preview}")
                        print(f"    0. Cancel (generate new)")
                        
                        try:
                            idx = int(input("\n  Select preset [1-{}]: ".format(len(presets))).strip())
                            if 1 <= idx <= len(presets):
                                preset = presets[idx - 1]
                                print(f"\n  📂 Using saved keywords: [{preset.name}]")
                                keywords = preset.keywords
                                self.keywords = keywords
                                self.topic_context = preset.topic_context
                                self.extracted_entities = preset.entities
                                self.voiceover_segments = self._load_voiceover_segments(voiceover_path)
                                print(f"  ✓ Parsed {len(self.voiceover_segments)} voiceover segments")
                                self._detect_location_chapters()
                                use_saved = True
                        except (ValueError, IndexError):
                            pass
                        break
                    elif choice == 'N':
                        break
                except (EOFError, KeyboardInterrupt):
                    print("\n  Generating new keywords...")
                    break
        
        # Extract keywords if not using saved
        if keywords is None:
            stage_start = time.time()
            keywords = self.stage_analyze_voiceover(voiceover_path, num_keywords)
            stage_duration = time.time() - stage_start
            if self.run_logger:
                self.run_logger.log_stage_complete("ANALYZE", stage_duration, {
                    "segments": len(self.voiceover_segments),
                    "keywords": len(keywords)
                })
                self.run_logger.set_stats(total_segments=len(self.voiceover_segments))
            
            # Save checkpoint
            self.checkpoint.set_voiceover(voiceover_path)
            self.checkpoint.save("ANALYZE", {
                'keywords': keywords,
                'segments': self.voiceover_segments,
                'segment_count': len(self.voiceover_segments),
                'topic_context': self.topic_context,
                'entities': self.extracted_entities
            })
            
            # Save keywords if requested
            if save_keywords:
                preset_name = None if save_keywords == 'auto' else save_keywords
                saved_name = self.keyword_manager.save_keywords(
                    keywords=keywords,
                    topic_context=self.topic_context,
                    entities=self.extracted_entities,
                    name=preset_name,
                    voiceover_path=voiceover_path
                )
                print(f"  💾 Keywords saved as: [{saved_name}]")
        
        # Reinitialize keyword remixer if needed
        if use_saved and self.topic_context and self.modules.get('keyword_remix'):
            try:
                from src.keyword_remix import KeywordRemixer
                self.keyword_remixer = KeywordRemixer(
                    topic_context=self.topic_context
                )
            except Exception as e:
                logger.warning(f"Could not reinitialize KeywordRemixer: {e}")
        
        if not match_only:
            # Get all user preferences BEFORE starting the pipeline
            self._configure_pipeline_upfront(keywords)
        
        # =====================================================================
        # PIPELINE EXECUTION - No pauses, runs end-to-end
        # =====================================================================
        
        if not match_only:
            # Stage 1.5: Entity image search (after keywords, before video download)
            if self.config.image_search.enabled and not self.config.pipeline.skip_image_search:
                if self.resume_mode and self.checkpoint.should_skip_stage("ENTITY_IMAGES"):
                    print(f"\n  ⏭ Skipping ENTITY_IMAGES (completed in previous run)")
                    logger.info("Skipping ENTITY_IMAGES stage (checkpoint resume)")
                else:
                    stage_start = time.time()
                    self.stage_image_search()
                    stage_duration = time.time() - stage_start
                    # EntityImageResult has .images attribute (list of paths)
                    entity_images = getattr(self, 'entity_images', {})
                    if entity_images:
                        entity_img_count = sum(
                            len(v.images) if hasattr(v, 'images') else (len(v) if isinstance(v, list) else 0)
                            for v in entity_images.values()
                        )
                    else:
                        entity_img_count = 0
                    if self.run_logger:
                        self.run_logger.log_stage_complete("ENTITY_IMAGES", stage_duration, {
                            "images": entity_img_count
                        })
                        self.run_logger.set_stats(entity_images_downloaded=entity_img_count)
                    
                    # Save checkpoint
                    self.checkpoint.save("ENTITY_IMAGES", {
                        'image_count': entity_img_count
                    })
            elif self.config.pipeline.skip_image_search:
                print(f"\n  ⏭ Skipping image search (config: skip_image_search=true)")
                logger.info("Skipping IMAGE_SEARCH stage (config: skip_image_search=true)")
            
            # Stage 1.6: Stock video search (Pexels/Pixabay)
            if self.config.image_search.enabled and self.config.image_search.use_stock_apis and not self.config.pipeline.skip_image_search:
                if self.resume_mode and self.checkpoint.should_skip_stage("ENTITY_VIDEOS"):
                    print(f"\n  ⏭ Skipping ENTITY_VIDEOS (completed in previous run)")
                    logger.info("Skipping ENTITY_VIDEOS stage (checkpoint resume)")
                else:
                    stage_start = time.time()
                    self.stage_stock_video()
                    stage_duration = time.time() - stage_start
                    # EntityVideoResult has .videos attribute (list of paths)
                    entity_videos = getattr(self, 'entity_videos', {})
                    if entity_videos:
                        entity_vid_count = sum(
                            len(v.videos) if hasattr(v, 'videos') else (len(v) if isinstance(v, list) else 0)
                            for v in entity_videos.values()
                        )
                    else:
                        entity_vid_count = 0
                    if self.run_logger:
                        self.run_logger.log_stage_complete("ENTITY_VIDEOS", stage_duration, {
                            "videos": entity_vid_count
                        })
                        self.run_logger.set_stats(entity_videos_downloaded=entity_vid_count)
                    
                    # Save checkpoint
                    self.checkpoint.save("ENTITY_VIDEOS", {
                        'video_count': entity_vid_count
                    })
            
            # Stage 1.7: List-based keyword detection
            # Detects numbered lists and prepends guaranteed keywords
            list_config = getattr(self.config.keyword, 'list_detection', None)
            if list_config and getattr(list_config, 'enabled', False):
                list_items = self._detect_list_items(self.voiceover_segments)
                
                if list_items:
                    # Get entity texts to check for coverage
                    entity_texts = set()
                    if getattr(list_config, 'skip_if_entity_covered', True):
                        for e in getattr(self, 'extracted_entities', []):
                            entity_text = e.get('text', '').lower()
                            if entity_text:
                                entity_texts.add(entity_text)
                    
                    # Generate keywords from list items
                    suffix = getattr(list_config, 'keyword_suffix', 'footage')
                    list_keywords = self._generate_list_keywords(list_items, entity_texts, suffix)
                    
                    if list_keywords:
                        print(f"\n  ─── List Items Detected ({len(list_items)}) ───")
                        for item in list_items[:5]:
                            print(f"    #{item['number']}: {item['text']}")
                        if len(list_items) > 5:
                            print(f"    ... and {len(list_items) - 5} more")
                        
                        print(f"  ✓ Generated {len(list_keywords)} list keywords")
                        if list_keywords:
                            print(f"    {', '.join(list_keywords[:3])}")
                            if len(list_keywords) > 3:
                                print(f"    ... and {len(list_keywords) - 3} more")
                        
                        # Prepend list keywords if download_first is enabled
                        if getattr(list_config, 'download_first', True):
                            keywords = list_keywords + [k for k in keywords if k not in list_keywords]
                            logger.info(f"List keywords prepended: {list_keywords}")
                        else:
                            keywords = keywords + [k for k in list_keywords if k not in keywords]
            
            # Stage 2: Download footage (or audio-only for audio-first mode)
            if not self.config.pipeline.skip_download:
                if self.resume_mode and self.checkpoint.should_skip_stage("DOWNLOAD"):
                    print(f"\n  ⏭ Skipping DOWNLOAD (completed in previous run)")
                    logger.info("Skipping DOWNLOAD stage (checkpoint resume)")
                    # Load existing videos
                    self._load_existing_videos()
                    # For audio-first mode, rebuild audio_downloads from disk
                    if self._is_audio_first_enabled():
                        self._rebuild_audio_downloads_from_disk()
                else:
                    stage_start = time.time()

                    # Check if audio-first mode is enabled
                    if self._is_audio_first_enabled():
                        print(f"\n  ─── Audio-First Mode ───")
                        print(f"  • Downloading audio only (MP3) for faster processing")
                        print(f"  • Video segments will be downloaded after matching")
                        self.audio_downloads = self.stage_download_audio(keywords)
                        dl_stats = {"downloaded": len(self.audio_downloads), "mode": "audio-first"}
                    else:
                        self.stage_download(keywords)
                        dl_stats = {"downloaded": len(getattr(self, 'downloaded_videos', []))}
                        if hasattr(self, 'downloader') and self.downloader:
                            dl_stats["skipped"] = getattr(self.downloader, 'skipped_count', 0)
                            dl_stats["failed"] = len(getattr(self, 'failed_keywords', []))

                    stage_duration = time.time() - stage_start

                    if self.run_logger:
                        self.run_logger.log_stage_complete("DOWNLOAD", stage_duration, dl_stats)
                        self.run_logger.set_stats(
                            videos_downloaded=dl_stats.get("downloaded", 0),
                            videos_skipped=dl_stats.get("skipped", 0),
                            videos_failed=dl_stats.get("failed", 0)
                        )

                    # Stage 2c: Stock footage (skip in audio-first mode)
                    if not self._is_audio_first_enabled():
                        self.stage_download_stock(keywords)

                        # Remix zero-download keywords (generate alternative keywords)
                        keywords = self.stage_remix_zero_downloads(keywords)

                    # Save checkpoint with video/audio paths
                    if self._is_audio_first_enabled():
                        audio_paths = [ad.audio_file for ad in getattr(self, 'audio_downloads', [])]
                        self.checkpoint.save("DOWNLOAD", {
                            'audio_paths': audio_paths,
                            'audio_count': len(audio_paths),
                            'keywords': keywords,
                            'mode': 'audio-first'
                        })
                    else:
                        video_paths = [str(v.get('path', v)) if isinstance(v, dict) else str(v)
                                       for v in self.downloaded_videos]
                        self.checkpoint.save("DOWNLOAD", {
                            'video_paths': video_paths,
                            'video_count': len(video_paths),
                            'keywords': keywords
                        })
            else:
                print(f"\n  ⏭ Skipping downloads (config: skip_download=true)")
                logger.info("Skipping DOWNLOAD stage (config: skip_download=true)")
                # Load existing videos from output directory
                self._load_existing_videos()
            
            # Stage 2.5: Zero-download video remix (filter videos by keyword relevance)
            if self.config.remix.enabled:
                if self.resume_mode and self.checkpoint.should_skip_stage("REMIX"):
                    print(f"\n  ⏭ Skipping REMIX (completed in previous run)")
                    logger.info("Skipping REMIX stage (checkpoint resume)")
                    # Load remixed videos from downloaded_videos dir
                    self._load_existing_videos()
                else:
                    stage_start = time.time()
                    self.stage_remix(keywords)
                    stage_duration = time.time() - stage_start
                    if self.run_logger:
                        self.run_logger.log_stage_complete("REMIX", stage_duration, {
                            "videos": len(getattr(self, 'remixed_video_paths', []))
                        })
                    
                    # Save checkpoint
                    self.checkpoint.save("REMIX", {
                        'remixed_paths': [str(p) for p in getattr(self, 'remixed_video_paths', [])],
                        'video_count': len(getattr(self, 'remixed_video_paths', []))
                    })

        # Stage 3: Transcribe & index
        # For match_only mode or resume, load from cache instead of recomputing
        skip_transcribe = False

        if match_only:
            # Match-only mode: skip transcription entirely, load from cache
            print(f"\n  ⏭ Match-only mode: loading transcripts and embeddings from cache")
            logger.info("Skipping TRANSCRIBE stage (match-only mode)")
            self._load_existing_videos()
            stage_start = time.time()
            self._load_transcripts_from_cache()
            stage_duration = time.time() - stage_start
            skip_transcribe = True
        elif self.resume_mode and self.checkpoint.should_skip_stage("TRANSCRIBE"):
            # Resume mode: load from cache
            print(f"\n  ⏭ Skipping TRANSCRIBE (completed in previous run)")
            logger.info("Skipping TRANSCRIBE stage (checkpoint resume)")
            self._load_existing_videos()
            stage_start = time.time()
            self._load_transcripts_from_cache()
            stage_duration = time.time() - stage_start
            skip_transcribe = True

        if not skip_transcribe:
            # Normal mode: run transcription
            stage_start = time.time()
            self.stage_transcribe()
            stage_duration = time.time() - stage_start

        transcribe_stats = {
            "transcribed": len(getattr(self, 'transcripts', {})),
            "embeddings": len(getattr(self, 'embeddings', [])) if hasattr(self, 'embeddings') and self.embeddings is not None else 0
        }
        if self.run_logger:
            self.run_logger.log_stage_complete("TRANSCRIBE", stage_duration, transcribe_stats)
            self.run_logger.set_stats(
                videos_transcribed=transcribe_stats["transcribed"],
                embeddings_computed=transcribe_stats["embeddings"]
            )

        # Save checkpoint after transcription (skip for match_only to avoid overwriting)
        if not match_only:
            self.checkpoint.save("TRANSCRIBE", {
                'transcribed_count': transcribe_stats["transcribed"],
                'embedding_count': transcribe_stats["embeddings"]
            })
        
        # Stage 4: Match
        stage_start = time.time()
        self.stage_match()
        stage_duration = time.time() - stage_start
        
        match_stats = {}
        confidences = []  # Initialize before use
        if self.matches:
            confidences = [m.primary_match.confidence for m in self.matches if m and m.primary_match]
            match_stats = {
                "matched": len(self.matches),
                "avg_conf": f"{sum(confidences)/len(confidences)*100:.0f}%" if confidences else "0%"
            }
        if self.run_logger:
            self.run_logger.log_stage_complete("MATCH", stage_duration, match_stats)
            self.run_logger.set_stats(
                total_matches=len(self.matches) if self.matches else 0,
                avg_confidence=sum(confidences)/len(confidences) if confidences else 0
            )
        
        # Save checkpoint after matching
        self.checkpoint.save("MATCH", {
            'match_count': len(self.matches) if self.matches else 0,
            'avg_confidence': sum(confidences)/len(confidences) if confidences else 0
        })

        # Stage 4.5: Download video segments (audio-first mode only)
        if self._is_audio_first_enabled() and self.matches:
            if self.resume_mode and self.checkpoint.should_skip_stage("VIDEO_SEGMENTS"):
                print(f"\n  ⏭ Skipping VIDEO_SEGMENTS (completed in previous run)")
                logger.info("Skipping VIDEO_SEGMENTS stage (checkpoint resume)")
            else:
                stage_start = time.time()
                print(f"\n  ─── Downloading Video Segments ───")
                print(f"  • Matched {len(self.matches)} voiceover segments")
                print(f"  • Downloading only required video portions")

                self.downloaded_segments = self.stage_download_video_segments()
                stage_duration = time.time() - stage_start

                segment_stats = {
                    "segments": len(self.downloaded_segments),
                    "total_matches": sum(len(seg.matches) for seg in self.downloaded_segments)
                }

                if self.run_logger:
                    self.run_logger.log_stage_complete("VIDEO_SEGMENTS", stage_duration, segment_stats)

                # Save checkpoint
                self.checkpoint.save("VIDEO_SEGMENTS", {
                    'segment_count': len(self.downloaded_segments),
                    'segment_files': [seg.file for seg in self.downloaded_segments]
                })

                # Run face detection on downloaded video segments (deferred from Stage 3)
                face_pref = getattr(self, 'face_preference', 'neutral')
                if face_pref != 'neutral' and self.downloaded_segments:
                    segment_files = [Path(seg.file) for seg in self.downloaded_segments]
                    print(f"\n  ─── Face Detection (on video segments) ───")
                    self._predetect_faces(segment_files)

                    # Map face scores from video segments to audio files
                    # This allows second-pass matching to use face scores
                    self._map_segment_face_scores_to_audio()

                    # Second-pass matching with face preference
                    print(f"\n  ─── Second-Pass Matching (with face preference: {face_pref}) ───")
                    self.matches = self.stage_match()
                    print(f"  ✓ Re-matched {len(self.matches)} segments with face preference")

        # Stage 5: Output
        stage_start = time.time()
        outputs = self.stage_output()
        stage_duration = time.time() - stage_start
        
        if self.run_logger:
            self.run_logger.log_stage_complete("OUTPUT", stage_duration, {
                "files": len(outputs)
            })
            # Log each output file
            for file_type, file_path in outputs.items():
                if isinstance(file_path, list):
                    for fp in file_path:
                        self.run_logger.log_file_generated(file_type, str(fp))
                else:
                    self.run_logger.log_file_generated(file_type, str(file_path))
        
        # Mark output stage complete and clear checkpoint (pipeline finished successfully)
        self.checkpoint.save("OUTPUT", {"files": len(outputs)})
        self.checkpoint.clear()  # Pipeline completed successfully - no need to resume

        # Finalize logger (prints comprehensive run summary)
        if self.run_logger:
            self.run_logger.set_stats(total_segments=len(self.srt_segments) if hasattr(self, 'srt_segments') else 0)
            self.run_logger.finalize()
            print(f"  Log file: {self.run_logger.log_file}")
    
    def _configure_pipeline_upfront(self, keywords: List[str]):
        """
        Configure all pipeline options UPFRONT before any processing starts.
        This ensures the pipeline runs end-to-end without pauses.
        
        Set enhanced.non_interactive=true in config.yaml to skip all prompts.
        """
        config = self.config
        
        # Non-interactive mode - skip all prompts, use defaults
        if config.enhanced.non_interactive:
            print(f"\n  ─── Non-Interactive Mode ───")
            print(f"  • Keywords: {len(keywords)}")
            print(f"  • Topic: {self.topic_context}")
            print(f"  • Face preference: {self.face_preference}")
            print(f"  • Video filtering: filtered (auto)")
            print(f"  • Using default settings (no prompts)")
            
            # Use defaults
            self.enhanced_enabled = config.enhanced.remix_enabled
            
            # Set default video filtering to 'filtered' (skip prompt)
            config.remix.auto_accept_filter = "filtered"
            
            # Disable interactive curation in remix
            config.remix.interactive_curation = False
            
            if self.enhanced_enabled and self.modules.get('keyword_remix'):
                try:
                    from src.keyword_remix import KeywordRemixer
                    self.keyword_remixer = KeywordRemixer(topic_context=self.topic_context)
                except:
                    pass
            return
        
        print(f"\n{'═' * 70}")
        print(f"  PIPELINE CONFIGURATION")
        print(f"{'═' * 70}")
        
        # Show detected topic
        if self.topic_context:
            print(f"\n  Detected topic: {self.topic_context}")
        
        # Show keywords summary
        print(f"  Keywords: {len(keywords)} extracted")
        print(f"    Top 5: {', '.join(keywords[:5])}")
        
        # Show entities with context
        if hasattr(self, 'extracted_entities') and self.extracted_entities:
            print(f"\n  ─── Named Entities ({len(self.extracted_entities)}) ───")
            entity_types = {}
            for entity in self.extracted_entities:
                etype = entity.get('type', 'OTHER')
                if etype not in entity_types:
                    entity_types[etype] = []
                entity_types[etype].append(entity)
            
            type_labels = {
                'PERSON': '👤 People',
                'GPE': '📍 Places', 
                'ORG': '🏢 Organizations',
                'DATE': '📅 Dates',
                'EVENT': '⚡ Events'
            }
            
            for etype, entities in entity_types.items():
                label = type_labels.get(etype, f'📌 {etype}')
                print(f"  {label}:")
                for e in entities[:3]:  # Show max 3 per type
                    name = e.get('text', '')
                    context = e.get('context', '')
                    search_kw = e.get('search_keyword', '')
                    if context:
                        print(f"    • {name} - {context}")
                    else:
                        print(f"    • {name}")
                    if search_kw:
                        print(f"      → Search: \"{search_kw}\"")
                if len(entities) > 3:
                    print(f"    ... and {len(entities) - 3} more")
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. KEYWORD REVIEW
        # ─────────────────────────────────────────────────────────────────────
        if config.enhanced.confirm_before_download:
            print(f"\n  ─── Keyword Review ───")
            
            # Separate entity-based keywords from general keywords
            entity_keywords = []
            general_keywords = []
            
            if hasattr(self, 'extracted_entities') and self.extracted_entities:
                entity_search_terms = {e.get('search_keyword', '').lower() for e in self.extracted_entities if e.get('search_keyword')}
                for kw in keywords:
                    if kw.lower() in entity_search_terms:
                        entity_keywords.append(kw)
                    else:
                        general_keywords.append(kw)
            else:
                general_keywords = keywords
            
            if entity_keywords:
                print(f"  Entity keywords ({len(entity_keywords)}):")
                print(f"    {', '.join(entity_keywords[:5])}")
                if len(entity_keywords) > 5:
                    print(f"    ... and {len(entity_keywords) - 5} more")
            
            print(f"  General keywords ({len(general_keywords)}):")
            print(f"    {', '.join(general_keywords[:8])}")
            if len(general_keywords) > 8:
                print(f"    ... and {len(general_keywords) - 8} more")
            
            if not self._get_user_confirmation("\n  Use these keywords?"):
                custom = self._get_user_input("  Enter custom keywords (comma-separated)")
                if custom:
                    self.keywords = [k.strip() for k in custom.split(',')]
                    keywords[:] = self.keywords  # Modify in place
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. TOPIC OVERRIDE
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  ─── Topic Context ───")
        print(f"  Current: {self.topic_context or 'Not detected'}")
        
        custom_topic = self._get_user_input("  Custom topic (Enter to keep)")
        if custom_topic:
            self.topic_context = custom_topic
            print(f"  ✓ Topic set to: {self.topic_context}")
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. ENHANCED FEATURES
        # ─────────────────────────────────────────────────────────────────────
        if config.enhanced.prompt_enhanced_features and any([
            self.modules['keyword_remix'],
            self.modules['pexels'],
            self.modules['pixabay'],
            self.modules['multi_style']
        ]):
            print(f"\n  ─── Enhanced Features ───")
            
            # Confidence enforcement
            if self.modules['keyword_remix'] and config.enhanced.remix_enabled:
                min_conf = config.enhanced.min_confidence
                self.enhanced_enabled = self._get_user_confirmation(
                    f"  Enable {min_conf:.0%} confidence enforcement?"
                )
                
                if self.enhanced_enabled:
                    # Initialize remixer with topic
                    try:
                        from src.keyword_remix import KeywordRemixer
                        self.keyword_remixer = KeywordRemixer(topic_context=self.topic_context)
                        print(f"  ✓ Keyword remixer ready")
                    except Exception as e:
                        logger.debug(f"Could not init remixer: {e}")
            
            # Multi-style OTIO
            if self.modules['multi_style'] and config.multi_style.enabled:
                if self._get_user_confirmation("  Generate multiple OTIO styles?", default=False):
                    from src.multi_style import prompt_for_second_style
                    self.second_style = prompt_for_second_style()
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. VIDEO FILTERING
        # ─────────────────────────────────────────────────────────────────────
        if config.remix.enabled and config.remix.interactive_curation:
            print(f"\n  ─── Video Filtering ───")
            print(f"  After download, videos are scored for keyword relevance.")
            print(f"    [A] Use ALL downloaded videos (skip filtering)")
            print(f"    [F] Use FILTERED videos (higher relevance, fewer options)")
            
            choice = self._get_user_input("  Select [A/F]", default="A").strip().upper()
            
            if choice == 'F':
                config.remix.auto_accept_filter = "filtered"
                print(f"  ✓ Will use filtered videos (min score: {config.remix.min_relevance_score})")
            else:
                config.remix.auto_accept_filter = "all"
                print(f"  ✓ Will use ALL videos")
        
        # ─────────────────────────────────────────────────────────────────────
        # 4.5 FACE PREFERENCE
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  ─── Face Preference ───")
        print(f"  Control whether matched clips should contain faces:")
        print(f"    [M] MORE faces - prefer clips with people/faces")
        print(f"    [N] NO faces - prefer clips without people (b-roll, scenery)")
        print(f"    [X] No preference - don't filter by faces")
        
        face_choice = self._get_user_input("  Select [M/N/X]", default="N").strip().upper()
        
        if face_choice == 'M':
            self.face_preference = "more"
            print(f"  ✓ Will prefer clips with faces")
        elif face_choice == 'N':
            self.face_preference = "none"
            print(f"  ✓ Will prefer clips without faces")
        else:
            self.face_preference = "neutral"
            print(f"  ✓ No face preference")
        
        # ─────────────────────────────────────────────────────────────────────
        # 5. FINAL CONFIRMATION
        # ─────────────────────────────────────────────────────────────────────
        entity_count = len(self.extracted_entities) if hasattr(self, 'extracted_entities') else 0
        
        print(f"\n  ─── Pipeline Summary ───")
        print(f"  • Keywords: {len(keywords)} ({entity_count} entity-based)")
        print(f"  • Topic: {self.topic_context}")
        print(f"  • Confidence enforcement: {'Yes' if self.enhanced_enabled else 'No'}")
        print(f"  • Zero-download remix: {'Yes' if config.zero_download_remix.enabled else 'No'}")
        print(f"  • Video filtering: {config.remix.auto_accept_filter.upper() if config.remix.enabled else 'Disabled'}")
        print(f"  • Face preference: {self.face_preference.upper()}")
        if self.second_style:
            print(f"  • Multi-style output: {self.second_style.name}")
        
        if not self._get_user_confirmation("\n  Start pipeline?"):
            print("\n  ❌ Pipeline cancelled")
            sys.exit(0)
        
        print(f"\n{'═' * 70}")
        print(f"  STARTING PIPELINE (End-to-End)")
        print(f"{'═' * 70}\n")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

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
    
    return parser.parse_args()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

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
    
    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run(
        voiceover_path=str(vo_path),
        num_keywords=args.keywords,
        match_only=args.match_only,
        resume=args.resume,
        fresh=getattr(args, 'fresh', False),
        use_keywords=getattr(args, 'use_keywords', None),
        save_keywords=getattr(args, 'save_keywords', None),
        force_rematch=getattr(args, 'force_rematch', False)
    )


if __name__ == '__main__':
    main()