#!/usr/bin/env python3
"""
Voiceover-Matcher: End-to-End Documentary Footage Pipeline v2.4

PERFORMANCE OPTIMIZATIONS (v2.4):
- Parallel transcription with ThreadPoolExecutor (4 workers)
- Delta-aware indexing (only process NEW videos on retries)
- Batch embedding API calls (100 texts per call)
- Selective vision processing (skip if transcript covers content)

Enhanced features (v2.3):
- 90% confidence enforcement with keyword remix
- Zero-download keyword remix with topic context
- Pexels/Pixabay stock footage integration
- Multi-style OTIO generation

One command to go from voiceover → matched timeline:
1. Extract keywords from voiceover (LLM)
2. Download footage from YouTube + Pexels + Pixabay
3. Transcribe & index videos (OPTIMIZED: parallel + delta-aware)
4. Match to voiceover segments (with 90% confidence enforcement)
5. Output DaVinci Resolve timeline (multiple styles)

Usage:
    python main.py --voiceover script.srt
    python main.py --voiceover script.srt --keywords 30
    python main.py --project "E:\\Projects\\MyDoc" --voiceover voiceover.srt
    python main.py --resume  # Resume interrupted run
    python main.py --match-only  # Skip download, just match existing footage
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Determine install directory (where this script lives)
INSTALL_DIR = Path(__file__).parent.resolve()

# Project directory (set later via --project argument)
PROJECT_DIR = None

# =============================================================================
# ENHANCED FEATURES CONFIGURATION
# =============================================================================
ENHANCED_MIN_CONFIDENCE = 0.90  # 90% minimum confidence
ENHANCED_MAX_RETRIES = 3        # Maximum keyword remix attempts
ENHANCED_ENABLE_PEXELS = True   # Download from Pexels
ENHANCED_ENABLE_PIXABAY = True  # Download from Pixabay
ENHANCED_STOCK_PER_KEYWORD = 2  # Videos per keyword from stock sources

# =============================================================================
# PERFORMANCE OPTIMIZATION FLAGS (v2.4)
# =============================================================================
PARALLEL_WORKERS = 4            # Number of parallel transcription workers
EMBEDDING_BATCH_SIZE = 100      # Texts per embedding API call
MAX_VISION_SCENES = 3           # Vision API scenes per video


def strip_extended_path_prefix(path: Path) -> Path:
    r"""
    Strip Windows extended-length path prefix (\\?\) from a Path.
    This prefix can cause issues with some applications.
    """
    path_str = str(path)
    
    # Remove Windows extended-length path prefix
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


def make_paths_project_relative(config, project_dir: Path):
    """Update config paths to be relative to project directory"""
    project_dir = project_dir.resolve()
    
    # These paths should be in the project folder
    project_relative_paths = [
        'downloaded_videos_dir',
        'otio_output_dir',
        'cache_dir',
    ]
    
    for attr in project_relative_paths:
        if hasattr(config, attr):
            current = getattr(config, attr)
            # If it's a relative path, make it relative to project
            if current and not Path(current).is_absolute():
                new_path = str(project_dir / current)
                setattr(config, attr, new_path)
    
    # Handle logging separately
    if hasattr(config, 'logging') and hasattr(config.logging, 'log_dir'):
        log_dir = config.logging.log_dir
        if log_dir and not Path(log_dir).is_absolute():
            config.logging.log_dir = str(project_dir / log_dir)
    
    # Voiceover path - if relative, make project-relative
    if hasattr(config, 'voiceover_path') and config.voiceover_path:
        vo_path = Path(config.voiceover_path)
        if not vo_path.is_absolute():
            config.voiceover_path = str(project_dir / vo_path)
    
    return config


def load_project_config(project_dir: Path, global_config_path: Path = None):
    """
    Load configuration with project overrides.
    """
    from config import load_config
    import yaml
    
    # Load global config
    if global_config_path is None:
        global_config_path = INSTALL_DIR / 'config.yaml'
    
    config = load_config(str(global_config_path))
    
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
                # Direct override
                setattr(config, key, value)
        else:
            # New attribute
            setattr(config, key, value)
    
    return config


# Add src to path
sys.path.insert(0, str(INSTALL_DIR / 'src'))

from config import Config, load_config
from downloader import VideoDownloader, DownloadCheckpoint
from keyword_extractor import LLMKeywordExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# ENHANCED FEATURES IMPORTS
# =============================================================================
try:
    from src.keyword_remix import KeywordRemixer
    ENHANCED_REMIX_AVAILABLE = True
except ImportError:
    ENHANCED_REMIX_AVAILABLE = False
    logger.debug("keyword_remix module not available")

try:
    from src.pexels import download_pexels_footage
    PEXELS_AVAILABLE = True
except ImportError:
    PEXELS_AVAILABLE = False
    logger.debug("pexels module not available")

try:
    from src.pixabay import download_pixabay_footage
    PIXABAY_AVAILABLE = True
except ImportError:
    PIXABAY_AVAILABLE = False
    logger.debug("pixabay module not available")

try:
    from src.multi_style import prompt_for_second_style, STYLE_DEFAULT, OTIOStyle
    MULTI_STYLE_AVAILABLE = True
except ImportError:
    MULTI_STYLE_AVAILABLE = False
    logger.debug("multi_style module not available")

# =============================================================================
# PERFORMANCE OPTIMIZATION IMPORTS (v2.4)
# =============================================================================
try:
    from src.transcription_optimized import (
        transcribe_videos_parallel,
        DeltaAwareIndex,
        transcribe_voiceover_media
    )
    OPTIMIZED_TRANSCRIPTION = True
    logger.info("✓ Using optimized parallel transcription")
except ImportError:
    OPTIMIZED_TRANSCRIPTION = False
    logger.debug("transcription_optimized module not available - using standard")

try:
    from src.embeddings_optimized import (
        compute_embeddings as compute_embeddings_optimized,
        get_embedding_provider as get_embedding_provider_optimized,
        build_embedding_index as build_embedding_index_optimized,
        EmbeddingCache
    )
    OPTIMIZED_EMBEDDINGS = True
    logger.info("✓ Using optimized batch embeddings")
except ImportError:
    OPTIMIZED_EMBEDDINGS = False
    logger.debug("embeddings_optimized module not available - using standard")

try:
    from src.vision_optimized import process_video_vision as process_video_vision_optimized
    OPTIMIZED_VISION = True
    logger.info("✓ Using selective vision processing")
except ImportError:
    OPTIMIZED_VISION = False
    logger.debug("vision_optimized module not available - using standard")


class Pipeline:
    """End-to-end documentary footage pipeline with enhanced features and performance optimizations"""
    
    def __init__(self, config: Config):
        self.config = config
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
        self.keyword_remixer = None
        self.enhanced_enabled = False
        self.second_style = None
        self.failed_keywords = []
        
        # Performance tracking (v2.4)
        self.stage_timings = {}
        self.use_delta_indexing = True
        
        # Auto-logging
        self.run_logger = None
        if getattr(config, 'logging', None) and getattr(config.logging, 'enabled', False):
            try:
                from src.logger import RunLogger
                log_dir = getattr(config.logging, 'log_dir', './logs')
                self.run_logger = RunLogger(log_dir=log_dir)
                self.run_logger.log_config(config)
                print(f"  ✓ Logging enabled: {self.run_logger.log_file}")
            except Exception as e:
                print(f"  ⚠ Could not initialize logger: {e}")
    
    def _print_banner(self):
        """Print startup banner"""
        print("=" * 70)
        print("  VOICEOVER-MATCHER: End-to-End Documentary Pipeline v2.4")
        print("=" * 70)
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if PROJECT_DIR and PROJECT_DIR != Path.cwd():
            print(f"  Project: {PROJECT_DIR.name}")
            print(f"  Location: {PROJECT_DIR}")
        print(f"  Videos: {self.config.downloaded_videos_dir}")
        print(f"  Output: {self.config.otio_output_dir}")
        if self.run_logger:
            print(f"  Log: {self.run_logger.log_file}")
        
        # Show enhanced features status
        enhanced_status = []
        if ENHANCED_REMIX_AVAILABLE:
            enhanced_status.append("Confidence Enforcement")
        if PEXELS_AVAILABLE:
            enhanced_status.append("Pexels")
        if PIXABAY_AVAILABLE:
            enhanced_status.append("Pixabay")
        if MULTI_STYLE_AVAILABLE:
            enhanced_status.append("Multi-Style OTIO")
        
        if enhanced_status:
            print(f"  Enhanced: {', '.join(enhanced_status)}")
        
        # Show optimization status (v2.4)
        opt_status = []
        if OPTIMIZED_TRANSCRIPTION:
            opt_status.append(f"Parallel({PARALLEL_WORKERS})")
        if OPTIMIZED_EMBEDDINGS:
            opt_status.append(f"Batch({EMBEDDING_BATCH_SIZE})")
        if OPTIMIZED_VISION:
            opt_status.append("SelectiveVision")
        
        if opt_status:
            print(f"  Optimized: {', '.join(opt_status)}")
        
        print("=" * 70)
    
    def _print_stage(self, stage, name: str):
        """Print stage header and log it"""
        print(f"\n{'─' * 70}")
        print(f"  STAGE {stage}: {name}")
        print(f"{'─' * 70}")
        
        # Log to file
        if self.run_logger:
            self.run_logger.file_logger.info(f"STAGE {stage}: {name}")
    
    def _parse_srt(self, srt_path: str) -> List[dict]:
        """Parse SRT file into segments"""
        try:
            import srt
            with open(srt_path, 'r', encoding='utf-8') as f:
                subtitles = list(srt.parse(f.read()))
            
            segments = []
            for sub in subtitles:
                segments.append({
                    'index': sub.index,
                    'start_time': sub.start.total_seconds(),
                    'end_time': sub.end.total_seconds(),
                    'text': sub.content.strip(),
                    'duration': (sub.end - sub.start).total_seconds()
                })
            return segments
        except ImportError:
            logger.error("srt package not installed. Install with: pip install srt")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to parse SRT: {e}")
            sys.exit(1)
    
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
    # ENHANCED FEATURES METHODS
    # =========================================================================
    
    def prompt_enhanced_features(self):
        """Prompt user for enhanced feature settings"""
        if not any([ENHANCED_REMIX_AVAILABLE, PEXELS_AVAILABLE, PIXABAY_AVAILABLE, MULTI_STYLE_AVAILABLE]):
            return
        
        print(f"\n{'─' * 70}")
        print("  ENHANCED FEATURES")
        print(f"{'─' * 70}")
        
        try:
            # Confidence enforcement
            if ENHANCED_REMIX_AVAILABLE:
                enable = input(f"  Enable {ENHANCED_MIN_CONFIDENCE:.0%} confidence enforcement? [Y/n]: ").strip().lower()
                self.enhanced_enabled = enable != 'n'
                
                if self.enhanced_enabled:
                    # Get topic context from voiceover
                    if self.voiceover_segments:
                        auto_topic = " ".join([s.get('text', '')[:50] for s in self.voiceover_segments[:5]])
                        print(f"  Auto-detected topic: {auto_topic[:80]}...")
                        custom = input("  Custom topic (Enter for auto): ").strip()
                        self.topic_context = custom if custom else auto_topic
                    
                    # Initialize remixer
                    try:
                        self.keyword_remixer = KeywordRemixer(topic_context=self.topic_context)
                        print(f"  ✓ Keyword remixer initialized")
                    except Exception as e:
                        print(f"  ⚠ Could not initialize remixer: {e}")
                        self.keyword_remixer = None
            
            # Multi-style OTIO
            if MULTI_STYLE_AVAILABLE:
                multi = input("  Generate multiple OTIO styles? [y/N]: ").strip().lower()
                if multi == 'y':
                    self.second_style = prompt_for_second_style()
            
            # Summary
            if self.enhanced_enabled or self.second_style:
                print(f"\n  ✓ Enhanced features configured:")
                if self.enhanced_enabled:
                    print(f"    • Confidence enforcement: {ENHANCED_MIN_CONFIDENCE:.0%}")
                    print(f"    • Max retries: {ENHANCED_MAX_RETRIES}")
                if PEXELS_AVAILABLE and ENHANCED_ENABLE_PEXELS:
                    print(f"    • Pexels: {ENHANCED_STOCK_PER_KEYWORD} videos/keyword")
                if PIXABAY_AVAILABLE and ENHANCED_ENABLE_PIXABAY:
                    print(f"    • Pixabay: {ENHANCED_STOCK_PER_KEYWORD} videos/keyword")
                if self.second_style:
                    print(f"    • Second OTIO style: {self.second_style.name}")
            
        except (EOFError, KeyboardInterrupt):
            print("\n  Enhanced features disabled")
            self.enhanced_enabled = False
    
    def stage_download_stock(self, keywords: List[str]) -> dict:
        """Stage 2c: Download stock footage from Pexels/Pixabay"""
        if not self.enhanced_enabled:
            return {'skipped': True}
        
        if not PEXELS_AVAILABLE and not PIXABAY_AVAILABLE:
            return {'skipped': True, 'reason': 'no stock modules'}
        
        self._print_stage("2c", "DOWNLOAD STOCK FOOTAGE")
        
        stock_dir = Path(self.config.downloaded_videos_dir) / "stock"
        stock_dir.mkdir(parents=True, exist_ok=True)
        
        total = 0
        stock_keywords = keywords[:15]  # Limit for rate limits
        
        # Pexels
        if PEXELS_AVAILABLE and ENHANCED_ENABLE_PEXELS:
            try:
                print(f"  Pexels: Searching {len(stock_keywords)} keywords...")
                paths, counts = download_pexels_footage(
                    stock_keywords, str(stock_dir), per_keyword=ENHANCED_STOCK_PER_KEYWORD
                )
                total += len(paths)
                print(f"  ✓ Pexels: {len(paths)} videos")
                
                # Track zero-result keywords
                for kw, count in counts.items():
                    if count == 0 and kw not in self.failed_keywords:
                        self.failed_keywords.append(kw)
            except Exception as e:
                print(f"  ⚠ Pexels error: {e}")
        
        # Pixabay
        if PIXABAY_AVAILABLE and ENHANCED_ENABLE_PIXABAY:
            try:
                print(f"  Pixabay: Searching {len(stock_keywords)} keywords...")
                paths, counts = download_pixabay_footage(
                    stock_keywords, str(stock_dir), per_keyword=ENHANCED_STOCK_PER_KEYWORD
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
        """Remix keywords that had zero downloads"""
        if not self.enhanced_enabled or not self.keyword_remixer:
            return keywords
        
        if not self.failed_keywords:
            return keywords
        
        print(f"\n  Remixing {len(self.failed_keywords)} zero-download keywords...")
        
        new_keywords = []
        for kw in self.failed_keywords[:10]:  # Limit
            try:
                remixed = self.keyword_remixer.remix_for_zero_downloads(kw, attempt=1)
                if remixed:
                    print(f"    '{kw}' → {remixed[:2]}")
                    new_keywords.extend(remixed)
            except Exception as e:
                logger.debug(f"Remix failed for '{kw}': {e}")
        
        if new_keywords:
            # Remove duplicates
            new_keywords = list(set(new_keywords) - set(keywords))
            print(f"  ✓ Generated {len(new_keywords)} new keywords")
            return keywords + new_keywords
        
        return keywords
    
    def stage_confidence_enforcement(self, keywords: List[str]) -> dict:
        """
        Stage 4b: Enforce minimum confidence by remixing and re-downloading
        
        OPTIMIZATION (v2.4): Uses force_reprocess=False to enable delta-aware
        indexing during retries. Only NEW stock footage is transcribed.
        """
        if not self.enhanced_enabled or not self.keyword_remixer:
            return {'skipped': True}
        
        if not hasattr(self, 'matches') or not self.matches:
            return {'skipped': True, 'reason': 'no matches'}
        
        self._print_stage("4b", "CONFIDENCE ENFORCEMENT")
        print(f"  Minimum confidence: {ENHANCED_MIN_CONFIDENCE:.0%}")
        print(f"  Max retries: {ENHANCED_MAX_RETRIES}")
        
        if OPTIMIZED_TRANSCRIPTION:
            print(f"  ✓ Delta-aware indexing enabled (only new files will be processed)")
        
        # Count low-confidence matches
        def count_low_confidence():
            low = []
            for i, m in enumerate(self.matches):
                if hasattr(m, 'primary_match') and m.primary_match:
                    conf = m.primary_match.confidence
                    if conf < ENHANCED_MIN_CONFIDENCE:
                        low.append({
                            'index': i,
                            'confidence': conf,
                            'vo_text': m.voiceover_segment.text[:80] if hasattr(m, 'voiceover_segment') else '',
                            'matched_text': m.primary_match.video_segment.text[:80] if hasattr(m.primary_match, 'video_segment') else ''
                        })
            return low
        
        low_conf = count_low_confidence()
        initial_low = len(low_conf)
        
        if not low_conf:
            print(f"  ✓ All {len(self.matches)} matches meet {ENHANCED_MIN_CONFIDENCE:.0%} confidence!")
            return {'low_confidence_initial': 0, 'low_confidence_final': 0, 'retries': 0}
        
        print(f"  ⚠ {len(low_conf)}/{len(self.matches)} matches below {ENHANCED_MIN_CONFIDENCE:.0%}")
        
        # Retry loop
        retries = 0
        all_new_keywords = set()
        
        while low_conf and retries < ENHANCED_MAX_RETRIES:
            retries += 1
            print(f"\n  ── Retry {retries}/{ENHANCED_MAX_RETRIES} ──")
            
            # Sample low-confidence matches
            samples = low_conf[:5]
            new_keywords = []
            
            for sample in samples:
                try:
                    remixed = self.keyword_remixer.remix_for_low_confidence(
                        keyword=self.topic_context.split()[0] if self.topic_context else "footage",
                        matched_text=sample['matched_text'],
                        voiceover_text=sample['vo_text'],
                        confidence=sample['confidence'],
                        attempt=retries
                    )
                    new_keywords.extend(remixed)
                except Exception as e:
                    logger.debug(f"Remix failed: {e}")
            
            # Remove duplicates
            new_keywords = list(set(new_keywords) - all_new_keywords - set(keywords))
            
            if not new_keywords:
                print(f"    No new keywords generated")
                break
            
            print(f"    Generated {len(new_keywords)} keywords: {new_keywords[:3]}...")
            all_new_keywords.update(new_keywords)
            
            # Download for new keywords
            stock_dir = Path(self.config.downloaded_videos_dir) / "stock"
            
            try:
                if PEXELS_AVAILABLE and ENHANCED_ENABLE_PEXELS:
                    pexels_paths, _ = download_pexels_footage(
                        new_keywords, str(stock_dir), per_keyword=2
                    )
                    print(f"    Pexels: +{len(pexels_paths)} videos")
                
                if PIXABAY_AVAILABLE and ENHANCED_ENABLE_PIXABAY:
                    pixabay_paths, _ = download_pixabay_footage(
                        new_keywords, str(stock_dir), per_keyword=2
                    )
                    print(f"    Pixabay: +{len(pixabay_paths)} videos")
            except Exception as e:
                print(f"    ⚠ Stock download error: {e}")
            
            # ═══════════════════════════════════════════════════════════════
            # OPTIMIZATION (v2.4): Delta-aware re-indexing
            # Only process NEWLY downloaded videos, not all videos
            # ═══════════════════════════════════════════════════════════════
            print(f"    Re-indexing (delta-aware - only new footage)...")
            retry_start = time.time()
            try:
                # force_reprocess=False enables delta-aware indexing
                self.stage_transcribe_index(force_reprocess=False)
                self.stage_match()
                retry_time = time.time() - retry_start
                print(f"    ✓ Re-indexed in {retry_time:.1f}s")
            except Exception as e:
                print(f"    ⚠ Re-match error: {e}")
                break
            
            # Recount
            low_conf = count_low_confidence()
            
            if not low_conf:
                print(f"\n  ✓ All matches now meet {ENHANCED_MIN_CONFIDENCE:.0%}!")
                break
            else:
                print(f"    Still {len(low_conf)} below threshold")
        
        final_low = len(low_conf)
        
        if low_conf:
            print(f"\n  ⚠ After {retries} retries: {final_low} matches still below {ENHANCED_MIN_CONFIDENCE:.0%}")
            print(f"    Consider reviewing these segments manually")
        
        return {
            'low_confidence_initial': initial_low,
            'low_confidence_final': final_low,
            'retries': retries,
            'new_keywords': len(all_new_keywords),
            'improved': initial_low - final_low
        }
    
    def stage_multi_style_otio(self) -> dict:
        """Generate additional OTIO styles"""
        if not self.second_style:
            return {'skipped': True}
        
        if not hasattr(self, 'matches') or not self.matches:
            return {'skipped': True, 'reason': 'no matches'}
        
        print(f"\n  Generating style 2: {self.second_style.name}...")
        
        try:
            from src.otio_builder import create_timeline, save_timeline
        except ImportError:
            print(f"  ⚠ Could not import otio_builder")
            return {'error': 'import failed'}
        
        output_dir = Path(self.config.otio_output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store original config values
        original_threshold = getattr(self.config.matching, 'confidence_threshold', 0.5)
        original_alternatives = getattr(self.config.output, 'num_alternatives', 2)
        
        try:
            # Apply style overrides
            if hasattr(self.second_style, 'confidence_threshold'):
                self.config.matching.confidence_threshold = self.second_style.confidence_threshold
            if hasattr(self.second_style, 'num_alternatives'):
                self.config.output.num_alternatives = self.second_style.num_alternatives
            
            # Create timeline
            voiceover_path = getattr(self.config, 'voiceover_path', None)
            timeline = create_timeline(self.matches, self.config, voiceover_path)
            
            # Save with style name
            style_name = self.second_style.name
            otio_path = output_dir / f"timeline_{style_name}_{timestamp}.otio"
            save_timeline(timeline, str(otio_path))
            print(f"  ✓ Style '{style_name}': {otio_path}")
            
            return {'file': str(otio_path), 'style': style_name}
            
        finally:
            # Restore original config
            self.config.matching.confidence_threshold = original_threshold
            self.config.output.num_alternatives = original_alternatives
    
    # =========================================================================
    # EXISTING STAGE METHODS
    # =========================================================================
    
    def stage_analyze_voiceover(self, voiceover_path: str) -> List[dict]:
        """Stage 1: Parse and analyze voiceover (supports SRT, audio, and video files)"""
        self._print_stage(1, "ANALYZE VOICEOVER")
        
        voiceover_path = Path(voiceover_path)
        
        if not voiceover_path.exists():
            logger.error(f"Voiceover file not found: {voiceover_path}")
            sys.exit(1)
        
        # Supported formats
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.opus'}
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v'}
        
        suffix = voiceover_path.suffix.lower()
        needs_transcription = suffix in audio_extensions or suffix in video_extensions
        
        if needs_transcription:
            file_type = "Video" if suffix in video_extensions else "Audio"
            print(f"  {file_type} file detected: {voiceover_path.name}")
            print(f"  Transcribing with faster-whisper (GPU)...")
            
            try:
                # Use optimized transcription if available
                if OPTIMIZED_TRANSCRIPTION:
                    from src.transcription_optimized import transcribe_voiceover_media
                else:
                    from src.transcription import transcribe_voiceover_media
                
                model_name = getattr(self.config.transcription, 'model', 'base')
                language = getattr(self.config.transcription, 'language', 'en')
                compute_type = getattr(self.config.transcription, 'compute_type', 'auto')
                
                srt_path = voiceover_path.with_suffix('.srt')
                
                if srt_path.exists():
                    print(f"  ✓ Using existing SRT: {srt_path.name}")
                else:
                    srt_path = transcribe_voiceover_media(
                        media_path=str(voiceover_path),
                        output_srt_path=str(srt_path),
                        model_name=model_name,
                        language=language,
                        compute_type=compute_type
                    )
                    print(f"  ✓ Transcribed to: {srt_path}")
                
                voiceover_path = Path(srt_path)
                
            except ImportError as e:
                logger.error(f"Cannot transcribe - missing dependency: {e}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to transcribe voiceover: {e}")
                sys.exit(1)
        
        # Parse SRT
        logger.info(f"Parsing: {voiceover_path}")
        segments = self._parse_srt(str(voiceover_path))
        
        # Calculate stats
        total_duration = sum(s['duration'] for s in segments)
        avg_duration = total_duration / len(segments) if segments else 0
        
        print(f"  ✓ Parsed {len(segments)} segments")
        print(f"  ✓ Total duration: {total_duration / 60:.1f} minutes")
        print(f"  ✓ Avg segment: {avg_duration:.1f} seconds")
        
        self.voiceover_segments = segments
        return segments
    
    def stage_extract_keywords(
        self,
        segments: List[dict],
        max_keywords: int = 50,
        review_enabled: bool = True
    ) -> List[str]:
        """Stage 1b: Extract search keywords and entities from voiceover"""
        print(f"\n  Extracting keywords (max {max_keywords})...")
        
        self.keyword_extractor = LLMKeywordExtractor(self.config)
        
        result = self.keyword_extractor.extract_keywords(
            segments,
            max_keywords=max_keywords,
            expand=True
        )
        
        keywords = result.keywords
        entities = getattr(result, 'entities', [])
        topic = getattr(result, 'topic', '')
        
        print(f"  ✓ Extracted {len(keywords)} keywords ({result.extraction_method})")
        if entities:
            print(f"  ✓ Found {len(entities)} named entities")
        
        self.entities = entities
        self.topic = topic
        
        # Store topic for enhanced features
        if topic and not self.topic_context:
            self.topic_context = topic
        
        # Show sample
        if keywords:
            print(f"\n  Sample keywords:")
            for kw in keywords[:10]:
                print(f"    • {kw}")
            if len(keywords) > 10:
                print(f"    ... and {len(keywords) - 10} more")
        
        if entities:
            print(f"\n  Named entities:")
            for ent in entities[:5]:
                ent_text = ent.get('text', ent.get('name', str(ent)))
                ent_type = ent.get('type', ent.get('label', ''))
                print(f"    • {ent_text} ({ent_type})")
        
        # Interactive keyword review
        if review_enabled and not getattr(self.config.pipeline, 'skip_keyword_review', False):
            try:
                from src.interactive import review_keywords
                
                project_dir = None
                if hasattr(self.config, 'cache_dir'):
                    project_dir = Path(self.config.cache_dir).parent
                
                print(f"\n  Launching keyword review...")
                keywords = review_keywords(
                    keywords=keywords,
                    entities=entities,
                    topic=topic,
                    project_dir=project_dir
                )
                print(f"  ✓ {len(keywords)} keywords after review")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Keyword review failed: {e}")
        
        self.keywords = keywords
        return keywords
    
    def stage_pre_run_summary(
        self,
        keywords: List[str],
        segments: List[dict]
    ) -> bool:
        """Show pre-run summary and get confirmation"""
        print(f"\n{'─' * 70}")
        print("  PRE-RUN SUMMARY")
        print(f"{'─' * 70}\n")
        
        self.downloader = VideoDownloader(self.config)
        
        ok, msg = self.downloader.check_dependencies()
        print(msg)
        if not ok:
            return False
        
        estimate = self.downloader.get_download_estimate(len(keywords))
        output_dir = Path(self.config.downloaded_videos_dir)
        inventory = self.downloader.get_inventory_report(output_dir)
        
        print(f"\n  Voiceover:")
        print(f"    • {len(segments)} segments")
        print(f"    • {sum(s['duration'] for s in segments) / 60:.1f} minutes total")
        
        print(f"\n  Keywords:")
        print(f"    • {len(keywords)} search terms")
        
        print(f"\n  Download Plan:")
        for tier, desc in estimate['tiers'].items():
            print(f"    • {tier}: {desc}")
        print(f"    • Total: ~{estimate['total_videos']} videos (YouTube)")
        
        # Show stock footage estimate
        if self.enhanced_enabled:
            stock_estimate = len(keywords[:15]) * ENHANCED_STOCK_PER_KEYWORD * 2  # Pexels + Pixabay
            print(f"    • Stock (Pexels/Pixabay): ~{stock_estimate} additional")
        
        print(f"    • Est. storage: {estimate['est_storage_gb']} GB")
        print(f"    • Est. time: {estimate['est_time_minutes']:.0f} minutes")
        
        if inventory['total_videos'] > 0:
            print(f"\n  Existing Footage:")
            print(f"    • {inventory['total_videos']} videos ({inventory['total_duration_hours']:.1f} hours)")
        
        print()
        
        if self.config.pipeline.confirm_before_download:
            return self._get_user_confirmation("Proceed with download?", default=True)
        return True
    
    def stage_download(
        self,
        keywords: List[str],
        resume: bool = False
    ) -> List[dict]:
        """Stage 2: Download footage from YouTube"""
        self._print_stage(2, "DOWNLOAD FOOTAGE")
        
        if not self.downloader:
            self.downloader = VideoDownloader(self.config)
        
        output_dir = Path(self.config.downloaded_videos_dir)
        
        downloaded, failed = self.downloader.download_all(
            keywords=keywords,
            output_dir=output_dir,
            max_concurrent=self.config.download.max_concurrent,
            resume=resume
        )
        
        print(f"\n  ✓ Downloaded {len(downloaded)} videos")
        if failed:
            print(f"  ⚠ {len(failed)} keywords with no results")
            self.failed_keywords.extend(failed)
            
            failed_file = output_dir / "no_results_keywords.txt"
            with open(failed_file, 'w') as f:
                f.write(f"# Keywords with 0 results - {datetime.now().isoformat()}\n")
                for kw in failed:
                    f.write(f"{kw}\n")
        
        self.downloaded_videos = [d.to_dict() for d in downloaded]
        return self.downloaded_videos
    
    def stage_deduplicate(self) -> dict:
        """Stage 2b: Deduplicate downloaded footage"""
        self._print_stage("2b", "DEDUPLICATE FOOTAGE")
        
        try:
            from src.deduplication import VideoDeduplicator
        except ImportError as e:
            print("  ⚠ Deduplication not available")
            return {'duplicates_deleted': 0}
        
        video_dir = Path(self.config.downloaded_videos_dir)
        deduplicator = VideoDeduplicator(self.config)
        
        if not deduplicator.is_available():
            print("  ⚠ imagehash not installed, skipping")
            return {'duplicates_deleted': 0}
        
        report_path = video_dir / "deduplication_report.json"
        report = deduplicator.deduplicate(
            video_dir=str(video_dir),
            auto_delete=True,
            report_path=str(report_path)
        )
        
        print(f"\n  Scanned: {report.total_videos} videos")
        print(f"  ✓ Unique: {report.unique_videos}")
        
        if report.duplicates_deleted > 0:
            print(f"  ✓ Deleted: {report.duplicates_deleted} duplicates")
            print(f"  ✓ Saved: {report.space_saved_mb:.1f} MB")
        
        return {
            'total_videos': report.total_videos,
            'duplicates_deleted': report.duplicates_deleted
        }
    
    def stage_transcribe_index(self, force_reprocess: bool = False) -> dict:
        """
        Stage 3: Transcribe and index videos (OPTIMIZED v2.4)
        
        OPTIMIZATIONS:
        1. Delta-aware indexing - only processes NEW videos
        2. Parallel transcription with ThreadPoolExecutor
        3. Batch embedding API calls (100 texts per call)
        4. Selective vision processing (skip if transcript covers content)
        
        Args:
            force_reprocess: If True, reprocess all videos (ignore cache)
        """
        self._print_stage(3, "TRANSCRIBE & INDEX")
        
        stage_start = time.time()
        
        # Import modules based on optimization availability
        try:
            if OPTIMIZED_TRANSCRIPTION:
                from src.transcription_optimized import transcribe_videos_parallel
            else:
                from src.transcription import transcribe_videos_parallel
            
            if OPTIMIZED_EMBEDDINGS:
                from src.embeddings_optimized import compute_embeddings, build_embedding_index, get_embedding_provider
            else:
                from src.embeddings import compute_embeddings, build_embedding_index, get_embedding_provider
            
            if OPTIMIZED_VISION:
                from src.vision_optimized import process_video_vision
            else:
                from src.vision import process_video_vision
            
            from src.utils import CacheManager, SRTSegment
        except ImportError as e:
            logger.warning(f"Could not import modules: {e}")
            return {'videos_indexed': 0}
        
        video_dir = Path(self.config.downloaded_videos_dir)
        
        # Find all video files
        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.mxf'}
        videos = []
        for ext in video_extensions:
            videos.extend(video_dir.rglob(f'*{ext}'))
        
        video_paths = [str(v) for v in videos]
        print(f"  Found {len(video_paths)} videos")
        
        # Initialize cache
        cache = CacheManager(self.config.cache_dir)
        self.cache = cache
        
        # =====================================================================
        # OPTIMIZATION 1: Delta-aware transcription
        # Only transcribe NEW videos, load existing from cache
        # =====================================================================
        transcribe_start = time.time()
        
        if OPTIMIZED_TRANSCRIPTION:
            print(f"  Using optimized parallel transcription ({PARALLEL_WORKERS} workers)...")
            print(f"  Delta-aware: {'disabled' if force_reprocess else 'enabled'}")
            transcripts = transcribe_videos_parallel(
                video_paths, 
                cache, 
                self.config,
                max_workers=PARALLEL_WORKERS,
                force_reprocess=force_reprocess
            )
        else:
            print(f"  Transcribing with {self.config.transcription.provider}...")
            transcripts = transcribe_videos_parallel(video_paths, cache, self.config)
        
        transcribe_time = time.time() - transcribe_start
        print(f"  ✓ Transcribed {len(transcripts)} videos in {transcribe_time:.1f}s")
        
        # =====================================================================
        # OPTIMIZATION 2: Selective vision processing
        # Only process scenes where transcript is sparse
        # =====================================================================
        vision_time = 0
        if self.config.vision.enabled:
            vision_start = time.time()
            
            if OPTIMIZED_VISION:
                print(f"  Selective vision processing (max {MAX_VISION_SCENES} scenes/video)...")
            else:
                print(f"  Describing scenes with vision API...")
            
            all_scenes = {}
            skipped_videos = 0
            
            for video_path in videos:
                video_str = str(video_path)
                video_transcripts = transcripts.get(video_str, [])
                
                # Convert to SRTSegment if needed
                segments = []
                for t in video_transcripts:
                    if hasattr(t, 'start_time'):
                        segments.append(t)
                    elif isinstance(t, dict):
                        segments.append(SRTSegment(
                            index=t.get('index', 0),
                            start_time=t.get('start_time', 0),
                            end_time=t.get('end_time', 0),
                            text=t.get('text', '')
                        ))
                
                # Process with selective vision
                if OPTIMIZED_VISION:
                    scenes = process_video_vision(
                        video_str, 
                        segments, 
                        cache, 
                        self.config,
                        max_scenes_per_video=MAX_VISION_SCENES
                    )
                else:
                    scenes = process_video_vision(video_str, segments, cache, self.config)
                
                if scenes:
                    all_scenes[video_str] = scenes
                else:
                    skipped_videos += 1
            
            vision_time = time.time() - vision_start
            total_scenes = sum(len(s) for s in all_scenes.values())
            
            if OPTIMIZED_VISION:
                print(f"  ✓ Vision: {total_scenes} scenes, {skipped_videos} videos skipped (good transcript)")
            else:
                print(f"  ✓ Described {total_scenes} scenes across {len(all_scenes)} videos")
            print(f"  ✓ Vision time: {vision_time:.1f}s")
        
        # =====================================================================
        # OPTIMIZATION 3: Batch embedding computation
        # Groups into batches of 100 for API efficiency
        # =====================================================================
        embed_start = time.time()
        
        # Extract texts for embedding
        all_texts = []
        text_metadata = []
        for video_path, segments in transcripts.items():
            for seg in segments:
                if isinstance(seg, dict):
                    text = seg.get('text', '')
                else:
                    text = getattr(seg, 'text', '')
                if text and text.strip():
                    all_texts.append(text)
                    text_metadata.append({'video': video_path, 'segment': seg})
        
        print(f"  Computing embeddings ({self.config.embedding.provider})...")
        print(f"    {len(all_texts)} text segments")
        
        if OPTIMIZED_EMBEDDINGS:
            print(f"    Using batch embedding (batch size {EMBEDDING_BATCH_SIZE})")
            embedding_provider = get_embedding_provider(self.config)
            embeddings = compute_embeddings(
                texts=all_texts,
                provider=embedding_provider,
                cache=cache,
                cache_key="video_segments",
                show_progress=True
            )
        else:
            embedding_provider = get_embedding_provider(self.config)
            embeddings = compute_embeddings(
                texts=all_texts,
                provider=embedding_provider,
                cache=cache,
                cache_key="video_segments"
            )
        
        embed_time = time.time() - embed_start
        print(f"  ✓ Computed {len(embeddings)} embeddings in {embed_time:.1f}s")
        
        # Build FAISS index
        if self.config.indexing.use_faiss:
            print(f"  Building FAISS index...")
            if OPTIMIZED_EMBEDDINGS:
                self.embedding_index = build_embedding_index(embeddings, self.config)
            else:
                self.embedding_index = build_embedding_index(embeddings, self.config)
            print(f"  ✓ Built index")
        
        # Store for matching
        self.transcripts = transcripts
        self.embeddings = embeddings
        self.text_metadata = text_metadata
        
        # Performance summary
        total_time = time.time() - stage_start
        self.stage_timings['stage3'] = {
            'transcribe': transcribe_time,
            'vision': vision_time,
            'embed': embed_time,
            'total': total_time
        }
        
        print(f"\n  ═══ Stage 3 Performance ═══")
        print(f"    Transcription: {transcribe_time:.1f}s")
        if self.config.vision.enabled:
            print(f"    Vision: {vision_time:.1f}s")
        print(f"    Embeddings: {embed_time:.1f}s")
        print(f"    Total: {total_time:.1f}s")
        
        return {
            'videos_indexed': len(videos),
            'embeddings': len(embeddings),
            'transcribe_time': transcribe_time,
            'embed_time': embed_time,
            'total_time': total_time
        }
    
    def stage_scene_detection(self) -> dict:
        """Stage 3b: Scene detection with audio analysis"""
        self._print_stage("3b", "SCENE DETECTION")
        
        try:
            from src.scene_detection import SceneDetector
        except ImportError as e:
            print("  ⚠ Scene detection module not available")
            return {'scenes': 0}
        
        video_dir = Path(self.config.downloaded_videos_dir)
        
        print(f"  Preset: {self.config.scene_detection.preset}")
        force_cuda = getattr(self.config.scene_detection, 'force_cuda', False)
        print(f"  CUDA: {'Forced' if force_cuda else 'Auto'}")
        print(f"  Timecode: {self.config.scene_detection.start_timecode}")
        
        detector = SceneDetector(self.config)
        
        min_duration = self.config.scene_detection.min_video_duration
        results = detector.process_all_videos(video_dir, min_duration=min_duration)
        
        stats = detector.get_stats()
        
        print(f"\n  ✓ Processed {stats['videos_processed']} videos")
        print(f"  ✓ Detected {stats['total_scenes']} total scenes")
        print(f"  ✓ Average: {stats['avg_scenes_per_video']} scenes/video")
        
        if stats.get('audio_analysis_enabled'):
            print(f"  ✓ Audio analysis: {stats['videos_with_speech']} videos with speech")
            print(f"  ✓ Audio cut points: {stats['total_audio_cut_points']}")
        
        if self.config.scene_detection.output_otio:
            print(f"  ✓ OTIO files: {detector.otio_output_dir}")
        if self.config.scene_detection.output_scene_index:
            print(f"  ✓ Scene index: {detector.scene_index_path}")
        
        self.scene_detector = detector
        
        return {
            'videos_processed': stats['videos_processed'],
            'total_scenes': stats['total_scenes']
        }
    
    def stage_match(self) -> dict:
        """Stage 4: Match footage to voiceover"""
        self._print_stage(4, "MATCH FOOTAGE")
        
        try:
            from src.matching import match_all_segments
            
            if OPTIMIZED_EMBEDDINGS:
                from src.embeddings_optimized import get_embedding_provider, compute_embeddings
            else:
                from src.embeddings import get_embedding_provider, compute_embeddings
            
            from src.utils import SRTSegment
        except ImportError as e:
            logger.warning(f"Could not import modules: {e}")
            return {'matches': 0}
        
        if not self.voiceover_segments:
            print("  ⚠ No voiceover segments (run stage 1 first)")
            return {'matches': 0}
        
        if not self.embeddings:
            print("  ⚠ No video embeddings (run stage 3 first)")
            return {'matches': 0}
        
        # Convert voiceover segments
        vo_segments = []
        for seg in self.voiceover_segments:
            if isinstance(seg, dict):
                text = seg.get('text', '').strip()
                if text:
                    vo_segments.append(SRTSegment(
                        index=seg.get('index', 0),
                        start_time=seg.get('start_time', 0),
                        end_time=seg.get('end_time', 0),
                        text=text
                    ))
            else:
                if seg.text and seg.text.strip():
                    vo_segments.append(seg)
        
        print(f"  Matching {len(vo_segments)} voiceover segments...")
        
        # Compute voiceover embeddings
        print(f"  Computing voiceover embeddings...")
        vo_texts = [seg.text for seg in vo_segments]
        embedding_provider = get_embedding_provider(self.config)
        
        if OPTIMIZED_EMBEDDINGS:
            vo_embeddings = compute_embeddings(
                texts=vo_texts,
                provider=embedding_provider,
                cache=self.cache,
                cache_key="voiceover_segments",
                show_progress=False
            )
        else:
            vo_embeddings = compute_embeddings(
                texts=vo_texts,
                provider=embedding_provider,
                cache=self.cache,
                cache_key="voiceover_segments"
            )
        print(f"  ✓ {len(vo_embeddings)} voiceover embeddings")
        
        # Convert video metadata
        video_segments = []
        for meta in self.text_metadata:
            seg = meta['segment']
            if isinstance(seg, dict):
                video_segments.append(SRTSegment(
                    index=seg.get('index', 0),
                    start_time=seg.get('start_time', 0),
                    end_time=seg.get('end_time', 0),
                    text=seg.get('text', ''),
                    source_file=meta.get('video', '')
                ))
            else:
                if not hasattr(seg, 'source_file') or not seg.source_file:
                    seg.source_file = meta.get('video', '')
                video_segments.append(seg)
        
        # Get scenes
        scenes = None
        if hasattr(self, 'scene_detector') and self.scene_detector:
            try:
                scenes = self.scene_detector.get_all_scenes()
            except:
                pass
        
        # Run matching
        print(f"  Running two-stage matching...")
        matches = match_all_segments(
            voiceover_segments=vo_segments,
            video_segments=video_segments,
            voiceover_embeddings=vo_embeddings,
            video_embeddings=self.embeddings,
            scenes=scenes,
            config=self.config,
            cache=self.cache,
            embedding_index=self.embedding_index
        )
        
        # Count stats
        matched = sum(1 for m in matches if hasattr(m, 'primary_match') and m.primary_match.confidence > 0.5)
        high_conf = sum(1 for m in matches if hasattr(m, 'primary_match') and m.primary_match.confidence >= ENHANCED_MIN_CONFIDENCE)
        
        print(f"  ✓ Matched {matched}/{len(vo_segments)} segments (>50% confidence)")
        print(f"  ✓ High confidence (≥{ENHANCED_MIN_CONFIDENCE:.0%}): {high_conf}/{len(vo_segments)}")
        
        self.matches = matches
        
        return {
            'matches': len(matches),
            'high_confidence': high_conf,
            'above_50': matched
        }
    
    def stage_output(self) -> dict:
        """Stage 5: Generate output files"""
        self._print_stage(5, "GENERATE OUTPUT")
        
        try:
            from src.otio_builder import (
                create_timeline, 
                save_timeline, 
                save_timeline_as_edl, 
                save_timeline_as_resolve_xml,
                generate_match_report
            )
        except ImportError as e:
            logger.warning(f"Could not import modules: {e}")
            return {'files': []}
        
        if not hasattr(self, 'matches') or not self.matches:
            print("  ⚠ No matches to export (run stage 4 first)")
            return {'files': []}
        
        output_dir = Path(self.config.otio_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = []
        
        voiceover_path = getattr(self.config, 'voiceover_path', None)
        timeline = create_timeline(self.matches, self.config, voiceover_path)
        
        # Save OTIO
        otio_path = output_dir / f"timeline_{timestamp}.otio"
        save_timeline(timeline, str(otio_path))
        print(f"  ✓ OTIO: {otio_path}")
        files.append(str(otio_path))
        
        # Export EDL
        edl_path = output_dir / f"timeline_{timestamp}.edl"
        save_timeline_as_edl(self.matches, str(edl_path))
        print(f"  ✓ EDL: {edl_path}")
        files.append(str(edl_path))
        
        # Export DaVinci XML
        xml_path = output_dir / f"timeline_{timestamp}.xml"
        save_timeline_as_resolve_xml(self.matches, str(xml_path), voiceover_path=voiceover_path)
        print(f"  ✓ DaVinci XML: {xml_path}")
        files.append(str(xml_path))
        
        # Generate match report
        report_path = output_dir / f"match_report_{timestamp}.md"
        generate_match_report(self.matches, str(report_path), self.config)
        print(f"  ✓ Match Report: {report_path}")
        files.append(str(report_path))
        
        return {'files': files}
    
    # =========================================================================
    # MAIN RUN METHOD
    # =========================================================================
    
    def run(
        self,
        voiceover_path: str,
        max_keywords: int = 50,
        resume: bool = False,
        skip_download: bool = False,
        skip_transcribe: bool = False,
        skip_scenes: bool = False,
        skip_match: bool = False,
        download_only: bool = False,
        match_only: bool = False
    ):
        """Run the full pipeline with enhanced features and performance optimizations"""
        self._print_banner()
        
        pipeline_start = time.time()
        
        # Stage 1: Analyze voiceover
        segments = self.stage_analyze_voiceover(voiceover_path)
        
        # Stage 1b: Extract keywords
        if not match_only:
            keywords = self.stage_extract_keywords(segments, max_keywords)
        else:
            keywords = []
            print("\n  Skipping keyword extraction (--match-only)")
        
        if match_only:
            skip_download = True
        
        # ═══════════════════════════════════════════════════════════════════
        # ENHANCED: Prompt for enhanced features (before download)
        # ═══════════════════════════════════════════════════════════════════
        if not skip_download and not match_only:
            self.prompt_enhanced_features()
        
        # Pre-run summary
        if not skip_download:
            if not self.stage_pre_run_summary(keywords, segments):
                print("\nAborted by user.")
                return
        
        # Stage 2: Download from YouTube
        if not skip_download:
            self.stage_download(keywords, resume=resume)
            
            # Stage 2b: Deduplicate
            self.stage_deduplicate()
            
            # ═══════════════════════════════════════════════════════════════
            # ENHANCED: Download stock footage from Pexels/Pixabay
            # ═══════════════════════════════════════════════════════════════
            if self.enhanced_enabled:
                self.stage_download_stock(keywords)
                
                # Remix zero-download keywords and retry
                if self.failed_keywords and self.keyword_remixer:
                    keywords = self.stage_remix_zero_downloads(keywords)
            
            if download_only:
                print("\n" + "=" * 70)
                print("  Download complete. Run with --match-only to continue.")
                print("=" * 70)
                return
        
        # Stage 3: Transcribe & Index
        if not skip_transcribe:
            self.stage_transcribe_index(force_reprocess=False)
        
        # Stage 3b: Scene Detection
        if not skip_transcribe and not skip_scenes and self.config.scene_detection.enabled:
            self.stage_scene_detection()
        
        # Stage 4: Match
        if not skip_match:
            self.stage_match()
            
            # ═══════════════════════════════════════════════════════════════
            # ENHANCED: Confidence enforcement (90% minimum)
            # ═══════════════════════════════════════════════════════════════
            if self.enhanced_enabled and self.keyword_remixer:
                enforcement_result = self.stage_confidence_enforcement(keywords)
                
                if enforcement_result.get('improved', 0) > 0:
                    print(f"  ✓ Improved {enforcement_result['improved']} matches via keyword remix")
        
        # Stage 5: Output
        self.stage_output()
        
        # ═══════════════════════════════════════════════════════════════════
        # ENHANCED: Generate additional OTIO styles
        # ═══════════════════════════════════════════════════════════════════
        if self.second_style:
            self.stage_multi_style_otio()
        
        # Stage 6: Clip Grading (optional)
        if not skip_match and self.matches:
            grading_config = getattr(self.config, 'clip_grading', None)
            if (grading_config and 
                getattr(grading_config, 'enabled', False) and
                getattr(grading_config, 'prompt_after_match', True)):
                
                try:
                    from src.interactive import grade_clips_after_match
                    
                    project_dir = None
                    if hasattr(self.config, 'cache_dir'):
                        project_dir = Path(self.config.cache_dir).parent
                    
                    print("\n" + "─" * 70)
                    print("  CLIP GRADING (Optional)")
                    print("─" * 70)
                    
                    allow_skip = getattr(grading_config, 'allow_skip', True)
                    grader = grade_clips_after_match(
                        self.matches,
                        project_dir=project_dir,
                        allow_skip=allow_skip
                    )
                    
                    # Merge into global cache
                    cross_cache = getattr(self.config, 'cross_project_cache', None)
                    if (cross_cache and 
                        getattr(cross_cache, 'enabled', False) and
                        getattr(cross_cache, 'share_clip_grades', True)):
                        try:
                            from src.interactive import GlobalCache
                            global_cache = GlobalCache(
                                global_cache_dir=getattr(cross_cache, 'global_cache_dir', None),
                                install_dir=str(Path(__file__).parent)
                            )
                            global_cache.merge_project_grades(str(grader.grades_file))
                        except Exception as e:
                            logger.warning(f"Could not sync grades: {e}")
                            
                except ImportError:
                    pass
                except Exception as e:
                    logger.warning(f"Clip grading failed: {e}")
        
        # Finalize logging
        if self.run_logger:
            try:
                self.run_logger.run_log.voiceover_segments = len(self.voiceover_segments)
                self.run_logger.run_log.video_files = len(self.transcripts)
                self.run_logger.run_log.video_segments = len(self.embeddings)
                
                for i, match in enumerate(self.matches):
                    if hasattr(match, 'primary_match'):
                        pm = match.primary_match
                        self.run_logger.log_match_decision(
                            segment_index=i,
                            voiceover_text=pm.voiceover_segment.text[:100] if pm.voiceover_segment else "",
                            selected_clip=Path(pm.video_segment.source_file).stem if pm.video_segment else "",
                            selected_source=pm.video_segment.source_file if pm.video_segment else "",
                            confidence=pm.confidence,
                            reasoning=pm.reasoning,
                            embedding_similarity=pm.embedding_similarity,
                            is_hybrid_match=pm.is_visual_match
                        )
                
                self.run_logger.finalize()
                print(f"  Log saved: {self.run_logger.log_file}")
            except Exception as e:
                print(f"  ⚠ Could not save log: {e}")
        
        # Pipeline complete
        pipeline_time = time.time() - pipeline_start
        
        # Final summary
        print("\n" + "=" * 70)
        print("  ✓ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Output: {self.config.otio_output_dir}")
        print(f"  Videos: {self.config.downloaded_videos_dir}")
        if self.downloader:
            print(f"  Sources: {self.downloader.sources_file}")
        if self.scene_detector:
            stats = self.scene_detector.get_stats()
            print(f"  Scenes: {stats['total_scenes']} across {stats['videos_processed']} videos")
        
        # Enhanced features summary
        if self.enhanced_enabled:
            high_conf = sum(1 for m in self.matches 
                          if hasattr(m, 'primary_match') and m.primary_match.confidence >= ENHANCED_MIN_CONFIDENCE)
            print(f"  High confidence (≥{ENHANCED_MIN_CONFIDENCE:.0%}): {high_conf}/{len(self.matches)}")
        
        # Performance summary (v2.4)
        print(f"\n  ═══ Performance Summary ═══")
        print(f"    Total pipeline time: {pipeline_time:.1f}s ({pipeline_time/60:.1f} min)")
        if 'stage3' in self.stage_timings:
            s3 = self.stage_timings['stage3']
            print(f"    Stage 3 breakdown:")
            print(f"      Transcription: {s3['transcribe']:.1f}s")
            if s3['vision'] > 0:
                print(f"      Vision: {s3['vision']:.1f}s")
            print(f"      Embeddings: {s3['embed']:.1f}s")
        
        if self.run_logger:
            print(f"  Log: {self.run_logger.log_file}")
        print("=" * 70)


def main():
    global PROJECT_DIR
    
    parser = argparse.ArgumentParser(
        description='Voiceover-Matcher: End-to-End Documentary Pipeline v2.4 (Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --voiceover script.srt
  python main.py --voiceover script.srt --keywords 30
  python main.py --project "E:\\Projects\\MyDoc" -v voiceover/script.srt
  python main.py --resume
  python main.py --match-only

Performance Optimizations (v2.4):
  - Parallel transcription (4 workers)
  - Delta-aware indexing (only new files on retry)
  - Batch embedding API calls (100/batch)
  - Selective vision processing

Enhanced Features:
  - 90% confidence enforcement with keyword remix
  - Pexels/Pixabay stock footage download
  - Multi-style OTIO generation
        """
    )
    
    parser.add_argument('--project', '-p', help='Project directory', default=None)
    parser.add_argument('--voiceover', '-v', help='Path to voiceover file', default=None)
    parser.add_argument('--config', '-c', help='Path to config.yaml', default=None)
    parser.add_argument('--keywords', '-k', type=int, help='Maximum keywords', default=None)
    parser.add_argument('--resume', action='store_true', help='Resume interrupted download')
    parser.add_argument('--skip-download', action='store_true', help='Skip download stage')
    parser.add_argument('--skip-transcribe', action='store_true', help='Skip transcription')
    parser.add_argument('--skip-match', action='store_true', help='Skip matching')
    parser.add_argument('--skip-scenes', action='store_true', help='Skip scene detection')
    parser.add_argument('--download-only', action='store_true', help='Only download')
    parser.add_argument('--match-only', action='store_true', help='Skip download, just match')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip prompts')
    
    args = parser.parse_args()
    
    # Determine project directory
    if args.project:
        PROJECT_DIR = strip_extended_path_prefix(Path(args.project).resolve())
        if str(PROJECT_DIR).endswith(('\\', '/')):
            PROJECT_DIR = PROJECT_DIR.parent / PROJECT_DIR.name
        
        print(f"\n📁 Project Mode")
        print(f"   Project: {PROJECT_DIR}")
        print(f"   Install: {INSTALL_DIR}")
        
        if not PROJECT_DIR.exists():
            print(f"\n⚠ Project directory does not exist: {PROJECT_DIR}")
            sys.exit(1)
    else:
        PROJECT_DIR = Path.cwd()
    
    # Load environment
    load_environment(PROJECT_DIR if args.project else None)
    
    # Load config
    if args.project:
        global_config_path = Path(args.config) if args.config else INSTALL_DIR / 'config.yaml'
        config = load_project_config(PROJECT_DIR, global_config_path)
    else:
        config_path = args.config if args.config else 'config.yaml'
        config = load_config(config_path)
    
    if args.yes:
        config.pipeline.confirm_before_download = False
    
    # Import interactive module
    try:
        from src.interactive import (
            select_voiceover_file, 
            review_keywords, 
            prompt_face_preference,
            grade_clips_after_match
        )
        has_interactive = True
    except ImportError:
        has_interactive = False
    
    # Get voiceover path
    voiceover_path = args.voiceover or getattr(config, 'voiceover_path', None)
    
    if args.project and voiceover_path and not Path(voiceover_path).is_absolute():
        voiceover_path = str(PROJECT_DIR / voiceover_path)
    
    if not voiceover_path or not Path(voiceover_path).exists():
        if has_interactive and not args.yes:
            voiceover_path = select_voiceover_file(
                project_dir=PROJECT_DIR if args.project else None
            )
            
            if not voiceover_path:
                print("No voiceover file selected. Exiting.")
                sys.exit(0)
        else:
            print("\nVoiceover file not specified.")
            voiceover_path = input("Enter path to voiceover file: ").strip()
            
            if args.project and voiceover_path and not Path(voiceover_path).is_absolute():
                voiceover_path = str(PROJECT_DIR / voiceover_path)
        
        if not voiceover_path or not Path(voiceover_path).exists():
            print(f"Error: Voiceover file not found: {voiceover_path}")
            sys.exit(1)
    
    # Face detection preference
    face_preference = getattr(config, 'face_detection', None)
    if (face_preference and 
        getattr(face_preference, 'enabled', False) and 
        getattr(face_preference, 'prompt_per_run', False) and
        has_interactive and not args.yes):
        
        face_pref = prompt_face_preference()
        config.face_detection.current_preference = face_pref
    
    # Get max keywords
    max_keywords = args.keywords
    if max_keywords is None and not args.match_only:
        print(f"\nKeyword extraction (default: {config.pipeline.max_keywords})")
        user_input = input(f"How many keywords to extract? [{config.pipeline.max_keywords}]: ").strip()
        if user_input:
            try:
                max_keywords = int(user_input)
            except ValueError:
                max_keywords = config.pipeline.max_keywords
        else:
            max_keywords = config.pipeline.max_keywords
    elif max_keywords is None:
        max_keywords = config.pipeline.max_keywords
    
    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run(
        voiceover_path=voiceover_path,
        max_keywords=max_keywords,
        resume=args.resume,
        skip_download=args.skip_download,
        skip_transcribe=args.skip_transcribe,
        skip_scenes=args.skip_scenes,
        skip_match=args.skip_match,
        download_only=args.download_only,
        match_only=args.match_only
    )


if __name__ == '__main__':
    main()
