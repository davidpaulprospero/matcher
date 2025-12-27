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
    python main.py --resume  # Resume interrupted run
    python main.py --match-only  # Skip download, just match existing footage
    python main.py --config custom_config.yaml  # Use custom config file
"""

import os
import sys
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
    
    # Update cache dir
    if config.cache.cache_dir and not Path(config.cache.cache_dir).is_absolute():
        config.cache.cache_dir = str(project_dir / config.cache.cache_dir)
    
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

def setup_logging(config: Config) -> logging.Logger:
    """Setup logging based on config"""
    log_level = getattr(logging, config.logging.log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    return logging.getLogger(__name__)


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
        self.keyword_remixer = None
        self.enhanced_enabled = config.enhanced.enabled
        self.second_style = None
        self.failed_keywords = []
        
        # Performance tracking
        self.stage_timings = {}
        self.use_delta_indexing = True
        
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
        
        # Determine file type and parse/transcribe
        vo_path = Path(voiceover_path)
        if vo_path.suffix.lower() == '.srt':
            print(f"  Parsing SRT: {vo_path.name}")
            self.voiceover_segments = self._parse_srt(str(vo_path))
        else:
            # Transcribe audio/video
            if self.modules['optimized_transcription']:
                from src.transcription import transcribe_voiceover_media
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
                
                # Now parse the generated SRT file
                print(f"  ✓ Transcription saved to: {Path(result_srt).name}")
                self.voiceover_segments = self._parse_srt(result_srt)
            else:
                logger.error("Transcription module not available for non-SRT files")
                sys.exit(1)
        
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
        
        return self.keywords
    
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
    
    def stage_download(self, keywords: List[str]) -> List[dict]:
        """
        Stage 2: Download footage from YouTube.
        Duration tiers and settings from config.
        Includes zero-download keyword remix for failed keywords.
        """
        config = self.config
        
        if config.pipeline.skip_download:
            print("  ⏭ Skipping download (config: skip_download=true)")
            return []
        
        self._print_stage("2", "DOWNLOAD FOOTAGE")
        
        try:
            from src.downloader import VideoDownloader, DownloadCheckpoint
            
            # Initialize downloader with config
            self.downloader = VideoDownloader(config=config)
            
            # Get duration tier settings from config
            tiers = config.duration_tiers
            
            print(f"  Duration tiers:")
            print(f"    • short:  {tiers.short.min_seconds}-{tiers.short.max_seconds}s ({tiers.short.videos_per_keyword}/kw)")
            print(f"    • medium: {tiers.medium.min_seconds}-{tiers.medium.max_seconds}s ({tiers.medium.videos_per_keyword}/kw)")
            print(f"    • long:   {tiers.long.min_seconds}-{tiers.long.max_seconds}s ({tiers.long.videos_per_keyword}/kw)")
            print(f"    • longer: {tiers.longer.min_seconds}-{tiers.longer.max_seconds}s ({tiers.longer.videos_per_keyword}/kw)")
            
            # Download using download_all() method
            output_dir = Path(config.downloaded_videos_dir)
            downloaded_videos, failed = self.downloader.download_all(
                keywords=keywords,
                output_dir=output_dir,
                resume=True,
                topic=self.topic_context or ""
            )
            self.downloaded_videos = downloaded_videos
            
            # Store failed keywords for potential remix
            self.failed_keywords = failed
            
            if failed:
                print(f"  ⚠ Failed keywords: {', '.join(failed[:5])}" + 
                      (f" (+{len(failed)-5} more)" if len(failed) > 5 else ""))
            
            print(f"\n  ✓ Downloaded {len(self.downloaded_videos)} videos")
            
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
            from src.keyword_remix import remix_downloaded_videos, RemixConfig, save_remix_report
            
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
            
            # Get video directory
            video_dir = Path(config.downloaded_videos_dir)
            
            if not video_dir.exists():
                print(f"  ⚠ Video directory not found: {video_dir}")
                return []
            
            # Run remix
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
    
    def stage_transcribe(self) -> dict:
        """
        Stage 3: Transcribe and index videos.
        Parallel processing settings from config.
        Uses remixed video paths if available from stage_remix.
        """
        config = self.config
        
        if config.pipeline.skip_transcription:
            print("  ⏭ Skipping transcription (config: skip_transcription=true)")
            return {}
        
        self._print_stage("3", "TRANSCRIBE & INDEX")
        
        # Check if we have remixed video paths from stage_remix
        if hasattr(self, 'remixed_video_paths') and self.remixed_video_paths:
            # Use filtered videos from remix stage
            video_files = [Path(p) for p in self.remixed_video_paths]
            print(f"  Using {len(video_files)} videos from remix stage")
        else:
            # Fall back to scanning directory
            videos_dir = Path(config.downloaded_videos_dir)
            video_files = list(videos_dir.rglob('*.mp4')) + list(videos_dir.rglob('*.webm'))
        
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
        
        return self.transcripts
    
    def stage_match(self) -> List[dict]:
        """
        Stage 4: Match voiceover to footage.
        All matching parameters from config.
        """
        config = self.config
        
        if config.pipeline.skip_matching:
            print("  ⏭ Skipping matching (config: skip_matching=true)")
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
            from src.utils import SRTSegment, CacheManager
            from src.embeddings import compute_embeddings
            
            print(f"  Matching settings (from config):")
            print(f"    • Min confidence: {config.matching.min_confidence}")
            print(f"    • High confidence threshold: {config.matching.high_confidence_threshold}")
            print(f"    • Embedding candidates: {config.matching.embedding_candidates}")
            print(f"    • LLM rerank candidates: {config.matching.llm_rerank_candidates}")
            print(f"    • Max clip reuse: {config.matching.max_clip_reuse}")
            
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
                        entities=seg.get('entities', [])
                    )
                else:
                    vo_segment = seg
                vo_segments.append(vo_segment)
            
            # Convert video metadata to SRTSegment objects  
            print(f"  Preparing video segments...")
            video_segments = []
            for i, meta in enumerate(self.text_metadata):
                if isinstance(meta, dict):
                    vid_segment = SRTSegment(
                        index=i,
                        start_time=meta.get('start_time', 0),
                        end_time=meta.get('end_time', 0),
                        text=meta.get('text', ''),
                        source_file=meta.get('video_path', ''),
                    )
                else:
                    vid_segment = meta
                video_segments.append(vid_segment)
            
            # Compute voiceover embeddings
            print(f"  Computing voiceover embeddings...")
            vo_texts = [seg.text for seg in vo_segments]
            
            # Get embedding provider and cache
            from src.embeddings import get_embedding_provider
            cache_dir = config.cache.cache_dir if hasattr(config.cache, 'cache_dir') else ".cache"
            provider = get_embedding_provider(config)
            cache = CacheManager(cache_dir)
            
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
            self.matches = match_all_segments(
                voiceover_segments=vo_segments,
                video_segments=video_segments,
                voiceover_embeddings=vo_embeddings,
                video_embeddings=self.embeddings,
                scenes=getattr(self, 'scenes', None),
                config=config,
                cache=cache,
                embedding_index=self.embedding_index
            )
            
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
        """
        config = self.config
        
        self._print_stage("5", "GENERATE OUTPUT")
        
        output_dir = Path(config.otio_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
                entity_videos=getattr(self, 'entity_videos', None)   # V10 stock videos
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
                edl_path = output_dir / "matched_timeline.edl"
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
                xml_base_path = output_dir / "xml"
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
    
    def run(self, voiceover_path: str, num_keywords: int = None, match_only: bool = False):
        """
        Run the complete pipeline.
        All settings from config.yaml.
        """
        start_time = time.time()
        
        self._print_banner()
        
        # =====================================================================
        # UPFRONT CONFIGURATION - All prompts happen here, then pipeline runs
        # =====================================================================
        
        # Stage 1: Analyze voiceover (extracts keywords + detects topic)
        keywords = self.stage_analyze_voiceover(voiceover_path, num_keywords)
        
        if not match_only:
            # Get all user preferences BEFORE starting the pipeline
            self._configure_pipeline_upfront(keywords)
        
        # =====================================================================
        # PIPELINE EXECUTION - No pauses, runs end-to-end
        # =====================================================================
        
        if not match_only:
            # Stage 1.5: Entity image search (after keywords, before video download)
            if self.config.image_search.enabled and not self.config.pipeline.skip_image_search:
                self.stage_image_search()
            elif self.config.pipeline.skip_image_search:
                print(f"\n  ⏭ Skipping image search (config: skip_image_search=true)")
            
            # Stage 1.6: Stock video search (Pexels/Pixabay)
            if self.config.image_search.enabled and self.config.image_search.use_stock_apis and not self.config.pipeline.skip_image_search:
                self.stage_stock_video()
            
            # Stage 2: Download footage
            if not self.config.pipeline.skip_download:
                self.stage_download(keywords)
                
                # Stage 2c: Stock footage
                self.stage_download_stock(keywords)
                
                # Remix zero-download keywords (generate alternative keywords)
                keywords = self.stage_remix_zero_downloads(keywords)
            else:
                print(f"\n  ⏭ Skipping downloads (config: skip_download=true)")
                # Load existing videos from output directory
                self._load_existing_videos()
            
            # Stage 2.5: Zero-download video remix (filter videos by keyword relevance)
            if self.config.remix.enabled:
                self.stage_remix(keywords)
        
        # Stage 3: Transcribe & index
        self.stage_transcribe()
        
        # Stage 4: Match
        self.stage_match()
        
        # Stage 5: Output
        outputs = self.stage_output()
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Outputs: {len(outputs)} files generated")
        
        # Config metrics
        metrics = get_config_metrics()
        if metrics['load_count'] > 0:
            print(f"  Config loads: {metrics['load_count']} (avg {metrics['load_time_total_ms']/metrics['load_count']:.1f}ms)")
        
        # Finalize logger
        if self.run_logger:
            self.run_logger.finalize()
            print(f"  Log: {self.run_logger.log_file}")
    
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
            print(f"  • Using default settings (no prompts)")
            
            # Use defaults
            self.enhanced_enabled = config.enhanced.remix_enabled
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
            print(f"    [F] Use FILTERED videos (recommended - higher relevance)")
            print(f"    [A] Use ALL downloaded videos (skip filtering)")
            
            choice = self._get_user_input("  Select [F/A]", default="F").strip().upper()
            
            if choice == 'A':
                config.remix.auto_accept_filter = "all"
                print(f"  ✓ Will use ALL videos")
            else:
                config.remix.auto_accept_filter = "filtered"
                print(f"  ✓ Will use filtered videos (min score: {config.remix.min_relevance_score})")
        
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
    python main.py --resume
    python main.py --match-only
    python main.py --config custom_config.yaml
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
        '--validate-config',
        action='store_true',
        help='Validate config file and exit'
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
    
    # Setup logging
    logger = setup_logging(config)
    
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
        match_only=args.match_only
    )


if __name__ == '__main__':
    main()