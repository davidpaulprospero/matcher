"""
Voiceover-to-Footage Matcher v3.0

Chain-of-thought: Modular architecture for maintainability and testability
Reasoning: Each module handles a specific concern, all configurable via config.yaml
Decision: Single source of truth configuration with lazy imports for optional modules

A comprehensive video matching system with:
- Centralized configuration via config.yaml
- FAISS indexing for fast similarity search
- Hybrid text + visual embeddings
- TF-IDF auto keyword weight detection
- Two-stage matching optimization (embedding retrieval → LLM reranking)
- Duration-aware scoring
- Source rotation strategy track
- Comprehensive logging with API cost tracking
- GPU-accelerated transcription and transcoding
- MP3/video voiceover transcription support
- GPU lock fix for parallel transcription
- Hot-reload configuration support

Configuration:
    All settings are loaded from config.yaml via the config module.
    Use load_config() at startup, get_config() anywhere else.

Example:
    from src.config import load_config, get_config
    
    config = load_config("config.yaml")
    threshold = config.matching.min_confidence
"""

__version__ = "3.0.0"

# =============================================================================
# CORE EXPORTS (Always Available)
# =============================================================================

from .config import (
    Config,
    load_config,
    get_config,
    set_config,
    reload_config,
    ensure_dirs,
    get_api_key,
    get_config_metrics,
    log_hardcoded_warning,
)

from .logger import (
    RunLogger,
    get_confidence_tier,
    get_confidence_color,
    log_config_access,
    log_hardcoded,
)

# =============================================================================
# LAZY IMPORTS (Module-Level Functions for Optional Components)
# =============================================================================

def get_transcription_functions():
    """Lazy import transcription functions"""
    try:
        from .transcription import (
            transcribe_voiceover_audio,
            transcribe_voiceover_media,
            transcribe_videos_parallel,
            DeltaAwareIndex
        )
        return {
            'transcribe_voiceover_audio': transcribe_voiceover_audio,
            'transcribe_voiceover_media': transcribe_voiceover_media,
            'transcribe_videos_parallel': transcribe_videos_parallel,
            'DeltaAwareIndex': DeltaAwareIndex,
        }
    except ImportError:
        return {}


def get_embedding_functions():
    """Lazy import embedding functions"""
    try:
        from .embeddings import (
            compute_embeddings,
            get_embedding_provider,
            build_embedding_index,
            EmbeddingCache
        )
        return {
            'compute_embeddings': compute_embeddings,
            'get_embedding_provider': get_embedding_provider,
            'build_embedding_index': build_embedding_index,
            'EmbeddingCache': EmbeddingCache,
        }
    except ImportError:
        return {}


def get_matching_functions():
    """Lazy import matching functions"""
    try:
        from .matching import TieredMatcher, ReuseTracker
        return {
            'TieredMatcher': TieredMatcher,
            'ReuseTracker': ReuseTracker,
        }
    except ImportError:
        return {}


# =============================================================================
# MODULE AVAILABILITY CHECK
# =============================================================================

def check_module_availability() -> dict:
    """Check which optional modules are available"""
    modules = {}
    
    optional_modules = [
        ('transcription', 'transcribe_videos_parallel'),
        ('embeddings', 'compute_embeddings'),
        ('matching', 'TieredMatcher'),
        ('downloader', 'VideoDownloader'),
        ('keyword_extractor', 'LLMKeywordExtractor'),
        ('keyword_remix', 'KeywordRemixer'),
        ('otio_builder', 'OTIOBuilder'),
        ('scene_detection', 'SceneDetector'),
        ('vision', 'process_video_vision'),
        ('audio_analysis', 'AudioAnalyzer'),
        ('deduplication', 'deduplicate_videos'),
        ('pexels', 'download_pexels_footage'),
        ('pixabay', 'download_pixabay_footage'),
        ('multi_style', 'OTIOStyle'),
        ('watcher', 'FileWatcher'),
    ]
    
    for module_name, test_export in optional_modules:
        try:
            module = __import__(f'.{module_name}', globals(), locals(), [test_export], 1)
            modules[module_name] = hasattr(module, test_export)
        except ImportError:
            modules[module_name] = False
    
    return modules


# =============================================================================
# VERSION AND CONFIG INFO
# =============================================================================

def get_version_info() -> dict:
    """Get version and configuration info"""
    return {
        'version': __version__,
        'config_loaded': get_config()._config_path if get_config() else None,
        'modules': check_module_availability(),
    }
