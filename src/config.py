"""
Configuration Management System v3.0

Chain-of-thought: Centralized configuration loader for matcher-alt pipeline
Reasoning: Single source of truth reduces maintenance, improves consistency, enables hot-reload
Decision: CSafeLoader for performance, schema validation, file change detection

Features:
- CSafeLoader (C-based YAML parser) for 40-60% faster parsing
- Schema validation with clear error messages
- Hot-reload with file change detection (hash-based)
- Caching mechanism for repeated config access
- Comprehensive dataclasses mirroring config.yaml structure
- Logging of config access patterns

Performance:
- Config loading: <100ms target
- Component init: <50ms target
- Cached access: <1ms

Usage:
    from src.config import load_config, get_config
    
    # At startup
    config = load_config("config.yaml")
    
    # Anywhere in codebase
    config = get_config()
    threshold = config.matching.min_confidence
"""

import os
import sys
import json
import logging
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List, Dict, Any, Tuple, Union, TypeVar, Type
from datetime import datetime
from functools import lru_cache
import threading

# Use CSafeLoader for performance (40-60% faster than pure Python)
try:
    import yaml
    from yaml import CSafeLoader as SafeLoader
    YAML_FAST = True
except ImportError:
    import yaml
    from yaml import SafeLoader
    YAML_FAST = False

logger = logging.getLogger(__name__)

# Type variable for dataclass building
T = TypeVar('T')

# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================

_config_metrics = {
    'load_count': 0,
    'load_time_total_ms': 0.0,
    'cache_hits': 0,
    'cache_misses': 0,
    'reload_count': 0,
    'validation_errors': 0,
}


def get_config_metrics() -> Dict[str, Any]:
    """Get configuration performance metrics"""
    return _config_metrics.copy()


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================
# Chain-of-thought: Each dataclass mirrors a section in config.yaml
# Reasoning: Type safety, IDE autocomplete, validation
# Decision: Use field(default_factory=...) for mutable defaults

@dataclass
class ProjectConfig:
    """Project identification settings"""
    name: str = "matcher-alt"
    version: str = "3.0.0"
    description: str = ""


@dataclass
@dataclass
class PauseSplitConfig:
    """Pause-based segment splitting settings
    
    Aggressive splitting for better video matching:
    - Sentence boundaries: Every sentence becomes its own segment
    - List markers: "1.", "2.", "8." isolates countdown items
    - Location patterns: "Atlanta, Georgia," splits at city/state pairs
    """
    enabled: bool = True
    min_gap_ms: int = 300  # Minimum gap to consider a pause
    split_at_sentences: bool = True  # Split at sentence boundaries (. ! ?)
    split_at_list_markers: bool = True  # Split before "1.", "2.", "Number 10", etc.
    split_at_locations: bool = True  # Split at "City, State" patterns  
    min_phrase_words: int = 2  # Minimum words to keep in a segment
    min_segment_duration: float = 0.5  # Minimum duration for split segments (seconds)


@dataclass
class TranscriptionConfig:
    """Transcription settings (faster-whisper)
    
    Chain-of-thought: GPU acceleration critical for performance
    Reasoning: Parallel audio extraction (CPU) + sequential GPU transcription
    Decision: max_workers controls CPU parallelism, GPU uses shared model
    """
    provider: str = "faster-whisper"
    model: str = "base"  # tiny, base, small, medium, large, large-v2, large-v3
    language: str = "en"
    use_gpu: bool = True
    compute_type: str = "auto"  # auto, float16, int8, float32
    
    # Parallel processing
    max_workers: int = 4
    batch_size: int = 10
    
    # VAD settings
    vad_filter: bool = True
    min_silence_duration_ms: int = 200
    speech_pad_ms: int = 10
    
    # Segment splitting (for voiceover optimization)
    split_threshold_multiplier: float = 1.5  # Split if > multiplier * median
    long_median_threshold: float = 8.0  # If median > this, use absolute threshold
    absolute_split_threshold: float = 12.0  # Absolute threshold for very long segments
    min_split_duration: float = 4.0  # Minimum duration to consider splitting
    
    # Pause-based splitting
    pause_split: PauseSplitConfig = None
    
    # Caching
    cache_transcriptions: bool = True
    cache_dir: str = "transcriptions"
    
    def __post_init__(self):
        if self.pause_split is None:
            self.pause_split = PauseSplitConfig()


@dataclass
class EmbeddingConfig:
    """Embedding generation settings
    
    Chain-of-thought: Gemini embeddings best quality, local for offline
    Reasoning: Batch processing reduces API calls by 100x
    Decision: batch_size=100 optimal for Gemini API limits
    """
    provider: str = "gemini"  # gemini, voyage, local
    
    # Model settings per provider
    gemini_model: str = "models/text-embedding-004"
    voyage_model: str = "voyage-2"
    local_model: str = "all-MiniLM-L6-v2"
    
    # Batch processing (Gemini supports up to 100)
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Caching
    cache_embeddings: bool = True
    cache_batch_results: bool = True


@dataclass
class IndexingConfig:
    """FAISS indexing settings
    
    Chain-of-thought: Vector index for fast similarity search
    Reasoning: Flat index exact but O(n), IVF approximate but O(sqrt(n))
    Decision: Use flat for <10k vectors, IVF for larger
    """
    index_type: str = "flat"  # flat, ivf
    use_faiss: bool = True
    
    # IVF settings (for large datasets)
    ivf_nlist: int = 100  # Number of clusters
    ivf_nprobe: int = 10  # Clusters to search
    
    # Similarity settings
    normalize_embeddings: bool = True
    similarity_metric: str = "cosine"  # cosine, dot, euclidean


@dataclass
class VisionConfig:
    """Vision processing settings
    
    Chain-of-thought: Vision API expensive, use selectively
    Reasoning: Only process scenes where transcript is sparse
    Decision: coverage_threshold=0.3 means skip if >30% transcribed
    """
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    enabled: bool = True
    
    # Selective processing
    min_words_per_scene: int = 5
    coverage_threshold: float = 0.3
    max_scenes_per_video: int = 50
    
    # Frame extraction
    frame_format: str = "jpg"
    frame_quality: int = 85
    
    # Cost management
    max_api_calls_per_run: int = 100
    estimated_cost_per_call: float = 0.001


@dataclass
class SceneDetectionConfig:
    """Scene detection settings (PySceneDetect)
    
    Chain-of-thought: Scene boundaries enable granular clip selection
    Reasoning: Presets balance speed vs accuracy for different use cases
    Decision: "balanced" default, "fast" for preview, "accurate" for final
    """
    enabled: bool = True
    preset: str = "balanced"  # fast, balanced, accurate
    
    # Individual settings (can override preset)
    threshold: float = 27.0
    min_scene_len: int = 15  # frames
    downscale_factor: int = 4
    frame_skip: int = 2
    
    # Minimum video duration to process (seconds, 0 = no minimum)
    min_video_duration: float = 0
    
    # Audio analysis integration
    audio_analysis: bool = True
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 0.3
    
    # Hardware acceleration
    use_gpu: bool = True
    force_gpu: bool = False


@dataclass
class AudioAnalysisConfig:
    """Audio analysis settings (librosa)"""
    enabled: bool = True
    sample_rate: int = 22050
    
    # Silence detection
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 0.3
    
    # Speech detection
    speech_threshold: float = 0.5
    min_speech_duration: float = 0.2


@dataclass
class MatchingConfig:
    """Matching engine settings
    
    Chain-of-thought: Two-stage matching optimizes cost vs quality
    Reasoning: Embedding search cheap (FAISS), LLM expensive
    Decision: Retrieve 20 candidates → rerank top 5 with LLM
    """
    # Confidence thresholds
    min_confidence: float = 0.7
    high_confidence_threshold: float = 0.85  # Skip LLM if above
    skip_llm_threshold: float = 0.85  # Alias for high_confidence_threshold (for matching.py compatibility)
    low_confidence_threshold: float = 0.5    # Use secondary LLM if below
    ambiguous_threshold: float = 0.6  # Use secondary LLM if confidence < this
    confidence_threshold: float = 0.5  # Legacy alias for min_confidence
    
    # Clip reuse prevention
    max_clip_reuse: int = 1
    reuse_penalty: float = 0.5
    smart_reuse: bool = True
    sequential_when_reuse: bool = True
    
    # Two-stage matching optimization
    embedding_candidates: int = 50  # Retrieve from FAISS (need 50+ for V1-V6)
    llm_rerank_candidates: int = 5  # Send to LLM
    top_k_candidates: int = 10  # Legacy alias
    
    # Context window
    context_window: int = 2  # Consider N segments before/after
    
    # Duration scoring
    duration_scoring_enabled: bool = True
    ideal_speed_min: float = 0.85
    ideal_speed_max: float = 1.15
    soft_speed_min: float = 0.70
    soft_speed_max: float = 1.50
    duration_penalty_factor: float = 0.1
    
    # Tuple versions for matching.py compatibility
    @property
    def ideal_speed_range(self) -> tuple:
        return (self.ideal_speed_min, self.ideal_speed_max)
    
    @property
    def soft_penalty_range(self) -> tuple:
        return (self.soft_speed_min, self.soft_speed_max)
    
    # Keyword/entity boosting
    keyword_boost: float = 0.05
    entity_boost: float = 0.08
    
    # LLM providers (tiered: primary → secondary → local)
    primary_provider: str = "gemini"
    secondary_provider: str = "anthropic"
    local_provider: str = "ollama"
    use_local_for_review: bool = True
    
    # Model names
    gemini_model: str = "gemini-2.0-flash"
    anthropic_model: str = "claude-3-haiku-20240307"
    ollama_model: str = "llama3.2"
    local_llm_model: str = "llama3.2"  # Alias for ollama_model
    ollama_host: str = "http://localhost:11434"  # Ollama API host
    
    # Caching
    cache_llm_responses: bool = True
    cache_ttl_hours: int = 24


@dataclass
class NegativeMatchingConfig:
    """Negative matching rules - what NOT to match"""
    enabled: bool = True
    rules: List[str] = field(default_factory=lambda: [
        "Don't match talking head shots to action narration",
        "Don't match static images to dynamic narration",
        "Avoid matching unrelated b-roll to specific statements"
    ])


@dataclass
class RemixConfig:
    """Configuration for zero-download keyword remix functionality.
    
    Enables filtering/remixing of downloaded videos based on keyword
    relevance before transcription, without additional downloads.
    """
    enabled: bool = True
    trigger_after_download: bool = True
    
    # Scoring thresholds
    min_relevance_score: float = 0.1  # Minimum score to include
    high_relevance_threshold: float = 0.5  # Score for "high relevance" label
    
    # Processing limits
    max_files_to_process: int = 500
    max_files_to_include: int = 100  # Maximum videos to pass to transcription
    
    # Matching settings
    fuzzy_match: bool = True  # Allow partial keyword matches
    case_sensitive: bool = False
    
    # Interactive mode
    interactive_curation: bool = True  # Ask user to confirm remix
    show_excluded: bool = True  # Show which files were excluded
    
    # Auto-accept filter results (set upfront to skip mid-pipeline prompt)
    # Options: "filtered" (use filtered), "all" (use all videos), "prompt" (ask during pipeline)
    auto_accept_filter: str = "prompt"
    
    # Logging
    log_level: str = "INFO"
    log_file_processing: bool = False  # Log each file (verbose)
    
    # Performance
    parallel_scoring: bool = True
    max_workers: int = 4


@dataclass
class ZeroDownloadRemixConfig:
    """Configuration for zero-download keyword remixing.
    
    When a keyword returns 0 downloads from YouTube, use LLM to 
    generate alternative search terms that are more likely to yield results.
    """
    enabled: bool = True
    max_retries: int = 2  # Maximum remix attempts per batch
    use_llm: bool = True  # Use LLM for smart remixing
    use_fallback: bool = True  # Use rule-based fallback if LLM fails
    cache_results: bool = True  # Cache remix results
    min_keywords_to_trigger: int = 1  # Minimum failed keywords to trigger remix
    max_keywords_per_batch: int = 20  # Maximum keywords to remix at once


@dataclass
@dataclass
class StockVideoConfig:
    """Configuration for stock video downloads."""
    min_duration: float = 3.0  # Minimum video duration in seconds
    max_duration: float = 30.0  # Maximum video duration in seconds
    prefer_hd: bool = True  # Prefer HD quality videos


@dataclass
class ImageSearchConfig:
    """Configuration for entity image and video search.
    
    Downloads images representing entities (people, places, organizations)
    mentioned in the voiceover for use as stills/overlays on V9 track.
    
    Also downloads stock videos from Pexels/Pixabay for V10 track.
    """
    enabled: bool = True
    
    # Short path settings (like download config)
    root_dir: str = ""  # Empty = use project_dir/images. Set to e.g. "E:/i"
    folder_name: str = "images"  # Subfolder name when root_dir is empty
    
    # Images per entity (5 recommended for variety and backup options)
    images_per_entity: int = 5
    
    # Videos per entity for stock footage (3 recommended)
    videos_per_entity: int = 3
    
    # Maximum number of entities to process (0 = no limit)
    max_entities: int = 0
    
    # Minimum file size in MB (1MB default for quality)
    min_size_mb: float = 1.0
    
    # Output directory (relative to project) - DEPRECATED, use root_dir instead
    output_dir: str = "images"
    
    # Search sources
    use_google: bool = True  # Google Images (requires pyimagedl library)
    use_bing: bool = False  # Bing Images (disabled by default - alive_progress conflicts)
    use_stock_apis: bool = True  # Pexels/Pixabay for images AND videos (recommended fallback)
    
    # Entity types to search for
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "GPE", "ORG", "DATE", "EVENT"
    ])
    
    # OTIO output settings
    image_track: str = "V9"  # Track for image stills
    stock_video_track: str = "V10"  # Track for stock videos
    default_duration: float = 0.0  # 0 = match segment duration
    
    # Download timeouts and limits
    download_timeout: int = 10  # Seconds per image download
    max_search_time: int = 300  # Max seconds for entire search per entity (5 min)
    max_results_to_check: int = 500  # Max search results to check per entity
    search_until_found: bool = True  # If true, keeps searching until images_per_entity found
    
    # Stock video settings
    stock_video: StockVideoConfig = field(default_factory=StockVideoConfig)


@dataclass
class ListDetectionConfig:
    """List-based keyword detection settings
    
    Detects numbered lists in voiceover (e.g., "Number 10 Austin, Texas")
    and ensures each list item gets a guaranteed download keyword.
    """
    enabled: bool = True
    download_first: bool = True  # Download list keywords before general keywords
    skip_if_entity_covered: bool = True  # Skip if already in entity extraction
    keyword_suffix: str = "footage"  # Suffix for generated keywords


@dataclass
class KeywordConfig:
    """Keyword extraction settings"""
    provider: str = "gemini"  # gemini, anthropic, tfidf
    
    # Extraction settings
    max_keywords: int = 30
    min_keyword_length: int = 3
    
    # List detection
    list_detection: ListDetectionConfig = None
    
    # Entity extraction
    extract_entities: bool = True
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "GPE", "ORG", "DATE", "EVENT"
    ])
    
    # TF-IDF fallback
    use_tfidf_weights: bool = True
    tfidf_max_features: int = 100
    
    # Footage suffixes
    add_footage_suffixes: bool = True
    footage_suffixes: List[str] = field(default_factory=lambda: [
        "4K footage", "news footage", "drone footage",
        "aerial footage", "stock footage", "documentary footage"
    ])
    
    # Batch processing
    batch_size: int = 50
    
    def __post_init__(self):
        if self.list_detection is None:
            self.list_detection = ListDetectionConfig()


@dataclass
class LLMConfig:
    """LLM settings for keyword extraction and other tasks
    
    Chain-of-thought: Centralizes LLM provider settings
    Reasoning: Multiple components need LLM access
    Decision: Single config section shared across components
    """
    provider: str = "google"  # google, anthropic
    model: str = "gemini-2.0-flash"
    api_key: str = ""  # Loaded from environment if empty
    
    # Anthropic specific
    anthropic_model: str = "claude-3-haiku-20240307"
    
    # Temperature and generation settings
    temperature: float = 0.7
    max_tokens: int = 2000
    
    def __post_init__(self):
        """Load API key from environment if not set"""
        import os
        if not self.api_key:
            if self.provider == "google":
                self.api_key = os.getenv("GEMINI_API_KEY", "")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class EnhancedFeaturesConfig:
    """Enhanced features (confidence enforcement, remix)
    
    Chain-of-thought: 90% confidence ensures quality output
    Reasoning: Keyword remix generates alternatives for low matches
    Decision: Enable by default, allow user override
    """
    enabled: bool = True
    min_confidence: float = 0.90  # 90% minimum
    max_retries: int = 3
    
    # Keyword remix
    remix_enabled: bool = True
    remix_keyword_count: int = 10
    
    # Stock footage integration
    enable_pexels: bool = True
    enable_pixabay: bool = True
    stock_per_keyword: int = 2
    
    # User prompts
    confirm_before_download: bool = True
    prompt_enhanced_features: bool = True
    non_interactive: bool = False  # Skip ALL prompts, use defaults
    
    # Face preference for matching
    # Options: "neutral" (no preference), "more" (prefer faces), "none" (avoid faces)
    face_preference: str = "neutral"


@dataclass
class LLMTitleFilterConfig:
    """Config for LLM-based title filtering before download."""
    enabled: bool = True
    provider: str = "gemini"  # gemini or anthropic
    model: str = "gemini-2.0-flash"  # or claude-3-haiku-20240307
    batch_size: int = 20  # Check multiple titles at once
    min_relevance: float = 0.7  # 0-1, reject if below


@dataclass
class DownloadConfig:
    """Download settings for yt-dlp (matches downloader.py expectations)
    
    Chain-of-thought: This config matches what VideoDownloader expects
    Reasoning: downloader.py accesses config.download with specific attributes
    Decision: Provide all attributes that downloader.py uses
    """
    # Short path settings - keeps paths under Windows 260 char limit
    # and improves NLE import performance
    root_dir: str = ""  # Empty = use project_dir/videos. Set to e.g. "E:/v"
    folder_name: str = "videos"  # Downloads folder name
    max_keyword_len: int = 6  # Max chars for keyword folder names
    max_filename_len: int = 8  # Max chars for video title in filename
    
    # Quality settings
    quality: str = "1080p"
    format: str = "mp4"
    
    # DaVinci Resolve compatibility
    davinci_mode: bool = True
    davinci_codec: str = "h264"
    davinci_prores_profile: str = "proxy"
    
    # Hardware acceleration
    hw_accel: str = "auto"  # auto, cuda, amf, qsv, videotoolbox, none
    hw_quality: str = "high"
    
    # Download behavior
    min_views: int = 0
    delete_original: bool = True  # Delete original after transcode
    delay_between_keywords: float = 1.0
    
    # Title blacklist - skip videos containing these terms (case-insensitive)
    title_blacklist: List[str] = field(default_factory=lambda: [
        "highlights", "basketball", "football", "soccer", "nba", "nfl",
        "mlb", "nhl", "ufc", "boxing", "wrestling", "vs.", "vs ",
        "match", "game recap", "full game", "full match", "sports",
        "espn", "goals", "touchdowns"
    ])
    
    # LLM Title Filter - use AI to check if video titles are relevant
    llm_title_filter: LLMTitleFilterConfig = field(default_factory=LLMTitleFilterConfig)
    
    # YouTube authentication
    # Required due to YouTube bot detection - export cookies from browser
    # See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp
    cookies_path: str = ""  # Path to cookies.txt (auto-detected if empty)
    cookies_from_browser: str = ""  # Browser to extract cookies from: chrome, firefox, edge, etc.
    
    # Search pool multiplier - ytsearch returns limited results, so we need to 
    # search more than we want to download to find videos matching duration filters
    search_pool_multiplier: int = 5  # Search 5x what we want to download
    min_search_pool: int = 40  # Minimum search pool size
    
    # Timeout settings
    search_timeout: int = 60  # Seconds for search metadata subprocess
    download_timeout: int = 120  # Seconds per video download (2 min) - retries with modified keyword on timeout
    
    # Duration tiers (can be overridden)
    tiers: dict = field(default_factory=lambda: {
        'short': {'min': 20, 'max': 120, 'per_keyword': 8},
        'medium': {'min': 120, 'max': 600, 'per_keyword': 8},
        'long': {'min': 600, 'max': 1500, 'per_keyword': 5},
        'longer': {'min': 1500, 'max': 3000, 'per_keyword': 5}
    })


@dataclass
class DownloadingConfig:
    """Video downloading settings (yt-dlp)
    
    Chain-of-thought: Duration tiers optimize for different content types
    Reasoning: Short clips for B-roll, long for documentary segments
    Decision: Prefer H.264 to avoid VP9/AV1 transcoding overhead
    """
    output_dir: str = "downloaded_videos"
    organize_by_keyword: bool = True
    
    # Quality settings
    preferred_quality: str = "1080"
    max_quality: str = "2160"
    prefer_h264: bool = True
    
    # DaVinci Resolve compatible codecs
    davinci_compatible_codecs: List[str] = field(default_factory=lambda: [
        "h264", "hevc", "prores", "dnxhd", "dnxhr"
    ])
    
    # Codecs requiring transcoding
    transcode_codecs: List[str] = field(default_factory=lambda: [
        "vp9", "av1", "vp8"
    ])
    
    # Transcoding settings
    transcode_codec: str = "h264"
    transcode_crf: int = 18
    use_hw_accel: bool = True
    hw_accel_type: str = "auto"  # auto, cuda, amf, qsv, videotoolbox
    
    # Rate limiting
    max_downloads_per_keyword: int = 10
    download_delay: float = 1.0
    
    # Checkpointing
    use_checkpoints: bool = True
    checkpoint_file: str = "download_checkpoint.json"


@dataclass
class DurationTierConfig:
    """Single duration tier configuration"""
    min_seconds: int = 0
    max_seconds: int = 120
    videos_per_keyword: int = 5


@dataclass
class DurationTiersConfig:
    """Duration tiers for downloading"""
    short: DurationTierConfig = field(default_factory=lambda: DurationTierConfig(20, 120, 8))
    medium: DurationTierConfig = field(default_factory=lambda: DurationTierConfig(120, 600, 8))
    long: DurationTierConfig = field(default_factory=lambda: DurationTierConfig(600, 1500, 5))
    longer: DurationTierConfig = field(default_factory=lambda: DurationTierConfig(1500, 3000, 5))


@dataclass
class StockFootageConfig:
    """Stock footage API settings (Pexels, Pixabay)"""
    enabled: bool = True
    pexels_enabled: bool = True
    pixabay_enabled: bool = True
    per_keyword: int = 3
    min_duration: int = 5
    max_duration: int = 60
    min_height: int = 720
    prefer_landscape: bool = True
    request_interval: float = 0.5


@dataclass
class DeduplicationConfig:
    """Video deduplication settings"""
    enabled: bool = True
    hash_threshold: int = 10  # Hamming distance (0-64, lower = stricter)
    use_first_frame: bool = True
    auto_delete: bool = True
    generate_report: bool = True
    frame_timeout: int = 30  # Seconds for FFmpeg frame extraction


@dataclass
class VarietyConfig:
    """Settings to ensure variety across tracks (for strategy matching)"""
    require_different_source: bool = True  # Each track must use different source video
    exclude_same_clip: bool = True  # Never use exact same clip on multiple tracks
    min_time_distance: float = 10.0  # Clips must be N seconds apart (same source)
    min_embedding_distance: float = 0.3  # V4+ must have distance > this from V1-V3
    
    # Timeline variety enforcement - prevents same source video from dominating
    # Within timeline_variety_window seconds, same source can only appear max_source_repeats times
    enforce_timeline_variety: bool = True  # Enable timeline-based variety enforcement
    timeline_variety_window: float = 600.0  # 10 minutes - no same source within this window
    max_source_repeats_in_window: int = 1  # Max times same source can appear in window


@dataclass
class OutputConfig:
    """Output generation settings
    
    Chain-of-thought: Multiple tracks give editors options
    Reasoning: V1-V3 alternatives, V4-V8 strategies
    Decision: Enable all by default, let editors disable unused
    """
    output_dir: str = "output"
    
    # File formats
    generate_otio: bool = True
    split_otio: bool = True  # Split OTIO into multiple files by clip batches
    otio_clips_per_file: int = 10  # Max clips per OTIO file
    generate_edl: bool = True
    generate_xml: bool = True  # DaVinci Resolve XML (fallback if OTIO fails)
    xml_parts: int = 2  # Split XML into multiple files (helps with large projects)
    generate_report: bool = True
    
    # Timeline settings
    frame_rate: float = 30.0
    timeline_start_tc: str = "01:00:00:00"  # Standard broadcast start
    
    # Track structure
    # V1: Primary, V2-V3: Alternatives
    # V4-V6: Secondary (different video files from V1-V3)
    # V7: Embedding-Diversity strategy
    num_alternatives: int = 2  # V2-V3
    include_alternatives: bool = True
    include_strategy_tracks: bool = True
    strategy_tracks: List[str] = field(default_factory=lambda: [
        "embedding_diversity"  # V7 only - V4-V6 are now secondary matches
    ])
    
    # Variety enforcement for strategy tracks
    variety: VarietyConfig = field(default_factory=VarietyConfig)
    
    # Markers
    include_speed_markers: bool = True
    max_recommended_speed: float = 1.5
    markers_for_low_confidence: bool = True
    markers_for_gaps: bool = True
    markers_for_speed: bool = True


@dataclass
class MultiStyleConfig:
    """Multi-style OTIO generation"""
    enabled: bool = False
    styles: List[str] = field(default_factory=lambda: ["default", "strict"])


@dataclass
class LoggingConfig:
    """Logging settings
    
    Chain-of-thought: Comprehensive logging aids debugging and auditing
    Reasoning: Dual output (file + JSON) for human and machine consumption
    Decision: Track API costs for budget management
    """
    enabled: bool = True
    log_dir: str = "logs"
    log_level: str = "INFO"
    
    # Output targets
    log_to_file: bool = True
    log_to_console: bool = True
    generate_json_log: bool = True
    
    # Log content
    log_api_calls: bool = True
    log_match_decisions: bool = True
    log_performance: bool = True
    
    # Cost tracking
    track_api_costs: bool = True
    
    # Config debugging
    log_config_access: bool = False
    warn_on_hardcoded: bool = True


@dataclass
class CacheConfig:
    """Cache settings"""
    cache_dir: str = ".cache"
    cache_transcriptions: bool = True
    cache_embeddings: bool = True
    cache_scenes: bool = True
    cache_llm_responses: bool = True
    cache_vision: bool = True
    
    # Cross-project cache
    cross_project_cache: bool = False
    cross_project_cache_dir: str = ""
    
    # Validation
    validate_cache_on_load: bool = True
    use_file_hash: bool = True


@dataclass
class PipelineConfig:
    """Pipeline automation settings"""
    # Stage control - skip individual stages
    skip_download: bool = False        # Skip video download (use existing videos)
    skip_image_search: bool = False    # Skip entity image search
    skip_transcription: bool = False   # Skip video transcription (use cached)
    skip_scene_detection: bool = False # Skip scene detection
    skip_matching: bool = False        # Skip matching stage
    
    # Video source directory (used when skip_download=true)
    # Set to absolute path of folder containing videos
    video_source_dir: str = ""
    
    # Resume/retry
    resume_enabled: bool = True
    max_retries: int = 3
    
    # Parallel processing
    parallel_transcription: bool = True
    parallel_embedding: bool = True


@dataclass
class APIKeysConfig:
    """API keys (loaded from environment)"""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    unsplash_api_key: str = ""
    
    def __post_init__(self):
        """Load from environment if not set"""
        self.gemini_api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.voyage_api_key = self.voyage_api_key or os.getenv("VOYAGE_API_KEY", "")
        self.pexels_api_key = self.pexels_api_key or os.getenv("PEXELS_API_KEY", "")
        self.pixabay_api_key = self.pixabay_api_key or os.getenv("PIXABAY_API_KEY", "")
        self.unsplash_api_key = self.unsplash_api_key or os.getenv("UNSPLASH_API_KEY", "")


# =============================================================================
# MAIN CONFIG CLASS
# =============================================================================

@dataclass
class Config:
    """
    Master configuration class - Single Source of Truth
    
    Chain-of-thought: All components access settings through this class
    Reasoning: Centralizes configuration, enables validation and hot-reload
    Decision: Load from config.yaml using Config.from_yaml(path)
    
    Usage:
        config = Config.from_yaml("config.yaml")
        threshold = config.matching.min_confidence
    """
    # Project settings
    project: ProjectConfig = field(default_factory=ProjectConfig)
    
    # Section configs
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    scene_detection: SceneDetectionConfig = field(default_factory=SceneDetectionConfig)
    audio_analysis: AudioAnalysisConfig = field(default_factory=AudioAnalysisConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    negative_matching: NegativeMatchingConfig = field(default_factory=NegativeMatchingConfig)
    remix: RemixConfig = field(default_factory=RemixConfig)
    zero_download_remix: ZeroDownloadRemixConfig = field(default_factory=ZeroDownloadRemixConfig)
    image_search: ImageSearchConfig = field(default_factory=ImageSearchConfig)
    keyword: KeywordConfig = field(default_factory=KeywordConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    enhanced: EnhancedFeaturesConfig = field(default_factory=EnhancedFeaturesConfig)
    downloading: DownloadingConfig = field(default_factory=DownloadingConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    duration_tiers: DurationTiersConfig = field(default_factory=DurationTiersConfig)
    stock_footage: StockFootageConfig = field(default_factory=StockFootageConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    multi_style: MultiStyleConfig = field(default_factory=MultiStyleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    
    # Convenience paths (resolved at load time)
    project_dir: str = "."
    downloaded_videos_dir: str = "downloaded_videos"
    otio_output_dir: str = "output"
    cache_dir: str = ".cache"
    
    # API key aliases (for matching.py compatibility)
    # These are populated from api_keys in __post_init__
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    
    # Metadata (private)
    _config_path: str = ""
    _config_hash: str = ""
    _loaded_at: str = ""
    _load_time_ms: float = 0.0
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        self._loaded_at = datetime.now().isoformat()
        self._resolve_paths()
        self._populate_api_keys()
    
    def _populate_api_keys(self):
        """Populate top-level API key aliases from api_keys config"""
        self.gemini_api_key = self.api_keys.gemini_api_key
        self.anthropic_api_key = self.api_keys.anthropic_api_key
        self.voyage_api_key = self.api_keys.voyage_api_key
    
    def _resolve_paths(self):
        """Resolve relative paths to absolute"""
        base = Path(self.project_dir).resolve()
        
        # Resolve all path attributes
        path_attrs = [
            ('downloaded_videos_dir', self.downloading.output_dir),
            ('otio_output_dir', self.output.output_dir),
        ]
        
        for attr_name, config_value in path_attrs:
            if config_value and not Path(config_value).is_absolute():
                setattr(self, attr_name, str(base / config_value))
            else:
                setattr(self, attr_name, config_value)
        
        # Cache dir
        if self.cache.cache_dir and not Path(self.cache.cache_dir).is_absolute():
            self.cache.cache_dir = str(base / self.cache.cache_dir)
        
        # Log dir
        if self.logging.log_dir and not Path(self.logging.log_dir).is_absolute():
            self.logging.log_dir = str(base / self.logging.log_dir)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML file with CSafeLoader for performance.
        
        Performance: ~50ms for typical config file
        """
        global _config_metrics
        start_time = time.perf_counter()
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = cls()
            config._config_path = str(config_path)
            _config_metrics['cache_misses'] += 1
            return config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                # Use CSafeLoader for 40-60% faster parsing
                data = yaml.load(f, Loader=SafeLoader) or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            _config_metrics['validation_errors'] += 1
            return cls()
        
        # Compute hash for change detection
        config_hash = hashlib.md5(
            yaml.dump(data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Build config object
        config = cls._from_dict(data)
        config._config_path = str(config_path)
        config._config_hash = config_hash
        
        # Track metrics
        load_time = (time.perf_counter() - start_time) * 1000
        config._load_time_ms = load_time
        _config_metrics['load_count'] += 1
        _config_metrics['load_time_total_ms'] += load_time
        
        logger.info(f"Loaded config from {config_path} in {load_time:.1f}ms (hash: {config_hash})")
        if YAML_FAST:
            logger.debug("Using CSafeLoader (C-based) for optimized parsing")
        
        return config
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build Config from dictionary with nested dataclass handling"""
        config = cls()
        
        # Project settings
        if 'project' in data:
            config.project = cls._build_dataclass(ProjectConfig, data['project'])
        
        config.project_dir = data.get('project_dir', config.project_dir)
        
        # Section mapping: yaml_key -> (dataclass_type, attr_name)
        section_mapping = {
            'transcription': (TranscriptionConfig, 'transcription'),
            'embedding': (EmbeddingConfig, 'embedding'),
            'indexing': (IndexingConfig, 'indexing'),
            'vision': (VisionConfig, 'vision'),
            'scene_detection': (SceneDetectionConfig, 'scene_detection'),
            'audio_analysis': (AudioAnalysisConfig, 'audio_analysis'),
            'matching': (MatchingConfig, 'matching'),
            'negative_matching': (NegativeMatchingConfig, 'negative_matching'),
            'remix': (RemixConfig, 'remix'),
            'zero_download_remix': (ZeroDownloadRemixConfig, 'zero_download_remix'),
            'image_search': (ImageSearchConfig, 'image_search'),
            'keyword': (KeywordConfig, 'keyword'),
            'llm': (LLMConfig, 'llm'),
            'enhanced': (EnhancedFeaturesConfig, 'enhanced'),
            'downloading': (DownloadingConfig, 'downloading'),
            'download': (DownloadConfig, 'download'),
            'stock_footage': (StockFootageConfig, 'stock_footage'),
            'deduplication': (DeduplicationConfig, 'deduplication'),
            'output': (OutputConfig, 'output'),
            'multi_style': (MultiStyleConfig, 'multi_style'),
            'logging': (LoggingConfig, 'logging'),
            'cache': (CacheConfig, 'cache'),
            'pipeline': (PipelineConfig, 'pipeline'),
            'api_keys': (APIKeysConfig, 'api_keys'),
        }
        
        for yaml_key, (dataclass_type, attr_name) in section_mapping.items():
            section_data = data.get(yaml_key, {})
            if section_data:
                section_config = cls._build_dataclass(dataclass_type, section_data)
                setattr(config, attr_name, section_config)
        
        # Handle duration_tiers specially (nested structure)
        if 'duration_tiers' in data:
            config.duration_tiers = cls._build_duration_tiers(data['duration_tiers'])
        
        return config
    
    @staticmethod
    def _build_dataclass(dataclass_type: Type[T], data: Dict) -> T:
        """Build a dataclass from dict, handling missing/extra fields and nested dataclasses"""
        if not data:
            return dataclass_type()
        
        valid_fields = {f.name: f for f in fields(dataclass_type)}
        filtered_data = {}
        
        for key, value in data.items():
            if key in valid_fields:
                field_info = valid_fields[key]
                field_type = field_info.type
                
                # Check if this field is a nested dataclass
                # Handle Optional types and get the actual type
                origin = getattr(field_type, '__origin__', None)
                if origin is not None:
                    # For Optional[X], Union[X, None], etc.
                    args = getattr(field_type, '__args__', ())
                    if args:
                        field_type = args[0]
                
                # Check if the field type is a dataclass
                if hasattr(field_type, '__dataclass_fields__') and isinstance(value, dict):
                    # Recursively build nested dataclass
                    filtered_data[key] = Config._build_dataclass(field_type, value)
                else:
                    filtered_data[key] = value
            else:
                logger.debug(f"Ignoring unknown config field in {dataclass_type.__name__}: {key}")
        
        try:
            return dataclass_type(**filtered_data)
        except TypeError as e:
            logger.warning(f"Error building {dataclass_type.__name__}: {e}")
            return dataclass_type()
    
    @staticmethod
    def _build_duration_tiers(data: Dict) -> DurationTiersConfig:
        """Build duration tiers from nested config"""
        tiers = DurationTiersConfig()
        
        for tier_name in ['short', 'medium', 'long', 'longer']:
            if tier_name in data:
                tier_data = data[tier_name]
                tier_config = DurationTierConfig(
                    min_seconds=tier_data.get('min', 0),
                    max_seconds=tier_data.get('max', 120),
                    videos_per_keyword=tier_data.get('count', 5)
                )
                setattr(tiers, tier_name, tier_config)
        
        return tiers
    
    def to_yaml(self, output_path: str = None) -> str:
        """Serialize config to YAML"""
        data = self._to_dict()
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
            logger.info(f"Saved config to {output_path}")
        
        return yaml_str
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding private fields)"""
        result = {
            'project': asdict(self.project),
            'project_dir': self.project_dir,
        }
        
        sections = [
            'transcription', 'embedding', 'indexing', 'vision',
            'scene_detection', 'audio_analysis', 'matching', 'negative_matching', 'remix', 'image_search', 'keyword', 'llm',
            'enhanced', 'downloading', 'download', 'stock_footage', 'deduplication',
            'output', 'multi_style', 'logging', 'cache', 'pipeline'
        ]
        
        for section in sections:
            section_config = getattr(self, section, None)
            if section_config:
                result[section] = asdict(section_config)
        
        # Don't include API keys in serialization
        return result
    
    def reload(self) -> bool:
        """
        Reload config from file if changed.
        
        Returns:
            True if config was reloaded
        """
        global _config_metrics
        
        if not self._config_path:
            return False
        
        config_path = Path(self._config_path)
        if not config_path.exists():
            return False
        
        # Check hash for changes
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=SafeLoader) or {}
        
        new_hash = hashlib.md5(
            yaml.dump(data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        if new_hash == self._config_hash:
            return False
        
        # Reload
        logger.info(f"Config changed, reloading (old: {self._config_hash}, new: {new_hash})")
        new_config = Config.from_yaml(self._config_path)
        
        # Copy all attributes
        for attr in dir(new_config):
            if not attr.startswith('_') and not callable(getattr(new_config, attr)):
                setattr(self, attr, getattr(new_config, attr))
        
        self._config_hash = new_hash
        self._loaded_at = datetime.now().isoformat()
        _config_metrics['reload_count'] += 1
        
        return True
    
    def validate(self) -> List[str]:
        """
        Validate configuration with clear error messages.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        global _config_metrics
        errors = []
        
        # API key checks
        api_checks = [
            (self.matching.primary_provider == "gemini", 
             self.api_keys.gemini_api_key, 
             "GEMINI_API_KEY required for gemini matching"),
            (self.matching.secondary_provider == "anthropic",
             self.api_keys.anthropic_api_key,
             "ANTHROPIC_API_KEY required for anthropic matching (secondary)"),
            (self.embedding.provider == "gemini",
             self.api_keys.gemini_api_key,
             "GEMINI_API_KEY required for gemini embeddings"),
            (self.stock_footage.pexels_enabled,
             self.api_keys.pexels_api_key,
             "PEXELS_API_KEY required for Pexels stock footage"),
            (self.stock_footage.pixabay_enabled,
             self.api_keys.pixabay_api_key,
             "PIXABAY_API_KEY required for Pixabay stock footage"),
        ]
        
        for condition, key, message in api_checks:
            if condition and not key:
                errors.append(message)
        
        # Value range checks
        range_checks = [
            (0 <= self.matching.min_confidence <= 1,
             f"matching.min_confidence must be 0-1, got {self.matching.min_confidence}"),
            (self.matching.max_clip_reuse >= 0,
             f"matching.max_clip_reuse must be >= 0, got {self.matching.max_clip_reuse}"),
            (self.transcription.max_workers >= 1,
             f"transcription.max_workers must be >= 1, got {self.transcription.max_workers}"),
            (self.embedding.batch_size >= 1,
             f"embedding.batch_size must be >= 1, got {self.embedding.batch_size}"),
            (self.keyword.max_keywords >= 1,
             f"keyword.max_keywords must be >= 1, got {self.keyword.max_keywords}"),
        ]
        
        for valid, message in range_checks:
            if not valid:
                errors.append(message)
        
        _config_metrics['validation_errors'] += len(errors)
        
        return errors
    
    def get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get a nested config value by dot-notation path.
        
        Example:
            config.get_nested("matching.min_confidence")
            config.get_nested("duration_tiers.short.min_seconds")
        """
        parts = path.split(".")
        obj = self
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                if self.logging.log_config_access:
                    logger.debug(f"Config path not found: {path}")
                return default
        
        return obj


# =============================================================================
# GLOBAL CONFIG INSTANCE (Thread-Safe Singleton)
# =============================================================================

_global_config: Optional[Config] = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """Get the global config instance (creates default if not loaded)"""
    global _global_config, _config_metrics
    
    with _config_lock:
        if _global_config is None:
            _global_config = Config()
            _config_metrics['cache_misses'] += 1
            logger.warning("Using default config - call load_config() to load from file")
        else:
            _config_metrics['cache_hits'] += 1
        
        return _global_config


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load config from file and set as global instance.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Loaded Config object
    """
    global _global_config
    
    with _config_lock:
        _global_config = Config.from_yaml(config_path)
        
        # Validate and warn
        errors = _global_config.validate()
        for error in errors:
            logger.warning(f"Config validation: {error}")
        
        return _global_config


def set_config(config: Config):
    """Set the global config instance"""
    global _global_config
    
    with _config_lock:
        _global_config = config


def reload_config() -> bool:
    """Reload global config if file changed"""
    global _global_config
    
    with _config_lock:
        if _global_config:
            return _global_config.reload()
        return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider from config"""
    config = get_config()
    key_map = {
        "gemini": config.api_keys.gemini_api_key,
        "anthropic": config.api_keys.anthropic_api_key,
        "voyage": config.api_keys.voyage_api_key,
        "pexels": config.api_keys.pexels_api_key,
        "pixabay": config.api_keys.pixabay_api_key,
        "unsplash": config.api_keys.unsplash_api_key,
    }
    return key_map.get(provider.lower())


def ensure_dirs(config: Config = None):
    """Create necessary directories from config"""
    config = config or get_config()
    
    dirs = [
        config.cache.cache_dir,
        config.output.output_dir,
        config.downloading.output_dir,
        config.logging.log_dir,
        config.downloaded_videos_dir,
        config.otio_output_dir,
    ]
    
    for dir_path in dirs:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def log_hardcoded_warning(component: str, value_name: str, value: Any):
    """Log warning when a component uses hardcoded value"""
    config = get_config()
    if config.logging.warn_on_hardcoded:
        logger.warning(
            f"HARDCODED VALUE in {component}: {value_name}={value} - "
            f"Consider adding to config.yaml"
        )


# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================

# Allow import of individual config classes
__all__ = [
    'Config', 'load_config', 'get_config', 'set_config', 'reload_config',
    'ensure_dirs', 'get_api_key', 'get_config_metrics', 'log_hardcoded_warning',
    'TranscriptionConfig', 'EmbeddingConfig', 'MatchingConfig', 'OutputConfig',
    'KeywordConfig', 'DownloadingConfig', 'LoggingConfig', 'CacheConfig',
]