"""Infrastructure configuration: Logging, caching, pipeline, API keys.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    'LoggingConfig',
    'CacheConfig',
    'GlobalCacheConfig',
    'PipelineConfig',
    'APIKeysConfig',
]


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

    # Cross-project cache (legacy - use GlobalCacheConfig)
    cross_project_cache: bool = False
    cross_project_cache_dir: str = ""

    # Validation
    validate_cache_on_load: bool = True
    use_file_hash: bool = True


@dataclass
class GlobalCacheConfig:
    """Global cache settings for cross-project video reuse

    Enables sharing of video cache (transcripts, embeddings, scenes)
    across multiple projects. Videos from past projects can be reused
    if they match the current project's keywords/topics.
    """
    enabled: bool = True
    cache_dir: str = "~/.matcher_global_cache"  # Expands ~ to home dir

    # Pre-download optimization
    check_before_download: bool = True  # Query cache before downloading
    redownload_deleted: bool = True     # Re-download if cached video was deleted
    prompt_reuse: bool = True           # Ask user before reusing (false = auto-reuse)

    # Relevance thresholds
    min_keyword_similarity: float = 0.8  # Fuzzy match threshold for keywords
    min_topic_overlap: float = 0.3       # Topic relevance threshold

    # Limits
    max_reuse_videos: int = 50           # Max videos to reuse from cache per project
    max_redownload: int = 10             # Max deleted videos to re-download

    # What to share globally
    share_transcripts: bool = True
    share_embeddings: bool = True
    share_scenes: bool = True
    share_face_detection: bool = True

    # Priority boost for current project videos in matching
    current_project_boost: float = 0.1


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
