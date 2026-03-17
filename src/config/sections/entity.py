"""Entity configuration: Entity image search and caching.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = [
    'StockVideoConfig',
    'SilentVideoConfig',
    'EntityCacheConfig',
    'ImageSearchConfig',
]


@dataclass
class StockVideoConfig:
    """Configuration for stock video downloads."""
    min_duration: float = 3.0  # Minimum video duration in seconds
    max_duration: float = 30.0  # Maximum video duration in seconds
    prefer_hd: bool = True  # Prefer HD quality videos


@dataclass
class SilentVideoConfig:
    """Configuration for handling silent/B-roll videos.

    Silent videos (no speech detected) are valuable B-roll footage.
    This config controls how they're processed for matching:
    - Vision API: Extract frame descriptions using Gemini Vision
    - LLM fallback: Generate descriptions from title/keyword
    - B-roll boost: Increase match confidence for silent videos
    """
    enabled: bool = True  # Enable silent video handling
    min_words_threshold: int = 10  # Videos with fewer words are considered silent
    use_vision_api: bool = True  # Use Vision API to describe video frames
    use_llm_fallback: bool = True  # Fall back to LLM description from title/keyword
    cache_descriptions: bool = True  # Cache generated descriptions


@dataclass
class EntityCacheConfig:
    """Configuration for cross-project entity image caching.

    Enables sharing entity images (people, places, organizations) across
    projects. When searching for an entity, checks global cache first.
    """
    # Enable global caching
    enabled: bool = False

    # Global cache directory (shared across all projects)
    # Expands ~ to home directory
    cache_dir: str = "~/.matcher_entity_cache"

    # Fuzzy matching threshold for entity names (0.0-1.0)
    # 0.0 = exact match only, 1.0 = match anything
    # Recommended: 0.85 for reasonable fuzzy matching
    fuzzy_threshold: float = 0.85

    # Maximum age of cached images in days (0 = never expire)
    max_age_days: int = 0

    # How to use cached images in projects:
    # "copy" = copy to project folder (default, most portable)
    # "symlink" = create symlink to cache (saves space, but Windows issues)
    # "reference" = use absolute paths to cache (least portable)
    cache_strategy: str = "copy"


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

    # US-99-008: Number of entities to display in summary output (0 = no limit)
    entity_display_limit: int = 5

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

    # Cross-project entity image caching
    entity_cache: EntityCacheConfig = field(default_factory=EntityCacheConfig)

    # Entity matching settings (V9/V10 track gap control)
    enable_sticky_matching: bool = False  # Reuse last entity when no match (creates continuous blocks)
    semantic_match_threshold: float = 0.15  # Minimum word overlap for semantic match (0.0-1.0)

    def __post_init__(self):
        """Convert nested dicts to dataclasses if needed"""
        if isinstance(self.stock_video, dict):
            self.stock_video = StockVideoConfig(**self.stock_video)
        if isinstance(self.entity_cache, dict):
            self.entity_cache = EntityCacheConfig(**self.entity_cache)
