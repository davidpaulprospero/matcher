"""Config sections package - modular configuration organized by domain.

Each section file contains related configuration dataclasses:
- infrastructure.py: Logging, caching, pipeline, API keys
- core.py: Project, transcription, embedding, indexing
- matching.py: Matching engine, location matching, negative matching
- llm.py: LLM providers, retry logic, caching
- download.py: Download settings, audio-first mode, speech screening
- keywords.py: Keyword and entity extraction
- entity.py: Entity image search and caching
- duration.py: Duration tiers and stock footage
- output.py: Timeline generation, deduplication, variety
- media.py: Vision, scene detection, audio analysis
"""

from __future__ import annotations

# Infrastructure
from .infrastructure import (
    LoggingConfig,
    CacheConfig,
    GlobalCacheConfig,
    PipelineConfig,
    APIKeysConfig,
    HealingConfig,
    HealingLoggingConfig,
    WatcherConfig,
    LLMHealerConfig,
    ProxyConfig,
)

# Core
from .core import (
    ProjectConfig,
    PauseSplitConfig,
    TranscriptionConfig,
    EmbeddingConfig,
    IndexingConfig,
)

# Matching
from .matching import (
    LocationMatchingConfig,
    NegativeMatchingConfig,
    ChapterDetectionConfig,
    PremiseScoringConfig,
    HighMatchesModeConfig,
    MatchingConfig,
)

# LLM
from .llm import (
    LLMRetryConfig,
    LLMCacheConfig,
    LLMProviderConfig,
    LLMConfig,
)

# Download
from .download import (
    RemixConfig,
    ZeroDownloadRemixConfig,
    EnhancedFeaturesConfig,
    LLMTitleFilterConfig,
    AudioFirstConfig,
    CaptionFirstConfig,
    SpeechScreeningConfig,
    DownloadConfig,
    DownloadingConfig,
)

# Keywords
from .keywords import (
    ListDetectionConfig,
    KeywordConfig,
)

# Entity
from .entity import (
    StockVideoConfig,
    SilentVideoConfig,
    EntityCacheConfig,
    ImageSearchConfig,
)

# Duration tiers
from .duration import (
    DurationTierConfig,
    DurationTiersConfig,
    StockFootageConfig,
)

# Output generation
from .output import (
    DeduplicationConfig,
    VarietyConfig,
    OutputConfig,
    MultiStyleConfig,
)

# Media processing
from .media import (
    VisionConfig,
    SceneDetectionConfig,
    AudioAnalysisConfig,
)

# B-roll
from .broll import (
    BrollSourceBoostConfig,
    BrollConfig,
)

# Keyword mode (no voiceover)
from .keyword_mode import (
    MontageConfig,
    ScriptLLMConfig,
    ScriptConfig,
    CollectionConfig,
    KeywordModeConfig,
)

# Feedback system (rejection learning, channel scoring)
from .feedback import (
    ChannelCategoriesConfig,
    ChannelScoringConfig,
    CrossProjectConfig,
    FeedbackConfig,
)

__all__ = [
    # Infrastructure
    'LoggingConfig',
    'CacheConfig',
    'GlobalCacheConfig',
    'PipelineConfig',
    'APIKeysConfig',
    'HealingConfig',
    'HealingLoggingConfig',
    'WatcherConfig',
    'LLMHealerConfig',
    'ProxyConfig',
    # Core
    'ProjectConfig',
    'PauseSplitConfig',
    'TranscriptionConfig',
    'EmbeddingConfig',
    'IndexingConfig',
    # Matching
    'LocationMatchingConfig',
    'NegativeMatchingConfig',
    'ChapterDetectionConfig',
    'PremiseScoringConfig',
    'HighMatchesModeConfig',
    'MatchingConfig',
    # LLM
    'LLMRetryConfig',
    'LLMCacheConfig',
    'LLMProviderConfig',
    'LLMConfig',
    # Download
    'RemixConfig',
    'ZeroDownloadRemixConfig',
    'EnhancedFeaturesConfig',
    'LLMTitleFilterConfig',
    'AudioFirstConfig',
    'CaptionFirstConfig',
    'SpeechScreeningConfig',
    'DownloadConfig',
    'DownloadingConfig',
    # Keywords
    'ListDetectionConfig',
    'KeywordConfig',
    # Entity
    'StockVideoConfig',
    'SilentVideoConfig',
    'EntityCacheConfig',
    'ImageSearchConfig',
    # Duration
    'DurationTierConfig',
    'DurationTiersConfig',
    'StockFootageConfig',
    # Output
    'DeduplicationConfig',
    'VarietyConfig',
    'OutputConfig',
    'MultiStyleConfig',
    # Media
    'VisionConfig',
    'SceneDetectionConfig',
    'AudioAnalysisConfig',
    # B-roll
    'BrollSourceBoostConfig',
    'BrollConfig',
    # Keyword mode
    'MontageConfig',
    'ScriptLLMConfig',
    'ScriptConfig',
    'CollectionConfig',
    'KeywordModeConfig',
    # Feedback system
    'ChannelCategoriesConfig',
    'ChannelScoringConfig',
    'CrossProjectConfig',
    'FeedbackConfig',
]
