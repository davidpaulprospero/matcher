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
    QualityGatesConfig,
    PipelineConfig,
    MetricsExportConfig,
    RetryStrategyConfig,
    StageRetryConfig,
    StageTimeoutConfig,
    DriftRuleConfig,
    DriftRulesConfig,
    APIKeysConfig,
    HealingConfig,
    HealingLoggingConfig,
    WatcherConfig,
    LLMHealerConfig,
    CheckpointCompressionConfig,
    CheckpointAutoRepairConfig,
    CheckpointAutoCleanupConfig,
    CheckpointHotBackupConfig,
    CloudBackupConfig,
    UnifiedErrorAggregationConfig,
    CrossKeywordRetryLearningConfig,
    ErrorRateThresholdConfig,
    ErrorRateTrackingConfig,
    ValidationWebhookConfig,
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
    CaptionFirstConfig,
    AudioFirstConfig,
    SpeechScreeningConfig,
    DownloadConfig,
    DownloadingConfig,
    RateLimitPredictorConfig,
    AdaptiveBackoffConfig,
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

# Iterative matching
from .iterative_matching import (
    IterativeMatchingConfig,
)

# Rate limiting
from .rate_limit import (
    RateLimitConfig,
)

# Video search
from .video_search import (
    SearchBudgetConfig,
    PerKeywordCircuitBreakerConfig,
    VideoSearchConfig,
)

__all__ = [
    # Infrastructure
    'LoggingConfig',
    'CacheConfig',
    'GlobalCacheConfig',
    'QualityGatesConfig',
    'PipelineConfig',
    'MetricsExportConfig',
    'RetryStrategyConfig',
    'StageRetryConfig',
    'StageTimeoutConfig',
    'DriftRuleConfig',
    'DriftRulesConfig',
    'APIKeysConfig',
    'HealingConfig',
    'HealingLoggingConfig',
    'WatcherConfig',
    'LLMHealerConfig',
    'CheckpointCompressionConfig',
    'CheckpointAutoRepairConfig',
    'CheckpointAutoCleanupConfig',
    'CheckpointHotBackupConfig',
    'CloudBackupConfig',
    'UnifiedErrorAggregationConfig',
    'CrossKeywordRetryLearningConfig',
    'ErrorRateThresholdConfig',
    'ErrorRateTrackingConfig',
    'ValidationWebhookConfig',
    # Core
    'ProjectConfig',
    'PauseSplitConfig',
    'TranscriptionConfig',
    'EmbeddingConfig',
    'IndexingConfig',
    # Matching
    'LocationMatchingConfig',
    'NegativeMatchingConfig',
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
    'CaptionFirstConfig',
    'AudioFirstConfig',
    'SpeechScreeningConfig',
    'DownloadConfig',
    'DownloadingConfig',
    'RateLimitPredictorConfig',
    'AdaptiveBackoffConfig',
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
    # Iterative matching
    'IterativeMatchingConfig',
    # Rate limiting
    'RateLimitConfig',
    # Video search
    'SearchBudgetConfig',
    'PerKeywordCircuitBreakerConfig',
    'VideoSearchConfig',
]
