"""
Config package - Modular configuration management for matcher-alt pipeline.

Refactored from monolithic config.py (1,929 lines) into organized sections.
All section configs imported from sections/ subpackage.

Usage:
    from src.config import load_config, get_config

    # At startup
    config = load_config("config.yaml")

    # Anywhere in codebase
    config = get_config()
    threshold = config.matching.min_confidence
"""

from __future__ import annotations

# Main config class and functions
from .base import (
    Config,
    ConfigError,
    FrozenConfigError,
    CRITICAL_SECTIONS,
    load_config,
    get_config,
    set_config,
    reload_config,
    ensure_dirs,
    get_api_key,
    get_config_metrics,
    log_hardcoded_warning,
)
from .utils import safe_get_config_value

# Re-export section configs for backward compatibility
from .sections import (
    # Infrastructure
    LoggingConfig,
    CacheConfig,
    GlobalCacheConfig,
    PipelineConfig,
    APIKeysConfig,
    HealingConfig,
    # Core
    ProjectConfig,
    PauseSplitConfig,
    TranscriptionConfig,
    EmbeddingConfig,
    IndexingConfig,
    # Matching
    LocationMatchingConfig,
    NegativeMatchingConfig,
    MatchingConfig,
    # LLM
    LLMRetryConfig,
    LLMCacheConfig,
    LLMProviderConfig,
    LLMConfig,
    # Download
    RemixConfig,
    ZeroDownloadRemixConfig,
    EnhancedFeaturesConfig,
    LLMTitleFilterConfig,
    CaptionFirstConfig,
    AudioFirstConfig,
    SpeechScreeningConfig,
    DownloadConfig,
    DownloadingConfig,
    # Keywords
    ListDetectionConfig,
    KeywordConfig,
    # Entity
    StockVideoConfig,
    SilentVideoConfig,
    EntityCacheConfig,
    ImageSearchConfig,
    # Duration
    DurationTierConfig,
    DurationTiersConfig,
    StockFootageConfig,
    # Output
    DeduplicationConfig,
    VarietyConfig,
    OutputConfig,
    MultiStyleConfig,
    # Media
    VisionConfig,
    SceneDetectionConfig,
    AudioAnalysisConfig,
)

__all__ = [
    # Main config and functions
    'Config',
    'ConfigError',
    'FrozenConfigError',
    'CRITICAL_SECTIONS',
    'load_config',
    'get_config',
    'set_config',
    'reload_config',
    'ensure_dirs',
    'get_api_key',
    'get_config_metrics',
    'log_hardcoded_warning',
    'safe_get_config_value',
    # Infrastructure
    'LoggingConfig',
    'CacheConfig',
    'GlobalCacheConfig',
    'PipelineConfig',
    'APIKeysConfig',
    'HealingConfig',
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
]
