"""
Caption fetcher package.

Provides YouTube caption fetching functionality with caching, batch processing,
and quality metrics.

Usage:
    from src.caption import CaptionFetcher, CaptionResult, CaptionCache

    fetcher = CaptionFetcher()
    result = fetcher.fetch_captions("dQw4w9WgXcQ")
    for segment in result.segments:
        print(f"{segment.start_time:.2f} -> {segment.end_time:.2f}: {segment.text}")

    # With caching:
    cache = CaptionCache(config.download.caption_first)
    cached = cache.get("dQw4w9WgXcQ", "en")
    if cached:
        print(f"Cache hit: {len(cached.segments)} segments")
    else:
        result = fetcher.fetch_captions("dQw4w9WgXcQ")
        cache.store(result)
"""

from __future__ import annotations

# Enums
from .enums import CaptionErrorCategory, StreamState, DEFAULT_RETRY_BUDGETS

# Constants
from .constants import (
    ISO_639_1_CODES,
    VALID_CAPTION_FORMATS,
    DEFAULT_PREFERRED_FORMATS,
    DEFAULT_CAPTION_TIMEOUT,
    DEFAULT_RETRY_DELAY,
    DEFAULT_MAX_RETRIES,
)

# Exceptions
from .exceptions import (
    CaptionError,
    CaptionUnavailableError,
    CaptionFetchError,
    CaptionFormatUnavailableError,
    CaptionFormatExhaustedError,
    CaptionParseWarning,
    ConfigValidationError,
    ErrorPatternAbortError,
    CaptionNormalizationError,
)

# Models
from .models import (
    CaptionSegment,
    CaptionResult,
    SegmentQualityMetrics,
    TimingValidationResult,
    StreamStateResult,
    ParseResult,
    ErrorPatternResult,
    AvailableLanguage,
    NormalizationConfig,
    CaptionConfigValidationResult,
    TestFetchResult,
    TestFetchSummary,
)

# Quality functions
from .quality import (
    determine_caption_quality,
    calculate_segment_metrics,
)

# Stream state classification
from .stream_state import classify_stream_state

# Validators
from .validators import (
    is_valid_language_code,
    validate_language_config,
)

# Error handling
from .error_handling import (
    categorize_caption_error,
    ErrorPatternDetector,
    # US-100-005: New error classification features
    NetworkErrorSubtype,
    classify_network_error,
    ErrorSeverity,
    get_error_severity,
    get_error_retry_score,
    CategoryRetryBudget,
    CategoryBudgetManager,
    ErrorPatternLearner,
    ErrorClassificationMetrics,
    log_classification_decision,
)

# Retry budget
from .retry_budget import BatchRetryBudget

# Worker progress
from .worker_progress import (
    WorkerProgress,
    WorkerProgressTracker,
)

# Batch checkpoint
from .batch_checkpoint import CaptionBatchCheckpoint

# Timeout management
from .timeout import (
    TimeoutEscalationLevel,
    RateLimitState,
    RateLimitTracker,
    TimeoutEscalationPolicy,
    ProgressiveTimeoutManager,
    FormatTimeoutPolicy,
    StalledOperationDetector,
)

# Cache models
from .cache_models import (
    CachedCaption,
    CacheValidationResult,
    ChannelCaptionPattern,
    BatchPreCheckResult,
)

# Cache
from .cache import CaptionCache

# Enhanced cache (from cache_enhanced.py)
from .cache_enhanced import (
    CacheMetrics,
    EnhancedCaptionCache,
)

# Normalizer
from .normalizer import CaptionNormalizer, coerce_segments

# Parsers
from .parsers import (
    parse_timestamp,
    parse_vtt,
    parse_srt,
    parse_json3,
    parse_caption_content,
)

# Metrics
from .metrics import CaptionMetrics

# Batch processing
from .batch_processor import (
    BatchProcessor,
    BatchProcessorConfig,
    BatchResult,
    ProgressCallback,
)

__all__ = [
    # Enums
    'CaptionErrorCategory',
    'StreamState',
    'DEFAULT_RETRY_BUDGETS',
    # Constants
    'ISO_639_1_CODES',
    'VALID_CAPTION_FORMATS',
    'DEFAULT_PREFERRED_FORMATS',
    'DEFAULT_CAPTION_TIMEOUT',
    'DEFAULT_RETRY_DELAY',
    'DEFAULT_MAX_RETRIES',
    # Exceptions
    'CaptionError',
    'CaptionUnavailableError',
    'CaptionFetchError',
    'CaptionFormatUnavailableError',
    'CaptionFormatExhaustedError',
    'CaptionParseWarning',
    'ConfigValidationError',
    'ErrorPatternAbortError',
    'CaptionNormalizationError',
    # Models
    'CaptionSegment',
    'CaptionResult',
    'SegmentQualityMetrics',
    'TimingValidationResult',
    'StreamStateResult',
    'ParseResult',
    'ErrorPatternResult',
    'AvailableLanguage',
    'NormalizationConfig',
    'CaptionConfigValidationResult',
    'TestFetchResult',
    'TestFetchSummary',
    # Quality functions
    'determine_caption_quality',
    'calculate_segment_metrics',
    # Stream state
    'classify_stream_state',
    # Validators
    'is_valid_language_code',
    'validate_language_config',
    # Error handling
    'categorize_caption_error',
    'ErrorPatternDetector',
    # US-100-005: New error classification features
    'NetworkErrorSubtype',
    'classify_network_error',
    'ErrorSeverity',
    'get_error_severity',
    'get_error_retry_score',
    'CategoryRetryBudget',
    'CategoryBudgetManager',
    'ErrorPatternLearner',
    'ErrorClassificationMetrics',
    'log_classification_decision',
    # Retry budget
    'BatchRetryBudget',
    # Worker progress
    'WorkerProgress',
    'WorkerProgressTracker',
    # Batch checkpoint
    'CaptionBatchCheckpoint',
    # Timeout management
    'TimeoutEscalationLevel',
    'RateLimitState',
    'RateLimitTracker',
    'TimeoutEscalationPolicy',
    'ProgressiveTimeoutManager',
    'FormatTimeoutPolicy',
    'StalledOperationDetector',
    # Cache models
    'CachedCaption',
    'CacheValidationResult',
    'ChannelCaptionPattern',
    'BatchPreCheckResult',
    # Cache
    'CaptionCache',
    # Enhanced cache
    'CacheMetrics',
    'EnhancedCaptionCache',
    # Normalizer
    'CaptionNormalizer',
    'coerce_segments',
    # Parsers
    'parse_timestamp',
    'parse_vtt',
    'parse_srt',
    'parse_json3',
    'parse_caption_content',
    # Metrics
    'CaptionMetrics',
    # Batch processing
    'BatchProcessor',
    'BatchProcessorConfig',
    'BatchResult',
    'ProgressCallback',
]
