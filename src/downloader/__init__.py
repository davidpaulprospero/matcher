"""
Video downloader package - YouTube download orchestration.

Refactored from monolithic downloader.py (2,906 lines) into modular architecture:
- core.py: Streamlined VideoDownloader orchestrator (~1,100 lines)
- utils.py: Utility functions (sanitize_filename_for_nle, format_time, get_cookies_args)
- types.py: Dataclass definitions (MatchedSegment, MergedSegment, DownloadedSegment)
- segment_utils.py: Segment management for audio-first pipeline
- checkpoint.py: Checkpoint and duration tier management
- transcoding.py: FFmpeg transcoding and codec detection
- speech_screening.py: Whisper VAD speech detection
- title_filter.py: LLM-based title filtering (Rule 9 compliant)
- keyword_remix.py: Download search optimization (SearchOptimizer)
- audio_first.py: Audio-first download pipeline (Phases 1 & 3)

Total code reduction: 2,906 → 3,150 lines across 10 focused modules.
Original VideoDownloader (~2,388 lines) → streamlined core (~1,100 lines) = 54% reduction.

**100% Backward Compatible:** from src.downloader import VideoDownloader works identically.
"""

# Core class (NEW - Phase 9 complete)
from .core import VideoDownloader
from .orchestrator import DownloadOrchestrator, RateLimitHooks

# Dataclasses and exceptions (from types.py)
from .types import (
    MatchedSegment,
    MergedSegment,
    DownloadedSegment,
    DownloadError,
    EscalationTier,
    EscalationState
)

# Checkpoint management
from .checkpoint import (
    CheckpointManager,
    DownloadCheckpoint
)

# Specialized managers
from .transcoding import TranscodingManager
from .speech_screening import SpeechScreener
from .title_filter import TitleFilter
from .keyword_remix import SearchOptimizer
from .audio_first import AudioFirstPipeline
from .cookie_rotator import CookieRotator
from .impersonation import ImpersonationManager, ImpersonationStats
from .escalation_manager import (
    EscalationManager, EscalationResult, is_escalation_trigger,
    classify_trigger, _TRIGGER_CATEGORIES,
)
from .escalation_metrics import EscalationMetrics
from .escalation_strategy import EscalationStrategy, EscalationDecision
from .errors import (
    ClassifiedDownloadError,
    NetworkError,
    BotDetectionError,
    RateLimitError,
    FormatError,
    AuthenticationError,
    TimeoutError_,
)
from .error_classification import (
    classify_error_category,
    classify_error_severity,
    classify_network_subcategory,
    is_network_failure,
    is_escalation_error,
    ERROR_PATTERNS,
    NETWORK_ERROR_PATTERNS,
    NETWORK_FAILURE_PATTERNS,
    NETWORK_FAILURE_THRESHOLD,
    BOT_DETECTION_ABORT_THRESHOLD,
    ERROR_SEVERITY_PATTERNS,
    SEVERITY_MULTIPLIERS,
)
from .cookie_method_fallback import CookieMethodFallback
from .vpn_manager import VPNManager
from .speed_tracker import DownloadSpeedTracker, DownloadSpeedConfig, DownloadRecord, RateLimitSignal
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerBuilder
from .retry_queue import RetryQueue, BatchRetryConfig, RetryItem
from .retry_stats import RetryQueueStats
from .rate_limit_metrics import RateLimitMetrics, RateLimitMetricsAggregator

# Segment utilities (public helpers)
from . import segment_utils
from .segment_utils import (
    collect_matched_segments,
    merge_segments_with_buffer,
    prepare_merged_segments,
    get_segment_filename,
    _extract_video_id
)

# Utilities
from . import utils
from .utils import sanitize_filename_for_nle

# Import AudioDownload from state (for backward compatibility with old tests)
from ..state import AudioDownload

__all__ = [
    # Core class
    'VideoDownloader',
    'DownloadOrchestrator',
    'RateLimitHooks',

    # Dataclasses and exceptions
    'MatchedSegment',
    'MergedSegment',
    'DownloadedSegment',
    'DownloadCheckpoint',
    'DownloadError',
    'EscalationTier',
    'EscalationState',

    # Managers
    'CheckpointManager',
    'TranscodingManager',
    'SpeechScreener',
    'TitleFilter',
    'SearchOptimizer',
    'AudioFirstPipeline',
    'CookieRotator',
    'ImpersonationManager',
    'ImpersonationStats',
    'EscalationManager',
    'EscalationResult',
    'is_escalation_trigger',
    'classify_trigger',
    '_TRIGGER_CATEGORIES',
    'EscalationMetrics',
    'EscalationStrategy',
    'EscalationDecision',
    'CookieMethodFallback',
    'VPNManager',
    'DownloadSpeedTracker',
    'DownloadSpeedConfig',
    'DownloadRecord',
    'RateLimitSignal',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitBreakerBuilder',
    'RetryQueue',
    'BatchRetryConfig',
    'RetryItem',
    'RetryQueueStats',
    'RateLimitMetrics',
    'RateLimitMetricsAggregator',

    # Typed error hierarchy (US-82-002)
    'ClassifiedDownloadError',
    'NetworkError',
    'BotDetectionError',
    'RateLimitError',
    'FormatError',
    'AuthenticationError',
    'TimeoutError_',

    # Error classification (shared module)
    'classify_error_category',
    'classify_error_severity',
    'classify_network_subcategory',
    'is_network_failure',
    'is_escalation_error',
    'ERROR_PATTERNS',
    'NETWORK_ERROR_PATTERNS',
    'NETWORK_FAILURE_PATTERNS',
    'NETWORK_FAILURE_THRESHOLD',
    'BOT_DETECTION_ABORT_THRESHOLD',
    'ERROR_SEVERITY_PATTERNS',
    'SEVERITY_MULTIPLIERS',

    # Segment utilities
    'collect_matched_segments',
    'merge_segments_with_buffer',
    'prepare_merged_segments',
    'get_segment_filename',
    '_extract_video_id',

    # Utilities
    'sanitize_filename_for_nle',

    # Backward compatibility
    'AudioDownload',

    # Module exports
    'segment_utils',
    'utils',
]
