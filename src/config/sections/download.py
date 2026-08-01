"""Download configuration: Video download settings, audio-first mode, speech screening.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Dict, Optional

__all__ = [
    'RemixConfig',
    'ZeroDownloadRemixConfig',
    'EnhancedFeaturesConfig',
    'LLMTitleFilterConfig',
    'CaptionCircuitBreakerConfig',
    'CaptionRetryBudgetConfig',
    'DownloadRetryBudgetConfig',
    'SegmentValidationConfig',
    'ChecksumValidationConfig',
    'CaptionFirstConfig',
    'AudioFirstConfig',
    'SpeechScreeningConfig',
    'CookieRotationConfig',
    'RateLimitConfig',
    'RateLimitBudgetConfig',
    'SpeedTrackingConfig',
    'CircuitBreakerConfig',
    'BatchRetryConfig',
    'VPNConfig',
    'MullvadConfig',
    'YouTubeAPIConfig',
    'ImpersonationConfig',
    'ExtractorArgsConfig',
    'ErrorPatternsConfig',
    'ErrorHandlingMapping',
    'ErrorHandlingAction',
    'AdaptiveSeverityConfig',
    'BandwidthThrottleConfig',
    'DownloadResumeConfig',
    'CheckpointCompressionConfig',
    'FormatPreferenceConfig',
    'TierSlotManagementConfig',
    'DownloadConfig',
    'DownloadingConfig',
    'RegionBackoffConfig',
    '_get_region_from_country',
    '_get_ip_geolocation',
]


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

    def __post_init__(self):
        if not (0.0 <= self.min_relevance_score <= 1.0):
            raise ValueError(
                f"RemixConfig.min_relevance_score must be in range 0.0-1.0, got {self.min_relevance_score}"
            )
        if self.max_workers < 1:
            raise ValueError(
                f"RemixConfig.max_workers must be positive, got {self.max_workers}"
            )
        valid_auto_accept = ("filtered", "all", "prompt")
        if self.auto_accept_filter not in valid_auto_accept:
            raise ValueError(
                f"RemixConfig.auto_accept_filter must be one of {valid_auto_accept}, got '{self.auto_accept_filter}'"
            )


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

    def __post_init__(self):
        if self.max_retries < 1:
            raise ValueError(
                f"ZeroDownloadRemixConfig.max_retries must be positive, got {self.max_retries}"
            )
        if self.max_keywords_per_batch < 1:
            raise ValueError(
                f"ZeroDownloadRemixConfig.max_keywords_per_batch must be positive, got {self.max_keywords_per_batch}"
            )


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

    # Retry budget control (US-42-012)
    reset_budget: bool = False  # Reset retry budget counters on resume

    # Face preference for matching
    # Options: "neutral" (no preference), "more" (prefer faces), "none" (avoid faces)
    face_preference: str = "neutral"

    def __post_init__(self):
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"EnhancedFeaturesConfig.min_confidence must be in range 0.0-1.0, got {self.min_confidence}"
            )
        if self.max_retries < 1:
            raise ValueError(
                f"EnhancedFeaturesConfig.max_retries must be positive, got {self.max_retries}"
            )
        valid_face_preferences = ("neutral", "more", "none")
        if self.face_preference not in valid_face_preferences:
            raise ValueError(
                f"EnhancedFeaturesConfig.face_preference must be one of {valid_face_preferences}, got '{self.face_preference}'"
            )


@dataclass
class LLMTitleFilterConfig:
    """Config for LLM-based title filtering before download."""
    enabled: bool = True
    provider: str = "gemini"  # gemini or anthropic
    model: str = "gemini-2.5-flash"  # or claude-3-haiku-20240307
    batch_size: int = 20  # Check multiple titles at once
    min_relevance: float = 0.7  # 0-1, reject if below

    def __post_init__(self):
        if not (0.0 <= self.min_relevance <= 1.0):
            raise ValueError(
                f"LLMTitleFilterConfig.min_relevance must be in range 0.0-1.0, got {self.min_relevance}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"LLMTitleFilterConfig.batch_size must be positive, got {self.batch_size}"
            )


@dataclass
class CaptionCircuitBreakerConfig:
    """Circuit breaker for repeated caption fetch failures (US-33-009).

    When multiple consecutive caption fetches fail, the circuit breaker
    trips and pauses all fetches for a duration. This prevents hammering
    YouTube during rate limit windows.

    Example with defaults:
      - 5 fetches fail in a row → circuit trips
      - Wait 60 seconds before allowing new fetches
      - On next successful fetch → circuit resets to closed state

    Configure in config.yaml under download.caption_first.circuit_breaker.
    """
    # Enable/disable circuit breaker
    enabled: bool = True

    # Number of consecutive failures before circuit trips (opens)
    threshold: int = 5

    # Duration to pause after circuit trips (seconds)
    pause_seconds: float = 60.0

    # Maximum pause duration cap (seconds) to prevent runaway pause scaling
    max_pause_seconds: float = 300.0

    # Circuit breaker cascade (US-61-003): when enabled, failures propagate to
    # the download circuit breaker (and vice versa) to speed up coordinated pausing
    # when YouTube is rate-limiting. Default: True.
    circuit_breaker_cascade: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values at load time (US-66-007)."""
        if self.pause_seconds <= 0:
            raise ValueError(
                f"CaptionCircuitBreakerConfig.pause_seconds must be > 0, "
                f"got {self.pause_seconds}. "
                f"Check config.yaml under download.caption_first.circuit_breaker.pause_seconds"
            )

        if self.max_pause_seconds < self.pause_seconds:
            raise ValueError(
                f"CaptionCircuitBreakerConfig.max_pause_seconds must be >= pause_seconds, "
                f"got max_pause_seconds={self.max_pause_seconds} < pause_seconds={self.pause_seconds}. "
                f"Check config.yaml under download.caption_first.circuit_breaker.max_pause_seconds"
            )

        if self.threshold < 1:
            raise ValueError(
                f"CaptionCircuitBreakerConfig.threshold must be >= 1, "
                f"got {self.threshold}. "
                f"Check config.yaml under download.caption_first.circuit_breaker.threshold"
            )


@dataclass
class CaptionRetryBudgetConfig:
    """Retry budget for caption fetching across all videos (US-33-010).

    Tracks total attempts and backoff time used across the entire batch.
    When limits are exceeded, remaining videos are skipped and use
    transcription fallback.

    Configure in config.yaml under download.caption_first.retry_budget.
    """
    # Enable/disable retry budget tracking
    enabled: bool = True

    # Maximum total fetch attempts across all videos in batch
    # Includes both successful and failed attempts
    # Set to 0 for unlimited attempts
    max_attempts: int = 100

    # Maximum cumulative backoff time (seconds) before exhaustion
    # This prevents spending too much time waiting between retries
    # Set to 0 for unlimited backoff
    max_backoff_time_seconds: float = 300.0

    # Automatic scaling settings (US-37-004)
    # When enabled, max_attempts scales up based on batch size
    auto_scale: bool = True

    # Attempts per video multiplier for auto-scaling
    # 2.0 = 1 attempt + 1 retry per video average
    # Only scales UP when batch > max_attempts / attempts_per_video
    attempts_per_video: float = 2.0

    # VPN rotation on rate limit exhaustion (US-37-008, US-61-011)
    # When budget exhausts with >50% RATE_LIMIT errors, trigger VPN rotation
    # This resets the budget and retries remaining videos with a new IP
    # Only works if Mullvad VPN is enabled (download.mullvad.enabled: true)
    # Default: false - VPN rotation is opt-in behavior
    vpn_rotation_on_caption_exhaustion: bool = False

    # Maximum VPN-triggered budget resets per session (US-37-008)
    # Prevents infinite loops if VPN rotation doesn't help
    # After this many resets, budget exhaustion is final
    max_vpn_resets_per_session: int = 2

    # Early termination settings (US-37-009)
    # When success rate drops below threshold after min_sample videos,
    # terminate early to save time instead of continuing to fail

    # Minimum success rate threshold (0.0 to 1.0)
    # If success rate falls below this, terminate early
    # 0.3 = terminate if less than 30% of videos succeed
    min_success_rate: float = 0.3

    # Minimum videos processed before checking success rate
    # Prevents premature termination on small samples
    min_sample_for_early_termination: int = 20

    # Reset on scale-up settings (US-41-009)
    # When enabled, scaling up the budget also resets usage counters
    # Useful when restoring from checkpoint with prior attempts used
    # and the batch needs a larger budget than the checkpoint had
    reset_on_scale: bool = False

    # Budget warning threshold (US-100-011)
    # When budget consumption exceeds this threshold, emit warnings
    # Default: 0.8 (80%) - warn when 80% of budget is consumed
    budget_warning_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Validate configuration values at load time.

        US-41-004: Validate attempts_per_video > 0
        US-41-011: Validate minimum values to prevent misconfiguration
        """
        # US-41-011: max_attempts must be >= 10 (or 0 for unlimited)
        # Setting max_attempts=5 for 175 videos guarantees failure
        if self.max_attempts != 0 and self.max_attempts < 10:
            raise ValueError(
                f"CaptionRetryBudgetConfig.max_attempts must be >= 10 (or 0 for unlimited), "
                f"got {self.max_attempts}. A value like 5 for a batch of 175 videos guarantees failure. "
                f"Check config.yaml under download.caption_first.retry_budget.max_attempts"
            )

        # US-41-011: attempts_per_video must be >= 1.0 (need at least 1 attempt per video)
        if self.attempts_per_video < 1.0:
            raise ValueError(
                f"CaptionRetryBudgetConfig.attempts_per_video must be >= 1.0, got {self.attempts_per_video}. "
                f"Values below 1.0 would allow fewer attempts than videos in the batch. "
                f"Check config.yaml under download.caption_first.retry_budget.attempts_per_video"
            )

        # US-41-011: max_backoff_time_seconds must be >= 30.0 (or 0 for unlimited)
        # 30s minimum gives enough time for reasonable retry delays
        if self.max_backoff_time_seconds != 0 and self.max_backoff_time_seconds < 30.0:
            raise ValueError(
                f"CaptionRetryBudgetConfig.max_backoff_time_seconds must be >= 30.0 (or 0 for unlimited), "
                f"got {self.max_backoff_time_seconds}. Values below 30s don't allow meaningful backoff delays. "
                f"Check config.yaml under download.caption_first.retry_budget.max_backoff_time_seconds"
            )

        # US-100-011: budget_warning_threshold must be between 0.5 and 1.0
        if not (0.5 <= self.budget_warning_threshold <= 1.0):
            raise ValueError(
                f"CaptionRetryBudgetConfig.budget_warning_threshold must be between 0.5 and 1.0, "
                f"got {self.budget_warning_threshold}. This threshold controls when to emit budget warnings. "
                f"Check config.yaml under download.caption_first.retry_budget.budget_warning_threshold"
            )


@dataclass
class CaptionFirstConfig:
    """Caption-first mode configuration.

    Caption fetching is ALWAYS enabled - YouTube captions are fetched before
    video download for faster matching with lower bandwidth. Videos without
    captions fall back to the TRANSCRIBE stage (Whisper).

    Benefits:
    - Faster: No need to download/process audio for transcription
    - Lower bandwidth: Only downloads video segments after matching
    - Quality indicators: Tracks human vs auto-generated captions

    Retry behavior (US-008):
    - Network errors trigger exponential backoff retries
    - CaptionUnavailableError (no captions exist) is NOT retried
    - CaptionFetchError (temporary failure) IS retried
    - After max_retries, video is marked for transcription fallback

    Live stream detection (US-002):
    - Detects live/was_live videos before caption fetch
    - Skips caption fetch for live streams to prevent hangs
    - Tracks skipped live streams in metrics separately
    """
    # DEPRECATED: Caption-first is now always enabled. This field is ignored.
    # Kept for backward compatibility with existing config files.
    enabled: bool = True

    # Fall back to Whisper transcription when captions unavailable
    fallback_to_transcription: bool = True

    # Preferred caption language (ISO 639-1 code)
    # Fallback chain: preferred_language -> fallback_languages -> 'en' -> any available
    preferred_language: str = "en"

    # Configurable language fallback chain (US-003)
    # List of ISO 639-1 codes to try after preferred_language fails
    # Example for multilingual projects: ["es", "pt", "fr"]
    # Empty list = use default behavior (preferred_language -> en -> any)
    fallback_languages: List[str] = field(default_factory=list)

    # Language priority chain for caption track selection (US-78-009)
    # Ordered list of language codes to try before falling back to auto-generated captions.
    # YouTube videos often have regional variants (en-US, en-GB) but not plain 'en'.
    # The fetcher tries each code in order; when a non-primary (index > 0) language is
    # used, language_confidence is reduced (0.8 for 2nd choice, 0.6 for 3rd+).
    # Empty list = use preferred_language only (existing behavior).
    language_priority: List[str] = field(default_factory=lambda: ['en', 'en-US', 'en-GB'])

    # Weighted language preferences (US-100-007)
    # Dictionary mapping language codes to preference weights (0.0-1.0).
    # When weights are provided, language selection uses weighted scoring instead of
    # strict ordering. Higher weights = higher preference.
    # Example: {"en": 1.0, "es": 0.8, "fr": 0.6}
    # Empty dict = use language_priority ordering (backward compatible).
    language_priority_weights: Dict[str, float] = field(default_factory=dict)

    # Language fallback strategy (US-100-007)
    # Strategy for selecting the best language when multiple options are available:
    # - 'sequential': Try languages in priority order, use first available (default)
    # - 'coverage_first': Prefer language with highest coverage ratio for the video
    # - 'confidence_weighted': Score languages by combining priority weight and coverage
    language_fallback_strategy: str = "sequential"

    # Multi-language caption aggregation (US-100-007)
    # When enabled, attempt to combine captions from multiple languages if available.
    # This can improve coverage for multilingual content by merging captions from
    # different language tracks. Only combines if primary language is incomplete.
    enable_multi_language_aggregation: bool = False

    # Language coverage metrics tracking (US-100-007)
    # When enabled, track per-language coverage statistics and selection metrics.
    # Metrics include: per_language_coverage, selected_language, fallback_count.
    language_coverage_metrics_enabled: bool = True

    # Timeout for caption fetch requests (seconds)
    timeout: int = 30

    # Prefer human-uploaded captions over auto-generated
    prefer_human_captions: bool = True

    # Cache captions for cross-project reuse
    cache_captions: bool = True

    # Cache directory for cross-project caption reuse
    # Supports ~ expansion (default: ~/.matcher_caption_cache)
    cache_dir: str = "~/.matcher_caption_cache"

    # Maximum cache age in days (0 = no expiration)
    # Captions older than this will be re-fetched
    max_cache_age_days: int = 30

    # Maximum number of entries in caption cache (US-90-010)
    # When set > 0, cache uses LRU eviction to maintain max entry count
    # Default 0 = no limit (cache grows indefinitely based on max_cache_age_days)
    max_cache_size: int = 0

    # Retry settings for caption fetch failures (US-008)
    # Max retries on network/temporary errors (CaptionFetchError)
    # CaptionUnavailableError (no captions exist) is NOT retried
    max_retries: int = 3

    # Base delay between retries in seconds (exponential backoff)
    # Actual delays: retry_delay, retry_delay*2, retry_delay*4, ...
    retry_delay: float = 2.0

    # Parallel caption fetching (US-001)
    # Number of concurrent caption fetches (1 = sequential)
    # Higher values speed up projects with 50+ videos
    max_parallel_fetches: int = 4

    # Adaptive worker count strategy (US-100-006)
    # Strategy for determining worker count based on batch size:
    # - 'static': Use max_parallel_fetches directly (default, backward compatible)
    # - 'adaptive': Scale workers based on batch size (more workers for larger batches)
    # - 'cpu_count': Use CPU count as worker count (up to max_parallel_fetches)
    worker_count_strategy: str = "static"

    # Minimum workers for adaptive mode
    # Used when worker_count_strategy is 'adaptive'
    min_workers: int = 4

    # Maximum workers for adaptive mode
    # Used when worker_count_strategy is 'adaptive', caps scaling at this limit
    max_workers_limit: int = 8

    # Batch size threshold for adaptive scaling (US-100-006)
    # When batch_size > this threshold, workers scale up from min_workers to max_workers_limit
    # Linear interpolation between min and max for batch sizes between 0 and threshold*2
    batch_size_threshold: int = 100

    # Rate limit feedback loop (US-100-006)
    # When enabled, reduce workers dynamically when rate limit errors increase
    # This helps avoid triggering rate limits by backing off concurrency
    rate_limit_feedback_enabled: bool = True

    # Rate limit error threshold for worker reduction (US-100-006)
    # When rate limit error rate exceeds this threshold (0.0-1.0), reduce workers
    rate_limit_error_threshold: float = 0.15

    # Live stream detection (US-002)
    # Skip caption fetch for live streams to prevent hangs
    # Live streams can hang indefinitely during caption fetch
    skip_live_streams: bool = True

    # Upcoming stream handling (US-007 Sprint 8)
    # How to handle UPCOMING and PREMIERE videos (scheduled but not yet live/available):
    # - 'skip': Skip these videos entirely (same as live streams)
    # - 'queue': Add to pending_streams list for processing later
    # - 'check_later': Skip for now but don't mark as failed (can retry next run)
    # UPCOMING streams will have captions available once they complete/premiere.
    handle_upcoming: str = "skip"

    # US-153-008: Prefer YouTube Data API for caption availability checks
    # When enabled, uses captions.list API to check if captions exist before
    # falling back to yt-dlp. Can significantly reduce yt-dlp subprocess calls.
    prefer_api_captions: bool = True

    # US-153-008: Cache TTL for caption availability results from API (hours)
    # Cached results avoid repeated API calls for the same video.
    # Default 24 hours - balance between freshness and API quota savings.
    caption_availability_cache_ttl_hours: int = 24

    # US-154-002: Fetch full captions via YouTube Data API
    # When enabled, fetches actual caption content via captions.list API instead of yt-dlp.
    # Default: False - disabled due to higher API quota cost (captions.download = 50 units vs yt-dlp = 0 cost).
    # Enable this for projects with limited yt-dlp access (frequent 403s/blocks) but sufficient API quota.
    fetch_captions_via_api: bool = False

    # Coverage threshold (US-004)
    # Minimum coverage ratio (0.0-1.0) for caption quality
    # Videos below this threshold are flagged for potential transcription fallback
    # High coverage = better matching accuracy
    min_coverage_threshold: float = 0.5

    # Coverage quality calculation weights (US-100-003)
    # These control how coverage quality is calculated:
    # segment_density_weight: Weight for content density (longer segments = higher quality)
    #   Higher values favor longer, more complete segments
    # gap_penalty_weight: Weight for silence/gap detection
    #   Higher values penalize captions with excessive pauses
    # confidence_weight: Weight for coverage confidence (video duration vs caption length)
    #   Higher values penalize captions that don't match video duration
    # These must sum to <=1.0 (remaining weight goes to legacy coverage_ratio)
    segment_density_weight: float = 0.5
    gap_penalty_weight: float = 0.3
    confidence_weight: float = 0.2

    # Gap threshold for coverage calculation (US-100-003)
    # Minimum gap duration in seconds to count as a "pause" in captions
    # Gaps larger than this contribute to the gap_penalty
    # Default 2.0 seconds - typical for sentence transitions
    coverage_gap_threshold: float = 2.0

    # Format preference (US-006)
    # Order of subtitle formats to try when fetching captions
    # Available formats: json3, srv3, vtt, srt
    # json3 is preferred by default since it's already structured (no conversion needed)
    # Falls back to next format on parse error or unavailability
    preferred_formats: List[str] = field(default_factory=lambda: ["json3", "vtt", "srt"])

    # Adaptive format ordering (US-002 Sprint 7)
    # When enabled, reorders formats based on historical success rates from previous runs.
    # Formats with higher success rates are tried first, reducing fetch latency.
    # Success rates are persisted in caption cache metadata for cross-run learning.
    # When disabled, uses static preferred_formats order.
    adaptive_format_order: bool = True

    # Caption availability pre-check (US-008)
    # When enabled, checks if captions exist before attempting fetch
    # This avoids wasted network calls for videos without captions
    # The check uses --list-subs which is faster than downloading captions
    pre_check_availability: bool = True

    # US-153-008: Prefer YouTube Data API for caption availability checks
    # When enabled, uses captions.list API to check caption availability before yt-dlp.
    # This is faster and more reliable than yt-dlp --list-subs.
    # Falls back to yt-dlp if API is unavailable or quota exceeded.
    prefer_api_captions: bool = True

    # US-154-002: Use API caption track info to skip yt-dlp --list-subs call
    # When enabled, uses captions.list API to get track IDs, then passes them to
    # yt-dlp --write-sub to skip the expensive --list-subs subprocess call.
    # This saves ~3 seconds per video by avoiding one subprocess call.
    # Default: False (off due to quota cost - each captions.list call uses quota)
    # Note: Requires prefer_api_captions=True to have any effect.
    use_api_for_caption_fetch: bool = False

    # US-153-008: Caption availability cache TTL in seconds
    # How long to cache caption availability results (both available and unavailable).
    # Default 24 hours (86400s) - captions don't change frequently.
    # Set to 0 to disable caching (always check fresh).
    availability_cache_ttl_seconds: int = 86400

    # Timing validation epsilon (US-007 Sprint 6)
    # Tolerance in milliseconds for floating-point precision at video duration boundary.
    # Captions ending within this epsilon of video duration are treated as valid.
    # Example: caption at 299.999s in 300s video is valid with 100ms epsilon.
    # Set to 0 for exact matching (may cause false positives from float precision).
    timing_epsilon_ms: float = 100.0

    # Timing validation mode (US-100-008)
    # Controls segment timing validation behavior:
    #   'strict': Validate gaps/overlaps between segments, reject on issues
    #   'lenient': Validate but only log warnings (default)
    #   'off': Skip all segment timing validation (fastest)
    # This validates segment continuity, not video duration boundaries.
    timing_validation_mode: str = "lenient"

    # Segment continuity validation threshold (US-100-008)
    # Minimum gap duration in seconds to count as a timing issue.
    # Gaps larger than this are flagged in continuity validation.
    # Only used when timing_validation_mode is 'strict' or 'lenient'.
    continuity_gap_threshold: float = 1.0

    # Overlap tolerance in seconds for segment validation (US-100-008)
    # Segments overlapping by more than this amount are flagged.
    # Only used when timing_validation_mode is 'strict' or 'lenient'.
    overlap_tolerance: float = 0.1

    # Cross-project cache validation (US-008 Sprint 6)
    # Validates cached caption data integrity before use.
    # Modes:
    #   'strict': Reject cache if segment_count differs >20% from expected (based on duration)
    #   'warn': Log warning and re-fetch on validation failure (default)
    #   'skip': Skip validation entirely (fastest, but risks corrupt data)
    # Validation checks: video_id match, language match, segment_count consistency
    cache_validation: str = "warn"

    # Cache validation segment count tolerance (US-008 Sprint 6)
    # Maximum allowed deviation in segment count as a ratio (0.0-1.0).
    # Default 0.2 means segment count can differ by up to 20% from expected.
    # Expected segments = duration_seconds / 3 (typical caption segment is ~3 seconds)
    cache_validation_tolerance: float = 0.2

    # Per-category retry budgets (US-003 Sprint 7)
    # Maps error category names to maximum retry attempts.
    # Categories: network, timeout, parse, unavailable, rate_limit
    # Network errors should retry aggressively; parse errors rarely succeed on retry.
    # Set to 0 to never retry a category.
    # Example log: "NETWORK error, retry 2/3" vs "PARSE error, no retry (budget: 1)"
    retry_budgets: Dict[str, int] = field(default_factory=lambda: {
        "network": 3,      # Network connectivity issues, DNS failures
        "timeout": 2,      # Request/connection timeouts
        "parse": 1,        # Caption content parsing failures (unlikely to succeed)
        "unavailable": 0,  # No captions exist - never retry
        "rate_limit": 2,   # API rate limiting, retry with backoff
    })

    # Batch pre-check by channel (US-006 Sprint 7)
    # Groups videos by YouTube channel and uses representative samples to determine
    # caption availability for the entire channel, reducing API calls.
    # For example, 50 videos from the same channel might only need 5 actual checks
    # if the channel pattern indicates >90% confidence.
    # Disable for maximum accuracy at the cost of more API calls.
    batch_precheck_by_channel: bool = True

    # Minimum confidence (0.0-1.0) for channel pattern to skip individual checks (US-006 Sprint 7)
    # When a channel has >= confidence_threshold success rate (or <= 1-threshold failure rate)
    # after min_samples_for_confidence videos, remaining videos skip pre-check.
    # Higher = more conservative (fewer skips), lower = more aggressive (more skips)
    batch_precheck_confidence: float = 0.9

    # Minimum videos checked per channel before trusting the pattern (US-006 Sprint 7)
    # Until this many videos have been checked for a channel, all are individually checked.
    batch_precheck_min_samples: int = 5

    # Sample size per channel when building the pattern (US-006 Sprint 7)
    # When a channel doesn't have high confidence yet, check this many videos
    # before applying the pattern to remaining videos in the batch.
    batch_precheck_sample_size: int = 5

    # Enable hierarchical clustering for channel grouping (US-100-010)
    # When enabled, similar channels (based on caption availability patterns) are
    # grouped together, allowing pattern sharing across related channels.
    # Disable for traditional per-channel behavior.
    precheck_clustering_enabled: bool = False

    # Minimum cluster size for hierarchical clustering (US-100-010)
    # Channels with fewer than this many videos are not clustered separately.
    # Higher values = fewer clusters, more aggressive sharing.
    precheck_min_cluster_size: int = 3

    # Prioritize fetch order by channel success rate (US-009 Sprint 7)
    # When enabled, videos from channels with higher caption availability are
    # fetched first. This improves average success rate early in the batch and
    # helps identify problematic channels faster.
    #
    # Requires channel patterns to be populated from previous runs or batch pre-check.
    # If no channel patterns are available, falls back to original video order.
    #
    # Example: Channels with 100% success fetched before channels with 50% success.
    prioritize_by_channel: bool = True

    # Error pattern detection and early abort (US-007 Sprint 7)
    # When 30%+ of the first N videos fail with the same error, detect the pattern
    # and take action based on this setting. Useful for detecting geoblocking,
    # API restrictions, or other systemic issues early.
    #
    # Modes:
    #   'abort': Stop batch fetch immediately with clear error message
    #   'warn': Log warning with pattern details, continue fetching (default)
    #   'skip': Disable pattern detection entirely
    #
    # Example log: "Pattern detected: 403 Forbidden (8/10 videos) - possible geoblocking"
    abort_on_error_pattern: str = "warn"

    # Threshold for triggering error pattern detection (US-007 Sprint 7)
    # Ratio of videos that must fail with the same error to trigger detection.
    # Default 0.3 = 30% of first sample_size videos must fail with same error.
    error_pattern_threshold: float = 0.3

    # Sample size for error pattern detection (US-007 Sprint 7)
    # Number of videos to check before evaluating error patterns.
    # Lower = faster detection, higher = more confidence in pattern.
    error_pattern_sample_size: int = 10

    # Batch progress reporting interval (US-66-005)
    # Log batch-level progress every N videos processed.
    # Lower values provide more frequent updates; higher values reduce log noise.
    progress_report_interval: int = 25

    # Unavailable early termination threshold (US-66-005)
    # When this fraction of a rolling window of recent videos lack captions,
    # switch remaining videos to preflight-only mode (lighter check).
    # 0.8 = switch when 80% of recent videos are unavailable.
    unavailable_threshold: float = 0.8

    # Rate limit backoff parameters (US-66-006)
    # Controls exponential backoff behavior for rate-limited caption fetches.
    # Used by RateLimitState in src/caption/timeout.py.
    rate_limit_base_backoff_seconds: float = 5.0      # Initial backoff delay (1st: 5s, 2nd: 10s, 3rd: 20s...)
    rate_limit_max_backoff_seconds: float = 300.0      # Maximum backoff cap (5 minutes)
    rate_limit_max_history: int = 10                   # Recent rate limit events to track

    # Worker-level progress tracking (US-008 Sprint 8)
    # Time threshold in seconds for considering a worker "stuck" on a video.
    # Workers exceeding this threshold are reported in progress callbacks.
    # Set higher for slow networks or videos with many caption tracks.
    stuck_worker_threshold: float = 60.0

    # Adaptive request spacing (US-33-004)
    # Minimum interval between caption fetch requests in milliseconds.
    # Increased automatically when error rate is high, decreased when stable.
    # This helps avoid triggering rate limits during batch processing.
    min_request_interval_ms: int = 500

    # Circuit breaker for repeated failures (US-33-009)
    # Pauses caption fetching when too many consecutive failures occur
    circuit_breaker: CaptionCircuitBreakerConfig = field(default_factory=CaptionCircuitBreakerConfig)

    # Retry budget tracking across all videos in batch (US-33-010)
    # Tracks total attempts and backoff time; skips remaining when exhausted
    retry_budget: CaptionRetryBudgetConfig = field(default_factory=CaptionRetryBudgetConfig)

    # Global rate limit coordinator integration (US-34-002)
    # When enabled, acquires slots from GlobalRateLimitCoordinator before each fetch
    # This provides unified rate limiting across caption fetching, downloading, and API calls
    # Disable to use the existing per-stage rate limiting only
    use_global_coordinator: bool = True

    # Per-format timeout overrides (US-59-005)
    # Maps subtitle format names to base timeout in seconds.
    # Used by FormatTimeoutPolicy for format-specific timeouts instead of flat timeout.
    # Defaults (in FormatTimeoutPolicy): json3=45s, srv3=30s, vtt=25s, srt=25s
    # Set to empty dict {} to use FormatTimeoutPolicy defaults.
    format_timeouts: Dict[str, float] = field(default_factory=dict)

    # Total format timeout budget (US-67-003)
    # Maximum total wall-clock seconds allowed across ALL format attempts per video.
    # When cumulative time exceeds this budget, remaining formats are skipped.
    # This prevents a video with no captions from wasting up to 95s (45+25+25)
    # trying every format before giving up. Default 60s caps total format time.
    # Set to 0 to disable the budget (unlimited time across formats).
    total_format_timeout_seconds: float = 60.0

    # Allow auto-generated caption fallback (US-59-007)
    # When True, _fetch_subtitle will retry with auto_generated=True if manual
    # captions are unavailable. When False, only manual captions are accepted.
    allow_auto_generated: bool = True

    # Negative cache TTL in hours (US-60-004) - DEPRECATED, use negative_cache_ttl_seconds
    # Cached "captions unavailable" entries expire after this TTL.
    # Shorter than max_cache_age_days since caption availability may change
    # (e.g., creator enables captions later). Default 1 hour.
    # Set to 0 to use max_cache_age_days for negative entries too.
    negative_cache_ttl_hours: float = 1.0

    # Negative cache TTL in seconds (US-63-005) - preferred over negative_cache_ttl_hours
    # When a video is found to have no captions, this result is cached to avoid
    # repeated expensive lookups. Default: 3600 seconds (1 hour).
    # If set, this takes precedence over negative_cache_ttl_hours.
    negative_cache_ttl_seconds: int = 3600

    # Per-category negative cache TTL (US-90-003):
    # Different TTLs for different types of negative cache entries:
    # - unavailable_ttl: For videos confirmed to have no captions (longer TTL ok)
    # - error_ttl: For transient errors (network, timeout) - shorter TTL for faster retry
    unavailable_ttl_seconds: int = 3600  # Default 1 hour - captions unlikely to appear soon
    error_ttl_seconds: int = 300  # Default 5 minutes - transient errors should retry faster

    # Video metadata language detection (US-100-002)
    # Detect video language from metadata (title, description, tags) before caption fetch.
    # This can help prioritize fetch order or skip pre-check for high-confidence predictions.
    enable_language_detection: bool = True

    # Confidence threshold (0.0-1.0) for language detection to skip pre-check (US-100-002)
    # When detected language confidence >= this threshold, skip pre-check availability
    # and go directly to fetching captions in that language.
    language_detection_confidence_threshold: float = 0.8

    # Minimum confidence (0.0-1.0) for language detection to affect fetch order (US-100-002)
    # Only videos with confidence >= this threshold are prioritized by detected language.
    language_detection_min_confidence: float = 0.6

    # Metrics export configuration (US-100-009)
    # Enable export of caption fetch metrics to JSON file
    metrics_export_enabled: bool = True

    # Path for metrics export (relative to project directory)
    # Supports ~ expansion (default: .cache/caption_metrics.json)
    metrics_export_path: str = ".cache/caption_metrics.json"

    # Number of metric runs to retain (US-100-009)
    # When auto_cleanup is enabled, older metric files are deleted
    # Default 10 = keep last 10 runs
    metrics_retention_runs: int = 10

    # Auto-cleanup old metric files (US-100-009)
    # When enabled, automatically delete metric files beyond retention limit
    metrics_auto_cleanup: bool = True

    def __post_init__(self):
        """Convert nested dicts to proper dataclass instances."""
        if isinstance(self.circuit_breaker, dict):
            self.circuit_breaker = CaptionCircuitBreakerConfig(**self.circuit_breaker)
        if isinstance(self.retry_budget, dict):
            self.retry_budget = CaptionRetryBudgetConfig(**self.retry_budget)
        # US-100-003: Validate coverage weights sum to <= 1.0
        weight_sum = self.segment_density_weight + self.gap_penalty_weight + self.confidence_weight
        if weight_sum > 1.0:
            raise ValueError(
                f"CaptionFirstConfig coverage weights must sum to <= 1.0, "
                f"got {weight_sum}: segment_density_weight={self.segment_density_weight}, "
                f"gap_penalty_weight={self.gap_penalty_weight}, confidence_weight={self.confidence_weight}"
            )
        if self.coverage_gap_threshold < 0:
            raise ValueError(
                f"CaptionFirstConfig.coverage_gap_threshold must be >= 0, got {self.coverage_gap_threshold}"
            )
        # US-100-006: Validate worker count strategy
        valid_strategies = ("static", "adaptive", "cpu_count")
        if self.worker_count_strategy not in valid_strategies:
            raise ValueError(
                f"CaptionFirstConfig.worker_count_strategy must be one of {valid_strategies}, "
                f"got '{self.worker_count_strategy}'"
            )
        if self.min_workers < 1:
            raise ValueError(
                f"CaptionFirstConfig.min_workers must be >= 1, got {self.min_workers}"
            )
        if self.max_workers_limit < self.min_workers:
            raise ValueError(
                f"CaptionFirstConfig.max_workers_limit must be >= min_workers, "
                f"got {self.max_workers_limit} < {self.min_workers}"
            )
        if not 0.0 <= self.rate_limit_error_threshold <= 1.0:
            raise ValueError(
                f"CaptionFirstConfig.rate_limit_error_threshold must be between 0.0 and 1.0, "
                f"got {self.rate_limit_error_threshold}"
            )
        # US-100-009: Validate metrics export config
        if self.metrics_retention_runs < 1:
            raise ValueError(
                f"CaptionFirstConfig.metrics_retention_runs must be >= 1, got {self.metrics_retention_runs}"
            )


@dataclass
class AudioFirstConfig:
    """Audio-first download pipeline configuration.

    When enabled, downloads audio (MP3) first for fast transcription/matching,
    then downloads only the matched video segments. This dramatically reduces
    download time and storage usage.

    Configure in config.yaml under download.audio_first.
    """
    # Enable/disable audio-first mode
    enabled: bool = False  # Disabled by default, enable per-project

    # Buffer around matched segments (seconds)
    # Adds padding before/after each match for editing flexibility
    buffer_seconds: float = 30.0

    # Merge segments if gap is smaller than this (seconds)
    # Reduces number of download requests
    merge_gap_seconds: float = 15.0

    # Audio quality for transcription (0=best, 9=worst)
    # 5 is ~128kbps, good enough for speech recognition
    audio_quality: int = 5

    # If segment download fails, download full video as fallback
    fallback_full_video: bool = True

    # Download alternatives from V2-V10 (not just V1 primary)
    download_all_tracks: bool = False

    # Keep audio files after video download (useful for re-matching)
    delete_audio_after_video: bool = False

    # Checkpoint between phases for resume capability
    checkpoint_phases: bool = True


@dataclass
class SpeechScreeningConfig:
    """Pre-screen videos by transcribing first N seconds to detect speech.

    Filters out videos with talking heads, commentary, or voiceover intros
    to ensure only true B-roll footage (no speech) gets downloaded.

    Only applied to configured tiers (default: long/longer videos 10+ min)
    since those are more likely to have speech intros and take longer to download.
    """
    # Enable/disable speech screening
    enabled: bool = False

    # Duration to check (seconds from start of video)
    screening_duration: float = 5.0

    # Minimum speech duration to trigger rejection (seconds)
    # Brief sounds/clicks under this threshold are ignored
    min_speech_duration: float = 0.5

    # If True, reject videos with speech; if False, just log detection
    reject_with_speech: bool = True

    # Whisper model size for screening (smaller = faster)
    # Options: tiny, base, small, medium, large
    whisper_model: str = "base"

    # Timeout per video for screening (seconds)
    # Bypasses the normal download timeout for quick screening
    timeout_per_video: int = 30

    # Fallback behavior on screening errors: "accept" or "reject"
    fallback_on_error: str = "accept"

    # Only apply to these duration tiers (skip short/medium for speed)
    tiers: List[str] = field(default_factory=lambda: ["long", "longer"])


@dataclass
class CookieRotationConfig:
    """Cookie rotation for YouTube rate limit evasion.

    When YouTube returns 429 (rate limit) or requires sign-in, rotate to
    a different cookie file. Each cookie file should be exported from a
    different browser profile or account.

    Cookie files should be in Netscape format (cookies.txt).
    Export using: "Get cookies.txt LOCALLY" extension or similar.
    """
    # Enable/disable cookie rotation
    enabled: bool = False

    # List of cookie file paths to rotate through
    # Example: ["cookies/main.txt", "cookies/backup1.txt", "cookies/backup2.txt"]
    cookie_files: List[str] = field(default_factory=list)

    # Rotation strategy:
    # - "on_error": Only rotate when hitting rate limit or sign-in errors
    # - "round_robin": Rotate after each download batch (proactive)
    # - "random": Randomly select cookie on each error
    rotation_strategy: str = "on_error"

    # Error patterns that trigger cookie rotation (case-insensitive matching)
    # Add custom patterns to match new YouTube error formats
    rotate_on_errors: List[str] = field(default_factory=lambda: [
        "429",
        "rate limit",
        "too many requests",
        "403",
        "forbidden",
        "sign in",
        "login required",
        "confirm your age",
        "bot detection",
    ])

    # Cooldown before reusing a rotated-out cookie (seconds)
    # Gives YouTube time to "forget" the rate limit
    cooldown_seconds: int = 300  # 5 minutes

    # Maximum rotations before giving up (0 = unlimited)
    max_rotations_per_session: int = 0

    # Health-based rotation (US-93-010): success rate threshold for auto-removal
    # Cookies with success rate below this threshold after health_min_attempts
    # will be automatically removed from rotation
    # Set to 0 to disable auto-removal
    success_rate_threshold: float = 0.3

    # Minimum attempts before evaluating health for auto-removal
    # Prevents premature removal of cookies that just started with bad luck
    health_min_attempts: int = 5

    # Maximum consecutive failures before auto-removal
    # Cookies failing this many times in a row are removed regardless of success rate
    max_consecutive_failures: int = 3

    # Cookie expiration handling (US-113-011)
    # Hours before cookie expiration to trigger warning/rotation
    # Cookies approaching expiration will be rotated proactively
    # Set to 0 to disable expiration checking
    cookie_expiry_warning_threshold_hours: int = 24

    # Enable proactive cookie rotation before expiration
    # When True, cookies approaching expiration threshold will be rotated
    rotate_before_expiry: bool = True

    # US-136-005: Proactive rotation threshold
    # Minimum hours before expiry to trigger proactive rotation
    # Only rotates proactively when cookie expires within this time AND
    # a fresher alternative is available
    # Set to 0 to use cookie_expiry_warning_threshold_hours as default
    proactive_rotation_threshold_hours: int = 0

    # Cookie pool health monitoring (US-114-006)
    # Minimum success rate threshold for health warnings
    # Cookies with success rate below this will trigger warnings but not auto-retire
    # Use this for monitoring while success_rate_threshold controls auto-retirement
    cookie_min_success_rate: float = 0.6

    # Health check interval (number of downloads between health checks)
    # Set to 0 to check on every download (more responsive but higher overhead)
    cookie_health_check_interval: int = 10

    # Browser-based cookie extraction (US-143-005)
    # List of browser profiles to extract cookies from
    # Each entry: {"browser": "chrome", "profile": "Default", "domain": ".youtube.com"}
    # Supported browsers: chrome, firefox, edge, safari, opera, brave
    browser_profiles: List[Dict[str, str]] = field(default_factory=list)

    # Browser priority order for cookie extraction
    # Higher priority browsers are tried first
    # Example: ["chrome", "firefox", "edge"]
    browser_priority_order: List[str] = field(default_factory=lambda: [
        "chrome", "firefox", "edge", "brave", "opera", "safari"
    ])

    # Auto-extract cookies from browsers on startup
    # When True, attempts to extract cookies from configured browser_profiles
    # and adds them to the cookie rotation pool
    auto_extract_cookies: bool = False

    # Cookie freshness validation before use
    # When True, validates cookie is not expired before each use
    # Expired cookies are automatically rotated
    cookie_freshness_validation: bool = True

    # Auto-refresh expired cookies from browser
    # When True, attempts to re-extract cookies from browser when current cookie expires
    auto_refresh_cookies: bool = True

    # Maximum time (seconds) to wait for browser cookie extraction
    browser_extraction_timeout: int = 30

    # Freshness threshold for extracted cookies (US-144-005)
    # If extracted cookie is older than this many minutes, trigger auto-refresh
    # Default: 5 minutes - ensures extracted cookies are still fresh
    freshness_threshold_minutes: int = 5

    def __post_init__(self):
        """Validate configuration values."""
        # Ensure rotate_on_errors is a non-empty list when rotation is enabled
        if self.enabled and not self.rotate_on_errors:
            raise ValueError(
                "cookie_rotation.rotate_on_errors must be a non-empty list when "
                "cookie rotation is enabled. Add at least one error pattern like '429'."
            )

        # Validate health-based rotation settings (US-93-010)
        if not (0.0 <= self.success_rate_threshold <= 1.0):
            raise ValueError(
                f"cookie_rotation.success_rate_threshold must be between 0.0 and 1.0, "
                f"got {self.success_rate_threshold}"
            )

        if self.health_min_attempts < 1:
            raise ValueError(
                f"cookie_rotation.health_min_attempts must be >= 1, "
                f"got {self.health_min_attempts}"
            )

        if self.max_consecutive_failures < 1:
            raise ValueError(
                f"cookie_rotation.max_consecutive_failures must be >= 1, "
                f"got {self.max_consecutive_failures}"
            )

        # Validate cookie expiration settings (US-113-011)
        if self.cookie_expiry_warning_threshold_hours < 0:
            raise ValueError(
                f"cookie_rotation.cookie_expiry_warning_threshold_hours must be >= 0, "
                f"got {self.cookie_expiry_warning_threshold_hours}"
            )

        # Validate cookie health monitoring settings (US-114-006)
        if not (0.0 <= self.cookie_min_success_rate <= 1.0):
            raise ValueError(
                f"cookie_rotation.cookie_min_success_rate must be between 0.0 and 1.0, "
                f"got {self.cookie_min_success_rate}"
            )

        if self.cookie_health_check_interval < 0:
            raise ValueError(
                f"cookie_rotation.cookie_health_check_interval must be >= 0, "
                f"got {self.cookie_health_check_interval}"
            )

        # Validate browser profile settings (US-143-005)
        if self.browser_extraction_timeout < 1:
            raise ValueError(
                f"cookie_rotation.browser_extraction_timeout must be >= 1, "
                f"got {self.browser_extraction_timeout}"
            )

        # Validate freshness threshold (US-144-005)
        if self.freshness_threshold_minutes < 1:
            raise ValueError(
                f"cookie_rotation.freshness_threshold_minutes must be >= 1, "
                f"got {self.freshness_threshold_minutes}"
            )


@dataclass
class RateLimitConfig:
    """Progressive backoff configuration for rate limit handling.

    When rate limit errors occur, apply exponential backoff BEFORE
    escalating to cookie rotation or VPN switching. This handles
    brief rate-limit windows without exhausting cookies.

    Backoff sequence example (with defaults):
      1st rate limit: wait 5s
      2nd rate limit: wait 10s
      3rd rate limit: wait 20s
      4th rate limit: wait 25s (capped at 60s total, rotate cookie)

    Cross-session cooldown:
      Rate limit events are saved to checkpoint with timestamp. On resume,
      if last rate limit was within cooldown period, the session starts
      with aggressive recovery mode (longer delays, faster escalation).

    Per-tier isolation:
      When downloading multiple tiers (short, medium, long, longer), rate limit
      state can be tracked separately per tier. This prevents a rate limit on
      one tier from affecting backoff counters for other tiers.

    Cross-keyword budget sharing:
      When downloading multiple keywords, rate limit recovery resources
      (rotations, VPN switches, backoff time) can be tracked across all
      keywords. If keyword A exhausts all cookie rotations, keyword B
      skips directly to VPN switching instead of trying rotations again.

    Adaptive backoff multiplier (US-008):
      When adaptive_multiplier is enabled, the backoff multiplier is adjusted
      based on the severity of the error. More severe errors (quota exceeded)
      use higher multipliers than mild errors (brief rate limit).
      Severity levels: low (1.5x), medium (2.0x), high (3.0x)
    """
    # Initial backoff delay on first rate limit error (seconds)
    initial_backoff_seconds: float = 5.0

    # Maximum cumulative delay before rotating cookie (seconds)
    # After this much total delay, escalate to cookie rotation
    max_backoff_before_rotate: float = 60.0

    # Backoff multiplier (exponential growth)
    backoff_multiplier: float = 2.0

    # Cross-session cooldown: minutes to wait before assuming rate limit cleared
    # If resuming within this period, start with recovery mode
    resume_cooldown_minutes: float = 15.0

    # Per-tier isolation: track rate limit state separately for each duration tier
    # When True, rate limit on 'long' tier won't affect 'short' tier backoff
    per_tier_isolation: bool = True

    # Cross-keyword budget sharing: track rate limit recovery resources across keywords
    # When True, if keyword A exhausts rotations, keyword B skips directly to VPN
    share_budget_across_keywords: bool = True

    # Maximum total backoff time budget per session (seconds)
    # After this much total backoff, skip backoff and escalate immediately
    # 0 = unlimited
    max_backoff_budget: float = 300.0  # 5 minutes

    # Adaptive backoff multiplier: adjust multiplier based on error severity (US-008)
    # When True, backoff_multiplier is adjusted based on error patterns:
    #   - low severity (brief rate limit): 1.5x multiplier
    #   - medium severity (too many requests): 2.0x multiplier (default)
    #   - high severity (quota exceeded, bot detection): 3.0x multiplier
    adaptive_multiplier: bool = True


@dataclass
class RateLimitBudgetConfig:
    """Cross-keyword rate limit budget configuration.

    Controls shared resource limits for rate limit recovery across all keywords
    in a download session. When one keyword exhausts a resource (e.g., cookie
    rotations), other keywords see the reduced budget and can skip to the next
    escalation level immediately.

    Example: If max_rotations=10 and keyword A uses 10 rotations, keyword B
    sees can_rotate()=False and skips directly to VPN switching.

    Tier-Isolated Budgets (US-109-007):
    - When tier_isolation_enabled=True, each escalation tier has independent budget
    - Cross-tier borrowing: exhausted tier can borrow from other tiers with unused capacity
    - Budget priority: tier4 (VPN) is last to borrow, tier1 is first to lend
    """
    # Enable cross-keyword budget sharing
    # When False, each keyword manages its own rate limit state independently
    enabled: bool = True

    # Maximum cookie rotations per session (0 = unlimited)
    # After this many rotations, skip directly to VPN switching
    max_rotations: int = 10

    # Maximum total backoff time per session (seconds)
    # After this much delay, skip backoff and escalate immediately
    # 0 = unlimited
    max_backoff_time: float = 600.0  # 10 minutes

    # Maximum VPN switches per session (0 = unlimited)
    # After this many switches, mark budget as exhausted
    max_vpn_switches: int = 3

    # Auto-scale budget limits based on keyword count
    # When True, budget limits scale with ceil(keyword_count / 5), capped at 5x
    # This ensures larger keyword sets have proportionally larger recovery budgets
    auto_scale_budget: bool = True

    # Tier-isolated budgets (US-109-007)
    # When True, each escalation tier has independent retry budget with cross-tier borrowing
    # Tiers: tier1=impersonate, tier2=extractor-args, tier3=cookie, tier4=VPN
    tier_isolation_enabled: bool = False

    # Tier-specific attempt limits (only used when tier_isolation_enabled=True)
    # Map of tier name to max attempts. If not specified, uses defaults:
    # tier1: 5, tier2: 5, tier3: 5, tier4: 3
    tier_max_attempts: Optional[Dict[str, int]] = None

    # US-123-007: Cross-session state persistence
    # Enable saving/loading rate limit state across pipeline sessions
    state_persistence_enabled: bool = False

    # Reset state entries older than this threshold (seconds)
    # State older than this will be skipped during restoration
    stale_state_threshold: float = 3600.0  # 1 hour default


@dataclass
class RateLimitPredictorConfig:
    """Rate limit predictor configuration for proactive budget adjustment (US-113-003).

    When enabled, the RateLimitPredictor analyzes historical rate limit patterns
    to predict the likelihood of rate limits occurring at the current time.
    Based on prediction, budgets are proactively adjusted:
    - likelihood > 0.6: Increase budget allocation by 1.5x
    - likelihood > 0.8: Increase budget allocation by 2.0x

    This allows the system to be more aggressive with recovery budgets during
    historically problematic time windows.

    US-143-004 Enhancements:
    - Hour-of-day granularity (24 bins) for fine-grained predictions
    - Weekend vs weekday separate tracking
    - Configurable prediction sensitivity
    """
    # Enable/disable rate limit prediction
    enabled: bool = True

    # Maximum days to keep in historical pattern database
    max_history_days: int = 30

    # Threshold for triggering budget increase (0.0-1.0)
    budget_increase_threshold: float = 0.6

    # Multiplier for budget increase when likelihood > threshold
    budget_increase_multiplier: float = 1.5

    # Higher threshold for maximum budget boost
    high_risk_threshold: float = 0.8

    # Maximum multiplier for high-risk scenarios
    max_budget_multiplier: float = 2.0

    # US-143-004: Prediction sensitivity (0.0-1.0)
    # Higher values give more weight to hourly patterns
    # Lower values give more weight to time window patterns
    sensitivity: float = 0.5

    # US-143-004: Minimum hourly events for reliable hourly prediction
    min_hourly_events: int = 10

    # US-143-004: Enable weekend vs weekday separate tracking
    enable_weekend_tracking: bool = True


@dataclass
class AdaptiveBackoffConfig:
    """Adaptive backoff configuration based on time-of-day patterns (US-114-004).

    This config enables tracking download success rates by hour of day and applying
    longer backoff during historically low-success hours. For example, US night hours
    (typically 0:00-6:00 UTC) often have higher rate limiting from YouTube.

    The multiplier is applied to the base backoff duration:
    - Higher multiplier (up to time_of_day_multiplier_range[1]) during low-success hours
    - Lower multiplier (down to time_of_day_multiplier_range[0]) during high-success hours

    US-136-009 Enhancements:
    - More granular hour bins (configurable bin size)
    - Weekend vs weekday differentiation
    - Exponential moving average for smoother transitions
    """
    # Enable/disable adaptive time-of-day backoff
    enabled: bool = True

    # Hour of day (0-23 UTC) multipliers map: hour -> success_rate (0.0-1.0)
    # YouTube tends to have more aggressive rate limiting during US night hours
    # These are historical defaults that get updated dynamically as data is collected
    hourly_success_rates: Optional[Dict[int, float]] = None

    # Multiplier range applied to backoff based on time-of-day
    # [min_multiplier, max_multiplier]: e.g., [1.0, 2.0] means 1x during peak hours,
    # up to 2x during low-success hours
    time_of_day_multiplier_range: Tuple[float, float] = (1.0, 2.0)

    # Hours considered "low success" (UTC) - default is US night (0-6 UTC)
    # When historical data shows success rate below this threshold, apply max multiplier
    low_success_hour_threshold: float = 0.5

    # Hours considered "high success" (UTC) - default is US daytime (14-22 UTC)
    # When historical data shows success rate above this threshold, apply min multiplier
    high_success_hour_threshold: float = 0.8

    # Maximum age of historical data before it's considered stale (hours)
    # After this, the data is reset to avoid using outdated patterns
    max_data_age_hours: int = 24

    # Minimum samples required before applying adaptive backoff
    # Avoids overfitting to small sample sizes early in a session
    min_samples_per_hour: int = 5

    # US-136-009: Use more granular hour bins instead of individual hours
    # When True, groups hours into bins (e.g., 2-hour bins: 0-1, 2-3, etc.)
    use_granular_bins: bool = True

    # Size of hour bins when use_granular_bins is True
    # Valid values: 1, 2, 3, 4, 6, 8, 12, 24
    # 2 = 12 bins per day (2-hour periods)
    # 4 = 6 bins per day (4-hour periods)
    bin_size_hours: int = 2

    # Enable weekend vs weekday differentiation
    # When True, tracks and applies separate multipliers for weekends and weekdays
    enable_weekend_diff: bool = True

    # Exponential moving average alpha for success rate smoothing
    # Higher values (closer to 1.0) give more weight to recent observations
    # Lower values (closer to 0.0) give more weight to historical data
    # Recommended range: 0.1 - 0.5
    ema_alpha: float = 0.3

    # Additional multiplier boost applied during weekends
    # 0.2 = 20% additional backoff during weekends
    weekend_multiplier_boost: float = 0.2

    def __post_init__(self) -> None:
        # Convert dict to dict if needed (YAML loads as dict)
        if self.hourly_success_rates is not None and isinstance(self.hourly_success_rates, dict):
            # Keep as-is, it's already a dict
            pass

        # Validate multiplier range
        min_mult, max_mult = self.time_of_day_multiplier_range
        if min_mult < 1.0 or max_mult < 1.0:
            raise ValueError(
                f"AdaptiveBackoffConfig.time_of_day_multiplier_range must have values >= 1.0, "
                f"got ({min_mult}, {max_mult})"
            )
        if min_mult > max_mult:
            raise ValueError(
                f"AdaptiveBackoffConfig.time_of_day_multiplier_range min ({min_mult}) "
                f"must be <= max ({max_mult})"
            )

        # US-136-009: Validate bin size
        valid_bin_sizes = {1, 2, 3, 4, 6, 8, 12, 24}
        if self.bin_size_hours not in valid_bin_sizes:
            raise ValueError(
                f"AdaptiveBackoffConfig.bin_size_hours must be one of {valid_bin_sizes}, "
                f"got {self.bin_size_hours}"
            )

        # Validate EMA alpha
        if not 0 < self.ema_alpha <= 1:
            raise ValueError(
                f"AdaptiveBackoffConfig.ema_alpha must be between 0 and 1, got {self.ema_alpha}"
            )


@dataclass
class RegionBackoffConfig:
    """Region-specific backoff multiplier configuration (US-109-011).

    Different geographic regions have different rate limit tolerance from YouTube.
    This config allows applying region-specific backoff multipliers when the
    current region is known (via VPN rotation country or IP geolocation).

    Multipliers are applied as a factor to the pause duration:
    - US: 1.0 (baseline)
    - EU: 1.2 (20% longer backoff)
    - ASIA: 1.5 (50% longer backoff)
    - OTHER: 2.0 (100% longer backoff)

    Region mapping uses ISO 3166-1 alpha-2 country codes:
    - US -> US
    - CA -> US
    - GB, DE, NL, SE, CH, FR, IT, ES, PL, BE, AT, IE, NO, FI, DK -> EU
    - JP, SG, KR, IN, ID, MY, TH, VN, PH, TW, HK -> ASIA
    - All others -> OTHER

    Usage:
      - VPN rotation automatically detects region from country code
      - IP geolocation can be used when not using VPN
      - Region-specific tracking maintains separate rate limit counters per region
    """
    # Enable region-specific backoff multipliers
    enabled: bool = False

    # Multiplier for US region (baseline = 1.0)
    us_multiplier: float = 1.0

    # Multiplier for European regions (default 1.2 = 20% longer)
    eu_multiplier: float = 1.2

    # Multiplier for Asian regions (default 1.5 = 50% longer)
    asia_multiplier: float = 1.5

    # Multiplier for other/unknown regions (default 2.0 = 100% longer)
    other_multiplier: float = 2.0

    # Enable regional rate limit tracking (separate counters per region)
    # When True, each region maintains independent rate limit counters
    track_per_region: bool = False

    # US-113-008: Dynamic region-based backoff
    # Minimum success rate threshold for a region (default 0.7 = 70%)
    # Regions with success rate below this threshold will have increased backoff
    # and may be avoided during VPN rotation
    region_success_rate_threshold: float = 0.7

    # Enable dynamic adjustment of backoff multipliers based on regional success rates
    # When True, regions with poor success rates get higher multipliers
    dynamic_region_adjustment: bool = False


def _get_region_from_country(country_code: str) -> str:
    """Map ISO country code to region for backoff multiplier.

    Args:
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'us', 'de')

    Returns:
        Region name: 'us', 'eu', 'asia', or 'other'
    """
    if not country_code:
        return 'other'

    country = country_code.lower()

    # US/Canada
    if country in ('us', 'ca'):
        return 'us'

    # European countries
    eu_countries = {
        'gb', 'de', 'nl', 'se', 'ch', 'fr', 'it', 'es', 'pl', 'be',
        'at', 'ie', 'no', 'fi', 'dk', 'pt', 'el', 'cz', 'hu', 'ro',
        'bg', 'sk', 'si', 'hr', 'lt', 'lv', 'ee', 'cy', 'lu', 'mt'
    }
    if country in eu_countries:
        return 'eu'

    # Asian countries
    asia_countries = {
        'jp', 'sg', 'kr', 'in', 'id', 'my', 'th', 'vn', 'ph', 'tw',
        'hk', 'cn', 'pk', 'bd', 'lk', 'np', 'mm', 'kh', 'la', 'mm'
    }
    if country in asia_countries:
        return 'asia'

    return 'other'


def _get_ip_geolocation(ip_address: Optional[str] = None) -> Optional[str]:
    """Get IP geolocation (country code) for the given IP address.

    Uses ip-api.com free service to look up country code.
    This is used as a fallback when VPN is not available.

    Args:
        ip_address: IP address to look up. If None, looks up current IP.

    Returns:
        ISO country code (e.g., 'us', 'de') or None if lookup fails.
    """
    import json
    import urllib.request
    import urllib.error

    if ip_address is None:
        # Look up current public IP
        url = "http://ip-api.com/json/"
    else:
        # Look up specific IP
        url = f"http://ip-api.com/json/{ip_address}"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data.get('status') == 'success':
                return data.get('countryCode', '').lower()
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        pass

    return None


@dataclass
class SpeedTrackingConfig:
    """Download speed monitoring for adaptive timeouts and rate limit detection.

    Tracks actual download speeds and adjusts timeouts dynamically based on
    network conditions. On slow networks, timeouts can be extended up to
    max_timeout_multiplier to prevent unnecessary timeout failures.

    Additionally, monitors for rate limit signals - sudden speed drops that
    often precede hard rate limit failures. When detected, triggers preemptive
    backoff via circuit breaker integration.

    How it works:
      1. After each download, records bytes downloaded and time taken
      2. Maintains a sliding window of the last N downloads (window_size)
      3. Calculates average speed across the window
      4. If speed < min_speed_mbps, extends timeouts proportionally
      5. If speed < rate_limit_signal_threshold for 3+ consecutive samples,
         emit a rate limit signal for preemptive backoff
      6. State is persisted in checkpoint for resume scenarios

    Example with defaults (min_speed_mbps=1.0, max_timeout_multiplier=2.0):
      - Network at 2.0 MB/s: no adjustment
      - Network at 0.5 MB/s: timeout extended 2x
      - Network at 0.25 MB/s: timeout extended 2x (capped)

    Rate limit signal detection (rate_limit_signal_threshold=0.1):
      - 3+ consecutive downloads below 0.1 MB/s suggests throttling
      - Triggers preemptive backoff before hard rate limit error
    """
    # Enable/disable speed tracking
    enabled: bool = True

    # Number of downloads to track in sliding window (running average)
    # Smaller = more responsive, larger = more stable
    # Default 10 provides good balance between responsiveness and stability
    window_size: int = 10

    # Minimum expected speed in MB/s
    # Below this, timeouts start getting extended
    min_speed_mbps: float = 1.0

    # Maximum timeout extension multiplier
    # 2.0 means timeout can be at most doubled
    max_timeout_multiplier: float = 2.0

    # Enable adaptive timeout (use speed data to extend timeouts)
    # If False, speeds are tracked but timeouts are not adjusted
    enable_adaptive_timeout: bool = True

    # Safety factor for adaptive timeout calculation (US-61-004)
    # When calculating timeout from estimated_size / avg_speed, multiply by this factor
    # Higher values provide more buffer for network variability
    # Default 1.5 gives 50% buffer over theoretical minimum
    safety_factor: float = 1.5

    # Rate limit signal threshold in MB/s
    # When speed drops below this for consecutive samples, signals potential rate limiting
    # Default 0.1 MB/s (100 KB/s) - near-stalled downloads indicate throttling
    rate_limit_signal_threshold: float = 0.1

    # Consecutive slow samples before emitting rate limit signal
    # Requires this many samples below threshold to trigger signal
    consecutive_slow_samples: int = 3

    # Variance detection configuration (US-93-012)
    # Coefficient of variation above which network is considered flaky
    # CV = std_dev / mean; above 0.5 (50%) = flaky network
    variance_threshold: float = 0.5

    # Speed in MB/s below which to warn about slow download
    # Individual download warnings help identify problematic videos
    slow_download_warning_threshold: float = 0.5

    # Enable variance detection for network issue identification
    # When enabled, detects flaky vs slow-but-consistent networks
    enable_variance_detection: bool = True


@dataclass
class SelfRegulateConfig:
    """Configuration for self-regulating download speed based on error patterns.

    This feature prevents rate limits by proactively throttling download concurrency
    when error patterns indicate YouTube is starting to throttle requests.

    Example:
      - 3 rate limit errors in 10 minutes → reduce concurrency by throttle_factor (0.5)
      - If errors continue → further reduce concurrency
      - When errors subside → auto_recover_speed gradually restores concurrency
    """
    # Enable/disable self-regulation of download speed
    enabled: bool = True

    # Number of errors within the window to trigger throttling
    # When errors exceed this threshold, reduce download concurrency
    # Default: 3 errors in rate_limit_prevention_window_seconds
    rate_limit_prevention_threshold: int = 3

    # Time window in seconds to track errors for throttling
    # Default: 600 seconds (10 minutes)
    rate_limit_prevention_window_seconds: int = 600

    # Factor to reduce concurrency by when throttling triggers
    # 0.5 = reduce by 50% (e.g., 4 → 2, 2 → 1)
    # Must be between 0.1 and 0.9
    throttle_factor: float = 0.5

    # Minimum concurrent downloads to maintain even when heavily throttled
    # Prevents complete stoppage; default: 1
    min_concurrent: int = 1

    # Time in seconds to wait before attempting to recover speed
    # After this period without errors, gradually restore concurrency
    # Default: 300 seconds (5 minutes)
    recovery_window_seconds: int = 300

    # Factor to increase concurrency by when recovering (gradual recovery)
    # 0.25 = increase by 25% each recovery step
    # Must be between 0.1 and 0.5
    recovery_factor: float = 0.25

    # Maximum concurrency to restore to (usually = parallel_workers)
    # Default: 0 means use the original max_concurrent value
    max_recovery_concurrency: int = 0

    def __post_init__(self):
        """Validate configuration values."""
        if self.throttle_factor < 0.1 or self.throttle_factor > 0.9:
            raise ValueError(
                f"SelfRegulateConfig.throttle_factor={self.throttle_factor} must be between 0.1 and 0.9"
            )
        if self.recovery_factor < 0.1 or self.recovery_factor > 0.5:
            raise ValueError(
                f"SelfRegulateConfig.recovery_factor={self.recovery_factor} must be between 0.1 and 0.5"
            )
        if self.min_concurrent < 1:
            raise ValueError(
                f"SelfRegulateConfig.min_concurrent={self.min_concurrent} must be >= 1"
            )


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker for repeated search failures.

    Implements the circuit breaker pattern: after a threshold of consecutive
    search failures, pauses all searches for a duration. This prevents
    hammering YouTube during rate limit windows or outages.

    Example with defaults:
      - 5 searches fail in a row → circuit trips
      - Wait 60 seconds before allowing new searches
      - On next successful search → circuit resets to closed state

    Download retry coordination:
      When block_download_retries is enabled (default), the download retry loop
      in _run_download_cmd will check the circuit breaker state before each retry
      attempt. If the circuit breaker is tripped during a retry sequence, the
      retry will wait for the circuit breaker to recover before continuing.
      The retry count is preserved across circuit breaker pauses.
    """
    # Enable/disable circuit breaker
    enabled: bool = True

    # Number of consecutive failures before circuit trips (opens)
    consecutive_failures_threshold: int = 5

    # Duration to pause after circuit trips (seconds)
    pause_seconds: float = 60.0

    # Block download retries when circuit breaker is tripped
    # When true, download retry loop waits for circuit breaker recovery
    block_download_retries: bool = True

    # Maximum pause duration cap (seconds) to prevent runaway pause scaling
    max_pause_seconds: float = 300.0

    # Circuit breaker cascade (US-61-003): when enabled, failures propagate to
    # the caption circuit breaker (and vice versa) to speed up coordinated pausing
    # when YouTube is rate-limiting. Default: True.
    circuit_breaker_cascade: bool = True

    # US-113-004: Enable circuit breaker state persistence across runs
    # When true, circuit breaker state (is_open, failure counts, timestamps)
    # is saved to checkpoint and restored on pipeline resume.
    # Default: True.
    persist_state: bool = True


@dataclass
class CascadeRuleConfig:
    """A cascade rule for multi-circuit coordination (US-89-009).

    Defines how one circuit breaker affects another when events occur.
    """
    source: str = ""  # Source circuit breaker name (e.g., "search", "caption")
    target: str = ""  # Target circuit breaker name (e.g., "download")
    on_trip: bool = True  # Propagate trip to target
    on_failure: bool = False  # Propagate failure count to target


@dataclass
class CircuitCoordinationConfig:
    """Multi-circuit coordination configuration (US-89-009).

    Configures cross-circuit trip propagation between multiple circuit breakers
    in the pipeline. When one circuit trips, it can trigger trips in other
    circuits based on configured cascade rules.

    Default cascade rules:
    - search -> caption: trip on failure (existing US-61-003)
    - caption -> search: trip on failure (existing US-61-003)
    - search -> download: trip on trip (NEW)
    """
    # Enable/disable multi-circuit coordination
    enabled: bool = True

    # Cascade rules for cross-circuit propagation
    cascade_rules: List[CascadeRuleConfig] = field(default_factory=list)


@dataclass
class BatchRetryConfig:
    """Batch-level retry queue for rate-limited videos.

    When rate limiting affects multiple videos in a batch, collect them
    and retry the entire batch after a delay. This is more effective than
    individual retries because it allows the rate limit window to pass.

    Example with defaults:
      - Video fails due to rate limit → added to retry queue
      - After batch completes, wait 120s
      - Retry all queued videos together (pass 1)
      - If still failing, wait and retry again (pass 2)
      - After 2 passes, give up on remaining failures

    Circuit breaker coordination:
      When respect_circuit_breaker is enabled (default), the batch retry queue
      will check the circuit breaker state before processing. If the circuit
      breaker is tripped, the retry queue will wait for it to recover before
      retrying. This prevents retries from being wasted during active rate limits.

    Cookie cooldown coordination:
      When wait_for_cookie_cooldown is enabled (default), the batch retry queue
      will check if any cookies are in cooldown before processing. If all cookies
      are in cooldown, it waits for the shortest cooldown to expire before retrying.
      This prevents retries from failing immediately due to cookie unavailability.
    """
    # Enable/disable batch retry queue
    enabled: bool = True

    # Delay before processing retry queue (seconds)
    # Should be long enough for rate limit window to pass
    delay_seconds: float = 120.0

    # Maximum retry passes per download session
    # After this many batch retries, give up on remaining failures
    max_passes: int = 2

    # US-51-010: Maximum retries per individual video across pipeline restarts.
    # Videos exceeding this count are permanently skipped on checkpoint restore.
    max_retries_per_video: int = 3

    # Respect circuit breaker state when processing retries
    # If True, wait for circuit breaker to recover before retrying
    # If False, retry immediately after delay_seconds regardless of circuit breaker
    respect_circuit_breaker: bool = True

    # Wait for cookie cooldown before processing retries
    # If True, check if any cookies are in cooldown and extend delay if needed
    # If False, proceed with retry even if cookies are in cooldown
    wait_for_cookie_cooldown: bool = True

    # Maximum combined wait time (seconds) when both circuit breaker and cookie
    # cooldown are blocking simultaneously. If exceeded, force-process the retry
    # queue with the current best-available cookie method instead of waiting
    # for both to clear. Prevents deadlock when CB and cooldown overlap.
    max_combined_wait_seconds: float = 300.0

    # US-114-012: Progressive retry delay - exponential backoff with max cap.
    # Initial delay for first retry pass (seconds). After each pass, the delay
    # grows exponentially: initial_delay * (backoff_multiplier ^ (pass - 1))
    initial_delay_seconds: float = 1.0

    # Maximum delay cap for progressive backoff (seconds). Prevents delays from
    # growing unbounded on extended rate limiting.
    max_delay_seconds: float = 60.0

    # Multiplier for exponential backoff growth. 2.0 means each pass doubles
    # the delay (1s -> 2s -> 4s -> 8s -> ... capped at max_delay_seconds).
    backoff_multiplier: float = 2.0

    # Jitter factor for delay randomization to prevent thundering herd.
    # Applied as delay * (1 + random.uniform(-jitter, +jitter)).
    # Set to 0.0 to disable jitter. Recommended: 0.1-0.3
    jitter_factor: float = 0.2

    # US-114-002: Priority weighting for smart retry queue prioritization.
    # Controls how segment_value_score is calculated:
    # - duration: Weight for segment duration tier (longer = higher priority)
    # - confidence: Weight for match confidence score (higher = higher priority)
    # - retry_count: Weight for retry count (lower = higher priority, ensures eventual retry)
    # Formula: score = (duration_score * duration_weight) + (confidence * confidence_weight) + ((max_retries - retry_count) * retry_weight)
    retry_priority_weighting: Dict[str, float] = field(default_factory=lambda: {
        "duration": 0.5,
        "confidence": 0.3,
        "retry_count": 0.2,
    })

    def __post_init__(self):
        """Convert nested dicts to proper dataclass instances."""
        if isinstance(self.retry_priority_weighting, dict):
            # Handle case where it loads as a dict from YAML
            self.retry_priority_weighting = self.retry_priority_weighting or {}
            # Ensure all keys exist with defaults
            self.retry_priority_weighting = {
                "duration": self.retry_priority_weighting.get("duration", 0.5),
                "confidence": self.retry_priority_weighting.get("confidence", 0.3),
                "retry_count": self.retry_priority_weighting.get("retry_count", 0.2),
            }


@dataclass
class CategoryBackoffConfig:
    """US-144-003: Category-specific backoff configuration.

    Defines exponential backoff parameters for different error categories.
    Each category can have its own base time and max backoff to allow
    more aggressive retrying for recoverable errors.

    Categories:
        rate_limit: 429 errors - longer backoff (YouTube rate limiting)
        network: DNS, connectivity errors - moderate backoff
        format: Format-specific errors (age-gated, unavailable) - shorter backoff
        server: 5xx errors - moderate backoff
        bot_detection: 403/bot detection - longer backoff
        timeout: Timeout errors - moderate backoff
        geo_blocked: Geographic blocking - longer backoff (needs VPN)
    """
    # Base backoff time in seconds (before exponential multiplier)
    rate_limit_base: float = 30.0
    network_base: float = 10.0
    format_base: float = 5.0
    server_base: float = 15.0
    bot_detection_base: float = 30.0
    timeout_base: float = 15.0
    geo_blocked_base: float = 60.0

    # Maximum backoff time per category (caps exponential growth)
    rate_limit_max: float = 300.0
    network_max: float = 120.0
    format_max: float = 60.0
    server_max: float = 180.0
    bot_detection_max: float = 300.0
    timeout_max: float = 120.0
    geo_blocked_max: float = 600.0

    # Exponential backoff multiplier (base * multiplier^attempts)
    exponential_base: float = 2.0


@dataclass
class DownloadRetryBudgetConfig:
    """Retry budget for video downloads (US-129-002).

    Tracks per-video retry attempts and backoff time to prevent infinite
    download loops on persistent failures. When limits are exceeded for a
    video, it's permanently skipped instead of endless retries.

    US-144-003: Added category-aware backoff with CategoryBackoffConfig.

    Configure in config.yaml under download.retry_budget.

    Example:
      - Video download fails → retry attempt recorded
      - After 5 failures (max_attempts) → skip video
      - Or after 300s cumulative backoff (max_backoff_time_seconds) → skip video
      - On successful download → budget resets for that video
    """
    # Enable/disable retry budget tracking
    enabled: bool = True

    # Maximum retry attempts per video before giving up
    # Set to 0 for unlimited attempts
    max_attempts: int = 5

    # Maximum cumulative backoff time per video (seconds)
    # Set to 0 for unlimited backoff
    max_backoff_time_seconds: float = 300.0

    # US-144-003: Category-aware backoff settings
    category_backoff: CategoryBackoffConfig = None

    # US-144-003: Enable reducing backoff after successful retries within a category
    reduce_backoff_on_success: bool = True

    # US-144-003: Factor to reduce backoff by after successful retry (0.5 = halve the backoff)
    success_backoff_reduction_factor: float = 0.5

    def __post_init__(self):
        """Validate configuration values at load time."""
        # Handle category_backoff as dict from YAML
        if self.category_backoff is None or isinstance(self.category_backoff, dict):
            self.category_backoff = CategoryBackoffConfig(
                **(self.category_backoff or {})
            )

        if self.max_attempts < 0:
            raise ValueError(
                f"DownloadRetryBudgetConfig.max_attempts must be >= 0, "
                f"got {self.max_attempts}. "
                f"Check config.yaml under download.retry_budget.max_attempts"
            )

        if self.max_backoff_time_seconds < 0:
            raise ValueError(
                f"DownloadRetryBudgetConfig.max_backoff_time_seconds must be >= 0, "
                f"got {self.max_backoff_time_seconds}. "
                f"Check config.yaml under download.retry_budget.max_backoff_time_seconds"
            )

        if self.success_backoff_reduction_factor <= 0 or self.success_backoff_reduction_factor >= 1.0:
            raise ValueError(
                f"DownloadRetryBudgetConfig.success_backoff_reduction_factor must be between 0 and 1, "
                f"got {self.success_backoff_reduction_factor}"
            )


@dataclass
class VPNConfig:
    """VPN integration for IP rotation on rate limits.

    When cookie rotation is exhausted or unavailable, switch VPN servers
    to get a new IP address. Requires a VPN client with CLI support.

    Examples:
    - NordVPN: "nordvpn connect random"
    - ExpressVPN: "expressvpn connect random"
    - Mullvad: "mullvad relay set location any && mullvad connect"
    - WireGuard: "wg-quick down wg0 && wg-quick up wg1"

    Connection verification:
    - By default, verifies connection by pinging Google (8.8.8.8) or curl to google.com
    - For privacy-conscious users, endpoints are configurable via verification_endpoint/verification_ip
    - Set skip_verification=True to disable verification entirely
    """
    # Enable/disable VPN switching
    enabled: bool = False

    # Command to switch/rotate VPN server
    # Should connect to a new server (ideally random location)
    switch_command: str = ""  # e.g., "nordvpn connect random"

    # Command to disconnect VPN (optional, for cleanup)
    disconnect_command: str = ""  # e.g., "nordvpn disconnect"

    # Trigger VPN switch on rate limit (after cookie rotation exhausted)
    rotate_on_rate_limit: bool = True

    # Wait time after VPN switch for connection to establish (seconds)
    switch_delay_seconds: int = 10

    # Maximum VPN switches per session (0 = unlimited)
    max_switches_per_session: int = 10

    # Verify connectivity after switch (ping test)
    verify_connection: bool = True

    # Timeout for connection verification (seconds)
    verify_timeout: int = 30

    # Skip verification entirely (for privacy or when endpoints are blocked)
    # When True, verification is skipped even if verify_connection=True
    skip_verification: bool = False

    # HTTPS endpoint for curl-based verification (default: Google)
    # Example alternatives: "https://cloudflare.com", "https://1.1.1.1"
    verification_endpoint: str = "https://www.google.com"

    # IP address for ping-based verification (default: Google DNS)
    # Example alternatives: "1.1.1.1" (Cloudflare), "208.67.222.222" (OpenDNS)
    verification_ip: str = "8.8.8.8"


@dataclass
class MullvadConfig:
    """Mullvad VPN-specific configuration for Tier 4 bypass.

    Extends VPN functionality with Mullvad-specific features:
    - Uses native mullvad CLI commands for connect/disconnect/rotate
    - Verifies connection via am.i.mullvad.net API
    - Supports geographic server rotation by country code

    Tier 4 escalation: After cookie rotation (Tier 3) is exhausted,
    MullvadVPN rotates servers to get a new IP address.

    Requires: Mullvad VPN client installed with CLI access.
    Install: winget install Mullvad.Mullvad (Windows)
    """
    # Enable/disable Mullvad VPN integration as Tier 4 bypass
    enabled: bool = False

    # Preferred countries for server rotation (ISO 3166-1 alpha-2 codes)
    # Mullvad rotates through these when escalation triggers VPN rotation
    # Empty list = use any available server (mullvad default)
    preferred_countries: List[str] = field(default_factory=lambda: [
        'us', 'gb', 'de', 'nl', 'se', 'ch'
    ])

    # Server rotation strategy:
    # - 'random': Select random country from preferred_countries
    # - 'sequential': Cycle through preferred_countries in order
    # - 'nearest': Let Mullvad choose nearest server (ignores preferred_countries)
    rotation_strategy: str = 'random'

    # Maximum VPN server rotations per session (prevents infinite rotation loops)
    max_rotations_per_session: int = 5

    # Timeout for am.i.mullvad.net verification request (seconds)
    # Falls back to generic ping verification if verification times out
    verification_timeout: int = 10

    # Initial delay before first rotation (seconds) - base for exponential backoff
    initial_rotation_delay_seconds: float = 5.0

    # Maximum rotation delay cap (seconds) - exponential backoff won't exceed this
    max_rotation_delay_seconds: float = 60.0

    # Wait time after rotation for connection to stabilize (seconds)
    rotation_delay_seconds: int = 5

    # Maximum consecutive VPN rotation failures before graceful degradation (US-113-012)
    # When exceeded, VPN will degrade to cookie-only mode (Tier 3)
    # Set to 0 to disable degradation (always try VPN)
    max_consecutive_vpn_failures: int = 5

    # Enable latency-based server selection (US-114-008)
    # When True, measure latency to servers and prefer lower-latency ones
    prefer_low_latency: bool = True

    # Maximum acceptable latency in milliseconds (US-114-008)
    # Servers with latency above this threshold are skipped
    # Set to 0 to disable threshold (accept any latency)
    max_latency_ms: int = 200

    # Latency measurement timeout per server (seconds)
    # If latency check times out, server is deprioritized but not excluded
    latency_measurement_timeout: float = 3.0


@dataclass
class YouTubeAPIConfig:
    """YouTube Data API configuration for programmatic access to YouTube.

    Provides an alternative to yt-dlp for video search, metadata retrieval,
    and caption enumeration using the official YouTube Data API v3.

    Quota costs (per operation):
    - search.list: 100 units
    - videos.list: 1 unit
    - channels.list: 1 unit
    - captions.list: 50 units

    Default daily quota: 10,000 units
    """
    # Master switch: enable YouTube Data API integration
    enabled: bool = True

    # YouTube Data API key from Google Cloud Console
    # Get at: https://console.cloud.google.com/apis/credentials
    api_key: str = ""

    # List of API keys for higher quota limits with automatic key rotation
    # When one key's quota is exhausted, the client automatically switches to the next key
    api_keys: List[str] = field(default_factory=list)

    # US-153-002: Rotation strategy when switching between API keys
    # US-154-006: Added "smart" option for quota-aware rotation
    # Options: "sequential" (next key in order), "random" (random key), "least_used" (key with lowest quota used), "smart" (key with most remaining quota)
    rotation_strategy: str = "sequential"

    # Daily quota limit (default: 10,000 units per Google Cloud free tier)
    # API will fallback to yt-dlp when quota exhausted
    quota_limit: int = 10000

    # US-153-003: Quota allocation strategy between search and caption operations
    # Options: "balanced" (equal distribution), "search_first" (prioritize search quota), "caption_first" (prioritize caption quota)
    quota_allocation_strategy: str = "balanced"

    # Warn when quota reaches this percentage of limit
    warn_at_percent: int = 80

    # US-150-007: Trigger proactive fallback when remaining quota falls below this percentage
    # When predicted remaining quota drops below this threshold, fallback to yt-dlp
    # Default: 10% - triggers fallback when <10% quota remains
    quota_fallback_threshold_percent: int = 10

    # US-155-003: Predictive quota fallback threshold (minutes)
    # Trigger fallback when predicted time until quota exhaustion is less than this value
    # Default: 30 minutes - gives time to gracefully switch to yt-dlp
    quota_fallback_prediction_minutes: int = 30

    # US-153-003: Quota allocation strategy for video search vs caption
    # Options: "balanced" (equal priority), "search_first" (prioritize search), "caption_first" (prioritize captions)
    quota_allocation_strategy: str = "balanced"

    # US-155-003: Enable adaptive quota fallback threshold based on time of day
    # When enabled, uses higher threshold during peak usage hours
    quota_fallback_adaptive_enabled: bool = True

    # US-155-003: Peak hours threshold multiplier
    # During peak hours, multiply the base threshold by this factor for earlier fallback
    # Default: 1.5x - triggers fallback earlier during busy periods
    quota_fallback_peak_multiplier: float = 1.5

    # US-155-003: Peak hours start time (24-hour format)
    # Start of peak usage hours (e.g., 9 = 9 AM, 18 = 6 PM)
    quota_fallback_peak_start_hour: int = 9

    # US-155-003: Peak hours end time (24-hour format)
    # End of peak usage hours
    quota_fallback_peak_end_hour: int = 21

    # US-155-003: Enable abnormal quota usage rate warning
    # Log warning when usage rate significantly deviates from historical patterns
    quota_abnormal_rate_warning_enabled: bool = True

    # US-155-003: Abnormal rate threshold (multiplier)
    # Consider rate abnormal if it exceeds this multiple of the hourly average
    # Default: 2.0x - warn if current rate is more than 2x the average
    quota_abnormal_rate_threshold: float = 2.0

    # US-150-008: Skip API key validation at startup
    # Set to true to skip the quick health check call on pipeline start
    skip_startup_validation: bool = False

    # Action when quota exhausted: "fallback" (use yt-dlp) or "pause" (stop)
    quota_exhausted_action: str = "fallback"

    # Retry configuration for transient errors
    max_retries: int = 3
    retry_delay_seconds: float = 2.0

    # US-152-008: Rate limit in requests per second
    # YouTube recommends 10 requests/second as the safe limit
    # Token bucket algorithm is used for rate limiting
    rate_limit_rps: float = 10.0

    # US-155-008: Adaptive rate limiting based on response times
    # Enable adaptive rate limiting that adjusts RPS based on API latency
    adaptive_rate_limiting_enabled: bool = False

    # High latency threshold (ms) - reduce RPS when average latency exceeds this
    # Default: 500ms - YouTube API typically responds in <500ms under normal load
    latency_high_threshold_ms: float = 500.0

    # Low latency threshold (ms) - increase RPS when average latency is below this
    # Default: 200ms - indicates low server load, safe to increase rate
    latency_low_threshold_ms: float = 200.0

    # Rate decrease factor when high latency detected (multiplier)
    # Default: 0.8 - reduce rate by 20% when latency is high
    rate_decrease_factor: float = 0.8

    # Rate increase factor when low latency detected (multiplier)
    # Default: 1.1 - increase rate by 10% when latency is low
    rate_increase_factor: float = 1.1

    # Minimum adaptive rate (RPS) - rate will not go below this
    min_adaptive_rate: float = 1.0

    # Maximum adaptive rate (RPS) - rate will not exceed this
    max_adaptive_rate: float = 10.0

    # Number of latency samples to use for averaging
    # Default: 10 - smooths out variance in response times
    latency_smoothing_window: int = 10

    # Request timeout in seconds
    timeout_seconds: int = 30

    # Minimum subscriber count for channel filtering
    # Videos from channels below this threshold are filtered out
    min_subscriber_count: int = 1000

    # Cache TTL for API responses (seconds)
    cache_ttl_seconds: int = 3600  # 1 hour

    # Cache TTL for channel metadata (seconds)
    # Channel metadata changes infrequently, so longer cache is beneficial
    channel_metadata_cache_ttl_seconds: int = 86400  # 24 hours

    # US-149-009: Cache TTL in days for SQLite-based query cache
    # Persists across restarts to reduce API quota usage
    cache_ttl_days: int = 7

    # US-156-003: Auto-invalidate cache when API returns stale/empty data
    # When enabled, automatically invalidates cached entries when the API returns
    # empty results or errors that indicate stale data (e.g., quota issues)
    auto_invalidate_on_error: bool = True

    # US-156-005: Enable search query sanitization and deduplication
    # When enabled, sanitizes queries (removes extra whitespace, special chars)
    # and deduplicates search terms to avoid redundant API calls
    deduplicate_searches: bool = True

    # US-148-009: Auto-scale quota based on project size
    # When enabled, quota_limit is automatically adjusted based on estimated project size
    quota_auto_scale_enabled: bool = False  # Enable auto-scaling
    auto_scale_budget: bool = True  # US-157-002: Scale budget based on keyword count
    quota_multiplier: float = 1.0  # Multiplier for quota scaling (e.g., 1.5 = 150% of default)
    quota_floor: int = 1000  # Minimum quota for small projects

    # US-153-005: Max concurrent requests for async batch video metadata operations
    # Controls parallel API calls when fetching video details
    # Default: 5, Min: 1, Max: 10
    max_concurrent_requests: int = 5
    quota_ceiling: int = 100000  # Maximum quota to prevent runaway

    # US-158-002: Search results ordering
    # Options: "relevance" (default), "date", "viewCount", "rating", "videoCount"
    # - relevance: Most relevant results (YouTube default)
    # - date: Most recently published
    # - viewCount: Highest view count
    # - rating: Highest rating
    # - videoCount: Channel with most videos
    order_by: str = "relevance"

    # US-158-003: Video duration filter for search results
    # Options: "any" (default), "short" (<4 min), "medium" (4-20 min), "long" (>20 min)
    # - any: No duration filter
    # - short: Videos less than 4 minutes
    # - medium: Videos between 4 and 20 minutes
    # - long: Videos longer than 20 minutes
    video_duration: str = "any"

    # US-158-004: Region code for localized search results
    # ISO 3166-1 alpha-2 country code (US, GB, DE, JP, etc.)
    # Default: "US" - returns results relevant to United States
    # Empty string = no region filter (uses YouTube default)
    region_code: str = "US"

    # US-158-005: Safe search level for family-friendly results
    # Options: "none" (no filtering), "moderate" (some results filtered), "strict" (most results filtered)
    # Default: "moderate" - filters explicit content while allowing most results
    # YouTube API: safeSearch parameter for search endpoint
    safe_search: str = "moderate"

    # US-158-010: Quality boost for video engagement metrics in result ranking
    # When enabled, calculates quality score using weighted formula: viewCount * 0.7 + likeCount * 0.2 + commentCount * 0.1
    # This score can be used to boost higher-quality videos in search results
    quality_boost_enabled: bool = False

    # Weights for quality score calculation
    # view_count_weight + like_count_weight + comment_count_weight should equal 1.0
    quality_view_weight: float = 0.7
    quality_like_weight: float = 0.2
    quality_comment_weight: float = 0.1

    # Estimated quota cost per operation (YouTube API units)
    # search.list: 100 units, captions.list: 50 units, videos.list: 1 unit
    estimated_quota_per_search: int = 100
    estimated_quota_per_caption: int = 50
    estimated_quota_per_metadata: int = 1

    # Enable pre-flight quota check to warn before pipeline starts
    enable_pre_flight_check: bool = True

    # Enable per-operation type quota tracking metrics
    enable_operation_metrics: bool = True

    # US-158-006: Batch caption fetching size
    # Number of videos to process in a single batch when fetching captions
    # Default: 10 - balances API efficiency with quota usage
    caption_batch_size: int = 10

    # US-153-009: Rate limit prediction based on time-of-day patterns
    # Enable predictive rate limiting using historical success/failure patterns
    rate_limit_prediction_enabled: bool = True

    # Prediction window in hours for historical analysis
    # How many hours of historical data to consider for predictions
    prediction_window_hours: int = 24

    # Backoff multiplier when high failure rate is predicted
    # Multiplier applied to base delay when likelihood > 0.6
    # e.g., 2.0 = double the delay when rate limit is likely
    backoff_multiplier: float = 2.0

    # Sensitivity for prediction (0.0-1.0)
    # Higher values give more weight to recent hourly patterns
    prediction_sensitivity: float = 0.5

    # US-155-008: Adaptive rate limiting based on response latency
    # When enabled, dynamically adjusts RPS based on API response times
    adaptive_rate_limiting_enabled: bool = True

    # Latency threshold (ms) above which to reduce RPS
    # When average latency exceeds this, rate is decreased
    # Default: 500ms - YouTube API typically responds in 100-300ms
    latency_high_threshold_ms: float = 500.0

    # Latency threshold (ms) below which to increase RPS
    # When average latency is below this, rate can be gradually increased
    # Default: 200ms - indicates healthy, fast API responses
    latency_low_threshold_ms: float = 200.0

    # Factor to multiply rate by when latency is high (reduce rate)
    # Default: 0.8 - reduce rate by 20% when latency spikes
    rate_decrease_factor: float = 0.8

    # Factor to multiply rate by when latency is low (increase rate)
    # Default: 1.1 - increase rate by 10% when latency is healthy
    rate_increase_factor: float = 1.1

    # Minimum RPS to prevent rate from going too low
    # Default: 1.0 - never go below 1 request per second
    min_adaptive_rate: float = 1.0

    # Maximum RPS to prevent rate from going too high
    # Default: 20.0 - never exceed double the default limit
    max_adaptive_rate: float = 20.0

    # Number of latency samples to smooth over
    # Higher values = more stable but slower to respond
    # Default: 10 samples
    latency_smoothing_window: int = 10

    # Minimum requests before adjusting rate
    # Wait for this many requests before making rate adjustments
    # Default: 5 - avoid adjusting on just a few samples
    min_requests_before_adjustment: int = 5

    # US-149-004: Retry budget to prevent infinite retry loops on transient errors
    # Configure in config.yaml under download.youtube_api.retry_budget
    retry_budget: Optional[Dict[str, Any]] = None

    # US-155-003: Preferred caption language for API-based caption fetching
    # ISO 639-1 language code (e.g., 'en', 'es', 'fr')
    # Fallback chain: preferred -> 'en' -> auto-generated -> any available
    preferred_caption_language: str = "en"

    # US-155-003: Enable fallback chain when preferred language is unavailable
    # When enabled: preferred -> 'en' -> auto-generated -> any available
    caption_language_fallback: bool = True

    # US-155-003: Track caption language distribution in API metrics
    # When enabled, records language distribution for captions fetched via YouTube API
    track_caption_language_metrics: bool = True

    # US-155-004: Date range filtering for YouTube API searches
    # Filter results by publication date
    # Use ISO 8601 format (e.g., '2020-01-01T00:00:00Z') or relative dates
    # Relative dates: 'today', '7days', '30days', '90days', '1year', '5years'
    date_range_enabled: bool = False
    published_after: str = ""  # Include videos published after this date
    published_before: str = ""  # Include videos published before this date

    # Default relative date preset (used when date_range_enabled is true)
    # Options: 'last_7_days', 'last_30_days', 'last_90_days', 'last_year', 'last_5_years'
    date_range_preset: str = "last_30_days"

    # US-155-005: Video category filtering
    # Filter search results by YouTube video category ID
    # See: https://gist.github.com/dgp/1b92a4b9c4c0ba62fe80a3a7c7066172
    # Common categories: 1=Film/Animation, 2=Autos, 10=Music, 15=Pets/Animals,
    # 17=Sports, 18=Short Movies, 19=Travel/Events, 20=Gaming, 21=Videoblogging,
    # 22=People/Blogs, 23=Comedy, 24=Entertainment, 25=News/Politics, 26=Howto/Style,
    # 27=Education, 28=Science/Technology, 29=Nonprofits, 30=Movies, 31=Anime/Action-Adventure,
    # 32=Action/Adventure, 33=Classics, 34=Comedy, 35=Documentary, 36=Drama, 37=Family,
    # 38=Foreign, 39=Horror, 40=Sci-Fi/Fantasy, 41=Thriller, 42=Shorts, 43=Shows, 44=Trailers
    video_category_enabled: bool = False  # Enable video category filtering
    default_video_category: str = ""  # Single category ID (e.g., "28" for Science & Technology)
    video_category_ids: List[str] = field(default_factory=list)  # Multiple category IDs for broader search

    # US-155-007: Quota alert webhook notifications
    # Send notifications when quota reaches warning threshold
    webhook_enabled: bool = False  # Enable webhook notifications
    webhook_urls: List[str] = field(default_factory=list)  # List of webhook URLs to notify
    webhook_timeout: int = 10  # Timeout for webhook requests in seconds
    webhook_retry_count: int = 3  # Number of retries for webhook delivery

    # US-155-005: Parallel video details fetching
    # Optimize batch video details fetching with parallel requests
    parallel_video_details_enabled: bool = True  # Enable parallel chunk execution
    video_details_chunk_size: int = 50  # Chunk size for batching (max 50 per API call)
    video_details_max_workers: int = 5  # Max parallel workers for chunk execution

    # US-155-009: Transcript timestamp optimization
    # Reduce timestamp precision to improve matching performance and reduce memory
    # Options: "millisecond" (default, full precision), "second" (rounded to 1s), "5_second" (rounded to 5s)
    timestamp_precision: str = "millisecond"

    # US-155-010: Per-channel API usage tracking and rate limiting
    # Track API calls per channel to avoid rate-limiting specific channels
    per_channel_tracking_enabled: bool = False  # Enable per-channel API tracking
    per_channel_rate_limit: int = 100  # Max API calls per channel per session
    per_channel_circuit_breaker_enabled: bool = False  # Enable circuit breaker per channel
    per_channel_circuit_breaker_threshold: int = 5  # Failures before pausing a channel
    per_channel_circuit_breaker_pause_seconds: float = 60.0  # Pause duration when circuit opens
    per_channel_graceful_no_videos: bool = True  # Handle channels with no published videos gracefully

    # US-155-010: Per-channel API usage tracking and limits
    # Track API calls per channel to avoid rate-limiting specific channels
    per_channel_tracking_enabled: bool = True  # Enable per-channel API call tracking
    per_channel_rate_limit: int = 100  # Max API calls per channel per session
    per_channel_circuit_breaker_enabled: bool = True  # Enable channel-level circuit breaker
    per_channel_circuit_breaker_threshold: int = 5  # Failures before channel circuit trips
    per_channel_circuit_breaker_pause: float = 60.0  # Pause duration when channel circuit opens

    # US-158-010: Video quality signals integration for result ranking
    # Enable quality boost based on engagement metrics (viewCount, likeCount, commentCount)
    # When enabled, calculates quality score and adds to video details
    # Formula: viewCount * 0.7 + likeCount * 0.2 + commentCount * 0.1
    quality_boost_enabled: bool = False

    # Weight for view count in quality score calculation (default: 0.7)
    quality_view_count_weight: float = 0.7

    # Weight for like count in quality score calculation (default: 0.2)
    quality_like_count_weight: float = 0.2

    # Weight for comment count in quality score calculation (default: 0.1)
    quality_comment_count_weight: float = 0.1

    def __post_init__(self):
        valid_actions = ("fallback", "pause")
        if self.quota_exhausted_action not in valid_actions:
            raise ValueError(
                f"YouTubeAPIConfig.quota_exhausted_action must be one of {valid_actions}, "
                f"got '{self.quota_exhausted_action}'"
            )
        if self.quota_limit < 1:
            raise ValueError(
                f"YouTubeAPIConfig.quota_limit must be positive, got {self.quota_limit}"
            )
        if not 0 <= self.quota_fallback_threshold_percent <= 100:
            raise ValueError(
                f"YouTubeAPIConfig.quota_fallback_threshold_percent must be 0-100, "
                f"got {self.quota_fallback_threshold_percent}"
            )
        # US-155-003: Validate quota_fallback_prediction_minutes
        if self.quota_fallback_prediction_minutes <= 0:
            raise ValueError(
                f"YouTubeAPIConfig.quota_fallback_prediction_minutes must be positive, "
                f"got {self.quota_fallback_prediction_minutes}"
            )
        # US-155-003: Validate quota_fallback_peak_multiplier
        if self.quota_fallback_peak_multiplier <= 0:
            raise ValueError(
                f"YouTubeAPIConfig.quota_fallback_peak_multiplier must be positive, "
                f"got {self.quota_fallback_peak_multiplier}"
            )
        # US-155-003: Validate peak hours
        if not 0 <= self.quota_fallback_peak_start_hour <= 23:
            raise ValueError(
                f"YouTubeAPIConfig.quota_fallback_peak_start_hour must be 0-23, "
                f"got {self.quota_fallback_peak_start_hour}"
            )
        if not 0 <= self.quota_fallback_peak_end_hour <= 23:
            raise ValueError(
                f"YouTubeAPIConfig.quota_fallback_peak_end_hour must be 0-23, "
                f"got {self.quota_fallback_peak_end_hour}"
            )
        # US-155-003: Validate quota_abnormal_rate_threshold
        if self.quota_abnormal_rate_threshold <= 0:
            raise ValueError(
                f"YouTubeAPIConfig.quota_abnormal_rate_threshold must be positive, "
                f"got {self.quota_abnormal_rate_threshold}"
            )
        # US-152-008: Validate rate_limit_rps
        if self.rate_limit_rps <= 0:
            raise ValueError(
                f"YouTubeAPIConfig.rate_limit_rps must be positive, got {self.rate_limit_rps}"
            )
        # US-155-008: Validate adaptive rate limiting config
        if self.adaptive_rate_limiting_enabled:
            if self.latency_high_threshold_ms <= 0:
                raise ValueError(
                    f"YouTubeAPIConfig.latency_high_threshold_ms must be positive, "
                    f"got {self.latency_high_threshold_ms}"
                )
            if self.latency_low_threshold_ms <= 0:
                raise ValueError(
                    f"YouTubeAPIConfig.latency_low_threshold_ms must be positive, "
                    f"got {self.latency_low_threshold_ms}"
                )
            if self.latency_low_threshold_ms >= self.latency_high_threshold_ms:
                raise ValueError(
                    f"YouTubeAPIConfig.latency_low_threshold_ms must be less than "
                    f"latency_high_threshold_ms, got low={self.latency_low_threshold_ms}, "
                    f"high={self.latency_high_threshold_ms}"
                )
            if self.rate_decrease_factor <= 0 or self.rate_decrease_factor >= 1.0:
                raise ValueError(
                    f"YouTubeAPIConfig.rate_decrease_factor must be between 0 and 1, "
                    f"got {self.rate_decrease_factor}"
                )
            if self.rate_increase_factor <= 1.0:
                raise ValueError(
                    f"YouTubeAPIConfig.rate_increase_factor must be greater than 1, "
                    f"got {self.rate_increase_factor}"
                )
            if self.min_adaptive_rate <= 0:
                raise ValueError(
                    f"YouTubeAPIConfig.min_adaptive_rate must be positive, "
                    f"got {self.min_adaptive_rate}"
                )
            if self.max_adaptive_rate <= 0:
                raise ValueError(
                    f"YouTubeAPIConfig.max_adaptive_rate must be positive, "
                    f"got {self.max_adaptive_rate}"
                )
            if self.min_adaptive_rate > self.max_adaptive_rate:
                raise ValueError(
                    f"YouTubeAPIConfig.min_adaptive_rate must be less than or equal to "
                    f"max_adaptive_rate, got min={self.min_adaptive_rate}, max={self.max_adaptive_rate}"
                )
            if self.latency_smoothing_window <= 0:
                raise ValueError(
                    f"YouTubeAPIConfig.latency_smoothing_window must be positive, "
                    f"got {self.latency_smoothing_window}"
                )
        # US-153-003: Validate quota_allocation_strategy
        valid_strategies = ("balanced", "search_first", "caption_first")
        if self.quota_allocation_strategy not in valid_strategies:
            raise ValueError(
                f"YouTubeAPIConfig.quota_allocation_strategy must be one of {valid_strategies}, "
                f"got '{self.quota_allocation_strategy}'"
            )
        # US-155-005: Validate video_category_ids
        if self.video_category_ids:
            # Ensure all IDs are valid category ID strings
            for cat_id in self.video_category_ids:
                if not cat_id.isdigit():
                    raise ValueError(
                        f"YouTubeAPIConfig.video_category_ids must contain numeric strings, "
                        f"got '{cat_id}'"
                    )
        # US-155-009: Validate timestamp_precision
        valid_precisions = ("millisecond", "second", "5_second")
        if self.timestamp_precision not in valid_precisions:
            raise ValueError(
                f"YouTubeAPIConfig.timestamp_precision must be one of {valid_precisions}, "
                f"got '{self.timestamp_precision}'"
            )
        # US-158-002: Validate order_by
        valid_orders = ("relevance", "date", "viewCount", "rating", "videoCount")
        if self.order_by not in valid_orders:
            raise ValueError(
                f"YouTubeAPIConfig.order_by must be one of {valid_orders}, "
                f"got '{self.order_by}'"
            )
        # US-158-006: Validate caption_batch_size
        if self.caption_batch_size < 1:
            raise ValueError(
                f"YouTubeAPIConfig.caption_batch_size must be positive, "
                f"got {self.caption_batch_size}"
            )


@dataclass
class ImpersonationConfig:
    """Browser impersonation configuration for yt-dlp TLS fingerprint bypass.

    Uses curl_cffi impersonation targets to spoof browser TLS fingerprints,
    defeating YouTube bot detection that relies on TLS ClientHello analysis.

    When enabled (default), every yt-dlp subprocess call includes
    ``--impersonate <target>`` with round-robin rotation across detected targets.

    Auto-detection runs ``yt-dlp --list-impersonate-targets`` at startup to
    discover available targets from the installed curl_cffi library.

    Browser fallback chain (US-113-005):
    When Chrome impersonation fails, automatically fall back to Firefox,
    then Safari. Configure the order via ``impersonation_fallback_order``.
    Success rates are tracked per browser type to prefer higher-success browsers.

    See also: ``src/downloader/impersonation.py`` for ImpersonationManager.
    """
    # Master switch: enable browser impersonation on all yt-dlp calls (Tier 1)
    # When True, every yt-dlp command includes --impersonate with a rotated target
    enabled: bool = True

    # Preferred targets: filter auto-detected targets to only these
    # Empty list = use all detected targets (recommended for maximum coverage)
    # Example: ["Chrome-136:Macos-15", "Safari-18.0:Ios-18.0"]
    preferred_targets: List[str] = field(default_factory=list)

    # Auto-detect available targets at startup via --list-impersonate-targets
    # When False, relies on preferred_targets list only
    detect_at_startup: bool = True

    # Timeout for the --list-impersonate-targets subprocess (seconds)
    # Increase if detection is timing out on slow systems
    detection_timeout: int = 10

    # Browser fallback order (US-113-005): try browsers in this order when
    # the current browser type fails. Each entry is a browser family name.
    # Default: Chrome -> Firefox -> Safari
    impersonation_fallback_order: List[str] = field(default_factory=lambda: [
        "chrome", "firefox", "safari"
    ])

    # Minimum success rate threshold (US-113-005): targets with success rates
    # below this threshold are deprioritized in rotation
    min_success_rate: float = 0.2

    # Enable success rate-based filtering (US-113-005): when True, skip
    # targets with success rates below min_success_rate
    enable_success_filtering: bool = True

    # Enable adaptive profile selection (US-123-004): when True, select the best
    # browser profile based on recent success rates instead of round-robin
    adaptive_profile_selection: bool = False

    # Number of recent attempts to consider for adaptive profile selection (US-123-004)
    # Only used when adaptive_profile_selection is True. Higher values give more
    # stable averages but react slower to changing conditions.
    profile_success_window: int = 20

    def __post_init__(self):
        if self.detection_timeout <= 0:
            raise ValueError(
                f"ImpersonationConfig.detection_timeout must be positive, "
                f"got {self.detection_timeout}"
            )
        if self.profile_success_window <= 0:
            raise ValueError(
                f"ImpersonationConfig.profile_success_window must be positive, "
                f"got {self.profile_success_window}"
            )
        if not (0 <= self.min_success_rate <= 1):
            raise ValueError(
                f"ImpersonationConfig.min_success_rate must be between 0 and 1, "
                f"got {self.min_success_rate}"
            )
        # Warn about unknown browser families in fallback order (non-fatal for extensibility)
        known_families = {'chrome', 'firefox', 'safari', 'edge', 'tor', 'opera'}
        for family in self.impersonation_fallback_order:
            if family.lower() not in known_families:
                import logging
                logging.getLogger(__name__).warning(
                    f"Unknown browser family '{family}' in impersonation_fallback_order"
                )


@dataclass
class ExtractorArgsConfig:
    """Tier 2 extractor-args bypass configuration for yt-dlp.

    When Tier 1 (browser impersonation) fails repeatedly with 403/bot errors,
    escalate to Tier 2 by adding ``--extractor-args "youtube:player_client=<client>"``
    to yt-dlp commands. Player clients are rotated round-robin per keyword.

    Tier progression:
      - Tier 1 (IMPERSONATE_ONLY): --impersonate only (handled by ImpersonationManager)
      - Tier 2 (EXTRACTOR_ARGS): --impersonate + --extractor-args player_client
      - Tier 3 (FULL_BYPASS): Both + cookie rotation (handled by CookieRotator)

    Browser-specific player clients (US-113-005):
    Different browsers work better with different player clients. This config
    allows specifying browser-specific player_client values that are used when
    falling back from one browser to another.

    See also: ``src/downloader/escalation_manager.py`` for EscalationManager.
    """
    # Master switch: enable extractor-args escalation (Tier 2)
    enabled: bool = True

    # Player client values to rotate through at Tier 2
    # These are passed as --extractor-args "youtube:player_client=<value>"
    # Available: web, web_safari, web_embedded, web_music, web_creator,
    #   mweb, ios, android, android_sdkless, tv, tv_simply, tv_downgraded, tv_embedded
    player_clients: List[str] = field(default_factory=lambda: [
        "web_safari", "tv_downgraded", "web", "ios", "android_vr"
    ])

    # Browser-specific player clients (US-113-005): Map browser family to
    # preferred player clients. When falling back from one browser to another,
    # use the corresponding player clients for better success rates.
    browser_specific_clients: Dict[str, List[str]] = field(default_factory=lambda: {
        "chrome": ["web", "web_embedded", "tv", "tv_downgraded"],
        "firefox": ["web_safari", "tv_simply", "web"],
        "safari": ["web_safari", "ios", "tv_downgraded"],
        "edge": ["web", "web_embedded", "tv"],
    })

    # Number of consecutive 403 errors before escalating to next tier
    escalation_threshold: int = 2

    # Cooldown before de-escalating back to a lower tier (seconds)
    # After this period without errors, tier may be lowered
    cooldown_seconds: float = 300.0

    # Maximum escalation tier (1=impersonate, 2=extractor-args, 3=full bypass)
    max_tier: int = 3

    # De-escalation: drop tier on sustained success (US-61-006)
    de_escalation_enabled: bool = True
    de_escalation_threshold: int = 5  # Consecutive successes needed to de-escalate

    # Dynamic player_client selection based on success rates (US-123-003)
    # Number of recent attempts to consider for success rate calculation
    # Set to 0 to disable dynamic selection
    success_rate_window: int = 50

    # Manual override for player_client fallback order
    # If provided, this order is used instead of dynamic selection
    # Empty list means use dynamic selection
    extractor_args_fallback_order: List[str] = field(default_factory=list)


@dataclass
class AdaptiveSeverityConfig:
    """Adaptive error severity configuration (US-89-012).

    Tracks error frequency per category over a sliding window and auto-escalates
    severity when the same error repeats. This improves backoff timing for
    recurring errors without manual tuning.

    Example: After 3 consecutive 429 errors, severity escalates from medium to high,
    resulting in longer backoff delays (3.0x multiplier instead of 2.0x).

    Configure in config.yaml under download.adaptive_severity.
    """
    # Enable/disable adaptive severity adjustment
    enabled: bool = True

    # Number of consecutive errors of the same category before escalating severity
    # Example: 3 means after 3x 429 errors, severity escalates
    escalation_threshold: int = 3

    # Maximum severity level to escalate to: 'medium' or 'high'
    # 'high' = 3.0x backoff multiplier, 'medium' = 2.0x
    max_severity: str = "high"

    # Sliding window size for tracking error frequency
    # Errors older than this window are forgotten
    window_size: int = 10

    # Categories to track for adaptive severity
    # Maps category name to whether it's tracked
    tracked_categories: Dict[str, bool] = field(default_factory=lambda: {
        "rate_limit": True,      # 429 errors
        "bot_detection": True,  # 403/bot errors
        "network": True,         # DNS/connection errors
        "timeout": True,         # Timeout errors
        "video_specific": False, # Video-specific errors (rarely retry)
    })

    # Reset severity tracking on success
    # When True, resets error count on successful download
    reset_on_success: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        valid_max_severity = ("medium", "high")
        if self.max_severity not in valid_max_severity:
            raise ValueError(
                f"AdaptiveSeverityConfig.max_severity must be one of {valid_max_severity}, "
                f"got '{self.max_severity}'"
            )
        if self.escalation_threshold < 1:
            raise ValueError(
                f"AdaptiveSeverityConfig.escalation_threshold must be >= 1, "
                f"got {self.escalation_threshold}"
            )
        if self.window_size < 1:
            raise ValueError(
                f"AdaptiveSeverityConfig.window_size must be >= 1, "
                f"got {self.window_size}"
            )


@dataclass
class BandwidthThrottleConfig:
    """Bandwidth throttling configuration for downloads (US-93-007, US-114-007).

    Limits download bandwidth to prevent consuming all available bandwidth
    during pipeline execution. Uses yt-dlp's --downloader-args to pass
    rate limiting to ffmpeg.

    Example with defaults:
      - Global limit: 5M (5 MB/s = 40 Mbps)
      - Per-download limit: disabled by default
      - Small file bypass: files under 10MB skip throttling
      - Peak/offpeak hours: 9am-5pm uses 1000 Kbps, offpeak uses 5000 Kbps
      - Logging: always logs when throttling is applied

    Configure in config.yaml under download.bandwidth_throttle.
    """
    # Enable/disable bandwidth throttling (US-114-007)
    # When False, all other settings are ignored
    enabled: bool = False

    # Integer bandwidth limit in Kbps (US-114-007)
    # Alternative to string-based global_limit
    # 0 = unlimited, default: 0
    # Takes precedence over global_limit if > 0
    bandwidth_limit_kbps: int = 0

    # Global bandwidth limit for all downloads
    # Format: number followed by K (KB/s) or M (MB/s)
    # Examples: "5M" = 5 MB/s, "1M" = 1 MB/s, "500K" = 500 KB/s
    # Set to "0" or empty to disable global limit
    global_limit: str = "5M"

    # Per-download bandwidth limit (optional override)
    # If set, applies this limit instead of global_limit for each download
    # Empty string = use global_limit
    per_download_limit: str = ""

    # File size threshold to bypass throttling (MB)
    # Files smaller than this will not have throttling applied
    # This speeds up small file downloads while limiting large ones
    # Set to 0 to always throttle, or very high to never throttle
    bypass_under_mb: float = 10.0

    # Minimum file size for per-download limit (MB)
    # When per_download_limit is set, only apply it to files >= this size
    # Smaller files use global_limit or no throttling
    per_download_min_size_mb: float = 50.0

    # Peak hours configuration (US-114-007)
    # Hour of day when peak pricing/limits apply (24-hour format)
    # Default: 9 AM
    peak_hours_start: int = 9

    # Hour of day when peak hours end (24-hour format)
    # Default: 5 PM (17:00)
    peak_hours_end: int = 17

    # Bandwidth limit during peak hours (Kbps)
    # Applied when current hour is within peak_hours_start to peak_hours_end
    # Default: 1000 Kbps (1 Mbps)
    peak_limit_kbps: int = 1000

    # Bandwidth limit during off-peak hours (Kbps)
    # Applied when current hour is outside peak hours
    # Default: 5000 Kbps (5 Mbps)
    offpeak_limit_kbps: int = 5000

    # Enable peak/offpeak hour-based throttling (US-114-007)
    # When True, uses peak_hours_start/end to determine which limit to apply
    peak_offpeak_enabled: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.bypass_under_mb < 0:
            raise ValueError(
                f"BandwidthThrottleConfig.bypass_under_mb must be >= 0, got {self.bypass_under_mb}"
            )
        if self.per_download_min_size_mb < 0:
            raise ValueError(
                f"BandwidthThrottleConfig.per_download_min_size_mb must be >= 0, got {self.per_download_min_size_mb}"
            )
        if self.bandwidth_limit_kbps < 0:
            raise ValueError(
                f"BandwidthThrottleConfig.bandwidth_limit_kbps must be >= 0, got {self.bandwidth_limit_kbps}"
            )
        if not 0 <= self.peak_hours_start <= 23:
            raise ValueError(
                f"BandwidthThrottleConfig.peak_hours_start must be 0-23, got {self.peak_hours_start}"
            )
        if not 0 <= self.peak_hours_end <= 23:
            raise ValueError(
                f"BandwidthThrottleConfig.peak_hours_end must be 0-23, got {self.peak_hours_end}"
            )
        if self.peak_limit_kbps < 0:
            raise ValueError(
                f"BandwidthThrottleConfig.peak_limit_kbps must be >= 0, got {self.peak_limit_kbps}"
            )
        if self.offpeak_limit_kbps < 0:
            raise ValueError(
                f"BandwidthThrottleConfig.offpeak_limit_kbps must be >= 0, got {self.offpeak_limit_kbps}"
            )

    def get_limit_for_size(self, file_size_mb: float) -> str | None:
        """
        Get the appropriate bandwidth limit for a given file size.

        Args:
            file_size_mb: Estimated file size in MB

        Returns:
            Bandwidth limit string (e.g., "5M") or None if throttling disabled
        """
        if not self.enabled:
            return None

        # Check if file is small enough to bypass
        if file_size_mb < self.bypass_under_mb:
            return None

        # Use per-download limit if file is large enough and per-download limit is set
        if self.per_download_limit and file_size_mb >= self.per_download_min_size_mb:
            return self.per_download_limit

        # Use global limit if enabled
        if self.global_limit and self.global_limit != "0":
            return self.global_limit

        return None

    def get_limit_for_time(self, hour: int = None) -> str | None:
        """
        Get bandwidth limit based on time of day (peak/offpeak hours).

        Args:
            hour: Current hour (0-23). If None, uses current system time.

        Returns:
            Bandwidth limit string (e.g., "1M") or None if throttling disabled
        """
        if not self.enabled:
            return None

        # Use current hour if not provided
        if hour is None:
            from datetime import datetime
            hour = datetime.now().hour

        # Determine if we're in peak hours
        if self.peak_offpeak_enabled:
            # Handle wrap-around case (e.g., peak from 22 to 6)
            if self.peak_hours_start <= self.peak_hours_end:
                is_peak = self.peak_hours_start <= hour < self.peak_hours_end
            else:
                is_peak = hour >= self.peak_hours_start or hour < self.peak_hours_end

            if is_peak:
                if self.peak_limit_kbps > 0:
                    return self._kbps_to_string(self.peak_limit_kbps)
                return None
            else:
                if self.offpeak_limit_kbps > 0:
                    return self._kbps_to_string(self.offpeak_limit_kbps)
                return None

        # Fall back to bandwidth_limit_kbps if set
        if self.bandwidth_limit_kbps > 0:
            return self._kbps_to_string(self.bandwidth_limit_kbps)

        # Fall back to global_limit
        if self.global_limit and self.global_limit != "0":
            return self.global_limit

        return None

    def _kbps_to_string(self, kbps: int) -> str:
        """
        Convert integer Kbps to yt-dlp format string.

        Args:
            kbps: Bandwidth in Kbps

        Returns:
            String like "1M" for 1000 Kbps or "500K" for 500 Kbps
        """
        if kbps >= 1000:
            # Convert to Mbps
            mbps = kbps / 1000
            if mbps == int(mbps):
                return f"{int(mbps)}M"
            else:
                return f"{mbps}M"
        else:
            return f"{kbps}K"

    def get_limit(self, file_size_mb: float = None, hour: int = None) -> str | None:
        """
        Get the appropriate bandwidth limit considering both file size and time of day.

        Args:
            file_size_mb: Estimated file size in MB (None to skip size check)
            hour: Current hour (0-23). If None, uses current system time.

        Returns:
            Bandwidth limit string (e.g., "1M") or None if throttling disabled
        """
        if not self.enabled:
            return None

        # First check if time-based limit applies
        if self.peak_offpeak_enabled or self.bandwidth_limit_kbps > 0:
            time_limit = self.get_limit_for_time(hour)
            if time_limit:
                # If we have a time-based limit and file is small, check bypass
                if file_size_mb is not None and file_size_mb < self.bypass_under_mb:
                    return None
                return time_limit

        # Fall back to size-based limit
        if file_size_mb is not None:
            return self.get_limit_for_size(file_size_mb)

        # No specific limit found
        return None


@dataclass
class DownloadResumeConfig:
    """Download resume configuration (US-93-008).

    Enables resuming partial downloads when interrupted, reducing bandwidth waste.
    Uses yt-dlp's --continue flag to resume from last byte position.

    Partial files are detected by:
    - File extension .part (yt-dlp default)
    - File size > 0 with no complete file present

    Configure in config.yaml under download.download_resume.
    """
    # Enable/disable download resume functionality
    # When False, partial files are cleaned up and downloads start fresh
    enabled: bool = True

    # Minimum partial file size to consider for resume (bytes)
    # Files smaller than this are treated as incomplete and discarded
    # This prevents resuming tiny partial files from failed downloads
    min_partial_size_bytes: int = 1024  # 1KB minimum

    # Partial file extension used by yt-dlp
    # yt-dlp appends this extension during download, removes on completion
    partial_extension: str = ".part"

    # Clean up partial files on final failure (after all retries exhausted)
    # When True, deletes incomplete downloads when retries are exhausted
    # When False, partial files remain for manual inspection
    cleanup_on_failure: bool = True

    # Log resume attempts for debugging
    # When True, logs when a resume is detected and attempted
    log_resume_attempts: bool = True


@dataclass
class CheckpointCompressionConfig:
    """Checkpoint compression configuration (US-114-009).

    Enables gzip compression for download checkpoint files to reduce
    disk space usage for large projects with many videos.

    Configure in config.yaml under download.checkpoint_compression.
    """
    # Enable/disable checkpoint compression
    # When True, checkpoint files are gzip compressed
    # When False, checkpoint files are saved as plain JSON
    enabled: bool = True

    # Compression level (1-9, where 1 is fastest, 9 is best compression)
    # 6 is a good balance between speed and compression ratio
    compression_level: int = 6

    def __post_init__(self):
        if not 1 <= self.compression_level <= 9:
            raise ValueError(
                f"CheckpointCompressionConfig.compression_level must be in range 1-9, got {self.compression_level}"
            )


@dataclass
class FormatPreferenceConfig:
    """Format preference configuration (US-93-009).

    Controls video format (container/codec) selection for yt-dlp downloads.
    Allows preferring specific formats (mp4, webm) over others and selecting
    quality preference (highest/best/worst).

    yt-dlp format selection works as a fallback chain:
    - First preferred format is tried
    - If unavailable, falls back to next format
    - If all preferred formats unavailable, uses yt-dlp default

    Examples:
      - preference_order: ["mp4", "webm"] + quality: "highest"
        → Tries mp4 first, then webm, prefers highest quality available
      - preference_order: ["mp4"] + quality: "best"
        → Tries mp4 only, uses "best" quality selection
      - preference_order: [] (empty) + quality: "best"
        → Uses yt-dlp default behavior (no format preference)

    Configure in config.yaml under download.format_preference.
    """
    # Enable/disable format preference
    # When False, uses yt-dlp default format selection
    enabled: bool = True

    # Preferred format order (first available is used)
    # yt-dlp format selectors: mp4, webm, mkv, mov, avi
    # Empty list = use yt-dlp default (no preference)
    preference_order: List[str] = field(default_factory=lambda: ["mp4", "webm"])

    # Quality selection strategy:
    # - "highest": Prefer highest resolution/quality available
    # - "best": Best quality (same as highest for most cases)
    # - "worst": Lowest quality (smallest file size)
    # Note: yt-dlp's "best" includes both resolution and codec quality
    # while "highest" is purely resolution-based
    quality: str = "highest"

    # Log format selection decisions for transparency
    # When True, logs which format was selected and why
    log_selection: bool = True

    # Fallback behavior when preferred formats unavailable:
    # - "any": Use any available format
    # - "fail": Skip download with error
    fallback_behavior: str = "any"

    def __post_init__(self):
        """Validate configuration values."""
        valid_qualities = ("highest", "best", "worst")
        if self.quality not in valid_qualities:
            raise ValueError(
                f"FormatPreferenceConfig.quality must be one of {valid_qualities}, "
                f"got '{self.quality}'"
            )

        valid_fallbacks = ("any", "fail")
        if self.fallback_behavior not in valid_fallbacks:
            raise ValueError(
                f"FormatPreferenceConfig.fallback_behavior must be one of {valid_fallbacks}, "
                f"got '{self.fallback_behavior}'"
            )

        # Validate preference_order contains valid formats
        valid_formats = {"mp4", "webm", "mkv", "mov", "avi", "m4a", "opus", "flac"}
        for fmt in self.preference_order:
            if fmt.lower() not in valid_formats:
                raise ValueError(
                    f"FormatPreferenceConfig.preference_order contains invalid format '{fmt}'. "
                    f"Valid formats: {sorted(valid_formats)}"
                )

        # Convert to lowercase for consistency
        self.preference_order = [fmt.lower() for fmt in self.preference_order]


@dataclass
class FormatFallbackConfig:
    """Multi-format download fallback configuration (US-114-005).

    Controls fallback behavior when primary video formats fail to download.
    Supports trying alternative formats in priority order, ultimately falling back
    to audio-only with automatic transcription.

    The fallback chain works as follows:
    1. Try formats in format_priority order (default: mp4 -> webm)
    2. If all video formats fail, fallback to audio-only
    3. Audio-only downloads use existing transcription pipeline

    Success rates are tracked per format for adaptive priority adjustment.

    Configure in config.yaml under download.format_fallback.
    """
    # Enable/disable multi-format fallback
    # When True, tries alternative formats on failure
    # When False, uses single format (faster but less resilient)
    enabled: bool = True

    # Format priority order (first successful format is used)
    # Valid formats: "mp4", "webm", "vp9.2", "avc", "audio-only", "m4a", "opus", "mp3", "thumbnail"
    # "audio-only" triggers transcription fallback
    # US-143-002: Added vp9.2 (HDR) to default priority
    # US-144-006: Added avc (H.264), mp3, thumbnail to default priority
    format_priority: List[str] = field(default_factory=lambda: ["mp4", "webm", "vp9.2", "avc", "mp3", "audio-only"])

    # Maximum retry attempts per format before moving to next
    # This is per-format, not total retries
    max_retries_per_format: int = 2

    # Enable adaptive priority adjustment based on success rates
    # When True, formats with higher success rates are tried first
    adaptive_priority: bool = True

    # Minimum sample size before adapting priority
    # Don't adjust priority until we have this many attempts per format
    min_samples_for_adaptation: int = 10

    # Track format success rates for logging/analysis
    # When True, logs format success rates periodically
    track_success_rates: bool = True

    # Fallback to audio-only on any video format error
    # When False, only falls back on specific "unavailable" errors
    fallback_on_any_error: bool = True

    # US-129-004: Enable adaptive format selection based on historical success rates
    # When True, dynamically reorder format_priority to prefer formats with higher success rates
    # This is the main toggle for the adaptive selection feature
    format_adaptive_selection: bool = True

    # US-129-004: Reset success rate after this many consecutive failures
    # This allows retrying formats that were marked as "bad" if they recover
    # Set to 0 to disable reset
    reset_attempts_threshold: int = 5

    # US-143-002: Enable region-based success rate tracking
    # When True, tracks success rates per region for adaptive priority
    # Regions are determined by VPN exit location or IP geolocation
    track_region_stats: bool = False

    # US-144-006: Enable thumbnail-only fallback for premium content
    # When True, falls back to thumbnail when all video/audio formats fail
    # for premium/members-only content
    enable_thumbnail_fallback: bool = False

    def __post_init__(self):
        """Validate and normalize configuration values."""
        # US-144-006: Added avc (H.264), mp3, thumbnail to valid formats
        valid_formats = {"mp4", "webm", "vp9.2", "avc", "audio-only", "m4a", "opus", "mp3", "thumbnail"}

        # Validate format_priority contains valid formats
        for fmt in self.format_priority:
            if fmt.lower() not in valid_formats:
                raise ValueError(
                    f"FormatFallbackConfig.format_priority contains invalid format '{fmt}'. "
                    f"Valid formats: {sorted(valid_formats)}"
                )

        # Ensure audio-only is at the end (it's the final fallback before thumbnail)
        if "audio-only" in self.format_priority:
            # Remove and re-append at end
            self.format_priority.remove("audio-only")
            self.format_priority.append("audio-only")

        # US-144-006: Ensure thumbnail is at the very end (final fallback)
        if "thumbnail" in self.format_priority:
            # Remove and re-append at end (after audio-only)
            self.format_priority.remove("thumbnail")
            self.format_priority.append("thumbnail")

        # Convert to lowercase for consistency
        self.format_priority = [fmt.lower() for fmt in self.format_priority]

        # Validate numeric values
        if self.max_retries_per_format < 0:
            raise ValueError(
                f"FormatFallbackConfig.max_retries_per_format must be >= 0, "
                f"got {self.max_retries_per_format}"
            )

        if self.min_samples_for_adaptation < 1:
            raise ValueError(
                f"FormatFallbackConfig.min_samples_for_adaptation must be >= 1, "
                f"got {self.min_samples_for_adaptation}"
            )


@dataclass
class TierSlotManagementConfig:
    """Per-tier concurrent download slot management (US-114-011).

    Controls maximum concurrent downloads per duration tier to optimize
    bandwidth utilization and prevent resource exhaustion.

    When enabled, each tier (short, medium, long, longer) maintains its own
    slot pool, allowing more short videos to download concurrently while
    limiting resource-intensive long video downloads.

    Configure in config.yaml under download.tier_slot_management.
    """
    # Enable/disable per-tier slot management
    # When True, each tier has its own concurrent download limit
    # When False, uses global max_concurrent limit
    enabled: bool = True

    # Maximum concurrent downloads per duration tier
    # Keys: 'short' (<2min), 'medium' (2-10min), 'long' (10-25min), 'longer' (>25min)
    # Defaults allow more short videos (faster downloads) while limiting long videos
    max_concurrent_per_tier: Dict[str, int] = field(default_factory=lambda: {
        'short': 2,    # More short videos can run concurrently
        'medium': 1,   # 1 medium video at a time
        'long': 1,     # 1 long video at a time
        'longer': 1,   # 1 longer video at a time
    })

    # Allow tier slot borrowing when one tier has available slots but another is waiting
    # When True: if short tier has 0/2 used, a long video can use an available short slot
    # When False: each tier is strictly isolated (may leave slots unused)
    allow_borrowing: bool = True

    # Maximum total concurrent downloads across all tiers (safety cap)
    # Set to None to use sum of per-tier limits instead
    max_total_concurrent: int = None

    def __post_init__(self):
        """Validate and normalize configuration values."""
        valid_tiers = {'short', 'medium', 'long', 'longer'}

        # Validate max_concurrent_per_tier keys
        for tier in self.max_concurrent_per_tier:
            if tier not in valid_tiers:
                raise ValueError(
                    f"TierSlotManagementConfig.max_concurrent_per_tier contains invalid tier '{tier}'. "
                    f"Valid tiers: {sorted(valid_tiers)}"
                )

        # Validate slot counts are positive
        for tier, count in self.max_concurrent_per_tier.items():
            if count < 1:
                raise ValueError(
                    f"TierSlotManagementConfig.max_concurrent_per_tier['{tier}'] must be >= 1, "
                    f"got {count}"
                )

        # Validate max_total_concurrent if set
        if self.max_total_concurrent is not None and self.max_total_concurrent < 1:
            raise ValueError(
                f"TierSlotManagementConfig.max_total_concurrent must be >= 1 or None, "
                f"got {self.max_total_concurrent}"
            )


@dataclass
class ErrorPatternsConfig:
    """Configurable error patterns for download error classification (US-89-008).

    Allows runtime adjustment of error patterns without code changes.
    Patterns are organized by sub-category: dns, tcp, tls, http, ffmpeg.

    Configure in config.yaml under download.error_patterns.
    """
    # DNS resolution error patterns
    dns: List[str] = field(default_factory=lambda: [
        'getaddrinfo failed',
        'Name or service not known',
        'Errno 11001',
        'nodename nor servname',
        'No address associated with hostname',
        'Temporary failure in name resolution',
        'Failed to resolve',
    ])

    # TCP connection error patterns
    tcp: List[str] = field(default_factory=lambda: [
        'Network is unreachable',
        'ConnectionResetError',
        'Connection refused',
        'Connection timed out',
    ])

    # TLS/SSL error patterns
    tls: List[str] = field(default_factory=lambda: [
        'SSL: CERTIFICATE_VERIFY_FAILED',
        'SSL: WRONG_VERSION_NUMBER',
        'SSLError',
        'SSLHandshakeError',
        'ssl_',
        'OpenSSL.SSL.Error',
        'certificate verify failed',
        'EOF occurred in violation of protocol',
        'no protocols available',
        'sslv3 alert handshake failure',
        'tlsv1 alert',
        'unsupported protocol',
        'bad_certificate',
        'certificate expired',
        'certificate has expired',
        'certificate not yet valid',
        'hostname mismatch',
        'SNI not enabled',
        'unsafe legacy renegotiation',
    ])

    # HTTP error patterns
    http: List[str] = field(default_factory=lambda: [
        'URLError',
        'HTTP Error 403',
        'HTTP Error 429',
        'HTTP Error 5',
    ])

    # FFmpeg exit code patterns (network-related)
    ffmpeg: List[str] = field(default_factory=lambda: [
        '4294967158',
        '-314',
    ])

    def to_dict(self) -> dict[str, list[str]]:
        """Convert to dict format expected by error_classification module."""
        return {
            'dns': self.dns,
            'tcp': self.tcp,
            'tls': self.tls,
            'http': self.http,
            'ffmpeg': self.ffmpeg,
        }


class ErrorHandlingAction(str, Enum):
    """Actions to take when a specific error type is encountered (US-129-005).

    Each error category can trigger a specific handling action:
    - RETRY: Retry the download (default for most errors)
    - RETRY_WITH_BACKOFF: Retry with exponential backoff
    - ESCALATE_TIER2: Escalate to Tier 2 (extractor args)
    - ESCALATE_TIER3: Escalate to Tier 3 (cookie rotation)
    - ESCALATE_TIER4: Escalate to Tier 4 (VPN rotation)
    - SKIP: Skip this video entirely
    - ABORT: Abort the entire download batch
    """
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    ESCALATE_TIER2 = "escalate_tier2"
    ESCALATE_TIER3 = "escalate_tier3"
    ESCALATE_TIER4 = "escalate_tier4"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class ErrorHandlingMapping:
    """Maps error categories to handling actions (US-129-005).

    Each field maps an error category to the action to take.
    Configure in config.yaml under download.error_handling_mapping.
    """
    # Network errors (DNS, TCP, TLS) - retry with backoff
    network: ErrorHandlingAction = ErrorHandlingAction.RETRY_WITH_BACKOFF

    # Geo-blocking errors - escalate to VPN (Tier 4)
    geo_blocked: ErrorHandlingAction = ErrorHandlingAction.ESCALATE_TIER4

    # Device limit errors - retry with backoff after delay
    device_limit: ErrorHandlingAction = ErrorHandlingAction.RETRY_WITH_BACKOFF

    # Login required errors - escalate to cookie rotation (Tier 3)
    login_required: ErrorHandlingAction = ErrorHandlingAction.ESCALATE_TIER3

    # Bot detection (403, captcha, etc.) - escalate tiers
    bot_detection: ErrorHandlingAction = ErrorHandlingAction.ESCALATE_TIER2

    # Rate limit (429) - retry with backoff
    rate_limit: ErrorHandlingAction = ErrorHandlingAction.RETRY_WITH_BACKOFF

    # Timeout errors - retry with backoff
    timeout: ErrorHandlingAction = ErrorHandlingAction.RETRY_WITH_BACKOFF

    # Video-specific errors (removed, unavailable) - skip
    video_specific: ErrorHandlingAction = ErrorHandlingAction.SKIP

    # Unknown errors - retry with backoff
    unknown: ErrorHandlingAction = ErrorHandlingAction.RETRY_WITH_BACKOFF


@dataclass
class MetricsExporterConfig:
    """Metrics exporter configuration for download metrics (US-93-011, US-143-009).

    Enables Prometheus and JSON export of download metrics for production monitoring.
    """
    # Enable/disable metrics exporter
    enabled: bool = True
    # Export interval in seconds (configurable)
    export_interval_seconds: float = 60.0
    # Output directory for exported metrics files
    output_dir: str = "output/download_metrics"
    # Export to Prometheus text format
    export_prometheus: bool = True
    # Export to JSON format
    export_json: bool = True
    # Export to CSV format (US-114-003)
    export_csv: bool = False
    # Export format: "json", "csv", or "both" (default: json) (US-114-003)
    metrics_export_format: str = "json"

    # US-143-009: HTTP server for /metrics endpoint
    # Enable HTTP server for Prometheus scraping
    enable_http_server: bool = False
    # HTTP port for metrics endpoint (default: 9090)
    http_port: int = 9090
    # HTTP host to bind to (default: "0.0.0.0")
    http_host: str = "0.0.0.0"


@dataclass
class SegmentValidationConfig:
    """Configuration for pre-queue segment validation (US-129-010).

    Validates segments before adding to download queue to prevent
    invalid segments from being processed.
    """
    # Minimum segment duration in seconds (default: 1.0s).
    # Segments shorter than this are skipped.
    # Set to 0 to disable duration validation.
    min_duration_seconds: float = 1.0

    # Maximum segment duration in seconds.
    # Segments longer than this are skipped.
    # Set to 0 to disable max duration validation.
    max_duration_seconds: float = 0.0

    # Validate segment start/end times are within video duration bounds.
    # Requires video duration info from video_search_results.
    validate_bounds: bool = True

    # Filter duplicate segments (same video_id, start, end).
    # Note: Deduplication is already done in _collect_matched_segments,
    # but this enables explicit duplicate detection with logging.
    filter_duplicates: bool = True

    # Validation strictness mode:
    # - "lenient": Skip invalid segments and continue (default)
    # - "strict": Raise error on first invalid segment
    strictness: str = "lenient"

    # Log skipped segments with reasons.
    # Valid reasons: "too_short", "too_long", "out_of_bounds", "duplicate"
    log_skipped: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if self.min_duration_seconds < 0:
            raise ValueError(
                f"min_duration_seconds must be >= 0, got {self.min_duration_seconds}"
            )
        if self.max_duration_seconds < 0:
            raise ValueError(
                f"max_duration_seconds must be >= 0, got {self.max_duration_seconds}"
            )
        if self.min_duration_seconds > 0 and self.max_duration_seconds > 0:
            if self.min_duration_seconds > self.max_duration_seconds:
                raise ValueError(
                    f"min_duration_seconds ({self.min_duration_seconds}) must be "
                    f"<= max_duration_seconds ({self.max_duration_seconds})"
                )
        valid_strictness = ("lenient", "strict")
        if self.strictness not in valid_strictness:
            raise ValueError(
                f"strictness must be one of {valid_strictness}, got '{self.strictness}'"
            )


@dataclass
class ChecksumValidationConfig:
    """Configuration for download segment checksum validation (US-143-012).

    Validates downloaded segment integrity using SHA256 checksums.
    """
    # Enable checksum validation after download.
    # When enabled, SHA256 hash is calculated for each downloaded segment
    # and verified against expected value (if available) or logged for metrics.
    enabled: bool = False

    # Algorithm to use for checksum calculation.
    # Supported: "sha256", "sha1", "md5"
    algorithm: str = "sha256"

    # Retry on checksum failure (download was corrupted).
    # Only retries if the checksum doesn't match, suggesting download corruption.
    retry_on_failure: bool = True

    # Maximum retries for checksum failures per segment.
    max_retries: int = 2

    # Minimum file size in bytes for checksum validation.
    # Very small files (<1KB) may not be worth validating.
    min_file_size_bytes: int = 1024

    # Log checksum validation results to metrics.
    log_to_metrics: bool = True

    # Verify file size matches expected (if provided by yt-dlp).
    verify_file_size: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        valid_algorithms = ("sha256", "sha1", "md5")
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"algorithm must be one of {valid_algorithms}, got '{self.algorithm}'"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {self.max_retries}"
            )
        if self.min_file_size_bytes < 0:
            raise ValueError(
                f"min_file_size_bytes must be >= 0, got {self.min_file_size_bytes}"
            )


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
    # Download alternatives from V2-V10 (not just V1 primary)
    download_all_tracks: bool = False

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
    max_search_pool: int = 100  # Maximum search pool size (for adaptive sizing)

    # Timeout settings
    # Search timeout: If YouTube search takes longer, trigger keyword remix (max 2 attempts)
    search_timeout: int = 30  # Seconds for search metadata subprocess (triggers remix on timeout)
    download_timeout: int = 120  # Default seconds per video download (used for 'short' tier)

    # Stall timeout: seconds of no download progress before killing the process.
    # Unlike download_timeout (total time), this only triggers when yt-dlp produces
    # no output, allowing slow-but-progressing downloads to continue.
    # Default: same as tier timeout (uses download_timeout/download_timeouts values)
    stall_timeout: int = 60  # Seconds of zero yt-dlp output before killing (0 = tier timeout, not recommended)

    # Socket timeout: seconds before yt-dlp gives up on a stalled TCP connection.
    # Used by both subprocess calls (--socket-timeout) and Python API calls (socket_timeout key).
    # Without this, ydl.download() can block indefinitely on stalled connections.
    socket_timeout: int = 30

    # Tier-specific download timeouts (longer videos need more time)
    # Keys: 'short', 'medium', 'long', 'longer'
    download_timeouts: Dict[str, int] = field(default_factory=lambda: {
        'short': 120,    # 2 min timeout for videos <2 min
        'medium': 300,   # 5 min timeout for videos 2-10 min
        'long': 600,     # 10 min timeout for videos 10-25 min
        'longer': 900,   # 15 min timeout for videos 25-50 min
    })

    # Adaptive timeout based on video file size (US-93-005)
    # Calculates: base_timeout + (estimated_size_mb * size_multiplier)
    # This prevents premature timeouts on large files while avoiding long waits on small files
    # Set enabled: false to disable and use tier-based timeouts only
    adaptive_timeout_enabled: bool = True

    # Base timeout in seconds (added to size-based calculation)
    # This provides a minimum baseline for connection/setup time
    adaptive_timeout_base: int = 30

    # Multiplier: seconds to add per MB of estimated file size
    # For 100MB video: 30 + (100 * 0.5) = 80 second timeout
    # For 500MB video: 30 + (500 * 0.5) = 280 second timeout
    adaptive_timeout_multiplier: float = 0.5

    # Maximum adaptive timeout cap (seconds)
    # Prevents extremely large videos from getting excessive timeouts
    adaptive_timeout_max: int = 600

    # Estimate file size from duration using these bitrate assumptions (MB/min):
    # Used when exact filesize is not available from metadata
    # Format: {resolution: MB per minute}
    adaptive_timeout_size_estimates: Dict[str, float] = field(default_factory=lambda: {
        '2160': 25.0,  # 4K
        '1440': 15.0,  # 2K
        '1080': 5.0,   # 1080p
        '720': 2.5,    # 720p
        '480': 1.5,    # 480p
        'default': 3.0  # Unknown resolution
    })

    # Audio-first download pipeline (enable per-project for faster downloads)
    audio_first: AudioFirstConfig = field(default_factory=AudioFirstConfig)

    # Caption-first mode: fetch YouTube captions before video download
    # Enables faster matching with lower bandwidth - falls back to Whisper if unavailable
    caption_first: CaptionFirstConfig = field(default_factory=CaptionFirstConfig)

    # Zero-download remix: auto-retry with alternative keywords when 0 results
    zero_download_remix: ZeroDownloadRemixConfig = field(default_factory=ZeroDownloadRemixConfig)

    # Speech screening: pre-screen videos by transcribing first N seconds
    # Rejects videos with speech in intro to ensure only B-roll footage
    speech_screening: SpeechScreeningConfig = field(default_factory=SpeechScreeningConfig)

    # Cookie rotation: rotate between multiple cookie files on rate limit
    cookie_rotation: CookieRotationConfig = field(default_factory=CookieRotationConfig)

    # Rate limit backoff: progressive delay before cookie rotation
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Region-specific backoff: apply geographic multipliers based on VPN country (US-109-011)
    region_backoff: RegionBackoffConfig = field(default_factory=RegionBackoffConfig)

    # VPN integration: switch VPN servers when cookies are exhausted
    vpn: VPNConfig = field(default_factory=VPNConfig)

    # Mullvad VPN: Tier 4 bypass using Mullvad CLI for IP rotation
    # Activates after cookie rotation (Tier 3) is exhausted
    mullvad: MullvadConfig = field(default_factory=MullvadConfig)

    # YouTube Data API: programmatic access for search, metadata, captions
    # Alternative to yt-dlp with quota tracking and auto-fallback
    youtube_api: YouTubeAPIConfig = field(default_factory=YouTubeAPIConfig)

    # Speed tracking: monitor download speeds for adaptive timeouts
    speed_tracking: SpeedTrackingConfig = field(default_factory=SpeedTrackingConfig)

    # US-123-010: Self-regulate download speed based on error patterns to prevent rate limits
    self_regulate: SelfRegulateConfig = field(default_factory=SelfRegulateConfig)

    # Circuit breaker: pause searches after consecutive failures
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # Batch retry: collect rate-limited videos and retry after delay
    batch_retry: BatchRetryConfig = field(default_factory=BatchRetryConfig)

    # US-129-002: Per-video retry budget to prevent infinite loops
    # Tracks remaining retry attempts and backoff time per video ID
    retry_budget: DownloadRetryBudgetConfig = field(default_factory=DownloadRetryBudgetConfig)

    # Browser impersonation: spoof TLS fingerprints to bypass bot detection
    # Always-on by default (Tier 1). Auto-detects curl_cffi targets at startup.
    impersonation: ImpersonationConfig = field(default_factory=ImpersonationConfig)

    # Extractor args escalation: Tier 2 bypass adds --extractor-args player_client
    # Activates after repeated 403 errors when impersonation alone isn't enough.
    extractor_args: ExtractorArgsConfig = field(default_factory=ExtractorArgsConfig)

    # Cross-keyword rate limit budget: shared resource limits across keywords
    # Controls how many rotations, backoff time, and VPN switches are available per session
    rate_limit_budget: RateLimitBudgetConfig = field(default_factory=RateLimitBudgetConfig)

    # Rate limit predictor: proactive budget adjustment based on historical patterns (US-113-003)
    # When enabled, predicts rate limit likelihood and increases budget allocation proactively
    rate_limit_predictor: RateLimitPredictorConfig = field(default_factory=RateLimitPredictorConfig)

    # Adaptive backoff: time-of-day based backoff adjustment (US-114-004)
    # When enabled, applies longer backoff during historically low-success hours
    adaptive_backoff: AdaptiveBackoffConfig = field(default_factory=AdaptiveBackoffConfig)

    # Error patterns for download error classification (US-89-008)
    # Allows runtime adjustment without code changes. If not provided, uses defaults.
    error_patterns: ErrorPatternsConfig | None = None

    # Error handling mapping (US-129-005)
    # Maps error categories to handling actions (retry, escalate, skip, abort)
    # If not provided, uses ErrorHandlingMapping with defaults.
    error_handling_mapping: ErrorHandlingMapping | None = None

    # Adaptive severity configuration (US-89-012)
    # Tracks error frequency and auto-escalates severity on repeated errors.
    # If not provided, uses AdaptiveSeverityConfig with defaults.
    adaptive_severity: AdaptiveSeverityConfig | None = None

    # Bandwidth throttling configuration (US-93-007)
    # Limits download bandwidth to prevent consuming all bandwidth during pipeline execution.
    # Uses yt-dlp --downloader-args to pass rate limiting to ffmpeg.
    # If not provided, uses BandwidthThrottleConfig with defaults.
    bandwidth_throttle: BandwidthThrottleConfig | None = None

    # Download resume configuration (US-93-008)
    # Enables resuming partial downloads when interrupted, reducing bandwidth waste.
    # Uses yt-dlp's --continue flag to resume from last byte position.
    # If not provided, uses DownloadResumeConfig with defaults.
    download_resume: DownloadResumeConfig | None = None

    # Checkpoint compression configuration (US-114-009)
    # Enables gzip compression for download checkpoint files to reduce disk space.
    # If not provided, uses CheckpointCompressionConfig with defaults.
    checkpoint_compression: CheckpointCompressionConfig | None = None

    # Format preference configuration (US-93-009)
    # Controls video format (container/codec) selection for yt-dlp downloads.
    # Allows preferring specific formats (mp4, webm) over others.
    # If not provided, uses FormatPreferenceConfig with defaults.
    format_preference: FormatPreferenceConfig | None = None

    # Multi-format download fallback configuration (US-114-005)
    # Controls fallback behavior when primary video formats fail.
    # Supports trying alternative formats in priority order, then audio-only with transcription.
    # If not provided, uses FormatFallbackConfig with defaults.
    format_fallback: FormatFallbackConfig | None = None

    # Per-tier concurrent download slot management (US-114-011)
    # Controls maximum concurrent downloads per duration tier to optimize bandwidth utilization.
    # If not provided, uses TierSlotManagementConfig with defaults.
    tier_slot_management: TierSlotManagementConfig | None = None

    # Metrics exporter configuration (US-93-011)
    # Enables Prometheus/JSON export of download metrics for production monitoring.
    # If not provided, uses MetricsExporterConfig with defaults.
    metrics_exporter: MetricsExporterConfig | None = None

    # Segment download settings (used by DOWNLOAD_SEGMENTS stage)
    # Delay between segment download requests (seconds).
    # Prevents YouTube rate-limiting when downloading many segments back-to-back.
    # Adaptive: doubles after each failure (capped at 30s), resets on success.
    # yt-dlp recommends `-t sleep` for rate limits; this is the equivalent.
    segment_request_delay: float = 1.0
    # Maximum adaptive delay after consecutive failures (seconds)
    segment_request_delay_max: float = 30.0
    # US-85-008: Random jitter factor applied to delay to prevent thundering herd.
    # Jitter is +/- this factor (e.g., 0.25 = +/-25%). Set to 0 to disable.
    segment_request_delay_jitter: float = 0.25
    # Buffer seconds to add before/after each matched segment for editing flexibility
    segment_buffer: float = 5.0
    # yt-dlp format string for segment downloads (height capped by segment_max_resolution)
    segment_format: str = "best[height<={segment_max_resolution}]"
    # Maximum video resolution (height) for segment downloads
    segment_max_resolution: int = 1080
    # Socket timeout for segment downloads (seconds). 0 = use main socket_timeout value.
    segment_socket_timeout: int = 0
    # Stall timeout for segment downloads (seconds). Wraps ydl.download() in a
    # ThreadPoolExecutor with this timeout to detect hangs where the download
    # blocks indefinitely (e.g., ffmpeg post-processing stalls, stream-level hangs).
    # Unlike socket_timeout (covers HTTP sockets only), this covers the entire
    # ydl.download() call including ffmpeg merging/remuxing. 0 = no stall detection.
    segment_stall_timeout: int = 120
    # US-81-003: Save incremental checkpoint every N segment downloads.
    # Enables partial resume for long-running DOWNLOAD_SEGMENTS stages.
    # 0 = checkpoint after every download (maximum resilience, more I/O).
    segment_checkpoint_every_n: int = 10
    # Number of concurrent segment download workers. Each worker uses a
    # separate cookie file from cookie_rotation.cookie_files (round-robin).
    # Set to 1 for sequential downloads (default). Recommended: match the
    # number of available cookie files (e.g., 3 cookies → 3 workers).
    segment_concurrent_workers: int = 1

    # US-129-010: Pre-queue segment validation configuration.
    # Validates segments before adding to download queue.
    # Validates: min/max duration, video bounds, duplicates.
    segment_validation: SegmentValidationConfig = field(default_factory=SegmentValidationConfig)
    # US-143-012: Checksum validation for downloaded segments
    checksum_validation: ChecksumValidationConfig = field(default_factory=ChecksumValidationConfig)

    # Bot-detection tier floor: after N consecutive 403/bot-detection errors across
    # ALL video IDs (stage-level), new downloads start at max escalation tier instead
    # of Tier 1. This prevents wasting time on doomed Tier 1 requests when YouTube
    # is broadly blocking. Resets on any successful download. 0 = disabled.
    bot_detection_tier_floor_threshold: int = 5

    # Bot-detection abort threshold: after N total bot-detection errors across all
    # videos in the stage, abort the entire download loop. This prevents the stage
    # from running to completion hitting YouTube's block wall for every remaining
    # segment when cookies are broken or impersonation is defeated. Progress is
    # checkpointed before aborting so --resume can pick up where it left off.
    # 0 = disabled (never abort on bot-detection errors).
    bot_detection_abort_threshold: int = 10

    # US-123-009: Graceful tier degradation - when a tier is struggling (>60% failure rate
    # in recent window), degrade gracefully by reducing parameters rather than skipping
    # to the next tier. This is less aggressive than full tier escalation.
    tier_graceful_degradation: bool = True

    # Failure rate threshold for graceful degradation: when a tier fails more than this
    # percentage in the recent window, mark it as struggling and trigger degradation.
    # Default: 0.6 (60% failure rate)
    tier_degradation_threshold: float = 0.6

    # Window size (in seconds) for tracking tier failure rates for graceful degradation
    tier_degradation_window: float = 300.0  # 5 minutes

    # Network failure threshold: after N consecutive network failures (DNS, connection
    # refused, etc.), abort the download loop. These indicate systemic network issues
    # that won't resolve by retrying more videos. Resets on any successful download.
    network_failure_threshold: int = 3

    # FFmpeg location (for segment downloads, set if not in PATH)
    # Example: "C:/ffmpeg/bin/ffmpeg.exe" or "/usr/local/bin/ffmpeg"
    ffmpeg_location: str = ""

    # Retry settings for failed downloads
    # Used by DownloadStage to retry failed video/audio downloads
    max_retries: int = 3  # Maximum retry attempts per video
    retry_delay: float = 2.0  # Base delay between retries (seconds)
    retry_backoff: float = 2.0  # Exponential backoff multiplier

    # Parallel download settings
    # Number of concurrent downloads to run simultaneously
    # Higher values = faster downloads but more bandwidth/CPU usage
    # Note: Currently sequential, this setting is prepared for future parallel support
    parallel_workers: int = 4  # Concurrent download workers (1-8 recommended)
    max_concurrent: int = None  # Max concurrent for DownloadCoordinator (defaults to parallel_workers)

    def __post_init__(self):
        """Convert nested dicts to proper dataclass instances."""
        if isinstance(self.llm_title_filter, dict):
            self.llm_title_filter = LLMTitleFilterConfig(**self.llm_title_filter)
        if isinstance(self.audio_first, dict):
            self.audio_first = AudioFirstConfig(**self.audio_first)
        if isinstance(self.caption_first, dict):
            self.caption_first = CaptionFirstConfig(**self.caption_first)
        if isinstance(self.zero_download_remix, dict):
            self.zero_download_remix = ZeroDownloadRemixConfig(**self.zero_download_remix)
        if isinstance(self.speech_screening, dict):
            self.speech_screening = SpeechScreeningConfig(**self.speech_screening)
        if isinstance(self.cookie_rotation, dict):
            self.cookie_rotation = CookieRotationConfig(**self.cookie_rotation)
        if isinstance(self.rate_limit, dict):
            self.rate_limit = RateLimitConfig(**self.rate_limit)
        if isinstance(self.region_backoff, dict):
            self.region_backoff = RegionBackoffConfig(**self.region_backoff)
        if isinstance(self.vpn, dict):
            self.vpn = VPNConfig(**self.vpn)
        if isinstance(self.mullvad, dict):
            self.mullvad = MullvadConfig(**self.mullvad)
        if isinstance(self.speed_tracking, dict):
            self.speed_tracking = SpeedTrackingConfig(**self.speed_tracking)
        # US-123-010: Handle self_regulate - can be None, dict, or SelfRegulateConfig
        if isinstance(self.self_regulate, dict):
            self.self_regulate = SelfRegulateConfig(**self.self_regulate)
        # If None or not provided, keep as None (backward compat - use defaults)
        if isinstance(self.circuit_breaker, dict):
            self.circuit_breaker = CircuitBreakerConfig(**self.circuit_breaker)
        if isinstance(self.batch_retry, dict):
            self.batch_retry = BatchRetryConfig(**self.batch_retry)
        if isinstance(self.impersonation, dict):
            self.impersonation = ImpersonationConfig(**self.impersonation)
        if isinstance(self.extractor_args, dict):
            self.extractor_args = ExtractorArgsConfig(**self.extractor_args)
        if isinstance(self.rate_limit_budget, dict):
            self.rate_limit_budget = RateLimitBudgetConfig(**self.rate_limit_budget)

        # US-89-008: Handle error_patterns - can be None, dict, or ErrorPatternsConfig
        if isinstance(self.error_patterns, dict):
            self.error_patterns = ErrorPatternsConfig(**self.error_patterns)
        # If None or not provided, keep as None (backward compat - use defaults)

        # US-129-005: Handle error_handling_mapping - can be None, dict, or ErrorHandlingMapping
        if isinstance(self.error_handling_mapping, dict):
            self.error_handling_mapping = ErrorHandlingMapping(**self.error_handling_mapping)
        # If None or not provided, keep as None (backward compat - use defaults)

        # US-89-012: Handle adaptive_severity - can be None, dict, or AdaptiveSeverityConfig
        if isinstance(self.adaptive_severity, dict):
            self.adaptive_severity = AdaptiveSeverityConfig(**self.adaptive_severity)
        # If None or not provided, keep as None (backward compat - use defaults)

        # US-93-007: Handle bandwidth_throttle - can be None, dict, or BandwidthThrottleConfig
        if isinstance(self.bandwidth_throttle, dict):
            self.bandwidth_throttle = BandwidthThrottleConfig(**self.bandwidth_throttle)
        # If None or not provided, keep as None (backward compat - use defaults)

        # US-93-009: Handle format_preference - can be None, dict, or FormatPreferenceConfig
        if isinstance(self.format_preference, dict):
            self.format_preference = FormatPreferenceConfig(**self.format_preference)
        # If None or not provided, keep as None (backward compat - use defaults)

        # US-114-005: Handle format_fallback - can be None, dict, or FormatFallbackConfig
        if isinstance(self.format_fallback, dict):
            self.format_fallback = FormatFallbackConfig(**self.format_fallback)
        # If None or not provided, keep as None (backward compat - use defaults)

        # US-93-011: Handle metrics_exporter - can be None, dict, or MetricsExporterConfig
        if isinstance(self.metrics_exporter, dict):
            self.metrics_exporter = MetricsExporterConfig(**self.metrics_exporter)
        # If None or not provided, keep as None (backward compat - use defaults)

        # US-114-009: Handle checkpoint_compression - can be None, dict, or CheckpointCompressionConfig
        if isinstance(self.checkpoint_compression, dict):
            self.checkpoint_compression = CheckpointCompressionConfig(**self.checkpoint_compression)
        # If None or not provided, keep as None (backward compat - use defaults)

        # Validate parallel_workers >= 1 (positive integer)
        if self.parallel_workers < 1:
            raise ValueError(
                f"DownloadConfig.parallel_workers={self.parallel_workers} must be >= 1. "
                f"Check download.parallel_workers in config.yaml"
            )

        # Default max_concurrent to parallel_workers if not set
        if self.max_concurrent is None:
            self.max_concurrent = self.parallel_workers

        # Validate max_concurrent >= 1
        if self.max_concurrent < 1:
            raise ValueError(
                f"DownloadConfig.max_concurrent={self.max_concurrent} must be >= 1. "
                f"Check download.max_concurrent in config.yaml"
            )

        # Validate cookies_from_browser is a known browser or empty (disabled)
        _valid_browsers = {'firefox', 'chrome', 'edge', 'safari', 'opera', 'brave', ''}
        if self.cookies_from_browser not in _valid_browsers:
            raise ValueError(
                f"DownloadConfig.cookies_from_browser must be one of {sorted(_valid_browsers - {''})} "
                f"or '' (empty to disable), got '{self.cookies_from_browser}'. "
                f"Check config.yaml under download.cookies_from_browser"
            )


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
    delay_between_downloads: float = 1.0

    # US-129-008: Priority boost for early voiceover segments
    # Earlier voiceover segments get higher priority in download queue
    # to enable faster iterative match feedback
    priority_boost_for_early_segments: float = 1.5
