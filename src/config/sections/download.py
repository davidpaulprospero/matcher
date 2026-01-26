"""Download configuration: Video download settings, audio-first mode, speech screening.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

__all__ = [
    'RemixConfig',
    'ZeroDownloadRemixConfig',
    'EnhancedFeaturesConfig',
    'LLMTitleFilterConfig',
    'CaptionFirstConfig',
    'AudioFirstConfig',
    'SpeechScreeningConfig',
    'CookieRotationConfig',
    'RateLimitConfig',
    'SpeedTrackingConfig',
    'CircuitBreakerConfig',
    'BatchRetryConfig',
    'VPNConfig',
    'ImpersonationConfig',
    'ExtractorArgsConfig',
    'DownloadConfig',
    'DownloadingConfig',
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
class CaptionFirstConfig:
    """Caption-first mode configuration.

    When enabled, fetches YouTube captions BEFORE video download, enabling
    faster matching with lower bandwidth. If captions are unavailable,
    falls back to Whisper transcription.

    Enable per-project in project_config.yaml:
        download:
          caption_first:
            enabled: true

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
    # Enable/disable caption-first mode
    enabled: bool = False  # Disabled by default, enable per-project

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

    # Coverage threshold (US-004)
    # Minimum coverage ratio (0.0-1.0) for caption quality
    # Videos below this threshold are flagged for potential transcription fallback
    # High coverage = better matching accuracy
    min_coverage_threshold: float = 0.5

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

    # Timing validation epsilon (US-007 Sprint 6)
    # Tolerance in milliseconds for floating-point precision at video duration boundary.
    # Captions ending within this epsilon of video duration are treated as valid.
    # Example: caption at 299.999s in 300s video is valid with 100ms epsilon.
    # Set to 0 for exact matching (may cause false positives from float precision).
    timing_epsilon_ms: float = 100.0

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

    # Worker-level progress tracking (US-008 Sprint 8)
    # Time threshold in seconds for considering a worker "stuck" on a video.
    # Workers exceeding this threshold are reported in progress callbacks.
    # Set higher for slow networks or videos with many caption tracks.
    stuck_worker_threshold: float = 60.0


@dataclass
class AudioFirstConfig:
    """Audio-first download pipeline configuration.

    When enabled, downloads audio (MP3) first for fast transcription/matching,
    then downloads only the matched video segments. This dramatically reduces
    download time and storage usage.

    Enable per-project in project_config.yaml:
        download:
          audio_first:
            enabled: true
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

    def __post_init__(self):
        """Validate configuration values."""
        # Ensure rotate_on_errors is a non-empty list when rotation is enabled
        if self.enabled and not self.rotate_on_errors:
            raise ValueError(
                "cookie_rotation.rotate_on_errors must be a non-empty list when "
                "cookie rotation is enabled. Add at least one error pattern like '429'."
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

    # Number of downloads to track in sliding window
    # Smaller = more responsive, larger = more stable
    window_size: int = 5

    # Minimum expected speed in MB/s
    # Below this, timeouts start getting extended
    min_speed_mbps: float = 1.0

    # Maximum timeout extension multiplier
    # 2.0 means timeout can be at most doubled
    max_timeout_multiplier: float = 2.0

    # Enable adaptive timeout (use speed data to extend timeouts)
    # If False, speeds are tracked but timeouts are not adjusted
    enable_adaptive_timeout: bool = True

    # Rate limit signal threshold in MB/s
    # When speed drops below this for consecutive samples, signals potential rate limiting
    # Default 0.1 MB/s (100 KB/s) - near-stalled downloads indicate throttling
    rate_limit_signal_threshold: float = 0.1

    # Consecutive slow samples before emitting rate limit signal
    # Requires this many samples below threshold to trigger signal
    consecutive_slow_samples: int = 3


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

    # Respect circuit breaker state when processing retries
    # If True, wait for circuit breaker to recover before retrying
    # If False, retry immediately after delay_seconds regardless of circuit breaker
    respect_circuit_breaker: bool = True

    # Wait for cookie cooldown before processing retries
    # If True, check if any cookies are in cooldown and extend delay if needed
    # If False, proceed with retry even if cookies are in cooldown
    wait_for_cookie_cooldown: bool = True


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
class ImpersonationConfig:
    """Browser impersonation configuration for yt-dlp TLS fingerprint bypass.

    Uses curl_cffi impersonation targets to spoof browser TLS fingerprints,
    defeating YouTube bot detection that relies on TLS ClientHello analysis.

    When enabled (default), every yt-dlp subprocess call includes
    ``--impersonate <target>`` with round-robin rotation across detected targets.

    Auto-detection runs ``yt-dlp --list-impersonate-targets`` at startup to
    discover available targets from the installed curl_cffi library.

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

    # Number of consecutive 403 errors before escalating to next tier
    escalation_threshold: int = 2

    # Cooldown before de-escalating back to a lower tier (seconds)
    # After this period without errors, tier may be lowered
    cooldown_seconds: float = 300.0

    # Maximum escalation tier (1=impersonate, 2=extractor-args, 3=full bypass)
    max_tier: int = 3


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
    max_search_pool: int = 100  # Maximum search pool size (for adaptive sizing)

    # Timeout settings
    # Search timeout: If YouTube search takes longer, trigger keyword remix (max 2 attempts)
    search_timeout: int = 30  # Seconds for search metadata subprocess (triggers remix on timeout)
    download_timeout: int = 120  # Default seconds per video download (used for 'short' tier)

    # Tier-specific download timeouts (longer videos need more time)
    # Keys: 'short', 'medium', 'long', 'longer'
    download_timeouts: Dict[str, int] = field(default_factory=lambda: {
        'short': 120,    # 2 min timeout for videos <2 min
        'medium': 300,   # 5 min timeout for videos 2-10 min
        'long': 600,     # 10 min timeout for videos 10-25 min
        'longer': 900,   # 15 min timeout for videos 25-50 min
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

    # VPN integration: switch VPN servers when cookies are exhausted
    vpn: VPNConfig = field(default_factory=VPNConfig)

    # Speed tracking: monitor download speeds for adaptive timeouts
    speed_tracking: SpeedTrackingConfig = field(default_factory=SpeedTrackingConfig)

    # Circuit breaker: pause searches after consecutive failures
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # Batch retry: collect rate-limited videos and retry after delay
    batch_retry: BatchRetryConfig = field(default_factory=BatchRetryConfig)

    # Browser impersonation: spoof TLS fingerprints to bypass bot detection
    # Always-on by default (Tier 1). Auto-detects curl_cffi targets at startup.
    impersonation: ImpersonationConfig = field(default_factory=ImpersonationConfig)

    # Extractor args escalation: Tier 2 bypass adds --extractor-args player_client
    # Activates after repeated 403 errors when impersonation alone isn't enough.
    extractor_args: ExtractorArgsConfig = field(default_factory=ExtractorArgsConfig)

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
        if isinstance(self.vpn, dict):
            self.vpn = VPNConfig(**self.vpn)
        if isinstance(self.speed_tracking, dict):
            self.speed_tracking = SpeedTrackingConfig(**self.speed_tracking)
        if isinstance(self.circuit_breaker, dict):
            self.circuit_breaker = CircuitBreakerConfig(**self.circuit_breaker)
        if isinstance(self.batch_retry, dict):
            self.batch_retry = BatchRetryConfig(**self.batch_retry)
        if isinstance(self.impersonation, dict):
            self.impersonation = ImpersonationConfig(**self.impersonation)
        if isinstance(self.extractor_args, dict):
            self.extractor_args = ExtractorArgsConfig(**self.extractor_args)


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
