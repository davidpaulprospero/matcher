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
    'AudioFirstConfig',
    'SpeechScreeningConfig',
    'CookieRotationConfig',
    'RateLimitConfig',
    'SpeedTrackingConfig',
    'CircuitBreakerConfig',
    'BatchRetryConfig',
    'VPNConfig',
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


@dataclass
class SpeedTrackingConfig:
    """Download speed monitoring for adaptive timeouts.

    Tracks actual download speeds and adjusts timeouts dynamically based on
    network conditions. On slow networks, timeouts can be extended up to
    max_timeout_multiplier to prevent unnecessary timeout failures.

    How it works:
      1. After each download, records bytes downloaded and time taken
      2. Maintains a sliding window of the last N downloads (window_size)
      3. Calculates average speed across the window
      4. If speed < min_speed_mbps, extends timeouts proportionally
      5. State is persisted in checkpoint for resume scenarios

    Example with defaults (min_speed_mbps=1.0, max_timeout_multiplier=2.0):
      - Network at 2.0 MB/s: no adjustment
      - Network at 0.5 MB/s: timeout extended 2x
      - Network at 0.25 MB/s: timeout extended 2x (capped)
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
    """
    # Enable/disable circuit breaker
    enabled: bool = True

    # Number of consecutive failures before circuit trips (opens)
    consecutive_failures_threshold: int = 5

    # Duration to pause after circuit trips (seconds)
    pause_seconds: float = 60.0


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
    """
    # Enable/disable batch retry queue
    enabled: bool = True

    # Delay before processing retry queue (seconds)
    # Should be long enough for rate limit window to pass
    delay_seconds: float = 120.0

    # Maximum retry passes per download session
    # After this many batch retries, give up on remaining failures
    max_passes: int = 2


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
