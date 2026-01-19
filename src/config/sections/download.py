"""Download configuration: Video download settings, audio-first mode, speech screening.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict
from enum import IntEnum

__all__ = [
    'RemixConfig',
    'ZeroDownloadRemixConfig',
    'EnhancedFeaturesConfig',
    'LLMTitleFilterConfig',
    'AudioFirstConfig',
    'CaptionFirstConfig',
    'SpeechScreeningConfig',
    'DownloadConfig',
    'DownloadingConfig',
    'RateLimitBypassConfig',
    'BypassTier',
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
class CaptionFirstConfig:
    """Caption-first download pipeline configuration.

    When enabled, fetches YouTube captions/subtitles first before any media download.
    Videos with captions skip audio download and Whisper transcription entirely.
    Only matched video segments are downloaded after matching.

    This is even faster than audio-first mode since no audio download or
    transcription is needed for videos that have captions.

    Enable per-project in project_config.yaml:
        download:
          caption_first:
            enabled: true
    """
    # Enable/disable caption-first mode
    enabled: bool = False  # Disabled by default, enable per-project

    # Prefer manual captions over auto-generated
    # Manual captions are typically higher quality
    prefer_manual_captions: bool = True

    # Languages to try for captions (in order of preference)
    languages: List[str] = field(default_factory=lambda: ["en", "en-US", "en-GB"])

    # Fall back to audio download + Whisper if no captions available
    fallback_to_audio: bool = True

    # Minimum caption coverage (fraction of video duration)
    # Videos with sparse captions may need Whisper fallback
    min_caption_coverage: float = 0.3

    # Confidence boost for manual captions (vs auto-captions and Whisper)
    # Auto-captions get no boost (neutral vs Whisper)
    confidence_boost_manual: float = 0.1

    # Cache downloaded captions for reuse
    cache_captions: bool = True

    # Timeout for caption fetch per video (seconds)
    fetch_timeout: int = 30


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


class BypassTier(IntEnum):
    """Rate limit bypass tier levels.

    Ordered from fastest/least intrusive to slowest/most compatible.
    System escalates through tiers when errors occur.
    """
    IMPERSONATE_CHROME = 1   # Chrome impersonation (fastest, no auth)
    IMPERSONATE_SAFARI = 2   # Safari impersonation (different fingerprint)
    TV_EMBEDDED = 3          # tv_embedded player (best for subtitles, no PO Token)
    BROWSER_COOKIES = 4      # Browser cookies + web player
    IOS_CREATOR = 5          # ios_creator player (different API endpoint)
    ANDROID_VR = 6           # android_vr player (alternative mobile)
    STANDARD = 7             # Plain yt-dlp (most compatible fallback)


@dataclass
class RateLimitBypassConfig:
    """Configuration for YouTube rate limit bypass strategies.

    Multi-tier system that cycles through different techniques when errors occur.
    Automatically escalates on 403, rate limits, PO Token errors, and bot checks.

    Tiers (in escalation order):
    1. Impersonate Chrome (fastest, curl_cffi, no auth)
    2. Impersonate Safari (different TLS fingerprint)
    3. tv_embedded player (works for subtitles without PO Token)
    4. Browser cookies + web player (for age-gated content)
    5. ios_creator player (different API, some geo-restrictions)
    6. android_vr player (alternative mobile endpoint)
    7. Standard yt-dlp (most compatible fallback)
    """
    # Tier 1: Chrome impersonation (preferred - fastest)
    tier1_enabled: bool = True
    tier1_target: str = "Chrome-131:Android-14"

    # Tier 2: Safari impersonation (different fingerprint)
    tier2_enabled: bool = True
    tier2_target: str = "Safari-18.2:macOS-15"

    # Tier 3: tv_embedded player (best for subtitles - no PO Token needed)
    tier3_enabled: bool = True
    tier3_player: str = "tv_embedded"

    # Tier 4: Browser cookies + web player
    tier4_enabled: bool = True
    tier4_browser: str = "firefox"  # firefox, chrome, edge, brave
    tier4_player: str = "web"

    # Tier 5: ios_creator player (different API endpoint)
    tier5_enabled: bool = True
    tier5_player: str = "ios_creator"

    # Tier 6: android_vr player (alternative mobile)
    tier6_enabled: bool = True
    tier6_player: str = "android_vr"

    # Tier 7: Standard yt-dlp (always available as fallback)

    # Escalation triggers
    escalate_on_403: bool = True           # 403 Forbidden
    escalate_on_429: bool = True           # 429 Too Many Requests
    escalate_on_bot_check: bool = True     # "Sign in to confirm you're not a bot"
    escalate_on_po_token: bool = True      # "PO Token required" / subtitle access
    escalate_on_geo_block: bool = True     # Geographic restrictions
    escalate_on_timeout: bool = True       # Connection timeouts / TLS hangs

    max_tier: int = 7                      # Maximum tier (7 = standard)

    # Subtitle-specific: always use tv_embedded for captions
    subtitle_always_tv_embedded: bool = True

    # Impersonation target rotation (cycle through on repeated failures)
    impersonate_targets: List[str] = field(default_factory=lambda: [
        "Chrome-131:Android-14",
        "Chrome-131:Windows-10",
        "Chrome-130:macOS-14",
        "Safari-18.2:macOS-15",
        "Safari-18.1:iOS-18",
        "Edge-131:Windows-10",
    ])
    _impersonate_index: int = field(default=0, repr=False)

    # Player client rotation for non-impersonation tiers
    player_clients: List[str] = field(default_factory=lambda: [
        "tv_embedded",   # Best for subtitles, no PO Token
        "web",           # Standard web player
        "mweb",          # Mobile web (may need PO Token for subs)
        "ios_creator",   # iOS creator app
        "android_vr",    # Android VR app
        "mediaconnect",  # Smart TV
    ])
    _player_index: int = field(default=0, repr=False)

    # Current runtime state (not persisted to YAML)
    _current_tier: int = field(default=1, repr=False)

    def get_current_tier(self) -> BypassTier:
        """Get current tier as enum."""
        try:
            return BypassTier(self._current_tier)
        except ValueError:
            return BypassTier.STANDARD

    def escalate(self) -> bool:
        """Escalate to next tier. Returns False if already at max."""
        if self._current_tier >= self.max_tier:
            return False
        self._current_tier += 1
        return True

    def rotate_impersonate_target(self) -> str:
        """Get next impersonation target (rotates through list)."""
        if not self.impersonate_targets:
            return self.tier1_target
        target = self.impersonate_targets[self._impersonate_index]
        self._impersonate_index = (self._impersonate_index + 1) % len(self.impersonate_targets)
        return target

    def rotate_player_client(self) -> str:
        """Get next player client (rotates through list)."""
        if not self.player_clients:
            return "tv_embedded"
        client = self.player_clients[self._player_index]
        self._player_index = (self._player_index + 1) % len(self.player_clients)
        return client

    def get_subtitle_player(self) -> str:
        """Get player client for subtitle fetching (tv_embedded recommended)."""
        if self.subtitle_always_tv_embedded:
            return "tv_embedded"
        return self.rotate_player_client()

    def reset(self):
        """Reset to tier 1 and rotation indices."""
        self._current_tier = 1
        self._impersonate_index = 0
        self._player_index = 0


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

    # Content filter preset - select from content_filter_presets in config.yaml
    # Options: "raw" (default), "documentary", "stock_footage"
    content_filter_preset: str = "raw"

    # Custom prompts that EXTEND the selected preset (not replace)
    # Useful for project-specific filtering requirements
    custom_rejection_prompt: str = ""  # Additional rejection criteria
    custom_acceptance_prompt: str = ""  # Additional acceptance criteria

    # YouTube Data API key for fetching video metadata (title, channel, description)
    # Much faster than yt-dlp for metadata: 50 videos per API call vs 1 per subprocess
    # Set here or via YOUTUBE_API_KEY environment variable
    youtube_api_key: str = ""

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
    search_timeout: int = 60  # Seconds for search metadata subprocess
    download_timeout: int = 120  # Default seconds per video download (used for 'short' tier)

    # Tier-specific download timeouts (longer videos need more time)
    # Keys: 'short', 'medium', 'long', 'longer'
    download_timeouts: Dict[str, int] = field(default_factory=lambda: {
        'short': 120,    # 2 min timeout for videos <2 min
        'medium': 300,   # 5 min timeout for videos 2-10 min
        'long': 600,     # 10 min timeout for videos 10-25 min
        'longer': 900,   # 15 min timeout for videos 25-50 min
    })

    # Per-video timeout for individual downloads (prevents batch hangs)
    # Used when downloading by specific video IDs (LLM filter mode)
    # Shorter than tier timeouts for faster failure on bad videos
    per_video_timeout: int = 60  # 1 min timeout per video

    # First-byte timeout for detecting connection hangs (faster than per_video_timeout)
    # If yt-dlp produces zero output within this time, assume connection hang and escalate tier
    # Critical for detecting --impersonate TLS hangs before full timeout
    first_byte_timeout: int = 30  # 30 sec timeout for first output

    # Validation timeout for checking video accessibility before download
    # Quick check to filter out dead/private/geoblocked videos from cache
    validation_timeout: int = 30  # 30 sec timeout per validation check

    # Audio-first download pipeline (enable per-project for faster downloads)
    audio_first: AudioFirstConfig = field(default_factory=AudioFirstConfig)

    # Caption-first download pipeline (even faster - uses YouTube captions instead of Whisper)
    caption_first: CaptionFirstConfig = field(default_factory=CaptionFirstConfig)

    # Zero-download remix: auto-retry with alternative keywords when 0 results
    zero_download_remix: ZeroDownloadRemixConfig = field(default_factory=ZeroDownloadRemixConfig)

    # Speech screening: pre-screen videos by transcribing first N seconds
    # Rejects videos with speech in intro to ensure only B-roll footage
    speech_screening: SpeechScreeningConfig = field(default_factory=SpeechScreeningConfig)

    # Rate limit bypass configuration
    rate_limit_bypass: RateLimitBypassConfig = field(default_factory=RateLimitBypassConfig)

    # FFmpeg location (for segment downloads, set if not in PATH)
    # Example: "C:/ffmpeg/bin/ffmpeg.exe" or "/usr/local/bin/ffmpeg"
    ffmpeg_location: str = ""

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
        if isinstance(self.rate_limit_bypass, dict):
            self.rate_limit_bypass = RateLimitBypassConfig(**self.rate_limit_bypass)


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
