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

    # Audio-first download pipeline (enable per-project for faster downloads)
    audio_first: AudioFirstConfig = field(default_factory=AudioFirstConfig)

    # Zero-download remix: auto-retry with alternative keywords when 0 results
    zero_download_remix: ZeroDownloadRemixConfig = field(default_factory=ZeroDownloadRemixConfig)

    # Speech screening: pre-screen videos by transcribing first N seconds
    # Rejects videos with speech in intro to ensure only B-roll footage
    speech_screening: SpeechScreeningConfig = field(default_factory=SpeechScreeningConfig)

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
