"""
Multi-format download fallback pipeline (US-114-005).

Handles fallback behavior when primary video formats (mp4, webm) fail to download.
Supports trying alternative formats in priority order, ultimately falling back
to audio-only with automatic transcription.

The fallback chain:
1. Try formats in format_priority order (default: mp4 -> webm)
2. If all video formats fail, fallback to audio-only
3. Audio-only downloads use existing transcription pipeline

Success rates are tracked per format for adaptive priority adjustment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .transcoding import TranscodingManager

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

from .errors import log_error


# Format identifiers for yt-dlp
# US-143-002: Added HDR format (vp9.2) to format fallback chain
# US-144-006: Added avc (H.264), mp3, thumbnail formats
FORMAT_IDENTIFIERS = {
    "mp4": "mp4",
    "webm": "webm",
    "vp9.2": "webm",  # HDR format using VP9 codec
    "avc": "mp4",  # US-144-006: H.264 codec fallback from VP9
    "audio-only": "bestaudio",
    "m4a": "bestaudio[ext=m4a]",
    "opus": "bestaudio[ext=opus]",
    "mp3": "bestaudio[ext=mp3]",  # US-144-006: MP3 fallback from m4a/aac
    "thumbnail": "thumbnail",  # US-144-006: Thumbnail-only fallback for premium content
}

# Audio-only formats that require transcription
# US-144-006: Added mp3 to audio formats
AUDIO_ONLY_FORMATS = {"audio-only", "m4a", "opus", "bestaudio", "mp3"}

# US-143-002: HDR formats that use VP9 codec
HDR_FORMATS = {"vp9.2"}

# US-144-006: Video codec formats (for codec-specific fallback)
VIDEO_CODEC_FORMATS = {"mp4", "webm", "vp9.2", "avc"}

# US-144-006: Thumbnail-only format for premium content
THUMBNAIL_ONLY_FORMAT = "thumbnail"

# US-144-006: Codec fallback mapping (when one codec fails, try the fallback)
CODEC_FALLBACK_MAP = {
    "vp9.2": "avc",  # VP9 HDR fails -> try H.264
    "vp9": "avc",    # VP9 fails -> try H.264
    "webm": "avc",   # WebM container fails -> try H.264 in MP4
    "m4a": "mp3",    # AAC fails -> try MP3
    "aac": "mp3",    # AAC fails -> try MP3
}


@dataclass
class FormatAttempt:
    """Track a single format download attempt."""
    video_id: str
    format: str
    success: bool
    error: Optional[str] = None
    file_path: Optional[str] = None


@dataclass
class FormatStats:
    """Track format success statistics for adaptive priority."""
    attempts: int = 0
    successes: int = 0
    consecutive_failures: int = 0  # US-129-004: Track consecutive failures for reset
    # US-143-002: Track fallback count per format per session
    fallback_count: int = 0
    # US-143-002: Region-based success rates for adaptive priority
    region_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def to_dict(self) -> Dict[str, int]:
        """Serialize to dict for checkpoint persistence."""
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "consecutive_failures": self.consecutive_failures,
            "fallback_count": self.fallback_count,
            "region_stats": self.region_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'FormatStats':
        """Deserialize from dict."""
        return cls(
            attempts=data.get("attempts", 0),
            successes=data.get("successes", 0),
            consecutive_failures=data.get("consecutive_failures", 0),
            fallback_count=data.get("fallback_count", 0),
            region_stats=data.get("region_stats", {}),
        )

    def get_region_success_rate(self, region: str) -> float:
        """Get success rate for a specific region.

        Args:
            region: Region code (e.g., 'us', 'eu', 'asia')

        Returns:
            Success rate for the region (0.0 if no data)
        """
        if region not in self.region_stats:
            return 0.0
        stats = self.region_stats[region]
        attempts = stats.get("attempts", 0)
        if attempts == 0:
            return 0.0
        successes = stats.get("successes", 0)
        return successes / attempts

    def record_region_attempt(self, region: str, success: bool):
        """Record an attempt for a specific region.

        Args:
            region: Region code (e.g., 'us', 'eu', 'asia')
            success: Whether the attempt succeeded
        """
        if region not in self.region_stats:
            self.region_stats[region] = {"attempts": 0, "successes": 0}
        self.region_stats[region]["attempts"] += 1
        if success:
            self.region_stats[region]["successes"] += 1


class FormatFallbackHandler:
    """Handles multi-format download fallback logic.

    Manages format priority, tracks success rates, and provides adaptive
    fallback behavior for the download pipeline.
    """

    def __init__(self, config: 'Config'):
        """Initialize FormatFallbackHandler.

        Args:
            config: Config object with format_fallback settings
        """
        self.config = config
        self.download_config = config.download
        self._format_fallback = self.download_config.format_fallback

        # Track stats per format for adaptive priority
        self._format_stats: Dict[str, FormatStats] = {}
        self._attempt_history: List[FormatAttempt] = []

        # Initialize stats for configured formats
        if self._format_fallback and self._format_fallback.format_priority:
            for fmt in self._format_fallback.format_priority:
                self._format_stats[fmt] = FormatStats()

        # Get transcoding manager for format string building
        self._transcoding_manager = TranscodingManager(config)

    @property
    def is_enabled(self) -> bool:
        """Check if format fallback is enabled."""
        if not self._format_fallback:
            return False
        return self._format_fallback.enabled

    def get_format_priority(self, region: Optional[str] = None) -> List[str]:
        """Get the current format priority order.

        US-143-002: If region is provided and track_region_stats is enabled,
        uses region-specific success rates for adaptive priority.

        Args:
            region: Optional region code for regional adaptation (e.g., 'us', 'eu', 'asia')

        Returns:
            List of formats in priority order (first to try)
        """
        if not self._format_fallback:
            # US-144-006: Added avc and mp3 to default priority
            return ["mp4", "webm", "vp9.2", "avc", "mp3", "audio-only"]

        priority = list(self._format_fallback.format_priority)

        # US-129-004: Apply adaptive priority adjustment if enabled
        # Check new format_adaptive_selection option first, then fall back to legacy
        if self.should_use_adaptive_selection():
            priority = self._adjust_priority_by_success_rate(priority, region)

        return priority

    def _adjust_priority_by_success_rate(self, priority: List[str], region: Optional[str] = None) -> List[str]:
        """Adjust format priority based on historical success rates.

        Only adjusts after reaching min_samples_for_adaptation attempts
        per format to ensure statistical significance.

        US-143-002: If region is provided and track_region_stats is enabled,
        uses region-specific success rates for adaptive priority.

        Args:
            priority: Original format priority order
            region: Optional region code for regional adaptation (e.g., 'us', 'eu', 'asia')

        Returns:
            Adjusted priority with higher-success formats first
        """
        if not self._format_fallback or not self.should_use_adaptive_selection():
            return priority

        min_samples = self._format_fallback.min_samples_for_adaptation
        use_region_stats = (region and self._format_fallback.track_region_stats)

        # Only consider formats with enough samples
        eligible_formats = []
        for fmt in priority:
            stats = self._format_stats.get(fmt)
            if not stats:
                continue

            # US-143-002: Use region-specific success rate if available
            # Also check region-specific attempt count, not just global
            if use_region_stats and region in stats.region_stats:
                region_attempts = stats.region_stats[region].get("attempts", 0)
                if region_attempts >= min_samples:
                    success_rate = stats.get_region_success_rate(region)
                    eligible_formats.append((fmt, success_rate))
            elif stats.attempts >= min_samples:
                eligible_formats.append((fmt, stats.success_rate))

        # If not enough data, return original priority
        if len(eligible_formats) < 2:
            return priority

        # Sort by success rate (highest first), but keep audio-only at end
        audio_only = "audio-only"
        non_audio = [(f, r) for f, r in eligible_formats if f != audio_only]
        non_audio.sort(key=lambda x: x[1], reverse=True)

        # Rebuild priority with audio-only at end
        result = [f for f, _ in non_audio]
        if audio_only in priority:
            result.append(audio_only)

        # Log the adjustment
        if self._format_fallback.track_success_rates:
            rate_str = ", ".join(f"{f}:{r:.1%}" for f, r in non_audio)
            region_info = f" (region: {region})" if region else ""
            logger.info(f"FormatFallback: Adaptive priority adjusted{region_info} - {rate_str}")

        return result

    def build_format_string(self, target_format: str) -> str:
        """Build yt-dlp format string for the target format.

        Args:
            target_format: Target format (mp4, webm, audio-only, etc.)

        Returns:
            yt-dlp format string
        """
        # Handle audio-only formats
        if target_format.lower() in AUDIO_ONLY_FORMATS:
            return "bestaudio"

        # For video formats, use transcoding manager
        if hasattr(self.download_config, 'format_preference') and self.download_config.format_preference:
            # Use existing format preference logic
            original_pref = list(self.download_config.format_preference.preference_order)
            self.download_config.format_preference.preference_order = [target_format]
            format_str = self._transcoding_manager.build_format_string()
            self.download_config.format_preference.preference_order = original_pref
            return format_str

        # Fallback to simple format string
        return f"bestvideo[ext={target_format}]+bestaudio/best[ext={target_format}]"

    def build_composite_format_string(self, quality: str = "1080") -> str:
        """Build composite format string with fallback using yt-dlp's / operator.

        Creates a format string that tries formats in priority order:
        - mp4 first (best compatibility with DaVinci)
        - webm second (VP9/AV1 fallback)
        - avc third (H.264 codec fallback from VP9)
        - audio-only last (transcription fallback)

        Uses yt-dlp's native fallback: format1/format2/format3

        US-144-006: Enhanced to include avc (H.264), mp3 audio, and thumbnail formats.

        Args:
            quality: Video quality/resolution (e.g., "1080", "720", "2160")

        Returns:
            Composite yt-dlp format string with fallback
        """
        priority = self.get_format_priority()
        davinci_mode = self.download_config.davinci_mode

        selectors = []

        for fmt in priority:
            # US-144-006: Handle thumbnail-only format
            if fmt == THUMBNAIL_ONLY_FORMAT:
                selectors.append("thumbnail")
                continue

            if fmt in AUDIO_ONLY_FORMATS:
                # US-144-006: Handle mp3 audio format specifically
                if fmt == "mp3":
                    selectors.append("bestaudio[ext=mp3]")
                elif fmt == "m4a":
                    selectors.append("bestaudio[ext=m4a]")
                else:
                    selectors.append("bestaudio")
                continue

            # Video format (mp4, webm, avc, etc.)
            # US-144-006: Handle avc format specially (it's a codec, not container)
            if fmt == "avc":
                # avc means H.264 codec in MP4 container
                if davinci_mode:  # Prefer H.264 for DaVinci compatibility
                    if quality.isdigit():
                        height = quality.rstrip('p')
                        video_sel = f'bestvideo[ext=mp4][height<={height}][vcodec^=avc1]'
                    else:
                        video_sel = 'bestvideo[ext=mp4][vcodec^=avc1]'
                else:
                    if quality.isdigit():
                        height = quality.rstrip('p')
                        video_sel = f'bestvideo[ext=mp4][height<={height}]'
                    else:
                        video_sel = 'bestvideo[ext=mp4]'
                audio_sel = "bestaudio[ext=m4a]"
                selectors.append(f'{video_sel}+{audio_sel}')
                continue

            if davinci_mode:
                # Prefer h264 for DaVinci compatibility
                if quality.isdigit():
                    height = quality.rstrip('p')
                    video_sel = f'bestvideo[ext={fmt}][height<={height}][vcodec^=avc1]'
                else:
                    video_sel = f'bestvideo[ext={fmt}][vcodec^=avc1]'
            else:
                if quality.isdigit():
                    height = quality.rstrip('p')
                    video_sel = f'bestvideo[ext={fmt}][height<={height}]'
                else:
                    video_sel = f'bestvideo[ext={fmt}]'

            # Audio extension for this container
            audio_ext = self._transcoding_manager._get_audio_extension_for_container(fmt)
            audio_sel = f'bestaudio[ext={audio_ext}]'

            # Add video+audio selector
            selectors.append(f'{video_sel}+{audio_sel}')

            # Also add video-only fallback for this format
            selectors.append(f'bestvideo[ext={fmt}]+bestaudio')

        # Join with / for yt-dlp fallback
        return '/'.join(selectors)

    def is_audio_only_format(self, format_str: str) -> bool:
        """Check if the format string represents audio-only.

        Args:
            format_str: yt-dlp format string

        Returns:
            True if audio-only format
        """
        format_lower = format_str.lower()
        return any(audio_fmt in format_lower for audio_fmt in AUDIO_ONLY_FORMATS)

    def record_attempt(self, video_id: str, format: str, success: bool,
                      error: Optional[str] = None, file_path: Optional[str] = None,
                      region: Optional[str] = None, is_fallback: bool = False):
        """Record a format download attempt for tracking.

        Args:
            video_id: YouTube video ID
            format: Format tried (mp4, webm, audio-only)
            success: Whether download succeeded
            error: Error message if failed
            file_path: Path to downloaded file if successful
            region: Optional region code for regional success tracking (US-143-002)
            is_fallback: Whether this was a fallback attempt (US-143-002)
        """
        attempt = FormatAttempt(
            video_id=video_id,
            format=format,
            success=success,
            error=error,
            file_path=file_path
        )
        self._attempt_history.append(attempt)

        # Update stats
        if format not in self._format_stats:
            self._format_stats[format] = FormatStats()

        stats = self._format_stats[format]
        stats.attempts += 1
        if success:
            stats.successes += 1
            stats.consecutive_failures = 0  # US-129-004: Reset on success
        else:
            stats.consecutive_failures += 1
            # US-129-004: Check if we need to reset stats due to consecutive failures
            self._maybe_reset_format_stats(format, stats)

        # US-143-002: Track fallback count per format
        if is_fallback:
            stats.fallback_count += 1

        # US-143-002: Track region-based success rates
        if region and self._format_fallback and self._format_fallback.track_region_stats:
            stats.record_region_attempt(region, success)

        # Log if tracking enabled
        if self._format_fallback and self._format_fallback.track_success_rates:
            if success:
                logger.debug(f"FormatFallback: {format} succeeded for {video_id}")
            else:
                logger.debug(f"FormatFallback: {format} failed for {video_id}: {error}")

    def _maybe_reset_format_stats(self, format: str, stats: FormatStats):
        """Check if format stats should be reset due to consecutive failures.

        US-129-004: When a format has too many consecutive failures, reset its
        stats so it can be retried (the format may have recovered).

        Args:
            format: Format name
            stats: FormatStats object
        """
        if not self._format_fallback:
            return

        reset_threshold = getattr(self._format_fallback, 'reset_attempts_threshold', 0)
        # Reset only AFTER exceeding threshold (not on the exact threshold)
        if reset_threshold > 0 and stats.consecutive_failures > reset_threshold:
            logger.info(f"FormatFallback: Resetting stats for {format} after {stats.consecutive_failures} consecutive failures")
            stats.attempts = 0
            stats.successes = 0
            stats.consecutive_failures = 0

    def get_format_stats(self) -> Dict[str, Dict[str, float]]:
        """Get format success statistics.

        Returns:
            Dict mapping format to {attempts, successes, success_rate, fallback_count}
        """
        result = {}
        for fmt, stats in self._format_stats.items():
            result[fmt] = {
                "attempts": stats.attempts,
                "successes": stats.successes,
                "success_rate": stats.success_rate,
                "fallback_count": stats.fallback_count,  # US-143-002
            }
        return result

    def get_fallback_counts(self) -> Dict[str, int]:
        """Get fallback counts per format for the current session.

        US-143-002: Returns the number of times each format was used as a fallback.

        Returns:
            Dict mapping format to fallback count
        """
        result = {}
        for fmt, stats in self._format_stats.items():
            result[fmt] = stats.fallback_count
        return result

    def should_fallback(self, error: Optional[str], current_format: str) -> bool:
        """Determine if we should fallback to next format.

        Args:
            error: Error message from failed download
            current_format: Current format being tried

        Returns:
            True if should try next format in priority
        """
        if not self._format_fallback:
            return False

        # If fallback disabled, don't fallback
        if not self._format_fallback.enabled:
            return False

        # If fallback on any error, always try next format
        if self._format_fallback.fallback_on_any_error:
            return error is not None

        # Otherwise, check for specific "unavailable" errors
        if not error:
            return False

        # Common errors that indicate format unavailability
        unavailable_patterns = [
            "unavailable",
            "not available",
            "format not available",
            "no format",
            "premium",
            "members only",
            "private video",
            "deleted",
            "removed",
        ]

        error_lower = error.lower()
        return any(pattern in error_lower for pattern in unavailable_patterns)

    def get_next_format(self, current_format: str) -> Optional[str]:
        """Get the next format to try in the priority chain.

        Args:
            current_format: Current format being tried

        Returns:
            Next format, or None if no more formats
        """
        priority = self.get_format_priority()

        try:
            current_idx = priority.index(current_format)
            if current_idx + 1 < len(priority):
                return priority[current_idx + 1]
        except ValueError:
            # current_format not in priority, return first format
            return priority[0] if priority else None

        return None

    # US-144-006: Codec-specific fallback methods

    def get_codec_fallback_format(self, current_format: str) -> Optional[str]:
        """Get the codec fallback format when current format fails.

        US-144-006: When VP9/H.265 fails, try H.264 (avc). When m4a/aac fails,
        try mp3 as a fallback.

        Args:
            current_format: Current format that failed

        Returns:
            Fallback format or None if no codec fallback available
        """
        return CODEC_FALLBACK_MAP.get(current_format)

    def should_use_codec_fallback(self, error: Optional[str], current_format: str) -> bool:
        """Determine if we should use codec-specific fallback.

        US-144-006: Used when a specific codec fails on certain formats/quality
        but another codec might work.

        Args:
            error: Error message from failed download
            current_format: Current format being tried

        Returns:
            True if codec fallback should be tried
        """
        if not error or not self._format_fallback:
            return False

        # Check if this format has a codec fallback defined
        if current_format not in CODEC_FALLBACK_MAP:
            return False

        # Check for codec-specific error patterns
        codec_error_patterns = [
            "codec",
            "encoder",
            "unsupported",
            "not available",
            "premium",
            "format",
            "unavailable",
        ]

        error_lower = error.lower()
        return any(pattern in error_lower for pattern in codec_error_patterns)

    def get_thumbnail_fallback_format(self) -> str:
        """Get thumbnail-only format for premium/unavailable content.

        US-144-006: When all video/audio formats fail for premium content,
        try to at least get the thumbnail for visual placeholder.

        Returns:
            Thumbnail format identifier
        """
        return THUMBNAIL_ONLY_FORMAT

    def should_use_thumbnail_fallback(self, error: Optional[str], formats_tried: List[str]) -> bool:
        """Determine if thumbnail-only fallback should be used.

        US-144-006: Used when content is premium/members-only and no video
        or audio is available. Falls back to thumbnail as last resort.

        Args:
            error: Error message from failed download
            formats_tried: List of formats already tried

        Returns:
            True if thumbnail fallback should be tried
        """
        if not error or not self._format_fallback:
            return False

        # Check if thumbnail fallback is enabled
        if not getattr(self._format_fallback, 'enable_thumbnail_fallback', False):
            return False

        # Check for premium/members-only content patterns
        premium_patterns = [
            "premium",
            "members only",
            "members-only",
            "not available",
            "unavailable",
            "region",
            "blocked",
        ]

        error_lower = error.lower()
        is_premium_error = any(pattern in error_lower for pattern in premium_patterns)

        # Only use thumbnail fallback if premium error and we've tried video/audio
        return is_premium_error and len(formats_tried) >= 2

    def is_thumbnail_only_format(self, format_str: str) -> bool:
        """Check if format is thumbnail-only.

        Args:
            format_str: Format string to check

        Returns:
            True if thumbnail-only format
        """
        return format_str.lower() == THUMBNAIL_ONLY_FORMAT

    def log_success_rates(self):
        """Log current format success rates (for debugging/monitoring)."""
        if not self._format_fallback or not self._format_fallback.track_success_rates:
            return

        logger.info("Format success rates:")
        for fmt in self.get_format_priority():
            stats = self._format_stats.get(fmt)
            if stats and stats.attempts > 0:
                logger.info(f"  {fmt}: {stats.successes}/{stats.attempts} ({stats.success_rate:.1%})")

    # US-129-004: Checkpoint persistence for format success history

    def get_format_stats_for_checkpoint(self) -> Dict[str, Dict[str, int]]:
        """Get format stats for checkpoint persistence.

        Returns:
            Dict mapping format name to serialized FormatStats dict
        """
        result = {}
        for fmt, stats in self._format_stats.items():
            result[fmt] = stats.to_dict()
        return result

    def load_format_stats_from_checkpoint(self, data: Dict[str, Dict[str, int]]):
        """Load format stats from checkpoint data.

        Args:
            data: Dict mapping format name to serialized FormatStats dict
        """
        if not data:
            return

        for fmt, stats_data in data.items():
            self._format_stats[fmt] = FormatStats.from_dict(stats_data)
        logger.info(f"FormatFallback: Loaded format stats from checkpoint: {list(data.keys())}")

    def should_use_adaptive_selection(self) -> bool:
        """Check if adaptive format selection should be used.

        Returns True if format_adaptive_selection is enabled (or legacy adaptive_priority).

        Returns:
            True if adaptive selection is enabled
        """
        if not self._format_fallback:
            return False

        # Check new format_adaptive_selection option first
        if hasattr(self._format_fallback, 'format_adaptive_selection'):
            value = self._format_fallback.format_adaptive_selection
            # Only use if it's a proper boolean (not a MagicMock or other object)
            if isinstance(value, bool):
                return value

        # Fall back to legacy adaptive_priority option
        return getattr(self._format_fallback, 'adaptive_priority', False)


class FormatFallbackPipeline:
    """Orchestrates the full format fallback download process.

    Coordinates between format fallback handler and the actual download
    to implement the retry-with-fallback logic.
    """

    def __init__(self, config: 'Config', download_func):
        """Initialize FormatFallbackPipeline.

        Args:
            config: Config object
            download_func: Function to call for actual download (video_id, format) -> file_path
        """
        self.config = config
        self.download_config = config.download
        self._handler = FormatFallbackHandler(config)
        self._download_func = download_func

    @property
    def is_enabled(self) -> bool:
        """Check if format fallback is enabled."""
        return self._handler.is_enabled

    def download_with_fallback(self, video_id: str,
                               max_total_retries: int = 3) -> Tuple[Optional[str], Optional[str], bool]:
        """Download video with format fallback on failure.

        Args:
            video_id: YouTube video ID
            max_total_retries: Maximum total retry attempts across all formats

        Returns:
            Tuple of (file_path, format_used, is_audio_only)
            - file_path: Path to downloaded file, or None if all failed
            - format_used: Format that succeeded (mp4/webm/audio-only), or None
            - is_audio_only: True if audio-only format was used (requires transcription)
        """
        if not self._handler.is_enabled:
            # Fallback disabled, use default behavior
            file_path = self._download_func(video_id, None)
            return file_path, "default", file_path is not None

        priority = self._handler.get_format_priority()
        total_attempts = 0
        last_error = None

        for format_idx, format_name in enumerate(priority):
            # Get retry config for this format
            max_retries = self._handler._format_fallback.max_retries_per_format if self._handler._format_fallback else 1
            max_retries = min(max_retries, max_total_retries - total_attempts)

            for retry in range(max_retries + 1):
                total_attempts += 1
                last_error = None

                # Build format string
                format_str = self._handler.build_format_string(format_name)
                is_audio = self._handler.is_audio_only_format(format_str)

                logger.info(f"FormatFallback: Trying {format_name} (attempt {retry + 1}/{max_retries + 1}) for {video_id}")

                try:
                    file_path = self._download_func(video_id, format_str)

                    if file_path:
                        # Success!
                        self._handler.record_attempt(video_id, format_name, success=True, file_path=file_path)
                        logger.info(f"FormatFallback: {format_name} succeeded for {video_id}")
                        return file_path, format_name, is_audio
                    else:
                        # Download returned None (no file)
                        last_error = "Download returned no file"
                        self._handler.record_attempt(video_id, format_name, success=False, error=last_error)

                except Exception as e:
                    last_error = str(e)
                    self._handler.record_attempt(video_id, format_name, success=False, error=last_error)
                    logger.warning(f"FormatFallback: {format_name} failed for {video_id}: {last_error}")

                # Check if we should fallback to next format
                if total_attempts >= max_total_retries:
                    break

                if retry < max_retries and self._handler.should_fallback(last_error, format_name):
                    # Continue to next retry of same format
                    continue
                elif format_idx < len(priority) - 1:
                    # Move to next format in priority
                    break

        # All formats failed
        log_error(logger, "FormatFallback", f"All formats failed for {video_id} after {total_attempts} attempts", error_code="E301")
        return None, None, False

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get format success statistics."""
        return self._handler.get_format_stats()
