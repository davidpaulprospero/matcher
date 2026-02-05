"""
Caption Healer - Intelligent caption format discovery and fallback.

Handles:
- Caption format unavailable errors (e.g., json3 not available)
- No captions available (fallback to auto-generated)
- yt-dlp subtitle extraction errors
- Format discovery using --list-subs
- Negative caching to prevent repeated failed lookups (US-64-003)

This healer addresses the ~40 second delay caused by format fallback loops
by proactively discovering available formats before attempting download.

US-64-003: Integrates with CaptionCache for persistent negative caching:
- Checks negative cache before attempting format discovery
- Stores "no captions" results with configurable TTL (default 1 hour)
- Propagates structured NoCaptionsStatus for downstream stage handling
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from ..base import Healer, HealerResult, HealerAction

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState
    from ...caption.cache import CaptionCache

logger = logging.getLogger(__name__)


@dataclass
class SubtitleInfo:
    """Information about an available subtitle track."""
    language: str
    format: str
    is_auto_generated: bool
    name: Optional[str] = None


@dataclass
class NoCaptionsStatus:
    """Structured status for videos with no captions available (US-64-003).

    This dataclass provides a structured way to communicate "no captions"
    status to downstream stages, allowing them to handle it gracefully
    (e.g., skip transcription fallback, use placeholder, etc.).

    Attributes:
        video_id: YouTube video ID
        language: Language that was checked
        reason: Why captions are unavailable
        cached: Whether this result came from negative cache
        cached_at: Timestamp when result was cached (if cached)
        ttl_seconds: TTL for this negative cache entry
        auto_subs_attempted: Whether auto-subs were also checked
        discovered_formats: List of formats that were found (empty for no captions)
    """
    video_id: str
    language: str = "en"
    reason: str = "no_captions_available"
    cached: bool = False
    cached_at: Optional[float] = None
    ttl_seconds: int = 3600  # Default 1 hour
    auto_subs_attempted: bool = False
    discovered_formats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'video_id': self.video_id,
            'language': self.language,
            'reason': self.reason,
            'cached': self.cached,
            'cached_at': self.cached_at,
            'ttl_seconds': self.ttl_seconds,
            'auto_subs_attempted': self.auto_subs_attempted,
            'discovered_formats': self.discovered_formats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoCaptionsStatus':
        """Create from dictionary."""
        return cls(
            video_id=data['video_id'],
            language=data.get('language', 'en'),
            reason=data.get('reason', 'no_captions_available'),
            cached=data.get('cached', False),
            cached_at=data.get('cached_at'),
            ttl_seconds=data.get('ttl_seconds', 3600),
            auto_subs_attempted=data.get('auto_subs_attempted', False),
            discovered_formats=data.get('discovered_formats', []),
        )

    @property
    def is_expired(self) -> bool:
        """Check if the cached status has expired."""
        if not self.cached or self.cached_at is None:
            return False
        return time.time() - self.cached_at > self.ttl_seconds


class CaptionHealer(Healer):
    """
    Heals caption-related errors.

    Recovery strategies:
    1. Format discovery: Use --list-subs to find available formats before retry
    2. Auto-caption fallback: Enable --write-auto-sub when manual captions unavailable
    3. Language fallback: Try alternative languages (en, en-US, a]auto] variants)
    4. Format negotiation: Suggest working format based on available options

    This healer reduces wasted time by proactively discovering what formats
    are available rather than trying each format sequentially.
    """

    name = "caption-healer"
    description = "Fix caption/subtitle fetch errors"

    error_patterns = [
        "no captions",
        "caption",
        "subtitle",
        "no subtitles",
        "format unavailable",
        "json3",
        "srv1",
        "srv2",
        "srv3",
        "vtt",
        "ttml",
        "there are no subtitles",
        "requested format is not available",
        "unable to download video subtitles",
        "--write-auto-sub",
    ]

    # Preferred format order (json3 is best for parsing, then srv variants, then vtt)
    FORMAT_PREFERENCE = ["json3", "srv3", "srv2", "srv1", "vtt", "ttml"]

    def __init__(self, config: 'Config', project_dir, caption_cache: Optional['CaptionCache'] = None):
        super().__init__(config, project_dir)
        # Cache discovered formats to avoid repeated --list-subs calls
        self._format_cache: Dict[str, List[SubtitleInfo]] = {}
        # Track videos with no captions at all (in-memory, per-session)
        self._no_captions_videos: Set[str] = set()
        # Track auto-caption enabled state
        self._auto_caption_enabled: bool = False
        # Persistent negative cache (US-64-003)
        self._caption_cache = caption_cache
        # Get TTL from config (default 1 hour = 3600 seconds)
        self._negative_cache_ttl_seconds = self._get_negative_cache_ttl()

    def _get_negative_cache_ttl(self) -> int:
        """Get negative cache TTL from config (US-64-003).

        Returns TTL in seconds. Checks negative_cache_ttl_seconds first,
        falls back to negative_cache_ttl_hours converted to seconds.
        """
        download_config = getattr(self.config, 'download', None)
        if download_config:
            caption_config = getattr(download_config, 'caption_first', None)
            if caption_config:
                # Prefer seconds (US-63-005)
                ttl_seconds = getattr(caption_config, 'negative_cache_ttl_seconds', None)
                if ttl_seconds and ttl_seconds > 0:
                    return int(ttl_seconds)
                # Fallback to hours
                ttl_hours = getattr(caption_config, 'negative_cache_ttl_hours', 1.0)
                return int(ttl_hours * 3600)
        return 3600  # Default 1 hour

    def set_caption_cache(self, caption_cache: 'CaptionCache') -> None:
        """Set the caption cache for persistent negative caching (US-64-003).

        This allows the healer to be configured with a cache after construction,
        useful when the cache is created later in the pipeline initialization.
        """
        self._caption_cache = caption_cache

    def can_handle(self, error: Exception, stage_name: str) -> bool:
        """
        Check if this healer can handle caption-related errors.

        Handles:
        - CaptionFormatUnavailableError: Specific format not available
        - CaptionUnavailableError: No captions at all
        - CaptionFetchError: General fetch failures that might be format-related
        - Generic errors with caption/subtitle keywords
        """
        # Check by exception type
        error_type = type(error).__name__
        if error_type in (
            "CaptionFormatUnavailableError",
            "CaptionUnavailableError",
            "CaptionFetchError",
        ):
            return True

        # Check error message patterns
        error_str = str(error).lower()
        for pattern in self.error_patterns:
            if pattern.lower() in error_str:
                return True

        return False

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """
        Attempt to fix caption-related errors.

        Strategy (US-64-003 enhanced):
        1. Extract video ID from error
        2. Check negative cache - if cached as unavailable, skip immediately
        3. Run --list-subs to discover available formats
        4. If formats available: suggest best format and retry
        5. If no manual captions: enable auto-caption and retry
        6. If no captions at all: store in negative cache and skip video
        """
        error_str = str(error)
        video_id = self._extract_video_id(error_str, error)

        if not video_id:
            self.log_attempt("Could not extract video ID from error")
            return HealerResult.failed(
                "Could not determine video ID for format discovery"
            )

        # US-64-003: Check persistent negative cache FIRST
        # This prevents the ~40 second failure cycle for known captionless videos
        negative_status = self._check_negative_cache(video_id)
        if negative_status:
            self.log_attempt(
                f"Negative cache hit: {video_id} has no captions "
                f"(cached {self._format_cache_age(negative_status.cached_at)} ago)"
            )
            return HealerResult.fixed(
                f"Video {video_id} cached as having no captions, skipping",
                action=HealerAction.SKIP,
                video_id=video_id,
                no_captions_status=negative_status.to_dict(),
                cached=True
            )

        # Check in-memory cache (for current session)
        if video_id in self._no_captions_videos:
            return HealerResult.fixed(
                f"Video {video_id} known to have no captions, skipping",
                action=HealerAction.SKIP,
                video_id=video_id
            )

        # Discover available formats
        self.log_attempt(f"Discovering subtitle formats for {video_id}...")
        available = self._discover_formats(video_id)

        if not available:
            # No subtitles at all - try enabling auto-captions first
            if not self._auto_caption_enabled:
                self.log_attempt("No subtitles found, enabling auto-caption fallback...")
                self._auto_caption_enabled = True
                return self._enable_auto_captions()

            # Auto-captions already enabled but still no luck
            self._no_captions_videos.add(video_id)

            # US-64-003: Store in persistent negative cache
            no_captions_status = self._store_negative_cache(video_id, auto_subs_attempted=True)

            self.log_attempt(f"No captions available for {video_id} even with auto-sub")
            return HealerResult.fixed(
                f"No captions available for {video_id}, skipping",
                action=HealerAction.SKIP,
                video_id=video_id,
                no_captions=True,
                no_captions_status=no_captions_status.to_dict() if no_captions_status else None
            )

        # Check for auto-generated captions
        auto_subs = [s for s in available if s.is_auto_generated]
        manual_subs = [s for s in available if not s.is_auto_generated]

        # Determine best available format
        best_format = self._select_best_format(available)

        if manual_subs:
            self.log_success(
                f"Found {len(manual_subs)} manual subtitle(s), "
                f"best format: {best_format.format}"
            )
        elif auto_subs:
            self.log_success(
                f"Found {len(auto_subs)} auto-generated subtitle(s), "
                f"best format: {best_format.format}"
            )
            # Ensure auto-caption mode is enabled
            if not self._auto_caption_enabled:
                self._auto_caption_enabled = True
                self._update_config_for_auto_subs()

        return HealerResult.config_changed(
            f"Discovered format '{best_format.format}' for {video_id}",
            video_id=video_id,
            discovered_format=best_format.format,
            language=best_format.language,
            is_auto_generated=best_format.is_auto_generated,
            available_formats=[s.format for s in available],
            manual_count=len(manual_subs),
            auto_count=len(auto_subs),
        )

    def _extract_video_id(self, error_str: str, error: Exception) -> Optional[str]:
        """Extract video ID from error message or exception attributes."""
        # Check exception attributes first
        if hasattr(error, 'video_id'):
            return error.video_id

        # Try common patterns in error message
        patterns = [
            r'video[_\s]?(?:id)?[:\s]+([a-zA-Z0-9_-]{11})',
            r'(?:v=|/)([a-zA-Z0-9_-]{11})(?:[&?/]|$)',
            r"'([a-zA-Z0-9_-]{11})'",
            r'"([a-zA-Z0-9_-]{11})"',
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _discover_formats(self, video_id: str) -> List[SubtitleInfo]:
        """
        Discover available subtitle formats using yt-dlp --list-subs.

        Args:
            video_id: YouTube video ID

        Returns:
            List of SubtitleInfo for available subtitles
        """
        # Check cache first
        if video_id in self._format_cache:
            return self._format_cache[video_id]

        try:
            cmd = [
                "yt-dlp",
                "--list-subs",
                "--skip-download",
                f"https://www.youtube.com/watch?v={video_id}",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
            )

            available = self._parse_list_subs_output(result.stdout, result.stderr)
            self._format_cache[video_id] = available
            return available

        except subprocess.TimeoutExpired:
            self.log_attempt(f"Timeout discovering formats for {video_id}")
            return []
        except Exception as e:
            self.log_attempt(f"Error discovering formats: {e}")
            return []

    def _parse_list_subs_output(
        self,
        stdout: str,
        stderr: str
    ) -> List[SubtitleInfo]:
        """
        Parse yt-dlp --list-subs output to extract available formats.

        Output format example:
        Available subtitles for VIDEO_ID:
        Language  formats
        en        json3, srv1, srv2, srv3, ttml, vtt

        Available automatic captions for VIDEO_ID:
        Language  formats
        af        json3, srv1, srv2, srv3, ttml, vtt
        en        json3, srv1, srv2, srv3, ttml, vtt
        ...
        """
        combined = stdout + "\n" + stderr
        available: List[SubtitleInfo] = []
        is_auto_section = False

        for line in combined.split('\n'):
            line = line.strip()

            # Detect section headers
            if 'automatic captions' in line.lower():
                is_auto_section = True
                continue
            elif 'available subtitles' in line.lower():
                is_auto_section = False
                continue
            elif line.startswith('Language') or not line:
                continue

            # Parse language/format lines: "en   json3, srv1, srv2, srv3, ttml, vtt"
            # or "en-US  json3, srv1, srv2, srv3, ttml, vtt  English (United States)"
            parts = line.split(None, 1)  # Split on first whitespace
            if len(parts) >= 2:
                language = parts[0]
                # Skip header-like entries
                if language.lower() in ('language', 'formats'):
                    continue

                rest = parts[1]
                # Extract formats (comma-separated list before any description)
                format_match = re.match(r'^([\w, ]+)', rest)
                if format_match:
                    formats_str = format_match.group(1)
                    formats = [f.strip() for f in formats_str.split(',') if f.strip()]

                    # Extract optional name (after formats)
                    name = None
                    name_match = re.search(r'\s+(\w[\w\s()]+)$', rest)
                    if name_match:
                        name = name_match.group(1).strip()

                    for fmt in formats:
                        # Skip if not a known format
                        if fmt not in self.FORMAT_PREFERENCE + ['json']:
                            continue
                        available.append(SubtitleInfo(
                            language=language,
                            format=fmt,
                            is_auto_generated=is_auto_section,
                            name=name,
                        ))

        return available

    def _select_best_format(self, available: List[SubtitleInfo]) -> SubtitleInfo:
        """
        Select the best subtitle format based on preference order.

        Prefers:
        1. Manual captions over auto-generated
        2. English variants (en, en-US, en-GB) over other languages
        3. json3 > srv3 > srv2 > srv1 > vtt > ttml
        """
        if not available:
            raise ValueError("No available formats to select from")

        # Separate by type
        manual = [s for s in available if not s.is_auto_generated]
        auto = [s for s in available if s.is_auto_generated]

        # Prefer manual if available
        candidates = manual if manual else auto

        # Prefer English variants
        english_candidates = [
            s for s in candidates
            if s.language.lower().startswith('en')
        ]
        if english_candidates:
            candidates = english_candidates

        # Sort by format preference
        def format_rank(sub: SubtitleInfo) -> int:
            try:
                return self.FORMAT_PREFERENCE.index(sub.format)
            except ValueError:
                return len(self.FORMAT_PREFERENCE)  # Unknown formats last

        candidates.sort(key=format_rank)
        return candidates[0]

    def _enable_auto_captions(self) -> HealerResult:
        """Enable auto-caption fallback in config."""
        download_config = getattr(self.config, 'download', None)
        if download_config:
            caption_config = getattr(download_config, 'caption_first', None)
            if caption_config:
                # Enable auto-sub fallback
                if hasattr(caption_config, 'enable_auto_subs'):
                    caption_config.enable_auto_subs = True
                elif isinstance(caption_config, dict):
                    caption_config['enable_auto_subs'] = True

        self.log_success("Enabled auto-caption fallback (--write-auto-sub)")
        return HealerResult.config_changed(
            "Enabled auto-caption fallback for videos without manual captions",
            auto_subs_enabled=True
        )

    def _update_config_for_auto_subs(self):
        """Update config to prefer auto-subs when no manual available."""
        download_config = getattr(self.config, 'download', None)
        if download_config:
            caption_config = getattr(download_config, 'caption_first', None)
            if caption_config:
                if hasattr(caption_config, 'prefer_auto_subs_if_no_manual'):
                    caption_config.prefer_auto_subs_if_no_manual = True
                elif isinstance(caption_config, dict):
                    caption_config['prefer_auto_subs_if_no_manual'] = True

    def get_discovered_formats(self, video_id: str) -> List[SubtitleInfo]:
        """Get cached format discovery results for a video."""
        return self._format_cache.get(video_id, [])

    def get_no_caption_videos(self) -> Set[str]:
        """Get set of video IDs known to have no captions."""
        return self._no_captions_videos.copy()

    def reset_discovery_cache(self):
        """Clear format discovery cache."""
        self._format_cache.clear()
        self._no_captions_videos.clear()
        self._auto_caption_enabled = False

    # US-64-003: Negative cache integration methods

    def _check_negative_cache(self, video_id: str, language: str = "en") -> Optional[NoCaptionsStatus]:
        """Check if video is known to have no captions (US-64-003).

        Checks the persistent negative cache to avoid repeated failed lookups.
        Returns NoCaptionsStatus if cached and not expired, None otherwise.
        """
        if not self._caption_cache:
            return None

        try:
            # Check if cached as unavailable
            if self._caption_cache.is_caption_unavailable(video_id, language):
                # Get cached entry details for the status
                key = self._caption_cache._make_cache_key(video_id, language)
                entry = self._caption_cache.get(key)
                cached_at = entry.cached_at if entry else time.time()

                return NoCaptionsStatus(
                    video_id=video_id,
                    language=language,
                    reason="negative_cache_hit",
                    cached=True,
                    cached_at=cached_at,
                    ttl_seconds=self._negative_cache_ttl_seconds,
                    auto_subs_attempted=True,  # Assume auto-subs were tried
                )
        except Exception as e:
            logger.debug(f"Error checking negative cache for {video_id}: {e}")

        return None

    def _store_negative_cache(
        self,
        video_id: str,
        language: str = "en",
        auto_subs_attempted: bool = False
    ) -> Optional[NoCaptionsStatus]:
        """Store video as having no captions in persistent cache (US-64-003).

        Returns NoCaptionsStatus describing the cached result.
        """
        status = NoCaptionsStatus(
            video_id=video_id,
            language=language,
            reason="no_captions_available",
            cached=True,
            cached_at=time.time(),
            ttl_seconds=self._negative_cache_ttl_seconds,
            auto_subs_attempted=auto_subs_attempted,
        )

        if self._caption_cache:
            try:
                self._caption_cache.store_unavailable(video_id, language)
                logger.info(
                    f"Stored negative cache for {video_id}: no captions available "
                    f"(TTL: {self._negative_cache_ttl_seconds}s)"
                )
            except Exception as e:
                logger.warning(f"Failed to store negative cache for {video_id}: {e}")
                status.cached = False
                status.cached_at = None

        return status

    def _format_cache_age(self, cached_at: Optional[float]) -> str:
        """Format cache age for logging."""
        if cached_at is None:
            return "unknown"
        age_seconds = time.time() - cached_at
        if age_seconds < 60:
            return f"{int(age_seconds)}s"
        elif age_seconds < 3600:
            return f"{int(age_seconds / 60)}m"
        else:
            return f"{age_seconds / 3600:.1f}h"

    def get_no_captions_status(self, video_id: str, language: str = "en") -> Optional[NoCaptionsStatus]:
        """Get structured no-captions status for a video (US-64-003).

        Returns NoCaptionsStatus if the video is known to have no captions,
        either from persistent cache or in-memory tracking. Returns None
        if caption availability is unknown.

        This method is intended for downstream stages to check caption
        availability before attempting operations that require captions.
        """
        # Check persistent cache first
        status = self._check_negative_cache(video_id, language)
        if status:
            return status

        # Check in-memory tracking
        if video_id in self._no_captions_videos:
            return NoCaptionsStatus(
                video_id=video_id,
                language=language,
                reason="session_cache_hit",
                cached=False,
                auto_subs_attempted=self._auto_caption_enabled,
            )

        return None
