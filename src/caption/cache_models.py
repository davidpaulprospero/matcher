"""
Cache-related models for caption storage and validation.

Contains dataclasses for cached caption data, validation results,
and channel-level pattern tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .models import CaptionSegment


@dataclass
class CachedCaption:
    """Cached caption data for cross-project reuse.

    Stores the full caption result along with fetch metadata.
    Also supports caching "unavailable" status to avoid re-checking videos
    that are known to have no captions.
    """
    video_id: str
    language: str
    segments: List[Dict[str, Any]]  # CaptionSegment.to_dict() format
    is_auto_generated: bool
    format_source: str
    fetch_timestamp: float  # Unix timestamp when fetched
    duration: float = 0.0  # Total caption duration
    caption_quality: str = "medium"  # 'high', 'medium', or 'low' (US-007)
    coverage_ratio: Optional[float] = None  # Caption coverage vs video duration (US-004)
    unavailable: bool = False  # True if captions are known to be unavailable (US-008 enhancement)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'video_id': self.video_id,
            'language': self.language,
            'segments': self.segments,
            'is_auto_generated': self.is_auto_generated,
            'format_source': self.format_source,
            'fetch_timestamp': self.fetch_timestamp,
            'duration': self.duration,
            'caption_quality': self.caption_quality,
            'coverage_ratio': self.coverage_ratio,
            'unavailable': self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CachedCaption':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_caption_result(self) -> 'CaptionResult':
        """Convert cached data back to CaptionResult.

        Note: Imports CaptionResult and CaptionSegment at call time to avoid
        circular imports.
        """
        from .models import CaptionSegment
        # Import CaptionResult from fetcher once it exists
        # For now, return a dict-like structure
        segments = [
            CaptionSegment(
                index=seg.get('index', i),
                start_time=seg.get('start', seg.get('start_time', 0.0)),
                end_time=seg.get('end', seg.get('end_time', 0.0)),
                text=seg.get('text', ''),
                source_file=seg.get('source_file', self.video_id),
            )
            for i, seg in enumerate(self.segments)
        ]
        # Import from models (extracted in refactoring)
        from .models import CaptionResult
        return CaptionResult(
            video_id=self.video_id,
            segments=segments,
            language=self.language,
            is_auto_generated=self.is_auto_generated,
            format_source=self.format_source,
        )

    @classmethod
    def from_caption_result(cls, result: 'CaptionResult', language: str) -> 'CachedCaption':
        """Create CachedCaption from a CaptionResult.

        Args:
            result: CaptionResult to cache.
            language: Language code for this caption.

        Returns:
            CachedCaption instance.
        """
        segments = [seg.to_dict() for seg in result.segments]
        duration = result.segments[-1].end_time if result.segments else 0.0

        return cls(
            video_id=result.video_id,
            language=language,
            segments=segments,
            is_auto_generated=result.is_auto_generated,
            format_source=result.format_source,
            fetch_timestamp=time.time(),
            duration=duration,
            caption_quality=result.caption_quality,
        )


@dataclass
class CacheValidationResult:
    """Result of cache entry validation (US-008 Sprint 6).

    Returned by CaptionCache.validate_cache_entry() to indicate whether
    a cached caption entry is valid and usable, or needs to be rejected/refetched.

    Attributes:
        is_valid: True if entry passes validation, False otherwise.
        video_id: Video ID that was validated.
        language: Language code that was validated.
        reason: Human-readable reason if validation failed.
        expected_segment_count: Expected segment count based on duration.
        actual_segment_count: Actual segment count in cached data.
        segment_count_deviation: Absolute deviation as ratio (0.0-1.0).
    """
    is_valid: bool
    video_id: str
    language: str
    reason: str = ""
    expected_segment_count: Optional[int] = None
    actual_segment_count: Optional[int] = None
    segment_count_deviation: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_valid': self.is_valid,
            'video_id': self.video_id,
            'language': self.language,
            'reason': self.reason,
            'expected_segment_count': self.expected_segment_count,
            'actual_segment_count': self.actual_segment_count,
            'segment_count_deviation': self.segment_count_deviation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheValidationResult':
        """Create from dictionary."""
        return cls(
            is_valid=data.get('is_valid', False),
            video_id=data.get('video_id', ''),
            language=data.get('language', ''),
            reason=data.get('reason', ''),
            expected_segment_count=data.get('expected_segment_count'),
            actual_segment_count=data.get('actual_segment_count'),
            segment_count_deviation=data.get('segment_count_deviation'),
        )


@dataclass
class ChannelCaptionPattern:
    """Channel-level caption availability pattern (US-006 Sprint 7).

    Tracks whether a YouTube channel typically has captions available.
    Used for batch pre-check optimization: if a channel has >90% confidence,
    skip individual pre-checks for remaining videos from that channel.

    Attributes:
        channel_id: YouTube channel ID (UCxxxx format).
        videos_checked: Number of videos checked for this channel.
        captions_found: Number of videos that had captions available.
        success_rate: Ratio of captions_found / videos_checked (0.0-1.0).
        last_updated: Unix timestamp when pattern was last updated.

    Example:
        >>> pattern = ChannelCaptionPattern(
        ...     channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
        ...     videos_checked=10,
        ...     captions_found=9,
        ...     success_rate=0.9,
        ...     last_updated=time.time()
        ... )
        >>> if pattern.success_rate > 0.9 and pattern.videos_checked >= 5:
        ...     print("High confidence - skip pre-checks")
    """
    channel_id: str
    videos_checked: int = 0
    captions_found: int = 0
    success_rate: float = 0.0
    last_updated: float = 0.0

    def update(self, has_captions: bool) -> None:
        """Update pattern with a new video check result.

        Args:
            has_captions: Whether the video had captions available.
        """
        self.videos_checked += 1
        if has_captions:
            self.captions_found += 1
        self.success_rate = self.captions_found / self.videos_checked if self.videos_checked > 0 else 0.0
        self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'channel_id': self.channel_id,
            'videos_checked': self.videos_checked,
            'captions_found': self.captions_found,
            'success_rate': self.success_rate,
            'last_updated': self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelCaptionPattern':
        """Create from dictionary."""
        return cls(
            channel_id=data.get('channel_id', ''),
            videos_checked=data.get('videos_checked', 0),
            captions_found=data.get('captions_found', 0),
            success_rate=data.get('success_rate', 0.0),
            last_updated=data.get('last_updated', 0.0),
        )


@dataclass
class CachedLanguageList:
    """Cached list-subs output for available languages (US-59-012).

    Caches the result of list_available_languages() to avoid redundant
    yt-dlp --list-subs subprocess calls on resume runs.

    Attributes:
        video_id: YouTube video ID (11 characters).
        languages: List of available language dicts with 'code', 'name', 'is_auto_generated'.
        cached_at: Unix timestamp when cached.
        ttl_hours: TTL in hours (default 1, since availability can change).
    """
    video_id: str
    languages: List[Dict[str, Any]]  # AvailableLanguage fields as dicts
    cached_at: float
    ttl_hours: float = 1.0  # 1 hour default TTL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'video_id': self.video_id,
            'languages': self.languages,
            'cached_at': self.cached_at,
            'ttl_hours': self.ttl_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedLanguageList':
        """Create from dictionary."""
        return cls(
            video_id=data.get('video_id', ''),
            languages=data.get('languages', []),
            cached_at=data.get('cached_at', 0.0),
            ttl_hours=data.get('ttl_hours', 1.0),
        )

    def is_expired(self) -> bool:
        """Check if cache entry is expired based on TTL."""
        age_seconds = time.time() - self.cached_at
        ttl_seconds = self.ttl_hours * 3600
        return age_seconds > ttl_seconds


@dataclass
class BatchPreCheckResult:
    """Result of batch pre-check by channel (US-006 Sprint 7).

    Contains the results of checking caption availability for a batch of videos
    grouped by channel, including metrics on API calls saved.

    Attributes:
        video_results: Dict mapping video_id -> has_captions (bool).
        channel_patterns: Updated channel patterns after checks.
        total_videos: Total number of videos processed.
        actual_checks: Number of actual API calls made.
        skipped_by_pattern: Number of videos skipped due to high-confidence pattern.
        api_calls_saved: Estimated API calls saved (total_videos - actual_checks).
        fetched_channel_info: Dict mapping video_id -> channel_id from fetched metadata.
    """
    video_results: Dict[str, bool] = field(default_factory=dict)
    channel_patterns: Dict[str, ChannelCaptionPattern] = field(default_factory=dict)
    total_videos: int = 0
    actual_checks: int = 0
    skipped_by_pattern: int = 0
    api_calls_saved: int = 0
    fetched_channel_info: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'video_results': self.video_results,
            'channel_patterns': {
                cid: pattern.to_dict()
                for cid, pattern in self.channel_patterns.items()
            },
            'total_videos': self.total_videos,
            'actual_checks': self.actual_checks,
            'skipped_by_pattern': self.skipped_by_pattern,
            'api_calls_saved': self.api_calls_saved,
            'fetched_channel_info': self.fetched_channel_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchPreCheckResult':
        """Create from dictionary."""
        channel_patterns = {}
        for cid, pattern_data in data.get('channel_patterns', {}).items():
            channel_patterns[cid] = ChannelCaptionPattern.from_dict(pattern_data)

        return cls(
            video_results=data.get('video_results', {}),
            channel_patterns=channel_patterns,
            total_videos=data.get('total_videos', 0),
            actual_checks=data.get('actual_checks', 0),
            skipped_by_pattern=data.get('skipped_by_pattern', 0),
            api_calls_saved=data.get('api_calls_saved', 0),
            fetched_channel_info=data.get('fetched_channel_info', {}),
        )
