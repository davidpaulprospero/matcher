"""
Caption cache for YouTube captions, enabling cross-project reuse.

Caches fetched captions by video_id + language code to avoid
re-fetching the same captions across different projects.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..cache import BaseCache, CacheEntry

from .cache_models import (
    CachedCaption,
    CacheValidationResult,
    ChannelCaptionPattern,
)

if TYPE_CHECKING:
    from ..config.sections.download import CaptionFirstConfig

logger = logging.getLogger(__name__)


class CaptionCache(BaseCache):
    """Cache for YouTube captions, enabling cross-project reuse.

    Caches fetched captions by video_id + language code to avoid
    re-fetching the same captions across different projects.

    Features:
    - JSON index for fast lookups
    - Age-based expiration (configurable days)
    - Cache key: video_id_language (e.g., 'dQw4w9WgXcQ_en')
    - Stores caption text, timing info, is_auto_generated, fetch_timestamp

    Usage:
        cache = CaptionCache(config.download.caption_first)

        # Check cache before fetching
        cached = cache.get_caption("dQw4w9WgXcQ", "en")
        if cached:
            result = cached.to_caption_result()
        else:
            result = fetcher.fetch_captions("dQw4w9WgXcQ")
            cache.store(result)

    Example with CaptionFetcher:
        fetcher = CaptionFetcher(config)
        cache = CaptionCache(config.download.caption_first)

        def fetch_with_cache(video_id: str, language: str = "en"):
            cached = cache.get_caption(video_id, language)
            if cached:
                return cached.to_caption_result()
            result = fetcher.fetch_captions(video_id, language=language)
            cache.store(result)
            return result
    """

    def __init__(self, config: Optional['CaptionFirstConfig'] = None):
        """Initialize the caption cache.

        Args:
            config: CaptionFirstConfig with cache settings. If None, uses defaults.
        """
        # Get config values with defaults
        if config:
            cache_dir = getattr(config, 'cache_dir', '~/.matcher_caption_cache')
            max_age_days = getattr(config, 'max_cache_age_days', 30)
            self.enabled = getattr(config, 'cache_captions', True)
            # Cache validation settings (US-008 Sprint 6)
            self.validation_mode = getattr(config, 'cache_validation', 'warn')
            self.validation_tolerance = getattr(config, 'cache_validation_tolerance', 0.2)
            # Negative cache TTL (US-60-004, US-63-005)
            # Separate TTL for "unavailable" entries (default 1 hour = 3600 seconds)
            # US-63-005: Prefer negative_cache_ttl_seconds if set, else use hours
            negative_cache_ttl_seconds = getattr(config, 'negative_cache_ttl_seconds', None)
            if isinstance(negative_cache_ttl_seconds, (int, float)) and negative_cache_ttl_seconds > 0:
                # Convert seconds to hours for internal storage
                self.negative_cache_ttl_hours = negative_cache_ttl_seconds / 3600.0
            else:
                self.negative_cache_ttl_hours = getattr(config, 'negative_cache_ttl_hours', 1.0)

            # Per-category negative cache TTL (US-90-003)
            # Different TTLs for different types of negative cache entries:
            # - unavailable_ttl_seconds: For videos with no captions (longer TTL)
            # - error_ttl_seconds: For transient errors (shorter TTL for faster retry)
            # Use isinstance check to handle Mock objects in tests
            # For backward compatibility, check if the new values differ from their defaults;
            # if at default, use the legacy negative_cache_ttl_seconds value
            unavailable_ttl = getattr(config, 'unavailable_ttl_seconds', None)
            legacy_ttl = getattr(config, 'negative_cache_ttl_seconds', None)
            if isinstance(unavailable_ttl, (int, float)) and unavailable_ttl != 3600:
                # New value explicitly set (different from default)
                self.unavailable_ttl_seconds = unavailable_ttl
            elif isinstance(legacy_ttl, (int, float)):
                # Fall back to legacy value
                self.unavailable_ttl_seconds = legacy_ttl
            else:
                self.unavailable_ttl_seconds = 3600

            error_ttl = getattr(config, 'error_ttl_seconds', None)
            if isinstance(error_ttl, (int, float)) and error_ttl != 300:
                # New value explicitly set (different from default)
                self.error_ttl_seconds = error_ttl
            else:
                # Use default
                self.error_ttl_seconds = 300

            # LRU eviction config (US-90-010)
            # Maximum number of entries in cache (0 = no limit)
            self.max_cache_size = getattr(config, 'max_cache_size', 0)
            # Validate max_cache_size if set (must be >= 100)
            if self.max_cache_size > 0 and self.max_cache_size < 100:
                raise ValueError(
                    f"max_cache_size must be >= 100 if set, got {self.max_cache_size}"
                )
        else:
            cache_dir = '~/.matcher_caption_cache'
            max_age_days = 30
            self.enabled = True
            self.validation_mode = 'warn'
            self.validation_tolerance = 0.2
            self.negative_cache_ttl_hours = 1.0  # 1 hour default = 3600 seconds
            # Per-category negative cache TTL (US-90-003)
            self.unavailable_ttl_seconds = 3600
            self.error_ttl_seconds = 300
            # LRU eviction config (US-90-010)
            self.max_cache_size = 0

        # Expand ~ in cache_dir
        cache_dir = Path(os.path.expanduser(cache_dir))

        # Convert max_age_days to TTL seconds (0 = no expiration)
        ttl_seconds = max_age_days * 24 * 3600 if max_age_days > 0 else 0

        # Initialize BaseCache
        super().__init__(
            cache_dir=cache_dir,
            index_name="caption_cache_index.json",
            ttl_seconds=ttl_seconds,
            auto_save=True
        )

        self.max_age_days = max_age_days

        # Track eviction statistics (US-100-004)
        self._eviction_count = 0

        logger.debug(f"CaptionCache initialized: dir={cache_dir}, "
                    f"ttl={max_age_days} days, enabled={self.enabled}, "
                    f"validation={self.validation_mode}, max_cache_size={self.max_cache_size}")

    def _evict_lru_entries(self, dry_run: bool = False) -> Dict[str, Any]:
        """Evict least-recently-used entries when cache exceeds max_cache_size (US-90-010).

        Args:
            dry_run: If True, only count entries to evict without removing.

        Returns:
            Dict with eviction results:
            - entries_evicted: Number of entries evicted
            - evicted_keys: List of evicted keys
            - current_size: Current entry count after eviction
            - dry_run: Whether this was a dry run
        """
        if self.max_cache_size <= 0:
            # No limit configured
            return {
                'entries_evicted': 0,
                'evicted_keys': [],
                'current_size': self._count_entries(),
                'dry_run': dry_run,
            }

        current_count = self._count_entries()
        if current_count <= self.max_cache_size:
            # Under limit, no eviction needed
            return {
                'entries_evicted': 0,
                'evicted_keys': [],
                'current_size': current_count,
                'dry_run': dry_run,
            }

        # Get entries sorted by last_used (LRU - oldest first)
        # BaseCache stores entries with 'cached_at', we track 'last_used' in metadata
        entries_to_evict = []
        entries_sorted = self.get_entries_sorted(sort_key='last_used', reverse=False)

        # Need to evict entries_count_to_evict
        entries_to_evict_count = current_count - self.max_cache_size

        for key, entry_data in entries_sorted:
            # Skip metadata entries
            if key.startswith('__'):
                continue
            if len(entries_to_evict) >= entries_to_evict_count:
                break
            entries_to_evict.append(key)

        if not dry_run:
            for key in entries_to_evict:
                if key in self.index:
                    del self.index[key]
            if entries_to_evict and self.auto_save:
                self._save_index()

            # Track eviction count (US-100-004)
            self._eviction_count += len(entries_to_evict)

            logger.debug(f"LRU eviction: removed {len(entries_to_evict)} entries "
                       f"(cache now has {self._count_entries()} entries)")

        return {
            'entries_evicted': len(entries_to_evict),
            'evicted_keys': entries_to_evict,
            'current_size': self._count_entries(),
            'dry_run': dry_run,
        }

    def _update_last_used(self, key: str) -> None:
        """Update last_used timestamp for LRU tracking (US-90-010).

        Args:
            key: Cache key to update.
        """
        if key not in self.index:
            return

        try:
            entry_data = self.index[key]
            # Update last_used in metadata
            metadata = entry_data.get('metadata', {})
            metadata['last_used'] = time.time()
            entry_data['metadata'] = metadata
            entry_data['last_used'] = time.time()
            # Don't auto-save here - it's too frequent; rely on periodic saves
        except Exception as e:
            logger.debug(f"Failed to update last_used for {key}: {e}")

    def _make_cache_key(self, video_id: str, language: str, is_auto_generated: bool = False) -> str:
        """Create cache key from video_id, language, and auto-generated flag.

        US-67-012: Auto-generated captions use a distinct '_autosub' suffix
        to avoid confusion with manual captions in the cache.

        Args:
            video_id: YouTube video ID (11 characters).
            language: ISO 639-1 language code (e.g., 'en').
            is_auto_generated: Whether caption is auto-generated.

        Returns:
            Cache key in format 'video_id_language' (e.g., 'dQw4w9WgXcQ_en')
            or 'video_id_language_autosub' for auto-generated captions.
        """
        key = f"{video_id}_{language}"
        if is_auto_generated:
            key += "_autosub"
        return key

    def _serialize_entry(self, entry: CacheEntry) -> Dict[str, Any]:
        """Serialize CachedCaption to dict with LRU tracking (US-90-010)."""
        # Ensure metadata has last_used for LRU tracking
        metadata = dict(entry.metadata) if entry.metadata else {}
        if 'last_used' not in metadata:
            metadata['last_used'] = entry.cached_at

        return {
            'data': entry.data,  # CachedCaption.to_dict()
            'cached_at': entry.cached_at,
            'metadata': metadata,
            'last_used': metadata.get('last_used', entry.cached_at),  # For sorting
        }

    def _deserialize_entry(self, data: Dict[str, Any]) -> CacheEntry:
        """Deserialize dict to CachedCaption entry with LRU tracking (US-90-010)."""
        metadata = data.get('metadata', {})
        # Ensure last_used exists for LRU sorting
        if 'last_used' not in metadata:
            metadata['last_used'] = data.get('cached_at', time.time())

        return CacheEntry(
            data=data.get('data', {}),
            cached_at=data.get('cached_at', 0.0),
            key='',
            metadata=metadata
        )

    def _is_valid_entry(self, entry: CacheEntry) -> bool:
        """Override to prevent auto-deletion of stale entries (US-004 Sprint 8).

        All validation modes (strict/warn/skip) return True here to prevent
        BaseCache.get() from auto-deleting entries. Staleness handling
        (skip/warn/reject) is done in get_caption() instead.

        This ensures stale entries are preserved for explicit cleanup via
        --cleanup-caption-cache rather than being silently deleted.
        """
        # Never auto-delete based on TTL - let get_caption() handle staleness
        return True

    def is_stale(self, entry: CacheEntry, max_age_days: Optional[int] = None) -> bool:
        """Check if a cache entry is stale based on age (US-004 Sprint 8).

        Args:
            entry: Cache entry to check.
            max_age_days: Override max age in days. If None, uses config value.

        Returns:
            True if the entry is older than max_age_days, False otherwise.
            Returns False if max_age_days is 0 (no expiration).
        """
        max_days = max_age_days if max_age_days is not None else self.max_age_days
        if max_days <= 0:
            return False

        age_seconds = time.time() - entry.cached_at
        max_age_seconds = max_days * 24 * 3600
        return age_seconds > max_age_seconds

    def get_entry_age_days(self, entry: CacheEntry) -> float:
        """Get the age of a cache entry in days (US-004 Sprint 8).

        Args:
            entry: Cache entry to check.

        Returns:
            Age in days (fractional).
        """
        age_seconds = time.time() - entry.cached_at
        return age_seconds / (24 * 3600)

    def cleanup_stale_entries(
        self,
        max_age_days: Optional[int] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Remove stale cache entries older than max_age_days (US-004 Sprint 8).

        Args:
            max_age_days: Maximum age in days. If None, uses config value.
            dry_run: If True, only count entries without removing them.

        Returns:
            Dict with cleanup results:
            - entries_removed: Number of stale entries removed
            - bytes_freed: Estimated bytes freed (from JSON serialization size)
            - oldest_removed_days: Age of oldest removed entry in days
            - dry_run: Whether this was a dry run

        Example:
            >>> cache = CaptionCache(config)
            >>> result = cache.cleanup_stale_entries(max_age_days=7)
            >>> print(f"Removed {result['entries_removed']} stale entries")
        """
        max_days = max_age_days if max_age_days is not None else self.max_age_days
        if max_days <= 0:
            return {
                'entries_removed': 0,
                'bytes_freed': 0,
                'oldest_removed_days': 0,
                'dry_run': dry_run,
            }

        stale_keys = []
        oldest_age_days = 0
        total_bytes = 0

        for key in list(self.index.keys()):
            # Skip metadata entries
            if key.startswith('__'):
                continue

            try:
                entry_data = self.index[key]
                entry = self._deserialize_entry(entry_data)

                if self.is_stale(entry, max_days):
                    stale_keys.append(key)
                    age_days = self.get_entry_age_days(entry)
                    if age_days > oldest_age_days:
                        oldest_age_days = age_days
                    # Estimate bytes from JSON serialization
                    total_bytes += len(json.dumps(entry_data))
            except Exception as e:
                logger.debug(f"Error checking entry {key}: {e}")
                # Mark corrupt entries for removal too
                stale_keys.append(key)

        if not dry_run:
            for key in stale_keys:
                if key in self.index:
                    del self.index[key]

            if stale_keys and self.auto_save:
                self._save_index()

            if stale_keys:
                logger.info(
                    f"Cleaned up {len(stale_keys)} stale caption cache entries "
                    f"(oldest: {oldest_age_days:.1f} days)"
                )

        return {
            'entries_removed': len(stale_keys),
            'bytes_freed': total_bytes,
            'oldest_removed_days': oldest_age_days,
            'dry_run': dry_run,
        }

    def validate_cache_entry(
        self,
        cached: CachedCaption,
        video_id: str,
        language: str
    ) -> CacheValidationResult:
        """Validate a cached caption entry for integrity (US-008 Sprint 6).

        Checks:
        1. video_id matches requested video_id
        2. language matches requested language
        3. segment_count is within tolerance of expected count (based on duration)

        The expected segment count is estimated as: duration / 3 (seconds per segment)
        since typical caption segments are ~3 seconds long.

        Args:
            cached: CachedCaption to validate.
            video_id: Expected video ID.
            language: Expected language code.

        Returns:
            CacheValidationResult with is_valid=True if all checks pass,
            or is_valid=False with reason explaining the failure.

        Example:
            >>> cache = CaptionCache(config)
            >>> cached = cache.get_caption("dQw4w9WgXcQ", "en")
            >>> if cached:
            ...     result = cache.validate_cache_entry(cached, "dQw4w9WgXcQ", "en")
            ...     if not result.is_valid:
            ...         logger.warning(f"Cache validation failed: {result.reason}")
        """
        # Check video_id match
        if cached.video_id != video_id:
            return CacheValidationResult(
                is_valid=False,
                video_id=video_id,
                language=language,
                reason=f"video_id mismatch: cached={cached.video_id}, expected={video_id}",
            )

        # Check language match
        if cached.language != language:
            return CacheValidationResult(
                is_valid=False,
                video_id=video_id,
                language=language,
                reason=f"language mismatch: cached={cached.language}, expected={language}",
            )

        # Check segment count consistency (if duration is available)
        actual_count = len(cached.segments)
        if cached.duration > 0:
            # Expect ~1 segment per 3 seconds of video
            expected_count = max(1, int(cached.duration / 3))
            deviation = abs(actual_count - expected_count) / max(expected_count, 1)

            if deviation > self.validation_tolerance:
                return CacheValidationResult(
                    is_valid=False,
                    video_id=video_id,
                    language=language,
                    reason=(
                        f"segment_count deviation {deviation:.1%} exceeds tolerance "
                        f"{self.validation_tolerance:.0%}: expected ~{expected_count}, "
                        f"got {actual_count}"
                    ),
                    expected_segment_count=expected_count,
                    actual_segment_count=actual_count,
                    segment_count_deviation=deviation,
                )
        else:
            # No duration info - can't validate segment count, just check non-empty
            if actual_count == 0:
                return CacheValidationResult(
                    is_valid=False,
                    video_id=video_id,
                    language=language,
                    reason="cached caption has 0 segments",
                    actual_segment_count=0,
                )

        # All checks passed
        return CacheValidationResult(
            is_valid=True,
            video_id=video_id,
            language=language,
            actual_segment_count=actual_count,
            expected_segment_count=int(cached.duration / 3) if cached.duration > 0 else None,
        )

    def get_caption(self, video_id: str, language: str, auto_generated: Optional[bool] = None) -> Optional[CachedCaption]:
        """Get cached caption for a video and language.

        Staleness handling depends on validation_mode (US-004 Sprint 8):
        - 'strict': Returns None for stale entries (triggers re-fetch)
        - 'warn': Logs warning for stale entries but returns them
        - 'skip': No staleness check, returns cached data as-is

        US-67-012: When auto_generated is None (default), tries the manual key
        first, then falls back to the '_autosub' key. When explicitly True/False,
        looks up only the specific key variant.

        Args:
            video_id: YouTube video ID.
            language: ISO 639-1 language code.
            auto_generated: If None, try both manual and auto keys.
                If True, only look up auto-generated key.
                If False, only look up manual key.

        Returns:
            CachedCaption if found and valid, None otherwise.
        """
        if not self.enabled:
            return None

        if auto_generated is None:
            # Try manual first, then auto-generated (US-67-012)
            key = self._make_cache_key(video_id, language, False)
            entry = self.get(key)
            if entry is None:
                # Undo the miss count from first lookup — this is a single
                # logical lookup that tries two keys (US-67-012)
                self._misses -= 1
                key = self._make_cache_key(video_id, language, True)
                entry = self.get(key)
            elif isinstance(entry.data, dict) and entry.data.get('unavailable', False):
                # Skip unavailable markers — the autosub key may have a positive entry
                self._hits -= 1
                key = self._make_cache_key(video_id, language, True)
                entry = self.get(key)
        else:
            key = self._make_cache_key(video_id, language, auto_generated)
            entry = self.get(key)

        if entry is None:
            logger.debug(f"Caption cache miss: {key}")
            return None

        # Check staleness based on validation_mode (US-004 Sprint 8)
        if self.validation_mode != 'skip' and self.is_stale(entry):
            age_days = self.get_entry_age_days(entry)
            if self.validation_mode == 'strict':
                logger.info(
                    f"Caption cache stale (strict mode): {key} "
                    f"(age: {age_days:.1f} days, max: {self.max_age_days} days)"
                )
                # Don't delete - let user investigate or cleanup explicitly
                return None
            else:  # 'warn' mode
                logger.warning(
                    f"Caption cache stale: {key} "
                    f"(age: {age_days:.1f} days, max: {self.max_age_days} days) - "
                    f"returning cached data, consider running --cleanup-caption-cache"
                )

        try:
            cached = CachedCaption.from_dict(entry.data)

            # Update last_used for LRU tracking (US-90-010)
            self._update_last_used(key)

            logger.debug(f"Caption cache hit: {key} "
                        f"({len(cached.segments)} segments, "
                        f"auto={cached.is_auto_generated})")
            return cached
        except Exception as e:
            logger.warning(f"Failed to deserialize cached caption {key}: {e}")
            self.delete(key)
            return None

    def get_validated_caption(
        self,
        video_id: str,
        language: str,
        metrics: Optional['CaptionMetrics'] = None
    ) -> tuple[Optional[CachedCaption], Optional[CacheValidationResult]]:
        """Get cached caption with validation (US-008 Sprint 6).

        Retrieves cached caption and validates its integrity based on the
        configured validation_mode:
        - 'skip': No validation, returns cached data as-is
        - 'warn': Validates and logs warning on failure, returns None
        - 'strict': Validates and returns None on failure (rejects cache)

        Args:
            video_id: YouTube video ID.
            language: ISO 639-1 language code.
            metrics: Optional CaptionMetrics to track validation results.

        Returns:
            Tuple of (CachedCaption or None, CacheValidationResult or None).
            First element is the cached data (None if miss or validation failed).
            Second element is the validation result (None if skip mode or miss).

        Example:
            >>> cache = CaptionCache(config)
            >>> cached, validation = cache.get_validated_caption("dQw4w9WgXcQ", "en")
            >>> if cached:
            ...     result = cached.to_caption_result()
            ... elif validation and not validation.is_valid:
            ...     logger.info(f"Cache rejected: {validation.reason}")
        """
        # Get raw cached data
        cached = self.get_caption(video_id, language)
        if cached is None:
            return None, None

        # Skip validation if mode is 'skip'
        if self.validation_mode == 'skip':
            return cached, None

        # Validate the cached entry
        validation_result = self.validate_cache_entry(cached, video_id, language)

        # Track validation results in metrics
        if metrics is not None:
            metrics.record_cache_validation(validation_result)

        if validation_result.is_valid:
            return cached, validation_result

        # Handle validation failure based on mode
        # US-67-012: Use auto_generated flag from cached entry for correct key
        key = self._make_cache_key(video_id, language, cached.is_auto_generated)
        if self.validation_mode == 'strict':
            logger.warning(
                f"Cache validation REJECTED {key}: {validation_result.reason}"
            )
            # Don't delete in strict mode - let user investigate
            return None, validation_result
        else:  # 'warn' mode
            logger.warning(
                f"Cache validation failed {key}: {validation_result.reason} "
                f"(will re-fetch)"
            )
            # Invalidate the corrupted entry so it gets re-fetched
            self.delete(key)
            return None, validation_result

    def store(self, result: 'CaptionResult') -> bool:
        """Store a CaptionResult in the cache.

        Args:
            result: CaptionResult to cache.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self.enabled:
            return False

        if not result.segments:
            logger.debug(f"Not caching empty caption result for {result.video_id}")
            return False

        # US-67-012: Use '_autosub' suffix for auto-generated captions
        key = self._make_cache_key(result.video_id, result.language, result.is_auto_generated)

        cached = CachedCaption(
            video_id=result.video_id,
            language=result.language,
            segments=[seg.to_dict() for seg in result.segments],
            is_auto_generated=result.is_auto_generated,
            format_source=result.format_source,
            fetch_timestamp=time.time(),
            duration=result.duration,
            caption_quality=result.caption_quality,
            coverage_ratio=result.coverage_ratio,
            unavailable=False,
            # US-78-007: Preserve metadata fields in cache
            video_description=result.video_description,
            video_chapters=result.video_chapters,
            video_tags=result.video_tags,
            language_confidence=result.language_confidence,
            fallback_language=result.fallback_language,
        )

        # Store with last_used metadata for LRU tracking (US-90-010)
        metadata = {'last_used': time.time()}
        self.set(key, cached.to_dict(), metadata=metadata)

        # Trigger LRU eviction if cache size exceeds max_cache_size (US-90-010)
        if self.max_cache_size > 0:
            self._evict_lru_entries()

        # Log size change at DEBUG (US-100-004)
        current_size = self._count_entries()
        logger.debug(f"Cache size after storing {key}: {current_size} entries")

        logger.info(f"Cached captions: {key} "
                   f"({len(result.segments)} segments, "
                   f"duration={result.duration:.1f}s, "
                   f"auto={result.is_auto_generated})")
        return True

    def store_unavailable(self, video_id: str, language: str = "en") -> bool:
        """Store a "captions unavailable" entry in the cache.

        This caches the knowledge that a video has no captions available,
        avoiding repeated API calls for videos known to lack captions.

        US-90-003: Uses unavailable_ttl_seconds for TTL (default 1 hour).

        Args:
            video_id: YouTube video ID.
            language: Language code that was checked.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self.enabled:
            return False

        key = self._make_cache_key(video_id, language)

        cached = CachedCaption(
            video_id=video_id,
            language=language,
            segments=[],  # Empty segments for unavailable
            is_auto_generated=False,
            format_source="unavailable",
            fetch_timestamp=time.time(),
            duration=0.0,
            caption_quality="low",
            coverage_ratio=0.0,
            unavailable=True,  # Mark as unavailable
        )

        self.set(key, cached.to_dict())

        # Trigger LRU eviction if cache size exceeds max_cache_size (US-90-010)
        if self.max_cache_size > 0:
            self._evict_lru_entries()

        # Log size change at DEBUG (US-100-004)
        logger.debug(f"Cache size after storing unavailable {key}: {self._count_entries()} entries")

        logger.info(f"Cached unavailable captions: {key} (TTL: {self.unavailable_ttl_seconds}s)")
        return True

    def store_error(self, video_id: str, language: str = "en") -> bool:
        """Store a transient error entry in the cache (US-90-003).

        This caches temporary failures (network errors, timeouts, etc.) with a
        shorter TTL than "unavailable" entries to allow faster retry.

        US-90-003: Uses error_ttl_seconds for TTL (default 5 minutes).

        Args:
            video_id: YouTube video ID.
            language: Language code that was checked.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self.enabled:
            return False

        key = self._make_cache_key(video_id, language)

        cached = CachedCaption(
            video_id=video_id,
            language=language,
            segments=[],  # Empty segments for error
            is_auto_generated=False,
            format_source="error",  # Mark as error type
            fetch_timestamp=time.time(),
            duration=0.0,
            caption_quality="low",
            coverage_ratio=0.0,
            unavailable=True,  # Still mark as unavailable since no captions
        )

        self.set(key, cached.to_dict())

        # Trigger LRU eviction if cache size exceeds max_cache_size (US-90-010)
        if self.max_cache_size > 0:
            self._evict_lru_entries()

        # Log size change at DEBUG (US-100-004)
        logger.debug(f"Cache size after storing error {key}: {self._count_entries()} entries")

        logger.info(f"Cached error status: {key} (TTL: {self.error_ttl_seconds}s)")
        return True

    def is_negative_entry_stale(self, entry: CacheEntry) -> bool:
        """Check if a negative cache entry is stale based on per-category TTL (US-90-003).

        Uses different TTLs for different types of negative cache entries:
        - "unavailable": Uses unavailable_ttl_seconds (default 1 hour)
        - "error": Uses error_ttl_seconds (default 5 minutes)

        This allows faster retry for transient errors while keeping longer
        cache for confirmed unavailable captions.

        Args:
            entry: Cache entry to check.

        Returns:
            True if the entry is older than the category-specific TTL, False otherwise.
            Returns False if both TTLs are 0 (uses max_age_days instead).
        """
        # Get the category from format_source (default to "unavailable" for backward compat)
        format_source = entry.data.get('format_source', 'unavailable')

        # Determine TTL based on category
        if format_source == 'error':
            # Error entries use shorter TTL
            if self.error_ttl_seconds <= 0:
                return self.is_stale(entry)
            max_age_seconds = self.error_ttl_seconds
        else:
            # Unavailable entries use longer TTL
            if self.unavailable_ttl_seconds <= 0:
                return self.is_stale(entry)
            max_age_seconds = self.unavailable_ttl_seconds

        age_seconds = time.time() - entry.cached_at
        return age_seconds > max_age_seconds

    def is_caption_unavailable(
        self,
        video_id: str,
        language: str = "en",
        check_staleness: bool = True
    ) -> bool:
        """Check if captions are known to be unavailable for a video.

        TTL handling (US-60-004):
        Uses negative_cache_ttl_hours (default 1 hour) instead of max_age_days
        for negative entries. This allows captions to be re-checked more frequently
        since availability may change.

        Staleness handling (when check_staleness=True):
        - 'strict': Returns False for stale entries (triggers re-check)
        - 'warn': Logs warning but returns True (cached unavailable is used)
        - 'skip': No staleness check, returns cached status as-is

        Args:
            video_id: YouTube video ID.
            language: Language code to check.
            check_staleness: If True, check if cache entry is stale before
                returning unavailable status. Default is True to prevent
                false negatives from stale cache entries.

        Returns:
            True if captions are cached as unavailable and not stale,
            False otherwise (not cached, not unavailable, or stale).
        """
        if not self.enabled:
            return False

        key = self._make_cache_key(video_id, language)
        entry = self.get(key)

        if entry is None:
            return False

        try:
            cached = CachedCaption.from_dict(entry.data)
            if not cached.unavailable:
                return False

            # US-60-004: Check staleness for negative entries using negative_cache_ttl_hours
            if check_staleness and self.validation_mode != 'skip' and self.is_negative_entry_stale(entry):
                age_hours = (time.time() - entry.cached_at) / 3600
                ttl_hours = self.negative_cache_ttl_hours if self.negative_cache_ttl_hours > 0 else (self.max_age_days * 24)
                if self.validation_mode == 'strict':
                    logger.info(
                        f"Caption unavailable cache expired (strict mode): {key} "
                        f"(age: {age_hours:.1f}h, max: {ttl_hours:.1f}h)"
                    )
                    # Treat as not unavailable - will trigger re-check
                    return False
                else:  # 'warn' mode
                    logger.warning(
                        f"Caption unavailable cache expired: {key} "
                        f"(age: {age_hours:.1f}h, max: {ttl_hours:.1f}h) - "
                        f"consider running --cleanup-caption-cache"
                    )
                    # In warn mode, still return True but log the warning

            return cached.unavailable
        except Exception:
            return False

    def get_or_fetch(
        self,
        fetcher: 'CaptionFetcher',
        video_id: str,
        language: str = "en",
        prefer_manual: bool = True
    ) -> 'CaptionResult':
        """Get from cache or fetch and cache.

        Convenience method that combines cache lookup and fetching.

        Args:
            fetcher: CaptionFetcher instance to use for fetching.
            video_id: YouTube video ID.
            language: Preferred language code.
            prefer_manual: Prefer manual captions over auto-generated.

        Returns:
            CaptionResult from cache or freshly fetched.

        Raises:
            CaptionUnavailableError: If no captions exist.
            CaptionFetchError: If fetch fails due to network/temporary error.
        """
        # Check cache first
        cached = self.get_caption(video_id, language)
        if cached:
            return cached.to_caption_result()

        # Fetch and cache
        result = fetcher.fetch_captions(video_id, language=language, prefer_manual=prefer_manual)
        self.store(result)
        return result

    def get_or_fetch_with_retry(
        self,
        fetcher: 'CaptionFetcher',
        video_id: str,
        language: str = "en",
        prefer_manual: bool = True,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ) -> 'CaptionResult':
        """Get from cache or fetch with retry logic and cache (US-008).

        Combines cache lookup with fetch retry logic. Uses exponential backoff
        for network errors (CaptionFetchError) but NOT for permanent errors
        (CaptionUnavailableError).

        Args:
            fetcher: CaptionFetcher instance to use for fetching.
            video_id: YouTube video ID.
            language: Preferred language code.
            prefer_manual: Prefer manual captions over auto-generated.
            max_retries: Override max retry attempts (default: fetcher config).
            retry_delay: Override base delay between retries (default: fetcher config).

        Returns:
            CaptionResult from cache or freshly fetched.

        Raises:
            CaptionUnavailableError: If no captions exist (NOT retried).
            CaptionFetchError: If all retries fail due to network/temporary errors.

        Example:
            cache = CaptionCache(config.download.caption_first)
            fetcher = CaptionFetcher(config)
            try:
                result = cache.get_or_fetch_with_retry(fetcher, "dQw4w9WgXcQ")
            except CaptionFetchError:
                # Mark for transcription fallback
                needs_transcription.append("dQw4w9WgXcQ")
        """
        # Check cache first
        cached = self.get_caption(video_id, language)
        if cached:
            return cached.to_caption_result()

        # Fetch with retry and cache
        result = fetcher.fetch_captions_with_retry(
            video_id,
            language=language,
            prefer_manual=prefer_manual,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        self.store(result)
        return result

    def invalidate(self, video_id: str, language: Optional[str] = None) -> int:
        """Invalidate cached captions for a video.

        Args:
            video_id: YouTube video ID.
            language: If provided, only invalidate for this language.
                     If None, invalidate all languages for this video.

        Returns:
            Number of cache entries invalidated.
        """
        if language:
            # US-67-012: Invalidate both manual and auto-generated keys
            invalidated = 0
            for auto in (False, True):
                key = self._make_cache_key(video_id, language, auto)
                if self.delete(key):
                    logger.debug(f"Invalidated caption cache: {key}")
                    invalidated += 1
            return invalidated

        # Invalidate all languages for this video
        invalidated = 0
        for key in list(self.index.keys()):
            if key.startswith(f"{video_id}_"):
                self.delete(key)
                invalidated += 1

        if invalidated:
            logger.debug(f"Invalidated {invalidated} caption cache entries for {video_id}")

        return invalidated

    def save_format_statistics(self, format_success_counts: Dict[str, int]) -> bool:
        """Save format success counts to cache metadata (US-002 Sprint 7).

        Persists format success statistics for cross-run learning.
        These statistics enable adaptive format ordering, trying
        historically successful formats first.

        Args:
            format_success_counts: Dict mapping format name -> success count.
                                   Example: {'json3': 95, 'vtt': 80, 'srt': 25}

        Returns:
            True if saved successfully, False otherwise.

        Example:
            >>> cache = CaptionCache(config)
            >>> metrics = CaptionMetrics()
            >>> # After batch fetch...
            >>> cache.save_format_statistics(metrics.format_success_counts)
            True
        """
        if not self.enabled:
            return False

        try:
            # Store in a special metadata entry
            metadata_key = "__format_statistics__"
            self.index[metadata_key] = {
                'format_success_counts': format_success_counts,
                'updated_at': time.time(),
                'total_samples': sum(format_success_counts.values()),
            }
            if self.auto_save:
                self._save_index()

            logger.debug(f"Saved format statistics: {format_success_counts}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save format statistics: {e}")
            return False

    def load_format_statistics(self) -> Dict[str, int]:
        """Load format success counts from cache metadata (US-002 Sprint 7).

        Retrieves persisted format success statistics for adaptive ordering.
        Returns empty dict if no historical data available.

        Returns:
            Dict mapping format name -> success count.
            Empty dict if no historical data exists.

        Example:
            >>> cache = CaptionCache(config)
            >>> historical = cache.load_format_statistics()
            >>> if historical:
            ...     metrics.format_success_counts = historical
            ...     optimal_order = metrics.get_optimal_format_order()
        """
        if not self.enabled:
            return {}

        try:
            metadata_key = "__format_statistics__"
            metadata = self.index.get(metadata_key, {})
            format_counts = metadata.get('format_success_counts', {})

            if format_counts:
                total = metadata.get('total_samples', sum(format_counts.values()))
                logger.debug(f"Loaded format statistics: {format_counts} ({total} samples)")

            return format_counts
        except Exception as e:
            logger.warning(f"Failed to load format statistics: {e}")
            return {}

    def save_channel_patterns(
        self,
        patterns: Dict[str, 'ChannelCaptionPattern']
    ) -> bool:
        """Save channel caption patterns to cache metadata (US-006 Sprint 7).

        Persists channel-level caption availability patterns for cross-run
        optimization. These patterns enable batch pre-check grouping where
        channels with high confidence patterns skip individual pre-checks.

        Args:
            patterns: Dict mapping channel_id -> ChannelCaptionPattern.

        Returns:
            True if saved successfully, False otherwise.

        Example:
            >>> cache = CaptionCache(config)
            >>> patterns = {
            ...     "UCabc": ChannelCaptionPattern("UCabc", 10, 9, 0.9, time.time()),
            ...     "UCdef": ChannelCaptionPattern("UCdef", 5, 0, 0.0, time.time()),
            ... }
            >>> cache.save_channel_patterns(patterns)
            True
        """
        if not self.enabled:
            return False

        try:
            metadata_key = "__channel_patterns__"
            self.index[metadata_key] = {
                'patterns': {
                    cid: pattern.to_dict()
                    for cid, pattern in patterns.items()
                },
                'updated_at': time.time(),
                'total_channels': len(patterns),
            }
            if self.auto_save:
                self._save_index()

            logger.debug(f"Saved {len(patterns)} channel caption patterns")
            return True
        except Exception as e:
            logger.warning(f"Failed to save channel patterns: {e}")
            return False

    def load_channel_patterns(self) -> Dict[str, 'ChannelCaptionPattern']:
        """Load channel caption patterns from cache metadata (US-006 Sprint 7).

        Retrieves persisted channel-level caption availability patterns for
        batch pre-check optimization. Channels with >90% confidence and
        sufficient samples can skip individual pre-checks.

        Returns:
            Dict mapping channel_id -> ChannelCaptionPattern.
            Empty dict if no historical data exists.

        Example:
            >>> cache = CaptionCache(config)
            >>> patterns = cache.load_channel_patterns()
            >>> for cid, pattern in patterns.items():
            ...     if pattern.success_rate > 0.9 and pattern.videos_checked >= 5:
            ...         print(f"Channel {cid}: high confidence ({pattern.success_rate:.0%})")
        """
        if not self.enabled:
            return {}

        try:
            metadata_key = "__channel_patterns__"
            metadata = self.index.get(metadata_key, {})
            patterns_data = metadata.get('patterns', {})

            patterns = {}
            for cid, data in patterns_data.items():
                patterns[cid] = ChannelCaptionPattern.from_dict(data)

            if patterns:
                total_channels = metadata.get('total_channels', len(patterns))
                logger.debug(f"Loaded {total_channels} channel caption patterns")

            return patterns
        except Exception as e:
            logger.warning(f"Failed to load channel patterns: {e}")
            return {}

    def get_channel_pattern(self, channel_id: str) -> Optional['ChannelCaptionPattern']:
        """Get a single channel's caption pattern (US-006 Sprint 7).

        Convenience method to retrieve pattern for a specific channel.

        Args:
            channel_id: YouTube channel ID (UCxxxx format).

        Returns:
            ChannelCaptionPattern if found, None otherwise.
        """
        patterns = self.load_channel_patterns()
        return patterns.get(channel_id)

    def update_channel_pattern(
        self,
        channel_id: str,
        has_captions: bool
    ) -> 'ChannelCaptionPattern':
        """Update a channel's caption pattern with a new observation (US-006 Sprint 7).

        Atomically loads existing patterns, updates the specified channel,
        and saves back to cache.

        Args:
            channel_id: YouTube channel ID (UCxxxx format).
            has_captions: Whether the checked video had captions.

        Returns:
            Updated ChannelCaptionPattern for the channel.

        Example:
            >>> cache = CaptionCache(config)
            >>> pattern = cache.update_channel_pattern("UCabc", has_captions=True)
            >>> print(f"Channel {pattern.channel_id}: {pattern.success_rate:.0%} success")
        """
        patterns = self.load_channel_patterns()

        if channel_id not in patterns:
            patterns[channel_id] = ChannelCaptionPattern(channel_id=channel_id)

        patterns[channel_id].update(has_captions)
        self.save_channel_patterns(patterns)

        return patterns[channel_id]

    def save_channel_info(self, channel_info: Dict[str, str]) -> bool:
        """Save video_id -> channel_id mapping to cache.

        This enables fast path in batch_precheck_by_channel() by avoiding
        redundant metadata fetches for videos we've already seen.

        Args:
            channel_info: Dict mapping video_id -> channel_id.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            # Validate video IDs (should be 11 chars for YouTube)
            valid_mappings = {
                vid: ch for vid, ch in channel_info.items()
                if vid and len(vid) == 11 and ch
            }

            metadata_key = "__meta:channel_info__"
            self.index[metadata_key] = {
                'mappings': valid_mappings,
                'updated_at': time.time(),
                'total_mappings': len(valid_mappings),
            }
            self._save_index()
            logger.debug(f"Saved {len(valid_mappings)} channel mappings to cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to save channel_info: {e}")
            return False  # Non-fatal, continue pipeline

    def load_channel_info(self, ttl_days: int = 30) -> Dict[str, str]:
        """Load video_id -> channel_id mapping from cache.

        Args:
            ttl_days: Maximum age of cached data in days. Default 30.

        Returns:
            Dict mapping video_id -> channel_id.
            Empty dict if no cached data exists or data is expired.
        """
        try:
            metadata_key = "__meta:channel_info__"
            metadata = self.index.get(metadata_key, {})

            updated_at = metadata.get('updated_at', 0)
            ttl_seconds = ttl_days * 24 * 60 * 60

            if time.time() - updated_at > ttl_seconds:
                logger.debug("Channel info cache expired")
                return {}

            mappings = metadata.get('mappings', {})
            logger.debug(f"Loaded {len(mappings)} channel mappings from cache")
            return mappings
        except Exception as e:
            logger.warning(f"Failed to load channel_info: {e}")
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache metrics including:
            - current_size: Number of entries in cache
            - eviction_count: Total number of entries evicted
            - oldest_entry_age: Age of oldest entry in days
        """
        base_stats = super().get_stats()

        # Add caption-specific stats
        total_segments = 0
        auto_generated_count = 0
        manual_count = 0

        # Track current_size and oldest_entry_age (US-100-004)
        current_size = 0
        oldest_cached_at = None

        for entry in self.get_all().values():
            try:
                cached = CachedCaption.from_dict(entry.data)
                total_segments += len(cached.segments)
                if cached.is_auto_generated:
                    auto_generated_count += 1
                else:
                    manual_count += 1

                # Track current_size and oldest entry (US-100-004)
                current_size += 1
                if oldest_cached_at is None or entry.cached_at < oldest_cached_at:
                    oldest_cached_at = entry.cached_at
            except Exception:
                pass

        # Calculate oldest_entry_age (US-100-004)
        oldest_entry_age = 0.0
        if oldest_cached_at is not None:
            oldest_entry_age = (time.time() - oldest_cached_at) / (24 * 3600)

        base_stats.update({
            'total_segments': total_segments,
            'auto_generated_entries': auto_generated_count,
            'manual_entries': manual_count,
            'max_age_days': self.max_age_days,
            'enabled': self.enabled,
            # LRU eviction statistics (US-100-004)
            'current_size': current_size,
            'eviction_count': self._eviction_count,
            'oldest_entry_age': oldest_entry_age,
            'max_cache_size': self.max_cache_size,
        })

        return base_stats


class ListSubsCache(BaseCache):
    """Cache for yt-dlp --list-subs output (US-59-012).

    Caches the result of list_available_languages() to avoid redundant
    subprocess calls when resuming pipeline runs. Each video's available
    languages are cached with a configurable TTL (default 1 hour) since
    subtitle availability can change.

    Features:
    - Persistent JSON storage in .cache/ directory
    - Configurable TTL (default 1 hour)
    - Compatible with CaptionCache location for unified caching

    Usage:
        cache = ListSubsCache(cache_dir=".cache/list_subs_cache")

        # Check cache before making subprocess call
        cached = cache.get_languages("dQw4w9WgXcQ")
        if cached:
            return cached  # List of AvailableLanguage objects

        # After fetching, store in cache
        languages = fetcher.list_available_languages("dQw4w9WgXcQ")
        cache.store("dQw4w9WgXcQ", languages)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_hours: float = 1.0,
        enabled: bool = True
    ):
        """Initialize the list-subs cache.

        Args:
            cache_dir: Cache directory. Defaults to ~/.matcher_caption_cache/list_subs.
            ttl_hours: TTL in hours for cache entries. Default 1 hour.
            enabled: Whether caching is enabled.
        """
        if cache_dir is None:
            cache_dir = Path(os.path.expanduser('~/.matcher_caption_cache/list_subs'))

        # Convert ttl_hours to seconds for BaseCache
        ttl_seconds = int(ttl_hours * 3600)

        super().__init__(
            cache_dir=cache_dir,
            index_name="list_subs_cache.json",
            ttl_seconds=ttl_seconds,
            auto_save=True
        )

        self.ttl_hours = ttl_hours
        self.enabled = enabled

        logger.debug(f"ListSubsCache initialized: dir={cache_dir}, "
                    f"ttl={ttl_hours}h, enabled={enabled}")

    def _serialize_entry(self, entry: CacheEntry) -> Dict[str, Any]:
        """Serialize CachedLanguageList to dict."""
        return {
            'data': entry.data,
            'cached_at': entry.cached_at,
            'metadata': entry.metadata
        }

    def _deserialize_entry(self, data: Dict[str, Any]) -> CacheEntry:
        """Deserialize dict to CachedLanguageList entry."""
        return CacheEntry(
            data=data.get('data', {}),
            cached_at=data.get('cached_at', 0.0),
            key='',
            metadata=data.get('metadata', {})
        )

    def get_languages(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached available languages for a video.

        Args:
            video_id: YouTube video ID.

        Returns:
            List of language dicts with 'code', 'name', 'is_auto_generated',
            or None if not cached or expired.
        """
        if not self.enabled:
            return None

        entry = self.get(video_id)
        if entry is None:
            logger.debug(f"ListSubsCache miss: {video_id}")
            return None

        try:
            cached_data = entry.data
            if isinstance(cached_data, dict):
                # Full CachedLanguageList format
                languages = cached_data.get('languages', [])
            else:
                # Direct list format (backward compat)
                languages = cached_data

            # US-62-004, US-63-005: Log specific message for negative cache hit at DEBUG level
            if len(languages) == 0:
                logger.debug(f"Negative cache hit: no captions for {video_id}")
            else:
                logger.debug(f"ListSubsCache hit: {video_id} ({len(languages)} languages)")
            return languages
        except Exception as e:
            logger.warning(f"Failed to deserialize list-subs cache {video_id}: {e}")
            self.delete(video_id)
            return None

    def store(
        self,
        video_id: str,
        languages: List[Any]  # List of AvailableLanguage or dicts
    ) -> bool:
        """Store available languages in cache.

        Args:
            video_id: YouTube video ID.
            languages: List of AvailableLanguage objects or dicts.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self.enabled:
            return False

        # Convert AvailableLanguage objects to dicts if needed
        language_dicts = []
        for lang in languages:
            if hasattr(lang, 'code'):
                # AvailableLanguage object
                language_dicts.append({
                    'code': lang.code,
                    'name': lang.name,
                    'is_auto_generated': lang.is_auto_generated
                })
            elif isinstance(lang, dict):
                language_dicts.append(lang)
            else:
                logger.warning(f"Unknown language type: {type(lang)}")
                continue

        cached_data = {
            'video_id': video_id,
            'languages': language_dicts,
            'cached_at': time.time(),
            'ttl_hours': self.ttl_hours,
        }

        self.set(video_id, cached_data)
        logger.debug(f"Cached list-subs: {video_id} ({len(language_dicts)} languages)")
        return True

    def is_no_captions_available(self, video_id: str) -> bool:
        """Check if video is known to have no captions available (US-62-004).

        This checks the negative cache (list-subs result was empty). Returns True
        only if there's a non-expired cache entry with an empty language list.

        Args:
            video_id: YouTube video ID.

        Returns:
            True if cached as having no captions, False otherwise
            (not cached or has captions).
        """
        if not self.enabled:
            return False

        entry = self.get(video_id)
        if entry is None:
            return False

        try:
            cached_data = entry.data
            if isinstance(cached_data, dict):
                languages = cached_data.get('languages', [])
            else:
                languages = cached_data

            return len(languages) == 0
        except Exception:
            return False
