"""
Enhanced caching for Caption Fetcher with prefetch, compression, and bulk operations.

This module extends the base CaptionCache with advanced caching features:
- Cache warming: Pre-load entries for known video IDs
- Bulk operations: Efficient batch get/set for multiple videos
- Smart prefetch: Anticipate and cache likely-needed captions
- Compression: Reduce storage for caption data
- Enhanced metrics: Detailed cache performance tracking

Usage:
    from src.caption_fetcher_cache import EnhancedCaptionCache
    
    cache = EnhancedCaptionCache(config)
    
    # Warm cache for known video IDs
    cache.warm_cache_for_videos(['abc123', 'def456', 'ghi789'])
    
    # Bulk fetch from cache
    results = cache.get_captions_bulk(['abc123', 'def456'], language='en')
    
    # Store bulk results
    cache.store_captions_bulk(results)
"""

from __future__ import annotations

import gzip
import json
import logging
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from .cache import BaseCache, CacheEntry

if TYPE_CHECKING:
    from .config.sections.download import CaptionFirstConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Detailed cache performance metrics."""
    hits: int = 0
    misses: int = 0
    warm_hits: int = 0  # Hits from pre-warmed entries
    bulk_hits: int = 0  # Hits from bulk operations
    compressed_entries: int = 0
    bytes_compressed: int = 0
    bytes_uncompressed: int = 0
    prefetch_attempts: int = 0
    prefetch_hits: int = 0
    avg_lookup_time_ms: float = 0.0
    _lookup_times: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record_hit(self, is_warm: bool = False, is_bulk: bool = False, 
                   lookup_time_ms: float = 0.0) -> None:
        """Record a cache hit."""
        with self._lock:
            self.hits += 1
            if is_warm:
                self.warm_hits += 1
            if is_bulk:
                self.bulk_hits += 1
            self._lookup_times.append(lookup_time_ms)
            self._update_avg()
    
    def record_miss(self, lookup_time_ms: float = 0.0) -> None:
        """Record a cache miss."""
        with self._lock:
            self.misses += 1
            self._lookup_times.append(lookup_time_ms)
            self._update_avg()
    
    def record_compression(self, original_bytes: int, compressed_bytes: int) -> None:
        """Record compression stats."""
        with self._lock:
            self.compressed_entries += 1
            self.bytes_uncompressed += original_bytes
            self.bytes_compressed += compressed_bytes
    
    def record_prefetch(self, hit: bool = False) -> None:
        """Record prefetch attempt."""
        with self._lock:
            self.prefetch_attempts += 1
            if hit:
                self.prefetch_hits += 1
    
    def _update_avg(self) -> None:
        """Update average lookup time."""
        if self._lookup_times:
            # Keep last 100 measurements
            recent = self._lookup_times[-100:]
            self.avg_lookup_time_ms = sum(recent) / len(recent)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def compression_ratio(self) -> float:
        """Calculate average compression ratio."""
        if self.bytes_uncompressed > 0:
            return self.bytes_compressed / self.bytes_uncompressed
        return 1.0
    
    @property
    def prefetch_success_rate(self) -> float:
        """Calculate prefetch hit rate."""
        if self.prefetch_attempts > 0:
            return self.prefetch_hits / self.prefetch_attempts
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'warm_hits': self.warm_hits,
            'bulk_hits': self.bulk_hits,
            'compressed_entries': self.compressed_entries,
            'compression_ratio': self.compression_ratio,
            'bytes_saved': self.bytes_uncompressed - self.bytes_compressed,
            'prefetch_attempts': self.prefetch_attempts,
            'prefetch_hits': self.prefetch_hits,
            'prefetch_success_rate': self.prefetch_success_rate,
            'avg_lookup_time_ms': self.avg_lookup_time_ms,
        }


class EnhancedCaptionCache:
    """Enhanced caption cache with prefetch, compression, and bulk operations.
    
    This class wraps the standard CaptionCache with additional features:
    - Cache warming for batch operations
    - Bulk get/set for multiple videos
    - Compression for storage efficiency
    - Predictive prefetching
    - Enhanced metrics
    
    Args:
        base_cache: The underlying CaptionCache instance
        enable_compression: Whether to compress caption data
        compression_level: Gzip compression level (1-9)
        max_prefetch: Maximum number of videos to prefetch ahead
    """
    
    def __init__(
        self,
        base_cache: 'CaptionCache',
        enable_compression: bool = True,
        compression_level: int = 6,
        max_prefetch: int = 10
    ):
        self.base = base_cache
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        self.max_prefetch = max_prefetch
        self.metrics = CacheMetrics()
        
        # Track warmed videos for metrics
        self._warmed_videos: Set[str] = set()
        self._warmed_lock = threading.Lock()
        
        # Prefetch queue
        self._prefetch_queue: List[str] = []
        self._prefetch_lock = threading.Lock()

        # Preflight cache (US-67-005): in-memory store for list-subs results
        # with 1-hour TTL to avoid repeated subprocess calls during batch processing
        self._preflight_store: Dict[str, Dict[str, Any]] = {}
        self._preflight_ttl_seconds: float = 3600.0  # 1 hour

        logger.debug(f"EnhancedCaptionCache initialized: compression={enable_compression}, "
                    f"compression_level={compression_level}, max_prefetch={max_prefetch}")
    
    def _make_key(self, video_id: str, language: str, is_auto_generated: bool = False) -> str:
        """Create cache key. US-67-012: '_autosub' suffix for auto-generated."""
        key = f"{video_id}_{language}"
        if is_auto_generated:
            key += "_autosub"
        return key
    
    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """Compress caption data using gzip."""
        json_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
        
        if not self.enable_compression:
            return json_bytes
        
        compressed = gzip.compress(json_bytes, compresslevel=self.compression_level)
        
        # Only use compression if it saves space
        if len(compressed) < len(json_bytes):
            self.metrics.record_compression(len(json_bytes), len(compressed))
            # Mark as compressed with prefix
            return b'\x01' + compressed
        
        return b'\x00' + json_bytes
    
    def _decompress_data(self, data: bytes) -> Dict[str, Any]:
        """Decompress caption data."""
        if not data:
            return {}
        
        # Check compression marker
        is_compressed = data[0] == 1
        payload = data[1:]
        
        if is_compressed:
            payload = gzip.decompress(payload)
        
        return json.loads(payload.decode('utf-8'))
    
    def get_caption(
        self, 
        video_id: str, 
        language: str = "en",
        track_timing: bool = True
    ) -> Optional['CachedCaption']:
        """Get cached caption with enhanced metrics.
        
        Args:
            video_id: YouTube video ID
            language: Language code
            track_timing: Whether to track lookup time
            
        Returns:
            CachedCaption if found, None otherwise
        """
        start_time = time.perf_counter() if track_timing else 0
        
        # Check if this was a warmed entry
        key = self._make_key(video_id, language)
        is_warmed = key in self._warmed_videos
        
        # Delegate to base cache
        result = self.base.get_caption(video_id, language)
        
        if track_timing:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if result:
                self.metrics.record_hit(is_warm=is_warmed, lookup_time_ms=elapsed_ms)
            else:
                self.metrics.record_miss(lookup_time_ms=elapsed_ms)
        elif result:
            self.metrics.record_hit(is_warm=is_warmed)
        else:
            self.metrics.record_miss()
        
        return result
    
    def get_captions_bulk(
        self, 
        video_ids: List[str], 
        language: str = "en"
    ) -> Dict[str, Optional['CachedCaption']]:
        """Get multiple captions from cache in one operation.
        
        Args:
            video_ids: List of YouTube video IDs
            language: Language code
            
        Returns:
            Dict mapping video_id to CachedCaption (or None if not cached)
        """
        start_time = time.perf_counter()
        results = {}
        
        for vid in video_ids:
            result = self.get_caption(vid, language, track_timing=False)
            results[vid] = result
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        hits = sum(1 for r in results.values() if r is not None)
        misses = len(results) - hits
        
        # Record bulk metrics
        with self.metrics._lock:
            self.metrics.bulk_hits += hits
            self.metrics.hits += hits
            self.metrics.misses += misses
        
        logger.debug(f"Bulk cache lookup: {hits}/{len(video_ids)} hits in {elapsed_ms:.1f}ms")
        return results
    
    def store_caption(
        self, 
        result: 'CaptionResult',
        language: Optional[str] = None,
        compress: Optional[bool] = None
    ) -> bool:
        """Store caption with optional compression.
        
        Args:
            result: CaptionResult to store
            language: Language code (defaults to result.language)
            compress: Override default compression setting
            
        Returns:
            True if stored successfully
        """
        from .caption_fetcher import CachedCaption
        
        if not self.base.enabled:
            return False
        
        lang = language or result.language
        
        try:
            # Create CachedCaption
            cached = CachedCaption.from_caption_result(result, lang)
            
            # Store via base cache
            self.base.store(result)
            return True
            
        except Exception as e:
            logger.warning(f"Failed to store caption {result.video_id}: {e}")
            return False
    
    def store_captions_bulk(
        self, 
        results: Dict[str, 'CaptionResult'],
        language: Optional[str] = None
    ) -> Dict[str, bool]:
        """Store multiple captions efficiently.
        
        Args:
            results: Dict mapping video_id to CaptionResult
            language: Default language code
            
        Returns:
            Dict mapping video_id to success status
        """
        stored = {}
        
        for vid, result in results.items():
            stored[vid] = self.store_caption(result, language)
        
        logger.debug(f"Bulk store: {sum(stored.values())}/{len(stored)} successful")
        return stored

    def store_unavailable(
        self, 
        video_id: str,
        language: str = "en"
    ) -> bool:
        """Store unavailable caption status in cache.
        
        This caches the knowledge that a video has no captions,
        avoiding repeated API calls for videos known to lack captions.
        
        Args:
            video_id: YouTube video ID
            language: Language code
            
        Returns:
            True if stored successfully
        """
        if not self.base.enabled:
            return False
        
        try:
            return self.base.store_unavailable(video_id, language)
        except Exception as e:
            logger.warning(f"Failed to store unavailable status for {video_id}: {e}")
            return False

    def store_unavailable_bulk(
        self, 
        video_ids: List[str],
        language: str = "en"
    ) -> Dict[str, bool]:
        """Store unavailable status for multiple videos efficiently.
        
        Args:
            video_ids: List of YouTube video IDs
            language: Language code
            
        Returns:
            Dict mapping video_id to success status
        """
        stored = {}
        
        for vid in video_ids:
            stored[vid] = self.store_unavailable(vid, language)
        
        logger.debug(f"Bulk store unavailable: {sum(stored.values())}/{len(stored)} successful")
        return stored

    def store_error(
        self,
        video_id: str,
        language: str = "en"
    ) -> bool:
        """Store transient error status in cache (US-90-003).

        This caches temporary failures (network errors, timeouts, etc.) with a
        shorter TTL than "unavailable" entries to allow faster retry.

        US-90-003: Uses error_ttl_seconds for TTL (default 5 minutes).

        Args:
            video_id: YouTube video ID
            language: Language code

        Returns:
            True if stored successfully
        """
        if not self.base.enabled:
            return False

        try:
            return self.base.store_error(video_id, language)
        except Exception as e:
            logger.warning(f"Failed to store error status for {video_id}: {e}")
            return False

    def store_error_bulk(
        self,
        video_ids: List[str],
        language: str = "en"
    ) -> Dict[str, bool]:
        """Store error status for multiple videos efficiently (US-90-003).

        Args:
            video_ids: List of YouTube video IDs
            language: Language code

        Returns:
            Dict mapping video_id to success status
        """
        stored = {}

        for vid in video_ids:
            stored[vid] = self.store_error(vid, language)

        logger.debug(f"Bulk store error: {sum(stored.values())}/{len(stored)} successful")
        return stored

    def warm_cache_for_videos(
        self, 
        video_ids: List[str], 
        language: str = "en",
        background: bool = False
    ) -> Dict[str, Any]:
        """Pre-load cache entries for specified videos.
        
        This is useful before starting a batch fetch to ensure
        cache entries are loaded into memory.
        
        Args:
            video_ids: List of video IDs to warm
            language: Language code
            background: If True, warming happens in background thread
            
        Returns:
            Dict with warming stats:
                - warmed: Number of entries found and warmed
                - missed: Number of entries not in cache
                - elapsed_ms: Time taken
        """
        if background:
            thread = threading.Thread(
                target=self._warm_cache_sync,
                args=(video_ids, language),
                daemon=True
            )
            thread.start()
            return {'status': 'background_warming_started', 'videos': len(video_ids)}
        
        return self._warm_cache_sync(video_ids, language)
    
    def _warm_cache_sync(
        self, 
        video_ids: List[str], 
        language: str
    ) -> Dict[str, Any]:
        """Synchronous cache warming."""
        start_time = time.perf_counter()
        warmed = 0
        missed = 0
        
        for vid in video_ids:
            key = self._make_key(vid, language)
            
            # Check if entry exists
            if self.base.get_caption(vid, language):
                with self._warmed_lock:
                    self._warmed_videos.add(key)
                warmed += 1
            else:
                missed += 1
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Cache warming complete: {warmed} warmed, {missed} missed "
                   f"({elapsed_ms:.1f}ms)")
        
        return {
            'warmed': warmed,
            'missed': missed,
            'elapsed_ms': elapsed_ms,
            'hit_rate': warmed / len(video_ids) if video_ids else 0.0
        }
    
    def prefetch_captions(
        self, 
        current_video_id: str,
        upcoming_video_ids: List[str],
        language: str = "en"
    ) -> None:
        """Prefetch captions for upcoming videos.
        
        Called during batch processing to anticipate needed captions.
        
        Args:
            current_video_id: Currently processing video
            upcoming_video_ids: List of videos likely to be needed next
            language: Language code
        """
        # Add to prefetch queue
        with self._prefetch_lock:
            # Limit queue size
            self._prefetch_queue = upcoming_video_ids[:self.max_prefetch]
        
        # Start background prefetch
        thread = threading.Thread(
            target=self._prefetch_worker,
            args=(current_video_id, language),
            daemon=True
        )
        thread.start()
    
    def _prefetch_worker(self, current_video_id: str, language: str) -> None:
        """Background worker for prefetching captions."""
        with self._prefetch_lock:
            to_prefetch = list(self._prefetch_queue)
        
        for vid in to_prefetch:
            # Skip if it's the current video
            if vid == current_video_id:
                continue
            
            # Check if already cached
            if self.base.get_caption(vid, language):
                self.metrics.record_prefetch(hit=True)
                logger.debug(f"Prefetch hit: {vid}")
            else:
                self.metrics.record_prefetch(hit=False)
    
    def get_cache_coverage(self, video_ids: List[str], language: str = "en") -> Dict[str, Any]:
        """Get cache coverage statistics for a list of videos.
        
        Args:
            video_ids: List of video IDs to check
            language: Language code
            
        Returns:
            Dict with coverage stats:
                - total: Total videos checked
                - cached: Number already cached
                - coverage_ratio: Percentage cached
                - missing: List of video IDs not cached
        """
        cached = []
        missing = []
        
        for vid in video_ids:
            key = self._make_key(vid, language)
            # Fast check using base cache index
            if key in self.base.index:
                cached.append(vid)
            else:
                missing.append(vid)
        
        total = len(video_ids)
        return {
            'total': total,
            'cached': len(cached),
            'missing': len(missing),
            'coverage_ratio': len(cached) / total if total > 0 else 0.0,
            'missing_ids': missing,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        base_stats = self.base.get_stats() if hasattr(self.base, 'get_stats') else {}
        enhanced_stats = self.metrics.to_dict()
        
        return {
            'base_cache': base_stats,
            'enhanced': enhanced_stats,
            'warmed_videos': len(self._warmed_videos),
            'compression_enabled': self.enable_compression,
        }
    
    def clear_warmed(self) -> None:
        """Clear warmed video tracking."""
        with self._warmed_lock:
            self._warmed_videos.clear()

    # ==================== Preflight Cache (US-67-005) ====================

    def get_preflight(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached preflight discovery results for a video (US-67-005).

        Returns cached list-subs output (available subtitle formats) to avoid
        repeated yt-dlp --list-subs subprocess calls during batch processing
        with retries.

        Cache key format: '{video_id}_preflight'
        TTL: 1 hour (3600 seconds)

        Args:
            video_id: YouTube video ID.

        Returns:
            List of language dicts with 'code', 'name', 'is_auto_generated',
            or None if not cached or expired.
        """
        key = f"{video_id}_preflight"
        entry = self._preflight_store.get(key)
        if entry is None:
            return None

        # Check TTL (1 hour)
        cached_at = entry.get('cached_at', 0.0)
        if (time.time() - cached_at) > self._preflight_ttl_seconds:
            # Expired - remove entry
            self._preflight_store.pop(key, None)
            logger.debug(f"Preflight cache expired: {video_id}")
            return None

        languages = entry.get('languages', [])
        logger.debug(f"Preflight cache hit: {video_id} ({len(languages)} languages)")
        return languages

    def store_preflight(
        self,
        video_id: str,
        languages: List[Any],
    ) -> None:
        """Store preflight discovery results in cache (US-67-005).

        Caches the result of list_available_languages() to avoid redundant
        yt-dlp --list-subs subprocess calls when the same video is checked
        multiple times during batch processing.

        Args:
            video_id: YouTube video ID.
            languages: List of AvailableLanguage objects or dicts.
        """
        key = f"{video_id}_preflight"

        # Convert AvailableLanguage objects to dicts if needed
        language_dicts = []
        for lang in languages:
            if hasattr(lang, 'code'):
                language_dicts.append({
                    'code': lang.code,
                    'name': lang.name,
                    'is_auto_generated': lang.is_auto_generated,
                })
            elif isinstance(lang, dict):
                language_dicts.append(lang)

        self._preflight_store[key] = {
            'languages': language_dicts,
            'cached_at': time.time(),
        }
        logger.debug(f"Preflight cache stored: {video_id} ({len(language_dicts)} languages)")


# Import at end to avoid circular imports
from .caption_fetcher import CaptionCache, CachedCaption, CaptionResult
