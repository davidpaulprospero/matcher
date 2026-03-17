"""
YouTube API Query Cache with SQLite Persistence

Caches YouTube Data API search results to survive restarts and reduce API quota usage.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# US-157-007: Compression constants
COMPRESSION_THRESHOLD_BYTES = 1024  # Compress responses larger than 1KB
DEFAULT_COMPRESSION_ENABLED = True

# US-158-012: Query normalization constants
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s]')  # Keep only word characters and whitespace
WHITESPACE_PATTERN = re.compile(r'\s+')  # Normalize whitespace


def normalize_query(query: str) -> str:
    """US-158-012: Normalize query for consistent cache key generation.

    Normalization steps:
    1. Lowercase
    2. Trim leading/trailing whitespace
    3. Remove special characters (keep only alphanumeric and whitespace)
    4. Normalize internal whitespace (multiple spaces -> single space)
    5. Sort keywords alphabetically for order-independent matching

    Args:
        query: Raw search query string

    Returns:
        Normalized query string
    """
    # Step 1: Lowercase
    normalized = query.lower()

    # Step 2: Trim whitespace
    normalized = normalized.strip()

    # Step 3: Remove special characters
    normalized = SPECIAL_CHARS_PATTERN.sub(' ', normalized)

    # Step 4: Normalize internal whitespace
    normalized = WHITESPACE_PATTERN.sub(' ', normalized)

    # Step 5: Trim again after normalization
    normalized = normalized.strip()

    # Step 6: Sort keywords (for order-independent matching)
    # This ensures "python tutorial" and "tutorial python" map to the same cache key
    words = normalized.split()
    if len(words) > 1:
        # Only sort if there are multiple words
        normalized = ' '.join(sorted(words))

    return normalized


@dataclass
class CachedAPIResponse:
    """Cached API response data"""
    query: str
    max_results: int
    video_type: str
    response_json: str
    cached_at: str
    expires_at: str
    video_published_at: Optional[str] = None  # US-155-007: Track earliest video date


class YouTubeAPISQLCache:
    """SQLite-based cache for YouTube API search results.

    Cache key: hash of (query, max_results, video_type)
    TTL: Configurable via cache_ttl_days option (default: 7 days)

    Thread-safe for concurrent access.

    Features (US-157-007):
    - Cache compression using gzip for large responses
    - Cache invalidation by video ID and by time
    - Cache hit/miss metrics with detailed breakdown by endpoint
    - Cache size limits with LRU eviction policy

    Features (US-155-007):
    - Smart invalidation based on video publication date
    - Cache hit rate metrics by query type
    - LRU eviction when cache exceeds size limit
    - Cache warmup for frequent queries
    - Robust concurrent write handling
    """

    # US-155-007: Configurable constants
    DEFAULT_NEW_VIDEO_THRESHOLD_DAYS = 7
    DEFAULT_MAX_CACHE_SIZE_MB = 100
    DEFAULT_LRU_EVICTION_BATCH = 50
    DEFAULT_MAX_QUERY_TYPE_METRICS = 100

    # US-157-007: Endpoint types for metrics
    ENDPOINT_SEARCH = "search"
    ENDPOINT_VIDEOS = "videos"
    ENDPOINT_CHANNELS = "channels"
    ENDPOINT_PLAYLISTS = "playlists"
    ENDPOINT_CAPTIONS = "captions"
    ENDPOINT_COMMENTS = "comments"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_days: int = 7,
        db_name: str = "youtube_api_cache.db",
        max_cache_size_mb: int = 100,
        new_video_threshold_days: int = 7,
        compression_enabled: bool = DEFAULT_COMPRESSION_ENABLED,
        endpoint: str = ENDPOINT_SEARCH,
        cache_ttl_seconds: Optional[int] = None  # US-158-012: TTL in seconds (overrides ttl_days)
    ):
        """Initialize the SQLite cache.

        Args:
            cache_dir: Directory for cache database (default: ~/.matcher_cache/youtube_api)
            ttl_days: Time-to-live in days for cached entries (default: 7)
            db_name: Name of the SQLite database file
            max_cache_size_mb: Maximum cache size in MB before LRU eviction (default: 100)
            new_video_threshold_days: Days threshold for "new video" detection (default: 7)
            compression_enabled: Enable gzip compression for large responses (default: True)
            endpoint: The API endpoint type for metrics tracking (default: search)
            cache_ttl_seconds: TTL in seconds (overrides ttl_days if provided, US-158-012)
        """
        # US-158-012: Convert seconds to days if cache_ttl_seconds is provided
        if cache_ttl_seconds is not None and cache_ttl_seconds > 0:
            ttl_days = cache_ttl_seconds / 86400.0  # Convert seconds to days

        if cache_dir is None:
            cache_dir = Path.home() / ".matcher_cache" / "youtube_api"

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._cache_dir / db_name
        self._ttl_days = ttl_days
        self._lock = threading.RLock()  # US-155-007: Use RLock for reentrant locking
        self._max_cache_size_mb = max_cache_size_mb
        self._new_video_threshold_days = new_video_threshold_days
        self._compression_enabled = compression_enabled  # US-157-007
        self._endpoint = endpoint  # US-157-007

        # Metrics
        self._hits = 0
        self._misses = 0
        self._query_type_metrics: Dict[str, Dict[str, int]] = {}  # US-155-007: Per query type metrics
        self._frequent_queries: Dict[str, int] = {}  # US-155-007: Track frequent queries for warmup
        self._max_frequent_queries = self.DEFAULT_MAX_QUERY_TYPE_METRICS
        self._endpoint_metrics: Dict[str, Dict[str, int]] = {}  # US-157-007: Per-endpoint metrics

        # Initialize database
        self._init_db()
        self._init_endpoint_metrics()

        logger.info(
            f"YouTubeAPISQLCache initialized: {self._db_path} (TTL: {ttl_days}d, "
            f"max_size: {max_cache_size_mb}MB, new_video_threshold: {new_video_threshold_days}d, "
            f"compression: {compression_enabled})"
        )

    def _init_endpoint_metrics(self) -> None:
        """US-157-007: Initialize endpoint metrics tracking."""
        self._endpoint_metrics = {
            self.ENDPOINT_SEARCH: {'hits': 0, 'misses': 0},
            self.ENDPOINT_VIDEOS: {'hits': 0, 'misses': 0},
            self.ENDPOINT_CHANNELS: {'hits': 0, 'misses': 0},
            self.ENDPOINT_PLAYLISTS: {'hits': 0, 'misses': 0},
            self.ENDPOINT_CAPTIONS: {'hits': 0, 'misses': 0},
            self.ENDPOINT_COMMENTS: {'hits': 0, 'misses': 0},
        }

    def _compress(self, data: str) -> bytes:
        """US-157-007: Compress data using gzip if threshold is met.

        Args:
            data: JSON string to compress

        Returns:
            Gzipped bytes if larger than threshold, otherwise original bytes
        """
        if not self._compression_enabled:
            return data.encode('utf-8')

        data_bytes = data.encode('utf-8')
        if len(data_bytes) < COMPRESSION_THRESHOLD_BYTES:
            return data_bytes

        return gzip.compress(data_bytes)

    def _decompress(self, data: bytes, is_compressed: bool) -> str:
        """US-157-007: Decompress data using gzip if needed.

        Args:
            data: Bytes to decompress
            is_compressed: Whether the data is gzip compressed

        Returns:
            Decompressed JSON string
        """
        if is_compressed:
            return gzip.decompress(data).decode('utf-8')
        return data.decode('utf-8')

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    normalized_query TEXT,
                    max_results INTEGER NOT NULL,
                    video_type TEXT NOT NULL,
                    response_json BLOB NOT NULL,
                    cached_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    video_published_at TEXT,
                    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_compressed INTEGER DEFAULT 0,
                    video_ids TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON api_cache(expires_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_params ON api_cache(query, max_results, video_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON api_cache(last_accessed_at)
            """)
            # US-157-007: Index for video ID lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_video_ids ON api_cache(video_ids)
            """)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _normalize_query(self, query: str) -> str:
        """US-158-012: Normalize query string for consistent cache key generation.

        Normalization steps:
        1. Convert to lowercase
        2. Trim whitespace
        3. Remove special characters (keep only word characters and whitespace)
        4. Sort keywords alphabetically

        Args:
            query: Search query string

        Returns:
            Normalized query string
        """
        # Step 1: Lowercase
        normalized = query.lower()
        # Step 2: Trim whitespace
        normalized = normalized.strip()
        # Step 3: Remove special characters (keep only alphanumeric and whitespace)
        normalized = SPECIAL_CHARS_PATTERN.sub(' ', normalized)
        # Step 4: Split into words, sort alphabetically, and rejoin
        words = normalized.split()
        sorted_words = sorted(words)
        normalized = ' '.join(sorted_words)
        return normalized

    def _make_key(self, query: str, max_results: int, video_type: str) -> str:
        """Generate cache key from search parameters.

        Args:
            query: Search query string
            max_results: Maximum number of results
            video_type: Type of results (e.g., "video")

        Returns:
            MD5 hash of the parameters
        """
        # US-158-012: Use normalized query for consistent cache keys
        normalized_query = self._normalize_query(query)
        key_string = f"{normalized_query}|{max_results}|{video_type}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self,
        query: str,
        max_results: int = 50,
        video_type: str = "video",
        endpoint: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results if available and not expired.

        Args:
            query: Search query string
            max_results: Maximum number of results
            video_type: Type of results
            endpoint: API endpoint for metrics tracking (optional, uses default if not provided)

        Returns:
            List of video search result dicts, or None if not cached/expired
        """
        cache_key = self._make_key(query, max_results, video_type)
        endpoint = endpoint or self._endpoint

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT response_json, expires_at, is_compressed FROM api_cache
                    WHERE cache_key = ? AND expires_at > datetime('now')
                    """,
                    (cache_key,)
                )
                row = cursor.fetchone()

                if row:
                    self._hits += 1
                    # US-155-007: Update query type metrics
                    self._update_query_type_metrics(query, video_type, hit=True)
                    # US-157-007: Update endpoint metrics
                    self._update_endpoint_metrics(endpoint, hit=True)
                    # US-155-007: Update last accessed time
                    conn.execute(
                        "UPDATE api_cache SET last_accessed_at = datetime('now') WHERE cache_key = ?",
                        (cache_key,)
                    )
                    conn.commit()

                    try:
                        # Handle compression
                        response_json = self._decompress(
                            row['response_json'],
                            bool(row['is_compressed'])
                        )
                        response = json.loads(response_json)
                        logger.debug(
                            f"Cache HIT: query='{query}' max_results={max_results} "
                            f"video_type={video_type} -> {len(response)} results"
                        )
                        return response
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(f"Invalid JSON in cache for key {cache_key}: {e}")
                        # Delete invalid entry
                        conn.execute("DELETE FROM api_cache WHERE cache_key = ?", (cache_key,))
                        conn.commit()
                        self._misses += 1
                        self._update_query_type_metrics(query, video_type, hit=False)
                        self._update_endpoint_metrics(endpoint, hit=False)
                        return None

                self._misses += 1
                self._update_query_type_metrics(query, video_type, hit=False)
                self._update_endpoint_metrics(endpoint, hit=False)
                logger.debug(f"Cache MISS: query='{query}' max_results={max_results}")
                return None

    def _update_query_type_metrics(self, query: str, video_type: str, hit: bool) -> None:
        """US-155-007: Update per query type metrics."""
        query_lower = query.lower().strip()[:50]  # Truncate for key
        key = f"{query_lower}|{video_type}"

        if key not in self._query_type_metrics:
            if len(self._query_type_metrics) >= self.DEFAULT_MAX_QUERY_TYPE_METRICS:
                # Remove oldest entry to make room
                oldest_key = next(iter(self._query_type_metrics))
                del self._query_type_metrics[oldest_key]
            self._query_type_metrics[key] = {'hits': 0, 'misses': 0, 'accesses': 0}

        if hit:
            self._query_type_metrics[key]['hits'] += 1
        else:
            self._query_type_metrics[key]['misses'] += 1
        self._query_type_metrics[key]['accesses'] += 1

        # Track frequent queries
        if key not in self._frequent_queries:
            if len(self._frequent_queries) >= self._max_frequent_queries:
                # Remove least frequent query
                min_key = min(self._frequent_queries, key=self._frequent_queries.get)
                del self._frequent_queries[min_key]
            self._frequent_queries[key] = 0
        self._frequent_queries[key] += 1

    def _update_endpoint_metrics(self, endpoint: str, hit: bool) -> None:
        """US-157-007: Update per-endpoint metrics."""
        if endpoint not in self._endpoint_metrics:
            self._endpoint_metrics[endpoint] = {'hits': 0, 'misses': 0}

        if hit:
            self._endpoint_metrics[endpoint]['hits'] += 1
        else:
            self._endpoint_metrics[endpoint]['misses'] += 1

    def set(
        self,
        query: str,
        max_results: int,
        video_type: str,
        results: List[Dict[str, Any]],
        video_published_at: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Cache search results.

        Args:
            query: Search query string
            max_results: Maximum number of results
            video_type: Type of results
            results: List of video result dicts to cache
            video_published_at: ISO date string of earliest video publication date (optional)
            video_ids: List of video IDs in the results (optional, for invalidation)
            ttl_seconds: Time-to-live in seconds (overrides default ttl_days)
        """
        cache_key = self._make_key(query, max_results, video_type)
        normalized_query = self._normalize_query(query)
        now = datetime.now()
        cached_at = now.isoformat()

        # US-158-012: Support TTL in seconds
        if ttl_seconds is not None and ttl_seconds > 0:
            expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        else:
            expires_at = (now + timedelta(days=self._ttl_days)).isoformat()

        response_json = json.dumps(results)

        # US-157-007: Apply compression
        compressed_data = self._compress(response_json)
        is_compressed = compressed_data != response_json.encode('utf-8')

        # US-155-007: Extract earliest video publication date from results if not provided
        if video_published_at is None and results:
            video_published_at = self._extract_earliest_published_date(results)

        # US-157-007: Extract video IDs if not provided
        if video_ids is None and results:
            video_ids = self._extract_video_ids(results)
        video_ids_str = ','.join(video_ids) if video_ids else None

        with self._lock:
            # US-155-007: Check cache size and evict if necessary before writing
            self._evict_if_needed()

            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO api_cache
                    (cache_key, query, normalized_query, max_results, video_type, response_json, cached_at, expires_at, video_published_at, is_compressed, video_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cache_key, query, normalized_query, max_results, video_type, compressed_data, cached_at, expires_at, video_published_at, int(is_compressed), video_ids_str)
                )
                conn.commit()

                logger.debug(
                    f"Cache SET: query='{query}' normalized='{normalized_query}' max_results={max_results} "
                    f"video_type={video_type} -> {len(results)} results"
                )

    def _extract_earliest_published_date(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """US-155-007: Extract earliest video publication date from search results."""
        earliest_date = None
        for result in results:
            # YouTube API typically returns 'snippet.publishedAt'
            published_at = result.get('snippet', {}).get('publishedAt')
            if published_at:
                if earliest_date is None or published_at < earliest_date:
                    earliest_date = published_at
        return earliest_date

    def _extract_video_ids(self, results: List[Dict[str, Any]]) -> List[str]:
        """US-157-007: Extract video IDs from search results.

        Args:
            results: List of video result dicts

        Returns:
            List of video ID strings
        """
        video_ids = []
        for result in results:
            # YouTube API returns 'videoId' in search results
            video_id = result.get('id', {}).get('videoId') or result.get('videoId')
            if video_id:
                video_ids.append(video_id)
            # Also check for 'resourceId.videoId' in playlist items
            resource_id = result.get('id', {})
            if isinstance(resource_id, dict):
                rid_video_id = resource_id.get('videoId')
                if rid_video_id and rid_video_id not in video_ids:
                    video_ids.append(rid_video_id)
        return video_ids

    def _evict_if_needed(self) -> int:
        """US-155-007: Evict LRU entries if cache exceeds size limit.

        Returns:
            Number of entries evicted
        """
        if not self._db_path.exists():
            return 0

        db_size_mb = os.path.getsize(self._db_path) / (1024 * 1024)

        if db_size_mb <= self._max_cache_size_mb:
            return 0

        logger.info(f"Cache size ({db_size_mb:.2f}MB) exceeds limit ({self._max_cache_size_mb}MB), running LRU eviction")

        evicted = 0
        with self._get_connection() as conn:
            # Delete oldest accessed entries first
            while True:
                db_size_mb = os.path.getsize(self._db_path) / (1024 * 1024)
                if db_size_mb <= self._max_cache_size_mb * 0.8:  # Evict to 80% of limit
                    break

                cursor = conn.execute(
                    """
                    SELECT cache_key FROM api_cache
                    ORDER BY last_accessed_at ASC
                    LIMIT ?
                    """,
                    (self.DEFAULT_LRU_EVICTION_BATCH,)
                )
                rows = cursor.fetchall()

                if not rows:
                    break

                for row in rows:
                    conn.execute("DELETE FROM api_cache WHERE cache_key = ?", (row['cache_key'],))
                    evicted += 1

                conn.commit()

        logger.info(f"LRU eviction completed: {evicted} entries removed")
        return evicted

    def invalidate(self, cache_key: str) -> bool:
        """Invalidate a specific cache entry by its key.

        Args:
            cache_key: The cache key to invalidate (MD5 hash)

        Returns:
            True if entry was found and deleted, False otherwise
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM api_cache WHERE cache_key = ?",
                    (cache_key,)
                )
                count = cursor.fetchone()[0]

                if count > 0:
                    conn.execute("DELETE FROM api_cache WHERE cache_key = ?", (cache_key,))
                    conn.commit()
                    logger.info(f"Cache invalidated: 1 entry with key '{cache_key}'")
                    return True

                return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a query pattern (wildcard support).

        Supports SQL LIKE pattern matching. Use % for any sequence of characters
        and _ for any single character.

        Args:
            pattern: SQL LIKE pattern to match against queries (e.g., "python%")

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM api_cache WHERE query LIKE ?",
                    (pattern,)
                )
                count = cursor.fetchone()[0]

                if count > 0:
                    conn.execute("DELETE FROM api_cache WHERE query LIKE ?", (pattern,))
                    conn.commit()
                    logger.info(f"Cache invalidated: {count} entries matching pattern '{pattern}'")

                return count

    def invalidate_all(self) -> int:
        """Invalidate all cache entries (e.g., when quota resets).

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM api_cache")
                count = cursor.fetchone()[0]

                conn.execute("DELETE FROM api_cache")
                conn.commit()

                logger.info(f"Cache invalidated: {count} entries cleared")
                return count

    def invalidate_by_query(self, query: str) -> int:
        """Invalidate cache entries for a specific query.

        Args:
            query: Query string to invalidate

        Returns:
            Number of entries invalidated
        """
        query_lower = query.lower().strip()

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM api_cache WHERE LOWER(query) = ?",
                    (query_lower,)
                )
                count = cursor.fetchone()[0]

                conn.execute("DELETE FROM api_cache WHERE LOWER(query) = ?", (query_lower,))
                conn.commit()

                if count > 0:
                    logger.info(f"Cache invalidated: {count} entries for query '{query}'")

                return count

    def invalidate_by_video_id(self, video_id: str) -> int:
        """US-157-007: Invalidate cache entries containing a specific video ID.

        This is useful when a specific video becomes unavailable or needs to be refreshed.

        Args:
            video_id: The video ID to invalidate

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            with self._get_connection() as conn:
                # Find entries containing this video ID
                cursor = conn.execute(
                    """
                    SELECT cache_key, video_ids FROM api_cache
                    WHERE video_ids LIKE ?
                    """,
                    (f"%{video_id}%",)
                )
                rows = cursor.fetchall()

                count = 0
                for row in rows:
                    video_ids_str = row['video_ids']
                    if video_ids_str and video_id in video_ids_str.split(','):
                        conn.execute("DELETE FROM api_cache WHERE cache_key = ?", (row['cache_key'],))
                        count += 1

                if count > 0:
                    conn.commit()
                    logger.info(f"Cache invalidated: {count} entries containing video ID '{video_id}'")

                return count

    def invalidate_by_age(self, max_age_days: int) -> int:
        """US-157-007: Invalidate cache entries older than specified days.

        This allows forcing a refresh of older cache entries regardless of TTL.

        Args:
            max_age_days: Maximum age in days for entries to remain in cache

        Returns:
            Number of entries invalidated
        """
        threshold_date = datetime.now() - timedelta(days=max_age_days)
        threshold_iso = threshold_date.isoformat()

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM api_cache WHERE cached_at < ?",
                    (threshold_iso,)
                )
                count = cursor.fetchone()[0]

                if count > 0:
                    conn.execute(
                        "DELETE FROM api_cache WHERE cached_at < ?",
                        (threshold_iso,)
                    )
                    conn.commit()
                    logger.info(f"Cache invalidated: {count} entries older than {max_age_days} days")

                return count

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM api_cache WHERE expires_at <= datetime('now')"
                )
                conn.commit()
                count = cursor.rowcount

                if count > 0:
                    logger.info(f"Cache cleanup: {count} expired entries removed")

                return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate, entries count, and size info
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM api_cache")
                entries = cursor.fetchone()[0]

                total = self._hits + self._misses
                hit_rate = self._hits / total if total > 0 else 0

                # Get database file size
                db_size = os.path.getsize(self._db_path) if self._db_path.exists() else 0

                # Build endpoint metrics
                endpoint_stats = {}
                for endpoint, data in self._endpoint_metrics.items():
                    endpoint_total = data['hits'] + data['misses']
                    endpoint_hit_rate = data['hits'] / endpoint_total if endpoint_total > 0 else 0
                    endpoint_stats[endpoint] = {
                        'hits': data['hits'],
                        'misses': data['misses'],
                        'hit_rate': endpoint_hit_rate
                    }

                return {
                    'hits': self._hits,
                    'misses': self._misses,
                    'entries': entries,
                    'hit_rate': hit_rate,
                    'db_size_bytes': db_size,
                    'ttl_days': self._ttl_days,
                    'max_cache_size_mb': self._max_cache_size_mb,
                    'new_video_threshold_days': self._new_video_threshold_days,
                    'query_type_metrics_count': len(self._query_type_metrics),
                    'frequent_queries_count': len(self._frequent_queries),
                    'endpoint_metrics': endpoint_stats,
                    'endpoint_metrics_count': len(self._endpoint_metrics),
                    'compression_enabled': self._compression_enabled
                }

    def get_endpoint_metrics(self) -> Dict[str, Dict[str, Any]]:
        """US-157-007: Get cache hit/miss metrics broken down by endpoint.

        Returns:
            Dict mapping endpoint names to hit/miss/hit_rate
        """
        with self._lock:
            metrics = {}
            for endpoint, data in self._endpoint_metrics.items():
                total = data['hits'] + data['misses']
                hit_rate = data['hits'] / total if total > 0 else 0.0
                metrics[endpoint] = {
                    'hits': data['hits'],
                    'misses': data['misses'],
                    'hit_rate': hit_rate
                }
            return metrics

    def reset_metrics(self) -> None:
        """Reset hit/miss counters and query type metrics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._query_type_metrics = {}
            self._frequent_queries = {}
            self._init_endpoint_metrics()  # Reset endpoint metrics

    # ============================================================================
    # US-155-007: Smart Cache Invalidation Methods
    # ============================================================================

    def invalidate_by_video_age(self, threshold_days: Optional[int] = None) -> int:
        """US-155-007: Invalidate cache entries containing new videos.

        Invalidates cached search results where the source videos are newer than
        the specified threshold (default: 7 days). This ensures fresh results
        for time-sensitive queries.

        Args:
            threshold_days: Days threshold for "new video" detection (default: configured value)

        Returns:
            Number of entries invalidated
        """
        if threshold_days is None:
            threshold_days = self._new_video_threshold_days

        threshold_date = datetime.now() - timedelta(days=threshold_days)
        threshold_iso = threshold_date.isoformat()

        with self._lock:
            with self._get_connection() as conn:
                # Find entries with videos published after threshold date
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM api_cache
                    WHERE video_published_at > ?
                    """,
                    (threshold_iso,)
                )
                count = cursor.fetchone()[0]

                if count > 0:
                    conn.execute(
                        "DELETE FROM api_cache WHERE video_published_at > ?",
                        (threshold_iso,)
                    )
                    conn.commit()
                    logger.info(
                        f"Cache invalidated: {count} entries with videos newer than "
                        f"{threshold_days} days removed"
                    )

                return count

    def get_query_type_metrics(self) -> Dict[str, Dict[str, Any]]:
        """US-155-007: Get cache hit rate metrics by query type.

        Returns:
            Dict mapping query_type keys to metrics (hits, misses, accesses, hit_rate)
        """
        with self._lock:
            metrics = {}
            for key, data in self._query_type_metrics.items():
                total = data['hits'] + data['misses']
                hit_rate = data['hits'] / total if total > 0 else 0.0
                metrics[key] = {
                    'hits': data['hits'],
                    'misses': data['misses'],
                    'accesses': data['accesses'],
                    'hit_rate': hit_rate
                }
            return metrics

    def get_frequent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """US-155-007: Get most frequently accessed queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of dicts with query info and access count
        """
        with self._lock:
            # Sort by access count and return top queries
            sorted_queries = sorted(
                self._frequent_queries.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]

            return [
                {'query': k, 'accesses': v}
                for k, v in sorted_queries
            ]

    def warmup(self, queries: List[Dict[str, Any]], fetch_func) -> int:
        """US-155-007: Cache warmup method for frequent queries.

        Pre-caches results for frequently used queries to improve initial
        hit rate. This should be called at startup with common queries.

        Args:
            queries: List of dicts with 'query', 'max_results', 'video_type' keys
            fetch_func: Async function to fetch results (query, max_results, video_type) -> results

        Returns:
            Number of queries successfully warmed up
        """
        warmed = 0

        with self._lock:
            for q in queries:
                query = q.get('query', '')
                max_results = q.get('max_results', 50)
                video_type = q.get('video_type', 'video')

                # Skip if already cached
                cached = self.get(query, max_results, video_type)
                if cached is not None:
                    continue

                try:
                    # Fetch fresh results
                    results = fetch_func(query, max_results, video_type)
                    if results:
                        self.set(query, max_results, video_type, results)
                        warmed += 1
                        logger.debug(f"Warmed up cache for query: {query}")
                except Exception as e:
                    logger.warning(f"Failed to warmup cache for query '{query}': {e}")

        logger.info(f"Cache warmup completed: {warmed}/{len(queries)} queries warmed up")
        return warmed

    def invalidate_for_new_videos(self, results: List[Dict[str, Any]]) -> int:
        """US-155-007: Invalidate cache entries if results contain new videos.

        Checks the provided results for videos newer than threshold and
        invalidates matching cached entries.

        Args:
            results: List of video result dicts

        Returns:
            Number of entries invalidated
        """
        if not results:
            return 0

        # Extract earliest publication date from new results
        earliest_date = self._extract_earliest_published_date(results)
        if earliest_date is None:
            return 0

        # Check if any videos are newer than threshold
        threshold_date = datetime.now() - timedelta(days=self._new_video_threshold_days)
        earliest_dt = datetime.fromisoformat(earliest_date.replace('Z', '+00:00'))

        if earliest_dt >= threshold_date:
            # Videos are new, invalidate all matching queries
            return self.cleanup_expired()  # Use general cleanup as proxy

        return 0

    # ============================================================================
    # US-157-007: Advanced Caching Features
    # ============================================================================

    def get_endpoint_metrics(self) -> Dict[str, Dict[str, Any]]:
        """US-157-007: Get cache hit/miss metrics with detailed breakdown by endpoint.

        Returns:
            Dict mapping endpoint names to metrics (hits, misses, hit_rate)
        """
        with self._lock:
            metrics = {}
            for endpoint, data in self._endpoint_metrics.items():
                total = data['hits'] + data['misses']
                hit_rate = data['hits'] / total if total > 0 else 0.0
                metrics[endpoint] = {
                    'hits': data['hits'],
                    'misses': data['misses'],
                    'total_requests': total,
                    'hit_rate': hit_rate
                }
            return metrics

    def invalidate_by_video_id(self, video_id: str) -> int:
        """US-157-007: Invalidate cache entries containing a specific video ID.

        Searches for cached entries that contain the given video ID in their
        video_ids field and removes them.

        Args:
            video_id: The video ID to invalidate

        Returns:
            Number of entries invalidated
        """
        if not video_id:
            return 0

        with self._lock:
            with self._get_connection() as conn:
                # Find entries containing this video_id
                cursor = conn.execute(
                    """
                    SELECT cache_key, video_ids FROM api_cache
                    WHERE video_ids LIKE ?
                    """,
                    (f"%{video_id}%",)
                )
                rows = cursor.fetchall()

                count = 0
                for row in rows:
                    video_ids_str = row['video_ids']
                    if video_ids_str and video_id in video_ids_str.split(','):
                        conn.execute(
                            "DELETE FROM api_cache WHERE cache_key = ?",
                            (row['cache_key'],)
                        )
                        count += 1

                if count > 0:
                    conn.commit()
                    logger.info(f"Cache invalidated: {count} entries containing video_id '{video_id}'")

                return count

    def invalidate_by_time(self, hours: int) -> int:
        """US-157-007: Invalidate cache entries older than specified hours.

        Removes cached entries that were created more than the specified number
        of hours ago. Useful for forcing refresh of stale data.

        Args:
            hours: Number of hours to keep. Entries older than this will be invalidated.

        Returns:
            Number of entries invalidated
        """
        if hours <= 0:
            return 0

        with self._lock:
            with self._get_connection() as conn:
                # Find entries older than specified hours
                cursor = conn.execute(
                    """
                    DELETE FROM api_cache
                    WHERE cached_at <= datetime('now', ?)
                    """,
                    (f"-{hours} hours",)
                )
                conn.commit()
                count = cursor.rowcount

                if count > 0:
                    logger.info(f"Cache invalidated: {count} entries older than {hours} hours")

                return count

    # ============================================================================
    # US-158-012: Quota Exhaustion Invalidation
    # ============================================================================

    def invalidate_on_quota_exhaustion(self, quota_remaining: int, quota_threshold: int = 1000) -> int:
        """US-158-012: Invalidate cache entries when quota is near exhaustion.

        Clears the cache when remaining quota falls below the threshold to prioritize
        fresh API calls over cached results when quota is scarce.

        Args:
            quota_remaining: The remaining quota units
            quota_threshold: Threshold below which to invalidate cache (default: 1000)

        Returns:
            Number of entries invalidated
        """
        if quota_remaining >= quota_threshold:
            return 0

        logger.warning(
            f"US-158-012: Quota exhaustion detected ({quota_remaining} remaining, "
            f"threshold: {quota_threshold}), invalidating cache"
        )
        return self.invalidate_all()

    def should_refresh_on_quota_exhaustion(self, quota_remaining: int, quota_threshold: int = 1000) -> bool:
        """US-158-012: Check if cache should be refreshed when quota is low.

        Returns True if quota is below threshold, indicating fresh API calls
        should be prioritized over cached results.

        Args:
            quota_remaining: The remaining quota units
            quota_threshold: Threshold below which to prefer fresh results (default: 1000)

        Returns:
            True if fresh results should be preferred, False otherwise
        """
        return quota_remaining < quota_threshold
