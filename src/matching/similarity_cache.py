"""
Similarity cache for memoizing cosine similarity computations.

Provides a thread-safe cache for embedding pair similarities to avoid
redundant computations in tight matching loops across multiple strategy tracks.

Cache key: (hash(embedding_a), hash(embedding_b)) in canonical order
Cache value: float cosine similarity

Performance gains:
- Same embedding pairs computed once across 6+ strategy tracks
- O(1) lookup vs O(n) dot product computation
- Memory-efficient hash keys vs storing full embeddings

Created during performance optimization (Jan 2026).
"""

import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Global instance for module-level access
_similarity_cache: Optional['SimilarityCache'] = None
_cache_lock = threading.Lock()


class SimilarityCache:
    """
    Thread-safe memoization cache for cosine similarity computations.

    Uses embedding hashes as keys to avoid storing full vectors.
    Automatically clears when size exceeds max_size.
    """

    def __init__(self, max_size: int = 100_000):
        """
        Initialize similarity cache.

        Args:
            max_size: Maximum cache entries before automatic clearing.
                     Default 100K entries ≈ 4MB memory.
        """
        self._cache: Dict[Tuple[str, str], float] = {}
        self._lock = threading.RLock()
        self._max_size = max_size

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._creation_time = time.time()

    def get(self, emb_a_hash: str, emb_b_hash: str) -> Optional[float]:
        """
        Get cached similarity for embedding pair.

        Args:
            emb_a_hash: Hash of first embedding
            emb_b_hash: Hash of second embedding

        Returns:
            Cached similarity or None if not in cache.
        """
        # Canonical key ordering (smaller hash first)
        key = self._make_key(emb_a_hash, emb_b_hash)

        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, emb_a_hash: str, emb_b_hash: str, similarity: float):
        """
        Store similarity for embedding pair.

        Args:
            emb_a_hash: Hash of first embedding
            emb_b_hash: Hash of second embedding
            similarity: Cosine similarity value
        """
        key = self._make_key(emb_a_hash, emb_b_hash)

        with self._lock:
            # Check if cache needs clearing
            if len(self._cache) >= self._max_size:
                self._evict()

            self._cache[key] = similarity

    def _make_key(self, hash_a: str, hash_b: str) -> Tuple[str, str]:
        """Create canonical key with smaller hash first."""
        if hash_a <= hash_b:
            return (hash_a, hash_b)
        return (hash_b, hash_a)

    def _evict(self):
        """Clear half the cache when full (simple eviction strategy)."""
        # In tight loops, all entries are used equally, so just clear
        old_size = len(self._cache)
        self._cache.clear()
        self._evictions += 1
        logger.debug(f"SimilarityCache evicted {old_size} entries (eviction #{self._evictions})")

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            age = time.time() - self._creation_time

            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'age_seconds': age,
            }

    def log_stats(self):
        """Log cache statistics at INFO level."""
        stats = self.get_stats()
        logger.info(
            f"SimilarityCache: {stats['size']}/{stats['max_size']} entries, "
            f"hit_rate={stats['hit_rate']:.1%} ({stats['hits']} hits, {stats['misses']} misses), "
            f"evictions={stats['evictions']}"
        )


def get_similarity_cache(max_size: int = 100_000) -> SimilarityCache:
    """
    Get or create the global similarity cache instance.

    Args:
        max_size: Maximum cache entries (only used on first call)

    Returns:
        Global SimilarityCache instance.
    """
    global _similarity_cache

    with _cache_lock:
        if _similarity_cache is None:
            _similarity_cache = SimilarityCache(max_size)
            logger.info(f"Created SimilarityCache with max_size={max_size}")
        return _similarity_cache


def clear_similarity_cache():
    """Clear the global similarity cache."""
    global _similarity_cache

    with _cache_lock:
        if _similarity_cache is not None:
            _similarity_cache.log_stats()
            _similarity_cache.clear()
            logger.info("SimilarityCache cleared")


def embedding_hash(embedding: Any) -> str:
    """
    Compute stable hash for an embedding vector.

    Args:
        embedding: Embedding vector (list or numpy array)

    Returns:
        Hex string hash (first 16 chars of MD5)
    """
    try:
        import numpy as np

        # Convert to numpy if needed
        if isinstance(embedding, list):
            arr = np.array(embedding, dtype='float32')
        elif isinstance(embedding, np.ndarray):
            arr = embedding.astype('float32')
        else:
            # Fallback for unknown types
            return hashlib.md5(str(embedding).encode()).hexdigest()[:16]

        # Use tobytes for stable hashing
        return hashlib.md5(arr.tobytes()).hexdigest()[:16]

    except Exception:
        # Fallback if numpy conversion fails
        return hashlib.md5(str(embedding).encode()).hexdigest()[:16]


class KeywordExtractionCache:
    """
    Cache for keyword extraction from text.

    Avoids re-extracting keywords from the same voiceover text
    across multiple candidate comparisons.
    """

    def __init__(self, max_size: int = 10_000):
        self._cache: Dict[str, set] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, text_hash: str) -> Optional[set]:
        """Get cached keywords for text hash."""
        with self._lock:
            if text_hash in self._cache:
                self._hits += 1
                return self._cache[text_hash]
            self._misses += 1
            return None

    def put(self, text_hash: str, keywords: set):
        """Store keywords for text hash."""
        with self._lock:
            if len(self._cache) >= self._max_size:
                # Simple eviction: clear all
                self._cache.clear()
                logger.debug(f"KeywordExtractionCache cleared at {self._max_size} entries")
            self._cache[text_hash] = keywords

    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
            }


# Global keyword cache instance
_keyword_cache: Optional[KeywordExtractionCache] = None


def get_keyword_cache(max_size: int = 10_000) -> KeywordExtractionCache:
    """Get or create global keyword extraction cache."""
    global _keyword_cache

    with _cache_lock:
        if _keyword_cache is None:
            _keyword_cache = KeywordExtractionCache(max_size)
            logger.info(f"Created KeywordExtractionCache with max_size={max_size}")
        return _keyword_cache


def text_hash(text: str) -> str:
    """Compute hash for text content."""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]


class TopicPenaltyCache:
    """
    Cache for topic penalty computations.

    Key: (vo_topics_hash, video_topics_hash)
    Value: penalty float
    """

    def __init__(self, max_size: int = 50_000):
        self._cache: Dict[Tuple[str, str], float] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, vo_hash: str, video_hash: str) -> Optional[float]:
        """Get cached penalty."""
        key = (vo_hash, video_hash)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, vo_hash: str, video_hash: str, penalty: float):
        """Store penalty."""
        key = (vo_hash, video_hash)
        with self._lock:
            if len(self._cache) >= self._max_size:
                self._cache.clear()
                logger.debug(f"TopicPenaltyCache cleared at {self._max_size} entries")
            self._cache[key] = penalty

    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                'size': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
            }


# Global topic penalty cache
_topic_penalty_cache: Optional[TopicPenaltyCache] = None


def get_topic_penalty_cache(max_size: int = 50_000) -> TopicPenaltyCache:
    """Get or create global topic penalty cache."""
    global _topic_penalty_cache

    with _cache_lock:
        if _topic_penalty_cache is None:
            _topic_penalty_cache = TopicPenaltyCache(max_size)
            logger.info(f"Created TopicPenaltyCache with max_size={max_size}")
        return _topic_penalty_cache


def topics_hash(topics: list) -> str:
    """Compute hash for topics list."""
    if not topics:
        return "empty"
    sorted_topics = sorted(str(t).lower() for t in topics if t)
    return hashlib.md5("|".join(sorted_topics).encode()).hexdigest()[:16]


class VideoContextCache:
    """
    Cache for built video context strings (US-134-011).

    Avoids rebuilding the same video context from metadata (title, description,
    tags, chapters) across multiple segment comparisons. Uses TTL to expire
    stale entries.

    Key: video_id or hash of (title, description, tags, chapters)
    Value: Built context string
    """

    def __init__(self, max_size: int = 10_000, ttl_seconds: float = 3600.0):
        """
        Initialize video context cache.

        Args:
            max_size: Maximum cache entries before automatic clearing.
            ttl_seconds: Time-to-live for cache entries in seconds (default: 1 hour).
        """
        self._cache: Dict[str, Tuple[str, float]] = {}  # key -> (context_string, timestamp)
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        self._creation_time = time.time()

    def get(self, cache_key: str) -> Optional[str]:
        """
        Get cached video context.

        Args:
            cache_key: Cache key (video_id or hash)

        Returns:
            Cached context string or None if not in cache or expired.
        """
        with self._lock:
            if cache_key in self._cache:
                context, timestamp = self._cache[cache_key]
                # Check TTL
                if time.time() - timestamp < self._ttl_seconds:
                    self._hits += 1
                    return context
                # Entry expired
                del self._cache[cache_key]
                self._expirations += 1

            self._misses += 1
            return None

    def put(self, cache_key: str, context: str):
        """
        Store video context.

        Args:
            cache_key: Cache key (video_id or hash)
            context: Built context string
        """
        with self._lock:
            # Check if cache needs clearing
            if len(self._cache) >= self._max_size:
                self._evict_expired()

            if len(self._cache) >= self._max_size:
                # Still full after eviction - clear oldest half
                self._cache.clear()
                self._evictions += 1

            self._cache[cache_key] = (context, time.time())

    def _evict_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self._ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]
            self._expirations += 1

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            age = time.time() - self._creation_time

            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'ttl_seconds': self._ttl_seconds,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'expirations': self._expirations,
                'age_seconds': age,
            }

    def log_stats(self):
        """Log cache statistics at INFO level."""
        stats = self.get_stats()
        logger.info(
            f"VideoContextCache: {stats['size']}/{stats['max_size']} entries, "
            f"hit_rate={stats['hit_rate']:.1%} ({stats['hits']} hits, {stats['misses']} misses), "
            f"evictions={stats['evictions']}, expirations={stats['expirations']}"
        )


# Global video context cache instance
_video_context_cache: Optional[VideoContextCache] = None


def get_video_context_cache(
    max_size: int = 10_000,
    ttl_seconds: float = 3600.0
) -> VideoContextCache:
    """
    Get or create global video context cache.

    Args:
        max_size: Maximum cache entries (only used on first call)
        ttl_seconds: Time-to-live in seconds (only used on first call)

    Returns:
        Global VideoContextCache instance.
    """
    global _video_context_cache

    with _cache_lock:
        if _video_context_cache is None:
            _video_context_cache = VideoContextCache(max_size, ttl_seconds)
            logger.info(f"Created VideoContextCache with max_size={max_size}, ttl_seconds={ttl_seconds}")
        return _video_context_cache


def clear_video_context_cache():
    """Clear the global video context cache."""
    global _video_context_cache

    with _cache_lock:
        if _video_context_cache is not None:
            _video_context_cache.log_stats()
            _video_context_cache.clear()
            logger.info("VideoContextCache cleared")


def video_context_cache_key(
    video_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    chapters: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generate cache key for video context.

    If video_id is provided, use it as the key. Otherwise, generate a hash
    from the metadata content.

    Args:
        video_id: Video ID (preferred key)
        title: Video title
        description: Video description
        tags: Video tags
        chapters: Video chapters

    Returns:
        Cache key string
    """
    if video_id:
        return f"vid:{video_id}"

    # Generate hash from metadata content
    content_parts = []
    if title:
        content_parts.append(f"t:{title}")
    if description:
        content_parts.append(f"d:{description[:200]}")  # Limit description length
    if tags:
        sorted_tags = sorted(str(t).lower() for t in tags if t)
        content_parts.append(f"tags:{','.join(sorted_tags[:10])}")  # Limit tags
    if chapters:
        chapter_titles = [c.get('title', '') for c in chapters if c.get('title')]
        content_parts.append(f"ch:{','.join(chapter_titles[:5])}")  # Limit chapters

    if not content_parts:
        return "vid:empty"

    content = "|".join(content_parts)
    return f"meta:{hashlib.md5(content.encode()).hexdigest()[:16]}"


def log_all_cache_stats():
    """Log statistics for all caches."""
    if _similarity_cache:
        _similarity_cache.log_stats()

    if _keyword_cache:
        stats = _keyword_cache.get_stats()
        logger.info(
            f"KeywordExtractionCache: {stats['size']}/{stats['max_size']} entries, "
            f"hit_rate={stats['hit_rate']:.1%}"
        )

    if _topic_penalty_cache:
        stats = _topic_penalty_cache.get_stats()
        logger.info(
            f"TopicPenaltyCache: {stats['size']} entries, "
            f"hit_rate={stats['hit_rate']:.1%}"
        )

    # US-134-011: Video context cache stats
    if _video_context_cache:
        _video_context_cache.log_stats()


def clear_all_caches():
    """Clear all matching caches."""
    clear_similarity_cache()

    global _keyword_cache, _topic_penalty_cache
    with _cache_lock:
        if _keyword_cache:
            _keyword_cache.clear()
        if _topic_penalty_cache:
            _topic_penalty_cache.clear()

    # US-134-011: Clear video context cache between projects
    clear_video_context_cache()

    logger.info("All matching caches cleared")
