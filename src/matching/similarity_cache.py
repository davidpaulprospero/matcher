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
from typing import Any, Dict, Optional, Tuple

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


def clear_all_caches():
    """Clear all matching caches."""
    clear_similarity_cache()

    global _keyword_cache, _topic_penalty_cache
    with _cache_lock:
        if _keyword_cache:
            _keyword_cache.clear()
        if _topic_penalty_cache:
            _topic_penalty_cache.clear()

    logger.info("All matching caches cleared")
