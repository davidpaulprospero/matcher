"""
Tests for src/matching/similarity_cache.py

Covers SimilarityCache, KeywordExtractionCache, TopicPenaltyCache,
global accessors, hash helpers, and thread safety.
"""

import hashlib
import threading
import time
from unittest.mock import patch

import pytest

from src.matching.similarity_cache import (
    KeywordExtractionCache,
    SimilarityCache,
    TopicPenaltyCache,
    clear_all_caches,
    clear_similarity_cache,
    embedding_hash,
    get_keyword_cache,
    get_similarity_cache,
    get_topic_penalty_cache,
    log_all_cache_stats,
    text_hash,
    topics_hash,
)


# ---------------------------------------------------------------------------
# SimilarityCache – basic get / put / stats
# ---------------------------------------------------------------------------

class TestSimilarityCacheBasic:
    """Core cache behaviour: store, retrieve, stats."""

    def test_put_and_get_returns_value(self):
        cache = SimilarityCache(max_size=100)
        cache.put("aaa", "bbb", 0.95)
        assert cache.get("aaa", "bbb") == 0.95

    def test_get_miss_returns_none(self):
        cache = SimilarityCache(max_size=100)
        assert cache.get("aaa", "bbb") is None

    def test_canonical_key_order(self):
        """(a, b) and (b, a) must resolve to the same entry."""
        cache = SimilarityCache(max_size=100)
        cache.put("zzz", "aaa", 0.75)
        assert cache.get("aaa", "zzz") == 0.75

    def test_identical_pair_consistent(self):
        """Same embedding pair always returns the same result."""
        cache = SimilarityCache(max_size=100)
        cache.put("h1", "h2", 0.42)
        for _ in range(10):
            assert cache.get("h1", "h2") == 0.42
            assert cache.get("h2", "h1") == 0.42

    def test_overwrite_value(self):
        cache = SimilarityCache(max_size=100)
        cache.put("a", "b", 0.1)
        cache.put("a", "b", 0.9)
        assert cache.get("a", "b") == 0.9

    def test_stats_initial(self):
        cache = SimilarityCache(max_size=50)
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 50
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["hit_rate"] == 0.0

    def test_stats_after_hits_and_misses(self):
        cache = SimilarityCache(max_size=100)
        cache.put("a", "b", 0.5)
        cache.get("a", "b")  # hit
        cache.get("a", "b")  # hit
        cache.get("x", "y")  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# SimilarityCache – eviction
# ---------------------------------------------------------------------------

class TestSimilarityCacheEviction:
    """Cache must not grow unbounded; eviction fires at max_size."""

    def test_eviction_at_max_size(self):
        cache = SimilarityCache(max_size=5)
        # Fill to max
        for i in range(5):
            cache.put(f"a{i}", f"b{i}", float(i))
        assert cache.get_stats()["size"] == 5

        # This put triggers eviction (cache clears then inserts new entry)
        cache.put("overflow_a", "overflow_b", 99.0)
        stats = cache.get_stats()
        assert stats["evictions"] == 1
        # After eviction the cache has only the new entry
        assert stats["size"] == 1
        assert cache.get("overflow_a", "overflow_b") == 99.0

    def test_old_entries_gone_after_eviction(self):
        cache = SimilarityCache(max_size=3)
        cache.put("k1", "k2", 0.1)
        cache.put("k3", "k4", 0.2)
        cache.put("k5", "k6", 0.3)
        # Trigger eviction
        cache.put("k7", "k8", 0.4)
        assert cache.get("k1", "k2") is None
        assert cache.get("k3", "k4") is None

    def test_multiple_evictions(self):
        cache = SimilarityCache(max_size=2)
        for cycle in range(4):
            cache.put(f"a{cycle}", f"b{cycle}", float(cycle))
            cache.put(f"c{cycle}", f"d{cycle}", float(cycle))
            # Third put in each cycle triggers eviction
            cache.put(f"e{cycle}", f"f{cycle}", float(cycle))
        assert cache.get_stats()["evictions"] >= 4


# ---------------------------------------------------------------------------
# SimilarityCache – clear
# ---------------------------------------------------------------------------

class TestSimilarityCacheClear:
    """Clearing resets stored entries."""

    def test_clear_removes_entries(self):
        cache = SimilarityCache(max_size=100)
        cache.put("a", "b", 0.5)
        cache.clear()
        assert cache.get("a", "b") is None
        assert cache.get_stats()["size"] == 0

    def test_clear_preserves_counters(self):
        """clear() only empties data; counters survive (by design)."""
        cache = SimilarityCache(max_size=100)
        cache.put("a", "b", 0.5)
        cache.get("a", "b")  # hit
        cache.clear()
        # hits counter is still 1 — clear doesn't reset stats
        assert cache.get_stats()["hits"] == 1


# ---------------------------------------------------------------------------
# SimilarityCache – thread safety
# ---------------------------------------------------------------------------

class TestSimilarityCacheThreadSafety:
    """Concurrent access must not corrupt state or raise."""

    def test_concurrent_puts_no_exception(self):
        cache = SimilarityCache(max_size=500)
        errors = []

        def writer(tid):
            try:
                for i in range(200):
                    cache.put(f"t{tid}_a{i}", f"t{tid}_b{i}", float(i))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Threads raised: {errors}"

    def test_concurrent_get_put_no_exception(self):
        cache = SimilarityCache(max_size=200)
        errors = []

        def mixed(tid):
            try:
                for i in range(200):
                    cache.put(f"m{tid}_{i}", f"n{tid}_{i}", 0.5)
                    cache.get(f"m{tid}_{i}", f"n{tid}_{i}")
                    cache.get("nonexistent", "key")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=mixed, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Threads raised: {errors}"

    def test_concurrent_eviction_no_corruption(self):
        """Small max_size forces many evictions under contention."""
        cache = SimilarityCache(max_size=10)
        errors = []

        def hammer(tid):
            try:
                for i in range(500):
                    cache.put(f"h{tid}_{i}", f"j{tid}_{i}", float(i))
                    val = cache.get(f"h{tid}_{i}", f"j{tid}_{i}")
                    # val may be None if another thread evicted, that's fine
                    if val is not None:
                        assert isinstance(val, float)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=hammer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Threads raised: {errors}"
        assert cache.get_stats()["evictions"] > 0

    def test_concurrent_clear_no_exception(self):
        cache = SimilarityCache(max_size=50)
        errors = []

        def writer(tid):
            try:
                for i in range(200):
                    cache.put(f"w{tid}_{i}", f"x{tid}_{i}", 0.1)
            except Exception as exc:
                errors.append(exc)

        def clearer():
            try:
                for _ in range(50):
                    cache.clear()
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        threads.append(threading.Thread(target=clearer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Threads raised: {errors}"


# ---------------------------------------------------------------------------
# SimilarityCache – log_stats
# ---------------------------------------------------------------------------

class TestSimilarityCacheLogStats:
    def test_log_stats_does_not_raise(self):
        cache = SimilarityCache(max_size=10)
        cache.put("a", "b", 0.5)
        cache.log_stats()  # should not raise


# ---------------------------------------------------------------------------
# KeywordExtractionCache
# ---------------------------------------------------------------------------

class TestKeywordExtractionCache:

    def test_put_and_get(self):
        cache = KeywordExtractionCache(max_size=100)
        cache.put("h1", {"word1", "word2"})
        assert cache.get("h1") == {"word1", "word2"}

    def test_miss(self):
        cache = KeywordExtractionCache(max_size=100)
        assert cache.get("missing") is None

    def test_eviction(self):
        cache = KeywordExtractionCache(max_size=3)
        for i in range(3):
            cache.put(f"k{i}", {f"w{i}"})
        # Trigger eviction
        cache.put("overflow", {"new"})
        assert cache.get("k0") is None
        assert cache.get("overflow") == {"new"}

    def test_clear(self):
        cache = KeywordExtractionCache(max_size=100)
        cache.put("h", {"a"})
        cache.clear()
        assert cache.get("h") is None

    def test_stats(self):
        cache = KeywordExtractionCache(max_size=100)
        cache.put("h", {"a"})
        cache.get("h")       # hit
        cache.get("miss")    # miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TopicPenaltyCache
# ---------------------------------------------------------------------------

class TestTopicPenaltyCache:

    def test_put_and_get(self):
        cache = TopicPenaltyCache(max_size=100)
        cache.put("vo1", "vid1", 0.3)
        assert cache.get("vo1", "vid1") == 0.3

    def test_miss(self):
        cache = TopicPenaltyCache(max_size=100)
        assert cache.get("vo1", "vid1") is None

    def test_eviction(self):
        cache = TopicPenaltyCache(max_size=2)
        cache.put("a", "b", 0.1)
        cache.put("c", "d", 0.2)
        cache.put("e", "f", 0.3)  # triggers eviction
        assert cache.get("a", "b") is None
        assert cache.get("e", "f") == 0.3

    def test_clear(self):
        cache = TopicPenaltyCache(max_size=100)
        cache.put("a", "b", 0.5)
        cache.clear()
        assert cache.get("a", "b") is None

    def test_stats(self):
        cache = TopicPenaltyCache(max_size=100)
        cache.put("a", "b", 0.1)
        cache.get("a", "b")  # hit
        cache.get("x", "y")  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

class TestHashHelpers:

    def test_embedding_hash_list(self):
        h = embedding_hash([1.0, 2.0, 3.0])
        assert isinstance(h, str)
        assert len(h) == 16

    def test_embedding_hash_numpy(self):
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0], dtype="float32")
        h = embedding_hash(arr)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_embedding_hash_deterministic(self):
        h1 = embedding_hash([1.0, 2.0, 3.0])
        h2 = embedding_hash([1.0, 2.0, 3.0])
        assert h1 == h2

    def test_embedding_hash_different_inputs(self):
        h1 = embedding_hash([1.0, 2.0, 3.0])
        h2 = embedding_hash([4.0, 5.0, 6.0])
        assert h1 != h2

    def test_embedding_hash_fallback_type(self):
        """Non-array type falls back to str-based hashing."""
        h = embedding_hash("not_an_embedding")
        assert isinstance(h, str) and len(h) == 16

    def test_text_hash_deterministic(self):
        assert text_hash("Hello World") == text_hash("hello world")

    def test_text_hash_strips(self):
        assert text_hash("  hi  ") == text_hash("hi")

    def test_topics_hash_empty(self):
        assert topics_hash([]) == "empty"

    def test_topics_hash_order_independent(self):
        assert topics_hash(["b", "a"]) == topics_hash(["a", "b"])

    def test_topics_hash_ignores_none(self):
        h1 = topics_hash(["a", None, "b"])
        h2 = topics_hash(["a", "b"])
        assert h1 == h2


# ---------------------------------------------------------------------------
# Global accessor functions
# ---------------------------------------------------------------------------

class TestGlobalAccessors:
    """Test module-level get_*_cache / clear_* functions."""

    def setup_method(self):
        """Reset global state before each test."""
        import src.matching.similarity_cache as mod
        with mod._cache_lock:
            mod._similarity_cache = None
            mod._keyword_cache = None
            mod._topic_penalty_cache = None

    def test_get_similarity_cache_creates_singleton(self):
        c1 = get_similarity_cache(max_size=50)
        c2 = get_similarity_cache(max_size=999)  # second call ignores max_size
        assert c1 is c2

    def test_clear_similarity_cache(self):
        cache = get_similarity_cache()
        cache.put("a", "b", 0.5)
        clear_similarity_cache()
        assert cache.get("a", "b") is None

    def test_get_keyword_cache_creates_singleton(self):
        c1 = get_keyword_cache()
        c2 = get_keyword_cache()
        assert c1 is c2

    def test_get_topic_penalty_cache_creates_singleton(self):
        c1 = get_topic_penalty_cache()
        c2 = get_topic_penalty_cache()
        assert c1 is c2

    def test_clear_all_caches(self):
        sc = get_similarity_cache()
        kc = get_keyword_cache()
        tc = get_topic_penalty_cache()
        sc.put("a", "b", 0.5)
        kc.put("h", {"w"})
        tc.put("v", "d", 0.1)
        clear_all_caches()
        assert sc.get("a", "b") is None
        assert kc.get("h") is None
        assert tc.get("v", "d") is None

    def test_log_all_cache_stats_no_raise(self):
        """Calling log_all when caches exist should not raise."""
        get_similarity_cache()
        get_keyword_cache()
        get_topic_penalty_cache()
        log_all_cache_stats()

    def test_log_all_cache_stats_no_caches(self):
        """Calling log_all when no caches created should not raise."""
        log_all_cache_stats()

    def test_clear_similarity_cache_when_none(self):
        """Clearing when no cache exists should not raise."""
        clear_similarity_cache()
