"""
Tests for downloader cache modules: LLMFilterCache and SearchResultsCache.

US-47-007: Add unit tests for downloader cache modules.
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from src.cache.base import CacheEntry
from src.downloader.llm_filter_cache import LLMFilterCache, CachedLLMFilterResult
from src.downloader.search_cache import SearchResultsCache, CachedSearchResult


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def llm_cache(tmp_path):
    """LLMFilterCache backed by a temp directory."""
    return LLMFilterCache(cache_dir=tmp_path / "llm_filter", ttl_days=7)


@pytest.fixture
def search_cache(tmp_path):
    """SearchResultsCache backed by a temp directory."""
    return SearchResultsCache(cache_dir=tmp_path / "search", ttl_hours=24)


def _make_videos(*ids):
    """Helper to build a list of video metadata dicts."""
    return [{"id": vid, "title": f"Video {vid}"} for vid in ids]


# ============================================================
# LLMFilterCache — set / get / miss
# ============================================================

class TestLLMFilterCacheSetGet:

    def test_set_then_get_returns_approved_videos(self, llm_cache):
        videos = _make_videos("v1", "v2", "v3")
        approved = [{"id": "v1", "relevance": 0.9}]

        llm_cache.set_filter_result("nature documentary", videos, approved)
        result = llm_cache.get_filter_result("nature documentary", videos)

        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "v1"

    def test_cache_miss_returns_none(self, llm_cache):
        videos = _make_videos("v1")
        assert llm_cache.get_filter_result("unknown keyword", videos) is None

    def test_different_keyword_is_separate_entry(self, llm_cache):
        videos = _make_videos("v1", "v2")
        llm_cache.set_filter_result("cats", videos, [{"id": "v1"}])
        llm_cache.set_filter_result("dogs", videos, [{"id": "v2"}])

        assert llm_cache.get_filter_result("cats", videos)[0]["id"] == "v1"
        assert llm_cache.get_filter_result("dogs", videos)[0]["id"] == "v2"

    def test_different_video_set_is_separate_entry(self, llm_cache):
        vids_a = _make_videos("v1", "v2")
        vids_b = _make_videos("v3", "v4")
        llm_cache.set_filter_result("nature", vids_a, [{"id": "v1"}])
        llm_cache.set_filter_result("nature", vids_b, [{"id": "v3"}])

        assert llm_cache.get_filter_result("nature", vids_a)[0]["id"] == "v1"
        assert llm_cache.get_filter_result("nature", vids_b)[0]["id"] == "v3"

    def test_keyword_case_insensitive(self, llm_cache):
        videos = _make_videos("v1")
        llm_cache.set_filter_result("Nature", videos, [{"id": "v1"}])
        result = llm_cache.get_filter_result("nature", videos)
        assert result is not None

    def test_empty_approved_list_cached(self, llm_cache):
        videos = _make_videos("v1", "v2")
        llm_cache.set_filter_result("irrelevant", videos, [])
        result = llm_cache.get_filter_result("irrelevant", videos)
        assert result == []


# ============================================================
# LLMFilterCache — expiry
# ============================================================

class TestLLMFilterCacheExpiry:

    def test_expired_entry_returns_none(self, tmp_path):
        cache = LLMFilterCache(cache_dir=tmp_path / "llm_exp", ttl_days=1)
        videos = _make_videos("v1")
        cache.set_filter_result("test", videos, [{"id": "v1"}])

        # Manually backdate the cached_at timestamp
        for key in cache.index:
            cache.index[key]["cached_at"] = time.time() - (2 * 86400)  # 2 days ago

        result = cache.get_filter_result("test", videos)
        assert result is None

    def test_non_expired_entry_still_valid(self, tmp_path):
        cache = LLMFilterCache(cache_dir=tmp_path / "llm_fresh", ttl_days=7)
        videos = _make_videos("v1")
        cache.set_filter_result("test", videos, [{"id": "v1"}])

        result = cache.get_filter_result("test", videos)
        assert result is not None

    def test_zero_ttl_never_expires(self, tmp_path):
        cache = LLMFilterCache(cache_dir=tmp_path / "llm_noexpire", ttl_days=0)
        videos = _make_videos("v1")
        cache.set_filter_result("test", videos, [{"id": "v1"}])

        # Backdate far in the past
        for key in cache.index:
            cache.index[key]["cached_at"] = 1000000.0

        result = cache.get_filter_result("test", videos)
        assert result is not None


# ============================================================
# LLMFilterCache — invalidation
# ============================================================

class TestLLMFilterCacheInvalidation:

    def test_invalidate_keyword_removes_entries(self, llm_cache):
        vids_a = _make_videos("v1", "v2")
        vids_b = _make_videos("v3", "v4")
        llm_cache.set_filter_result("nature", vids_a, [{"id": "v1"}])
        llm_cache.set_filter_result("nature", vids_b, [{"id": "v3"}])
        llm_cache.set_filter_result("cats", vids_a, [{"id": "v2"}])

        removed = llm_cache.invalidate_keyword("nature")
        assert removed == 2

        assert llm_cache.get_filter_result("nature", vids_a) is None
        assert llm_cache.get_filter_result("nature", vids_b) is None
        # cats entry still present
        assert llm_cache.get_filter_result("cats", vids_a) is not None

    def test_invalidate_nonexistent_keyword_returns_zero(self, llm_cache):
        assert llm_cache.invalidate_keyword("nonexistent") == 0

    def test_clear_removes_all(self, llm_cache):
        videos = _make_videos("v1")
        llm_cache.set_filter_result("a", videos, [])
        llm_cache.set_filter_result("b", videos, [])
        llm_cache.clear()
        assert llm_cache._count_entries() == 0


# ============================================================
# SearchResultsCache — set / get / miss
# ============================================================

class TestSearchCacheSetGet:

    def test_set_then_get_returns_videos(self, search_cache):
        videos = _make_videos("v1", "v2")
        search_cache.set_search_result("cats", "short", 10, videos)
        result = search_cache.get_search_result("cats", "short", 10)

        assert result is not None
        assert len(result) == 2
        assert result[0]["id"] == "v1"

    def test_cache_miss_returns_none(self, search_cache):
        assert search_cache.get_search_result("unknown", "short", 10) is None

    def test_different_tier_is_separate_entry(self, search_cache):
        short_vids = _make_videos("s1")
        long_vids = _make_videos("l1")
        search_cache.set_search_result("cats", "short", 10, short_vids)
        search_cache.set_search_result("cats", "long", 10, long_vids)

        assert search_cache.get_search_result("cats", "short", 10)[0]["id"] == "s1"
        assert search_cache.get_search_result("cats", "long", 10)[0]["id"] == "l1"

    def test_different_pool_size_is_separate_entry(self, search_cache):
        vids_10 = _make_videos("a1")
        vids_20 = _make_videos("b1")
        search_cache.set_search_result("cats", "short", 10, vids_10)
        search_cache.set_search_result("cats", "short", 20, vids_20)

        assert search_cache.get_search_result("cats", "short", 10)[0]["id"] == "a1"
        assert search_cache.get_search_result("cats", "short", 20)[0]["id"] == "b1"

    def test_keyword_case_insensitive(self, search_cache):
        videos = _make_videos("v1")
        search_cache.set_search_result("Cats", "short", 10, videos)
        result = search_cache.get_search_result("cats", "short", 10)
        assert result is not None

    def test_empty_video_list_cached(self, search_cache):
        search_cache.set_search_result("noresults", "short", 10, [])
        result = search_cache.get_search_result("noresults", "short", 10)
        assert result == []


# ============================================================
# SearchResultsCache — expiry
# ============================================================

class TestSearchCacheExpiry:

    def test_expired_entry_returns_none(self, tmp_path):
        cache = SearchResultsCache(cache_dir=tmp_path / "search_exp", ttl_hours=1)
        cache.set_search_result("test", "short", 10, _make_videos("v1"))

        # Backdate by 2 hours
        for key in cache.index:
            cache.index[key]["cached_at"] = time.time() - 7200

        result = cache.get_search_result("test", "short", 10)
        assert result is None

    def test_non_expired_entry_valid(self, tmp_path):
        cache = SearchResultsCache(cache_dir=tmp_path / "search_fresh", ttl_hours=24)
        cache.set_search_result("test", "short", 10, _make_videos("v1"))
        result = cache.get_search_result("test", "short", 10)
        assert result is not None

    def test_zero_ttl_never_expires(self, tmp_path):
        cache = SearchResultsCache(cache_dir=tmp_path / "search_noexp", ttl_hours=0)
        cache.set_search_result("test", "short", 10, _make_videos("v1"))

        for key in cache.index:
            cache.index[key]["cached_at"] = 1000000.0

        result = cache.get_search_result("test", "short", 10)
        assert result is not None


# ============================================================
# SearchResultsCache — invalidation
# ============================================================

class TestSearchCacheInvalidation:

    def test_invalidate_keyword_removes_entries(self, search_cache):
        search_cache.set_search_result("cats", "short", 10, _make_videos("v1"))
        search_cache.set_search_result("cats", "long", 10, _make_videos("v2"))
        search_cache.set_search_result("dogs", "short", 10, _make_videos("v3"))

        removed = search_cache.invalidate_keyword("cats")
        assert removed == 2

        assert search_cache.get_search_result("cats", "short", 10) is None
        assert search_cache.get_search_result("cats", "long", 10) is None
        assert search_cache.get_search_result("dogs", "short", 10) is not None

    def test_invalidate_nonexistent_returns_zero(self, search_cache):
        assert search_cache.invalidate_keyword("nothing") == 0

    def test_clear_removes_all(self, search_cache):
        search_cache.set_search_result("a", "short", 10, [])
        search_cache.set_search_result("b", "long", 20, [])
        search_cache.clear()
        assert search_cache._count_entries() == 0


# ============================================================
# Serialization roundtrip — save to disk and reload
# ============================================================

class TestSerializationRoundtrip:

    def test_llm_cache_persists_to_disk(self, tmp_path):
        cache_dir = tmp_path / "llm_persist"
        cache1 = LLMFilterCache(cache_dir=cache_dir, ttl_days=7)
        videos = _make_videos("v1", "v2")
        approved = [{"id": "v1", "relevance": 0.95}]
        cache1.set_filter_result("nature", videos, approved)

        # Create a new cache instance reading from the same directory
        cache2 = LLMFilterCache(cache_dir=cache_dir, ttl_days=7)
        result = cache2.get_filter_result("nature", videos)

        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "v1"
        assert result[0]["relevance"] == 0.95

    def test_search_cache_persists_to_disk(self, tmp_path):
        cache_dir = tmp_path / "search_persist"
        cache1 = SearchResultsCache(cache_dir=cache_dir, ttl_hours=24)
        videos = _make_videos("v1", "v2", "v3")
        cache1.set_search_result("cats", "medium", 15, videos)

        cache2 = SearchResultsCache(cache_dir=cache_dir, ttl_hours=24)
        result = cache2.get_search_result("cats", "medium", 15)

        assert result is not None
        assert len(result) == 3
        assert [v["id"] for v in result] == ["v1", "v2", "v3"]

    def test_index_file_is_valid_json(self, tmp_path):
        cache_dir = tmp_path / "json_check"
        cache = LLMFilterCache(cache_dir=cache_dir, ttl_days=7)
        cache.set_filter_result("test", _make_videos("v1"), [{"id": "v1"}])

        index_path = cache_dir / "llm_filter_cache_index.json"
        assert index_path.exists()

        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert len(data) == 1

    def test_search_index_file_is_valid_json(self, tmp_path):
        cache_dir = tmp_path / "json_check_search"
        cache = SearchResultsCache(cache_dir=cache_dir, ttl_hours=24)
        cache.set_search_result("test", "short", 10, _make_videos("v1"))

        index_path = cache_dir / "search_cache_index.json"
        assert index_path.exists()

        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert len(data) == 1

    def test_corrupted_index_recovers_gracefully(self, tmp_path):
        cache_dir = tmp_path / "corrupted"
        cache_dir.mkdir(parents=True)
        index_path = cache_dir / "llm_filter_cache_index.json"
        index_path.write_text("{invalid json!!!", encoding="utf-8")

        # Should not raise — starts fresh
        cache = LLMFilterCache(cache_dir=cache_dir, ttl_days=7)
        assert cache._count_entries() == 0

        # Should work normally after recovery
        cache.set_filter_result("test", _make_videos("v1"), [{"id": "v1"}])
        assert cache.get_filter_result("test", _make_videos("v1")) is not None


# ============================================================
# Eviction behavior
# ============================================================

class TestEvictionBehavior:

    def test_cleanup_expired_removes_old_entries(self, tmp_path):
        cache = LLMFilterCache(cache_dir=tmp_path / "evict", ttl_days=1)
        videos = _make_videos("v1")
        cache.set_filter_result("old", videos, [{"id": "v1"}])
        cache.set_filter_result("new", videos, [{"id": "v1"}])

        # Backdate only one entry
        first_key = list(cache.index.keys())[0]
        cache.index[first_key]["cached_at"] = time.time() - (3 * 86400)

        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache._count_entries() == 1

    def test_evict_to_size_removes_oldest_first(self, tmp_path):
        cache_dir = tmp_path / "evict_size"
        cache = SearchResultsCache(cache_dir=cache_dir, ttl_hours=24)

        # Add several entries with increasing timestamps
        for i in range(5):
            cache.set_search_result(f"kw{i}", "short", 10, _make_videos(f"v{i}"))

        # Verify all 5 entries exist
        assert cache._count_entries() == 5

        # Evict with a very small target size (0 MB) to force removal
        result = cache.evict_to_size(target_size_mb=0.0, strategy="lru")
        assert result.entries_removed > 0
        assert len(result.evicted_keys) > 0

    def test_evict_dry_run_does_not_modify(self, tmp_path):
        cache_dir = tmp_path / "evict_dry"
        cache = SearchResultsCache(cache_dir=cache_dir, ttl_hours=24)
        for i in range(3):
            cache.set_search_result(f"kw{i}", "short", 10, _make_videos(f"v{i}"))

        before_count = cache._count_entries()
        result = cache.evict_to_size(target_size_mb=0.0, strategy="lru", dry_run=True)
        assert result.dry_run is True
        assert cache._count_entries() == before_count

    def test_evict_when_under_target_does_nothing(self, tmp_path):
        cache = LLMFilterCache(cache_dir=tmp_path / "evict_ok", ttl_days=7)
        cache.set_filter_result("test", _make_videos("v1"), [{"id": "v1"}])

        result = cache.evict_to_size(target_size_mb=1000.0)
        assert result.entries_removed == 0


# ============================================================
# Statistics tracking
# ============================================================

class TestStatistics:

    def test_hit_miss_tracking(self, llm_cache):
        videos = _make_videos("v1")
        llm_cache.set_filter_result("test", videos, [{"id": "v1"}])

        # 1 hit
        llm_cache.get_filter_result("test", videos)
        # 1 miss
        llm_cache.get_filter_result("nonexistent", videos)

        stats = llm_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_search_cache_stats(self, search_cache):
        search_cache.set_search_result("cats", "short", 10, _make_videos("v1"))
        search_cache.get_search_result("cats", "short", 10)  # hit
        search_cache.get_search_result("dogs", "short", 10)  # miss

        stats = search_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert 0 < stats["hit_rate"] < 1.0


# ============================================================
# Dataclass helpers
# ============================================================

class TestDataclassHelpers:

    def test_cached_llm_filter_result_roundtrip(self):
        original = CachedLLMFilterResult(
            keyword="nature",
            video_ids_hash="abc123",
            approved_videos=[{"id": "v1"}],
            cached_at="2026-01-01T00:00:00",
        )
        d = original.to_dict()
        restored = CachedLLMFilterResult.from_dict(d)
        assert restored.keyword == original.keyword
        assert restored.video_ids_hash == original.video_ids_hash
        assert restored.approved_videos == original.approved_videos

    def test_cached_search_result_roundtrip(self):
        original = CachedSearchResult(
            keyword="cats",
            tier="short",
            search_pool=10,
            videos=[{"id": "v1"}],
            cached_at="2026-01-01T00:00:00",
        )
        d = original.to_dict()
        restored = CachedSearchResult.from_dict(d)
        assert restored.keyword == original.keyword
        assert restored.tier == original.tier
        assert restored.search_pool == original.search_pool
        assert restored.videos == original.videos

    def test_from_dict_ignores_extra_fields(self):
        d = {
            "keyword": "test",
            "video_ids_hash": "abc",
            "approved_videos": [],
            "cached_at": "2026-01-01",
            "extra_field": "should be ignored",
        }
        result = CachedLLMFilterResult.from_dict(d)
        assert result.keyword == "test"
        assert not hasattr(result, "extra_field")
