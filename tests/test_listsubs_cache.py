"""
Tests for US-59-012: Cache list-subs output to avoid redundant pre-flight calls on resume.

Verifies that list_available_languages() checks a persistent cache before
making yt-dlp --list-subs subprocess calls, with configurable TTL.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.caption_fetcher import (
    CaptionFetcher,
    AvailableLanguage,
)
from src.caption.cache import ListSubsCache


ENGLISH_AVAILABLE = [
    AvailableLanguage(code='en', name='English', is_auto_generated=False),
    AvailableLanguage(code='en', name='English (auto-generated)', is_auto_generated=True),
]


@pytest.mark.fast
class TestListSubsCache:
    """Tests for ListSubsCache persistence (US-59-012)."""

    def test_cache_stores_and_retrieves_languages(self, tmp_path):
        """Cache stores available languages and retrieves them correctly.

        AC: The cache key is video_id and the cached value includes the list
        of AvailableLanguage objects and a timestamp.
        """
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)

        # Store languages
        cache.store("dQw4w9WgXcQ", ENGLISH_AVAILABLE)

        # Retrieve from cache
        cached = cache.get_languages("dQw4w9WgXcQ")

        assert cached is not None
        assert len(cached) == 2
        assert cached[0]['code'] == 'en'
        assert cached[0]['name'] == 'English'
        assert cached[0]['is_auto_generated'] is False
        assert cached[1]['is_auto_generated'] is True

    def test_cache_persists_to_disk(self, tmp_path):
        """Cache persists to .cache/ directory structure.

        AC: The cache is stored in the existing .cache/ directory structure
        (not in-memory only) so it persists across --resume runs.
        """
        cache_dir = tmp_path / "list_subs_cache"
        cache1 = ListSubsCache(cache_dir=cache_dir, ttl_hours=1.0)

        # Store languages
        cache1.store("dQw4w9WgXcQ", ENGLISH_AVAILABLE)

        # Verify index file exists
        index_file = cache_dir / "list_subs_cache.json"
        assert index_file.exists()

        # Create a new cache instance (simulating process restart)
        cache2 = ListSubsCache(cache_dir=cache_dir, ttl_hours=1.0)

        # Should retrieve from persisted cache
        cached = cache2.get_languages("dQw4w9WgXcQ")
        assert cached is not None
        assert len(cached) == 2

    def test_cache_ttl_expiration(self, tmp_path):
        """Cache entries expire after configurable TTL.

        AC: Cache entries expire after a configurable TTL (default 1 hour)
        since subtitle availability can change.
        """
        # Use 1 second TTL for testing
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1/3600)

        # Store languages
        cache.store("dQw4w9WgXcQ", ENGLISH_AVAILABLE)

        # Immediately retrieve - should work
        cached = cache.get_languages("dQw4w9WgXcQ")
        assert cached is not None

        # Wait for TTL to expire
        time.sleep(1.5)

        # Should be expired now
        cached = cache.get_languages("dQw4w9WgXcQ")
        assert cached is None

    def test_cache_miss_returns_none(self, tmp_path):
        """Cache returns None for uncached video IDs."""
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)

        cached = cache.get_languages("nonexistent")
        assert cached is None

    def test_disabled_cache_always_returns_none(self, tmp_path):
        """Disabled cache always returns None."""
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", enabled=False)

        # Store attempt should return False
        assert cache.store("dQw4w9WgXcQ", ENGLISH_AVAILABLE) is False

        # Get should return None
        cached = cache.get_languages("dQw4w9WgXcQ")
        assert cached is None


@pytest.mark.fast
class TestCaptionFetcherListSubsCaching:
    """Tests for CaptionFetcher integration with ListSubsCache (US-59-012)."""

    def test_list_available_languages_checks_cache_before_subprocess(self, tmp_path):
        """Calling list_available_languages() twice results in exactly 1 subprocess call.

        AC: list_available_languages() checks a cache before making the yt-dlp
        --list-subs subprocess call.

        AC: Unit test verifies that calling list_available_languages() twice for
        the same video within the TTL results in exactly 1 subprocess call.
        """
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)
        fetcher = CaptionFetcher(list_subs_cache=cache)

        subprocess_call_count = 0

        def mock_subprocess_run(*args, **kwargs):
            nonlocal subprocess_call_count
            subprocess_call_count += 1
            result = Mock()
            result.stdout = """
[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3

[info] Available automatic captions for dQw4w9WgXcQ:
Language  Name                              Formats
en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
"""
            result.stderr = ""
            return result

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            # First call - should hit subprocess
            languages1 = fetcher.list_available_languages("dQw4w9WgXcQ")
            assert len(languages1) >= 1
            assert subprocess_call_count == 1

            # Second call - should use cache, no subprocess
            languages2 = fetcher.list_available_languages("dQw4w9WgXcQ")
            assert len(languages2) >= 1
            assert subprocess_call_count == 1  # Still 1, not 2

    def test_list_available_languages_without_cache_makes_subprocess_call_each_time(self):
        """Without cache, each call makes a subprocess call."""
        fetcher = CaptionFetcher(list_subs_cache=None)

        subprocess_call_count = 0

        def mock_subprocess_run(*args, **kwargs):
            nonlocal subprocess_call_count
            subprocess_call_count += 1
            result = Mock()
            result.stdout = """
[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3
"""
            result.stderr = ""
            return result

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            # First call
            fetcher.list_available_languages("dQw4w9WgXcQ")
            assert subprocess_call_count == 1

            # Second call - no cache, so should make another subprocess call
            fetcher.list_available_languages("dQw4w9WgXcQ")
            assert subprocess_call_count == 2

    def test_cached_languages_match_fresh_fetch(self, tmp_path):
        """Cached result has same structure as freshly fetched result."""
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)
        fetcher = CaptionFetcher(list_subs_cache=cache)

        list_subs_output = """
[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3
es        Spanish              vtt, ttml, srv3, srv2, srv1, json3

[info] Available automatic captions for dQw4w9WgXcQ:
Language  Name                              Formats
en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
"""

        def mock_subprocess_run(*args, **kwargs):
            result = Mock()
            result.stdout = list_subs_output
            result.stderr = ""
            return result

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            # Fresh fetch
            fresh_languages = fetcher.list_available_languages("dQw4w9WgXcQ")

            # Cached fetch
            cached_languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        # Both should have same structure
        assert len(fresh_languages) == len(cached_languages)
        for fresh, cached in zip(fresh_languages, cached_languages):
            assert fresh.code == cached.code
            assert fresh.name == cached.name
            assert fresh.is_auto_generated == cached.is_auto_generated

    def test_different_videos_make_separate_cache_entries(self, tmp_path):
        """Different video IDs are cached separately."""
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)
        fetcher = CaptionFetcher(list_subs_cache=cache)

        subprocess_call_count = 0

        # Valid YouTube video IDs are exactly 11 characters
        video_id_1 = "aaaaaa11111"
        video_id_2 = "bbbbbb22222"

        def mock_subprocess_run(*args, **kwargs):
            nonlocal subprocess_call_count
            subprocess_call_count += 1
            # Determine which video from the command
            cmd = args[0]
            video_id = video_id_1 if video_id_1 in str(cmd) else video_id_2
            result = Mock()
            result.stdout = f"""
[info] Available subtitles for {video_id}:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3
"""
            result.stderr = ""
            return result

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            # First video - first call
            fetcher.list_available_languages(video_id_1)
            assert subprocess_call_count == 1

            # Second video - needs fresh call
            fetcher.list_available_languages(video_id_2)
            assert subprocess_call_count == 2

            # First video again - from cache
            fetcher.list_available_languages(video_id_1)
            assert subprocess_call_count == 2

            # Second video again - from cache
            fetcher.list_available_languages(video_id_2)
            assert subprocess_call_count == 2

    def test_cache_survives_process_restart(self, tmp_path):
        """Cache survives CaptionFetcher re-instantiation (simulating --resume).

        AC: The cache is stored in the existing .cache/ directory structure
        (not in-memory only) so it persists across --resume runs.
        """
        cache_dir = tmp_path / "list_subs_cache"

        # First "run"
        cache1 = ListSubsCache(cache_dir=cache_dir, ttl_hours=1.0)
        fetcher1 = CaptionFetcher(list_subs_cache=cache1)

        subprocess_call_count = 0

        def mock_subprocess_run(*args, **kwargs):
            nonlocal subprocess_call_count
            subprocess_call_count += 1
            result = Mock()
            result.stdout = """
[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3
"""
            result.stderr = ""
            return result

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            # First fetch - makes subprocess call
            fetcher1.list_available_languages("dQw4w9WgXcQ")
            assert subprocess_call_count == 1

        # Second "run" (simulating --resume with new instances)
        cache2 = ListSubsCache(cache_dir=cache_dir, ttl_hours=1.0)
        fetcher2 = CaptionFetcher(list_subs_cache=cache2)

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            # Should use persisted cache, no new subprocess call
            languages = fetcher2.list_available_languages("dQw4w9WgXcQ")
            assert subprocess_call_count == 1  # Still 1
            assert len(languages) >= 1


@pytest.mark.fast
class TestListSubsNegativeCache:
    """Tests for US-62-004: Negative cache for videos with no captions."""

    def test_store_and_retrieve_empty_languages(self, tmp_path):
        """Storing empty language list creates a negative cache entry.

        AC: When --list-subs returns empty, store negative cache entry with video_id.
        """
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)

        # Store empty languages (no captions available)
        cache.store("no_caps_vid1", [])

        # Retrieve should return empty list, not None
        cached = cache.get_languages("no_caps_vid1")
        assert cached is not None, "Empty list should be cached, not None"
        assert cached == []

    def test_is_no_captions_available_returns_true_for_empty_cache(self, tmp_path):
        """is_no_captions_available returns True for videos cached with empty list.

        AC: Subsequent requests for same video_id check negative cache before network call.
        """
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)

        # Store empty (negative cache)
        cache.store("no_caps_vid2", [])

        # Should return True
        assert cache.is_no_captions_available("no_caps_vid2") is True

    def test_is_no_captions_available_returns_false_for_videos_with_captions(self, tmp_path):
        """is_no_captions_available returns False for videos with cached captions."""
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)

        # Store with captions
        cache.store("has_caps_vid", ENGLISH_AVAILABLE)

        assert cache.is_no_captions_available("has_caps_vid") is False

    def test_is_no_captions_available_returns_false_for_uncached_videos(self, tmp_path):
        """is_no_captions_available returns False for videos not in cache."""
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)

        # Never cached
        assert cache.is_no_captions_available("never_seen_vid") is False

    def test_negative_cache_hit_logs_message(self, tmp_path, caplog):
        """Negative cache hit logs 'Negative cache hit: no captions'.

        AC: Negative cache hit logs 'Negative cache hit: no captions' and returns immediately.
        """
        import logging
        caplog.set_level(logging.INFO)

        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)

        # Store empty (negative cache)
        cache.store("log_test_vid", [])

        # Clear logs before the get
        caplog.clear()

        # Retrieve should log the message
        cached = cache.get_languages("log_test_vid")
        assert cached == []

        # Check log message
        assert any("Negative cache hit: no captions" in record.message for record in caplog.records)

    def test_zero_ytdlp_calls_for_negative_cached_video(self, tmp_path):
        """Second list_available_languages call for negative-cached video makes 0 yt-dlp calls.

        AC: Test verifies 0 yt-dlp calls for negative-cached video.
        """
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1.0)
        fetcher = CaptionFetcher(list_subs_cache=cache)

        subprocess_call_count = 0

        def mock_subprocess_run(*args, **kwargs):
            nonlocal subprocess_call_count
            subprocess_call_count += 1
            result = Mock()
            # Return empty subtitles
            result.stdout = "[info] Available subtitles for emptycapvid:\n"
            result.stderr = ""
            return result

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            # First call - should hit subprocess, get empty result
            languages1 = fetcher.list_available_languages("emptycapvid")
            assert len(languages1) == 0
            assert subprocess_call_count == 1

            # Second call - should use negative cache, zero subprocess calls
            languages2 = fetcher.list_available_languages("emptycapvid")
            assert len(languages2) == 0
            assert subprocess_call_count == 1, "Should be 1, not 2 - negative cache should prevent second call"

    def test_negative_cache_expires_after_ttl(self, tmp_path):
        """Negative cache entries expire after TTL.

        AC: Test verifies negative cache expires after TTL.
        AC: Negative cache entries use configurable TTL (default 1 hour).
        """
        # Use 1 second TTL for testing
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=1/3600)

        # Store empty (negative cache)
        cache.store("expire_test_vid", [])

        # Immediately should be available
        assert cache.is_no_captions_available("expire_test_vid") is True

        # Wait for TTL to expire
        time.sleep(1.5)

        # Should be expired now - get_languages returns None
        cached = cache.get_languages("expire_test_vid")
        assert cached is None, "Negative cache should expire after TTL"

        # is_no_captions_available should also return False
        assert cache.is_no_captions_available("expire_test_vid") is False

    def test_negative_cache_configurable_ttl(self, tmp_path):
        """Negative cache TTL is configurable.

        AC: Negative cache entries use configurable TTL (default 1 hour).
        """
        # Use 2 hour TTL
        cache = ListSubsCache(cache_dir=tmp_path / "list_subs_cache", ttl_hours=2.0)
        assert cache.ttl_hours == 2.0

        # Use 0.5 hour TTL
        cache2 = ListSubsCache(cache_dir=tmp_path / "list_subs_cache2", ttl_hours=0.5)
        assert cache2.ttl_hours == 0.5
