"""
Tests for US-67-005: Cache caption preflight discovery results to avoid
repeated list-subs calls.

Verifies that preflight results (available formats per video ID) are cached
in EnhancedCaptionCache with 1-hour TTL, and that on cache hit the
list_available_languages subprocess call is skipped entirely.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionCache,
    AvailableLanguage,
)
from src.caption_fetcher_cache import EnhancedCaptionCache


ENGLISH_AVAILABLE = [
    AvailableLanguage(code='en', name='English', is_auto_generated=False),
    AvailableLanguage(code='es', name='Spanish', is_auto_generated=True),
]


@pytest.mark.fast
class TestPreflightCacheInEnhancedCaptionCache:
    """Tests for preflight cache methods on EnhancedCaptionCache (US-67-005)."""

    def _make_enhanced_cache(self):
        """Create an EnhancedCaptionCache with a mock base cache."""
        base = MagicMock(spec=CaptionCache)
        base.enabled = True
        return EnhancedCaptionCache(base)

    def test_cache_key_format_is_video_id_preflight(self):
        """AC: Cache key format is '{video_id}_preflight'."""
        cache = self._make_enhanced_cache()

        cache.store_preflight("dQw4w9WgXcQ", ENGLISH_AVAILABLE)

        # Internal store should use {video_id}_preflight key
        assert "dQw4w9WgXcQ_preflight" in cache._preflight_store

    def test_store_and_retrieve_preflight(self):
        """AC: Preflight results are cached and retrievable."""
        cache = self._make_enhanced_cache()

        cache.store_preflight("dQw4w9WgXcQ", ENGLISH_AVAILABLE)
        result = cache.get_preflight("dQw4w9WgXcQ")

        assert result is not None
        assert len(result) == 2
        assert result[0]['code'] == 'en'
        assert result[0]['is_auto_generated'] is False
        assert result[1]['code'] == 'es'
        assert result[1]['is_auto_generated'] is True

    def test_cache_miss_returns_none(self):
        """Cache miss for unknown video returns None."""
        cache = self._make_enhanced_cache()
        assert cache.get_preflight("unknown_video") is None

    def test_store_converts_available_language_objects_to_dicts(self):
        """AvailableLanguage objects are converted to dicts for storage."""
        cache = self._make_enhanced_cache()

        cache.store_preflight("vid123", ENGLISH_AVAILABLE)
        result = cache.get_preflight("vid123")

        assert isinstance(result, list)
        for entry in result:
            assert isinstance(entry, dict)
            assert 'code' in entry
            assert 'name' in entry
            assert 'is_auto_generated' in entry

    def test_store_accepts_dict_languages(self):
        """Store also accepts pre-serialized dicts."""
        cache = self._make_enhanced_cache()
        dict_langs = [
            {'code': 'fr', 'name': 'French', 'is_auto_generated': False},
        ]

        cache.store_preflight("vid456", dict_langs)
        result = cache.get_preflight("vid456")

        assert result is not None
        assert len(result) == 1
        assert result[0]['code'] == 'fr'

    def test_ttl_is_one_hour(self):
        """AC: Cache entries expire after 1-hour TTL."""
        cache = self._make_enhanced_cache()
        assert cache._preflight_ttl_seconds == 3600.0

    def test_cache_expires_after_ttl(self):
        """AC: Cache entry expires after TTL and triggers fresh --list-subs call."""
        cache = self._make_enhanced_cache()

        cache.store_preflight("dQw4w9WgXcQ", ENGLISH_AVAILABLE)

        # Manually backdate the cached_at timestamp beyond TTL
        key = "dQw4w9WgXcQ_preflight"
        cache._preflight_store[key]['cached_at'] = time.time() - 3601  # 1 hour + 1 second ago

        result = cache.get_preflight("dQw4w9WgXcQ")
        assert result is None, "Should return None for expired entry"

        # Entry should be removed from store
        assert key not in cache._preflight_store

    def test_store_empty_list_for_no_captions(self):
        """Empty language list is cached (negative cache for no-captions videos)."""
        cache = self._make_enhanced_cache()

        cache.store_preflight("noSubs123", [])
        result = cache.get_preflight("noSubs123")

        assert result is not None
        assert result == []


@pytest.mark.fast
class TestPreflightCacheIntegration:
    """Tests for preflight cache wired into CaptionFetcher (US-67-005)."""

    def _make_fetcher_with_preflight_cache(self):
        """Create CaptionFetcher with an EnhancedCaptionCache for preflight."""
        base = MagicMock(spec=CaptionCache)
        base.enabled = True
        preflight_cache = EnhancedCaptionCache(base)
        fetcher = CaptionFetcher(preflight_cache=preflight_cache)
        return fetcher, preflight_cache

    def test_second_call_skips_subprocess(self):
        """AC: On cache hit, _check_captions_preflight skips the yt-dlp
        --list-subs subprocess call entirely.

        Unit test verifies second call for same video_id does not spawn
        subprocess (mock subprocess, check call count).
        """
        fetcher, preflight_cache = self._make_fetcher_with_preflight_cache()

        list_subs_stdout = (
            "[info] Available subtitles for dQw4w9WgXcQ:\n"
            "Language  Name     Formats\n"
            "en        English  vtt, ttml, srv3, srv2, srv1, json3\n"
        )

        with patch('src.caption_fetcher.subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout=list_subs_stdout,
                stderr="",
                returncode=0,
            )

            # First call - should invoke subprocess
            result1 = fetcher.list_available_languages("dQw4w9WgXcQ")
            assert mock_run.call_count == 1
            assert len(result1) >= 1

            # Second call - should use preflight cache, skip subprocess
            result2 = fetcher.list_available_languages("dQw4w9WgXcQ")
            assert mock_run.call_count == 1, (
                "Second call should NOT spawn subprocess (cache hit)"
            )
            assert len(result2) == len(result1)

    def test_cache_miss_triggers_subprocess(self):
        """Cache miss for new video triggers subprocess call."""
        fetcher, preflight_cache = self._make_fetcher_with_preflight_cache()

        with patch('src.caption_fetcher.subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="",
                stderr="",
                returncode=0,
            )

            # First video (11-char YouTube ID format)
            fetcher.list_available_languages("aaaaaaaaaaa")
            assert mock_run.call_count == 1

            # Different video - should also trigger subprocess
            fetcher.list_available_languages("bbbbbbbbbbb")
            assert mock_run.call_count == 2

    def test_expired_cache_triggers_fresh_subprocess(self):
        """AC: Cache entry expires after TTL and triggers fresh --list-subs call."""
        fetcher, preflight_cache = self._make_fetcher_with_preflight_cache()

        list_subs_stdout = (
            "[info] Available subtitles for dQw4w9WgXcQ:\n"
            "Language  Name     Formats\n"
            "en        English  vtt, ttml, srv3, srv2, srv1, json3\n"
        )

        with patch('src.caption_fetcher.subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout=list_subs_stdout,
                stderr="",
                returncode=0,
            )

            # First call populates cache
            fetcher.list_available_languages("dQw4w9WgXcQ")
            assert mock_run.call_count == 1

            # Expire the cache entry manually
            key = "dQw4w9WgXcQ_preflight"
            preflight_cache._preflight_store[key]['cached_at'] = (
                time.time() - 3601  # 1 hour + 1 second ago
            )

            # Third call - cache expired, should trigger new subprocess
            fetcher.list_available_languages("dQw4w9WgXcQ")
            assert mock_run.call_count == 2, (
                "Expired cache should trigger fresh --list-subs subprocess call"
            )

    def test_fetcher_without_preflight_cache_still_works(self):
        """CaptionFetcher without preflight_cache works normally."""
        fetcher = CaptionFetcher()  # No preflight_cache

        with patch('src.caption_fetcher.subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="",
                stderr="",
                returncode=0,
            )

            fetcher.list_available_languages("dQw4w9WgXcQ")
            assert mock_run.call_count == 1

            # Second call also triggers subprocess (no cache)
            fetcher.list_available_languages("dQw4w9WgXcQ")
            assert mock_run.call_count == 2

    def test_preflight_cache_stores_empty_for_no_captions(self):
        """Videos with no captions cache the empty result."""
        fetcher, preflight_cache = self._make_fetcher_with_preflight_cache()

        with patch('src.caption_fetcher.subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="",
                stderr="",
                returncode=0,
            )

            # First call - subprocess returns no captions (11-char YouTube ID)
            result1 = fetcher.list_available_languages("noSubs12345")
            assert mock_run.call_count == 1
            assert result1 == []

            # Second call - should use cache
            result2 = fetcher.list_available_languages("noSubs12345")
            assert mock_run.call_count == 1, "Empty result should be cached"
            assert result2 == []
