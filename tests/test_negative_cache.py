"""
Tests for US-59-003: Integrate negative cache (store_unavailable) into caption fetcher.

Verifies that:
1. When fetch_captions determines no captions exist, store_unavailable is called
2. A second fetch attempt for the same video hits the negative cache (zero yt-dlp calls)
3. Negative cache entries respect staleness policy
4. Integration works with both single-video and batch fetch paths
"""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionCache,
    CachedCaption,
    CaptionResult,
    CaptionSegment,
    CaptionUnavailableError,
    CaptionFetchError,
    AvailableLanguage,
)


@pytest.mark.fast
class TestNegativeCacheIntegration:
    """Tests for negative cache integration in CaptionFetcher (US-59-003)."""

    def _make_cache(self, enabled=True, validation_mode='warn'):
        """Create a CaptionCache with a temp directory."""
        config = Mock()
        config.cache_dir = tempfile.mkdtemp()
        config.max_cache_age_days = 30
        config.cache_captions = enabled
        config.cache_validation = validation_mode
        config.cache_validation_tolerance = 0.2
        cache = CaptionCache(config)
        cache.enabled = enabled
        cache.validation_mode = validation_mode
        return cache

    def _make_fetcher(self, cache=None):
        """Create a CaptionFetcher with optional cache."""
        return CaptionFetcher(caption_cache=cache)

    def test_store_unavailable_on_no_captions(self):
        """AC: When _fetch_subtitle() determines no captions exist (via pre-flight or
        exhausted formats), it calls cache.store_unavailable(video_id, language)
        before raising CaptionUnavailableError.
        """
        cache = self._make_cache()
        fetcher = self._make_fetcher(cache=cache)

        # Mock list_available_languages to return empty (no captions)
        with patch.object(fetcher, 'list_available_languages', return_value=[]):
            with pytest.raises(CaptionUnavailableError):
                fetcher.fetch_captions("dQw4w9WgXcQ", language="en")

        # Verify negative cache was stored
        assert cache.is_caption_unavailable("dQw4w9WgXcQ", "en")

    def test_second_fetch_hits_negative_cache_zero_ytdlp_calls(self):
        """AC: Unit test verifies that a second fetch attempt for a known-unavailable
        video hits the negative cache and makes zero yt-dlp subprocess calls.
        """
        cache = self._make_cache()
        fetcher = self._make_fetcher(cache=cache)

        # First fetch: pre-flight returns empty → stores negative cache
        with patch.object(fetcher, 'list_available_languages', return_value=[]) as mock_list:
            with pytest.raises(CaptionUnavailableError):
                fetcher.fetch_captions("dQw4w9WgXcQ", language="en")
            # First fetch did call list_available_languages
            assert mock_list.call_count == 1

        # Second fetch: should hit negative cache, zero subprocess calls
        with patch('src.caption_fetcher.subprocess.run') as mock_subprocess:
            with patch.object(fetcher, 'list_available_languages') as mock_list2:
                with pytest.raises(CaptionUnavailableError) as exc_info:
                    fetcher.fetch_captions("dQw4w9WgXcQ", language="en")

                assert "Negative cache" in exc_info.value.reason
                # Zero yt-dlp subprocess calls
                assert mock_subprocess.call_count == 0
                # Zero list_available_languages calls
                assert mock_list2.call_count == 0

    def test_negative_cache_check_at_start_of_fetch_captions(self):
        """AC: At the start of fetch_captions(), cache.is_caption_unavailable()
        is checked and returns early if True.
        """
        cache = self._make_cache()
        fetcher = self._make_fetcher(cache=cache)

        # Pre-populate negative cache
        cache.store_unavailable("abc12345678", "en")

        # fetch_captions should raise immediately without any network calls
        with patch.object(fetcher, '_is_valid_video_id', return_value=True):
            with patch.object(fetcher, 'list_available_languages') as mock_list:
                with pytest.raises(CaptionUnavailableError) as exc_info:
                    fetcher.fetch_captions("abc12345678", language="en")

                assert "Negative cache" in exc_info.value.reason
                assert mock_list.call_count == 0

    def test_negative_cache_respects_staleness_strict(self):
        """AC: Negative cache entries use the existing staleness policy
        (strict/warn/skip modes) from CaptionCache.
        """
        cache = self._make_cache(validation_mode='strict')
        cache.max_age_days = 0.00001  # Very short TTL to force staleness
        fetcher = self._make_fetcher(cache=cache)

        # Store unavailable entry with old timestamp
        cache.store_unavailable("stale_video1", "en")

        # Manually make the entry stale by backdating
        key = cache._make_cache_key("stale_video1", "en")
        entry_data = cache.index[key]
        entry_data['cached_at'] = time.time() - 86400 * 365  # 1 year ago
        cache.index[key] = entry_data

        # Strict mode: stale negative cache should return False
        assert not cache.is_caption_unavailable("stale_video1", "en")

    def test_negative_cache_respects_staleness_skip(self):
        """Skip mode ignores staleness for negative cache entries."""
        cache = self._make_cache(validation_mode='skip')
        cache.max_age_days = 0.00001
        fetcher = self._make_fetcher(cache=cache)

        # Store unavailable entry with old timestamp
        cache.store_unavailable("stale_video2", "en")

        # Manually backdate
        key = cache._make_cache_key("stale_video2", "en")
        entry_data = cache.index[key]
        entry_data['cached_at'] = time.time() - 86400 * 365
        cache.index[key] = entry_data

        # Skip mode: stale entries still return True
        assert cache.is_caption_unavailable("stale_video2", "en")

    def test_negative_cache_disabled_does_not_block(self):
        """When cache is disabled, negative cache check is skipped."""
        cache = self._make_cache(enabled=False)
        fetcher = self._make_fetcher(cache=cache)

        # Store should return False when disabled
        assert not cache.store_unavailable("vid123456789", "en")
        assert not cache.is_caption_unavailable("vid123456789", "en")

    def test_no_cache_fetcher_works_normally(self):
        """CaptionFetcher without caption_cache works as before (no regression)."""
        fetcher = self._make_fetcher(cache=None)

        # No cache means _caption_cache is None, negative check is skipped
        with patch.object(fetcher, 'list_available_languages', return_value=[]):
            with pytest.raises(CaptionUnavailableError) as exc_info:
                fetcher.fetch_captions("dQw4w9WgXcQ", language="en")

            # Should NOT mention "Negative cache"
            assert "Negative cache" not in exc_info.value.reason

    def test_batch_fetch_uses_negative_cache(self):
        """AC: Negative cache integration works with batch fetch paths.

        When fetch_captions_batch calls fetch_captions for a known-unavailable
        video, it should hit the negative cache.
        """
        cache = self._make_cache()
        fetcher = self._make_fetcher(cache=cache)

        # Pre-populate negative cache for one video
        cache.store_unavailable("unavail_vid1", "en")

        # Patch fetch_captions_auto_language_with_retry which is called by batch
        original_fetch = fetcher.fetch_captions

        def mock_auto_lang_with_retry(video_id, preferred_language=None, **kwargs):
            """Simulate what fetch_captions_auto_language_with_retry does."""
            lang = preferred_language or "en"
            return original_fetch(video_id, language=lang)

        with patch.object(
            fetcher, 'fetch_captions_auto_language_with_retry',
            side_effect=mock_auto_lang_with_retry
        ):
            results = fetcher.fetch_captions_batch(
                video_ids=["unavail_vid1"],
                preferred_language="en",
                max_workers=1,
            )

        # The batch should have caught the CaptionUnavailableError from negative cache
        assert "unavail_vid1" in results
        result = results["unavail_vid1"]
        assert isinstance(result, dict)
        assert result.get('unavailable') is True or result.get('error') is True

    def test_store_unavailable_creates_proper_entry(self):
        """Verify store_unavailable creates a proper CachedCaption entry."""
        cache = self._make_cache()

        result = cache.store_unavailable("test_vid_001", "en")
        assert result is True

        # Verify the entry
        key = cache._make_cache_key("test_vid_001", "en")
        assert key in cache.index

        entry_data = cache.index[key]
        assert entry_data['data']['unavailable'] is True
        assert entry_data['data']['format_source'] == "unavailable"
        assert entry_data['data']['segments'] == []
        assert entry_data['data']['video_id'] == "test_vid_001"

    def test_positive_cache_not_treated_as_unavailable(self):
        """Positive cache entries (with actual captions) are NOT treated as unavailable."""
        cache = self._make_cache()

        # Store a normal caption result
        result = CaptionResult(
            video_id="good_vid_001",
            language="en",
            segments=[
                CaptionSegment(index=0, start_time=0.0, end_time=5.0, text="Hello"),
            ],
            is_auto_generated=False,
            format_source="json3",
        )
        cache.store(result)

        # Should NOT be treated as unavailable
        assert not cache.is_caption_unavailable("good_vid_001", "en")
