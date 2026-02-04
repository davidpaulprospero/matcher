"""
Tests for US-59-003 and US-60-004: Negative cache integration and configurable TTL.

US-59-003 verifies:
1. When fetch_captions determines no captions exist, store_unavailable is called
2. A second fetch attempt for the same video hits the negative cache (zero yt-dlp calls)
3. Negative cache entries respect staleness policy
4. Integration works with both single-video and batch fetch paths

US-60-004 verifies:
1. store_unavailable() persists negative results correctly
2. negative_cache_ttl_hours parameter configures TTL (default 1 hour)
3. is_caption_unavailable() respects TTL and returns False for expired entries
4. Negative cache check short-circuits fetches for known captionless videos
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
        config.negative_cache_ttl_hours = 1.0  # US-60-004
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


@pytest.mark.fast
class TestNegativeCacheTTLConfig:
    """Tests for US-60-004: Configurable negative cache TTL."""

    def test_default_negative_ttl_is_one_hour(self):
        """Default negative_cache_ttl_hours is 1.0 (1 hour).

        AC: Add configurable negative_cache_ttl_hours parameter to CaptionFirstConfig (default 1 hour).
        """
        from src.config.sections.download import CaptionFirstConfig
        config = CaptionFirstConfig()
        assert config.negative_cache_ttl_hours == 1.0

    def test_custom_negative_ttl_is_respected(self):
        """Custom negative_cache_ttl_hours is used by CaptionCache."""
        from src.config.sections.download import CaptionFirstConfig
        config = CaptionFirstConfig(
            cache_dir=tempfile.mkdtemp(),
            negative_cache_ttl_hours=2.5
        )
        cache = CaptionCache(config)

        assert cache.negative_cache_ttl_hours == 2.5

    def test_zero_negative_ttl_uses_max_age_days(self):
        """When negative_cache_ttl_hours=0, is_negative_entry_stale uses max_age_days."""
        from src.config.sections.download import CaptionFirstConfig
        config = CaptionFirstConfig(
            cache_dir=tempfile.mkdtemp(),
            negative_cache_ttl_hours=0,
            max_cache_age_days=30
        )
        cache = CaptionCache(config)

        cache.store_unavailable("dQw4w9WgXcQ", "en")

        # Should not be stale immediately (uses 30 day TTL)
        assert cache.is_caption_unavailable("dQw4w9WgXcQ", "en") is True


@pytest.mark.fast
class TestNegativeCacheTTLExpiration:
    """Tests for US-60-004: is_caption_unavailable() respecting TTL."""

    def _make_cache_with_ttl(self, ttl_hours, validation_mode='strict'):
        """Create a CaptionCache with specific TTL and validation mode."""
        from src.config.sections.download import CaptionFirstConfig
        config = CaptionFirstConfig(
            cache_dir=tempfile.mkdtemp(),
            negative_cache_ttl_hours=ttl_hours,
            cache_validation=validation_mode,
            max_cache_age_days=30
        )
        return CaptionCache(config)

    def test_fresh_negative_entry_returns_true(self):
        """Fresh negative entry returns True for is_caption_unavailable().

        AC: Ensure is_caption_unavailable() respects the TTL and returns False for expired entries.
        """
        cache = self._make_cache_with_ttl(ttl_hours=1.0)

        cache.store_unavailable("dQw4w9WgXcQ", "en")

        # Fresh entry should return True
        assert cache.is_caption_unavailable("dQw4w9WgXcQ", "en") is True

    def test_expired_negative_entry_returns_false_in_strict_mode(self):
        """Expired negative entry returns False in strict validation mode.

        AC: Ensure is_caption_unavailable() respects the TTL and returns False for expired entries.
        """
        # Use 1 second TTL for testing
        cache = self._make_cache_with_ttl(ttl_hours=1 / 3600, validation_mode='strict')

        cache.store_unavailable("dQw4w9WgXcQ", "en")

        # Immediately should be available
        assert cache.is_caption_unavailable("dQw4w9WgXcQ", "en") is True

        # Wait for TTL to expire
        time.sleep(1.5)

        # Expired entry should return False in strict mode
        assert cache.is_caption_unavailable("dQw4w9WgXcQ", "en") is False

    def test_expired_negative_entry_warns_but_returns_true_in_warn_mode(self):
        """Expired negative entry logs warning but returns True in warn mode."""
        # Use 1 second TTL for testing
        cache = self._make_cache_with_ttl(ttl_hours=1 / 3600, validation_mode='warn')

        cache.store_unavailable("warntest0001", "en")

        # Wait for TTL to expire
        time.sleep(1.5)

        # Warn mode still returns True but logs warning
        # The warning is logged via src.caption_fetcher logger
        with patch('src.caption_fetcher.logger') as mock_logger:
            result = cache.is_caption_unavailable("warntest0001", "en")
            assert result is True
            assert mock_logger.warning.called

    def test_skip_mode_ignores_ttl(self):
        """Skip validation mode ignores TTL for negative entries."""
        # Use 1 second TTL for testing
        cache = self._make_cache_with_ttl(ttl_hours=1 / 3600, validation_mode='skip')

        cache.store_unavailable("skiptest0001", "en")

        # Wait for TTL to expire
        time.sleep(1.5)

        # Skip mode should still return True (ignores staleness)
        assert cache.is_caption_unavailable("skiptest0001", "en") is True

    def test_is_negative_entry_stale_method(self):
        """is_negative_entry_stale() correctly identifies expired entries.

        AC: Ensure is_caption_unavailable() respects the TTL and returns False for expired entries.
        """
        # Use 1 second TTL for testing
        cache = self._make_cache_with_ttl(ttl_hours=1 / 3600)

        cache.store_unavailable("staletest001", "en")

        # Get the entry
        key = cache._make_cache_key("staletest001", "en")
        entry = cache.get(key)

        # Fresh entry should not be stale
        assert cache.is_negative_entry_stale(entry) is False

        # Wait for TTL to expire
        time.sleep(1.5)

        # Entry should now be stale
        assert cache.is_negative_entry_stale(entry) is True


@pytest.mark.fast
class TestNegativeCacheTTLIntegration:
    """Tests for US-60-004: Integration of negative cache TTL with fetch_captions."""

    def test_negative_cache_hit_prevents_fetch_attempt(self):
        """Negative cache hit prevents subprocess calls in fetch_captions().

        AC: Integrate negative cache check at start of fetch_captions() to
        short-circuit fetches for known captionless videos.
        AC: Add test verifying negative cache hit prevents fetch attempts.
        """
        from src.config.sections.download import CaptionFirstConfig

        config = CaptionFirstConfig(
            cache_dir=tempfile.mkdtemp(),
            negative_cache_ttl_hours=1.0
        )
        cache = CaptionCache(config)

        # Pre-populate negative cache
        cache.store_unavailable("dQw4w9WgXcQ", "en")

        # Create fetcher with the cache
        fetcher = CaptionFetcher(caption_cache=cache)

        # Track subprocess calls
        subprocess_call_count = 0

        def mock_subprocess_run(*args, **kwargs):
            nonlocal subprocess_call_count
            subprocess_call_count += 1
            raise Exception("Should not be called")

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            # fetch_captions should raise CaptionUnavailableError without subprocess
            with pytest.raises(CaptionUnavailableError) as exc_info:
                fetcher.fetch_captions("dQw4w9WgXcQ", "en")

            # Verify no subprocess calls were made
            assert subprocess_call_count == 0

            # Verify error mentions negative cache
            assert "Negative cache" in str(exc_info.value) or "unavailable" in str(exc_info.value).lower()

    def test_expired_negative_cache_triggers_refetch(self):
        """Expired negative cache allows fetch attempt.

        AC: Add test verifying negative cache hit prevents fetch attempts
        and respects TTL expiration.
        """
        from src.config.sections.download import CaptionFirstConfig

        # Use 1 second TTL for testing
        config = CaptionFirstConfig(
            cache_dir=tempfile.mkdtemp(),
            negative_cache_ttl_hours=1 / 3600,  # 1 second
            cache_validation="strict"
        )
        cache = CaptionCache(config)

        # Pre-populate negative cache
        cache.store_unavailable("refetchtest", "en")

        # Create fetcher with the cache
        fetcher = CaptionFetcher(caption_cache=cache)

        # Track subprocess calls
        subprocess_call_count = 0

        def mock_subprocess_run(*args, **kwargs):
            nonlocal subprocess_call_count
            subprocess_call_count += 1
            result = Mock()
            result.stdout = "[info] Available subtitles for refetchtest:\n"
            result.stderr = ""
            result.returncode = 0
            return result

        # Wait for TTL to expire
        time.sleep(1.5)

        with patch('subprocess.run', side_effect=mock_subprocess_run):
            # fetch_captions should attempt fetch since cache is expired
            with pytest.raises(CaptionUnavailableError):
                fetcher.fetch_captions("refetchtest", "en")

            # Verify subprocess was called (cache expired, so re-fetch attempted)
            assert subprocess_call_count >= 1
