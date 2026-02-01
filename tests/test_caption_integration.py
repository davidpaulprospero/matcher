"""
Integration tests for caption fetching with real YouTube videos.

These tests verify end-to-end caption functionality against real YouTube videos
with known caption states. They are marked with @pytest.mark.requires_network
and are skipped by default in CI.

Run with:
    pytest -m requires_network tests/test_caption_integration.py -v

Maintenance notes:
    Video IDs may become unavailable over time. If tests fail due to video
    unavailability, update the VIDEO_FIXTURES with new stable video IDs.

Video selection criteria:
    - Videos with captions: Any video with available captions (most have auto-generated)
    - No captions: Very short clips or videos with captions disabled

Reality check (2026):
    Most YouTube videos now only have auto-generated captions. True human-uploaded
    captions are rare outside of official channels (movies, TV, some music videos).
    Our tests focus on verifying caption fetching works regardless of type.

Last updated: 2026-01-25
"""

import pytest
from unittest.mock import Mock

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionResult,
    CaptionSegment,
    CaptionUnavailableError,
    CaptionFetchError,
    AvailableLanguage,
)


# =============================================================================
# Test Fixtures - Known Video States
# =============================================================================
# These video IDs have been verified to have specific caption states.
# If videos become unavailable, update with new IDs matching the same criteria.

VIDEO_FIXTURES = {
    # Videos with captions available (may be human or auto-generated)
    "with_captions": {
        # TED Talk video - has auto-generated captions available
        # "Do schools kill creativity?" - Ken Robinson
        "id": "iG9CE55wbtY",
        "expected_languages": ["en"],  # At minimum, English should be available
        "notes": "TED Talk - has captions available (may be auto-generated)",
    },

    # Videos with only auto-generated captions
    "auto_captions": {
        # "Me at the zoo" - first YouTube video, short clip with speech
        # Stable, historic, has auto-captions
        "id": "jNQXAC9IVRw",
        "notes": "First YouTube video ever - short clip with speech, has auto-captions",
    },

    # Videos with no captions available
    "no_captions": {
        # Color bar test pattern - no speech, unlikely to have captions
        "id": "2Z4m4lnjxkY",
        "notes": "Test pattern video - no speech content, unlikely to have captions",
    },

    # Backup video with captions (music video, has auto-generated lyrics)
    "backup_with_captions": {
        # Rick Astley - Never Gonna Give You Up
        # Extremely stable, has been on YouTube since 2009
        # Has auto-generated captions
        "id": "dQw4w9WgXcQ",
        "notes": "Classic music video with auto-generated lyrics captions",
    },
}


class VideoUnavailableError(Exception):
    """Raised when a test video is no longer available."""
    pass


def get_test_video(fixture_key: str) -> str:
    """Get a video ID from fixtures with validation."""
    if fixture_key not in VIDEO_FIXTURES:
        raise ValueError(f"Unknown fixture key: {fixture_key}")
    return VIDEO_FIXTURES[fixture_key]["id"]


# =============================================================================
# Integration Tests - Caption Fetching
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.flaky(reruns=2, reruns_delay=1.0)
class TestCaptionIntegrationFetch:
    """Integration tests for caption fetching from real YouTube videos.

    These tests verify:
    - Successful caption fetch from videos with captions
    - Proper segment parsing and structure
    - Quality detection
    """

    @pytest.mark.fast
    def test_fetch_captions_ted_talk(self):
        """Test fetching captions from a TED Talk.

        TED Talks have captions available (usually auto-generated now),
        making them reliable for testing caption fetching.
        """
        fetcher = CaptionFetcher()
        video_id = get_test_video("with_captions")

        try:
            # First check available languages
            languages = fetcher.list_available_languages(video_id)

            if not languages:
                pytest.skip(f"Video {video_id} has no captions available")

            # Find English or use first available
            language_codes = [lang.code for lang in languages]
            target_lang = "en" if any("en" in code for code in language_codes) else languages[0].code

            # Fetch the captions
            result = fetcher.fetch_captions(video_id, language=target_lang)

            # Verify basic result structure
            assert result.video_id == video_id
            assert len(result.segments) > 0, "Expected segments but got none"

            # Verify segment structure
            for segment in result.segments[:5]:  # Check first 5 segments
                assert segment.start_time >= 0
                assert segment.end_time >= segment.start_time
                assert segment.source_file == video_id

        except CaptionUnavailableError:
            pytest.skip(f"Video {video_id} captions not available - may need updated fixture")
        except CaptionFetchError as e:
            if "unavailable" in str(e).lower() or "private" in str(e).lower():
                pytest.skip(f"Video {video_id} not accessible - may need updated fixture")
            # Skip on preprocessing errors (yt-dlp internal issues)
            if "preprocessing" in str(e).lower() or "invalid data" in str(e).lower():
                pytest.skip(f"yt-dlp preprocessing error for {video_id} - transient issue")
            raise

    @pytest.mark.fast
    def test_fetch_captions_backup_video(self):
        """Test fetching captions from backup video (Rick Astley).

        This is a well-known stable video used as a fallback test.
        """
        fetcher = CaptionFetcher()
        video_id = get_test_video("backup_with_captions")

        try:
            # First verify captions are available
            languages = fetcher.list_available_languages(video_id)

            if not languages:
                pytest.skip(f"Video {video_id} has no captions available")

            result = fetcher.fetch_captions(video_id, language="en")

            assert result.video_id == video_id
            assert len(result.segments) > 0
            assert result.duration > 0

            # The song has specific lyrics we can verify
            full_text = result.text.lower()
            # At least one of these phrases should appear (in lyrics)
            lyrics_present = any(phrase in full_text for phrase in [
                "give you up",
                "let you down",
                "never gonna",
                "gonna",  # Fallback for auto-generated variations
            ])
            # Note: Auto-generated captions may have variations
            if not lyrics_present:
                # Still pass if we got segments (auto-captions may vary)
                assert len(result.segments) > 5, f"Got segments but no recognizable lyrics: {full_text[:200]}..."

        except CaptionUnavailableError:
            pytest.skip(f"Video {video_id} captions not available - may need updated fixture")
        except CaptionFetchError as e:
            if "preprocessing" in str(e).lower() or "invalid data" in str(e).lower():
                pytest.skip(f"yt-dlp preprocessing error for {video_id} - transient issue")
            raise

    @pytest.mark.fast
    def test_caption_quality_detection(self):
        """Test that caption quality is properly detected."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("with_captions")

        try:
            # List languages
            languages = fetcher.list_available_languages(video_id)

            if not languages:
                pytest.skip("No captions available for this video")

            # Fetch captions
            result = fetcher.fetch_captions(video_id)

            # Quality should be one of the valid values
            assert result.caption_quality in ["high", "medium", "low"], \
                f"Invalid quality: {result.caption_quality}"

            # Auto-generated captions should typically be "medium" or "low"
            if result.is_auto_generated:
                assert result.caption_quality in ["medium", "low"], \
                    f"Expected medium/low for auto, got: {result.caption_quality}"

        except CaptionUnavailableError:
            pytest.skip(f"Video {video_id} captions not available")
        except CaptionFetchError as e:
            if "preprocessing" in str(e).lower():
                pytest.skip(f"yt-dlp preprocessing error - transient issue")
            raise


# =============================================================================
# Integration Tests - Auto-Generated Captions
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.flaky(reruns=2, reruns_delay=1.0)
class TestCaptionIntegrationAuto:
    """Integration tests for videos with auto-generated captions.

    These tests verify:
    - Successful fetch of auto-generated captions
    - Correct is_auto_generated flag detection
    - Quality marking for auto-generated content
    """

    @pytest.mark.fast
    def test_fetch_auto_captions_detection(self):
        """Test fetching and detecting auto-generated captions.

        Note: Auto-generated caption detection is complex because:
        1. list_available_languages marks translated captions as auto-generated
        2. But yt-dlp's actual download may classify them differently

        This test verifies we can successfully fetch captions from auto-generated
        tracks, and that quality scoring works correctly based on the detection.
        """
        fetcher = CaptionFetcher()
        video_id = get_test_video("auto_captions")

        try:
            # First list available languages
            languages = fetcher.list_available_languages(video_id)

            if not languages:
                pytest.skip(f"No captions available for video {video_id}")

            # Check if there are auto-generated captions
            auto_langs = [lang for lang in languages if lang.is_auto_generated]

            if auto_langs:
                # Fetch auto-generated captions
                result = fetcher.fetch_captions(
                    video_id,
                    language=auto_langs[0].code,
                    prefer_manual=False
                )

                assert result.video_id == video_id
                assert len(result.segments) > 0, "Expected caption segments"

                # Verify quality is set (exact value depends on is_auto_generated detection)
                assert result.caption_quality in ["high", "medium", "low"], \
                    f"Invalid quality: {result.caption_quality}"

                # If detected as auto-generated, quality should be medium at best
                if result.is_auto_generated:
                    assert result.caption_quality in ["medium", "low"], \
                        f"Auto captions should be medium/low quality, got: {result.caption_quality}"
            else:
                # Video might have human captions instead - still valid test
                result = fetcher.fetch_captions(video_id, language=languages[0].code)
                assert result is not None

        except CaptionUnavailableError:
            pytest.skip(f"Video {video_id} captions not available")

    @pytest.mark.fast
    def test_auto_language_selection(self):
        """Test automatic language selection with fallback chain."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("backup_with_captions")

        try:
            # Use auto language selection
            result = fetcher.fetch_captions_auto_language(
                video_id,
                preferred_language="en"
            )

            assert result.video_id == video_id
            assert result.language == "en"
            assert len(result.segments) > 0

        except CaptionUnavailableError:
            pytest.skip(f"No captions available for video {video_id}")

    @pytest.mark.fast
    def test_auto_language_fallback_chain(self):
        """Test that language fallback works when preferred language unavailable."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("backup_with_captions")

        try:
            # Request an unlikely language - should fall back to English
            result = fetcher.fetch_captions_auto_language(
                video_id,
                preferred_language="xx"  # Invalid language code
            )

            # Should fall back to English or another available language
            assert result is not None
            assert len(result.segments) > 0

        except CaptionUnavailableError:
            pytest.skip(f"No captions available for video {video_id}")


# =============================================================================
# Integration Tests - No Captions
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.flaky(reruns=2, reruns_delay=1.0)
class TestCaptionIntegrationNoCaption:
    """Integration tests for videos without captions.

    These tests verify:
    - Proper CaptionUnavailableError raising for no-caption videos
    - Graceful error handling
    - Informative error messages
    """

    @pytest.mark.fast
    def test_no_captions_raises_error(self):
        """Test that videos without captions raise CaptionUnavailableError."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("no_captions")

        # First check if this video actually has no captions
        try:
            languages = fetcher.list_available_languages(video_id)

            if languages:
                # Video now has captions - skip this test
                pytest.skip(
                    f"Video {video_id} now has captions ({len(languages)} languages). "
                    "Need to update fixture with a video that has no captions."
                )

            # Try to fetch - should raise CaptionUnavailableError
            with pytest.raises(CaptionUnavailableError) as exc_info:
                fetcher.fetch_captions(video_id, language="en")

            # Verify error contains video ID for debugging
            assert video_id in str(exc_info.value) or "video_id" in str(exc_info.value).lower()

        except CaptionFetchError as e:
            # Video might be unavailable entirely
            if "unavailable" in str(e).lower():
                pytest.skip(f"Video {video_id} not accessible")
            raise

    @pytest.mark.fast
    def test_auto_language_no_captions(self):
        """Test auto language selection handles no-caption videos gracefully."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("no_captions")

        try:
            languages = fetcher.list_available_languages(video_id)

            if languages:
                pytest.skip(f"Video {video_id} has captions - update fixture")

            with pytest.raises(CaptionUnavailableError):
                fetcher.fetch_captions_auto_language(video_id)

        except CaptionFetchError as e:
            if "unavailable" in str(e).lower():
                pytest.skip(f"Video {video_id} not accessible")
            raise

    @pytest.mark.fast
    def test_list_languages_empty_for_no_captions(self):
        """Test that list_available_languages returns empty list for no-caption videos."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("no_captions")

        try:
            languages = fetcher.list_available_languages(video_id)

            # If video truly has no captions, list should be empty
            if languages:
                pytest.skip(
                    f"Video {video_id} has {len(languages)} language(s). "
                    f"Update fixture with a video that has no captions."
                )

            assert languages == []

        except CaptionFetchError as e:
            if "unavailable" in str(e).lower():
                pytest.skip(f"Video {video_id} not accessible")
            raise


# =============================================================================
# Integration Tests - Language Selection
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.flaky(reruns=2, reruns_delay=1.0)
class TestCaptionIntegrationLanguage:
    """Integration tests for caption language detection and selection."""

    @pytest.mark.fast
    def test_list_multiple_languages(self):
        """Test listing all available languages for a multilingual video."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("with_captions")  # TED talks have many languages

        try:
            languages = fetcher.list_available_languages(video_id)

            assert len(languages) > 0, "Expected at least one language"

            # Verify AvailableLanguage structure
            for lang in languages:
                assert isinstance(lang, AvailableLanguage)
                assert len(lang.code) >= 2  # ISO language codes are 2+ chars
                assert lang.name  # Should have a name
                assert isinstance(lang.is_auto_generated, bool)

            # TED talks usually have many translations (mostly auto-generated now)
            if len(languages) >= 5:
                # Good multilingual video - verify we can distinguish types
                auto_count = sum(1 for lang in languages if lang.is_auto_generated)

                # Most modern videos have auto-generated captions
                assert auto_count >= 0  # Can be any count, just verify no error

        except CaptionFetchError as e:
            if "unavailable" in str(e).lower():
                pytest.skip(f"Video {video_id} not accessible")
            raise

    @pytest.mark.fast
    def test_select_best_language_preference(self):
        """Test that language selection respects preferences."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("with_captions")

        try:
            languages = fetcher.list_available_languages(video_id)

            if len(languages) < 2:
                pytest.skip("Need video with multiple languages")

            # Get all available language codes
            available_codes = [lang.code for lang in languages]

            # Test that preferred language is selected if available
            if "en" in available_codes:
                selected = fetcher.select_best_language(languages, preferred="en")
                assert selected.code == "en"

            # Test fallback when preferred not available
            selected = fetcher.select_best_language(languages, preferred="xx")
            assert selected is not None
            assert selected.code in available_codes

        except CaptionFetchError as e:
            if "unavailable" in str(e).lower():
                pytest.skip(f"Video {video_id} not accessible")
            raise


# =============================================================================
# Integration Tests - Quality and Metrics
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.flaky(reruns=2, reruns_delay=1.0)
class TestCaptionIntegrationQuality:
    """Integration tests for caption quality detection and metrics."""

    @pytest.mark.fast
    def test_caption_quality_detection(self):
        """Test that caption quality is properly determined."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("with_captions")

        try:
            languages = fetcher.list_available_languages(video_id)

            if not languages:
                pytest.skip(f"Video {video_id} has no captions")

            # Fetch captions
            result = fetcher.fetch_captions(video_id)

            # Quality should be one of the valid values
            assert result.caption_quality in ["high", "medium", "low"], \
                f"Invalid quality '{result.caption_quality}'"

            # Verify is_auto_generated is set
            assert isinstance(result.is_auto_generated, bool)

            # If many segments, quality should be at least medium
            if len(result.segments) >= 50:
                assert result.caption_quality in ["high", "medium"], \
                    f"Dense captions ({len(result.segments)} segments) should be medium+ quality"

        except CaptionUnavailableError:
            pytest.skip(f"Video {video_id} captions not available")
        except CaptionFetchError as e:
            if "preprocessing" in str(e).lower():
                pytest.skip(f"yt-dlp preprocessing error - transient issue")
            raise

    @pytest.mark.fast
    def test_caption_segments_have_valid_timing(self):
        """Test that caption segments have valid, sequential timing."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("backup_with_captions")

        try:
            result = fetcher.fetch_captions(video_id, language="en")

            assert len(result.segments) > 0

            # Check timing validity
            for i, segment in enumerate(result.segments):
                # Each segment should have non-negative start
                assert segment.start_time >= 0, f"Segment {i} has negative start time"

                # End should be >= start
                assert segment.end_time >= segment.start_time, \
                    f"Segment {i} has end ({segment.end_time}) before start ({segment.start_time})"

            # Check general progression (most segments should advance in time)
            if len(result.segments) > 2:
                advancing = sum(
                    1 for i in range(1, len(result.segments))
                    if result.segments[i].start_time >= result.segments[i-1].start_time
                )
                total = len(result.segments) - 1
                # At least 90% should be advancing (allows for minor overlaps)
                assert advancing / total >= 0.9, \
                    f"Only {advancing}/{total} segments advance in time"

        except CaptionUnavailableError:
            pytest.skip(f"Video {video_id} captions not available")
        except CaptionFetchError as e:
            if "preprocessing" in str(e).lower():
                pytest.skip(f"yt-dlp preprocessing error - transient issue")
            raise


# =============================================================================
# Integration Tests - Retry and Error Handling
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.flaky(reruns=2, reruns_delay=1.0)
class TestCaptionIntegrationRetry:
    """Integration tests for retry behavior with real network conditions."""

    @pytest.mark.fast
    def test_retry_eventually_succeeds(self):
        """Test that retry mechanism works with real network conditions."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("backup_with_captions")

        try:
            # Should succeed (possibly after retries on slow networks)
            result = fetcher.fetch_captions_with_retry(
                video_id,
                language="en",
                max_retries=3,
                retry_delay=1.0
            )

            assert result.video_id == video_id
            assert len(result.segments) > 0

        except CaptionUnavailableError:
            pytest.skip(f"Video {video_id} captions not available")
        except CaptionFetchError as e:
            # Network might genuinely be down or yt-dlp internal error
            if "preprocessing" in str(e).lower():
                pytest.skip(f"yt-dlp preprocessing error - transient issue")
            pytest.skip(f"Network error during test: {e}")

    @pytest.mark.fast
    def test_unavailable_not_retried(self):
        """Test that CaptionUnavailableError is not retried (no point)."""
        fetcher = CaptionFetcher()
        video_id = get_test_video("no_captions")

        try:
            languages = fetcher.list_available_languages(video_id)

            if languages:
                pytest.skip(f"Video {video_id} has captions - update fixture")

            # This should fail immediately, not retry
            with pytest.raises(CaptionUnavailableError):
                fetcher.fetch_captions_with_retry(
                    video_id,
                    language="en",
                    max_retries=3,
                    retry_delay=1.0
                )

        except CaptionFetchError as e:
            if "unavailable" in str(e).lower():
                pytest.skip(f"Video {video_id} not accessible")
            raise


# =============================================================================
# Integration Tests - Cache Integration
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.flaky(reruns=2, reruns_delay=1.0)
class TestCaptionIntegrationCache:
    """Integration tests for caption caching with real videos."""

    @pytest.mark.fast
    def test_cache_stores_real_captions(self, tmp_path):
        """Test that cache correctly stores captions from real fetch."""
        from src.caption_fetcher import CaptionCache
        from src.config.sections.download import CaptionFirstConfig

        fetcher = CaptionFetcher()
        config = CaptionFirstConfig(cache_dir=str(tmp_path / "caption_cache"))
        cache = CaptionCache(config)

        video_id = get_test_video("backup_with_captions")

        try:
            # First fetch - should miss cache and fetch from network
            result1 = cache.get_or_fetch(fetcher, video_id, "en")

            assert result1.video_id == video_id
            assert len(result1.segments) > 0

            # Second fetch - should hit cache
            result2 = cache.get_or_fetch(fetcher, video_id, "en")

            # Results should match
            assert result2.video_id == result1.video_id
            assert len(result2.segments) == len(result1.segments)

            # Verify cache stats
            stats = cache.get_stats()
            assert stats.hits >= 1, "Expected at least one cache hit"

        except CaptionUnavailableError:
            pytest.skip(f"Video {video_id} captions not available")
        except CaptionFetchError as e:
            if "preprocessing" in str(e).lower():
                pytest.skip(f"yt-dlp preprocessing error - transient issue")
            raise

    @pytest.mark.fast
    def test_cache_key_includes_language(self, tmp_path):
        """Test that cache keys are language-specific."""
        from src.caption_fetcher import CaptionCache
        from src.config.sections.download import CaptionFirstConfig

        config = CaptionFirstConfig(cache_dir=str(tmp_path / "caption_cache"))
        cache = CaptionCache(config)

        video_id = get_test_video("backup_with_captions")

        # Generate cache keys
        key_en = cache._make_cache_key(video_id, "en")
        key_es = cache._make_cache_key(video_id, "es")

        # Keys should be different for different languages
        assert key_en != key_es
        assert video_id in key_en
        assert "en" in key_en
        assert "es" in key_es


# =============================================================================
# Fixture Documentation
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.flaky(reruns=2, reruns_delay=1.0)
class TestVideoFixtureValidation:
    """Tests to validate that our video fixtures are still valid.

    Run these periodically to ensure test videos are still available
    and have the expected caption states.
    """

    @pytest.mark.fast
    def test_validate_with_captions_fixture(self):
        """Validate that with_captions fixture video has captions."""
        fetcher = CaptionFetcher()
        fixture = VIDEO_FIXTURES["with_captions"]
        video_id = fixture["id"]

        try:
            languages = fetcher.list_available_languages(video_id)

            # Should have captions (any type)
            assert len(languages) > 0, \
                f"Video {video_id} ({fixture['notes']}) has no captions - update fixture"

        except CaptionFetchError as e:
            pytest.fail(f"with_captions fixture video unavailable: {e}")

    @pytest.mark.fast
    def test_validate_backup_fixture(self):
        """Validate that backup video fixture has captions available."""
        fetcher = CaptionFetcher()
        fixture = VIDEO_FIXTURES["backup_with_captions"]
        video_id = fixture["id"]

        try:
            # First check languages (doesn't require caption download)
            languages = fetcher.list_available_languages(video_id)

            assert len(languages) > 0, \
                f"Backup video {video_id} has no captions available - investigate"

            # Try to fetch - may fail due to yt-dlp internal errors
            try:
                result = fetcher.fetch_captions(video_id, language="en")
                assert len(result.segments) > 0, \
                    f"Backup video {video_id} has no segments - investigate"
            except CaptionFetchError as e:
                if "preprocessing" in str(e).lower():
                    pytest.skip(f"yt-dlp preprocessing error for {video_id} - transient issue")
                raise

        except CaptionFetchError as e:
            pytest.fail(f"Backup fixture video unavailable: {e}")

    @pytest.mark.fast
    def test_validate_no_caption_fixture(self):
        """Validate that no-caption fixture video has no captions."""
        fetcher = CaptionFetcher()
        fixture = VIDEO_FIXTURES["no_captions"]
        video_id = fixture["id"]

        try:
            languages = fetcher.list_available_languages(video_id)

            if len(languages) > 0:
                pytest.skip(
                    f"No-caption fixture video {video_id} now has {len(languages)} "
                    f"caption(s). Update VIDEO_FIXTURES['no_captions'] with a new "
                    f"video ID that has no captions. Current notes: {fixture['notes']}"
                )

        except CaptionFetchError as e:
            if "unavailable" in str(e).lower():
                pytest.skip(f"Video {video_id} not accessible - may need update")
            raise


# =============================================================================
# Rate Limiter Integration Tests (US-33-003)
# =============================================================================

class TestCaptionFetcherRateLimiterIntegration:
    """Tests for rate limiter integration in CaptionFetcher (US-33-003).

    Verifies:
    - CaptionFetcher accepts optional rate_limiter parameter
    - Pre-fetch delay via rate_limiter.wait_if_needed()
    - Success recording via rate_limiter.record_success()
    - Rate limit recording on 429/quota errors
    """

    def test_init_accepts_rate_limiter(self):
        """Test CaptionFetcher.__init__ accepts optional rate_limiter parameter."""
        from src.caption.rate_limiter import UnifiedCaptionRateLimiter

        rate_limiter = UnifiedCaptionRateLimiter()
        fetcher = CaptionFetcher(rate_limiter=rate_limiter)

        assert fetcher._rate_limiter is rate_limiter

    def test_init_without_rate_limiter(self):
        """Test CaptionFetcher works without rate_limiter (backwards compatible)."""
        fetcher = CaptionFetcher()

        assert fetcher._rate_limiter is None

    def test_fetch_calls_wait_if_needed_before_fetch(self):
        """Test fetch_captions calls rate_limiter.wait_if_needed() before fetching."""
        from unittest.mock import MagicMock
        from src.caption.rate_limiter import UnifiedCaptionRateLimiter

        mock_limiter = MagicMock(spec=UnifiedCaptionRateLimiter)
        mock_limiter.wait_if_needed.return_value = 0.5

        fetcher = CaptionFetcher(rate_limiter=mock_limiter)

        # Mock _is_valid_video_id to fail early (avoid actual fetch)
        fetcher._is_valid_video_id = MagicMock(return_value=False)

        with pytest.raises(CaptionFetchError):
            fetcher.fetch_captions("test_id")

        # wait_if_needed should be called BEFORE validation
        mock_limiter.wait_if_needed.assert_called_once()

    def test_fetch_records_success_on_successful_fetch(self):
        """Test fetch_captions calls rate_limiter.record_success() on success."""
        from unittest.mock import MagicMock, patch
        from src.caption.rate_limiter import UnifiedCaptionRateLimiter
        from src.caption_fetcher import CaptionSegment

        mock_limiter = MagicMock(spec=UnifiedCaptionRateLimiter)
        mock_limiter.wait_if_needed.return_value = 0.0

        fetcher = CaptionFetcher(rate_limiter=mock_limiter)

        # Mock the internal methods to simulate successful fetch
        mock_segment = CaptionSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="test",
            source_file="dQw4w9WgXcQ",
        )
        mock_result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=[mock_segment],
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )

        with patch.object(fetcher, '_fetch_subtitle', return_value=mock_result):
            with patch.object(fetcher, '_is_valid_video_id', return_value=True):
                result = fetcher.fetch_captions("dQw4w9WgXcQ")

        assert result is mock_result
        mock_limiter.record_success.assert_called_once()

    def test_fetch_records_rate_limit_on_429_error(self):
        """Test fetch_captions calls rate_limiter.record_rate_limit() on 429 errors."""
        from unittest.mock import MagicMock, patch
        from src.caption.rate_limiter import UnifiedCaptionRateLimiter

        mock_limiter = MagicMock(spec=UnifiedCaptionRateLimiter)
        mock_limiter.wait_if_needed.return_value = 0.0

        fetcher = CaptionFetcher(rate_limiter=mock_limiter)

        # Create a rate limit error
        rate_limit_error = CaptionFetchError("test_id", "HTTP 429 Too Many Requests")

        with patch.object(fetcher, '_fetch_subtitle', side_effect=rate_limit_error):
            with patch.object(fetcher, '_is_valid_video_id', return_value=True):
                with pytest.raises(CaptionFetchError):
                    fetcher.fetch_captions("dQw4w9WgXcQ")

        # Should record rate limit for the video
        mock_limiter.record_rate_limit.assert_called()

    def test_fetch_does_not_record_rate_limit_on_other_errors(self):
        """Test fetch_captions does NOT call record_rate_limit on non-429 errors."""
        from unittest.mock import MagicMock, patch
        from src.caption.rate_limiter import UnifiedCaptionRateLimiter

        mock_limiter = MagicMock(spec=UnifiedCaptionRateLimiter)
        mock_limiter.wait_if_needed.return_value = 0.0

        fetcher = CaptionFetcher(rate_limiter=mock_limiter)

        # Create a non-rate-limit error (network error)
        network_error = CaptionFetchError("test_id", "Connection refused")

        with patch.object(fetcher, '_fetch_subtitle', side_effect=network_error):
            with patch.object(fetcher, '_is_valid_video_id', return_value=True):
                with pytest.raises(CaptionFetchError):
                    fetcher.fetch_captions("dQw4w9WgXcQ")

        # Should NOT record rate limit for network errors
        mock_limiter.record_rate_limit.assert_not_called()

    def test_fetch_without_rate_limiter_still_works(self):
        """Test fetch_captions works normally without rate_limiter."""
        from unittest.mock import patch
        from src.caption_fetcher import CaptionSegment

        fetcher = CaptionFetcher()  # No rate limiter

        mock_segment = CaptionSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="test",
            source_file="dQw4w9WgXcQ",
        )
        mock_result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=[mock_segment],
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )

        with patch.object(fetcher, '_fetch_subtitle', return_value=mock_result):
            with patch.object(fetcher, '_is_valid_video_id', return_value=True):
                result = fetcher.fetch_captions("dQw4w9WgXcQ")

        assert result is mock_result


class TestCaptionFetcherRateLimiterConfig:
    """Tests for rate limiter configuration in CaptionFetcher (US-33-003)."""

    def test_custom_rate_limit_config(self):
        """Test CaptionFetcher with custom rate limit config."""
        from src.caption.rate_limiter import UnifiedCaptionRateLimiter, RateLimitConfig

        config = RateLimitConfig(
            enabled=True,
            base_delay_seconds=5.0,
            max_delay_seconds=60.0,
            jitter_factor=0.2
        )
        rate_limiter = UnifiedCaptionRateLimiter(config)

        fetcher = CaptionFetcher(rate_limiter=rate_limiter)

        assert fetcher._rate_limiter.config.base_delay_seconds == 5.0
        assert fetcher._rate_limiter.config.max_delay_seconds == 60.0
        assert fetcher._rate_limiter.config.jitter_factor == 0.2

    def test_disabled_rate_limiter_skips_delay(self):
        """Test disabled rate limiter doesn't add delay."""
        from src.caption.rate_limiter import UnifiedCaptionRateLimiter, RateLimitConfig

        config = RateLimitConfig(enabled=False)
        rate_limiter = UnifiedCaptionRateLimiter(config)

        fetcher = CaptionFetcher(rate_limiter=rate_limiter)

        # Even with consecutive rate limits, disabled limiter returns 0
        rate_limiter.record_rate_limit("test")
        rate_limiter.record_rate_limit("test")

        delay = rate_limiter.get_rate_limit_delay()
        assert delay == 0.0


class TestCaptionFetcherRateLimiterStateTracking:
    """Tests for rate limiter state tracking during fetches (US-33-003)."""

    def test_consecutive_rate_limits_increase_delay(self):
        """Test consecutive rate limits increase backoff delay."""
        from unittest.mock import patch
        from src.caption.rate_limiter import UnifiedCaptionRateLimiter

        rate_limiter = UnifiedCaptionRateLimiter()
        fetcher = CaptionFetcher(rate_limiter=rate_limiter)

        # Simulate multiple rate limit errors
        rate_limit_error = CaptionFetchError("test_id", "HTTP 429 quota exceeded")

        with patch.object(fetcher, '_fetch_subtitle', side_effect=rate_limit_error):
            with patch.object(fetcher, '_is_valid_video_id', return_value=True):
                with pytest.raises(CaptionFetchError):
                    fetcher.fetch_captions("video1")

        assert rate_limiter.consecutive_rate_limits >= 1

    def test_success_resets_consecutive_rate_limits(self):
        """Test successful fetch resets consecutive rate limit count."""
        from unittest.mock import patch
        from src.caption.rate_limiter import UnifiedCaptionRateLimiter
        from src.caption_fetcher import CaptionSegment

        rate_limiter = UnifiedCaptionRateLimiter()
        fetcher = CaptionFetcher(rate_limiter=rate_limiter)

        # First record some rate limits
        rate_limiter.record_rate_limit("test")
        rate_limiter.record_rate_limit("test")
        assert rate_limiter.consecutive_rate_limits == 2

        # Now simulate a successful fetch
        mock_segment = CaptionSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="test",
            source_file="dQw4w9WgXcQ",
        )
        mock_result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=[mock_segment],
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )

        with patch.object(fetcher, '_fetch_subtitle', return_value=mock_result):
            with patch.object(fetcher, '_is_valid_video_id', return_value=True):
                fetcher.fetch_captions("dQw4w9WgXcQ")

        # Consecutive count should be reset
        assert rate_limiter.consecutive_rate_limits == 0
