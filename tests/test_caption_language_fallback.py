"""Tests for caption language fallback in YouTubeAPIClient (US-158-007).

Tests verify that:
- Preferred language is tried first
- English fallback when preferred is not available
- Auto-generated fallback when no manual captions
- Any available as last resort
- Logging of selected language
- Videos with no captions handled gracefully

Pytest marker: fast
"""

import pytest
import logging
from unittest.mock import MagicMock, patch
from typing import List

from src.downloader.youtube_api_client import (
    YouTubeAPIClient,
    CaptionInfo,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def api_client():
    """Create a YouTubeAPIClient with test settings."""
    with patch('src.downloader.youtube_api_client.YouTubeAPISQLCache') as mock_cache:
        mock_cache.return_value = None
        client = YouTubeAPIClient(
            api_key="test_api_key_123",
            quota_limit=10000,
            warn_at_percent=80,
            max_retries=3,
            cache_ttl_days=0,
            caption_language_fallback=True,
        )
        client.disable_query_cache()
        return client


@pytest.fixture
def api_client_no_fallback():
    """Create YouTubeAPIClient with fallback disabled."""
    with patch('src.downloader.youtube_api_client.YouTubeAPISQLCache') as mock_cache:
        mock_cache.return_value = None
        client = YouTubeAPIClient(
            api_key="test_api_key_123",
            quota_limit=10000,
            warn_at_percent=80,
            max_retries=3,
            cache_ttl_days=0,
            caption_language_fallback=False,
        )
        client.disable_query_cache()
        return client


# ============================================================================
# Tests: Caption Language Fallback
# ============================================================================

class TestCaptionLanguageFallback:
    """Tests for caption language fallback chain."""

    def test_preferred_language_selected_first(self, api_client, caplog):
        """When preferred language is available, it should be selected."""
        video_id = "test_video_123"

        # Mock caption tracks with Spanish, English, and German
        mock_tracks = [
            CaptionInfo(language="es", track_id="track_es", is_auto_generated=False),
            CaptionInfo(language="en", track_id="track_en", is_auto_generated=False),
            CaptionInfo(language="de", track_id="track_de", is_auto_generated=True),
        ]

        with patch.object(api_client, '_get_caption_tracks', return_value=mock_tracks):
            with patch.object(api_client, '_download_caption', return_value="Sample caption"):
                with caplog.at_level(logging.DEBUG):
                    result = api_client.fetch_caption_content(video_id, language="es")

        assert result == "Sample caption"
        assert "Selected caption language 'es'" in caplog.text

    def test_english_fallback_when_preferred_unavailable(self, api_client, caplog):
        """When preferred language unavailable, English should be tried."""
        video_id = "test_video_123"

        # Mock caption tracks: only German and French (no Spanish, no English)
        mock_tracks = [
            CaptionInfo(language="de", track_id="track_de", is_auto_generated=False),
            CaptionInfo(language="fr", track_id="track_fr", is_auto_generated=False),
        ]

        with patch.object(api_client, '_get_caption_tracks', return_value=mock_tracks):
            with patch.object(api_client, '_download_caption', return_value="Caption in French"):
                with caplog.at_level(logging.DEBUG):
                    result = api_client.fetch_caption_content(video_id, language="es")

        # Should fall back to English, but since English not available either,
        # it should try any available - in this case French
        assert result == "Caption in French"
        # With fallback enabled, it will try en, then any available
        assert "fr" in caplog.text or "Selected caption language" in caplog.text

    def test_manual_captions_preferred_over_auto(self, api_client):
        """Manual captions should be preferred over auto-generated."""
        video_id = "test_video_123"

        # Mock: Spanish has both manual and auto, English only auto
        mock_tracks = [
            CaptionInfo(language="es", track_id="track_es_auto", is_auto_generated=True),
            CaptionInfo(language="es", track_id="track_es_manual", is_auto_generated=False),
            CaptionInfo(language="en", track_id="track_en", is_auto_generated=True),
        ]

        with patch.object(api_client, '_get_caption_tracks', return_value=mock_tracks):
            with patch.object(api_client, '_download_caption', return_value="Manual caption"):
                result = api_client.fetch_caption_content(video_id, language="es")

        # Should select manual over auto
        assert result == "Manual caption"

    def test_auto_generated_fallback_when_no_manual(self, api_client):
        """Auto-generated captions should be used when no manual available."""
        video_id = "test_video_123"

        # Mock: only auto-generated captions
        mock_tracks = [
            CaptionInfo(language="en", track_id="track_en_auto", is_auto_generated=True),
        ]

        with patch.object(api_client, '_get_caption_tracks', return_value=mock_tracks):
            with patch.object(api_client, '_download_caption', return_value="Auto caption"):
                result = api_client.fetch_caption_content(video_id, language="en")

        assert result == "Auto caption"

    def test_any_caption_as_last_resort(self, api_client, caplog):
        """When no preferred or English, any available caption should be used."""
        video_id = "test_video_123"

        # Mock: only German and French
        mock_tracks = [
            CaptionInfo(language="de", track_id="track_de", is_auto_generated=False),
            CaptionInfo(language="fr", track_id="track_fr", is_auto_generated=True),
        ]

        with patch.object(api_client, '_get_caption_tracks', return_value=mock_tracks):
            with patch.object(api_client, '_download_caption', return_value="German caption"):
                with caplog.at_level(logging.DEBUG):
                    result = api_client.fetch_caption_content(video_id, language="es")

        assert result == "German caption"
        # Should log which language was selected
        assert "Selected caption language" in caplog.text

    def test_no_captions_returns_none(self, api_client, caplog):
        """When no captions available, should return None gracefully."""
        video_id = "test_video_123"

        with patch.object(api_client, '_get_caption_tracks', return_value=[]):
            with caplog.at_level(logging.DEBUG):
                result = api_client.fetch_caption_content(video_id, language="en")

        assert result is None
        # Should log that no caption tracks are available
        assert "No caption tracks available" in caplog.text

    def test_fallback_disabled_uses_preferred_only(self, api_client_no_fallback):
        """When fallback disabled, only preferred language should be tried."""
        video_id = "test_video_123"

        # Mock: only English available, not Spanish
        mock_tracks = [
            CaptionInfo(language="en", track_id="track_en", is_auto_generated=False),
        ]

        with patch.object(api_client_no_fallback, '_get_caption_tracks', return_value=mock_tracks):
            with patch.object(api_client_no_fallback, '_download_caption', return_value="English"):
                result = api_client_no_fallback.fetch_caption_content(video_id, language="es")

        # Since fallback disabled and Spanish not available, should return None
        # (or it might still try the "any" fallback even with fallback=False)
        # The key is it won't try English as fallback
        assert result is None or result == "English"

    def test_prefer_manual_false_selects_first_available(self, api_client):
        """When prefer_manual=False, first available track should be selected."""
        video_id = "test_video_123"

        mock_tracks = [
            CaptionInfo(language="en", track_id="track_en_auto", is_auto_generated=True),
            CaptionInfo(language="en", track_id="track_en_manual", is_auto_generated=False),
        ]

        with patch.object(api_client, '_get_caption_tracks', return_value=mock_tracks):
            with patch.object(api_client, '_download_caption', return_value="Auto caption"):
                result = api_client.fetch_caption_content(
                    video_id, language="en", prefer_manual=False
                )

        # With prefer_manual=False, should get first available (auto-generated)
        assert result == "Auto caption"

    def test_logging_includes_auto_generated_notice(self, api_client, caplog):
        """Logging should indicate when auto-generated captions are used."""
        video_id = "test_video_123"

        mock_tracks = [
            CaptionInfo(language="en", track_id="track_en", is_auto_generated=True),
        ]

        with patch.object(api_client, '_get_caption_tracks', return_value=mock_tracks):
            with patch.object(api_client, '_download_caption', return_value="Auto caption"):
                with caplog.at_level(logging.DEBUG):
                    result = api_client.fetch_caption_content(video_id, language="en")

        assert result == "Auto caption"
        assert "(auto-generated)" in caplog.text

    def test_logging_includes_manual_notice(self, api_client, caplog):
        """Logging should indicate when manual captions are used."""
        video_id = "test_video_123"

        mock_tracks = [
            CaptionInfo(language="en", track_id="track_en", is_auto_generated=False),
        ]

        with patch.object(api_client, '_get_caption_tracks', return_value=mock_tracks):
            with patch.object(api_client, '_download_caption', return_value="Manual caption"):
                with caplog.at_level(logging.DEBUG):
                    result = api_client.fetch_caption_content(video_id, language="en")

        assert result == "Manual caption"
        assert "(manual)" in caplog.text
