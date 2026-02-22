"""Integration tests for CaptionFetcher with YouTubeAPIClient (US-148-004).

Tests verify that CaptionFetcher uses YouTubeAPIClient.check_captions_available()
as first option, falls back to yt-dlp on API failure, and tracks metrics.

Run with:
    pytest tests/test_caption_fetcher_youtube_api.py -v

Pytest marker: fast
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Optional

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionMetrics,
    CaptionResult,
    CaptionSegment,
)
from src.downloader.youtube_api_client import (
    YouTubeAPIClient,
    CaptionInfo,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_youtube_api_client():
    """Create a mock YouTubeAPIClient."""
    mock_client = MagicMock(spec=YouTubeAPIClient)
    return mock_client


@pytest.fixture
def caption_metrics():
    """Create CaptionMetrics for testing."""
    return CaptionMetrics()


# ============================================================================
# Tests: CaptionFetcher with YouTubeAPIClient
# ============================================================================

class TestCaptionFetcherYouTubeAPI:
    """Tests for CaptionFetcher integration with YouTubeAPIClient."""

    def test_fetcher_accepts_youtube_api_client(self, mock_youtube_api_client):
        """CaptionFetcher should accept YouTubeAPIClient in constructor."""
        fetcher = CaptionFetcher(youtube_api_client=mock_youtube_api_client)
        assert fetcher._youtube_api_client is mock_youtube_api_client

    def test_check_caption_api_first_returns_languages_when_api_available(
        self, mock_youtube_api_client
    ):
        """_check_caption_api_first should return languages when API has captions."""
        # Setup mock to return caption info
        mock_youtube_api_client.check_captions_available.return_value = [
            CaptionInfo(language="en", track_id="test_track_1", is_auto_generated=False),
            CaptionInfo(language="es", track_id="test_track_2", is_auto_generated=True),
        ]

        fetcher = CaptionFetcher(youtube_api_client=mock_youtube_api_client)
        result = fetcher._check_caption_api_first("test_video_id")

        assert result == ["en", "es"]
        mock_youtube_api_client.check_captions_available.assert_called_once_with("test_video_id")

    def test_check_caption_api_first_returns_empty_when_no_captions(
        self, mock_youtube_api_client
    ):
        """_check_caption_api_first should return empty list when API shows no captions."""
        mock_youtube_api_client.check_captions_available.return_value = []

        fetcher = CaptionFetcher(youtube_api_client=mock_youtube_api_client)
        result = fetcher._check_caption_api_first("test_video_id")

        assert result == []

    def test_check_caption_api_first_falls_back_on_quota_exceeded(
        self, mock_youtube_api_client
    ):
        """_check_caption_api_first should fallback to yt-dlp when quota exceeded."""
        from src.downloader.youtube_api_client import QuotaExceededError
        mock_youtube_api_client.check_captions_available.side_effect = QuotaExceededError("Quota exceeded")

        fetcher = CaptionFetcher(youtube_api_client=mock_youtube_api_client)
        result = fetcher._check_caption_api_first("test_video_id")

        assert result is None  # Should fallback to yt-dlp

    def test_check_caption_api_first_falls_back_on_exception(
        self, mock_youtube_api_client
    ):
        """_check_caption_api_first should fallback to yt-dlp on any exception."""
        mock_youtube_api_client.check_captions_available.side_effect = Exception("API error")

        fetcher = CaptionFetcher(youtube_api_client=mock_youtube_api_client)
        result = fetcher._check_caption_api_first("test_video_id")

        assert result is None  # Should fallback to yt-dlp

    def test_check_caption_api_first_returns_none_when_no_client(self):
        """_check_caption_api_first should return None when no API client configured."""
        fetcher = CaptionFetcher(youtube_api_client=None)
        result = fetcher._check_caption_api_first("test_video_id")

        assert result is None


class TestCaptionFetcherYouTubeAPIMetrics:
    """Tests for CaptionFetcher metrics tracking with YouTubeAPIClient."""

    def test_metrics_record_api_check_available(self, caption_metrics):
        """CaptionMetrics should record API check with available captions."""
        caption_metrics.record_caption_api_check("video_1", ["en", "es"])

        assert caption_metrics.caption_api_checks == 1
        assert caption_metrics.caption_api_available == 1
        assert caption_metrics.caption_api_unavailable == 0
        assert caption_metrics.caption_api_fallback == 0

    def test_metrics_record_api_check_unavailable(self, caption_metrics):
        """CaptionMetrics should record API check with no captions."""
        caption_metrics.record_caption_api_check("video_1", [])

        assert caption_metrics.caption_api_checks == 1
        assert caption_metrics.caption_api_available == 0
        assert caption_metrics.caption_api_unavailable == 1
        assert caption_metrics.caption_api_fallback == 0

    def test_metrics_record_api_check_fallback(self, caption_metrics):
        """CaptionMetrics should record API fallback to yt-dlp."""
        caption_metrics.record_caption_api_check("video_1", None)

        assert caption_metrics.caption_api_checks == 1
        assert caption_metrics.caption_api_available == 0
        assert caption_metrics.caption_api_unavailable == 0
        assert caption_metrics.caption_api_fallback == 1

    def test_metrics_track_multiple_checks(self, caption_metrics):
        """CaptionMetrics should correctly track multiple API checks."""
        caption_metrics.record_caption_api_check("video_1", ["en"])
        caption_metrics.record_caption_api_check("video_2", [])
        caption_metrics.record_caption_api_check("video_3", None)  # Fallback

        assert caption_metrics.caption_api_checks == 3
        assert caption_metrics.caption_api_available == 1
        assert caption_metrics.caption_api_unavailable == 1
        assert caption_metrics.caption_api_fallback == 1


class TestCaptionFetcherYouTubeAPIIntegration:
    """Integration tests for CaptionFetcher using YouTubeAPIClient."""

    def test_fetcher_uses_api_when_available(self, mock_youtube_api_client):
        """CaptionFetcher should use API first when client is available."""
        # Setup mock to return caption info (API available)
        mock_youtube_api_client.check_captions_available.return_value = [
            CaptionInfo(language="en", track_id="test_track", is_auto_generated=False),
        ]

        fetcher = CaptionFetcher(youtube_api_client=mock_youtube_api_client)

        # Call the internal method directly
        result = fetcher._check_caption_api_first("test_video_123")

        # Verify API was called and returned languages
        mock_youtube_api_client.check_captions_available.assert_called_once_with("test_video_123")
        assert result == ["en"]

    def test_fetcher_falls_back_to_ytdlp_on_api_failure(self, mock_youtube_api_client):
        """CaptionFetcher should fallback to yt-dlp when API fails."""
        from src.downloader.youtube_api_client import QuotaExceededError
        mock_youtube_api_client.check_captions_available.side_effect = QuotaExceededError("Quota")

        fetcher = CaptionFetcher(youtube_api_client=mock_youtube_api_client)

        # Should return None to trigger yt-dlp fallback
        result = fetcher._check_caption_api_first("test_video_123")

        assert result is None
        mock_youtube_api_client.check_captions_available.assert_called_once()

    def test_fetcher_records_metrics_on_api_check(self, mock_youtube_api_client):
        """CaptionFetcher should record metrics when API check succeeds."""
        mock_youtube_api_client.check_captions_available.return_value = [
            CaptionInfo(language="en", track_id="test", is_auto_generated=False),
        ]

        # Create fetcher with metrics
        metrics = CaptionMetrics()
        fetcher = CaptionFetcher(
            youtube_api_client=mock_youtube_api_client,
        )
        # Set metrics on fetcher
        fetcher._active_metrics = metrics

        # Perform API check
        result = fetcher._check_caption_api_first("test_video_123")

        # Verify metrics recorded
        assert metrics.caption_api_checks == 1
        assert metrics.caption_api_available == 1

    def test_fetcher_records_metrics_on_api_unavailable(self, mock_youtube_api_client):
        """CaptionFetcher should record metrics when API reports no captions."""
        mock_youtube_api_client.check_captions_available.return_value = []

        metrics = CaptionMetrics()
        fetcher = CaptionFetcher(youtube_api_client=mock_youtube_api_client)
        fetcher._active_metrics = metrics

        result = fetcher._check_caption_api_first("test_video_123")

        assert metrics.caption_api_checks == 1
        assert metrics.caption_api_unavailable == 1
        assert metrics.caption_api_available == 0

    def test_fetcher_records_metrics_on_api_fallback(self, mock_youtube_api_client):
        """CaptionFetcher should record metrics when API fails and falls back to yt-dlp."""
        mock_youtube_api_client.check_captions_available.side_effect = Exception("API Error")

        metrics = CaptionMetrics()
        fetcher = CaptionFetcher(youtube_api_client=mock_youtube_api_client)
        fetcher._active_metrics = metrics

        result = fetcher._check_caption_api_first("test_video_123")

        assert metrics.caption_api_checks == 1
        assert metrics.caption_api_fallback == 1
