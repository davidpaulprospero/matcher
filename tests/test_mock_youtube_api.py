"""Tests for MockYouTubeAPIClient (US-151-005, US-154-012).

Tests the mock YouTube API client that returns canned responses
without making actual API calls. Enable via MOCK_YOUTUBE_API=1.

Also tests fixture loading from files (US-154-012).

Pytest marker: fast
"""

import os
import pytest
from pathlib import Path
from typing import List

from src.testing.mocks.youtube_api import (
    MockYouTubeAPIClient,
    get_mock_client,
    is_mock_enabled,
    create_youtube_api_client,
    DEFAULT_MOCK_VIDEO_IDS,
    DEFAULT_FIXTURE_DIR,
)
from src.downloader.youtube_api_client import VideoSearchResult, VideoDetails, CaptionInfo


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_client():
    """Create a MockYouTubeAPIClient instance."""
    return get_mock_client()


# ============================================================================
# Tests - Core Functionality
# ============================================================================

class TestMockYouTubeAPIClient:
    """Tests for MockYouTubeAPIClient class."""

    def test_search_videos_returns_results(self, mock_client):
        """Test that search_videos returns mock results."""
        results = mock_client.search_videos("test query", max_results=10)

        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 10

        # Check result structure
        first = results[0]
        assert isinstance(first, VideoSearchResult)
        assert first.video_id in DEFAULT_MOCK_VIDEO_IDS
        assert first.title
        assert first.channel_id
        assert first.channel_title

    def test_search_videos_respects_max_results(self, mock_client):
        """Test that max_results parameter is respected."""
        results = mock_client.search_videos("test query", max_results=2)

        assert len(results) == 2

    def test_search_videos_caches_results(self, mock_client):
        """Test that search results are cached per query."""
        query = "unique_query_12345"

        # First call
        results1 = mock_client.search_videos(query, max_results=5)

        # Second call should return same results
        results2 = mock_client.search_videos(query, max_results=5)

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.video_id == r2.video_id
            assert r1.title == r2.title

    def test_get_video_details_returns_mock_data(self, mock_client):
        """Test that get_video_details returns mock data."""
        video_ids = ["dQw4w9WgXcQ", "jNQXAC9IVRw"]

        details = mock_client.get_video_details(video_ids)

        # Now returns Dict[str, VideoDetails]
        assert len(details) == 2
        for detail in details.values():
            assert isinstance(detail, VideoDetails)
            assert detail.video_id in video_ids
            assert detail.duration_seconds > 0
            assert detail.caption_available is True

    def test_check_captions_available_returns_mock_data(self, mock_client):
        """Test that check_captions_available returns mock captions."""
        video_id = "dQw4w9WgXcQ"

        captions = mock_client.check_captions_available(video_id)

        assert len(captions) > 0
        assert all(isinstance(c, CaptionInfo) for c in captions)
        # Should include English and Spanish
        languages = [c.language for c in captions]
        assert "en" in languages

    def test_get_remaining_quota_returns_full(self, mock_client):
        """Test that get_remaining_quota returns full limit."""
        remaining = mock_client.get_remaining_quota()

        assert remaining == mock_client.quota_limit
        assert remaining == 10000

    def test_get_quota_status_returns_mock_status(self, mock_client):
        """Test that get_quota_status returns mock status."""
        status = mock_client.get_quota_status()

        assert status["remaining"] == 10000
        assert status["limit"] == 10000
        assert status["used"] == 0
        assert status["mock_mode"] is True

    def test_call_count_tracking(self, mock_client):
        """Test that call counts are tracked."""
        # Reset counts
        mock_client.reset_call_counts()

        # Make some calls
        mock_client.search_videos("query1")
        mock_client.search_videos("query2")
        mock_client.get_video_details(["vid1"])
        mock_client.check_captions_available("vid1")

        # Check metrics
        metrics = mock_client.get_api_metrics()

        assert metrics["search_calls"] == 2
        assert metrics["videos_calls"] == 1
        assert metrics["captions_calls"] == 1
        assert metrics["mock_mode"] is True


class TestMockClientCustomization:
    """Tests for customizing mock client with custom data."""

    def test_set_custom_search_results(self, mock_client):
        """Test setting custom search results."""
        custom_results = [
            VideoSearchResult(
                video_id="custom_vid_1",
                title="Custom Video 1",
                channel_id="custom_channel",
                channel_title="Custom Channel",
                published_at="2024-01-01T00:00:00Z",
                description="Custom description",
            ),
        ]

        mock_client.set_search_results("my_custom_query", custom_results)

        results = mock_client.search_videos("my_custom_query")

        assert len(results) == 1
        assert results[0].video_id == "custom_vid_1"
        assert results[0].title == "Custom Video 1"

    def test_set_custom_video_details(self, mock_client):
        """Test setting custom video details."""
        custom_details = VideoDetails(
            video_id="custom_vid",
            duration="PT5M",
            duration_seconds=300,
            tags=["custom", "test"],
            category_id="1",
            caption_available=False,
        )

        mock_client.set_video_details("custom_vid", custom_details)

        details = mock_client.get_video_details(["custom_vid"])

        assert len(details) == 1
        assert details[0].video_id == "custom_vid"
        assert details[0].duration_seconds == 300
        assert details[0].caption_available is False


class TestMockEnvironmentVariable:
    """Tests for MOCK_YOUTUBE_API environment variable."""

    def test_is_mock_enabled_when_set(self, monkeypatch):
        """Test that is_mock_enabled returns True when set."""
        monkeypatch.setenv("MOCK_YOUTUBE_API", "1")

        assert is_mock_enabled() is True

    def test_is_mock_enabled_when_not_set(self, monkeypatch):
        """Test that is_mock_enabled returns False when not set."""
        monkeypatch.delenv("MOCK_YOUTUBE_API", raising=False)

        assert is_mock_enabled() is False

    def test_is_mock_enabled_when_set_to_0(self, monkeypatch):
        """Test that is_mock_enabled returns False when set to 0."""
        monkeypatch.setenv("MOCK_YOUTUBE_API", "0")

        assert is_mock_enabled() is False

    def test_create_youtube_api_client_returns_mock_when_enabled(self, monkeypatch):
        """Test that create_youtube_api_client returns mock when enabled."""
        monkeypatch.setenv("MOCK_YOUTUBE_API", "1")

        client = create_youtube_api_client("any_key")

        assert isinstance(client, MockYouTubeAPIClient)

    def test_create_youtube_api_client_returns_real_when_disabled(self, monkeypatch):
        """Test that create_youtube_api_client returns real client when disabled."""
        monkeypatch.setenv("MOCK_YOUTUBE_API", "0")

        # Import here to avoid issues when real client can't be created
        from src.downloader.youtube_api_client import YouTubeAPIClient

        # This will fail because we need a valid API key
        # but we're testing that it doesn't return MockYouTubeAPIClient
        # We'll just verify the mock doesn't return itself
        client = create_youtube_api_client("test_key_12345")
        # When disabled, it should try to return real client (may fail on init)
        # but shouldn't return mock
        assert not isinstance(client, MockYouTubeAPIClient)


class TestDefaultMockVideoIds:
    """Tests for default mock video IDs."""

    def test_default_video_ids_exist(self):
        """Test that default mock video IDs are defined."""
        assert len(DEFAULT_MOCK_VIDEO_IDS) > 0

    def test_default_video_ids_are_strings(self):
        """Test that default video IDs are valid YouTube ID format."""
        for vid in DEFAULT_MOCK_VIDEO_IDS:
            assert isinstance(vid, str)
            assert len(vid) == 11  # YouTube IDs are 11 chars


class TestFixtureLoading:
    """Tests for fixture loading (US-154-012)."""

    def test_default_fixture_dir_exists(self):
        """Test that default fixture directory exists."""
        assert DEFAULT_FIXTURE_DIR.exists()

    def test_load_fixtures_from_directory(self):
        """Test loading fixtures from test fixture directory."""
        client = MockYouTubeAPIClient()
        loaded = client.load_fixtures(DEFAULT_FIXTURE_DIR)

        # Should load at least the existing fixtures
        assert loaded > 0

    def test_fixture_search_loaded(self):
        """Test that search fixtures are loaded."""
        client = MockYouTubeAPIClient()
        client.load_fixtures(DEFAULT_FIXTURE_DIR)

        # Check if search results for "nature documentary" are loaded
        results = client.search_videos("nature documentary")
        assert len(results) > 0

        # Should contain fixture data
        first = results[0]
        assert first.video_id == "vid001abc123"

    def test_fixture_video_details_loaded(self):
        """Test that video details fixtures are loaded."""
        client = MockYouTubeAPIClient()
        client.load_fixtures(DEFAULT_FIXTURE_DIR)

        details = client.get_video_details(["vid001abc123"])
        assert "vid001abc123" in details

        # Check fixture data
        detail = details["vid001abc123"]
        assert detail.video_id == "vid001abc123"
        assert detail.duration_seconds == 930  # 15M30S = 930 seconds

    def test_fixture_captions_loaded(self):
        """Test that caption fixtures are loaded."""
        client = MockYouTubeAPIClient()
        client.load_fixtures(DEFAULT_FIXTURE_DIR)

        captions = client.check_captions_available("vid001abc123")
        assert len(captions) > 0

        # Should have multiple languages from fixture
        languages = [c.language for c in captions]
        assert "en" in languages
        assert "es" in languages

    def test_fixture_channel_loaded(self):
        """Test that channel fixtures are loaded."""
        client = MockYouTubeAPIClient()
        client.load_fixtures(DEFAULT_FIXTURE_DIR)

        channel = client.get_channel_details("UCTestChannel001")
        assert channel is not None

    def test_fixture_errors_loaded(self):
        """Test that error fixtures are loaded."""
        client = MockYouTubeAPIClient()
        client.load_fixtures(DEFAULT_FIXTURE_DIR)

        error = client.get_error_response("quota_exceeded")
        assert error is not None
        assert "error" in error

    def test_create_youtube_api_client_with_test_mode(self):
        """Test that create_youtube_api_client supports test_mode parameter."""
        client = create_youtube_api_client("test_key", test_mode=True)

        assert isinstance(client, MockYouTubeAPIClient)

        # Should have loaded fixtures
        results = client.search_videos("nature documentary")
        assert len(results) > 0


class TestSchemaValidation:
    """Tests for schema validation of mock responses (US-154-012)."""

    def test_search_result_schema(self):
        """Test that search results match expected schema."""
        client = get_mock_client()
        client.load_fixtures(DEFAULT_FIXTURE_DIR)

        results = client.search_videos("nature documentary")
        assert len(results) > 0

        for r in results:
            # Required fields
            assert r.video_id
            assert r.title
            assert r.channel_id
            assert r.channel_title

            # Optional fields can be None or have values
            assert r.published_at is None or isinstance(r.published_at, str)
            assert r.description is None or isinstance(r.description, str)

    def test_video_details_schema(self):
        """Test that video details match expected schema."""
        client = get_mock_client()
        client.load_fixtures(DEFAULT_FIXTURE_DIR)

        details = client.get_video_details(["vid001abc123"])
        assert "vid001abc123" in details

        d = details["vid001abc123"]
        # Required fields
        assert d.video_id
        assert d.duration_seconds > 0
        assert isinstance(d.caption_available, bool)

    def test_caption_info_schema(self):
        """Test that caption info matches expected schema."""
        client = get_mock_client()
        client.load_fixtures(DEFAULT_FIXTURE_DIR)

        captions = client.check_captions_available("vid001abc123")
        assert len(captions) > 0

        for c in captions:
            # Required fields
            assert c.language
            assert c.track_id
            assert isinstance(c.is_auto_generated, bool)

    def test_mock_client_has_channel_method(self):
        """Test that mock client has get_channel_details method."""
        client = get_mock_client()

        assert hasattr(client, 'get_channel_details')
        assert callable(client.get_channel_details)

    def test_mock_client_has_error_method(self):
        """Test that mock client has get_error_response method."""
        client = get_mock_client()

        assert hasattr(client, 'get_error_response')
        assert callable(client.get_error_response)
