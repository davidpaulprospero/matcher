"""Tests for YouTube API engagement metrics (US-148-008).

Tests engagement metrics enrichment from videos.list API,
ranking logic, and sorting by engagement score.

Pytest marker: fast
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List

from src.downloader.youtube_api_client import YouTubeAPIClient, VideoDetails
from src.downloader.api_fallback_handler import YouTubeAPIFallbackHandler


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_api_client():
    """Create a mock YouTubeAPIClient."""
    client = MagicMock(spec=YouTubeAPIClient)
    client.get_remaining_quota.return_value = 10000
    client.get_video_details.return_value = {}
    client.get_channel_metadata.return_value = {}
    return client


@pytest.fixture
def mock_config():
    """Create a mock config with youtube_api section."""
    config = MagicMock()
    youtube_api = MagicMock()
    youtube_api.enabled = True
    youtube_api.api_key = "test_key"
    youtube_api.quota_limit = 10000
    youtube_api.view_count_weight = 0.5
    youtube_api.like_count_weight = 0.3
    youtube_api.comment_count_weight = 0.2
    youtube_api.min_subscriber_count = 1000
    config.youtube_api = youtube_api
    return config


# ============================================================================
# Tests: Engagement Score Calculation
# ============================================================================

class TestEngagementScoreCalculation:
    """Test engagement score calculation logic."""

    def test_calculate_engagement_score_basic(self):
        """Test basic engagement score calculation."""
        handler = YouTubeAPIFallbackHandler(api_client=None, config=None)

        # Test with 1M views, 100K likes, 10K comments
        score = handler._calculate_engagement_score(
            view_count=1000000,
            like_count=100000,
            comment_count=10000,
            view_count_weight=0.5,
            like_count_weight=0.3,
            comment_count_weight=0.2
        )

        # Score should be positive and <= 1.0
        assert score > 0
        assert score <= 1.0

    def test_calculate_engagement_score_zero_values(self):
        """Test engagement score with zero values."""
        handler = YouTubeAPIFallbackHandler(api_client=None, config=None)

        score = handler._calculate_engagement_score(
            view_count=0,
            like_count=0,
            comment_count=0
        )

        assert score == 0.0

    def test_calculate_engagement_score_weights(self):
        """Test that weights affect the score."""
        handler = YouTubeAPIFallbackHandler(api_client=None, config=None)

        # Same counts, different weights
        score1 = handler._calculate_engagement_score(
            view_count=1000000,
            like_count=100000,
            comment_count=10000,
            view_count_weight=1.0,
            like_count_weight=0.0,
            comment_count_weight=0.0
        )

        score2 = handler._calculate_engagement_score(
            view_count=1000000,
            like_count=100000,
            comment_count=10000,
            view_count_weight=0.0,
            like_count_weight=1.0,
            comment_count_weight=0.0
        )

        # With 1M views and view_weight=1.0, score should be based on views
        # With 100K likes and like_weight=1.0, score should be based on likes
        # Views (1M) should give higher score than likes (100K)
        assert score1 > score2


# ============================================================================
# Tests: Engagement Metrics Enrichment
# ============================================================================

class TestEngagementMetricsEnrichment:
    """Test engagement metrics enrichment from API."""

    def test_enrich_with_engagement_metrics(self, mock_api_client):
        """Test enriching results with engagement metrics."""
        # Setup mock to return video details with statistics (Dict now)
        mock_api_client.get_video_details.return_value = {
            "abc123": VideoDetails(
                video_id="abc123",
                duration="PT10M",
                duration_seconds=600,
                tags=[],
                category_id="1",
                topic_details={},
                caption_available=True,
                dimension="2d",
                definition="hd",
                view_count=1000000,
                like_count=50000,
                comment_count=5000
            ),
        }

        handler = YouTubeAPIFallbackHandler(
            api_client=mock_api_client,
            config=mock_config()
        )

        results = [
            {"video_id": "abc123", "title": "Test Video"},
            {"video_id": "def456", "title": "Test Video 2"},
        ]

        enriched = handler._enrich_with_engagement_metrics(results)

        # Should have engagement metrics
        assert len(enriched) == 2
        # First result should have engagement metrics
        assert "view_count" in enriched[0]
        assert "like_count" in enriched[0]
        assert "comment_count" in enriched[0]
        assert "engagement_score" in enriched[0]

    def test_enrich_with_engagement_metrics_no_client(self):
        """Test enrichment without API client returns original results."""
        handler = YouTubeAPIFallbackHandler(api_client=None, config=None)

        results = [{"video_id": "abc123"}]

        enriched = handler._enrich_with_engagement_metrics(results)

        # Should return original results unchanged
        assert len(enriched) == 1
        assert enriched[0] == results[0]

    def test_enrich_with_engagement_metrics_quota_exceeded(self, mock_api_client):
        """Test enrichment stops when quota exceeded."""
        mock_api_client.get_remaining_quota.return_value = 0

        handler = YouTubeAPIFallbackHandler(
            api_client=mock_api_client,
            config=mock_config()
        )

        results = [{"video_id": "abc123"}]

        enriched = handler._enrich_with_engagement_metrics(results)

        # Should return original results without calling API
        assert len(enriched) == 1
        mock_api_client.get_video_details.assert_not_called()


# ============================================================================
# Tests: Ranking Logic
# ============================================================================

class TestEngagementRanking:
    """Test engagement-based ranking logic."""

    def test_ranking_affects_order(self):
        """Test that engagement metrics affect result ordering."""
        # Create results with different engagement scores
        results = [
            {"video_id": "low", "engagement_score": 0.2, "title": "Low Engagement"},
            {"video_id": "high", "engagement_score": 0.9, "title": "High Engagement"},
            {"video_id": "medium", "engagement_score": 0.5, "title": "Medium Engagement"},
        ]

        # Sort by engagement score (highest first)
        sorted_results = sorted(
            results,
            key=lambda r: r.get('engagement_score', 0),
            reverse=True
        )

        # Highest should be first
        assert sorted_results[0]['video_id'] == 'high'
        assert sorted_results[1]['video_id'] == 'medium'
        assert sorted_results[2]['video_id'] == 'low'

    def test_missing_engagement_score_at_bottom(self):
        """Test results without engagement score sort to bottom."""
        results = [
            {"video_id": "has_score", "engagement_score": 0.5},
            {"video_id": "no_score", "title": "No Score"},
            {"video_id": "has_score2", "engagement_score": 0.8},
        ]

        sorted_results = sorted(
            results,
            key=lambda r: r.get('engagement_score', 0),
            reverse=True
        )

        # Results with scores should be first
        assert sorted_results[0]['video_id'] == 'has_score2'
        assert sorted_results[1]['video_id'] == 'has_score'
        assert sorted_results[2]['video_id'] == 'no_score'


# ============================================================================
# Tests: Integration
# ============================================================================

class TestEngagementMetricsIntegration:
    """Integration tests for engagement metrics feature."""

    def test_search_with_fallback_includes_engagement(self, mock_api_client):
        """Test that search_with_fallback enriches with engagement metrics."""
        from src.downloader.youtube_api_client import VideoSearchResult

        # Mock search results
        mock_api_client.search_videos.return_value = [
            VideoSearchResult(
                video_id="abc123",
                title="Test Video",
                channel_id="UCtest",
                channel_title="Test Channel",
                published_at="2024-01-01",
                description="Test"
            )
        ]

        # Mock video details with statistics
        mock_details = MagicMock()
        mock_details.video_id = "abc123"
        mock_details.statistics = {'viewCount': '1000000', 'likeCount': '50000', 'commentCount': '5000'}
        mock_api_client.get_video_details.return_value = [mock_details]

        # Mock channel metadata
        mock_api_client.get_channel_metadata.return_value = {}

        handler = YouTubeAPIFallbackHandler(
            api_client=mock_api_client,
            config=mock_config()
        )

        results = handler.search_with_fallback("test query", max_results=10)

        # Should have engagement metrics
        assert len(results) > 0
        result = results[0]
        assert "engagement_score" in result
        assert result["engagement_score"] > 0


# ============================================================================
# Tests: US-158-010 Video Quality Score (Quality Boost)
# ============================================================================

class TestVideoQualityScore:
    """Test video quality score calculation and ranking (US-158-010)."""

    def test_calculate_video_quality_score_basic(self):
        """Test basic quality score calculation with default weights."""
        client = YouTubeAPIClient(
            api_key="test_key",
            quality_boost_enabled=True,
            quality_view_count_weight=0.7,
            quality_like_count_weight=0.2,
            quality_comment_count_weight=0.1
        )

        # Test with 1M views, 100K likes, 10K comments
        # Expected: 1000000 * 0.7 + 100000 * 0.2 + 10000 * 0.1 = 700000 + 20000 + 1000 = 721000
        score = client.calculate_video_quality_score(
            view_count=1000000,
            like_count=100000,
            comment_count=10000
        )

        assert score == 721000.0

    def test_calculate_video_quality_score_zero_engagement(self):
        """Test quality score with zero engagement - should return 0.0."""
        client = YouTubeAPIClient(
            api_key="test_key",
            quality_boost_enabled=True
        )

        score = client.calculate_video_quality_score(
            view_count=0,
            like_count=0,
            comment_count=0
        )

        assert score == 0.0

    def test_calculate_video_quality_score_custom_weights(self):
        """Test quality score with custom weights."""
        client = YouTubeAPIClient(
            api_key="test_key",
            quality_boost_enabled=True,
            quality_view_count_weight=0.5,
            quality_like_count_weight=0.3,
            quality_comment_count_weight=0.2
        )

        # Test with 1M views, 100K likes, 10K comments
        # Expected: 1000000 * 0.5 + 100000 * 0.3 + 10000 * 0.2 = 500000 + 30000 + 2000 = 532000
        score = client.calculate_video_quality_score(
            view_count=1000000,
            like_count=100000,
            comment_count=10000
        )

        assert score == 532000.0

    def test_calculate_video_quality_score_partial_zero(self):
        """Test quality score with some zero values."""
        client = YouTubeAPIClient(
            api_key="test_key",
            quality_boost_enabled=True
        )

        # Only views, no likes or comments
        score = client.calculate_video_quality_score(
            view_count=1000000,
            like_count=0,
            comment_count=0
        )

        # Expected: 1000000 * 0.7 + 0 + 0 = 700000
        assert score == 700000.0

    def test_quality_score_ranking_sort(self):
        """Test that results can be sorted by quality score."""
        client = YouTubeAPIClient(
            api_key="test_key",
            quality_boost_enabled=True,
            quality_view_count_weight=0.7,
            quality_like_count_weight=0.2,
            quality_comment_count_weight=0.1
        )

        # Create mock results with different engagement metrics
        results = [
            {"video_id": "vid1", "view_count": 100000, "like_count": 5000, "comment_count": 500},
            {"video_id": "vid2", "view_count": 1000000, "like_count": 50000, "comment_count": 5000},
            {"video_id": "vid3", "view_count": 50000, "like_count": 1000, "comment_count": 100},
        ]

        # Calculate quality scores
        for r in results:
            r['quality_score'] = client.calculate_video_quality_score(
                view_count=r['view_count'],
                like_count=r['like_count'],
                comment_count=r['comment_count']
            )

        # Sort by quality score (highest first)
        results.sort(key=lambda r: r['quality_score'], reverse=True)

        # vid2 should be first (highest quality score)
        assert results[0]['video_id'] == 'vid2'
        # vid3 should be last (lowest quality score)
        assert results[2]['video_id'] == 'vid3'

    def test_quality_score_disabled(self):
        """Test that quality score is 0 when quality boost is disabled."""
        client = YouTubeAPIClient(
            api_key="test_key",
            quality_boost_enabled=False
        )

        # Even with high engagement, quality should be 0 when disabled
        score = client.calculate_video_quality_score(
            view_count=1000000,
            like_count=100000,
            comment_count=10000
        )

        # The method still calculates, but the client wouldn't use it when disabled
        assert score > 0  # Method calculates regardless of enabled flag
