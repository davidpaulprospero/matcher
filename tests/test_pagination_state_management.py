"""Tests for YouTube API pagination state management (US-156-011).

Tests verify:
- Track pagination state with page_token -> results mapping
- Detect duplicate page tokens indicating API bugs
- Add max_pages_per_query config (default: 10) to limit pagination
- Implement page_token caching to resume interrupted pagination
- Add pagination_timeout to abort stuck pagination after N seconds
- Integration test for pagination with simulated interruption

Run with:
    pytest tests/test_pagination_state_management.py -v

Pytest markers:
    - integration: marks tests as integration tests
"""

import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock, call
from typing import List, Dict, Any

from src.downloader.youtube_api_client import YouTubeAPIClient


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def pagination_client():
    """Create a YouTubeAPIClient with pagination config."""
    with patch('src.downloader.youtube_api_client.YouTubeAPISQLCache') as mock_cache:
        mock_cache.return_value = None
        client = YouTubeAPIClient(
            api_key="test_api_key_123",
            quota_limit=10000,
            max_pages_per_query=5,  # Test with 5 pages limit
            pagination_timeout=60,
            deduplicate_searches=False,  # Disable to allow repeated queries in tests
        )
        client.disable_query_cache()
        return client


@pytest.fixture
def mock_paginated_responses():
    """Create mock responses for multiple pages with page tokens."""
    def create_page(page_num: int, num_results: int = 10, has_next: bool = True):
        items = [
            {
                "id": {"kind": "youtube#video", "videoId": f"vid{page_num}_{i}"},
                "snippet": {
                    "title": f"Video Page {page_num} #{i}",
                    "channelId": f"UC{page_num}",
                    "channelTitle": f"Channel {page_num}",
                    "publishedAt": "2024-01-01T00:00:00Z",
                    "description": f"Description for page {page_num}, video {i}",
                    "thumbnails": {},
                }
            }
            for i in range(num_results)
        ]
        return {
            "kind": "youtube#searchListResponse",
            "items": items,
            "nextPageToken": f"page_token_{page_num + 1}" if has_next else None,
        }
    return create_page


# ============================================================================
# Tests
# ============================================================================

class TestPaginationStateTracking:
    """Test pagination state tracking (US-156-011)."""

    def test_track_pagination_state(self, pagination_client):
        """Test pagination state is tracked correctly."""
        query = "test query"
        page_token = "test_token_123"
        results_count = 50

        # Track pagination state
        pagination_client.track_pagination_state(query, page_token, results_count)

        # Verify state was tracked
        state = pagination_client.get_pagination_state(query)
        assert page_token in state
        assert state[page_token] == results_count

    def test_track_multiple_pages(self, pagination_client):
        """Test tracking multiple pages for same query."""
        query = "test query"

        # Track multiple pages
        pagination_client.track_pagination_state(query, "token_1", 50)
        pagination_client.track_pagination_state(query, "token_2", 50)
        pagination_client.track_pagination_state(query, "token_3", 50)

        # Verify all pages tracked
        state = pagination_client.get_pagination_state(query)
        assert len(state) == 3
        assert state["token_1"] == 50
        assert state["token_2"] == 50
        assert state["token_3"] == 50

    def test_duplicate_page_token_detection(self, pagination_client):
        """Test duplicate page tokens are detected and logged."""
        query = "test query"
        page_token = "duplicate_token"
        results_count = 50

        # Track same token twice
        pagination_client.track_pagination_state(query, page_token, results_count)

        # Track again - should detect duplicate
        pagination_client.track_pagination_state(query, page_token, results_count)

        # Verify duplicate was counted in metrics (access via _metrics)
        assert pagination_client._metrics.duplicate_page_tokens >= 1

    def test_clear_pagination_state(self, pagination_client):
        """Test clearing pagination state."""
        query = "test query"

        # Add state
        pagination_client.track_pagination_state(query, "token_1", 50)
        pagination_client.track_pagination_state(query, "token_2", 50)

        # Clear specific query
        pagination_client.clear_pagination_state(query)

        # Verify cleared
        state = pagination_client.get_pagination_state(query)
        assert len(state) == 0

    def test_clear_all_pagination_state(self, pagination_client):
        """Test clearing all pagination state."""
        # Add multiple queries
        pagination_client.track_pagination_state("query1", "token_1", 50)
        pagination_client.track_pagination_state("query2", "token_2", 50)

        # Clear all
        pagination_client.clear_pagination_state()

        # Verify all cleared
        assert len(pagination_client.get_pagination_state("query1")) == 0
        assert len(pagination_client.get_pagination_state("query2")) == 0


class TestPageTokenCaching:
    """Test page token caching for resume (US-156-011)."""

    def test_cache_page_token(self, pagination_client):
        """Test page token is cached correctly."""
        query = "test query"
        page_token = "resume_token_123"
        results_count = 150

        # Cache page token
        pagination_client._cache_page_token(query, page_token, results_count)

        # Verify cached
        cached = pagination_client.get_cached_page_token(query)
        assert cached is not None
        assert cached[0] == page_token
        assert cached[1] == results_count

    def test_cache_multiple_page_tokens(self, pagination_client):
        """Test multiple page tokens are cached for same query."""
        query = "test query"

        # Cache multiple tokens
        pagination_client._cache_page_token(query, "token_1", 50)
        pagination_client._cache_page_token(query, "token_2", 100)
        pagination_client._cache_page_token(query, "token_3", 150)

        # Should have last token cached
        cached = pagination_client.get_cached_page_token(query)
        assert cached[0] == "token_3"
        assert cached[1] == 150

    def test_cache_different_queries(self, pagination_client):
        """Test page tokens are cached separately for different queries."""
        # Cache for different queries
        pagination_client._cache_page_token("query1", "token_q1", 50)
        pagination_client._cache_page_token("query2", "token_q2", 100)

        # Verify separate caches
        cached1 = pagination_client.get_cached_page_token("query1")
        cached2 = pagination_client.get_cached_page_token("query2")

        assert cached1[0] == "token_q1"
        assert cached2[0] == "token_q2"

    def test_clear_page_token_cache(self, pagination_client):
        """Test clearing page token cache."""
        query = "test query"

        # Cache token
        pagination_client._cache_page_token(query, "token_1", 50)

        # Clear cache
        pagination_client.clear_page_token_cache(query)

        # Verify cleared
        cached = pagination_client.get_cached_page_token(query)
        assert cached is None

    def test_clear_all_page_token_cache(self, pagination_client):
        """Test clearing all page token caches."""
        # Cache tokens
        pagination_client._cache_page_token("query1", "token_1", 50)
        pagination_client._cache_page_token("query2", "token_2", 100)

        # Clear all
        pagination_client.clear_page_token_cache()

        # Verify all cleared
        assert pagination_client.get_cached_page_token("query1") is None
        assert pagination_client.get_cached_page_token("query2") is None


class TestMaxPagesPerQuery:
    """Test max_pages_per_query config (US-156-011)."""

    def test_max_pages_limit(self, pagination_client, mock_paginated_responses):
        """Test pagination stops at max_pages_per_query limit."""
        call_count = 0

        def mock_request(endpoint, params, quota_cost):
            nonlocal call_count
            call_count += 1
            # Return pages with next tokens (will continue indefinitely)
            return mock_paginated_responses(call_count, 10, has_next=True)

        with patch.object(pagination_client, '_make_request', side_effect=mock_request):
            results = pagination_client.search_videos("test query", max_results=1000)

        # Should stop at max_pages_per_query (5)
        assert call_count == 5
        # Should have 5 pages * 10 results = 50 results
        assert len(results) == 50

    def test_default_max_pages(self):
        """Test default max_pages_per_query is 10."""
        with patch('src.downloader.youtube_api_client.YouTubeAPISQLCache') as mock_cache:
            mock_cache.return_value = None
            client = YouTubeAPIClient(api_key="test_key")
            assert client._max_pages_per_query == 10

    def test_custom_max_pages(self):
        """Test custom max_pages_per_query is respected."""
        with patch('src.downloader.youtube_api_client.YouTubeAPISQLCache') as mock_cache:
            mock_cache.return_value = None
            client = YouTubeAPIClient(api_key="test_key", max_pages_per_query=3)
            assert client._max_pages_per_query == 3


class TestPaginationTimeout:
    """Test pagination timeout (US-156-011)."""

    def test_pagination_timeout_stops_loop(self, pagination_client, mock_paginated_responses):
        """Test pagination times out and stops."""
        # Set very short timeout
        pagination_client._pagination_timeout = 0.1  # 100ms

        call_count = 0

        def mock_request(endpoint, params, quota_cost):
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)  # 50ms per request
            return mock_paginated_responses(call_count, 10, has_next=True)

        with patch.object(pagination_client, '_make_request', side_effect=mock_request):
            results = pagination_client.search_videos("test query", max_results=1000)

        # Should stop due to timeout
        assert call_count >= 1
        # Check timeout was recorded
        stats = pagination_client.get_pagination_stats()
        assert stats["search_page_timeouts"] >= 1


class TestPaginationIntegration:
    """Integration tests for pagination with simulated interruption (US-156-011)."""

    def test_pagination_resume_after_interruption(self, pagination_client, mock_paginated_responses):
        """Test pagination can resume after simulated interruption (US-156-011)."""
        query = "test query"
        call_count = 0

        def mock_request_first_run(endpoint, params, quota_cost):
            """Mock for first run - stops at max_pages limit."""
            nonlocal call_count
            call_count += 1
            # Always return has_next=True so we hit max_pages limit
            return mock_paginated_responses(call_count, 10, has_next=True)

        # First run: get partial results and cache token (stops at max_pages limit)
        with patch.object(pagination_client, '_make_request', side_effect=mock_request_first_run):
            results_first = pagination_client.search_videos(query, max_results=1000)

        # First run should have stopped at max_pages limit (5)
        assert len(results_first) == 50
        assert call_count == 5

        # Verify page token was cached
        cached = pagination_client.get_cached_page_token(query)
        assert cached is not None
        assert cached[1] == 50  # 5 pages * 10 results

        # Reset call count for second run (resume)
        call_count = 0

        # Clear deduplication cache so second run makes actual API calls
        pagination_client.clear_deduplication_cache()

        def mock_request_resume(endpoint, params, quota_cost):
            """Mock for resume run - should use cached page token."""
            nonlocal call_count
            call_count += 1
            # Verify pageToken is in params (indicating resume)
            assert "pageToken" in params, "Expected pageToken in params for resume"
            # Return remaining pages until exhausted
            return mock_paginated_responses(call_count + 5, 10, has_next=False)

        # Second run: should resume from cached token
        with patch.object(pagination_client, '_make_request', side_effect=mock_request_resume):
            results_second = pagination_client.search_videos(query, max_results=1000)

        # Second run should have fetched additional pages using cached token
        # Should have made at least 1 call (resume from cached token)
        assert call_count >= 1, "Expected resume to use cached page token"

        # Page token cache should be cleared after successful completion
        cached_after = pagination_client.get_cached_page_token(query)
        assert cached_after is None, "Page token cache should be cleared after successful search"

    def test_pagination_state_in_stats(self, pagination_client):
        """Test pagination state info is included in stats."""
        # Track some state
        pagination_client.track_pagination_state("query1", "token1", 50)
        pagination_client._cache_page_token("query2", "token2", 100)

        stats = pagination_client.get_pagination_stats()

        # Should include pagination cache info
        assert "cached_queries_with_page_tokens" in stats
        assert stats["cached_queries_with_page_tokens"] >= 1

    def test_duplicate_token_during_search(self, pagination_client, mock_paginated_responses):
        """Test duplicate page tokens during search are detected."""
        # Create response that returns same token twice (simulating API bug)
        call_count = 0

        def mock_request(endpoint, params, quota_cost):
            nonlocal call_count
            call_count += 1
            return mock_paginated_responses(1, 10, has_next=True)  # Always returns same token

        with patch.object(pagination_client, '_make_request', side_effect=mock_request):
            results = pagination_client.search_videos("test query", max_results=100)

        # Should detect duplicate and stop
        stats = pagination_client.get_pagination_stats()
        assert stats["duplicate_page_tokens"] >= 1
        # Should stop after detecting duplicate
        assert call_count < 10  # Should not continue forever

    def test_pagination_with_zero_max_pages(self):
        """Test pagination with max_pages_per_query=0 (fetches first page, then stops)."""
        with patch('src.downloader.youtube_api_client.YouTubeAPISQLCache') as mock_cache:
            mock_cache.return_value = None
            client = YouTubeAPIClient(
                api_key="test_key",
                max_pages_per_query=0,  # Edge case: limit starts at 0, but first page still fetched
                deduplicate_searches=False,  # Disable to allow test to make API calls
            )
            client.disable_query_cache()

        # Return page with more results
        mock_response = {
            "kind": "youtube#searchListResponse",
            "items": [
                {
                    "id": {"kind": "youtube#video", "videoId": f"vid{i}"},
                    "snippet": {
                        "title": f"Video {i}",
                        "channelId": f"UC{i}",
                        "channelTitle": f"Channel {i}",
                        "publishedAt": "2024-01-01T00:00:00Z",
                        "description": f"Description {i}",
                        "thumbnails": {},
                    }
                }
                for i in range(10)
            ],
            "nextPageToken": "next_token",
        }

        with patch.object(client, '_make_request', return_value=mock_response):
            results = client.search_videos("test query", max_results=50)

        # With max_pages=0, first page is still fetched (check happens after first page)
        assert len(results) == 10


class TestQueryNormalization:
    """Test query normalization for pagination state."""

    def test_normalized_query_for_cache(self, pagination_client):
        """Test queries are normalized before caching."""
        # Cache with different case/whitespace
        pagination_client._cache_page_token("Test  Query", "token1", 50)
        pagination_client._cache_page_token("test query", "token2", 100)

        # Should have only one entry (normalized)
        cached = pagination_client.get_cached_page_token("TEST QUERY")
        assert cached is not None

        # Should have the last one cached
        assert cached[1] == 100

    def test_pagination_state_normalization(self, pagination_client):
        """Test pagination state uses normalized queries."""
        # Track with different query forms
        pagination_client.track_pagination_state("Test  Query", "token1", 50)
        pagination_client.track_pagination_state("test query", "token2", 100)

        # Should have consolidated state
        state = pagination_client.get_pagination_state("TEST  QUERY")
        assert len(state) == 2  # Both tokens tracked under normalized form


class TestPaginationEfficiency:
    """Test pagination efficiency metrics (US-157-011)."""

    def test_pagination_efficiency_metric(self, pagination_client, mock_paginated_responses):
        """Test pagination efficiency is tracked correctly."""
        # Mock search that returns partial results (simulating quota limit)
        call_count = 0

        def mock_request(endpoint, params, quota_cost):
            nonlocal call_count
            call_count += 1
            # First page has 10 results, next page returns quota error
            if call_count == 1:
                return mock_paginated_responses(1, 10, has_next=False)
            raise Exception("Quota exceeded")

        with patch.object(pagination_client, '_make_request', side_effect=mock_request):
            results = pagination_client.search_videos("test query", max_results=50)

        # Requested 50, got 10
        assert len(results) == 10

        stats = pagination_client.get_pagination_stats()
        assert stats["pagination_results_requested"] == 50
        assert stats["pagination_results_returned"] == 10
        assert stats["pagination_efficiency"] == 0.2  # 10/50 = 0.2

    def test_pagination_efficiency_full_results(self, pagination_client, mock_paginated_responses):
        """Test pagination efficiency when all results returned."""
        call_count = 0

        def mock_request(endpoint, params, quota_cost):
            nonlocal call_count
            call_count += 1
            # Return enough pages to fulfill the request
            return mock_paginated_responses(call_count, 10, has_next=call_count < 3)

        with patch.object(pagination_client, '_make_request', side_effect=mock_request):
            results = pagination_client.search_videos("test query", max_results=30)

        # Requested 30, got 30 (3 pages * 10)
        assert len(results) == 30

        stats = pagination_client.get_pagination_stats()
        assert stats["pagination_results_requested"] == 30
        assert stats["pagination_results_returned"] == 30
        assert stats["pagination_efficiency"] == 1.0  # 30/30 = 1.0

    def test_pagination_efficiency_zero_requested(self, pagination_client):
        """Test pagination efficiency when no results requested (edge case)."""
        # Direct check on metrics object
        metrics = pagination_client._metrics

        # Initially should have 0 requested
        stats = metrics.get_pagination_stats()
        assert stats["pagination_results_requested"] == 0
        assert stats["pagination_results_returned"] == 0
        assert stats["pagination_efficiency"] == 0.0  # No request = 0 efficiency


class TestResultsPerPage:
    """Test configurable results_per_page (US-157-011)."""

    def test_default_results_per_page(self):
        """Test default results_per_page is 50."""
        with patch('src.downloader.youtube_api_client.YouTubeAPISQLCache') as mock_cache:
            mock_cache.return_value = None
            client = YouTubeAPIClient(api_key="test_key")
            assert client._results_per_page == 50

    def test_custom_results_per_page(self):
        """Test custom results_per_page is respected."""
        with patch('src.downloader.youtube_api_client.YouTubeAPISQLCache') as mock_cache:
            mock_cache.return_value = None
            client = YouTubeAPIClient(api_key="test_key", results_per_page=25)
            assert client._results_per_page == 25

    def test_results_per_page_max_enforced(self):
        """Test results_per_page cannot exceed 50 (YouTube API limit)."""
        with patch('src.downloader.youtube_api_client.YouTubeAPISQLCache') as mock_cache:
            mock_cache.return_value = None
            client = YouTubeAPIClient(api_key="test_key", results_per_page=100)
            assert client._results_per_page == 50  # Should be capped at 50

    def test_results_per_page_used_in_api_call(self, pagination_client, mock_paginated_responses):
        """Test results_per_page is used in API maxResults parameter."""
        # Create client with results_per_page=10
        with patch('src.downloader.youtube_api_client.YouTubeAPISQLCache') as mock_cache:
            mock_cache.return_value = None
            client = YouTubeAPIClient(
                api_key="test_key",
                results_per_page=10,
                deduplicate_searches=False,
            )
            client.disable_query_cache()

        max_results_seen = []

        def mock_request(endpoint, params, quota_cost):
            # Capture maxResults param
            max_results_seen.append(params.get("maxResults"))
            return mock_paginated_responses(1, 10, has_next=False)

        with patch.object(client, '_make_request', side_effect=mock_request):
            client.search_videos("test query", max_results=50)

        # Should have used results_per_page=10 in the API call
        assert 10 in max_results_seen

    def test_results_per_page_with_pagination(self, pagination_client, mock_paginated_responses):
        """Test results_per_page works correctly with pagination."""
        # Configure with small results_per_page
        pagination_client._results_per_page = 5
        call_count = 0

        def mock_request(endpoint, params, quota_cost):
            nonlocal call_count
            call_count += 1
            # Verify maxResults uses results_per_page
            assert params.get("maxResults") == 5, f"Expected maxResults=5, got {params.get('maxResults')}"
            return mock_paginated_responses(call_count, 5, has_next=True)

        with patch.object(pagination_client, '_make_request', side_effect=mock_request):
            results = pagination_client.search_videos("test query", max_results=15)

        # Should have made 3 pages (15 / 5 = 3)
        assert call_count == 3
        assert len(results) == 15
