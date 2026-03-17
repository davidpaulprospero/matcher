"""Mock YouTube API client for testing.

Provides a mock implementation of the YouTubeAPIClient interface
that returns canned responses without making actual API calls.

Enable via environment variable: MOCK_YOUTUBE_API=1

Fixture Support:
    The mock client can load responses from fixture files located in:
    tests/fixtures/youtube_api/

    Fixture files are named by endpoint and parameters:
    - search_{query}.json - Search results for a query
    - video_details_{video_id}.json - Video metadata
    - channel_{channel_id}.json - Channel metadata
    - captions_{video_id}.json - Available captions
    - error_{error_type}.json - Error responses

    Use load_fixtures() method to load fixtures from a directory.

Recording New Fixtures:
    To record new fixtures for testing:

    1. Make an actual API call and capture the response:
    ```python
    import requests
    import json

    API_KEY = "your-api-key"
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {"part": "snippet", "q": "nature documentary", "maxResults": 50, "key": API_KEY}

    response = requests.get(url, params=params)
    data = response.json()

    # Save to fixture file
    with open("tests/fixtures/youtube_api/search_nature_documentary.json", "w") as f:
        json.dump(data, f, indent=2)
    ```

    2. For error responses, use the error type in the filename:
    - error_quota_exceeded.json
    - error_403_forbidden.json
    - error_rate_limited.json

    3. Use the mock client with fixtures:
    ```python
    from src.testing.mocks.youtube_api import create_youtube_api_client

    # With test_mode=True, loads fixtures automatically
    client = create_youtube_api_client("test_key", test_mode=True)
    ```
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...downloader.youtube_api_client import (
    VideoSearchResult,
    VideoDetails,
    CaptionInfo,
    YouTubeAPIMetrics,
)

logger = logging.getLogger(__name__)

# Default fixture directory
DEFAULT_FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "youtube_api"


# Default canned video IDs for mock responses
DEFAULT_MOCK_VIDEO_IDS = [
    "dQw4w9WgXcQ",  # Rick Astley - Never Gonna Give You Up
    "jNQXAC9IVRw",  # Me at the zoo (first YouTube video)
    "9bZkp7q19f0",  # PSY - Gangnam Style
    "L_jWHffIx5E",  # Smash Mouth - All Star
    "3tmd-ClpJxA",  # Evolution of the柳's
]


def get_mock_client() -> "MockYouTubeAPIClient":
    """Factory function to create a mock YouTube API client.

    Returns:
        MockYouTubeAPIClient instance

    Args:
        api_key: Ignored in mock mode
        quota_limit: Ignored in mock mode
    """
    return MockYouTubeAPIClient(
        api_key="mock_api_key",
        quota_limit=10000,
    )


def is_mock_enabled() -> bool:
    """Check if mock mode is enabled via environment variable.

    Returns:
        True if MOCK_YOUTUBE_API=1 is set
    """
    return os.environ.get("MOCK_YOUTUBE_API", "0") == "1"


@dataclass
class MockYouTubeAPIClient:
    """Mock YouTube API client for testing.

    Returns canned responses for all API calls without making
    actual network requests.

    Attributes:
        api_key: API key (ignored in mock mode)
        quota_limit: Quota limit (ignored in mock mode)
        warn_at_percent: Warning threshold (ignored in mock mode)
        max_retries: Max retries (ignored in mock mode)
        fixture_dir: Directory to load fixture files from (optional)
    """

    api_key: str = "mock_api_key"
    quota_limit: int = 10000
    warn_at_percent: int = 80
    max_retries: int = 3
    fixture_dir: Optional[Path] = None

    # Store custom mock data for testing
    _mock_search_results: Dict[str, List[VideoSearchResult]] = field(default_factory=dict)
    _mock_video_details: Dict[str, VideoDetails] = field(default_factory=dict)
    _mock_captions: Dict[str, List[CaptionInfo]] = field(default_factory=dict)
    _mock_channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _mock_errors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _search_call_count: int = 0
    _details_call_count: int = 0
    _captions_call_count: int = 0
    _channel_call_count: int = 0

    def __post_init__(self):
        """Initialize with default mock data."""
        self._initialize_default_mock_data()
        # Try to load fixtures from default directory if it exists
        if self.fixture_dir is None and DEFAULT_FIXTURE_DIR.exists():
            self.fixture_dir = DEFAULT_FIXTURE_DIR
        if self.fixture_dir and self.fixture_dir.exists():
            self.load_fixtures(self.fixture_dir)

    def _initialize_default_mock_data(self):
        """Set up default mock responses."""
        # Default search results for common queries
        default_queries = [
            "technology", "science", "history", "nature", "music",
            "documentary", "education", "tutorial", "news", "sports"
        ]

        for query in default_queries:
            self._mock_search_results[query] = self._generate_search_results(query)

    def load_fixtures(self, fixture_dir: Path) -> int:
        """Load mock responses from fixture files.

        Fixture files should be named:
        - search_{query}.json - Search results
        - video_details_{video_id}.json - Video metadata
        - channel_{channel_id}.json - Channel metadata
        - captions_{video_id}.json - Caption list
        - error_{error_type}.json - Error responses

        Args:
            fixture_dir: Path to directory containing fixture JSON files

        Returns:
            Number of fixtures loaded
        """
        loaded = 0
        fixture_path = Path(fixture_dir)

        if not fixture_path.exists():
            logger.warning(f"Fixture directory does not exist: {fixture_dir}")
            return 0

        # Load search fixtures
        for f in fixture_path.glob("search_*.json"):
            query = f.stem.replace("search_", "")
            try:
                with open(f) as fp:
                    data = json.load(fp)
                results = self._parse_search_fixture(data)
                if results:
                    self._mock_search_results[query] = results
                    loaded += 1
                    logger.info(f"Loaded search fixture: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to load fixture {f.name}: {e}")

        # Load video details fixtures
        for f in fixture_path.glob("video_details_*.json"):
            video_id = f.stem.replace("video_details_", "")
            try:
                with open(f) as fp:
                    data = json.load(fp)
                details = self._parse_video_details_fixture(data, video_id)
                if details:
                    self._mock_video_details[video_id] = details
                    loaded += 1
                    logger.info(f"Loaded video_details fixture: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to load fixture {f.name}: {e}")

        # Load channel fixtures
        for f in fixture_path.glob("channel_*.json"):
            channel_id = f.stem.replace("channel_", "")
            try:
                with open(f) as fp:
                    data = json.load(fp)
                self._mock_channels[channel_id] = data
                loaded += 1
                logger.info(f"Loaded channel fixture: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to load fixture {f.name}: {e}")

        # Load caption fixtures
        for f in fixture_path.glob("captions_*.json"):
            video_id = f.stem.replace("captions_", "")
            try:
                with open(f) as fp:
                    data = json.load(fp)
                captions = self._parse_captions_fixture(data)
                if captions:
                    self._mock_captions[video_id] = captions
                    loaded += 1
                    logger.info(f"Loaded captions fixture: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to load fixture {f.name}: {e}")

        # Load error fixtures
        for f in fixture_path.glob("error_*.json"):
            error_type = f.stem.replace("error_", "")
            try:
                with open(f) as fp:
                    data = json.load(fp)
                self._mock_errors[error_type] = data
                loaded += 1
                logger.info(f"Loaded error fixture: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to load fixture {f.name}: {e}")

        logger.info(f"Loaded {loaded} fixtures from {fixture_dir}")
        return loaded

    def _parse_search_fixture(self, data: Dict[str, Any]) -> List[VideoSearchResult]:
        """Parse search response fixture into VideoSearchResult objects."""
        results = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            results.append(VideoSearchResult(
                video_id=item.get("id", {}).get("videoId", ""),
                title=snippet.get("title", ""),
                channel_id=snippet.get("channelId", ""),
                channel_title=snippet.get("channelTitle", ""),
                published_at=snippet.get("publishedAt", ""),
                description=snippet.get("description", ""),
                thumbnail_url=snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            ))
        return results

    def _parse_video_details_fixture(self, data: Dict[str, Any], video_id: str) -> Optional[VideoDetails]:
        """Parse video details fixture into VideoDetails object."""
        items = data.get("items", [])
        if not items:
            return None
        item = items[0]
        snippet = item.get("snippet", {})
        content = item.get("contentDetails", {})
        stats = item.get("statistics", {})
        topics = item.get("topicDetails", {})

        # Parse ISO 8601 duration to seconds
        duration_str = content.get("duration", "PT0S")
        duration_seconds = self._parse_duration(duration_str)

        return VideoDetails(
            video_id=video_id,
            duration=duration_str,
            duration_seconds=duration_seconds,
            tags=snippet.get("tags", []),
            category_id=snippet.get("categoryId", ""),
            topic_details=topics.get("topicCategories", []),
            topic_categories=topics.get("topicCategories", []),
            caption_available=content.get("caption") == "true",
            dimension=content.get("dimension", "2d"),
            definition=content.get("definition", "hd"),
            view_count=int(stats.get("viewCount", 0)),
            like_count=int(stats.get("likeCount", 0)),
            comment_count=int(stats.get("commentCount", 0)),
        )

    def _parse_captions_fixture(self, data: Dict[str, Any]) -> List[CaptionInfo]:
        """Parse captions fixture into CaptionInfo objects."""
        captions = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            captions.append(CaptionInfo(
                language=snippet.get("language", ""),
                track_id=item.get("id", ""),
                is_auto_generated=snippet.get("trackKind") == "ASR",
            ))
        return captions

    def _parse_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to seconds."""
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if not match:
            return 0
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    def _generate_search_results(self, query: str) -> List[VideoSearchResult]:
        """Generate mock search results for a query.

        Args:
            query: Search query string

        Returns:
            List of mock VideoSearchResult objects
        """
        results = []
        for i, video_id in enumerate(DEFAULT_MOCK_VIDEO_IDS):
            results.append(VideoSearchResult(
                video_id=video_id,
                title=f"Mock Video {i+1} - {query.title()}",
                channel_id=f"channel_{i+1}",
                channel_title=f"Mock Channel {i+1}",
                published_at="2024-01-15T10:00:00Z",
                description=f"This is a mock video about {query}",
                thumbnail_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                view_count=100000 + i * 10000,
                subscriber_count=50000 + i * 5000,
                total_views=1000000 + i * 100000,
                channel_created_date="2020-01-01T00:00:00Z",
            ))
        return results

    def _generate_video_details(self, video_id: str) -> VideoDetails:
        """Generate mock video details for a video ID.

        Args:
            video_id: YouTube video ID

        Returns:
            Mock VideoDetails object
        """
        return VideoDetails(
            video_id=video_id,
            duration="PT10M30S",
            duration_seconds=630,
            tags=["mock", "test", "video"],
            category_id="22",
            topic_details={
                "topic_categories": ["https://en.wikipedia.org/wiki/Technology"],
                "relevant_topic_ids": ["/m/07sbk9k"],
            },
            topic_categories=["https://en.wikipedia.org/wiki/Technology"],
            caption_available=True,
            dimension="2d",
            definition="hd",
            view_count=150000,
            like_count=5000,
            comment_count=250,
        )

    def _generate_captions(self, video_id: str) -> List[CaptionInfo]:
        """Generate mock caption info for a video ID.

        Args:
            video_id: YouTube video ID

        Returns:
            List of mock CaptionInfo objects
        """
        return [
            CaptionInfo(
                language="en",
                track_id="a.en",
                is_auto_generated=False,
            ),
            CaptionInfo(
                language="es",
                track_id="a.es",
                is_auto_generated=True,
            ),
        ]

    def search_videos(
        self,
        query: str,
        max_results: int = 50,
        video_type: str = "video",
    ) -> List[VideoSearchResult]:
        """Mock search for videos.

        Args:
            query: Search query string
            max_results: Maximum number of results
            video_type: Type of results (default: "video")

        Returns:
            List of mock VideoSearchResult objects
        """
        self._search_call_count += 1

        # Generate results for this query if not cached
        if query not in self._mock_search_results:
            self._mock_search_results[query] = self._generate_search_results(query)

        results = self._mock_search_results[query][:max_results]
        logger.info(f"MockYouTubeAPIClient: search_videos('{query}') returned {len(results)} results")
        return results

    def get_video_details(
        self,
        video_ids: List[str],
        part: str = "contentDetails,statistics,topicDetails",
    ) -> Dict[str, VideoDetails]:
        """Mock get video details.

        Args:
            video_ids: List of YouTube video IDs
            part: API parts to request

        Returns:
            Dict mapping video_id to mock VideoDetails objects
        """
        self._details_call_count += 1

        results: Dict[str, VideoDetails] = {}
        for video_id in video_ids:
            if video_id in self._mock_video_details:
                results[video_id] = self._mock_video_details[video_id]
            else:
                details = self._generate_video_details(video_id)
                self._mock_video_details[video_id] = details
                results[video_id] = details

        logger.info(f"MockYouTubeAPIClient: get_video_details({len(video_ids)} IDs) returned {len(results)} results")
        return results

    def check_captions_available(self, video_id: str) -> List[CaptionInfo]:
        """Mock check for available captions.

        Args:
            video_id: YouTube video ID

        Returns:
            List of mock CaptionInfo objects
        """
        self._captions_call_count += 1

        if video_id in self._mock_captions:
            return self._mock_captions[video_id]

        captions = self._generate_captions(video_id)
        self._mock_captions[video_id] = captions
        return captions

    def get_channel_details(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Mock get channel details.

        Args:
            channel_id: YouTube channel ID

        Returns:
            Dict with channel details or None if not found
        """
        self._channel_call_count += 1

        if channel_id in self._mock_channels:
            return self._mock_channels[channel_id]

        # Generate basic channel data if not found
        return {
            "kind": "youtube#channelListResponse",
            "items": [{
                "id": channel_id,
                "snippet": {
                    "title": f"Channel {channel_id}",
                    "description": f"Mock channel {channel_id}",
                },
                "statistics": {
                    "viewCount": "0",
                    "subscriberCount": "0",
                    "videoCount": "0",
                }
            }]
        }

    def get_error_response(self, error_type: str) -> Optional[Dict[str, Any]]:
        """Get mock error response by type.

        Args:
            error_type: Type of error (e.g., "quota_exceeded", "403_forbidden")

        Returns:
            Dict with error response or None if not found
        """
        if error_type in self._mock_errors:
            return self._mock_errors[error_type]
        return None

    def get_api_metrics(self) -> Dict[str, Any]:
        """Get mock API metrics.

        Returns:
            Dict with mock metrics
        """
        return {
            "search_calls": self._search_call_count,
            "videos_calls": self._details_call_count,
            "captions_calls": self._captions_call_count,
            "channel_calls": self._channel_call_count,
            "mock_mode": True,
        }

    def get_remaining_quota(self) -> int:
        """Get remaining quota (mock always returns full).

        Returns:
            Mock quota remaining
        """
        return self.quota_limit

    def get_quota_status(self) -> Dict[str, Any]:
        """Get quota status (mock).

        Returns:
            Dict with mock quota status
        """
        return {
            "remaining": self.quota_limit,
            "limit": self.quota_limit,
            "used": 0,
            "percent_used": 0.0,
            "mock_mode": True,
        }

    def get_usage_summary(self) -> str:
        """Get usage summary (mock).

        Returns:
            Mock usage summary string
        """
        return f"Mock Mode - Search calls: {self._search_call_count}, Details calls: {self._details_call_count}"

    # Methods for testing - allow injecting custom mock data

    def set_search_results(self, query: str, results: List[VideoSearchResult]) -> None:
        """Set custom search results for a query.

        Args:
            query: Search query
            results: List of VideoSearchResult to return
        """
        self._mock_search_results[query] = results

    def set_video_details(self, video_id: str, details: VideoDetails) -> None:
        """Set custom video details for a video ID.

        Args:
            video_id: YouTube video ID
            details: VideoDetails to return
        """
        self._mock_video_details[video_id] = details

    def set_captions(self, video_id: str, captions: List[CaptionInfo]) -> None:
        """Set custom captions for a video ID.

        Args:
            video_id: YouTube video ID
            captions: List of CaptionInfo to return
        """
        self._mock_captions[video_id] = captions

    def reset_call_counts(self) -> None:
        """Reset internal call counters."""
        self._search_call_count = 0
        self._details_call_count = 0
        self._captions_call_count = 0
        self._channel_call_count = 0


# Module-level function to create client based on environment
def create_youtube_api_client(
    api_key: str,
    quota_limit: int = 10000,
    warn_at_percent: int = 80,
    max_retries: int = 3,
    test_mode: bool = False,
    fixture_dir: Optional[Path] = None,
) -> Any:
    """Create YouTube API client, optionally using mock.

    Checks MOCK_YOUTUBE_API environment variable or test_mode parameter.
    If either is set, returns a mock client with fixture support.

    Args:
        api_key: YouTube API key
        quota_limit: Quota limit
        warn_at_percent: Warning threshold
        max_retries: Max retries
        test_mode: If True, use mock client (useful for --test-mode flag)
        fixture_dir: Optional directory to load fixtures from

    Returns:
        YouTubeAPIClient or MockYouTubeAPIClient instance
    """
    if is_mock_enabled() or test_mode:
        logger.info(f"Using MockYouTubeAPIClient (mock_env={is_mock_enabled()}, test_mode={test_mode})")
        client = get_mock_client()
        if fixture_dir:
            client.load_fixtures(fixture_dir)
        elif DEFAULT_FIXTURE_DIR.exists():
            client.load_fixtures(DEFAULT_FIXTURE_DIR)
        return client

    # Import real client
    from ...downloader.youtube_api_client import YouTubeAPIClient
    return YouTubeAPIClient(
        api_key=api_key,
        quota_limit=quota_limit,
        warn_at_percent=warn_at_percent,
        max_retries=max_retries,
    )
