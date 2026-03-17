"""
Shared Fixtures for YouTube API Tests

US-119-009: Add pytest fixtures for mock YouTube API responses

This module provides reusable mock factories and fixtures for testing
YouTube API interactions. These fixtures cover:
- Video metadata responses
- Caption API responses
- Search API responses

Usage:
    from tests.fixtures.youtube_api_fixtures import (
        create_mock_youtube_video_metadata,
        create_mock_caption_response,
        create_mock_youtube_search_result,
    )

    def test_something():
        video = create_mock_youtube_video_metadata(video_id="abc123")
        caption = create_mock_caption_response(video_id="abc123")

Pytest fixtures are also available when this module is imported into conftest.py:
    @pytest.fixture
    def mock_youtube_video():
        return create_mock_youtube_video_metadata()
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


# =============================================================================
# Video Metadata Fixtures
# =============================================================================

def create_mock_youtube_video_metadata(
    video_id: str = "dQw4w9WgXcQ",
    title: str = "Sample Video Title",
    channel: str = "Test Channel",
    channel_id: str = "UC123456",
    duration: int = 600,  # seconds
    view_count: int = 1000000,
    like_count: int = 50000,
    comment_count: int = 1000,
    published_at: str = "2024-01-15T10:00:00Z",
    description: str = "This is a sample video description with relevant content.",
    tags: Optional[List[str]] = None,
    category_id: str = "22",
    live_broadcast: bool = False,
    privacy_status: str = "public",
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube video metadata response.

    Mimics the structure returned by YouTube Data API v3 videos endpoint.

    Args:
        video_id: YouTube video ID
        title: Video title
        channel: Channel name
        channel_id: YouTube channel ID
        duration: Video duration in seconds
        view_count: View count
        like_count: Like count
        comment_count: Comment count
        published_at: ISO 8601 publish timestamp
        description: Video description
        tags: Video tags
        category_id: YouTube category ID
        live_broadcast: Is live broadcast
        privacy_status: Video privacy status
        **overrides: Additional fields to override

    Returns:
        Dict with video metadata matching YouTube API structure

    Example:
        >>> video = create_mock_youtube_video_metadata(video_id="abc123")
        >>> assert video["id"] == "abc123"
        >>> assert video["snippet"]["title"] == "Sample Video Title"
    """
    if tags is None:
        tags = ["test", "sample", "video"]

    result = {
        "kind": "youtube#video",
        "etag": "etag123",
        "id": video_id,
        "snippet": {
            "publishedAt": published_at,
            "channelId": channel_id,
            "title": title,
            "description": description,
            "thumbnails": {
                "default": {"url": f"https://i.ytimg.com/vi/{video_id}/default.jpg", "width": 120, "height": 90},
                "medium": {"url": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg", "width": 320, "height": 180},
                "high": {"url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg", "width": 480, "height": 360},
                "standard": {"url": f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg", "width": 640, "height": 480},
                "maxres": {"url": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg", "width": 1280, "height": 720},
            },
            "channelTitle": channel,
            "tags": tags,
            "categoryId": category_id,
            "liveBroadcastContent": "none" if not live_broadcast else "live",
            "localized": {
                "title": title,
                "description": description,
            },
        },
        "contentDetails": {
            "duration": _seconds_to_iso8601(duration),
            "dimension": "2d",
            "definition": "hd",
            "caption": "true",
            "licensedContent": False,
            "regionRestriction": {
                "allowed": ["US", "GB", "CA"],
            },
        },
        "statistics": {
            "viewCount": str(view_count),
            "likeCount": str(like_count),
            "commentCount": str(comment_count),
            "favoriteCount": "0",
        },
        "status": {
            "uploadStatus": "processed",
            "privacyStatus": privacy_status,
            "license": "youtube",
            "embeddable": True,
            "publicStatsViewable": True,
        },
    }

    # Apply overrides
    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like "snippet.title"
            parts = key.split(".")
            obj = result
            for part in parts[:-1]:
                obj = obj.setdefault(part, {})
            obj[parts[-1]] = value
        else:
            result[key] = value

    return result


def _seconds_to_iso8601(seconds: int) -> str:
    """Convert seconds to ISO 8601 duration format (PT#H#M#S)."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}H")
    if minutes > 0:
        parts.append(f"{minutes}M")
    if secs > 0 or not parts:
        parts.append(f"{secs}S")

    return "PT" + "".join(parts)


def create_mock_video_list_response(
    videos: Optional[List[Dict[str, Any]]] = None,
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube videos list API response.

    Args:
        videos: List of video metadata dicts (uses create_mock_youtube_video_metadata if None)
        **overrides: Additional response fields to override

    Returns:
        Dict with YouTube API video list response structure
    """
    if videos is None:
        videos = [create_mock_youtube_video_metadata()]

    result = {
        "kind": "youtube#videoListResponse",
        "etag": "etag123",
        "pageInfo": {
            "totalResults": len(videos),
            "resultsPerPage": len(videos),
        },
        "items": videos,
    }

    for key, value in overrides.items():
        result[key] = value

    return result


# =============================================================================
# Search API Fixtures
# =============================================================================

def create_mock_youtube_search_result(
    video_id: str = "dQw4w9WgXcQ",
    title: str = "Search Result Video",
    channel: str = "Test Channel",
    channel_id: str = "UC123456",
    published_at: str = "2024-01-15T10:00:00Z",
    description: str = "Video description from search results",
    duration: str = "PT10M",  # ISO 8601 duration
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube search result.

    Mimics the structure returned by YouTube Data API v3 search endpoint.

    Args:
        video_id: YouTube video ID
        title: Video title
        channel: Channel name
        channel_id: YouTube channel ID
        published_at: ISO 8601 publish timestamp
        description: Video description snippet
        duration: ISO 8601 duration string
        **overrides: Additional fields to override

    Returns:
        Dict with search result structure

    Example:
        >>> result = create_mock_youtube_search_result(video_id="abc123")
        >>> assert result["id"]["videoId"] == "abc123"
    """
    result = {
        "kind": "youtube#searchResult",
        "etag": "etag_search_123",
        "id": {
            "kind": "youtube#video",
            "videoId": video_id,
        },
        "snippet": {
            "publishedAt": published_at,
            "channelId": channel_id,
            "channelTitle": channel,
            "title": title,
            "description": description,
            "thumbnails": {
                "default": {"url": f"https://i.ytimg.com/vi/{video_id}/default.jpg", "width": 120, "height": 90},
                "medium": {"url": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg", "width": 320, "height": 180},
                "high": {"url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg", "width": 480, "height": 360},
            },
        },
    }

    if duration:
        result["contentDetails"] = {
            "duration": duration,
            "dimension": "2d",
            "definition": "hd",
        }

    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            obj = result
            for part in parts[:-1]:
                obj = obj.setdefault(part, {})
            obj[parts[-1]] = value
        else:
            result[key] = value

    return result


def create_mock_search_list_response(
    results: Optional[List[Dict[str, Any]]] = None,
    total_results: int = 100,
    results_per_page: int = 50,
    next_page_token: Optional[str] = None,
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube search list API response.

    Args:
        results: List of search result dicts (uses create_mock_youtube_search_result if None)
        total_results: Total number of results available
        results_per_page: Number of results per page
        next_page_token: Token for next page (None for last page)
        **overrides: Additional response fields to override

    Returns:
        Dict with YouTube API search list response structure
    """
    if results is None:
        results = [create_mock_youtube_search_result()]

    result = {
        "kind": "youtube#searchListResponse",
        "etag": "etag_search_list",
        "pageInfo": {
            "totalResults": total_results,
            "resultsPerPage": results_per_page,
        },
        "items": results,
    }

    if next_page_token:
        result["nextPageToken"] = next_page_token

    for key, value in overrides.items():
        result[key] = value

    return result


# =============================================================================
# Caption API Fixtures
# =============================================================================

def create_mock_caption_track(
    video_id: str = "dQw4w9WgXcQ",
    track_id: str = "a1b2c3d4e5",
    language: str = "en",
    name: str = "English (auto-generated)",
    is_auto_generated: bool = True,
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube caption track.

    Mimics structure from YouTube caption API.

    Args:
        video_id: YouTube video ID
        track_id: Caption track ID
        language: ISO 639-1 language code
        name: Display name of caption track
        is_auto_generated: Whether captions are auto-generated
        **overrides: Additional fields to override

    Returns:
        Dict with caption track structure
    """
    result = {
        "kind": "youtube#captionTrack",
        "etag": "etag_caption_123",
        "id": track_id,
        "snippet": {
            "videoId": video_id,
            "language": language,
            "name": name,
            "trackKind": "ASR" if is_auto_generated else "standard",
            "isAutoSynced": is_auto_generated,
        },
    }

    for key, value in overrides.items():
        result[key] = value

    return result


def create_mock_caption_list_response(
    video_id: str = "dQw4w9WgXcQ",
    tracks: Optional[List[Dict[str, Any]]] = None,
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube caption list API response.

    Args:
        video_id: YouTube video ID
        tracks: List of caption track dicts
        **overrides: Additional response fields to override

    Returns:
        Dict with YouTube API caption list response structure
    """
    if tracks is None:
        tracks = [
            create_mock_caption_track(video_id=video_id, language="en", name="English", is_auto_generated=False),
            create_mock_caption_track(video_id=video_id, language="en", name="English (auto-generated)", is_auto_generated=True),
            create_mock_caption_track(video_id=video_id, language="es", name="Spanish", is_auto_generated=False),
        ]

    result = {
        "kind": "youtube#captionListResponse",
        "etag": "etag_caption_list",
        "items": tracks,
    }

    for key, value in overrides.items():
        result[key] = value

    return result


def create_mock_caption_response(
    video_id: str = "dQw4w9WgXcQ",
    language: str = "en",
    is_auto_generated: bool = False,
    segments: Optional[List[Dict[str, Any]]] = None,
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube caption data response (the actual caption content).

    Args:
        video_id: YouTube video ID
        language: ISO 639-1 language code
        is_auto_generated: Whether captions are auto-generated
        segments: List of caption segment dicts
        **overrides: Additional fields to override

    Returns:
        Dict with caption content structure (similar to VTT format parsed)

    Example:
        >>> caption = create_mock_caption_response(video_id="abc123")
        >>> assert caption["video_id"] == "abc123"
        >>> assert len(caption["segments"]) > 0
    """
    if segments is None:
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "text": "Hello and welcome to this video."},
            {"start_time": 5.0, "end_time": 10.0, "text": "Today we're going to discuss an interesting topic."},
            {"start_time": 10.0, "end_time": 15.0, "text": "Let's get started with the introduction."},
            {"start_time": 15.0, "end_time": 20.0, "text": "First, let me explain the basics."},
            {"start_time": 20.0, "end_time": 25.0, "text": "Now we can move on to more advanced concepts."},
        ]

    result = {
        "video_id": video_id,
        "language": language,
        "is_auto_generated": is_auto_generated,
        "segments": segments,
        "format": "vtt" if not is_auto_generated else "srv3",
    }

    for key, value in overrides.items():
        result[key] = value

    return result


def create_mock_no_captions_response(
    video_id: str = "dQw4w9WgXcQ",
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock response indicating no captions available.

    Args:
        video_id: YouTube video ID
        **overrides: Additional fields to override

    Returns:
        Dict indicating no captions available
    """
    result = {
        "video_id": video_id,
        "error": "no_captions_available",
        "message": "This video does not have captions",
        "segments": [],
    }

    for key, value in overrides.items():
        result[key] = value

    return result


# =============================================================================
# Channel API Fixtures
# =============================================================================

def create_mock_youtube_channel(
    channel_id: str = "UC123456",
    title: str = "Test Channel",
    description: str = "This is a test channel",
    subscriber_count: str = "1000000",
    video_count: str = "500",
    view_count: str = "100000000",
    country: str = "US",
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube channel response.

    Args:
        channel_id: YouTube channel ID
        title: Channel title
        description: Channel description
        subscriber_count: Subscriber count string
        video_count: Total video count string
        view_count: Total view count string
        country: Country code
        **overrides: Additional fields to override

    Returns:
        Dict with channel structure
    """
    result = {
        "kind": "youtube#channel",
        "etag": "etag_channel",
        "id": channel_id,
        "snippet": {
            "title": title,
            "description": description,
            "customUrl": f"@{title.lower().replace(' ', '')}",
            "publishedAt": "2020-01-01T00:00:00Z",
            "thumbnails": {
                "default": {"url": f"https://yt3.ggpht.com/ytc/{channel_id}-default", "width": 88, "height": 88},
                "medium": {"url": f"https://yt3.ggpht.com/ytc/{channel_id}-medium", "width": 240, "height": 240},
                "high": {"url": f"https://yt3.ggpht.com/ytc/{channel_id}-high", "width": 800, "height": 800},
            },
        },
        "statistics": {
            "viewCount": view_count,
            "subscriberCount": subscriber_count,
            "videoCount": video_count,
            "hiddenSubscriberCount": False,
        },
        "contentDetails": {
            "relatedPlaylists": {
                "uploads": f"UU{channel_id[2:]}",
                "likes": f"LL{channel_id[2:]}",
            },
        },
    }

    if country:
        result["snippet"]["country"] = country

    for key, value in overrides.items():
        result[key] = value

    return result


# =============================================================================
# Error Response Fixtures
# =============================================================================

def create_mock_youtube_api_error(
    error_code: int = 403,
    error_message: str = "Forbidden",
    error_reason: str = "rateLimitExceeded",
    domain: str = "youtube.api",
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube API error response.

    Args:
        error_code: HTTP error code
        error_message: Error message
        error_reason: YouTube API error reason
        domain: API domain
        **overrides: Additional fields to override

    Returns:
        Dict with error structure
    """
    result = {
        "error": {
            "code": error_code,
            "message": error_message,
            "errors": [
                {
                    "domain": domain,
                    "reason": error_reason,
                    "message": error_message,
                }
            ],
        }
    }

    for key, value in overrides.items():
        if key in result["error"]:
            result["error"][key] = value
        else:
            result[key] = value

    return result


def create_mock_rate_limit_error() -> Dict[str, Any]:
    """Create a mock rate limit (429) error response."""
    return create_mock_youtube_api_error(
        error_code=429,
        error_message="The request quota for this method has been exhausted.",
        error_reason="rateLimitExceeded",
    )


def create_mock_quota_exceeded_error() -> Dict[str, Any]:
    """Create a mock quota exceeded (403) error response."""
    return create_mock_youtube_api_error(
        error_code=403,
        error_message="The request quota for this project has been exhausted.",
        error_reason="quotaExceeded",
    )


def create_mock_not_found_error(video_id: str = "dQw4w9WgXcQ") -> Dict[str, Any]:
    """Create a mock video not found (404) error response."""
    return create_mock_youtube_api_error(
        error_code=404,
        error_message=f"Video not found: {video_id}",
        error_reason="videoNotFound",
    )


# =============================================================================
# Playlist API Fixtures
# =============================================================================

def create_mock_playlist_item(
    video_id: str = "dQw4w9WgXcQ",
    title: str = "Playlist Video",
    channel: str = "Test Channel",
    position: int = 0,
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube playlist item.

    Args:
        video_id: YouTube video ID
        title: Video title
        channel: Channel name
        position: Position in playlist
        **overrides: Additional fields to override

    Returns:
        Dict with playlist item structure
    """
    result = {
        "kind": "youtube#playlistItem",
        "etag": "etag_playlist",
        "id": f"PL_{video_id}_{position}",
        "snippet": {
            "publishedAt": "2024-01-15T10:00:00Z",
            "channelId": "UC123456",
            "title": title,
            "description": f"Video {video_id} in playlist",
            "thumbnails": {
                "default": {"url": f"https://i.ytimg.com/vi/{video_id}/default.jpg", "width": 120, "height": 90},
                "medium": {"url": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg", "width": 320, "height": 180},
                "high": {"url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg", "width": 480, "height": 360},
            },
            "channelTitle": channel,
            "videoOwnerChannelTitle": channel,
            "videoOwnerChannelId": "UC123456",
        },
        "contentDetails": {
            "videoId": video_id,
            "videoPublishedAt": "2024-01-15T10:00:00Z",
        },
    }

    if "snippet" not in overrides:
        overrides["snippet"] = {}
    overrides["snippet"]["position"] = position

    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            obj = result
            for part in parts[:-1]:
                obj = obj.setdefault(part, {})
            obj[parts[-1]] = value
        else:
            result[key] = value

    return result


def create_mock_playlist_list_response(
    playlist_id: str = "PL123456",
    items: Optional[List[Dict[str, Any]]] = None,
    total_results: int = 100,
    **overrides,
) -> Dict[str, Any]:
    """
    Create a mock YouTube playlist items API response.

    Args:
        playlist_id: YouTube playlist ID
        items: List of playlist item dicts
        total_results: Total number of items
        **overrides: Additional response fields to override

    Returns:
        Dict with YouTube API playlist items response structure
    """
    if items is None:
        items = [create_mock_playlist_item(video_id="vid1", position=0)]

    result = {
        "kind": "youtube#playlistItemsListResponse",
        "etag": "etag_playlist_list",
        "pageInfo": {
            "totalResults": total_results,
            "resultsPerPage": len(items),
        },
        "items": items,
    }

    for key, value in overrides.items():
        result[key] = value

    return result


# =============================================================================
# Video Search Result Conversion Helper
# =============================================================================

def youtube_search_to_video_search_result(
    search_result: Dict[str, Any],
    keyword: str = "",
) -> Dict[str, Any]:
    """
    Convert a YouTube search API result to the format expected by VideoSearchResult.

    Args:
        search_result: YouTube search API result dict
        keyword: Keyword used for search

    Returns:
        Dict with VideoSearchResult-compatible fields

    Example:
        >>> search = create_mock_youtube_search_result(video_id="abc")
        >>> result = youtube_search_to_video_search_result(search, keyword="travel")
        >>> assert result["video_id"] == "abc"
        >>> assert result["keyword"] == "travel"
    """
    video_id = search_result.get("id", {}).get("videoId", "")
    snippet = search_result.get("snippet", {})
    content = search_result.get("contentDetails", {})

    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": snippet.get("title", ""),
        "channel": snippet.get("channelTitle", ""),
        "duration": _iso8601_to_seconds(content.get("duration", "PT0S")),
        "keyword": keyword,
        "description": snippet.get("description", ""),
    }


def _iso8601_to_seconds(duration: str) -> int:
    """Convert ISO 8601 duration string to seconds."""
    import re
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Video metadata
    "create_mock_youtube_video_metadata",
    "create_mock_video_list_response",
    # Search
    "create_mock_youtube_search_result",
    "create_mock_search_list_response",
    # Captions
    "create_mock_caption_track",
    "create_mock_caption_list_response",
    "create_mock_caption_response",
    "create_mock_no_captions_response",
    # Channel
    "create_mock_youtube_channel",
    # Errors
    "create_mock_youtube_api_error",
    "create_mock_rate_limit_error",
    "create_mock_quota_exceeded_error",
    "create_mock_not_found_error",
    # Playlist
    "create_mock_playlist_item",
    "create_mock_playlist_list_response",
    # Helpers
    "youtube_search_to_video_search_result",
]
