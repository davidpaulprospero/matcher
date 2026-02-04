"""
Caption fetcher enumerations.

Contains CaptionErrorCategory, StreamState, and other enums
extracted from caption_fetcher.py.
"""

from __future__ import annotations

from enum import Enum, auto


class CaptionErrorCategory(Enum):
    """Error categories for caption fetch failures (US-003 Sprint 7).

    Different error types have different retry strategies:
    - NETWORK: Transient network issues, should retry aggressively
    - TIMEOUT: Request timeouts, may retry but with longer delays
    - PARSE: Content parsing failed, unlikely to succeed on retry
    - UNAVAILABLE: Video has no captions, should not retry
    - RATE_LIMIT: API rate limit hit, should retry after delay

    Each category has a configurable retry budget in CaptionFirstConfig.
    The retry decision logged includes the category for debugging.

    Example log output:
        "NETWORK error, retry 2/3"
        "PARSE error, no retry (budget: 1)"
        "RATE_LIMIT error, retry 1/2 after 30s backoff"
    """
    NETWORK = auto()    # Network connectivity issues, DNS failures
    TIMEOUT = auto()    # Request/connection timeouts
    PARSE = auto()      # Caption content parsing failures
    UNAVAILABLE = auto()  # No captions exist for the video
    FORMAT_UNAVAILABLE = auto()  # Specific format unavailable, others may exist (US-59-004)
    RATE_LIMIT = auto()   # API rate limiting (429, quota exceeded)


class StreamState(Enum):
    """Stream state classification for YouTube videos (US-007 Sprint 8).

    Classifies videos into stream states to determine appropriate handling:
    - LIVE: Currently streaming, captions being generated in real-time
    - UPCOMING: Scheduled stream/premiere, will have captions when complete
    - VOD: Regular video-on-demand (completed video with full captions)
    - PREMIERE: Scheduled premiere (pre-recorded video with countdown)
    - UNKNOWN: Could not determine state (metadata fetch failed)

    Stream state affects caption handling:
    - LIVE: Skip (captions incomplete/unavailable)
    - UPCOMING: Based on config.handle_upcoming ('skip', 'queue', 'check_later')
    - VOD: Normal caption fetch
    - PREMIERE: Same as UPCOMING (will become VOD when complete)
    - UNKNOWN: Assume VOD (default to normal behavior)

    yt-dlp metadata fields used:
    - is_live: True if currently streaming
    - was_live: True if video was previously live
    - live_status: 'is_live', 'is_upcoming', 'was_live', 'post_live', 'not_live'
    - release_timestamp: Unix timestamp for scheduled premiere/stream

    Example:
        >>> state = classify_stream_state(metadata)
        >>> if state == StreamState.LIVE:
        ...     print("Skipping live stream")
        >>> elif state == StreamState.UPCOMING:
        ...     print(f"Queuing for later: scheduled {metadata.get('release_timestamp')}")
    """
    LIVE = auto()       # Currently streaming
    UPCOMING = auto()   # Scheduled stream/premiere (not yet started)
    VOD = auto()        # Video-on-demand (regular completed video)
    PREMIERE = auto()   # Scheduled premiere (pre-recorded, with countdown)
    UNKNOWN = auto()    # Could not determine state


# Default retry budgets per error category (US-003 Sprint 7)
# These can be overridden in CaptionFirstConfig.retry_budgets
DEFAULT_RETRY_BUDGETS = {
    CaptionErrorCategory.NETWORK: 3,      # Network errors retry aggressively
    CaptionErrorCategory.TIMEOUT: 2,      # Timeouts get moderate retries
    CaptionErrorCategory.PARSE: 1,        # Parse errors rarely succeed on retry
    CaptionErrorCategory.UNAVAILABLE: 0,  # Never retry - video has no captions
    CaptionErrorCategory.FORMAT_UNAVAILABLE: 0,  # Don't retry - format loop handles this
    CaptionErrorCategory.RATE_LIMIT: 2,   # Rate limits retry with backoff
}
