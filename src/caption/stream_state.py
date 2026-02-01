"""
Stream state classification for YouTube videos.

Provides functionality for classifying video stream states (live, upcoming, VOD, etc.)
to determine appropriate caption handling.
"""

from __future__ import annotations

from typing import Any, Dict

from .enums import StreamState
from .models import StreamStateResult


def classify_stream_state(
    metadata: Dict[str, Any],
    video_id: str = ""
) -> StreamStateResult:
    """Classify a video's stream state from yt-dlp metadata (US-007 Sprint 8).

    Uses multiple metadata fields to accurately classify the video's state:
    1. live_status (most reliable when present)
    2. is_live / was_live boolean flags
    3. release_timestamp (for scheduled premieres)
    4. duration (VOD videos have known duration)

    Args:
        metadata: Dict from yt-dlp --dump-json output.
        video_id: YouTube video ID for logging (optional).

    Returns:
        StreamStateResult with classified state and metadata.

    Classification logic:
        1. live_status='is_live' OR is_live=True -> LIVE
        2. live_status='is_upcoming' -> UPCOMING or PREMIERE (check release_timestamp)
        3. live_status in ('was_live', 'post_live') -> VOD (completed live)
        4. live_status='not_live' with duration -> VOD
        5. was_live=True with duration -> VOD (archived live stream)
        6. release_timestamp in future with no duration -> PREMIERE
        7. Default with duration -> VOD
        8. Default without duration -> UNKNOWN

    Examples:
        >>> # Currently live
        >>> classify_stream_state({'is_live': True, 'live_status': 'is_live'}, 'abc')
        StreamStateResult(state=StreamState.LIVE, video_id='abc', ...)

        >>> # Scheduled premiere
        >>> classify_stream_state({
        ...     'live_status': 'is_upcoming',
        ...     'release_timestamp': 1706400000
        ... }, 'xyz')
        StreamStateResult(state=StreamState.PREMIERE, video_id='xyz', ...)

        >>> # Regular video
        >>> classify_stream_state({'duration': 300, 'live_status': 'not_live'}, 'def')
        StreamStateResult(state=StreamState.VOD, video_id='def', ...)
    """
    # Extract relevant fields
    is_live = metadata.get('is_live', False)
    was_live = metadata.get('was_live', False)
    live_status = metadata.get('live_status')
    release_timestamp = metadata.get('release_timestamp')
    duration = metadata.get('duration')

    # Format scheduled start time if available
    scheduled_start = None
    if release_timestamp:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(float(release_timestamp), tz=timezone.utc)
            scheduled_start = dt.strftime('%Y-%m-%d %H:%M UTC')
        except (ValueError, TypeError, OSError, OverflowError):
            # Invalid timestamp (string, None, out of range), ignore
            pass

    # Build result with common fields
    def make_result(state: StreamState) -> StreamStateResult:
        return StreamStateResult(
            state=state,
            video_id=video_id,
            is_live=bool(is_live),
            was_live=bool(was_live),
            live_status=live_status,
            scheduled_start=scheduled_start,
            duration=duration,
        )

    # 1. Check live_status first (most reliable)
    if live_status:
        live_status_lower = str(live_status).lower()

        if live_status_lower == 'is_live':
            return make_result(StreamState.LIVE)

        if live_status_lower == 'is_upcoming':
            # Distinguish UPCOMING (live stream) vs PREMIERE (pre-recorded)
            # Premieres typically have a release_timestamp and may have duration
            if release_timestamp:
                return make_result(StreamState.PREMIERE)
            return make_result(StreamState.UPCOMING)

        if live_status_lower in ('was_live', 'post_live'):
            # Completed live stream - now VOD
            return make_result(StreamState.VOD)

        if live_status_lower == 'not_live':
            # Definitely not live - VOD if has duration
            if duration is not None and duration > 0:
                return make_result(StreamState.VOD)

    # 2. Check boolean flags if live_status not conclusive
    if is_live:
        return make_result(StreamState.LIVE)

    if was_live:
        # Was live but now complete - VOD with full captions
        if duration is not None and duration > 0:
            return make_result(StreamState.VOD)
        # was_live but no duration - might still be processing
        return make_result(StreamState.UNKNOWN)

    # 3. Check for scheduled premiere without live_status
    if release_timestamp and not duration:
        # Has future release but no duration = not yet available
        import time
        try:
            if float(release_timestamp) > time.time():
                return make_result(StreamState.PREMIERE)
        except (ValueError, TypeError):
            # Invalid release_timestamp, skip this check
            pass

    # 4. Default: VOD if has duration, UNKNOWN otherwise
    if duration is not None and duration > 0:
        return make_result(StreamState.VOD)

    return make_result(StreamState.UNKNOWN)
