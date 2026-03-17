"""
Caption fetcher exceptions.

Contains custom exception classes for caption fetching operations
extracted from caption_fetcher.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .models import ErrorPatternResult


class CaptionError(Exception):
    """Base exception for caption-related errors."""
    pass


class CaptionUnavailableError(CaptionError):
    """Raised when captions are not available for a video.

    This indicates the video genuinely has no captions (auto or manual),
    as opposed to a temporary fetch failure.
    """
    def __init__(self, video_id: str, reason: str = ""):
        self.video_id = video_id
        self.reason = reason
        message = f"No captions available for video {video_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class CaptionFetchError(CaptionError):
    """Raised when caption fetch fails due to a temporary/network error.

    This indicates a potentially retryable failure, not that captions
    don't exist.
    """
    def __init__(self, video_id: str, reason: str = ""):
        self.video_id = video_id
        self.reason = reason
        message = f"Failed to fetch captions for video {video_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class CaptionFormatUnavailableError(CaptionFetchError):
    """Raised when a specific caption format is not available (US-59-004).

    This is a subclass of CaptionFetchError that indicates the requested
    format (e.g., json3) is not available, but other formats may exist.
    Unlike CaptionUnavailableError (video has NO captions at all), this
    error means the video may have captions in a different format.

    The format loop should continue trying other formats when this is raised.
    """
    def __init__(self, video_id: str, fmt: str, reason: str = ""):
        self.format = fmt
        super().__init__(video_id, reason or f"Requested format '{fmt}' is not available")


class CaptionParseWarning(CaptionError):
    """Non-fatal warning for caption parsing issues (US-005).

    This indicates a segment could not be parsed but other segments
    may still be usable. Use for graceful degradation with partial recovery.

    Attributes:
        video_id: YouTube video ID.
        segment_index: Index of the problematic segment.
        reason: Description of the parsing issue.
    """
    def __init__(self, video_id: str, segment_index: int, reason: str = ""):
        self.video_id = video_id
        self.segment_index = segment_index
        self.reason = reason
        message = f"Parse warning for video {video_id} segment {segment_index}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ConfigValidationError(CaptionError):
    """Raised when language configuration is invalid (US-005 Sprint 6).

    This indicates a configuration error that should be fixed before
    running the pipeline. Invalid configurations will cause silent failures
    during fetch.

    Attributes:
        field: The config field with the issue (e.g., 'fallback_languages').
        value: The invalid value.
        reason: Description of why validation failed.
        suggestion: Suggested fix for the issue.
    """
    def __init__(
        self,
        field: str,
        value: Any,
        reason: str = "",
        suggestion: str = ""
    ):
        self.field = field
        self.value = value
        self.reason = reason
        self.suggestion = suggestion
        message = f"Invalid language configuration for '{field}': {value}"
        if reason:
            message += f" - {reason}"
        if suggestion:
            message += f". Suggestion: {suggestion}"
        super().__init__(message)


class ErrorPatternAbortError(CaptionError):
    """Raised when batch fetch is aborted due to detected error pattern (US-007 Sprint 7).

    This exception is raised when abort_on_error_pattern='abort' and a pattern
    is detected (e.g., 30%+ of videos failing with the same error).

    Attributes:
        pattern_result: The ErrorPatternResult with detection details.
        partial_results: Dict of results collected before abort.
    """
    def __init__(
        self,
        pattern_result: 'ErrorPatternResult',
        partial_results: Optional[Dict[str, Any]] = None
    ):
        self.pattern_result = pattern_result
        self.partial_results = partial_results or {}
        message = str(pattern_result)
        super().__init__(message)


class CaptionFormatExhaustedError(CaptionFetchError):
    """Raised when all unique caption formats have been tried and failed (US-67-002).

    Unlike CaptionFormatUnavailableError (single format not available) or
    CaptionUnavailableError (no captions at all), this error means every
    format in the preference list was attempted exactly once and none succeeded.

    Attributes:
        formats_tried: List of format names that were attempted.
        formats_skipped: Number of formats skipped (e.g., due to budget exhaustion).
        last_error: The last error encountered during format attempts.
    """
    def __init__(
        self,
        video_id: str,
        formats_tried: Optional[list] = None,
        formats_skipped: int = 0,
        reason: str = "",
        last_error: Optional[Exception] = None
    ):
        self.formats_tried = formats_tried or []
        self.formats_skipped = formats_skipped
        self.last_error = last_error
        detail = f"All {len(self.formats_tried)} formats exhausted: {self.formats_tried}"
        if formats_skipped:
            detail += f" ({formats_skipped} skipped)"
        super().__init__(video_id, reason or detail)


class CaptionNormalizationError(CaptionError):
    """Raised when caption normalization fails.

    This indicates that the normalization process could not complete,
    typically due to too many malformed segments exceeding the partial
    recovery threshold.

    Attributes:
        reason: Description of why normalization failed.
    """
    def __init__(self, reason: str = ""):
        self.reason = reason
        message = "Caption normalization failed"
        if reason:
            message += f": {reason}"
        super().__init__(message)
