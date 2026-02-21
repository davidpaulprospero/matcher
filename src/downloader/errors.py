"""
Typed exception hierarchy for download errors.

US-82-002: Replaces string-based error classification with a typed
exception hierarchy. Each subclass stores structured fields (category,
severity, retryable, original_message) so downstream code can use
isinstance() instead of string comparison or regex re-matching.

Extends the existing DownloadError from types.py (which carries retry
context) with classification metadata. This means all typed errors
are also valid DownloadError instances for backward compatibility.

The hierarchy:
    DownloadError (from types.py — base with retry context)
    └── ClassifiedDownloadError (adds category/severity/retryable)
        ├── NetworkError          - DNS, TCP, TLS, ffmpeg network failures
        ├── BotDetectionError     - 403, captcha, sign-in, bot blocks
        ├── RateLimitError        - 429, quota exceeded, throttling
        ├── FormatError           - Requested format unavailable, video-specific
        ├── AuthenticationError   - Age-gate, login required
        └── TimeoutError_         - Stall/socket timeouts (underscore avoids builtin)

US-93-006: Enhanced with structured error response class, error codes,
user-friendly messages, and recovery suggestions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from .types import DownloadError


class ClassifiedDownloadError(DownloadError):
    """Download error with classification metadata.

    Extends the base DownloadError (retry context) with structured
    fields for category, severity, and retryability. classify_error_category()
    returns instances of subclasses of this class.
    """

    category: str = 'video_specific'
    severity: str = 'medium'
    retryable: bool = True

    def __init__(
        self,
        original_message: str,
        *,
        category: str | None = None,
        severity: str | None = None,
        retryable: bool | None = None,
        # US-153-007: Accept error_type to pass through to DownloadError
        error_type: str | None = None,
    ) -> None:
        self.original_message = original_message
        if category is not None:
            self.category = category
        if severity is not None:
            self.severity = severity
        if retryable is not None:
            self.retryable = retryable
        # US-153-007: Pass error_type through to parent (DownloadError)
        super().__init__(original_message, error_type=error_type)

    def __str__(self) -> str:
        return self.original_message

    def __eq__(self, other: object) -> bool:
        """Support comparison with category strings for backward compatibility.

        This allows ``classify_error_category(msg) == 'network'`` to keep
        working even though the return type is now a ClassifiedDownloadError.
        """
        if isinstance(other, str):
            return self.category == other
        if isinstance(other, ClassifiedDownloadError):
            return self.category == other.category and self.original_message == other.original_message
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return hash(self.category)


class NetworkError(ClassifiedDownloadError):
    """Systemic network failure: DNS, TCP, TLS, ffmpeg network codes.

    These affect ALL downloads (no internet, DNS down) and should not
    be retried individually — the whole batch is doomed.
    """

    category = 'network'
    severity = 'high'
    retryable = False


class BotDetectionError(ClassifiedDownloadError):
    """Bot detection / access block: 403, captcha, sign-in prompts.

    Warrants escalation to a higher bypass tier (Tier 2/3/4).
    """

    category = 'bot_detection'
    severity = 'high'
    retryable = True


class RateLimitError(ClassifiedDownloadError):
    """Rate limiting: 429, quota exceeded, throttling.

    US-114-010: Added retry_after field to store Retry-After header value.
    Retry-After takes precedence over calculated exponential backoff.

    Args:
        original_message: The error message string.
        retry_after: Optional seconds to wait before retry (from Retry-After header).
    """

    category = 'bot_detection'  # Matches existing classification behavior
    severity = 'medium'
    retryable = True
    retry_after: Optional[float] = None

    def __init__(
        self,
        original_message: str,
        *,
        retry_after: Optional[float] = None,
        severity: str | None = None,
        **kwargs
    ) -> None:
        # Call ClassifiedDownloadError directly to properly handle severity
        ClassifiedDownloadError.__init__(
            self,
            original_message,
            severity=severity,
            **kwargs
        )
        self.retry_after = retry_after


class FormatError(ClassifiedDownloadError):
    """Format unavailable or video-specific error.

    The video itself has an issue (removed, unavailable format, etc.).
    """

    category = 'video_specific'
    severity = 'low'
    retryable = False


class AuthenticationError(ClassifiedDownloadError):
    """Authentication required: age-gate, login required.

    Retryable with cookies/auth but not without.
    """

    category = 'bot_detection'  # Matches existing: sign-in triggers escalation
    severity = 'low'
    retryable = True


class TimeoutError_(ClassifiedDownloadError):
    """Timeout: stall timeout, socket timeout, deadline exceeded.

    Trailing underscore avoids shadowing the builtin TimeoutError.
    """

    category = 'timeout'
    severity = 'medium'
    retryable = True


class YouTubeAPIError(ClassifiedDownloadError):
    """YouTube Data API specific error with classification and actionable guidance.

    US-148-012: Extends ClassifiedDownloadError to integrate with the
    existing error classification system. Provides specific error types
    with actionable recovery suggestions.

    US-156-010: Added error_code field for YouTube Data API error codes (100-110).

    Error types:
    - quota_exceeded: API quota exhausted, can reset at midnight PST
    - rate_limited: HTTP 429, wait before retrying
    - invalid_key: API key invalid or lacks permissions
    - permission_denied: Access denied for requested scope
    - network_error: Network connectivity issues
    """

    category = 'youtube_api'
    severity = 'medium'
    retryable = True

    # Error type constants
    TYPE_QUOTA_EXCEEDED = 'quota_exceeded'
    TYPE_RATE_LIMITED = 'rate_limited'
    TYPE_INVALID_KEY = 'invalid_key'
    TYPE_PERMISSION_DENIED = 'permission_denied'
    TYPE_NETWORK_ERROR = 'network_error'

    def __init__(
        self,
        original_message: str,
        *,
        error_type: str = 'youtube_api',
        retry_after: Optional[float] = None,
        endpoint: str = "",
        error_code: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize YouTubeAPIError.

        Args:
            original_message: The error message string.
            error_type: Specific error type (quota_exceeded, rate_limited, etc.)
            retry_after: Optional seconds to wait before retry (for rate limiting).
            endpoint: Optional API endpoint where error occurred.
            error_code: Optional YouTube Data API error code (100-110).
        """
        self.error_type = error_type
        self.retry_after = retry_after
        self.endpoint = endpoint
        self.error_code = error_code  # US-156-010: Add error_code field
        self.actionable_guidance = self._get_guidance(error_type)
        # US-153-007: Explicitly pass error_type to parent to avoid default 'unknown'
        super().__init__(original_message, error_type=error_type, **kwargs)

    def _get_guidance(self, error_type: str) -> str:
        """Get actionable guidance for the error type.

        Args:
            error_type: The specific error type.

        Returns:
            Actionable guidance string for recovery.
        """
        guidance_map = {
            self.TYPE_QUOTA_EXCEEDED: (
                "Switch to yt-dlp fallback for remaining searches. "
                "API quota resets at midnight PST. Check Google Cloud Console "
                "(https://console.cloud.google.com/apis/dashboard) for quota usage. "
                "Consider requesting quota increase or using multiple API keys."
            ),
            self.TYPE_RATE_LIMITED: (
                "Rate limited (HTTP 429). Wait before retrying. "
                "Suggested next steps: (1) Wait 60-100 seconds before retry, "
                "(2) Reduce request frequency, (3) Use multiple API keys for rotation, "
                "(4) Enable yt-dlp fallback in config.yaml (video_search.youtube_api.fallback_to_yt_dlp: true)."
            ),
            self.TYPE_INVALID_KEY: (
                "API key is invalid or lacks required permissions. "
                "Suggested next steps: (1) Verify key in Google Cloud Console "
                "(https://console.cloud.google.com/apis/credentials), "
                "(2) Ensure YouTube Data API v3 is enabled, "
                "(3) Check key has no IP or HTTP referrer restrictions, "
                "(4) Create new API key if needed."
            ),
            self.TYPE_PERMISSION_DENIED: (
                "API key lacks required scopes or permissions. "
                "Suggested next steps: (1) Enable YouTube Data API v3 in Google Cloud Console, "
                "(2) Verify key has no API restrictions, "
                "(3) Check project billing is enabled, "
                "(4) Consider falling back to yt-dlp if issues persist."
            ),
            self.TYPE_NETWORK_ERROR: (
                "Network connectivity issues detected. "
                "Suggested next steps: (1) Check internet connection, "
                "(2) Verify firewall/proxy settings allow Google APIs, "
                "(3) Try enabling yt-dlp fallback in config.yaml, "
                "(4) Check for VPN/proxy that may be blocking requests."
            ),
        }
        return guidance_map.get(error_type, "Refer to YouTube Data API documentation for error details.")

    def __str__(self) -> str:
        """Return error string with actionable guidance."""
        guidance = f" - {self.actionable_guidance}" if self.actionable_guidance else ""
        error_code_str = f" [Error {self.error_code}]" if self.error_code is not None else ""
        return f"{self.original_message}{error_code_str}{guidance}"


# =============================================================================
# US-156-010: YouTube Data API Error Code Mapping
# =============================================================================


class YouTubeAPIErrorCode(Enum):
    """YouTube Data API specific error codes.

    US-156-010: Maps YouTube Data API error codes to specific error types
    and provides human-readable descriptions for each error code.

    Reference: https://developers.google.com/youtube/v3/docs/errors
    """

    # 1xx - Authentication and permission errors
    ERR_API_DISABLED = 100  # The API is not enabled for the project
    ERR_NOT_FOUND = 101  # Video or resource not found
    ERR_INVALID_PARAMETER = 102  # Request contains invalid parameter
    ERR_PERMISSION_DENIED = 103  # Permission denied for the request

    # 2xx - Request errors
    ERR_MISSING_REQUIRED_PARAMETER = 200  # Missing required parameter
    ERR_INVALID_FILTER = 400  # Invalid filter parameter in request

    # 4xx - Client errors
    ERR_INVALID_API_KEY = 401  # API key is invalid
    ERR_QUOTA_EXCEEDED = 402  # API quota exceeded
    ERR_RATE_LIMIT_EXCEEDED = 403  # Rate limit exceeded (HTTP 429)
    ERR_RESOURCE_NOT_FOUND = 404  # Requested resource not found
    ERR_RESOURCE_LENGTH_REQUIRED = 411  # Content-Length required for upload
    ERR_INVALID_CONTENT_RANGE = 416  # Invalid content range

    # 5xx - Server errors
    ERR_INTERNAL_ERROR = 500  # Internal server error
    ERR_BACKEND_ERROR = 503  # Backend error
    ERR_UNAVAILABLE = 501  # Not implemented

    # Unknown error code
    ERR_UNKNOWN = 999


# =============================================================================
# US-158-008: Additional YouTube API Error Type Mappings
# =============================================================================

# Error "reason" values from YouTube API error response (errors[].reason field)
# These are the specific error types mentioned in the acceptance criteria
YOUTUBE_API_ERROR_REASONS: Dict[str, Dict[str, Any]] = {
    # notFound - Resource not found (video deleted, channel removed, etc.)
    "notFound": {
        "error_type": "not_found",
        "description": "The requested resource was not found. The video may have been removed or the channel may have been deleted.",
        "severity": "low",
        "retryable": False,
    },
    # closed - Live stream or video is closed/ended
    "closed": {
        "error_type": "closed",
        "description": "The live stream or video is no longer available. It may have ended or been closed.",
        "severity": "low",
        "retryable": False,
    },
    # embeddingDisabled - Video embedding is disabled by owner
    "embeddingDisabled": {
        "error_type": "embedding_disabled",
        "description": "Video embedding is disabled. The video owner has restricted embedding.",
        "severity": "low",
        "retryable": False,
    },
    # uploadFailed - Video upload failed
    "uploadFailed": {
        "error_type": "upload_failed",
        "description": "The video upload failed. There may be an issue with the video file or upload parameters.",
        "severity": "medium",
        "retryable": True,
    },
    # invalidRequest - General invalid request
    "invalidRequest": {
        "error_type": "invalid_request",
        "description": "The request was invalid. Check request parameters and format.",
        "severity": "medium",
        "retryable": False,
    },
    # playbackQuotaExceeded - Playback quota exceeded
    "playbackQuotaExceeded": {
        "error_type": "quota_exceeded",
        "description": "Playback quota exceeded. Too many requests from this source.",
        "severity": "high",
        "retryable": False,
    },
    # dailyLimitExceeded - Daily quota limit exceeded
    "dailyLimitExceeded": {
        "error_type": "daily_quota_exceeded",
        "description": "Daily quota limit exceeded. Quota resets at midnight PST.",
        "severity": "high",
        "retryable": False,
    },
    # quotaExceeded - General quota exceeded
    "quotaExceeded": {
        "error_type": "quota_exceeded",
        "description": "API quota exceeded. Quota resets at midnight PST.",
        "severity": "high",
        "retryable": False,
    },
    # rateLimitExceeded - Rate limit exceeded
    "rateLimitExceeded": {
        "error_type": "rate_limited",
        "description": "Rate limit exceeded. Wait before retrying.",
        "severity": "medium",
        "retryable": True,
    },
    # accessNotConfigured - API not configured for the project
    "accessNotConfigured": {
        "error_type": "permission_denied",
        "description": "API access not configured. Enable YouTube Data API v3 in Google Cloud Console.",
        "severity": "critical",
        "retryable": False,
    },
    # forbidden - Access forbidden
    "forbidden": {
        "error_type": "permission_denied",
        "description": "Access forbidden. Check API key permissions and scopes.",
        "severity": "high",
        "retryable": False,
    },
    # asyncError - Asynchronous operation failed
    "asyncError": {
        "error_type": "temporary_error",
        "description": "Asynchronous operation failed. This is usually temporary, retry later.",
        "severity": "low",
        "retryable": True,
    },
    # serviceUnavailable - Service temporarily unavailable
    "serviceUnavailable": {
        "error_type": "temporary_error",
        "description": "Service temporarily unavailable. This is usually temporary, retry later.",
        "severity": "low",
        "retryable": True,
    },
}

# Error domain to human-readable description mapping
YOUTUBE_API_ERROR_DOMAINS: Dict[str, str] = {
    # YouTube API service domains
    "youtube": "YouTube API",
    "youtubePartner": "YouTube Partner API",
    "ytks": "YouTube Toolkit Service",

    # Specific API domains
    "youtube.search": "YouTube Search API",
    "youtube.video": "YouTube Video API",
    "youtube.channel": "YouTube Channel API",
    "youtube.playlist": "YouTube Playlist API",
    "youtube.comment": "YouTube Comment API",
    "youtube.caption": "YouTube Caption API",
    "youtubeStreaming": "YouTube Streaming API",

    # Upload domains
    "youtube.upload": "YouTube Upload API",
    "resumableUpload": "Resumable Upload API",

    # Generic domains
    "global": "Global API Error",
    "backendError": "Backend Error",
    "authError": "Authentication Error",
    "quotaError": "Quota Error",
}


def get_error_domain_info(domain: str) -> str:
    """Get human-readable description for a YouTube API error domain.

    US-158-008: Maps YouTube API error domains to readable descriptions
    for better debugging and logging.

    Args:
        domain: The error domain string (e.g., 'youtube.search').

    Returns:
        Human-readable domain description.
    """
    if domain in YOUTUBE_API_ERROR_DOMAINS:
        return YOUTUBE_API_ERROR_DOMAINS[domain]
    return f"Unknown domain: {domain}"


def get_error_reason_info(reason: str) -> Dict[str, Any]:
    """Get error information for a YouTube API error reason string.

    US-158-008: Maps YouTube Data API error "reason" values to structured
    information including error type, description, severity, and retryability.

    Args:
        reason: The YouTube API error reason string (e.g., 'notFound', 'closed').

    Returns:
        Dict with error_type, description, severity, and retryable keys.
        Returns default values for unknown reasons.
    """
    if reason in YOUTUBE_API_ERROR_REASONS:
        return YOUTUBE_API_ERROR_REASONS[reason].copy()

    # Default for unknown error reasons
    return {
        "error_type": "unknown",
        "description": f"Unknown YouTube API error reason: {reason}",
        "severity": "medium",
        "retryable": True,
    }


def parse_youtube_api_error_response(error_response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse YouTube API error response for comprehensive error information.

    US-158-008: Extracts error code, reason, domain, and message from the
    YouTube API error response format.

    Args:
        error_response: The JSON error response from YouTube API.

    Returns:
        Dict with parsed error information: code, reason, domain, message.
    """
    result = {
        "code": None,
        "reason": None,
        "domain": None,
        "message": "",
    }

    error = error_response.get("error", {})
    if not error:
        return result

    # Extract HTTP-like error code
    result["code"] = error.get("code")

    # Extract main error message
    result["message"] = error.get("message", "")

    # Extract from errors array (most detailed)
    errors = error.get("errors", [])
    if errors:
        first_error = errors[0]
        result["reason"] = first_error.get("reason")
        result["domain"] = first_error.get("domain")
        # Use more specific message if available
        if first_error.get("message"):
            result["message"] = first_error["message"]

    return result


# Mapping from YouTube API error codes to error type strings and descriptions
ERROR_CODE_MAPPING: Dict[int, Dict[str, Any]] = {
    # 1xx - Authentication and permission errors
    100: {
        "error_type": YouTubeAPIError.TYPE_PERMISSION_DENIED,
        "description": "The API is not enabled for the project. Enable YouTube Data API v3 in Google Cloud Console.",
        "severity": "critical",
        "retryable": False,
    },
    101: {
        "error_type": "not_found",
        "description": "The requested resource was not found. The video may have been removed.",
        "severity": "low",
        "retryable": False,
    },
    102: {
        "error_type": "invalid_parameter",
        "description": "The request contains an invalid parameter. Check request parameters.",
        "severity": "medium",
        "retryable": False,
    },
    103: {
        "error_type": YouTubeAPIError.TYPE_PERMISSION_DENIED,
        "description": "Permission denied for the request. Check API key permissions and scopes.",
        "severity": "high",
        "retryable": False,
    },

    # 2xx - Request errors
    200: {
        "error_type": "missing_parameter",
        "description": "Missing required parameter in request. Check required parameters for the API call.",
        "severity": "medium",
        "retryable": False,
    },
    400: {
        "error_type": "invalid_filter",
        "description": "Invalid filter parameter in request. Check filter syntax.",
        "severity": "medium",
        "retryable": False,
    },

    # 4xx - Client errors
    401: {
        "error_type": YouTubeAPIError.TYPE_INVALID_KEY,
        "description": "API key is invalid. Verify API key in Google Cloud Console.",
        "severity": "critical",
        "retryable": False,
    },
    402: {
        "error_type": YouTubeAPIError.TYPE_QUOTA_EXCEEDED,
        "description": "API quota exceeded. Quota resets at midnight PST. Use yt-dlp fallback.",
        "severity": "high",
        "retryable": False,
    },
    403: {
        "error_type": YouTubeAPIError.TYPE_RATE_LIMITED,
        "description": "Rate limit exceeded. Wait before retrying or use multiple API keys.",
        "severity": "medium",
        "retryable": True,
    },
    404: {
        "error_type": "not_found",
        "description": "Requested resource not found. The video may have been removed.",
        "severity": "low",
        "retryable": False,
    },
    411: {
        "error_type": "invalid_request",
        "description": "Content-Length required for upload request.",
        "severity": "medium",
        "retryable": False,
    },
    416: {
        "error_type": "invalid_request",
        "description": "Invalid content range in request.",
        "severity": "medium",
        "retryable": False,
    },

    # 5xx - Server errors
    500: {
        "error_type": "temporary_error",
        "description": "Internal server error. This is usually temporary, retry later.",
        "severity": "low",
        "retryable": True,
    },
    501: {
        "error_type": "not_implemented",
        "description": "The requested functionality is not implemented.",
        "severity": "medium",
        "retryable": False,
    },
    503: {
        "error_type": "temporary_error",
        "description": "Backend error. This is usually temporary, retry later.",
        "severity": "low",
        "retryable": True,
    },
}


# =============================================================================
# US-158-008: YouTube API Error Reason Mapping
# =============================================================================

# Mapping from YouTube API error "reason" field to error information
# These reasons appear in the "errors[].reason" field of API error responses
ERROR_REASON_MAPPING: Dict[str, Dict[str, Any]] = {
    # notFound - Resource not found
    "notFound": {
        "error_type": "not_found",
        "description": "The requested resource was not found. The video may have been removed or set to private.",
        "severity": "low",
        "retryable": False,
        "http_status": 404,
    },
    # invalidRequest - Invalid request parameters
    "invalidRequest": {
        "error_type": "invalid_request",
        "description": "The request contains invalid parameters. Check request parameters and try again.",
        "severity": "medium",
        "retryable": False,
        "http_status": 400,
    },
    # closed - Video has been closed/removed
    "closed": {
        "error_type": "closed",
        "description": "The video has been closed and is no longer available.",
        "severity": "low",
        "retryable": False,
        "http_status": 404,
    },
    # embeddingDisabled - Video embedding is disabled
    "embeddingDisabled": {
        "error_type": "embedding_disabled",
        "description": "Embedding is disabled for this video. The video cannot be embedded.",
        "severity": "low",
        "retryable": False,
        "http_status": 403,
    },
    # uploadFailed - Upload failed
    "uploadFailed": {
        "error_type": "upload_failed",
        "description": "The upload failed. This may be a temporary issue with the upload process.",
        "severity": "medium",
        "retryable": True,
        "http_status": 500,
    },
    # forbidden - Access forbidden
    "forbidden": {
        "error_type": "forbidden",
        "description": "Access to this resource is forbidden. Check API key permissions.",
        "severity": "high",
        "retryable": False,
        "http_status": 403,
    },
    # quotaExceeded - API quota exceeded
    "quotaExceeded": {
        "error_type": "quota_exceeded",
        "description": "API quota exceeded. Quota resets at midnight PST.",
        "severity": "high",
        "retryable": False,
        "http_status": 403,
    },
    # rateLimitExceeded - Rate limit exceeded
    "rateLimitExceeded": {
        "error_type": "rate_limited",
        "description": "Rate limit exceeded. Wait before retrying.",
        "severity": "medium",
        "retryable": True,
        "http_status": 429,
    },
    # backendError - Backend error
    "backendError": {
        "error_type": "temporary_error",
        "description": "Backend error. This is usually temporary, retry later.",
        "severity": "low",
        "retryable": True,
        "http_status": 503,
    },
    # internalError - Internal server error
    "internalError": {
        "error_type": "temporary_error",
        "description": "Internal server error. This is usually temporary, retry later.",
        "severity": "low",
        "retryable": True,
        "http_status": 500,
    },
    # unknown - Fallback for unknown reasons
    "unknown": {
        "error_type": "unknown",
        "description": "An unknown error occurred. Check logs for details.",
        "severity": "medium",
        "retryable": True,
        "http_status": 0,
    },
}


def get_error_by_reason(reason: str) -> Dict[str, Any]:
    """Get error information by YouTube API error reason.

    US-158-008: Maps YouTube Data API error "reason" field to structured
    information including error type, description, severity, and retryability.

    Args:
        reason: The error reason string from YouTube API error response.

    Returns:
        Dict with error_type, description, severity, retryable, and http_status.
        Returns default values for unknown reasons.
    """
    if reason in ERROR_REASON_MAPPING:
        return ERROR_REASON_MAPPING[reason].copy()

    # Default for unknown reasons
    return {
        "error_type": "unknown",
        "description": f"Unknown YouTube API error reason: {reason}",
        "severity": "medium",
        "retryable": True,
        "http_status": 0,
    }


def get_error_reason_description(reason: str) -> str:
    """Get human-readable description for a YouTube API error reason.

    Args:
        reason: The error reason string.

    Returns:
        Human-readable error description string.
    """
    info = get_error_by_reason(reason)
    return info.get("description", "Unknown error")


def get_error_reason_type(reason: str) -> str:
    """Get error type string for a YouTube API error reason.

    Args:
        reason: The error reason string.

    Returns:
        Error type string (e.g., 'not_found', 'quota_exceeded').
    """
    info = get_error_by_reason(reason)
    return info.get("error_type", "unknown")


def get_error_by_reason_action(reason: str) -> ErrorAction:
    """Get the recommended action for a YouTube API error reason.

    US-158-008: Maps error reasons to recovery actions.

    Args:
        reason: The error reason string.

    Returns:
        The ErrorAction to take for this error reason.
    """
    # Map reason strings to ErrorAction
    reason_to_action: Dict[str, ErrorAction] = {
        "notFound": ErrorAction.ABORT,
        "closed": ErrorAction.ABORT,
        "embeddingDisabled": ErrorAction.ABORT,
        "invalidRequest": ErrorAction.ABORT,
        "forbidden": ErrorAction.ABORT,
        "quotaExceeded": ErrorAction.FALLBACK,
        "rateLimitExceeded": ErrorAction.BACKOFF,
        "backendError": ErrorAction.BACKOFF,
        "internalError": ErrorAction.BACKOFF,
        "uploadFailed": ErrorAction.RETRY,
        "unknown": ErrorAction.BACKOFF,
    }
    return reason_to_action.get(reason, ErrorAction.BACKOFF)


def get_error_category(error_code: int) -> Dict[str, str]:
    """Get error category information for a YouTube API error code.

    US-156-010: Maps YouTube Data API error codes to structured information
    including error type, description, severity, and retryability.

    Args:
        error_code: The YouTube Data API error code (100-110 or standard HTTP codes).

    Returns:
        Dict with error_type, description, severity, and retryable keys.
        Returns default values for unknown error codes.
    """
    if error_code in ERROR_CODE_MAPPING:
        return ERROR_CODE_MAPPING[error_code].copy()

    # Default for unknown error codes
    return {
        "error_type": "unknown",
        "description": f"Unknown YouTube API error code: {error_code}",
        "severity": "medium",
        "retryable": True,
    }


def get_error_description(error_code: int) -> str:
    """Get human-readable description for a YouTube API error code.

    Args:
        error_code: The YouTube Data API error code.

    Returns:
        Human-readable error description string.
    """
    category = get_error_category(error_code)
    return category.get("description", "Unknown error")


def get_error_type_for_code(error_code: int) -> str:
    """Get error type string for a YouTube API error code.

    Args:
        error_code: The YouTube Data API error code.

    Returns:
        Error type string (e.g., 'quota_exceeded', 'rate_limited').
    """
    category = get_error_category(error_code)
    return category.get("error_type", "unknown")


# =============================================================================
# US-157-005: Error Code to Action Mapping
# =============================================================================


class ErrorAction(Enum):
    """Actions to take when encountering specific error codes.

    US-157-005: Maps YouTube Data API error codes to specific actions
    that determine the next step in error recovery.
    """

    # Retry immediately with the same request
    RETRY = "retry"

    # Retry with exponential backoff
    BACKOFF = "backoff"

    # Skip this item and continue with the next
    SKIP = "skip"

    # Switch to fallback (yt-dlp)
    FALLBACK = "fallback"

    # Rotate to a different API key
    ROTATE_KEY = "rotate_key"

    # Abort the operation entirely
    ABORT = "abort"

    # Wait and retry (for rate limits with Retry-After)
    WAIT = "wait"


# Mapping from error codes to recommended actions
ERROR_CODE_TO_ACTION: Dict[int, ErrorAction] = {
    # 1xx - Authentication and permission errors
    100: ErrorAction.ABORT,  # API disabled - cannot proceed
    101: ErrorAction.SKIP,   # Not found - skip this item
    102: ErrorAction.ABORT,  # Invalid parameter - fix code
    103: ErrorAction.ABORT,  # Permission denied - cannot proceed

    # 2xx - Request errors
    200: ErrorAction.ABORT,   # Missing parameter - fix code
    400: ErrorAction.ABORT,  # Invalid filter - fix code

    # 4xx - Client errors
    401: ErrorAction.ROTATE_KEY,  # Invalid key - try different key
    402: ErrorAction.FALLBACK,    # Quota exceeded - use yt-dlp
    403: ErrorAction.BACKOFF,     # Rate limited - backoff and retry
    404: ErrorAction.SKIP,        # Not found - skip this item
    411: ErrorAction.ABORT,      # Invalid request - fix code
    416: ErrorAction.ABORT,       # Invalid content range - fix code

    # 5xx - Server errors
    500: ErrorAction.BACKOFF,  # Internal error - retry with backoff
    501: ErrorAction.ABORT,   # Not implemented - cannot proceed
    503: ErrorAction.BACKOFF,  # Backend error - retry with backoff
}


def get_error_action(error_code: int) -> ErrorAction:
    """Get the recommended action for a YouTube API error code.

    US-157-005: Maps error codes to specific recovery actions.

    Args:
        error_code: The YouTube Data API error code.

    Returns:
        The ErrorAction to take for this error code.
    """
    return ERROR_CODE_TO_ACTION.get(error_code, ErrorAction.RETRY)


# Error type to action mapping (more general than specific error codes)
ERROR_TYPE_TO_ACTION: Dict[str, ErrorAction] = {
    # Network errors
    "network": ErrorAction.BACKOFF,
    "dns_timeout": ErrorAction.BACKOFF,
    "dns_resolution": ErrorAction.ABORT,
    "connection_timeout": ErrorAction.BACKOFF,
    "tls_handshake": ErrorAction.BACKOFF,
    "timeout": ErrorAction.BACKOFF,

    # Rate limit errors
    "rate_limited": ErrorAction.WAIT,
    "per_second_limit": ErrorAction.WAIT,
    "429": ErrorAction.WAIT,

    # Quota errors
    "quota_exceeded": ErrorAction.FALLBACK,
    "daily_quota_exceeded": ErrorAction.FALLBACK,
    "quota_error": ErrorAction.ROTATE_KEY,

    # Authentication errors
    "invalid_key": ErrorAction.ROTATE_KEY,
    "invalid_project": ErrorAction.ABORT,
    "disabled_project": ErrorAction.ABORT,
    "permission_denied": ErrorAction.FALLBACK,

    # Network errors from YouTube API
    "network_error": ErrorAction.BACKOFF,

    # Bot detection errors
    "bot_detection": ErrorAction.FALLBACK,

    # Temporary errors - retry with backoff
    "temporary_error": ErrorAction.BACKOFF,

    # Geo-blocking - escalate to VPN
    "geo_blocked": ErrorAction.FALLBACK,

    # Premium required - no retry without premium
    "premium_required": ErrorAction.SKIP,

    # Device limit - wait and retry
    "device_limit": ErrorAction.WAIT,

    # Not found - skip
    "not_found": ErrorAction.SKIP,

    # Unknown - default to retry
    "unknown": ErrorAction.RETRY,
}


def get_action_for_error_type(error_type: str) -> ErrorAction:
    """Get the recommended action for an error type.

    US-157-005: Maps error types to specific recovery actions.

    Args:
        error_type: The error type string.

    Returns:
        The ErrorAction to take for this error type.
    """
    return ERROR_TYPE_TO_ACTION.get(error_type, ErrorAction.RETRY)


class YouTubeAPIQuotaExceededError(YouTubeAPIError):
    """Raised when YouTube Data API quota is exhausted.

    US-148-012: Specific error type for quota exhaustion with actionable guidance.
    """

    category = 'youtube_api'
    severity = 'high'
    retryable = False
    error_type = YouTubeAPIError.TYPE_QUOTA_EXCEEDED

    def __init__(
        self,
        original_message: str,
        *,
        retry_after: Optional[float] = None,
        endpoint: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            original_message,
            error_type=self.TYPE_QUOTA_EXCEEDED,
            retry_after=retry_after,
            endpoint=endpoint,
            **kwargs
        )


class YouTubeAPIRateLimitedError(YouTubeAPIError):
    """Raised when API rate limit is exceeded (HTTP 429).

    US-148-012: Specific error type for rate limiting with retry-after support.
    """

    category = 'youtube_api'
    severity = 'medium'
    retryable = True
    error_type = YouTubeAPIError.TYPE_RATE_LIMITED

    def __init__(
        self,
        original_message: str,
        *,
        retry_after: Optional[float] = None,
        endpoint: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            original_message,
            error_type=self.TYPE_RATE_LIMITED,
            retry_after=retry_after,
            endpoint=endpoint,
            **kwargs
        )
        self.retry_after = retry_after


class YouTubeAPIInvalidKeyError(YouTubeAPIError):
    """Raised when API key is invalid or lacks required permissions.

    US-148-012: Specific error type for 403 errors related to authentication.
    """

    category = 'youtube_api'
    severity = 'high'
    retryable = False
    error_type = YouTubeAPIError.TYPE_INVALID_KEY

    def __init__(
        self,
        original_message: str,
        *,
        endpoint: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            original_message,
            error_type=self.TYPE_INVALID_KEY,
            endpoint=endpoint,
            **kwargs
        )


class YouTubeAPIPermissionDeniedError(YouTubeAPIError):
    """Raised when API access is denied due to permissions.

    US-148-012: Specific error type for permission-related access denied.
    """

    category = 'youtube_api'
    severity = 'high'
    retryable = False
    error_type = YouTubeAPIError.TYPE_PERMISSION_DENIED

    def __init__(
        self,
        original_message: str,
        *,
        endpoint: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            original_message,
            error_type=self.TYPE_PERMISSION_DENIED,
            endpoint=endpoint,
            **kwargs
        )


class YouTubeAPINetworkError(YouTubeAPIError):
    """Raised when network connectivity issues occur with YouTube API.

    US-148-012: Specific error type for network-related errors.
    """

    category = 'network'
    severity = 'medium'
    retryable = True
    error_type = YouTubeAPIError.TYPE_NETWORK_ERROR

    def __init__(
        self,
        original_message: str,
        *,
        endpoint: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            original_message,
            error_type=self.TYPE_NETWORK_ERROR,
            endpoint=endpoint,
            **kwargs
        )


class YouTubeAPITemporaryError(YouTubeAPIError):
    """Raised when YouTube API returns transient 5xx errors.

    US-149-007: These errors are retryable with backoff, but should eventually
    fallback if they persist. 5xx errors indicate server-side issues that
    are usually temporary.
    """

    category = 'youtube_api'
    severity = 'low'
    retryable = True
    error_type = 'temporary_error'

    def __init__(
        self,
        original_message: str,
        *,
        status_code: Optional[int] = None,
        endpoint: str = "",
        **kwargs
    ) -> None:
        self.status_code = status_code
        super().__init__(
            original_message,
            error_type='temporary_error',
            endpoint=endpoint,
            **kwargs
        )


class YouTubeAPIQuotaError(YouTubeAPIError):
    """Raised when YouTube API quota-related errors need immediate fallback.

    US-149-007: This is a more specific error type for quota errors that
    should trigger immediate fallback without retrying. It's similar to
    YouTubeAPIQuotaExceededError but indicates quota is exhausted across
    all keys and should fail fast.
    """

    category = 'youtube_api'
    severity = 'high'
    retryable = False
    error_type = 'quota_error'

    def __init__(
        self,
        original_message: str,
        *,
        endpoint: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            original_message,
            error_type='quota_error',
            endpoint=endpoint,
            **kwargs
        )


# =============================================================================
# US-153-007: Advanced error classification with recovery strategies
# =============================================================================


class YouTubeAPIDailyQuotaExceededError(YouTubeAPIError):
    """Raised when YouTube API daily quota is exhausted.

    US-153-007: Specific error type for daily quota limits. Quota resets
    at midnight PST. This is not retryable until the quota resets.
    """

    category = 'youtube_api'
    severity = 'high'
    retryable = False
    error_type = 'daily_quota_exceeded'

    def __init__(
        self,
        original_message: str,
        *,
        retry_after: Optional[float] = None,
        endpoint: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            original_message,
            error_type=self.error_type,
            retry_after=retry_after,
            endpoint=endpoint,
            **kwargs
        )

    def _get_guidance(self, error_type: str) -> str:
        return (
            "Daily API quota exhausted. Quota resets at midnight PST. "
            "Suggested next steps: (1) Wait until quota resets, "
            "(2) Request quota increase in Google Cloud Console, "
            "(3) Use yt-dlp fallback for remaining searches, "
            "(4) Consider using multiple API keys with rotation."
        )


class YouTubeAPIPerSecondLimitError(YouTubeAPIError):
    """Raised when YouTube API per-second request limit is exceeded.

    US-153-007: Specific error type for per-second rate limits. These
    are transient and retryable after a short delay.
    """

    category = 'youtube_api'
    severity = 'medium'
    retryable = True
    error_type = 'per_second_limit'

    def __init__(
        self,
        original_message: str,
        *,
        retry_after: Optional[float] = None,
        endpoint: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            original_message,
            error_type=self.error_type,
            retry_after=retry_after,
            endpoint=endpoint,
            **kwargs
        )

    def _get_guidance(self, error_type: str) -> str:
        return (
            "Per-second API limit exceeded. "
            "Suggested next steps: (1) Wait 1-2 seconds before retry, "
            "(2) Reduce request frequency, "
            "(3) Use request batching to reduce API calls, "
            "(4) Enable yt-dlp fallback in config.yaml."
        )


class YouTubeAPIDisabledProjectError(YouTubeAPIError):
    """Raised when the Google Cloud project is disabled.

    US-153-007: Specific error type for disabled projects. This is a
    configuration error that requires manual intervention.
    """

    category = 'youtube_api'
    severity = 'critical'
    retryable = False
    error_type = 'disabled_project'

    def __init__(
        self,
        original_message: str,
        *,
        endpoint: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            original_message,
            error_type=self.error_type,
            endpoint=endpoint,
            **kwargs
        )

    def _get_guidance(self, error_type: str) -> str:
        return (
            "Google Cloud project is disabled. "
            "Suggested next steps: (1) Check Google Cloud Console for project status, "
            "(2) Enable the project if you have access, "
            "(3) Create a new project if needed, "
            "(4) Verify billing is enabled for the project."
        )


class YouTubeAPIInvalidProjectError(YouTubeAPIError):
    """Raised when the Google Cloud project is invalid or not found.

    US-153-007: Specific error type for invalid project errors.
    """

    category = 'youtube_api'
    severity = 'critical'
    retryable = False
    error_type = 'invalid_project'

    def __init__(
        self,
        original_message: str,
        *,
        endpoint: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            original_message,
            error_type=self.error_type,
            endpoint=endpoint,
            **kwargs
        )

    def _get_guidance(self, error_type: str) -> str:
        return (
            "Google Cloud project is invalid or not found. "
            "Suggested next steps: (1) Verify project ID in config.yaml, "
            "(2) Check Google Cloud Console for correct project ID, "
            "(3) Ensure YouTube Data API v3 is enabled for the project, "
            "(4) Check project has valid credentials."
        )


class DNSTimeoutError(NetworkError):
    """DNS resolution timeout error.

    US-153-007: Specific error for DNS timeouts. This is a network
    error that may be transient.
    """

    category = 'network'
    severity = 'medium'
    retryable = True
    error_type = 'dns_timeout'

    def __init__(
        self,
        original_message: str,
        **kwargs
    ) -> None:
        super().__init__(original_message, **kwargs)


class DNSResolutionError(NetworkError):
    """DNS resolution failure error.

    US-153-007: Specific error for DNS resolution failures. This
    indicates DNS server issues.
    """

    category = 'network'
    severity = 'high'
    retryable = False
    error_type = 'dns_resolution'

    def __init__(
        self,
        original_message: str,
        **kwargs
    ) -> None:
        super().__init__(original_message, **kwargs)


class ConnectionTimeoutError(NetworkError):
    """TCP connection timeout error.

    US-153-007: Specific error for connection timeouts. This is a
    network error that may be transient.
    """

    category = 'network'
    severity = 'medium'
    retryable = True
    error_type = 'connection_timeout'

    def __init__(
        self,
        original_message: str,
        **kwargs
    ) -> None:
        super().__init__(original_message, **kwargs)


class TLSHandshakeError(NetworkError):
    """TLS handshake failure error.

    US-153-007: Specific error for TLS handshake failures. This may
    indicate certificate or protocol issues.
    """

    category = 'network'
    severity = 'medium'
    retryable = True
    error_type = 'tls_handshake'

    def __init__(
        self,
        original_message: str,
        **kwargs
    ) -> None:
        super().__init__(original_message, **kwargs)


# =============================================================================
# Retry Strategy Mapping (US-153-007)
# =============================================================================


class RetryStrategy(Enum):
    """Retry strategies mapped to error types.

    US-153-007: Defines retry behavior for different error types.
    """

    # No retry - fail immediately
    NO_RETRY = "no_retry"

    # Retry with exponential backoff
    EXPONENTIAL_BACKOFF = "exponential_backoff"

    # Retry after fixed delay
    FIXED_DELAY = "fixed_delay"

    # Retry after specified delay (from Retry-After header)
    RETRY_AFTER = "retry_after"

    # Retry with circuit breaker pause
    CIRCUIT_BREAKER = "circuit_breaker"

    # Switch to fallback (yt-dlp)
    FALLBACK = "fallback"

    # Switch API key and retry
    ROTATE_KEY = "rotate_key"

    # Retry with VPN/escalation
    ESCALATE = "escalate"


# Mapping of error types to retry strategies
ERROR_TYPE_RETRY_STRATEGY: dict[str, RetryStrategy] = {
    # Network errors
    'network': RetryStrategy.EXPONENTIAL_BACKOFF,
    'dns_timeout': RetryStrategy.FIXED_DELAY,
    'dns_resolution': RetryStrategy.NO_RETRY,
    'connection_timeout': RetryStrategy.EXPONENTIAL_BACKOFF,
    'tls_handshake': RetryStrategy.EXPONENTIAL_BACKOFF,
    'timeout': RetryStrategy.EXPONENTIAL_BACKOFF,

    # Rate limit errors
    'rate_limited': RetryStrategy.RETRY_AFTER,
    'per_second_limit': RetryStrategy.FIXED_DELAY,
    '429': RetryStrategy.RETRY_AFTER,

    # Quota errors
    'quota_exceeded': RetryStrategy.FALLBACK,
    'daily_quota_exceeded': RetryStrategy.FALLBACK,
    'quota_error': RetryStrategy.ROTATE_KEY,

    # Authentication errors
    'invalid_key': RetryStrategy.ROTATE_KEY,
    'invalid_project': RetryStrategy.NO_RETRY,
    'disabled_project': RetryStrategy.NO_RETRY,
    'permission_denied': RetryStrategy.FALLBACK,

    # Network errors from YouTube API
    'network_error': RetryStrategy.EXPONENTIAL_BACKOFF,

    # Bot detection errors
    'bot_detection': RetryStrategy.ESCALATE,

    # Temporary errors - retry with backoff
    'temporary_error': RetryStrategy.EXPONENTIAL_BACKOFF,

    # Geo-blocking - escalate to VPN
    'geo_blocked': RetryStrategy.ESCALATE,

    # Premium required - no retry without premium
    'premium_required': RetryStrategy.NO_RETRY,

    # Device limit - wait and retry
    'device_limit': RetryStrategy.FIXED_DELAY,

    # Unknown - default to exponential backoff
    'unknown': RetryStrategy.EXPONENTIAL_BACKOFF,
}

# Default retry parameters per strategy
RETRY_STRATEGY_CONFIG: dict[RetryStrategy, dict] = {
    RetryStrategy.NO_RETRY: {
        'max_attempts': 0,
        'base_delay': 0,
        'max_delay': 0,
    },
    RetryStrategy.EXPONENTIAL_BACKOFF: {
        'max_attempts': 5,
        'base_delay': 2.0,
        'max_delay': 120.0,
        'jitter_factor': 0.2,
    },
    RetryStrategy.FIXED_DELAY: {
        'max_attempts': 3,
        'base_delay': 5.0,
        'max_delay': 10.0,
    },
    RetryStrategy.RETRY_AFTER: {
        'max_attempts': 3,
        'base_delay': 0,  # Use Retry-After header
        'max_delay': 300.0,
    },
    RetryStrategy.CIRCUIT_BREAKER: {
        'max_attempts': 1,
        'base_delay': 60.0,
        'max_delay': 300.0,
    },
    RetryStrategy.FALLBACK: {
        'max_attempts': 1,
        'base_delay': 0,
        'max_delay': 0,
    },
    RetryStrategy.ROTATE_KEY: {
        'max_attempts': 3,
        'base_delay': 1.0,
        'max_delay': 5.0,
    },
    RetryStrategy.ESCALATE: {
        'max_attempts': 2,
        'base_delay': 10.0,
        'max_delay': 60.0,
    },
}


def get_retry_strategy(error_type: str) -> RetryStrategy:
    """Get the retry strategy for an error type.

    Args:
        error_type: The error type string.

    Returns:
        The RetryStrategy to use for this error type.
    """
    return ERROR_TYPE_RETRY_STRATEGY.get(error_type, RetryStrategy.EXPONENTIAL_BACKOFF)


def get_retry_config(error_type: str) -> dict:
    """Get retry configuration for an error type.

    Args:
        error_type: The error type string.

    Returns:
        Dict with retry configuration (max_attempts, base_delay, etc.).
    """
    strategy = get_retry_strategy(error_type)
    return RETRY_STRATEGY_CONFIG.get(strategy, RETRY_STRATEGY_CONFIG[RetryStrategy.EXPONENTIAL_BACKOFF])


# =============================================================================
# US-157-005: Error Code to Action Mapping
# =============================================================================


class ErrorAction(Enum):
    """Actions to take when encountering an error.

    US-157-005: High-level action mapping for error handling decisions.
    Maps error codes to specific actions: retry, backoff, fallback, or abort.
    """

    # Retry the request immediately or after a short delay
    RETRY = "retry"

    # Apply exponential backoff before retrying
    BACKOFF = "backoff"

    # Switch to fallback method (yt-dlp)
    FALLBACK = "fallback"

    # Abort the operation entirely (no retry)
    ABORT = "abort"


# Mapping from error codes to recommended actions
# US-157-005: Maps YouTube Data API error codes to high-level actions
ERROR_CODE_TO_ACTION: dict[int, ErrorAction] = {
    # 1xx - Authentication and permission errors
    100: ErrorAction.ABORT,  # API disabled - cannot proceed
    101: ErrorAction.ABORT,  # Not found - won't be found on retry
    102: ErrorAction.ABORT,  # Invalid parameter - request is malformed
    103: ErrorAction.ABORT,  # Permission denied - won't change without config change

    # 2xx - Request errors
    200: ErrorAction.ABORT,  # Missing required parameter - request is malformed
    400: ErrorAction.ABORT,  # Invalid filter - request is malformed

    # 4xx - Client errors
    401: ErrorAction.FALLBACK,  # Invalid API key - switch to yt-dlp
    402: ErrorAction.FALLBACK,  # Quota exceeded - switch to yt-dlp
    403: ErrorAction.BACKOFF,  # Rate limited - apply backoff before retry
    404: ErrorAction.ABORT,  # Resource not found - won't be found on retry
    411: ErrorAction.ABORT,  # Invalid request format
    416: ErrorAction.ABORT,  # Invalid content range

    # 5xx - Server errors
    500: ErrorAction.BACKOFF,  # Internal error - retry with backoff
    501: ErrorAction.ABORT,  # Not implemented - won't work
    503: ErrorAction.BACKOFF,  # Backend error - retry with backoff
}


def error_code_to_action(error_code: int) -> ErrorAction:
    """Get the recommended action for a YouTube API error code.

    US-157-005: Maps error codes to high-level actions for decision making.
    Provides a simplified interface for determining what to do when an
    error occurs, abstracted away from retry strategies.

    Args:
        error_code: The YouTube Data API error code (100-110 or HTTP codes).

    Returns:
        The ErrorAction to take: RETRY, BACKOFF, FALLBACK, or ABORT.
    """
    return ERROR_CODE_TO_ACTION.get(error_code, ErrorAction.BACKOFF)


def get_action_for_error_type(error_type: str) -> ErrorAction:
    """Get the recommended action for an error type string.

    US-157-005: Maps error type strings to high-level actions.
    This is useful when error type is known but not the specific error code.

    Args:
        error_type: The error type string (e.g., 'quota_exceeded', 'rate_limited').

    Returns:
        The ErrorAction to take: RETRY, BACKOFF, FALLBACK, or ABORT.
    """
    # Map error types to actions
    error_type_to_action: dict[str, ErrorAction] = {
        # Retryable errors - retry immediately
        'temporary_error': ErrorAction.BACKOFF,
        'timeout': ErrorAction.BACKOFF,
        'network_error': ErrorAction.BACKOFF,
        'connection_timeout': ErrorAction.BACKOFF,
        'dns_timeout': ErrorAction.BACKOFF,
        'tls_handshake': ErrorAction.BACKOFF,

        # Rate limited - apply backoff
        'rate_limited': ErrorAction.BACKOFF,
        'per_second_limit': ErrorAction.RETRY,
        '429': ErrorAction.BACKOFF,

        # Fallback errors - switch to yt-dlp
        'quota_exceeded': ErrorAction.FALLBACK,
        'daily_quota_exceeded': ErrorAction.FALLBACK,
        'quota_error': ErrorAction.FALLBACK,
        'invalid_key': ErrorAction.FALLBACK,
        'invalid_project': ErrorAction.ABORT,
        'disabled_project': ErrorAction.ABORT,

        # Abort errors - do not retry
        'permission_denied': ErrorAction.ABORT,
        'not_found': ErrorAction.ABORT,
        'invalid_parameter': ErrorAction.ABORT,
        'missing_parameter': ErrorAction.ABORT,
        'invalid_filter': ErrorAction.ABORT,
        'invalid_request': ErrorAction.ABORT,
        'not_implemented': ErrorAction.ABORT,
        'geo_blocked': ErrorAction.FALLBACK,
        'premium_required': ErrorAction.ABORT,
        'device_limit': ErrorAction.RETRY,

        # Bot detection - escalate or fallback
        'bot_detection': ErrorAction.FALLBACK,

        # Unknown - default to backoff
        'unknown': ErrorAction.BACKOFF,
    }
    return error_type_to_action.get(error_type, ErrorAction.BACKOFF)


class GeoBlockedError(ClassifiedDownloadError):
    """Geographic blocking: content not available in user's region.

    US-113-006: New error class for geo-blocking errors from YouTube.
    Requires VPN rotation (Tier 4) to bypass.
    """

    category = 'geo_blocked'
    severity = 'high'
    retryable = True


class DeviceLimitError(ClassifiedDownloadError):
    """Device limit exceeded: too many devices streaming simultaneously.

    US-113-006: New error class for device limit errors from YouTube.
    Retryable after waiting for existing streams to finish.
    """

    category = 'device_limit'
    severity = 'medium'
    retryable = True


class LoginRequiredError(ClassifiedDownloadError):
    """Login required: content requires authentication to access.

    US-113-006: New error class for login-required errors.
    Retryable with cookies/authentication.
    """

    category = 'login_required'
    severity = 'low'
    retryable = True


class PremiumRequiredError(ClassifiedDownloadError):
    """Premium required: content requires YouTube Premium subscription.

    US-143-010: New error class for premium/members-only errors.
    These errors are not retryable without Premium subscription.
    """

    category = 'premium_required'
    severity = 'high'
    retryable = False


class UnknownError(ClassifiedDownloadError):
    """Unclassified error: error doesn't match any known pattern.

    US-120-002: New error class for Unknown errors that don't match
    any known category. Captures original message and optionally
    stack trace for later analysis.
    """

    category = 'unknown'
    severity = 'medium'
    retryable = True

    def __init__(
        self,
        original_message: str,
        *,
        stack_trace: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize Unknown error.

        Args:
            original_message: The error message string.
            stack_trace: Optional stack trace for diagnostics.
        """
        super().__init__(original_message, **kwargs)
        self.stack_trace = stack_trace

    def __str__(self) -> str:
        return self.original_message


# =============================================================================
# US-146-009: YouTube Data API error handling
# =============================================================================


class APIError(Exception):
    """Base exception for YouTube Data API errors.

    US-146-009: Unified exception hierarchy for YouTube API errors.
    """

    def __init__(
        self,
        message: str,
        *,
        endpoint: str = "",
        params: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ) -> None:
        self.message = message
        self.endpoint = endpoint
        self.params = params or {}
        self.status_code = status_code
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


class QuotaExceededError(APIError):
    """Raised when YouTube Data API quota is exhausted.

    US-146-009: Specific error for quota exhaustion.
    """

    def __init__(
        self,
        message: str,
        *,
        endpoint: str = "",
        params: Optional[Dict[str, Any]] = None,
        status_code: int = 403,
    ) -> None:
        super().__init__(
            message,
            endpoint=endpoint,
            params=params,
            status_code=status_code,
        )


class InvalidCredentialsError(APIError):
    """Raised when API key is invalid or lacks required permissions.

    US-146-009: Specific error for 403 errors related to authentication.
    """

    def __init__(
        self,
        message: str,
        *,
        endpoint: str = "",
        params: Optional[Dict[str, Any]] = None,
        status_code: int = 403,
    ) -> None:
        super().__init__(
            message,
            endpoint=endpoint,
            params=params,
            status_code=status_code,
        )


class YouTubeAPIRateLimitError(APIError):
    """Raised when YouTube API rate limit is exceeded (HTTP 429).

    US-146-009: Specific error for rate limiting with retry-after support.
    """

    retry_after: Optional[float] = None

    def __init__(
        self,
        message: str,
        *,
        endpoint: str = "",
        params: Optional[Dict[str, Any]] = None,
        status_code: int = 429,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(
            message,
            endpoint=endpoint,
            params=params,
            status_code=status_code,
        )
        self.retry_after = retry_after


# =============================================================================
# US-93-006: Enhanced error handling with error codes and suggestions
# =============================================================================


class DownloadErrorCode(Enum):
    """Structured error codes for download failures.

    Each code maps to a user-friendly message and recovery suggestion.
    """

    # Network errors (0xx)
    ERR_NETWORK_DNS = "E001"  # DNS resolution failure
    ERR_NETWORK_CONNECTION = "E002"  # Connection refused/reset
    ERR_NETWORK_TLS = "E003"  # TLS handshake failure
    ERR_NETWORK_TIMEOUT = "E004"  # Network timeout
    ERR_NETWORK_UNREACHABLE = "E005"  # Host unreachable

    # Bot detection errors (1xx)
    ERR_BOT_403 = "E101"  # HTTP 403 Forbidden
    ERR_BOT_CAPTCHA = "E102"  # CAPTCHA required
    ERR_BOT_SIGNIN = "E103"  # Sign-in required
    ERR_BOT_BLOCKED = "E104"  # Bot detection/blocked

    # Rate limit errors (2xx)
    ERR_RATE_LIMIT = "E201"  # HTTP 429 Too Many Requests
    ERR_QUOTA_EXCEEDED = "E202"  # API quota exceeded

    # Format errors (3xx)
    ERR_FORMAT_UNAVAILABLE = "E301"  # Requested format not available
    ERR_FORMAT_MISSING = "E302"  # No formats found
    ERR_VIDEO_UNAVAILABLE = "E303"  # Video unavailable/removed
    ERR_VIDEO_PRIVATE = "E304"  # Video is private
    ERR_VIDEO_AGE_RESTRICTED = "E305"  # Age-restricted content

    # Authentication errors (4xx)
    ERR_AUTH_AGE_GATE = "E401"  # Age-gate requires authentication
    ERR_AUTH_LOGIN_REQUIRED = "E402"  # Login required for content
    ERR_AUTH_PREMIUM = "E403"  # Premium membership required

    # Timeout errors (5xx)
    ERR_TIMEOUT_STALL = "E501"  # Download stalled
    ERR_TIMEOUT_SOCKET = "E502"  # Socket timeout
    ERR_TIMEOUT_DEADLINE = "E503"  # Deadline exceeded

    # US-113-006: New error codes for enhanced error classification
    # Geo-blocking errors (6xx)
    ERR_GEO_BLOCKED = "E601"  # Content not available in region
    ERR_GEO_RESTRICTED = "E602"  # Video geo-restricted

    # Device limit errors (7xx)
    ERR_DEVICE_LIMIT_EXCEEDED = "E701"  # Too many devices streaming

    # Login required errors (8xx)
    ERR_LOGIN_REQUIRED = "E801"  # Login required for content

    # Premium required errors (9xx)
    ERR_PREMIUM_REQUIRED = "E901"  # Premium subscription required

    # Unknown errors (xxx) - keep as fallback
    ERR_UNKNOWN = "E999"  # Unclassified error
    ERR_PARSE_ERROR = "E902"  # Failed to parse yt-dlp output


# Error code to error class mapping
_ERROR_CODE_TO_CLASS = {
    # Network errors
    "DNS": DownloadErrorCode.ERR_NETWORK_DNS,
    "connection": DownloadErrorCode.ERR_NETWORK_CONNECTION,
    "tls": DownloadErrorCode.ERR_NETWORK_TLS,
    "timed out": DownloadErrorCode.ERR_NETWORK_TIMEOUT,
    "unreachable": DownloadErrorCode.ERR_NETWORK_UNREACHABLE,
    # Bot detection
    "403": DownloadErrorCode.ERR_BOT_403,
    "forbidden": DownloadErrorCode.ERR_BOT_403,
    "captcha": DownloadErrorCode.ERR_BOT_CAPTCHA,
    "sign in": DownloadErrorCode.ERR_BOT_SIGNIN,
    "bot": DownloadErrorCode.ERR_BOT_BLOCKED,
    # US-113-006: Geo-blocking must come BEFORE generic "blocked"
    "geo block": DownloadErrorCode.ERR_GEO_BLOCKED,
    "geo blocked": DownloadErrorCode.ERR_GEO_BLOCKED,
    "geo-restricted": DownloadErrorCode.ERR_GEO_RESTRICTED,
    # Generic "blocked" must come AFTER geo-blocking patterns
    "blocked": DownloadErrorCode.ERR_BOT_BLOCKED,
    # Rate limit
    "429": DownloadErrorCode.ERR_RATE_LIMIT,
    "rate limit": DownloadErrorCode.ERR_RATE_LIMIT,
    "quota": DownloadErrorCode.ERR_QUOTA_EXCEEDED,
    # Format errors
    "requested format": DownloadErrorCode.ERR_FORMAT_UNAVAILABLE,
    "no video formats": DownloadErrorCode.ERR_FORMAT_MISSING,
    "unavailable": DownloadErrorCode.ERR_VIDEO_UNAVAILABLE,
    "removed": DownloadErrorCode.ERR_VIDEO_UNAVAILABLE,
    "private": DownloadErrorCode.ERR_VIDEO_PRIVATE,
    "age-restricted": DownloadErrorCode.ERR_VIDEO_AGE_RESTRICTED,
    # Authentication
    "age gate": DownloadErrorCode.ERR_AUTH_AGE_GATE,
    "login required": DownloadErrorCode.ERR_AUTH_LOGIN_REQUIRED,
    "members only": DownloadErrorCode.ERR_AUTH_PREMIUM,
    # Timeout
    "stalled": DownloadErrorCode.ERR_TIMEOUT_STALL,
    "socket timeout": DownloadErrorCode.ERR_TIMEOUT_SOCKET,
    "deadline": DownloadErrorCode.ERR_TIMEOUT_DEADLINE,
    # US-113-006: Geo-blocking errors
    "geo block": DownloadErrorCode.ERR_GEO_BLOCKED,
    "geo blocked": DownloadErrorCode.ERR_GEO_BLOCKED,
    "geo-restricted": DownloadErrorCode.ERR_GEO_RESTRICTED,
    "not available in your country": DownloadErrorCode.ERR_GEO_BLOCKED,
    "not available in your region": DownloadErrorCode.ERR_GEO_BLOCKED,
    # Device limit errors
    "device limit": DownloadErrorCode.ERR_DEVICE_LIMIT_EXCEEDED,
    "too many devices": DownloadErrorCode.ERR_DEVICE_LIMIT_EXCEEDED,
    "playback on other": DownloadErrorCode.ERR_DEVICE_LIMIT_EXCEEDED,
    # Login required errors
    "login required": DownloadErrorCode.ERR_LOGIN_REQUIRED,
    "sign in to watch": DownloadErrorCode.ERR_LOGIN_REQUIRED,
    # US-143-010: Premium required errors
    "premium required": DownloadErrorCode.ERR_PREMIUM_REQUIRED,
    "members only": DownloadErrorCode.ERR_PREMIUM_REQUIRED,
    "youtube premium": DownloadErrorCode.ERR_PREMIUM_REQUIRED,
    "premium only": DownloadErrorCode.ERR_PREMIUM_REQUIRED,
}


# User-friendly messages for each error code
_ERROR_MESSAGES = {
    DownloadErrorCode.ERR_NETWORK_DNS: "Unable to resolve video host. Check your internet connection and DNS settings.",
    DownloadErrorCode.ERR_NETWORK_CONNECTION: "Could not connect to video host. The server may be down or blocking connections.",
    DownloadErrorCode.ERR_NETWORK_TLS: "Secure connection failed. Some sites require specific TLS versions.",
    DownloadErrorCode.ERR_NETWORK_TIMEOUT: "Connection timed out. The server took too long to respond.",
    DownloadErrorCode.ERR_NETWORK_UNREACHABLE: "Video host is unreachable. Check your internet connection.",
    DownloadErrorCode.ERR_BOT_403: "Access denied (403). The site is blocking automated requests.",
    DownloadErrorCode.ERR_BOT_CAPTCHA: "CAPTCHA challenge required. The site suspects automated access.",
    DownloadErrorCode.ERR_BOT_SIGNIN: "Sign-in required. Some content requires authentication.",
    DownloadErrorCode.ERR_BOT_BLOCKED: "Access blocked. The site detected suspicious activity.",
    DownloadErrorCode.ERR_RATE_LIMIT: "Rate limited (429). Too many requests - wait before retrying.",
    DownloadErrorCode.ERR_QUOTA_EXCEEDED: "API quota exceeded. YouTube API limit reached for today.",
    DownloadErrorCode.ERR_FORMAT_UNAVAILABLE: "Requested video format is not available. Try a different quality option.",
    DownloadErrorCode.ERR_FORMAT_MISSING: "No downloadable formats found. The video may be unavailable.",
    DownloadErrorCode.ERR_VIDEO_UNAVAILABLE: "Video has been removed or is unavailable.",
    DownloadErrorCode.ERR_VIDEO_PRIVATE: "Video is private and cannot be downloaded.",
    DownloadErrorCode.ERR_VIDEO_AGE_RESTRICTED: "Video is age-restricted and requires authentication.",
    DownloadErrorCode.ERR_AUTH_AGE_GATE: "Age-restricted content requires authentication.",
    DownloadErrorCode.ERR_AUTH_LOGIN_REQUIRED: "This content requires a login to access.",
    DownloadErrorCode.ERR_AUTH_PREMIUM: "This content requires a premium subscription.",
    DownloadErrorCode.ERR_TIMEOUT_STALL: "Download stalled - no progress for extended period.",
    DownloadErrorCode.ERR_TIMEOUT_SOCKET: "Socket timeout - server stopped responding.",
    DownloadErrorCode.ERR_TIMEOUT_DEADLINE: "Operation exceeded maximum time limit.",
    # US-113-006: New error messages for enhanced classification
    DownloadErrorCode.ERR_GEO_BLOCKED: "This content is not available in your country or region.",
    DownloadErrorCode.ERR_GEO_RESTRICTED: "This video is geo-restricted and not available in your location.",
    DownloadErrorCode.ERR_DEVICE_LIMIT_EXCEEDED: "Too many devices are playing this content. Please wait and try again.",
    DownloadErrorCode.ERR_LOGIN_REQUIRED: "This content requires you to be logged in to watch.",
    # US-143-010: Premium required error message
    DownloadErrorCode.ERR_PREMIUM_REQUIRED: "This content requires a YouTube Premium subscription to access.",
    DownloadErrorCode.ERR_UNKNOWN: "An unexpected error occurred during download.",
    DownloadErrorCode.ERR_PARSE_ERROR: "Failed to process video information.",
}


# Recovery suggestions for each error code
_ERROR_SUGGESTIONS = {
    DownloadErrorCode.ERR_NETWORK_DNS: [
        "Check your internet connection",
        "Try using a different DNS server (e.g., Google DNS 8.8.8.8)",
    ],
    DownloadErrorCode.ERR_NETWORK_CONNECTION: [
        "Check your internet connection",
        "The video host may be experiencing issues - try again later",
    ],
    DownloadErrorCode.ERR_NETWORK_TLS: [
        "Update your SSL/TLS libraries",
        "Try using curl_cffi for TLS fingerprinting",
    ],
    DownloadErrorCode.ERR_NETWORK_TIMEOUT: [
        "Increase download timeout in settings",
        "Check your internet connection speed",
    ],
    DownloadErrorCode.ERR_NETWORK_UNREACHABLE: [
        "Check your internet connection",
        "The video host may be blocked in your region",
    ],
    DownloadErrorCode.ERR_BOT_403: [
        "Try using cookies for authentication",
        "Use different impersonation target (Tier 2 escalation)",
        "Wait an hour before retrying",
    ],
    DownloadErrorCode.ERR_BOT_CAPTCHA: [
        "Manual verification required - open video in browser",
        "Try using cookies from a logged-in session",
    ],
    DownloadErrorCode.ERR_BOT_SIGNIN: [
        "Provide YouTube cookies for authentication",
        "Use browser cookies from a logged-in session",
    ],
    DownloadErrorCode.ERR_BOT_BLOCKED: [
        "Try different impersonation target",
        "Use cookies from a regular browser session",
        "Consider using VPN (Tier 4 escalation)",
    ],
    DownloadErrorCode.ERR_RATE_LIMIT: [
        "Wait before retrying (rate limit window)",
        "Reduce request frequency",
        "Try again in a few minutes",
    ],
    DownloadErrorCode.ERR_QUOTA_EXCEEDED: [
        "YouTube API quota reset typically at midnight PST",
        "Use alternative video sources if possible",
    ],
    DownloadErrorCode.ERR_FORMAT_UNAVAILABLE: [
        "Try a different video quality (720p, 1080p, etc.)",
        "Audio-only may be available as fallback",
    ],
    DownloadErrorCode.ERR_FORMAT_MISSING: [
        "Video may be unavailable in your region",
        "Try using a VPN to access different regions",
    ],
    DownloadErrorCode.ERR_VIDEO_UNAVAILABLE: [
        "Video has been removed by the uploader",
        "Search for alternative videos with similar content",
    ],
    DownloadErrorCode.ERR_VIDEO_PRIVATE: [
        "Cannot download private videos",
        "Request the video creator to make it public",
    ],
    DownloadErrorCode.ERR_VIDEO_AGE_RESTRICTED: [
        "Provide age-gate bypass cookies",
        "Use cookies from an age-verified session",
    ],
    DownloadErrorCode.ERR_AUTH_AGE_GATE: [
        "Provide authentication cookies",
        "Use cookies from an age-verified YouTube session",
    ],
    DownloadErrorCode.ERR_AUTH_LOGIN_REQUIRED: [
        "Provide YouTube cookies for authentication",
        "Log in through browser first, then export cookies",
    ],
    DownloadErrorCode.ERR_AUTH_PREMIUM: [
        "This requires a YouTube Premium subscription",
        "Download may not be possible without Premium",
    ],
    DownloadErrorCode.ERR_TIMEOUT_STALL: [
        "Increase stall timeout in settings",
        "Check internet connection stability",
        "Try downloading at different time (off-peak)",
    ],
    DownloadErrorCode.ERR_TIMEOUT_SOCKET: [
        "Check internet connection",
        "Try increasing socket timeout",
    ],
    DownloadErrorCode.ERR_TIMEOUT_DEADLINE: [
        "Increase operation timeout in settings",
        "Try with smaller video segments",
    ],
    # US-113-006: New error suggestions for enhanced classification
    DownloadErrorCode.ERR_GEO_BLOCKED: [
        "Use VPN to connect from an allowed region (Tier 4 escalation)",
        "Try connecting to a VPN server in US, UK, or other allowed country",
        "Some content may only be available in specific regions",
    ],
    DownloadErrorCode.ERR_GEO_RESTRICTED: [
        "Use VPN to connect from an allowed region (Tier 4 escalation)",
        "Try connecting to a VPN server in a different country",
        "Content may be restricted to certain geographic locations",
    ],
    DownloadErrorCode.ERR_DEVICE_LIMIT_EXCEEDED: [
        "Wait for other devices to stop playing this content",
        "Close other tabs or apps playing YouTube videos",
        "YouTube Premium members can stream on more devices",
    ],
    DownloadErrorCode.ERR_LOGIN_REQUIRED: [
        "Provide YouTube cookies for authentication",
        "Log in through browser first, then export cookies",
        "Some content requires a YouTube account to access",
    ],
    # US-143-010: Premium required suggestions
    DownloadErrorCode.ERR_PREMIUM_REQUIRED: [
        "This content requires a YouTube Premium subscription",
        "Download may not be possible without Premium",
        "Try searching for similar free content as alternative",
    ],
    DownloadErrorCode.ERR_UNKNOWN: [
        "Check logs for detailed error information",
        "Try running with debug logging enabled",
        "Report issue if persists",
    ],
    DownloadErrorCode.ERR_PARSE_ERROR: [
        "Update yt-dlp to latest version",
        "Try with different extractor arguments",
    ],
}


def get_error_code(error_message: str) -> DownloadErrorCode:
    """Map an error message to a structured error code.

    Args:
        error_message: The raw error message from yt-dlp.

    Returns:
        The corresponding DownloadErrorCode enum value.
    """
    error_lower = error_message.lower()

    # Check each keyword in order of specificity
    for keyword, code in _ERROR_CODE_TO_CLASS.items():
        if keyword.lower() in error_lower:
            return code

    return DownloadErrorCode.ERR_UNKNOWN


def get_error_message(error_code: DownloadErrorCode) -> str:
    """Get user-friendly message for an error code.

    Args:
        error_code: The error code enum value.

    Returns:
        Human-readable error message.
    """
    return _ERROR_MESSAGES.get(error_code, "An unknown error occurred.")


def get_error_suggestions(error_code: DownloadErrorCode) -> list[str]:
    """Get recovery suggestions for an error code.

    Args:
        error_code: The error code enum value.

    Returns:
        List of actionable suggestions for recovery.
    """
    return _ERROR_SUGGESTIONS.get(error_code, ["Try again later.", "Check logs for details."])


class StructuredDownloadError:
    """Structured error response with error code, message, and suggestions.

    US-93-006: Provides user-friendly error messages and actionable recovery
    suggestions for download failures. This class wraps the existing
    classified errors and adds structured error codes and logging context.
    """

    def __init__(
        self,
        error: ClassifiedDownloadError,
        video_id: Optional[str] = None,
        keyword: Optional[str] = None,
    ):
        """Initialize structured download error.

        Args:
            error: The classified download error instance.
            video_id: Optional video ID for context.
            keyword: Optional search keyword for context.
        """
        self._error = error
        self.video_id = video_id
        self.keyword = keyword
        self.error_code = get_error_code(error.original_message)
        self.user_message = get_error_message(self.error_code)
        self.suggestions = get_error_suggestions(self.error_code)

    @property
    def category(self) -> str:
        """Get the error category."""
        return self._error.category

    @property
    def severity(self) -> str:
        """Get the error severity."""
        return self._error.severity

    @property
    def retryable(self) -> bool:
        """Whether the error is retryable."""
        return self._error.retryable

    @property
    def original_message(self) -> str:
        """Get the original error message."""
        return self._error.original_message

    def __str__(self) -> str:
        """Return user-friendly error string."""
        return f"[{self.error_code.value}] {self.user_message}"

    def __repr__(self) -> str:
        """Return detailed error representation with context."""
        parts = [
            f"StructuredDownloadError(",
            f"code={self.error_code.value}",
            f"category={self.category}",
            f"severity={self.severity}",
            f"retryable={self.retryable}",
        ]
        if self.video_id:
            parts.append(f"video_id={self.video_id}")
        if self.keyword:
            parts.append(f"keyword={self.keyword}")
        parts.append(f"message={self.user_message!r}")
        parts.append(")")
        return " ".join(parts)

    def get_log_message(self) -> str:
        """Get detailed message for logging with context.

        Returns:
            Formatted string suitable for logging with video context.
        """
        parts = [
            f"Download error [{self.error_code.value}]",
            f"category={self.category}",
            f"severity={self.severity}",
        ]
        if self.video_id:
            parts.append(f"video_id={self.video_id}")
        if self.keyword:
            parts.append(f"keyword={self.keyword}")
        parts.append(f"message={self.original_message}")
        return " ".join(parts)
