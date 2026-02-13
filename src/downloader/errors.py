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
from typing import Optional

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
    ) -> None:
        self.original_message = original_message
        if category is not None:
            self.category = category
        if severity is not None:
            self.severity = severity
        if retryable is not None:
            self.retryable = retryable
        super().__init__(original_message)

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

    Retryable after backoff delay.
    """

    category = 'bot_detection'  # Matches existing classification behavior
    severity = 'medium'
    retryable = True


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

    # Unknown errors (9xx)
    ERR_UNKNOWN = "E901"  # Unclassified error
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
