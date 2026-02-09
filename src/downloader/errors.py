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
"""

from __future__ import annotations

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
