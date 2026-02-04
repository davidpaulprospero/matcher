"""
Caption error handling and pattern detection.

Provides error categorization and pattern detection for batch caption fetching.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

from .enums import CaptionErrorCategory
from .exceptions import CaptionFormatUnavailableError, CaptionUnavailableError
from .models import ErrorPatternResult

logger = logging.getLogger(__name__)


def categorize_caption_error(error: Exception, reason: str = "") -> CaptionErrorCategory:
    """Categorize an exception into a CaptionErrorCategory (US-003 Sprint 7).

    Maps exception types and error message patterns to categories for
    determining appropriate retry strategy.

    Args:
        error: The exception that occurred during caption fetch.
        reason: Optional additional context string (e.g., from CaptionFetchError.reason).

    Returns:
        CaptionErrorCategory indicating the type of failure.

    Examples:
        >>> categorize_caption_error(CaptionUnavailableError("abc", "no subs"))
        CaptionErrorCategory.UNAVAILABLE

        >>> categorize_caption_error(TimeoutError())
        CaptionErrorCategory.TIMEOUT

        >>> e = CaptionFetchError("abc", "HTTP 429 Too Many Requests")
        >>> categorize_caption_error(e, e.reason)
        CaptionErrorCategory.RATE_LIMIT
    """
    # First check for specific exception types (most specific first)
    if isinstance(error, CaptionFormatUnavailableError):
        return CaptionErrorCategory.FORMAT_UNAVAILABLE

    if isinstance(error, CaptionUnavailableError):
        return CaptionErrorCategory.UNAVAILABLE

    if isinstance(error, (TimeoutError, )):
        return CaptionErrorCategory.TIMEOUT

    if isinstance(error, (json.JSONDecodeError, UnicodeDecodeError)):
        return CaptionErrorCategory.PARSE

    # Check reason string for patterns
    reason_lower = reason.lower() if reason else ""
    error_str = str(error).lower()
    combined = f"{reason_lower} {error_str}"

    # Rate limit patterns
    rate_limit_patterns = [
        "429", "too many requests", "rate limit", "quota exceeded",
        "rate-limit", "throttle", "slow down"
    ]
    if any(p in combined for p in rate_limit_patterns):
        return CaptionErrorCategory.RATE_LIMIT

    # Timeout patterns
    timeout_patterns = [
        "timeout", "timed out", "deadline exceeded", "connection timed out"
    ]
    if any(p in combined for p in timeout_patterns):
        return CaptionErrorCategory.TIMEOUT

    # Parse error patterns
    parse_patterns = [
        "parse", "decode", "invalid json", "malformed", "syntax error",
        "unexpected token", "invalid format", "corrupt"
    ]
    if any(p in combined for p in parse_patterns):
        return CaptionErrorCategory.PARSE

    # Format-specific unavailable (US-59-004) — check before general unavailable
    format_unavailable_patterns = [
        "requested format is not available",
    ]
    if any(p in combined for p in format_unavailable_patterns):
        return CaptionErrorCategory.FORMAT_UNAVAILABLE

    # Unavailable patterns
    unavailable_patterns = [
        "not available", "unavailable", "no subtitles", "no captions",
        "subtitles disabled", "captions disabled", "not found"
    ]
    if any(p in combined for p in unavailable_patterns):
        return CaptionErrorCategory.UNAVAILABLE

    # Network patterns (catch-all for connectivity issues)
    network_patterns = [
        "network", "connection", "dns", "resolve", "unreachable",
        "refused", "reset", "broken pipe", "http error", "ssl",
        "certificate", "socket", "eof"
    ]
    if any(p in combined for p in network_patterns):
        return CaptionErrorCategory.NETWORK

    # Default to NETWORK for unknown errors (most likely to benefit from retry)
    return CaptionErrorCategory.NETWORK


class ErrorPatternDetector:
    """Detects repeated error patterns during batch caption fetching (US-007 Sprint 7).

    Tracks error signatures as videos are processed and triggers pattern detection
    when the same error affects a configurable threshold of videos. This enables
    early detection of systemic issues like:
    - Geoblocking (403 Forbidden from many videos)
    - Rate limiting (429 Too Many Requests)
    - Network issues (connection timeouts across batch)
    - API restrictions (specific error messages)

    The detector is thread-safe for use with concurrent caption fetching.

    Args:
        threshold: Ratio of videos that must fail with same error to trigger.
            Default 0.3 = 30% of sample_size videos must fail with same error.
        sample_size: Number of videos to check before evaluating patterns.
            Default 10 = check first 10 videos for patterns.
    """

    # Common error patterns and their likely causes
    ERROR_CAUSES = {
        "403": "possible geoblocking or access restriction",
        "forbidden": "possible geoblocking or access restriction",
        "429": "rate limiting - consider reducing parallel fetches",
        "too many requests": "rate limiting - consider reducing parallel fetches",
        "rate limit": "rate limiting - consider reducing parallel fetches",
        "timeout": "network issues - check connection or increase timeout",
        "connection": "network connectivity issues",
        "dns": "DNS resolution failure - check network",
        "ssl": "SSL/TLS certificate issue",
        "unavailable": "captions disabled or unavailable for region",
        "not found": "videos may be deleted or private",
    }

    def __init__(self, threshold: float = 0.3, sample_size: int = 10):
        """Initialize the detector.

        Args:
            threshold: Ratio of videos that must fail with same error to trigger.
            sample_size: Number of videos to check before evaluating patterns.
        """
        self.threshold = threshold
        self.sample_size = sample_size
        self._errors: Dict[str, List[str]] = {}  # signature -> [video_ids]
        self._successes: List[str] = []
        self._total_processed = 0
        self._pattern_checked = False
        self._pattern_result: Optional[ErrorPatternResult] = None
        self._lock = threading.Lock()

    def _extract_signature(self, error_reason: str) -> str:
        """Extract a canonical error signature from an error reason.

        Normalizes error messages to group similar errors together.
        For example, "HTTP Error 403: Forbidden" and "403 access denied"
        both become "403 Forbidden".
        """
        reason_lower = error_reason.lower()

        # Check for HTTP status codes
        http_patterns = [
            (r"403|forbidden", "403 Forbidden"),
            (r"429|too many requests", "429 Too Many Requests"),
            (r"404|not found", "404 Not Found"),
            (r"500|internal server error", "500 Internal Server Error"),
            (r"502|bad gateway", "502 Bad Gateway"),
            (r"503|service unavailable", "503 Service Unavailable"),
            (r"timeout|timed out", "Timeout"),
            (r"connection refused|refused", "Connection Refused"),
            (r"connection reset|reset", "Connection Reset"),
            (r"dns|resolve", "DNS Error"),
            (r"ssl|certificate", "SSL Error"),
            (r"rate limit|throttle", "Rate Limited"),
            (r"unavailable|no subtitles|no captions", "Captions Unavailable"),
            (r"private|not available", "Video Private/Unavailable"),
        ]

        for pattern, signature in http_patterns:
            if re.search(pattern, reason_lower):
                return signature

        # Truncate long messages and return as-is
        if len(error_reason) > 50:
            return error_reason[:50] + "..."
        return error_reason

    def _infer_cause(self, signature: str) -> str:
        """Infer likely cause from error signature."""
        sig_lower = signature.lower()
        for pattern, cause in self.ERROR_CAUSES.items():
            if pattern in sig_lower:
                return cause
        return "unknown cause - check error details"

    def record_error(self, video_id: str, error_reason: str) -> None:
        """Record a failed video with its error reason.

        Thread-safe method to track errors during batch processing.
        """
        signature = self._extract_signature(error_reason)
        with self._lock:
            if signature not in self._errors:
                self._errors[signature] = []
            self._errors[signature].append(video_id)
            self._total_processed += 1

    def record_success(self, video_id: str) -> None:
        """Record a successfully processed video.

        Thread-safe method to track successes during batch processing.
        """
        with self._lock:
            self._successes.append(video_id)
            self._total_processed += 1

    def should_check_pattern(self) -> bool:
        """Check if enough videos have been processed to evaluate patterns."""
        with self._lock:
            return (
                self._total_processed >= self.sample_size
                and not self._pattern_checked
            )

    def check_pattern(self) -> ErrorPatternResult:
        """Evaluate error patterns and return detection result.

        Checks if any error signature affects more than threshold ratio
        of processed videos. Should be called after sample_size videos
        have been processed (use should_check_pattern() to verify).
        """
        with self._lock:
            # Return cached result if already checked
            if self._pattern_checked and self._pattern_result:
                return self._pattern_result

            self._pattern_checked = True

            # Find the most common error signature
            if not self._errors:
                self._pattern_result = ErrorPatternResult(
                    detected=False,
                    sample_size=self._total_processed
                )
                return self._pattern_result

            # Find signature with most affected videos
            max_sig = max(self._errors.keys(), key=lambda s: len(self._errors[s]))
            affected = self._errors[max_sig]
            ratio = len(affected) / self._total_processed if self._total_processed > 0 else 0

            if ratio >= self.threshold:
                self._pattern_result = ErrorPatternResult(
                    detected=True,
                    error_signature=max_sig,
                    affected_video_ids=list(affected),
                    sample_size=self._total_processed,
                    ratio=ratio,
                    likely_cause=self._infer_cause(max_sig)
                )
            else:
                self._pattern_result = ErrorPatternResult(
                    detected=False,
                    error_signature=max_sig,
                    affected_video_ids=list(affected),
                    sample_size=self._total_processed,
                    ratio=ratio,
                    likely_cause=""
                )

            return self._pattern_result

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for debugging."""
        with self._lock:
            return {
                "total_processed": self._total_processed,
                "success_count": len(self._successes),
                "error_counts": {sig: len(vids) for sig, vids in self._errors.items()},
                "pattern_checked": self._pattern_checked,
            }
