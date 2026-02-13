"""
Caption error handling and pattern detection.

Provides error categorization and pattern detection for batch caption fetching.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..common.error_patterns import (
    FORMAT_UNAVAILABLE_PATTERNS,
    NETWORK_PATTERNS,
    PARSE_PATTERNS,
    RATE_LIMIT_PATTERNS,
    TIMEOUT_PATTERNS,
    UNAVAILABLE_PATTERNS,
)
from .enums import CaptionErrorCategory
from .exceptions import CaptionFormatUnavailableError, CaptionUnavailableError
from .models import ErrorPatternResult

logger = logging.getLogger(__name__)


# =============================================================================
# NETWORK ERROR SUB-CLASSIFICATION (US-100-005)
# =============================================================================

class NetworkErrorSubtype(Enum):
    """Sub-types of NETWORK errors for more granular retry decisions."""
    DNS = "dns"                    # DNS resolution failures
    CONNECTION = "connection"       # Connection refused/reset
    TIMEOUT = "timeout"            # Request timeouts
    SSL = "ssl"                    # SSL certificate issues
    UNKNOWN = "unknown"            # Unclassified network errors


# Specific patterns for NETWORK sub-classification
DNS_PATTERNS = ["dns", "name or service not known", "could not resolve", "resolve host"]
CONNECTION_PATTERNS = ["connection refused", "connection reset", "connection closed", "broken pipe", "connect:"]
SSL_PATTERNS = ["ssl", "certificate", "tls", "sslcontext", "ssl verification"]


def classify_network_error(error: Exception, reason: str = "") -> NetworkErrorSubtype:
    """Classify NETWORK errors into sub-types for granular retry decisions (US-100-005).

    Args:
        error: The exception that occurred.
        reason: Optional additional context string.

    Returns:
        NetworkErrorSubtype indicating the specific network issue.
    """
    combined = f"{reason} {error}".lower()

    if any(p in combined for p in DNS_PATTERNS):
        return NetworkErrorSubtype.DNS
    if any(p in combined for p in SSL_PATTERNS):
        return NetworkErrorSubtype.SSL
    if any(p in combined for p in CONNECTION_PATTERNS):
        return NetworkErrorSubtype.CONNECTION
    if any(p in combined for p in TIMEOUT_PATTERNS):
        return NetworkErrorSubtype.TIMEOUT

    return NetworkErrorSubtype.UNKNOWN


# =============================================================================
# ERROR SEVERITY SCORING (US-100-005)
# =============================================================================

class ErrorSeverity(Enum):
    """Error severity levels for retry priority decisions."""
    HIGH = "high"      # Should retry aggressively (network issues)
    MEDIUM = "medium"  # Retry with backoff (rate limits, timeouts)
    LOW = "low"        # Retry sparingly (parse errors)
    NONE = "none"      # Don't retry (unavailable)


# Severity scores: higher = more retryable
ERROR_SEVERITY_SCORES = {
    ErrorSeverity.HIGH: 1.0,
    ErrorSeverity.MEDIUM: 0.6,
    ErrorSeverity.LOW: 0.3,
    ErrorSeverity.NONE: 0.0,
}


def get_error_severity(category: CaptionErrorCategory) -> ErrorSeverity:
    """Get severity level for an error category (US-100-005).

    Args:
        category: The error category.

    Returns:
        ErrorSeverity indicating retry priority.
    """
    severity_map = {
        CaptionErrorCategory.NETWORK: ErrorSeverity.HIGH,
        CaptionErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
        CaptionErrorCategory.RATE_LIMIT: ErrorSeverity.MEDIUM,
        CaptionErrorCategory.PARSE: ErrorSeverity.LOW,
        CaptionErrorCategory.UNAVAILABLE: ErrorSeverity.NONE,
        CaptionErrorCategory.FORMAT_UNAVAILABLE: ErrorSeverity.NONE,
    }
    return severity_map.get(category, ErrorSeverity.MEDIUM)


def get_error_retry_score(category: CaptionErrorCategory, network_subtype: NetworkErrorSubtype = None) -> float:
    """Get numeric retry score for an error (US-100-005).

    Used to prioritize retryable errors over non-retryable ones.

    Args:
        category: The error category.
        network_subtype: Optional sub-type if category is NETWORK.

    Returns:
        Float between 0.0 and 1.0 indicating retry priority.
    """
    base_score = ERROR_SEVERITY_SCORES.get(get_error_severity(category), 0.5)

    # Adjust for network sub-types (DNS and SSL often transient, connection issues may persist)
    if category == CaptionErrorCategory.NETWORK and network_subtype:
        if network_subtype == NetworkErrorSubtype.DNS:
            base_score *= 1.0   # DNS often transient
        elif network_subtype == NetworkErrorSubtype.SSL:
            base_score *= 0.8  # SSL issues may persist
        elif network_subtype == NetworkErrorSubtype.CONNECTION:
            base_score *= 0.7  # Connection issues may need more time

    return base_score


# =============================================================================
# PER-ERROR-CATEGORY RETRY BUDGETS (US-100-005)
# =============================================================================

@dataclass
class CategoryRetryBudget:
    """Per-error-category retry budget tracker (US-100-005).

    Tracks retries per error category to enable granular budget enforcement.
    """
    category: CaptionErrorCategory
    max_retries: int
    current_retries: int = 0

    def can_retry(self) -> bool:
        """Check if retries remain for this category."""
        return self.current_retries < self.max_retries

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.current_retries += 1

    def get_remaining(self) -> int:
        """Get remaining retry count."""
        return max(0, self.max_retries - self.current_retries)


class CategoryBudgetManager:
    """Manages per-category retry budgets (US-100-005)."""

    def __init__(self, category_budgets: Dict[CaptionErrorCategory, int] = None):
        """Initialize with category budgets.

        Args:
            category_budgets: Dict mapping category to max retries.
        """
        self._budgets: Dict[CaptionErrorCategory, CategoryRetryBudget] = {}
        if category_budgets:
            for cat, max_retries in category_budgets.items():
                self._budgets[cat] = CategoryRetryBudget(cat, max_retries)

    def get_budget(self, category: CaptionErrorCategory, default_max: int = 3) -> CategoryRetryBudget:
        """Get or create budget for a category."""
        if category not in self._budgets:
            self._budgets[category] = CategoryRetryBudget(category, default_max)
        return self._budgets[category]

    def can_retry(self, category: CaptionErrorCategory) -> bool:
        """Check if retries remain for category."""
        return self.get_budget(category).can_retry()

    def record_retry(self, category: CaptionErrorCategory) -> None:
        """Record a retry for category."""
        self.get_budget(category).record_retry()

    def get_remaining(self, category: CaptionErrorCategory) -> int:
        """Get remaining retries for category."""
        return self.get_budget(category).get_remaining()

    def get_stats(self) -> Dict[str, Any]:
        """Get budget statistics."""
        return {
            cat.name: {"current": b.current_retries, "max": b.max_retries}
            for cat, b in self._budgets.items()
        }


# =============================================================================
# ERROR PATTERN LEARNING (US-100-005)
# =============================================================================

@dataclass
class ErrorPatternEntry:
    """Tracks error occurrence for pattern learning."""
    video_id: str
    channel_id: str
    category: CaptionErrorCategory
    network_subtype: Optional[NetworkErrorSubtype]
    timestamp: float
    retry_succeeded: bool = False


class ErrorPatternLearner:
    """Learns error patterns by tracking frequency per video/channel (US-100-005).

    Enables adaptive retry decisions based on historical error patterns.
    """

    def __init__(self, max_history: int = 1000):
        """Initialize the learner.

        Args:
            max_history: Maximum error entries to retain.
        """
        self.max_history = max_history
        self._errors: List[ErrorPatternEntry] = []
        self._lock = threading.Lock()

    def record_error(self, video_id: str, channel_id: str,
                     category: CaptionErrorCategory,
                     network_subtype: NetworkErrorSubtype = None,
                     timestamp: float = None) -> None:
        """Record an error occurrence.

        Args:
            video_id: Video ID where error occurred.
            channel_id: Channel ID for the video.
            category: Error category.
            network_subtype: Optional network sub-type.
            timestamp: Error timestamp (defaults to now).
        """
        import time
        entry = ErrorPatternEntry(
            video_id=video_id,
            channel_id=channel_id,
            category=category,
            network_subtype=network_subtype,
            timestamp=timestamp or time.time()
        )
        with self._lock:
            self._errors.append(entry)
            # Trim history if needed
            if len(self._errors) > self.max_history:
                self._errors = self._errors[-self.max_history:]

    def record_retry_outcome(self, video_id: str, succeeded: bool) -> None:
        """Record whether a retry succeeded.

        Args:
            video_id: Video ID that was retried.
            succeeded: Whether the retry succeeded.
        """
        with self._lock:
            for entry in reversed(self._errors):
                if entry.video_id == video_id and not entry.retry_succeeded:
                    entry.retry_succeeded = succeeded
                    break

    def get_channel_error_rate(self, channel_id: str) -> float:
        """Get error rate for a channel.

        Returns:
            Error rate as ratio (0.0 to 1.0).
        """
        with self._lock:
            channel_errors = [e for e in self._errors if e.channel_id == channel_id]
            if not channel_errors:
                return 0.0
            return len(channel_errors) / len(self._errors)

    def get_video_error_count(self, video_id: str) -> int:
        """Get error count for a specific video."""
        with self._lock:
            return sum(1 for e in self._errors if e.video_id == video_id)

    def get_category_frequency(self, category: CaptionErrorCategory) -> float:
        """Get frequency of a category in recent errors.

        Returns:
            Frequency as ratio (0.0 to 1.0).
        """
        with self._lock:
            if not self._errors:
                return 0.0
            recent = self._errors[-100:]  # Last 100 errors
            return sum(1 for e in recent if e.category == category) / len(recent)

    def get_retry_effectiveness(self, category: CaptionErrorCategory) -> float:
        """Calculate retry success rate for a category.

        Returns:
            Retry success rate (0.0 to 1.0).
        """
        with self._lock:
            category_errors = [e for e in self._errors if e.category == category]
            if not category_errors:
                return 0.5  # Default to 50% if no data
            succeeded = sum(1 for e in category_errors if e.retry_succeeded)
            return succeeded / len(category_errors)

    def get_adaptive_retry_limit(self, category: CaptionErrorCategory,
                                 base_limit: int, video_id: str = None) -> int:
        """Get adaptive retry limit based on learned patterns.

        Args:
            category: Error category.
            base_limit: Default retry limit.
            video_id: Optional video ID for video-specific adjustment.

        Returns:
            Adjusted retry limit.
        """
        # Increase limit for high-retryability errors
        retry_rate = get_error_retry_score(category)

        if video_id:
            video_count = self.get_video_error_count(video_id)
            # If video has multiple errors, reduce retries (likely persistent)
            if video_count > 2:
                retry_rate *= 0.5

        # Scale base limit by retry score
        return max(1, int(base_limit * retry_rate))

    def get_stats(self) -> Dict[str, Any]:
        """Get pattern learning statistics."""
        with self._lock:
            return {
                "total_errors": len(self._errors),
                "category_distribution": {
                    cat.name: sum(1 for e in self._errors if e.category == cat)
                    for cat in CaptionErrorCategory
                },
                "retry_effectiveness": {
                    cat.name: self.get_retry_effectiveness(cat)
                    for cat in CaptionErrorCategory
                }
            }


# =============================================================================
# ERROR CLASSIFICATION METRICS (US-100-005)
# =============================================================================

class ErrorClassificationMetrics:
    """Tracks error classification metrics (US-100-005).

    Provides metrics for error_type_distribution and retry_effectiveness_by_type.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self._error_counts: Dict[CaptionErrorCategory, int] = {}
        self._retry_attempts: Dict[CaptionErrorCategory, int] = {}
        self._retry_successes: Dict[CaptionErrorCategory, int] = {}
        self._lock = threading.Lock()

    def record_error(self, category: CaptionErrorCategory) -> None:
        """Record an error occurrence."""
        with self._lock:
            self._error_counts[category] = self._error_counts.get(category, 0) + 1

    def record_retry_attempt(self, category: CaptionErrorCategory) -> None:
        """Record a retry attempt for category."""
        with self._lock:
            self._retry_attempts[category] = self._retry_attempts.get(category, 0) + 1

    def record_retry_success(self, category: CaptionErrorCategory) -> None:
        """Record a successful retry."""
        with self._lock:
            self._retry_successes[category] = self._retry_successes.get(category, 0) + 1

    def get_error_type_distribution(self) -> Dict[str, float]:
        """Get error type distribution as ratios.

        Returns:
            Dict mapping category names to ratios.
        """
        with self._lock:
            total = sum(self._error_counts.values())
            if total == 0:
                return {cat.name: 0.0 for cat in CaptionErrorCategory}
            return {
                cat.name: self._error_counts.get(cat, 0) / total
                for cat in CaptionErrorCategory
            }

    def get_retry_effectiveness_by_type(self) -> Dict[str, float]:
        """Get retry success rate by error type.

        Returns:
            Dict mapping category names to success rates (0.0 to 1.0).
        """
        with self._lock:
            result = {}
            for cat in CaptionErrorCategory:
                attempts = self._retry_attempts.get(cat, 0)
                successes = self._retry_successes.get(cat, 0)
                if attempts == 0:
                    result[cat.name] = 0.5  # Default if no data
                else:
                    result[cat.name] = successes / attempts
            return result

    def get_summary(self) -> Dict[str, Any]:
        """Get full metrics summary."""
        with self._lock:
            return {
                "error_type_distribution": self.get_error_type_distribution(),
                "retry_effectiveness_by_type": self.get_retry_effectiveness_by_type(),
                "total_errors": sum(self._error_counts.values()),
                "total_retries": sum(self._retry_attempts.values()),
            }


# =============================================================================
# LOGGING ERROR CLASSIFICATION DECISIONS (US-100-005)
# =============================================================================

def log_classification_decision(category: CaptionErrorCategory,
                               network_subtype: NetworkErrorSubtype = None,
                               video_id: str = "",
                               severity: ErrorSeverity = None,
                               retry_decision: str = "") -> None:
    """Log error classification decision at WARNING level with reasoning (US-100-005).

    Args:
        category: The error category.
        network_subtype: Optional network sub-type.
        video_id: Video ID for context.
        severity: Error severity level.
        retry_decision: Human-readable retry decision.
    """
    if severity is None:
        severity = get_error_severity(category)

    # Build context string
    context_parts = []
    if video_id:
        context_parts.append(f"video={video_id}")
    context_parts.append(f"category={category.name}")
    if network_subtype and category == CaptionErrorCategory.NETWORK:
        context_parts.append(f"subtype={network_subtype.value}")
    context_parts.append(f"severity={severity.value}")

    context = ", ".join(context_parts)
    reasoning = f"Retry decision: {retry_decision}" if retry_decision else "No retry"

    logger.warning(f"CaptionErrorClassification: {context} | {reasoning}")


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

    # Rate limit patterns (US-67-009: shared from src/common/error_patterns.py)
    if any(p in combined for p in RATE_LIMIT_PATTERNS):
        return CaptionErrorCategory.RATE_LIMIT

    # Timeout patterns (US-67-009: shared from src/common/error_patterns.py)
    if any(p in combined for p in TIMEOUT_PATTERNS):
        return CaptionErrorCategory.TIMEOUT

    # Parse error patterns (US-67-009: shared from src/common/error_patterns.py)
    if any(p in combined for p in PARSE_PATTERNS):
        return CaptionErrorCategory.PARSE

    # Format-specific unavailable (US-59-004, US-67-009: shared)
    if any(p in combined for p in FORMAT_UNAVAILABLE_PATTERNS):
        return CaptionErrorCategory.FORMAT_UNAVAILABLE

    # Unavailable patterns (US-67-009: shared from src/common/error_patterns.py)
    if any(p in combined for p in UNAVAILABLE_PATTERNS):
        return CaptionErrorCategory.UNAVAILABLE

    # Network patterns (US-67-009: shared from src/common/error_patterns.py)
    if any(p in combined for p in NETWORK_PATTERNS):
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
