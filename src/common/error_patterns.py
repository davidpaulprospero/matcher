"""
Shared error severity patterns for yt-dlp error classification.

US-67-009: Extracted from downloader/error_classification.py and
caption/error_handling.py to provide a single source of truth for
error pattern matching across both the download and caption systems.

Both systems previously maintained independent copies of overlapping
patterns (403, 429, bot detection, rate limiting, network errors).
Changes to one didn't propagate to the other. This module centralizes
the shared patterns so both systems stay in sync.

This module provides:
- RATE_LIMIT_PATTERNS: Patterns indicating API rate limiting / throttling
- BOT_DETECTION_PATTERNS: Patterns indicating bot detection / access blocks
- TIMEOUT_PATTERNS: Patterns indicating request/connection timeouts
- NETWORK_PATTERNS: Patterns indicating network connectivity issues
- PARSE_PATTERNS: Patterns indicating response parsing failures
- UNAVAILABLE_PATTERNS: Patterns indicating content is unavailable
"""

from __future__ import annotations

# Rate limit / throttling patterns (429, quota, etc.)
# Used by: downloader severity classification (medium/high), caption error categorization
RATE_LIMIT_PATTERNS: list[str] = [
    "429",
    "too many requests",
    "rate limit",
    "rate-limit",
    "quota exceeded",
    "throttle",
    "slow down",
]

# Bot detection / severe access block patterns
# Used by: downloader severity classification (high)
BOT_DETECTION_PATTERNS: list[str] = [
    "bot detection",
    "automated",
    "suspicious activity",
    "account suspended",
    "ip blocked",
    "ip has been blocked",
    "permanently banned",
    "daily quota",
]

# Timeout patterns
# Used by: downloader category classification, caption error categorization
TIMEOUT_PATTERNS: list[str] = [
    "timeout",
    "timed out",
    "deadline exceeded",
    "connection timed out",
]

# Network connectivity patterns
# Used by: caption error categorization (catch-all network issues)
NETWORK_PATTERNS: list[str] = [
    "network",
    "connection",
    "dns",
    "resolve",
    "unreachable",
    "refused",
    "reset",
    "broken pipe",
    "http error",
    "ssl",
    "certificate",
    "socket",
    "eof",
    "getaddrinfo",
]

# Parse / decode error patterns
# Used by: caption error categorization
PARSE_PATTERNS: list[str] = [
    "parse",
    "decode",
    "invalid json",
    "malformed",
    "syntax error",
    "unexpected token",
    "invalid format",
    "corrupt",
]

# Content unavailable patterns
# Used by: caption error categorization
UNAVAILABLE_PATTERNS: list[str] = [
    "not available",
    "unavailable",
    "no subtitles",
    "no captions",
    "subtitles disabled",
    "captions disabled",
    "not found",
]

# Low severity auth patterns (age-gate, login)
# Used by: downloader severity classification (low)
AUTH_PATTERNS: list[str] = [
    "sign in",
    "login required",
    "confirm your age",
]

# Format-specific unavailable patterns (US-59-004)
FORMAT_UNAVAILABLE_PATTERNS: list[str] = [
    "requested format is not available",
]

# US-113-006: New patterns for enhanced error classification
# Geo-blocking patterns
GEO_BLOCKED_PATTERNS: list[str] = [
    "geo block",
    "geo-restricted",
    "not available in your country",
    "not available in your region",
    "this content is not available",
    "video is not available",
]

# Device limit patterns
DEVICE_LIMIT_PATTERNS: list[str] = [
    "device limit",
    "too many devices",
    "playback on other",
    "exceeded the limit",
]

# US-120-002: Unknown error patterns for unclassified yt-dlp errors
# These are fallback patterns when no known category matches
UNKNOWN_PATTERNS: list[str] = [
    "error",
    "failed",
    "exception",
    "unhandled",
    "unexpected",
    "unknown",
]

# US-136-002: Additional patterns to reduce Unknown error rate
# These patterns capture common yt-dlp errors that weren't previously classified
# HTTP 5xx server errors
HTTP_5XX_PATTERNS: list[str] = [
    "500 internal server error",
    "500 internal server error",
    "501 not implemented",
    "502 bad gateway",
    "503 service unavailable",
    "503 service temporarily unavailable",
    "504 gateway timeout",
    "505 http version not supported",
    "520 web server is returning an unknown error",
    "521 web server is down",
    "522 connection timed out",
    "523 origin is unreachable",
    "524 a timeout occurred",
]

# Extractor-specific errors from yt-dlp
EXTRACTOR_ERROR_PATTERNS: list[str] = [
    "extractor error",
    "unable to extract",
    "could not extract",
    "no suitable extractor",
    "no extractor found",
    "module not found",
    "requested format not found",
    "no format found",
    "no video found",
    "video not found",
    "playlist not found",
    "channel not found",
    "username not found",
    "requested data not found",
    "no entries found",
    "no results found",
]

# yt-dlp specific error messages
YTDLP_ERROR_PATTERNS: list[str] = [
    "postprocessing error",
    "download error",
    "encoding error",
    "fragment error",
    "regex error",
    "resolver error",
    "captions not found",
    "no subtitles",
    "no captions available",
    "metadata not found",
    "thumbnail not found",
    "unable to download",
    "unable to extract",
]

# Additional network/connection patterns
EXTENDED_NETWORK_PATTERNS: list[str] = [
    "connection aborted",
    "connection reset by peer",
    "broken pipe",
    "network change",
    "interrupted",
    "operation timed out",
    "send failed",
    "recv failed",
    "ssl bad",
    "ssl routines",
    "tls handshake failure",
    "read error",
    "write error",
    "connection closed",
    "remote end closed",
    "peer closed",
]

# Permission/auth related patterns
PERMISSION_PATTERNS: list[str] = [
    "permission denied",
    "access denied",
    "not allowed",
    "unauthorized",
    "forbidden",
    "insufficient permissions",
]

# Content ID/copyright patterns
# US-136-002: Made more specific to avoid false positives
CONTENT_ID_PATTERNS: list[str] = [
    "content id",
    "copyright",
    "blocked by",
    "administrator",
    "terms of service violation",
    "copyright violation",
]

# US-144-002: Additional yt-dlp extractor patterns to reduce Unknown errors
# More specific extractor errors that weren't previously classified
YTDLP_EXTRACTOR_PATTERNS: list[str] = [
    # Specific extractor error codes
    "extractor error",
    "unable to extract",
    "could not extract",
    "no suitable extractor",
    "no extractor found",
    "module not found",
    # Video/playlist extraction failures
    "requested format not found",
    "no format found",
    "no video found",
    "video not found",
    "playlist not found",
    "channel not found",
    "username not found",
    "requested data not found",
    "no entries found",
    "no results found",
    # Additional extraction errors
    "unable to extract data",
    "unable to extract player",
    "no player response",
    "no playability data",
    "no embed data",
    "embed code not found",
    "thumbnail not available",
    "no subtitles found",
    "subtitles not available",
    "annotations not found",
    "infojson not found",
    "cannot download info",
]

# US-144-002: Additional HTTP error patterns (4xx client errors)
HTTP_4XX_PATTERNS: list[str] = [
    "400 bad request",
    "401 unauthorized",
    "402 payment required",
    "403 forbidden",
    "404 not found",
    "405 method not allowed",
    "406 not acceptable",
    "407 proxy authentication required",
    "408 request timeout",
    "409 conflict",
    "410 gone",
    "411 length required",
    "412 precondition failed",
    "413 payload too large",
    "414 uri too long",
    "415 unsupported media type",
    "416 range not satisfiable",
    "417 expectation failed",
    "418 i'm a teapot",
    "421 misdirected request",
    "422 unprocessable entity",
    "423 locked",
    "424 failed dependency",
    "425 too early",
    "426 upgrade required",
    "428 precondition required",
    "429 too many requests",
]

# US-144-002: Platform-specific error patterns
PLATFORM_ERROR_PATTERNS: list[str] = [
    # Windows-specific
    "windows error",
    "winerror",
    "error_win",
    # Linux-specific
    "linux error",
    "errno",
    # macOS-specific
    "darwin",
    "macos error",
    # Common OS-level errors
    "no such file",
    "file not found",
    "directory not found",
    "disk full",
    "no space left",
    "quota exceeded",
    "disk quota exceeded",
    "input/output error",
    "io error",
    "device not ready",
    "media not found",
]

# US-144-002: Format-specific error patterns
FORMAT_ERROR_PATTERNS: list[str] = [
    "requested format is not available",
    "format not available",
    "no such format",
    "unsupported format",
    "invalid format",
    "unknown format",
    "video format not found",
    "audio format not found",
    "subtitles format not available",
    "no compatible format",
    "mimetype not supported",
    "codec not supported",
    "container not supported",
    "resolution not available",
    "bitrate not available",
]

# US-144-002: Download/streaming error patterns
DOWNLOAD_ERROR_PATTERNS: list[str] = [
    "download failed",
    "download error",
    "download interrupted",
    "download timeout",
    "download cancelled",
    "download aborted",
    "unable to download",
    "could not download",
    "failed to download",
    "error downloading",
    "download stalled",
    "download incomplete",
    "partial download",
    "connection lost during download",
    "http error",
    "https error",
    "transfer failed",
    "send failed",
    "receive failed",
    "retries exhausted",
    "max retries exceeded",
    "too many redirects",
    "redirect limit exceeded",
    "follow failed",
    "could not follow",
    "too many errors",
]

# US-144-002: Authentication/session error patterns
SESSION_ERROR_PATTERNS: list[str] = [
    "session expired",
    "session invalid",
    "session not found",
    "cookie expired",
    "cookie invalid",
    "cookie required",
    "authentication required",
    "auth expired",
    "auth invalid",
    "token expired",
    "token invalid",
    "refresh failed",
    "oauth failed",
    "signin required",
    "account error",
    "profile not found",
    "not authenticated",
]

# US-144-002: Additional media-specific error patterns
MEDIA_ERROR_PATTERNS: list[str] = [
    "video unavailable",
    "video removed",
    "video deleted",
    "video private",
    "video age restricted",
    "video not available",
    "video blocked",
    "content warning",
    "harmful content",
    "spam",
    "scam",
    "misinformation",
    "medical misinformation",
    "restricted mode",
    "family safe",
    "kids safe",
    "livestream not available",
    "live not available",
    "premiere not available",
    "video ended",
    "age restricted",
]

# US-144-002: Quality/resolution error patterns
QUALITY_ERROR_PATTERNS: list[str] = [
    "quality not available",
    "resolution not available",
    "quality unavailable",
    "requested quality not found",
    "best quality unavailable",
    "worst quality not found",
    "no quality",
    "quality limit",
    "bitrate limit",
    "framerate not available",
    "hdr not available",
    "vp9 not available",
    "av1 not available",
    "h264 not available",
    "audio not available",
    "video not available",
]

# US-144-002: Region/location error patterns
REGION_ERROR_PATTERNS: list[str] = [
    "not available in your country",
    "not available in your region",
    "geo restricted",
    "geo blocked",
    "location blocked",
    "country blocked",
    "region blocked",
    "access restricted",
    "content restricted",
    "video is not available",
    "this content is not available",
    "playback restricted",
    "embed not allowed",
    "embed disabled",
    "embedding disabled",
    "third party blocked",
]

# US-144-002: Generic fallback error patterns that should catch remaining Unknowns
# These are lower-priority catch-alls
GENERIC_ERROR_PATTERNS: list[str] = [
    "error occurred",
    "an error occurred",
    "something went wrong",
    "operation failed",
    "request failed",
    "call failed",
    "execution failed",
    "processing failed",
    "failed with",
    "failed to",
    "unable to",
    "could not",
    "cannot",
    "unsupported",
    "invalid",
    "malformed",
    "corrupted",
    "unexpected",
    "unknown",
]


# =============================================================================
# Severity-classified pattern groups (US-82-012)
# =============================================================================
# These provide the shared base patterns for error severity classification.
# Consumers (e.g. downloader/error_classification.py) import these and may
# append module-specific patterns.

# High severity: bot detection + quota exceeded (subset of RATE_LIMIT_PATTERNS)
# US-136-002: Added HTTP 5xx errors, content ID/copyright blocks (serious blocks)
# US-144-002: Added region errors, media errors as high severity (serious blocks)
HIGH_SEVERITY_PATTERNS: list[str] = BOT_DETECTION_PATTERNS + [
    "quota exceeded",
    # US-136-002: HTTP 5xx server errors indicate serious issues
] + HTTP_5XX_PATTERNS + [
    # US-136-002: Content copyright/blocking is serious
] + CONTENT_ID_PATTERNS + [
    # US-144-002: Region blocking is serious
] + REGION_ERROR_PATTERNS + [
    # US-144-002: Media unavailability is serious
] + MEDIA_ERROR_PATTERNS

# Medium severity: core rate-limit patterns (excluding those assigned elsewhere)
# 'quota exceeded' -> high, 'throttle'/'rate-limit'/'slow down' -> low or excluded
# US-136-002: Added extractor errors, yt-dlp errors, extended network patterns
# US-144-002: Added download errors, quality errors, platform errors as medium severity
MEDIUM_SEVERITY_PATTERNS: list[str] = [
    p for p in RATE_LIMIT_PATTERNS
    if p not in ("quota exceeded", "throttle", "rate-limit", "slow down")
] + EXTRACTOR_ERROR_PATTERNS + YTDLP_ERROR_PATTERNS + EXTENDED_NETWORK_PATTERNS + [
    # US-144-002: More extractor patterns
] + YTDLP_EXTRACTOR_PATTERNS + [
    # US-144-002: HTTP 4xx errors are medium severity (retryable)
] + HTTP_4XX_PATTERNS + [
    # US-144-002: Download errors are medium severity
] + DOWNLOAD_ERROR_PATTERNS + [
    # US-144-002: Quality errors are medium severity
] + QUALITY_ERROR_PATTERNS + [
    # US-144-002: Platform errors are medium severity
] + PLATFORM_ERROR_PATTERNS

# Low severity: auth patterns (age-gate, login)
# US-136-002: Added permission patterns
# US-144-002: Added session errors and format errors as low severity (easily retryable)
LOW_SEVERITY_PATTERNS: list[str] = list(AUTH_PATTERNS) + PERMISSION_PATTERNS + [
    # US-144-002: Session errors are low severity
] + SESSION_ERROR_PATTERNS + [
    # US-144-002: Format errors are low severity (fallback available)
] + FORMAT_ERROR_PATTERNS


# =============================================================================
# US-120-002: Unknown Error Handler
# =============================================================================

import logging
import traceback
from typing import Optional

logger = logging.getLogger(__name__)


class UnknownErrorHandler:
    """Handler for unclassified yt-dlp errors.

    US-120-002: Captures and logs Unknown errors with full context
    (original message and stack trace) for later analysis.

    This handler:
    - Attempts to extract meaningful categories from Unknown errors
    - Logs Unknown errors with full stack traces for diagnostics
    - Records errors to error_aggregator for pattern analysis
    """

    def __init__(self, enable_stack_trace_capture: bool = True):
        """Initialize UnknownErrorHandler.

        Args:
            enable_stack_trace_capture: Whether to capture stack traces (default True)
        """
        self._enable_stack_trace_capture = enable_stack_trace_capture
        self._captured_errors: list[dict] = []

    def handle_unknown_error(
        self,
        error_msg: str,
        exception: Optional[Exception] = None,
    ) -> dict:
        """Handle an unclassified error with full context capture.

        Args:
            error_msg: The error message string
            exception: Optional exception object for stack trace capture

        Returns:
            Dict with error details including original message and stack trace
        """
        # Capture stack trace if exception provided
        stack_trace: Optional[str] = None
        if exception and self._enable_stack_trace_capture:
            stack_trace = ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))

        # Attempt to extract meaningful categories
        extracted_categories = self._extract_categories(error_msg)

        # Build error record
        error_record = {
            'original_message': error_msg,
            'stack_trace': stack_trace,
            'extracted_categories': extracted_categories,
            'is_unknown': True,
        }

        # Store for later analysis
        self._captured_errors.append(error_record)

        # Log with full context
        self._log_unknown_error(error_record)

        return error_record

    def _extract_categories(self, error_msg: str) -> list[str]:
        """Attempt to extract meaningful categories from Unknown error.

        Falls back to known patterns to provide hints about what the error
        might actually be, even if not matched by primary classification.

        US-136-002: Extended to include HTTP_5XX_PATTERNS, EXTRACTOR_ERROR_PATTERNS,
        YTDLP_ERROR_PATTERNS, EXTENDED_NETWORK_PATTERNS, PERMISSION_PATTERNS,
        and CONTENT_ID_PATTERNS.

        Args:
            error_msg: The error message string

        Returns:
            List of potential categories extracted from the message
        """
        categories: list[str] = []
        error_lower = error_msg.lower()

        # Check each known pattern category for partial matches
        # that might help understand the Unknown error
        for pattern in RATE_LIMIT_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"rate_limit ({pattern})")
                break

        for pattern in TIMEOUT_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"timeout ({pattern})")
                break

        for pattern in NETWORK_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"network ({pattern})")
                break

        for pattern in BOT_DETECTION_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"bot_detection ({pattern})")
                break

        for pattern in PARSE_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"parse ({pattern})")
                break

        for pattern in UNAVAILABLE_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"unavailable ({pattern})")
                break

        for pattern in GEO_BLOCKED_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"geo_blocked ({pattern})")
                break

        for pattern in DEVICE_LIMIT_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"device_limit ({pattern})")
                break

        for pattern in AUTH_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"auth ({pattern})")
                break

        # US-136-002: Extended categories
        for pattern in HTTP_5XX_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"server_error ({pattern})")
                break

        for pattern in EXTRACTOR_ERROR_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"extractor ({pattern})")
                break

        for pattern in YTDLP_ERROR_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"ytdlp_error ({pattern})")
                break

        for pattern in EXTENDED_NETWORK_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"network ({pattern})")
                break

        for pattern in PERMISSION_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"permission ({pattern})")
                break

        for pattern in CONTENT_ID_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"content_block ({pattern})")
                break

        # US-144-002: Extended categories for new patterns
        for pattern in YTDLP_EXTRACTOR_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"extractor ({pattern})")
                break

        for pattern in HTTP_4XX_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"client_error ({pattern})")
                break

        for pattern in PLATFORM_ERROR_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"platform ({pattern})")
                break

        for pattern in FORMAT_ERROR_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"format ({pattern})")
                break

        for pattern in DOWNLOAD_ERROR_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"download ({pattern})")
                break

        for pattern in SESSION_ERROR_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"session ({pattern})")
                break

        for pattern in MEDIA_ERROR_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"media_unavailable ({pattern})")
                break

        for pattern in QUALITY_ERROR_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"quality ({pattern})")
                break

        for pattern in REGION_ERROR_PATTERNS:
            if pattern.lower() in error_lower:
                categories.append(f"region_block ({pattern})")
                break

        return categories

    def _log_unknown_error(self, error_record: dict) -> None:
        """Log Unknown error with full context.

        Args:
            error_record: Dict containing error details
        """
        original_msg = error_record['original_message']
        stack_trace = error_record.get('stack_trace')
        extracted = error_record.get('extracted_categories', [])

        # Log at WARNING level since this is an unclassified error
        logger.warning("Unknown error encountered: %s", original_msg[:200])

        if extracted:
            logger.warning(
                "Unknown error - potential categories: %s",
                ', '.join(extracted)
            )

        if stack_trace:
            logger.debug("Unknown error stack trace:\n%s", stack_trace)

    def get_captured_errors(self) -> list[dict]:
        """Get all captured Unknown errors for analysis.

        Returns:
            List of error record dicts
        """
        return list(self._captured_errors)

    def clear(self) -> None:
        """Clear captured errors."""
        self._captured_errors.clear()


# =============================================================================
# US-123-005: Unknown Error Pattern Learning System
# =============================================================================

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


# Default path for persisting learned patterns
DEFAULT_LEARNED_PATTERNS_PATH = os.path.expanduser("~/.matcher_learned_error_patterns.json")


@dataclass
class LearnedPattern:
    """A learned error pattern with its classification.

    Attributes:
        pattern: The error pattern text
        proposed_category: The proposed category (e.g., 'rate_limit', 'network')
        confidence: Confidence score (0.0-1.0) based on frequency
        occurrences: Number of times this pattern was observed
        first_seen: Timestamp of first occurrence
        last_seen: Timestamp of most recent occurrence
    """
    pattern: str
    proposed_category: str
    confidence: float = 0.0
    occurrences: int = 1
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())


class UnknownPatternLearner:
    """Learns from Unknown errors to build a classification database.

    US-123-005: Tracks previously unknown error messages and learns to classify
    them based on:
    - User-provided classifications (via learn_from_unknown)
    - Pattern matching against existing known patterns
    - Frequency of occurrence (more frequent = higher confidence)

    This enables the system to gradually learn to classify errors that were
    previously Unknown, reducing the 40x "Unknown" errors from lessons learned.

    Example:
        learner = UnknownPatternLearner()

        # Record a new unknown error with proposed category
        learner.learn_from_unknown(
            error_msg="yt-dlp extractor failed with code 12345",
            proposed_category="extractor_error"
        )

        # Later, auto-classify similar errors
        category = learner.auto_classify_unknown(
            "yt-dlp extractor failed with code 99999"
        )
        # Returns 'extractor_error' if pattern matches

        # Persist for future runs
        learner.persist_learned_patterns()
    """

    def __init__(
        self,
        storage_path: str = DEFAULT_LEARNED_PATTERNS_PATH,
        min_confidence_threshold: float = 0.3,
        max_patterns: int = 1000,
    ):
        """Initialize UnknownPatternLearner.

        Args:
            storage_path: Path to persist learned patterns (JSON file)
            min_confidence_threshold: Minimum confidence to use for auto-classification
            max_patterns: Maximum number of patterns to store
        """
        self._storage_path = storage_path
        self._min_confidence_threshold = min_confidence_threshold
        self._max_patterns = max_patterns
        self._learned_patterns: dict[str, LearnedPattern] = {}
        self._load_learned_patterns()

    def learn_from_unknown(
        self,
        error_msg: str,
        proposed_category: str,
    ) -> LearnedPattern:
        """Record a new error pattern with proposed category.

        If the pattern already exists, increments the occurrence count and
        updates the confidence score.

        Args:
            error_msg: The error message string to learn
            proposed_category: The proposed category (e.g., 'rate_limit', 'network')

        Returns:
            The LearnedPattern that was created or updated
        """
        # Normalize the pattern (use first 200 chars as key)
        pattern_key = error_msg[:200].lower().strip()

        if pattern_key in self._learned_patterns:
            # Increment occurrence and update confidence
            existing = self._learned_patterns[pattern_key]
            existing.occurrences += 1
            existing.last_seen = datetime.now().isoformat()
            # Confidence increases with more occurrences, capped at 1.0
            existing.confidence = min(1.0, existing.occurrences / 10.0)
            return existing
        else:
            # Create new pattern
            # If user proposes a category that exists in our known patterns,
            # boost initial confidence
            initial_confidence = 0.5 if proposed_category else 0.1

            new_pattern = LearnedPattern(
                pattern=error_msg[:200],
                proposed_category=proposed_category,
                confidence=initial_confidence,
            )

            # Enforce max patterns limit (remove lowest confidence)
            if len(self._learned_patterns) >= self._max_patterns:
                self._evict_lowest_confidence()

            self._learned_patterns[pattern_key] = new_pattern
            return new_pattern

    def _evict_lowest_confidence(self) -> None:
        """Remove the lowest confidence pattern to make room for new ones."""
        if not self._learned_patterns:
            return

        lowest_key = min(
            self._learned_patterns.keys(),
            key=lambda k: self._learned_patterns[k].confidence
        )
        del self._learned_patterns[lowest_key]

    def get_learned_patterns(
        self,
        min_confidence: Optional[float] = None,
    ) -> list[LearnedPattern]:
        """Retrieve learned patterns for classification.

        Args:
            min_confidence: Optional minimum confidence threshold to filter patterns

        Returns:
            List of LearnedPattern objects, sorted by confidence (highest first)
        """
        threshold = min_confidence if min_confidence is not None else 0.0
        patterns = [
            p for p in self._learned_patterns.values()
            if p.confidence >= threshold
        ]
        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def auto_classify_unknown(self, error_msg: str) -> Optional[str]:
        """Attempt to classify an unknown error based on learned patterns.

        Uses pattern matching against learned patterns, returning the
        category with highest confidence if above threshold.

        Args:
            error_msg: The error message to classify

        Returns:
            The learned category if confidence is above threshold, None otherwise
        """
        error_lower = error_msg.lower()

        best_match: Optional[LearnedPattern] = None

        for pattern in self._learned_patterns.values():
            if pattern.confidence < self._min_confidence_threshold:
                continue

            # Check if the learned pattern is a substring of the error
            if pattern.pattern.lower() in error_lower:
                if best_match is None or pattern.confidence > best_match.confidence:
                    best_match = pattern

        if best_match:
            logger.info(
                "Auto-classified unknown error as '%s' (confidence=%.2f)",
                best_match.proposed_category,
                best_match.confidence
            )
            return best_match.proposed_category

        return None

    def persist_learned_patterns(self, path: Optional[str] = None) -> bool:
        """Save learned patterns to disk.

        Args:
            path: Optional custom path, uses default if not provided

        Returns:
            True if successfully saved, False otherwise
        """
        save_path = path or self._storage_path

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Convert to dict for JSON serialization
            patterns_dict = {
                key: asdict(pattern)
                for key, pattern in self._learned_patterns.items()
            }

            data = {
                "version": "1.0",
                "patterns": patterns_dict,
                "updated_at": datetime.now().isoformat(),
            }

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(
                "Persisted %d learned error patterns to %s",
                len(self._learned_patterns),
                save_path
            )
            return True

        except Exception as e:
            logger.error("Failed to persist learned patterns: %s", e)
            return False

    def _load_learned_patterns(self) -> bool:
        """Load learned patterns from disk.

        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(self._storage_path):
            logger.debug("No learned patterns file found, starting fresh")
            return False

        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            patterns_dict = data.get("patterns", {})
            for key, pattern_data in patterns_dict.items():
                self._learned_patterns[key] = LearnedPattern(**pattern_data)

            logger.info(
                "Loaded %d learned error patterns from %s",
                len(self._learned_patterns),
                self._storage_path
            )
            return True

        except Exception as e:
            logger.warning("Failed to load learned patterns: %s", e)
            return False

    def clear_learned_patterns(self) -> int:
        """Clear all learned patterns.

        Returns:
            Number of patterns that were cleared
        """
        count = len(self._learned_patterns)
        self._learned_patterns.clear()
        logger.info("Cleared %d learned error patterns", count)
        return count

    def get_stats(self) -> dict:
        """Get statistics about learned patterns.

        Returns:
            Dict with pattern counts, categories, etc.
        """
        categories: dict[str, int] = {}
        total_occurrences = 0

        for pattern in self._learned_patterns.values():
            cat = pattern.proposed_category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1
            total_occurrences += pattern.occurrences

        return {
            "total_patterns": len(self._learned_patterns),
            "total_occurrences": total_occurrences,
            "by_category": categories,
            "storage_path": self._storage_path,
        }


# Global learner instance
_unknown_pattern_learner: Optional[UnknownPatternLearner] = None


def get_unknown_pattern_learner() -> UnknownPatternLearner:
    """Get the global UnknownPatternLearner instance.

    Returns:
        The global UnknownPatternLearner
    """
    global _unknown_pattern_learner
    if _unknown_pattern_learner is None:
        _unknown_pattern_learner = UnknownPatternLearner()
    return _unknown_pattern_learner


# =============================================================================
# US-123-008: Error Pattern Auto-Discovery
# =============================================================================

import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


# Default path for discovered patterns that need manual review
DEFAULT_DISCOVERY_PATH = os.path.expanduser("~/.matcher_discovered_error_patterns.json")

# Default confidence threshold for auto-adding patterns
DEFAULT_DISCOVERY_CONFIDENCE = 0.7


@dataclass
class DiscoveredPattern:
    """A discovered error pattern pending manual review.

    Attributes:
        pattern: The error pattern text (extracted from error message)
        proposed_category: The proposed category (e.g., 'rate_limit', 'network')
        confidence: Confidence score (0.0-1.0) based on occurrence frequency
        occurrences: Number of times this pattern was observed
        sample_messages: Example error messages containing this pattern
        first_seen: Timestamp of first occurrence
        last_seen: Timestamp of most recent occurrence
        status: 'pending', 'approved', 'rejected'
    """
    pattern: str
    proposed_category: str
    confidence: float = 0.0
    occurrences: int = 1
    sample_messages: list[str] = field(default_factory=list)
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"


class ErrorPatternDiscovery:
    """Discovers new error patterns from unknown yt-dlp errors.

    US-123-008: Analyzes unknown error messages to extract potential new patterns
    that can be added to the error classification system.

    This enables automatic discovery of new YouTube/yt-dlp error patterns by:
    - Analyzing unknown errors for common substrings/patterns
    - Clustering similar error messages
    - Proposing categories based on error text analysis
    - Maintaining a review queue for manual approval

    Example:
        discovery = ErrorPatternDiscovery()

        # Analyze new unknown errors
        discovery.discover_patterns([
            "ERROR: [youtube] extractor error: HTTP Error 503: Service Unavailable",
            "ERROR: [youtube] extractor error: HTTP Error 503: Backend connection failed",
        ])

        # Get patterns pending review
        pending = discovery.get_pending_patterns()

        # Approve a pattern after review
        discovery.review_discovered_patterns([
            {"pattern": "http error 503", "approved": True}
        ])
    """

    def __init__(
        self,
        storage_path: str = DEFAULT_DISCOVERY_PATH,
        discovery_confidence: float = DEFAULT_DISCOVERY_CONFIDENCE,
        min_occurrences: int = 2,
        max_patterns: int = 500,
    ):
        """Initialize ErrorPatternDiscovery.

        Args:
            storage_path: Path to persist discovered patterns (JSON file)
            discovery_confidence: Minimum confidence to auto-suggest patterns
            min_occurrences: Minimum occurrences before proposing a pattern
            max_patterns: Maximum number of patterns to store
        """
        self._storage_path = storage_path
        self._discovery_confidence = discovery_confidence
        self._min_occurrences = min_occurrences
        self._max_patterns = max_patterns
        self._discovered_patterns: dict[str, DiscoveredPattern] = {}
        self._load_discovered_patterns()

    def discover_patterns(self, error_messages: list[str]) -> list[DiscoveredPattern]:
        """Extract potential patterns from unknown error messages.

        Analyzes error messages to find common substrings that might represent
        new error patterns worth adding to the classification system.

        Args:
            error_messages: List of error message strings to analyze

        Returns:
            List of DiscoveredPattern objects that meet the threshold
        """
        # Track pattern candidates by extracting meaningful substrings
        pattern_candidates: dict[str, dict] = {}

        for error_msg in error_messages:
            if not error_msg:
                continue

            # Extract potential patterns from the error message
            candidates = self._extract_pattern_candidates(error_msg)

            for candidate in candidates:
                if candidate not in pattern_candidates:
                    pattern_candidates[candidate] = {
                        'count': 0,
                        'samples': [],
                    }
                pattern_candidates[candidate]['count'] += 1
                if len(pattern_candidates[candidate]['samples']) < 3:
                    pattern_candidates[candidate]['samples'].append(error_msg)

        # Convert candidates to DiscoveredPattern objects
        discovered = []
        now = datetime.now().isoformat()

        for pattern_text, data in pattern_candidates.items():
            if data['count'] < self._min_occurrences:
                continue

            # Calculate confidence based on occurrence count
            confidence = min(1.0, data['count'] / 10.0)

            # Propose category based on the error text
            proposed_category = self.propose_pattern_category(pattern_text, data['samples'])

            # Check if we already have this pattern
            if pattern_text in self._discovered_patterns:
                existing = self._discovered_patterns[pattern_text]
                existing.occurrences += data['count']
                existing.last_seen = now
                existing.confidence = min(1.0, existing.occurrences / 10.0)
                # Add new samples if we have room
                for sample in data['samples']:
                    if sample not in existing.sample_messages and len(existing.sample_messages) < 5:
                        existing.sample_messages.append(sample)
                continue

            # Create new discovered pattern
            discovered_pattern = DiscoveredPattern(
                pattern=pattern_text,
                proposed_category=proposed_category,
                confidence=confidence,
                occurrences=data['count'],
                sample_messages=data['samples'][:5],
            )

            # Enforce max patterns limit
            if len(self._discovered_patterns) >= self._max_patterns:
                self._evict_lowest_confidence()

            self._discovered_patterns[pattern_text] = discovered_pattern
            discovered.append(discovered_pattern)

        return discovered

    def _extract_pattern_candidates(self, error_msg: str) -> list[str]:
        """Extract potential pattern candidates from an error message.

        Args:
            error_msg: The error message to analyze

        Returns:
            List of potential pattern candidates
        """
        candidates = []
        error_lower = error_msg.lower()

        # Extract HTTP error codes
        http_matches = re.findall(r'http\s*error\s*(\d{3})', error_lower)
        for code in http_matches:
            candidates.append(f"http error {code}")

        # Extract yt-dlp error codes
        yt_matches = re.findall(r'error[:\s]+(\w+\s*\d+)', error_lower)
        for match in yt_matches:
            candidates.append(f"error: {match}")

        # Extract extractor-specific errors
        extractor_matches = re.findall(r'\[(\w+)\]\s*(\w+\s*error|\w+\s*failed)', error_lower)
        for extractor, error_type in extractor_matches:
            candidates.append(f"[{extractor}] {error_type}")

        # Extract specific error keywords
        error_keywords = [
            'extractor error',
            'download error',
            'connection error',
            'authentication error',
            'permission denied',
            'service unavailable',
            'internal server error',
            'bad gateway',
            'gateway timeout',
            'backend error',
            'connection refused',
            'connection reset',
            'ssl error',
            'certificate error',
            'timeout error',
        ]

        for keyword in error_keywords:
            if keyword in error_lower:
                candidates.append(keyword)

        return candidates

    def propose_pattern_category(
        self,
        pattern: str,
        sample_messages: Optional[list[str]] = None,
    ) -> str:
        """Suggest a category based on error text analysis.

        Analyzes the pattern and optional sample messages to propose
        an appropriate error category.

        Args:
            pattern: The error pattern text
            sample_messages: Optional sample error messages for context

        Returns:
            Proposed category name (e.g., 'rate_limit', 'network', 'extractor')
        """
        pattern_lower = pattern.lower()

        # Check against known patterns to infer category
        # Rate limit patterns
        if any(p in pattern_lower for p in ['429', 'rate limit', 'quota', 'too many']):
            return 'rate_limit'

        # Network patterns
        if any(p in pattern_lower for p in [
            'connection', 'network', 'timeout', 'dns', 'ssl', 'certificate',
            'refused', 'reset', 'unreachable', 'unavailable'
        ]):
            return 'network'

        # Bot detection / auth patterns
        if any(p in pattern_lower for p in [
            '403', 'forbidden', 'bot', 'captcha', 'blocked', 'banned', 'suspended'
        ]):
            return 'bot_detection'

        # Extractor-specific errors
        if any(p in pattern_lower for p in [
            'extractor', '[youtube]', '[bandcamp]', '[vimeo]'
        ]):
            return 'extractor'

        # Server errors (5xx)
        if '5' in pattern and 'error' in pattern_lower:
            return 'server_error'

        # Default to unknown
        return 'unknown'

    def get_pending_patterns(
        self,
        min_confidence: Optional[float] = None,
    ) -> list[DiscoveredPattern]:
        """Get discovered patterns pending manual review.

        Args:
            min_confidence: Optional minimum confidence threshold

        Returns:
            List of DiscoveredPattern objects with 'pending' status
        """
        threshold = min_confidence if min_confidence is not None else 0.0
        return [
            p for p in self._discovered_patterns.values()
            if p.status == "pending" and p.confidence >= threshold
        ]

    def review_discovered_patterns(
        self,
        reviews: list[dict],
    ) -> dict:
        """Process manual review decisions for discovered patterns.

        Args:
            reviews: List of dicts with 'pattern' and 'approved' keys
                    Optional 'category' key to override proposed category

        Returns:
            Dict with counts of approved/rejected patterns
        """
        results = {"approved": 0, "rejected": 0, "not_found": 0}

        for review in reviews:
            pattern = review.get("pattern", "")
            approved = review.get("approved", False)
            override_category = review.get("category")

            if pattern not in self._discovered_patterns:
                results["not_found"] += 1
                continue

            discovered = self._discovered_patterns[pattern]

            if approved:
                discovered.status = "approved"
                if override_category:
                    discovered.proposed_category = override_category
                results["approved"] += 1

                # Optionally: auto-add to known patterns
                # This would require modifying the actual pattern lists
                logger.info(
                    "Approved pattern '%s' for category '%s'",
                    pattern, discovered.proposed_category
                )
            else:
                discovered.status = "rejected"
                results["rejected"] += 1

        # Persist after review
        self.persist_discovered_patterns()

        return results

    def get_approved_patterns(self) -> list[DiscoveredPattern]:
        """Get all approved discovered patterns.

        Returns:
            List of DiscoveredPattern objects with 'approved' status
        """
        return [
            p for p in self._discovered_patterns.values()
            if p.status == "approved"
        ]

    def get_all_patterns(self) -> list[DiscoveredPattern]:
        """Get all discovered patterns regardless of status.

        Returns:
            List of all DiscoveredPattern objects
        """
        return list(self._discovered_patterns.values())

    def _evict_lowest_confidence(self) -> None:
        """Remove the lowest confidence pattern to make room for new ones."""
        if not self._discovered_patterns:
            return

        lowest_key = min(
            self._discovered_patterns.keys(),
            key=lambda k: self._discovered_patterns[k].confidence
        )
        del self._discovered_patterns[lowest_key]

    def persist_discovered_patterns(self, path: Optional[str] = None) -> bool:
        """Save discovered patterns to disk.

        Args:
            path: Optional custom path, uses default if not provided

        Returns:
            True if successfully saved, False otherwise
        """
        save_path = path or self._storage_path

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Convert to dict for JSON serialization
            patterns_dict = {
                key: asdict(pattern)
                for key, pattern in self._discovered_patterns.items()
            }

            data = {
                "version": "1.0",
                "patterns": patterns_dict,
                "discovery_confidence": self._discovery_confidence,
                "updated_at": datetime.now().isoformat(),
            }

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(
                "Persisted %d discovered error patterns to %s",
                len(self._discovered_patterns),
                save_path
            )
            return True

        except Exception as e:
            logger.error("Failed to persist discovered patterns: %s", e)
            return False

    def _load_discovered_patterns(self) -> bool:
        """Load discovered patterns from disk.

        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(self._storage_path):
            logger.debug("No discovered patterns file found, starting fresh")
            return False

        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            patterns_dict = data.get("patterns", {})
            for key, pattern_data in patterns_dict.items():
                self._discovered_patterns[key] = DiscoveredPattern(**pattern_data)

            # Load confidence threshold if present
            if "discovery_confidence" in data:
                self._discovery_confidence = data["discovery_confidence"]

            logger.info(
                "Loaded %d discovered error patterns from %s",
                len(self._discovered_patterns),
                self._storage_path
            )
            return True

        except Exception as e:
            logger.warning("Failed to load discovered patterns: %s", e)
            return False

    def clear_patterns(self, status: Optional[str] = None) -> int:
        """Clear discovered patterns.

        Args:
            status: Optional status to filter clearing (e.g., 'rejected', 'pending')
                   If None, clears all patterns

        Returns:
            Number of patterns cleared
        """
        if status is None:
            count = len(self._discovered_patterns)
            self._discovered_patterns.clear()
        else:
            to_remove = [
                k for k, p in self._discovered_patterns.items()
                if p.status == status
            ]
            count = len(to_remove)
            for key in to_remove:
                del self._discovered_patterns[key]

        logger.info("Cleared %d discovered error patterns", count)
        return count

    def get_stats(self) -> dict:
        """Get statistics about discovered patterns.

        Returns:
            Dict with pattern counts, categories, etc.
        """
        categories: dict[str, int] = {}
        statuses: dict[str, int] = {}
        total_occurrences = 0

        for pattern in self._discovered_patterns.values():
            cat = pattern.proposed_category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1
            statuses[pattern.status] = statuses.get(pattern.status, 0) + 1
            total_occurrences += pattern.occurrences

        return {
            "total_patterns": len(self._discovered_patterns),
            "total_occurrences": total_occurrences,
            "by_category": categories,
            "by_status": statuses,
            "discovery_confidence": self._discovery_confidence,
            "storage_path": self._storage_path,
        }


# Global discovery instance
_error_pattern_discovery: Optional[ErrorPatternDiscovery] = None


def get_error_pattern_discovery() -> ErrorPatternDiscovery:
    """Get the global ErrorPatternDiscovery instance.

    Returns:
        The global ErrorPatternDiscovery
    """
    global _error_pattern_discovery
    if _error_pattern_discovery is None:
        _error_pattern_discovery = ErrorPatternDiscovery()
    return _error_pattern_discovery


# Integrate with UnknownPatternLearner to discover from captured errors
def discover_from_captured_errors() -> list[DiscoveredPattern]:
    """Discover patterns from all errors captured by UnknownErrorHandler.

    This can be called periodically to batch-discover from accumulated Unknown errors.

    Returns:
        List of DiscoveredPattern objects
    """
    handler = get_unknown_error_handler()
    discovery = get_error_pattern_discovery()

    captured = handler.get_captured_errors()
    messages = [e.get("original_message", "") for e in captured if e.get("original_message")]

    return discovery.discover_patterns(messages)


# Integrate with UnknownPatternLearner to auto-learn approved patterns
def learn_approved_patterns() -> int:
    """Learn patterns from all approved discovered patterns.

    Returns:
        Number of patterns learned
    """
    discovery = get_error_pattern_discovery()
    learner = get_unknown_pattern_learner()

    approved = discovery.get_approved_patterns()
    learned_count = 0

    for pattern in approved:
        learner.learn_from_unknown(pattern.pattern, pattern.proposed_category)
        learned_count += 1

    if learned_count > 0:
        learner.persist_learned_patterns()

    return learned_count


# Global handler instance
_unknown_error_handler: Optional[UnknownErrorHandler] = None


def get_unknown_error_handler() -> UnknownErrorHandler:
    """Get the global UnknownErrorHandler instance.

    Returns:
        The global UnknownErrorHandler
    """
    global _unknown_error_handler
    if _unknown_error_handler is None:
        _unknown_error_handler = UnknownErrorHandler()
    return _unknown_error_handler


def is_unknown_error(error_msg: str) -> bool:
    """Check if an error message is unclassified (Unknown).

    Returns True if the error doesn't match any known pattern categories.
    An error is Unknown if it doesn't match any of the specific error patterns
    (rate limit, bot detection, timeout, network, etc.).

    US-136-002: Extended with HTTP_5XX_PATTERNS, EXTRACTOR_ERROR_PATTERNS,
    YTDLP_ERROR_PATTERNS, EXTENDED_NETWORK_PATTERNS, PERMISSION_PATTERNS,
    and CONTENT_ID_PATTERNS to reduce Unknown error rate.

    US-144-002: Extended with additional patterns to reduce Unknown errors further:
    - YTDLP_EXTRACTOR_PATTERNS: More specific yt-dlp extractor errors
    - HTTP_4XX_PATTERNS: HTTP client errors
    - PLATFORM_ERROR_PATTERNS: Platform-specific errors
    - FORMAT_ERROR_PATTERNS: Format-specific errors
    - DOWNLOAD_ERROR_PATTERNS: Download/streaming errors
    - SESSION_ERROR_PATTERNS: Session/authentication errors
    - MEDIA_ERROR_PATTERNS: Media unavailability errors
    - QUALITY_ERROR_PATTERNS: Quality/resolution errors
    - REGION_ERROR_PATTERNS: Region/location errors

    Args:
        error_msg: The error message string

    Returns:
        True if the error is unclassified (doesn't match any known pattern)
    """
    error_lower = error_msg.lower()

    # Check against all known pattern categories (excluding UNKNOWN_PATTERNS)
    # UNKNOWN_PATTERNS is a catch-all that shouldn't be used for classification
    all_patterns = (
        RATE_LIMIT_PATTERNS +
        BOT_DETECTION_PATTERNS +
        TIMEOUT_PATTERNS +
        NETWORK_PATTERNS +
        PARSE_PATTERNS +
        UNAVAILABLE_PATTERNS +
        GEO_BLOCKED_PATTERNS +
        DEVICE_LIMIT_PATTERNS +
        AUTH_PATTERNS +
        FORMAT_UNAVAILABLE_PATTERNS +
        # US-136-002: Extended patterns to reduce Unknown error rate
        HTTP_5XX_PATTERNS +
        EXTRACTOR_ERROR_PATTERNS +
        YTDLP_ERROR_PATTERNS +
        EXTENDED_NETWORK_PATTERNS +
        PERMISSION_PATTERNS +
        CONTENT_ID_PATTERNS +
        # US-144-002: Additional patterns to reduce Unknown errors
        YTDLP_EXTRACTOR_PATTERNS +
        HTTP_4XX_PATTERNS +
        PLATFORM_ERROR_PATTERNS +
        FORMAT_ERROR_PATTERNS +
        DOWNLOAD_ERROR_PATTERNS +
        SESSION_ERROR_PATTERNS +
        MEDIA_ERROR_PATTERNS +
        QUALITY_ERROR_PATTERNS +
        REGION_ERROR_PATTERNS
    )

    for pattern in all_patterns:
        if pattern.lower() in error_lower:
            return False

    # No known pattern matched - it's Unknown
    return True
