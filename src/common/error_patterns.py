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


# =============================================================================
# Severity-classified pattern groups (US-82-012)
# =============================================================================
# These provide the shared base patterns for error severity classification.
# Consumers (e.g. downloader/error_classification.py) import these and may
# append module-specific patterns.

# High severity: bot detection + quota exceeded (subset of RATE_LIMIT_PATTERNS)
HIGH_SEVERITY_PATTERNS: list[str] = BOT_DETECTION_PATTERNS + [
    "quota exceeded",
]

# Medium severity: core rate-limit patterns (excluding those assigned elsewhere)
# 'quota exceeded' -> high, 'throttle'/'rate-limit'/'slow down' -> low or excluded
MEDIUM_SEVERITY_PATTERNS: list[str] = [
    p for p in RATE_LIMIT_PATTERNS
    if p not in ("quota exceeded", "throttle", "rate-limit", "slow down")
]

# Low severity: auth patterns (age-gate, login)
LOW_SEVERITY_PATTERNS: list[str] = list(AUTH_PATTERNS)
