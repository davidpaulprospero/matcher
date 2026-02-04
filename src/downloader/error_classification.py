"""
Unified error classification for download errors.

US-52-006: Extracted from download_segments.py and core.py to provide a single
source of truth for error classification across the pipeline.

This module provides:
- classify_error_category(): Classify errors into 'network', 'bot_detection',
  'timeout', or 'video_specific' categories.
- is_network_failure(): Check if an error indicates systemic network failure.
- is_escalation_error(): Check if an error warrants bypass tier escalation.
- classify_error_severity(): Classify error severity for adaptive backoff.
- ERROR_SEVERITY_PATTERNS / SEVERITY_MULTIPLIERS: Severity pattern data.
- NETWORK_FAILURE_THRESHOLD / BOT_DETECTION_ABORT_THRESHOLD: Abort thresholds.
"""

from __future__ import annotations

# Error patterns indicating systemic network failures (not video-specific).
# These match both subprocess stderr AND Python API DownloadError messages,
# which wrap exceptions as "ERROR: [youtube] ID: <original exception text>".
NETWORK_FAILURE_PATTERNS = (
    'getaddrinfo failed',
    'Name or service not known',
    'Errno 11001',           # Windows DNS resolution failure
    'nodename nor servname',  # macOS DNS failure
    'Network is unreachable',
    'No address associated with hostname',
    'Temporary failure in name resolution',
    'Failed to resolve',     # curl/curl_cffi DNS failure message
    'URLError',              # Python urllib wrapper (e.g. URLError: <urlopen error ...>)
    'ConnectionResetError',  # Python API: connection dropped mid-transfer
    'Connection refused',    # Server rejecting connections (systemic when widespread)
    'Connection timed out',  # TCP connection timeout (systemic when widespread)
)

# ffmpeg exit code 0xFFFFFEC6 = 4294967158 unsigned = -314 signed (network error)
# Some systems report the unsigned value, others report the signed value.
FFMPEG_NETWORK_EXIT_CODES = ('4294967158', '-314')

# Default threshold for consecutive network failures before aborting
NETWORK_FAILURE_THRESHOLD = 3

# Default threshold for total bot-detection errors before aborting the stage.
# When YouTube is broadly blocking (broken cookies, defeated impersonation),
# every remaining segment hits the block wall with doomed Tier 1 requests.
# This threshold triggers an early abort with actionable guidance.
BOT_DETECTION_ABORT_THRESHOLD = 10


# Error severity mapping for adaptive backoff multiplier (US-008)
# Maps error patterns (case-insensitive) to severity levels
# Severity determines backoff multiplier: low=1.5x, medium=2.0x, high=3.0x
ERROR_SEVERITY_PATTERNS = {
    # High severity: quota exceeded, bot detection, severe blocks
    'high': [
        'quota exceeded',
        'daily quota',
        'bot detection',
        'automated',
        'suspicious activity',
        'account suspended',
        'ip blocked',
        'ip has been blocked',
        'permanently banned',
    ],
    # Medium severity: standard rate limits, too many requests
    'medium': [
        'too many requests',
        '429',
        'rate limit',
        'please try again later',
        'temporarily unavailable',
    ],
    # Low severity: brief rate limits, minor throttling
    'low': [
        'sign in',
        'login required',
        'confirm your age',
        'slow down',
    ],
}

# Multipliers for each severity level
SEVERITY_MULTIPLIERS = {
    'low': 1.5,
    'medium': 2.0,
    'high': 3.0,
}


def is_network_failure(error_msg: str) -> bool:
    """Check if an error message indicates a systemic network failure.

    These are failures that affect ALL downloads (DNS down, no internet),
    as opposed to video-specific errors (403, removed, age-gated).

    Args:
        error_msg: The exception message string.

    Returns:
        True if the error indicates a systemic network issue.
    """
    error_lower = error_msg.lower()
    for pattern in NETWORK_FAILURE_PATTERNS:
        if pattern.lower() in error_lower:
            return True
    # Check for ffmpeg network exit codes (unsigned 4294967158 or signed -314)
    for code in FFMPEG_NETWORK_EXIT_CODES:
        if code in error_msg:
            return True
    return False


def is_escalation_error(error_msg: str) -> bool:
    """Check if an error message indicates a 403/bot-detection/auth error.

    These errors warrant escalation to a higher bypass tier via the
    EscalationManager (Tier 2 extractor_args, Tier 3 cookies).

    Args:
        error_msg: The exception message string.

    Returns:
        True if the error matches 403/bot/auth patterns.
    """
    try:
        from .escalation_manager import is_escalation_trigger
        return is_escalation_trigger(error_msg)
    except ImportError:
        # Fallback: simple pattern match if escalation_manager unavailable
        lower = error_msg.lower()
        return any(p in lower for p in ('403', 'forbidden', 'sign in', 'bot', 'captcha'))


def classify_error_category(error_msg: str) -> str:
    """Classify a download error into a diagnostic category.

    Categories (most specific first):
        'network'       - DNS failure, no connectivity (systemic)
        'bot_detection' - 403/bot/captcha/sign-in errors
        'timeout'       - stall timeouts, socket timeouts
        'video_specific'- removed, age-gated, unavailable, etc.

    Args:
        error_msg: The exception message string.

    Returns:
        One of 'network', 'bot_detection', 'timeout', 'video_specific'.
    """
    if is_network_failure(error_msg):
        return 'network'
    if is_escalation_error(error_msg):
        return 'bot_detection'
    lower = error_msg.lower()
    if any(p in lower for p in ('timeout', 'timed out', 'stalled')):
        return 'timeout'
    return 'video_specific'


def classify_error_severity(error_message: str) -> str:
    """Classify error message severity for adaptive backoff.

    Examines the error message for known patterns and returns the
    severity level that should determine the backoff multiplier.

    Args:
        error_message: Error string from yt-dlp or YouTube

    Returns:
        Severity level: 'low', 'medium', or 'high'
        Defaults to 'medium' if no pattern matches.
    """
    error_lower = error_message.lower()

    # Check high severity first (most impactful)
    for pattern in ERROR_SEVERITY_PATTERNS['high']:
        if pattern in error_lower:
            return 'high'

    # Check medium severity (standard rate limits)
    for pattern in ERROR_SEVERITY_PATTERNS['medium']:
        if pattern in error_lower:
            return 'medium'

    # Check low severity (minor issues)
    for pattern in ERROR_SEVERITY_PATTERNS['low']:
        if pattern in error_lower:
            return 'low'

    # Default to medium if no pattern matches
    return 'medium'
