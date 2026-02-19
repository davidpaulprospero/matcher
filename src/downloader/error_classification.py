"""
Unified error classification for download errors.

US-52-006: Extracted from download_segments.py and core.py to provide a single
source of truth for error classification across the pipeline.

US-57-008: Error patterns restructured into categorized dict (ERROR_PATTERNS)
with sub-categories: dns, tcp, tls, http, ffmpeg. Enables diagnostic reporting
like 'DNS errors: 3, HTTP errors: 12'. Flat tuple NETWORK_FAILURE_PATTERNS
derived for backward compatibility.

US-82-002: classify_error_category() now returns a typed DownloadError instance
instead of a plain string. The .category attribute preserves the string value
for backward compatibility. is_network_failure() and is_escalation_error() use
isinstance() checks when given a DownloadError, avoiding redundant regex re-parsing.

US-89-008: Error patterns are now configurable via config.yaml download.error_patterns.
The module loads patterns from config when available, with fallback to defaults.
Use reload_patterns() to re-read patterns from config without restart.

US-113-006: Added GeoBlockedError, DeviceLimitError, LoginRequiredError with
corresponding error codes E601, E701, E801.

US-143-010: Added PremiumRequiredError (E901) for YouTube Premium required errors.
These are terminal errors - not retryable without Premium subscription.

Error Code Reference (see src/downloader/errors.py for full list):
    E001-E005: Network errors (DNS, connection, TLS, timeout, unreachable)
    E101-E104: Bot detection (403, captcha, sign-in, blocked)
    E201-E202: Rate limit / quota exceeded
    E301-E305: Format errors (unavailable, missing, removed, private, age-restricted)
    E401-E403: Auth errors (age-gate, login required, premium required)
    E501-E503: Timeout errors
    E601-E602: Geo-blocking errors
    E701: Device limit exceeded
    E801: Login required
    E901: Premium required (US-143-010)
    E999: Unknown error

This module provides:
- classify_error_category(): Returns a DownloadError subclass instance with
  .category, .severity, .retryable, .original_message fields.
- is_network_failure(): Check if an error indicates systemic network failure.
- classify_network_subcategory(): Get the error sub-category (dns/tcp/tls/http/ffmpeg).
- is_escalation_error(): Check if an error warrants bypass tier escalation.
- classify_error_severity(): Classify error severity for adaptive backoff.
- ERROR_PATTERNS: Categorized dict of error patterns by sub-category.
- NETWORK_ERROR_PATTERNS: Alias for ERROR_PATTERNS (backward compat).
- NETWORK_FAILURE_PATTERNS: Flat tuple (backward compat, derived from above).
- ERROR_SEVERITY_PATTERNS / SEVERITY_MULTIPLIERS: Severity pattern data.
- NETWORK_FAILURE_THRESHOLD / BOT_DETECTION_ABORT_THRESHOLD: Abort thresholds.
- load_patterns_from_config(): Load patterns from DownloadConfig.
- reload_patterns(): Re-load patterns from current config (runtime update).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..common.error_patterns import (
    AUTH_PATTERNS,
    BOT_DETECTION_PATTERNS,
    CONTENT_ID_PATTERNS,
    EXTRACTOR_ERROR_PATTERNS,
    EXTENDED_NETWORK_PATTERNS,
    HIGH_SEVERITY_PATTERNS,
    HTTP_5XX_PATTERNS,
    LOW_SEVERITY_PATTERNS,
    MEDIUM_SEVERITY_PATTERNS,
    PERMISSION_PATTERNS,
    RATE_LIMIT_PATTERNS,
    YTDLP_ERROR_PATTERNS,
    is_unknown_error,
)
from .errors import (
    AuthenticationError,
    BotDetectionError,
    ClassifiedDownloadError,
    DeviceLimitError,
    FormatError,
    GeoBlockedError,
    LoginRequiredError,
    NetworkError,
    PremiumRequiredError,
    RateLimitError,
    TimeoutError_,
    UnknownError,
)

logger = logging.getLogger(__name__)

# Module-level storage for config-loaded patterns (US-89-008)
_config_patterns: dict[str, list[str]] | None = None
_config_source: str = "default"  # Track where patterns came from for debugging


def _get_error_patterns() -> dict[str, list[str]]:
    """Get the current error patterns, using config if available.

    Returns:
        The config-loaded patterns if available, otherwise the default patterns.
    """
    global _config_patterns
    if _config_patterns is not None:
        return _config_patterns
    return _DEFAULT_ERROR_PATTERNS


def load_patterns_from_config(download_config) -> None:
    """Load error patterns from a DownloadConfig instance (US-89-008).

    If download_config.error_patterns is provided, uses those patterns.
    Otherwise, uses the default hardcoded patterns.

    Also updates module-level ERROR_PATTERNS and NETWORK_ERROR_PATTERNS for
    backward compatibility with external code that references them directly.

    Args:
        download_config: A DownloadConfig instance with optional error_patterns field.
    """
    global _config_patterns, _config_source, ERROR_PATTERNS, NETWORK_ERROR_PATTERNS

    # Check if config has error_patterns and it's not None
    error_patterns_attr = getattr(download_config, 'error_patterns', None)
    if error_patterns_attr is not None:
        # Convert to dict format
        if hasattr(error_patterns_attr, 'to_dict'):
            _config_patterns = error_patterns_attr.to_dict()
        elif isinstance(error_patterns_attr, dict):
            _config_patterns = error_patterns_attr
        else:
            logger.warning(
                "error_patterns is not a dict or ErrorPatternsConfig, using defaults"
            )
            _config_patterns = None
            _config_source = "default"
            # Update module-level for backward compat
            ERROR_PATTERNS = _DEFAULT_ERROR_PATTERNS
            NETWORK_ERROR_PATTERNS = ERROR_PATTERNS
            return

        _config_source = "config"
        # Update module-level aliases for backward compatibility
        ERROR_PATTERNS = _config_patterns
        NETWORK_ERROR_PATTERNS = _config_patterns
        logger.info("Loaded error patterns from config: %s", list(_config_patterns.keys()))
    else:
        _config_patterns = None
        _config_source = "default"
        # Update module-level for backward compat
        ERROR_PATTERNS = _DEFAULT_ERROR_PATTERNS
        NETWORK_ERROR_PATTERNS = ERROR_PATTERNS
        logger.debug("No error_patterns in config, using defaults")


def reload_patterns(download_config = None) -> None:
    """Re-load error patterns from config (US-89-008).

    Use this to refresh patterns at runtime without restarting the application.
    If download_config is provided, reloads from it.
    Otherwise, re-attempts to load from the global config.

    Args:
        download_config: Optional DownloadConfig instance. If not provided,
                        tries to get from config system.
    """
    global _config_patterns, _config_source

    if download_config is not None:
        load_patterns_from_config(download_config)
        return

    # Try to get config from the config module
    try:
        from ..config import get_config
        config = get_config()
        if config and hasattr(config, 'download'):
            load_patterns_from_config(config.download)
            return
    except Exception as e:
        logger.debug("Could not reload from config: %s", e)

    # If we get here, couldn't reload - keep existing patterns
    logger.warning("Could not reload patterns, keeping existing: %s", _config_source)


# Categorized error patterns for download failures.
# These are the DEFAULT patterns used when no config is provided.
# These match both subprocess stderr AND Python API DownloadError messages,
# which wrap exceptions as "ERROR: [youtube] ID: <original exception text>".
#
# Sub-categories enable diagnostic reporting like 'DNS errors: 3, HTTP errors: 12'.
_DEFAULT_ERROR_PATTERNS: dict[str, list[str]] = {
    'dns': [
        'getaddrinfo failed',
        'Name or service not known',
        'Errno 11001',               # Windows DNS resolution failure
        'nodename nor servname',      # macOS DNS failure
        'No address associated with hostname',
        'Temporary failure in name resolution',
        'Failed to resolve',          # curl/curl_cffi DNS failure message
    ],
    'tcp': [
        'Network is unreachable',
        'ConnectionResetError',       # Python API: connection dropped mid-transfer
        'Connection refused',         # Server rejecting connections (systemic when widespread)
        'Connection timed out',       # TCP connection timeout (systemic when widespread)
        # US-136-002: Extended network patterns
        'Connection aborted',
        'Connection reset by peer',
        'Broken pipe',
        'Connection closed',
    ],
    'tls': [
        # TLS-specific patterns (certificate errors, handshake failures, SSL errors)
        # Common production patterns from yt-dlp, curl_cffi, and urllib3
        'SSL: CERTIFICATE_VERIFY_FAILED',  # Certificate validation failure
        'SSL: WRONG_VERSION_NUMBER',      # TLS version mismatch
        'SSLError',                        # Python ssl module errors
        'SSLHandshakeError',               # TLS handshake failure
        'ssl_',                            # Generic ssl_ prefix (ssl.SSLError, etc.)
        'OpenSSL.SSL.Error',              # PyOpenSSL errors
        'certificate verify failed',       # Certificate validation message
        'EOF occurred in violation of protocol',  # TLS protocol error
        'no protocols available',         # No common TLS protocol
        'sslv3 alert handshake failure',  # TLS handshake alert
        'tlsv1 alert',                    # TLS version-specific alerts
        'unsupported protocol',           # Protocol not supported
        'bad_certificate',                 # Malformed certificate
        'certificate expired',             # Expired certificate
        'certificate has expired',        # Expired certificate variation
        'certificate not yet valid',      # Not yet valid certificate
        'hostname mismatch',               # Certificate hostname mismatch
        'SNI not enabled',                 # Server Name Indication not supported
        'unsafe legacy renegotiation',    # Legacy TLS renegotiation disabled
    ],
    'http': [
        'URLError',                   # Python urllib wrapper (e.g. URLError: <urlopen error ...>)
        'HTTP Error 403',             # Forbidden (bot detection / access block)
        'HTTP Error 429',             # Too Many Requests (rate limiting)
        'HTTP Error 5',               # Server errors (500, 502, 503, etc.)
        # US-136-002: Extended HTTP error codes
        'HTTP Error 500',             # Internal Server Error
        'HTTP Error 501',             # Not Implemented
        'HTTP Error 502',             # Bad Gateway
        'HTTP Error 503',             # Service Unavailable
        'HTTP Error 504',             # Gateway Timeout
    ],
    # US-136-002: New subcategory for extractor errors
    'extractor': [
        'extractor error',
        'unable to extract',
        'could not extract',
        'no suitable extractor',
        'no extractor found',
        'video not found',
        'playlist not found',
        'channel not found',
        'no entries found',
        'no results found',
    ],
    # US-136-002: New subcategory for yt-dlp specific errors
    'ytdlp': [
        'postprocessing error',
        'download error',
        'encoding error',
        'fragment error',
        'captions not found',
        'no subtitles',
        'metadata not found',
        'thumbnail not found',
    ],
    'ffmpeg': [
        # ffmpeg exit code 0xFFFFFEC6 = 4294967158 unsigned = -314 signed (network error)
        '4294967158',
        '-314',
    ],
}

# Backward-compatible alias - points to default patterns (use _get_error_patterns() for runtime)
# This is kept for external code that might reference it directly.
ERROR_PATTERNS = _DEFAULT_ERROR_PATTERNS

# Backward-compatible alias
NETWORK_ERROR_PATTERNS = ERROR_PATTERNS

# Categories that indicate systemic network failures (not video-specific).
# 'http' is excluded because HTTP status errors (403, 429, 5xx) are handled
# by is_escalation_error, not is_network_failure. URLError is a transport-level
# wrapper that indicates connectivity issues, so it remains in the flat tuple.
_NETWORK_FAILURE_CATEGORIES = ('dns', 'tcp', 'tls')


def _get_network_failure_patterns() -> tuple[str, ...]:
    """Get the current network failure patterns as a flat tuple.

    Returns:
        Flat tuple of all network failure patterns (dns + tcp + tls + URLError).
    """
    patterns = _get_error_patterns()
    return tuple(
        pattern
        for category, pattern_list in patterns.items()
        if category in _NETWORK_FAILURE_CATEGORIES
        for pattern in pattern_list
    ) + ('URLError',)


def _get_ffmpeg_exit_codes() -> tuple[str, ...]:
    """Get the current FFmpeg network exit codes.

    Returns:
        Tuple of FFmpeg exit codes that indicate network errors.
    """
    patterns = _get_error_patterns()
    return tuple(patterns.get('ffmpeg', []))


# Backward-compatible flat tuple - computed from defaults at module load.
# Use _get_network_failure_patterns() for runtime-updated patterns.
NETWORK_FAILURE_PATTERNS: tuple[str, ...] = tuple(
    pattern
    for category, patterns in _DEFAULT_ERROR_PATTERNS.items()
    if category in _NETWORK_FAILURE_CATEGORIES
    for pattern in patterns
) + ('URLError',)

# ffmpeg network exit codes - computed from defaults at module load.
# Use _get_ffmpeg_exit_codes() for runtime-updated patterns.
FFMPEG_NETWORK_EXIT_CODES: tuple[str, ...] = tuple(_DEFAULT_ERROR_PATTERNS['ffmpeg'])

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
#
# US-67-009: High-severity bot detection patterns and medium-severity rate limit
# patterns are sourced from src/common/error_patterns.py (shared with caption system).
# US-82-012: Base patterns imported from common module; downloader-specific patterns appended.
# US-89-004: TLS-specific severity patterns added for certificate/handshake errors.
# US-113-006: Added geo-blocking and device limit patterns for new error types.
ERROR_SEVERITY_PATTERNS = {
    # High severity: quota exceeded, bot detection, severe blocks, geo-blocking
    'high': HIGH_SEVERITY_PATTERNS + [
        # US-113-006: Geo-blocking requires VPN (Tier 4) - high severity
        'geo block',
        'geo-restricted',
        'not available in your country',
        'not available in your region',
        # US-143-010: Premium required - not retryable without Premium
        'premium required',
        'premium only',
        'members only',
        'youtube premium',
    ],
    # Medium severity: standard rate limits + downloader-specific patterns
    # US-89-004: TLS errors are generally transient network issues (medium severity)
    # US-113-006: Device limit errors are medium severity
    'medium': MEDIUM_SEVERITY_PATTERNS + [
        'please try again later',
        'temporarily unavailable',
        # TLS-specific: certificate/handshake/version errors (transient network issues)
        'sslerror',
        'ssl handshake',
        'certificate verify failed',
        'wrong version number',
        'eof occurred in violation',
        'unsupported protocol',
        'tlsv1 alert',
        'handshake failure',
        # US-113-006: Device limit errors - medium severity (wait and retry)
        'device limit',
        'too many devices',
        'playback on other',
    ],
    # Low severity: auth patterns + downloader-specific patterns
    # US-113-006: Login required is low severity
    'low': LOW_SEVERITY_PATTERNS + [
        'slow down',
        # US-113-006: Login required - low severity (requires auth, not blocking)
        'login required',
        'sign in to watch',
    ],
}

# Multipliers for each severity level
SEVERITY_MULTIPLIERS = {
    'low': 1.5,
    'medium': 2.0,
    'high': 3.0,
}


def classify_network_subcategory(error_msg: str) -> str | None:
    """Classify an error into its sub-category from error patterns.

    Checks the error message against current patterns (config or defaults) and returns
    the sub-category (e.g. 'dns', 'tcp', 'tls', 'http', 'ffmpeg').

    Args:
        error_msg: The exception message string.

    Returns:
        Sub-category string if a pattern matches, None otherwise.
    """
    error_lower = error_msg.lower()
    patterns = _get_error_patterns()
    for subcategory, pattern_list in patterns.items():
        if subcategory == 'ffmpeg':
            # ffmpeg codes are matched case-sensitively as exact substrings
            for pattern in pattern_list:
                if pattern in error_msg:
                    return 'ffmpeg'
        else:
            for pattern in pattern_list:
                if pattern.lower() in error_lower:
                    return subcategory
    return None


def is_network_failure(error_msg: str | ClassifiedDownloadError) -> bool:
    """Check if an error message indicates a systemic network failure.

    These are failures that affect ALL downloads (DNS down, no internet),
    as opposed to video-specific errors (403, removed, age-gated).
    Only dns, tcp, tls, ffmpeg, and URLError patterns qualify as network failures.

    US-82-002: When passed a DownloadError instance, uses isinstance()
    instead of re-parsing patterns — O(1) vs O(n) pattern matching.

    Args:
        error_msg: The exception message string or a DownloadError instance.

    Returns:
        True if the error indicates a systemic network issue.
    """
    if isinstance(error_msg, ClassifiedDownloadError):
        return isinstance(error_msg, NetworkError)
    error_lower = error_msg.lower()
    # Use dynamic patterns that can be updated from config
    for pattern in _get_network_failure_patterns():
        if pattern.lower() in error_lower:
            return True
    # Check for ffmpeg network exit codes (unsigned 4294967158 or signed -314)
    for code in _get_ffmpeg_exit_codes():
        if code in error_msg:
            return True
    return False


def is_escalation_error(error_msg: str | ClassifiedDownloadError) -> bool:
    """Check if an error message indicates a 403/bot-detection/auth error.

    These errors warrant escalation to a higher bypass tier via the
    EscalationManager (Tier 2 extractor_args, Tier 3 cookies).

    US-82-002: When passed a DownloadError instance, uses isinstance()
    instead of re-parsing patterns.

    US-113-006: Now also detects GeoBlockedError (needs VPN - Tier 4) and
    LoginRequiredError (needs auth - Tier 3).

    Args:
        error_msg: The exception message string or a DownloadError instance.

    Returns:
        True if the error matches 403/bot/auth/geo-block/login patterns.
    """
    if isinstance(error_msg, ClassifiedDownloadError):
        return isinstance(error_msg, (
            BotDetectionError, AuthenticationError, GeoBlockedError, LoginRequiredError
        ))
    try:
        from .escalation_manager import is_escalation_trigger
        return is_escalation_trigger(error_msg)
    except ImportError:
        # Fallback: simple pattern match if escalation_manager unavailable
        lower = error_msg.lower()
        return any(p in lower for p in (
            '403', 'forbidden', 'sign in', 'bot', 'captcha',
            'geo block', 'geo-restricted', 'login required'
        ))


def parse_retry_after(error_msg: str) -> Optional[float]:
    """Parse Retry-After header value from error message.

    US-114-010: Extracts the Retry-After duration from error messages.
    Supports formats:
    - "Retry-After: 120" (seconds)
    - "retry-after: 120" (case insensitive)
    - "X-Retry-After: 120" (some CDNs use X- prefix)
    - "retry after: 120 seconds"

    Also extracts from yt-dlp error format:
    - "HTTP Error 429: Too Many Requests. Retry-After: 120"
    - "ERROR: [youtube] 429: ..."

    Args:
        error_msg: The error message string potentially containing Retry-After.

    Returns:
        Seconds to wait as float, or None if not found.
    """
    error_lower = error_msg.lower()

    # Pattern 1: "Retry-After: 120" or "retry-after: 120" (header format)
    match = re.search(r'retry[- ]after[:\s]+(\d+(?:\.\d+)?)', error_lower)
    if match:
        return float(match.group(1))

    # Pattern 2: "X-Retry-After: 120" (CDN header format)
    match = re.search(r'x[-_]retry[-_]after[:\s]+(\d+(?:\.\d+)?)', error_lower)
    if match:
        return float(match.group(1))

    # Pattern 3: "wait 120 seconds" or "wait 120s"
    match = re.search(r'wait[:\s]+(\d+(?:\.\d+)?)\s*(?:seconds?|s(?:ec)?)?', error_lower)
    if match:
        return float(match.group(1))

    # Pattern 4: "retry in 120 seconds" or "retry in 120s"
    match = re.search(r'retry\s+in\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s(?:ec)?)?', error_lower)
    if match:
        return float(match.group(1))

    return None


def classify_error_category(error_msg: str) -> ClassifiedDownloadError:
    """Classify a download error into a typed DownloadError subclass.

    US-82-002: Returns a DownloadError instance instead of a plain string.
    The .category attribute preserves the string value for backward
    compatibility (e.g. 'network', 'bot_detection', 'timeout', 'video_specific').

    US-113-006: Added GeoBlockedError, DeviceLimitError, LoginRequiredError.

    US-120-002: Added UnknownError for unclassified errors that don't match
    any known pattern category.

    US-136-002: Added HTTP 5xx, extractor, and yt-dlp error classification
    to reduce Unknown error rate.

    Categories (most specific first):
        NetworkError       ('network')        - DNS failure, no connectivity
        GeoBlockedError    ('geo_blocked')   - Geographic blocking (VPN required)
        DeviceLimitError   ('device_limit')  - Too many devices streaming
        LoginRequiredError ('login_required') - Login/authentication required
        PremiumRequiredError ('premium_required') - YouTube Premium required (US-143-010)
        BotDetectionError  ('bot_detection') - 403/bot/captcha/sign-in errors
        TimeoutError_      ('timeout')       - stall timeouts, socket timeouts
        UnknownError       ('unknown')       - Unclassified errors (US-120-002)
        FormatError        ('video_specific')- removed, age-gated, unavailable

    Args:
        error_msg: The exception message string.

    Returns:
        A DownloadError subclass instance with .category, .severity,
        .retryable, and .original_message attributes.
    """
    severity = classify_error_severity(error_msg)
    if is_network_failure(error_msg):
        error = NetworkError(error_msg, severity=severity)
        _log_error_classification(error)
        return error
    # US-114-010: Check for HTTP 429 rate limit errors (before other escalation errors)
    # This allows extracting Retry-After header value when present
    lower = error_msg.lower()
    if '429' in lower or 'too many requests' in lower or 'rate limit' in lower:
        retry_after = parse_retry_after(error_msg)
        error = RateLimitError(error_msg, retry_after=retry_after, severity=severity)
        _log_error_classification(error)
        return error
    # US-136-002: Check for HTTP 5xx server errors
    # Check for numeric codes and error patterns
    if any(p in lower for p in (
        '500', '501', '502', '503', '504', '505',
        'internal server error', 'not implemented', 'bad gateway',
        'service unavailable', 'service temporarily unavailable',
        'gateway timeout', 'http version not supported',
        '520', '521', '522', '523', '524'
    )):
        error = NetworkError(error_msg, severity='high')
        _log_error_classification(error)
        return error
    # US-113-006: Check for geo-blocking first (VPN tier escalation)
    lower = error_msg.lower()
    if any(p in lower for p in (
        'geo block', 'geo-restricted', 'not available in your country',
        'not available in your region', 'this content is not available'
    )):
        error = GeoBlockedError(error_msg, severity=severity)
        _log_error_classification(error)
        return error
    # US-113-006: Check for device limit errors
    if any(p in lower for p in (
        'device limit', 'too many devices', 'playback on other', 'exceeded the limit'
    )):
        error = DeviceLimitError(error_msg, severity=severity)
        _log_error_classification(error)
        return error
    # US-113-006: Check for login required errors (before generic bot detection)
    if any(p in lower for p in ('login required', 'sign in to watch')):
        error = LoginRequiredError(error_msg, severity=severity)
        _log_error_classification(error)
        return error
    # US-143-010: Check for premium required errors (YouTube Premium subscription needed)
    if any(p in lower for p in (
        'premium required', 'premium only', 'members only',
        'youtube premium', 'premium subscription'
    )):
        error = PremiumRequiredError(error_msg, severity=severity)
        _log_error_classification(error)
        return error
    if is_escalation_error(error_msg):
        error = BotDetectionError(error_msg, severity=severity)
        _log_error_classification(error)
        return error
    if any(p in lower for p in ('timeout', 'timed out', 'stalled')):
        error = TimeoutError_(error_msg, severity=severity)
        _log_error_classification(error)
        return error
    # US-120-002: Check for unavailable/removed/private patterns before Unknown
    if any(p in lower for p in ('not available', 'unavailable', 'removed', 'deleted', 'private')):
        error = FormatError(error_msg, severity=severity)
        _log_error_classification(error)
        return error
    # US-136-002: Check for extractor errors
    if any(p in lower for p in (
        'extractor error', 'unable to extract', 'could not extract',
        'no suitable extractor', 'no extractor found', 'video not found',
        'playlist not found', 'channel not found', 'no entries found',
        'no results found'
    )):
        error = FormatError(error_msg, severity=severity)
        _log_error_classification(error)
        return error
    # US-136-002: Check for yt-dlp specific errors
    if any(p in lower for p in (
        'postprocessing error', 'download error', 'encoding error',
        'fragment error', 'captions not found', 'no subtitles',
        'metadata not found', 'thumbnail not found'
    )):
        error = FormatError(error_msg, severity=severity)
        _log_error_classification(error)
        return error
    # US-120-002: Check if this is an unclassified error
    if is_unknown_error(error_msg):
        # Import here to avoid circular import
        from ..common.error_patterns import get_unknown_error_handler
        handler = get_unknown_error_handler()
        error_record = handler.handle_unknown_error(error_msg)
        error = UnknownError(error_msg, severity=severity, stack_trace=error_record.get('stack_trace'))
        _log_error_classification(error)
        return error
    error = FormatError(error_msg, severity=severity)
    _log_error_classification(error)
    return error


def _log_error_classification(error: ClassifiedDownloadError) -> None:
    """Log error classification with original error and category.

    US-129-005: Logs the classified error with full context for debugging.

    Args:
        error: The classified download error instance
    """
    logger.debug(
        "Error classified: category=%s, severity=%s, retryable=%s",
        error.category,
        error.severity,
        error.retryable
    )
    # Log the original error message at INFO level for significant errors
    if error.severity == 'high':
        logger.info(
            "High severity error classified: category=%s, message=%s",
            error.category,
            error.original_message[:150] if error.original_message else ""
        )


def classify_error_severity(error_message: str) -> str:
    """Classify error message severity for adaptive backoff.

    Examines the error message for known patterns and returns the
    severity level that should determine the backoff multiplier.
    Also logs the network error sub-category when applicable.

    Args:
        error_message: Error string from yt-dlp or YouTube

    Returns:
        Severity level: 'low', 'medium', or 'high'
        Defaults to 'medium' if no pattern matches.
    """
    error_lower = error_message.lower()

    # Log network error sub-category for diagnostics
    subcategory = classify_network_subcategory(error_message)
    if subcategory:
        logger.debug("Network error sub-category: %s", subcategory)

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


# =============================================================================
# Adaptive Error Severity Tracking (US-89-012)
# =============================================================================

from collections import deque, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSeverityTracker:
    """Tracks error frequency per category over a sliding window.

    US-89-012: Tracks consecutive errors by category and auto-escalates
    severity when the same error repeats. This improves backoff timing
    for recurring errors without manual tuning.

    Example:
        tracker = AdaptiveSeverityTracker(window_size=10, escalation_threshold=3)
        tracker.register_error("rate_limit")  # count=1
        tracker.register_error("rate_limit")  # count=2
        tracker.register_error("rate_limit")  # count=3 -> escalates
        severity = tracker.get_adjusted_severity("medium", "rate_limit")
        # Returns "high" after 3 consecutive rate_limit errors
    """
    # Configuration
    window_size: int = 10  # Maximum errors to track in sliding window
    escalation_threshold: int = 3  # Errors before escalating severity
    max_severity: str = "high"  # Maximum severity level to escalate to
    reset_on_success: bool = True  # Reset count on success

    # Error counts per category (maps category -> deque of timestamps)
    _error_history: dict[str, Deque[datetime]] = field(default_factory=dict)

    def _get_category_key(self, category: str) -> str:
        """Normalize category name for tracking.

        Maps category names to tracked keys.
        """
        # Normalize category names
        category_map = {
            "rate_limit": "rate_limit",
            "rate limit": "rate_limit",
            "429": "rate_limit",
            "bot_detection": "bot_detection",
            "bot detection": "bot_detection",
            "403": "bot_detection",
            "network": "network",
            "timeout": "timeout",
            "video_specific": "video_specific",
        }
        return category_map.get(category.lower(), category.lower())

    def register_error(self, category: str) -> None:
        """Register an error occurrence for the given category.

        Args:
            category: The error category (e.g., 'rate_limit', 'bot_detection')
        """
        key = self._get_category_key(category)
        now = datetime.now()

        if key not in self._error_history:
            self._error_history[key] = deque(maxlen=self.window_size)

        self._error_history[key].append(now)
        logger.debug(
            "AdaptiveSeverity: Registered error for '%s' (count=%d)",
            key,
            len(self._error_history[key])
        )

    def register_success(self) -> None:
        """Reset error counts on successful operation.

        Called when a download succeeds to reset the error counters.
        """
        if self.reset_on_success:
            self._error_history.clear()
            logger.debug("AdaptiveSeverity: Reset error counts on success")

    def get_consecutive_count(self, category: str) -> int:
        """Get the number of recent consecutive errors for a category.

        Only counts errors within the sliding window that are close together
        (within the window_size timeframe).

        Args:
            category: The error category

        Returns:
            Number of consecutive errors in the window
        """
        key = self._get_category_key(category)
        if key not in self._error_history:
            return 0

        history = self._error_history[key]
        if not history:
            return 0

        # Count errors in the sliding window
        # For simplicity, we just return the count of errors in the deque
        return len(history)

    def get_adjusted_severity(self, base_severity: str, category: str) -> str:
        """Get severity adjusted based on error frequency.

        If the same error has occurred repeatedly (more than escalation_threshold),
        escalates the severity to improve backoff timing.

        Args:
            base_severity: The base severity from pattern matching
            category: The error category

        Returns:
            Adjusted severity: 'low', 'medium', or 'high'
        """
        if not self._is_tracked_category(category):
            return base_severity

        count = self.get_consecutive_count(category)

        # Check if we've hit the escalation threshold
        if count >= self.escalation_threshold:
            # Only escalate if we're below max_severity
            severity_order = ['low', 'medium', 'high']
            base_index = severity_order.index(base_severity) if base_severity in severity_order else 1
            max_index = severity_order.index(self.max_severity) if self.max_severity in severity_order else 2

            if base_index < max_index:
                escalated = severity_order[max_index]
                logger.info(
                    "AdaptiveSeverity: Escalating severity from '%s' to '%s' "
                    "after %d consecutive '%s' errors",
                    base_severity, escalated, count, category
                )
                return escalated

        return base_severity

    def _is_tracked_category(self, category: str) -> bool:
        """Check if a category is tracked for adaptive severity.

        Args:
            category: The error category

        Returns:
            True if the category is tracked
        """
        # Track rate_limit, bot_detection, network, timeout by default
        tracked = {"rate_limit", "bot_detection", "network", "timeout"}
        key = self._get_category_key(category)
        return key in tracked

    def get_stats(self) -> dict:
        """Get current tracking statistics.

        Returns:
            Dict with error counts per category
        """
        return {
            key: len(timestamps)
            for key, timestamps in self._error_history.items()
        }

    def reset(self) -> None:
        """Reset all error tracking."""
        self._error_history.clear()
        logger.debug("AdaptiveSeverity: Reset all tracking")


# Global tracker instance (can be reinitialized with config)
_adaptive_tracker: AdaptiveSeverityTracker | None = None


def get_adaptive_tracker() -> AdaptiveSeverityTracker:
    """Get the global adaptive severity tracker instance.

    Returns:
        The global AdaptiveSeverityTracker instance
    """
    global _adaptive_tracker
    if _adaptive_tracker is None:
        _adaptive_tracker = AdaptiveSeverityTracker()
    return _adaptive_tracker


def init_adaptive_tracker(
    enabled: bool = True,
    escalation_threshold: int = 3,
    max_severity: str = "high",
    window_size: int = 10,
    reset_on_success: bool = True,
) -> None:
    """Initialize or update the global adaptive severity tracker.

    Args:
        enabled: Whether adaptive severity is enabled
        escalation_threshold: Errors before escalating severity
        max_severity: Maximum severity level ('medium' or 'high')
        window_size: Sliding window size for error tracking
        reset_on_success: Reset counts on successful download
    """
    global _adaptive_tracker
    if enabled:
        _adaptive_tracker = AdaptiveSeverityTracker(
            window_size=window_size,
            escalation_threshold=escalation_threshold,
            max_severity=max_severity,
            reset_on_success=reset_on_success,
        )
        logger.info(
            "AdaptiveSeverity: Initialized with threshold=%d, max_severity=%s, window_size=%d",
            escalation_threshold, max_severity, window_size
        )
    else:
        _adaptive_tracker = None
        logger.info("AdaptiveSeverity: Disabled")


def register_error(category: str) -> None:
    """Register an error for adaptive severity tracking.

    Args:
        category: The error category (e.g., 'rate_limit', '429')
    """
    tracker = get_adaptive_tracker()
    tracker.register_error(category)


def register_success() -> None:
    """Register a success for adaptive severity tracking.

    Resets error counts if reset_on_success is enabled.
    """
    tracker = get_adaptive_tracker()
    tracker.register_success()


def get_adjusted_severity(base_severity: str, category: str) -> str:
    """Get severity adjusted for repeated errors.

    If the same error has occurred repeatedly, returns an escalated severity
    to improve backoff timing.

    Args:
        base_severity: The base severity from pattern matching
        category: The error category

    Returns:
        Adjusted severity: 'low', 'medium', or 'high'
    """
    tracker = get_adaptive_tracker()
    return tracker.get_adjusted_severity(base_severity, category)


def classify_error_severity_with_adaptive(
    error_message: str,
    category: str | None = None,
) -> str:
    """Classify error severity with adaptive adjustment.

    Combines pattern-based severity classification with adaptive tracking
    for repeated errors. This is the recommended function to use instead
    of classify_error_severity when adaptive severity is enabled.

    Args:
        error_message: Error string from yt-dlp or YouTube
        category: Optional category for adaptive tracking. If not provided,
                 will be derived from error_message.

    Returns:
        Severity level: 'low', 'medium', or 'high' (possibly escalated)
    """
    # Get base severity from patterns
    base_severity = classify_error_severity(error_message)

    # If adaptive tracking is disabled or no category, return base
    tracker = get_adaptive_tracker()
    if tracker is None:
        return base_severity

    # Determine category for tracking
    if category is None:
        # Derive category from error message
        classified = classify_error_category(error_message)
        category = classified.category

    # Get adjusted severity based on error frequency
    return tracker.get_adjusted_severity(base_severity, category)


# =============================================================================
# Error Metrics Tracking (US-129-005)
# =============================================================================


class ErrorMetricsTracker:
    """Tracks error counts per category for metrics and diagnostics.

    US-129-005: Tracks error counts by category to provide metrics on
    which error types are occurring most frequently. This enables:
    - Error rate monitoring per category
    - Identifying trending error patterns
    - Generating error distribution reports

    Example:
        tracker = ErrorMetricsTracker()
        tracker.record_error("geo_blocked")
        tracker.record_error("rate_limit")
        tracker.record_error("geo_blocked")

        counts = tracker.get_counts()
        # Returns {'geo_blocked': 2, 'rate_limit': 1, ...}

        summary = tracker.get_summary()
        # Returns {'total': 3, 'by_category': {...}, 'most_common': ...}
    """

    def __init__(self):
        """Initialize the error metrics tracker."""
        self._counts: Counter[str] = Counter()
        self._first_seen: dict[str, datetime] = {}
        self._last_seen: dict[str, datetime] = {}

    def record_error(self, category: str) -> None:
        """Record an error occurrence for the given category.

        Args:
            category: The error category (e.g., 'geo_blocked', 'rate_limit')
        """
        now = datetime.now()

        # Track first and last seen timestamps
        if category not in self._first_seen:
            self._first_seen[category] = now
        self._last_seen[category] = now

        # Increment count
        self._counts[category] += 1

        # Log the classification
        logger.debug(
            "Error classified: category=%s, total_count=%d",
            category,
            self._counts[category]
        )

    def record_classified_error(self, error: ClassifiedDownloadError) -> None:
        """Record an error from a ClassifiedDownloadError instance.

        This is the preferred method for recording errors as it captures
        both the category and the original message for logging.

        Args:
            error: The classified download error instance
        """
        # Record by category
        self.record_error(error.category)

        # Log with original error message
        logger.info(
            "Error classified: category=%s, severity=%s, retryable=%s, message=%s",
            error.category,
            error.severity,
            error.retryable,
            error.original_message[:100] if error.original_message else ""
        )

    def get_counts(self) -> dict[str, int]:
        """Get error counts per category.

        Returns:
            Dict mapping category names to error counts
        """
        return dict(self._counts)

    def get_category_count(self, category: str) -> int:
        """Get the error count for a specific category.

        Args:
            category: The error category

        Returns:
            Number of errors for that category
        """
        return self._counts.get(category, 0)

    def get_summary(self) -> dict:
        """Get a summary of error metrics.

        Returns:
            Dict with total errors, counts by category, and most common category
        """
        if not self._counts:
            return {
                "total": 0,
                "by_category": {},
                "most_common": None,
                "first_seen": None,
                "last_seen": None,
            }

        most_common = self._counts.most_common(1)
        most_common_category = most_common[0][0] if most_common else None

        return {
            "total": sum(self._counts.values()),
            "by_category": dict(self._counts),
            "most_common": most_common_category,
            "most_common_count": most_common[0][1] if most_common else 0,
            "first_seen": min(self._first_seen.values()) if self._first_seen else None,
            "last_seen": max(self._last_seen.values()) if self._last_seen else None,
        }

    def get_percentages(self) -> dict[str, float]:
        """Get error counts as percentages of total.

        Returns:
            Dict mapping category names to percentage of total (0-100)
        """
        total = sum(self._counts.values())
        if total == 0:
            return {}

        return {
            category: (count / total) * 100
            for category, count in self._counts.items()
        }

    def reset(self) -> None:
        """Reset all error counts."""
        self._counts.clear()
        self._first_seen.clear()
        self._last_seen.clear()
        logger.debug("ErrorMetricsTracker: Reset all counts")


# Global metrics tracker instance
_error_metrics: ErrorMetricsTracker | None = None


def get_error_metrics() -> ErrorMetricsTracker:
    """Get the global error metrics tracker instance.

    Returns:
        The global ErrorMetricsTracker instance
    """
    global _error_metrics
    if _error_metrics is None:
        _error_metrics = ErrorMetricsTracker()
    return _error_metrics


def init_error_metrics() -> None:
    """Initialize or reset the global error metrics tracker."""
    global _error_metrics
    _error_metrics = ErrorMetricsTracker()
    logger.info("ErrorMetricsTracker: Initialized")


def record_error(category: str) -> None:
    """Record an error occurrence for metrics tracking.

    Args:
        category: The error category (e.g., 'geo_blocked', 'rate_limit')
    """
    tracker = get_error_metrics()
    tracker.record_error(category)


def record_classified_error(error: ClassifiedDownloadError) -> None:
    """Record a classified error for metrics tracking.

    This also logs the classification with original error message.

    Args:
        error: The classified download error instance
    """
    tracker = get_error_metrics()
    tracker.record_classified_error(error)


def get_error_counts() -> dict[str, int]:
    """Get error counts per category.

    Returns:
        Dict mapping category names to error counts
    """
    tracker = get_error_metrics()
    return tracker.get_counts()


def get_error_summary() -> dict:
    """Get a summary of error metrics.

    Returns:
        Dict with total errors, counts by category, and most common category
    """
    tracker = get_error_metrics()
    return tracker.get_summary()
