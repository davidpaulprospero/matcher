"""
Stage-level error aggregation with categorized failure summaries.

US-51-011: Provides an ErrorCategory enum and ErrorAggregator class that
pipeline stages can use to collect, categorize, and summarize errors during
execution. The completion summary logs a table of category -> count -> sample.

US-106-009: Enhanced with error category hierarchy (network.timeout -> network -> all),
pipeline-level error aggregation, grouping similar errors, and trend analysis.

Usage:
    from src.stages.error_aggregator import ErrorCategory, ErrorAggregator

    agg = ErrorAggregator()
    agg.record("HTTP Error 403: Forbidden", ErrorCategory.AUTH)
    agg.record("getaddrinfo failed", ErrorCategory.NETWORK)
    agg.log_summary()  # logs table at INFO level
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categorized error buckets for pipeline stage failures.

    Categories are ordered from most specific/actionable to least:
    - NETWORK: Systemic connectivity issues (DNS, unreachable)
    - AUTH: Bot-detection, 403, captcha, sign-in blocks
    - RATE_LIMIT: Rate limiting, quota exceeded
    - RESOURCE_EXHAUSTION: Memory, disk space, CPU limits
    - CONFIG: Configuration errors, invalid settings
    - TIMEOUT: Stall timeouts, socket timeouts
    - VALIDATION: Input/config validation failures
    - CONTRACT: Stage input/output contract violations (US-108-011)
    - UNKNOWN: Uncategorized errors

    Inherits from str so values serialize cleanly to JSON/checkpoint dicts.
    """
    NETWORK = 'network'
    AUTH = 'auth'
    RATE_LIMIT = 'rate_limit'
    RESOURCE_EXHAUSTION = 'resource_exhaustion'
    CONFIG = 'config'
    TIMEOUT = 'timeout'
    VALIDATION = 'validation'
    CONTRACT = 'contract'  # US-108-011: Stage contract violations
    UNKNOWN = 'unknown'

    @classmethod
    def get_parent(cls, category: str) -> Optional[str]:
        """Get parent category for hierarchical grouping.

        Args:
            category: Category string (may include subcategory like 'network.timeout')

        Returns:
            Parent category string (e.g., 'network' for 'network.timeout'), or None.
        """
        # Handle hierarchical categories like 'network.timeout'
        if '.' in category:
            parts = category.split('.')
            if len(parts) >= 2:
                return parts[0]
        return None

    @classmethod
    def get_all_ancestors(cls, category: str) -> List[str]:
        """Get all ancestor categories in hierarchy.

        Args:
            category: Category string (may include subcategories)

        Returns:
            List of ancestor categories from most specific to 'all'.
        """
        ancestors = []
        current = category
        while True:
            parent = cls.get_parent(current)
            if parent is None:
                break
            ancestors.append(parent)
            current = parent
        ancestors.append('all')
        return ancestors


# Subcategories for more granular error tracking
ERROR_SUBCATEGORIES: Dict[str, Set[str]] = {
    'network': {'timeout', 'dns', 'connection', 'unreachable', 'ssl'},
    'auth': {'bot_detection', '403', 'captcha', 'rate_limit'},
    'rate_limit': {'429', 'quota', 'throttle'},
    'resource_exhaustion': {'memory', 'disk', 'cpu', 'file_handles'},
    'config': {'missing', 'invalid', 'schema'},
    'timeout': {'socket', 'stall', 'read', 'connect'},
    'validation': {'input', 'config', 'schema'},
}


# US-123-011: Error sources for multi-source aggregation
class ErrorSource(str, Enum):
    """Pipeline error sources for unified rate limit analysis.

    Tracks errors from different pipeline stages to identify cross-source
    correlation patterns (e.g., same IP/cookie causing issues across stages).

    Inherits from str so values serialize cleanly to JSON/checkpoint dicts.
    """
    DOWNLOAD = 'download'
    CAPTION = 'caption'
    TRANSCRIPTION = 'transcription'

    @classmethod
    def from_stage_name(cls, stage_name: str) -> 'ErrorSource':
        """Map stage name to error source.

        Args:
            stage_name: Stage name like 'DOWNLOAD_SEGMENTS', 'CAPTION', etc.

        Returns:
            Corresponding ErrorSource enum value.
        """
        stage_lower = stage_name.lower()
        if 'download' in stage_lower:
            return cls.DOWNLOAD
        elif 'caption' in stage_lower:
            return cls.CAPTION
        elif 'transcribe' in stage_lower or 'transcription' in stage_lower:
            return cls.TRANSCRIPTION
        # Default to DOWNLOAD for unknown stages as it's the most common
        return cls.DOWNLOAD


# US-108-006: Suggestion engine for actionable error insights
# Maps error categories to fix recommendations
ERROR_SUGGESTIONS: Dict[str, List[str]] = {
    'network': [
        "Check your internet connection and DNS settings",
        "Try using a different network or VPN if issues persist",
        "Check firewall settings that may block outbound connections",
    ],
    'network.timeout': [
        "Increase timeout values in config.yaml download section",
        "Check network latency to YouTube servers",
        "Consider using a faster/stable connection",
    ],
    'network.dns': [
        "Check DNS resolver configuration",
        "Try using Google's DNS (8.8.8.8) or Cloudflare (1.1.1.1)",
        "DNS issues may be temporary - retry later",
    ],
    'network.ssl': [
        "Check SSL certificate configuration",
        "Update ca-certificates package",
        "SSL errors may indicate MITM - verify your network security",
    ],
    'auth': [
        "YouTube may be blocking requests - try cookie-based authentication",
        "Use --cookies-from-browser to import browser cookies",
        "Consider using a VPN to rotate IP addresses",
    ],
    'auth.403': [
        "HTTP 403 indicates access denied - video may be private or region-locked",
        "Try using --cookies-from-browser to authenticate",
        "Video may have been deleted or made private",
    ],
    'auth.captcha': [
        "YouTube is showing captcha - rate limit your requests",
        "Wait before retrying or use VPN to get new IP",
        "Consider using cookies from a logged-in browser session",
    ],
    'auth.bot_detection': [
        "Bot detection triggered - reduce request frequency",
        "Enable impersonation in config.yaml (already default)",
        "Try using --cookies-from-browser with a YouTube account",
    ],
    'auth.rate_limit': [
        "Rate limit exceeded - wait before retrying",
        "Implement delays between requests in config.yaml",
        "Consider reducing concurrent downloads",
    ],
    'rate_limit': [
        "Too many requests - implement rate limiting",
        "Wait and retry with exponential backoff",
        "Reduce number of concurrent operations",
    ],
    'resource_exhaustion': [
        "System resources exhausted - check disk space and memory",
        "Free up disk space or reduce batch sizes",
        "Close other applications to free up memory",
    ],
    'resource_exhaustion.disk': [
        "Insufficient disk space - free up space in project/output directories",
        "Clean up old downloads in .cache folder",
        "Move project to drive with more free space",
    ],
    'resource_exhaustion.memory': [
        "Out of memory - reduce batch sizes or increase swap",
        "Close other applications using memory",
        "Consider processing fewer videos at once",
    ],
    'config': [
        "Check config.yaml for invalid or missing settings",
        "Run --validate-config to check configuration",
        "Review config.yaml comments for valid values",
    ],
    'config.api_key': [
        "Invalid or missing API key in config.yaml",
        "Verify API keys are set in config.yaml",
        "Check API key hasn't expired or been revoked",
    ],
    'validation': [
        "Input validation failed - check input files and parameters",
        "Verify voiceover file format (SRT, VTT supported)",
        "Check that project directory exists and is accessible",
    ],
    'validation.input': [
        "Invalid input file - verify file format and path",
        "SRT files should be properly formatted UTF-8 text",
        "Check that file paths don't contain special characters",
    ],
    'validation.config': [
        "Config validation failed - check config.yaml syntax",
        "Run --validate-config for detailed error messages",
        "Ensure YAML indentation is correct",
    ],
    'contract': [  # US-108-011
        "Stage input/output contract violated - checkpoint data may be corrupted",
        "Run pipeline with --fresh to start fresh or check checkpoint integrity",
        "Run with --validate-only to see detailed contract violations",
    ],
    'contract.input': [  # US-108-011
        "Stage is missing required input data from checkpoint",
        "Previous stage may have failed - check pipeline logs",
        "Try running with --resume to complete previous stages",
    ],
    'contract.output': [  # US-108-011
        "Stage did not produce expected output in checkpoint",
        "Stage may have failed silently - check pipeline logs",
        "Run pipeline fresh to regenerate all stage outputs",
    ],
    'timeout': [
        "Operation timed out - check network connectivity",
        "Increase timeout values in config.yaml",
        "Try again with --resume if checkpoint exists",
    ],
    'timeout.socket': [
        "Socket timeout - network may be slow or blocked",
        "Increase socket_timeout in config.yaml download section",
        "Check firewall/antivirus that may block connections",
    ],
    'timeout.stall': [
        "Download stalled - connection may be unstable",
        "Increase stall_timeout in config.yaml",
        "Try downloading a single video to diagnose",
    ],
    'unknown': [  # US-120-002
        "Unknown error - check logs for details",
        "Enable debug logging with --verbose for more information",
        "Report issue if persists - include error message and stack trace",
    ],
}


class SuggestionEngine:
    """US-108-006: Provides actionable fix suggestions based on error categories.

    Generates recommendations for fixing errors based on:
    - Error category and subcategory
    - Error frequency (recurring errors get priority suggestions)
    - Available healing strategies

    Usage:
        from src.stages.error_aggregator import SuggestionEngine, ErrorAggregator

        engine = SuggestionEngine()
        agg = ErrorAggregator()
        agg.record("HTTP 403", ErrorCategory.AUTH)
        suggestions = engine.get_suggestions(agg)
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    """

    def __init__(self, enable_healing_integration: bool = True):
        """Initialize suggestion engine.

        Args:
            enable_healing_integration: Include healing strategy suggestions
        """
        self._enable_healing_integration = enable_healing_integration

    def get_suggestions(
        self,
        aggregator: ErrorAggregator,
        min_count: int = 1,
    ) -> List[str]:
        """Get fix suggestions based on recorded errors.

        Args:
            aggregator: ErrorAggregator with recorded errors
            min_count: Minimum error count to include category (default 1)

        Returns:
            List of actionable suggestion strings
        """
        suggestions: List[str] = []
        seen_suggestions: Set[str] = set()

        # Get hierarchical counts
        hierarchical = aggregator.get_hierarchical_counts()

        # Get top repeated patterns for priority suggestions
        top_patterns = aggregator.get_top_similar_errors(limit=3, min_count=2)

        # Add suggestions for repeated patterns first (higher priority)
        for pattern, count, sample in top_patterns:
            pattern_suggestions = self._get_suggestions_for_pattern(pattern, sample)
            for suggestion in pattern_suggestions:
                if suggestion not in seen_suggestions:
                    suggestions.append(f"[Repeated {count}x] {suggestion}")
                    seen_suggestions.add(suggestion)

        # Add suggestions for each category with errors
        for category, count in sorted(hierarchical.items(), key=lambda x: x[1], reverse=True):
            if category == 'all':
                continue
            if count < min_count:
                continue

            # Get suggestions for this category
            cat_suggestions = self._get_suggestions_for_category(category)
            for suggestion in cat_suggestions:
                if suggestion not in seen_suggestions:
                    # Add count context
                    if count > 1:
                        suggestions.append(f"[{count} errors] {suggestion}")
                    else:
                        suggestions.append(suggestion)
                    seen_suggestions.add(suggestion)

        return suggestions

    def _get_suggestions_for_category(self, category: str) -> List[str]:
        """Get suggestions for a specific category.

        Args:
            category: Error category or subcategory (e.g., 'network', 'auth.403')

        Returns:
            List of suggestion strings
        """
        suggestions: List[str] = []

        # Check exact match first
        if category in ERROR_SUGGESTIONS:
            suggestions.extend(ERROR_SUGGESTIONS[category])

        # Check parent category (e.g., 'auth.403' -> 'auth')
        if '.' in category:
            parent = category.split('.')[0]
            if parent in ERROR_SUGGESTIONS:
                for s in ERROR_SUGGESTIONS[parent]:
                    if s not in suggestions:
                        suggestions.append(s)

        # If no specific suggestions, add generic ones from main category
        if not suggestions:
            main_category = category.split('.')[0] if '.' in category else category
            if main_category in ERROR_SUGGESTIONS:
                suggestions.extend(ERROR_SUGGESTIONS[main_category])
            else:
                suggestions.append(f"Unknown error category: {category}")

        return suggestions

    def _get_suggestions_for_pattern(self, pattern: str, sample: str) -> List[str]:
        """Get suggestions based on error pattern.

        Args:
            pattern: Normalized error pattern
            sample: Sample error message

        Returns:
            List of suggestion strings
        """
        suggestions: List[str] = []
        pattern_lower = pattern.lower()

        # Pattern-based suggestions for specific error types
        if '403' in sample or 'forbidden' in pattern_lower:
            suggestions.append("HTTP 403 errors - video may be private or region-locked")
            suggestions.append("Try using authenticated cookies (--cookies-from-browser)")
        elif 'timeout' in pattern_lower or 'timed out' in pattern_lower:
            suggestions.append("Timeout errors - check network stability")
            suggestions.append("Increase timeout values in config.yaml")
        elif 'dns' in pattern_lower or 'getaddrinfo' in pattern_lower:
            suggestions.append("DNS resolution failed - check network/DNS settings")
            suggestions.append("Try using alternative DNS (8.8.8.8)")
        elif 'connection' in pattern_lower:
            suggestions.append("Connection errors - check network connectivity")
            suggestions.append("Firewall may be blocking connections")
        elif 'rate limit' in pattern_lower or 'too many request' in pattern_lower:
            suggestions.append("Rate limited - implement request throttling")
            suggestions.append("Wait before retrying or use VPN for new IP")

        return suggestions

    def get_healing_strategy_suggestions(
        self,
        aggregator: ErrorAggregator,
    ) -> List[str]:
        """Get healing strategy suggestions based on error categories.

        US-108-006: Integration with self-healing system.

        Args:
            aggregator: ErrorAggregator with recorded errors

        Returns:
            List of healing strategy suggestions
        """
        if not self._enable_healing_integration:
            return []

        suggestions: List[str] = []
        hierarchical = aggregator.get_hierarchical_counts()

        # Map categories to healer names
        category_healers = {
            'network': ['download-healer', 'api-healer'],
            'auth': ['api-healer', 'download-healer'],
            'rate_limit': ['api-healer'],
            'resource_exhaustion': ['disk-healer'],
            'config': ['path-healer'],
            'validation': ['checkpoint-healer'],
            'timeout': ['download-healer'],
        }

        for category in hierarchical:
            if category == 'all':
                continue
            main_cat = category.split('.')[0] if '.' in category else category
            if main_cat in category_healers:
                healers = category_healers[main_cat]
                for healer in healers:
                    suggestion = f"Consider using {healer} for {category} errors"
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)

        return suggestions


def classify_into_subcategory(category: ErrorCategory, error_msg: str) -> str:
    """Classify error into more specific subcategory based on message content.

    Args:
        category: The main error category
        error_msg: The error message to analyze

    Returns:
        Subcategory string (e.g., 'network.timeout') or main category if no match
    """
    error_lower = error_msg.lower()

    if category == ErrorCategory.NETWORK:
        if 'timeout' in error_lower or 'timed out' in error_lower:
            return 'network.timeout'
        elif 'dns' in error_lower or 'getaddrinfo' in error_lower or 'name or service not known' in error_lower:
            return 'network.dns'
        elif 'connection refused' in error_lower or 'connection reset' in error_lower:
            return 'network.connection'
        elif 'ssl' in error_lower or 'tls' in error_lower or 'certificate' in error_lower:
            return 'network.ssl'
        elif 'unreachable' in error_lower or 'network is unreachable' in error_lower:
            return 'network.unreachable'
        return 'network'

    elif category == ErrorCategory.AUTH:
        if '403' in error_lower or 'forbidden' in error_lower:
            return 'auth.403'
        elif 'captcha' in error_lower:
            return 'auth.captcha'
        elif 'rate limit' in error_lower or 'too many requests' in error_lower:
            return 'auth.rate_limit'
        elif 'bot' in error_lower or 'detected' in error_lower:
            return 'auth.bot_detection'
        return 'auth'

    elif category == ErrorCategory.RATE_LIMIT:
        if '429' in error_lower:
            return 'rate_limit.429'
        elif 'quota' in error_lower:
            return 'rate_limit.quota'
        elif 'throttle' in error_lower or 'throttled' in error_lower:
            return 'rate_limit.throttle'
        return 'rate_limit'

    elif category == ErrorCategory.RESOURCE_EXHAUSTION:
        if 'memory' in error_lower or 'out of memory' in error_lower or 'oom' in error_lower:
            return 'resource_exhaustion.memory'
        elif 'disk' in error_lower or 'no space' in error_lower or 'quota' in error_lower:
            return 'resource_exhaustion.disk'
        elif 'cpu' in error_lower or 'too many processes' in error_lower:
            return 'resource_exhaustion.cpu'
        elif 'file' in error_lower or 'handle' in error_lower or 'too many open' in error_lower:
            return 'resource_exhaustion.file_handles'
        return 'resource_exhaustion'

    elif category == ErrorCategory.CONFIG:
        if 'missing' in error_lower or 'not found' in error_lower:
            return 'config.missing'
        elif 'invalid' in error_lower or 'incorrect' in error_lower:
            return 'config.invalid'
        elif 'schema' in error_lower:
            return 'config.schema'
        return 'config'

    elif category == ErrorCategory.TIMEOUT:
        if 'socket' in error_lower:
            return 'timeout.socket'
        elif 'stall' in error_lower:
            return 'timeout.stall'
        elif 'read' in error_lower:
            return 'timeout.read'
        elif 'connect' in error_lower:
            return 'timeout.connect'
        return 'timeout'

    elif category == ErrorCategory.VALIDATION:
        if 'input' in error_lower:
            return 'validation.input'
        elif 'config' in error_lower:
            return 'validation.config'
        elif 'schema' in error_lower:
            return 'validation.schema'
        return 'validation'

    elif category == ErrorCategory.CONTRACT:  # US-108-011
        if 'input' in error_lower:
            return 'contract.input'
        elif 'output' in error_lower:
            return 'contract.output'
        return 'contract'

    return category.value


# Backwards-compatible mapping from legacy category strings used in
# download_segments.classify_error_category() to ErrorCategory values.
_LEGACY_CATEGORY_MAP: Dict[str, ErrorCategory] = {
    'network': ErrorCategory.NETWORK,
    'bot_detection': ErrorCategory.AUTH,
    'rate_limit': ErrorCategory.RATE_LIMIT,
    'resource_exhaustion': ErrorCategory.RESOURCE_EXHAUSTION,
    'config': ErrorCategory.CONFIG,
    'timeout': ErrorCategory.TIMEOUT,
    'video_specific': ErrorCategory.UNKNOWN,
    'unknown': ErrorCategory.UNKNOWN,  # US-120-002: Unknown error category
}


def normalize_category(category: str) -> ErrorCategory:
    """Convert a category string to an ErrorCategory enum value.

    Handles both new ErrorCategory values and legacy category strings
    from classify_error_category() (e.g. 'bot_detection' -> AUTH).

    Args:
        category: Category string (ErrorCategory value or legacy name).

    Returns:
        Matching ErrorCategory enum value, or UNKNOWN if unrecognized.
    """
    # Direct enum value match
    try:
        return ErrorCategory(category)
    except ValueError:
        pass
    # Legacy string match
    return _LEGACY_CATEGORY_MAP.get(category, ErrorCategory.UNKNOWN)


def record_contract_violation(
    aggregator: 'ErrorAggregator',
    stage_name: str,
    violation_type: str,
    field_name: str,
    expected: str,
    actual: str,
) -> None:
    """US-108-011: Record a contract violation to an ErrorAggregator.

    Converts a ContractViolation into error aggregator records.

    Args:
        aggregator: ErrorAggregator to record the violation to
        stage_name: Name of the stage with the violation
        violation_type: 'input' or 'output'
        field_name: Name of the field that violated the contract
        expected: Description of what was expected
        actual: Description of what was actually found
    """
    # Create error message with details
    error_msg = (
        f"[{stage_name}] {violation_type} contract violation: "
        f"field '{field_name}' - expected {expected}, got {actual}"
    )

    # Determine subcategory based on violation type
    category = ErrorCategory.CONTRACT
    subcategory = f"contract.{violation_type}" if violation_type in ('input', 'output') else 'contract'

    # Record the error
    aggregator.record(error_msg, category)


class ErrorAggregator:
    """Collects and categorizes errors during stage execution.

    Each recorded error is filed into an ErrorCategory bucket. The aggregator
    tracks per-category counts and retains one sample message per category
    for the completion summary.

    US-106-009: Enhanced with subcategory tracking for hierarchical grouping.

    Thread-safety: NOT thread-safe. Use one aggregator per stage execution
    (stages run sequentially in the current pipeline).
    """

    def __init__(self, enable_subcategories: bool = True) -> None:
        self._counts: Dict[ErrorCategory, int] = {}
        self._samples: Dict[ErrorCategory, str] = {}
        self._subcategory_counts: Dict[str, int] = {}  # US-106-009: subcategory -> count
        self._subcategory_samples: Dict[str, str] = {}  # US-106-009: subcategory -> sample
        self._similar_groups: Dict[str, List[str]] = {}  # US-106-009: pattern -> [error_messages]
        self._enable_subcategories = enable_subcategories
        # US-120-005: Temporal tracking for error patterns
        self._temporal_errors: List[Dict[str, Any]] = []  # [{timestamp, category, error_msg}]

    def record_with_timestamp(self, error_msg: str, category: ErrorCategory | str, timestamp: Optional[datetime] = None) -> None:
        """Record an error with timestamp for temporal analysis.

        US-120-005: Enables time-of-day and day-of-week pattern analysis.

        Args:
            error_msg: The error message string.
            category: ErrorCategory enum or string.
            timestamp: Optional timestamp (defaults to now).
        """
        # Record normally first
        self.record(error_msg, category)

        # Record with timestamp for temporal analysis
        if timestamp is None:
            timestamp = datetime.now()

        self._temporal_errors.append({
            'timestamp': timestamp,
            'category': category.value if isinstance(category, ErrorCategory) else category,
            'error_msg': error_msg[:200],
        })

    def record(self, error_msg: str, category: ErrorCategory | str) -> None:
        """Record an error with its category.

        Args:
            error_msg: The error message string.
            category: ErrorCategory enum or string (legacy names accepted).
        """
        if isinstance(category, str):
            cat = normalize_category(category)
        else:
            cat = category

        self._counts[cat] = self._counts.get(cat, 0) + 1
        # Keep only the first sample per category (most representative)
        if cat not in self._samples:
            self._samples[cat] = error_msg[:200]  # truncate long messages

        # US-106-009: Track subcategories
        if self._enable_subcategories:
            subcategory = classify_into_subcategory(cat, error_msg)
            self._subcategory_counts[subcategory] = self._subcategory_counts.get(subcategory, 0) + 1
            if subcategory not in self._subcategory_samples:
                self._subcategory_samples[subcategory] = error_msg[:200]
            # Note: Parent category totals are computed in get_hierarchical_counts(),
            # not incremented here, to avoid double-counting

    def record_similar(self, error_msg: str, category: ErrorCategory | str) -> None:
        """Record an error with similarity grouping for repetition detection.

        Groups errors by extracting a normalized pattern (e.g., 'video_id' -> '<ID>').

        Args:
            error_msg: The error message string.
            category: ErrorCategory enum or string.
        """
        # First record the error normally
        self.record(error_msg, category)

        # US-106-009: Group similar errors
        normalized = self._normalize_for_grouping(error_msg)
        if normalized not in self._similar_groups:
            self._similar_groups[normalized] = []
        self._similar_groups[normalized].append(error_msg[:200])

    def _normalize_for_grouping(self, error_msg: str) -> str:
        """Normalize error message for grouping similar errors.

        Replaces variable parts (video IDs, URLs, numbers) with placeholders.

        Args:
            error_msg: Original error message

        Returns:
            Normalized pattern string
        """
        normalized = error_msg.lower()

        # Replace video IDs (e.g., 'abc123def456', 'xyz789uvw012', 'qrs345tuv678')
        normalized = re.sub(r'\b[a-z0-9]{8,}\b', '<VIDEO_ID>', normalized)

        # Replace URLs
        normalized = re.sub(r'https?://[^\s]+', '<URL>', normalized)

        # Replace IP addresses
        normalized = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', normalized)

        # Replace numeric IDs
        normalized = re.sub(r'\b\d{8,}\b', '<NUMERIC_ID>', normalized)

        # Replace common path patterns
        normalized = re.sub(r'/[a-zA-Z0-9_\-./]+', '<PATH>', normalized)

        # Truncate to 100 chars for grouping key
        return normalized[:100]

    def get_similar_group_count(self, pattern: str) -> int:
        """Get count of errors matching a normalized pattern.

        Args:
            pattern: Normalized pattern from _normalize_for_grouping

        Returns:
            Number of similar errors
        """
        return len(self._similar_groups.get(pattern, []))

    @property
    def total_errors(self) -> int:
        """Total number of errors recorded across all categories."""
        return sum(self._counts.values())

    @property
    def categories(self) -> Dict[ErrorCategory, int]:
        """Per-category error counts."""
        return dict(self._counts)

    def get_count(self, category: ErrorCategory) -> int:
        """Get the error count for a specific category."""
        return self._counts.get(category, 0)

    def get_sample(self, category: ErrorCategory) -> Optional[str]:
        """Get the sample error message for a category, if any."""
        return self._samples.get(category)

    def to_dict(self) -> Dict[str, int]:
        """Export counts as a plain dict with string keys.

        Compatible with StageMetrics.error_categories format.
        """
        return {cat.value: count for cat, count in self._counts.items()}

    def summary_rows(self) -> List[Tuple[str, int, str]]:
        """Build summary rows sorted by count descending.

        Returns:
            List of (category_name, count, sample_message) tuples.
        """
        rows = []
        for cat in sorted(self._counts, key=lambda c: self._counts[c], reverse=True):
            rows.append((
                cat.value,
                self._counts[cat],
                self._samples.get(cat, ''),
            ))
        return rows

    def log_summary(self, stage_name: str = '') -> None:
        """Log a categorized error summary table at stage completion.

        Logs at INFO level with category -> count -> sample_message format.
        Only logs if there are errors to report.

        Args:
            stage_name: Optional stage name for log prefix.
        """
        if not self._counts:
            return

        prefix = f"[{stage_name}] " if stage_name else ''
        rows = self.summary_rows()

        # Header
        logger.info(
            f"{prefix}Error summary ({self.total_errors} total errors "
            f"across {len(self._counts)} categories):"
        )
        # Table rows
        for cat_name, count, sample in rows:
            # Truncate sample for log readability
            sample_display = sample[:120] + '...' if len(sample) > 120 else sample
            logger.info(
                f"{prefix}  {cat_name:<12} count={count:<4} sample: {sample_display}"
            )

    def merge(self, other: 'ErrorAggregator') -> None:
        """Merge another aggregator's data into this one.

        Useful for combining errors from sub-stages or retry passes.
        """
        for cat, count in other._counts.items():
            self._counts[cat] = self._counts.get(cat, 0) + count
            if cat not in self._samples and cat in other._samples:
                self._samples[cat] = other._samples[cat]

    def clear(self) -> None:
        """Reset all collected errors."""
        self._counts.clear()
        self._samples.clear()
        self._subcategory_counts.clear()
        self._subcategory_samples.clear()
        self._similar_groups.clear()
        self._temporal_errors.clear()

    def get_hierarchical_counts(self) -> Dict[str, int]:
        """Get counts with hierarchical aggregation.

        US-106-009: Returns counts that include both specific subcategories
        and their parent categories.

        Returns:
            Dict mapping category/subcategory to count (includes 'all' total)
        """
        result = dict(self._subcategory_counts) if self._subcategory_counts else self.to_dict()

        # Add parent category totals (e.g., 'network' = sum of all 'network.*')
        parent_totals: Dict[str, int] = {}
        for subcat, count in result.items():
            if '.' in subcat:
                parent = subcat.split('.')[0]
                parent_totals[parent] = parent_totals.get(parent, 0) + count

        # Merge parent totals into result (only if not already present as explicit category)
        for parent, total in parent_totals.items():
            if parent not in result:
                result[parent] = total
            else:
                # Use the higher of explicit or computed
                result[parent] = max(result[parent], total)

        result['all'] = self.total_errors
        return result

    def get_top_similar_errors(self, limit: int = 5, min_count: int = 2) -> List[Tuple[str, int, str]]:
        """Get most frequent similar error patterns.

        US-106-009: Groups similar errors and returns top patterns.

        Args:
            limit: Maximum number of patterns to return
            min_count: Minimum number of occurrences to include (default 2 for repeated patterns)

        Returns:
            List of (pattern, count, sample_message) tuples
        """
        results = []
        for pattern, messages in self._similar_groups.items():
            if len(messages) >= min_count:
                results.append((pattern, len(messages), messages[0]))

        # Sort by count descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def generate_frequency_report(self, top_n: int = 10) -> Dict[str, Any]:
        """Generate error frequency report with top N errors.

        US-120-005: Provides error analytics with counts and percentages.

        Args:
            top_n: Number of top errors to include (default 10).

        Returns:
            Dict with frequency analysis: top_errors, category_breakdown,
            total_errors, timestamp_range.
        """
        if self.total_errors == 0:
            return {
                'top_errors': [],
                'category_breakdown': {},
                'total_errors': 0,
                'percentages': {},
            }

        total = self.total_errors

        # Get category breakdown with percentages
        hierarchical = self.get_hierarchical_counts()
        category_breakdown = {}
        percentages = {}

        for cat, count in hierarchical.items():
            if cat != 'all':
                category_breakdown[cat] = count
                percentages[cat] = round((count / total) * 100, 1) if total > 0 else 0

        # Get top similar error patterns (from record_similar calls)
        top_patterns = self.get_top_similar_errors(limit=top_n, min_count=1)

        # Format top errors with percentages
        top_errors = []
        for pattern, count, sample in top_patterns:
            top_errors.append({
                'pattern': pattern,
                'count': count,
                'percentage': round((count / total) * 100, 1),
                'sample': sample,
            })

        # If no similar patterns found, fall back to category-level data
        if not top_errors:
            for cat, count in sorted(self._counts.items(), key=lambda x: x[1], reverse=True)[:top_n]:
                sample = self._samples.get(cat, '')
                top_errors.append({
                    'pattern': cat.value,
                    'count': count,
                    'percentage': round((count / total) * 100, 1),
                    'sample': sample,
                })

        # Get timestamp range if temporal data exists
        timestamp_range = None
        if self._temporal_errors:
            timestamps = [e['timestamp'] for e in self._temporal_errors]
            timestamp_range = {
                'earliest': min(timestamps).isoformat(),
                'latest': max(timestamps).isoformat(),
            }

        return {
            'top_errors': top_errors,
            'category_breakdown': category_breakdown,
            'total_errors': total,
            'percentages': percentages,
            'timestamp_range': timestamp_range,
        }

    def get_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in errors.

        US-120-005: Returns time-of-day and day-of-week error distributions.

        Returns:
            Dict with time_of_day and day_of_week distributions.
        """
        if not self._temporal_errors:
            return {
                'time_of_day': {},
                'day_of_week': {},
                'total_temporal_errors': 0,
            }

        # Time of day buckets: morning (6-12), afternoon (12-18), evening (18-22), overnight (22-6)
        time_buckets = {'morning': 0, 'afternoon': 0, 'evening': 0, 'overnight': 0}
        # Day of week: 0=Monday, 6=Sunday
        day_buckets = {i: 0 for i in range(7)}

        for entry in self._temporal_errors:
            ts = entry['timestamp']
            hour = ts.hour

            # Bucket into time of day
            if 6 <= hour < 12:
                time_buckets['morning'] += 1
            elif 12 <= hour < 18:
                time_buckets['afternoon'] += 1
            elif 18 <= hour < 22:
                time_buckets['evening'] += 1
            else:
                time_buckets['overnight'] += 1

            # Bucket into day of week
            day_buckets[ts.weekday()] += 1

        day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_of_week = {day_names[k]: v for k, v in day_buckets.items()}

        return {
            'time_of_day': time_buckets,
            'day_of_week': day_of_week,
            'total_temporal_errors': len(self._temporal_errors),
        }

    def export_as_json(self, include_temporal: bool = True) -> Dict[str, Any]:
        """Export error stats as JSON for external analysis tools.

        US-120-005: Provides JSON export for programmatic use.

        Args:
            include_temporal: Include temporal pattern data (default True).

        Returns:
            Dict suitable for JSON serialization.
        """
        report = self.generate_frequency_report()

        if include_temporal:
            report['temporal_patterns'] = self.get_temporal_patterns()

        return report

    def merge(self, other: 'ErrorAggregator') -> None:
        """Merge another aggregator's data into this one.

        Useful for combining errors from sub-stages or retry passes.
        """
        for cat, count in other._counts.items():
            self._counts[cat] = self._counts.get(cat, 0) + count
            if cat not in self._samples and cat in other._samples:
                self._samples[cat] = other._samples[cat]

        # Merge subcategories
        for subcat, count in other._subcategory_counts.items():
            self._subcategory_counts[subcat] = self._subcategory_counts.get(subcat, 0) + count
            if subcat not in self._subcategory_samples and subcat in other._subcategory_samples:
                self._subcategory_samples[subcat] = other._subcategory_samples[subcat]

        # Merge similar groups
        for pattern, messages in other._similar_groups.items():
            if pattern not in self._similar_groups:
                self._similar_groups[pattern] = []
            self._similar_groups[pattern].extend(messages)

        # Merge temporal errors (US-120-005)
        self._temporal_errors.extend(other._temporal_errors)


# =============================================================================
# US-108-006: Error suggestion engine
# =============================================================================

# Suggestions for each error category - actionable fix recommendations
_ERROR_SUGGESTIONS: Dict[ErrorCategory, List[str]] = {
    ErrorCategory.NETWORK: [
        "Check your internet connection and DNS settings",
        "Verify firewall isn't blocking network requests",
        "Try using a different network (e.g., mobile hotspot)",
    ],
    ErrorCategory.AUTH: [
        "Provide authentication cookies for the video source",
        "Use browser cookies from a logged-in session",
        "Try using a different user agent or impersonation target",
        "Wait an hour before retrying (rate limit may have reset)",
    ],
    ErrorCategory.RATE_LIMIT: [
        "Wait before retrying (rate limit window)",
        "Reduce request frequency in config.yaml",
        "Use --reset-budget flag to start with fresh limits",
        "Consider using a VPN to get a new IP address",
    ],
    ErrorCategory.RESOURCE_EXHAUSTION: [
        "Free up disk space or increase cache cleanup threshold",
        "Reduce batch size in config.yaml",
        "Close other applications to free up memory",
        "Clear old project caches manually",
    ],
    ErrorCategory.CONFIG: [
        "Check config.yaml for invalid or missing values",
        "Run --validate-config to check for errors",
        "Restore config.yaml from a working backup",
    ],
    ErrorCategory.TIMEOUT: [
        "Increase timeout values in config.yaml",
        "Check internet connection stability",
        "Try downloading at different time (off-peak)",
    ],
    ErrorCategory.VALIDATION: [
        "Check input files for correct format",
        "Verify file paths exist and are accessible",
        "Run with --non-interactive to skip prompts",
    ],
    ErrorCategory.UNKNOWN: [
        "Check logs for detailed error information",
        "Try running with --verbose for more details",
        "Report issue if persists",
    ],
}

# Self-healing strategy mapping based on error category
_ERROR_TO_HEALING_STRATEGY: Dict[ErrorCategory, List[str]] = {
    ErrorCategory.NETWORK: ["network-healer", "api-healer"],
    ErrorCategory.AUTH: ["api-healer", "download-healer"],
    ErrorCategory.RATE_LIMIT: ["api-healer", "download-healer"],
    ErrorCategory.RESOURCE_EXHAUSTION: ["disk-healer", "memory-healer"],
    ErrorCategory.CONFIG: ["config-healer", "path-healer"],
    ErrorCategory.TIMEOUT: ["download-healer", "api-healer"],
    ErrorCategory.VALIDATION: ["config-healer", "path-healer"],
    ErrorCategory.UNKNOWN: ["api-healer", "checkpoint-healer"],
}


class ErrorSuggestionEngine:
    """Provides actionable suggestions based on error aggregation.

    US-108-006: Analyzes aggregated errors and generates fix recommendations
    and healing strategy suggestions.

    Usage:
        from src.stages.error_aggregator import ErrorAggregator, ErrorSuggestionEngine

        agg = ErrorAggregator()
        # ... record errors ...
        engine = ErrorSuggestionEngine(agg)
        suggestions = engine.get_suggestions()
        healing_strategies = engine.get_healing_strategies()
    """

    def __init__(self, aggregator: ErrorAggregator):
        """Initialize with an ErrorAggregator.

        Args:
            aggregator: ErrorAggregator instance to analyze
        """
        self._agg = aggregator

    def get_suggestions(self) -> List[Tuple[str, str, List[str]]]:
        """Get actionable suggestions based on recorded errors.

        Returns:
            List of (category, summary, suggestions) tuples sorted by priority
        """
        suggestions: List[Tuple[str, str, List[str]]] = []

        # Get hierarchical counts to identify dominant categories
        hierarchical = self._agg.get_hierarchical_counts()

        # Sort categories by count
        category_counts = [
            (cat, cnt)
            for cat, cnt in hierarchical.items()
            if cat != 'all' and cnt > 0
        ]
        category_counts.sort(key=lambda x: x[1], reverse=True)

        for category, count in category_counts:
            # Find the main category (first part before '.')
            main_cat = category.split('.')[0] if '.' in category else category

            try:
                error_cat = ErrorCategory(main_cat)
            except ValueError:
                error_cat = ErrorCategory.UNKNOWN

            # Get suggestions for this category
            cat_suggestions = _ERROR_SUGGESTIONS.get(error_cat, _ERROR_SUGGESTIONS[ErrorCategory.UNKNOWN])

            # Generate summary message
            total = hierarchical.get('all', 1)
            percentage = (count / total) * 100 if total > 0 else 0
            summary = f"{count} errors ({percentage:.0f}%)"

            suggestions.append((category, summary, cat_suggestions))

        return suggestions

    def get_healing_strategies(self) -> List[Tuple[str, List[str]]]:
        """Get recommended healing strategies based on error categories.

        Returns:
            List of (category, healer_names) tuples
        """
        strategies: List[Tuple[str, List[str]]] = []

        # Get hierarchical counts
        hierarchical = self._agg.get_hierarchical_counts()

        # Find dominant category
        dominant_category = None
        max_count = 0

        for cat, count in hierarchical.items():
            if cat != 'all' and count > max_count:
                max_count = count
                dominant_category = cat

        if dominant_category:
            main_cat = dominant_category.split('.')[0] if '.' in dominant_category else dominant_category

            try:
                error_cat = ErrorCategory(main_cat)
            except ValueError:
                error_cat = ErrorCategory.UNKNOWN

            healers = _ERROR_TO_HEALING_STRATEGY.get(error_cat, _ERROR_TO_HEALING_STRATEGY[ErrorCategory.UNKNOWN])
            strategies.append((dominant_category, healers))

        return strategies

    def format_summary(self) -> str:
        """Format error summary with suggestions as a string.

        Returns:
            Formatted summary string suitable for display
        """
        lines = []
        suggestions = self.get_suggestions()

        if not suggestions:
            return "No errors recorded."

        lines.append("=" * 60)
        lines.append("ERROR SUMMARY WITH SUGGESTIONS")
        lines.append("=" * 60)

        for category, summary, cat_suggestions in suggestions:
            lines.append(f"\n[{category}] - {summary}")
            for i, suggestion in enumerate(cat_suggestions, 1):
                lines.append(f"  {i}. {suggestion}")

        # Add healing strategies
        healing = self.get_healing_strategies()
        if healing:
            lines.append("\n" + "-" * 60)
            lines.append("RECOMMENDED HEALING STRATEGIES:")
            for category, healers in healing:
                lines.append(f"  For {category}: try {' or '.join(healers)}")

        lines.append("=" * 60)
        return "\n".join(lines)


class PipelineErrorSuggestionEngine:
    """Provides suggestions based on pipeline-level error aggregation.

    US-108-006: Analyzes errors across all pipeline stages and generates
    comprehensive fix recommendations.
    """

    def __init__(self, pipeline_aggregator: PipelineErrorAggregator):
        """Initialize with a PipelineErrorAggregator.

        Args:
            pipeline_aggregator: PipelineErrorAggregator instance to analyze
        """
        self._pipeline = pipeline_aggregator

    def get_suggestions(self) -> List[Tuple[str, str, List[str]]]:
        """Get actionable suggestions based on pipeline-level errors.

        Returns:
            List of (stage/category, summary, suggestions) tuples
        """
        suggestions: List[Tuple[str, str, List[str]]] = []

        # Get stage-level errors
        for stage_name in self._pipeline.stages_with_errors:
            stage_agg = self._pipeline._stage_errors.get(stage_name)
            if stage_agg:
                engine = ErrorSuggestionEngine(stage_agg)
                stage_suggestions = engine.get_suggestions()

                for category, summary, cat_suggestions in stage_suggestions:
                    suggestions.append((f"{stage_name}:{category}", summary, cat_suggestions))

        # Add overall category suggestions
        hierarchical = self._pipeline.get_hierarchical_totals()
        total_errors = hierarchical.get('all', 0)

        if total_errors > 0:
            # Find top categories
            top_cats = sorted(
                [(k, v) for k, v in hierarchical.items() if k != 'all'],
                key=lambda x: x[1],
                reverse=True
            )[:3]

            for cat, count in top_cats:
                main_cat = cat.split('.')[0] if '.' in cat else cat
                try:
                    error_cat = ErrorCategory(main_cat)
                except ValueError:
                    error_cat = ErrorCategory.UNKNOWN

                cat_suggestions = _ERROR_SUGGESTIONS.get(error_cat, _ERROR_SUGGESTIONS[ErrorCategory.UNKNOWN])
                suggestions.append((cat, f"{count} errors", cat_suggestions))

        return suggestions

    def format_summary(self) -> str:
        """Format pipeline error summary with suggestions.

        Returns:
            Formatted summary string
        """
        lines = []
        suggestions = self.get_suggestions()

        lines.append("=" * 60)
        lines.append("PIPELINE ERROR SUMMARY WITH SUGGESTIONS")
        lines.append("=" * 60)
        lines.append(f"Total errors: {self._pipeline.total_errors}")
        lines.append(f"Stages with errors: {len(self._pipeline.stages_with_errors)}")

        if not suggestions:
            lines.append("\nNo actionable suggestions (no errors recorded).")
            return "\n".join(lines)

        lines.append("\n" + "-" * 60)
        lines.append("ACTIONABLE SUGGESTIONS:")

        for location, summary, cat_suggestions in suggestions:
            lines.append(f"\n[{location}] - {summary}")
            for i, suggestion in enumerate(cat_suggestions, 1):
                lines.append(f"  {i}. {suggestion}")

        # Add healing strategies
        strategies = self._get_healing_strategies()
        if strategies:
            lines.append("\n" + "-" * 60)
            lines.append("RECOMMENDED HEALING STRATEGIES:")
            for category, healers in strategies:
                lines.append(f"  {category}: try {' or '.join(healers)}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _get_healing_strategies(self) -> List[Tuple[str, List[str]]]:
        """Get healing strategies based on pipeline errors.

        Returns:
            List of (category, healer_names) tuples
        """
        strategies: List[Tuple[str, List[str]]] = []

        # Get category totals
        category_totals = self._pipeline.get_category_totals()

        if not category_totals:
            return strategies

        # Find dominant category
        dominant = max(category_totals.items(), key=lambda x: x[1])

        if dominant[1] > 0:
            try:
                error_cat = ErrorCategory(dominant[0])
            except ValueError:
                error_cat = ErrorCategory.UNKNOWN

            healers = _ERROR_TO_HEALING_STRATEGY.get(error_cat, _ERROR_TO_HEALING_STRATEGY[ErrorCategory.UNKNOWN])
            strategies.append((dominant[0], healers))

        return strategies


class PipelineErrorAggregator:
    """Aggregates errors across all pipeline stages for pipeline-level analysis.

    US-106-009: Provides:
    - Aggregation of errors across all stages
    - Error trend analysis (increasing/decreasing over stages)
    - Pipeline-level error summary

    Usage:
        from src.stages.error_aggregator import PipelineErrorAggregator

        pipeline_agg = PipelineErrorAggregator()
        pipeline_agg.add_stage_errors('DOWNLOAD_SEGMENTS', stage_aggregator)
        pipeline_agg.add_stage_errors('MATCH', match_aggregator)
        pipeline_agg.log_pipeline_summary()
    """

    def __init__(self) -> None:
        # stage_name -> ErrorAggregator
        self._stage_errors: Dict[str, ErrorAggregator] = {}
        # Ordered list of stage names in execution order
        self._stage_order: List[str] = []

    def add_stage_errors(self, stage_name: str, aggregator: ErrorAggregator) -> None:
        """Add errors from a stage.

        Args:
            stage_name: Name of the stage
            aggregator: ErrorAggregator from the stage
        """
        if stage_name not in self._stage_order:
            self._stage_order.append(stage_name)

        if stage_name not in self._stage_errors:
            self._stage_errors[stage_name] = ErrorAggregator()

        self._stage_errors[stage_name].merge(aggregator)

    @property
    def total_errors(self) -> int:
        """Total errors across all stages."""
        return sum(
            agg.total_errors for agg in self._stage_errors.values()
        )

    @property
    def stages_with_errors(self) -> List[str]:
        """List of stage names that had errors."""
        return [
            name for name, agg in self._stage_errors.items()
            if agg.total_errors > 0
        ]

    def get_stage_count(self, stage_name: str) -> int:
        """Get error count for a specific stage."""
        if stage_name in self._stage_errors:
            return self._stage_errors[stage_name].total_errors
        return 0

    def get_category_totals(self) -> Dict[str, int]:
        """Get totals by error category across all stages."""
        totals: Dict[str, int] = {}
        for agg in self._stage_errors.values():
            for cat, count in agg.categories.items():
                cat_str = cat.value if isinstance(cat, ErrorCategory) else str(cat)
                totals[cat_str] = totals.get(cat_str, 0) + count
        return totals

    def get_hierarchical_totals(self) -> Dict[str, int]:
        """Get hierarchical totals including subcategories."""
        totals: Dict[str, int] = {'all': 0}
        for agg in self._stage_errors.values():
            hierarchical = agg.get_hierarchical_counts()
            for cat, count in hierarchical.items():
                totals[cat] = totals.get(cat, 0) + count
        return totals

    def analyze_trends(self) -> Dict[str, str]:
        """Analyze error trends across stage execution order.

        US-106-009: Determines if errors are increasing, decreasing, or stable.

        Returns:
            Dict mapping category to trend: 'increasing', 'decreasing', 'stable', or 'insufficient_data'
        """
        trends: Dict[str, str] = {}

        # Get all categories
        all_categories: Set[str] = set()
        for agg in self._stage_errors.values():
            all_categories.update(agg.categories.keys())

        for cat in all_categories:
            counts_by_stage: List[int] = []
            for stage_name in self._stage_order:
                if stage_name in self._stage_errors:
                    agg = self._stage_errors[stage_name]
                    if isinstance(cat, ErrorCategory):
                        counts_by_stage.append(agg.get_count(cat))
                    else:
                        counts_by_stage.append(agg.get_count(ErrorCategory(cat)))

            if len(counts_by_stage) < 2:
                trends[cat.value if isinstance(cat, ErrorCategory) else cat] = 'insufficient_data'
                continue

            # Calculate trend
            first_half = sum(counts_by_stage[:len(counts_by_stage)//2])
            second_half = sum(counts_by_stage[len(counts_by_stage)//2:])

            if second_half > first_half * 1.2:
                trends[cat.value if isinstance(cat, ErrorCategory) else cat] = 'increasing'
            elif first_half > second_half * 1.2:
                trends[cat.value if isinstance(cat, ErrorCategory) else cat] = 'decreasing'
            else:
                trends[cat.value if isinstance(cat, ErrorCategory) else cat] = 'stable'

        return trends

    def get_repeated_error_patterns(self) -> Dict[str, List[Tuple[str, int]]]:
        """Get error patterns that repeat across stages.

        US-106-009: Identifies errors that occur multiple times across pipeline.

        Returns:
            Dict mapping error pattern to [(stage, count), ...]
        """
        pattern_stages: Dict[str, List[Tuple[str, int]]] = {}

        for stage_name, agg in self._stage_errors.items():
            for pattern, messages in agg._similar_groups.items():
                if len(messages) > 1:
                    if pattern not in pattern_stages:
                        pattern_stages[pattern] = []
                    pattern_stages[pattern].append((stage_name, len(messages)))

        return pattern_stages

    def generate_frequency_report(self, top_n: int = 10) -> Dict[str, Any]:
        """Generate pipeline-level error frequency report.

        US-120-005: Aggregates error frequencies across all stages.

        Args:
            top_n: Number of top errors to include per stage (default 10).

        Returns:
            Dict with stage-level and overall frequency analysis.
        """
        if self.total_errors == 0:
            return {
                'total_errors': 0,
                'stages': {},
                'overall': {'top_errors': [], 'category_breakdown': {}},
            }

        # Generate per-stage frequency reports
        stages_report = {}
        overall_aggregator = ErrorAggregator()

        for stage_name, agg in self._stage_errors.items():
            if agg.total_errors > 0:
                stages_report[stage_name] = agg.generate_frequency_report(top_n)
                overall_aggregator.merge(agg)

        # Generate overall frequency report
        overall_report = overall_aggregator.generate_frequency_report(top_n)

        return {
            'total_errors': self.total_errors,
            'stages': stages_report,
            'overall': overall_report,
            'stages_with_errors': self.stages_with_errors,
        }

    def get_temporal_patterns(self) -> Dict[str, Any]:
        """Get temporal patterns across all pipeline stages.

        US-120-005: Aggregates temporal patterns from all stages.

        Returns:
            Dict with time_of_day and day_of_week distributions.
        """
        # Aggregate temporal data from all stages
        combined_time_buckets = {'morning': 0, 'afternoon': 0, 'evening': 0, 'overnight': 0}
        combined_day_buckets = {i: 0 for i in range(7)}
        total_temporal = 0

        day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_name_to_idx = {name: idx for idx, name in enumerate(day_names)}

        for agg in self._stage_errors.values():
            patterns = agg.get_temporal_patterns()
            if patterns.get('total_temporal_errors', 0) > 0:
                # Merge time of day
                for bucket, count in patterns.get('time_of_day', {}).items():
                    combined_time_buckets[bucket] += count
                # Merge day of week - handle both string and int keys
                for day_key, count in patterns.get('day_of_week', {}).items():
                    if isinstance(day_key, int) and 0 <= day_key <= 6:
                        combined_day_buckets[day_key] += count
                    elif isinstance(day_key, str) and day_key in day_name_to_idx:
                        combined_day_buckets[day_name_to_idx[day_key]] += count
                total_temporal += patterns['total_temporal_errors']

        return {
            'time_of_day': combined_time_buckets,
            'day_of_week': {day_names[k]: v for k, v in combined_day_buckets.items()},
            'total_temporal_errors': total_temporal,
        }

    def export_as_json(self, include_temporal: bool = True) -> Dict[str, Any]:
        """Export pipeline error stats as JSON for external analysis tools.

        US-120-005: Provides JSON export for programmatic use.

        Args:
            include_temporal: Include temporal pattern data (default True).

        Returns:
            Dict suitable for JSON serialization.
        """
        report = self.generate_frequency_report()

        if include_temporal:
            report['temporal_patterns'] = self.get_temporal_patterns()

        return report

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """Export as dict for serialization."""
        return {
            stage: agg.to_dict()
            for stage, agg in self._stage_errors.items()
        }

    def log_pipeline_summary(self) -> None:
        """Log pipeline-level error summary.

        US-106-009: Logs comprehensive summary including:
        - Total errors by category
        - Errors per stage
        - Trend analysis
        - Repeated patterns
        """
        if not self._stage_errors:
            logger.info("Pipeline completed with no errors recorded")
            return

        logger.info("=" * 60)
        logger.info("PIPELINE ERROR SUMMARY")
        logger.info("=" * 60)

        # Total errors
        logger.info(f"Total errors: {self.total_errors}")
        logger.info(f"Stages with errors: {len(self.stages_with_errors)}")

        # Category breakdown
        category_totals = self.get_category_totals()
        if category_totals:
            logger.info("Errors by category:")
            for cat, count in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {cat:<12}: {count}")

        # Hierarchical breakdown
        hierarchical = self.get_hierarchical_totals()
        if hierarchical and len(hierarchical) > len(category_totals):
            logger.info("Errors by category (hierarchical):")
            for cat, count in sorted(hierarchical.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {cat:<20}: {count}")

        # Stage breakdown
        logger.info("Errors per stage:")
        for stage_name in self._stage_order:
            count = self.get_stage_count(stage_name)
            if count > 0:
                logger.info(f"  {stage_name:<20}: {count}")

        # Trend analysis
        trends = self.analyze_trends()
        if trends:
            trend_categories = {k: v for k, v in trends.items() if v != 'insufficient_data'}
            if trend_categories:
                logger.info("Error trends:")
                for cat, trend in sorted(trend_categories.items()):
                    emoji = {
                        'increasing': '↑',
                        'decreasing': '↓',
                        'stable': '→',
                    }.get(trend, '?')
                    logger.info(f"  {cat:<12}: {trend:<12} {emoji}")

        # Repeated patterns
        repeated = self.get_repeated_error_patterns()
        if repeated:
            logger.info(f"Repeated error patterns ({len(repeated)} patterns):")
            for pattern, stage_counts in sorted(repeated.items(), key=lambda x: sum(c for _, c in x[1]), reverse=True)[:5]:
                total = sum(c for _, c in stage_counts)
                stages_str = ', '.join(f"{s}({c})" for s, c in stage_counts)
                logger.info(f"  Pattern: {pattern[:50]}...")
                logger.info(f"    Total: {total}, Stages: {stages_str}")

        logger.info("=" * 60)


# =============================================================================
# US-123-011: Multi-Source Error Aggregation for Unified Rate Limit Analysis
# =============================================================================


class UnifiedErrorAggregator:
    """Aggregates errors across multiple pipeline sources for unified analysis.

    US-123-011: Provides:
    - Aggregation of errors from DOWNLOAD, CAPTION, TRANSCRIPTION sources
    - Unified rate limit statistics across all sources
    - Cross-source correlation to detect when same IP/cookie causes issues

    Usage:
        from src.stages.error_aggregator import UnifiedErrorAggregator, ErrorSource

        unified = UnifiedErrorAggregator()
        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_aggregator)
        unified.aggregate_from_source(ErrorSource.CAPTION, caption_aggregator)
        stats = unified.get_unified_rate_limit_stats()
        correlations = unified.cross_source_correlation()
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the unified error aggregator.

        Args:
            enabled: Whether cross-source tracking is enabled (default True).
        """
        self._enabled = enabled
        # source -> ErrorAggregator
        self._source_errors: Dict[ErrorSource, ErrorAggregator] = {}
        # Track error timestamps for correlation analysis
        self._error_timestamps: List[Dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        """Whether cross-source tracking is enabled."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable cross-source tracking.

        Args:
            enabled: True to enable, False to disable.
        """
        self._enabled = enabled

    def aggregate_from_source(
        self,
        source: ErrorSource,
        aggregator: ErrorAggregator,
    ) -> None:
        """Aggregate errors from a specific pipeline source.

        Args:
            source: The error source (DOWNLOAD, CAPTION, TRANSCRIPTION).
            aggregator: ErrorAggregator containing the errors to merge.
        """
        if not self._enabled:
            return

        if source not in self._source_errors:
            self._source_errors[source] = ErrorAggregator()

        self._source_errors[source].merge(aggregator)

        # Record timestamps for correlation analysis
        for entry in aggregator._temporal_errors:
            self._error_timestamps.append({
                'source': source.value,
                'timestamp': entry.get('timestamp'),
                'category': entry.get('category'),
                'error_msg': entry.get('error_msg', '')[:100],
            })

    def get_unified_rate_limit_stats(self) -> Dict[str, Any]:
        """Get unified rate limit statistics across all sources.

        Returns:
            Dict with rate limit counts per source and total, plus percentages.
        """
        if not self._enabled:
            return {'enabled': False}

        rate_limit_key = ErrorCategory.RATE_LIMIT.value
        result: Dict[str, Any] = {
            'enabled': True,
            'sources': {},
            'total_rate_limits': 0,
            'by_source': {},
        }

        for source, agg in self._source_errors.items():
            count = agg.get_count(ErrorCategory.RATE_LIMIT)
            # Also check hierarchical for subcategories like 'rate_limit.429'
            hierarchical = agg.get_hierarchical_counts()
            hierarchical_count = sum(
                v for k, v in hierarchical.items()
                if k.startswith('rate_limit')
            )

            result['sources'][source.value] = {
                'count': count,
                'hierarchical_count': hierarchical_count,
                'total_errors': agg.total_errors,
            }
            result['total_rate_limits'] += hierarchical_count

        # Calculate percentages
        total_errors = sum(
            agg.total_errors for agg in self._source_errors.values()
        )
        if total_errors > 0:
            result['rate_limit_percentage'] = round(
                (result['total_rate_limits'] / total_errors) * 100, 2
            )
        else:
            result['rate_limit_percentage'] = 0.0

        return result

    def cross_source_correlation(self) -> List[Dict[str, Any]]:
        """Detect when same IP/cookie causes issues across sources.

        Looks for temporal patterns where rate limits occur around the same time
        across different sources, suggesting a common cause (e.g., same IP).

        Returns:
            List of correlation findings with source pairs and timing.
        """
        if not self._enabled or len(self._source_errors) < 2:
            return []

        correlations: List[Dict[str, Any]] = []

        # Group rate limit errors by time window (5-minute windows)
        time_windows: Dict[str, List[Dict[str, Any]]] = {}

        for entry in self._error_timestamps:
            if entry.get('category', '').startswith('rate_limit'):
                ts = entry.get('timestamp')
                if ts:
                    # Round to 5-minute window
                    window_key = f"{ts.year}-{ts.month:02d}-{ts.day:02d}-{ts.hour:02d}-{(ts.minute // 5) * 5:02d}"
                    if window_key not in time_windows:
                        time_windows[window_key] = []
                    time_windows[window_key].append(entry)

        # Find windows with multiple sources
        for window, entries in time_windows.items():
            sources_in_window = set(e['source'] for e in entries)
            if len(sources_in_window) >= 2:
                correlations.append({
                    'window': window,
                    'sources': list(sources_in_window),
                    'error_count': len(entries),
                    'likely_cause': 'common_ip_or_cookie',
                })

        # Also correlate by similar error patterns across sources
        source_patterns: Dict[str, Dict[str, int]] = {}
        for source, agg in self._source_errors.items():
            for pattern, messages in agg._similar_groups.items():
                if len(messages) >= 2:  # Repeated pattern
                    if source.value not in source_patterns:
                        source_patterns[source.value] = {}
                    source_patterns[source.value][pattern] = len(messages)

        # Find patterns that appear in multiple sources
        all_patterns: Set[str] = set()
        for patterns in source_patterns.values():
            all_patterns.update(patterns.keys())

        for pattern in all_patterns:
            sources_with_pattern = [
                s for s, patterns in source_patterns.items()
                if pattern in patterns
            ]
            if len(sources_with_pattern) >= 2:
                correlations.append({
                    'pattern': pattern[:50],
                    'sources': sources_with_pattern,
                    'likely_cause': 'shared_infrastructure',
                })

        return correlations

    def get_all_source_stats(self) -> Dict[str, Any]:
        """Get complete statistics across all sources.

        Returns:
            Dict with breakdown by source and category.
        """
        if not self._enabled:
            return {'enabled': False}

        result: Dict[str, Any] = {
            'enabled': True,
            'sources': {},
            'total_errors': 0,
        }

        for source, agg in self._source_errors.items():
            hierarchical = agg.get_hierarchical_counts()
            result['sources'][source.value] = {
                'total_errors': agg.total_errors,
                'categories': hierarchical,
            }
            result['total_errors'] += agg.total_errors

        return result

    def export_as_json(self) -> Dict[str, Any]:
        """Export unified error stats as JSON.

        Returns:
            Dict suitable for JSON serialization.
        """
        return {
            'unified_stats': self.get_all_source_stats(),
            'rate_limit_stats': self.get_unified_rate_limit_stats(),
            'correlations': self.cross_source_correlation(),
            'enabled': self._enabled,
        }

    def log_unified_summary(self) -> None:
        """Log unified error summary across all sources."""
        if not self._enabled:
            logger.info("Unified error aggregation is disabled")
            return

        if not self._source_errors:
            logger.info("No errors aggregated from any source")
            return

        logger.info("=" * 60)
        logger.info("UNIFIED ERROR SUMMARY (Multi-Source)")
        logger.info("=" * 60)

        # Errors by source
        logger.info("Errors by source:")
        for source, agg in self._source_errors.items():
            logger.info(f"  {source.value:<15}: {agg.total_errors} errors")

        # Rate limit stats
        rl_stats = self.get_unified_rate_limit_stats()
        logger.info(f"Total rate limits: {rl_stats.get('total_rate_limits', 0)}")
        logger.info(f"Rate limit %: {rl_stats.get('rate_limit_percentage', 0)}%")

        # Cross-source correlations
        correlations = self.cross_source_correlation()
        if correlations:
            logger.info(f"Cross-source correlations detected: {len(correlations)}")
            for corr in correlations[:3]:
                logger.info(f"  - {corr.get('sources', [])}: {corr.get('likely_cause', 'unknown')}")

        logger.info("=" * 60)
