"""
Stage-level error aggregation with categorized failure summaries.

US-51-011: Provides an ErrorCategory enum and ErrorAggregator class that
pipeline stages can use to collect, categorize, and summarize errors during
execution. The completion summary logs a table of category -> count -> sample.

Usage:
    from src.stages.error_aggregator import ErrorCategory, ErrorAggregator

    agg = ErrorAggregator()
    agg.record("HTTP Error 403: Forbidden", ErrorCategory.AUTH)
    agg.record("getaddrinfo failed", ErrorCategory.NETWORK)
    agg.log_summary()  # logs table at INFO level
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categorized error buckets for pipeline stage failures.

    Categories are ordered from most specific/actionable to least:
    - NETWORK: Systemic connectivity issues (DNS, unreachable)
    - AUTH: Bot-detection, 403, captcha, sign-in blocks
    - TIMEOUT: Stall timeouts, socket timeouts
    - VALIDATION: Input/config validation failures
    - UNKNOWN: Uncategorized errors

    Inherits from str so values serialize cleanly to JSON/checkpoint dicts.
    """
    NETWORK = 'network'
    AUTH = 'auth'
    TIMEOUT = 'timeout'
    VALIDATION = 'validation'
    UNKNOWN = 'unknown'


# Backwards-compatible mapping from legacy category strings used in
# download_segments.classify_error_category() to ErrorCategory values.
_LEGACY_CATEGORY_MAP: Dict[str, ErrorCategory] = {
    'network': ErrorCategory.NETWORK,
    'bot_detection': ErrorCategory.AUTH,
    'timeout': ErrorCategory.TIMEOUT,
    'video_specific': ErrorCategory.UNKNOWN,
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


class ErrorAggregator:
    """Collects and categorizes errors during stage execution.

    Each recorded error is filed into an ErrorCategory bucket. The aggregator
    tracks per-category counts and retains one sample message per category
    for the completion summary.

    Thread-safety: NOT thread-safe. Use one aggregator per stage execution
    (stages run sequentially in the current pipeline).
    """

    def __init__(self) -> None:
        self._counts: Dict[ErrorCategory, int] = {}
        self._samples: Dict[ErrorCategory, str] = {}

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
