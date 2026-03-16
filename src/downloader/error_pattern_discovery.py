"""
Runtime error pattern auto-discovery for download errors.

US-144-011: Automatically discovers new error patterns from unclassified yt-dlp errors
at runtime. Patterns are stored in the user config directory (not in the codebase)
and must match 3+ occurrences before being promoted for use in error classification.

This module provides:
- RuntimePatternDiscovery: Discovers new patterns from unclassified errors
- PatternValidator: Validates patterns meet occurrence threshold
- Integration with error_classification.py for pattern usage

Example:
    from src.downloader.error_pattern_discovery import get_runtime_pattern_discovery

    discovery = get_runtime_pattern_discovery()

    # Record an unclassified error
    discovery.record_unclassified_error(
        "yt-dlp: specific error message that wasn't classified",
        context={"video_id": "abc123", "stage": "download"}
    )

    # After 3+ occurrences, pattern gets promoted
    promoted = discovery.get_promoted_patterns()
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# User config directory for storing discovered patterns
DEFAULT_PATTERN_DIR = os.path.expanduser("~/.matcher/error_patterns")
DEFAULT_PATTERN_FILE = os.path.join(DEFAULT_PATTERN_DIR, "discovered_patterns.json")

# Minimum occurrences before a pattern can be promoted
DEFAULT_MIN_OCCURRENCES = 3

# Maximum patterns to store
DEFAULT_MAX_PATTERNS = 500

# Confidence threshold for promotion (0.0-1.0)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DiscoveredDownloadPattern:
    """A discovered error pattern from download operations.

    Attributes:
        pattern_id: Unique identifier for the pattern
        pattern_regex: The regex pattern extracted from error messages
        pattern_sample: Sample error message that generated this pattern
        proposed_category: The proposed category (e.g., 'rate_limit', 'network')
        occurrences: Number of times this pattern was observed
        confidence: Confidence score (0.0-1.0) based on occurrence frequency
        first_seen: Timestamp of first occurrence
        last_seen: Timestamp of most recent occurrence
        status: 'discovered', 'validating', 'promoted', 'rejected'
        sample_messages: Example error messages containing this pattern
    """
    pattern_id: str
    pattern_regex: str
    pattern_sample: str
    proposed_category: str
    occurrences: int = 1
    confidence: float = 0.0
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "discovered"
    sample_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DiscoveredDownloadPattern:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class UnclassifiedErrorRecord:
    """Record of an unclassified error for pattern analysis.

    Attributes:
        error_message: The raw error message
        timestamp: When the error occurred
        context: Additional context (video_id, stage, etc.)
    """
    error_message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: dict = field(default_factory=dict)


# =============================================================================
# Pattern Discovery Engine
# =============================================================================


class RuntimePatternDiscovery:
    """Discovers new error patterns from unclassified download errors.

    US-144-011: This class provides runtime auto-discovery of error patterns
    that can be used to improve error classification. It:
    1. Records unclassified errors as they occur
    2. Extracts potential patterns using regex
    3. Validates patterns against the 3+ occurrence threshold
    4. Promotes validated patterns for use in classification

    The key difference from src/common/error_patterns.py:
    - Stores patterns in user config directory (~/.matcher/error_patterns/)
    - Uses stricter 3+ occurrence validation before promotion
    - Focuses specifically on download errors (not caption/other errors)
    - Uses regex-based pattern extraction

    Example:
        discovery = RuntimePatternDiscovery()

        # Record unclassified errors
        discovery.record_unclassified_error("HTTP Error 518: custom server error")

        # After 3+ occurrences, check for promoted patterns
        promoted = discovery.get_promoted_patterns()

        # Export promoted patterns for use
        if promoted:
            export_patterns_for_classification(promoted)
    """

    def __init__(
        self,
        pattern_dir: str = DEFAULT_PATTERN_DIR,
        min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
        max_patterns: int = DEFAULT_MAX_PATTERNS,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """Initialize RuntimePatternDiscovery.

        Args:
            pattern_dir: Directory to store discovered patterns
            min_occurrences: Minimum occurrences before promotion (default 3)
            max_patterns: Maximum patterns to store
            confidence_threshold: Confidence threshold for promotion (0.0-1.0)
        """
        self._pattern_dir = Path(pattern_dir)
        self._min_occurrences = min_occurrences
        self._max_patterns = max_patterns
        self._confidence_threshold = confidence_threshold

        # In-memory storage for patterns and unclassified errors
        self._discovered_patterns: dict[str, DiscoveredDownloadPattern] = {}
        self._unclassified_errors: list[UnclassifiedErrorRecord] = []

        # Load existing patterns from disk
        self._load_patterns()

    def record_unclassified_error(
        self,
        error_message: str,
        context: Optional[dict] = None,
    ) -> Optional[DiscoveredDownloadPattern]:
        """Record an unclassified error for pattern discovery.

        Args:
            error_message: The unclassified error message
            context: Optional context (video_id, stage, etc.)

        Returns:
            DiscoveredDownloadPattern if pattern was promoted, None otherwise
        """
        if not error_message:
            return None

        # Record the unclassified error
        record = UnclassifiedErrorRecord(
            error_message=error_message,
            context=context or {},
        )
        self._unclassified_errors.append(record)

        # Extract potential patterns from this error
        extracted = self._extract_patterns(error_message)

        promoted = None
        for pattern_info in extracted:
            pattern_key = pattern_info["key"]
            pattern_regex = pattern_info["regex"]
            proposed_category = pattern_info["category"]

            # Check if we already have this pattern
            if pattern_key in self._discovered_patterns:
                pattern = self._discovered_patterns[pattern_key]
                pattern.occurrences += 1
                pattern.last_seen = datetime.now().isoformat()

                # Update confidence
                pattern.confidence = self._calculate_confidence(pattern.occurrences)

                # Add sample message if we have room
                if len(pattern.sample_messages) < 5:
                    pattern.sample_messages.append(error_message)

                # Check if promotion threshold met
                if (
                    pattern.status == "discovered"
                    and pattern.occurrences >= self._min_occurrences
                    and pattern.confidence >= self._confidence_threshold
                ):
                    pattern.status = "promoted"
                    promoted = pattern
                    logger.info(
                        "Pattern promoted: %s (occurrences=%d, confidence=%.2f)",
                        pattern_key,
                        pattern.occurrences,
                        pattern.confidence,
                    )
            else:
                # Create new discovered pattern
                new_pattern = DiscoveredDownloadPattern(
                    pattern_id=pattern_key,
                    pattern_regex=pattern_regex,
                    pattern_sample=error_message[:200],
                    proposed_category=proposed_category,
                    occurrences=1,
                    confidence=self._calculate_confidence(1),
                    sample_messages=[error_message] if len(error_message) <= 500 else [],
                )

                # Enforce max patterns limit
                if len(self._discovered_patterns) >= self._max_patterns:
                    self._evict_lowest_confidence()

                self._discovered_patterns[pattern_key] = new_pattern

        # Persist patterns if any were promoted
        if promoted:
            self._persist_patterns()

        return promoted

    def _calculate_confidence(self, occurrences: int) -> float:
        """Calculate confidence based on occurrence count.

        Args:
            occurrences: Number of times pattern was observed

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # More aggressive at lower occurrences:
        # 1 occurrence = 0.3 (30%)
        # 2 occurrences = 0.5 (50%)
        # 3 occurrences = 0.7 (70%) - enough to exceed 0.5 threshold
        # 5+ occurrences = 1.0 (100%)
        if occurrences >= 5:
            return 1.0
        elif occurrences >= 3:
            return 0.7
        elif occurrences >= 2:
            return 0.5
        else:
            return 0.3

    def _extract_patterns(self, error_message: str) -> list[dict]:
        """Extract potential patterns from error message using regex.

        Args:
            error_message: The error message to analyze

        Returns:
            List of dicts with pattern info (key, regex, category)
        """
        patterns = []
        error_lower = error_message.lower()

        # Extract HTTP error codes (e.g., "HTTP Error 503", "HTTP/2 518")
        http_matches = re.findall(
            r'(?:http\s*error\s*(\d{3})|http/[1-3]\s+(\d{3}))',
            error_lower,
            re.IGNORECASE
        )
        for match in http_matches:
            code = match[0] or match[1]
            if code:
                key = f"http_error_{code}"
                patterns.append({
                    "key": key,
                    "regex": rf"http\s*error\s*{re.escape(code)}",
                    "category": self._categorize_http_code(code),
                })

        # Extract yt-dlp extractor errors: "[youtube] error message"
        extractor_matches = re.findall(
            r'\[(\w+)\]\s+(.+?)(?:\s*error|\s*failed)',
            error_lower
        )
        for extractor, msg in extractor_matches:
            key = f"extractor_{extractor}_{msg[:30]}"
            patterns.append({
                "key": key,
                "regex": rf"\[{re.escape(extractor)}\]\s+.*{re.escape(msg[:30])}",
                "category": "extractor",
            })

        # Extract specific error keywords/phrases
        error_keywords = [
            # Network-related
            (r'connection\s+(?:refused|reset|timeout|failed)', 'network'),
            (r'dns\s+(?:resolution\s+)?failed', 'network'),
            (r'ssl\s+(?:handshake\s+)?failed', 'network'),
            (r'tls\s+(?:handshake\s+)?failed', 'network'),
            (r'socket\s+error', 'network'),
            # Rate limiting
            (r'rate\s+limit', 'rate_limit'),
            (r'too\s+many\s+requests', 'rate_limit'),
            (r'quota\s+exceeded', 'rate_limit'),
            # Server errors
            (r'5\d{2}\s+(?:internal\s+server\s+error|service\s+unavailable|bad\s+gateway)', 'server_error'),
            (r'5\d{2}', 'server_error'),
            # Client errors
            (r'4\d{2}', 'client_error'),
            # Bot detection
            (r'bot\s+detection', 'bot_detection'),
            (r'captcha\s+required', 'bot_detection'),
            (r'blocked\s+by\s+bot', 'bot_detection'),
            # Content issues
            (r'video\s+(?:unavailable|removed|deleted|private)', 'content_unavailable'),
            (r'content\s+(?:blocked|restricted)', 'content_blocked'),
            # Auth issues
            (r'authentication\s+required', 'auth_required'),
            (r'login\s+required', 'auth_required'),
            # Extractor issues
            (r'extractor\s+error', 'extractor'),
            (r'unable\s+to\s+extract', 'extractor'),
        ]

        for regex, category in error_keywords:
            if re.search(regex, error_lower):
                key = f"pattern_{regex[:30]}"
                patterns.append({
                    "key": key,
                    "regex": regex,
                    "category": category,
                })

        # If no specific patterns found, use the full message as a generic pattern
        if not patterns:
            # Extract a meaningful substring (first 100 chars, normalized)
            normalized = error_lower[:100].strip()
            normalized = re.sub(r'[^a-z0-9\s]', '', normalized)  # Remove special chars
            if len(normalized) >= 10:
                patterns.append({
                    "key": f"generic_{normalized[:30]}",
                    "regex": re.escape(error_message[:100]),
                    "category": "unknown",
                })

        return patterns

    def _categorize_http_code(self, code: str) -> str:
        """Categorize HTTP error code.

        Args:
            code: HTTP status code

        Returns:
            Category string
        """
        code_int = int(code) if code.isdigit() else 0

        if 500 <= code_int < 600:
            return "server_error"
        elif 400 <= code_int < 500:
            if code_int == 429:
                return "rate_limit"
            elif code_int == 403:
                return "bot_detection"
            elif code_int == 401:
                return "auth_required"
            else:
                return "client_error"
        else:
            return "unknown"

    def _evict_lowest_confidence(self) -> None:
        """Remove the lowest confidence pattern to make room for new ones."""
        if not self._discovered_patterns:
            return

        lowest_key = min(
            self._discovered_patterns.keys(),
            key=lambda k: self._discovered_patterns[k].confidence
        )
        del self._discovered_patterns[lowest_key]
        logger.debug("Evicted lowest confidence pattern: %s", lowest_key)

    def get_promoted_patterns(
        self,
        min_confidence: Optional[float] = None,
    ) -> list[DiscoveredDownloadPattern]:
        """Get patterns that have been promoted for use in classification.

        Args:
            min_confidence: Optional minimum confidence threshold

        Returns:
            List of promoted DiscoveredDownloadPattern objects
        """
        threshold = min_confidence if min_confidence is not None else self._confidence_threshold
        return [
            p for p in self._discovered_patterns.values()
            if p.status == "promoted" and p.confidence >= threshold
        ]

    def get_discovered_patterns(
        self,
        status: Optional[str] = None,
    ) -> list[DiscoveredDownloadPattern]:
        """Get all discovered patterns, optionally filtered by status.

        Args:
            status: Optional status filter ('discovered', 'validating', 'promoted', 'rejected')

        Returns:
            List of DiscoveredDownloadPattern objects
        """
        if status:
            return [
                p for p in self._discovered_patterns.values()
                if p.status == status
            ]
        return list(self._discovered_patterns.values())

    def get_unclassified_errors(self) -> list[UnclassifiedErrorRecord]:
        """Get all recorded unclassified errors.

        Returns:
            List of UnclassifiedErrorRecord objects
        """
        return list(self._unclassified_errors)

    def review_pattern(
        self,
        pattern_id: str,
        approved: bool,
        override_category: Optional[str] = None,
    ) -> bool:
        """Manually review a discovered pattern.

        Args:
            pattern_id: The pattern ID to review
            approved: Whether to approve the pattern
            override_category: Optional category override

        Returns:
            True if pattern was found and updated
        """
        if pattern_id not in self._discovered_patterns:
            return False

        pattern = self._discovered_patterns[pattern_id]

        if approved:
            pattern.status = "promoted"
            if override_category:
                pattern.proposed_category = override_category
            pattern.confidence = 1.0  # Manual approval = max confidence
        else:
            pattern.status = "rejected"

        self._persist_patterns()
        return True

    def _load_patterns(self) -> bool:
        """Load discovered patterns from disk.

        Returns:
            True if patterns were loaded successfully
        """
        pattern_file = Path(self._pattern_dir) / "discovered_patterns.json"

        if not pattern_file.exists():
            logger.debug("No discovered patterns file found, starting fresh")
            return False

        try:
            with open(pattern_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            patterns_list = data.get("patterns", [])
            for pattern_data in patterns_list:
                pattern = DiscoveredDownloadPattern.from_dict(pattern_data)
                self._discovered_patterns[pattern.pattern_id] = pattern

            logger.info(
                "Loaded %d discovered patterns from %s",
                len(self._discovered_patterns),
                pattern_file
            )
            return True

        except Exception as e:
            logger.warning("Failed to load discovered patterns: %s", e)
            return False

    def _persist_patterns(self) -> bool:
        """Save discovered patterns to disk.

        Returns:
            True if patterns were saved successfully
        """
        try:
            # Ensure directory exists
            self._pattern_dir.mkdir(parents=True, exist_ok=True)

            pattern_file = self._pattern_dir / "discovered_patterns.json"

            # Convert patterns to dicts
            patterns_list = [
                p.to_dict() for p in self._discovered_patterns.values()
            ]

            data = {
                "version": "1.0",
                "min_occurrences": self._min_occurrences,
                "confidence_threshold": self._confidence_threshold,
                "patterns": patterns_list,
                "updated_at": datetime.now().isoformat(),
            }

            with open(pattern_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.debug(
                "Persisted %d patterns to %s",
                len(self._discovered_patterns),
                pattern_file
            )
            return True

        except Exception as e:
            logger.error("Failed to persist discovered patterns: %s", e)
            return False

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
            "total_unclassified_errors": len(self._unclassified_errors),
            "total_occurrences": total_occurrences,
            "by_category": categories,
            "by_status": statuses,
            "min_occurrences_required": self._min_occurrences,
            "confidence_threshold": self._confidence_threshold,
            "pattern_dir": str(self._pattern_dir),
        }

    def export_promoted_patterns(self) -> list[dict]:
        """Export promoted patterns in a format usable by error classification.

        Returns:
            List of dicts with pattern info suitable for classification use
        """
        promoted = self.get_promoted_patterns()

        return [
            {
                "pattern": p.pattern_regex,
                "category": p.proposed_category,
                "confidence": p.confidence,
                "occurrences": p.occurrences,
            }
            for p in promoted
        ]

    def clear_patterns(self, status: Optional[str] = None) -> int:
        """Clear discovered patterns.

        Args:
            status: Optional status to filter clearing

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

        if count > 0:
            self._persist_patterns()

        return count


# =============================================================================
# Global Instance
# =============================================================================

_runtime_pattern_discovery: Optional[RuntimePatternDiscovery] = None


def get_runtime_pattern_discovery() -> RuntimePatternDiscovery:
    """Get the global RuntimePatternDiscovery instance.

    Returns:
        The global RuntimePatternDiscovery
    """
    global _runtime_pattern_discovery
    if _runtime_pattern_discovery is None:
        _runtime_pattern_discovery = RuntimePatternDiscovery()
    return _runtime_pattern_discovery


def reset_runtime_pattern_discovery() -> None:
    """Reset the global RuntimePatternDiscovery instance.

    Useful for testing.
    """
    global _runtime_pattern_discovery
    _runtime_pattern_discovery = None


# =============================================================================
# Integration with error_classification.py
# =============================================================================

def get_pattern_for_classification() -> list[tuple[str, str]]:
    """Get promoted patterns formatted for use in error classification.

    Returns:
        List of (pattern, category) tuples suitable for adding to
        error classification patterns
    """
    discovery = get_runtime_pattern_discovery()
    exported = discovery.export_promoted_patterns()

    return [
        (p["pattern"], p["category"])
        for p in exported
    ]
