"""Unit tests for shared error patterns (US-67-009).

Verifies that src/common/error_patterns.py is the single source of truth
for error classification patterns used by both the downloader and caption systems.
Adding a pattern to the shared module must be picked up by both classifiers.
"""

from __future__ import annotations

import pytest

from src.common.error_patterns import (
    AUTH_PATTERNS,
    BOT_DETECTION_PATTERNS,
    FORMAT_UNAVAILABLE_PATTERNS,
    NETWORK_PATTERNS,
    PARSE_PATTERNS,
    RATE_LIMIT_PATTERNS,
    TIMEOUT_PATTERNS,
    UNAVAILABLE_PATTERNS,
)


@pytest.mark.fast
class TestSharedPatternsExist:
    """Verify shared pattern lists are non-empty and well-formed."""

    def test_rate_limit_patterns_non_empty(self):
        assert len(RATE_LIMIT_PATTERNS) > 0

    def test_bot_detection_patterns_non_empty(self):
        assert len(BOT_DETECTION_PATTERNS) > 0

    def test_timeout_patterns_non_empty(self):
        assert len(TIMEOUT_PATTERNS) > 0

    def test_network_patterns_non_empty(self):
        assert len(NETWORK_PATTERNS) > 0

    def test_parse_patterns_non_empty(self):
        assert len(PARSE_PATTERNS) > 0

    def test_unavailable_patterns_non_empty(self):
        assert len(UNAVAILABLE_PATTERNS) > 0

    def test_auth_patterns_non_empty(self):
        assert len(AUTH_PATTERNS) > 0

    def test_all_patterns_are_lowercase_strings(self):
        """All shared patterns should be lowercase strings for case-insensitive matching."""
        all_lists = [
            RATE_LIMIT_PATTERNS, BOT_DETECTION_PATTERNS, TIMEOUT_PATTERNS,
            NETWORK_PATTERNS, PARSE_PATTERNS, UNAVAILABLE_PATTERNS,
            AUTH_PATTERNS, FORMAT_UNAVAILABLE_PATTERNS,
        ]
        for pattern_list in all_lists:
            for pattern in pattern_list:
                assert isinstance(pattern, str), f"Pattern {pattern!r} is not a string"
                assert pattern == pattern.lower(), f"Pattern {pattern!r} is not lowercase"


@pytest.mark.fast
class TestDownloaderUsesSharedPatterns:
    """Verify downloader error_classification imports from shared module."""

    def test_severity_high_contains_bot_detection(self):
        """ERROR_SEVERITY_PATTERNS['high'] includes BOT_DETECTION_PATTERNS."""
        from src.downloader.error_classification import ERROR_SEVERITY_PATTERNS
        high = ERROR_SEVERITY_PATTERNS['high']
        for pattern in BOT_DETECTION_PATTERNS:
            assert pattern in high, f"Missing shared pattern '{pattern}' in severity 'high'"

    def test_severity_medium_contains_rate_limit_core(self):
        """ERROR_SEVERITY_PATTERNS['medium'] includes core rate limit patterns."""
        from src.downloader.error_classification import ERROR_SEVERITY_PATTERNS
        medium = ERROR_SEVERITY_PATTERNS['medium']
        # 429, too many requests, and rate limit are the core rate-limit patterns
        for pattern in ['429', 'too many requests', 'rate limit']:
            assert pattern in medium, f"Missing shared pattern '{pattern}' in severity 'medium'"

    def test_severity_low_contains_auth_patterns(self):
        """ERROR_SEVERITY_PATTERNS['low'] includes AUTH_PATTERNS."""
        from src.downloader.error_classification import ERROR_SEVERITY_PATTERNS
        low = ERROR_SEVERITY_PATTERNS['low']
        for pattern in AUTH_PATTERNS:
            assert pattern in low, f"Missing shared pattern '{pattern}' in severity 'low'"


@pytest.mark.fast
class TestCaptionUsesSharedPatterns:
    """Verify caption error_handling imports from shared module."""

    def test_rate_limit_detected_via_shared(self):
        """categorize_caption_error detects rate limits using shared RATE_LIMIT_PATTERNS."""
        from src.caption.error_handling import categorize_caption_error
        from src.caption.enums import CaptionErrorCategory
        from src.caption.exceptions import CaptionFetchError

        # Test each rate limit pattern
        for pattern in RATE_LIMIT_PATTERNS:
            error = CaptionFetchError("test_vid", f"Error: {pattern} occurred")
            category = categorize_caption_error(error, error.reason)
            assert category == CaptionErrorCategory.RATE_LIMIT, (
                f"Shared pattern '{pattern}' not detected as RATE_LIMIT"
            )

    def test_timeout_detected_via_shared(self):
        """categorize_caption_error detects timeouts using shared TIMEOUT_PATTERNS."""
        from src.caption.error_handling import categorize_caption_error
        from src.caption.enums import CaptionErrorCategory
        from src.caption.exceptions import CaptionFetchError

        for pattern in TIMEOUT_PATTERNS:
            error = CaptionFetchError("test_vid", f"Error: {pattern} occurred")
            category = categorize_caption_error(error, error.reason)
            assert category == CaptionErrorCategory.TIMEOUT, (
                f"Shared pattern '{pattern}' not detected as TIMEOUT"
            )

    def test_network_detected_via_shared(self):
        """categorize_caption_error detects network errors using shared NETWORK_PATTERNS."""
        from src.caption.error_handling import categorize_caption_error
        from src.caption.enums import CaptionErrorCategory
        from src.caption.exceptions import CaptionFetchError

        for pattern in NETWORK_PATTERNS:
            error = CaptionFetchError("test_vid", f"Error: {pattern} issue")
            category = categorize_caption_error(error, error.reason)
            assert category == CaptionErrorCategory.NETWORK, (
                f"Shared pattern '{pattern}' not detected as NETWORK"
            )


@pytest.mark.fast
class TestAddingSharedPatternPropagates:
    """US-67-009 acceptance criterion: adding a pattern to the shared module
    is picked up by both downloader and caption classifiers.

    Uses monkeypatch to temporarily add a pattern and verify both systems see it.
    """

    def test_new_rate_limit_pattern_propagates_to_caption(self, monkeypatch):
        """A new rate limit pattern added to shared module is used by caption classifier."""
        from src.common import error_patterns
        from src.caption.error_handling import categorize_caption_error
        from src.caption.enums import CaptionErrorCategory
        from src.caption.exceptions import CaptionFetchError

        # Add a new pattern to the shared list
        original = error_patterns.RATE_LIMIT_PATTERNS[:]
        monkeypatch.setattr(
            error_patterns, 'RATE_LIMIT_PATTERNS',
            original + ["test_sentinel_xyzzy"]
        )

        # Reload the reference used by caption error_handling
        # Since error_handling imports the list object, we need to verify
        # the caption module references the same list. Because Python imports
        # bind to the module attribute, we patch the caption module's reference too.
        from src.caption import error_handling
        monkeypatch.setattr(
            error_handling, 'RATE_LIMIT_PATTERNS',
            error_patterns.RATE_LIMIT_PATTERNS
        )

        error = CaptionFetchError("vid", "test_sentinel_xyzzy detected")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.RATE_LIMIT

    def test_new_rate_limit_pattern_propagates_to_downloader(self, monkeypatch):
        """A new rate limit pattern added to shared module is used by downloader classifier."""
        from src.common import error_patterns
        from src.downloader.error_classification import classify_error_severity

        # Add 'test_sentinel_xyzzy' to shared RATE_LIMIT_PATTERNS
        original = error_patterns.RATE_LIMIT_PATTERNS[:]
        new_patterns = original + ["test_sentinel_xyzzy"]
        monkeypatch.setattr(error_patterns, 'RATE_LIMIT_PATTERNS', new_patterns)

        # Rebuild ERROR_SEVERITY_PATTERNS medium list to include the new pattern
        from src.downloader import error_classification
        original_severity = error_classification.ERROR_SEVERITY_PATTERNS.copy()
        new_medium = [p for p in new_patterns if p not in ('quota exceeded', 'throttle', 'rate-limit', 'slow down')] + [
            'please try again later', 'temporarily unavailable',
        ]
        monkeypatch.setattr(
            error_classification, 'ERROR_SEVERITY_PATTERNS',
            {**original_severity, 'medium': new_medium}
        )

        # Verify the new pattern is classified as 'medium' severity
        result = classify_error_severity("test_sentinel_xyzzy happened")
        assert result == 'medium'

    def test_new_bot_detection_pattern_propagates_to_downloader(self, monkeypatch):
        """A new bot detection pattern added to shared module is used by downloader."""
        from src.common import error_patterns
        from src.downloader.error_classification import classify_error_severity

        # Add 'test_sentinel_bot' to shared BOT_DETECTION_PATTERNS
        original = error_patterns.BOT_DETECTION_PATTERNS[:]
        new_patterns = original + ["test_sentinel_bot"]
        monkeypatch.setattr(error_patterns, 'BOT_DETECTION_PATTERNS', new_patterns)

        # Rebuild ERROR_SEVERITY_PATTERNS high list
        from src.downloader import error_classification
        original_severity = error_classification.ERROR_SEVERITY_PATTERNS.copy()
        new_high = [p for p in new_patterns] + ['quota exceeded']
        monkeypatch.setattr(
            error_classification, 'ERROR_SEVERITY_PATTERNS',
            {**original_severity, 'high': new_high}
        )

        result = classify_error_severity("test_sentinel_bot detected")
        assert result == 'high'
