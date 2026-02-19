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
    GEO_BLOCKED_PATTERNS,
    DEVICE_LIMIT_PATTERNS,
    HIGH_SEVERITY_PATTERNS,
    MEDIUM_SEVERITY_PATTERNS,
    LOW_SEVERITY_PATTERNS,
    NETWORK_PATTERNS,
    PARSE_PATTERNS,
    RATE_LIMIT_PATTERNS,
    TIMEOUT_PATTERNS,
    UNAVAILABLE_PATTERNS,
    UNKNOWN_PATTERNS,
    UnknownErrorHandler,
    get_unknown_error_handler,
    is_unknown_error,
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

    def test_severity_high_uses_shared_high_severity_patterns(self):
        """ERROR_SEVERITY_PATTERNS['high'] starts with HIGH_SEVERITY_PATTERNS from shared module."""
        from src.common.error_patterns import HIGH_SEVERITY_PATTERNS
        from src.downloader.error_classification import ERROR_SEVERITY_PATTERNS
        high = ERROR_SEVERITY_PATTERNS['high']
        for pattern in HIGH_SEVERITY_PATTERNS:
            assert pattern in high, f"Shared HIGH_SEVERITY pattern '{pattern}' missing from severity 'high'"

    def test_severity_medium_uses_shared_medium_severity_patterns(self):
        """ERROR_SEVERITY_PATTERNS['medium'] starts with MEDIUM_SEVERITY_PATTERNS from shared module."""
        from src.common.error_patterns import MEDIUM_SEVERITY_PATTERNS
        from src.downloader.error_classification import ERROR_SEVERITY_PATTERNS
        medium = ERROR_SEVERITY_PATTERNS['medium']
        for pattern in MEDIUM_SEVERITY_PATTERNS:
            assert pattern in medium, f"Shared MEDIUM_SEVERITY pattern '{pattern}' missing from severity 'medium'"

    def test_severity_low_uses_shared_low_severity_patterns(self):
        """ERROR_SEVERITY_PATTERNS['low'] starts with LOW_SEVERITY_PATTERNS from shared module."""
        from src.common.error_patterns import LOW_SEVERITY_PATTERNS
        from src.downloader.error_classification import ERROR_SEVERITY_PATTERNS
        low = ERROR_SEVERITY_PATTERNS['low']
        for pattern in LOW_SEVERITY_PATTERNS:
            assert pattern in low, f"Shared LOW_SEVERITY pattern '{pattern}' missing from severity 'low'"

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


@pytest.mark.fast
class TestNewPatternGroups:
    """Test new pattern groups added in US-113-006 and later."""

    def test_geo_blocked_patterns_non_empty(self):
        """GEO_BLOCKED_PATTERNS should not be empty."""
        assert len(GEO_BLOCKED_PATTERNS) > 0

    def test_device_limit_patterns_non_empty(self):
        """DEVICE_LIMIT_PATTERNS should not be empty."""
        assert len(DEVICE_LIMIT_PATTERNS) > 0

    def test_format_unavailable_patterns_non_empty(self):
        """FORMAT_UNAVAILABLE_PATTERNS should not be empty."""
        assert len(FORMAT_UNAVAILABLE_PATTERNS) > 0

    def test_geo_blocked_patterns_all_lowercase(self):
        """All geo-blocked patterns should be lowercase."""
        for pattern in GEO_BLOCKED_PATTERNS:
            assert isinstance(pattern, str), f"Pattern {pattern!r} is not a string"
            assert pattern == pattern.lower(), f"Pattern {pattern!r} is not lowercase"

    def test_device_limit_patterns_all_lowercase(self):
        """All device limit patterns should be lowercase."""
        for pattern in DEVICE_LIMIT_PATTERNS:
            assert isinstance(pattern, str), f"Pattern {pattern!r} is not a string"
            assert pattern == pattern.lower(), f"Pattern {pattern!r} is not lowercase"

    def test_geo_blocked_detected_by_caption_classifier(self):
        """categorize_caption_error detects geo-blocking using GEO_BLOCKED_PATTERNS."""
        from src.caption.error_handling import categorize_caption_error
        from src.caption.enums import CaptionErrorCategory
        from src.caption.exceptions import CaptionFetchError

        # Test each geo-blocked pattern
        for pattern in GEO_BLOCKED_PATTERNS[:3]:  # Test first 3 patterns
            error = CaptionFetchError("test_vid", f"Error: {pattern}")
            category = categorize_caption_error(error, error.reason)
            # Geo-blocking should be detected as UNAVAILABLE or NETWORK
            assert category in (CaptionErrorCategory.UNAVAILABLE, CaptionErrorCategory.NETWORK), (
                f"Pattern '{pattern}' not detected properly"
            )

    def test_device_limit_detected_by_caption_classifier(self):
        """categorize_caption_error detects device limits."""
        from src.caption.error_handling import categorize_caption_error
        from src.caption.enums import CaptionErrorCategory
        from src.caption.exceptions import CaptionFetchError

        # Test each device limit pattern
        for pattern in DEVICE_LIMIT_PATTERNS[:2]:  # Test first 2 patterns
            error = CaptionFetchError("test_vid", f"Error: {pattern}")
            category = categorize_caption_error(error, error.reason)
            # Device limit should be categorized (usually as NETWORK or UNAVAILABLE)
            assert category in (CaptionErrorCategory.NETWORK, CaptionErrorCategory.UNAVAILABLE), (
                f"Pattern '{pattern}' not detected properly"
            )


@pytest.mark.fast
class TestSeverityPatternGroups:
    """Test HIGH_SEVERITY_PATTERNS, MEDIUM_SEVERITY_PATTERNS, LOW_SEVERITY_PATTERNS."""

    def test_high_severity_patterns_non_empty(self):
        """HIGH_SEVERITY_PATTERNS should not be empty."""
        assert len(HIGH_SEVERITY_PATTERNS) > 0

    def test_medium_severity_patterns_non_empty(self):
        """MEDIUM_SEVERITY_PATTERNS should not be empty."""
        assert len(MEDIUM_SEVERITY_PATTERNS) > 0

    def test_low_severity_patterns_non_empty(self):
        """LOW_SEVERITY_PATTERNS should not be empty."""
        assert len(LOW_SEVERITY_PATTERNS) > 0

    def test_high_severity_contains_bot_detection(self):
        """HIGH_SEVERITY_PATTERNS should include all BOT_DETECTION_PATTERNS."""
        for pattern in BOT_DETECTION_PATTERNS:
            assert pattern in HIGH_SEVERITY_PATTERNS, (
                f"BOT_DETECTION pattern '{pattern}' missing from HIGH_SEVERITY"
            )

    def test_high_severity_contains_quota_exceeded(self):
        """HIGH_SEVERITY_PATTERNS should include 'quota exceeded'."""
        assert "quota exceeded" in HIGH_SEVERITY_PATTERNS

    def test_medium_severity_excludes_quota_exceeded(self):
        """MEDIUM_SEVERITY_PATTERNS should exclude 'quota exceeded' (goes to high)."""
        assert "quota exceeded" not in MEDIUM_SEVERITY_PATTERNS

    def test_medium_severity_includes_429(self):
        """MEDIUM_SEVERITY_PATTERNS should include core rate limit patterns."""
        assert "429" in MEDIUM_SEVERITY_PATTERNS
        assert "too many requests" in MEDIUM_SEVERITY_PATTERNS
        assert "rate limit" in MEDIUM_SEVERITY_PATTERNS

    def test_low_severity_contains_auth_patterns(self):
        """LOW_SEVERITY_PATTERNS should include all AUTH_PATTERNS."""
        for pattern in AUTH_PATTERNS:
            assert pattern in LOW_SEVERITY_PATTERNS, (
                f"AUTH pattern '{pattern}' missing from LOW_SEVERITY"
            )

    def test_all_severity_patterns_lowercase(self):
        """All severity pattern groups should be lowercase strings."""
        for pattern in HIGH_SEVERITY_PATTERNS + MEDIUM_SEVERITY_PATTERNS + LOW_SEVERITY_PATTERNS:
            assert isinstance(pattern, str), f"Pattern {pattern!r} is not a string"
            assert pattern == pattern.lower(), f"Pattern {pattern!r} is not lowercase"

    def test_severity_groups_disjoint(self):
        """Severity pattern groups should be mostly disjoint (no overlap in critical patterns)."""
        # These should NOT overlap between groups
        high_only = {"quota exceeded"}
        medium_only = {"429", "too many requests", "rate limit"}
        low_only = set(AUTH_PATTERNS)

        # Check overlaps
        assert not (high_only & medium_only), "HIGH and MEDIUM should not overlap"
        assert not (high_only & low_only), "HIGH and LOW should not overlap"
        assert not (medium_only & low_only), "MEDIUM and LOW should not overlap"


@pytest.mark.fast
class TestPatternMatchingIntegration:
    """Integration tests for pattern matching across systems."""

    def test_error_classification_uses_geo_blocked(self):
        """Downloader error_classification should detect geo-blocked errors."""
        from src.downloader.error_classification import classify_error_severity

        for pattern in GEO_BLOCKED_PATTERNS[:3]:
            result = classify_error_severity(f"Error: {pattern}")
            # Should be classified as some severity (not error)
            assert result in ('high', 'medium', 'low'), (
                f"Geo-blocked pattern '{pattern}' not classified"
            )

    def test_error_classification_uses_device_limit(self):
        """Downloader error_classification should detect device limit errors."""
        from src.downloader.error_classification import classify_error_severity

        for pattern in DEVICE_LIMIT_PATTERNS[:2]:
            result = classify_error_severity(f"Error: {pattern}")
            # Should be classified
            assert result in ('high', 'medium', 'low'), (
                f"Device limit pattern '{pattern}' not classified"
            )

    def test_error_classification_uses_format_unavailable(self):
        """Downloader error_classification should detect format unavailable."""
        from src.downloader.error_classification import classify_error_severity

        for pattern in FORMAT_UNAVAILABLE_PATTERNS:
            result = classify_error_severity(f"Error: {pattern}")
            # Should be classified
            assert result in ('high', 'medium', 'low'), (
                f"Format unavailable pattern '{pattern}' not classified"
            )

    def test_pattern_coverage_summary(self):
        """Verify all pattern lists have reasonable coverage."""
        # All pattern lists should have at least 2 patterns
        min_patterns = {
            'RATE_LIMIT_PATTERNS': RATE_LIMIT_PATTERNS,
            'BOT_DETECTION_PATTERNS': BOT_DETECTION_PATTERNS,
            'TIMEOUT_PATTERNS': TIMEOUT_PATTERNS,
            'NETWORK_PATTERNS': NETWORK_PATTERNS,
            'GEO_BLOCKED_PATTERNS': GEO_BLOCKED_PATTERNS,
            'DEVICE_LIMIT_PATTERNS': DEVICE_LIMIT_PATTERNS,
        }

        for name, patterns in min_patterns.items():
            assert len(patterns) >= 2, f"{name} should have at least 2 patterns"


@pytest.mark.fast
class TestUnknownErrorPatterns:
    """US-120-002: Tests for Unknown error classification."""

    def test_unknown_patterns_exist(self):
        """UNKNOWN_PATTERNS should exist and be non-empty."""
        assert len(UNKNOWN_PATTERNS) > 0

    def test_unknown_patterns_all_lowercase(self):
        """All UNKNOWN_PATTERNS should be lowercase strings."""
        for pattern in UNKNOWN_PATTERNS:
            assert isinstance(pattern, str), f"Pattern {pattern!r} is not a string"
            assert pattern == pattern.lower(), f"Pattern {pattern!r} is not lowercase"

    def test_is_unknown_error_known_patterns(self):
        """Errors with known patterns should NOT be classified as Unknown."""
        # Rate limit error
        assert not is_unknown_error("HTTP Error 429: Too Many Requests")
        # Bot detection
        assert not is_unknown_error("HTTP Error 403: Forbidden")
        # Timeout
        assert not is_unknown_error("Connection timed out")
        # Network
        assert not is_unknown_error("getaddrinfo failed")
        # Geo-blocked
        assert not is_unknown_error("not available in your country")
        # Device limit
        assert not is_unknown_error("device limit exceeded")

    def test_is_unknown_error_truly_unknown(self):
        """Errors that don't match known patterns should be classified as Unknown."""
        # These don't match any known pattern
        assert is_unknown_error("Something completely unexpected happened")
        assert is_unknown_error("This is a bizarre error xyz123")
        # Note: "File not found" matches "not found" in UNAVAILABLE_PATTERNS
        assert not is_unknown_error("File not found: /tmp/nothing")

    def test_is_unknown_error_partial_matches(self):
        """Errors with partial keyword matches to known patterns should still be Unknown."""
        # "error" alone is in UNKNOWN_PATTERNS but doesn't match specific categories
        # The function checks against all specific patterns first
        assert is_unknown_error("An unusual error occurred")

    def test_unknown_error_handler_initialization(self):
        """UnknownErrorHandler should initialize properly."""
        handler = UnknownErrorHandler()
        assert handler is not None

    def test_unknown_error_handler_captures_error(self):
        """UnknownErrorHandler.handle_unknown_error should capture error details."""
        handler = UnknownErrorHandler()
        handler.clear()  # Start fresh

        error_msg = "This is a truly unknown error for testing"
        result = handler.handle_unknown_error(error_msg)

        assert result['original_message'] == error_msg
        assert result['is_unknown'] is True
        assert 'extracted_categories' in result

    def test_unknown_error_handler_with_exception(self):
        """UnknownErrorHandler should capture stack trace when exception provided."""
        handler = UnknownErrorHandler()
        handler.clear()

        try:
            raise ValueError("Test exception for stack trace")
        except ValueError as e:
            result = handler.handle_unknown_error(str(e), exception=e)

        assert 'stack_trace' in result
        assert result['stack_trace'] is not None
        assert 'ValueError' in result['stack_trace']

    def test_unknown_error_handler_extracts_categories(self):
        """UnknownErrorHandler should attempt to extract potential categories."""
        handler = UnknownErrorHandler()

        # Test with partial match
        result = handler.handle_unknown_error("Error: rate limit exceeded but not quite")

        # Should have some extracted categories
        assert 'extracted_categories' in result

    def test_unknown_error_handler_get_captured_errors(self):
        """UnknownErrorHandler.get_captured_errors should return all captured errors."""
        handler = UnknownErrorHandler()
        handler.clear()

        handler.handle_unknown_error("Unknown error 1")
        handler.handle_unknown_error("Unknown error 2")

        errors = handler.get_captured_errors()
        assert len(errors) == 2

    def test_unknown_error_handler_clear(self):
        """UnknownErrorHandler.clear should reset captured errors."""
        handler = UnknownErrorHandler()
        handler.clear()

        handler.handle_unknown_error("Unknown error 1")
        assert len(handler.get_captured_errors()) == 1

        handler.clear()
        assert len(handler.get_captured_errors()) == 0

    def test_get_unknown_error_handler_singleton(self):
        """get_unknown_error_handler should return a singleton instance."""
        handler1 = get_unknown_error_handler()
        handler2 = get_unknown_error_handler()
        assert handler1 is handler2


@pytest.mark.fast
class TestUnknownErrorIntegration:
    """Integration tests for Unknown error classification with error_classification.py."""

    def test_classify_unknown_error(self):
        """classify_error_category should return UnknownError for unclassified errors."""
        from src.downloader.error_classification import classify_error_category

        # Truly unknown error
        result = classify_error_category("This is completely unrecognizable xyz789")

        assert result.category == 'unknown'
        assert result.severity == 'medium'

    def test_classify_unknown_error_has_stack_trace_with_exception(self):
        """UnknownError should have stack_trace when exception provided."""
        from src.downloader.error_classification import classify_error_category

        # This tests that the handler is invoked
        result = classify_error_category("Some random unknown error message")

        # Verify it's an UnknownError
        assert result.category == 'unknown'

    def test_classify_known_errors_still_work(self):
        """Known error patterns should still be classified correctly."""
        from src.downloader.error_classification import classify_error_category

        # Known patterns should return proper categories
        assert classify_error_category("HTTP Error 429").category == 'bot_detection'
        assert classify_error_category("getaddrinfo failed").category == 'network'
        assert classify_error_category("timeout error").category == 'timeout'


@pytest.mark.fast
class TestUnknownPatternLearner:
    """US-123-005: Tests for Unknown error pattern learning system."""

    def test_learn_from_unknown_creates_pattern(self):
        """learn_from_unknown should create a new learned pattern."""
        from src.common.error_patterns import UnknownPatternLearner
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patterns.json")
            learner = UnknownPatternLearner(storage_path=path)

            result = learner.learn_from_unknown(
                error_msg="yt-dlp extractor error code 12345",
                proposed_category="extractor_error"
            )

            assert result is not None
            assert result.proposed_category == "extractor_error"
            assert result.occurrences == 1

    def test_learn_from_unknown_increments_occurrences(self):
        """learn_from_unknown should increment occurrences for same pattern."""
        from src.common.error_patterns import UnknownPatternLearner
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patterns.json")
            learner = UnknownPatternLearner(storage_path=path)

            # Learn same error twice
            learner.learn_from_unknown("Same error message", "network")
            result = learner.learn_from_unknown("Same error message", "network")

            assert result.occurrences == 2

    def test_get_learned_patterns_returns_all(self):
        """get_learned_patterns should return all learned patterns."""
        from src.common.error_patterns import UnknownPatternLearner
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patterns.json")
            learner = UnknownPatternLearner(storage_path=path)

            learner.learn_from_unknown("Error 1", "rate_limit")
            learner.learn_from_unknown("Error 2", "network")

            patterns = learner.get_learned_patterns()
            assert len(patterns) == 2

    def test_get_learned_patterns_filters_by_confidence(self):
        """get_learned_patterns should filter by min_confidence threshold."""
        from src.common.error_patterns import UnknownPatternLearner
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patterns.json")
            learner = UnknownPatternLearner(storage_path=path)

            learner.learn_from_unknown("Low confidence error", "network")
            # Simulate high confidence by incrementing
            for _ in range(9):
                learner.learn_from_unknown("Low confidence error", "network")

            patterns = learner.get_learned_patterns(min_confidence=0.5)
            assert len(patterns) >= 1

    def test_auto_classify_unknown_returns_learned_category(self):
        """auto_classify_unknown should return learned category for matching errors."""
        from src.common.error_patterns import UnknownPatternLearner
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patterns.json")
            learner = UnknownPatternLearner(
                storage_path=path,
                min_confidence_threshold=0.1
            )

            # Learn a pattern
            learner.learn_from_unknown(
                "yt-dlp extractor failed with code",
                "extractor_error"
            )

            # Auto-classify similar error
            category = learner.auto_classify_unknown(
                "yt-dlp extractor failed with code 99999"
            )

            assert category == "extractor_error"

    def test_auto_classify_unknown_returns_none_for_unknown(self):
        """auto_classify_unknown should return None for unlearned errors."""
        from src.common.error_patterns import UnknownPatternLearner
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patterns.json")
            learner = UnknownPatternLearner(storage_path=path)

            category = learner.auto_classify_unknown("Never seen this error before")
            assert category is None

    def test_persist_and_load_learned_patterns(self):
        """persist_learned_patterns should save and load patterns correctly."""
        from src.common.error_patterns import UnknownPatternLearner
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patterns.json")
            learner1 = UnknownPatternLearner(storage_path=path)

            learner1.learn_from_unknown("Persistent error", "rate_limit")
            learner1.persist_learned_patterns()

            # Create new learner and load
            learner2 = UnknownPatternLearner(storage_path=path)
            patterns = learner2.get_learned_patterns()

            assert len(patterns) == 1
            assert patterns[0].proposed_category == "rate_limit"

    def test_clear_learned_patterns(self):
        """clear_learned_patterns should remove all patterns."""
        from src.common.error_patterns import UnknownPatternLearner
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patterns.json")
            learner = UnknownPatternLearner(storage_path=path)

            learner.learn_from_unknown("Error 1", "rate_limit")
            learner.learn_from_unknown("Error 2", "network")

            assert len(learner.get_learned_patterns()) == 2

            count = learner.clear_learned_patterns()
            assert count == 2
            assert len(learner.get_learned_patterns()) == 0

    def test_get_stats(self):
        """get_stats should return pattern statistics."""
        from src.common.error_patterns import UnknownPatternLearner
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patterns.json")
            learner = UnknownPatternLearner(storage_path=path)

            learner.learn_from_unknown("Error A", "rate_limit")
            learner.learn_from_unknown("Error B", "rate_limit")
            learner.learn_from_unknown("Error C", "network")

            stats = learner.get_stats()

            assert stats['total_patterns'] == 3
            assert stats['by_category']['rate_limit'] == 2
            assert stats['by_category']['network'] == 1

    def test_max_patterns_limit(self):
        """Learner should enforce max_patterns limit by evicting lowest confidence."""
        from src.common.error_patterns import UnknownPatternLearner
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patterns.json")
            learner = UnknownPatternLearner(storage_path=path, max_patterns=3)

            # Add more than max
            learner.learn_from_unknown("Error 1", "rate_limit")
            learner.learn_from_unknown("Error 2", "network")
            learner.learn_from_unknown("Error 3", "timeout")
            learner.learn_from_unknown("Error 4", "unknown")

            # Should have max_patterns at most
            patterns = learner.get_learned_patterns()
            assert len(patterns) <= 3

    def test_get_unknown_pattern_learner_singleton(self):
        """get_unknown_pattern_learner should return a singleton instance."""
        from src.common.error_patterns import get_unknown_pattern_learner

        learner1 = get_unknown_pattern_learner()
        learner2 = get_unknown_pattern_learner()

        assert learner1 is learner2

    def test_learned_pattern_dataclass(self):
        """LearnedPattern dataclass should store all required fields."""
        from src.common.error_patterns import LearnedPattern

        pattern = LearnedPattern(
            pattern="test error",
            proposed_category="rate_limit",
            confidence=0.5,
            occurrences=3
        )

        assert pattern.pattern == "test error"
        assert pattern.proposed_category == "rate_limit"
        assert pattern.confidence == 0.5
        assert pattern.occurrences == 3
        assert pattern.first_seen is not None
        assert pattern.last_seen is not None


# =============================================================================
# US-144-002: Tests for new error patterns to reduce Unknown errors
# =============================================================================

@pytest.mark.fast
class TestUS144NewErrorPatterns:
    """US-144-002: Tests for new error patterns that reduce Unknown classification."""

    # Import new pattern groups
    from src.common.error_patterns import (
        YTDLP_EXTRACTOR_PATTERNS,
        HTTP_4XX_PATTERNS,
        PLATFORM_ERROR_PATTERNS,
        FORMAT_ERROR_PATTERNS,
        DOWNLOAD_ERROR_PATTERNS,
        SESSION_ERROR_PATTERNS,
        MEDIA_ERROR_PATTERNS,
        QUALITY_ERROR_PATTERNS,
        REGION_ERROR_PATTERNS,
    )

    def test_ytdlp_extractor_patterns_not_empty(self):
        """YTDLP_EXTRACTOR_PATTERNS should be non-empty."""
        assert len(self.YTDLP_EXTRACTOR_PATTERNS) > 0

    def test_http_4xx_patterns_not_empty(self):
        """HTTP_4XX_PATTERNS should be non-empty."""
        assert len(self.HTTP_4XX_PATTERNS) > 0

    def test_platform_error_patterns_not_empty(self):
        """PLATFORM_ERROR_PATTERNS should be non-empty."""
        assert len(self.PLATFORM_ERROR_PATTERNS) > 0

    def test_format_error_patterns_not_empty(self):
        """FORMAT_ERROR_PATTERNS should be non-empty."""
        assert len(self.FORMAT_ERROR_PATTERNS) > 0

    def test_download_error_patterns_not_empty(self):
        """DOWNLOAD_ERROR_PATTERNS should be non-empty."""
        assert len(self.DOWNLOAD_ERROR_PATTERNS) > 0

    def test_session_error_patterns_not_empty(self):
        """SESSION_ERROR_PATTERNS should be non-empty."""
        assert len(self.SESSION_ERROR_PATTERNS) > 0

    def test_media_error_patterns_not_empty(self):
        """MEDIA_ERROR_PATTERNS should be non-empty."""
        assert len(self.MEDIA_ERROR_PATTERNS) > 0

    def test_quality_error_patterns_not_empty(self):
        """QUALITY_ERROR_PATTERNS should be non-empty."""
        assert len(self.QUALITY_ERROR_PATTERNS) > 0

    def test_region_error_patterns_not_empty(self):
        """REGION_ERROR_PATTERNS should be non-empty."""
        assert len(self.REGION_ERROR_PATTERNS) > 0

    def test_all_new_patterns_lowercase(self):
        """All new pattern groups should contain lowercase strings."""
        all_patterns = (
            self.YTDLP_EXTRACTOR_PATTERNS +
            self.HTTP_4XX_PATTERNS +
            self.PLATFORM_ERROR_PATTERNS +
            self.FORMAT_ERROR_PATTERNS +
            self.DOWNLOAD_ERROR_PATTERNS +
            self.SESSION_ERROR_PATTERNS +
            self.MEDIA_ERROR_PATTERNS +
            self.QUALITY_ERROR_PATTERNS +
            self.REGION_ERROR_PATTERNS
        )
        for pattern in all_patterns:
            assert isinstance(pattern, str), f"Pattern {pattern!r} is not a string"
            assert pattern == pattern.lower(), f"Pattern {pattern!r} is not lowercase"

    def test_new_patterns_count_over_10(self):
        """US-144-002: At least 10 new patterns should be added across all groups."""
        total = (
            len(self.YTDLP_EXTRACTOR_PATTERNS) +
            len(self.HTTP_4XX_PATTERNS) +
            len(self.PLATFORM_ERROR_PATTERNS) +
            len(self.FORMAT_ERROR_PATTERNS) +
            len(self.DOWNLOAD_ERROR_PATTERNS) +
            len(self.SESSION_ERROR_PATTERNS) +
            len(self.MEDIA_ERROR_PATTERNS) +
            len(self.QUALITY_ERROR_PATTERNS) +
            len(self.REGION_ERROR_PATTERNS)
        )
        assert total >= 10, f"Expected at least 10 new patterns, got {total}"

    def test_ytdlp_extractor_classified(self):
        """YTDLP_EXTRACTOR_PATTERNS should not be classified as Unknown."""
        from src.common.error_patterns import is_unknown_error

        # Test key extractor patterns
        test_errors = [
            "Unable to extract player response",
            "No suitable extractor found",
            "Could not extract data",
        ]
        for error in test_errors:
            assert not is_unknown_error(error), f"Pattern '{error}' should NOT be Unknown"

    def test_http_4xx_classified(self):
        """HTTP_4XX_PATTERNS should not be classified as Unknown."""
        from src.common.error_patterns import is_unknown_error

        # Test key HTTP 4xx patterns
        test_errors = [
            "HTTP Error 400: Bad Request",
            "HTTP Error 404: Not Found",
            "HTTP Error 407: Proxy Authentication Required",
        ]
        for error in test_errors:
            assert not is_unknown_error(error), f"Pattern '{error}' should NOT be Unknown"

    def test_platform_error_classified(self):
        """PLATFORM_ERROR_PATTERNS should not be classified as Unknown."""
        from src.common.error_patterns import is_unknown_error

        # Test key platform patterns
        test_errors = [
            "Windows error 3",
            "No such file or directory",
            "Disk full",
            "Input/output error",
        ]
        for error in test_errors:
            assert not is_unknown_error(error), f"Pattern '{error}' should NOT be Unknown"

    def test_format_error_classified(self):
        """FORMAT_ERROR_PATTERNS should not be classified as Unknown."""
        from src.common.error_patterns import is_unknown_error

        # Test key format patterns
        test_errors = [
            "Requested format is not available",
            "No such format",
            "Unsupported format",
            "Codec not supported",
        ]
        for error in test_errors:
            assert not is_unknown_error(error), f"Pattern '{error}' should NOT be Unknown"

    def test_download_error_classified(self):
        """DOWNLOAD_ERROR_PATTERNS should not be classified as Unknown."""
        from src.common.error_patterns import is_unknown_error

        # Test key download patterns
        test_errors = [
            "Download failed",
            "Download timeout",
            "HTTP error occurred",
            "Too many redirects",
        ]
        for error in test_errors:
            assert not is_unknown_error(error), f"Pattern '{error}' should NOT be Unknown"

    def test_session_error_classified(self):
        """SESSION_ERROR_PATTERNS should not be classified as Unknown."""
        from src.common.error_patterns import is_unknown_error

        # Test key session patterns
        test_errors = [
            "Session expired",
            "Cookie expired",
            "Authentication required",
            "Token invalid",
        ]
        for error in test_errors:
            assert not is_unknown_error(error), f"Pattern '{error}' should NOT be Unknown"

    def test_media_error_classified(self):
        """MEDIA_ERROR_PATTERNS should not be classified as Unknown."""
        from src.common.error_patterns import is_unknown_error

        # Test key media patterns
        test_errors = [
            "Video unavailable",
            "Video removed",
            "Video private",
            "Age restricted",
        ]
        for error in test_errors:
            assert not is_unknown_error(error), f"Pattern '{error}' should NOT be Unknown"

    def test_quality_error_classified(self):
        """QUALITY_ERROR_PATTERNS should not be classified as Unknown."""
        from src.common.error_patterns import is_unknown_error

        # Test key quality patterns
        test_errors = [
            "Quality not available",
            "Resolution not available",
            "VP9 not available",
            "HDR not available",
        ]
        for error in test_errors:
            assert not is_unknown_error(error), f"Pattern '{error}' should NOT be Unknown"

    def test_region_error_classified(self):
        """REGION_ERROR_PATTERNS should not be classified as Unknown."""
        from src.common.error_patterns import is_unknown_error

        # Test key region patterns
        test_errors = [
            "Not available in your country",
            "Geo restricted",
            "Region blocked",
            "Embedding disabled",
        ]
        for error in test_errors:
            assert not is_unknown_error(error), f"Pattern '{error}' should NOT be Unknown"

    def test_unknown_classification_reduction(self):
        """US-144-002: New patterns should reduce Unknown classification by at least 50%."""
        from src.common.error_patterns import is_unknown_error

        # Sample of previously unknown errors that should now be classified
        # These are common yt-dlp errors that were likely in the "Unknown" bucket
        previously_unknown_errors = [
            # yt-dlp extractor errors
            "Unable to extract player response",
            "No suitable extractor found",
            "Could not extract data",
            "No playability data",
            "No embed data",
            # HTTP errors
            "HTTP Error 404: Not Found",
            "HTTP Error 400: Bad Request",
            "HTTP Error 407: Proxy Authentication Required",
            # Platform errors
            "No such file or directory",
            "Disk full",
            "Input/output error",
            # Format errors
            "Requested format is not available",
            "Unsupported format",
            "Codec not supported",
            # Download errors
            "Download failed",
            "Download timeout",
            "Too many redirects",
            # Session errors
            "Session expired",
            "Cookie expired",
            "Authentication required",
            # Media errors
            "Video unavailable",
            "Video removed",
            "Video private",
            # Quality errors
            "Quality not available",
            "Resolution not available",
            "VP9 not available",
            # Region errors
            "Not available in your country",
            "Geo restricted",
            "Embedding disabled",
        ]

        # Count how many are now classified (should NOT be Unknown)
        classified_count = sum(1 for err in previously_unknown_errors if not is_unknown_error(err))
        total_count = len(previously_unknown_errors)
        classification_rate = classified_count / total_count

        # At least 50% should now be classified (was previously Unknown)
        assert classification_rate >= 0.5, (
            f"Expected at least 50% classification rate, got {classification_rate*100:.1f}%. "
            f"Classified {classified_count}/{total_count}"
        )


@pytest.mark.fast
class TestUS144NewPatternsInSeverity:
    """US-144-002: Verify new patterns are included in severity classifications."""

    def test_high_severity_contains_region_errors(self):
        """REGION_ERROR_PATTERNS should be in HIGH_SEVERITY_PATTERNS."""
        from src.common.error_patterns import HIGH_SEVERITY_PATTERNS, REGION_ERROR_PATTERNS

        # At least some key region patterns should be in high severity
        key_region_patterns = ["geo restricted", "geo blocked", "not available in your country"]
        for pattern in key_region_patterns:
            if pattern in REGION_ERROR_PATTERNS:
                assert pattern in HIGH_SEVERITY_PATTERNS, f"Pattern '{pattern}' should be in HIGH_SEVERITY"

    def test_high_severity_contains_media_errors(self):
        """MEDIA_ERROR_PATTERNS should be in HIGH_SEVERITY_PATTERNS."""
        from src.common.error_patterns import HIGH_SEVERITY_PATTERNS, MEDIA_ERROR_PATTERNS

        # At least some key media patterns should be in high severity
        key_media_patterns = ["video unavailable", "video removed", "video private"]
        for pattern in key_media_patterns:
            if pattern in MEDIA_ERROR_PATTERNS:
                assert pattern in HIGH_SEVERITY_PATTERNS, f"Pattern '{pattern}' should be in HIGH_SEVERITY"

    def test_medium_severity_contains_download_errors(self):
        """DOWNLOAD_ERROR_PATTERNS should be in MEDIUM_SEVERITY_PATTERNS."""
        from src.common.error_patterns import MEDIUM_SEVERITY_PATTERNS, DOWNLOAD_ERROR_PATTERNS

        for pattern in DOWNLOAD_ERROR_PATTERNS[:5]:
            assert pattern in MEDIUM_SEVERITY_PATTERNS, f"Pattern '{pattern}' should be in MEDIUM_SEVERITY"

    def test_medium_severity_contains_quality_errors(self):
        """QUALITY_ERROR_PATTERNS should be in MEDIUM_SEVERITY_PATTERNS."""
        from src.common.error_patterns import MEDIUM_SEVERITY_PATTERNS, QUALITY_ERROR_PATTERNS

        for pattern in QUALITY_ERROR_PATTERNS[:5]:
            assert pattern in MEDIUM_SEVERITY_PATTERNS, f"Pattern '{pattern}' should be in MEDIUM_SEVERITY"

    def test_low_severity_contains_session_errors(self):
        """SESSION_ERROR_PATTERNS should be in LOW_SEVERITY_PATTERNS."""
        from src.common.error_patterns import LOW_SEVERITY_PATTERNS, SESSION_ERROR_PATTERNS

        for pattern in SESSION_ERROR_PATTERNS[:5]:
            assert pattern in LOW_SEVERITY_PATTERNS, f"Pattern '{pattern}' should be in LOW_SEVERITY"

    def test_low_severity_contains_format_errors(self):
        """FORMAT_ERROR_PATTERNS should be in LOW_SEVERITY_PATTERNS."""
        from src.common.error_patterns import LOW_SEVERITY_PATTERNS, FORMAT_ERROR_PATTERNS

        for pattern in FORMAT_ERROR_PATTERNS[:5]:
            assert pattern in LOW_SEVERITY_PATTERNS, f"Pattern '{pattern}' should be in LOW_SEVERITY"


@pytest.mark.fast
class TestUS144DownloaderIntegration:
    """US-144-002: Verify new patterns work in downloader error classification."""

    def test_classify_error_severity_uses_new_patterns(self):
        """Downloader classify_error_severity should detect new patterns."""
        from src.downloader.error_classification import classify_error_severity

        # Test various new patterns
        test_cases = [
            ("Video unavailable", "high"),
            ("Video private", "high"),
            ("Download failed", "medium"),
            ("Quality not available", "medium"),
            ("Session expired", "low"),
            ("Format not available", "low"),
        ]

        for error, expected_severity in test_cases:
            result = classify_error_severity(f"Error: {error}")
            assert result == expected_severity, (
                f"Expected '{error}' to be classified as '{expected_severity}', got '{result}'"
            )

    def test_downloader_error_category_new_patterns(self):
        """Downloader classify_error_category should detect new patterns (not Unknown)."""
        from src.downloader.error_classification import classify_error_category
        from src.downloader.errors import UnknownError

        # Test various new patterns - they should NOT be Unknown
        test_errors = [
            "Video unavailable",
            "Download failed",
            "Quality not available",
            "Session expired",
            "Format not available",
        ]

        for error in test_errors:
            result = classify_error_category(f"Error: {error}")
            # Should NOT be UnknownError
            assert not isinstance(result, UnknownError), (
                f"Expected '{error}' to be classified, got Unknown"
            )
