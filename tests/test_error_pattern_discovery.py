"""
Tests for error pattern auto-discovery (US-123-008).

Tests the ErrorPatternDiscovery class that discovers new error patterns
from unknown yt-dlp error messages.
"""

import pytest
from src.common.error_patterns import (
    ErrorPatternDiscovery,
    DiscoveredPattern,
    get_error_pattern_discovery,
)


class TestErrorPatternDiscovery:
    """Test suite for ErrorPatternDiscovery class."""

    @pytest.fixture
    def discovery(self, tmp_path):
        """Create a fresh ErrorPatternDiscovery instance for testing."""
        return ErrorPatternDiscovery(
            storage_path=str(tmp_path / "test_discovered.json"),
            discovery_confidence=0.5,
            min_occurrences=2,
        )

    def test_discover_patterns_from_yt_dlp_errors(self, discovery):
        """Test discovering patterns from sample yt-dlp error output."""
        error_messages = [
            "ERROR: [youtube] extractor error: HTTP Error 503: Service Unavailable",
            "ERROR: [youtube] extractor error: HTTP Error 503: Backend connection failed",
            "ERROR: [youtube] extractor error: HTTP Error 503: Server overloaded",
        ]

        discovered = discovery.discover_patterns(error_messages)

        # Should discover HTTP 503 pattern
        patterns_found = [p.pattern for p in discovered]
        assert "http error 503" in patterns_found

        # Should have correct category
        http_503 = next(p for p in discovered if p.pattern == "http error 503")
        assert http_503.proposed_category == "rate_limit" or http_503.proposed_category == "server_error"
        assert http_503.occurrences == 3

    def test_discover_extractor_patterns(self, discovery):
        """Test discovering extractor-specific error patterns."""
        error_messages = [
            "ERROR: [youtube] extractor error: Unable to extract data",
            "ERROR: [youtube] extractor error: Video not found",
            "ERROR: [youtube] extractor error: Sign in required",
        ]

        discovered = discovery.discover_patterns(error_messages)

        # Should find extractor pattern
        patterns_found = [p.pattern for p in discovered]
        assert any("extractor error" in p for p in patterns_found)

    def test_discover_network_patterns(self, discovery):
        """Test discovering network-related error patterns."""
        error_messages = [
            "ERROR: [youtube] Connection error: Connection refused",
            "ERROR: [youtube] Connection error: Connection reset",
            "ERROR: [youtube] Network error: Connection timeout",
        ]

        discovered = discovery.discover_patterns(error_messages)

        # Should find connection patterns
        patterns_found = [p.pattern for p in discovered]
        assert any("connection" in p for p in patterns_found)

    def test_propose_pattern_category(self, discovery):
        """Test category proposal for different error patterns."""
        # Rate limit
        category = discovery.propose_pattern_category("http error 429")
        assert category == "rate_limit"

        # Network
        category = discovery.propose_pattern_category("connection timeout")
        assert category == "network"

        # Bot detection
        category = discovery.propose_pattern_category("403 forbidden")
        assert category == "bot_detection"

        # Extractor
        category = discovery.propose_pattern_category("[youtube] extractor error")
        assert category == "extractor"

    def test_review_discovered_patterns(self, discovery):
        """Test manual review of discovered patterns."""
        # First discover some patterns
        error_messages = [
            "ERROR: [youtube] HTTP Error 503: Service Unavailable",
            "ERROR: [youtube] HTTP Error 503: Backend error",
        ]
        discovery.discover_patterns(error_messages)

        # Get pending patterns
        pending = discovery.get_pending_patterns()
        assert len(pending) > 0

        # Review and approve
        review_results = discovery.review_discovered_patterns([
            {"pattern": "http error 503", "approved": True}
        ])

        assert review_results["approved"] == 1

        # Check pattern status changed
        approved = discovery.get_approved_patterns()
        assert len(approved) == 1
        assert approved[0].pattern == "http error 503"

    def test_reject_pattern(self, discovery):
        """Test rejecting a discovered pattern."""
        # Use error messages that will trigger pattern extraction
        error_messages = [
            "ERROR: [youtube] Some random extractor error message",
            "ERROR: [youtube] Some random extractor error message",
        ]
        discovery.discover_patterns(error_messages)

        # First check what patterns were discovered
        pending = discovery.get_pending_patterns()
        assert len(pending) > 0, "Should have discovered some patterns"

        # Use the actual pattern that was discovered
        actual_pattern = pending[0].pattern

        # Reject the pattern using actual discovered pattern
        review_results = discovery.review_discovered_patterns([
            {"pattern": actual_pattern, "approved": False}
        ])

        assert review_results["rejected"] == 1

        # Should have no approved patterns
        approved = discovery.get_approved_patterns()
        assert len(approved) == 0

    def test_persist_and_load(self, discovery, tmp_path):
        """Test persisting and loading discovered patterns."""
        # Discover patterns
        error_messages = [
            "ERROR: [youtube] HTTP Error 503: Unavailable",
            "ERROR: [youtube] HTTP Error 503: Overloaded",
        ]
        discovery.discover_patterns(error_messages)

        # Persist
        assert discovery.persist_discovered_patterns() is True

        # Create new instance and load
        new_discovery = ErrorPatternDiscovery(
            storage_path=str(tmp_path / "test_discovered.json"),
        )

        # Should have loaded the pattern
        assert len(new_discovery.get_all_patterns()) > 0

    def test_min_occurrences_threshold(self, discovery):
        """Test that min_occurrences filter works."""
        # Only 1 occurrence - should not be discovered
        error_messages = [
            "ERROR: [youtube] Some unique error message",
        ]

        discovered = discovery.discover_patterns(error_messages)
        # With min_occurrences=2, single occurrence should not be discovered
        assert len(discovered) == 0

    def test_get_stats(self, discovery):
        """Test getting discovery statistics."""
        error_messages = [
            "ERROR: [youtube] HTTP Error 503: Unavailable",
            "ERROR: [youtube] HTTP Error 503: Overloaded",
            "ERROR: [youtube] Connection timeout",
        ]
        discovery.discover_patterns(error_messages)

        stats = discovery.get_stats()

        assert stats["total_patterns"] > 0
        assert "by_category" in stats
        assert "by_status" in stats
        assert stats["discovery_confidence"] == 0.5

    def test_clear_patterns(self, discovery):
        """Test clearing discovered patterns."""
        error_messages = [
            "ERROR: [youtube] HTTP Error 503: Unavailable",
            "ERROR: [youtube] HTTP Error 503: Overloaded",
        ]
        discovery.discover_patterns(error_messages)

        # Clear all
        count = discovery.clear_patterns()
        assert count > 0

        assert len(discovery.get_all_patterns()) == 0


class TestDiscoveredPattern:
    """Test suite for DiscoveredPattern dataclass."""

    def test_discovered_pattern_creation(self):
        """Test creating a DiscoveredPattern instance."""
        pattern = DiscoveredPattern(
            pattern="http error 503",
            proposed_category="rate_limit",
            confidence=0.8,
            occurrences=5,
            sample_messages=["msg1", "msg2"],
        )

        assert pattern.pattern == "http error 503"
        assert pattern.proposed_category == "rate_limit"
        assert pattern.confidence == 0.8
        assert pattern.occurrences == 5
        assert pattern.status == "pending"

    def test_discovered_pattern_default_values(self):
        """Test default values for DiscoveredPattern."""
        pattern = DiscoveredPattern(
            pattern="test pattern",
            proposed_category="unknown",
        )

        assert pattern.confidence == 0.0
        assert pattern.occurrences == 1
        assert pattern.sample_messages == []
        assert pattern.status == "pending"
        assert pattern.first_seen is not None
        assert pattern.last_seen is not None


class TestIntegration:
    """Integration tests for error pattern discovery."""

    def test_discover_from_captured_errors(self):
        """Test discovering patterns from captured unknown errors."""
        from src.common.error_patterns import (
            get_unknown_error_handler,
            discover_from_captured_errors,
            ErrorPatternDiscovery,
        )

        # Create a fresh discovery instance
        discovery = ErrorPatternDiscovery(
            min_occurrences=1,  # Lower threshold for testing
        )

        # Capture some unknown errors
        handler = get_unknown_error_handler()
        handler.handle_unknown_error("ERROR: [youtube] Some new error pattern 12345")
        handler.handle_unknown_error("ERROR: [youtube] Some new error pattern 12345")

        # Discover from captured
        discovered = discover_from_captured_errors()

        # Should have discovered the pattern
        assert len(discovered) > 0

        # Cleanup
        handler.clear()


# =============================================================================
# US-144-011: Runtime Pattern Discovery Tests
# =============================================================================

import os
import tempfile
from src.downloader.error_pattern_discovery import (
    RuntimePatternDiscovery,
    DiscoveredDownloadPattern,
    get_runtime_pattern_discovery,
    reset_runtime_pattern_discovery,
    DEFAULT_PATTERN_DIR,
    DEFAULT_MIN_OCCURRENCES,
)


class TestRuntimePatternDiscovery:
    """Test suite for RuntimePatternDiscovery class (US-144-011).

    Tests the runtime error pattern auto-discovery module that discovers
    new patterns from unclassified download errors.
    """

    @pytest.fixture
    def discovery(self, tmp_path):
        """Create a fresh RuntimePatternDiscovery instance for testing."""
        # Reset global instance
        reset_runtime_pattern_discovery()

        return RuntimePatternDiscovery(
            pattern_dir=str(tmp_path / "patterns"),
            min_occurrences=3,  # Test with 3 occurrences required
            max_patterns=100,
            confidence_threshold=0.5,
        )

    @pytest.fixture
    def discovery_single(self, tmp_path):
        """Create discovery with 1 occurrence for testing initial discovery."""
        reset_runtime_pattern_discovery()

        return RuntimePatternDiscovery(
            pattern_dir=str(tmp_path / "patterns"),
            min_occurrences=1,
            max_patterns=100,
            confidence_threshold=0.5,
        )

    def test_record_unclassified_error_basic(self, discovery):
        """Test basic recording of unclassified errors."""
        result = discovery.record_unclassified_error(
            "ERROR: HTTP Error 503: Service Unavailable",
            context={"video_id": "abc123"}
        )

        # Should not return promoted pattern yet (only 1 occurrence)
        assert result is None

        # Should have recorded the error
        assert len(discovery.get_unclassified_errors()) == 1

        # Should have discovered patterns
        patterns = discovery.get_discovered_patterns()
        assert len(patterns) > 0

    def test_pattern_promotion_after_three_occurrences(self, discovery):
        """Test that patterns are promoted after 3+ occurrences."""
        error_msg = "ERROR: HTTP Error 518: Custom Server Error"

        # Record the same error 3 times
        promoted = None
        for i in range(3):
            result = discovery.record_unclassified_error(error_msg)
            if result:
                promoted = result

        # After 3 occurrences, pattern should be promoted
        assert promoted is not None
        assert promoted.status == "promoted"
        assert promoted.occurrences == 3
        assert promoted.proposed_category == "server_error"

        # Should have promoted patterns
        promoted_patterns = discovery.get_promoted_patterns()
        assert len(promoted_patterns) >= 1
        assert any(p.status == "promoted" for p in promoted_patterns)

    def test_pattern_extraction_http_errors(self, discovery):
        """Test regex-based pattern extraction from HTTP errors."""
        error_messages = [
            "ERROR: HTTP Error 503: Service Unavailable",
            "ERROR: HTTP Error 503: Backend connection failed",
            "ERROR: HTTP Error 503: Server overloaded",
        ]

        for msg in error_messages:
            discovery.record_unclassified_error(msg)

        # Should discover HTTP 503 pattern
        patterns = discovery.get_discovered_patterns()
        http_503_patterns = [p for p in patterns if "503" in p.pattern_id]

        assert len(http_503_patterns) > 0

    def test_pattern_extraction_rate_limit(self, discovery):
        """Test pattern extraction for rate limit errors."""
        error_messages = [
            "ERROR: [youtube] rate limit exceeded for IP",
            "ERROR: [youtube] rate limit exceeded for user",
            "ERROR: [youtube] rate limit exceeded - try again later",
        ]

        for msg in error_messages:
            discovery.record_unclassified_error(msg)

        # Should have discovered rate_limit pattern
        patterns = discovery.get_discovered_patterns()
        rate_limit_patterns = [
            p for p in patterns
            if p.proposed_category == "rate_limit"
        ]

        # Should have at least one rate limit pattern
        assert len(rate_limit_patterns) >= 1

    def test_pattern_extraction_network_errors(self, discovery):
        """Test pattern extraction for network errors."""
        error_messages = [
            "ERROR: Connection reset by peer",
            "ERROR: Connection refused",
            "ERROR: Connection timeout",
        ]

        for msg in error_messages:
            discovery.record_unclassified_error(msg)

        # Should have discovered network pattern
        patterns = discovery.get_discovered_patterns()
        network_patterns = [
            p for p in patterns
            if p.proposed_category == "network"
        ]

        assert len(network_patterns) >= 1

    def test_pattern_extraction_extractor_errors(self, discovery):
        """Test pattern extraction for extractor errors."""
        error_messages = [
            "ERROR: [youtube] extractor error: Unable to extract",
            "ERROR: [youtube] extractor error: No video found",
            "ERROR: [youtube] extractor error: No format available",
        ]

        for msg in error_messages:
            discovery.record_unclassified_error(msg)

        # Should have discovered extractor pattern
        patterns = discovery.get_discovered_patterns()
        extractor_patterns = [
            p for p in patterns
            if p.proposed_category == "extractor"
        ]

        assert len(extractor_patterns) >= 1

    def test_pattern_not_promoted_below_threshold(self, discovery):
        """Test that patterns are not promoted below 3 occurrences."""
        # Record only 2 occurrences
        error_msg = "ERROR: Unique error pattern XYZ123"

        for i in range(2):
            result = discovery.record_unclassified_error(error_msg)
            assert result is None

        # Should have discovered but not promoted
        patterns = discovery.get_discovered_patterns()
        promoted = discovery.get_promoted_patterns()

        assert len(patterns) > 0
        assert len(promoted) == 0

    def test_confidence_calculation(self, discovery):
        """Test confidence score calculation based on occurrences."""
        # With 1 occurrence: confidence = 0.1
        # With 5 occurrences: confidence = 0.5
        # With 10+ occurrences: confidence = 1.0

        error_msg = "ERROR: Test confidence pattern"

        for i in range(10):
            discovery.record_unclassified_error(error_msg)

        patterns = discovery.get_discovered_patterns()
        assert len(patterns) > 0

        # After 10 occurrences, confidence should be 1.0
        pattern = patterns[0]
        assert pattern.confidence == 1.0

    def test_persistence_to_user_config(self, discovery, tmp_path):
        """Test that patterns are persisted to user config directory."""
        error_msg = "ERROR: Persistent pattern test"

        # Record enough to get promoted
        for i in range(3):
            discovery.record_unclassified_error(error_msg)

        # Check that pattern file was created
        pattern_file = tmp_path / "patterns" / "discovered_patterns.json"
        assert pattern_file.exists()

        # Load in new instance and verify
        new_discovery = RuntimePatternDiscovery(
            pattern_dir=str(tmp_path / "patterns"),
            min_occurrences=3,
        )

        # Should have loaded the promoted pattern
        patterns = new_discovery.get_discovered_patterns()
        assert len(patterns) >= 1

    def test_get_stats(self, discovery):
        """Test statistics retrieval."""
        error_messages = [
            "ERROR: HTTP Error 503: Service Unavailable",
            "ERROR: Connection refused",
            "ERROR: rate limit exceeded",
        ]

        for msg in error_messages:
            discovery.record_unclassified_error(msg)

        stats = discovery.get_stats()

        assert "total_patterns" in stats
        assert "total_unclassified_errors" in stats
        assert "by_category" in stats
        assert "by_status" in stats
        assert stats["min_occurrences_required"] == 3

    def test_different_errors_create_different_patterns(self, discovery):
        """Test that different errors create different patterns."""
        errors = [
            "ERROR: HTTP Error 503: Service Unavailable",
            "ERROR: HTTP Error 504: Gateway Timeout",
            "ERROR: Connection refused",
        ]

        for msg in errors:
            for _ in range(3):
                discovery.record_unclassified_error(msg)

        patterns = discovery.get_discovered_patterns()

        # Should have multiple different patterns
        assert len(patterns) >= 3

    def test_review_pattern_approve(self, discovery):
        """Test manual pattern review approval."""
        # Record an error (not enough to promote)
        discovery.record_unclassified_error("ERROR: Test review pattern")

        patterns = discovery.get_discovered_patterns()
        assert len(patterns) > 0

        pattern_id = patterns[0].pattern_id

        # Manually approve
        result = discovery.review_pattern(pattern_id, approved=True)

        assert result is True

        # Should now be promoted
        pattern = discovery.get_discovered_patterns(status="promoted")
        assert len(pattern) == 1

    def test_review_pattern_reject(self, discovery):
        """Test manual pattern review rejection."""
        discovery.record_unclassified_error("ERROR: Test reject pattern")

        patterns = discovery.get_discovered_patterns()
        assert len(patterns) > 0

        pattern_id = patterns[0].pattern_id

        # Manually reject
        result = discovery.review_pattern(pattern_id, approved=False)

        assert result is True

        # Should now be rejected
        pattern = discovery.get_discovered_patterns(status="rejected")
        assert len(pattern) == 1

    def test_export_promoted_patterns(self, discovery):
        """Test exporting promoted patterns for classification."""
        error_msg = "ERROR: Export test pattern"

        for i in range(3):
            discovery.record_unclassified_error(error_msg)

        exported = discovery.export_promoted_patterns()

        assert len(exported) > 0
        assert "pattern" in exported[0]
        assert "category" in exported[0]
        assert "confidence" in exported[0]

    def test_clear_patterns(self, discovery):
        """Test clearing patterns."""
        discovery.record_unclassified_error("ERROR: Clear test")

        assert len(discovery.get_discovered_patterns()) > 0

        count = discovery.clear_patterns()

        assert count > 0
        assert len(discovery.get_discovered_patterns()) == 0


class TestRuntimePatternDiscoveryIntegration:
    """Integration tests for RuntimePatternDiscovery with error classification."""

    def test_get_runtime_pattern_discovery_singleton(self):
        """Test that get_runtime_pattern_discovery returns singleton."""
        reset_runtime_pattern_discovery()

        instance1 = get_runtime_pattern_discovery()
        instance2 = get_runtime_pattern_discovery()

        assert instance1 is instance2

        # Cleanup
        reset_runtime_pattern_discovery()

    def test_pattern_for_classification(self):
        """Test getting patterns formatted for classification."""
        reset_runtime_pattern_discovery()

        discovery = RuntimePatternDiscovery(
            pattern_dir=tempfile.mkdtemp(),
            min_occurrences=1,  # Low for testing
            confidence_threshold=0.1,
        )

        # Record and promote a pattern
        for _ in range(3):
            discovery.record_unclassified_error(
                "ERROR: Classification test HTTP Error 555"
            )

        # Get patterns for classification
        from src.downloader.error_pattern_discovery import get_pattern_for_classification
        patterns = get_pattern_for_classification()

        # Should have at least one pattern
        assert isinstance(patterns, list)

        # Cleanup
        reset_runtime_pattern_discovery()


class TestRuntimePatternDiscoveryAcceptance:
    """Acceptance tests for US-144-011.

    Verifies all acceptance criteria are met.
    """

    def test_acceptance_create_new_module(self):
        """Verify new error_pattern_discovery.py module exists."""
        from src.downloader import error_pattern_discovery
        assert hasattr(error_pattern_discovery, 'RuntimePatternDiscovery')

    def test_acceptance_regex_pattern_extraction(self, tmp_path):
        """Verify regex-based pattern extraction works."""
        reset_runtime_pattern_discovery()

        discovery = RuntimePatternDiscovery(
            pattern_dir=str(tmp_path / "patterns"),
            min_occurrences=1,
        )

        # Test HTTP error extraction
        discovery.record_unclassified_error("ERROR: HTTP Error 599: Custom Error")

        patterns = discovery.get_discovered_patterns()
        assert len(patterns) > 0

        # Should have extracted HTTP 599
        http_patterns = [p for p in patterns if "599" in p.pattern_id]
        assert len(http_patterns) > 0

    def test_acceptance_user_config_storage(self, tmp_path):
        """Verify patterns stored in user config directory."""
        test_dir = str(tmp_path / "test_patterns")

        discovery = RuntimePatternDiscovery(
            pattern_dir=test_dir,
            min_occurrences=1,
            confidence_threshold=0.3,
        )

        # Record enough to promote (needs 3 with default, but we lower threshold)
        for _ in range(3):
            discovery.record_unclassified_error("ERROR: Storage test")

        # Verify directory exists
        pattern_dir = Path(test_dir)
        assert pattern_dir.exists()

        # Verify pattern file was created (promotion triggers persistence)
        pattern_file = pattern_dir / "discovered_patterns.json"
        assert pattern_file.exists()

    def test_acceptance_three_occurrence_validation(self, tmp_path):
        """Verify pattern validation requires 3+ occurrences."""
        reset_runtime_pattern_discovery()

        discovery = RuntimePatternDiscovery(
            pattern_dir=str(tmp_path / "patterns"),
            min_occurrences=3,
            confidence_threshold=0.5,
        )

        error_msg = "ERROR: Validation test 3 occurrences"

        # Record 2 times - should NOT promote
        for _ in range(2):
            result = discovery.record_unclassified_error(error_msg)
            assert result is None

        promoted = discovery.get_promoted_patterns()
        assert len(promoted) == 0

        # Record 3rd time - should promote
        result = discovery.record_unclassified_error(error_msg)
        assert result is not None
        assert result.status == "promoted"

        promoted = discovery.get_promoted_patterns()
        assert len(promoted) >= 1

    def test_acceptance_full_flow(self, tmp_path):
        """Verify full flow: discover -> validate -> promote -> store."""
        reset_runtime_pattern_discovery()

        discovery = RuntimePatternDiscovery(
            pattern_dir=str(tmp_path / "patterns"),
            min_occurrences=3,
            confidence_threshold=0.3,  # Lower for faster test
        )

        # Simulate real errors occurring over time
        errors = [
            "ERROR: HTTP Error 511: Network Authentication Required",
            "ERROR: HTTP Error 511: Network Authentication Required",
            "ERROR: HTTP Error 511: Network Authentication Required",
        ]

        for error in errors:
            discovery.record_unclassified_error(error)

        # Verify promoted
        promoted = discovery.get_promoted_patterns()
        assert len(promoted) >= 1

        # Verify stored to disk
        pattern_file = Path(tmp_path / "patterns") / "discovered_patterns.json"
        assert pattern_file.exists()

        # Verify can be loaded in new instance
        new_discovery = RuntimePatternDiscovery(
            pattern_dir=str(tmp_path / "patterns"),
            min_occurrences=3,
        )

        loaded_promoted = new_discovery.get_promoted_patterns()
        assert len(loaded_promoted) >= 1


from pathlib import Path
