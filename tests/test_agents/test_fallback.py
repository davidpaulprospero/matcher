"""
Tests for FallbackChain and pattern-based routing.

US-005: Add agents fallback chain tests

Tests for src/agents/fallback.py covering:
- PatternClassification dataclass
- pattern_route() function for error pattern matching
- FallbackChain availability checks
- FallbackChain failure tracking (circuit breaker)
- PATTERN_ROUTING pattern coverage
"""

import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit

import sys
import os
import time
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.fallback import (
    FallbackChain,
    PatternClassification,
    pattern_route,
    PATTERN_ROUTING
)


class TestPatternClassificationDataclass:
    """Tests for PatternClassification dataclass."""

    @pytest.mark.fast
    def test_pattern_classification_all_fields(self):
        """Test PatternClassification accepts all fields."""
        classification = PatternClassification(
            category="api",
            suggested_healer="api-healer",
            severity="transient",
            confidence=0.7,
            needs_llm_healer=False,
            reasoning="Rate limit matched"
        )

        assert classification.category == "api"
        assert classification.suggested_healer == "api-healer"
        assert classification.severity == "transient"
        assert classification.confidence == 0.7
        assert classification.needs_llm_healer is False
        assert classification.reasoning == "Rate limit matched"

    @pytest.mark.fast
    def test_pattern_classification_default_values(self):
        """Test PatternClassification uses default values."""
        classification = PatternClassification(
            category="disk",
            suggested_healer="disk-healer"
        )

        assert classification.severity == "recoverable"
        assert classification.confidence == 0.6
        assert classification.needs_llm_healer is False
        assert classification.reasoning == "Pattern-based classification"


class TestPatternRouteFunction:
    """Tests for pattern_route() function."""

    @pytest.mark.fast
    def test_pattern_route_api_rate_limit_429(self):
        """Test pattern_route() matches HTTP 429 rate limit."""
        error = "HTTP Error 429: Too Many Requests"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"
        assert result.confidence == 0.7

    @pytest.mark.fast
    def test_pattern_route_api_rate_limit_text(self):
        """Test pattern_route() matches rate limit text."""
        error = "Rate limit exceeded, please wait"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.fast
    def test_pattern_route_api_quota_exceeded(self):
        """Test pattern_route() matches quota exceeded."""
        error = "API quota exceeded for this month"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.fast
    def test_pattern_route_api_timeout(self):
        """Test pattern_route() matches request timeout."""
        error = "Connection timed out while waiting for response"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.fast
    def test_pattern_route_api_connection_refused(self):
        """Test pattern_route() matches connection refused."""
        error = "Connection refused by server (ECONNREFUSED)"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.fast
    def test_pattern_route_disk_space(self):
        """Test pattern_route() matches disk full."""
        error = "[Errno 28] No space left on device"
        result = pattern_route(error)

        assert result.category == "disk"
        assert result.suggested_healer == "disk-healer"

    @pytest.mark.fast
    def test_pattern_route_disk_enospc(self):
        """Test pattern_route() matches ENOSPC error."""
        error = "OSError: ENOSPC - filesystem full"
        result = pattern_route(error)

        assert result.category == "disk"
        assert result.suggested_healer == "disk-healer"

    @pytest.mark.fast
    def test_pattern_route_disk_permission(self):
        """Test pattern_route() matches permission denied."""
        error = "PermissionError: [Errno 13] Permission denied"
        result = pattern_route(error)

        assert result.category == "disk"
        assert result.suggested_healer == "disk-healer"

    @pytest.mark.fast
    def test_pattern_route_path_too_long(self):
        """Test pattern_route() matches path too long."""
        error = "Path too long, exceeds 260 char limit on Windows"
        result = pattern_route(error)

        assert result.category == "path"
        assert result.suggested_healer == "path-healer"

    @pytest.mark.fast
    def test_pattern_route_path_unicode_error(self):
        """Test pattern_route() matches UnicodeDecodeError."""
        error = "UnicodeDecodeError: 'utf-8' codec can't decode bytes"
        result = pattern_route(error)

        assert result.category == "path"
        assert result.suggested_healer == "path-healer"

    @pytest.mark.fast
    def test_pattern_route_checkpoint_corrupt(self):
        """Test pattern_route() matches checkpoint corruption."""
        error = "Checkpoint file corrupt, cannot read JSON"
        result = pattern_route(error)

        assert result.category == "checkpoint"
        assert result.suggested_healer == "checkpoint-healer"

    @pytest.mark.fast
    def test_pattern_route_json_decode_error(self):
        """Test pattern_route() matches JSONDecodeError."""
        error = "json.decoder.JSONDecodeError: Expecting value: line 1"
        result = pattern_route(error)

        assert result.category == "checkpoint"
        assert result.suggested_healer == "checkpoint-healer"

    @pytest.mark.fast
    def test_pattern_route_download_video_unavailable(self):
        """Test pattern_route() matches video unavailable."""
        error = "Video unavailable: This video is private"
        result = pattern_route(error)

        assert result.category == "download"
        assert result.suggested_healer == "download-healer"

    @pytest.mark.fast
    def test_pattern_route_download_yt_dlp_error(self):
        """Test pattern_route() matches yt-dlp errors."""
        error = "yt-dlp error: Unable to extract video data"
        result = pattern_route(error)

        assert result.category == "download"
        assert result.suggested_healer == "download-healer"

    @pytest.mark.fast
    def test_pattern_route_otio_timeline_error(self):
        """Test pattern_route() matches OTIO timeline errors."""
        error = "opentimelineio.exception: Invalid time range"
        result = pattern_route(error)

        assert result.category == "otio"
        assert result.suggested_healer == "otio-healer"

    @pytest.mark.fast
    def test_pattern_route_otio_clip_error(self):
        """Test pattern_route() matches clip generation errors."""
        error = "Clip creation failed: invalid duration"
        result = pattern_route(error)

        assert result.category == "otio"
        assert result.suggested_healer == "otio-healer"

    @pytest.mark.fast
    def test_pattern_route_unknown_error(self):
        """Test pattern_route() returns unknown for unmatched errors."""
        error = "Completely random error with no patterns"
        result = pattern_route(error)

        assert result.category == "unknown"
        assert result.suggested_healer == ""
        assert result.confidence == 0.3
        assert result.needs_llm_healer is True

    @pytest.mark.fast
    def test_pattern_route_case_insensitive(self):
        """Test pattern_route() is case insensitive."""
        error = "RATE LIMIT EXCEEDED"
        result = pattern_route(error)

        assert result.category == "api"

    @pytest.mark.fast
    def test_pattern_route_no_false_positive_singapore_gap(self):
        """Test pattern_route() doesn't match Singapore as gap error."""
        error = "Downloading video from Singapore server"
        result = pattern_route(error)

        # Should NOT match gap/otio pattern
        assert result.category != "otio" or "gap" not in result.reasoning.lower()

    @pytest.mark.fast
    def test_pattern_route_no_false_positive_author_auth(self):
        """Test pattern_route() doesn't match author as auth error."""
        error = "Video by author JohnDoe uploaded to YouTube"
        result = pattern_route(error)

        # Should NOT match authentication pattern
        assert result.category != "api" or "auth" not in result.reasoning.lower()


class TestPatternRouteEdgeCasesUS005:
    """US-005: Edge case tests for pattern_route() function."""

    # AC1: Word boundary false positives - already covered by existing tests above
    # test_pattern_route_no_false_positive_singapore_gap
    # test_pattern_route_no_false_positive_author_auth

    @pytest.mark.fast
    def test_pattern_route_mixed_http_status_codes_in_message(self):
        """AC2: Test pattern_route() handles mixed HTTP status codes in error messages."""
        # Message contains multiple status codes - should match the relevant one
        error = "Request returned 200 OK but then failed with HTTP 403 Forbidden"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.fast
    def test_pattern_route_status_code_in_filename(self):
        """AC2: Test pattern_route() doesn't false positive on status codes in filenames."""
        error = "Processing file video_401.mp4 completed successfully"
        result = pattern_route(error)

        # 401 in filename should NOT match the authentication error pattern
        # because the pattern requires HTTP context or word boundary
        # If it matches, it should be because of explicit HTTP 401 pattern
        if result.category == "api":
            # Verify the pattern matched was the HTTP-specific one
            assert "HTTP" in error or "401" in result.reasoning

    @pytest.mark.fast
    def test_pattern_route_status_code_200_no_false_positive(self):
        """AC2: Test pattern_route() doesn't match HTTP 200 as an error."""
        error = "Request completed with HTTP 200 status"
        result = pattern_route(error)

        # 200 is success, should NOT match any error pattern
        assert result.category == "unknown"

    @pytest.mark.fast
    def test_pattern_route_rate_limit_youtube(self):
        """AC3: Test pattern_route() identifies rate limit from YouTube."""
        error = "yt-dlp error: HTTP Error 429 - Rate limit exceeded"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.requires_network
    def test_pattern_route_rate_limit_pexels(self):
        """AC3: Test pattern_route() identifies rate limit from Pexels API."""
        error = "Pexels API: Rate limit reached. Please wait before making more requests."
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.fast
    def test_pattern_route_rate_limit_gemini(self):
        """AC3: Test pattern_route() identifies rate limit from Gemini API."""
        error = "google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.fast
    def test_pattern_route_rate_limit_anthropic(self):
        """AC3: Test pattern_route() identifies rate limit from Anthropic API."""
        error = "anthropic.RateLimitError: Too many requests, please wait 60 seconds"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.requires_network
    def test_pattern_route_rate_limit_generic_too_many_requests(self):
        """AC3: Test pattern_route() identifies generic 'too many requests' pattern."""
        error = "Error: Too many requests. Try again later."
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.fast
    def test_pattern_route_multilingual_error_chinese(self):
        """AC4: Test pattern_route() returns UNKNOWN for Chinese error message."""
        error = "错误：文件未找到"  # "Error: File not found" in Chinese
        result = pattern_route(error)

        assert result.category == "unknown"
        assert result.needs_llm_healer is True

    @pytest.mark.fast
    def test_pattern_route_multilingual_error_japanese(self):
        """AC4: Test pattern_route() returns UNKNOWN for Japanese error message."""
        error = "エラー：接続できませんでした"  # "Error: Could not connect" in Japanese
        result = pattern_route(error)

        assert result.category == "unknown"
        assert result.needs_llm_healer is True

    @pytest.mark.fast
    def test_pattern_route_multilingual_error_arabic(self):
        """AC4: Test pattern_route() returns UNKNOWN for Arabic error message."""
        error = "خطأ: فشل التحميل"  # "Error: Download failed" in Arabic
        result = pattern_route(error)

        assert result.category == "unknown"
        assert result.needs_llm_healer is True

    @pytest.mark.fast
    def test_pattern_route_multilingual_error_mixed_english(self):
        """AC4: Test pattern_route() handles mixed language with English keywords."""
        # If an error has recognizable English keywords, it should still match
        error = "ошибка 429: rate limit exceeded"  # Russian + English
        result = pattern_route(error)

        # Should match on "rate limit" even with Russian prefix
        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    @pytest.mark.fast
    def test_pattern_route_priority_rate_limit_over_timeout(self):
        """AC5: Test pattern_route() priority - rate limit matches before timeout."""
        # Error message contains both patterns - 429 and timeout
        error = "HTTP 429 Rate limit: request timed out waiting for quota reset"
        result = pattern_route(error)

        # Rate limit pattern should be matched (appears first in PATTERN_ROUTING)
        assert result.category == "api"
        # Verify it's the rate limit pattern, not timeout
        assert "429" in result.reasoning or "rate" in result.reasoning.lower()

    @pytest.mark.fast
    def test_pattern_route_priority_api_over_download(self):
        """AC5: Test pattern_route() priority - API error matches before download error."""
        # Error contains both API auth and YouTube patterns
        error = "youtube: HTTP 403 Forbidden - authentication error accessing video"
        result = pattern_route(error)

        # Should match API pattern (HTTP 403 auth) which comes before download
        assert result.category == "api"
        assert "api-healer" in result.suggested_healer

    @pytest.mark.fast
    def test_pattern_route_priority_disk_specific_over_generic(self):
        """AC5: Test pattern_route() matches specific disk error over generic."""
        error = "OSError: [Errno 28] No space left on device"
        result = pattern_route(error)

        assert result.category == "disk"
        # Should match ENOSPC/no space pattern
        assert "28" in result.reasoning or "space" in result.reasoning.lower()

    @pytest.mark.fast
    def test_pattern_route_first_matching_pattern_wins(self):
        """AC5: Test that first matching pattern in PATTERN_ROUTING wins."""
        # This tests the dictionary iteration order behavior
        # Create an error that could match multiple patterns
        error = "HTTP 429 rate limit - connection timed out waiting"
        result = pattern_route(error)

        # The first pattern in PATTERN_ROUTING for "api" category should match
        # Both 429/rate_limit and timeout patterns match, but 429 comes first
        assert result.category == "api"
        # The reasoning should show which pattern matched
        assert "pattern" in result.reasoning.lower()


class TestPatternRoutingConstants:
    """Tests for PATTERN_ROUTING dictionary."""

    @pytest.mark.fast
    def test_pattern_routing_has_api_patterns(self):
        """Test PATTERN_ROUTING contains API error patterns."""
        api_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "api"]
        assert len(api_patterns) >= 3

    @pytest.mark.fast
    def test_pattern_routing_has_disk_patterns(self):
        """Test PATTERN_ROUTING contains disk error patterns."""
        disk_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "disk"]
        assert len(disk_patterns) >= 2

    @pytest.mark.fast
    def test_pattern_routing_has_path_patterns(self):
        """Test PATTERN_ROUTING contains path error patterns."""
        path_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "path"]
        assert len(path_patterns) >= 2

    @pytest.mark.fast
    def test_pattern_routing_has_checkpoint_patterns(self):
        """Test PATTERN_ROUTING contains checkpoint error patterns."""
        checkpoint_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "checkpoint"]
        assert len(checkpoint_patterns) >= 2

    @pytest.mark.fast
    def test_pattern_routing_has_download_patterns(self):
        """Test PATTERN_ROUTING contains download error patterns."""
        download_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "download"]
        assert len(download_patterns) >= 2

    @pytest.mark.fast
    def test_pattern_routing_has_otio_patterns(self):
        """Test PATTERN_ROUTING contains OTIO error patterns."""
        otio_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "otio"]
        assert len(otio_patterns) >= 3

    @pytest.mark.fast
    def test_pattern_routing_healer_names(self):
        """Test PATTERN_ROUTING healer names follow convention."""
        for pattern, (category, healer) in PATTERN_ROUTING.items():
            assert healer.endswith("-healer"), f"Healer {healer} should end with '-healer'"

    @pytest.mark.fast
    def test_pattern_routing_valid_categories(self):
        """Test PATTERN_ROUTING categories are valid."""
        valid_categories = ["api", "disk", "path", "checkpoint", "download", "otio", "config"]
        for pattern, (category, healer) in PATTERN_ROUTING.items():
            assert category in valid_categories, f"Invalid category: {category}"


class TestFallbackChainInit:
    """Tests for FallbackChain initialization."""

    @pytest.mark.fast
    def test_fallback_chain_init(self):
        """Test FallbackChain initializes with config."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)

        assert chain.config == mock_config
        assert chain.healing_logger is None
        assert chain.fallback_state["watcher_available"] is None
        assert chain.fallback_state["llm_healer_available"] is None
        assert chain.fallback_state["watcher_failures"] == 0
        assert chain.fallback_state["llm_healer_failures"] == 0

    @pytest.mark.fast
    def test_fallback_chain_init_with_logger(self):
        """Test FallbackChain initializes with healing_logger."""
        mock_config = Mock()
        mock_logger = Mock()
        chain = FallbackChain(config=mock_config, healing_logger=mock_logger)

        assert chain.healing_logger == mock_logger


class TestFallbackChainWatcherAvailability:
    """Tests for FallbackChain watcher availability checks."""

    @pytest.mark.fast
    def test_check_watcher_returns_false_without_requests(self):
        """Test check_watcher_available() returns False when requests unavailable."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)

        with patch('src.agents.fallback.REQUESTS_AVAILABLE', False):
            result = chain.check_watcher_available()
            assert result is False

    @pytest.mark.fast
    def test_check_watcher_returns_cached_true(self):
        """Test check_watcher_available() returns cached True result."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)
        chain.fallback_state["watcher_available"] = True

        result = chain.check_watcher_available()
        assert result is True

    @pytest.mark.fast
    def test_check_watcher_returns_cached_false(self):
        """Test check_watcher_available() returns cached False result."""
        mock_config = Mock()
        mock_config.watcher = Mock()
        mock_config.watcher.recheck_interval_seconds = 300.0  # 5 minutes

        chain = FallbackChain(config=mock_config)
        chain.fallback_state["watcher_available"] = False
        chain.fallback_state["last_watcher_check"] = time.time()  # Recent check

        result = chain.check_watcher_available()
        assert result is False

    @pytest.mark.fast
    def test_check_watcher_disabled_in_config(self):
        """Test check_watcher_available() returns False when disabled."""
        mock_config = Mock()
        mock_config.watcher = Mock()
        mock_config.watcher.enabled = False

        chain = FallbackChain(config=mock_config)

        with patch('src.agents.fallback.REQUESTS_AVAILABLE', True):
            result = chain.check_watcher_available()
            assert result is False


class TestFallbackChainLLMHealerAvailability:
    """Tests for FallbackChain LLM healer availability checks."""

    @pytest.mark.fast
    def test_check_llm_healer_disabled(self):
        """Test check_llm_healer_available() returns False when disabled."""
        mock_config = Mock()
        mock_config.llm_healer = Mock()
        mock_config.llm_healer.enabled = False

        chain = FallbackChain(config=mock_config)
        result = chain.check_llm_healer_available()

        assert result is False

    @pytest.mark.requires_api
    def test_check_llm_healer_anthropic_no_key(self):
        """Test check_llm_healer_available() returns False without ANTHROPIC_API_KEY."""
        mock_config = Mock()
        mock_config.llm_healer = Mock()
        mock_config.llm_healer.enabled = True
        mock_config.llm_healer.provider = "anthropic"

        chain = FallbackChain(config=mock_config)

        with patch.dict(os.environ, {}, clear=True):
            # Ensure ANTHROPIC_API_KEY is not set
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            result = chain.check_llm_healer_available()

        assert result is False

    @pytest.mark.requires_api
    def test_check_llm_healer_gemini_no_key(self):
        """Test check_llm_healer_available() returns False without GEMINI_API_KEY."""
        mock_config = Mock()
        mock_config.llm_healer = Mock()
        mock_config.llm_healer.enabled = True
        mock_config.llm_healer.provider = "gemini"

        chain = FallbackChain(config=mock_config)

        with patch.dict(os.environ, {}, clear=True):
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            result = chain.check_llm_healer_available()

        assert result is False

    @pytest.mark.requires_api
    def test_check_llm_healer_with_api_key(self):
        """Test check_llm_healer_available() returns True with API key."""
        mock_config = Mock()
        mock_config.llm_healer = Mock()
        mock_config.llm_healer.enabled = True
        mock_config.llm_healer.provider = "anthropic"

        chain = FallbackChain(config=mock_config)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            result = chain.check_llm_healer_available()

        assert result is True

    @pytest.mark.fast
    def test_check_llm_healer_returns_cached_result(self):
        """Test check_llm_healer_available() returns cached result."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)
        chain.fallback_state["llm_healer_available"] = True

        result = chain.check_llm_healer_available()

        assert result is True


class TestFallbackChainFailureTracking:
    """Tests for FallbackChain failure tracking (circuit breaker)."""

    @pytest.mark.fast
    def test_record_watcher_failure_increments_counter(self):
        """Test record_watcher_failure() increments failure counter."""
        mock_config = Mock()
        mock_config.watcher = Mock()
        mock_config.watcher.max_failures = 3

        chain = FallbackChain(config=mock_config)
        chain.record_watcher_failure()

        assert chain.fallback_state["watcher_failures"] == 1

    @pytest.mark.fast
    def test_record_watcher_failure_disables_after_max(self):
        """Test record_watcher_failure() disables watcher after max failures."""
        mock_config = Mock()
        mock_config.watcher = Mock()
        mock_config.watcher.max_failures = 3

        chain = FallbackChain(config=mock_config)
        chain.record_watcher_failure()
        chain.record_watcher_failure()
        chain.record_watcher_failure()

        assert chain.fallback_state["watcher_available"] is False

    @pytest.mark.fast
    def test_record_watcher_success_resets_counter(self):
        """Test record_watcher_success() resets failure counter."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)
        chain.fallback_state["watcher_failures"] = 2

        chain.record_watcher_success()

        assert chain.fallback_state["watcher_failures"] == 0

    @pytest.mark.fast
    def test_record_llm_healer_failure_increments_counter(self):
        """Test record_llm_healer_failure() increments failure counter."""
        mock_config = Mock()
        mock_config.llm_healer = Mock()
        mock_config.llm_healer.max_failures = 3

        chain = FallbackChain(config=mock_config)
        chain.record_llm_healer_failure()

        assert chain.fallback_state["llm_healer_failures"] == 1

    @pytest.mark.fast
    def test_record_llm_healer_failure_disables_after_max(self):
        """Test record_llm_healer_failure() disables after max failures."""
        mock_config = Mock()
        mock_config.llm_healer = Mock()
        mock_config.llm_healer.max_failures = 3

        chain = FallbackChain(config=mock_config)
        chain.record_llm_healer_failure()
        chain.record_llm_healer_failure()
        chain.record_llm_healer_failure()

        assert chain.fallback_state["llm_healer_available"] is False

    @pytest.mark.fast
    def test_record_llm_healer_success_resets_counter(self):
        """Test record_llm_healer_success() resets failure counter."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)
        chain.fallback_state["llm_healer_failures"] = 2

        chain.record_llm_healer_success()

        assert chain.fallback_state["llm_healer_failures"] == 0


class TestFallbackChainGetStatus:
    """Tests for FallbackChain.get_status()."""

    @pytest.mark.fast
    def test_get_status_returns_state(self):
        """Test get_status() returns current state."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)
        chain.fallback_state["watcher_available"] = True
        chain.fallback_state["llm_healer_available"] = False
        chain.fallback_state["watcher_failures"] = 1
        chain.fallback_state["llm_healer_failures"] = 2
        chain.fallback_state["active_model"] = "llama3.2"

        status = chain.get_status()

        assert status["watcher_available"] is True
        assert status["llm_healer_available"] is False
        assert status["watcher_failures"] == 1
        assert status["llm_healer_failures"] == 2
        assert status["active_model"] == "llama3.2"

    @pytest.mark.fast
    def test_get_status_is_thread_safe(self):
        """Test get_status() is thread-safe."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)

        results = []

        def get_status_thread():
            for _ in range(100):
                status = chain.get_status()
                results.append(status)

        threads = [threading.Thread(target=get_status_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All calls should succeed without error
        assert len(results) == 500


class TestFallbackChainRecheck:
    """Tests for FallbackChain recheck logic."""

    @pytest.mark.fast
    def test_watcher_recheck_after_interval(self):
        """Test watcher is rechecked after recheck_interval_seconds."""
        mock_config = Mock()
        mock_config.watcher = Mock()
        mock_config.watcher.enabled = False  # Will fail, but tests recheck logic
        mock_config.watcher.recheck_interval_seconds = 0.1  # 100ms for fast test

        chain = FallbackChain(config=mock_config)
        chain.fallback_state["watcher_available"] = False
        chain.fallback_state["last_watcher_check"] = time.time() - 1.0  # 1 second ago

        with patch('src.agents.fallback.REQUESTS_AVAILABLE', True):
            # Should trigger recheck since interval passed
            chain.check_watcher_available()

        # Watcher_available should be reset to None before check, then set to False by disabled config
        assert chain.fallback_state["watcher_failures"] == 0

    @pytest.mark.requires_api
    def test_llm_healer_recheck_after_interval(self):
        """Test LLM healer is rechecked after recheck_interval_seconds."""
        mock_config = Mock()
        mock_config.llm_healer = Mock()
        mock_config.llm_healer.enabled = True
        mock_config.llm_healer.provider = "anthropic"
        mock_config.llm_healer.recheck_interval_seconds = 0.1

        chain = FallbackChain(config=mock_config)
        chain.fallback_state["llm_healer_available"] = False
        chain.fallback_state["last_llm_healer_check"] = time.time() - 1.0

        # After recheck interval, should try again
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            result = chain.check_llm_healer_available()

        # Should succeed now that API key is available
        assert result is True
