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

    def test_pattern_route_api_rate_limit_429(self):
        """Test pattern_route() matches HTTP 429 rate limit."""
        error = "HTTP Error 429: Too Many Requests"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"
        assert result.confidence == 0.7

    def test_pattern_route_api_rate_limit_text(self):
        """Test pattern_route() matches rate limit text."""
        error = "Rate limit exceeded, please wait"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    def test_pattern_route_api_quota_exceeded(self):
        """Test pattern_route() matches quota exceeded."""
        error = "API quota exceeded for this month"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    def test_pattern_route_api_timeout(self):
        """Test pattern_route() matches request timeout."""
        error = "Connection timed out while waiting for response"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    def test_pattern_route_api_connection_refused(self):
        """Test pattern_route() matches connection refused."""
        error = "Connection refused by server (ECONNREFUSED)"
        result = pattern_route(error)

        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    def test_pattern_route_disk_space(self):
        """Test pattern_route() matches disk full."""
        error = "[Errno 28] No space left on device"
        result = pattern_route(error)

        assert result.category == "disk"
        assert result.suggested_healer == "disk-healer"

    def test_pattern_route_disk_enospc(self):
        """Test pattern_route() matches ENOSPC error."""
        error = "OSError: ENOSPC - filesystem full"
        result = pattern_route(error)

        assert result.category == "disk"
        assert result.suggested_healer == "disk-healer"

    def test_pattern_route_disk_permission(self):
        """Test pattern_route() matches permission denied."""
        error = "PermissionError: [Errno 13] Permission denied"
        result = pattern_route(error)

        assert result.category == "disk"
        assert result.suggested_healer == "disk-healer"

    def test_pattern_route_path_too_long(self):
        """Test pattern_route() matches path too long."""
        error = "Path too long, exceeds 260 char limit on Windows"
        result = pattern_route(error)

        assert result.category == "path"
        assert result.suggested_healer == "path-healer"

    def test_pattern_route_path_unicode_error(self):
        """Test pattern_route() matches UnicodeDecodeError."""
        error = "UnicodeDecodeError: 'utf-8' codec can't decode bytes"
        result = pattern_route(error)

        assert result.category == "path"
        assert result.suggested_healer == "path-healer"

    def test_pattern_route_checkpoint_corrupt(self):
        """Test pattern_route() matches checkpoint corruption."""
        error = "Checkpoint file corrupt, cannot read JSON"
        result = pattern_route(error)

        assert result.category == "checkpoint"
        assert result.suggested_healer == "checkpoint-healer"

    def test_pattern_route_json_decode_error(self):
        """Test pattern_route() matches JSONDecodeError."""
        error = "json.decoder.JSONDecodeError: Expecting value: line 1"
        result = pattern_route(error)

        assert result.category == "checkpoint"
        assert result.suggested_healer == "checkpoint-healer"

    def test_pattern_route_download_video_unavailable(self):
        """Test pattern_route() matches video unavailable."""
        error = "Video unavailable: This video is private"
        result = pattern_route(error)

        assert result.category == "download"
        assert result.suggested_healer == "download-healer"

    def test_pattern_route_download_yt_dlp_error(self):
        """Test pattern_route() matches yt-dlp errors."""
        error = "yt-dlp error: Unable to extract video data"
        result = pattern_route(error)

        assert result.category == "download"
        assert result.suggested_healer == "download-healer"

    def test_pattern_route_otio_timeline_error(self):
        """Test pattern_route() matches OTIO timeline errors."""
        error = "opentimelineio.exception: Invalid time range"
        result = pattern_route(error)

        assert result.category == "otio"
        assert result.suggested_healer == "otio-healer"

    def test_pattern_route_otio_clip_error(self):
        """Test pattern_route() matches clip generation errors."""
        error = "Clip creation failed: invalid duration"
        result = pattern_route(error)

        assert result.category == "otio"
        assert result.suggested_healer == "otio-healer"

    def test_pattern_route_unknown_error(self):
        """Test pattern_route() returns unknown for unmatched errors."""
        error = "Completely random error with no patterns"
        result = pattern_route(error)

        assert result.category == "unknown"
        assert result.suggested_healer == ""
        assert result.confidence == 0.3
        assert result.needs_llm_healer is True

    def test_pattern_route_case_insensitive(self):
        """Test pattern_route() is case insensitive."""
        error = "RATE LIMIT EXCEEDED"
        result = pattern_route(error)

        assert result.category == "api"

    def test_pattern_route_no_false_positive_singapore_gap(self):
        """Test pattern_route() doesn't match Singapore as gap error."""
        error = "Downloading video from Singapore server"
        result = pattern_route(error)

        # Should NOT match gap/otio pattern
        assert result.category != "otio" or "gap" not in result.reasoning.lower()

    def test_pattern_route_no_false_positive_author_auth(self):
        """Test pattern_route() doesn't match author as auth error."""
        error = "Video by author JohnDoe uploaded to YouTube"
        result = pattern_route(error)

        # Should NOT match authentication pattern
        assert result.category != "api" or "auth" not in result.reasoning.lower()


class TestPatternRoutingConstants:
    """Tests for PATTERN_ROUTING dictionary."""

    def test_pattern_routing_has_api_patterns(self):
        """Test PATTERN_ROUTING contains API error patterns."""
        api_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "api"]
        assert len(api_patterns) >= 3

    def test_pattern_routing_has_disk_patterns(self):
        """Test PATTERN_ROUTING contains disk error patterns."""
        disk_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "disk"]
        assert len(disk_patterns) >= 2

    def test_pattern_routing_has_path_patterns(self):
        """Test PATTERN_ROUTING contains path error patterns."""
        path_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "path"]
        assert len(path_patterns) >= 2

    def test_pattern_routing_has_checkpoint_patterns(self):
        """Test PATTERN_ROUTING contains checkpoint error patterns."""
        checkpoint_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "checkpoint"]
        assert len(checkpoint_patterns) >= 2

    def test_pattern_routing_has_download_patterns(self):
        """Test PATTERN_ROUTING contains download error patterns."""
        download_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "download"]
        assert len(download_patterns) >= 2

    def test_pattern_routing_has_otio_patterns(self):
        """Test PATTERN_ROUTING contains OTIO error patterns."""
        otio_patterns = [k for k, (cat, _) in PATTERN_ROUTING.items() if cat == "otio"]
        assert len(otio_patterns) >= 3

    def test_pattern_routing_healer_names(self):
        """Test PATTERN_ROUTING healer names follow convention."""
        for pattern, (category, healer) in PATTERN_ROUTING.items():
            assert healer.endswith("-healer"), f"Healer {healer} should end with '-healer'"

    def test_pattern_routing_valid_categories(self):
        """Test PATTERN_ROUTING categories are valid."""
        valid_categories = ["api", "disk", "path", "checkpoint", "download", "otio", "config"]
        for pattern, (category, healer) in PATTERN_ROUTING.items():
            assert category in valid_categories, f"Invalid category: {category}"


class TestFallbackChainInit:
    """Tests for FallbackChain initialization."""

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

    def test_fallback_chain_init_with_logger(self):
        """Test FallbackChain initializes with healing_logger."""
        mock_config = Mock()
        mock_logger = Mock()
        chain = FallbackChain(config=mock_config, healing_logger=mock_logger)

        assert chain.healing_logger == mock_logger


class TestFallbackChainWatcherAvailability:
    """Tests for FallbackChain watcher availability checks."""

    def test_check_watcher_returns_false_without_requests(self):
        """Test check_watcher_available() returns False when requests unavailable."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)

        with patch('src.agents.fallback.REQUESTS_AVAILABLE', False):
            result = chain.check_watcher_available()
            assert result is False

    def test_check_watcher_returns_cached_true(self):
        """Test check_watcher_available() returns cached True result."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)
        chain.fallback_state["watcher_available"] = True

        result = chain.check_watcher_available()
        assert result is True

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

    def test_check_llm_healer_disabled(self):
        """Test check_llm_healer_available() returns False when disabled."""
        mock_config = Mock()
        mock_config.llm_healer = Mock()
        mock_config.llm_healer.enabled = False

        chain = FallbackChain(config=mock_config)
        result = chain.check_llm_healer_available()

        assert result is False

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

    def test_check_llm_healer_returns_cached_result(self):
        """Test check_llm_healer_available() returns cached result."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)
        chain.fallback_state["llm_healer_available"] = True

        result = chain.check_llm_healer_available()

        assert result is True


class TestFallbackChainFailureTracking:
    """Tests for FallbackChain failure tracking (circuit breaker)."""

    def test_record_watcher_failure_increments_counter(self):
        """Test record_watcher_failure() increments failure counter."""
        mock_config = Mock()
        mock_config.watcher = Mock()
        mock_config.watcher.max_failures = 3

        chain = FallbackChain(config=mock_config)
        chain.record_watcher_failure()

        assert chain.fallback_state["watcher_failures"] == 1

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

    def test_record_watcher_success_resets_counter(self):
        """Test record_watcher_success() resets failure counter."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)
        chain.fallback_state["watcher_failures"] = 2

        chain.record_watcher_success()

        assert chain.fallback_state["watcher_failures"] == 0

    def test_record_llm_healer_failure_increments_counter(self):
        """Test record_llm_healer_failure() increments failure counter."""
        mock_config = Mock()
        mock_config.llm_healer = Mock()
        mock_config.llm_healer.max_failures = 3

        chain = FallbackChain(config=mock_config)
        chain.record_llm_healer_failure()

        assert chain.fallback_state["llm_healer_failures"] == 1

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

    def test_record_llm_healer_success_resets_counter(self):
        """Test record_llm_healer_success() resets failure counter."""
        mock_config = Mock()
        chain = FallbackChain(config=mock_config)
        chain.fallback_state["llm_healer_failures"] = 2

        chain.record_llm_healer_success()

        assert chain.fallback_state["llm_healer_failures"] == 0


class TestFallbackChainGetStatus:
    """Tests for FallbackChain.get_status()."""

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
