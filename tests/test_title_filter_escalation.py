"""Integration tests for TitleFilter escalation args wiring (Sprint 14 US-009).

Tests that TitleFilter.search_video_metadata() correctly integrates with
EscalationManager:
- Includes escalation args from get_escalation_args(keyword) in yt-dlp command
- Handles 403 during search by recording failure and retrying with escalated args
- Timeout does NOT trigger escalation (timeout is search-specific, not rate-limit)
- JSON parse error in yt-dlp output does not crash escalation tracking
- Works correctly with escalation_manager=None (no AttributeError)

Pytest marker: fast
"""

import json
import subprocess
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from src.downloader.title_filter import TitleFilter, SearchResult
from src.downloader.escalation_manager import (
    EscalationManager,
    EscalationResult,
    is_escalation_trigger,
)
from src.downloader.types import EscalationTier


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeExtractorArgsConfig:
    """Minimal stand-in for ExtractorArgsConfig."""
    enabled: bool = True
    player_clients: List[str] = field(
        default_factory=lambda: ["web_safari", "tv_downgraded", "web"]
    )
    escalation_threshold: int = 2
    cooldown_seconds: float = 0.0  # No cooldown for fast tests
    max_tier: int = 3


def _make_mock_config():
    """Create a mock Config with download settings."""
    config = Mock()
    config.download = Mock()
    config.download.search_timeout = 30
    return config


def _make_tier_value_func():
    """Create a tier value getter function."""
    def getter(tier, key, default):
        tiers = {
            'short': {'min': 0, 'max': 60},
            'medium': {'min': 60, 'max': 300},
            'long': {'min': 300, 'max': 900},
        }
        return tiers.get(tier, {}).get(key, default)
    return getter


def _make_impersonation_manager():
    """Create a mock ImpersonationManager."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = ["--impersonate", "Chrome-136:Macos-15"]
    mgr.target_count = 5
    return mgr


def _make_escalation_manager():
    """Create a real EscalationManager with fast test settings."""
    imp_mgr = _make_impersonation_manager()
    ext_config = FakeExtractorArgsConfig()
    return EscalationManager(
        impersonation_manager=imp_mgr,
        extractor_args_config=ext_config,
    )


def _make_title_filter(escalation_manager=None, impersonation_manager=None):
    """Create a TitleFilter with optional escalation/impersonation managers."""
    config = _make_mock_config()
    return TitleFilter(
        config=config,
        cookies_args=["--cookies", "cookies.txt"],
        get_tier_value_func=_make_tier_value_func(),
        impersonation_manager=impersonation_manager,
        escalation_manager=escalation_manager,
    )


def _mock_subprocess_success(stdout="", stderr=""):
    """Create a mock subprocess result for success."""
    return Mock(returncode=0, stdout=stdout, stderr=stderr)


def _mock_subprocess_403(stderr="ERROR: HTTP Error 403: Forbidden"):
    """Create a mock subprocess result for 403."""
    return Mock(returncode=1, stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# AC1: search_video_metadata includes escalation args in yt-dlp command
# ---------------------------------------------------------------------------

class TestSearchIncludesEscalationArgs:
    """Test that search_video_metadata() includes escalation args from
    get_escalation_args(keyword) in the yt-dlp --dump-json search command."""

    @patch("subprocess.run")
    def test_tier1_impersonate_args_included(self, mock_run):
        """At Tier 1, --impersonate args appear in yt-dlp command."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        mock_run.return_value = _mock_subprocess_success()

        tf.search_video_metadata("wildlife", "medium")

        cmd = mock_run.call_args[0][0]
        assert "--impersonate" in cmd, "Tier 1 should include --impersonate arg"
        # Verify --dump-json is also present (search command)
        assert "--dump-json" in cmd

    @patch("subprocess.run")
    def test_tier2_extractor_args_included(self, mock_run):
        """At Tier 2, --extractor-args appear alongside --impersonate."""
        esc_mgr = _make_escalation_manager()
        # Escalate to Tier 2 by recording 2 failures
        esc_mgr.record_failure("wildlife", "HTTP Error 403")
        esc_mgr.record_failure("wildlife", "HTTP Error 403")

        tf = _make_title_filter(escalation_manager=esc_mgr)

        mock_run.return_value = _mock_subprocess_success()

        tf.search_video_metadata("wildlife", "medium")

        cmd = mock_run.call_args[0][0]
        assert "--impersonate" in cmd
        assert "--extractor-args" in cmd, "Tier 2 should include --extractor-args"

    @patch("subprocess.run")
    def test_escalation_args_before_cookies(self, mock_run):
        """Escalation args appear before cookie args in command (correct ordering)."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        mock_run.return_value = _mock_subprocess_success()

        tf.search_video_metadata("wildlife", "medium")

        cmd = mock_run.call_args[0][0]
        # Find indices
        impersonate_idx = cmd.index("--impersonate")
        cookies_idx = cmd.index("--cookies")
        assert impersonate_idx < cookies_idx, \
            "Escalation args must come before cookie args"


# ---------------------------------------------------------------------------
# AC2: 403 during search — record_failure + retried with escalated args
# ---------------------------------------------------------------------------

class TestSearchHandles403WithEscalation:
    """Test that 403 errors during search are handled by recording failure
    on escalation_manager, so that subsequent searches use escalated args."""

    @patch("subprocess.run")
    def test_403_stderr_detected_as_escalation_trigger(self, mock_run):
        """Verify that is_escalation_trigger() detects 403 in search stderr."""
        mock_run.return_value = _mock_subprocess_403()

        result = is_escalation_trigger("ERROR: HTTP Error 403: Forbidden")
        assert result is True

    @patch("subprocess.run")
    def test_caller_can_record_failure_on_403_search(self, mock_run):
        """After search returns 403 stderr, caller records failure on
        escalation_manager which escalates for next call."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        # First search returns 403
        mock_run.return_value = _mock_subprocess_403()
        result = tf.search_video_metadata("wildlife", "medium")

        # Caller checks stderr and records failure
        stderr = mock_run.return_value.stderr
        if is_escalation_trigger(stderr):
            esc_mgr.record_failure("wildlife", stderr)

        # Record again to reach threshold (2 consecutive 403s -> Tier 2)
        esc_mgr.record_failure("wildlife", "HTTP Error 403")

        # Second search should use Tier 2 args
        mock_run.return_value = _mock_subprocess_success(
            stdout=json.dumps({"id": "abc", "title": "Test", "duration": 120})
        )
        result2 = tf.search_video_metadata("wildlife", "medium")

        cmd2 = mock_run.call_args[0][0]
        assert "--extractor-args" in cmd2, \
            "After 2 failures, search should use Tier 2 (--extractor-args)"

    @patch("subprocess.run")
    def test_403_does_not_crash_search_return(self, mock_run):
        """403 search still returns a valid SearchResult (empty videos)."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        mock_run.return_value = _mock_subprocess_403()
        result = tf.search_video_metadata("wildlife", "medium")

        # SearchResult is returned (not exception)
        assert hasattr(result, "videos")
        assert len(result.videos) == 0


# ---------------------------------------------------------------------------
# AC3: Timeout does NOT trigger escalation
# ---------------------------------------------------------------------------

class TestTimeoutDoesNotTriggerEscalation:
    """Test that search timeout does NOT trigger escalation (timeout is
    search-specific, not rate-limit related)."""

    @patch("subprocess.run")
    def test_timeout_does_not_escalate(self, mock_run):
        """TimeoutExpired in search doesn't call record_failure on escalation."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        mock_run.side_effect = subprocess.TimeoutExpired("yt-dlp", 30)

        result = tf.search_video_metadata("wildlife", "medium")

        # Verify result is a timed-out SearchResult
        assert result.timed_out is True
        assert result.videos == []

        # Verify escalation state unchanged (still Tier 1, no failures)
        esc_result = esc_mgr.get_escalation_args("wildlife")
        assert esc_result.tier == EscalationTier.IMPERSONATE_ONLY
        assert "--extractor-args" not in esc_result.args

    @patch("subprocess.run")
    def test_timeout_returns_timed_out_search_result(self, mock_run):
        """Timeout returns SearchResult with timed_out=True for caller to handle."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        mock_run.side_effect = subprocess.TimeoutExpired("yt-dlp", 30)

        result = tf.search_video_metadata("wildlife", "medium")

        assert result.timed_out is True
        assert result.error is not None
        assert "timeout" in result.error.lower()

    @patch("subprocess.run")
    def test_timeout_preserves_tier_for_next_search(self, mock_run):
        """After timeout, next search still uses same tier (no escalation)."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        # First call: timeout
        mock_run.side_effect = subprocess.TimeoutExpired("yt-dlp", 30)
        tf.search_video_metadata("wildlife", "medium")

        # Second call: success
        mock_run.side_effect = None
        mock_run.return_value = _mock_subprocess_success()
        tf.search_video_metadata("wildlife", "medium")

        # Should still be Tier 1 (timeout didn't escalate)
        cmd = mock_run.call_args[0][0]
        assert "--impersonate" in cmd
        assert "--extractor-args" not in cmd


# ---------------------------------------------------------------------------
# AC4: JSON parse error doesn't crash escalation tracking
# ---------------------------------------------------------------------------

class TestJsonParseErrorPreservesEscalation:
    """Test that JSON parse error in yt-dlp output does not crash escalation
    tracking — verify escalation state unchanged after parse failures."""

    @patch("subprocess.run")
    def test_malformed_json_output_no_escalation_crash(self, mock_run):
        """Malformed JSON in stdout doesn't affect escalation state."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        mock_run.return_value = _mock_subprocess_success(
            stdout="not json at all\n{broken json\nmore garbage"
        )

        result = tf.search_video_metadata("wildlife", "medium")

        # Should parse 0 videos (all lines invalid)
        assert len(result.videos) == 0
        assert result.timed_out is False

        # Escalation state should be unchanged
        esc_result = esc_mgr.get_escalation_args("wildlife")
        assert esc_result.tier == EscalationTier.IMPERSONATE_ONLY

    @patch("subprocess.run")
    def test_mixed_valid_invalid_json_preserves_escalation(self, mock_run):
        """Mix of valid and invalid JSON lines — valid ones parsed, escalation intact."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        stdout = "\n".join([
            json.dumps({"id": "abc", "title": "Valid Video", "duration": 120}),
            "not a json line",
            json.dumps({"id": "def", "title": "Also Valid", "duration": 180}),
        ])
        mock_run.return_value = _mock_subprocess_success(stdout=stdout)

        result = tf.search_video_metadata("wildlife", "medium")

        assert len(result.videos) == 2
        assert result.videos[0]["id"] == "abc"
        assert result.videos[1]["id"] == "def"

        # Escalation unchanged
        esc_result = esc_mgr.get_escalation_args("wildlife")
        assert esc_result.tier == EscalationTier.IMPERSONATE_ONLY

    @patch("subprocess.run")
    def test_empty_json_output_escalation_unchanged(self, mock_run):
        """Empty yt-dlp stdout — escalation state unchanged."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        mock_run.return_value = _mock_subprocess_success(stdout="")

        result = tf.search_video_metadata("wildlife", "medium")

        assert len(result.videos) == 0
        esc_result = esc_mgr.get_escalation_args("wildlife")
        assert esc_result.tier == EscalationTier.IMPERSONATE_ONLY

    @patch("subprocess.run")
    def test_generic_exception_no_escalation_change(self, mock_run):
        """Generic exception in subprocess — escalation state intact."""
        esc_mgr = _make_escalation_manager()
        tf = _make_title_filter(escalation_manager=esc_mgr)

        mock_run.side_effect = OSError("Network unreachable")

        result = tf.search_video_metadata("wildlife", "medium")

        assert len(result.videos) == 0
        assert result.error is not None

        # Escalation unaffected
        esc_result = esc_mgr.get_escalation_args("wildlife")
        assert esc_result.tier == EscalationTier.IMPERSONATE_ONLY


# ---------------------------------------------------------------------------
# AC5: escalation_manager=None skips all escalation logic without error
# ---------------------------------------------------------------------------

class TestEscalationManagerNone:
    """Test that escalation_manager=None skips all escalation logic
    without AttributeError."""

    @patch("subprocess.run")
    def test_search_works_without_escalation_manager(self, mock_run):
        """search_video_metadata works with no escalation manager."""
        tf = _make_title_filter(escalation_manager=None, impersonation_manager=None)

        stdout = json.dumps({"id": "abc", "title": "Test", "duration": 120})
        mock_run.return_value = _mock_subprocess_success(stdout=stdout)

        result = tf.search_video_metadata("wildlife", "medium")

        assert len(result.videos) == 1
        assert result.videos[0]["id"] == "abc"

    @patch("subprocess.run")
    def test_no_impersonate_args_when_both_none(self, mock_run):
        """No --impersonate or --extractor-args when both managers are None."""
        tf = _make_title_filter(escalation_manager=None, impersonation_manager=None)

        mock_run.return_value = _mock_subprocess_success()

        tf.search_video_metadata("wildlife", "medium")

        cmd = mock_run.call_args[0][0]
        assert "--impersonate" not in cmd
        assert "--extractor-args" not in cmd

    @patch("subprocess.run")
    def test_impersonation_fallback_when_escalation_none(self, mock_run):
        """When escalation_manager=None but impersonation_manager exists,
        impersonation args are still added."""
        imp_mgr = _make_impersonation_manager()
        tf = _make_title_filter(
            escalation_manager=None,
            impersonation_manager=imp_mgr,
        )

        mock_run.return_value = _mock_subprocess_success()

        tf.search_video_metadata("wildlife", "medium")

        cmd = mock_run.call_args[0][0]
        assert "--impersonate" in cmd, \
            "Should fall back to impersonation_manager when escalation is None"
        assert "--extractor-args" not in cmd

    @patch("subprocess.run")
    def test_no_attribute_error_on_403_without_manager(self, mock_run):
        """403 search with no escalation_manager doesn't raise AttributeError."""
        tf = _make_title_filter(escalation_manager=None, impersonation_manager=None)

        mock_run.return_value = _mock_subprocess_403()

        # Should not raise
        result = tf.search_video_metadata("wildlife", "medium")
        assert len(result.videos) == 0

    @patch("subprocess.run")
    def test_no_attribute_error_on_timeout_without_manager(self, mock_run):
        """Timeout with no escalation_manager doesn't raise AttributeError."""
        tf = _make_title_filter(escalation_manager=None, impersonation_manager=None)

        mock_run.side_effect = subprocess.TimeoutExpired("yt-dlp", 30)

        # Should not raise
        result = tf.search_video_metadata("wildlife", "medium")
        assert result.timed_out is True
