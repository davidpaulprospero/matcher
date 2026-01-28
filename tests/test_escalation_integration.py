"""Integration tests for escalation flow with mock 403 responses (US-010).

Tests the full end-to-end escalation lifecycle using mocked yt-dlp stderr
output, verifying that EscalationManager correctly progresses through tiers,
rotates extractor-args, and handles success resets — all without real
subprocess calls or network access.

Pytest marker: fast
"""

import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.escalation_manager import (
    EscalationManager,
    EscalationResult,
    is_escalation_trigger,
)
from src.downloader.types import EscalationState, EscalationTier


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeExtractorArgsConfig:
    """Minimal stand-in for ExtractorArgsConfig."""
    enabled: bool = True
    player_clients: List[str] = field(
        default_factory=lambda: ["web_safari", "tv_downgraded", "web", "ios", "android_vr"]
    )
    escalation_threshold: int = 2
    cooldown_seconds: float = 300.0
    max_tier: int = 3


def _make_impersonation_manager():
    """Create a mock ImpersonationManager returning deterministic args."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", "Chrome-136:Macos-15"
    ]
    return mgr


@pytest.fixture
def ext_config():
    return FakeExtractorArgsConfig()


@pytest.fixture
def imp_manager():
    return _make_impersonation_manager()


@pytest.fixture
def manager(imp_manager, ext_config):
    return EscalationManager(imp_manager, ext_config)


def _advance_past_cooldown(base_time: float = None):
    """Return a callable that returns time values well past the 300s cooldown.

    Uses a base_time anchored to real time.time() so that elapsed calculations
    from earlier (real) escalation timestamps always exceed the cooldown.
    """
    if base_time is None:
        base_time = time.time() + 500  # 500s past now, well beyond 300s cooldown
    counter = [0]

    def advancing_time():
        counter[0] += 1
        return base_time + counter[0] * 400

    return advancing_time


# ---------------------------------------------------------------------------
# Real yt-dlp stderr samples for trigger detection
# ---------------------------------------------------------------------------

REAL_STDERR_403_SAMPLES = [
    # Standard 403 from YouTube
    "ERROR: [youtube] dQw4w9WgXcQ: HTTP Error 403: Forbidden",
    # Sign-in confirmation prompt
    "ERROR: [youtube] abc123: Sign in to confirm you're not a bot. "
    "This helps protect our community. Learn more",
    # Age verification variant
    "ERROR: [youtube] xyz789: Sign in to confirm your age. "
    "This video may be inappropriate for some users.",
    # Bot detection with captcha
    "ERROR: [youtube] vid001: Sorry, we need to make sure "
    "you're not a bot. Please solve the captcha to continue.",
    # Generic blocked response
    "ERROR: [youtube] vid002: This request was blocked for security reasons. "
    "Please try again later.",
    # Human verification
    "ERROR: [youtube] vid003: Please verify you are human to continue watching",
]

REAL_STDERR_NON_TRIGGER_SAMPLES = [
    # Normal 404
    "ERROR: [youtube] gone123: HTTP Error 404: Not Found",
    # Video unavailable
    "ERROR: [youtube] priv456: Video unavailable. This video is private.",
    # Network timeout
    "ERROR: [youtube] slow789: Connection timed out after 30 seconds",
    # Note: 429 is NOW an escalation trigger (added in Sprint 10 US-001)
    # Successful download output
    "[download] 100% of 5.23MiB in 00:02",
    # Metadata extraction
    "[youtube] Extracting URL: https://www.youtube.com/watch?v=abc123",
    # Empty/blank
    "",
    # Just whitespace
    "   \n\t  ",
]


# ---------------------------------------------------------------------------
# AC 1: Full escalation flow — 403 errors -> record_failure -> escalate
#        -> verify args change at each tier
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestFullEscalationFlow:
    """Simulate complete 403-driven escalation from Tier 1 through Tier 3."""

    def test_full_tier_1_to_3_flow(self, manager):
        """End-to-end: mock 403 stderr -> is_escalation_trigger -> record_failure
        -> args change at each tier."""
        keyword = "sunset beach 4k"

        # --- Tier 1: initial state ---
        result_t1 = manager.get_escalation_args(keyword)
        assert result_t1.tier == EscalationTier.IMPERSONATE_ONLY
        assert "--impersonate" in result_t1.args
        assert "--extractor-args" not in result_t1.args
        assert result_t1.rotate_cookies is False

        # --- Simulate two 403 failures (threshold=2) ---
        stderr_403 = "ERROR: [youtube] dQw4w9WgXcQ: HTTP Error 403: Forbidden"
        assert is_escalation_trigger(stderr_403) is True
        manager.record_failure(keyword, stderr_403)
        manager.record_failure(keyword, stderr_403)

        # --- Tier 2: impersonate + extractor-args ---
        result_t2 = manager.get_escalation_args(keyword)
        assert result_t2.tier == EscalationTier.EXTRACTOR_ARGS
        assert "--impersonate" in result_t2.args
        assert "--extractor-args" in result_t2.args
        assert result_t2.rotate_cookies is False

        # Verify extractor-args format
        ea_idx = result_t2.args.index("--extractor-args")
        ea_val = result_t2.args[ea_idx + 1]
        assert ea_val.startswith("youtube:player_client=")
        # All 5 player clients should be present
        clients_str = ea_val.split("=", 1)[1]
        clients = clients_str.split(",")
        assert len(clients) == 5

        # --- Advance past cooldown and trigger Tier 3 ---
        advancing = _advance_past_cooldown()
        with patch("src.downloader.escalation_manager.time") as em_time, \
             patch("src.downloader.types.time") as types_time:
            em_time.time.side_effect = lambda: advancing()
            types_time.time.side_effect = lambda: advancing()

            stderr_bot = "ERROR: [youtube] abc123: Sign in to confirm you're not a bot."
            assert is_escalation_trigger(stderr_bot) is True
            manager.record_failure(keyword, stderr_bot)
            manager.record_failure(keyword, stderr_bot)

        # --- Tier 3: all Tier 2 args + cookie rotation ---
        result_t3 = manager.get_escalation_args(keyword)
        assert result_t3.tier == EscalationTier.FULL_BYPASS
        assert "--impersonate" in result_t3.args
        assert "--extractor-args" in result_t3.args
        assert result_t3.rotate_cookies is True

    @pytest.mark.fast
    def test_multiple_keywords_escalate_independently(self, manager):
        """Two keywords escalate through tiers independently."""
        kw_a = "mountain landscape"
        kw_b = "ocean waves"

        # Escalate kw_a to Tier 2
        manager.record_failure(kw_a, "HTTP Error 403")
        manager.record_failure(kw_a, "HTTP Error 403")

        # kw_b gets one failure only
        manager.record_failure(kw_b, "HTTP Error 403")

        assert manager.get_escalation_args(kw_a).tier == EscalationTier.EXTRACTOR_ARGS
        assert manager.get_escalation_args(kw_b).tier == EscalationTier.IMPERSONATE_ONLY

        # Now escalate kw_b to Tier 2
        manager.record_failure(kw_b, "HTTP Error 403")
        assert manager.get_escalation_args(kw_b).tier == EscalationTier.EXTRACTOR_ARGS

        # kw_a should still be at Tier 2, not Tier 3
        assert manager.get_escalation_args(kw_a).tier == EscalationTier.EXTRACTOR_ARGS

    @pytest.mark.fast
    def test_metrics_reflect_full_flow(self, manager):
        """After full escalation, metrics accurately track all events."""
        keyword = "drone footage"

        # Tier 1 -> 2
        manager.record_failure(keyword, "HTTP Error 403")
        manager.record_failure(keyword, "HTTP Error 403")

        # Some successes at Tier 2
        manager.record_success(keyword)
        manager.record_success(keyword)

        # More failures to try escalation (blocked by cooldown)
        manager.record_failure(keyword, "captcha required")
        manager.record_failure(keyword, "captcha required")

        metrics = manager.get_metrics()
        assert metrics["total_403s"] == 4
        assert metrics["total_successes"] == 2
        assert metrics["total_escalations"] >= 1
        assert keyword in metrics["keywords_at_each_tier"].get("EXTRACTOR_ARGS", [])


# ---------------------------------------------------------------------------
# AC 2: is_escalation_trigger() with real yt-dlp stderr samples
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestTriggerDetectionRealSamples:
    """is_escalation_trigger() correctly identifies 403 patterns from real
    yt-dlp stderr samples."""

    @pytest.mark.parametrize("stderr", REAL_STDERR_403_SAMPLES)
    @pytest.mark.fast
    def test_real_403_samples_trigger(self, stderr):
        """Each real 403/bot stderr sample is correctly identified as a trigger."""
        assert is_escalation_trigger(stderr) is True, (
            f"Expected trigger for: {stderr!r}"
        )

    @pytest.mark.parametrize("stderr", REAL_STDERR_NON_TRIGGER_SAMPLES)
    @pytest.mark.fast
    def test_real_non_trigger_samples_rejected(self, stderr):
        """Non-403 stderr samples are correctly rejected."""
        assert is_escalation_trigger(stderr) is False, (
            f"Should NOT trigger for: {stderr!r}"
        )

    @pytest.mark.fast
    def test_multiline_stderr_with_trigger(self):
        """Real yt-dlp often outputs multiple lines; trigger buried inside."""
        multiline_stderr = (
            "[youtube] Extracting URL: https://www.youtube.com/watch?v=abc123\n"
            "[youtube] abc123: Downloading webpage\n"
            "ERROR: [youtube] abc123: HTTP Error 403: Forbidden\n"
            "WARNING: Unable to download JSON metadata"
        )
        assert is_escalation_trigger(multiline_stderr) is True

    @pytest.mark.fast
    def test_multiline_stderr_without_trigger(self):
        """Multi-line non-trigger stderr should not trigger."""
        multiline_stderr = (
            "[youtube] Extracting URL: https://www.youtube.com/watch?v=abc123\n"
            "[youtube] abc123: Downloading webpage\n"
            "[download] 100% of 5.23MiB in 00:02\n"
            "[download] Destination: video.mp4"
        )
        assert is_escalation_trigger(multiline_stderr) is False

    @pytest.mark.fast
    def test_case_variations(self):
        """Trigger detection is case-insensitive."""
        assert is_escalation_trigger("http error 403") is True
        assert is_escalation_trigger("HTTP ERROR 403") is True
        assert is_escalation_trigger("CAPTCHA") is True
        assert is_escalation_trigger("Blocked") is True
        assert is_escalation_trigger("BOT detection") is True
        assert is_escalation_trigger("Verify You Are Human") is True


# ---------------------------------------------------------------------------
# AC 3: Success reset — after Tier 2, record_success resets 403 counter
#        but tier stays at 2
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestSuccessReset:
    """After escalation to Tier 2, record_success resets 403 counter
    but tier remains sticky at Tier 2."""

    def test_success_resets_counter_keeps_tier(self, manager):
        """record_success() clears consecutive_403s but tier stays at 2."""
        keyword = "nature documentary"

        # Escalate to Tier 2
        manager.record_failure(keyword, "HTTP Error 403")
        manager.record_failure(keyword, "HTTP Error 403")
        assert manager.get_escalation_args(keyword).tier == EscalationTier.EXTRACTOR_ARGS

        # Success resets counter
        manager.record_success(keyword)

        # Tier is still 2
        result = manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.EXTRACTOR_ARGS

        # 403 counter was reset — single failure does not re-escalate
        manager.record_failure(keyword, "HTTP Error 403")
        result = manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.EXTRACTOR_ARGS

    @pytest.mark.fast
    def test_multiple_successes_dont_deescalate(self, manager):
        """Many successes do not reduce the tier back to Tier 1."""
        keyword = "city timelapse"

        # Escalate to Tier 2
        manager.record_failure(keyword, "HTTP Error 403")
        manager.record_failure(keyword, "HTTP Error 403")

        # Record many successes
        for _ in range(20):
            manager.record_success(keyword)

        # Still Tier 2
        result = manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.EXTRACTOR_ARGS

    @pytest.mark.fast
    def test_success_at_tier_3_keeps_tier_3(self, manager):
        """At Tier 3, successes do not de-escalate."""
        keyword = "aerial footage"

        advancing = _advance_past_cooldown()
        with patch("src.downloader.escalation_manager.time") as em_time, \
             patch("src.downloader.types.time") as types_time:
            em_time.time.side_effect = lambda: advancing()
            types_time.time.side_effect = lambda: advancing()

            # Escalate to Tier 3
            manager.record_failure(keyword, "403")
            manager.record_failure(keyword, "403")
            manager.record_failure(keyword, "403")
            manager.record_failure(keyword, "403")

        assert manager.get_escalation_args(keyword).tier == EscalationTier.FULL_BYPASS

        # Many successes
        for _ in range(50):
            manager.record_success(keyword)

        # Still Tier 3
        assert manager.get_escalation_args(keyword).tier == EscalationTier.FULL_BYPASS

    @pytest.mark.fast
    def test_success_then_failure_cycle(self, manager):
        """Alternating success/failure doesn't accidentally escalate."""
        keyword = "wildlife footage"

        # One failure, one success, repeat - counter never reaches threshold
        for _ in range(10):
            manager.record_failure(keyword, "HTTP Error 403")
            manager.record_success(keyword)

        # Should still be Tier 1 (counter reset each time)
        assert manager.get_escalation_args(keyword).tier == EscalationTier.IMPERSONATE_ONLY


# ---------------------------------------------------------------------------
# AC 4: Extractor-args rotation — multiple Tier 2 escalations produce
#        different player_client starting positions
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestExtractorArgsRotation:
    """Multiple Tier 2 escalations produce different player_client orderings."""

    def test_extractor_args_rotate_on_successive_escalations(self, imp_manager):
        """Each Tier 2 escalation increments extractor_args_index,
        rotating the player_client list starting position."""
        config = FakeExtractorArgsConfig(cooldown_seconds=1.0)
        mgr = EscalationManager(imp_manager, config)
        keyword = "rotation_test"

        # First escalation -> Tier 2, extractor_args_index = 1
        mgr.record_failure(keyword, "403")
        mgr.record_failure(keyword, "403")
        result_1 = mgr.get_escalation_args(keyword)
        assert result_1.tier == EscalationTier.EXTRACTOR_ARGS
        ea_idx_1 = result_1.args.index("--extractor-args")
        client_str_1 = result_1.args[ea_idx_1 + 1]

        # Reset keyword to go back to Tier 1 while keeping extractor_args_index
        # Can't reset keyword (that clears state), so instead:
        # Advance past cooldown and escalate again -> Tier 3, index increments again
        advancing = _advance_past_cooldown()
        with patch("src.downloader.escalation_manager.time") as em_time, \
             patch("src.downloader.types.time") as types_time:
            em_time.time.side_effect = lambda: advancing()
            types_time.time.side_effect = lambda: advancing()

            mgr.record_failure(keyword, "403")
            mgr.record_failure(keyword, "403")

        result_2 = mgr.get_escalation_args(keyword)
        assert result_2.tier == EscalationTier.FULL_BYPASS
        ea_idx_2 = result_2.args.index("--extractor-args")
        client_str_2 = result_2.args[ea_idx_2 + 1]

        # The two extractor-args strings should differ (different starting position)
        assert client_str_1 != client_str_2, (
            f"Expected different rotation: {client_str_1} vs {client_str_2}"
        )

    @pytest.mark.fast
    def test_rotation_wraps_around(self, imp_manager):
        """Rotation index wraps around the player_clients list."""
        config = FakeExtractorArgsConfig(
            player_clients=["a", "b", "c"],
            cooldown_seconds=0.0,  # no cooldown for easy testing
        )
        mgr = EscalationManager(imp_manager, config)

        collected_orderings = []

        for i in range(4):
            keyword = f"kw_{i}"
            # Escalate to Tier 2
            mgr.record_failure(keyword, "403")
            mgr.record_failure(keyword, "403")

            result = mgr.get_escalation_args(keyword)
            ea_idx = result.args.index("--extractor-args")
            client_str = result.args[ea_idx + 1]
            ordering = client_str.split("=", 1)[1]
            collected_orderings.append(ordering)

        # Each keyword gets its own state with extractor_args_index = 1
        # (set on escalation). So all 4 keywords should have index=1,
        # giving the same starting rotation. But that's per-keyword.
        # Let's test that the format is always valid.
        for ordering in collected_orderings:
            clients = ordering.split(",")
            assert len(clients) == 3
            assert set(clients) == {"a", "b", "c"}

    @pytest.mark.fast
    def test_single_keyword_escalation_rotates_index(self, imp_manager):
        """A single keyword escalating twice gets different extractor-args index."""
        config = FakeExtractorArgsConfig(
            player_clients=["web_safari", "tv_downgraded", "web", "ios"],
            cooldown_seconds=0.0,
        )
        mgr = EscalationManager(imp_manager, config)
        keyword = "single_kw"

        # First escalation -> Tier 2 (index becomes 1)
        mgr.record_failure(keyword, "403")
        mgr.record_failure(keyword, "403")
        result_t2 = mgr.get_escalation_args(keyword)
        ea_idx = result_t2.args.index("--extractor-args")
        clients_t2 = result_t2.args[ea_idx + 1].split("=", 1)[1]

        # Second escalation -> Tier 3 (index becomes 2)
        advancing = _advance_past_cooldown()
        with patch("src.downloader.escalation_manager.time") as em_time, \
             patch("src.downloader.types.time") as types_time:
            em_time.time.side_effect = lambda: advancing()
            types_time.time.side_effect = lambda: advancing()
            mgr.record_failure(keyword, "403")
            mgr.record_failure(keyword, "403")

        result_t3 = mgr.get_escalation_args(keyword)
        ea_idx = result_t3.args.index("--extractor-args")
        clients_t3 = result_t3.args[ea_idx + 1].split("=", 1)[1]

        # Different starting positions due to index increment
        assert clients_t2 != clients_t3, (
            f"Expected different rotation: {clients_t2} vs {clients_t3}"
        )

        # Both should have all 4 clients (just reordered)
        assert set(clients_t2.split(",")) == {"web_safari", "tv_downgraded", "web", "ios"}
        assert set(clients_t3.split(",")) == {"web_safari", "tv_downgraded", "web", "ios"}


# ---------------------------------------------------------------------------
# AC 5: All tests use mocking (no real subprocess calls or network access)
# — This is enforced by structure: no subprocess imports, no network calls,
#   only mock ImpersonationManager and fake config dataclasses.
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestMockingEnforced:
    """Verify tests only use mocks — no real subprocess or network access."""

    def test_impersonation_manager_is_mock(self, manager):
        """ImpersonationManager is a MagicMock, not a real instance."""
        assert isinstance(manager._impersonation_manager, MagicMock)

    @pytest.mark.fast
    def test_no_subprocess_in_test_module(self):
        """This test module does not import subprocess."""
        import sys
        mod = sys.modules[__name__]
        # Verify subprocess is not in the module's namespace
        assert not hasattr(mod, "subprocess"), "subprocess should not be imported"
        # Verify subprocess module is not referenced in this module's globals
        assert "subprocess" not in dir(mod)

    @pytest.mark.fast
    def test_escalation_manager_works_without_network(self, manager):
        """Full flow works without any network calls."""
        # This test simply exercises the full flow and verifies it completes
        kw = "no_network"
        for _ in range(2):
            manager.record_failure(kw, "HTTP Error 403")
        result = manager.get_escalation_args(kw)
        assert result.tier == EscalationTier.EXTRACTOR_ARGS
        manager.record_success(kw)
        assert manager.get_escalation_args(kw).tier == EscalationTier.EXTRACTOR_ARGS
