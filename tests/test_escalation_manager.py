"""Unit tests for EscalationManager core logic (US-009).

Tests cover:
  - Tier progression via record_failure() threshold
  - get_escalation_args() returns correct args per tier
  - Cooldown suppression and expiry (mocked time.time())
  - Per-keyword isolation
  - Thread safety with concurrent access
"""

import threading
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
        default_factory=lambda: ["web_safari", "tv_downgraded", "web"]
    )
    escalation_threshold: int = 2
    cooldown_seconds: float = 300.0
    max_tier: int = 3


def _make_impersonation_manager(targets=None):
    """Create a mock ImpersonationManager that returns deterministic args."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", targets[0] if targets else "Chrome-136:Macos-15"
    ]
    return mgr


@pytest.fixture
def ext_config():
    return FakeExtractorArgsConfig()


@pytest.fixture
def imp_manager():
    return _make_impersonation_manager(["Chrome-136:Macos-15"])


@pytest.fixture
def manager(imp_manager, ext_config):
    return EscalationManager(imp_manager, ext_config)


# ---------------------------------------------------------------------------
# AC 1: Test tier progression - record_failure() x threshold triggers escalate
# ---------------------------------------------------------------------------

class TestTierProgression:
    """record_failure() x threshold triggers escalation Tier 1 -> 2 -> 3."""

    def test_starts_at_tier_1(self, manager):
        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY

    def test_single_failure_no_escalation(self, manager):
        manager.record_failure("kw", "HTTP Error 403")
        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY

    def test_threshold_failures_escalate_to_tier_2(self, manager):
        """Two 403 failures (threshold=2) should escalate from Tier 1 to Tier 2."""
        manager.record_failure("kw", "HTTP Error 403")
        manager.record_failure("kw", "HTTP Error 403")
        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.EXTRACTOR_ARGS

    def test_tier_2_to_tier_3_escalation(self, manager):
        """After reaching Tier 2, additional threshold failures -> Tier 3."""
        # Escalate to Tier 2
        manager.record_failure("kw", "HTTP Error 403")
        manager.record_failure("kw", "HTTP Error 403")

        # Now at Tier 2 - escalate() resets consecutive_403s,
        # so we need threshold more failures for Tier 3.
        # But cooldown may suppress - mock time to skip cooldown.
        # Need to patch time in BOTH modules:
        # - escalation_manager.time for _should_escalate()
        # - types.time for EscalationState.escalate()
        far_future = time.time() + 1000
        with patch("src.downloader.escalation_manager.time") as em_time, \
             patch("src.downloader.types.time") as types_time:
            em_time.time.return_value = far_future
            types_time.time.return_value = far_future

            manager.record_failure("kw", "HTTP Error 403")
            manager.record_failure("kw", "HTTP Error 403")

        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.FULL_BYPASS

    def test_no_escalation_beyond_tier_3(self, manager):
        """Tier 3 is the maximum - further failures don't crash."""
        # Fast-track to Tier 3 by mocking time to bypass cooldown
        with patch("src.downloader.escalation_manager.time") as em_time, \
             patch("src.downloader.types.time") as types_time:
            call_count = [0]
            base = 1000.0

            def advancing_time():
                call_count[0] += 1
                return base + call_count[0] * 400  # always past cooldown

            em_time.time.side_effect = lambda: advancing_time()
            types_time.time.side_effect = lambda: advancing_time()

            # Tier 1 -> 2
            manager.record_failure("kw", "403")
            manager.record_failure("kw", "403")
            # Tier 2 -> 3
            manager.record_failure("kw", "403")
            manager.record_failure("kw", "403")
            # Should not crash at Tier 3
            manager.record_failure("kw", "403")
            manager.record_failure("kw", "403")

        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.FULL_BYPASS

    def test_success_resets_403_counter_but_keeps_tier(self, manager):
        """record_success() resets 403 counter but keeps tier sticky."""
        manager.record_failure("kw", "HTTP Error 403")
        manager.record_failure("kw", "HTTP Error 403")
        # Now at Tier 2
        manager.record_success("kw")
        # Tier should stay at 2
        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.EXTRACTOR_ARGS
        # 403 counter should be reset - single failure should not escalate
        manager.record_failure("kw", "403")
        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.EXTRACTOR_ARGS


# ---------------------------------------------------------------------------
# AC 2: Test get_escalation_args() returns correct args per tier
# ---------------------------------------------------------------------------

class TestEscalationArgs:
    """get_escalation_args() returns correct args per tier."""

    def test_tier_1_impersonate_only(self, manager):
        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        assert "--impersonate" in result.args
        assert "--extractor-args" not in result.args
        assert result.rotate_cookies is False

    def test_tier_2_includes_extractor_args(self, manager):
        """Tier 2 should include both --impersonate and --extractor-args."""
        manager.record_failure("kw", "403")
        manager.record_failure("kw", "403")

        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.EXTRACTOR_ARGS
        assert "--impersonate" in result.args
        assert "--extractor-args" in result.args
        # Find the extractor-args value
        ea_idx = result.args.index("--extractor-args")
        ea_val = result.args[ea_idx + 1]
        assert ea_val.startswith("youtube:player_client=")
        assert result.rotate_cookies is False

    def test_tier_3_includes_cookie_flag(self, manager):
        """Tier 3 includes all Tier 2 args plus rotate_cookies=True."""
        with patch("src.downloader.escalation_manager.time") as em_time, \
             patch("src.downloader.types.time") as types_time:
            counter = [0]

            def advancing():
                counter[0] += 1
                return 1000.0 + counter[0] * 400

            em_time.time.side_effect = lambda: advancing()
            types_time.time.side_effect = lambda: advancing()

            # Tier 1 -> 2
            manager.record_failure("kw", "403")
            manager.record_failure("kw", "403")
            # Tier 2 -> 3
            manager.record_failure("kw", "403")
            manager.record_failure("kw", "403")

        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.FULL_BYPASS
        assert "--impersonate" in result.args
        assert "--extractor-args" in result.args
        assert result.rotate_cookies is True

    def test_disabled_extractor_config_tier_2_no_extractor_args(self, imp_manager):
        """With extractor config disabled, Tier 2 has no --extractor-args."""
        config = FakeExtractorArgsConfig(enabled=False)
        mgr = EscalationManager(imp_manager, config)

        mgr.record_failure("kw", "403")
        mgr.record_failure("kw", "403")

        result = mgr.get_escalation_args("kw")
        assert result.tier == EscalationTier.EXTRACTOR_ARGS
        assert "--impersonate" in result.args
        assert "--extractor-args" not in result.args

    def test_empty_player_clients_no_extractor_args(self, imp_manager):
        """Empty player_clients list means no --extractor-args at Tier 2."""
        config = FakeExtractorArgsConfig(player_clients=[])
        mgr = EscalationManager(imp_manager, config)

        mgr.record_failure("kw", "403")
        mgr.record_failure("kw", "403")

        result = mgr.get_escalation_args("kw")
        assert result.tier == EscalationTier.EXTRACTOR_ARGS
        assert "--extractor-args" not in result.args

    def test_no_extractor_config_returns_impersonate_only(self, imp_manager):
        """None extractor config: only impersonate args at any tier."""
        mgr = EscalationManager(imp_manager, None)
        mgr.record_failure("kw", "403")
        mgr.record_failure("kw", "403")

        result = mgr.get_escalation_args("kw")
        assert "--impersonate" in result.args
        assert "--extractor-args" not in result.args


# ---------------------------------------------------------------------------
# AC 3: Test cooldown - escalation suppressed during cooldown, allowed after
# ---------------------------------------------------------------------------

class TestCooldown:
    """Escalation is suppressed during cooldown, allowed after expiry."""

    def test_cooldown_suppresses_escalation(self, manager):
        """After escalation, further failures during cooldown do NOT escalate again."""
        # First escalation: Tier 1 -> 2
        manager.record_failure("kw", "403")
        manager.record_failure("kw", "403")
        assert manager.get_escalation_args("kw").tier == EscalationTier.EXTRACTOR_ARGS

        # More failures within cooldown should NOT escalate to Tier 3
        # (cooldown is 300s, time.time() advances very little between calls)
        manager.record_failure("kw", "403")
        manager.record_failure("kw", "403")
        assert manager.get_escalation_args("kw").tier == EscalationTier.EXTRACTOR_ARGS

    def test_escalation_after_cooldown_expiry(self, manager):
        """After cooldown expires, further failures CAN escalate."""
        # Escalate to Tier 2
        manager.record_failure("kw", "403")
        manager.record_failure("kw", "403")
        assert manager.get_escalation_args("kw").tier == EscalationTier.EXTRACTOR_ARGS

        # Mock time past cooldown (300s) in both modules
        far_future = time.time() + 500
        with patch("src.downloader.escalation_manager.time") as em_time, \
             patch("src.downloader.types.time") as types_time:
            em_time.time.return_value = far_future
            types_time.time.return_value = far_future

            manager.record_failure("kw", "403")
            manager.record_failure("kw", "403")

        assert manager.get_escalation_args("kw").tier == EscalationTier.FULL_BYPASS

    def test_get_cooldown_remaining(self, manager):
        """get_cooldown_remaining() reports correct remaining time."""
        # No state yet -> 0.0
        assert manager.get_cooldown_remaining("kw") == 0.0

        # Trigger escalation
        manager.record_failure("kw", "403")
        manager.record_failure("kw", "403")

        # Cooldown should be > 0 immediately after escalation
        remaining = manager.get_cooldown_remaining("kw")
        assert remaining > 0

    def test_cooldown_returns_zero_after_expiry(self, manager):
        """get_cooldown_remaining() returns 0.0 after cooldown expires."""
        manager.record_failure("kw", "403")
        manager.record_failure("kw", "403")

        with patch("src.downloader.escalation_manager.time") as mock_time:
            mock_time.time.return_value = time.time() + 500
            remaining = manager.get_cooldown_remaining("kw")
            assert remaining == 0.0

    def test_custom_cooldown_seconds(self, imp_manager):
        """Custom cooldown_seconds is respected."""
        config = FakeExtractorArgsConfig(cooldown_seconds=10.0)
        mgr = EscalationManager(imp_manager, config)

        # Escalate to Tier 2
        mgr.record_failure("kw", "403")
        mgr.record_failure("kw", "403")

        # Within 10s cooldown - should NOT escalate
        mgr.record_failure("kw", "403")
        mgr.record_failure("kw", "403")
        assert mgr.get_escalation_args("kw").tier == EscalationTier.EXTRACTOR_ARGS

        # After 10s cooldown - SHOULD escalate
        far_future = time.time() + 20
        with patch("src.downloader.escalation_manager.time") as em_time, \
             patch("src.downloader.types.time") as types_time:
            em_time.time.return_value = far_future
            types_time.time.return_value = far_future
            mgr.record_failure("kw", "403")
            mgr.record_failure("kw", "403")

        assert mgr.get_escalation_args("kw").tier == EscalationTier.FULL_BYPASS


# ---------------------------------------------------------------------------
# AC 4: Test per-keyword isolation
# ---------------------------------------------------------------------------

class TestPerKeywordIsolation:
    """keyword_a at Tier 3 does not affect keyword_b at Tier 1."""

    def test_different_keywords_have_independent_state(self, manager):
        """Escalating keyword_a should not affect keyword_b."""
        # Escalate keyword_a to Tier 2
        manager.record_failure("keyword_a", "403")
        manager.record_failure("keyword_a", "403")

        # keyword_b should still be Tier 1
        result_a = manager.get_escalation_args("keyword_a")
        result_b = manager.get_escalation_args("keyword_b")

        assert result_a.tier == EscalationTier.EXTRACTOR_ARGS
        assert result_b.tier == EscalationTier.IMPERSONATE_ONLY

    def test_keyword_a_tier_3_keyword_b_tier_1(self, manager):
        """Full isolation: keyword_a at Tier 3, keyword_b still at Tier 1."""
        with patch("src.downloader.escalation_manager.time") as em_time, \
             patch("src.downloader.types.time") as types_time:
            counter = [0]

            def advancing():
                counter[0] += 1
                return 1000.0 + counter[0] * 400

            em_time.time.side_effect = lambda: advancing()
            types_time.time.side_effect = lambda: advancing()

            # keyword_a -> Tier 3
            manager.record_failure("keyword_a", "403")
            manager.record_failure("keyword_a", "403")
            manager.record_failure("keyword_a", "403")
            manager.record_failure("keyword_a", "403")

        assert manager.get_escalation_args("keyword_a").tier == EscalationTier.FULL_BYPASS
        assert manager.get_escalation_args("keyword_b").tier == EscalationTier.IMPERSONATE_ONLY

    def test_success_on_one_keyword_doesnt_affect_other(self, manager):
        """record_success('a') does not affect keyword 'b'."""
        manager.record_failure("a", "403")
        manager.record_failure("b", "403")

        manager.record_success("a")

        # 'a' counter reset, 'b' still has 1 failure
        state_a = manager._get_state("a")
        state_b = manager._get_state("b")
        assert state_a.consecutive_403s == 0
        assert state_b.consecutive_403s == 1

    def test_reset_keyword_only_clears_target(self, manager):
        """reset_keyword('a') clears 'a' but not 'b'."""
        manager.record_failure("a", "403")
        manager.record_failure("a", "403")
        manager.record_failure("b", "403")

        manager.reset_keyword("a")

        # 'a' is gone, 'b' untouched
        assert "a" not in manager._keyword_states
        state_b = manager._get_state("b")
        assert state_b.consecutive_403s == 1

    def test_reset_all_clears_everything(self, manager):
        """reset_all() clears all keyword states."""
        manager.record_failure("a", "403")
        manager.record_failure("b", "403")
        manager.record_failure("c", "403")

        manager.reset_all()
        assert len(manager._keyword_states) == 0


# ---------------------------------------------------------------------------
# AC 5: Test thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Concurrent get_escalation_args() calls do not corrupt state."""

    def test_concurrent_get_escalation_args(self, manager):
        """Multiple threads calling get_escalation_args concurrently."""
        results = []
        errors = []

        def worker(keyword, n_calls):
            try:
                for _ in range(n_calls):
                    r = manager.get_escalation_args(keyword)
                    results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"kw_{i}", 50))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 500

    def test_concurrent_record_failure_and_get_args(self, manager):
        """Concurrent record_failure + get_escalation_args on same keyword."""
        errors = []

        def fail_worker(keyword, n):
            try:
                for _ in range(n):
                    manager.record_failure(keyword, "HTTP Error 403")
            except Exception as e:
                errors.append(e)

        def read_worker(keyword, n):
            try:
                for _ in range(n):
                    manager.get_escalation_args(keyword)
            except Exception as e:
                errors.append(e)

        keyword = "shared_kw"
        threads = [
            threading.Thread(target=fail_worker, args=(keyword, 100)),
            threading.Thread(target=read_worker, args=(keyword, 100)),
            threading.Thread(target=fail_worker, args=(keyword, 100)),
            threading.Thread(target=read_worker, args=(keyword, 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_different_keywords(self, manager):
        """Each keyword's state is independent under concurrent access."""
        errors = []

        def escalate_keyword(keyword, n_failures):
            try:
                for _ in range(n_failures):
                    manager.record_failure(keyword, "403")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=escalate_keyword, args=(f"kw_{i}", 5))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        # All 20 keywords should exist
        assert len(manager._keyword_states) == 20

    def test_concurrent_record_success(self, manager):
        """Concurrent record_success + record_failure don't corrupt state."""
        errors = []

        def mixed_worker(keyword):
            try:
                for i in range(50):
                    if i % 3 == 0:
                        manager.record_success(keyword)
                    else:
                        manager.record_failure(keyword, "403")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=mixed_worker, args=(f"kw_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# Bonus: is_escalation_trigger() tests
# ---------------------------------------------------------------------------

class TestIsEscalationTrigger:
    """Verify is_escalation_trigger() pattern matching."""

    @pytest.mark.parametrize("stderr", [
        "ERROR: HTTP Error 403: Forbidden",
        "Sign in to confirm you're not a bot",
        "Please verify you are human",
        "Access blocked by captcha",
        "Request blocked",
        "ERROR: Sign in to confirm your age",
    ])
    def test_matches_known_patterns(self, stderr):
        assert is_escalation_trigger(stderr) is True

    @pytest.mark.parametrize("stderr", [
        "ERROR: HTTP Error 404: Not Found",
        "ERROR: Video unavailable",
        "Download complete",
        "",
    ])
    def test_rejects_non_trigger_patterns(self, stderr):
        assert is_escalation_trigger(stderr) is False

    def test_case_insensitive(self):
        assert is_escalation_trigger("http error 403") is True
        assert is_escalation_trigger("CAPTCHA REQUIRED") is True

    def test_none_safe(self):
        """Empty string returns False (no crash)."""
        assert is_escalation_trigger("") is False


# ---------------------------------------------------------------------------
# Bonus: Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """get_metrics() returns correct summary."""

    def test_initial_metrics(self, manager):
        metrics = manager.get_metrics()
        assert metrics["total_escalations"] == 0
        assert metrics["total_403s"] == 0
        assert metrics["total_successes"] == 0
        assert metrics["average_tier"] == 1.0

    def test_metrics_after_escalation(self, manager):
        manager.record_failure("kw", "403")
        manager.record_failure("kw", "403")
        manager.record_success("kw")

        metrics = manager.get_metrics()
        assert metrics["total_403s"] == 2
        assert metrics["total_successes"] == 1
        assert metrics["total_escalations"] == 1
        assert "EXTRACTOR_ARGS" in metrics["escalations_per_tier"]
        assert "kw" in metrics["keywords_at_each_tier"].get("EXTRACTOR_ARGS", [])
