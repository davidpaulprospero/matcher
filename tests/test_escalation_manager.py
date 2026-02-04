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
    classify_trigger,
    is_escalation_trigger,
)
from src.downloader.types import EscalationState, EscalationTier
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@contextmanager
def patch_time_modules(time_func):
    """Patch time.time() in all escalation-related modules.

    Since US-35-009 extracted EscalationStrategy from EscalationManager,
    we now need to patch time in three modules:
    - escalation_manager.time
    - escalation_strategy.time
    - types.time

    Args:
        time_func: A callable that returns the mocked time value.
            Can be a lambda, a function, or a MagicMock side_effect.
    """
    with patch("src.downloader.escalation_manager.time") as em_time, \
         patch("src.downloader.escalation_strategy.time") as strat_time, \
         patch("src.downloader.types.time") as types_time:

        if callable(time_func):
            em_time.time.side_effect = time_func
            strat_time.time.side_effect = time_func
            types_time.time.side_effect = time_func
        else:
            em_time.time.return_value = time_func
            strat_time.time.return_value = time_func
            types_time.time.return_value = time_func

        yield (em_time, strat_time, types_time)

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
# AC 1: Test tier progression - record_failure() x threshold triggers escalate
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestTierProgression:
    """record_failure() x threshold triggers escalation Tier 1 -> 2 -> 3."""

    def test_starts_at_tier_1(self, manager):
        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY

    @pytest.mark.fast
    def test_single_failure_no_escalation(self, manager):
        manager.record_failure("kw", "HTTP Error 403")
        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY

    @pytest.mark.fast
    def test_threshold_failures_escalate_to_tier_2(self, manager):
        """Two 403 failures (threshold=2) should escalate from Tier 1 to Tier 2."""
        manager.record_failure("kw", "HTTP Error 403")
        manager.record_failure("kw", "HTTP Error 403")
        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.EXTRACTOR_ARGS

    @pytest.mark.fast
    def test_tier_2_to_tier_3_escalation(self, manager):
        """After reaching Tier 2, additional threshold failures -> Tier 3."""
        # Escalate to Tier 2
        manager.record_failure("kw", "HTTP Error 403")
        manager.record_failure("kw", "HTTP Error 403")

        # Now at Tier 2 - escalate() resets consecutive_403s,
        # so we need threshold more failures for Tier 3.
        # But cooldown may suppress - mock time to skip cooldown.
        far_future = time.time() + 1000
        with patch_time_modules(far_future):
            manager.record_failure("kw", "HTTP Error 403")
            manager.record_failure("kw", "HTTP Error 403")

        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.FULL_BYPASS

    @pytest.mark.fast
    def test_no_escalation_beyond_tier_3(self, manager):
        """Tier 3 is the maximum - further failures don't crash."""
        # Fast-track to Tier 3 by mocking time to bypass cooldown
        call_count = [0]
        base = 1000.0

        def advancing_time():
            call_count[0] += 1
            return base + call_count[0] * 400  # always past cooldown

        with patch_time_modules(advancing_time):
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

    @pytest.mark.fast
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

@pytest.mark.fast
class TestEscalationArgs:
    """get_escalation_args() returns correct args per tier."""

    def test_tier_1_impersonate_only(self, manager):
        result = manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        assert "--impersonate" in result.args
        assert "--extractor-args" not in result.args
        assert result.rotate_cookies is False

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_tier_3_includes_cookie_flag(self, manager):
        """Tier 3 includes all Tier 2 args plus rotate_cookies=True."""
        counter = [0]

        def advancing():
            counter[0] += 1
            return 1000.0 + counter[0] * 400

        with patch_time_modules(advancing):
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_empty_player_clients_no_extractor_args(self, imp_manager):
        """Empty player_clients list means no --extractor-args at Tier 2."""
        config = FakeExtractorArgsConfig(player_clients=[])
        mgr = EscalationManager(imp_manager, config)

        mgr.record_failure("kw", "403")
        mgr.record_failure("kw", "403")

        result = mgr.get_escalation_args("kw")
        assert result.tier == EscalationTier.EXTRACTOR_ARGS
        assert "--extractor-args" not in result.args

    @pytest.mark.fast
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

@pytest.mark.fast
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

    @pytest.mark.fast
    def test_escalation_after_cooldown_expiry(self, manager):
        """After cooldown expires, further failures CAN escalate."""
        # Escalate to Tier 2
        manager.record_failure("kw", "403")
        manager.record_failure("kw", "403")
        assert manager.get_escalation_args("kw").tier == EscalationTier.EXTRACTOR_ARGS

        # Mock time past cooldown (300s)
        far_future = time.time() + 500
        with patch_time_modules(far_future):
            manager.record_failure("kw", "403")
            manager.record_failure("kw", "403")

        assert manager.get_escalation_args("kw").tier == EscalationTier.FULL_BYPASS

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_cooldown_returns_zero_after_expiry(self, manager):
        """get_cooldown_remaining() returns 0.0 after cooldown expires."""
        manager.record_failure("kw", "403")
        manager.record_failure("kw", "403")

        far_future = time.time() + 500
        with patch_time_modules(far_future):
            remaining = manager.get_cooldown_remaining("kw")
            assert remaining == 0.0

    @pytest.mark.fast
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
        with patch_time_modules(far_future):
            mgr.record_failure("kw", "403")
            mgr.record_failure("kw", "403")

        assert mgr.get_escalation_args("kw").tier == EscalationTier.FULL_BYPASS


# ---------------------------------------------------------------------------
# AC 4: Test per-keyword isolation
# ---------------------------------------------------------------------------

@pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

@pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

@pytest.mark.fast
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
    @pytest.mark.fast
    def test_matches_known_patterns(self, stderr):
        assert is_escalation_trigger(stderr) is True

    # -- New: 429 / rate-limit patterns --
    @pytest.mark.parametrize("stderr", [
        "ERROR: HTTP Error 429: Too Many Requests",
        "HTTP Error 429",
        "Too Many Requests",
        "rate limit exceeded for this IP",
        "rate-limit reached",
    ])
    @pytest.mark.fast
    def test_matches_429_rate_limit_patterns(self, stderr):
        assert is_escalation_trigger(stderr) is True

    # -- New: IP-blocked patterns --
    @pytest.mark.parametrize("stderr", [
        "Your IP address has been blocked",
        "ip blocked due to abuse",
        "access denied from your region",
        "geo-blocked content",
        "geoblock: video not available in your country",
    ])
    @pytest.mark.fast
    def test_matches_ip_blocked_patterns(self, stderr):
        assert is_escalation_trigger(stderr) is True

    # -- New: Age-gate patterns --
    @pytest.mark.parametrize("stderr", [
        "This video is age-gated",
        "age-restricted content",
        "Sign in to confirm your age. This video may be inappropriate",
        "age gate verification required",
        "agerestricted: please sign in",
    ])
    @pytest.mark.fast
    def test_matches_age_gate_patterns(self, stderr):
        assert is_escalation_trigger(stderr) is True

    @pytest.mark.parametrize("stderr", [
        "ERROR: HTTP Error 404: Not Found",
        "ERROR: Video unavailable",
        "Download complete",
        "",
    ])
    @pytest.mark.fast
    def test_rejects_non_trigger_patterns(self, stderr):
        assert is_escalation_trigger(stderr) is False

    @pytest.mark.fast
    def test_case_insensitive(self):
        assert is_escalation_trigger("http error 403") is True
        assert is_escalation_trigger("CAPTCHA REQUIRED") is True

    @pytest.mark.fast
    def test_none_safe(self):
        """Empty string returns False (no crash)."""
        assert is_escalation_trigger("") is False


# ---------------------------------------------------------------------------
# Bonus: Metrics
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestMetrics:
    """get_metrics() returns correct summary."""

    def test_initial_metrics(self, manager):
        metrics = manager.get_metrics()
        assert metrics["total_escalations"] == 0
        assert metrics["total_403s"] == 0
        assert metrics["total_successes"] == 0
        assert metrics["average_tier"] == 1.0

    @pytest.mark.fast
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


# ---------------------------------------------------------------------------
# classify_trigger() tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestClassifyTrigger:
    """Verify classify_trigger() returns correct category strings."""

    @pytest.mark.parametrize("stderr,expected", [
        # 429 / rate-limit
        ("ERROR: HTTP Error 429: Too Many Requests", "429"),
        ("Too Many Requests", "429"),
        ("rate limit exceeded", "429"),
        ("rate-limit reached for this IP", "429"),
        # IP-blocked
        ("Your IP address has been blocked", "ip_blocked"),
        ("ip blocked due to abuse", "ip_blocked"),
        ("access denied from your region", "ip_blocked"),
        ("geo-blocked content", "ip_blocked"),
        # Bot detection
        ("you're not a bot, are you?", "bot_detection"),
        ("captcha verification required", "bot_detection"),
        ("Please verify you are human", "bot_detection"),
        # Age-gate
        ("This video is age-gated", "age_gate"),
        ("age-restricted content", "age_gate"),
        ("Sign in to confirm your age", "age_gate"),
        # Bot detection (contains 'bot')
        ("Sign in to confirm you're not a bot", "bot_detection"),
        # 403 (generic blocking)
        ("ERROR: HTTP Error 403: Forbidden", "403"),
        ("Request blocked", "403"),
        ("Sign in to confirm identity", "403"),
    ])
    @pytest.mark.fast
    def test_classify_known_patterns(self, stderr, expected):
        assert classify_trigger(stderr) == expected

    @pytest.mark.fast
    def test_classify_returns_none_for_non_triggers(self):
        assert classify_trigger("") is None
        assert classify_trigger("ERROR: HTTP Error 404: Not Found") is None
        assert classify_trigger("Download complete") is None
        assert classify_trigger("Video unavailable") is None

    @pytest.mark.fast
    def test_classify_case_insensitive(self):
        assert classify_trigger("HTTP ERROR 429") == "429"
        assert classify_trigger("AGE-GATED VIDEO") == "age_gate"
        assert classify_trigger("ACCESS DENIED") == "ip_blocked"

    @pytest.mark.fast
    def test_classify_priority_429_over_403(self):
        """429 patterns match '429' category, not '403'."""
        assert classify_trigger("HTTP Error 429") == "429"

    @pytest.mark.fast
    def test_classify_priority_age_gate_over_403(self):
        """'Sign in to confirm your age' matches 'age_gate', not '403'."""
        # This matches age_gate because age_gate is checked before 403
        assert classify_trigger("Sign in to confirm your age") == "age_gate"

    @pytest.mark.fast
    def test_classify_priority_ip_blocked_over_403(self):
        """IP block patterns match 'ip_blocked', not '403'."""
        assert classify_trigger("ip blocked") == "ip_blocked"


# ---------------------------------------------------------------------------
# Budget-Aware Escalation Tests (US-002)
# ---------------------------------------------------------------------------

from src.downloader.rate_limit_budget import RateLimitBudget


@pytest.fixture
def budget():
    """Create a RateLimitBudget with limited rotations for testing."""
    b = RateLimitBudget()
    b.max_rotations = 3
    b.max_backoff_time = 60.0
    return b


@pytest.fixture
def exhausted_budget():
    """Create an exhausted budget (can_rotate() returns False)."""
    b = RateLimitBudget()
    b.max_rotations = 2
    b.rotations_used = 2  # At limit
    b.max_backoff_time = 10.0
    b.backoff_time_spent = 10.0  # At limit
    return b


@pytest.fixture
def budget_manager(imp_manager, ext_config, budget):
    """EscalationManager with a budget attached."""
    return EscalationManager(imp_manager, ext_config, budget=budget)


@pytest.fixture
def exhausted_budget_manager(imp_manager, ext_config, exhausted_budget):
    """EscalationManager with an exhausted budget."""
    return EscalationManager(imp_manager, ext_config, budget=exhausted_budget)


@pytest.mark.fast
class TestBudgetAwareEscalation:
    """Tests for US-002: Wire RateLimitBudget into EscalationManager."""

    def test_constructor_stores_budget(self, budget_manager, budget):
        """EscalationManager stores the budget reference."""
        assert budget_manager._budget is budget

    @pytest.mark.fast
    def test_constructor_budget_none_is_safe(self, imp_manager, ext_config):
        """EscalationManager works fine without a budget (None)."""
        mgr = EscalationManager(imp_manager, ext_config, budget=None)
        assert mgr._budget is None
        # Normal operations still work
        result = mgr.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        mgr.record_failure("kw", "HTTP Error 403")
        mgr.record_success("kw")

    @pytest.mark.fast
    def test_get_escalation_args_calls_record_attempt(self, imp_manager, ext_config):
        """get_escalation_args() calls budget.record_attempt()."""
        mock_budget = MagicMock()
        mock_budget.can_rotate.return_value = True
        mgr = EscalationManager(imp_manager, ext_config, budget=mock_budget)
        mgr.get_escalation_args("sunset")
        mock_budget.record_attempt.assert_called_once_with("sunset")

    @pytest.mark.fast
    def test_record_failure_records_rotation_on_tier_advance(self, budget_manager, budget):
        """record_failure() calls budget.record_rotation() when tier advances."""
        assert budget.rotations_used == 0
        # Trigger escalation: 2 consecutive failures (threshold=2)
        budget_manager.record_failure("kw", "HTTP Error 403")
        budget_manager.record_failure("kw", "HTTP Error 403")
        # Should have recorded a rotation for the Tier 1 -> Tier 2 advance
        assert budget.rotations_used == 1

    @pytest.mark.fast
    def test_record_failure_records_rotation_on_each_advance(self, budget_manager, budget):
        """Each tier advance records a rotation in the budget."""
        # Escalate Tier 1 -> Tier 2 (2 failures)
        budget_manager.record_failure("kw", "HTTP Error 403")
        budget_manager.record_failure("kw", "HTTP Error 403")
        assert budget.rotations_used == 1

        # Advance past cooldown so second escalation can proceed
        far_future = time.time() + 400  # Past 300s cooldown
        with patch_time_modules(far_future):
            # Escalate Tier 2 -> Tier 3 (2 more failures)
            budget_manager.record_failure("kw", "HTTP Error 403")
            budget_manager.record_failure("kw", "HTTP Error 403")
        assert budget.rotations_used == 2

    @pytest.mark.fast
    def test_exhausted_budget_skips_to_max_tier(self, exhausted_budget_manager):
        """When budget is exhausted, record_failure skips to FULL_BYPASS."""
        mgr = exhausted_budget_manager
        # Trigger escalation threshold (2 failures)
        mgr.record_failure("kw", "HTTP Error 403")
        mgr.record_failure("kw", "HTTP Error 403")
        # Should have skipped straight to FULL_BYPASS
        result = mgr.get_escalation_args("kw")
        assert result.tier == EscalationTier.FULL_BYPASS
        assert result.rotate_cookies is True

    @pytest.mark.fast
    def test_exhausted_budget_skips_from_tier1_to_tier3(self, exhausted_budget_manager):
        """Exhausted budget skips from Tier 1 directly to Tier 3."""
        mgr = exhausted_budget_manager
        # Verify starts at Tier 1
        result = mgr.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        # 2 failures should skip Tier 2 and go directly to Tier 3
        mgr.record_failure("kw", "HTTP Error 403")
        mgr.record_failure("kw", "HTTP Error 403")
        result = mgr.get_escalation_args("kw")
        assert result.tier == EscalationTier.FULL_BYPASS

    @pytest.mark.fast
    def test_budget_none_normal_escalation(self, manager):
        """Without budget, escalation proceeds normally through all tiers."""
        assert manager._budget is None
        # Tier 1 -> 2
        manager.record_failure("kw", "HTTP Error 403")
        manager.record_failure("kw", "HTTP Error 403")
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.EXTRACTOR_ARGS
        # Advance past cooldown for second escalation
        far_future = time.time() + 400
        with patch_time_modules(far_future):
            # Tier 2 -> 3
            manager.record_failure("kw", "HTTP Error 403")
            manager.record_failure("kw", "HTTP Error 403")
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.FULL_BYPASS

    @pytest.mark.fast
    def test_budget_tracks_rotation_with_keyword(self, budget_manager, budget):
        """Budget rotation is recorded with the correct keyword."""
        budget_manager.record_failure("ocean", "HTTP Error 403")
        budget_manager.record_failure("ocean", "HTTP Error 403")
        assert "ocean" in budget.keywords_rate_limited


# ---------------------------------------------------------------------------
# US-006: Speed tracker signal consumption for preemptive escalation
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestSpeedTriggeredEscalation:
    """Tests for record_slow_speed() preemptive escalation."""

    def test_three_slow_speed_signals_escalate_tier(self, manager):
        """3+ slow speed signals for same keyword escalate one tier."""
        # Starts at Tier 1
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.IMPERSONATE_ONLY

        # 1st and 2nd signals: no escalation yet
        manager.record_slow_speed("kw", speed_mbps=0.05)
        manager.record_slow_speed("kw", speed_mbps=0.03)
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.IMPERSONATE_ONLY

        # 3rd signal: triggers escalation to Tier 2
        manager.record_slow_speed("kw", speed_mbps=0.08)
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.EXTRACTOR_ARGS

    @pytest.mark.fast
    def test_speed_signals_do_not_affect_budget(self, imp_manager, ext_config):
        """Speed-triggered escalations do NOT call budget.record_rotation()."""
        budget = MagicMock()
        budget.can_rotate.return_value = True
        mgr = EscalationManager(imp_manager, ext_config, budget=budget)

        # Trigger 3 slow speed signals -> escalation
        mgr.record_slow_speed("kw", speed_mbps=0.05)
        mgr.record_slow_speed("kw", speed_mbps=0.05)
        mgr.record_slow_speed("kw", speed_mbps=0.05)

        # Budget should NOT have record_rotation called
        budget.record_rotation.assert_not_called()

    @pytest.mark.fast
    def test_speed_escalations_counter_in_metrics(self, manager):
        """speed_escalations counter appears in get_metrics()."""
        metrics = manager.get_metrics()
        assert 'speed_escalations' in metrics
        assert metrics['speed_escalations'] == 0

        # Trigger a speed escalation
        manager.record_slow_speed("kw", 0.05)
        manager.record_slow_speed("kw", 0.05)
        manager.record_slow_speed("kw", 0.05)

        metrics = manager.get_metrics()
        assert metrics['speed_escalations'] == 1

    @pytest.mark.fast
    def test_speed_escalation_resets_counter_after_trigger(self, manager):
        """After speed escalation, counter resets so 3 more signals needed."""
        # First escalation: Tier 1 -> 2
        for _ in range(3):
            manager.record_slow_speed("kw", 0.05)
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.EXTRACTOR_ARGS

        # Advance past cooldown for second escalation
        advancing = _advance_past_cooldown()
        with patch_time_modules(advancing):
            # 1-2 more signals: no escalation yet
            manager.record_slow_speed("kw", 0.05)
            manager.record_slow_speed("kw", 0.05)
            r = manager.get_escalation_args("kw")
            assert r.tier == EscalationTier.EXTRACTOR_ARGS

            # 3rd signal after reset: Tier 2 -> 3
            manager.record_slow_speed("kw", 0.05)
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.FULL_BYPASS

    @pytest.mark.fast
    def test_speed_escalation_stops_at_max_tier(self, manager):
        """Speed escalation does not go beyond FULL_BYPASS."""
        # Escalate to Tier 3 via two rounds of speed signals (with cooldown advance)
        for _ in range(3):
            manager.record_slow_speed("kw", 0.05)

        advancing = _advance_past_cooldown()
        with patch_time_modules(advancing):
            for _ in range(3):
                manager.record_slow_speed("kw", 0.05)

        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.FULL_BYPASS

        # More speed signals don't crash or go beyond Tier 3
        for _ in range(3):
            manager.record_slow_speed("kw", 0.05)
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.FULL_BYPASS

        # Counter should be 2 (two successful speed escalations: 1->2, 2->3)
        metrics = manager.get_metrics()
        assert metrics['speed_escalations'] == 2

    @pytest.mark.fast
    def test_speed_signals_per_keyword_isolation(self, manager):
        """Speed signals are tracked per-keyword independently."""
        # 2 signals for kw1
        manager.record_slow_speed("kw1", 0.05)
        manager.record_slow_speed("kw1", 0.05)

        # 3 signals for kw2 -> triggers escalation for kw2 only
        manager.record_slow_speed("kw2", 0.05)
        manager.record_slow_speed("kw2", 0.05)
        manager.record_slow_speed("kw2", 0.05)

        r1 = manager.get_escalation_args("kw1")
        r2 = manager.get_escalation_args("kw2")
        assert r1.tier == EscalationTier.IMPERSONATE_ONLY
        assert r2.tier == EscalationTier.EXTRACTOR_ARGS

    @pytest.mark.fast
    def test_speed_escalation_logged_at_info(self, manager):
        """Speed-triggered escalation is logged at INFO level."""
        with patch("src.downloader.escalation_manager.logger") as mock_logger:
            manager.record_slow_speed("kw", 0.05)
            manager.record_slow_speed("kw", 0.05)
            manager.record_slow_speed("kw", 0.05)

            # Check that INFO log was emitted with expected message
            mock_logger.info.assert_called()
            log_msg = mock_logger.info.call_args[0][0]
            assert "Preemptive escalation" in log_msg
            assert "kw" in log_msg
            assert "low speed" in log_msg

    @pytest.mark.fast
    def test_reset_all_clears_speed_state(self, manager):
        """reset_all() clears speed counters and escalation count."""
        manager.record_slow_speed("kw", 0.05)
        manager.record_slow_speed("kw", 0.05)
        manager.record_slow_speed("kw", 0.05)
        assert manager.get_metrics()['speed_escalations'] == 1

        manager.reset_all()

        assert manager.get_metrics()['speed_escalations'] == 0
        # After reset, 3 more signals needed for escalation
        manager.record_slow_speed("kw", 0.05)
        manager.record_slow_speed("kw", 0.05)
        # Only 2 signals - should still be at Tier 1
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.IMPERSONATE_ONLY


# ---------------------------------------------------------------------------
# Speed escalation cooldown (Sprint 12 US-001)
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestSpeedEscalationCooldown:
    """Verify record_slow_speed() respects the cooldown window."""

    def test_speed_escalation_blocked_during_cooldown(self, imp_manager):
        """Speed escalation after initial escalation is suppressed within cooldown."""
        config = FakeExtractorArgsConfig(cooldown_seconds=300.0)
        manager = EscalationManager(imp_manager, config)

        # First round: escalate Tier 1 -> 2 via 3 slow signals
        for _ in range(3):
            manager.record_slow_speed("kw", 0.05)
        assert manager.get_escalation_args("kw").tier == EscalationTier.EXTRACTOR_ARGS
        assert manager.get_metrics()['speed_escalations'] == 1

        # Second round immediately (still in cooldown): should NOT escalate
        for _ in range(3):
            manager.record_slow_speed("kw", 0.05)
        # Tier should still be EXTRACTOR_ARGS (not FULL_BYPASS)
        assert manager.get_escalation_args("kw").tier == EscalationTier.EXTRACTOR_ARGS
        assert manager.get_metrics()['speed_escalations'] == 1

    @pytest.mark.fast
    def test_speed_escalation_proceeds_after_cooldown(self, imp_manager):
        """Speed escalation proceeds once cooldown has elapsed."""
        config = FakeExtractorArgsConfig(cooldown_seconds=300.0)
        manager = EscalationManager(imp_manager, config)

        # First escalation: Tier 1 -> 2
        for _ in range(3):
            manager.record_slow_speed("kw", 0.05)
        assert manager.get_escalation_args("kw").tier == EscalationTier.EXTRACTOR_ARGS

        # Advance past cooldown
        advancing = _advance_past_cooldown()
        with patch_time_modules(advancing):
            # Second escalation after cooldown: Tier 2 -> 3
            for _ in range(3):
                manager.record_slow_speed("kw", 0.05)

        assert manager.get_escalation_args("kw").tier == EscalationTier.FULL_BYPASS
        assert manager.get_metrics()['speed_escalations'] == 2

    @pytest.mark.fast
    def test_speed_cooldown_zero_allows_immediate_escalation(self, imp_manager):
        """With cooldown_seconds=0, speed escalation is never suppressed."""
        config = FakeExtractorArgsConfig(cooldown_seconds=0.0)
        manager = EscalationManager(imp_manager, config)

        # First: Tier 1 -> 2
        for _ in range(3):
            manager.record_slow_speed("kw", 0.05)
        assert manager.get_escalation_args("kw").tier == EscalationTier.EXTRACTOR_ARGS

        # Immediately: Tier 2 -> 3
        for _ in range(3):
            manager.record_slow_speed("kw", 0.05)
        assert manager.get_escalation_args("kw").tier == EscalationTier.FULL_BYPASS
        assert manager.get_metrics()['speed_escalations'] == 2


# ---------------------------------------------------------------------------
# Checkpoint Persistence (Sprint 10 US-007)
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestEscalationCheckpointPersistence:
    """Tests for to_dict() / from_dict() serialization and resume support."""

    def test_to_dict_empty_manager(self, imp_manager, ext_config):
        """to_dict() with no keyword states returns valid structure."""
        manager = EscalationManager(imp_manager, ext_config)
        data = manager.to_dict()

        assert 'keyword_states' in data
        assert data['keyword_states'] == {}
        assert data['total_403s'] == 0
        assert data['total_successes'] == 0
        assert data['total_escalations'] == 0
        assert 'saved_at' in data
        assert isinstance(data['saved_at'], float)

    @pytest.mark.fast
    def test_to_dict_with_keyword_states(self, imp_manager, ext_config):
        """to_dict() serializes per-keyword tier, 403 count, etc."""
        manager = EscalationManager(imp_manager, ext_config)
        # Escalate "alpha" to Tier 2
        manager.record_failure("alpha")
        manager.record_failure("alpha")
        # "beta" stays at Tier 1
        manager.record_success("beta")

        data = manager.to_dict()

        assert "alpha" in data['keyword_states']
        assert "beta" in data['keyword_states']
        alpha = data['keyword_states']['alpha']
        assert alpha['tier'] == EscalationTier.EXTRACTOR_ARGS.value  # 2
        assert alpha['consecutive_403s'] == 0  # reset after escalation
        assert 'extractor_args_index' in alpha
        assert 'last_escalation_time' in alpha

        beta = data['keyword_states']['beta']
        assert beta['tier'] == EscalationTier.IMPERSONATE_ONLY.value  # 1
        assert beta['consecutive_403s'] == 0  # reset on success

        assert data['total_403s'] == 2
        assert data['total_successes'] == 1
        assert data['total_escalations'] == 1

    @pytest.mark.fast
    def test_round_trip_serialize_deserialize(self, imp_manager):
        """Serialize then deserialize preserves all keyword states and counters."""
        # Use 0s cooldown to allow rapid escalation in test
        no_cooldown_config = FakeExtractorArgsConfig(cooldown_seconds=0.0)
        manager = EscalationManager(imp_manager, no_cooldown_config)
        # Escalate "kw1" to Tier 2, "kw2" to Tier 3
        manager.record_failure("kw1")
        manager.record_failure("kw1")  # -> Tier 2
        manager.record_failure("kw2")
        manager.record_failure("kw2")  # -> Tier 2
        manager.record_failure("kw2")
        manager.record_failure("kw2")  # -> Tier 3
        manager.record_success("kw1")

        data = manager.to_dict()

        # Restore
        restored = EscalationManager.from_dict(
            data=data,
            impersonation_manager=imp_manager,
            extractor_args_config=no_cooldown_config,
            stale_threshold=999999.0,  # Not stale
        )

        # Check kw1 is still at Tier 2
        r1 = restored.get_escalation_args("kw1")
        assert r1.tier == EscalationTier.EXTRACTOR_ARGS

        # Check kw2 is still at Tier 3
        r2 = restored.get_escalation_args("kw2")
        assert r2.tier == EscalationTier.FULL_BYPASS
        assert r2.rotate_cookies is True

        # Check global counters
        metrics = restored.get_metrics()
        assert metrics['total_403s'] == 6  # 2 (kw1) + 4 (kw2)
        assert metrics['total_successes'] == 1
        assert metrics['total_escalations'] == 3  # kw1 once, kw2 twice

    @pytest.mark.fast
    def test_stale_data_de_escalates_by_one_tier(self, imp_manager):
        """If checkpoint is >1 hour old, all keywords de-escalate by one tier."""
        no_cooldown_config = FakeExtractorArgsConfig(cooldown_seconds=0.0)
        manager = EscalationManager(imp_manager, no_cooldown_config)
        # Escalate "kw1" to Tier 2, "kw2" to Tier 3
        manager.record_failure("kw1")
        manager.record_failure("kw1")  # -> Tier 2
        manager.record_failure("kw2")
        manager.record_failure("kw2")  # -> Tier 2
        manager.record_failure("kw2")
        manager.record_failure("kw2")  # -> Tier 3

        data = manager.to_dict()
        # Simulate stale data: saved_at = 2 hours ago
        data['saved_at'] = time.time() - 7200

        restored = EscalationManager.from_dict(
            data=data,
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
            stale_threshold=3600.0,  # 1 hour
        )

        # kw1 was Tier 2, should be Tier 1 now
        r1 = restored.get_escalation_args("kw1")
        assert r1.tier == EscalationTier.IMPERSONATE_ONLY

        # kw2 was Tier 3, should be Tier 2 now
        r2 = restored.get_escalation_args("kw2")
        assert r2.tier == EscalationTier.EXTRACTOR_ARGS

    @pytest.mark.fast
    def test_stale_tier1_stays_at_tier1(self, imp_manager, ext_config):
        """Stale de-escalation does not go below Tier 1."""
        manager = EscalationManager(imp_manager, ext_config)
        manager.record_success("kw1")  # Tier 1

        data = manager.to_dict()
        data['saved_at'] = time.time() - 7200  # Stale

        restored = EscalationManager.from_dict(
            data=data,
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
            stale_threshold=3600.0,
        )

        r = restored.get_escalation_args("kw1")
        assert r.tier == EscalationTier.IMPERSONATE_ONLY  # Still Tier 1

    @pytest.mark.fast
    def test_missing_checkpoint_data_safe_fallback(self, imp_manager, ext_config):
        """from_dict() with None/empty/invalid data returns fresh manager."""
        # None data
        m1 = EscalationManager.from_dict(
            data=None,
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
        )
        assert m1.get_metrics()['total_403s'] == 0
        r = m1.get_escalation_args("test")
        assert r.tier == EscalationTier.IMPERSONATE_ONLY

        # Empty dict
        m2 = EscalationManager.from_dict(
            data={},
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
        )
        assert m2.get_metrics()['total_403s'] == 0

        # Invalid type
        m3 = EscalationManager.from_dict(
            data="not a dict",
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
        )
        assert m3.get_metrics()['total_403s'] == 0

    @pytest.mark.fast
    def test_from_dict_with_invalid_tier_value_clamped(self, imp_manager, ext_config):
        """Invalid tier values in checkpoint are clamped to valid range."""
        data = {
            'keyword_states': {
                'kw_high': {'tier': 99, 'consecutive_403s': 0, 'extractor_args_index': 0, 'last_escalation_time': None},
                'kw_low': {'tier': -5, 'consecutive_403s': 0, 'extractor_args_index': 0, 'last_escalation_time': None},
            },
            'total_403s': 0,
            'total_successes': 0,
            'total_escalations': 0,
            'escalations_per_tier': {},
            'speed_escalations': 0,
            'saved_at': time.time(),
        }

        restored = EscalationManager.from_dict(
            data=data,
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
            stale_threshold=999999.0,
        )

        # High value clamped to FULL_BYPASS (3)
        r_high = restored.get_escalation_args("kw_high")
        assert r_high.tier == EscalationTier.FULL_BYPASS

        # Low value clamped to IMPERSONATE_ONLY (1)
        r_low = restored.get_escalation_args("kw_low")
        assert r_low.tier == EscalationTier.IMPERSONATE_ONLY

    @pytest.mark.fast
    def test_speed_escalations_counter_persisted(self, imp_manager, ext_config):
        """speed_escalations counter survives round-trip."""
        manager = EscalationManager(imp_manager, ext_config)
        manager.record_slow_speed("kw", 0.05)
        manager.record_slow_speed("kw", 0.05)
        manager.record_slow_speed("kw", 0.05)  # Triggers speed escalation

        data = manager.to_dict()
        assert data['speed_escalations'] == 1

        restored = EscalationManager.from_dict(
            data=data,
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
            stale_threshold=999999.0,
        )

        metrics = restored.get_metrics()
        assert metrics['speed_escalations'] == 1

    @pytest.mark.fast
    def test_from_dict_missing_saved_at_treats_as_stale(self, imp_manager, ext_config):
        """Missing 'saved_at' key defaults to epoch 0, treating data as stale and de-escalating."""
        data = {
            'keyword_states': {
                'kw_tier2': {'tier': 2, 'consecutive_403s': 1, 'extractor_args_index': 1, 'last_escalation_time': None},
                'kw_tier3': {'tier': 3, 'consecutive_403s': 0, 'extractor_args_index': 2, 'last_escalation_time': None},
                'kw_tier1': {'tier': 1, 'consecutive_403s': 0, 'extractor_args_index': 0, 'last_escalation_time': None},
            },
            'total_403s': 5,
            'total_successes': 2,
            'total_escalations': 3,
            'escalations_per_tier': {},
            'speed_escalations': 0,
            # NOTE: no 'saved_at' key — should default to 0.0 (epoch), making it very stale
        }

        restored = EscalationManager.from_dict(
            data=data,
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
            stale_threshold=3600.0,
        )

        # Tier 2 → Tier 1 (de-escalated by 1)
        r2 = restored.get_escalation_args("kw_tier2")
        assert r2.tier == EscalationTier.IMPERSONATE_ONLY

        # Tier 3 → Tier 2 (de-escalated by 1)
        r3 = restored.get_escalation_args("kw_tier3")
        assert r3.tier == EscalationTier.EXTRACTOR_ARGS

        # Tier 1 stays at Tier 1 (can't go below)
        r1 = restored.get_escalation_args("kw_tier1")
        assert r1.tier == EscalationTier.IMPERSONATE_ONLY

    @pytest.mark.fast
    def test_from_dict_recent_saved_at_preserves_tiers(self, imp_manager, ext_config):
        """Checkpoint saved within stale_threshold preserves tiers exactly."""
        data = {
            'keyword_states': {
                'kw_tier2': {'tier': 2, 'consecutive_403s': 1, 'extractor_args_index': 1, 'last_escalation_time': None},
                'kw_tier3': {'tier': 3, 'consecutive_403s': 0, 'extractor_args_index': 2, 'last_escalation_time': None},
            },
            'total_403s': 4,
            'total_successes': 1,
            'total_escalations': 2,
            'escalations_per_tier': {},
            'speed_escalations': 0,
            'saved_at': time.time() - 60,  # 1 minute ago, well within 1 hour threshold
        }

        restored = EscalationManager.from_dict(
            data=data,
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
            stale_threshold=3600.0,
        )

        # Tiers preserved exactly
        r2 = restored.get_escalation_args("kw_tier2")
        assert r2.tier == EscalationTier.EXTRACTOR_ARGS

        r3 = restored.get_escalation_args("kw_tier3")
        assert r3.tier == EscalationTier.FULL_BYPASS

    @pytest.mark.fast
    def test_from_dict_empty_keyword_states_returns_fresh(self, imp_manager, ext_config):
        """from_dict() with empty keyword_states dict returns working manager with no crash."""
        data = {
            'keyword_states': {},
            'total_403s': 0,
            'total_successes': 0,
            'total_escalations': 0,
            'escalations_per_tier': {},
            'speed_escalations': 0,
            'saved_at': time.time(),
        }

        restored = EscalationManager.from_dict(
            data=data,
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
        )

        # Should have no tracked keywords
        assert restored.get_active_keyword_count() == 0
        assert restored.get_metrics()['total_403s'] == 0

        # Should still be functional for new keywords
        result = restored.get_escalation_args("new_keyword")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY


# ---------------------------------------------------------------------------
# Factory Method: create_with_mullvad (Sprint 35 US-003)
# ---------------------------------------------------------------------------

@dataclass
class FakeMullvadConfig:
    """Minimal stand-in for MullvadConfig."""
    enabled: bool = False
    preferred_countries: List[str] = field(default_factory=lambda: ['us', 'gb'])
    rotation_strategy: str = 'random'
    max_rotations_per_session: int = 5
    verification_timeout: int = 10
    rotation_delay_seconds: int = 5


@pytest.mark.fast
class TestCreateWithMullvad:
    """Tests for EscalationManager.create_with_mullvad() factory method (US-35-003)."""

    def test_create_with_mullvad_enabled(self, imp_manager, ext_config):
        """Factory method creates EscalationManager with MullvadVPN when enabled."""
        mullvad_config = FakeMullvadConfig(enabled=True)

        # Patch at module level where the import happens inside the factory method
        with patch('src.downloader.mullvad_vpn.MullvadVPN') as MockVPN:
            mock_vpn_instance = MagicMock()
            MockVPN.return_value = mock_vpn_instance

            manager = EscalationManager.create_with_mullvad(
                impersonation_manager=imp_manager,
                extractor_args_config=ext_config,
                mullvad_config=mullvad_config,
            )

            # MullvadVPN should be instantiated with the config
            MockVPN.assert_called_once_with(mullvad_config)
            # set_mullvad_vpn should have been called
            assert manager._mullvad_vpn is mock_vpn_instance

    def test_create_with_mullvad_disabled(self, imp_manager, ext_config):
        """Factory method does NOT create MullvadVPN when disabled."""
        mullvad_config = FakeMullvadConfig(enabled=False)

        with patch('src.downloader.mullvad_vpn.MullvadVPN') as MockVPN:
            manager = EscalationManager.create_with_mullvad(
                impersonation_manager=imp_manager,
                extractor_args_config=ext_config,
                mullvad_config=mullvad_config,
            )

            # MullvadVPN should NOT be instantiated
            MockVPN.assert_not_called()
            assert manager._mullvad_vpn is None

    def test_create_with_mullvad_none_config(self, imp_manager, ext_config):
        """Factory method handles None mullvad_config gracefully."""
        with patch('src.downloader.mullvad_vpn.MullvadVPN') as MockVPN:
            manager = EscalationManager.create_with_mullvad(
                impersonation_manager=imp_manager,
                extractor_args_config=ext_config,
                mullvad_config=None,
            )

            # MullvadVPN should NOT be instantiated
            MockVPN.assert_not_called()
            assert manager._mullvad_vpn is None

    def test_create_with_mullvad_functional(self, imp_manager, ext_config):
        """Factory-created manager is fully functional for escalation."""
        mullvad_config = FakeMullvadConfig(enabled=True)

        with patch('src.downloader.mullvad_vpn.MullvadVPN') as MockVPN:
            MockVPN.return_value = MagicMock()

            manager = EscalationManager.create_with_mullvad(
                impersonation_manager=imp_manager,
                extractor_args_config=ext_config,
                mullvad_config=mullvad_config,
            )

            # Manager should be functional
            result = manager.get_escalation_args("test_keyword")
            assert result.tier == EscalationTier.IMPERSONATE_ONLY

            # Can record failure and escalate
            manager.record_failure("test_keyword")
            manager.record_failure("test_keyword")
            result2 = manager.get_escalation_args("test_keyword")
            assert result2.tier == EscalationTier.EXTRACTOR_ARGS

    def test_create_with_mullvad_passes_all_params(self, imp_manager, ext_config):
        """Factory method passes budget and strategy to EscalationManager."""
        mullvad_config = FakeMullvadConfig(enabled=True)
        mock_budget = MagicMock()
        mock_strategy = MagicMock()
        # Make sure strategy returns expected behavior
        mock_strategy.should_escalate.return_value = False
        mock_strategy.get_extractor_args.return_value = []

        with patch('src.downloader.mullvad_vpn.MullvadVPN') as MockVPN:
            MockVPN.return_value = MagicMock()

            manager = EscalationManager.create_with_mullvad(
                impersonation_manager=imp_manager,
                extractor_args_config=ext_config,
                mullvad_config=mullvad_config,
                budget=mock_budget,
                strategy=mock_strategy,
            )

            # Verify budget and strategy were passed
            assert manager._budget is mock_budget
            assert manager._strategy is mock_strategy


# ---------------------------------------------------------------------------
# US-55-007: Circuit breaker wiring and tier floor propagation
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestCircuitBreakerWiring:
    """Test circuit breaker integration path and tier floor propagation.

    Verifies that set_circuit_breaker() and set_tier_floor() correctly
    influence get_escalation_args() behavior.
    """

    def test_circuit_breaker_open_shortcuts_to_full_bypass(self, imp_manager, ext_config):
        """After set_circuit_breaker(), open CB causes get_escalation_args to return max-tier args."""
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        manager = EscalationManager(imp_manager, ext_config)
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=True))

        # Wire circuit breaker
        manager.set_circuit_breaker(cb)

        # Trip the circuit breaker (make it open)
        for _ in range(cb.config.consecutive_failures_threshold):
            cb.record_failure()
        assert cb.is_open, "Circuit breaker should be open after threshold failures"

        # get_escalation_args should shortcut to FULL_BYPASS for a keyword at Tier 1
        result = manager.get_escalation_args("test_keyword")
        assert result.tier == EscalationTier.FULL_BYPASS
        assert result.rotate_cookies is True
        # Should also have extractor args (Tier 2+)
        assert any("--extractor-args" in arg for arg in result.args)

    def test_tier_floor_new_keywords_start_at_floor(self, imp_manager, ext_config):
        """set_tier_floor() causes new keywords to start at floor tier instead of Tier 1."""
        manager = EscalationManager(imp_manager, ext_config)

        # Set tier floor to EXTRACTOR_ARGS (Tier 2)
        manager.set_tier_floor(EscalationTier.EXTRACTOR_ARGS)

        # A brand new keyword should start at Tier 2
        result = manager.get_escalation_args("new_keyword_1")
        assert result.tier == EscalationTier.EXTRACTOR_ARGS
        # Should have extractor args since it's Tier 2
        assert any("--extractor-args" in arg for arg in result.args)

        # Another new keyword also starts at floor
        result2 = manager.get_escalation_args("new_keyword_2")
        assert result2.tier == EscalationTier.EXTRACTOR_ARGS

    def test_existing_keywords_below_floor_elevated(self, imp_manager, ext_config):
        """Existing keywords below tier floor are elevated to floor on next get_escalation_args."""
        manager = EscalationManager(imp_manager, ext_config)

        # Create a keyword at Tier 1 (default)
        result1 = manager.get_escalation_args("existing_keyword")
        assert result1.tier == EscalationTier.IMPERSONATE_ONLY

        # Now set tier floor to EXTRACTOR_ARGS (Tier 2)
        manager.set_tier_floor(EscalationTier.EXTRACTOR_ARGS)

        # Existing keyword should now be elevated to floor tier
        result2 = manager.get_escalation_args("existing_keyword")
        assert result2.tier == EscalationTier.EXTRACTOR_ARGS
        assert any("--extractor-args" in arg for arg in result2.args)

    def test_no_circuit_breaker_no_shortcut(self, imp_manager, ext_config):
        """When circuit_breaker is None (default), get_escalation_args returns normal tier-based args."""
        manager = EscalationManager(imp_manager, ext_config)

        # No circuit breaker set - _circuit_breaker is None
        assert manager._circuit_breaker is None

        # Should get normal Tier 1 args
        result = manager.get_escalation_args("keyword_a")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        assert result.rotate_cookies is False

        # Escalate to Tier 2 via failures
        manager.record_failure("keyword_a")
        manager.record_failure("keyword_a")
        result2 = manager.get_escalation_args("keyword_a")
        assert result2.tier == EscalationTier.EXTRACTOR_ARGS
        # No shortcut happened - normal progression
        assert result2.rotate_cookies is False

    def test_circuit_breaker_closed_no_shortcut(self, imp_manager, ext_config):
        """When circuit breaker is wired but closed, no shortcut occurs."""
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        manager = EscalationManager(imp_manager, ext_config)
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=True))

        manager.set_circuit_breaker(cb)
        assert not cb.is_open, "Circuit breaker should be closed by default"

        # Should get normal Tier 1 args (no shortcut)
        result = manager.get_escalation_args("keyword_b")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        assert result.rotate_cookies is False
