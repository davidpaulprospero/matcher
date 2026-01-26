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

    # -- New: 429 / rate-limit patterns --
    @pytest.mark.parametrize("stderr", [
        "ERROR: HTTP Error 429: Too Many Requests",
        "HTTP Error 429",
        "Too Many Requests",
        "rate limit exceeded for this IP",
        "rate-limit reached",
    ])
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
    def test_matches_age_gate_patterns(self, stderr):
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


# ---------------------------------------------------------------------------
# classify_trigger() tests
# ---------------------------------------------------------------------------

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
    def test_classify_known_patterns(self, stderr, expected):
        assert classify_trigger(stderr) == expected

    def test_classify_returns_none_for_non_triggers(self):
        assert classify_trigger("") is None
        assert classify_trigger("ERROR: HTTP Error 404: Not Found") is None
        assert classify_trigger("Download complete") is None
        assert classify_trigger("Video unavailable") is None

    def test_classify_case_insensitive(self):
        assert classify_trigger("HTTP ERROR 429") == "429"
        assert classify_trigger("AGE-GATED VIDEO") == "age_gate"
        assert classify_trigger("ACCESS DENIED") == "ip_blocked"

    def test_classify_priority_429_over_403(self):
        """429 patterns match '429' category, not '403'."""
        assert classify_trigger("HTTP Error 429") == "429"

    def test_classify_priority_age_gate_over_403(self):
        """'Sign in to confirm your age' matches 'age_gate', not '403'."""
        # This matches age_gate because age_gate is checked before 403
        assert classify_trigger("Sign in to confirm your age") == "age_gate"

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


class TestBudgetAwareEscalation:
    """Tests for US-002: Wire RateLimitBudget into EscalationManager."""

    def test_constructor_stores_budget(self, budget_manager, budget):
        """EscalationManager stores the budget reference."""
        assert budget_manager._budget is budget

    def test_constructor_budget_none_is_safe(self, imp_manager, ext_config):
        """EscalationManager works fine without a budget (None)."""
        mgr = EscalationManager(imp_manager, ext_config, budget=None)
        assert mgr._budget is None
        # Normal operations still work
        result = mgr.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        mgr.record_failure("kw", "HTTP Error 403")
        mgr.record_success("kw")

    def test_get_escalation_args_calls_record_attempt(self, imp_manager, ext_config):
        """get_escalation_args() calls budget.record_attempt()."""
        mock_budget = MagicMock()
        mock_budget.can_rotate.return_value = True
        mgr = EscalationManager(imp_manager, ext_config, budget=mock_budget)
        mgr.get_escalation_args("sunset")
        mock_budget.record_attempt.assert_called_once_with("sunset")

    def test_record_failure_records_rotation_on_tier_advance(self, budget_manager, budget):
        """record_failure() calls budget.record_rotation() when tier advances."""
        assert budget.rotations_used == 0
        # Trigger escalation: 2 consecutive failures (threshold=2)
        budget_manager.record_failure("kw", "HTTP Error 403")
        budget_manager.record_failure("kw", "HTTP Error 403")
        # Should have recorded a rotation for the Tier 1 -> Tier 2 advance
        assert budget.rotations_used == 1

    def test_record_failure_records_rotation_on_each_advance(self, budget_manager, budget):
        """Each tier advance records a rotation in the budget."""
        # Escalate Tier 1 -> Tier 2 (2 failures)
        budget_manager.record_failure("kw", "HTTP Error 403")
        budget_manager.record_failure("kw", "HTTP Error 403")
        assert budget.rotations_used == 1

        # Advance past cooldown so second escalation can proceed
        with patch("src.downloader.escalation_manager.time") as mock_time:
            mock_time.time.return_value = time.time() + 400  # Past 300s cooldown
            # Escalate Tier 2 -> Tier 3 (2 more failures)
            budget_manager.record_failure("kw", "HTTP Error 403")
            budget_manager.record_failure("kw", "HTTP Error 403")
        assert budget.rotations_used == 2

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

    def test_budget_none_normal_escalation(self, manager):
        """Without budget, escalation proceeds normally through all tiers."""
        assert manager._budget is None
        # Tier 1 -> 2
        manager.record_failure("kw", "HTTP Error 403")
        manager.record_failure("kw", "HTTP Error 403")
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.EXTRACTOR_ARGS
        # Advance past cooldown for second escalation
        with patch("src.downloader.escalation_manager.time") as mock_time:
            mock_time.time.return_value = time.time() + 400
            # Tier 2 -> 3
            manager.record_failure("kw", "HTTP Error 403")
            manager.record_failure("kw", "HTTP Error 403")
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.FULL_BYPASS

    def test_budget_tracks_rotation_with_keyword(self, budget_manager, budget):
        """Budget rotation is recorded with the correct keyword."""
        budget_manager.record_failure("ocean", "HTTP Error 403")
        budget_manager.record_failure("ocean", "HTTP Error 403")
        assert "ocean" in budget.keywords_rate_limited


# ---------------------------------------------------------------------------
# US-006: Speed tracker signal consumption for preemptive escalation
# ---------------------------------------------------------------------------

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

    def test_speed_escalation_resets_counter_after_trigger(self, manager):
        """After speed escalation, counter resets so 3 more signals needed."""
        # First escalation: Tier 1 -> 2
        for _ in range(3):
            manager.record_slow_speed("kw", 0.05)
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.EXTRACTOR_ARGS

        # 1-2 more signals: no escalation yet
        manager.record_slow_speed("kw", 0.05)
        manager.record_slow_speed("kw", 0.05)
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.EXTRACTOR_ARGS

        # 3rd signal after reset: Tier 2 -> 3
        manager.record_slow_speed("kw", 0.05)
        r = manager.get_escalation_args("kw")
        assert r.tier == EscalationTier.FULL_BYPASS

    def test_speed_escalation_stops_at_max_tier(self, manager):
        """Speed escalation does not go beyond FULL_BYPASS."""
        # Escalate to Tier 3 via two rounds of speed signals
        for _ in range(3):
            manager.record_slow_speed("kw", 0.05)
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
