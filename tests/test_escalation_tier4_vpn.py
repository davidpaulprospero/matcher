"""Integration tests for Tier 4 VPN rotation escalation path.

Tests the full escalation path from Tier 1 through Tier 4,
verifying VPN rotation triggers, budget resets, max rotation limits,
and cooldown enforcement — all with mocked Mullvad CLI calls.

Story: US-67-011
"""

import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch, call

import pytest

from src.downloader.escalation_manager import EscalationManager, EscalationResult
from src.downloader.types import EscalationTier
from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.downloader.rate_limit_budget import RateLimitBudget
from src.downloader.mullvad_vpn import MullvadVPN


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeExtractorArgsConfig:
    """Minimal stand-in for ExtractorArgsConfig with Tier 4 support."""
    enabled: bool = True
    player_clients: List[str] = field(
        default_factory=lambda: ["web_safari", "tv_downgraded", "web"]
    )
    escalation_threshold: int = 2
    cooldown_seconds: float = 0.0  # No cooldown for fast tests
    max_tier: int = 4  # Allow Tier 4 (VPN_ROTATION)


def _make_impersonation_manager():
    """Create a mock ImpersonationManager that returns deterministic args."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", "Chrome-136:Macos-15"
    ]
    return mgr


@dataclass
class FakeMullvadConfig:
    """Mock config combining VPNConfig base fields + MullvadConfig fields.

    MullvadConfig doesn't inherit from VPNConfig, but MullvadVPN.__init__
    calls VPNManager.__init__ which accesses switch_command, etc.
    This mock provides all fields needed by both classes.
    """
    # VPNConfig base fields (needed by VPNManager.__init__)
    enabled: bool = True
    switch_command: str = "mullvad connect"
    disconnect_command: str = "mullvad disconnect"
    rotate_on_rate_limit: bool = True
    switch_delay_seconds: int = 0
    max_switches_per_session: int = 10
    verify_connection: bool = False
    verify_timeout: int = 10
    skip_verification: bool = True
    verification_endpoint: str = "https://www.google.com"
    verification_ip: str = "8.8.8.8"
    # MullvadConfig fields
    preferred_countries: List[str] = field(
        default_factory=lambda: ['us', 'gb', 'de']
    )
    rotation_strategy: str = 'random'
    max_rotations_per_session: int = 5
    verification_timeout_mullvad: int = 10
    initial_rotation_delay_seconds: float = 0.0
    max_rotation_delay_seconds: float = 0.0
    rotation_delay_seconds: int = 0


def _make_mullvad_config(**overrides) -> FakeMullvadConfig:
    """Create a FakeMullvadConfig with test-friendly defaults."""
    return FakeMullvadConfig(**overrides)


def _escalate_to_tier(manager: EscalationManager, keyword: str, target_tier: int):
    """Helper: escalate a keyword to the given tier number (2, 3, or 4).

    With threshold=2, each tier takes 2 failures:
    - 2 failures -> Tier 2
    - 4 failures -> Tier 3
    - 6 failures -> Tier 4
    """
    failures_needed = (target_tier - 1) * 2
    for _ in range(failures_needed):
        manager.record_failure(keyword, "HTTP Error 403: Forbidden")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ext_config():
    return FakeExtractorArgsConfig()


@pytest.fixture
def imp_manager():
    return _make_impersonation_manager()


@pytest.fixture
def budget():
    return RateLimitBudget(
        max_rotations=10,
        max_backoff_time=300.0,
        max_vpn_switches=5,
    )


# ---------------------------------------------------------------------------
# Test 1: Full escalation path T1 -> T2 -> T3 -> T4 with VPN rotation trigger
# ---------------------------------------------------------------------------

class TestFullEscalationPathToTier4:
    """Verify the full T1->T2->T3->T4 escalation with VPN rotation trigger."""

    def test_full_escalation_path_reaches_tier4(self, imp_manager, ext_config):
        """Each tier pair of 403s escalates: T1->T2->T3->T4 with rotate_vpn=True."""
        mock_mullvad = MagicMock()
        mock_mullvad.can_switch.return_value = True

        manager = EscalationManager(
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
        )
        manager.set_mullvad_vpn(mock_mullvad)

        keyword = "path_test"

        # Tier 1 (initial)
        result = manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        assert result.rotate_vpn is False
        assert result.rotate_cookies is False

        # Tier 2 after 2 failures
        for _ in range(2):
            manager.record_failure(keyword, "HTTP Error 403: Forbidden")
        result = manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.EXTRACTOR_ARGS
        assert result.rotate_vpn is False
        assert result.rotate_cookies is False

        # Tier 3 after 2 more failures
        for _ in range(2):
            manager.record_failure(keyword, "HTTP Error 403: Forbidden")
        result = manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.FULL_BYPASS
        assert result.rotate_vpn is False
        assert result.rotate_cookies is True

        # Tier 4 after 2 more failures
        for _ in range(2):
            manager.record_failure(keyword, "HTTP Error 403: Forbidden")
        result = manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.VPN_ROTATION
        assert result.rotate_vpn is True
        assert result.rotate_cookies is True

    def test_tier4_without_mullvad_does_not_set_rotate_vpn(
        self, imp_manager, ext_config
    ):
        """When no MullvadVPN is linked, Tier 4 should NOT set rotate_vpn=True."""
        manager = EscalationManager(
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
        )
        # Do NOT set_mullvad_vpn

        keyword = "no_vpn_test"
        _escalate_to_tier(manager, keyword, 4)

        result = manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.VPN_ROTATION
        assert result.rotate_vpn is False  # No Mullvad linked


# ---------------------------------------------------------------------------
# Test 2: Budget reset after VPN rotation
# ---------------------------------------------------------------------------

class TestBudgetResetAfterVPNRotation:
    """Verify cookie budget and backoff budget refresh after VPN rotation."""

    def test_budget_resets_on_ip_change(self, budget):
        """reset_on_ip_change() clears cookie rotations + backoff, preserves VPN count."""
        # Simulate heavy usage
        budget.rotations_used = 8
        budget.backoff_time_spent = 250.0
        budget.last_escalation_level = "cookie"
        budget.record_vpn_rotation()  # 1 VPN switch

        # Simulate VPN rotation completing -> reset
        budget.reset_on_ip_change()

        # Cookie and backoff budgets reset
        assert budget.rotations_used == 0
        assert budget.backoff_time_spent == 0.0
        assert budget.last_escalation_level == "none"

        # VPN count preserved (not reset)
        assert budget.vpn_switches_used == 1

    def test_end_to_end_escalation_with_budget_reset(
        self, imp_manager, ext_config, budget
    ):
        """Full integration: escalate to T4, VPN rotation callback resets budget."""
        reset_called = []

        def vpn_rotation_callback(keyword: str):
            """Simulate the pipeline's VPN rotation handler."""
            budget.record_vpn_rotation(keyword)
            budget.reset_on_ip_change()
            reset_called.append(keyword)

        manager = EscalationManager(
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
            budget=budget,
            on_vpn_rotation_needed=vpn_rotation_callback,
        )
        mock_mullvad = MagicMock()
        mock_mullvad.can_switch.return_value = True
        manager.set_mullvad_vpn(mock_mullvad)

        # Simulate cookie budget usage before escalation
        budget.rotations_used = 5
        budget.backoff_time_spent = 100.0

        # Escalate to Tier 4
        keyword = "budget_reset_test"
        _escalate_to_tier(manager, keyword, 4)

        # Callback should have been called
        assert len(reset_called) == 1
        assert reset_called[0] == keyword

        # Budget should be reset (from callback)
        assert budget.rotations_used == 0
        assert budget.backoff_time_spent == 0.0

        # VPN switch should be recorded
        assert budget.vpn_switches_used == 1


# ---------------------------------------------------------------------------
# Test 3: Max rotations per session enforcement
# ---------------------------------------------------------------------------

class TestMaxRotationsPerSession:
    """Verify max_rotations_per_session limit is enforced on MullvadVPN."""

    @patch('src.downloader.mullvad_vpn.subprocess.run')
    def test_max_rotations_enforced(self, mock_run):
        """rotate_server() returns False once max_rotations_per_session reached."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = _make_mullvad_config(max_rotations_per_session=2)
        vpn = MullvadVPN(config)

        # First rotation succeeds
        assert vpn.rotate_server() is True
        assert vpn.switch_count == 1

        # Second rotation succeeds
        assert vpn.rotate_server() is True
        assert vpn.switch_count == 2

        # Third rotation blocked (limit reached)
        assert vpn.rotate_server() is False
        assert vpn.switch_count == 2  # No increment

    @patch('src.downloader.mullvad_vpn.subprocess.run')
    def test_max_rotations_with_escalation_integration(self, mock_run, imp_manager, ext_config):
        """After max rotations exhausted, escalation still shows Tier 4 but VPN can't rotate."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = _make_mullvad_config(max_rotations_per_session=1)
        vpn = MullvadVPN(config)

        manager = EscalationManager(
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
        )
        manager.set_mullvad_vpn(vpn)

        # Escalate to Tier 4
        keyword = "max_rot_test"
        _escalate_to_tier(manager, keyword, 4)

        result = manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.VPN_ROTATION
        assert result.rotate_vpn is True  # Signal is still set

        # First VPN rotation succeeds
        assert vpn.rotate_server() is True

        # Second VPN rotation blocked by limit
        assert vpn.rotate_server() is False


# ---------------------------------------------------------------------------
# Test 4: VPN rotation cooldown prevents rapid re-rotation
# ---------------------------------------------------------------------------

class TestVPNRotationCooldown:
    """Verify rotation_delay_seconds cooldown prevents rapid re-rotation."""

    @patch('src.downloader.mullvad_vpn.subprocess.run')
    @patch('src.downloader.mullvad_vpn.time.sleep')
    def test_cooldown_prevents_rapid_rotation(self, mock_sleep, mock_run):
        """Second rotation within cooldown period returns False."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = _make_mullvad_config(
            rotation_delay_seconds=60,  # 60s cooldown
            max_rotations_per_session=10,
        )
        vpn = MullvadVPN(config)

        # First rotation succeeds
        assert vpn.rotate_server() is True

        # Immediate second rotation blocked by cooldown
        assert vpn.rotate_server() is False
        assert vpn.switch_count == 1  # Only first counted

    @patch('src.downloader.mullvad_vpn.subprocess.run')
    @patch('src.downloader.mullvad_vpn.time.sleep')
    @patch('src.downloader.mullvad_vpn.time.time')
    def test_rotation_succeeds_after_cooldown_expires(
        self, mock_time, mock_sleep, mock_run
    ):
        """Rotation succeeds once cooldown period has elapsed."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_sleep.return_value = None

        # Simulate time progression: t=0, then t=70 (past 60s cooldown)
        time_values = [
            0.0,    # __init__ check
            0.0,    # rotate_server check
            0.0,    # _last_rotation_time set (line 293)
            0.0,    # _last_switch_time set (line 292)
            70.0,   # 2nd rotate_server elapsed check
            70.0,   # can_switch
            70.0,   # _last_rotation_time set
            70.0,   # _last_switch_time set
        ]
        mock_time.side_effect = time_values

        config = _make_mullvad_config(
            rotation_delay_seconds=60,
            max_rotations_per_session=10,
        )
        vpn = MullvadVPN(config)

        # First rotation at t=0
        assert vpn.rotate_server() is True

        # Second rotation at t=70 (past 60s cooldown)
        assert vpn.rotate_server() is True
        assert vpn.switch_count == 2


# ---------------------------------------------------------------------------
# Test 5: Mocked Mullvad CLI subprocess calls
# ---------------------------------------------------------------------------

class TestMockedMullvadCLI:
    """Verify all Mullvad CLI subprocess calls are mocked (no real VPN)."""

    @patch('src.downloader.mullvad_vpn.subprocess.run')
    def test_rotate_server_calls_mullvad_relay_and_reconnect(self, mock_run):
        """rotate_server() calls 'mullvad relay set location' then 'mullvad reconnect'."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = _make_mullvad_config()
        vpn = MullvadVPN(config)

        vpn.rotate_server(country="de")

        # Should have called mullvad relay set location AND mullvad reconnect
        calls = mock_run.call_args_list
        assert len(calls) >= 2

        # First call: mullvad relay set location de
        relay_call = calls[0]
        assert relay_call[0][0] == ["mullvad", "relay", "set", "location", "de"]

        # Second call: mullvad reconnect
        reconnect_call = calls[1]
        assert reconnect_call[0][0] == ["mullvad", "reconnect"]

    @patch('src.downloader.mullvad_vpn.subprocess.run')
    def test_relay_failure_returns_false(self, mock_run):
        """rotate_server() returns False when 'mullvad relay set location' fails."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="relay error")

        config = _make_mullvad_config()
        vpn = MullvadVPN(config)

        assert vpn.rotate_server(country="us") is False
        assert vpn.switch_count == 0

    @patch('src.downloader.mullvad_vpn.subprocess.run')
    def test_circuit_breaker_reset_on_successful_rotation(self, mock_run):
        """Circuit breaker is reset when VPN rotation succeeds."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = _make_mullvad_config()
        vpn = MullvadVPN(config)

        # Create and trip circuit breaker
        cb_config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3,
            pause_seconds=60,
        )
        cb = CircuitBreaker(cb_config)
        for _ in range(5):
            cb.record_failure()
        assert cb.is_open

        # Rotate with circuit breaker
        assert vpn.rotate_server(circuit_breaker=cb) is True

        # CB should be reset
        assert not cb.is_open
        assert cb.state.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Test 6: Exponential backoff on rotation
# ---------------------------------------------------------------------------

class TestRotationExponentialBackoff:
    """Verify exponential backoff delay on successive VPN rotations."""

    def test_backoff_delay_computation(self):
        """_compute_backoff_delay() follows base * 2^n capped at max."""
        config = _make_mullvad_config(
            initial_rotation_delay_seconds=5.0,
            max_rotation_delay_seconds=60.0,
        )
        vpn = MullvadVPN(config)

        # backoff_count=0: 5 * 2^0 = 5
        assert vpn._compute_backoff_delay() == 5.0

        vpn._backoff_count = 1
        # 5 * 2^1 = 10
        assert vpn._compute_backoff_delay() == 10.0

        vpn._backoff_count = 2
        # 5 * 2^2 = 20
        assert vpn._compute_backoff_delay() == 20.0

        vpn._backoff_count = 3
        # 5 * 2^3 = 40
        assert vpn._compute_backoff_delay() == 40.0

        vpn._backoff_count = 4
        # 5 * 2^4 = 80 -> capped at 60
        assert vpn._compute_backoff_delay() == 60.0

    def test_reset_backoff_clears_counter(self):
        """reset_backoff() resets counter to 0."""
        config = _make_mullvad_config(
            initial_rotation_delay_seconds=5.0,
            max_rotation_delay_seconds=60.0,
        )
        vpn = MullvadVPN(config)

        vpn._backoff_count = 3
        assert vpn._compute_backoff_delay() == 40.0

        vpn.reset_backoff()
        assert vpn._backoff_count == 0
        assert vpn._compute_backoff_delay() == 5.0


# ---------------------------------------------------------------------------
# Test 7: Multiple keywords share escalation state independently
# ---------------------------------------------------------------------------

class TestMultipleKeywordEscalation:
    """Verify different keywords escalate independently to Tier 4."""

    def test_keywords_escalate_independently(self, imp_manager, ext_config):
        """Each keyword has its own escalation state."""
        mock_mullvad = MagicMock()
        mock_mullvad.can_switch.return_value = True

        callback_keywords = []

        def track_callback(kw):
            callback_keywords.append(kw)

        manager = EscalationManager(
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
            on_vpn_rotation_needed=track_callback,
        )
        manager.set_mullvad_vpn(mock_mullvad)

        # Escalate keyword_a to Tier 4
        _escalate_to_tier(manager, "keyword_a", 4)
        result_a = manager.get_escalation_args("keyword_a")
        assert result_a.tier == EscalationTier.VPN_ROTATION

        # keyword_b should still be at Tier 1
        result_b = manager.get_escalation_args("keyword_b")
        assert result_b.tier == EscalationTier.IMPERSONATE_ONLY

        # Escalate keyword_b to Tier 3 only
        _escalate_to_tier(manager, "keyword_b", 3)
        result_b = manager.get_escalation_args("keyword_b")
        assert result_b.tier == EscalationTier.FULL_BYPASS
        assert result_b.rotate_vpn is False

        # Callback should only have fired for keyword_a
        assert callback_keywords == ["keyword_a"]


# ---------------------------------------------------------------------------
# Test 8: VPN rotation with budget tracking end-to-end
# ---------------------------------------------------------------------------

class TestVPNBudgetTracking:
    """End-to-end: budget tracks VPN rotations and enforces limits."""

    def test_budget_exhaustion_blocks_further_vpn_rotation(self):
        """Once max_vpn_switches is reached, can_rotate_vpn() returns False."""
        budget = RateLimitBudget(max_vpn_switches=2)

        assert budget.can_rotate_vpn() is True

        budget.record_vpn_rotation("kw1")
        assert budget.can_rotate_vpn() is True
        assert budget.vpn_switches_remaining() == 1

        budget.record_vpn_rotation("kw2")
        assert budget.can_rotate_vpn() is False
        assert budget.vpn_switches_remaining() == 0

    def test_vpn_switches_preserved_across_ip_resets(self):
        """reset_on_ip_change() preserves vpn_switches_used count."""
        budget = RateLimitBudget(max_vpn_switches=5)

        budget.record_vpn_rotation("kw1")
        budget.record_vpn_rotation("kw2")
        budget.rotations_used = 10
        budget.backoff_time_spent = 200.0

        budget.reset_on_ip_change()

        # VPN count preserved
        assert budget.vpn_switches_used == 2
        # Cookie/backoff reset
        assert budget.rotations_used == 0
        assert budget.backoff_time_spent == 0.0

    @patch('src.downloader.mullvad_vpn.subprocess.run')
    def test_full_vpn_rotation_flow_with_budget(
        self, mock_run, imp_manager, ext_config
    ):
        """Full flow: escalate -> VPN rotate -> budget tracks -> budget resets."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        budget = RateLimitBudget(
            max_rotations=5,
            max_vpn_switches=3,
            max_backoff_time=300.0,
        )

        config = _make_mullvad_config(max_rotations_per_session=3)
        vpn = MullvadVPN(config)

        rotation_events = []

        def vpn_callback(keyword):
            success = vpn.rotate_server()
            if success:
                budget.record_vpn_rotation(keyword)
                budget.reset_on_ip_change()
                rotation_events.append(keyword)

        manager = EscalationManager(
            impersonation_manager=imp_manager,
            extractor_args_config=ext_config,
            budget=budget,
            on_vpn_rotation_needed=vpn_callback,
        )
        manager.set_mullvad_vpn(vpn)

        # Simulate cookie budget usage
        budget.rotations_used = 5
        budget.backoff_time_spent = 200.0

        # Escalate to Tier 4 -> triggers callback -> VPN rotation
        _escalate_to_tier(manager, "vid_123", 4)

        assert len(rotation_events) == 1
        assert rotation_events[0] == "vid_123"

        # Budget should be reset
        assert budget.rotations_used == 0
        assert budget.backoff_time_spent == 0.0

        # VPN rotation tracked
        assert budget.vpn_switches_used == 1
        assert vpn.switch_count == 1
