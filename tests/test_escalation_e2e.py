"""End-to-end tests for full Tier 1→4 escalation chain.

These tests verify the complete escalation flow through all 4 tiers:
- Tier 1: Impersonate only
- Tier 2: + Extractor args
- Tier 3: + Cookie rotation
- Tier 4: + VPN rotation

The tests simulate 403 errors that force escalation through each tier.
"""

import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.escalation_manager import EscalationManager, EscalationResult
from src.downloader.types import EscalationTier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeExtractorArgsConfig:
    """Minimal stand-in for ExtractorArgsConfig with all tiers enabled."""
    enabled: bool = True
    player_clients: List[str] = field(
        default_factory=lambda: ["web_safari", "tv_downgraded", "web"]
    )
    escalation_threshold: int = 2  # Low threshold for fast escalation
    cooldown_seconds: float = 0.0  # No cooldown for tests
    max_tier: int = 4  # Allow all 4 tiers


def _make_impersonation_manager():
    """Create a mock ImpersonationManager that returns deterministic args."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", "Chrome-136:Macos-15"
    ]
    return mgr


@pytest.fixture
def ext_config():
    """Config that allows all 4 tiers."""
    return FakeExtractorArgsConfig()


@pytest.fixture
def imp_manager():
    return _make_impersonation_manager()


@pytest.fixture
def full_escalation_manager(imp_manager, ext_config):
    """EscalationManager configured for full Tier 1→4 escalation."""
    mgr = EscalationManager(imp_manager, ext_config)
    # Add mock Mullvad VPN
    mock_mullvad = MagicMock()
    mock_mullvad.can_switch.return_value = True
    mgr.set_mullvad_vpn(mock_mullvad)
    return mgr


# ---------------------------------------------------------------------------
# E2E Escalation Chain Tests
# ---------------------------------------------------------------------------

class TestFullEscalationChain:
    """Tests for complete Tier 1→4 escalation sequence."""

    def test_starts_at_tier1_with_impersonate_args(self, full_escalation_manager):
        """Verify Tier 1 (IMPERSONATE_ONLY) args are present in first attempts."""
        keyword = "e2e_test"

        # No failures yet - should be at Tier 1
        result = full_escalation_manager.get_escalation_args(keyword)

        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        assert "--impersonate" in result.args
        assert result.rotate_cookies is False
        assert result.rotate_vpn is False

    def test_escalates_to_tier2_with_extractor_args(self, full_escalation_manager):
        """Verify Tier 2 (EXTRACTOR_ARGS) added after threshold failures."""
        keyword = "e2e_test"

        # Record threshold failures (2) to trigger Tier 1→2 escalation
        full_escalation_manager.record_failure(keyword, "HTTP Error 403: Forbidden")
        full_escalation_manager.record_failure(keyword, "HTTP Error 403: Forbidden")

        result = full_escalation_manager.get_escalation_args(keyword)

        assert result.tier == EscalationTier.EXTRACTOR_ARGS
        # Still has impersonate args
        assert "--impersonate" in result.args
        # Now has extractor-args
        assert "--extractor-args" in result.args
        assert result.rotate_cookies is False
        assert result.rotate_vpn is False

    def test_escalates_to_tier3_triggers_cookie_rotation(self, full_escalation_manager):
        """Verify Tier 3 (FULL_BYPASS) triggers cookie rotation after Tier 2 failures."""
        keyword = "e2e_test"

        # Escalate to Tier 2 (2 failures)
        full_escalation_manager.record_failure(keyword, "HTTP Error 403: Forbidden")
        full_escalation_manager.record_failure(keyword, "HTTP Error 403: Forbidden")

        # Escalate to Tier 3 (2 more failures)
        full_escalation_manager.record_failure(keyword, "HTTP Error 403: Forbidden")
        full_escalation_manager.record_failure(keyword, "HTTP Error 403: Forbidden")

        result = full_escalation_manager.get_escalation_args(keyword)

        assert result.tier == EscalationTier.FULL_BYPASS
        assert "--impersonate" in result.args
        assert "--extractor-args" in result.args
        assert result.rotate_cookies is True  # Cookie rotation triggered
        assert result.rotate_vpn is False

    def test_escalates_to_tier4_triggers_vpn_rotation(self, full_escalation_manager):
        """Verify Tier 4 (VPN_ROTATION) triggers VPN rotation after cookie exhaustion."""
        keyword = "e2e_test"

        # Escalate through Tiers 1→2→3→4 (6 failures total with threshold=2)
        for _ in range(6):
            full_escalation_manager.record_failure(keyword, "HTTP Error 403: Forbidden")

        result = full_escalation_manager.get_escalation_args(keyword)

        assert result.tier == EscalationTier.VPN_ROTATION
        assert "--impersonate" in result.args
        assert "--extractor-args" in result.args
        # At Tier 4, both cookie and VPN rotation are available
        assert result.rotate_vpn is True  # VPN rotation triggered

    def test_full_escalation_chain_sequence(self, imp_manager, ext_config):
        """Simulate complete Tier 1→4 escalation with 403 errors in sequence."""
        # Create fresh manager for clean sequence
        mgr = EscalationManager(imp_manager, ext_config)
        mock_mullvad = MagicMock()
        mock_mullvad.can_switch.return_value = True
        mgr.set_mullvad_vpn(mock_mullvad)

        keyword = "sequence_test"
        tier_sequence = []

        # Start at Tier 1
        result = mgr.get_escalation_args(keyword)
        tier_sequence.append(result.tier)
        assert result.tier == EscalationTier.IMPERSONATE_ONLY

        # First failure - still Tier 1
        mgr.record_failure(keyword, "HTTP Error 403: Forbidden")
        result = mgr.get_escalation_args(keyword)
        # May still be Tier 1 (below threshold)

        # Second failure - escalate to Tier 2
        mgr.record_failure(keyword, "HTTP Error 403: Forbidden")
        result = mgr.get_escalation_args(keyword)
        tier_sequence.append(result.tier)
        assert result.tier == EscalationTier.EXTRACTOR_ARGS

        # Third failure - still Tier 2
        mgr.record_failure(keyword, "HTTP Error 403: Forbidden")

        # Fourth failure - escalate to Tier 3
        mgr.record_failure(keyword, "HTTP Error 403: Forbidden")
        result = mgr.get_escalation_args(keyword)
        tier_sequence.append(result.tier)
        assert result.tier == EscalationTier.FULL_BYPASS
        assert result.rotate_cookies is True

        # Fifth failure - still Tier 3
        mgr.record_failure(keyword, "HTTP Error 403: Forbidden")

        # Sixth failure - escalate to Tier 4
        mgr.record_failure(keyword, "HTTP Error 403: Forbidden")
        result = mgr.get_escalation_args(keyword)
        tier_sequence.append(result.tier)
        assert result.tier == EscalationTier.VPN_ROTATION
        assert result.rotate_vpn is True

        # Verify we went through all tiers
        assert EscalationTier.IMPERSONATE_ONLY in tier_sequence
        assert EscalationTier.EXTRACTOR_ARGS in tier_sequence
        assert EscalationTier.FULL_BYPASS in tier_sequence
        assert EscalationTier.VPN_ROTATION in tier_sequence


class TestEscalationAtMaxTier:
    """Tests for behavior at maximum tier (Tier 4)."""

    def test_stays_at_tier4_after_max(self, full_escalation_manager):
        """Verify escalation stops at Tier 4 (max tier)."""
        keyword = "max_tier_test"

        # Escalate to Tier 4
        for _ in range(6):
            full_escalation_manager.record_failure(keyword, "HTTP Error 403: Forbidden")

        result = full_escalation_manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.VPN_ROTATION

        # More failures - should stay at Tier 4
        for _ in range(10):
            full_escalation_manager.record_failure(keyword, "HTTP Error 403: Forbidden")

        result = full_escalation_manager.get_escalation_args(keyword)
        assert result.tier == EscalationTier.VPN_ROTATION


class TestMultipleKeywordsEscalation:
    """Tests for per-keyword escalation isolation."""

    def test_keywords_escalate_independently(self, full_escalation_manager):
        """Verify each keyword has independent escalation state."""
        keyword1 = "keyword_alpha"
        keyword2 = "keyword_beta"

        # Escalate keyword1 to Tier 4
        for _ in range(6):
            full_escalation_manager.record_failure(keyword1, "HTTP Error 403: Forbidden")

        # keyword2 should still be at Tier 1
        result1 = full_escalation_manager.get_escalation_args(keyword1)
        result2 = full_escalation_manager.get_escalation_args(keyword2)

        assert result1.tier == EscalationTier.VPN_ROTATION
        assert result2.tier == EscalationTier.IMPERSONATE_ONLY

    def test_multiple_keywords_at_different_tiers(self, full_escalation_manager):
        """Verify multiple keywords can be at different tiers simultaneously."""
        keywords = ["kw1", "kw2", "kw3", "kw4"]

        # Escalate each keyword to a different tier
        # kw1: 0 failures -> Tier 1
        # kw2: 2 failures -> Tier 2
        for _ in range(2):
            full_escalation_manager.record_failure("kw2", "HTTP Error 403")
        # kw3: 4 failures -> Tier 3
        for _ in range(4):
            full_escalation_manager.record_failure("kw3", "HTTP Error 403")
        # kw4: 6 failures -> Tier 4
        for _ in range(6):
            full_escalation_manager.record_failure("kw4", "HTTP Error 403")

        # Verify each keyword is at expected tier
        assert full_escalation_manager.get_escalation_args("kw1").tier == EscalationTier.IMPERSONATE_ONLY
        assert full_escalation_manager.get_escalation_args("kw2").tier == EscalationTier.EXTRACTOR_ARGS
        assert full_escalation_manager.get_escalation_args("kw3").tier == EscalationTier.FULL_BYPASS
        assert full_escalation_manager.get_escalation_args("kw4").tier == EscalationTier.VPN_ROTATION
