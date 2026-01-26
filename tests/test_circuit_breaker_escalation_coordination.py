"""Tests for circuit breaker + escalation manager coordination (Sprint 10 US-005).

Tests cover:
  - Circuit breaker pause extension when >50% keywords at Tier 3
  - EscalationManager returns Tier 3 args during circuit breaker pause
  - Extended reset requirement (3 consecutive successes at Tier 3)
  - get_active_keyword_count() and get_keywords_at_tier() query methods
  - INFO logging for circuit breaker adjustments
  - Bidirectional link (circuit breaker <-> escalation manager)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from src.downloader.escalation_manager import EscalationManager
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
    cooldown_seconds: float = 300.0
    max_tier: int = 3


def _make_impersonation_manager():
    """Create a mock ImpersonationManager."""
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
def escalation_manager(imp_manager, ext_config):
    return EscalationManager(imp_manager, ext_config)


@pytest.fixture
def circuit_breaker():
    return CircuitBreaker(CircuitBreakerConfig(
        enabled=True,
        consecutive_failures_threshold=3,
        pause_seconds=60.0,
    ))


@pytest.fixture
def linked_pair(circuit_breaker, escalation_manager):
    """Create a linked circuit breaker + escalation manager pair."""
    circuit_breaker.set_escalation_manager(escalation_manager)
    escalation_manager.set_circuit_breaker(circuit_breaker)
    return circuit_breaker, escalation_manager


# ---------------------------------------------------------------------------
# AC 1: Pause extension when >50% keywords at Tier 3
# ---------------------------------------------------------------------------


class TestPauseExtension:
    """When >50% of active keywords are at Tier 3, circuit breaker doubles pause."""

    def test_no_extension_without_escalation_manager(self, circuit_breaker):
        """Without linked escalation manager, pause is base value."""
        effective = circuit_breaker._get_effective_pause_seconds()
        assert effective == 60.0

    def test_no_extension_when_no_keywords(self, linked_pair):
        """When no keywords tracked, pause is base value."""
        cb, _ = linked_pair
        effective = cb._get_effective_pause_seconds()
        assert effective == 60.0

    def test_no_extension_when_few_at_tier3(self, linked_pair):
        """When <=50% keywords at Tier 3, pause is base value."""
        cb, em = linked_pair

        # Set up 3 keywords at Tier 1, 1 at Tier 3 (25% < 50%)
        em.get_escalation_args("kw1")
        em.get_escalation_args("kw2")
        em.get_escalation_args("kw3")

        # Force kw4 to Tier 3 by escalating past cooldown
        base_time = time.time()
        with patch('src.downloader.types.time') as mock_types_time, \
             patch('src.downloader.escalation_manager.time') as mock_em_time:
            mock_types_time.time.return_value = base_time
            mock_em_time.time.return_value = base_time
            em.record_failure("kw4", "HTTP Error 403")
            em.record_failure("kw4", "HTTP Error 403")
            # Advance past cooldown for second escalation
            mock_types_time.time.return_value = base_time + 400
            mock_em_time.time.return_value = base_time + 400
            em.record_failure("kw4", "HTTP Error 403")
            em.record_failure("kw4", "HTTP Error 403")

        effective = cb._get_effective_pause_seconds()
        assert effective == 60.0  # Only 1/4 at Tier 3 (25%)

    def test_extension_when_majority_at_tier3(self, linked_pair):
        """When >50% keywords at Tier 3, pause is doubled."""
        cb, em = linked_pair

        # Force 3 out of 4 keywords to Tier 3 (75% > 50%)
        base_time = time.time()
        with patch('src.downloader.types.time') as mock_types_time, \
             patch('src.downloader.escalation_manager.time') as mock_em_time:
            for kw in ["kw1", "kw2", "kw3"]:
                mock_types_time.time.return_value = base_time
                mock_em_time.time.return_value = base_time
                em.record_failure(kw, "HTTP Error 403")
                em.record_failure(kw, "HTTP Error 403")
                # Advance past cooldown
                mock_types_time.time.return_value = base_time + 400
                mock_em_time.time.return_value = base_time + 400
                em.record_failure(kw, "HTTP Error 403")
                em.record_failure(kw, "HTTP Error 403")

        # kw4 stays at Tier 1
        em.get_escalation_args("kw4")

        effective = cb._get_effective_pause_seconds()
        assert effective == 120.0  # 3/4 = 75% at Tier 3, doubled

    def test_extension_logged_at_info(self, linked_pair, caplog):
        """Extension should be logged at INFO level."""
        cb, em = linked_pair

        # Force 2 out of 2 keywords to Tier 3 (100%)
        base_time = time.time()
        with patch('src.downloader.types.time') as mock_types_time, \
             patch('src.downloader.escalation_manager.time') as mock_em_time:
            for kw in ["kw1", "kw2"]:
                mock_types_time.time.return_value = base_time
                mock_em_time.time.return_value = base_time
                em.record_failure(kw, "HTTP Error 403")
                em.record_failure(kw, "HTTP Error 403")
                mock_types_time.time.return_value = base_time + 400
                mock_em_time.time.return_value = base_time + 400
                em.record_failure(kw, "HTTP Error 403")
                em.record_failure(kw, "HTTP Error 403")

        with caplog.at_level(logging.INFO, logger='src.downloader.circuit_breaker'):
            effective = cb._get_effective_pause_seconds()

        assert effective == 120.0
        assert any("Circuit breaker extended" in r.message for r in caplog.records)
        assert any("keywords at Tier 3" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# AC 2: Tier 3 shortcut during circuit breaker pause
# ---------------------------------------------------------------------------


class TestTier3Shortcut:
    """When circuit breaker is open, escalation returns Tier 3 immediately."""

    def test_tier3_shortcut_when_circuit_open(self, linked_pair):
        """get_escalation_args returns Tier 3 when circuit breaker is open."""
        cb, em = linked_pair

        # Trip the circuit breaker
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open

        # Even though keyword hasn't had any failures, get Tier 3 args
        result = em.get_escalation_args("fresh_keyword")
        assert result.tier == EscalationTier.FULL_BYPASS
        assert result.rotate_cookies is True

    def test_no_shortcut_when_circuit_closed(self, linked_pair):
        """get_escalation_args returns normal tier when circuit breaker is closed."""
        cb, em = linked_pair

        # Circuit breaker is closed (no failures)
        assert not cb.is_open

        result = em.get_escalation_args("fresh_keyword")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        assert result.rotate_cookies is False

    def test_shortcut_logged_at_info(self, linked_pair, caplog):
        """Tier 3 shortcut should be logged at INFO."""
        cb, em = linked_pair

        for _ in range(3):
            cb.record_failure()

        with caplog.at_level(logging.INFO, logger='src.downloader.escalation_manager'):
            em.get_escalation_args("kw")

        assert any("Circuit breaker open" in r.message for r in caplog.records)
        assert any("shortcutting" in r.message for r in caplog.records)

    def test_no_shortcut_without_circuit_breaker(self, escalation_manager):
        """Without linked circuit breaker, normal behavior."""
        result = escalation_manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY

    def test_shortcut_preserves_actual_keyword_tier(self, linked_pair):
        """Shortcut changes returned tier but doesn't modify stored state."""
        cb, em = linked_pair

        # Trip circuit breaker
        for _ in range(3):
            cb.record_failure()

        em.get_escalation_args("kw")

        # Close circuit breaker
        cb.reset()
        assert not cb.is_open

        # Keyword should still be at Tier 1 (shortcut didn't change state)
        result = em.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY


# ---------------------------------------------------------------------------
# AC 3: Extended reset requirement (3 consecutive successes at Tier 3)
# ---------------------------------------------------------------------------


class TestExtendedReset:
    """At Tier 3, circuit breaker needs 3 consecutive successes to close."""

    def test_single_success_closes_without_tier3(self, circuit_breaker):
        """Without escalation manager, single success resets failures."""
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        assert circuit_breaker.state.consecutive_failures == 2

        circuit_breaker.record_success()
        assert circuit_breaker.state.consecutive_failures == 0

    def test_single_success_not_enough_at_tier3(self, linked_pair):
        """With Tier 3 escalation, single success doesn't reset failures."""
        cb, em = linked_pair

        # Force keyword to Tier 3
        base_time = time.time()
        with patch('src.downloader.types.time') as mock_types_time, \
             patch('src.downloader.escalation_manager.time') as mock_em_time:
            mock_types_time.time.return_value = base_time
            mock_em_time.time.return_value = base_time
            em.record_failure("kw", "HTTP Error 403")
            em.record_failure("kw", "HTTP Error 403")
            mock_types_time.time.return_value = base_time + 400
            mock_em_time.time.return_value = base_time + 400
            em.record_failure("kw", "HTTP Error 403")
            em.record_failure("kw", "HTTP Error 403")

        # Verify keyword at Tier 3
        tier3_kws = em.get_keywords_at_tier(EscalationTier.FULL_BYPASS)
        assert "kw" in tier3_kws

        # Record failures on circuit breaker
        cb.record_failure()
        cb.record_failure()
        assert cb.state.consecutive_failures == 2

        # One success should NOT reset due to Tier 3
        cb.record_success()
        assert cb.state.consecutive_failures == 2  # Not reset yet

    def test_three_successes_close_at_tier3(self, linked_pair):
        """With Tier 3, 3 consecutive successes reset failures."""
        cb, em = linked_pair

        # Force keyword to Tier 3
        base_time = time.time()
        with patch('src.downloader.types.time') as mock_types_time, \
             patch('src.downloader.escalation_manager.time') as mock_em_time:
            mock_types_time.time.return_value = base_time
            mock_em_time.time.return_value = base_time
            em.record_failure("kw", "HTTP Error 403")
            em.record_failure("kw", "HTTP Error 403")
            mock_types_time.time.return_value = base_time + 400
            mock_em_time.time.return_value = base_time + 400
            em.record_failure("kw", "HTTP Error 403")
            em.record_failure("kw", "HTTP Error 403")

        # Add failures
        cb.record_failure()
        cb.record_failure()

        # 3 consecutive successes should reset
        cb.record_success()
        cb.record_success()
        assert cb.state.consecutive_failures == 2  # Still not reset after 2
        cb.record_success()
        assert cb.state.consecutive_failures == 0  # Reset after 3

    def test_failure_resets_success_streak(self, linked_pair):
        """A failure between successes resets the streak counter."""
        cb, em = linked_pair

        # Force keyword to Tier 3
        base_time = time.time()
        with patch('src.downloader.types.time') as mock_types_time, \
             patch('src.downloader.escalation_manager.time') as mock_em_time:
            mock_types_time.time.return_value = base_time
            mock_em_time.time.return_value = base_time
            em.record_failure("kw", "HTTP Error 403")
            em.record_failure("kw", "HTTP Error 403")
            mock_types_time.time.return_value = base_time + 400
            mock_em_time.time.return_value = base_time + 400
            em.record_failure("kw", "HTTP Error 403")
            em.record_failure("kw", "HTTP Error 403")

        cb.record_failure()
        cb.record_failure()

        # 2 successes, then a failure breaks the streak
        cb.record_success()
        cb.record_success()
        cb.record_failure()  # Streak broken
        assert cb._consecutive_successes == 0

        # Need fresh 3 successes
        cb.record_success()
        cb.record_success()
        assert cb.state.consecutive_failures == 3  # Not reset yet (only 2 successes)
        cb.record_success()
        assert cb.state.consecutive_failures == 0  # Reset after 3


# ---------------------------------------------------------------------------
# AC 4: get_active_keyword_count() and get_keywords_at_tier()
# ---------------------------------------------------------------------------


class TestEscalationQueryMethods:
    """New query methods on EscalationManager for circuit breaker consultation."""

    def test_active_keyword_count_empty(self, escalation_manager):
        """No keywords tracked initially."""
        assert escalation_manager.get_active_keyword_count() == 0

    def test_active_keyword_count_tracks_gets(self, escalation_manager):
        """get_escalation_args creates keyword state."""
        escalation_manager.get_escalation_args("kw1")
        escalation_manager.get_escalation_args("kw2")
        assert escalation_manager.get_active_keyword_count() == 2

    def test_active_keyword_count_tracks_failures(self, escalation_manager):
        """record_failure creates keyword state."""
        escalation_manager.record_failure("kw_a", "HTTP Error 403")
        assert escalation_manager.get_active_keyword_count() == 1

    def test_keywords_at_tier_initially_empty(self, escalation_manager):
        """No keywords at any tier initially."""
        result = escalation_manager.get_keywords_at_tier(EscalationTier.IMPERSONATE_ONLY)
        assert result == []

    def test_keywords_at_tier_1(self, escalation_manager):
        """New keywords start at Tier 1."""
        escalation_manager.get_escalation_args("kw1")
        escalation_manager.get_escalation_args("kw2")

        tier1 = escalation_manager.get_keywords_at_tier(EscalationTier.IMPERSONATE_ONLY)
        assert sorted(tier1) == ["kw1", "kw2"]

        tier3 = escalation_manager.get_keywords_at_tier(EscalationTier.FULL_BYPASS)
        assert tier3 == []

    def test_keywords_at_tier_3_after_escalation(self, escalation_manager):
        """Keywords at Tier 3 after full escalation."""
        base_time = time.time()
        with patch('src.downloader.types.time') as mock_types_time, \
             patch('src.downloader.escalation_manager.time') as mock_em_time:
            mock_types_time.time.return_value = base_time
            mock_em_time.time.return_value = base_time
            escalation_manager.record_failure("kw", "HTTP Error 403")
            escalation_manager.record_failure("kw", "HTTP Error 403")
            mock_types_time.time.return_value = base_time + 400
            mock_em_time.time.return_value = base_time + 400
            escalation_manager.record_failure("kw", "HTTP Error 403")
            escalation_manager.record_failure("kw", "HTTP Error 403")

        tier3 = escalation_manager.get_keywords_at_tier(EscalationTier.FULL_BYPASS)
        assert "kw" in tier3

    def test_reset_clears_keyword_counts(self, escalation_manager):
        """reset_all() clears keyword tracking."""
        escalation_manager.get_escalation_args("kw1")
        escalation_manager.get_escalation_args("kw2")
        assert escalation_manager.get_active_keyword_count() == 2

        escalation_manager.reset_all()
        assert escalation_manager.get_active_keyword_count() == 0


# ---------------------------------------------------------------------------
# AC 5: Logging
# ---------------------------------------------------------------------------


class TestCoordinationLogging:
    """Verify INFO-level logging for coordination events."""

    def test_extension_log_includes_percentage(self, linked_pair, caplog):
        """Extension log mentions the percentage of keywords at Tier 3."""
        cb, em = linked_pair

        # All keywords at Tier 3
        base_time = time.time()
        with patch('src.downloader.types.time') as mock_types_time, \
             patch('src.downloader.escalation_manager.time') as mock_em_time:
            mock_types_time.time.return_value = base_time
            mock_em_time.time.return_value = base_time
            em.record_failure("kw1", "HTTP Error 403")
            em.record_failure("kw1", "HTTP Error 403")
            mock_types_time.time.return_value = base_time + 400
            mock_em_time.time.return_value = base_time + 400
            em.record_failure("kw1", "HTTP Error 403")
            em.record_failure("kw1", "HTTP Error 403")

        with caplog.at_level(logging.INFO, logger='src.downloader.circuit_breaker'):
            cb._get_effective_pause_seconds()

        # Should contain percentage and "Tier 3"
        extension_logs = [r for r in caplog.records if "extended" in r.message.lower()]
        assert len(extension_logs) >= 1
        assert "Tier 3" in extension_logs[0].message

    def test_shortcut_log_includes_keyword(self, linked_pair, caplog):
        """Shortcut log mentions the affected keyword."""
        cb, em = linked_pair

        for _ in range(3):
            cb.record_failure()

        with caplog.at_level(logging.INFO, logger='src.downloader.escalation_manager'):
            em.get_escalation_args("test_keyword")

        shortcut_logs = [r for r in caplog.records if "shortcutting" in r.message.lower()]
        assert len(shortcut_logs) >= 1
        assert "test_keyword" in shortcut_logs[0].message


# ---------------------------------------------------------------------------
# AC 6: Bidirectional link and safe fallback
# ---------------------------------------------------------------------------


class TestBidirectionalLink:
    """Verify both directions of the link work independently and safely."""

    def test_circuit_breaker_works_without_escalation_manager(self):
        """CircuitBreaker functions normally without linked escalation manager."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            enabled=True, consecutive_failures_threshold=2, pause_seconds=30.0
        ))
        # No set_escalation_manager() call

        cb.record_failure()
        cb.record_failure()
        assert cb.is_open  # Should still trip

        cb.record_success()
        assert cb.state.consecutive_failures == 0  # Single success resets

    def test_escalation_manager_works_without_circuit_breaker(self, escalation_manager):
        """EscalationManager functions normally without linked circuit breaker."""
        result = escalation_manager.get_escalation_args("kw")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY

    def test_set_escalation_manager_method(self, circuit_breaker, escalation_manager):
        """set_escalation_manager() stores the reference."""
        circuit_breaker.set_escalation_manager(escalation_manager)
        assert circuit_breaker._escalation_manager is escalation_manager

    def test_set_circuit_breaker_method(self, escalation_manager, circuit_breaker):
        """set_circuit_breaker() stores the reference."""
        escalation_manager.set_circuit_breaker(circuit_breaker)
        assert escalation_manager._circuit_breaker is circuit_breaker
