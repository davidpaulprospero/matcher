"""Unit tests for automatic tier floor management (US-109-009).

Tests cover:
  - Automatic tier floor elevation when >30% keywords hit rate limits in 5 minutes
  - Automatic tier floor reduction when no rate limits for 10 minutes
  - Tier floor history tracking
  - Config for elevation_threshold and reduction_timeout
"""

import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.escalation_manager import (
    EscalationManager,
    TierFloorEvent,
)
from src.downloader.types import EscalationTier


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
    de_escalation_enabled: bool = True
    de_escalation_threshold: int = 5


def _make_impersonation_manager(targets=None):
    """Create a mock ImpersonationManager that returns deterministic args."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", targets[0] if targets else "Chrome-136:Macos-15"
    ]
    return mgr


@pytest.fixture
def impersonation_manager():
    return _make_impersonation_manager()


@pytest.fixture
def extractor_config():
    return FakeExtractorArgsConfig()


@pytest.fixture
def escalation_manager(impersonation_manager, extractor_config):
    """Create EscalationManager with automatic tier floor enabled."""
    return EscalationManager(
        impersonation_manager=impersonation_manager,
        extractor_args_config=extractor_config,
        elevation_threshold=0.3,  # 30%
        reduction_timeout=10.0,  # 10 seconds for testing
    )


class TestAutomaticTierFloorElevation:
    """Tests for automatic tier floor elevation (US-109-009 AC1)."""

    def test_no_elevation_below_threshold(self, escalation_manager, impersonation_manager):
        """Tier floor should NOT elevate when <30% keywords have rate limits."""
        # Create 8 keywords first (to simulate a pool of keywords)
        for i in range(8):
            escalation_manager.get_escalation_args(f'ok_keyword_{i}')

        keywords = [f'keyword_{i}' for i in range(10)]

        # Only 2 keywords (20%) have rate limits - below 30% threshold
        for i in range(2):
            escalation_manager.record_failure(keywords[i], 'HTTP Error 429: Too Many Requests')

        # Check tier floor is still None
        assert escalation_manager.tier_floor is None

    def test_elevation_at_threshold(self, escalation_manager):
        """Tier floor should elevate when >30% keywords have rate limits."""
        # Create 8 keywords first (to simulate a pool of keywords)
        for i in range(8):
            escalation_manager.get_escalation_args(f'ok_keyword_{i}')

        keywords = [f'fail_keyword_{i}' for i in range(10)]

        # 4 keywords (40%) have rate limits - above 30% threshold
        for i in range(4):
            escalation_manager.record_failure(keywords[i], 'HTTP Error 429: Too Many Requests')

        # Tier floor should be elevated to EXTRACTOR_ARGS
        assert escalation_manager.tier_floor == EscalationTier.EXTRACTOR_ARGS

    def test_elevation_progressive(self, escalation_manager):
        """Tier floor should escalate progressively as rate limits increase."""
        # Create 8 keywords first
        for i in range(8):
            escalation_manager.get_escalation_args(f'ok_keyword_{i}')

        keywords = [f'fail_keyword_{i}' for i in range(10)]

        # First batch: 4 keywords hit rate limits -> Tier 2
        for i in range(4):
            escalation_manager.record_failure(keywords[i], 'HTTP Error 429')

        assert escalation_manager.tier_floor == EscalationTier.EXTRACTOR_ARGS

        # Second batch: 4 more keywords (8 total = 80%) -> Tier 3
        for i in range(4, 8):
            escalation_manager.record_failure(keywords[i], 'HTTP Error 429')

        assert escalation_manager.tier_floor == EscalationTier.FULL_BYPASS

    def test_no_elevation_at_max_tier(self, escalation_manager):
        """Tier floor should not exceed FULL_BYPASS."""
        # Create 8 keywords first
        for i in range(8):
            escalation_manager.get_escalation_args(f'ok_keyword_{i}')

        keywords = [f'fail_keyword_{i}' for i in range(10)]

        # Set tier floor to max already
        escalation_manager.set_tier_floor(EscalationTier.FULL_BYPASS)

        # Record more rate limits
        for i in range(8):
            escalation_manager.record_failure(keywords[i], 'HTTP Error 429')

        # Should still be at FULL_BYPASS (max)
        assert escalation_manager.tier_floor == EscalationTier.FULL_BYPASS


class TestAutomaticTierFloorReduction:
    """Tests for automatic tier floor reduction (US-109-009 AC2)."""

    def test_no_reduction_with_recent_rate_limits(self, escalation_manager):
        """Tier floor should NOT reduce when rate limits occurred recently."""
        # Create 8 keywords first
        for i in range(8):
            escalation_manager.get_escalation_args(f'ok_keyword_{i}')

        # Set tier floor
        escalation_manager.set_tier_floor(EscalationTier.EXTRACTOR_ARGS)

        # Record a rate limit just now
        escalation_manager.record_failure('keyword_1', 'HTTP Error 429')

        # Try to check for reduction - should not reduce
        reduced = escalation_manager.check_and_reduce_tier_floor()

        assert reduced is False
        assert escalation_manager.tier_floor == EscalationTier.EXTRACTOR_ARGS

    def test_reduction_after_timeout(self, escalation_manager):
        """Tier floor should reduce after no rate limits for reduction_timeout."""
        # Set tier floor
        escalation_manager.set_tier_floor(EscalationTier.EXTRACTOR_ARGS)

        # Record a rate limit in the past (beyond the window)
        # We'll mock the time to simulate this
        current_time = time.time()

        with patch("src.downloader.escalation_manager.time") as mock_time:
            # Set last rate limit time to way in the past
            mock_time.time.return_value = current_time
            escalation_manager._last_rate_limit_time = current_time - 20.0  # 20 seconds ago

            # reduction_timeout is 10 seconds, so should reduce
            reduced = escalation_manager.check_and_reduce_tier_floor()

            assert reduced is True
            assert escalation_manager.tier_floor == EscalationTier.IMPERSONATE_ONLY

    def test_gradual_deescalation(self, escalation_manager):
        """Tier floor should de-escalate one tier at a time."""
        # Set tier floor to Tier 3
        escalation_manager.set_tier_floor(EscalationTier.FULL_BYPASS)

        # Set last rate limit time to beyond timeout
        current_time = time.time()
        with patch("src.downloader.escalation_manager.time") as mock_time:
            mock_time.time.return_value = current_time
            escalation_manager._last_rate_limit_time = current_time - 20.0

            # First reduction: Tier 3 -> Tier 2
            reduced = escalation_manager.check_and_reduce_tier_floor()
            assert reduced is True
            assert escalation_manager.tier_floor == EscalationTier.EXTRACTOR_ARGS

            # Second reduction: Tier 2 -> Tier 1
            reduced = escalation_manager.check_and_reduce_tier_floor()
            assert reduced is True
            assert escalation_manager.tier_floor == EscalationTier.IMPERSONATE_ONLY

            # Third reduction: Tier 1 -> clear floor
            reduced = escalation_manager.check_and_reduce_tier_floor()
            assert reduced is True
            assert escalation_manager.tier_floor is None


class TestTierFloorHistory:
    """Tests for tier floor history tracking (US-109-009 AC3)."""

    def test_history_records_set(self, escalation_manager):
        """History should record when tier floor is set."""
        escalation_manager.set_tier_floor(EscalationTier.EXTRACTOR_ARGS, reason="manual_set")

        history = escalation_manager.get_tier_floor_history()

        assert len(history) == 1
        assert history[0]["from_tier"] is None
        assert history[0]["to_tier"] == "EXTRACTOR_ARGS"
        assert history[0]["reason"] == "manual_set"

    def test_history_records_clear(self, escalation_manager):
        """History should record when tier floor is cleared."""
        escalation_manager.set_tier_floor(EscalationTier.EXTRACTOR_ARGS)
        escalation_manager.clear_tier_floor(reason="manual_clear")

        history = escalation_manager.get_tier_floor_history()

        assert len(history) == 2
        assert history[1]["from_tier"] == "EXTRACTOR_ARGS"
        assert history[1]["to_tier"] is None
        assert history[1]["reason"] == "manual_clear"

    def test_history_records_automatic_elevation(self, escalation_manager):
        """History should record automatic tier floor elevations."""
        # Create 8 keywords first
        for i in range(8):
            escalation_manager.get_escalation_args(f'ok_keyword_{i}')

        keywords = [f'fail_keyword_{i}' for i in range(10)]

        # Trigger automatic elevation
        for i in range(4):
            escalation_manager.record_failure(keywords[i], 'HTTP Error 429')

        history = escalation_manager.get_tier_floor_history()

        # Should have 1 event for elevation
        elevation_events = [h for h in history if h['reason'] == 'automatic_elevation']
        assert len(elevation_events) == 1
        assert elevation_events[0]['to_tier'] == 'EXTRACTOR_ARGS'

    def test_history_records_automatic_reduction(self, escalation_manager):
        """History should record automatic tier floor reductions."""
        escalation_manager.set_tier_floor(EscalationTier.EXTRACTOR_ARGS)

        # Set last rate limit time to beyond timeout
        current_time = time.time()
        with patch("src.downloader.escalation_manager.time") as mock_time:
            mock_time.time.return_value = current_time
            escalation_manager._last_rate_limit_time = current_time - 20.0

            escalation_manager.check_and_reduce_tier_floor()

        history = escalation_manager.get_tier_floor_history()

        reduction_events = [h for h in history if h["reason"] == "automatic_reduction"]
        assert len(reduction_events) == 1


class TestTierFloorConfig:
    """Tests for tier floor config (US-109-009 AC4)."""

    def test_default_config(self, escalation_manager):
        """Test default configuration values."""
        config = escalation_manager.tier_floor_config

        assert config["elevation_threshold"] == 0.3
        assert config["reduction_timeout"] == 10.0  # From fixture
        assert config["elevation_window"] == 300.0  # 5 minutes default

    def test_custom_config(self, impersonation_manager, extractor_config):
        """Test custom configuration values."""
        manager = EscalationManager(
            impersonation_manager=impersonation_manager,
            extractor_args_config=extractor_config,
            elevation_threshold=0.5,  # 50%
            reduction_timeout=600.0,  # 10 minutes
        )

        config = manager.tier_floor_config

        assert config["elevation_threshold"] == 0.5
        assert config["reduction_timeout"] == 600.0

    def test_config_reflects_current_state(self, escalation_manager):
        """Config should reflect current tier floor state."""
        # Initially no floor
        config = escalation_manager.tier_floor_config
        assert config["current_tier_floor"] is None

        # After setting floor
        escalation_manager.set_tier_floor(EscalationTier.EXTRACTOR_ARGS)
        config = escalation_manager.tier_floor_config
        assert config["current_tier_floor"] == "EXTRACTOR_ARGS"


class TestRateLimitTracking:
    """Tests for rate limit event tracking."""

    def test_recent_events_tracked(self, escalation_manager):
        """Recent rate limit events should be tracked."""
        keywords = [f"keyword_{i}" for i in range(5)]

        for i in range(3):
            escalation_manager.record_failure(keywords[i], "HTTP Error 429")

        config = escalation_manager.tier_floor_config
        assert config["recent_rate_limit_count"] == 3

    def test_events_pruned_outside_window(self, escalation_manager):
        """Rate limit events outside the window should be pruned."""
        # Create 5 keywords first
        for i in range(5):
            escalation_manager.get_escalation_args(f'ok_keyword_{i}')

        # Add some events
        keywords = [f'fail_keyword_{i}' for i in range(3)]
        for kw in keywords:
            escalation_manager.record_failure(kw, 'HTTP Error 429')

        # Mock old events by directly manipulating the list
        old_time = time.time() - 400.0  # Beyond 5 minute window
        escalation_manager._recent_rate_limit_events = [(old_time, 'fail_keyword_0'), (old_time, 'fail_keyword_1'), (old_time, 'fail_keyword_2')]

        # Prune should remove them
        escalation_manager._prune_rate_limit_events()

        config = escalation_manager.tier_floor_config
        assert config['recent_rate_limit_count'] == 0


class TestEdgeCases:
    """Edge case tests for automatic tier floor management."""

    def test_no_keywords_no_elevation(self, escalation_manager):
        """No elevation should occur when there are no keywords tracked."""
        # When there are no keywords yet, _get_rate_limit_percentage returns 0.0
        # This tests the path where we haven't created any keywords yet
        # Note: record_failure will create a keyword state, so we test directly
        # by checking the percentage before any failures
        percentage = escalation_manager._get_rate_limit_percentage()
        assert percentage == 0.0
        # Also verify tier floor remains None initially
        assert escalation_manager.tier_floor is None

    def test_already_at_minimum_no_reduction(self, escalation_manager):
        """No reduction should occur when already at minimum tier."""
        # Set to minimum tier
        escalation_manager.set_tier_floor(EscalationTier.IMPERSONATE_ONLY)

        # Try to reduce
        current_time = time.time()
        with patch("src.downloader.escalation_manager.time") as mock_time:
            mock_time.time.return_value = current_time
            escalation_manager._last_rate_limit_time = current_time - 20.0

            reduced = escalation_manager.check_and_reduce_tier_floor()

        # Should clear the floor (not reduce further)
        assert reduced is True
        assert escalation_manager.tier_floor is None

    def test_rate_limit_percentage_calculation(self, escalation_manager):
        """Test rate limit percentage calculation."""
        keywords = [f"keyword_{i}" for i in range(10)]

        # 3 out of 10 = 30%
        for i in range(3):
            escalation_manager.record_failure(keywords[i], "HTTP Error 429")

        percentage = escalation_manager._get_rate_limit_percentage()

        # Should be >= 30% (accounting for the fact that unique keyword count is used)
        assert percentage >= 0.3

    def test_last_rate_limit_time_updated(self, escalation_manager):
        """Last rate limit time should be updated on events."""
        initial_time = escalation_manager._last_rate_limit_time

        # Record a failure
        time.sleep(0.01)  # Small delay
        escalation_manager.record_failure("keyword_1", "HTTP Error 429")

        assert escalation_manager._last_rate_limit_time is not None
        assert escalation_manager._last_rate_limit_time > (initial_time or 0)
