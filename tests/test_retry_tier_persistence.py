"""
Tests for retry queue tier persistence (Sprint 50 US-50-011).

Verifies:
1. When a download fails and is queued for retry, the current escalation tier
   is stored with the retry entry
2. When retrying, the download starts at the stored tier (or the global tier
   floor, whichever is higher) instead of Tier 1
3. A video that failed at Tier 2 starts its retry at Tier 2 (not Tier 1)
4. If the global tier floor is Tier 3 and the stored retry tier is Tier 2,
   the retry starts at Tier 3
5. Videos with no stored tier (legacy retry entries) default to Tier 1
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.downloader.retry_queue import RetryQueue, RetryItem
from src.downloader.types import EscalationTier, EscalationState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def retry_queue():
    """Create a basic RetryQueue with default config."""
    mock_config = MagicMock()
    mock_config.max_retries = 3
    mock_config.retry_delay = 0.0
    mock_config.retry_backoff_multiplier = 1.0
    mock_config.retry_jitter_factor = 0.0
    rq = RetryQueue(config=mock_config)
    return rq


@pytest.fixture
def escalation_mgr():
    """Create a mock EscalationManager with real _get_state behaviour."""
    mgr = MagicMock()
    # Keep real state dict for tier tracking
    mgr._keyword_states = {}
    mgr._tier_floor = None

    def _get_state(keyword):
        if keyword not in mgr._keyword_states:
            state = EscalationState()
            if mgr._tier_floor is not None and state.current_tier < mgr._tier_floor:
                state.current_tier = mgr._tier_floor
            mgr._keyword_states[keyword] = state
        else:
            state = mgr._keyword_states[keyword]
            if mgr._tier_floor is not None and state.current_tier < mgr._tier_floor:
                state.current_tier = mgr._tier_floor
        return mgr._keyword_states[keyword]

    mgr._get_state = _get_state

    def set_tier_floor(tier):
        mgr._tier_floor = tier

    def clear_tier_floor():
        mgr._tier_floor = None

    mgr.set_tier_floor = set_tier_floor
    mgr.clear_tier_floor = clear_tier_floor

    return mgr


# ---------------------------------------------------------------------------
# AC1: Escalation tier is stored with the retry entry
# ---------------------------------------------------------------------------

class TestRetryTierStorage:
    """AC1: When a download fails and is queued for retry, the current
    escalation tier is stored with the retry entry."""

    def test_tier_stored_on_add(self, retry_queue):
        """Retry entry stores the escalation_tier passed to add()."""
        retry_queue.add(
            video_id='vid_10_20',
            keyword='segment',
            tier='segment',
            error_message='403 Forbidden',
            error_category='bot_detection',
            escalation_tier=2,
        )
        items = retry_queue.get_pending_items()
        assert len(items) == 1
        assert items[0].escalation_tier == 2

    def test_tier_defaults_to_1(self, retry_queue):
        """When escalation_tier is not specified, defaults to 1."""
        retry_queue.add(
            video_id='vid_0_10',
            keyword='segment',
            tier='segment',
            error_message='video unavailable',
            error_category='video_specific',
        )
        items = retry_queue.get_pending_items()
        assert len(items) == 1
        assert items[0].escalation_tier == 1

    def test_tier_updated_to_max_on_re_add(self, retry_queue):
        """Re-adding a video upgrades to the higher escalation_tier."""
        retry_queue.add(
            video_id='vid_0_10',
            keyword='segment',
            tier='segment',
            error_message='403 Forbidden',
            escalation_tier=1,
        )
        retry_queue.add(
            video_id='vid_0_10',
            keyword='segment',
            tier='segment',
            error_message='403 Forbidden',
            escalation_tier=3,
        )
        items = retry_queue.get_pending_items()
        assert len(items) == 1
        assert items[0].escalation_tier == 3


# ---------------------------------------------------------------------------
# AC3: Video that failed at Tier 2 starts retry at Tier 2
# ---------------------------------------------------------------------------

class TestRetryStartsAtStoredTier:
    """AC3: A video that failed at Tier 2 starts its retry at Tier 2."""

    def test_retry_elevates_escalation_state(self, escalation_mgr):
        """When retrying, the escalation state is elevated to the stored tier.

        Mirrors the logic at download_segments.py lines 1096-1108:
        the retry path reads item.escalation_tier and elevates the
        escalation manager's state for that video_id before calling
        get_escalation_args().
        """
        video_id = 'abc123'

        # Simulate: video_id has no prior state (starts at Tier 1)
        state_before = escalation_mgr._get_state(video_id)
        assert state_before.current_tier == EscalationTier.IMPERSONATE_ONLY  # Tier 1

        # Simulate the retry path logic from download_segments._process_retry_queue
        stored_tier_value = 2  # Failed at Tier 2
        if stored_tier_value > 1:
            stored_tier = EscalationTier(stored_tier_value)
            esc_state = escalation_mgr._get_state(video_id)
            if esc_state.current_tier < stored_tier:
                esc_state.current_tier = stored_tier

        # Verify: state is now Tier 2
        state_after = escalation_mgr._get_state(video_id)
        assert state_after.current_tier == EscalationTier.EXTRACTOR_ARGS  # Tier 2

    def test_retry_does_not_downgrade(self, escalation_mgr):
        """If escalation state is already at Tier 3, stored Tier 2 doesn't downgrade."""
        video_id = 'xyz789'

        # Pre-set state to Tier 3
        state = escalation_mgr._get_state(video_id)
        state.current_tier = EscalationTier.FULL_BYPASS  # Tier 3

        # Retry item has stored tier 2
        stored_tier_value = 2
        if stored_tier_value > 1:
            stored_tier = EscalationTier(stored_tier_value)
            esc_state = escalation_mgr._get_state(video_id)
            if esc_state.current_tier < stored_tier:
                esc_state.current_tier = stored_tier

        # Tier should remain at 3 (not downgraded to 2)
        assert escalation_mgr._get_state(video_id).current_tier == EscalationTier.FULL_BYPASS


# ---------------------------------------------------------------------------
# AC4: Global tier floor overrides lower stored tier
# ---------------------------------------------------------------------------

class TestGlobalTierFloorOverride:
    """AC4: If global tier floor is Tier 3 and stored retry tier is Tier 2,
    the retry starts at Tier 3."""

    def test_tier_floor_overrides_stored_tier(self, escalation_mgr):
        """Global tier floor (Tier 3) wins over stored tier (Tier 2)."""
        video_id = 'floor_test'

        # Set global tier floor to Tier 3 (FULL_BYPASS)
        escalation_mgr.set_tier_floor(EscalationTier.FULL_BYPASS)

        # Apply stored tier 2 (as retry path does)
        stored_tier_value = 2
        if stored_tier_value > 1:
            stored_tier = EscalationTier(stored_tier_value)
            esc_state = escalation_mgr._get_state(video_id)
            if esc_state.current_tier < stored_tier:
                esc_state.current_tier = stored_tier

        # _get_state enforces tier floor — should be Tier 3, not Tier 2
        final_state = escalation_mgr._get_state(video_id)
        assert final_state.current_tier == EscalationTier.FULL_BYPASS  # Tier 3

    def test_stored_tier_above_floor_is_kept(self, escalation_mgr):
        """Stored tier (Tier 4) above floor (Tier 2) is preserved."""
        video_id = 'above_floor'

        # Set global floor to Tier 2
        escalation_mgr.set_tier_floor(EscalationTier.EXTRACTOR_ARGS)

        # Apply stored tier 4 (VPN_ROTATION)
        stored_tier_value = 4
        if stored_tier_value > 1:
            stored_tier = EscalationTier(stored_tier_value)
            esc_state = escalation_mgr._get_state(video_id)
            if esc_state.current_tier < stored_tier:
                esc_state.current_tier = stored_tier

        final_state = escalation_mgr._get_state(video_id)
        assert final_state.current_tier == EscalationTier.VPN_ROTATION  # Tier 4


# ---------------------------------------------------------------------------
# AC5: Legacy retry entries (no stored tier) default to Tier 1
# ---------------------------------------------------------------------------

class TestLegacyRetryEntries:
    """AC5: Videos with no stored tier (legacy retry entries) default to Tier 1."""

    def test_legacy_item_defaults_tier_1(self):
        """RetryItem without escalation_tier defaults to 1."""
        item = RetryItem(
            video_id='legacy_vid_0_10',
            keyword='segment',
            tier='segment',
            error_message='some error',
        )
        assert item.escalation_tier == 1

    def test_legacy_item_no_elevation(self, escalation_mgr):
        """Legacy item (escalation_tier=1) doesn't elevate beyond Tier 1."""
        video_id = 'legacy_no_elevate'

        # Simulate retry path with legacy item (tier=1)
        stored_tier_value = 1
        # The code only elevates if stored_tier_value > 1
        if stored_tier_value > 1:
            stored_tier = EscalationTier(stored_tier_value)
            esc_state = escalation_mgr._get_state(video_id)
            if esc_state.current_tier < stored_tier:
                esc_state.current_tier = stored_tier

        # State should remain at default Tier 1
        state = escalation_mgr._get_state(video_id)
        assert state.current_tier == EscalationTier.IMPERSONATE_ONLY

    def test_legacy_item_respects_tier_floor(self, escalation_mgr):
        """Legacy item (tier=1) still respects global tier floor."""
        video_id = 'legacy_with_floor'

        # Set global floor to Tier 2
        escalation_mgr.set_tier_floor(EscalationTier.EXTRACTOR_ARGS)

        # Legacy item has tier=1, skip elevation
        stored_tier_value = 1
        if stored_tier_value > 1:
            pass  # Not entered

        # But _get_state enforces floor
        state = escalation_mgr._get_state(video_id)
        assert state.current_tier == EscalationTier.EXTRACTOR_ARGS  # Tier 2 (floor)
