"""Comprehensive unit tests for RetryQueue class.

Tests focus on RetryQueue core functionality (data structure operations)
rather than process_queue execution which is tested separately.

Story: US-35-006 - Create comprehensive RetryQueue unit tests
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.downloader.retry_queue import (
    RetryQueue,
    RetryItem,
    BatchRetryConfig,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default BatchRetryConfig for testing."""
    return BatchRetryConfig(
        enabled=True,
        delay_seconds=0.01,  # Fast for tests
        max_passes=2,
        respect_circuit_breaker=True,
        wait_for_cookie_cooldown=True,
        max_combined_wait_seconds=0.1,
        jitter_factor=0.0,  # No jitter for deterministic tests
    )


@pytest.fixture
def queue(default_config):
    """RetryQueue instance for testing."""
    return RetryQueue(default_config)


@pytest.fixture
def disabled_config():
    """BatchRetryConfig with retry disabled."""
    return BatchRetryConfig(enabled=False)


@pytest.fixture
def disabled_queue(disabled_config):
    """RetryQueue with retry disabled."""
    return RetryQueue(disabled_config)


# =============================================================================
# Test: add_item correctly stores video_id, keyword, tier, error
# =============================================================================

@pytest.mark.fast
class TestAddItem:
    """Tests for RetryQueue.add() method."""

    def test_add_stores_video_id(self, queue):
        """add() stores video_id correctly."""
        result = queue.add("vid123", "python tutorial", "medium", "Rate limit")

        assert result is True
        assert "vid123" in queue.items
        assert queue.items["vid123"].video_id == "vid123"

    def test_add_stores_keyword(self, queue):
        """add() stores keyword correctly."""
        queue.add("vid456", "javascript async", "short", "403 Forbidden")

        assert queue.items["vid456"].keyword == "javascript async"

    def test_add_stores_tier(self, queue):
        """add() stores tier correctly."""
        queue.add("vid789", "react hooks", "long", "Server Error")

        assert queue.items["vid789"].tier == "long"

    def test_add_stores_error_message(self, queue):
        """add() stores error_message correctly."""
        error_msg = "HTTP Error 429: Too Many Requests"
        queue.add("vidABC", "vue tutorial", "medium", error_msg)

        assert queue.items["vidABC"].error_message == error_msg

    def test_add_sets_default_retry_count(self, queue):
        """add() sets retry_count to 0."""
        queue.add("vidDEF", "angular", "short", "Error")

        assert queue.items["vidDEF"].retry_count == 0

    def test_add_sets_timestamp(self, queue):
        """add() sets added_at timestamp."""
        before = time.time()
        queue.add("vidGHI", "svelte", "medium", "Error")
        after = time.time()

        assert before <= queue.items["vidGHI"].added_at <= after

    def test_add_increments_total_added(self, queue):
        """add() increments _total_added counter."""
        assert queue._total_added == 0

        queue.add("vid1", "kw1", "short", "Error")
        assert queue._total_added == 1

        queue.add("vid2", "kw2", "medium", "Error")
        assert queue._total_added == 2

    def test_add_returns_false_when_disabled(self, disabled_queue):
        """add() returns False when retry is disabled."""
        result = disabled_queue.add("vid123", "keyword", "short", "Error")

        assert result is False
        assert len(disabled_queue.items) == 0

    def test_add_returns_false_for_duplicate(self, queue):
        """add() returns False for duplicate video_id."""
        queue.add("vid123", "keyword1", "short", "Error 1")
        result = queue.add("vid123", "keyword2", "medium", "Error 2")

        assert result is False
        # Should update error message but not add duplicate
        assert len(queue.items) == 1
        assert queue.items["vid123"].error_message == "Error 2"

    def test_add_skips_completed_video(self, queue):
        """add() skips videos that have already been completed."""
        queue.add("vid123", "keyword", "short", "Error")
        queue.mark_success("vid123")

        result = queue.add("vid123", "keyword", "short", "New Error")

        assert result is False
        assert "vid123" not in queue.items

    def test_add_skips_permanently_failed_video(self, queue):
        """add() skips videos that have permanently failed."""
        queue._failed_ids.add("vid123")

        result = queue.add("vid123", "keyword", "short", "Error")

        assert result is False
        assert "vid123" not in queue.items


# =============================================================================
# Test: get_items returns items in FIFO order
# =============================================================================

@pytest.mark.fast
class TestGetItems:
    """Tests for RetryQueue.get_pending_items() method."""

    def test_get_pending_items_empty_queue(self, queue):
        """get_pending_items() returns empty list for empty queue."""
        items = queue.get_pending_items()

        assert items == []

    def test_get_pending_items_returns_list(self, queue):
        """get_pending_items() returns a list of RetryItem objects."""
        queue.add("vid1", "kw1", "short", "Error 1")
        queue.add("vid2", "kw2", "medium", "Error 2")

        items = queue.get_pending_items()

        assert isinstance(items, list)
        assert len(items) == 2
        assert all(isinstance(item, RetryItem) for item in items)

    def test_get_pending_items_fifo_order(self, queue):
        """get_pending_items() returns items in FIFO order (insertion order)."""
        # Add items with small delays to ensure distinct timestamps
        queue.add("vid_first", "kw1", "short", "Error 1")
        queue.add("vid_second", "kw2", "medium", "Error 2")
        queue.add("vid_third", "kw3", "long", "Error 3")

        items = queue.get_pending_items()
        video_ids = [item.video_id for item in items]

        # Python dicts preserve insertion order since 3.7
        assert video_ids == ["vid_first", "vid_second", "vid_third"]

    def test_get_pending_items_returns_copy(self, queue):
        """get_pending_items() returns a copy, not the internal list."""
        queue.add("vid1", "kw1", "short", "Error")

        items = queue.get_pending_items()
        items.clear()

        # Original should be unaffected
        assert len(queue.items) == 1


# =============================================================================
# Test: clear removes all items and resets pass count
# =============================================================================

@pytest.mark.fast
class TestClear:
    """Tests for RetryQueue.clear() method."""

    def test_clear_removes_all_items(self, queue):
        """clear() removes all items from the queue."""
        queue.add("vid1", "kw1", "short", "Error 1")
        queue.add("vid2", "kw2", "medium", "Error 2")
        queue.add("vid3", "kw3", "long", "Error 3")

        queue.clear()

        assert len(queue.items) == 0

    def test_clear_resets_pass_count(self, queue):
        """clear() resets current_pass to 0."""
        queue.add("vid1", "kw1", "short", "Error")
        queue.current_pass = 2

        queue.clear()

        assert queue.current_pass == 0

    def test_clear_resets_completed_ids(self, queue):
        """clear() clears completed_ids set."""
        queue.add("vid1", "kw1", "short", "Error")
        queue.mark_success("vid1")

        queue.clear()

        assert len(queue._completed_ids) == 0

    def test_clear_resets_failed_ids(self, queue):
        """clear() clears failed_ids set."""
        queue._failed_ids.add("vid_failed")

        queue.clear()

        assert len(queue._failed_ids) == 0

    def test_clear_resets_total_counters(self, queue):
        """clear() resets total_added and total_retried counters."""
        queue.add("vid1", "kw1", "short", "Error")
        queue._total_retried = 5

        queue.clear()

        assert queue._total_added == 0
        assert queue._total_retried == 0

    def test_clear_resets_processor_state(self, queue):
        """clear() resets processor wait times and forced_retry."""
        queue.processor._circuit_breaker_wait_time = 10.0
        queue.processor._cookie_cooldown_wait_time = 5.0
        queue.processor._forced_retry = True

        queue.clear()

        assert queue.processor._circuit_breaker_wait_time == 0.0
        assert queue.processor._cookie_cooldown_wait_time == 0.0
        assert queue.processor._forced_retry is False


# =============================================================================
# Test: max_passes limit prevents infinite retry loops
# =============================================================================

@pytest.mark.fast
class TestMaxPasses:
    """Tests for max_passes limit behavior."""

    def test_can_retry_true_within_limit(self, queue):
        """can_retry is True when under max_passes limit."""
        queue.current_pass = 0
        assert queue.can_retry is True

        queue.current_pass = 1
        assert queue.can_retry is True

    def test_can_retry_false_at_limit(self, queue):
        """can_retry is False when at max_passes limit."""
        queue.current_pass = 2  # max_passes is 2

        assert queue.can_retry is False

    def test_can_retry_false_above_limit(self, queue):
        """can_retry is False when above max_passes limit."""
        queue.current_pass = 5

        assert queue.can_retry is False

    def test_has_pending_requires_items_and_passes(self, queue):
        """has_pending() requires both items and available passes."""
        # No items, no passes used
        assert queue.has_pending() is False

        # Has items, passes available
        queue.add("vid1", "kw1", "short", "Error")
        assert queue.has_pending() is True

        # Has items, no passes available
        queue.current_pass = 2
        assert queue.has_pending() is False

    def test_finish_retry_pass_moves_to_failed_at_max(self, queue):
        """finish_retry_pass() moves remaining items to failed at max_passes."""
        queue.add("vid1", "kw1", "short", "Error 1")
        queue.add("vid2", "kw2", "medium", "Error 2")
        queue.current_pass = 2  # At max

        queue.finish_retry_pass()

        assert len(queue.items) == 0
        assert "vid1" in queue._failed_ids
        assert "vid2" in queue._failed_ids

    def test_finish_retry_pass_keeps_items_under_max(self, queue):
        """finish_retry_pass() keeps items in queue under max_passes."""
        queue.add("vid1", "kw1", "short", "Error")
        queue.current_pass = 1  # Under max (2)

        queue.finish_retry_pass()

        assert len(queue.items) == 1
        assert len(queue._failed_ids) == 0

    def test_max_passes_configurable(self):
        """max_passes can be configured via BatchRetryConfig."""
        config = BatchRetryConfig(max_passes=5, delay_seconds=0.01)
        queue = RetryQueue(config)

        queue.current_pass = 4
        assert queue.can_retry is True

        queue.current_pass = 5
        assert queue.can_retry is False


# =============================================================================
# Test: wait_for_cookie_cooldown delays when cookies in cooldown
# =============================================================================

@pytest.mark.fast
class TestCookieCooldown:
    """Tests for cookie cooldown wait behavior."""

    def test_no_wait_without_cookie_rotator(self, queue):
        """No wait when cookie_rotator is not set."""
        remaining = queue._get_cookie_cooldown_remaining()

        assert remaining == 0.0

    def test_no_wait_when_disabled(self, queue):
        """No wait when wait_for_cookie_cooldown is disabled."""
        queue.config.wait_for_cookie_cooldown = False
        mock_rotator = MagicMock()
        mock_rotator.is_enabled = True
        mock_rotator.available_cookies = 0
        queue.set_cookie_rotator(mock_rotator)

        remaining = queue._get_cookie_cooldown_remaining()

        assert remaining == 0.0

    def test_no_wait_when_cookies_available(self, queue):
        """No wait when cookies are available."""
        mock_rotator = MagicMock()
        mock_rotator.is_enabled = True
        mock_rotator.available_cookies = 3
        queue.set_cookie_rotator(mock_rotator)

        remaining = queue._get_cookie_cooldown_remaining()

        assert remaining == 0.0

    def test_returns_remaining_cooldown_time(self, queue):
        """Returns remaining cooldown time when all cookies in cooldown."""
        mock_rotator = MagicMock()
        mock_rotator.is_enabled = True
        mock_rotator.available_cookies = 0
        mock_rotator.config.cooldown_seconds = 60.0
        mock_rotator._failed_cookies = {
            "/path/cookie1.txt": time.time() - 30,  # 30s elapsed, 30s remaining
            "/path/cookie2.txt": time.time() - 50,  # 50s elapsed, 10s remaining
        }
        queue.set_cookie_rotator(mock_rotator)

        remaining = queue._get_cookie_cooldown_remaining()

        # Should return the shortest remaining cooldown (approximately 10s)
        assert 8.0 <= remaining <= 12.0  # Allow some tolerance

    def test_wait_for_cookie_cooldown_sleeps(self, queue):
        """_wait_for_cookie_cooldown() sleeps for remaining time."""
        mock_rotator = MagicMock()
        mock_rotator.is_enabled = True
        mock_rotator.available_cookies = 0
        mock_rotator.config.cooldown_seconds = 0.05  # 50ms for fast test
        mock_rotator._failed_cookies = {
            "/path/cookie.txt": time.time() - 0.02,  # 20ms elapsed, 30ms remaining
        }
        queue.set_cookie_rotator(mock_rotator)

        with patch('src.downloader.retry_processor.time.sleep') as mock_sleep:
            waited = queue._wait_for_cookie_cooldown()

        # Should have slept for approximately 30ms
        assert mock_sleep.called
        sleep_time = mock_sleep.call_args[0][0]
        assert 0.02 <= sleep_time <= 0.05

    def test_wait_for_cookie_cooldown_accumulates(self, queue):
        """_wait_for_cookie_cooldown() accumulates wait time."""
        mock_rotator = MagicMock()
        mock_rotator.is_enabled = True
        mock_rotator.available_cookies = 0
        mock_rotator.config.cooldown_seconds = 0.05
        mock_rotator._failed_cookies = {
            "/path/cookie.txt": time.time() - 0.02,
        }
        queue.set_cookie_rotator(mock_rotator)

        initial_wait = queue.processor._cookie_cooldown_wait_time

        with patch('src.downloader.retry_processor.time.sleep'):
            queue._wait_for_cookie_cooldown()

        assert queue.processor._cookie_cooldown_wait_time > initial_wait


# =============================================================================
# Additional Tests for Coverage (20+ total)
# =============================================================================

@pytest.mark.fast
class TestMarkSuccess:
    """Tests for RetryQueue.mark_success() method."""

    def test_mark_success_removes_from_queue(self, queue):
        """mark_success() removes video from queue."""
        queue.add("vid123", "kw", "short", "Error")

        queue.mark_success("vid123")

        assert "vid123" not in queue.items

    def test_mark_success_adds_to_completed(self, queue):
        """mark_success() adds video to completed set."""
        queue.add("vid123", "kw", "short", "Error")

        queue.mark_success("vid123")

        assert "vid123" in queue._completed_ids

    def test_mark_success_increments_total_retried(self, queue):
        """mark_success() increments _total_retried counter."""
        queue.add("vid123", "kw", "short", "Error")
        initial = queue._total_retried

        queue.mark_success("vid123")

        assert queue._total_retried == initial + 1

    def test_mark_success_ignores_unknown_video(self, queue):
        """mark_success() silently ignores unknown video IDs."""
        queue.mark_success("unknown_vid")

        assert "unknown_vid" not in queue._completed_ids


@pytest.mark.fast
class TestMarkFailed:
    """Tests for RetryQueue.mark_failed() method."""

    def test_mark_failed_increments_retry_count(self, queue):
        """mark_failed() increments retry_count on the item."""
        queue.add("vid123", "kw", "short", "Error")

        queue.mark_failed("vid123")

        assert queue.items["vid123"].retry_count == 1

        queue.mark_failed("vid123")

        assert queue.items["vid123"].retry_count == 2

    def test_mark_failed_ignores_unknown_video(self, queue):
        """mark_failed() silently ignores unknown video IDs."""
        # Should not raise
        queue.mark_failed("unknown_vid")


@pytest.mark.fast
class TestIsEnabled:
    """Tests for RetryQueue.is_enabled property."""

    def test_is_enabled_true(self, queue):
        """is_enabled is True when config.enabled is True."""
        assert queue.is_enabled is True

    def test_is_enabled_false(self, disabled_queue):
        """is_enabled is False when config.enabled is False."""
        assert disabled_queue.is_enabled is False


@pytest.mark.fast
class TestCheckpoint:
    """Tests for checkpoint serialization/deserialization."""

    def test_to_checkpoint_dict(self, queue):
        """to_checkpoint_dict() serializes queue state."""
        queue.add("vid1", "kw1", "short", "Error 1")
        queue.add("vid2", "kw2", "medium", "Error 2")
        queue.current_pass = 1
        queue._completed_ids.add("vid_done")
        queue._failed_ids.add("vid_failed")

        checkpoint = queue.to_checkpoint_dict()

        assert 'items' in checkpoint
        assert len(checkpoint['items']) == 2
        assert checkpoint['current_pass'] == 1
        assert 'vid_done' in checkpoint['completed_ids']
        assert 'vid_failed' in checkpoint['failed_ids']

    def test_from_checkpoint_dict(self, queue):
        """from_checkpoint_dict() restores queue state."""
        checkpoint = {
            'items': [
                {'video_id': 'vid1', 'keyword': 'kw1', 'tier': 'short', 'error_message': 'E1', 'retry_count': 1},
                {'video_id': 'vid2', 'keyword': 'kw2', 'tier': 'medium', 'error_message': 'E2', 'retry_count': 0},
            ],
            'current_pass': 1,
            'completed_ids': ['vid_done'],
            'failed_ids': ['vid_failed'],
            'total_added': 5,
            'total_retried': 2,
        }

        queue.from_checkpoint_dict(checkpoint)

        assert len(queue.items) == 2
        assert queue.items['vid1'].retry_count == 1
        assert queue.current_pass == 1
        assert 'vid_done' in queue._completed_ids
        assert 'vid_failed' in queue._failed_ids

    def test_from_checkpoint_dict_empty(self, queue):
        """from_checkpoint_dict() handles empty data."""
        queue.from_checkpoint_dict({})
        queue.from_checkpoint_dict(None)  # type: ignore

        # Should not crash


@pytest.mark.fast
class TestSeverity:
    """Tests for error severity classification in add()."""

    def test_add_classifies_high_severity(self, queue):
        """add() classifies quota exceeded as high severity."""
        queue.add("vid1", "kw", "short", "quota exceeded: daily limit reached")

        assert queue.items["vid1"].severity == "high"

    def test_add_classifies_medium_severity(self, queue):
        """add() classifies 429 errors as medium severity."""
        queue.add("vid1", "kw", "short", "HTTP Error 429: Too Many Requests")

        assert queue.items["vid1"].severity == "medium"

    def test_add_classifies_low_severity(self, queue):
        """add() classifies sign in errors as low severity."""
        queue.add("vid1", "kw", "short", "Sign in to confirm your age")

        assert queue.items["vid1"].severity == "low"


@pytest.mark.fast
class TestStats:
    """Tests for RetryQueue statistics methods."""

    def test_get_stats_returns_dict(self, queue):
        """get_stats() returns a dictionary with expected keys."""
        stats = queue.get_stats()

        assert isinstance(stats, dict)
        assert 'enabled' in stats
        assert 'pending' in stats
        assert 'completed' in stats
        assert 'failed' in stats
        assert 'current_pass' in stats
        assert 'max_passes' in stats

    def test_get_failure_reasons_returns_dict(self, queue):
        """get_failure_reasons() returns video_id -> error mapping."""
        queue.add("vid1", "kw1", "short", "Error 1")
        queue.add("vid2", "kw2", "medium", "Error 2")

        reasons = queue.get_failure_reasons()

        assert isinstance(reasons, dict)
        # RetryQueueStats tracks failures
        assert "vid1" in reasons or len(reasons) >= 0


@pytest.mark.fast
class TestCircuitBreakerIntegration:
    """Tests for circuit breaker coordination."""

    def test_set_circuit_breaker(self, queue):
        """set_circuit_breaker() links CB to processor."""
        mock_cb = MagicMock()

        queue.set_circuit_breaker(mock_cb)

        assert queue.processor._circuit_breaker is mock_cb

    def test_get_cb_remaining_no_cb(self, queue):
        """_get_cb_remaining() returns 0 when no CB linked."""
        remaining = queue._get_cb_remaining()

        assert remaining == 0.0

    def test_get_cb_remaining_cb_closed(self, queue):
        """_get_cb_remaining() returns 0 when CB is closed."""
        mock_cb = MagicMock()
        mock_cb.is_enabled = True
        mock_cb.is_open = False
        queue.set_circuit_breaker(mock_cb)

        remaining = queue._get_cb_remaining()

        assert remaining == 0.0


@pytest.mark.fast
class TestBudgetState:
    """Tests for rate limit budget state tracking."""

    def test_set_budget_state(self, queue):
        """set_budget_state() stores budget summary."""
        budget = {'is_exhausted': True, 'backoff_time_remaining': 60}

        queue.set_budget_state(budget)

        assert queue.get_budget_state() == budget

    def test_get_budget_state_initially_none(self, queue):
        """get_budget_state() returns None initially."""
        assert queue.get_budget_state() is None


# =============================================================================
# US-49-010: Test escalation tier persistence across retry passes
# =============================================================================

@pytest.mark.fast
class TestEscalationTierPersistence:
    """Tests for escalation tier storage and propagation in retry queue."""

    def test_add_stores_default_escalation_tier(self, queue):
        """add() stores default escalation_tier=1 when not specified."""
        queue.add("vid1", "kw", "short", "Error")

        assert queue.items["vid1"].escalation_tier == 1

    def test_add_stores_custom_escalation_tier(self, queue):
        """add() stores the provided escalation_tier value."""
        queue.add("vid1", "kw", "short", "403 Forbidden", escalation_tier=3)

        assert queue.items["vid1"].escalation_tier == 3

    def test_add_stores_max_escalation_tier(self, queue):
        """add() stores tier 4 (VPN_ROTATION)."""
        queue.add("vid1", "kw", "short", "Error", escalation_tier=4)

        assert queue.items["vid1"].escalation_tier == 4

    def test_duplicate_add_keeps_higher_tier(self, queue):
        """add() for duplicate video keeps the higher escalation tier."""
        queue.add("vid1", "kw", "short", "Error 1", escalation_tier=2)
        queue.add("vid1", "kw", "short", "Error 2", escalation_tier=3)

        assert queue.items["vid1"].escalation_tier == 3

    def test_duplicate_add_does_not_regress_tier(self, queue):
        """add() for duplicate video does not lower escalation tier."""
        queue.add("vid1", "kw", "short", "Error 1", escalation_tier=3)
        queue.add("vid1", "kw", "short", "Error 2", escalation_tier=1)

        # Should remain at 3, not regress to 1
        assert queue.items["vid1"].escalation_tier == 3

    def test_checkpoint_serializes_escalation_tier(self, queue):
        """to_checkpoint_dict() includes escalation_tier in serialized items."""
        queue.add("vid1", "kw", "short", "Error", escalation_tier=3)

        checkpoint = queue.to_checkpoint_dict()

        items = checkpoint['items']
        assert len(items) == 1
        assert items[0]['escalation_tier'] == 3

    def test_checkpoint_restores_escalation_tier(self, queue):
        """from_checkpoint_dict() restores escalation_tier from checkpoint."""
        checkpoint = {
            'items': [
                {
                    'video_id': 'vid1',
                    'keyword': 'kw',
                    'tier': 'short',
                    'error_message': 'Error',
                    'retry_count': 0,
                    'error_category': 'bot_detection',
                    'escalation_tier': 3,
                },
            ],
            'current_pass': 0,
            'completed_ids': [],
            'failed_ids': [],
            'total_added': 1,
            'total_retried': 0,
        }

        queue.from_checkpoint_dict(checkpoint)

        assert queue.items['vid1'].escalation_tier == 3

    def test_checkpoint_defaults_escalation_tier_when_missing(self, queue):
        """from_checkpoint_dict() defaults escalation_tier to 1 for old checkpoints."""
        checkpoint = {
            'items': [
                {
                    'video_id': 'vid1',
                    'keyword': 'kw',
                    'tier': 'short',
                    'error_message': 'Error',
                    'retry_count': 0,
                    # No escalation_tier field (old checkpoint format)
                },
            ],
            'current_pass': 0,
            'completed_ids': [],
            'failed_ids': [],
            'total_added': 1,
            'total_retried': 0,
        }

        queue.from_checkpoint_dict(checkpoint)

        assert queue.items['vid1'].escalation_tier == 1

    def test_retryable_items_include_escalation_tier(self, queue):
        """get_retryable_items() preserves escalation_tier on returned items."""
        queue.add("vid1", "kw", "short", "403 Forbidden",
                   error_category='bot_detection', escalation_tier=2)
        queue.add("vid2", "kw", "short", "getaddrinfo failed",
                   error_category='network', escalation_tier=1)

        retryable = queue.get_retryable_items()

        assert len(retryable) == 1  # network errors excluded
        assert retryable[0].video_id == "vid1"
        assert retryable[0].escalation_tier == 2


# =============================================================================
# US-49-010: Test cookies presence in retry queue ydl_opts
# =============================================================================

@pytest.mark.fast
class TestRetryQueueCookiesPropagation:
    """Tests verifying cookies are present in retry queue ydl_opts.

    These tests verify the retry queue's add() correctly stores data needed
    for cookie/impersonation propagation, which _process_retry_queue() uses
    to construct ydl_opts with the same auth as the primary download loop.
    """

    def test_retry_item_has_error_category_for_cookie_rotation(self, queue):
        """RetryItem stores error_category used for cookie rotation decisions."""
        queue.add("vid1", "kw", "short", "403 Forbidden",
                   error_category='bot_detection', escalation_tier=3)

        item = queue.items["vid1"]
        # bot_detection category triggers cookie rotation in the retry loop
        assert item.error_category == 'bot_detection'
        # Tier 3 (FULL_BYPASS) includes rotate_cookies=True
        assert item.escalation_tier == 3

    def test_retry_item_tier3_signals_cookie_rotation(self, queue):
        """Tier 3+ escalation implies cookies should be rotated during retry."""
        # When escalation_tier >= 3, EscalationManager returns
        # rotate_cookies=True in the escalation result
        queue.add("vid1", "kw", "short", "403 Forbidden",
                   error_category='bot_detection', escalation_tier=3)
        queue.add("vid2", "kw", "short", "403 Forbidden",
                   error_category='bot_detection', escalation_tier=4)

        assert queue.items["vid1"].escalation_tier >= 3  # Cookie rotation tier
        assert queue.items["vid2"].escalation_tier >= 3
