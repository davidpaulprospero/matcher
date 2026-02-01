"""Unit tests for RetryQueueProcessor (US-36-008).

Focused tests for processor logic:
- Circuit breaker state checking before processing
- Delay calculation with jitter bounds
- Cookie cooldown honoring
- Severity-based delay multipliers
- Budget exhaustion handling
"""

import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock

from src.downloader.retry_queue import RetryQueue, BatchRetryConfig, RetryItem
from src.downloader.retry_processor import RetryQueueProcessor
from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


# =============================================================================
# Test: Processor respects circuit breaker state before retrying
# =============================================================================

class TestCircuitBreakerStateRespect:
    """Tests that processor respects circuit breaker state before retrying."""

    @pytest.mark.fast
    def test_skips_processing_when_cb_open_and_respect_enabled(self):
        """Processor should wait when CB is open and respect_circuit_breaker=True."""
        config = BatchRetryConfig(delay_seconds=0.01, respect_circuit_breaker=True)
        queue = RetryQueue(config)
        queue.add('video1', 'keyword', 'short', 'rate limit')
        processor = RetryQueueProcessor(queue)

        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=60.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        processor.set_circuit_breaker(cb)

        remaining = processor._get_cb_remaining()
        assert remaining > 0, "Should report time remaining when CB is open"

    @pytest.mark.fast
    def test_proceeds_when_cb_open_but_respect_disabled(self):
        """Processor should proceed immediately when respect_circuit_breaker=False."""
        config = BatchRetryConfig(delay_seconds=0.01, respect_circuit_breaker=False)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=60.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        processor.set_circuit_breaker(cb)

        remaining = processor._get_cb_remaining()
        assert remaining == 0.0, "Should not wait when respect_circuit_breaker=False"

    @pytest.mark.fast
    def test_proceeds_when_cb_closed(self):
        """Processor should proceed immediately when CB is closed."""
        config = BatchRetryConfig(delay_seconds=0.01, respect_circuit_breaker=True)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        cb = CircuitBreaker()
        cb.state.is_open = False
        processor.set_circuit_breaker(cb)

        remaining = processor._get_cb_remaining()
        assert remaining == 0.0, "Should not wait when CB is closed"

    @pytest.mark.fast
    def test_proceeds_when_no_cb_linked(self):
        """Processor should proceed when no circuit breaker is linked."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))
        processor = RetryQueueProcessor(queue)

        remaining = processor._get_cb_remaining()
        assert remaining == 0.0, "Should not wait when no CB linked"

    @pytest.mark.fast
    def test_cb_disabled_means_no_wait(self):
        """Processor should not wait when CB exists but is disabled."""
        config = BatchRetryConfig(delay_seconds=0.01, respect_circuit_breaker=True)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        cb = CircuitBreaker(CircuitBreakerConfig(enabled=False))
        cb.state.is_open = True  # Open but disabled
        cb.state.opened_at = time.time()
        processor.set_circuit_breaker(cb)

        remaining = processor._get_cb_remaining()
        assert remaining == 0.0, "Should not wait when CB is disabled"


# =============================================================================
# Test: Delay calculation includes jitter within bounds
# =============================================================================

class TestJitterBounds:
    """Tests that delay jitter stays within expected bounds."""

    @pytest.mark.fast
    def test_jitter_within_factor_bounds(self):
        """Jittered delay should be within ±jitter_factor of base delay."""
        config = BatchRetryConfig(delay_seconds=100.0, jitter_factor=0.2)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        # Run multiple times to test randomness
        min_expected = 100.0 * (1 - 0.2)  # 80
        max_expected = 100.0 * (1 + 0.2)  # 120

        for _ in range(20):
            jittered = processor._apply_jitter(100.0)
            assert min_expected <= jittered <= max_expected, \
                f"Jitter {jittered} outside bounds [{min_expected}, {max_expected}]"

    @pytest.mark.fast
    def test_zero_jitter_returns_exact_delay(self):
        """Zero jitter factor should return exact input delay."""
        config = BatchRetryConfig(delay_seconds=100.0, jitter_factor=0.0)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        jittered = processor._apply_jitter(100.0)
        assert jittered == 100.0, "Zero jitter should return exact delay"

    @pytest.mark.fast
    def test_jitter_capped_at_max_wait(self):
        """Jittered delay should never exceed max_combined_wait_seconds."""
        config = BatchRetryConfig(
            delay_seconds=500.0,
            jitter_factor=0.5,  # Could be up to 750s
            max_combined_wait_seconds=300.0
        )
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        for _ in range(20):
            jittered = processor._apply_jitter(500.0)
            assert jittered <= 300.0, f"Jitter {jittered} exceeds max 300.0"

    @pytest.mark.fast
    def test_negative_jitter_factor_clamped_to_zero(self):
        """Negative jitter factor should be clamped to 0 (deterministic)."""
        config = BatchRetryConfig(delay_seconds=100.0, jitter_factor=-0.5)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        jittered = processor._apply_jitter(100.0)
        assert jittered == 100.0, "Negative jitter factor should clamp to 0"

    @pytest.mark.fast
    def test_jitter_factor_over_one_clamped(self):
        """Jitter factor > 1.0 should be clamped to 1.0."""
        config = BatchRetryConfig(delay_seconds=100.0, jitter_factor=2.0)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        # With clamped factor of 1.0, range is 0 to 200
        for _ in range(20):
            jittered = processor._apply_jitter(100.0)
            assert 0 <= jittered <= 200.0, \
                f"Jitter {jittered} outside clamped bounds [0, 200]"


# =============================================================================
# Test: Cookie cooldown is honored during processing
# =============================================================================

class TestCookieCooldownHonored:
    """Tests that processor honors cookie cooldown before processing."""

    @pytest.mark.fast
    def test_waits_when_all_cookies_in_cooldown(self):
        """Processor should wait when all cookies are in cooldown."""
        config = BatchRetryConfig(delay_seconds=0.01, wait_for_cookie_cooldown=True)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        # Mock cookie rotator with all cookies in cooldown
        rotator = MagicMock()
        rotator.is_enabled = True
        rotator.available_cookies = 0
        rotator._failed_cookies = {'/path/cookie1.txt': time.time()}
        rotator.config.cooldown_seconds = 60.0
        processor.set_cookie_rotator(rotator)

        remaining = processor._get_cookie_cooldown_remaining()
        assert remaining > 0, "Should have remaining cooldown time"

    @pytest.mark.fast
    def test_proceeds_when_cookies_available(self):
        """Processor should proceed when cookies are available."""
        config = BatchRetryConfig(delay_seconds=0.01, wait_for_cookie_cooldown=True)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        rotator = MagicMock()
        rotator.is_enabled = True
        rotator.available_cookies = 2
        processor.set_cookie_rotator(rotator)

        remaining = processor._get_cookie_cooldown_remaining()
        assert remaining == 0.0, "Should not wait when cookies available"

    @pytest.mark.fast
    def test_proceeds_when_cooldown_wait_disabled(self):
        """Processor should proceed when wait_for_cookie_cooldown=False."""
        config = BatchRetryConfig(delay_seconds=0.01, wait_for_cookie_cooldown=False)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        rotator = MagicMock()
        rotator.is_enabled = True
        rotator.available_cookies = 0  # No cookies available
        processor.set_cookie_rotator(rotator)

        remaining = processor._get_cookie_cooldown_remaining()
        assert remaining == 0.0, "Should not wait when cooldown wait disabled"

    @pytest.mark.fast
    def test_proceeds_when_rotator_disabled(self):
        """Processor should proceed when cookie rotator is disabled."""
        config = BatchRetryConfig(delay_seconds=0.01, wait_for_cookie_cooldown=True)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)

        rotator = MagicMock()
        rotator.is_enabled = False
        processor.set_cookie_rotator(rotator)

        remaining = processor._get_cookie_cooldown_remaining()
        assert remaining == 0.0, "Should not wait when rotator disabled"


# =============================================================================
# Test: Severity-based delay multipliers are applied correctly
# =============================================================================

class TestSeverityMultipliers:
    """Tests that severity-based delay multipliers are applied correctly."""

    @pytest.mark.fast
    def test_low_severity_multiplier_is_1_5(self):
        """Low severity items should use 1.5x multiplier."""
        queue = RetryQueue()
        queue.items['video1'] = RetryItem(
            video_id='video1', keyword='kw', tier='short',
            error_message='error', severity='low'
        )
        processor = RetryQueueProcessor(queue)

        assert processor.get_max_severity() == 'low'
        assert processor.get_severity_multiplier() == 1.5

    @pytest.mark.fast
    def test_medium_severity_multiplier_is_2_0(self):
        """Medium severity items should use 2.0x multiplier."""
        queue = RetryQueue()
        queue.items['video1'] = RetryItem(
            video_id='video1', keyword='kw', tier='short',
            error_message='error', severity='medium'
        )
        processor = RetryQueueProcessor(queue)

        assert processor.get_max_severity() == 'medium'
        assert processor.get_severity_multiplier() == 2.0

    @pytest.mark.fast
    def test_high_severity_multiplier_is_3_0(self):
        """High severity items should use 3.0x multiplier."""
        queue = RetryQueue()
        queue.items['video1'] = RetryItem(
            video_id='video1', keyword='kw', tier='short',
            error_message='error', severity='high'
        )
        processor = RetryQueueProcessor(queue)

        assert processor.get_max_severity() == 'high'
        assert processor.get_severity_multiplier() == 3.0

    @pytest.mark.fast
    def test_max_severity_from_multiple_items(self):
        """Should use the highest severity from all items in queue."""
        queue = RetryQueue()
        queue.items['video1'] = RetryItem(
            video_id='video1', keyword='kw', tier='short',
            error_message='error', severity='low'
        )
        queue.items['video2'] = RetryItem(
            video_id='video2', keyword='kw', tier='short',
            error_message='error', severity='high'
        )
        queue.items['video3'] = RetryItem(
            video_id='video3', keyword='kw', tier='short',
            error_message='error', severity='medium'
        )
        processor = RetryQueueProcessor(queue)

        assert processor.get_max_severity() == 'high'
        assert processor.get_severity_multiplier() == 3.0

    @pytest.mark.fast
    def test_empty_queue_defaults_to_medium(self):
        """Empty queue should default to medium severity."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)

        assert processor.get_max_severity() == 'medium'
        assert processor.get_severity_multiplier() == 2.0


# =============================================================================
# Test: Processing stops when budget exhausted
# =============================================================================

class TestBudgetExhaustion:
    """Tests that processor respects budget exhaustion state."""

    @pytest.mark.fast
    def test_budget_state_stored_correctly(self):
        """set_budget_state should store budget summary."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)

        budget = {'is_exhausted': True, 'backoff_time_remaining': 45.0}
        processor.set_budget_state(budget)

        assert processor.get_budget_state() == budget

    @pytest.mark.fast
    def test_budget_state_tracks_exhaustion(self):
        """Budget state should indicate when budget is exhausted."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)

        # Exhausted budget
        exhausted = {'is_exhausted': True, 'backoff_time_remaining': 30.0}
        processor.set_budget_state(exhausted)
        state = processor.get_budget_state()
        assert state['is_exhausted'] is True

        # Non-exhausted budget
        available = {'is_exhausted': False, 'backoff_time_remaining': 0.0}
        processor.set_budget_state(available)
        state = processor.get_budget_state()
        assert state['is_exhausted'] is False

    @pytest.mark.fast
    def test_budget_state_in_processor_stats(self):
        """get_processor_stats should include budget state."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)

        budget = {'is_exhausted': True, 'remaining_downloads': 0}
        processor.set_budget_state(budget)

        stats = processor.get_processor_stats()
        assert stats['budget_state'] == budget

    @pytest.mark.fast
    def test_budget_state_cleared_on_clear(self):
        """clear() should reset budget state to None."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)

        processor.set_budget_state({'is_exhausted': True})
        assert processor.get_budget_state() is not None

        processor.clear()
        assert processor.get_budget_state() is None


# =============================================================================
# Additional focused unit tests
# =============================================================================

class TestProcessorCombinedWaitStrategy:
    """Tests for combined wait strategy when CB and cooldown overlap."""

    @pytest.mark.fast
    def test_combined_wait_forces_retry_on_timeout(self):
        """Should force retry when combined wait exceeds max."""
        config = BatchRetryConfig(
            delay_seconds=0.01,
            max_combined_wait_seconds=10.0  # Low threshold
        )
        queue = RetryQueue(config)
        queue.add('video1', 'kw', 'short', 'error')
        processor = RetryQueueProcessor(queue)

        # Set up both blockers with long wait times
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=120.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        processor.set_circuit_breaker(cb)

        rotator = MagicMock()
        rotator.is_enabled = True
        rotator.available_cookies = 0
        rotator._failed_cookies = {'/path/c.txt': time.time()}
        rotator.config.cooldown_seconds = 120.0
        processor.set_cookie_rotator(rotator)

        # Combined wait would exceed max, should force
        wait_time = processor._wait_combined()
        assert wait_time == 0.0, "Should not wait when forcing"
        assert processor.forced_retry is True, "Should set forced_retry flag"

    @pytest.mark.fast
    def test_combined_wait_no_force_when_within_limit(self):
        """Should wait normally when combined wait is within limit."""
        config = BatchRetryConfig(
            delay_seconds=0.01,
            max_combined_wait_seconds=300.0
        )
        queue = RetryQueue(config)
        queue.add('video1', 'kw', 'short', 'error')
        processor = RetryQueueProcessor(queue)

        # Set up both blockers with short wait times
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        processor.set_circuit_breaker(cb)

        rotator = MagicMock()
        rotator.is_enabled = True
        rotator.available_cookies = 0
        rotator._failed_cookies = {'/path/c.txt': time.time()}
        rotator.config.cooldown_seconds = 0.1
        processor.set_cookie_rotator(rotator)

        with patch('time.sleep'):
            wait_time = processor._wait_combined()

        assert wait_time > 0, "Should wait when within limit"
        assert processor.forced_retry is False, "Should not force"


class TestProcessorHasPending:
    """Tests for has_pending delegation."""

    @pytest.mark.fast
    def test_has_pending_true_with_items(self):
        """has_pending should return True when queue has items."""
        queue = RetryQueue()
        queue.add('video1', 'kw', 'short', 'error')
        processor = RetryQueueProcessor(queue)

        assert processor.has_pending() is True

    @pytest.mark.fast
    def test_has_pending_false_when_empty(self):
        """has_pending should return False when queue is empty."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)

        assert processor.has_pending() is False
