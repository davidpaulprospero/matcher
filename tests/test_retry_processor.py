"""Tests for RetryQueueProcessor class (US-35-004).

Verifies the extracted processing logic in RetryQueueProcessor:
- RetryQueueProcessor handles delay calculation, circuit breaker wait, cookie cooldown
- RetryQueue delegates to processor for execution logic
- Processor respects circuit breaker state before processing
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.downloader.retry_queue import RetryQueue, BatchRetryConfig
from src.downloader.retry_processor import RetryQueueProcessor
from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


class TestRetryQueueProcessorInit:
    """Tests for RetryQueueProcessor initialization."""

    @pytest.mark.fast
    def test_processor_created_with_queue(self):
        """Processor should be created with a RetryQueue instance."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        assert processor.queue is queue

    @pytest.mark.fast
    def test_processor_has_config_from_queue(self):
        """Processor should have access to queue's config."""
        config = BatchRetryConfig(delay_seconds=60.0, max_passes=3)
        queue = RetryQueue(config)
        processor = RetryQueueProcessor(queue)
        assert processor.config.delay_seconds == 60.0
        assert processor.config.max_passes == 3

    @pytest.mark.fast
    def test_processor_initial_state(self):
        """Processor should initialize with default state."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        assert processor.circuit_breaker_wait_time == 0.0
        assert processor.cookie_cooldown_wait_time == 0.0
        assert processor.forced_retry is False


class TestRetryQueueProcessorCircuitBreaker:
    """Tests for processor respecting circuit breaker state."""

    @pytest.mark.fast
    def test_set_circuit_breaker_links_instance(self):
        """set_circuit_breaker should store circuit breaker reference."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        cb = CircuitBreaker()

        processor.set_circuit_breaker(cb)

        assert processor._circuit_breaker is cb

    @pytest.mark.fast
    def test_get_cb_remaining_returns_zero_when_closed(self):
        """Should return 0 when circuit breaker is not tripped."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        cb = CircuitBreaker()
        cb.state.is_open = False
        processor.set_circuit_breaker(cb)

        assert processor._get_cb_remaining() == 0.0

    @pytest.mark.fast
    def test_get_cb_remaining_returns_time_when_open(self):
        """Should return remaining time when circuit breaker is open."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=60.0))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        processor.set_circuit_breaker(cb)

        remaining = processor._get_cb_remaining()
        # Should be close to 60 seconds (minus any elapsed time)
        assert 55 < remaining <= 60

    @pytest.mark.fast
    def test_wait_for_circuit_breaker_when_open(self):
        """Processor should wait for circuit breaker to recover."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.1))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        processor.set_circuit_breaker(cb)

        with patch('time.sleep') as mock_sleep:
            wait_time = processor._wait_for_circuit_breaker()

            mock_sleep.assert_called_once()
            assert wait_time > 0
            assert processor.circuit_breaker_wait_time > 0

    @pytest.mark.fast
    def test_start_retry_pass_respects_circuit_breaker(self):
        """start_retry_pass should wait for circuit breaker before processing."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))
        queue.add('video1', 'keyword', 'short', 'error')
        processor = RetryQueueProcessor(queue)

        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.05))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        processor.set_circuit_breaker(cb)

        with patch('time.sleep') as mock_sleep:
            processor.start_retry_pass()

            # Should have called sleep for both CB wait and delay
            assert mock_sleep.call_count == 2


class TestRetryQueueProcessorJitter:
    """Tests for jitter calculation in processor."""

    @pytest.mark.fast
    def test_apply_jitter_returns_delay_within_bounds(self):
        """Jittered delay should be within expected bounds."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=100.0, jitter_factor=0.2))
        processor = RetryQueueProcessor(queue)

        # Run multiple times to test randomness
        for _ in range(10):
            jittered = processor._apply_jitter(100.0)
            # With 20% jitter, should be between 80 and 120
            assert 80 <= jittered <= 120

    @pytest.mark.fast
    def test_apply_jitter_zero_factor_deterministic(self):
        """Zero jitter factor should return exact delay."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=100.0, jitter_factor=0.0))
        processor = RetryQueueProcessor(queue)

        jittered = processor._apply_jitter(100.0)
        assert jittered == 100.0

    @pytest.mark.fast
    def test_apply_jitter_capped_at_max(self):
        """Jittered delay should be capped at max_combined_wait_seconds."""
        queue = RetryQueue(BatchRetryConfig(
            delay_seconds=1000.0,
            jitter_factor=0.5,
            max_combined_wait_seconds=300.0
        ))
        processor = RetryQueueProcessor(queue)

        jittered = processor._apply_jitter(1000.0)
        assert jittered <= 300.0


class TestRetryQueueProcessorCookieCooldown:
    """Tests for cookie cooldown coordination."""

    @pytest.mark.fast
    def test_set_cookie_rotator_links_instance(self):
        """set_cookie_rotator should store cookie rotator reference."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        rotator = MagicMock()

        processor.set_cookie_rotator(rotator)

        assert processor._cookie_rotator is rotator

    @pytest.mark.fast
    def test_get_cookie_cooldown_remaining_zero_when_no_rotator(self):
        """Should return 0 when no cookie rotator is linked."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)

        assert processor._get_cookie_cooldown_remaining() == 0.0


class TestRetryQueueProcessorBudgetState:
    """Tests for budget state tracking."""

    @pytest.mark.fast
    def test_set_budget_state(self):
        """set_budget_state should store budget summary."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        budget_summary = {'is_exhausted': True, 'backoff_time_remaining': 30.0}

        processor.set_budget_state(budget_summary)

        assert processor.get_budget_state() == budget_summary

    @pytest.mark.fast
    def test_budget_state_none_by_default(self):
        """Budget state should be None by default."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)

        assert processor.get_budget_state() is None


class TestRetryQueueProcessorCheckpoint:
    """Tests for checkpoint persistence."""

    @pytest.mark.fast
    def test_to_checkpoint_dict(self):
        """to_checkpoint_dict should serialize processor state."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        processor._circuit_breaker_wait_time = 30.0
        processor._cookie_cooldown_wait_time = 15.0

        data = processor.to_checkpoint_dict()

        assert data['circuit_breaker_wait_time'] == 30.0
        assert data['cookie_cooldown_wait_time'] == 15.0

    @pytest.mark.fast
    def test_from_checkpoint_dict(self):
        """from_checkpoint_dict should restore processor state."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        data = {
            'circuit_breaker_wait_time': 25.0,
            'cookie_cooldown_wait_time': 10.0,
        }

        processor.from_checkpoint_dict(data)

        assert processor.circuit_breaker_wait_time == 25.0
        assert processor.cookie_cooldown_wait_time == 10.0

    @pytest.mark.fast
    def test_checkpoint_roundtrip(self):
        """Checkpoint roundtrip should preserve state."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        processor._circuit_breaker_wait_time = 42.5
        processor._cookie_cooldown_wait_time = 17.3

        data = processor.to_checkpoint_dict()

        processor2 = RetryQueueProcessor(RetryQueue())
        processor2.from_checkpoint_dict(data)

        assert processor2.circuit_breaker_wait_time == 42.5
        assert processor2.cookie_cooldown_wait_time == 17.3


class TestRetryQueueProcessorClear:
    """Tests for clear method."""

    @pytest.mark.fast
    def test_clear_resets_wait_times(self):
        """clear should reset wait times to zero."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        processor._circuit_breaker_wait_time = 60.0
        processor._cookie_cooldown_wait_time = 30.0
        processor._forced_retry = True

        processor.clear()

        assert processor.circuit_breaker_wait_time == 0.0
        assert processor.cookie_cooldown_wait_time == 0.0
        assert processor.forced_retry is False


class TestRetryQueueDelegation:
    """Tests for RetryQueue delegating to processor."""

    @pytest.mark.fast
    def test_queue_creates_processor_lazily(self):
        """RetryQueue should create processor on first access."""
        queue = RetryQueue()
        # Initially no processor
        assert queue._processor is None

        # Accessing processor property creates it
        processor = queue.processor
        assert processor is not None
        assert queue._processor is processor

    @pytest.mark.fast
    def test_queue_set_circuit_breaker_delegates_to_processor(self):
        """set_circuit_breaker should delegate to processor."""
        queue = RetryQueue()
        cb = CircuitBreaker()

        queue.set_circuit_breaker(cb)

        assert queue.processor._circuit_breaker is cb
        # Backward compat property should also work
        assert queue._circuit_breaker is cb

    @pytest.mark.fast
    def test_queue_start_retry_pass_delegates_to_processor(self):
        """start_retry_pass should delegate to processor."""
        queue = RetryQueue(BatchRetryConfig(delay_seconds=0.01))
        queue.add('video1', 'keyword', 'short', 'error')

        with patch('time.sleep'):
            pass_num = queue.start_retry_pass()

        assert pass_num == 1
        assert queue.current_pass == 1

    @pytest.mark.fast
    def test_queue_wait_combined_delegates_to_processor(self):
        """_wait_combined should delegate to processor."""
        queue = RetryQueue()
        cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=0.05))
        cb.state.is_open = True
        cb.state.opened_at = time.time()
        queue.set_circuit_breaker(cb)

        with patch('time.sleep'):
            wait_time = queue._wait_combined()

        assert wait_time > 0


class TestRetryQueueProcessorStats:
    """Tests for processor stats."""

    @pytest.mark.fast
    def test_get_processor_stats(self):
        """get_processor_stats should return processor-specific stats."""
        queue = RetryQueue()
        processor = RetryQueueProcessor(queue)
        processor._circuit_breaker_wait_time = 30.0
        processor._cookie_cooldown_wait_time = 15.0
        processor._forced_retry = True

        stats = processor.get_processor_stats()

        assert stats['circuit_breaker_wait_time'] == 30.0
        assert stats['cookie_cooldown_wait_time'] == 15.0
        assert stats['forced_retry'] is True
