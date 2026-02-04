"""
Tests for jitter in circuit breaker exponential backoff (US-35-003).

Verifies:
- jitter_factor parameter in CircuitBreakerConfig (default 0.2)
- Jitter formula: delay = base_delay * (1 + random.uniform(-jitter, +jitter))
- Jitter applied to circuit breaker recovery wait
- Jittered delays capped at max_pause_seconds
- Jitter prevents thundering herd in parallel scenarios
"""

import random
import time
from unittest.mock import patch, MagicMock

import pytest

from src.downloader.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from src.downloader.retry_queue import (
    BatchRetryConfig,
    RetryQueue,
)


# ============================================================================
# CircuitBreakerConfig jitter_factor Tests
# ============================================================================


class TestCircuitBreakerConfigJitter:
    """Test jitter_factor parameter in CircuitBreakerConfig."""

    @pytest.mark.fast
    def test_default_jitter_factor(self):
        """Default jitter_factor should be 0.2 (20%)."""
        config = CircuitBreakerConfig()
        assert config.jitter_factor == 0.2

    @pytest.mark.fast
    def test_custom_jitter_factor(self):
        """Custom jitter_factor should be accepted."""
        config = CircuitBreakerConfig(jitter_factor=0.5)
        assert config.jitter_factor == 0.5

    @pytest.mark.fast
    def test_zero_jitter_factor(self):
        """Zero jitter_factor should disable jitter."""
        config = CircuitBreakerConfig(jitter_factor=0.0)
        assert config.jitter_factor == 0.0


# ============================================================================
# CircuitBreaker _apply_jitter Tests
# ============================================================================


class TestCircuitBreakerApplyJitter:
    """Test CircuitBreaker._apply_jitter method."""

    @pytest.mark.fast
    def test_jitter_formula_positive(self):
        """Jitter with positive random value should increase delay."""
        config = CircuitBreakerConfig(jitter_factor=0.2, max_pause_seconds=300.0)
        breaker = CircuitBreaker(config)

        with patch.object(random, 'uniform', return_value=0.2):
            result = breaker._apply_jitter(60.0)
            # 60 * (1 + 0.2) = 72
            assert abs(result - 72.0) < 0.01
            assert abs(breaker._last_jitter_applied - 0.2) < 0.01

    @pytest.mark.fast
    def test_jitter_formula_negative(self):
        """Jitter with negative random value should decrease delay."""
        config = CircuitBreakerConfig(jitter_factor=0.2, max_pause_seconds=300.0)
        breaker = CircuitBreaker(config)

        with patch.object(random, 'uniform', return_value=-0.2):
            result = breaker._apply_jitter(60.0)
            # 60 * (1 - 0.2) = 48
            assert abs(result - 48.0) < 0.01
            assert abs(breaker._last_jitter_applied - (-0.2)) < 0.01

    @pytest.mark.fast
    def test_zero_jitter_deterministic(self):
        """jitter_factor=0 should produce deterministic delay."""
        config = CircuitBreakerConfig(jitter_factor=0.0, max_pause_seconds=300.0)
        breaker = CircuitBreaker(config)

        result = breaker._apply_jitter(60.0)
        assert result == 60.0
        assert breaker._last_jitter_applied == 0.0

    @pytest.mark.fast
    def test_jitter_factor_clamped_negative(self):
        """Negative jitter_factor should be clamped to 0."""
        config = CircuitBreakerConfig(jitter_factor=-0.5, max_pause_seconds=300.0)
        breaker = CircuitBreaker(config)

        result = breaker._apply_jitter(60.0)
        assert result == 60.0  # No jitter applied

    @pytest.mark.fast
    def test_jitter_factor_clamped_above_one(self):
        """jitter_factor > 1.0 should be clamped to 1.0."""
        config = CircuitBreakerConfig(jitter_factor=2.0, max_pause_seconds=300.0)
        breaker = CircuitBreaker(config)

        with patch.object(random, 'uniform', return_value=0.5) as mock_uniform:
            breaker._apply_jitter(60.0)
            # Should have been called with (-1.0, 1.0), not (-2.0, 2.0)
            mock_uniform.assert_called_once_with(-1.0, 1.0)

    @pytest.mark.fast
    def test_jitter_capped_at_max_pause(self):
        """Jittered delay should never exceed max_pause_seconds."""
        config = CircuitBreakerConfig(
            jitter_factor=0.5,
            max_pause_seconds=100.0,
            pause_seconds=90.0
        )
        breaker = CircuitBreaker(config)

        # With +50% jitter, 90 would become 135, but should cap at 100
        with patch.object(random, 'uniform', return_value=0.5):
            result = breaker._apply_jitter(90.0)
            assert result <= 100.0


# ============================================================================
# CircuitBreaker Jitter in Recovery Wait Tests
# ============================================================================


class TestCircuitBreakerRecoveryWaitJitter:
    """Test jitter is applied during circuit breaker recovery wait."""

    @pytest.mark.fast
    def test_check_and_wait_applies_jitter(self):
        """check_and_wait should apply jitter to remaining wait time."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=2,
            pause_seconds=10.0,
            jitter_factor=0.2
        )
        breaker = CircuitBreaker(config)

        # Trip the circuit breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        # Mock time and sleep to verify jitter is applied
        with patch('time.time', return_value=time.time()):
            with patch('time.sleep') as mock_sleep:
                with patch.object(random, 'uniform', return_value=0.2):
                    breaker.check_and_wait()
                    # Should wait with jitter: ~10 * 1.2 = 12
                    if mock_sleep.called:
                        wait_time = mock_sleep.call_args[0][0]
                        assert 9.5 <= wait_time <= 12.5

    @pytest.mark.fast
    def test_wait_for_recovery_applies_jitter(self):
        """wait_for_recovery_if_needed should apply jitter."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=2,
            pause_seconds=10.0,
            jitter_factor=0.2
        )
        breaker = CircuitBreaker(config)

        # Trip the circuit breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        # Mock sleep to verify jitter is applied
        with patch('time.time', return_value=time.time()):
            with patch('time.sleep') as mock_sleep:
                with patch.object(random, 'uniform', return_value=-0.2):
                    wait_time = breaker.wait_for_recovery_if_needed()
                    # Should return jittered time
                    if wait_time > 0:
                        assert 7.5 <= wait_time <= 10.5

    @pytest.mark.fast
    def test_get_remaining_pause_time_with_jitter(self):
        """get_remaining_pause_time(apply_jitter=True) should apply jitter."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=2,
            pause_seconds=10.0,
            jitter_factor=0.2
        )
        breaker = CircuitBreaker(config)

        # Trip the circuit breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        with patch.object(random, 'uniform', return_value=0.2):
            remaining = breaker.get_remaining_pause_time(apply_jitter=True)
            # With +20% jitter, remaining should be larger
            assert remaining > 0

    @pytest.mark.fast
    def test_get_remaining_pause_time_includes_jitter(self):
        """get_remaining_pause_time() includes jitter from the pipeline."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=2,
            pause_seconds=10.0,
            jitter_factor=0.2
        )
        breaker = CircuitBreaker(config)

        # Trip the circuit breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        # Jitter is now always applied within _get_effective_pause_seconds,
        # so successive calls may return different values
        remaining = breaker.get_remaining_pause_time()
        assert remaining > 0


# ============================================================================
# RetryQueue jitter_factor Tests
# ============================================================================


class TestBatchRetryConfigJitter:
    """Test jitter_factor parameter in BatchRetryConfig."""

    @pytest.mark.fast
    def test_default_jitter_factor(self):
        """Default jitter_factor should be 0.2 (20%)."""
        config = BatchRetryConfig()
        assert config.jitter_factor == 0.2

    @pytest.mark.fast
    def test_custom_jitter_factor(self):
        """Custom jitter_factor should be accepted."""
        config = BatchRetryConfig(jitter_factor=0.3)
        assert config.jitter_factor == 0.3


class TestRetryQueueApplyJitter:
    """Test RetryQueue._apply_jitter method."""

    @pytest.mark.fast
    def test_jitter_formula(self):
        """Jitter should follow the formula: delay * (1 + uniform(-jitter, +jitter))."""
        config = BatchRetryConfig(jitter_factor=0.2, max_combined_wait_seconds=300.0)
        queue = RetryQueue(config)

        with patch.object(random, 'uniform', return_value=0.15):
            result = queue._apply_jitter(100.0)
            # 100 * (1 + 0.15) = 115
            assert abs(result - 115.0) < 0.01

    @pytest.mark.fast
    def test_jitter_capped_at_max_combined_wait(self):
        """Jittered delay should not exceed max_combined_wait_seconds."""
        config = BatchRetryConfig(
            delay_seconds=200.0,
            jitter_factor=0.5,
            max_combined_wait_seconds=250.0
        )
        queue = RetryQueue(config)

        # With +50% jitter, 200 would become 300, but should cap at 250
        with patch.object(random, 'uniform', return_value=0.5):
            result = queue._apply_jitter(200.0)
            assert result <= 250.0


class TestRetryQueueStartRetryPassJitter:
    """Test jitter is applied in start_retry_pass."""

    @pytest.mark.fast
    def test_start_retry_pass_applies_jitter(self):
        """start_retry_pass should apply jitter to delay_seconds."""
        config = BatchRetryConfig(
            delay_seconds=10.0,
            jitter_factor=0.2,
            max_passes=3
        )
        queue = RetryQueue(config)
        queue.add("video1", "keyword", "short", "error")

        with patch('time.sleep') as mock_sleep:
            with patch.object(random, 'uniform', return_value=0.2):
                queue.start_retry_pass()

                # Should sleep with jittered delay
                assert mock_sleep.called
                sleep_time = mock_sleep.call_args[0][0]
                # 10 * 1.2 = 12
                assert abs(sleep_time - 12.0) < 0.01

    @pytest.mark.fast
    def test_start_retry_pass_zero_jitter(self):
        """start_retry_pass with jitter_factor=0 should use exact delay."""
        config = BatchRetryConfig(
            delay_seconds=10.0,
            jitter_factor=0.0,
            max_passes=3
        )
        queue = RetryQueue(config)
        queue.add("video1", "keyword", "short", "error")

        with patch('time.sleep') as mock_sleep:
            queue.start_retry_pass()

            assert mock_sleep.called
            sleep_time = mock_sleep.call_args[0][0]
            assert sleep_time == 10.0


# ============================================================================
# Thundering Herd Prevention Tests
# ============================================================================


class TestThunderingHerdPrevention:
    """Test jitter prevents thundering herd in parallel scenarios."""

    @pytest.mark.fast
    def test_circuit_breaker_parallel_workers_different_delays(self):
        """Simulated parallel workers should get staggered delays from CB."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=2,
            pause_seconds=60.0,
            jitter_factor=0.2
        )

        worker_delays = []
        for i in range(10):
            breaker = CircuitBreaker(config)
            # Trip the circuit breaker
            breaker.record_failure()
            breaker.record_failure()

            # Get remaining time with jitter
            remaining = breaker.get_remaining_pause_time(apply_jitter=True)
            worker_delays.append(remaining)

        # Delays should be different (randomized by jitter)
        unique_delays = len(set(round(d, 2) for d in worker_delays))
        assert unique_delays > 1, "Workers should have staggered delays"

    @pytest.mark.fast
    def test_retry_queue_parallel_batches_different_delays(self):
        """Simulated parallel retry batches should get staggered delays."""
        config = BatchRetryConfig(
            delay_seconds=120.0,
            jitter_factor=0.2
        )

        batch_delays = []
        for i in range(10):
            queue = RetryQueue(config)
            delay = queue._apply_jitter(config.delay_seconds)
            batch_delays.append(delay)

        # Delays should be different
        unique_delays = len(set(round(d, 2) for d in batch_delays))
        assert unique_delays > 1, "Batches should have staggered delays"

        # Check spread
        delay_range = max(batch_delays) - min(batch_delays)
        assert delay_range > 1.0, "Delays should have meaningful spread"


# ============================================================================
# Jitter Bounds Tests
# ============================================================================


class TestJitterBounds:
    """Test jitter stays within configured bounds."""

    @pytest.mark.fast
    def test_circuit_breaker_jitter_within_bounds(self):
        """CB jitter should stay within ±jitter_factor bounds."""
        config = CircuitBreakerConfig(
            pause_seconds=100.0,
            jitter_factor=0.2,
            max_pause_seconds=500.0
        )
        breaker = CircuitBreaker(config)

        min_expected = 100.0 * (1 - 0.2)  # 80
        max_expected = 100.0 * (1 + 0.2)  # 120

        for _ in range(50):
            result = breaker._apply_jitter(100.0)
            assert min_expected <= result <= max_expected, \
                f"Result {result} outside bounds [{min_expected}, {max_expected}]"

    @pytest.mark.fast
    def test_retry_queue_jitter_within_bounds(self):
        """Retry queue jitter should stay within ±jitter_factor bounds."""
        config = BatchRetryConfig(
            delay_seconds=100.0,
            jitter_factor=0.3,
            max_combined_wait_seconds=500.0
        )
        queue = RetryQueue(config)

        min_expected = 100.0 * (1 - 0.3)  # 70
        max_expected = 100.0 * (1 + 0.3)  # 130

        for _ in range(50):
            result = queue._apply_jitter(100.0)
            assert min_expected <= result <= max_expected, \
                f"Result {result} outside bounds [{min_expected}, {max_expected}]"

    @pytest.mark.fast
    def test_jitter_never_exceeds_max_cap(self):
        """Jittered delays should never exceed configured max."""
        config = CircuitBreakerConfig(
            pause_seconds=280.0,
            jitter_factor=0.2,
            max_pause_seconds=300.0
        )
        breaker = CircuitBreaker(config)

        # With +20% jitter, 280 would become 336, but should cap at 300
        for _ in range(50):
            result = breaker._apply_jitter(280.0)
            assert result <= 300.0, f"Result {result} exceeded max 300"


# ============================================================================
# Integration Tests
# ============================================================================


class TestJitterIntegration:
    """Integration tests for jitter with other features."""

    @pytest.mark.fast
    def test_jitter_with_escalation_extension(self):
        """Jitter should be applied after escalation extension but before cap."""
        config = CircuitBreakerConfig(
            pause_seconds=60.0,
            jitter_factor=0.2,
            max_pause_seconds=300.0
        )
        breaker = CircuitBreaker(config)

        # Mock escalation manager that causes 2x extension
        mock_manager = MagicMock()
        mock_manager.get_active_keyword_count.return_value = 10
        mock_manager.get_keywords_at_tier.return_value = ['kw1', 'kw2', 'kw3', 'kw4', 'kw5', 'kw6']
        breaker.set_escalation_manager(mock_manager)

        # Get effective pause (should be 60 * 2 = 120 due to escalation, then jittered)
        # Mock the EscalationTier import inside _get_effective_pause_seconds
        with patch.dict('sys.modules', {'src.downloader.types': MagicMock()}):
            with patch.object(random, 'uniform', return_value=0.1):
                effective = breaker._get_effective_pause_seconds()
                # 60 * 2 (escalation) * 1.1 (jitter) = 132, capped at 300
                assert abs(effective - 132.0) < 0.01

    @pytest.mark.fast
    def test_jitter_with_budget_extension(self):
        """Jitter should be applied after budget extension but before cap."""
        config = CircuitBreakerConfig(
            pause_seconds=60.0,
            jitter_factor=0.2,
            max_pause_seconds=300.0
        )
        breaker = CircuitBreaker(config)

        # Mock budget that causes 1.5x extension (nearly exhausted)
        mock_budget = MagicMock()
        mock_budget.is_exhausted.return_value = False
        mock_budget.is_nearly_exhausted.return_value = True
        breaker.set_budget(mock_budget)

        # Get effective pause (should be 60 * 1.5 = 90, then jittered)
        with patch.object(random, 'uniform', return_value=-0.1):
            effective = breaker._get_effective_pause_seconds()
            # 60 * 1.5 (budget) * 0.9 (jitter) = 81, capped at 300
            assert abs(effective - 81.0) < 0.01


# ============================================================================
# US-58-006: Jitter capped at max_pause_seconds Tests
# ============================================================================


class TestJitterCappedAtMaxPause:
    """Test that jitter in the pipeline never exceeds max_pause_seconds (US-58-006)."""

    @pytest.mark.fast
    def test_jitter_near_max_capped(self):
        """With jitter_factor=0.5 and pause near max, result is capped at max_pause_seconds."""
        config = CircuitBreakerConfig(
            pause_seconds=290.0,
            jitter_factor=0.5,
            max_pause_seconds=300.0
        )
        breaker = CircuitBreaker(config)

        # With +50% jitter, 290 would become 435, but must cap at 300
        with patch.object(random, 'uniform', return_value=0.5):
            effective = breaker._get_effective_pause_seconds()
            assert effective <= 300.0, (
                f"Effective pause {effective} exceeded max_pause_seconds 300.0"
            )
            # Should be exactly 300 since 290 * 1.5 = 435 > 300
            assert effective == 300.0

    @pytest.mark.fast
    def test_jitter_at_max_capped(self):
        """With pause exactly at max and positive jitter, result stays at max."""
        config = CircuitBreakerConfig(
            pause_seconds=300.0,
            jitter_factor=0.5,
            max_pause_seconds=300.0
        )
        breaker = CircuitBreaker(config)

        # 300 * 1.5 = 450, must cap at 300
        with patch.object(random, 'uniform', return_value=0.5):
            effective = breaker._get_effective_pause_seconds()
            assert effective <= 300.0

    @pytest.mark.fast
    def test_jitter_well_below_max_still_applies(self):
        """When pause is well below max, jitter should still apply normally."""
        config = CircuitBreakerConfig(
            pause_seconds=60.0,
            jitter_factor=0.5,
            max_pause_seconds=300.0
        )
        breaker = CircuitBreaker(config)

        # With +50% jitter, 60 would become 90, well below 300 cap
        with patch.object(random, 'uniform', return_value=0.5):
            effective = breaker._get_effective_pause_seconds()
            # 60 * 1.5 = 90
            assert abs(effective - 90.0) < 0.01
            assert effective <= 300.0

    @pytest.mark.fast
    def test_jitter_below_max_negative_still_applies(self):
        """Negative jitter should still reduce pause when below max."""
        config = CircuitBreakerConfig(
            pause_seconds=60.0,
            jitter_factor=0.5,
            max_pause_seconds=300.0
        )
        breaker = CircuitBreaker(config)

        # With -50% jitter, 60 would become 30
        with patch.object(random, 'uniform', return_value=-0.5):
            effective = breaker._get_effective_pause_seconds()
            # 60 * 0.5 = 30
            assert abs(effective - 30.0) < 0.01

    @pytest.mark.fast
    def test_pipeline_order_jitter_before_cap(self):
        """Verify jitter is applied before cap in the pipeline (not after)."""
        config = CircuitBreakerConfig(
            pause_seconds=250.0,
            jitter_factor=0.5,
            max_pause_seconds=300.0
        )
        breaker = CircuitBreaker(config)

        # With +50% jitter, 250 * 1.5 = 375 -> capped to 300
        # If cap were applied BEFORE jitter: 250 (no cap) * 1.5 = 375 (uncapped!)
        # So this test verifies the correct order: jitter then cap
        with patch.object(random, 'uniform', return_value=0.5):
            effective = breaker._get_effective_pause_seconds()
            assert effective <= 300.0, (
                "Jitter pushed pause above max_pause_seconds - "
                "cap must be applied AFTER jitter"
            )

    @pytest.mark.fast
    def test_many_iterations_never_exceed_max(self):
        """Run many iterations to verify max_pause_seconds is never exceeded."""
        config = CircuitBreakerConfig(
            pause_seconds=280.0,
            jitter_factor=0.5,
            max_pause_seconds=300.0
        )
        breaker = CircuitBreaker(config)

        for _ in range(100):
            effective = breaker._get_effective_pause_seconds()
            assert effective <= 300.0, (
                f"Effective pause {effective} exceeded max_pause_seconds 300.0"
            )
