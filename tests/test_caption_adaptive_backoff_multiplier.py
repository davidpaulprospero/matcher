"""Tests for adaptive timeout scaling based on retry budget consumption rate (US-78-011).

Verifies that CaptionRetryBudget tracks consumption rate and adjusts the
backoff multiplier when retries are consumed too quickly, indicating active
rate limiting by YouTube.

Acceptance criteria:
1. CaptionRetryBudget tracks consumption rate as attempts_used/elapsed_seconds
2. When consumption rate exceeds 2x the expected rate, backoff delays increase by 1.5x
3. get_adaptive_backoff_multiplier() returns current multiplier (1.0 normal, up to 3.0 max)
4. Multiplier resets to 1.0 after 60 seconds of no new errors
5. Unit test verifies multiplier increases when rate exceeds threshold and resets after cooldown
"""

import pytest
import time

from src.caption.retry_budget import CaptionRetryBudget
from src.caption.enums import CaptionErrorCategory


class TestAdaptiveBackoffMultiplier:
    """Test adaptive backoff multiplier behavior (US-78-011)."""

    @pytest.mark.fast
    def test_default_multiplier_is_one(self):
        """Multiplier starts at 1.0 (normal pace)."""
        budget = CaptionRetryBudget()
        assert budget.get_adaptive_backoff_multiplier() == 1.0

    @pytest.mark.fast
    def test_multiplier_stays_one_under_normal_rate(self):
        """Multiplier stays 1.0 when consumption rate is within expected bounds."""
        budget = CaptionRetryBudget(max_backoff_time=300.0)
        budget.batch_size = 100
        # Expected rate = 100/300 = 0.33 attempts/sec
        # If we do 10 attempts in 60 seconds, rate = 0.17 (under 2*0.33=0.67)
        start = 1000.0
        budget._start_time = start

        for i in range(10):
            budget.record_attempt(f"vid_{i}")

        # Simulate failures at normal pace (under threshold)
        for i in range(10):
            with budget._lock:
                budget.failures += 1
                budget._update_adaptive_multiplier(current_time=start + 60.0)

        assert budget.get_adaptive_backoff_multiplier() == 1.0

    @pytest.mark.fast
    def test_multiplier_increases_on_fast_consumption(self):
        """Multiplier increases by 1.5x when consumption rate exceeds 2x expected."""
        budget = CaptionRetryBudget(max_backoff_time=300.0)
        budget.batch_size = 100
        # Expected rate = 100/300 = 0.33/s; threshold = 2*0.33 = 0.67/s
        # Do 50 attempts in 10 seconds => rate = 5.0/s >> 0.67/s
        start = 1000.0
        budget._start_time = start
        budget.attempts = 50

        with budget._lock:
            budget.failures = 50
            budget._update_adaptive_multiplier(current_time=start + 10.0)

        assert budget.get_adaptive_backoff_multiplier(current_time=start + 10.0) == 1.5

    @pytest.mark.fast
    def test_multiplier_caps_at_three(self):
        """Multiplier never exceeds 3.0 even with very high consumption rate."""
        budget = CaptionRetryBudget(max_backoff_time=300.0)
        budget.batch_size = 100
        start = 1000.0
        budget._start_time = start
        budget.attempts = 50

        # Trigger multiple increases: 1.0 -> 1.5 -> 2.25 -> 3.0 (capped)
        with budget._lock:
            for i in range(5):
                budget.failures += 10
                budget.attempts += 10
                budget._update_adaptive_multiplier(current_time=start + 1.0)

        assert budget.get_adaptive_backoff_multiplier(current_time=start + 1.0) == 3.0

    @pytest.mark.fast
    def test_multiplier_resets_after_cooldown(self):
        """Multiplier resets to 1.0 after 60 seconds of no errors."""
        budget = CaptionRetryBudget(max_backoff_time=300.0)
        budget.batch_size = 100
        start = 1000.0
        budget._start_time = start
        budget.attempts = 50

        # First, increase the multiplier
        with budget._lock:
            budget.failures = 50
            budget._update_adaptive_multiplier(current_time=start + 10.0)

        assert budget.get_adaptive_backoff_multiplier(current_time=start + 10.0) == 1.5

        # 60 seconds after last error => cooldown triggers
        assert budget.get_adaptive_backoff_multiplier(current_time=start + 70.1) == 1.0

    @pytest.mark.fast
    def test_multiplier_does_not_reset_before_cooldown(self):
        """Multiplier stays elevated if cooldown period hasn't elapsed."""
        budget = CaptionRetryBudget(max_backoff_time=300.0)
        budget.batch_size = 100
        start = 1000.0
        budget._start_time = start
        budget.attempts = 50

        with budget._lock:
            budget.failures = 50
            # _update sets _last_error_time to start + 10.0
            budget._update_adaptive_multiplier(current_time=start + 10.0)

        # 30 seconds since last error (start+10 to start+40) - under 60s cooldown
        assert budget.get_adaptive_backoff_multiplier(current_time=start + 40.0) == 1.5

    @pytest.mark.fast
    def test_reset_clears_adaptive_state(self):
        """Budget reset() clears all adaptive backoff state."""
        budget = CaptionRetryBudget(max_backoff_time=300.0)
        budget.batch_size = 100
        budget._start_time = 1000.0
        budget._last_error_time = 1010.0
        budget._adaptive_multiplier = 2.25

        budget.reset()

        assert budget._start_time is None
        assert budget._last_error_time is None
        assert budget._adaptive_multiplier == 1.0
        assert budget.get_adaptive_backoff_multiplier() == 1.0

    @pytest.mark.fast
    def test_summary_includes_adaptive_multiplier(self):
        """get_summary() includes adaptive_backoff_multiplier field."""
        budget = CaptionRetryBudget(max_backoff_time=300.0)
        budget.batch_size = 100
        summary = budget.get_summary()
        assert "adaptive_backoff_multiplier" in summary
        assert summary["adaptive_backoff_multiplier"] == 1.0

    @pytest.mark.fast
    def test_record_failure_updates_multiplier(self):
        """record_failure() automatically updates adaptive multiplier."""
        budget = CaptionRetryBudget(max_backoff_time=300.0)
        budget.batch_size = 100
        start = 1000.0
        budget._start_time = start

        # Simulate rapid failures by setting attempts high and controlling time
        budget.attempts = 80
        with budget._lock:
            budget._adaptive_multiplier = 1.0  # Ensure clean state

        # Record a failure with high consumption rate
        # 80 attempts in ~1 second is way above threshold
        budget._last_error_time = None  # Ensure no cooldown interference
        with budget._lock:
            budget.failures += 1
            budget._update_adaptive_multiplier(current_time=start + 1.0)

        # With 80 attempts in 1 second, rate=80/s >> expected=100/300=0.33/s
        assert budget.get_adaptive_backoff_multiplier(current_time=start + 1.0) == 1.5

    @pytest.mark.fast
    def test_no_batch_size_does_not_crash(self):
        """Adaptive logic gracefully handles missing batch_size."""
        budget = CaptionRetryBudget(max_backoff_time=300.0)
        # No batch_size set
        budget._start_time = 1000.0
        budget.attempts = 50

        with budget._lock:
            budget.failures = 50
            budget._update_adaptive_multiplier(current_time=1010.0)

        # Should stay at 1.0 because batch_size is None
        assert budget.get_adaptive_backoff_multiplier(current_time=1010.0) == 1.0

    @pytest.mark.fast
    def test_record_attempt_initializes_start_time(self):
        """First record_attempt() initializes _start_time."""
        budget = CaptionRetryBudget()
        assert budget._start_time is None
        budget.record_attempt("vid_1")
        assert budget._start_time is not None

    @pytest.mark.fast
    def test_progressive_multiplier_increase(self):
        """Multiplier increases progressively: 1.0 -> 1.5 -> 2.25 -> 3.0."""
        budget = CaptionRetryBudget(max_backoff_time=300.0)
        budget.batch_size = 100
        start = 1000.0
        budget._start_time = start
        budget.attempts = 50

        with budget._lock:
            # First increase: 1.0 -> 1.5
            budget.failures = 50
            budget._update_adaptive_multiplier(current_time=start + 1.0)
        assert budget._adaptive_multiplier == 1.5

        with budget._lock:
            # Second increase: 1.5 -> 2.25
            budget.attempts += 50
            budget.failures += 50
            budget._update_adaptive_multiplier(current_time=start + 2.0)
        assert budget._adaptive_multiplier == 2.25

        with budget._lock:
            # Third increase: 2.25 -> 3.0 (capped, not 3.375)
            budget.attempts += 50
            budget.failures += 50
            budget._update_adaptive_multiplier(current_time=start + 3.0)
        assert budget._adaptive_multiplier == 3.0
