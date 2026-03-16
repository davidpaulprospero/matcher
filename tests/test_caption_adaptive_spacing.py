"""Tests for caption adaptive request spacing.

Tests the AdaptiveSpacingController class that adjusts delay between
caption fetch requests based on error rate patterns.

Part of US-33-004: Add adaptive request spacing to caption batch processing.
"""

import pytest
import time
from unittest.mock import patch

from src.caption.adaptive_spacing import (
    AdaptiveSpacingController,
    AdaptiveSpacingConfig,
    RequestResult,
)


class TestAdaptiveSpacingController:
    """Tests for AdaptiveSpacingController class."""

    def test_init_with_defaults(self):
        """Controller initializes with default values."""
        controller = AdaptiveSpacingController()

        assert controller.config.min_interval_ms == 500
        assert controller.config.max_interval_ms == 5000
        assert controller.current_interval_ms == 500

    def test_init_with_custom_values(self):
        """Controller initializes with custom min/max values."""
        controller = AdaptiveSpacingController(
            min_interval_ms=100,
            max_interval_ms=2000
        )

        assert controller.config.min_interval_ms == 100
        assert controller.config.max_interval_ms == 2000
        assert controller.current_interval_ms == 100

    def test_init_with_config(self):
        """Controller initializes with full config object."""
        config = AdaptiveSpacingConfig(
            min_interval_ms=250,
            max_interval_ms=3000,
            error_threshold_high=0.3,
            error_threshold_low=0.1
        )
        controller = AdaptiveSpacingController(config=config)

        assert controller.config.min_interval_ms == 250
        assert controller.config.max_interval_ms == 3000
        assert controller.config.error_threshold_high == 0.3
        assert controller.config.error_threshold_low == 0.1

    def test_get_delay_seconds_first_request(self):
        """First request has no delay."""
        controller = AdaptiveSpacingController(min_interval_ms=500)
        delay = controller.get_delay_seconds()

        assert delay == 0.0

    def test_get_delay_seconds_after_request(self):
        """Delay calculated correctly after recording request start."""
        controller = AdaptiveSpacingController(min_interval_ms=500)

        controller.record_request_start()
        delay = controller.get_delay_seconds()

        # Should be close to 0.5 seconds (500ms)
        assert 0.4 <= delay <= 0.5

    def test_record_result_success(self):
        """Successful results are tracked correctly."""
        controller = AdaptiveSpacingController()

        controller.record_result(success=True)
        controller.record_result(success=True)

        stats = controller.get_stats()
        assert stats['total_requests'] == 2
        assert stats['error_rate'] == 0.0
        assert stats['consecutive_successes'] == 2

    def test_record_result_failure_resets_consecutive(self):
        """Failed result resets consecutive success counter."""
        controller = AdaptiveSpacingController()

        controller.record_result(success=True)
        controller.record_result(success=True)
        controller.record_result(success=False, error_type='rate_limit')

        stats = controller.get_stats()
        assert stats['consecutive_successes'] == 0

    def test_spacing_doubles_on_high_error_rate(self):
        """Spacing doubles when error rate exceeds 20% in last 10 requests."""
        config = AdaptiveSpacingConfig(
            min_interval_ms=500,
            max_interval_ms=5000,
            error_threshold_high=0.2,
            window_size_error=10
        )
        controller = AdaptiveSpacingController(config=config)

        # Record 7 successes and 3 failures (30% error rate > 20%)
        for _ in range(7):
            controller.record_result(success=True)
        for _ in range(3):
            controller.record_result(success=False, error_type='429')

        # Spacing should have doubled from 500 to 1000
        assert controller.current_interval_ms == 1000

    def test_spacing_doubles_multiple_times(self):
        """Spacing can double multiple times up to max."""
        config = AdaptiveSpacingConfig(
            min_interval_ms=500,
            max_interval_ms=5000,
            error_threshold_high=0.2,
            window_size_error=10
        )
        controller = AdaptiveSpacingController(config=config)

        # First high error batch: 500 -> 1000
        for _ in range(7):
            controller.record_result(success=True)
        for _ in range(3):
            controller.record_result(success=False)
        assert controller.current_interval_ms == 1000

        # Reset and trigger again to test doubling behavior
        controller.reset()
        controller._current_interval_ms = 1000  # Start from 1000

        # Second high error batch: 1000 -> 2000
        for _ in range(7):
            controller.record_result(success=True)
        for _ in range(3):
            controller.record_result(success=False)
        assert controller.current_interval_ms == 2000

    def test_spacing_capped_at_max(self):
        """Spacing cannot exceed max_interval_ms."""
        config = AdaptiveSpacingConfig(
            min_interval_ms=500,
            max_interval_ms=2000,
            error_threshold_high=0.2,
            window_size_error=10
        )
        controller = AdaptiveSpacingController(config=config)

        # Multiple high error batches
        for batch in range(5):
            for _ in range(7):
                controller.record_result(success=True)
            for _ in range(3):
                controller.record_result(success=False)

        # Should be capped at 2000
        assert controller.current_interval_ms == 2000

    def test_spacing_halves_on_low_error_rate(self):
        """Spacing halves when error rate below 5% for 20 requests."""
        config = AdaptiveSpacingConfig(
            min_interval_ms=500,
            max_interval_ms=5000,
            error_threshold_low=0.05,
            window_size_success=20
        )
        controller = AdaptiveSpacingController(config=config)

        # Start with elevated spacing
        controller._current_interval_ms = 2000

        # Record 20 successes (0% error rate < 5%)
        for _ in range(20):
            controller.record_result(success=True)

        # Spacing should have halved from 2000 to 1000
        assert controller.current_interval_ms == 1000

    def test_spacing_halves_multiple_times(self):
        """Spacing can halve multiple times down to min."""
        config = AdaptiveSpacingConfig(
            min_interval_ms=500,
            max_interval_ms=5000,
            error_threshold_low=0.05,
            window_size_success=20
        )
        controller = AdaptiveSpacingController(config=config)

        # Start with high spacing
        controller._current_interval_ms = 4000

        # First success batch: 4000 -> 2000
        for _ in range(20):
            controller.record_result(success=True)
        assert controller.current_interval_ms == 2000

        # Reset and test second halving
        controller.reset()
        controller._current_interval_ms = 2000

        # Second success batch: 2000 -> 1000
        for _ in range(20):
            controller.record_result(success=True)
        assert controller.current_interval_ms == 1000

    def test_spacing_cannot_go_below_min(self):
        """Spacing cannot go below min_interval_ms."""
        config = AdaptiveSpacingConfig(
            min_interval_ms=500,
            max_interval_ms=5000,
            error_threshold_low=0.05,
            window_size_success=20
        )
        controller = AdaptiveSpacingController(config=config)

        # Start at minimum
        controller._current_interval_ms = 500

        # Record many successes
        for _ in range(40):
            controller.record_result(success=True)

        # Should stay at minimum
        assert controller.current_interval_ms == 500

    def test_get_error_rate(self):
        """Error rate calculated correctly."""
        controller = AdaptiveSpacingController()

        for _ in range(8):
            controller.record_result(success=True)
        for _ in range(2):
            controller.record_result(success=False)

        # 2/10 = 0.2 = 20% error rate
        error_rate = controller.get_error_rate(window_size=10)
        assert error_rate == 0.2

    def test_get_error_rate_empty(self):
        """Error rate is 0 when no requests tracked."""
        controller = AdaptiveSpacingController()
        assert controller.get_error_rate() == 0.0

    def test_get_stats(self):
        """Stats dictionary contains all expected fields."""
        controller = AdaptiveSpacingController(min_interval_ms=500, max_interval_ms=5000)

        controller.record_result(success=True)
        controller.record_result(success=True)
        controller.record_result(success=False)

        stats = controller.get_stats()

        assert 'current_interval_ms' in stats
        assert 'min_interval_ms' in stats
        assert 'max_interval_ms' in stats
        assert 'total_requests' in stats
        assert 'error_rate' in stats
        assert 'consecutive_successes' in stats

        assert stats['total_requests'] == 3
        assert stats['min_interval_ms'] == 500
        assert stats['max_interval_ms'] == 5000

    def test_reset(self):
        """Reset returns controller to initial state."""
        controller = AdaptiveSpacingController(min_interval_ms=500)

        # Make some changes
        controller._current_interval_ms = 2000
        controller.record_result(success=True)
        controller.record_result(success=True)

        # Reset
        controller.reset()

        assert controller.current_interval_ms == 500
        assert controller.get_stats()['total_requests'] == 0
        assert controller.get_stats()['consecutive_successes'] == 0


class TestIntegrationSimulatedRateLimiting:
    """Integration tests with simulated rate limiting scenarios."""

    def test_spacing_increases_during_rate_limit_burst(self):
        """Spacing increases during simulated rate limit burst."""
        config = AdaptiveSpacingConfig(
            min_interval_ms=100,
            max_interval_ms=1000,
            error_threshold_high=0.2,
            window_size_error=10
        )
        controller = AdaptiveSpacingController(config=config)

        initial_spacing = controller.current_interval_ms

        # Simulate initial successful requests
        for _ in range(5):
            controller.record_result(success=True)

        # Simulate rate limit burst (5 failures in a row)
        for _ in range(5):
            controller.record_result(success=False, error_type='429')

        # Spacing should have increased
        assert controller.current_interval_ms > initial_spacing

    def test_spacing_recovers_after_rate_limit_clears(self):
        """Spacing recovers after rate limit clears."""
        config = AdaptiveSpacingConfig(
            min_interval_ms=100,
            max_interval_ms=2000,
            error_threshold_high=0.2,
            error_threshold_low=0.05,
            window_size_error=10,
            window_size_success=20
        )
        controller = AdaptiveSpacingController(config=config)

        # Set elevated spacing manually (simulating after rate limit)
        controller._current_interval_ms = 400
        elevated_spacing = controller.current_interval_ms

        # Simulate rate limit clearing with sustained success (only successes)
        for _ in range(20):
            controller.record_result(success=True)

        # Spacing should have decreased
        assert controller.current_interval_ms < elevated_spacing

    def test_full_cycle_increase_and_decrease(self):
        """Full cycle of spacing increase followed by decrease."""
        config = AdaptiveSpacingConfig(
            min_interval_ms=100,
            max_interval_ms=800,
            error_threshold_high=0.2,
            error_threshold_low=0.05,
            window_size_error=10,
            window_size_success=20
        )
        controller = AdaptiveSpacingController(config=config)

        # Phase 1: Normal operation
        assert controller.current_interval_ms == 100

        # Phase 2: Rate limiting hits (30% error rate in window)
        for i in range(10):
            if i < 7:
                controller.record_result(success=True)
            else:
                controller.record_result(success=False, error_type='rate_limit')

        # Should have doubled
        assert controller.current_interval_ms == 200

        # Phase 3: Reset for clean test of doubling from 200
        controller.reset()
        controller._current_interval_ms = 200

        for i in range(10):
            if i < 7:
                controller.record_result(success=True)
            else:
                controller.record_result(success=False, error_type='rate_limit')

        # Should have doubled again
        assert controller.current_interval_ms == 400

        # Phase 4: Reset for clean test of recovery
        controller.reset()
        controller._current_interval_ms = 400

        # Recovery (all successes)
        for _ in range(20):
            controller.record_result(success=True)

        # Should have halved
        assert controller.current_interval_ms == 200

        # Phase 5: Reset for second recovery test
        controller.reset()
        controller._current_interval_ms = 200

        for _ in range(20):
            controller.record_result(success=True)

        # Should have halved again
        assert controller.current_interval_ms == 100
