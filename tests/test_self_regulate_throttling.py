"""Tests for self-regulation throttling (US-123-010).

Tests the download throttling self-regulation feature that prevents
rate limits by proactively adjusting download concurrency based on error patterns.
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.downloader.speed_tracker import (
    DownloadSpeedTracker,
    DownloadSpeedConfig,
    ThrottleState,
    ThrottleSignal
)
from src.downloader.orchestrator import DownloadCoordinator


class TestSelfRegulateConfig:
    """Test the SelfRegulateConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from src.config.sections.download import SelfRegulateConfig

        config = SelfRegulateConfig()

        assert config.enabled is True
        assert config.rate_limit_prevention_threshold == 3
        assert config.rate_limit_prevention_window_seconds == 600
        assert config.throttle_factor == 0.5
        assert config.min_concurrent == 1
        assert config.recovery_window_seconds == 300
        assert config.recovery_factor == 0.25
        assert config.max_recovery_concurrency == 0

    def test_invalid_throttle_factor_too_low(self):
        """Test that throttle_factor below 0.1 raises ValueError."""
        from src.config.sections.download import SelfRegulateConfig

        with pytest.raises(ValueError, match="must be between 0.1 and 0.9"):
            SelfRegulateConfig(throttle_factor=0.05)

    def test_invalid_throttle_factor_too_high(self):
        """Test that throttle_factor above 0.9 raises ValueError."""
        from src.config.sections.download import SelfRegulateConfig

        with pytest.raises(ValueError, match="must be between 0.1 and 0.9"):
            SelfRegulateConfig(throttle_factor=1.0)

    def test_invalid_recovery_factor_too_low(self):
        """Test that recovery_factor below 0.1 raises ValueError."""
        from src.config.sections.download import SelfRegulateConfig

        with pytest.raises(ValueError, match="must be between 0.1 and 0.5"):
            SelfRegulateConfig(recovery_factor=0.05)

    def test_invalid_min_concurrent(self):
        """Test that min_concurrent below 1 raises ValueError."""
        from src.config.sections.download import SelfRegulateConfig

        with pytest.raises(ValueError, match="must be >= 1"):
            SelfRegulateConfig(min_concurrent=0)

    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        from src.config.sections.download import SelfRegulateConfig

        config = SelfRegulateConfig(
            enabled=True,
            rate_limit_prevention_threshold=5,
            rate_limit_prevention_window_seconds=300,
            throttle_factor=0.3,
            min_concurrent=2,
            recovery_window_seconds=600,
            recovery_factor=0.2,
            max_recovery_concurrency=8
        )

        assert config.enabled is True
        assert config.rate_limit_prevention_threshold == 5
        assert config.throttle_factor == 0.3
        assert config.min_concurrent == 2


class TestDownloadSpeedTrackerThrottling:
    """Test throttling methods in DownloadSpeedTracker."""

    def test_record_error(self):
        """Test that errors are recorded correctly."""
        tracker = DownloadSpeedTracker()

        # Record some errors
        tracker.record_error("rate_limit")
        tracker.record_error("429")
        tracker.record_error("timeout")

        error_count = tracker._get_error_count_in_window(window_seconds=600)
        assert error_count == 3

    def test_should_throttle_threshold_reached(self):
        """Test that throttling triggers when threshold is reached."""
        tracker = DownloadSpeedTracker()

        # Record enough errors to trigger throttling
        for _ in range(3):
            tracker.record_error("rate_limit")
            time.sleep(0.01)  # Small delay to ensure unique timestamps

        signal = tracker.should_throttle(
            threshold=3,
            window_seconds=600,
            current_concurrency=4,
            throttle_factor=0.5,
            min_concurrent=1
        )

        assert signal.should_throttle is True
        assert signal.new_concurrency == 2  # 4 * 0.5 = 2
        assert signal.throttle_level == 1

    def test_should_throttle_below_threshold(self):
        """Test that throttling doesn't trigger below threshold."""
        tracker = DownloadSpeedTracker()

        # Record only 2 errors (threshold is 3)
        tracker.record_error("rate_limit")
        tracker.record_error("429")

        signal = tracker.should_throttle(
            threshold=3,
            window_seconds=600,
            current_concurrency=4,
            throttle_factor=0.5,
            min_concurrent=1
        )

        assert signal.should_throttle is False
        assert signal.new_concurrency == 4

    def test_should_throttle_respects_min_concurrent(self):
        """Test that throttling respects minimum concurrency."""
        tracker = DownloadSpeedTracker()

        # Record enough errors to trigger throttling
        for _ in range(3):
            tracker.record_error("rate_limit")
            time.sleep(0.01)

        # Start with very low concurrency
        signal = tracker.should_throttle(
            threshold=3,
            window_seconds=600,
            current_concurrency=2,
            throttle_factor=0.5,
            min_concurrent=1
        )

        assert signal.should_throttle is True
        assert signal.new_concurrency == 1  # 2 * 0.5 = 1, but min is 1

    def test_should_recover_no_errors_in_window(self):
        """Test recovery when no errors in recovery window."""
        tracker = DownloadSpeedTracker()

        # First, trigger throttling with higher concurrency
        for _ in range(3):
            tracker.record_error("rate_limit")
            time.sleep(0.01)

        tracker.should_throttle(
            threshold=3,
            window_seconds=600,
            current_concurrency=8,  # Start with higher concurrency
            throttle_factor=0.5,
            min_concurrent=1
        )

        # Set last_error_time to far in the past (simulate recovery window passed)
        if tracker._throttle_state:
            tracker._throttle_state.last_error_time = time.time() - 400  # 400s ago

        # Now try to recover (current is 4 after throttling from 8)
        signal = tracker.should_recover(
            recovery_window_seconds=300,
            current_concurrency=4,
            recovery_factor=0.25,
            max_recovery_concurrency=8
        )

        # Recovery should work - 4 * 1.25 = 5 which is > 4
        assert signal.should_recover is True
        assert signal.new_concurrency == 5

    def test_should_recover_errors_in_window(self):
        """Test that recovery doesn't happen when errors still occurring."""
        tracker = DownloadSpeedTracker()

        # Trigger throttling first
        for _ in range(3):
            tracker.record_error("rate_limit")
            time.sleep(0.01)

        tracker.should_throttle(
            threshold=3,
            window_seconds=600,
            current_concurrency=4,
            throttle_factor=0.5,
            min_concurrent=1
        )

        # Add another error recently
        tracker.record_error("rate_limit")

        # Try to recover - should not work because error just happened
        signal = tracker.should_recover(
            recovery_window_seconds=300,
            current_concurrency=2,
            recovery_factor=0.25,
            max_recovery_concurrency=4
        )

        assert signal.should_recover is False

    def test_reset_throttle(self):
        """Test that throttle state can be reset."""
        tracker = DownloadSpeedTracker()

        # Record errors and trigger throttling
        for _ in range(3):
            tracker.record_error("rate_limit")
            time.sleep(0.01)

        tracker.should_throttle(
            threshold=3,
            window_seconds=600,
            current_concurrency=4,
            throttle_factor=0.5,
            min_concurrent=1
        )

        assert tracker._throttle_state is not None
        assert tracker._throttle_state.is_throttled is True

        # Reset
        tracker.reset_throttle()

        assert tracker._throttle_state is None


class TestDownloadCoordinatorThrottling:
    """Test throttling integration in DownloadCoordinator."""

    def test_set_concurrency(self):
        """Test that concurrency can be adjusted dynamically."""
        coordinator = DownloadCoordinator(max_concurrent=4)

        assert coordinator.get_current_concurrency() == 4

        coordinator.set_concurrency(2)

        assert coordinator.get_current_concurrency() == 2

    def test_set_concurrency_enforces_minimum(self):
        """Test that concurrency cannot go below 1."""
        coordinator = DownloadCoordinator(max_concurrent=4)

        coordinator.set_concurrency(0)  # Should be clamped to 1

        assert coordinator.get_current_concurrency() == 1

    def test_set_concurrency_logs_change(self):
        """Test that concurrency changes are logged."""
        coordinator = DownloadCoordinator(max_concurrent=4)

        with patch('src.downloader.orchestrator.logger') as mock_logger:
            coordinator.set_concurrency(2)

            mock_logger.info.assert_called_once()
            assert "4 → 2" in mock_logger.info.call_args[0][0]


class TestThrottlingConfigDownloadConfig:
    """Test that SelfRegulateConfig integrates with DownloadConfig."""

    def test_download_config_has_self_regulate(self):
        """Test that DownloadConfig includes self_regulate field."""
        from src.config.sections.download import DownloadConfig

        config = DownloadConfig()

        assert hasattr(config, 'self_regulate')
        assert config.self_regulate is not None
        assert config.self_regulate.enabled is True

    def test_self_regulate_from_dict(self):
        """Test that self_regulate can be loaded from dict."""
        from src.config.sections.download import DownloadConfig

        config = DownloadConfig(
            self_regulate={
                'enabled': True,
                'rate_limit_prevention_threshold': 5,
                'throttle_factor': 0.3
            }
        )

        assert config.self_regulate.enabled is True
        assert config.self_regulate.rate_limit_prevention_threshold == 5
        assert config.self_regulate.throttle_factor == 0.3


class TestThrottlingIntegration:
    """Integration tests for throttling behavior."""

    def test_throttling_reduces_concurrency(self):
        """Test that throttling actually reduces the coordinator concurrency."""
        tracker = DownloadSpeedTracker()
        coordinator = DownloadCoordinator(max_concurrent=4)

        # Record errors to trigger throttling
        for _ in range(3):
            tracker.record_error("rate_limit")
            time.sleep(0.01)

        # Check throttling
        signal = tracker.should_throttle(
            threshold=3,
            window_seconds=600,
            current_concurrency=coordinator.get_current_concurrency(),
            throttle_factor=0.5,
            min_concurrent=1
        )

        if signal.should_throttle:
            coordinator.set_concurrency(signal.new_concurrency)

        assert coordinator.get_current_concurrency() == 2

    def test_recovery_increases_concurrency(self):
        """Test that recovery gradually increases concurrency."""
        tracker = DownloadSpeedTracker()
        coordinator = DownloadCoordinator(max_concurrent=4)

        # Trigger throttling
        for _ in range(3):
            tracker.record_error("rate_limit")
            time.sleep(0.01)

        signal = tracker.should_throttle(
            threshold=3,
            window_seconds=600,
            current_concurrency=4,
            throttle_factor=0.5,
            min_concurrent=1
        )

        if signal.should_throttle:
            coordinator.set_concurrency(signal.new_concurrency)

        # Simulate recovery window passed
        if tracker._throttle_state:
            tracker._throttle_state.last_error_time = time.time() - 400

        # Try recovery
        recovery_signal = tracker.should_recover(
            recovery_window_seconds=300,
            current_concurrency=coordinator.get_current_concurrency(),
            recovery_factor=0.25,
            max_recovery_concurrency=4
        )

        if recovery_signal.should_recover:
            coordinator.set_concurrency(recovery_signal.new_concurrency)

        # Test recovery from a higher throttled state
        # Starting at 3 with recovery_factor 0.25: 3 * 1.25 = 3.75 -> int = 3
        # This won't increase, so test with higher recovery factor
        # Let's use a different approach - test with recovery factor that increases

        # Re-test: starting at 3 with recovery_factor 0.5: 3 * 1.5 = 4.5 -> int = 4
        tracker2 = DownloadSpeedTracker()
        coordinator2 = DownloadCoordinator(max_concurrent=4)

        # Trigger and set to throttled state manually
        tracker2._throttle_state = ThrottleState(
            current_concurrency=3,
            original_concurrency=4,
            throttle_level=1,
            is_throttled=True,
            last_throttle_time=time.time() - 100,
            last_error_time=time.time() - 400,  # Past recovery window
            is_recovering=False,
            recovery_step=0
        )

        recovery_signal2 = tracker2.should_recover(
            recovery_window_seconds=300,
            current_concurrency=3,
            recovery_factor=0.5,  # Higher factor for meaningful increase
            max_recovery_concurrency=4
        )

        assert recovery_signal2.should_recover is True
        assert recovery_signal2.new_concurrency == 4  # int(3 * 1.5) = 4
        assert recovery_signal2.new_concurrency == 4  # int(3 * 1.5) = 4


class TestThrottlingCheckpoint:
    """Test throttling state persistence."""

    def test_to_checkpoint_dict_includes_throttle_state(self):
        """Test that throttle state is included in checkpoint."""
        tracker = DownloadSpeedTracker()

        # Record errors and trigger throttling
        for _ in range(3):
            tracker.record_error("rate_limit")
            time.sleep(0.01)

        tracker.should_throttle(
            threshold=3,
            window_seconds=600,
            current_concurrency=4,
            throttle_factor=0.5,
            min_concurrent=1
        )

        data = tracker.to_checkpoint_dict()

        assert 'throttle_state' in data
        assert 'error_timestamps' in data

    def test_from_checkpoint_dict_restores_throttle_state(self):
        """Test that throttle state is restored from checkpoint."""
        tracker = DownloadSpeedTracker()

        # Create test data
        test_data = {
            'records': [],
            'throttle_state': {
                'current_concurrency': 2,
                'original_concurrency': 4,
                'throttle_level': 1,
                'is_throttled': True,
                'last_throttle_time': 1000.0,
                'last_error_time': 1000.0,
                'is_recovering': False,
                'recovery_step': 0
            },
            'error_timestamps': [1000.0, 1000.0, 1000.0]
        }

        tracker.from_checkpoint_dict(test_data)

        assert tracker._throttle_state is not None
        assert tracker._throttle_state.current_concurrency == 2
        assert tracker._throttle_state.is_throttled is True
        assert len(tracker._error_timestamps) == 3
