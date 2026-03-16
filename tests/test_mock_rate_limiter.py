"""Tests for mock rate limiter functionality.

Tests the mock rate limiting feature that bypasses actual delays in test mode.
"""

import os
import time
import pytest

from src.downloader.rate_limit_budget import (
    set_mock_rate_limits,
    is_mock_rate_limits_enabled,
    get_mock_delay_seconds,
    mock_rate_limit_delay,
)


class TestMockRateLimiter:
    """Tests for mock rate limiter functions."""

    def setup_method(self):
        """Reset mock state before each test."""
        # Reset global state
        set_mock_rate_limits(False, 0.01)
        # Ensure env var is not set
        if "MOCK_RATE_LIMITS" in os.environ:
            del os.environ["MOCK_RATE_LIMITS"]

    def teardown_method(self):
        """Clean up after each test."""
        # Reset state
        set_mock_rate_limits(False, 0.01)
        if "MOCK_RATE_LIMITS" in os.environ:
            del os.environ["MOCK_RATE_LIMITS"]

    def test_mock_disabled_by_default(self):
        """Test that mock rate limiting is disabled by default."""
        assert is_mock_rate_limits_enabled() is False

    def test_enable_mock_rate_limits(self):
        """Test programmatically enabling mock rate limits."""
        set_mock_rate_limits(True, 0.05)
        assert is_mock_rate_limits_enabled() is True
        assert get_mock_delay_seconds() == 0.05

    def test_disable_mock_rate_limits(self):
        """Test programmatically disabling mock rate limits."""
        set_mock_rate_limits(True)
        set_mock_rate_limits(False)
        assert is_mock_rate_limits_enabled() is False

    def test_env_var_enables_mock(self):
        """Test that MOCK_RATE_LIMITS=1 environment variable enables mock mode."""
        os.environ["MOCK_RATE_LIMITS"] = "1"
        assert is_mock_rate_limits_enabled() is True

    def test_env_var_true_enables_mock(self):
        """Test that MOCK_RATE_LIMITS=true environment variable enables mock mode."""
        os.environ["MOCK_RATE_LIMITS"] = "true"
        assert is_mock_rate_limits_enabled() is True

    def test_env_var_yes_enables_mock(self):
        """Test that MOCK_RATE_LIMITS=yes environment variable enables mock mode."""
        os.environ["MOCK_RATE_LIMITS"] = "yes"
        assert is_mock_rate_limits_enabled() is True

    def test_env_var_zero_does_not_enable_mock(self):
        """Test that MOCK_RATE_LIMITS=0 does not enable mock mode."""
        os.environ["MOCK_RATE_LIMITS"] = "0"
        assert is_mock_rate_limits_enabled() is False

    def test_mock_delay_with_env_var(self):
        """Test that mock delay works with environment variable enabled."""
        os.environ["MOCK_RATE_LIMITS"] = "1"

        start = time.time()
        mock_rate_limit_delay(5.0)  # Would sleep 5 seconds without mock
        elapsed = time.time() - start

        # Should take ~0.01 seconds, not 5 seconds
        assert elapsed < 0.5, f"Mock delay took too long: {elapsed}s"

    def test_mock_delay_with_programmatic_enable(self):
        """Test that mock delay works with programmatic enable."""
        set_mock_rate_limits(True, 0.02)

        start = time.time()
        mock_rate_limit_delay(10.0)  # Would sleep 10 seconds without mock
        elapsed = time.time() - start

        # Should take ~0.02 seconds, not 10 seconds
        assert elapsed < 0.5, f"Mock delay took too long: {elapsed}s"

    def test_real_delay_when_disabled(self):
        """Test that real delay is performed when mock is disabled."""
        set_mock_rate_limits(False)

        start = time.time()
        mock_rate_limit_delay(0.1)  # 100ms delay
        elapsed = time.time() - start

        # Should take at least 0.1 seconds
        assert elapsed >= 0.05, f"Real delay too short: {elapsed}s"
        assert elapsed < 0.5, f"Real delay too long: {elapsed}s"

    def test_config_mock_rate_limits(self):
        """Test that config.test_mode.mock_rate_limits enables mock mode."""
        # Create a mock config object
        class MockTestMode:
            mock_rate_limits = True
            mock_delay_seconds = 0.02

        class MockConfig:
            test_mode = MockTestMode()

        start = time.time()
        mock_rate_limit_delay(5.0, config=MockConfig())
        elapsed = time.time() - start

        # Should take ~0.02 seconds, not 5 seconds
        assert elapsed < 0.5, f"Mock delay took too long: {elapsed}s"

    def test_config_without_mock_rate_limits(self):
        """Test that config without mock_rate_limits doesn't enable mock."""
        class MockTestMode:
            mock_rate_limits = False

        class MockConfig:
            test_mode = MockTestMode()

        start = time.time()
        mock_rate_limit_delay(0.1, config=MockConfig())
        elapsed = time.time() - start

        # Should take at least 0.1 seconds (real delay)
        assert elapsed >= 0.05, f"Real delay too short: {elapsed}s"

    def test_config_without_test_mode(self):
        """Test that config without test_mode uses real delay."""
        class MockConfig:
            pass

        start = time.time()
        mock_rate_limit_delay(0.1, config=MockConfig())
        elapsed = time.time() - start

        # Should take at least 0.1 seconds (real delay)
        assert elapsed >= 0.05, f"Real delay too short: {elapsed}s"

    def test_env_var_overrides_programmatic(self):
        """Test that environment variable takes precedence."""
        set_mock_rate_limits(False)  # Programmatic disable

        os.environ["MOCK_RATE_LIMITS"] = "1"  # Env var enable

        assert is_mock_rate_limits_enabled() is True

    def test_zero_delay_no_sleep(self):
        """Test that zero delay doesn't cause sleep."""
        set_mock_rate_limits(True)

        start = time.time()
        mock_rate_limit_delay(0.0)
        elapsed = time.time() - start

        assert elapsed < 0.1, f"Zero delay took time: {elapsed}s"

    def test_negative_delay_no_sleep(self):
        """Test that negative delay doesn't cause sleep."""
        set_mock_rate_limits(True)

        start = time.time()
        mock_rate_limit_delay(-1.0)
        elapsed = time.time() - start

        assert elapsed < 0.1, f"Negative delay took time: {elapsed}s"
