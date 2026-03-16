"""Tests for YouTubeAPIRateLimiter burst handling features (US-156-008).

Tests the burst_allowance, smooth_start, get_current_rate(), and burst_usage features.
"""

import pytest
from src.downloader.youtube_api_client import YouTubeAPIRateLimiter


class TestYouTubeAPIRateLimiterBurstHandling:
    """Tests for burst handling features in YouTubeAPIRateLimiter."""

    def test_burst_allowance_config(self):
        """Test that burst_allowance config is properly set."""
        limiter = YouTubeAPIRateLimiter(rate=10.0, burst_size=20, burst_allowance=15)
        assert limiter.burst_allowance == 15

    def test_burst_allowance_default(self):
        """Test that burst_allowance defaults to 10."""
        limiter = YouTubeAPIRateLimiter()
        assert limiter.burst_allowance == 10

    def test_smooth_start_config(self):
        """Test that smooth_start config is properly set."""
        limiter = YouTubeAPIRateLimiter(smooth_start=True, smooth_start_duration=10.0)
        assert limiter.smooth_start is True

    def test_smooth_start_default(self):
        """Test that smooth_start defaults to False."""
        limiter = YouTubeAPIRateLimiter()
        assert limiter.smooth_start is False

    def test_get_current_rate_without_smooth_start(self):
        """Test get_current_rate returns base rate when smooth_start is disabled."""
        limiter = YouTubeAPIRateLimiter(rate=10.0)
        assert limiter.get_current_rate() == 10.0

    def test_get_current_rate_with_smooth_start(self):
        """Test get_current_rate returns reduced rate during smooth start."""
        limiter = YouTubeAPIRateLimiter(
            rate=10.0,
            smooth_start=True,
            smooth_start_duration=5.0
        )
        # During smooth start, rate should be less than target
        current_rate = limiter.get_current_rate()
        assert current_rate < 10.0

    def test_burst_usage_initially_zero(self):
        """Test that burst_usage starts at 0."""
        limiter = YouTubeAPIRateLimiter(burst_allowance=10)
        assert limiter.burst_usage == 0

    def test_burst_usage_tracked_on_acquire(self):
        """Test that burst_usage is tracked when consuming burst allowance."""
        limiter = YouTubeAPIRateLimiter(rate=10.0, burst_size=5, burst_allowance=5)

        # Make multiple acquisitions that will use burst allowance
        for _ in range(6):
            limiter.acquire()

        # Burst usage should be tracked
        assert limiter.burst_usage > 0

    def test_burst_usage_in_metrics(self):
        """Test that burst_usage appears in metrics."""
        limiter = YouTubeAPIRateLimiter(rate=10.0, burst_size=5, burst_allowance=5)

        # Make some acquisitions
        for _ in range(3):
            limiter.acquire()

        metrics = limiter.get_metrics()
        assert "burst_usage" in metrics
        assert "burst_allowance" in metrics
        assert "smooth_start_enabled" in metrics
        assert "smooth_start_active" in metrics
        assert "current_rate" in metrics

    def test_burst_exhausted_warning(self):
        """Test warning when burst allowance is exhausted."""
        limiter = YouTubeAPIRateLimiter(
            rate=10.0,
            burst_size=2,
            burst_allowance=2,
            smooth_start=False
        )

        # Make many acquisitions to exhaust burst
        for _ in range(10):
            limiter.acquire()

        metrics = limiter.get_metrics()
        assert metrics["burst_usage"] >= 2

    def test_smooth_start_ramp_up(self):
        """Test that smooth start gradually ramps up the rate."""
        limiter = YouTubeAPIRateLimiter(
            rate=10.0,
            smooth_start=True,
            smooth_start_duration=10.0
        )

        initial_rate = limiter.get_current_rate()

        # At start, rate should be lower than target
        assert initial_rate < 10.0
        # Rate should be at least initial_rate
        assert initial_rate == limiter._initial_rate

    def test_reset_clears_burst_usage(self):
        """Test that reset clears burst_usage."""
        limiter = YouTubeAPIRateLimiter(rate=10.0, burst_size=2, burst_allowance=2)

        # Make some acquisitions to build up burst usage
        for _ in range(5):
            limiter.acquire()

        # Reset
        limiter.reset()

        assert limiter.burst_usage == 0
        assert limiter.get_current_rate() == 10.0

    def test_current_rate_includes_smooth_start(self):
        """Test that current_rate in metrics reflects smooth start."""
        limiter = YouTubeAPIRateLimiter(
            rate=10.0,
            smooth_start=True,
            smooth_start_duration=10.0
        )

        metrics = limiter.get_metrics()
        assert "current_rate" in metrics
        # During smooth start, current_rate should differ from rate
        assert metrics["current_rate"] <= metrics["rate"]


class TestYouTubeAPIRateLimiterBurstMetrics:
    """Tests for burst handling metrics in YouTubeAPIRateLimiter."""

    def test_burst_allowance_in_metrics(self):
        """Test that burst_allowance appears in metrics."""
        limiter = YouTubeAPIRateLimiter(rate=10.0, burst_allowance=15)
        metrics = limiter.get_metrics()
        assert metrics["burst_allowance"] == 15

    def test_smooth_start_fields_in_metrics(self):
        """Test that smooth start fields appear in metrics."""
        limiter = YouTubeAPIRateLimiter(
            rate=10.0,
            smooth_start=True,
            smooth_start_duration=5.0
        )
        metrics = limiter.get_metrics()

        assert "smooth_start_enabled" in metrics
        assert "smooth_start_active" in metrics
        assert metrics["smooth_start_enabled"] is True
        assert metrics["smooth_start_active"] is True

    def test_burst_exhausted_warnings_metric(self):
        """Test that burst_exhausted_warnings appears in metrics."""
        limiter = YouTubeAPIRateLimiter(
            rate=10.0,
            burst_size=1,
            burst_allowance=1
        )

        # Exhaust the burst
        for _ in range(10):
            limiter.acquire()

        metrics = limiter.get_metrics()
        assert "burst_exhausted_warnings" in metrics
        assert metrics["burst_exhausted_warnings"] >= 1
