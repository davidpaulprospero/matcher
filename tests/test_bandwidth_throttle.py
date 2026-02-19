"""Unit tests for bandwidth throttling configuration (US-114-007).

Tests bandwidth_limit_kbps config, peak/offpeak hour limits, and yt-dlp integration.
"""

import pytest

from src.config.sections.download import BandwidthThrottleConfig


class TestBandwidthThrottleConfigDefaults:
    """Tests for default configuration values."""

    def test_default_enabled(self):
        """Test default enabled is False."""
        config = BandwidthThrottleConfig()
        assert config.enabled is False

    def test_default_bandwidth_limit_kbps(self):
        """Test default bandwidth_limit_kbps is 0 (unlimited)."""
        config = BandwidthThrottleConfig()
        assert config.bandwidth_limit_kbps == 0

    def test_default_global_limit(self):
        """Test default global_limit is '5M'."""
        config = BandwidthThrottleConfig()
        assert config.global_limit == "5M"

    def test_default_peak_offpeak_disabled(self):
        """Test peak/offpeak is disabled by default."""
        config = BandwidthThrottleConfig()
        assert config.peak_offpeak_enabled is False

    def test_default_peak_hours(self):
        """Test default peak hours are 9 AM to 5 PM."""
        config = BandwidthThrottleConfig()
        assert config.peak_hours_start == 9
        assert config.peak_hours_end == 17

    def test_default_peak_limit_kbps(self):
        """Test default peak limit is 1000 Kbps."""
        config = BandwidthThrottleConfig()
        assert config.peak_limit_kbps == 1000

    def test_default_offpeak_limit_kbps(self):
        """Test default offpeak limit is 5000 Kbps."""
        config = BandwidthThrottleConfig()
        assert config.offpeak_limit_kbps == 5000


class TestBandwidthThrottleConfigValidation:
    """Tests for configuration validation."""

    def test_negative_bandwidth_limit_kbps_raises(self):
        """Test negative bandwidth_limit_kbps raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BandwidthThrottleConfig(bandwidth_limit_kbps=-1)
        assert "bandwidth_limit_kbps" in str(exc_info.value)

    def test_invalid_peak_hours_start_raises(self):
        """Test invalid peak_hours_start raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BandwidthThrottleConfig(peak_hours_start=25)
        assert "peak_hours_start" in str(exc_info.value)

    def test_invalid_peak_hours_end_raises(self):
        """Test invalid peak_hours_end raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BandwidthThrottleConfig(peak_hours_end=24)
        assert "peak_hours_end" in str(exc_info.value)

    def test_negative_peak_limit_kbps_raises(self):
        """Test negative peak_limit_kbps raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BandwidthThrottleConfig(peak_limit_kbps=-1)
        assert "peak_limit_kbps" in str(exc_info.value)

    def test_negative_offpeak_limit_kbps_raises(self):
        """Test negative offpeak_limit_kbps raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BandwidthThrottleConfig(offpeak_limit_kbps=-1)
        assert "offpeak_limit_kbps" in str(exc_info.value)


class TestBandwidthThrottleKbpsConversion:
    """Tests for integer Kbps to string conversion."""

    def test_kbps_to_string_500k(self):
        """Test 500 Kbps converts to '500K'."""
        config = BandwidthThrottleConfig()
        assert config._kbps_to_string(500) == "500K"

    def test_kbps_to_string_1000k(self):
        """Test 1000 Kbps converts to '1M'."""
        config = BandwidthThrottleConfig()
        assert config._kbps_to_string(1000) == "1M"

    def test_kbps_to_string_1500k(self):
        """Test 1500 Kbps converts to '1.5M'."""
        config = BandwidthThrottleConfig()
        assert config._kbps_to_string(1500) == "1.5M"

    def test_kbps_to_string_5000k(self):
        """Test 5000 Kbps converts to '5M'."""
        config = BandwidthThrottleConfig()
        assert config._kbps_to_string(5000) == "5M"

    def test_kbps_to_string_100k(self):
        """Test 100 Kbps converts to '100K'."""
        config = BandwidthThrottleConfig()
        assert config._kbps_to_string(100) == "100K"


class TestBandwidthThrottleSizeBased:
    """Tests for size-based throttling."""

    def test_disabled_returns_none(self):
        """Test disabled config returns None."""
        config = BandwidthThrottleConfig(enabled=False)
        assert config.get_limit_for_size(100.0) is None

    def test_small_file_bypasses(self):
        """Test files smaller than bypass_under_mb return None."""
        config = BandwidthThrottleConfig(enabled=True, bypass_under_mb=10.0)
        assert config.get_limit_for_size(5.0) is None

    def test_large_file_uses_global_limit(self):
        """Test large files use global_limit."""
        config = BandwidthThrottleConfig(enabled=True, global_limit="5M", bypass_under_mb=10.0)
        assert config.get_limit_for_size(100.0) == "5M"

    def test_per_download_limit_applies(self):
        """Test per_download_limit applies for large files."""
        config = BandwidthThrottleConfig(
            enabled=True,
            global_limit="5M",
            per_download_limit="2M",
            per_download_min_size_mb=50.0,
            bypass_under_mb=10.0
        )
        # File < 50MB should use global limit
        assert config.get_limit_for_size(30.0) == "5M"
        # File >= 50MB should use per_download limit
        assert config.get_limit_for_size(60.0) == "2M"

    def test_zero_global_limit_disables(self):
        """Test global_limit '0' returns None."""
        config = BandwidthThrottleConfig(enabled=True, global_limit="0")
        assert config.get_limit_for_size(100.0) is None


class TestBandwidthThrottleTimeBased:
    """Tests for time-based (peak/offpeak) throttling."""

    def test_peak_offpeak_disabled_uses_kbps(self):
        """Test peak_offpeak disabled uses bandwidth_limit_kbps."""
        config = BandwidthThrottleConfig(
            enabled=True,
            bandwidth_limit_kbps=1000,
            peak_offpeak_enabled=False
        )
        # Should use bandwidth_limit_kbps when peak_offpeak is disabled
        assert config.get_limit_for_time(hour=12) == "1M"

    def test_peak_hours_returns_peak_limit(self):
        """Test during peak hours returns peak_limit_kbps."""
        config = BandwidthThrottleConfig(
            enabled=True,
            peak_offpeak_enabled=True,
            peak_hours_start=9,
            peak_hours_end=17,
            peak_limit_kbps=1000,
            offpeak_limit_kbps=5000
        )
        # 12 PM is within peak hours (9-17)
        assert config.get_limit_for_time(hour=12) == "1M"

    def test_offpeak_hours_returns_offpeak_limit(self):
        """Test during offpeak hours returns offpeak_limit_kbps."""
        config = BandwidthThrottleConfig(
            enabled=True,
            peak_offpeak_enabled=True,
            peak_hours_start=9,
            peak_hours_end=17,
            peak_limit_kbps=1000,
            offpeak_limit_kbps=5000
        )
        # 8 AM is before peak hours
        assert config.get_limit_for_time(hour=8) == "5M"
        # 8 PM is after peak hours
        assert config.get_limit_for_time(hour=20) == "5M"

    def test_wrap_around_peak_hours(self):
        """Test wrap-around peak hours (e.g., 22 to 6)."""
        config = BandwidthThrottleConfig(
            enabled=True,
            peak_offpeak_enabled=True,
            peak_hours_start=22,
            peak_hours_end=6,
            peak_limit_kbps=500,
            offpeak_limit_kbps=5000
        )
        # 23 (11 PM) is within peak (22-6 wrap-around)
        assert config.get_limit_for_time(hour=23) == "500K"
        # 2 AM is within peak (22-6 wrap-around)
        assert config.get_limit_for_time(hour=2) == "500K"
        # 12 PM is outside peak
        assert config.get_limit_for_time(hour=12) == "5M"

    def test_zero_limit_returns_none(self):
        """Test zero limit returns None."""
        config = BandwidthThrottleConfig(
            enabled=True,
            peak_offpeak_enabled=True,
            peak_limit_kbps=0,
            offpeak_limit_kbps=0
        )
        assert config.get_limit_for_time(hour=12) is None


class TestBandwidthThrottleUnified:
    """Tests for unified get_limit method combining size and time."""

    def test_disabled_returns_none(self):
        """Test disabled config returns None."""
        config = BandwidthThrottleConfig(enabled=False)
        assert config.get_limit(file_size_mb=100.0, hour=12) is None

    def test_size_bypass_with_time_based(self):
        """Test small file bypasses even with time-based config."""
        config = BandwidthThrottleConfig(
            enabled=True,
            peak_offpeak_enabled=True,
            peak_limit_kbps=1000,
            bypass_under_mb=10.0
        )
        # Small file should bypass
        assert config.get_limit(file_size_mb=5.0, hour=12) is None
        # Large file should use peak limit
        assert config.get_limit(file_size_mb=100.0, hour=12) == "1M"

    def test_time_based_fallback_to_size(self):
        """Test fallback to size-based when time-based not enabled."""
        config = BandwidthThrottleConfig(
            enabled=True,
            bandwidth_limit_kbps=0,
            global_limit="5M",
            peak_offpeak_enabled=False,
            bypass_under_mb=10.0
        )
        # Should use global_limit for large file
        assert config.get_limit(file_size_mb=100.0, hour=12) == "5M"

    def test_time_based_takes_precedence(self):
        """Test time-based limit takes precedence over size-based."""
        config = BandwidthThrottleConfig(
            enabled=True,
            bandwidth_limit_kbps=1000,  # Would be "1M"
            global_limit="10M",  # Would be "10M"
            peak_offpeak_enabled=True,
            peak_hours_start=9,
            peak_hours_end=17,
            peak_limit_kbps=500,  # Should be "500K" during peak
            offpeak_limit_kbps=2000,
            bypass_under_mb=10.0
        )
        # During peak hours, should use peak limit (500K)
        assert config.get_limit(file_size_mb=100.0, hour=12) == "500K"
        # During offpeak, should use offpeak limit (2000K = 2M)
        assert config.get_limit(file_size_mb=100.0, hour=8) == "2M"


class TestBandwidthThrottleEdgeCases:
    """Edge case tests for bandwidth throttling."""

    def test_boundary_peak_hours_start(self):
        """Test boundary at peak_hours_start."""
        config = BandwidthThrottleConfig(
            enabled=True,
            peak_offpeak_enabled=True,
            peak_hours_start=9,
            peak_hours_end=17,
            peak_limit_kbps=1000,
            offpeak_limit_kbps=5000
        )
        # At exactly start time (9 AM) - should be peak
        assert config.get_limit_for_time(hour=9) == "1M"

    def test_boundary_peak_hours_end(self):
        """Test boundary at peak_hours_end."""
        config = BandwidthThrottleConfig(
            enabled=True,
            peak_offpeak_enabled=True,
            peak_hours_start=9,
            peak_hours_end=17,
            peak_limit_kbps=1000,
            offpeak_limit_kbps=5000
        )
        # At exactly end time (5 PM) - should be offpeak
        assert config.get_limit_for_time(hour=17) == "5M"

    def test_hour_0_midnight(self):
        """Test midnight hour."""
        config = BandwidthThrottleConfig(
            enabled=True,
            peak_offpeak_enabled=True,
            peak_hours_start=22,
            peak_hours_end=6,
            peak_limit_kbps=500,
            offpeak_limit_kbps=5000
        )
        # Midnight is in peak (22-6 wrap-around)
        assert config.get_limit_for_time(hour=0) == "500K"

    def test_hour_23(self):
        """Test 11 PM hour."""
        config = BandwidthThrottleConfig(
            enabled=True,
            peak_offpeak_enabled=True,
            peak_hours_start=22,
            peak_hours_end=6,
            peak_limit_kbps=500,
            offpeak_limit_kbps=5000
        )
        # 11 PM is in peak (22-6 wrap-around)
        assert config.get_limit_for_time(hour=23) == "500K"
