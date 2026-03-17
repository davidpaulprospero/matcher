"""
Tests for download progress ETA and bandwidth utilities (US-113-010).

Verifies:
- ETA calculation from speed and remaining bytes
- Bandwidth utilization calculation
- Progress line formatting consistent with pipeline logging

US-113-010: Enhanced Download Progress Reporting with ETA
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages.download_segments import (
    calculate_eta_seconds,
    calculate_bandwidth_utilization,
    format_eta_display,
    format_progress_line,
    calculate_network_congestion_factor,
    calculate_eta_confidence_interval,
    format_eta_confidence_display,
    ETAHistory,
)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================================
# Test ETA Calculation
# ============================================================================

class TestCalculateEtaSeconds:
    """Test ETA calculation from download speed and remaining bytes."""

    def test_eta_with_valid_speed(self):
        """Test ETA calculation with valid speed."""
        # 10 MB downloaded at 1 MB/s, 90 MB remaining = 90 seconds
        eta = calculate_eta_seconds(10 * 1024 * 1024, 100 * 1024 * 1024, 1024 * 1024)
        assert eta is not None
        assert 89 <= eta <= 91  # Allow small floating point variance

    def test_eta_with_none_speed(self):
        """Test ETA returns None when speed is None."""
        eta = calculate_eta_seconds(10 * 1024 * 1024, 100 * 1024 * 1024, None)
        assert eta is None

    def test_eta_with_zero_speed(self):
        """Test ETA returns None when speed is zero."""
        eta = calculate_eta_seconds(10 * 1024 * 1024, 100 * 1024 * 1024, 0)
        assert eta is None

    def test_eta_with_negative_speed(self):
        """Test ETA returns None when speed is negative."""
        eta = calculate_eta_seconds(10 * 1024 * 1024, 100 * 1024 * 1024, -100)
        assert eta is None

    def test_eta_near_completion(self):
        """Test ETA when near completion."""
        # 99 MB downloaded at 1 MB/s, 1 MB remaining = ~1 second
        eta = calculate_eta_seconds(99 * 1024 * 1024, 100 * 1024 * 1024, 1024 * 1024)
        assert eta is not None
        assert 0.5 <= eta <= 1.5

    def test_eta_zero_remaining(self):
        """Test ETA when already complete."""
        eta = calculate_eta_seconds(100 * 1024 * 1024, 100 * 1024 * 1024, 1024 * 1024)
        assert eta == 0.0

    def test_eta_already_complete(self):
        """Test ETA when downloaded exceeds total."""
        eta = calculate_eta_seconds(110 * 1024 * 1024, 100 * 1024 * 1024, 1024 * 1024)
        assert eta == 0.0

    def test_eta_high_speed(self):
        """Test ETA with high download speed."""
        # 10 MB at 50 MB/s = 0.2 seconds
        eta = calculate_eta_seconds(10 * 1024 * 1024, 100 * 1024 * 1024, 50 * 1024 * 1024)
        assert eta is not None
        assert 1.5 <= eta <= 2.5


# ============================================================================
# Test Bandwidth Utilization
# ============================================================================

class TestCalculateBandwidthUtilization:
    """Test bandwidth utilization percentage calculation."""

    def test_utilization_at_max(self):
        """Test utilization at maximum bandwidth."""
        # 100 MB/s at 100 MB/s default max = 100%
        util = calculate_bandwidth_utilization(100 * 1024 * 1024)
        assert util is not None
        assert util == 100.0

    def test_utilization_at_half(self):
        """Test utilization at half bandwidth."""
        # 50 MB/s at 100 MB/s default max = 50%
        util = calculate_bandwidth_utilization(50 * 1024 * 1024)
        assert util is not None
        assert util == 50.0

    def test_utilization_at_quarter(self):
        """Test utilization at quarter bandwidth."""
        # 25 MB/s at 100 MB/s default max = 25%
        util = calculate_bandwidth_utilization(25 * 1024 * 1024)
        assert util is not None
        assert util == 25.0

    def test_utilization_custom_max(self):
        """Test utilization with custom max bandwidth."""
        # 25 MB/s at 50 MB/s max = 50%
        util = calculate_bandwidth_utilization(25 * 1024 * 1024, max_bandwidth_mbps=50.0)
        assert util is not None
        assert util == 50.0

    def test_utilization_none_speed(self):
        """Test utilization returns None when speed is None."""
        util = calculate_bandwidth_utilization(None)
        assert util is None

    def test_utilization_zero_speed(self):
        """Test utilization returns None when speed is zero."""
        util = calculate_bandwidth_utilization(0)
        assert util is None

    def test_utilization_exceeds_max(self):
        """Test utilization caps at 100% when exceeding max."""
        # 150 MB/s at 100 MB/s max should cap at 100%
        util = calculate_bandwidth_utilization(150 * 1024 * 1024)
        assert util is not None
        assert util == 100.0

    def test_utilization_low_speed(self):
        """Test utilization with low speed."""
        # 1 MB/s at 100 MB/s = 1%
        util = calculate_bandwidth_utilization(1 * 1024 * 1024)
        assert util is not None
        assert util == 1.0


# ============================================================================
# Test ETA Display Formatting
# ============================================================================

class TestFormatEtaDisplay:
    """Test ETA display string formatting."""

    def test_eta_none(self):
        """Test formatting None returns 'unknown'."""
        assert format_eta_display(None) == "unknown"

    def test_eta_zero(self):
        """Test formatting zero returns '< 1s'."""
        assert format_eta_display(0) == "< 1s"

    def test_eta_under_second(self):
        """Test formatting under 1 second."""
        assert format_eta_display(0.5) == "< 1s"

    def test_eta_seconds_only(self):
        """Test formatting seconds only."""
        assert format_eta_display(30) == "30s"

    def test_eta_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        result = format_eta_display(90)  # 1m 30s
        assert result == "1m 30s"

    def test_eta_minutes_only(self):
        """Test formatting minutes only (no seconds)."""
        result = format_eta_display(120)  # 2m
        assert result == "2m"

    def test_eta_hours_minutes(self):
        """Test formatting hours and minutes."""
        result = format_eta_display(3600)  # 1h
        assert result == "1h"

    def test_eta_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds."""
        result = format_eta_display(3723)  # 1h 2m 3s
        assert result == "1h 2m"

    def test_eta_large_hours(self):
        """Test formatting large hour values."""
        result = format_eta_display(7200)  # 2h
        assert result == "2h"


# ============================================================================
# Test Progress Line Formatting
# ============================================================================

class TestFormatProgressLine:
    """Test progress line formatting for consistent pipeline logging."""

    def test_progress_line_basic(self):
        """Test basic progress line format."""
        line = format_progress_line(
            video_id="dQw4w9WgXcQ",
            downloaded_mb=10.5,
            total_mb=100.0,
            speed_kbps=1024.0,
            eta_seconds=90.0,
            bandwidth_util=10.0,
            tier=1,
            elapsed=10.0,
        )
        assert "dQw4w9WgXcQ" in line
        assert "10.5MB" in line
        assert "100.0MB" in line
        assert "1024KB/s" in line
        assert "ETA:" in line
        assert "1m 30s" in line
        assert "BW:" in line
        assert "10%" in line
        assert "T1" in line

    def test_progress_line_no_tier(self):
        """Test progress line without tier."""
        line = format_progress_line(
            video_id="abc123",
            downloaded_mb=50.0,
            total_mb=100.0,
            speed_kbps=2048.0,
            eta_seconds=25.0,
            bandwidth_util=20.0,
            tier=None,
            elapsed=50.0,
        )
        assert "abc123" in line
        assert "T1" in line  # Defaults to T1

    def test_progress_line_high_tier(self):
        """Test progress line with high tier."""
        line = format_progress_line(
            video_id="xyz789",
            downloaded_mb=5.0,
            total_mb=50.0,
            speed_kbps=512.0,
            eta_seconds=90.0,
            bandwidth_util=5.0,
            tier=4,
            elapsed=5.0,
        )
        assert "T4" in line

    def test_progress_line_unknown_eta(self):
        """Test progress line with unknown ETA."""
        line = format_progress_line(
            video_id="test",
            downloaded_mb=5.0,
            total_mb=50.0,
            speed_kbps=0.0,
            eta_seconds=None,
            bandwidth_util=None,
            tier=1,
            elapsed=1.0,
        )
        assert "unknown" in line
        assert "N/A" in line

    def test_progress_line_format(self):
        """Test progress line has consistent [DOWNLOAD] prefix."""
        line = format_progress_line(
            video_id="vid",
            downloaded_mb=10.0,
            total_mb=20.0,
            speed_kbps=500.0,
            eta_seconds=20.0,
            bandwidth_util=5.0,
            tier=2,
            elapsed=10.0,
        )
        assert line.startswith("[DOWNLOAD]")


# ============================================================================
# Test Network Congestion Factor (US-143-007)
# ============================================================================

class TestNetworkCongestionFactor:
    """Test network congestion factor calculation based on speed variance."""

    def test_congestion_stable_network(self):
        """Test congestion factor with very stable speeds."""
        # Speeds: 100, 102, 98, 101, 99 MB/s (very stable, CV < 0.1)
        speeds = [100 * 1024 * 1024, 102 * 1024 * 1024, 98 * 1024 * 1024, 101 * 1024 * 1024, 99 * 1024 * 1024]
        factor = calculate_network_congestion_factor(speeds)
        assert factor == 0.9  # Very stable = optimistic

    def test_congestion_normal_network(self):
        """Test congestion factor with normal variance."""
        # Speeds: 80, 90, 100, 110, 120 MB/s (normal variance)
        speeds = [80 * 1024 * 1024, 90 * 1024 * 1024, 100 * 1024 * 1024, 110 * 1024 * 1024, 120 * 1024 * 1024]
        factor = calculate_network_congestion_factor(speeds)
        assert 0.95 <= factor <= 1.0

    def test_congestion_variable_network(self):
        """Test congestion factor with high variance."""
        # Speeds: 20, 40, 60, 80, 100 MB/s (high variance)
        speeds = [20 * 1024 * 1024, 40 * 1024 * 1024, 60 * 1024 * 1024, 80 * 1024 * 1024, 100 * 1024 * 1024]
        factor = calculate_network_congestion_factor(speeds)
        assert factor >= 1.25  # Variable = pessimistic

    def test_congestion_very_variable_network(self):
        """Test congestion factor with very high variance."""
        # Speeds: 10, 30, 50, 70, 90 MB/s (very high variance)
        speeds = [10 * 1024 * 1024, 30 * 1024 * 1024, 50 * 1024 * 1024, 70 * 1024 * 1024, 90 * 1024 * 1024]
        factor = calculate_network_congestion_factor(speeds)
        assert factor >= 1.5  # Very variable = add significant padding

    def test_congestion_empty_speeds(self):
        """Test congestion factor with empty speeds."""
        factor = calculate_network_congestion_factor([])
        assert factor == 1.0  # Default to normal

    def test_congestion_single_speed(self):
        """Test congestion factor with single speed."""
        factor = calculate_network_congestion_factor([100 * 1024 * 1024])
        assert factor == 1.0  # Default to normal

    def test_congestion_two_speeds(self):
        """Test congestion factor with two speeds."""
        speeds = [90 * 1024 * 1024, 110 * 1024 * 1024]
        factor = calculate_network_congestion_factor(speeds)
        assert 0.9 <= factor <= 1.0  # Slight variance


# ============================================================================
# Test ETA Confidence Interval (US-143-007)
# ============================================================================

class TestETAConfidenceInterval:
    """Test ETA confidence interval calculation."""

    def test_confidence_interval_stable(self):
        """Test confidence interval with stable speeds."""
        speeds = [100 * 1024 * 1024, 102 * 1024 * 1024, 98 * 1024 * 1024, 101 * 1024 * 1024, 99 * 1024 * 1024]
        eta_seconds = 100.0
        lower, upper = calculate_eta_confidence_interval(eta_seconds, speeds, 0.8)
        assert lower is not None
        assert upper is not None
        # With stable speeds, margin should be small (~10%)
        assert lower >= eta_seconds * 0.8
        assert upper <= eta_seconds * 1.2

    def test_confidence_interval_variable(self):
        """Test confidence interval with variable speeds."""
        speeds = [20 * 1024 * 1024, 40 * 1024 * 1024, 60 * 1024 * 1024, 80 * 1024 * 1024, 100 * 1024 * 1024]
        eta_seconds = 100.0
        lower, upper = calculate_eta_confidence_interval(eta_seconds, speeds, 0.8)
        assert lower is not None
        assert upper is not None
        # With variable speeds, margin should be wider
        assert lower < eta_seconds * 0.7
        assert upper > eta_seconds * 1.3

    def test_confidence_interval_default_margin(self):
        """Test confidence interval with no speed history."""
        eta_seconds = 100.0
        lower, upper = calculate_eta_confidence_interval(eta_seconds, [], 0.8)
        # Should use default 20% margin
        assert lower == 80.0
        assert upper == 120.0

    def test_confidence_interval_high_confidence(self):
        """Test confidence interval with 90% confidence level."""
        speeds = [50 * 1024 * 1024, 60 * 1024 * 1024, 70 * 1024 * 1024]
        eta_seconds = 100.0
        lower, upper = calculate_eta_confidence_interval(eta_seconds, speeds, 0.9)
        # Higher confidence = wider interval
        lower_default, upper_default = calculate_eta_confidence_interval(eta_seconds, speeds, 0.8)
        assert lower <= lower_default
        assert upper >= upper_default

    def test_confidence_interval_none_eta(self):
        """Test confidence interval with None ETA."""
        lower, upper = calculate_eta_confidence_interval(None, [100, 200, 300], 0.8)
        assert lower is None
        assert upper is None


# ============================================================================
# Test ETA Confidence Display (US-143-007)
# ============================================================================

class TestETAConfidenceDisplay:
    """Test ETA with confidence interval display formatting."""

    def test_confidence_display_with_bounds(self):
        """Test display with confidence bounds."""
        result = format_eta_confidence_display(100.0, 80.0, 120.0)
        assert "1m" in result  # Base ETA formatted as minutes
        assert "±" in result  # Has confidence interval

    def test_confidence_display_none_eta(self):
        """Test display with None ETA."""
        result = format_eta_confidence_display(None, None, None)
        assert result == "unknown"

    def test_confidence_display_no_bounds(self):
        """Test display with no bounds data."""
        result = format_eta_confidence_display(60.0, None, None)
        assert "1m" in result  # Just shows base ETA (formatted)


# ============================================================================
# Test ETA History (US-143-007)
# ============================================================================

class TestETAHistory:
    """Test ETA history tracking for accuracy analysis."""

    def test_history_record_prediction(self):
        """Test recording ETA predictions."""
        history = ETAHistory(max_history=10)
        history.record_prediction(
            predicted_eta=100.0,
            remaining_bytes=1000000,
            current_speed=10000.0,
            timestamp=1000.0
        )
        stats = history.get_accuracy_stats()
        assert stats['sample_count'] == 0  # No actual recorded yet

    def test_history_record_actual(self):
        """Test recording actual duration."""
        history = ETAHistory(max_history=10)
        history.record_prediction(100.0, 1000000, 10000.0, 1000.0)
        history.record_actual(95.0)
        stats = history.get_accuracy_stats()
        assert stats['sample_count'] == 1

    def test_history_accuracy_calculation(self):
        """Test accuracy calculation."""
        history = ETAHistory(max_history=10)
        # Record prediction of 100s, actual was 100s = 0% error
        history.record_prediction(100.0, 1000000, 10000.0, 1000.0)
        history.record_actual(100.0)
        stats = history.get_accuracy_stats()
        assert stats['mean_error_pct'] == 0.0
        assert stats['accuracy_score'] == 100.0

    def test_history_accuracy_with_error(self):
        """Test accuracy calculation with error."""
        history = ETAHistory(max_history=10)
        # Predicted 100s, actual was 80s = 25% error
        history.record_prediction(100.0, 1000000, 10000.0, 1000.0)
        history.record_actual(80.0)
        stats = history.get_accuracy_stats()
        assert stats['mean_error_pct'] > 0.0  # Should have some error
        assert stats['accuracy_score'] < 100.0

    def test_history_clear(self):
        """Test clearing history."""
        history = ETAHistory(max_history=10)
        history.record_prediction(100.0, 1000000, 10000.0, 1000.0)
        history.record_actual(80.0)
        history.clear()
        stats = history.get_accuracy_stats()
        assert stats['sample_count'] == 0

    def test_history_insufficient_data(self):
        """Test accuracy with insufficient data."""
        history = ETAHistory(max_history=10)
        # Less than 3 samples - should still calculate error
        history.record_prediction(100.0, 1000000, 10000.0, 1000.0)
        history.record_actual(80.0)  # 100 predicted, 80 actual = 25% error
        stats = history.get_accuracy_stats()
        # Should calculate error even with 1 sample
        assert stats['accuracy_score'] == 75.0  # 100 - 25 = 75
        assert stats['sample_count'] == 1
