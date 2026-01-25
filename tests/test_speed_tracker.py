"""Tests for DownloadSpeedTracker - US-005: Download speed monitoring for adaptive timeouts."""

import time
import pytest
from unittest.mock import patch, MagicMock

from src.downloader.speed_tracker import (
    DownloadSpeedTracker,
    DownloadSpeedConfig,
    DownloadRecord,
    RateLimitSignal
)


class TestDownloadSpeedConfig:
    """Tests for DownloadSpeedConfig defaults and values."""

    def test_default_values(self):
        """Test default config values are set correctly."""
        config = DownloadSpeedConfig()
        assert config.enabled is True
        assert config.window_size == 5
        assert config.min_speed_mbps == 1.0
        assert config.max_timeout_multiplier == 2.0
        assert config.enable_adaptive_timeout is True

    def test_custom_values(self):
        """Test config can be customized."""
        config = DownloadSpeedConfig(
            enabled=False,
            window_size=10,
            min_speed_mbps=2.0,
            max_timeout_multiplier=3.0,
            enable_adaptive_timeout=False
        )
        assert config.enabled is False
        assert config.window_size == 10
        assert config.min_speed_mbps == 2.0
        assert config.max_timeout_multiplier == 3.0
        assert config.enable_adaptive_timeout is False


class TestDownloadRecord:
    """Tests for DownloadRecord dataclass."""

    def test_speed_calculation(self):
        """Test speed_mbps property calculation."""
        record = DownloadRecord(
            video_id="test123",
            bytes_downloaded=10 * 1024 * 1024,  # 10 MB
            duration_seconds=5.0,  # 5 seconds
            timestamp=time.time(),
            tier="short"
        )
        # 10 MB / 5s = 2 MB/s
        assert record.speed_mbps == pytest.approx(2.0, rel=0.01)

    def test_speed_with_zero_duration(self):
        """Test speed_mbps returns 0 for zero duration."""
        record = DownloadRecord(
            video_id="test123",
            bytes_downloaded=10 * 1024 * 1024,
            duration_seconds=0,
            timestamp=time.time(),
            tier="short"
        )
        assert record.speed_mbps == 0.0


class TestTrackerInitialization:
    """Tests for DownloadSpeedTracker initialization."""

    def test_default_config(self):
        """Test tracker initializes with default config."""
        tracker = DownloadSpeedTracker()
        assert tracker.config.enabled is True
        assert tracker.config.window_size == 5

    def test_custom_config(self):
        """Test tracker initializes with custom config."""
        config = DownloadSpeedConfig(window_size=10)
        tracker = DownloadSpeedTracker(config)
        assert tracker.config.window_size == 10

    def test_empty_records_on_init(self):
        """Test tracker starts with no records."""
        tracker = DownloadSpeedTracker()
        assert len(tracker._records) == 0


class TestRecordDownload:
    """Tests for record_download method."""

    def test_records_valid_download(self):
        """Test valid download is recorded."""
        tracker = DownloadSpeedTracker()
        tracker.record_download(
            video_id="abc123",
            bytes_downloaded=50 * 1024 * 1024,  # 50 MB
            duration_seconds=10.0,
            tier="medium"
        )
        assert len(tracker._records) == 1
        assert tracker._records[0].video_id == "abc123"
        assert tracker._records[0].tier == "medium"

    def test_skips_invalid_values(self):
        """Test invalid values are skipped."""
        tracker = DownloadSpeedTracker()

        # Zero bytes
        tracker.record_download("test1", 0, 10.0, "short")
        assert len(tracker._records) == 0

        # Zero duration
        tracker.record_download("test2", 1000, 0, "short")
        assert len(tracker._records) == 0

        # Negative values
        tracker.record_download("test3", -1000, 10.0, "short")
        assert len(tracker._records) == 0

    def test_skips_when_disabled(self):
        """Test recording is skipped when tracker is disabled."""
        config = DownloadSpeedConfig(enabled=False)
        tracker = DownloadSpeedTracker(config)
        tracker.record_download("test123", 50 * 1024 * 1024, 10.0, "short")
        assert len(tracker._records) == 0

    def test_sliding_window_limit(self):
        """Test records are limited to window_size."""
        config = DownloadSpeedConfig(window_size=3)
        tracker = DownloadSpeedTracker(config)

        # Add 5 records, should keep only last 3
        for i in range(5):
            tracker.record_download(
                f"video{i}",
                10 * 1024 * 1024,
                5.0,
                "short"
            )

        assert len(tracker._records) == 3
        # Should have video2, video3, video4 (oldest dropped)
        video_ids = [r.video_id for r in tracker._records]
        assert video_ids == ["video2", "video3", "video4"]


class TestAverageSpeed:
    """Tests for get_average_speed_mbps method."""

    def test_no_records(self):
        """Test returns 0 with no records."""
        tracker = DownloadSpeedTracker()
        assert tracker.get_average_speed_mbps() == 0.0

    def test_single_record(self):
        """Test average with single record."""
        tracker = DownloadSpeedTracker()
        tracker.record_download(
            "video1",
            10 * 1024 * 1024,  # 10 MB
            5.0,  # 5 seconds
            "short"
        )
        # 10 MB / 5s = 2 MB/s
        assert tracker.get_average_speed_mbps() == pytest.approx(2.0, rel=0.01)

    def test_multiple_records(self):
        """Test average across multiple records."""
        tracker = DownloadSpeedTracker()

        # 10 MB in 5s = 2 MB/s
        tracker.record_download("video1", 10 * 1024 * 1024, 5.0, "short")
        # 20 MB in 5s = 4 MB/s
        tracker.record_download("video2", 20 * 1024 * 1024, 5.0, "short")

        # Total: 30 MB in 10s = 3 MB/s average
        assert tracker.get_average_speed_mbps() == pytest.approx(3.0, rel=0.01)


class TestAdaptiveTimeout:
    """Tests for get_adjusted_timeout method."""

    def test_no_adjustment_without_data(self):
        """Test no adjustment with insufficient data."""
        tracker = DownloadSpeedTracker()
        assert tracker.get_adjusted_timeout(120) == 120

        # Single record not enough
        tracker.record_download("video1", 10 * 1024 * 1024, 5.0, "short")
        assert tracker.get_adjusted_timeout(120) == 120

    def test_no_adjustment_when_fast(self):
        """Test no adjustment when speed is above minimum."""
        config = DownloadSpeedConfig(min_speed_mbps=1.0)
        tracker = DownloadSpeedTracker(config)

        # 2 MB/s - above 1.0 threshold
        tracker.record_download("video1", 20 * 1024 * 1024, 10.0, "short")
        tracker.record_download("video2", 20 * 1024 * 1024, 10.0, "short")

        assert tracker.get_adjusted_timeout(120) == 120

    def test_timeout_extended_on_slow_network(self):
        """Test timeout is extended when network is slow."""
        config = DownloadSpeedConfig(
            min_speed_mbps=2.0,  # Expect 2 MB/s
            max_timeout_multiplier=2.0
        )
        tracker = DownloadSpeedTracker(config)

        # Simulate 1 MB/s (50% of expected)
        tracker.record_download("video1", 10 * 1024 * 1024, 10.0, "short")  # 1 MB/s
        tracker.record_download("video2", 10 * 1024 * 1024, 10.0, "short")  # 1 MB/s

        # Speed is 1 MB/s, expected is 2 MB/s
        # Multiplier = 2.0 / 1.0 = 2.0x
        adjusted = tracker.get_adjusted_timeout(120)
        assert adjusted == 240  # 120 * 2.0

    def test_timeout_capped_at_max_multiplier(self):
        """Test timeout extension is capped at max_timeout_multiplier."""
        config = DownloadSpeedConfig(
            min_speed_mbps=4.0,  # Expect 4 MB/s
            max_timeout_multiplier=2.0  # Max 2x extension
        )
        tracker = DownloadSpeedTracker(config)

        # Simulate 0.5 MB/s (1/8 of expected)
        tracker.record_download("video1", 5 * 1024 * 1024, 10.0, "short")  # 0.5 MB/s
        tracker.record_download("video2", 5 * 1024 * 1024, 10.0, "short")  # 0.5 MB/s

        # Without cap: 4.0 / 0.5 = 8x, but capped at 2x
        adjusted = tracker.get_adjusted_timeout(120)
        assert adjusted == 240  # 120 * 2.0 (capped)

    def test_adaptive_timeout_disabled(self):
        """Test no adjustment when adaptive timeout is disabled."""
        config = DownloadSpeedConfig(
            min_speed_mbps=2.0,
            enable_adaptive_timeout=False
        )
        tracker = DownloadSpeedTracker(config)

        # Slow network
        tracker.record_download("video1", 5 * 1024 * 1024, 10.0, "short")
        tracker.record_download("video2", 5 * 1024 * 1024, 10.0, "short")

        # No adjustment because adaptive timeout is disabled
        assert tracker.get_adjusted_timeout(120) == 120


class TestSpeedStats:
    """Tests for get_speed_stats method."""

    def test_empty_stats(self):
        """Test stats with no records."""
        tracker = DownloadSpeedTracker()
        stats = tracker.get_speed_stats()

        assert stats['samples'] == 0
        assert stats['avg_speed_mbps'] == 0.0
        assert stats['min_speed_mbps'] == 0.0
        assert stats['max_speed_mbps'] == 0.0
        assert stats['total_bytes'] == 0
        assert stats['total_duration'] == 0.0
        assert stats['records'] == []

    def test_stats_with_records(self):
        """Test stats with multiple records."""
        tracker = DownloadSpeedTracker()

        tracker.record_download("video1", 10 * 1024 * 1024, 5.0, "short")
        tracker.record_download("video2", 20 * 1024 * 1024, 10.0, "medium")

        stats = tracker.get_speed_stats()

        assert stats['samples'] == 2
        assert stats['avg_speed_mbps'] == pytest.approx(2.0, rel=0.01)  # 30 MB / 15s
        assert stats['min_speed_mbps'] == pytest.approx(2.0, rel=0.01)  # Both 2 MB/s
        assert stats['max_speed_mbps'] == pytest.approx(2.0, rel=0.01)
        assert stats['total_bytes'] == 30 * 1024 * 1024
        assert stats['total_duration'] == pytest.approx(15.0, rel=0.01)
        assert len(stats['records']) == 2


class TestCheckpointPersistence:
    """Tests for checkpoint save/restore functionality."""

    def test_to_checkpoint_dict(self):
        """Test serialization to checkpoint dict."""
        tracker = DownloadSpeedTracker()
        tracker.record_download("video1", 10 * 1024 * 1024, 5.0, "short")
        tracker.record_download("video2", 20 * 1024 * 1024, 10.0, "medium")

        checkpoint = tracker.to_checkpoint_dict()

        assert 'records' in checkpoint
        assert len(checkpoint['records']) == 2
        assert checkpoint['records'][0]['video_id'] == "video1"
        assert checkpoint['records'][1]['video_id'] == "video2"

    def test_from_checkpoint_dict(self):
        """Test restoration from checkpoint dict."""
        tracker = DownloadSpeedTracker()

        checkpoint_data = {
            'records': [
                {
                    'video_id': 'video1',
                    'bytes_downloaded': 10 * 1024 * 1024,
                    'duration_seconds': 5.0,
                    'timestamp': time.time() - 100,
                    'tier': 'short'
                },
                {
                    'video_id': 'video2',
                    'bytes_downloaded': 20 * 1024 * 1024,
                    'duration_seconds': 10.0,
                    'timestamp': time.time() - 50,
                    'tier': 'medium'
                }
            ]
        }

        tracker.from_checkpoint_dict(checkpoint_data)

        assert len(tracker._records) == 2
        assert tracker._records[0].video_id == "video1"
        assert tracker._records[1].video_id == "video2"
        assert tracker.get_average_speed_mbps() == pytest.approx(2.0, rel=0.01)

    def test_from_checkpoint_dict_handles_missing_tier(self):
        """Test restoration handles missing tier field gracefully."""
        tracker = DownloadSpeedTracker()

        checkpoint_data = {
            'records': [
                {
                    'video_id': 'video1',
                    'bytes_downloaded': 10 * 1024 * 1024,
                    'duration_seconds': 5.0,
                    'timestamp': time.time()
                    # No tier field
                }
            ]
        }

        tracker.from_checkpoint_dict(checkpoint_data)

        assert len(tracker._records) == 1
        assert tracker._records[0].tier == "unknown"

    def test_from_checkpoint_dict_handles_invalid_records(self):
        """Test restoration skips invalid records."""
        tracker = DownloadSpeedTracker()

        checkpoint_data = {
            'records': [
                {
                    'video_id': 'valid',
                    'bytes_downloaded': 10 * 1024 * 1024,
                    'duration_seconds': 5.0,
                    'timestamp': time.time(),
                    'tier': 'short'
                },
                {
                    'invalid': 'record'  # Missing required fields
                },
                {
                    'video_id': 'also_valid',
                    'bytes_downloaded': 20 * 1024 * 1024,
                    'duration_seconds': 10.0,
                    'timestamp': time.time(),
                    'tier': 'medium'
                }
            ]
        }

        tracker.from_checkpoint_dict(checkpoint_data)

        # Should have 2 valid records, skipped 1 invalid
        assert len(tracker._records) == 2

    def test_clear_removes_all_records(self):
        """Test clear() removes all records."""
        tracker = DownloadSpeedTracker()
        tracker.record_download("video1", 10 * 1024 * 1024, 5.0, "short")
        tracker.record_download("video2", 20 * 1024 * 1024, 10.0, "medium")

        assert len(tracker._records) == 2

        tracker.clear()

        assert len(tracker._records) == 0
        assert tracker.get_speed_stats()['samples'] == 0


class TestIntegration:
    """Integration tests simulating real-world usage."""

    def test_sliding_window_with_varying_speeds(self):
        """Test tracker behavior with varying download speeds."""
        config = DownloadSpeedConfig(
            window_size=3,
            min_speed_mbps=2.0,
            max_timeout_multiplier=2.0
        )
        tracker = DownloadSpeedTracker(config)

        # Start fast (4 MB/s)
        tracker.record_download("v1", 40 * 1024 * 1024, 10.0, "short")
        tracker.record_download("v2", 40 * 1024 * 1024, 10.0, "short")

        # No adjustment when fast
        assert tracker.get_adjusted_timeout(120) == 120

        # Network slows down (1 MB/s)
        tracker.record_download("v3", 10 * 1024 * 1024, 10.0, "short")

        # Average: (40+40+10) / (10+10+10) = 90 MB / 30s = 3 MB/s
        # Still above 2.0 threshold, no adjustment
        assert tracker.get_adjusted_timeout(120) == 120

        # More slow downloads
        tracker.record_download("v4", 10 * 1024 * 1024, 10.0, "short")
        tracker.record_download("v5", 10 * 1024 * 1024, 10.0, "short")

        # Window now has v3, v4, v5 (all 1 MB/s)
        # Average: 1 MB/s, threshold: 2 MB/s
        # Multiplier: 2.0 / 1.0 = 2.0x
        assert tracker.get_adjusted_timeout(120) == 240

    def test_roundtrip_checkpoint(self):
        """Test full save/restore checkpoint cycle."""
        # Create tracker with records
        tracker1 = DownloadSpeedTracker()
        tracker1.record_download("v1", 10 * 1024 * 1024, 5.0, "short")
        tracker1.record_download("v2", 20 * 1024 * 1024, 10.0, "medium")

        # Save to checkpoint
        checkpoint = tracker1.to_checkpoint_dict()

        # Create new tracker and restore
        tracker2 = DownloadSpeedTracker()
        tracker2.from_checkpoint_dict(checkpoint)

        # Verify state matches
        stats1 = tracker1.get_speed_stats()
        stats2 = tracker2.get_speed_stats()

        assert stats1['samples'] == stats2['samples']
        assert stats1['avg_speed_mbps'] == pytest.approx(stats2['avg_speed_mbps'], rel=0.01)
        assert stats1['total_bytes'] == stats2['total_bytes']


# ============================================================================
# US-002: Rate limit signal detection tests
# ============================================================================


class TestRateLimitSignalDataclass:
    """Tests for RateLimitSignal dataclass."""

    def test_signal_attributes(self):
        """Test RateLimitSignal has all expected attributes."""
        signal = RateLimitSignal(
            detected=True,
            consecutive_slow_count=3,
            recent_speeds=[0.05, 0.08, 0.03],
            threshold=0.1,
            message="Test signal"
        )
        assert signal.detected is True
        assert signal.consecutive_slow_count == 3
        assert signal.recent_speeds == [0.05, 0.08, 0.03]
        assert signal.threshold == 0.1
        assert signal.message == "Test signal"

    def test_signal_false_detected(self):
        """Test RateLimitSignal with no detection."""
        signal = RateLimitSignal(
            detected=False,
            consecutive_slow_count=1,
            recent_speeds=[0.05],
            threshold=0.1,
            message="Insufficient samples"
        )
        assert signal.detected is False
        assert signal.consecutive_slow_count == 1


class TestRateLimitSignalConfigDefaults:
    """Tests for rate limit signal config defaults."""

    def test_default_threshold(self):
        """Test default rate limit signal threshold is 0.1 MB/s."""
        config = DownloadSpeedConfig()
        assert config.rate_limit_signal_threshold == 0.1

    def test_default_consecutive_samples(self):
        """Test default consecutive slow samples is 3."""
        config = DownloadSpeedConfig()
        assert config.consecutive_slow_samples == 3

    def test_custom_threshold(self):
        """Test custom rate limit signal threshold."""
        config = DownloadSpeedConfig(rate_limit_signal_threshold=0.05)
        assert config.rate_limit_signal_threshold == 0.05

    def test_custom_consecutive_samples(self):
        """Test custom consecutive slow samples."""
        config = DownloadSpeedConfig(consecutive_slow_samples=5)
        assert config.consecutive_slow_samples == 5


class TestDetectRateLimitSignals:
    """Tests for detect_rate_limit_signals method."""

    def test_insufficient_samples_returns_no_signal(self):
        """Test no signal when not enough samples."""
        config = DownloadSpeedConfig(consecutive_slow_samples=3)
        tracker = DownloadSpeedTracker(config)

        # Only 2 samples, need 3
        tracker.record_download("v1", 500 * 1024, 10.0, "short")  # 0.05 MB/s
        tracker.record_download("v2", 500 * 1024, 10.0, "short")  # 0.05 MB/s

        signal = tracker.detect_rate_limit_signals()
        assert signal.detected is False
        assert "Insufficient samples" in signal.message

    def test_signal_detected_with_consecutive_slow_downloads(self):
        """Test signal detected when 3+ consecutive downloads are slow."""
        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,  # 0.1 MB/s threshold
            consecutive_slow_samples=3
        )
        tracker = DownloadSpeedTracker(config)

        # 3 downloads all below 0.1 MB/s (throttled)
        tracker.record_download("v1", 50 * 1024, 10.0, "short")  # 0.005 MB/s
        tracker.record_download("v2", 80 * 1024, 10.0, "short")  # 0.008 MB/s
        tracker.record_download("v3", 30 * 1024, 10.0, "short")  # 0.003 MB/s

        signal = tracker.detect_rate_limit_signals()
        assert signal.detected is True
        assert signal.consecutive_slow_count >= 3
        assert "Rate limit signal" in signal.message

    def test_no_signal_when_downloads_fast(self):
        """Test no signal when downloads are above threshold."""
        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,
            consecutive_slow_samples=3
        )
        tracker = DownloadSpeedTracker(config)

        # 3 downloads all above 0.1 MB/s (healthy)
        tracker.record_download("v1", 10 * 1024 * 1024, 10.0, "short")  # 1 MB/s
        tracker.record_download("v2", 20 * 1024 * 1024, 10.0, "short")  # 2 MB/s
        tracker.record_download("v3", 15 * 1024 * 1024, 10.0, "short")  # 1.5 MB/s

        signal = tracker.detect_rate_limit_signals()
        assert signal.detected is False
        assert "No rate limit signal" in signal.message

    def test_signal_requires_consecutive_slow(self):
        """Test signal only fires when slow samples are consecutive (most recent)."""
        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,
            consecutive_slow_samples=3
        )
        tracker = DownloadSpeedTracker(config)

        # Mix of fast and slow - only 2 consecutive slow at end
        tracker.record_download("v1", 50 * 1024, 10.0, "short")  # 0.005 MB/s (slow)
        tracker.record_download("v2", 10 * 1024 * 1024, 10.0, "short")  # 1 MB/s (fast)
        tracker.record_download("v3", 50 * 1024, 10.0, "short")  # 0.005 MB/s (slow)
        tracker.record_download("v4", 50 * 1024, 10.0, "short")  # 0.005 MB/s (slow)

        signal = tracker.detect_rate_limit_signals()
        assert signal.detected is False  # Only 2 consecutive slow, need 3
        assert signal.consecutive_slow_count == 2

    def test_signal_triggers_on_three_consecutive_at_end(self):
        """Test signal fires when last 3 are slow, even after fast ones."""
        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,
            consecutive_slow_samples=3
        )
        tracker = DownloadSpeedTracker(config)

        # Fast then slow (network degrading)
        tracker.record_download("v1", 10 * 1024 * 1024, 10.0, "short")  # 1 MB/s (fast)
        tracker.record_download("v2", 50 * 1024, 10.0, "short")  # 0.005 MB/s (slow)
        tracker.record_download("v3", 50 * 1024, 10.0, "short")  # 0.005 MB/s (slow)
        tracker.record_download("v4", 50 * 1024, 10.0, "short")  # 0.005 MB/s (slow)

        signal = tracker.detect_rate_limit_signals()
        assert signal.detected is True
        assert signal.consecutive_slow_count == 3

    def test_signal_includes_recent_speeds(self):
        """Test signal includes recent speeds for debugging."""
        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,
            consecutive_slow_samples=3,
            window_size=5
        )
        tracker = DownloadSpeedTracker(config)

        # Record 4 downloads
        tracker.record_download("v1", 50 * 1024, 10.0, "short")
        tracker.record_download("v2", 60 * 1024, 10.0, "short")
        tracker.record_download("v3", 70 * 1024, 10.0, "short")
        tracker.record_download("v4", 80 * 1024, 10.0, "short")

        signal = tracker.detect_rate_limit_signals()
        assert len(signal.recent_speeds) == 4
        assert signal.threshold == 0.1


class TestSignalLogging:
    """Tests for rate limit signal logging."""

    def test_signal_logs_warning_when_detected(self):
        """Test WARNING log emitted when signal detected."""
        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,
            consecutive_slow_samples=3
        )
        tracker = DownloadSpeedTracker(config)

        # 3 slow downloads
        tracker.record_download("v1", 50 * 1024, 10.0, "short")
        tracker.record_download("v2", 50 * 1024, 10.0, "short")
        tracker.record_download("v3", 50 * 1024, 10.0, "short")

        with patch('src.downloader.speed_tracker.logger') as mock_logger:
            signal = tracker.detect_rate_limit_signals()
            assert signal.detected is True
            mock_logger.warning.assert_called_once()
            assert "Rate limit signal" in mock_logger.warning.call_args[0][0]

    def test_no_warning_when_no_signal(self):
        """Test no WARNING log when no signal detected."""
        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,
            consecutive_slow_samples=3
        )
        tracker = DownloadSpeedTracker(config)

        # Fast downloads
        tracker.record_download("v1", 10 * 1024 * 1024, 10.0, "short")
        tracker.record_download("v2", 10 * 1024 * 1024, 10.0, "short")
        tracker.record_download("v3", 10 * 1024 * 1024, 10.0, "short")

        with patch('src.downloader.speed_tracker.logger') as mock_logger:
            signal = tracker.detect_rate_limit_signals()
            assert signal.detected is False
            mock_logger.warning.assert_not_called()


class TestSignalWithCircuitBreaker:
    """Integration tests for signal with circuit breaker."""

    def test_signal_can_trigger_circuit_breaker_failure(self):
        """Test signal detection can be used to trigger circuit breaker."""
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,
            consecutive_slow_samples=3
        )
        tracker = DownloadSpeedTracker(config)

        breaker_config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3
        )
        breaker = CircuitBreaker(breaker_config)

        # 3 slow downloads trigger signal
        tracker.record_download("v1", 50 * 1024, 10.0, "short")
        tracker.record_download("v2", 50 * 1024, 10.0, "short")
        tracker.record_download("v3", 50 * 1024, 10.0, "short")

        signal = tracker.detect_rate_limit_signals()
        if signal.detected:
            breaker.record_failure()

        # Verify failure was recorded
        assert breaker.state.consecutive_failures == 1

    def test_multiple_signals_can_trip_circuit_breaker(self):
        """Test multiple signals can accumulate to trip circuit breaker."""
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,
            consecutive_slow_samples=2,
            window_size=3
        )

        breaker_config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=2
        )
        breaker = CircuitBreaker(breaker_config)

        # First batch of slow downloads
        tracker = DownloadSpeedTracker(config)
        tracker.record_download("v1", 50 * 1024, 10.0, "short")
        tracker.record_download("v2", 50 * 1024, 10.0, "short")
        signal = tracker.detect_rate_limit_signals()
        if signal.detected:
            breaker.record_failure()

        # Simulate recovery (would need new tracker in real scenario)
        # but continue with more slow downloads
        tracker.record_download("v3", 50 * 1024, 10.0, "short")
        signal = tracker.detect_rate_limit_signals()
        if signal.detected:
            tripped = breaker.record_failure()
            assert tripped is True  # Circuit should trip now
            assert breaker.is_open is True


class TestSignalEdgeCases:
    """Edge case tests for rate limit signal detection."""

    def test_empty_tracker(self):
        """Test signal detection with no records."""
        tracker = DownloadSpeedTracker()
        signal = tracker.detect_rate_limit_signals()
        assert signal.detected is False
        assert signal.consecutive_slow_count == 0
        assert signal.recent_speeds == []

    def test_exactly_at_threshold(self):
        """Test download speed exactly at threshold is NOT considered slow."""
        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,
            consecutive_slow_samples=3
        )
        tracker = DownloadSpeedTracker(config)

        # Exactly 0.1 MB/s = 1 MB in 10s = 1048576 bytes in 10s
        tracker.record_download("v1", 1024 * 1024, 10.0, "short")  # 0.1 MB/s exactly
        tracker.record_download("v2", 1024 * 1024, 10.0, "short")
        tracker.record_download("v3", 1024 * 1024, 10.0, "short")

        signal = tracker.detect_rate_limit_signals()
        # At threshold should NOT be considered slow (< not <=)
        assert signal.detected is False

    def test_just_below_threshold(self):
        """Test download speed just below threshold IS considered slow."""
        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=0.1,
            consecutive_slow_samples=3
        )
        tracker = DownloadSpeedTracker(config)

        # Just below 0.1 MB/s (0.099 MB/s)
        bytes_for_099 = int(0.099 * 1024 * 1024 * 10)  # for 10 seconds
        tracker.record_download("v1", bytes_for_099, 10.0, "short")
        tracker.record_download("v2", bytes_for_099, 10.0, "short")
        tracker.record_download("v3", bytes_for_099, 10.0, "short")

        signal = tracker.detect_rate_limit_signals()
        assert signal.detected is True

    def test_custom_high_threshold(self):
        """Test with higher threshold (more sensitive detection)."""
        config = DownloadSpeedConfig(
            rate_limit_signal_threshold=1.0,  # 1 MB/s - much higher
            consecutive_slow_samples=2
        )
        tracker = DownloadSpeedTracker(config)

        # 0.5 MB/s is slow when threshold is 1.0
        tracker.record_download("v1", 5 * 1024 * 1024, 10.0, "short")  # 0.5 MB/s
        tracker.record_download("v2", 5 * 1024 * 1024, 10.0, "short")  # 0.5 MB/s

        signal = tracker.detect_rate_limit_signals()
        assert signal.detected is True
        assert signal.threshold == 1.0
