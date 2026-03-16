"""Tests for US-109-006: Multi-signal correlation for rate limit detection.

Tests for:
- Sustained degradation detection
- Signal correlation (speed + error patterns)
- Signal weight configuration
"""

import pytest
from unittest.mock import patch, MagicMock

from src.downloader.speed_tracker import (
    DownloadSpeedTracker,
    DownloadSpeedConfig,
    SustainedDegradationSignal,
    CorrelatedSignal,
    ErrorPatternSignal,
    PerKeywordSpeedTracker,
)


@pytest.mark.fast
class TestSustainedDegradationConfig:
    """Tests for sustained degradation config defaults."""

    def test_default_enabled(self):
        """Test sustained degradation is enabled by default."""
        config = DownloadSpeedConfig()
        assert config.enable_sustained_degradation is True

    def test_default_samples_required(self):
        """Test default degradation samples required is 3."""
        config = DownloadSpeedConfig()
        assert config.degradation_samples_required == 3

    def test_default_degradation_threshold(self):
        """Test default degradation threshold is 50%."""
        config = DownloadSpeedConfig()
        assert config.degradation_threshold == 0.5

    def test_custom_config(self):
        """Test custom config values."""
        config = DownloadSpeedConfig(
            enable_sustained_degradation=False,
            degradation_samples_required=5,
            degradation_threshold=0.3
        )
        assert config.enable_sustained_degradation is False
        assert config.degradation_samples_required == 5
        assert config.degradation_threshold == 0.3


@pytest.mark.fast
class TestSignalCorrelationConfig:
    """Tests for signal correlation config defaults."""

    def test_default_enabled(self):
        """Test signal correlation is enabled by default."""
        config = DownloadSpeedConfig()
        assert config.enable_signal_correlation is True

    def test_default_weights(self):
        """Test default signal weights: 60% speed, 40% error."""
        config = DownloadSpeedConfig()
        assert config.speed_signal_weight == 0.6
        assert config.error_signal_weight == 0.4

    def test_default_correlation_threshold(self):
        """Test default correlation threshold is 0.7."""
        config = DownloadSpeedConfig()
        assert config.correlation_threshold == 0.7

    def test_custom_weights(self):
        """Test custom signal weights."""
        config = DownloadSpeedConfig(
            speed_signal_weight=0.7,
            error_signal_weight=0.3,
            correlation_threshold=0.8
        )
        assert config.speed_signal_weight == 0.7
        assert config.error_signal_weight == 0.3
        assert config.correlation_threshold == 0.8


@pytest.mark.fast
class TestSustainedDegradationDetection:
    """Tests for detect_sustained_degradation method."""

    def test_insufficient_samples(self):
        """Test returns insufficient data when fewer than required samples."""
        config = DownloadSpeedConfig(degradation_samples_required=3)
        tracker = DownloadSpeedTracker(config)

        # Only 2 samples, need 3
        tracker.record_download("v1", 10 * 1024 * 1024, 5.0, "short")
        tracker.record_download("v2", 10 * 1024 * 1024, 5.0, "short")

        signal = tracker.detect_sustained_degradation()
        assert signal.detected is False
        assert "Insufficient samples" in signal.message

    def test_no_degradation_stable_speeds(self):
        """Test no degradation when speeds are stable."""
        config = DownloadSpeedConfig(degradation_samples_required=3)
        tracker = DownloadSpeedTracker(config)

        # All similar speeds - no degradation
        tracker.record_download("v1", 10 * 1024 * 1024, 5.0, "short")  # 2 MB/s
        tracker.record_download("v2", 10 * 1024 * 1024, 5.0, "short")  # 2 MB/s
        tracker.record_download("v3", 10 * 1024 * 1024, 5.0, "short")  # 2 MB/s

        signal = tracker.detect_sustained_degradation()
        assert signal.detected is False
        assert signal.trend == 'stable'

    def test_degradation_detected(self):
        """Test degradation detected when speeds drop significantly."""
        config = DownloadSpeedConfig(
            degradation_samples_required=3,
            degradation_threshold=0.5  # 50% drop triggers
        )
        tracker = DownloadSpeedTracker(config)

        # Start fast, then slow down
        tracker.record_download("v1", 20 * 1024 * 1024, 5.0, "short")  # 4 MB/s
        tracker.record_download("v2", 15 * 1024 * 1024, 5.0, "short")  # 3 MB/s
        tracker.record_download("v3", 5 * 1024 * 1024, 5.0, "short")   # 1 MB/s

        signal = tracker.detect_sustained_degradation()
        assert signal.detected is True
        assert signal.degradation_percentage >= 0.5  # At least 50% drop
        assert signal.trend in ('degrading', 'severe')

    def test_improving_trend(self):
        """Test improving trend detected when speeds increase."""
        config = DownloadSpeedConfig(degradation_samples_required=3)
        tracker = DownloadSpeedTracker(config)

        # Speeds improving
        tracker.record_download("v1", 5 * 1024 * 1024, 5.0, "short")   # 1 MB/s
        tracker.record_download("v2", 10 * 1024 * 1024, 5.0, "short")  # 2 MB/s
        tracker.record_download("v3", 20 * 1024 * 1024, 5.0, "short")  # 4 MB/s

        signal = tracker.detect_sustained_degradation()
        assert signal.detected is False
        assert signal.trend == 'improving'

    def test_disabled_returns_no_signal(self):
        """Test disabled config returns no signal."""
        config = DownloadSpeedConfig(enable_sustained_degradation=False)
        tracker = DownloadSpeedTracker(config)

        tracker.record_download("v1", 20 * 1024 * 1024, 5.0, "short")
        tracker.record_download("v2", 5 * 1024 * 1024, 5.0, "short")
        tracker.record_download("v3", 5 * 1024 * 1024, 5.0, "short")

        signal = tracker.detect_sustained_degradation()
        assert signal.detected is False
        assert "disabled" in signal.message.lower()


@pytest.mark.fast
class TestSpeedSignalScore:
    """Tests for get_speed_signal_score method."""

    def test_no_data_returns_zero(self):
        """Test no data returns zero score."""
        tracker = DownloadSpeedTracker()
        assert tracker.get_speed_signal_score() == 0.0

    def test_fast_speeds_returns_low_score(self):
        """Test fast speeds return low signal score."""
        config = DownloadSpeedConfig(min_speed_mbps=1.0)
        tracker = DownloadSpeedTracker(config)

        # All fast downloads
        tracker.record_download("v1", 20 * 1024 * 1024, 5.0, "short")  # 4 MB/s
        tracker.record_download("v2", 20 * 1024 * 1024, 5.0, "short")  # 4 MB/s

        score = tracker.get_speed_signal_score()
        assert score < 0.3  # Low score for fast speeds

    def test_slow_speeds_returns_high_score(self):
        """Test slow speeds return high signal score."""
        config = DownloadSpeedConfig(rate_limit_signal_threshold=0.1)
        tracker = DownloadSpeedTracker(config)

        # All slow downloads
        tracker.record_download("v1", 50 * 1024, 10.0, "short")  # 0.005 MB/s
        tracker.record_download("v2", 50 * 1024, 10.0, "short")  # 0.005 MB/s
        tracker.record_download("v3", 50 * 1024, 10.0, "short")  # 0.005 MB/s

        score = tracker.get_speed_signal_score()
        assert score >= 0.7  # High score for slow speeds


@pytest.mark.fast
class TestCorrelatedSignalDetection:
    """Tests for detect_correlated_signals method."""

    def test_disabled_returns_no_signal(self):
        """Test disabled config returns no signal."""
        config = DownloadSpeedConfig(enable_signal_correlation=False)
        tracker = DownloadSpeedTracker(config)

        tracker.record_download("v1", 50 * 1024, 10.0, "short")
        tracker.record_download("v2", 50 * 1024, 10.0, "short")
        tracker.record_download("v3", 50 * 1024, 10.0, "short")

        # With high severity error signal
        error_signal = ErrorPatternSignal(
            error_count=3,
            error_types=['rate_limit', '429'],
            rate_limit_error_count=2,
            severity='high',
            message="Rate limit detected"
        )

        correlated = tracker.detect_correlated_signals(error_signal)
        assert correlated.detected is False
        assert correlated.recommended_action == 'none'

    def test_speed_only_correlation(self):
        """Test correlation with only speed signal (no error signal)."""
        config = DownloadSpeedConfig(
            correlation_threshold=0.5,
            speed_signal_weight=1.0,
            error_signal_weight=0.0
        )
        tracker = DownloadSpeedTracker(config)

        # All slow downloads - high speed signal
        tracker.record_download("v1", 50 * 1024, 10.0, "short")
        tracker.record_download("v2", 50 * 1024, 10.0, "short")
        tracker.record_download("v3", 50 * 1024, 10.0, "short")

        correlated = tracker.detect_correlated_signals()

        assert correlated.speed_signal_score >= 0.5
        assert correlated.speed_signal_detected is True
        assert correlated.error_signal_detected is False
        # Score should exceed threshold
        assert correlated.correlation_score >= 0.5

    def test_speed_plus_error_correlation(self):
        """Test correlation with both speed and error signals."""
        config = DownloadSpeedConfig(
            speed_signal_weight=0.6,
            error_signal_weight=0.4,
            correlation_threshold=0.5
        )
        tracker = DownloadSpeedTracker(config)

        # Slow downloads
        tracker.record_download("v1", 50 * 1024, 10.0, "short")
        tracker.record_download("v2", 50 * 1024, 10.0, "short")
        tracker.record_download("v3", 50 * 1024, 10.0, "short")

        # With high severity error signal
        error_signal = ErrorPatternSignal(
            error_count=3,
            error_types=['rate_limit', '429'],
            rate_limit_error_count=2,
            severity='high',
            message="Rate limit detected"
        )

        correlated = tracker.detect_correlated_signals(error_signal)

        assert correlated.speed_signal_detected is True
        assert correlated.error_signal_detected is True
        # Combined score should be high
        assert correlated.correlation_score >= 0.6

    def test_no_signal_fast_and_no_errors(self):
        """Test no correlated signal when speeds are fast and no errors."""
        config = DownloadSpeedConfig(
            speed_signal_weight=0.6,
            error_signal_weight=0.4,
            correlation_threshold=0.7
        )
        tracker = DownloadSpeedTracker(config)

        # Fast downloads
        tracker.record_download("v1", 20 * 1024 * 1024, 5.0, "short")
        tracker.record_download("v2", 20 * 1024 * 1024, 5.0, "short")
        tracker.record_download("v3", 20 * 1024 * 1024, 5.0, "short")

        # Low severity error
        error_signal = ErrorPatternSignal(
            error_count=1,
            error_types=['timeout'],
            rate_limit_error_count=0,
            severity='low',
            message="Minor timeout"
        )

        correlated = tracker.detect_correlated_signals(error_signal)

        assert correlated.detected is False
        assert correlated.recommended_action == 'none'

    def test_escalate_action_high_score(self):
        """Test escalate action when correlation score is very high."""
        config = DownloadSpeedConfig(
            speed_signal_weight=0.6,
            error_signal_weight=0.4,
            correlation_threshold=0.5  # Low threshold to trigger
        )
        tracker = DownloadSpeedTracker(config)

        # Very slow downloads
        tracker.record_download("v1", 10 * 1024, 10.0, "short")
        tracker.record_download("v2", 10 * 1024, 10.0, "short")
        tracker.record_download("v3", 10 * 1024, 10.0, "short")

        # High severity errors
        error_signal = ErrorPatternSignal(
            error_count=5,
            error_types=['rate_limit', '429', 'quota'],
            rate_limit_error_count=4,
            severity='high',
            message="Severe rate limit"
        )

        correlated = tracker.detect_correlated_signals(error_signal)

        # High combined score should trigger escalate
        assert correlated.correlation_score >= 0.85
        assert correlated.recommended_action == 'escalate'

    def test_watch_action_moderate_score(self):
        """Test watch action when correlation score is moderate."""
        config = DownloadSpeedConfig(
            speed_signal_weight=0.6,
            error_signal_weight=0.4,
            correlation_threshold=0.35  # Lower threshold for moderate combined score
        )
        tracker = DownloadSpeedTracker(config)

        # Moderate slow downloads (below rate limit threshold)
        tracker.record_download("v1", 1 * 1024 * 1024, 10.0, "short")  # 0.1 MB/s
        tracker.record_download("v2", 1 * 1024 * 1024, 10.0, "short")  # 0.1 MB/s
        tracker.record_download("v3", 1 * 1024 * 1024, 10.0, "short")  # 0.1 MB/s

        # Medium severity errors
        error_signal = ErrorPatternSignal(
            error_count=2,
            error_types=['rate_limit'],
            rate_limit_error_count=1,
            severity='medium',
            message="Moderate rate limit"
        )

        correlated = tracker.detect_correlated_signals(error_signal)

        # Should be detected but not escalate (score is around 0.36)
        assert correlated.detected is True
        assert correlated.recommended_action == 'watch'


@pytest.mark.fast
class TestPerKeywordSustainedDegradation:
    """Tests for per-keyword sustained degradation detection."""

    def test_keyword_isolation(self):
        """Test one keyword's degradation doesn't affect another."""
        config = DownloadSpeedConfig(
            degradation_samples_required=3,
            degradation_threshold=0.5
        )
        tracker = PerKeywordSpeedTracker(config)

        # Keyword A: fast (no degradation)
        tracker.record_download("cats", "v1", 20 * 1024 * 1024, 5.0, "short")
        tracker.record_download("cats", "v2", 20 * 1024 * 1024, 5.0, "short")
        tracker.record_download("cats", "v3", 20 * 1024 * 1024, 5.0, "short")

        # Keyword B: degrading
        tracker.record_download("dogs", "v1", 20 * 1024 * 1024, 5.0, "short")
        tracker.record_download("dogs", "v2", 10 * 1024 * 1024, 5.0, "short")
        tracker.record_download("dogs", "v3", 5 * 1024 * 1024, 5.0, "short")

        signal_cats = tracker.detect_sustained_degradation("cats")
        signal_dogs = tracker.detect_sustained_degradation("dogs")

        assert signal_cats.detected is False
        assert signal_dogs.detected is True

    def test_unknown_keyword_returns_no_signal(self):
        """Test unknown keyword returns no signal."""
        tracker = PerKeywordSpeedTracker()

        signal = tracker.detect_sustained_degradation("unknown")
        assert signal.detected is False
        assert "No data" in signal.message


@pytest.mark.fast
class TestPerKeywordCorrelatedSignals:
    """Tests for per-keyword correlated signal detection."""

    def test_keyword_correlation_isolated(self):
        """Test correlation is isolated per keyword."""
        config = DownloadSpeedConfig(
            speed_signal_weight=0.6,
            error_signal_weight=0.4,
            correlation_threshold=0.5
        )
        tracker = PerKeywordSpeedTracker(config)

        # Keyword A: slow with errors
        tracker.record_download("cats", "v1", 50 * 1024, 10.0, "short")
        tracker.record_download("cats", "v2", 50 * 1024, 10.0, "short")
        tracker.record_download("cats", "v3", 50 * 1024, 10.0, "short")

        # Keyword B: fast, no errors
        tracker.record_download("dogs", "v1", 20 * 1024 * 1024, 5.0, "short")
        tracker.record_download("dogs", "v2", 20 * 1024 * 1024, 5.0, "short")
        tracker.record_download("dogs", "v3", 20 * 1024 * 1024, 5.0, "short")

        # Error signal for cats only
        error_signal = ErrorPatternSignal(
            error_count=2,
            error_types=['rate_limit'],
            rate_limit_error_count=2,
            severity='high',
            message="Rate limit"
        )

        signal_cats = tracker.detect_correlated_signals("cats", error_signal)
        signal_dogs = tracker.detect_correlated_signals("dogs", None)

        assert signal_cats.detected is True
        assert signal_dogs.detected is False

    def test_unknown_keyword_returns_no_signal(self):
        """Test unknown keyword returns no correlation."""
        tracker = PerKeywordSpeedTracker()

        error_signal = ErrorPatternSignal(
            error_count=3,
            error_types=['rate_limit'],
            rate_limit_error_count=3,
            severity='high',
            message="Rate limit"
        )

        correlated = tracker.detect_correlated_signals("unknown", error_signal)
        assert correlated.detected is False
        assert "No data" in correlated.message


@pytest.mark.fast
class TestSignalDataclasses:
    """Tests for signal dataclasses."""

    def test_sustained_degradation_signal_attributes(self):
        """Test SustainedDegradationSignal has all required attributes."""
        signal = SustainedDegradationSignal(
            detected=True,
            degradation_percentage=0.75,
            samples_analyzed=5,
            peak_speed_mbps=4.0,
            current_speed_mbps=1.0,
            trend='degrading',
            message="Test degradation"
        )
        assert signal.detected is True
        assert signal.degradation_percentage == 0.75
        assert signal.samples_analyzed == 5
        assert signal.peak_speed_mbps == 4.0
        assert signal.current_speed_mbps == 1.0
        assert signal.trend == 'degrading'

    def test_correlated_signal_attributes(self):
        """Test CorrelatedSignal has all required attributes."""
        signal = CorrelatedSignal(
            detected=True,
            correlation_score=0.85,
            speed_signal_score=0.9,
            error_signal_score=0.8,
            speed_signal_detected=True,
            error_signal_detected=True,
            recommended_action='escalate',
            message="Test correlation"
        )
        assert signal.detected is True
        assert signal.correlation_score == 0.85
        assert signal.speed_signal_score == 0.9
        assert signal.error_signal_score == 0.8
        assert signal.recommended_action == 'escalate'

    def test_error_pattern_signal_attributes(self):
        """Test ErrorPatternSignal has all required attributes."""
        signal = ErrorPatternSignal(
            error_count=3,
            error_types=['rate_limit', '429'],
            rate_limit_error_count=2,
            severity='medium',
            message="Test errors"
        )
        assert signal.error_count == 3
        assert signal.error_types == ['rate_limit', '429']
        assert signal.rate_limit_error_count == 2
        assert signal.severity == 'medium'


@pytest.mark.fast
class TestSignalWeightNormalization:
    """Tests for signal weight normalization."""

    def test_unnormalized_weights_normalized(self):
        """Test weights that don't sum to 1.0 get normalized."""
        config = DownloadSpeedConfig(
            speed_signal_weight=0.3,
            error_signal_weight=0.3  # Sum = 0.6, not 1.0
        )
        tracker = DownloadSpeedTracker(config)

        # Add some slow downloads
        tracker.record_download("v1", 50 * 1024, 10.0, "short")
        tracker.record_download("v2", 50 * 1024, 10.0, "short")
        tracker.record_download("v3", 50 * 1024, 10.0, "short")

        error_signal = ErrorPatternSignal(
            error_count=2,
            error_types=['rate_limit'],
            rate_limit_error_count=2,
            severity='high',
            message="Test"
        )

        correlated = tracker.detect_correlated_signals(error_signal)

        # Should still work - weights normalized internally
        assert correlated.correlation_score > 0
        # Speed weight should be normalized to 0.5 (0.3/0.6)
        assert 0 < correlated.speed_signal_score <= 1.0


@pytest.mark.fast
class TestTrendClassification:
    """Tests for speed trend classification."""

    def test_improving_trend(self):
        """Test improving trend when speeds increase."""
        config = DownloadSpeedConfig(degradation_samples_required=5)
        tracker = DownloadSpeedTracker(config)

        # Clear improvement over time
        speeds = [1.0, 1.5, 2.0, 2.5, 3.0]  # MB/s
        for i, speed in enumerate(speeds):
            bytes_downloaded = int(speed * 1024 * 1024 * 5)  # 5 seconds
            tracker.record_download(f"v{i}", bytes_downloaded, 5.0, "short")

        signal = tracker.detect_sustained_degradation()
        assert signal.trend == 'improving'
        assert signal.detected is False

    def test_severe_degradation(self):
        """Test severe trend when speeds drop dramatically."""
        config = DownloadSpeedConfig(degradation_samples_required=4)
        tracker = DownloadSpeedTracker(config)

        # Dramatic drop
        speeds = [5.0, 4.0, 2.0, 0.5]  # MB/s
        for i, speed in enumerate(speeds):
            bytes_downloaded = int(speed * 1024 * 1024 * 5)
            tracker.record_download(f"v{i}", bytes_downloaded, 5.0, "short")

        signal = tracker.detect_sustained_degradation()
        assert signal.trend in ('degrading', 'severe')
        assert signal.detected is True
