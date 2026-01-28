"""
Tests for adaptive backoff multiplier based on error severity (US-008).

The adaptive backoff mechanism:
1. Classifies error messages into severity levels (low, medium, high)
2. Adjusts the backoff multiplier based on severity
3. Tracks backoff events by severity in metrics
4. Can be disabled via config to use fixed multiplier
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@dataclass
class MockRateLimitConfig:
    """Mock config for adaptive backoff testing."""
    initial_backoff_seconds: float = 5.0
    max_backoff_before_rotate: float = 60.0
    backoff_multiplier: float = 2.0
    adaptive_multiplier: bool = True
    per_tier_isolation: bool = False
    share_budget_across_keywords: bool = False


def create_mock_config(tmp_path, rate_limit_config=None, **overrides):
    """Create a mock config for testing."""
    mock_config = MagicMock()
    mock_config.cache_dir = str(tmp_path / ".cache")
    mock_config.downloaded_videos_dir = str(tmp_path / "videos")
    mock_config.download = MagicMock()
    mock_config.download.davinci_mode = False
    mock_config.download.cookies = None
    mock_config.download.cookies_from_browser = None
    mock_config.download.download_timeout = 120
    mock_config.download.download_timeouts = {}
    mock_config.download.delete_original = False
    mock_config.download.max_retries = 3
    mock_config.download.retry_delay = 2.0
    mock_config.download.retry_backoff = 2.0
    mock_config.download.rate_limit = rate_limit_config or MockRateLimitConfig()
    mock_config.download.cookie_rotation = None
    mock_config.download.vpn = None
    mock_config.download.rate_limit_budget = None
    mock_config.llm = MagicMock()
    mock_config.llm.provider = 'gemini'
    mock_config.llm.model = 'gemini-pro'

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(mock_config.download, key):
            setattr(mock_config.download, key, value)
        elif hasattr(mock_config, key):
            setattr(mock_config, key, value)

    return mock_config


def create_downloader(tmp_path, rate_limit_config=None, **overrides):
    """Create a VideoDownloader with mocked dependencies."""
    from src.downloader.core import VideoDownloader

    config = create_mock_config(tmp_path, rate_limit_config, **overrides)

    with patch('src.downloader.core.CheckpointManager'):
        with patch('src.downloader.core.TranscodingManager'):
            with patch('src.downloader.core.TitleFilter'):
                with patch('src.downloader.core.SpeechScreener'):
                    with patch('src.downloader.core.SearchOptimizer'):
                        with patch('src.downloader.core.AudioFirstPipeline'):
                            with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                return VideoDownloader(config)


class TestErrorSeverityClassification:
    """Test error message severity classification."""

    @pytest.mark.fast
    def test_classify_high_severity_quota_exceeded(self):
        """Test that 'quota exceeded' errors are classified as high severity."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("HTTP 429: Quota exceeded") == 'high'
        assert classify_error_severity("Daily quota has been exceeded") == 'high'

    @pytest.mark.fast
    def test_classify_high_severity_bot_detection(self):
        """Test that bot detection errors are classified as high severity."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("Bot detection triggered") == 'high'
        assert classify_error_severity("Automated traffic detected") == 'high'
        assert classify_error_severity("Suspicious activity on your account") == 'high'

    @pytest.mark.fast
    def test_classify_high_severity_blocks(self):
        """Test that severe blocks are classified as high severity."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("Your IP has been blocked") == 'high'
        assert classify_error_severity("Account suspended") == 'high'
        assert classify_error_severity("You have been permanently banned") == 'high'

    @pytest.mark.fast
    def test_classify_medium_severity_429(self):
        """Test that standard 429 errors are classified as medium severity."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("429 Too Many Requests") == 'medium'
        assert classify_error_severity("HTTP Error 429") == 'medium'

    @pytest.mark.fast
    def test_classify_medium_severity_rate_limit(self):
        """Test that rate limit messages are classified as medium severity."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("Rate limit reached") == 'medium'
        assert classify_error_severity("Too many requests, please wait") == 'medium'
        assert classify_error_severity("Please try again later") == 'medium'

    @pytest.mark.fast
    def test_classify_medium_severity_temporarily_unavailable(self):
        """Test that temporary unavailability is medium severity."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("Service temporarily unavailable") == 'medium'

    @pytest.mark.fast
    def test_classify_low_severity_sign_in(self):
        """Test that sign-in errors are classified as low severity."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("Sign in to confirm you're not a bot") == 'low'
        assert classify_error_severity("Login required") == 'low'

    @pytest.mark.fast
    def test_classify_low_severity_age_confirm(self):
        """Test that age confirmation errors are classified as low severity."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("Please confirm your age") == 'low'

    @pytest.mark.fast
    def test_classify_low_severity_slow_down(self):
        """Test that 'slow down' messages are classified as low severity."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("Slow down please") == 'low'

    @pytest.mark.fast
    def test_classify_unknown_defaults_to_medium(self):
        """Test that unknown error patterns default to medium severity."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("Some unknown error") == 'medium'
        assert classify_error_severity("Random failure xyz123") == 'medium'

    @pytest.mark.fast
    def test_classification_case_insensitive(self):
        """Test that classification is case-insensitive."""
        from src.downloader.core import classify_error_severity

        assert classify_error_severity("QUOTA EXCEEDED") == 'high'
        assert classify_error_severity("Bot Detection") == 'high'
        assert classify_error_severity("RATE LIMIT") == 'medium'
        assert classify_error_severity("Sign In Required") == 'low'


class TestSeverityMultipliers:
    """Test severity multiplier values."""

    @pytest.mark.fast
    def test_multiplier_values(self):
        """Test that multipliers match specification: low=1.5, medium=2.0, high=3.0."""
        from src.downloader.core import SEVERITY_MULTIPLIERS

        assert SEVERITY_MULTIPLIERS['low'] == 1.5
        assert SEVERITY_MULTIPLIERS['medium'] == 2.0
        assert SEVERITY_MULTIPLIERS['high'] == 3.0


class TestAdaptiveBackoffEnabled:
    """Test adaptive backoff when enabled (default)."""

    @pytest.mark.fast
    def test_low_severity_uses_1_5x_multiplier(self, tmp_path):
        """Test that low severity errors use 1.5x multiplier."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=10.0,
            adaptive_multiplier=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep') as mock_sleep:
            # "Sign in" triggers low severity (1.5x multiplier)
            # First backoff: 10 * (1.5 ^ 0) = 10
            downloader.handle_rate_limit_error("Please sign in to continue")
            assert abs(mock_sleep.call_args[0][0] - 10.0) < 0.01

            # Second backoff: 10 * (1.5 ^ 1) = 15
            downloader.handle_rate_limit_error("Please sign in to continue")
            assert abs(mock_sleep.call_args[0][0] - 15.0) < 0.01

    @pytest.mark.fast
    def test_medium_severity_uses_2x_multiplier(self, tmp_path):
        """Test that medium severity errors use 2.0x multiplier."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=10.0,
            max_backoff_before_rotate=200.0,
            adaptive_multiplier=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep') as mock_sleep:
            # "429" triggers medium severity (2.0x multiplier)
            # First backoff: 10 * (2 ^ 0) = 10
            downloader.handle_rate_limit_error("429 Too Many Requests")
            assert abs(mock_sleep.call_args[0][0] - 10.0) < 0.01

            # Second backoff: 10 * (2 ^ 1) = 20
            downloader.handle_rate_limit_error("429 Too Many Requests")
            assert abs(mock_sleep.call_args[0][0] - 20.0) < 0.01

    @pytest.mark.fast
    def test_high_severity_uses_3x_multiplier(self, tmp_path):
        """Test that high severity errors use 3.0x multiplier."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=200.0,
            adaptive_multiplier=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep') as mock_sleep:
            # "quota exceeded" triggers high severity (3.0x multiplier)
            # First backoff: 5 * (3 ^ 0) = 5
            downloader.handle_rate_limit_error("Quota exceeded for today")
            assert abs(mock_sleep.call_args[0][0] - 5.0) < 0.01

            # Second backoff: 5 * (3 ^ 1) = 15
            downloader.handle_rate_limit_error("Quota exceeded for today")
            assert abs(mock_sleep.call_args[0][0] - 15.0) < 0.01

            # Third backoff: 5 * (3 ^ 2) = 45
            downloader.handle_rate_limit_error("Quota exceeded for today")
            assert abs(mock_sleep.call_args[0][0] - 45.0) < 0.01

    @pytest.mark.fast
    def test_different_errors_different_multipliers_same_session(self, tmp_path):
        """Test that different error types in same session use appropriate multipliers."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=10.0,
            max_backoff_before_rotate=500.0,
            adaptive_multiplier=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep') as mock_sleep:
            # Low severity first: 10 * (1.5 ^ 0) = 10
            downloader.handle_rate_limit_error("sign in required")
            delay1 = mock_sleep.call_args[0][0]

            # High severity second: starts fresh severity but continues count
            # 10 * (3 ^ 1) = 30 (backoff_count is now 1)
            downloader.handle_rate_limit_error("quota exceeded")
            delay2 = mock_sleep.call_args[0][0]

            # High severity should give longer delay than low severity
            assert delay2 > delay1


class TestAdaptiveBackoffDisabled:
    """Test behavior when adaptive multiplier is disabled."""

    @pytest.mark.fast
    def test_uses_fixed_multiplier_when_disabled(self, tmp_path):
        """Test that fixed multiplier is used when adaptive is disabled."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=10.0,
            backoff_multiplier=2.0,  # Fixed multiplier
            max_backoff_before_rotate=200.0,
            adaptive_multiplier=False
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep') as mock_sleep:
            # Even with "quota exceeded" (high severity), should use fixed 2.0x
            downloader.handle_rate_limit_error("Quota exceeded")
            assert abs(mock_sleep.call_args[0][0] - 10.0) < 0.01  # 10 * (2 ^ 0)

            downloader.handle_rate_limit_error("Quota exceeded")
            assert abs(mock_sleep.call_args[0][0] - 20.0) < 0.01  # 10 * (2 ^ 1)

    @pytest.mark.fast
    def test_fixed_multiplier_ignores_severity(self, tmp_path):
        """Test that all severities use same multiplier when adaptive disabled."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=10.0,
            backoff_multiplier=2.5,  # Custom fixed multiplier
            max_backoff_before_rotate=500.0,
            adaptive_multiplier=False
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep') as mock_sleep:
            # First backoff with any error: 10 * (2.5 ^ 0) = 10
            downloader.handle_rate_limit_error("sign in")
            assert abs(mock_sleep.call_args[0][0] - 10.0) < 0.01

            # Reset and try high severity
            downloader._reset_rate_limit_backoff()
            downloader.handle_rate_limit_error("quota exceeded")
            assert abs(mock_sleep.call_args[0][0] - 10.0) < 0.01  # Same initial


class TestSeverityMetricsTracking:
    """Test that metrics track backoff events by severity."""

    @pytest.mark.fast
    def test_records_severity_in_metrics(self, tmp_path):
        """Test that backoff events are recorded by severity in metrics."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=200.0,
            adaptive_multiplier=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep'):
            # Trigger various severity levels
            downloader.handle_rate_limit_error("sign in required")  # low
            downloader.handle_rate_limit_error("429 error")  # medium
            downloader.handle_rate_limit_error("quota exceeded")  # high
            downloader.handle_rate_limit_error("quota exceeded")  # high again

        metrics = downloader.rate_limit_metrics
        assert metrics.backoff_events_by_severity.get('low', 0) == 1
        assert metrics.backoff_events_by_severity.get('medium', 0) == 1
        assert metrics.backoff_events_by_severity.get('high', 0) == 2

    @pytest.mark.fast
    def test_metrics_summary_includes_severity(self, tmp_path):
        """Test that metrics summary includes severity breakdown."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=200.0,
            adaptive_multiplier=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep'):
            downloader.handle_rate_limit_error("quota exceeded")  # high
            downloader.handle_rate_limit_error("429")  # medium

        summary = downloader.rate_limit_metrics.summary()
        assert "By severity:" in summary
        assert "high" in summary
        assert "medium" in summary

    @pytest.mark.fast
    def test_metrics_persist_severity_to_checkpoint(self, tmp_path):
        """Test that severity tracking persists through checkpoint save/load."""
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        # Create metrics with severity data
        metrics = RateLimitMetrics()
        metrics.record_backoff(5.0, severity='high')
        metrics.record_backoff(3.0, severity='medium')
        metrics.record_backoff(2.0, severity='low')
        metrics.record_backoff(4.0, severity='high')

        # Serialize and deserialize
        data = metrics.to_dict()
        restored = RateLimitMetrics.from_dict(data)

        assert restored.backoff_events_by_severity == {
            'high': 2,
            'medium': 1,
            'low': 1
        }


class TestSeverityLogging:
    """Test that severity classification is logged."""

    @pytest.mark.fast
    def test_logs_severity_classification(self, tmp_path, caplog):
        """Test that severity is logged with rate limit events."""
        import logging
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=100.0,
            adaptive_multiplier=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with caplog.at_level(logging.INFO):
            with patch('time.sleep'):
                downloader.handle_rate_limit_error("Quota exceeded for the day")

        log_text = caplog.text.lower()
        assert "high" in log_text
        assert "severity" in log_text
        assert "multiplier" in log_text

    @pytest.mark.fast
    def test_logs_multiplier_value(self, tmp_path, caplog):
        """Test that the multiplier value is logged."""
        import logging
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=100.0,
            adaptive_multiplier=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with caplog.at_level(logging.INFO):
            with patch('time.sleep'):
                downloader.handle_rate_limit_error("quota exceeded")

        log_text = caplog.text
        assert "3.0x" in log_text or "3x" in log_text

    @pytest.mark.fast
    def test_logs_severity_in_backoff_message(self, tmp_path, caplog):
        """Test that backoff log message includes severity label."""
        import logging
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=100.0,
            adaptive_multiplier=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with caplog.at_level(logging.INFO):
            with patch('time.sleep'):
                downloader.handle_rate_limit_error("sign in required")

        log_text = caplog.text
        assert "[low]" in log_text


class TestConfigOption:
    """Test the adaptive_multiplier config option."""

    @pytest.mark.fast
    def test_config_default_is_true(self):
        """Test that adaptive_multiplier defaults to True."""
        from src.config.sections.download import RateLimitConfig

        config = RateLimitConfig()
        assert config.adaptive_multiplier is True

    @pytest.mark.fast
    def test_config_can_be_disabled(self):
        """Test that adaptive_multiplier can be set to False."""
        from src.config.sections.download import RateLimitConfig

        config = RateLimitConfig(adaptive_multiplier=False)
        assert config.adaptive_multiplier is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.fast
    def test_empty_error_message(self):
        """Test classification of empty error message."""
        from src.downloader.core import classify_error_severity

        # Empty string should default to medium
        assert classify_error_severity("") == 'medium'

    @pytest.mark.fast
    def test_very_long_error_message(self):
        """Test classification with very long error message."""
        from src.downloader.core import classify_error_severity

        long_msg = "x" * 10000 + " quota exceeded " + "y" * 10000
        assert classify_error_severity(long_msg) == 'high'

    @pytest.mark.fast
    def test_none_rate_limit_config(self, tmp_path):
        """Test behavior when rate_limit config is None."""
        config = create_mock_config(tmp_path)
        config.download.rate_limit = None

        with patch('src.downloader.core.CheckpointManager'):
            with patch('src.downloader.core.TranscodingManager'):
                with patch('src.downloader.core.TitleFilter'):
                    with patch('src.downloader.core.SpeechScreener'):
                        with patch('src.downloader.core.SearchOptimizer'):
                            with patch('src.downloader.core.AudioFirstPipeline'):
                                with patch('src.downloader.core.utils.get_cookies_args', return_value=[]):
                                    from src.downloader.core import VideoDownloader
                                    downloader = VideoDownloader(config)

                                    with patch('time.sleep') as mock_sleep:
                                        # Should use default adaptive=True and classify severity
                                        result = downloader.handle_rate_limit_error("quota exceeded")
                                        assert result is True
                                        # High severity: 5 * (3 ^ 0) = 5 (defaults)
                                        assert abs(mock_sleep.call_args[0][0] - 5.0) < 0.01

    @pytest.mark.fast
    def test_multiple_patterns_match_uses_highest_severity(self):
        """Test that when multiple patterns match, highest severity wins."""
        from src.downloader.core import classify_error_severity

        # Message contains both medium and high patterns
        msg = "429 quota exceeded"  # Contains "429" (medium) and "quota exceeded" (high)
        # High is checked first, so should return high
        assert classify_error_severity(msg) == 'high'

    @pytest.mark.fast
    def test_metrics_clear_resets_severity_tracking(self):
        """Test that metrics.clear() resets severity tracking."""
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        metrics = RateLimitMetrics()
        metrics.record_backoff(5.0, severity='high')
        metrics.record_backoff(3.0, severity='medium')

        assert len(metrics.backoff_events_by_severity) == 2

        metrics.clear()

        assert metrics.backoff_events_by_severity == {}
