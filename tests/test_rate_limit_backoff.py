"""
Tests for progressive rate limit backoff before cookie rotation (US-002).

The rate limit backoff mechanism:
1. On rate limit error, applies exponential backoff before rotating cookies
2. Uses initial_backoff_seconds * (backoff_multiplier ^ attempt) for delays
3. After max_backoff_before_rotate total delay, escalates to cookie rotation
4. Resets backoff state on successful download
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.config.sections.download import (
    DownloadConfig,
    RateLimitConfig as RealRateLimitConfig,
    ImpersonationConfig,
    ExtractorArgsConfig,
    RateLimitBudgetConfig,
    CircuitBreakerConfig,
)


@dataclass
class MockRateLimitConfig:
    """Mock config for rate limit backoff testing."""
    initial_backoff_seconds: float = 5.0
    max_backoff_before_rotate: float = 60.0
    backoff_multiplier: float = 2.0


def create_mock_config(tmp_path, rate_limit_config=None, **overrides):
    """Create a mock config for testing.

    Uses spec=DownloadConfig on the download mock to catch phantom attributes.
    Attributes that should NOT exist are explicitly set to None.
    """
    mock_config = MagicMock()
    mock_config.cache_dir = str(tmp_path / ".cache")
    mock_config.downloaded_videos_dir = str(tmp_path / "videos")
    mock_config.download = MagicMock(spec=DownloadConfig)
    mock_config.download.davinci_mode = False
    mock_config.download.cookies_path = ""  # Real attribute (was 'cookies' - phantom)
    mock_config.download.cookies_from_browser = ""  # Real attribute, use empty string not None
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
    mock_config.download.impersonation = MagicMock(spec=ImpersonationConfig)
    mock_config.download.impersonation.enabled = False
    mock_config.download.extractor_args = MagicMock(spec=ExtractorArgsConfig)
    mock_config.download.extractor_args.enabled = False
    mock_config.download.circuit_breaker = MagicMock(spec=CircuitBreakerConfig)
    mock_config.download.circuit_breaker.enabled = False
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


class TestRateLimitConfigAccess:
    """Test that rate limit config is correctly accessed."""

    def test_uses_config_initial_backoff(self, tmp_path):
        """Test that initial_backoff_seconds is read from config."""
        rate_config = MockRateLimitConfig(initial_backoff_seconds=10.0)
        downloader = create_downloader(tmp_path, rate_config)
        assert downloader.download_config.rate_limit.initial_backoff_seconds == 10.0

    def test_uses_config_max_backoff(self, tmp_path):
        """Test that max_backoff_before_rotate is read from config."""
        rate_config = MockRateLimitConfig(max_backoff_before_rotate=120.0)
        downloader = create_downloader(tmp_path, rate_config)
        assert downloader.download_config.rate_limit.max_backoff_before_rotate == 120.0

    def test_uses_config_backoff_multiplier(self, tmp_path):
        """Test that backoff_multiplier is read from config."""
        rate_config = MockRateLimitConfig(backoff_multiplier=3.0)
        downloader = create_downloader(tmp_path, rate_config)
        assert downloader.download_config.rate_limit.backoff_multiplier == 3.0


class TestBackoffInitialization:
    """Test that backoff state is properly initialized."""

    def test_backoff_count_starts_at_zero(self, tmp_path):
        """Test that backoff count is initialized to zero."""
        downloader = create_downloader(tmp_path)
        assert downloader._rate_limit_backoff_count == 0

    def test_total_delay_starts_at_zero(self, tmp_path):
        """Test that total delay is initialized to zero."""
        downloader = create_downloader(tmp_path)
        assert downloader._rate_limit_total_delay == 0.0


class TestProgressiveBackoff:
    """Test the progressive backoff behavior."""

    def test_first_backoff_uses_initial_delay(self, tmp_path):
        """Test that first rate limit uses initial_backoff_seconds."""
        rate_config = MockRateLimitConfig(initial_backoff_seconds=5.0)
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep') as mock_sleep:
            result = downloader.handle_rate_limit_error("429 Too Many Requests")

            assert result is True
            mock_sleep.assert_called_once()
            assert abs(mock_sleep.call_args[0][0] - 5.0) < 0.01

    def test_backoff_is_exponential(self, tmp_path):
        """Test that backoff delay doubles each attempt."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=2.0,
            backoff_multiplier=2.0,
            max_backoff_before_rotate=100.0
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep') as mock_sleep:
            # First call: 2.0 * (2.0 ^ 0) = 2.0
            downloader.handle_rate_limit_error("429")
            assert abs(mock_sleep.call_args[0][0] - 2.0) < 0.01

            # Second call: 2.0 * (2.0 ^ 1) = 4.0
            downloader.handle_rate_limit_error("429")
            assert abs(mock_sleep.call_args[0][0] - 4.0) < 0.01

            # Third call: 2.0 * (2.0 ^ 2) = 8.0
            downloader.handle_rate_limit_error("429")
            assert abs(mock_sleep.call_args[0][0] - 8.0) < 0.01

    def test_backoff_count_increments(self, tmp_path):
        """Test that backoff count increments with each call."""
        rate_config = MockRateLimitConfig(max_backoff_before_rotate=100.0)
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep'):
            assert downloader._rate_limit_backoff_count == 0

            downloader.handle_rate_limit_error("429")
            assert downloader._rate_limit_backoff_count == 1

            downloader.handle_rate_limit_error("429")
            assert downloader._rate_limit_backoff_count == 2

    def test_total_delay_accumulates(self, tmp_path):
        """Test that total delay accumulates across calls."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            backoff_multiplier=2.0,
            max_backoff_before_rotate=100.0
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep'):
            # First: 5s delay, total = 5
            downloader.handle_rate_limit_error("429")
            assert downloader._rate_limit_total_delay == 5.0

            # Second: 10s delay, total = 15
            downloader.handle_rate_limit_error("429")
            assert downloader._rate_limit_total_delay == 15.0

            # Third: 20s delay, total = 35
            downloader.handle_rate_limit_error("429")
            assert downloader._rate_limit_total_delay == 35.0


class TestBackoffEscalation:
    """Test that backoff escalates to cookie rotation after max delay."""

    def test_escalates_after_max_backoff(self, tmp_path):
        """Test that cookie rotation is triggered after max backoff."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=30.0,
            max_backoff_before_rotate=60.0,
            backoff_multiplier=2.0
        )
        downloader = create_downloader(tmp_path, rate_config)
        downloader.cookie_rotator = MagicMock()
        downloader.cookie_rotator.should_rotate.return_value = True
        downloader.cookie_rotator.rotate.return_value = "/path/to/cookie.txt"

        with patch('time.sleep'):
            # First call: 30s delay (total 30, under max)
            result = downloader.handle_rate_limit_error("429")
            assert result is True
            assert downloader.cookie_rotator.rotate.call_count == 0

            # Second call: 60s delay would exceed max, cap at 30 more (total 60)
            result = downloader.handle_rate_limit_error("429")
            assert result is True
            assert downloader.cookie_rotator.rotate.call_count == 0

            # Third call: backoff exhausted, escalate to rotation
            result = downloader.handle_rate_limit_error("429")
            assert result is True
            assert downloader.cookie_rotator.rotate.call_count == 1

    def test_delay_capped_at_remaining(self, tmp_path):
        """Test that delay is capped so total doesn't exceed max."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=40.0,
            max_backoff_before_rotate=60.0,
            backoff_multiplier=2.0
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep') as mock_sleep:
            # First call: 40s delay
            downloader.handle_rate_limit_error("429")
            assert abs(mock_sleep.call_args[0][0] - 40.0) < 0.01

            # Second call: would be 80s but capped at remaining 20s
            downloader.handle_rate_limit_error("429")
            assert abs(mock_sleep.call_args[0][0] - 20.0) < 0.01


class TestBackoffReset:
    """Test that backoff state resets correctly."""

    def test_reset_clears_count(self, tmp_path):
        """Test that reset clears backoff count."""
        downloader = create_downloader(tmp_path)
        downloader._rate_limit_backoff_count = 5
        downloader._reset_rate_limit_backoff()
        assert downloader._rate_limit_backoff_count == 0

    def test_reset_clears_total_delay(self, tmp_path):
        """Test that reset clears total delay."""
        downloader = create_downloader(tmp_path)
        downloader._rate_limit_total_delay = 45.0
        downloader._reset_rate_limit_backoff()
        assert downloader._rate_limit_total_delay == 0.0

    def test_backoff_resets_before_rotation(self, tmp_path):
        """Test that backoff is reset before cookie rotation."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=5.0  # Immediate escalation after first backoff
        )
        downloader = create_downloader(tmp_path, rate_config)
        downloader.cookie_rotator = MagicMock()
        downloader.cookie_rotator.should_rotate.return_value = True
        downloader.cookie_rotator.rotate.return_value = "/path/to/cookie.txt"

        with patch('time.sleep'):
            # First call: uses up all backoff
            downloader.handle_rate_limit_error("429")
            assert downloader._rate_limit_total_delay == 5.0

            # Second call: triggers rotation and resets
            downloader.handle_rate_limit_error("429")
            assert downloader._rate_limit_backoff_count == 0
            assert downloader._rate_limit_total_delay == 0.0


class TestBackoffLogging:
    """Test that backoff progress is logged."""

    def test_backoff_logged_with_progress(self, tmp_path, caplog):
        """Test that backoff logs include attempt number and total delay."""
        import logging
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=100.0
        )
        downloader = create_downloader(tmp_path, rate_config)

        with caplog.at_level(logging.INFO):
            with patch('time.sleep'):
                downloader.handle_rate_limit_error("429")

                log_text = caplog.text.lower()
                assert "backoff" in log_text
                assert "waiting" in log_text

    def test_escalation_logged(self, tmp_path, caplog):
        """Test that escalation to cookie rotation is logged."""
        import logging
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=5.0
        )
        downloader = create_downloader(tmp_path, rate_config)
        downloader.cookie_rotator = MagicMock()
        downloader.cookie_rotator.should_rotate.return_value = True
        downloader.cookie_rotator.rotate.return_value = "/path/to/cookie.txt"

        with caplog.at_level(logging.INFO):
            with patch('time.sleep'):
                # First call uses up backoff
                downloader.handle_rate_limit_error("429")
                # Second call triggers escalation
                downloader.handle_rate_limit_error("429")

                log_text = caplog.text.lower()
                assert "exhausted" in log_text or "escalating" in log_text


class TestBackoffWithCookieRotation:
    """Test backoff interaction with cookie rotation."""

    def test_backoff_before_rotation_no_rotator(self, tmp_path):
        """Test backoff works when cookie rotator is not enabled."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=60.0
        )
        downloader = create_downloader(tmp_path, rate_config)
        downloader.cookie_rotator = None

        with patch('time.sleep') as mock_sleep:
            result = downloader.handle_rate_limit_error("429")

            assert result is True
            mock_sleep.assert_called_once()

    def test_rotation_only_after_backoff_exhausted(self, tmp_path):
        """Test that rotation is only called after backoff is exhausted."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=10.0,
            max_backoff_before_rotate=25.0,
            backoff_multiplier=2.0
        )
        downloader = create_downloader(tmp_path, rate_config)
        downloader.cookie_rotator = MagicMock()
        downloader.cookie_rotator.should_rotate.return_value = True
        downloader.cookie_rotator.rotate.return_value = "/path/to/cookie.txt"

        with patch('time.sleep'):
            # First: 10s delay (total 10)
            downloader.handle_rate_limit_error("429")
            assert downloader.cookie_rotator.rotate.call_count == 0

            # Second: 15s delay capped (total 25, at max)
            downloader.handle_rate_limit_error("429")
            assert downloader.cookie_rotator.rotate.call_count == 0

            # Third: backoff exhausted, rotate
            downloader.handle_rate_limit_error("429")
            assert downloader.cookie_rotator.rotate.call_count == 1


class TestBackoffWithVPN:
    """Test backoff interaction with VPN switching."""

    def test_vpn_only_after_cookies_exhausted(self, tmp_path):
        """Test that VPN switch is only after cookies exhausted."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=5.0
        )
        downloader = create_downloader(tmp_path, rate_config)
        downloader.cookie_rotator = MagicMock()
        downloader.cookie_rotator.should_rotate.return_value = True
        downloader.cookie_rotator.rotate.return_value = None  # Cookies exhausted
        downloader.vpn_manager = MagicMock()
        downloader.vpn_manager.can_switch.return_value = True
        downloader.vpn_manager.switch.return_value = True

        with patch('time.sleep'):
            # First call: uses backoff
            result = downloader.handle_rate_limit_error("429")
            assert result is True
            assert downloader.vpn_manager.switch.call_count == 0

            # Second call: backoff exhausted, cookies fail, VPN called
            result = downloader.handle_rate_limit_error("429")
            assert result is True
            assert downloader.vpn_manager.switch.call_count == 1


class TestNoRateLimitConfig:
    """Test fallback behavior when rate_limit config is missing."""

    def test_uses_defaults_when_config_missing(self, tmp_path):
        """Test that default values are used when rate_limit config is None."""
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
                                        result = downloader.handle_rate_limit_error("429")
                                        assert result is True
                                        # Should use default 5.0 seconds
                                        assert abs(mock_sleep.call_args[0][0] - 5.0) < 0.01
