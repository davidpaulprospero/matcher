"""
Tests for per-tier rate limit state isolation (US-001).

When downloading multiple tiers simultaneously, a rate limit on 'long' tier
should not affect 'short' tier backoff. This test suite verifies:
1. TierRateLimitState class tracks backoff state per tier
2. RateLimitConfig has per_tier_isolation option (default: true)
3. handle_rate_limit_error() uses tier-specific state when isolation enabled
4. _reset_rate_limit_backoff() resets only the affected tier's state
5. Rate limit metrics track events per tier in tier_rate_limit_events dict
6. Tier A rate limit does not affect tier B backoff
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@dataclass
class MockRateLimitConfig:
    """Mock config for rate limit testing."""
    initial_backoff_seconds: float = 5.0
    max_backoff_before_rotate: float = 60.0
    backoff_multiplier: float = 2.0
    resume_cooldown_minutes: float = 15.0
    per_tier_isolation: bool = True


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


class TestTierRateLimitState:
    """Test the TierRateLimitState dataclass."""

    def test_state_initializes_with_zeros(self):
        """Test that state initializes with default zero values."""
        from src.downloader.core import TierRateLimitState

        state = TierRateLimitState()
        assert state.backoff_count == 0
        assert state.total_delay == 0.0
        assert state.in_recovery is False
        assert state.last_event_time is None

    def test_state_reset_clears_values(self):
        """Test that reset() clears backoff state but not recovery mode."""
        from src.downloader.core import TierRateLimitState

        state = TierRateLimitState(
            backoff_count=5,
            total_delay=45.0,
            in_recovery=True,
            last_event_time="2026-01-25T10:00:00"
        )
        state.reset()

        assert state.backoff_count == 0
        assert state.total_delay == 0.0
        # in_recovery is NOT reset - it's session-level
        assert state.in_recovery is True

    def test_state_to_dict(self):
        """Test serialization to dictionary."""
        from src.downloader.core import TierRateLimitState

        state = TierRateLimitState(
            backoff_count=3,
            total_delay=15.0,
            in_recovery=True,
            last_event_time="2026-01-25T10:00:00"
        )
        d = state.to_dict()

        assert d == {
            'backoff_count': 3,
            'total_delay': 15.0,
            'in_recovery': True,
            'last_event_time': "2026-01-25T10:00:00"
        }

    def test_state_from_dict(self):
        """Test deserialization from dictionary."""
        from src.downloader.core import TierRateLimitState

        d = {
            'backoff_count': 2,
            'total_delay': 10.0,
            'in_recovery': False,
            'last_event_time': "2026-01-25T09:00:00"
        }
        state = TierRateLimitState.from_dict(d)

        assert state.backoff_count == 2
        assert state.total_delay == 10.0
        assert state.in_recovery is False
        assert state.last_event_time == "2026-01-25T09:00:00"

    def test_state_from_dict_handles_empty(self):
        """Test that from_dict handles empty dict."""
        from src.downloader.core import TierRateLimitState

        state = TierRateLimitState.from_dict({})
        assert state.backoff_count == 0
        assert state.total_delay == 0.0

    def test_state_from_dict_handles_none(self):
        """Test that from_dict handles None."""
        from src.downloader.core import TierRateLimitState

        state = TierRateLimitState.from_dict(None)
        assert state.backoff_count == 0


class TestPerTierIsolationConfig:
    """Test per_tier_isolation configuration option."""

    def test_config_defaults_to_true(self, tmp_path):
        """Test that per_tier_isolation defaults to True."""
        rate_config = MockRateLimitConfig()
        downloader = create_downloader(tmp_path, rate_config)

        assert downloader._per_tier_isolation is True

    def test_config_can_be_disabled(self, tmp_path):
        """Test that per_tier_isolation can be set to False."""
        rate_config = MockRateLimitConfig(per_tier_isolation=False)
        downloader = create_downloader(tmp_path, rate_config)

        assert downloader._per_tier_isolation is False

    def test_tier_states_initialized_for_all_tiers(self, tmp_path):
        """Test that tier states are initialized for all duration tiers."""
        downloader = create_downloader(tmp_path)

        assert 'short' in downloader._tier_rate_limit_states
        assert 'medium' in downloader._tier_rate_limit_states
        assert 'long' in downloader._tier_rate_limit_states
        assert 'longer' in downloader._tier_rate_limit_states


class TestTierIsolatedBackoff:
    """Test that rate limits are isolated per tier."""

    def test_tier_a_backoff_does_not_affect_tier_b(self, tmp_path):
        """Test that rate limit on tier A doesn't affect tier B backoff state."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=100.0,
            per_tier_isolation=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep'):
            # Trigger rate limit on 'long' tier multiple times
            downloader.handle_rate_limit_error("429", tier='long')
            downloader.handle_rate_limit_error("429", tier='long')
            downloader.handle_rate_limit_error("429", tier='long')

            # Check 'long' tier has accumulated state
            long_state = downloader._tier_rate_limit_states['long']
            assert long_state.backoff_count == 3
            assert long_state.total_delay > 0

            # Check 'short' tier is unaffected
            short_state = downloader._tier_rate_limit_states['short']
            assert short_state.backoff_count == 0
            assert short_state.total_delay == 0.0

    def test_each_tier_tracks_independently(self, tmp_path):
        """Test that each tier maintains independent backoff counters."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=100.0,
            per_tier_isolation=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep'):
            # Rate limit on different tiers
            downloader.handle_rate_limit_error("429", tier='short')
            downloader.handle_rate_limit_error("429", tier='short')
            downloader.handle_rate_limit_error("429", tier='medium')
            downloader.handle_rate_limit_error("429", tier='long')
            downloader.handle_rate_limit_error("429", tier='long')
            downloader.handle_rate_limit_error("429", tier='long')

            assert downloader._tier_rate_limit_states['short'].backoff_count == 2
            assert downloader._tier_rate_limit_states['medium'].backoff_count == 1
            assert downloader._tier_rate_limit_states['long'].backoff_count == 3
            assert downloader._tier_rate_limit_states['longer'].backoff_count == 0

    def test_isolation_disabled_uses_global_state(self, tmp_path):
        """Test that when per_tier_isolation is False, global state is used."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            max_backoff_before_rotate=100.0,
            per_tier_isolation=False
        )
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep'):
            # Rate limit on 'long' tier
            downloader.handle_rate_limit_error("429", tier='long')
            downloader.handle_rate_limit_error("429", tier='long')

            # Global state should be updated
            assert downloader._rate_limit_backoff_count == 2
            assert downloader._rate_limit_total_delay > 0

            # Tier states should remain at zero (unused)
            assert downloader._tier_rate_limit_states['long'].backoff_count == 0


class TestTierSpecificReset:
    """Test that reset only affects the specified tier."""

    def test_reset_only_affects_specified_tier(self, tmp_path):
        """Test that _reset_rate_limit_backoff resets only the affected tier."""
        rate_config = MockRateLimitConfig(per_tier_isolation=True)
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep'):
            # Build up state in multiple tiers
            downloader.handle_rate_limit_error("429", tier='short')
            downloader.handle_rate_limit_error("429", tier='short')
            downloader.handle_rate_limit_error("429", tier='long')
            downloader.handle_rate_limit_error("429", tier='long')

            # Reset only 'long' tier
            downloader._reset_rate_limit_backoff(tier='long')

            # 'short' should be unaffected
            assert downloader._tier_rate_limit_states['short'].backoff_count == 2
            assert downloader._tier_rate_limit_states['short'].total_delay > 0

            # 'long' should be reset
            assert downloader._tier_rate_limit_states['long'].backoff_count == 0
            assert downloader._tier_rate_limit_states['long'].total_delay == 0.0

    def test_reset_without_tier_resets_global(self, tmp_path):
        """Test that reset without tier parameter resets global state."""
        rate_config = MockRateLimitConfig(per_tier_isolation=False)
        downloader = create_downloader(tmp_path, rate_config)

        downloader._rate_limit_backoff_count = 5
        downloader._rate_limit_total_delay = 45.0

        downloader._reset_rate_limit_backoff()

        assert downloader._rate_limit_backoff_count == 0
        assert downloader._rate_limit_total_delay == 0.0


class TestTierRateLimitMetrics:
    """Test per-tier tracking in metrics."""

    def test_metrics_track_events_per_tier(self, tmp_path):
        """Test that metrics track rate limit events per tier."""
        rate_config = MockRateLimitConfig(per_tier_isolation=True)
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep'):
            downloader.handle_rate_limit_error("429", tier='short')
            downloader.handle_rate_limit_error("429", tier='short')
            downloader.handle_rate_limit_error("429", tier='long')

            metrics = downloader.rate_limit_metrics
            assert metrics.tier_rate_limit_events.get('short', 0) == 2
            assert metrics.tier_rate_limit_events.get('long', 0) == 1
            assert metrics.tier_rate_limit_events.get('medium', 0) == 0

    def test_metrics_total_includes_all_tiers(self, tmp_path):
        """Test that total rate limit events includes all tiers."""
        rate_config = MockRateLimitConfig(per_tier_isolation=True)
        downloader = create_downloader(tmp_path, rate_config)

        with patch('time.sleep'):
            downloader.handle_rate_limit_error("429", tier='short')
            downloader.handle_rate_limit_error("429", tier='medium')
            downloader.handle_rate_limit_error("429", tier='long')

            assert downloader.rate_limit_metrics.rate_limit_events == 3


class TestMetricsTierDict:
    """Test tier_rate_limit_events dict in RateLimitMetrics."""

    def test_record_rate_limit_event_with_tier(self):
        """Test that record_rate_limit_event accepts tier parameter."""
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(tier='short')
        metrics.record_rate_limit_event(tier='short')
        metrics.record_rate_limit_event(tier='long')

        assert metrics.tier_rate_limit_events == {'short': 2, 'long': 1}
        assert metrics.rate_limit_events == 3

    def test_record_rate_limit_event_without_tier(self):
        """Test that record_rate_limit_event works without tier."""
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event()
        metrics.record_rate_limit_event()

        assert metrics.rate_limit_events == 2
        assert metrics.tier_rate_limit_events == {}

    def test_tier_events_in_summary(self):
        """Test that tier events appear in summary output."""
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(tier='short')
        metrics.record_rate_limit_event(tier='long')
        metrics.record_rate_limit_event(tier='long')

        summary = metrics.summary()
        assert "By tier:" in summary
        assert "long: 2" in summary
        assert "short: 1" in summary

    def test_tier_events_persistence(self):
        """Test that tier events are saved/loaded correctly."""
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        # Create metrics with tier events
        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(tier='short')
        metrics.record_rate_limit_event(tier='long')

        # Save and restore
        data = metrics.to_dict()
        restored = RateLimitMetrics.from_dict(data)

        assert restored.tier_rate_limit_events == {'short': 1, 'long': 1}

    def test_tier_events_cleared_on_reset(self):
        """Test that clear() clears tier events."""
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        metrics = RateLimitMetrics()
        metrics.record_rate_limit_event(tier='short')
        metrics.record_rate_limit_event(tier='long')

        metrics.clear()

        assert metrics.tier_rate_limit_events == {}


class TestTierBackoffLogging:
    """Test logging includes tier information."""

    def test_backoff_log_includes_tier(self, tmp_path, caplog):
        """Test that backoff log messages include tier label."""
        import logging
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=5.0,
            per_tier_isolation=True
        )
        downloader = create_downloader(tmp_path, rate_config)

        with caplog.at_level(logging.INFO):
            with patch('time.sleep'):
                downloader.handle_rate_limit_error("429", tier='long')

                log_text = caplog.text
                assert "[long]" in log_text


class TestTierWithCookieRotation:
    """Test tier isolation with cookie rotation escalation."""

    def test_tier_escalation_independent(self, tmp_path):
        """Test that one tier can escalate to rotation without affecting others."""
        rate_config = MockRateLimitConfig(
            initial_backoff_seconds=30.0,
            max_backoff_before_rotate=60.0,
            per_tier_isolation=True
        )
        downloader = create_downloader(tmp_path, rate_config)
        downloader.cookie_rotator = MagicMock()
        downloader.cookie_rotator.should_rotate.return_value = True
        downloader.cookie_rotator.rotate.return_value = "/path/to/cookie.txt"

        with patch('time.sleep'):
            # 'long' tier: 2 backoffs (30 + 30 = 60) then rotation
            downloader.handle_rate_limit_error("429", tier='long')  # 30s
            downloader.handle_rate_limit_error("429", tier='long')  # 30s
            downloader.handle_rate_limit_error("429", tier='long')  # rotation

            # 'short' tier: still at beginning
            downloader.handle_rate_limit_error("429", tier='short')  # 30s

            # 'long' should have triggered rotation
            assert downloader.cookie_rotator.rotate.call_count == 1

            # 'short' should still be in backoff mode
            assert downloader._tier_rate_limit_states['short'].backoff_count == 1
            assert downloader._tier_rate_limit_states['short'].total_delay == 30.0


class TestConfigYamlIntegration:
    """Test config.yaml includes per_tier_isolation setting."""

    def test_dataclass_has_per_tier_isolation(self):
        """Test that RateLimitConfig dataclass has per_tier_isolation field."""
        from src.config.sections.download import RateLimitConfig

        config = RateLimitConfig()
        assert hasattr(config, 'per_tier_isolation')
        assert config.per_tier_isolation is True  # Default

    def test_dataclass_per_tier_isolation_configurable(self):
        """Test that per_tier_isolation can be set."""
        from src.config.sections.download import RateLimitConfig

        config = RateLimitConfig(per_tier_isolation=False)
        assert config.per_tier_isolation is False
