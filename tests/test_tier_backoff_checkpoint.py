"""
Tests for per-tier backoff state persistence to checkpoint (Sprint 12 US-003).

Verifies:
1. TierRateLimitState dict serialized to checkpoint under 'tier_backoff_state' key
2. Tier backoff state restored from checkpoint on resume with staleness de-escalation
3. Missing/malformed tier_backoff_state handled gracefully (no crash)
4. tier_backoff_state included in _save_checkpoint() alongside escalation_manager state
5. Save/restore round-trip preserves backoff delays
6. Stale checkpoint (>30min) halves backoff delays
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.downloader.core import TierRateLimitState
from src.downloader.checkpoint import DownloadCheckpoint

from src.config.sections.download import (
    DownloadConfig,
    RateLimitConfig as RealRateLimitConfig,
    ImpersonationConfig,
    ExtractorArgsConfig,
    CircuitBreakerConfig,
)


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
    mock_config.download = MagicMock(spec=DownloadConfig)
    mock_config.download.davinci_mode = False
    mock_config.download.cookies_path = ""
    mock_config.download.cookies_from_browser = ""
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


# ============================================================================
# Test DownloadCheckpoint tier_backoff_state field
# ============================================================================

class TestDownloadCheckpointTierBackoffField:
    """Test tier_backoff_state field in DownloadCheckpoint."""

    def test_field_defaults_to_none(self):
        """Test that tier_backoff_state defaults to None."""
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp="2026-01-27T10:00:00"
        )
        assert checkpoint.tier_backoff_state is None

    def test_field_serialized_in_to_dict(self):
        """Test that tier_backoff_state is included in to_dict output."""
        state_data = {
            'tiers': {'short': {'backoff_count': 2, 'total_delay': 10.0}},
            'saved_at': '2026-01-27T10:00:00'
        }
        checkpoint = DownloadCheckpoint(
            completed_keywords=[], completed_videos=[], failed_keywords=[],
            current_keyword=None, current_tier=None, timestamp="2026-01-27T10:00:00",
            tier_backoff_state=state_data
        )
        data = checkpoint.to_dict()
        assert 'tier_backoff_state' in data
        assert data['tier_backoff_state'] == state_data

    def test_from_dict_with_tier_backoff_state(self):
        """Test from_dict restores tier_backoff_state."""
        data = {
            'completed_keywords': [], 'completed_videos': [], 'failed_keywords': [],
            'current_keyword': None, 'current_tier': None, 'timestamp': '2026-01-27T10:00:00',
            'tier_backoff_state': {
                'tiers': {'long': {'backoff_count': 3, 'total_delay': 45.0}},
                'saved_at': '2026-01-27T10:00:00'
            }
        }
        checkpoint = DownloadCheckpoint.from_dict(data)
        assert checkpoint.tier_backoff_state is not None
        assert checkpoint.tier_backoff_state['tiers']['long']['backoff_count'] == 3

    def test_from_dict_missing_tier_backoff_state(self):
        """Test from_dict handles missing tier_backoff_state (backward compat)."""
        data = {
            'completed_keywords': [], 'completed_videos': [], 'failed_keywords': [],
            'current_keyword': None, 'current_tier': None, 'timestamp': '2026-01-27T10:00:00'
        }
        checkpoint = DownloadCheckpoint.from_dict(data)
        assert checkpoint.tier_backoff_state is None


# ============================================================================
# Test save checkpoint includes tier backoff state
# ============================================================================

class TestSaveCheckpointIncludesTierState:
    """Test that _save_checkpoint() serializes tier backoff state."""

    def test_save_includes_tier_backoff_state(self, tmp_path):
        """Test _save_checkpoint includes tier state when per_tier_isolation is enabled."""
        downloader = create_downloader(tmp_path)
        assert downloader._per_tier_isolation is True

        # Set up some backoff state
        downloader._tier_rate_limit_states['long'].backoff_count = 3
        downloader._tier_rate_limit_states['long'].total_delay = 45.0
        downloader._tier_rate_limit_states['short'].backoff_count = 1
        downloader._tier_rate_limit_states['short'].total_delay = 5.0

        # Create a checkpoint
        downloader.checkpoint = DownloadCheckpoint(
            completed_keywords=[], completed_videos=[], failed_keywords=[],
            current_keyword=None, current_tier=None,
            timestamp=datetime.now().isoformat()
        )

        # Call _save_checkpoint
        downloader._save_checkpoint()

        # Verify tier_backoff_state was set on checkpoint
        assert downloader.checkpoint.tier_backoff_state is not None
        tier_data = downloader.checkpoint.tier_backoff_state
        assert 'tiers' in tier_data
        assert 'saved_at' in tier_data
        assert tier_data['tiers']['long']['backoff_count'] == 3
        assert tier_data['tiers']['long']['total_delay'] == 45.0
        assert tier_data['tiers']['short']['backoff_count'] == 1

    def test_save_no_tier_state_when_isolation_disabled(self, tmp_path):
        """Test _save_checkpoint skips tier state when per_tier_isolation is disabled."""
        rate_config = MockRateLimitConfig(per_tier_isolation=False)
        downloader = create_downloader(tmp_path, rate_config)
        assert downloader._per_tier_isolation is False

        downloader.checkpoint = DownloadCheckpoint(
            completed_keywords=[], completed_videos=[], failed_keywords=[],
            current_keyword=None, current_tier=None,
            timestamp=datetime.now().isoformat()
        )

        downloader._save_checkpoint()

        # Should not have tier_backoff_state
        assert downloader.checkpoint.tier_backoff_state is None


# ============================================================================
# Test restore tier backoff state
# ============================================================================

class TestRestoreTierBackoffState:
    """Test _restore_tier_backoff_state method."""

    def test_restore_preserves_backoff_delays(self, tmp_path):
        """Test that restore preserves backoff counts and delays."""
        downloader = create_downloader(tmp_path)

        tier_state_data = {
            'tiers': {
                'short': {'backoff_count': 2, 'total_delay': 10.0, 'in_recovery': False, 'last_event_time': None},
                'medium': {'backoff_count': 0, 'total_delay': 0.0, 'in_recovery': False, 'last_event_time': None},
                'long': {'backoff_count': 5, 'total_delay': 75.0, 'in_recovery': True, 'last_event_time': '2026-01-27T10:00:00'},
                'longer': {'backoff_count': 0, 'total_delay': 0.0, 'in_recovery': False, 'last_event_time': None},
            },
            'saved_at': datetime.now().isoformat()  # Recent — no staleness reduction
        }

        downloader._restore_tier_backoff_state(tier_state_data)

        assert downloader._tier_rate_limit_states['short'].backoff_count == 2
        assert downloader._tier_rate_limit_states['short'].total_delay == 10.0
        assert downloader._tier_rate_limit_states['long'].backoff_count == 5
        assert downloader._tier_rate_limit_states['long'].total_delay == 75.0
        assert downloader._tier_rate_limit_states['long'].in_recovery is True
        assert downloader._tier_rate_limit_states['medium'].backoff_count == 0
        assert downloader._tier_rate_limit_states['longer'].backoff_count == 0

    def test_restore_stale_checkpoint_halves_delays(self, tmp_path):
        """Test that stale checkpoint (>30min) halves backoff delays."""
        downloader = create_downloader(tmp_path)

        # Create state data saved 45 minutes ago
        saved_time = datetime.now() - timedelta(minutes=45)
        tier_state_data = {
            'tiers': {
                'short': {'backoff_count': 2, 'total_delay': 20.0, 'in_recovery': False, 'last_event_time': None},
                'long': {'backoff_count': 4, 'total_delay': 100.0, 'in_recovery': True, 'last_event_time': None},
            },
            'saved_at': saved_time.isoformat()
        }

        downloader._restore_tier_backoff_state(tier_state_data)

        # Delays should be halved due to staleness
        assert downloader._tier_rate_limit_states['short'].total_delay == 10.0  # 20.0 * 0.5
        assert downloader._tier_rate_limit_states['long'].total_delay == 50.0   # 100.0 * 0.5
        # Backoff counts preserved (only delay is de-escalated)
        assert downloader._tier_rate_limit_states['short'].backoff_count == 2
        assert downloader._tier_rate_limit_states['long'].backoff_count == 4

    def test_restore_exactly_30min_not_stale(self, tmp_path):
        """Test that checkpoint exactly 30min old is not considered stale."""
        downloader = create_downloader(tmp_path)

        saved_time = datetime.now() - timedelta(minutes=30)
        tier_state_data = {
            'tiers': {
                'short': {'backoff_count': 2, 'total_delay': 20.0, 'in_recovery': False, 'last_event_time': None},
            },
            'saved_at': saved_time.isoformat()
        }

        downloader._restore_tier_backoff_state(tier_state_data)

        # 30min exactly should NOT trigger de-escalation (only >30 triggers it)
        assert downloader._tier_rate_limit_states['short'].total_delay == 20.0

    def test_restore_handles_missing_tiers_key(self, tmp_path):
        """Test graceful handling of missing 'tiers' key."""
        downloader = create_downloader(tmp_path)

        # No 'tiers' key
        tier_state_data = {
            'saved_at': datetime.now().isoformat()
        }

        # Should not crash
        downloader._restore_tier_backoff_state(tier_state_data)

        # State should remain at defaults
        assert downloader._tier_rate_limit_states['short'].backoff_count == 0

    def test_restore_handles_none_data(self, tmp_path):
        """Test graceful handling of None values."""
        downloader = create_downloader(tmp_path)

        # Empty dict
        downloader._restore_tier_backoff_state({})

        # State should remain at defaults
        assert downloader._tier_rate_limit_states['short'].backoff_count == 0

    def test_restore_handles_malformed_saved_at(self, tmp_path, caplog):
        """Test graceful handling of invalid saved_at timestamp."""
        downloader = create_downloader(tmp_path)

        tier_state_data = {
            'tiers': {
                'short': {'backoff_count': 3, 'total_delay': 30.0, 'in_recovery': False, 'last_event_time': None},
            },
            'saved_at': 'not-a-valid-timestamp'
        }

        with caplog.at_level(logging.WARNING):
            downloader._restore_tier_backoff_state(tier_state_data)

        # Should warn and start fresh
        assert "Invalid saved_at" in caplog.text
        # State remains at default (fresh)
        assert downloader._tier_rate_limit_states['short'].backoff_count == 0

    def test_restore_handles_unknown_tier_names(self, tmp_path):
        """Test that unknown tier names in checkpoint are ignored."""
        downloader = create_downloader(tmp_path)

        tier_state_data = {
            'tiers': {
                'short': {'backoff_count': 2, 'total_delay': 10.0, 'in_recovery': False, 'last_event_time': None},
                'unknown_tier': {'backoff_count': 99, 'total_delay': 999.0, 'in_recovery': False, 'last_event_time': None},
            },
            'saved_at': datetime.now().isoformat()
        }

        downloader._restore_tier_backoff_state(tier_state_data)

        # Known tier restored
        assert downloader._tier_rate_limit_states['short'].backoff_count == 2
        # Unknown tier not added
        assert 'unknown_tier' not in downloader._tier_rate_limit_states

    def test_restore_handles_corrupt_tier_data(self, tmp_path, caplog):
        """Test graceful handling of corrupt tier data that causes exceptions."""
        downloader = create_downloader(tmp_path)

        tier_state_data = {
            'tiers': "not-a-dict",  # Should be a dict
            'saved_at': datetime.now().isoformat()
        }

        with caplog.at_level(logging.WARNING):
            downloader._restore_tier_backoff_state(tier_state_data)

        assert "Could not restore tier backoff state" in caplog.text
        # State should remain at defaults
        assert downloader._tier_rate_limit_states['short'].backoff_count == 0

    def test_restore_logs_active_tiers(self, tmp_path, caplog):
        """Test that restore logs the number of active tiers."""
        downloader = create_downloader(tmp_path)

        tier_state_data = {
            'tiers': {
                'short': {'backoff_count': 2, 'total_delay': 10.0, 'in_recovery': False, 'last_event_time': None},
                'long': {'backoff_count': 1, 'total_delay': 5.0, 'in_recovery': False, 'last_event_time': None},
            },
            'saved_at': datetime.now().isoformat()
        }

        with caplog.at_level(logging.INFO):
            downloader._restore_tier_backoff_state(tier_state_data)

        assert "2 tiers with active backoff" in caplog.text


# ============================================================================
# Test round-trip save/restore
# ============================================================================

class TestTierBackoffRoundTrip:
    """Test full save → restore round-trip."""

    def test_round_trip_preserves_state(self, tmp_path):
        """Test that save then restore preserves all tier backoff state."""
        downloader = create_downloader(tmp_path)

        # Build up state via handle_rate_limit_error
        with patch('time.sleep'):
            downloader.handle_rate_limit_error("429", tier='short')
            downloader.handle_rate_limit_error("429", tier='short')
            downloader.handle_rate_limit_error("429", tier='long')

        # Capture current state
        original_short = downloader._tier_rate_limit_states['short'].to_dict()
        original_long = downloader._tier_rate_limit_states['long'].to_dict()

        # Create checkpoint and save
        downloader.checkpoint = DownloadCheckpoint(
            completed_keywords=[], completed_videos=[], failed_keywords=[],
            current_keyword=None, current_tier=None,
            timestamp=datetime.now().isoformat()
        )
        downloader._save_checkpoint()

        # Get the saved tier state
        saved_state = downloader.checkpoint.tier_backoff_state

        # Create a fresh downloader and restore
        downloader2 = create_downloader(tmp_path)
        downloader2._restore_tier_backoff_state(saved_state)

        # Verify state matches
        assert downloader2._tier_rate_limit_states['short'].backoff_count == original_short['backoff_count']
        assert downloader2._tier_rate_limit_states['short'].total_delay == original_short['total_delay']
        assert downloader2._tier_rate_limit_states['long'].backoff_count == original_long['backoff_count']
        assert downloader2._tier_rate_limit_states['long'].total_delay == original_long['total_delay']
        # Medium and longer should be clean
        assert downloader2._tier_rate_limit_states['medium'].backoff_count == 0
        assert downloader2._tier_rate_limit_states['longer'].backoff_count == 0

    def test_stale_round_trip_halves_delays(self, tmp_path):
        """Test round-trip with stale checkpoint halves total_delay."""
        downloader = create_downloader(tmp_path)

        with patch('time.sleep'):
            downloader.handle_rate_limit_error("429", tier='long')
            downloader.handle_rate_limit_error("429", tier='long')
            downloader.handle_rate_limit_error("429", tier='long')

        original_delay = downloader._tier_rate_limit_states['long'].total_delay
        original_count = downloader._tier_rate_limit_states['long'].backoff_count

        # Save checkpoint
        downloader.checkpoint = DownloadCheckpoint(
            completed_keywords=[], completed_videos=[], failed_keywords=[],
            current_keyword=None, current_tier=None,
            timestamp=datetime.now().isoformat()
        )
        downloader._save_checkpoint()

        # Manually age the saved_at timestamp to 45 minutes ago
        saved_state = downloader.checkpoint.tier_backoff_state
        saved_state['saved_at'] = (datetime.now() - timedelta(minutes=45)).isoformat()

        # Restore into fresh downloader
        downloader2 = create_downloader(tmp_path)
        downloader2._restore_tier_backoff_state(saved_state)

        # Delay should be halved, count preserved
        assert downloader2._tier_rate_limit_states['long'].total_delay == pytest.approx(original_delay * 0.5)
        assert downloader2._tier_rate_limit_states['long'].backoff_count == original_count


# ============================================================================
# Test no-saved_at still works
# ============================================================================

class TestMissingSavedAt:
    """Test behavior when saved_at is missing or None."""

    def test_missing_saved_at_no_de_escalation(self, tmp_path):
        """Test that missing saved_at means no staleness factor (assume fresh)."""
        downloader = create_downloader(tmp_path)

        tier_state_data = {
            'tiers': {
                'short': {'backoff_count': 3, 'total_delay': 30.0, 'in_recovery': False, 'last_event_time': None},
            }
            # No 'saved_at' key
        }

        downloader._restore_tier_backoff_state(tier_state_data)

        # No staleness de-escalation — full delay preserved
        assert downloader._tier_rate_limit_states['short'].total_delay == 30.0
        assert downloader._tier_rate_limit_states['short'].backoff_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
