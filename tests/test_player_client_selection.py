"""
Tests for dynamic extractor-args player_client selection (US-123-003).

Tests:
- Success rate tracking per player_client
- get_best_extractor_args() method selection
- Config options: success_rate_window, extractor_args_fallback_order
- Selection reset on config reload
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.config.sections.download import ExtractorArgsConfig
from src.downloader.escalation_manager import (
    PlayerClientTracker,
    EscalationManager,
)
from src.downloader.types import EscalationState, EscalationTier


class TestPlayerClientTracker:
    """Tests for PlayerClientTracker class."""

    def test_record_attempt_and_success_rate(self):
        """Test recording attempts and calculating success rates."""
        tracker = PlayerClientTracker(enabled=True, window_size=10, min_samples=3)

        # Record some successes
        tracker.record_attempt('web_safari', success=True)
        tracker.record_attempt('web_safari', success=True)
        tracker.record_attempt('web_safari', success=False)
        tracker.record_attempt('web_safari', success=True)

        # Should have 75% success rate (3/4)
        assert tracker.get_success_rate('web_safari') == 0.75
        assert tracker.get_sample_count('web_safari') == 4

    def test_not_enough_samples(self):
        """Test that not enough samples returns None."""
        tracker = PlayerClientTracker(enabled=True, min_samples=3)

        tracker.record_attempt('web_safari', success=True)
        tracker.record_attempt('web_safari', success=True)

        # Only 2 samples, need 3
        assert tracker.get_success_rate('web_safari') is None

    def test_get_best_client(self):
        """Test getting the best client based on success rates."""
        tracker = PlayerClientTracker(enabled=True, min_samples=2)

        # web_safari: 100% (2/2)
        tracker.record_attempt('web_safari', success=True)
        tracker.record_attempt('web_safari', success=True)

        # tv_downgraded: 50% (1/2)
        tracker.record_attempt('tv_downgraded', success=True)
        tracker.record_attempt('tv_downgraded', success=False)

        # web: 0% (0/2)
        tracker.record_attempt('web', success=False)
        tracker.record_attempt('web', success=False)

        clients = ['web_safari', 'tv_downgraded', 'web']
        best = tracker.get_best_client(clients)

        assert best == 'web_safari'

    def test_disabled_tracker(self):
        """Test that disabled tracker returns None."""
        tracker = PlayerClientTracker(enabled=False)

        tracker.record_attempt('web_safari', success=True)

        assert tracker.get_success_rate('web_safari') is None
        assert tracker.get_best_client(['web_safari']) is None

    def test_window_size_trimming(self):
        """Test that window size trimming works."""
        tracker = PlayerClientTracker(enabled=True, window_size=3, min_samples=1)

        # Record 5 attempts
        for i in range(5):
            tracker.record_attempt('web_safari', success=(i < 3))

        # Should only keep last 3
        assert tracker.get_sample_count('web_safari') == 3

    def test_reset(self):
        """Test resetting tracker clears all data."""
        tracker = PlayerClientTracker(enabled=True, min_samples=1)

        tracker.record_attempt('web_safari', success=True)
        tracker.reset()

        assert tracker.get_sample_count('web_safari') == 0

    def test_serialization(self):
        """Test to_dict and from_dict."""
        tracker = PlayerClientTracker(enabled=True, window_size=50, min_samples=3)
        tracker.record_attempt('web_safari', success=True)
        tracker.record_attempt('web_safari', success=False)

        data = tracker.to_dict()
        assert data['enabled'] is True
        assert data['window_size'] == 50

        restored = PlayerClientTracker.from_dict(data)
        assert restored.enabled is True
        assert restored.window_size == 50
        assert restored.get_sample_count('web_safari') == 2


class TestExtractorArgsConfig:
    """Tests for ExtractorArgsConfig with new options."""

    def test_default_values(self):
        """Test default config values."""
        config = ExtractorArgsConfig()

        assert config.success_rate_window == 50
        assert config.extractor_args_fallback_order == []

    def test_custom_values(self):
        """Test custom config values."""
        config = ExtractorArgsConfig(
            success_rate_window=100,
            extractor_args_fallback_order=['web_safari', 'tv_downgraded', 'web']
        )

        assert config.success_rate_window == 100
        assert config.extractor_args_fallback_order == ['web_safari', 'tv_downgraded', 'web']


class TestEscalationManagerPlayerClient:
    """Tests for EscalationManager player_client selection."""

    @pytest.fixture
    def mock_impersonation_manager(self):
        """Create a mock impersonation manager."""
        manager = MagicMock()
        manager.get_impersonate_args.return_value = ['--impersonate', 'chrome-136']
        return manager

    @pytest.fixture
    def extractor_config(self):
        """Create extractor args config with dynamic selection enabled."""
        return ExtractorArgsConfig(
            enabled=True,
            player_clients=['web_safari', 'tv_downgraded', 'web'],
            success_rate_window=10,
            extractor_args_fallback_order=[],  # Use dynamic selection
        )

    @pytest.fixture
    def extractor_config_with_manual_order(self):
        """Create extractor args config with manual override."""
        return ExtractorArgsConfig(
            enabled=True,
            player_clients=['web_safari', 'tv_downgraded', 'web'],
            success_rate_window=10,
            extractor_args_fallback_order=['tv_downgraded', 'web', 'web_safari'],
        )

    def test_get_best_extractor_args_dynamic_selection(
        self, mock_impersonation_manager, extractor_config
    ):
        """Test dynamic selection returns best client based on success rates."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            extractor_args_config=extractor_config,
        )

        # Record successes for web_safari
        for _ in range(5):
            manager.record_extractor_args_result('web_safari', success=True)

        # Record failures for tv_downgraded
        for _ in range(5):
            manager.record_extractor_args_result('tv_downgraded', success=False)

        clients = ['web_safari', 'tv_downgraded', 'web']
        best = manager.get_best_extractor_args(clients)

        assert best == 'web_safari'

    def test_get_best_extractor_args_manual_override(
        self, mock_impersonation_manager, extractor_config_with_manual_order
    ):
        """Test manual override takes precedence over dynamic selection."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            extractor_args_config=extractor_config_with_manual_order,
        )

        # Even with dynamic selection data, manual override should be used
        for _ in range(5):
            manager.record_extractor_args_result('web_safari', success=True)
            manager.record_extractor_args_result('tv_downgraded', success=False)

        clients = ['web_safari', 'tv_downgraded', 'web']
        best = manager.get_best_extractor_args(clients)

        # Should use manual override order (first in list that's available)
        assert best == 'tv_downgraded'

    def test_get_best_extractor_args_disabled(
        self, mock_impersonation_manager
    ):
        """Test dynamic selection disabled when success_rate_window is 0."""
        config = ExtractorArgsConfig(
            enabled=True,
            player_clients=['web_safari', 'tv_downgraded'],
            success_rate_window=0,  # Disabled
        )
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            extractor_args_config=config,
        )

        # Record some data
        manager.record_extractor_args_result('web_safari', success=True)

        clients = ['web_safari', 'tv_downgraded']
        best = manager.get_best_extractor_args(clients)

        # Should return None when disabled
        assert best is None

    def test_build_extractor_args_uses_best_client(
        self, mock_impersonation_manager, extractor_config
    ):
        """Test _build_extractor_args uses the best client."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            extractor_args_config=extractor_config,
        )

        # Record successes for web_safari
        for _ in range(5):
            manager.record_extractor_args_result('web_safari', success=True)

        # Record failures for tv_downgraded
        for _ in range(5):
            manager.record_extractor_args_result('tv_downgraded', success=False)

        # Get escalation args for a keyword
        state = EscalationState()
        args = manager._build_extractor_args(state)

        assert '--extractor-args' in args
        assert 'youtube:player_client=' in args[1]
        # Best client should be first
        assert args[1].startswith('youtube:player_client=web_safari')

    def test_record_success_with_player_client(
        self, mock_impersonation_manager, extractor_config
    ):
        """Test record_success tracks player_client."""
        # Set min_samples to 1 for easier testing
        extractor_config.success_rate_window = 10
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            extractor_args_config=extractor_config,
        )
        # Override min_samples for testing
        manager._player_client_tracker._min_samples = 1

        # Record success with player_client
        manager.record_success('test_keyword', player_client='web_safari')

        # Check stats
        stats = manager.get_extractor_args_stats()
        assert 'web_safari' in stats['client_sample_counts']
        assert stats['client_sample_counts']['web_safari'] == 1

    def test_record_failure_with_player_client(
        self, mock_impersonation_manager, extractor_config
    ):
        """Test record_failure tracks player_client."""
        # Set min_samples to 1 for easier testing
        extractor_config.success_rate_window = 10
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            extractor_args_config=extractor_config,
        )
        # Override min_samples for testing
        manager._player_client_tracker._min_samples = 1

        # Record failure with player_client
        manager.record_failure('test_keyword', player_client='tv_downgraded')

        # Check stats
        stats = manager.get_extractor_args_stats()
        assert 'tv_downgraded' in stats['client_sample_counts']
        assert stats['client_sample_counts']['tv_downgraded'] == 1

    def test_reset_player_client_tracker(
        self, mock_impersonation_manager, extractor_config
    ):
        """Test reset_player_client_tracker clears tracking data."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            extractor_args_config=extractor_config,
        )

        # Record some data
        manager.record_extractor_args_result('web_safari', success=True)

        # Reset
        manager.reset_player_client_tracker()

        # Check data is cleared
        stats = manager.get_extractor_args_stats()
        assert stats['client_success_rates'] == {}

    def test_player_client_tracker_in_checkpoint(
        self, mock_impersonation_manager, extractor_config
    ):
        """Test player_client_tracker is included in to_dict."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            extractor_args_config=extractor_config,
        )

        manager.record_extractor_args_result('web_safari', success=True)

        data = manager.to_dict()

        assert 'player_client_tracker' in data
        assert 'web_safari' in data['player_client_tracker']['client_results']

    def test_player_client_tracker_restored_from_checkpoint(
        self, mock_impersonation_manager, extractor_config
    ):
        """Test player_client_tracker is restored in from_dict."""
        # Set min_samples to 1 for easier testing
        extractor_config.success_rate_window = 10
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            extractor_args_config=extractor_config,
        )
        # Override min_samples for testing
        manager._player_client_tracker._min_samples = 1

        manager.record_extractor_args_result('web_safari', success=True)

        # Get checkpoint data
        data = manager.to_dict()

        # Restore
        restored = EscalationManager.from_dict(
            data,
            impersonation_manager=mock_impersonation_manager,
            extractor_args_config=extractor_config,
        )

        # Override min_samples on restored as well
        restored._player_client_tracker._min_samples = 1

        # Check data is restored
        stats = restored.get_extractor_args_stats()
        assert 'web_safari' in stats['client_sample_counts']
