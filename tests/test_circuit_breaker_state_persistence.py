"""Tests for circuit breaker state persistence (US-109-010).

Tests cover:
- Circuit breaker state serialization to checkpoint
- State restoration on pipeline resume
- Stale checkpoint handling (>1 hour old)
- Metrics tracking for restoration frequency
"""

import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta

from src.checkpoint import CheckpointManager, CheckpointData
from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


class TestCircuitBreakerStatePersistence:
    """Test circuit breaker state persistence and restoration."""

    @pytest.mark.fast
    def test_to_checkpoint_dict(self):
        """Circuit breaker should serialize state to dict."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config=config)

        # Record some failures and trips
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()

        # Trip the circuit
        breaker.record_failure()
        breaker.record_failure()  # This should trip

        state_dict = breaker.to_checkpoint_dict()

        assert state_dict['consecutive_failures'] == 5
        assert state_dict['is_open'] is True
        assert state_dict['total_trips'] == 1
        assert state_dict['total_paused_seconds'] >= 0

    @pytest.mark.fast
    def test_from_checkpoint_dict_fresh(self):
        """Should restore state from fresh checkpoint."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config=config)

        # Simulate checkpoint data
        checkpoint_data = {
            'consecutive_failures': 3,
            'is_open': True,
            'opened_at': time.time() - 30,  # Opened 30 seconds ago
            'total_trips': 2,
            'total_paused_seconds': 60.0,
        }

        # Fresh checkpoint (age = 0)
        result = breaker.from_checkpoint_dict(checkpoint_data, checkpoint_age_seconds=0)

        assert result['restored'] is True
        assert result['restored_state'] == 'open'
        assert result['was_stale'] is False
        assert breaker.state.consecutive_failures == 3
        assert breaker.state.is_open is True
        assert breaker.state.total_trips == 2

    @pytest.mark.fast
    def test_from_checkpoint_dict_stale(self):
        """Stale checkpoint (>1 hour) should restore to half-open state."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config=config)

        # Simulate stale checkpoint data (age > 1 hour)
        checkpoint_data = {
            'consecutive_failures': 3,
            'is_open': True,
            'opened_at': time.time() - 7200,  # Opened 2 hours ago
            'total_trips': 2,
            'total_paused_seconds': 120.0,
        }

        # Stale checkpoint (age = 2 hours = 7200 seconds)
        result = breaker.from_checkpoint_dict(checkpoint_data, checkpoint_age_seconds=7200)

        assert result['restored'] is True
        assert result['was_stale'] is True
        assert result['restored_state'] == 'half_open'
        assert breaker.state.is_open is False
        assert breaker.state.consecutive_failures == 3  # Failure count preserved

    @pytest.mark.fast
    def test_from_checkpoint_dict_empty(self):
        """Should handle empty checkpoint data gracefully."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config=config)

        result = breaker.from_checkpoint_dict({})

        assert result['restored'] is False
        assert result['restored_state'] == 'closed'


class TestCheckpointCircuitBreakerIntegration:
    """Test integration between CheckpointManager and CircuitBreaker."""

    @pytest.mark.integration
    def test_save_and_load_circuit_breaker_state(self):
        """Should save and restore circuit breaker state via checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create checkpoint manager
            cm = CheckpointManager(tmpdir, config_hash="test123")

            # Create and configure circuit breaker
            config = CircuitBreakerConfig()
            breaker = CircuitBreaker(config=config)

            # Trip the circuit
            for _ in range(5):
                breaker.record_failure()

            # Save state to checkpoint
            cm.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash="test123"
            )
            cm.save_circuit_breaker_state(breaker, name="search")

            # Verify state is saved
            assert cm.has_circuit_breaker_state("search")
            saved_state = cm.get_circuit_breaker_state("search")
            assert saved_state['is_open'] is True
            assert saved_state['consecutive_failures'] == 5

            # Create new circuit breaker and restore from checkpoint
            breaker2 = CircuitBreaker(config=config)
            result = cm.load_circuit_breaker_state(breaker2, name="search")

            assert result['restored'] is True
            assert breaker2.state.is_open is True
            assert breaker2.state.consecutive_failures == 5

    @pytest.mark.integration
    def test_stale_checkpoint_half_open_recovery(self):
        """Stale checkpoint should restore to half-open state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create checkpoint manager
            cm = CheckpointManager(tmpdir, config_hash="test123")

            # Create circuit breaker and trip it
            config = CircuitBreakerConfig()
            breaker = CircuitBreaker(config=config)
            for _ in range(5):
                breaker.record_failure()

            # Save state with old timestamp (simulate stale checkpoint)
            cm.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash="test123"
            )
            cm.save_circuit_breaker_state(breaker, name="search")

            # Manually set saved_at to 2 hours ago to simulate stale checkpoint
            old_time = (datetime.now() - timedelta(hours=2)).isoformat()
            cm.data.circuit_breaker_state['search']['saved_at'] = old_time

            # Restore - should get half-open state
            breaker2 = CircuitBreaker(config=config)
            result = cm.load_circuit_breaker_state(breaker2, name="search")

            assert result['restored'] is True
            assert result['was_stale'] is True
            assert result['restored_state'] == 'half_open'
            assert breaker2.state.is_open is False
            assert breaker2.state.consecutive_failures == 5

    @pytest.mark.integration
    def test_restoration_metrics_tracking(self):
        """Should track restoration frequency in metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            cm = CheckpointManager(tmpdir, config_hash="test123")
            cm.data = CheckpointData(
                created_at=datetime.now().isoformat(),
                config_hash="test123"
            )

            # Create and restore circuit breaker twice
            config = CircuitBreakerConfig()
            for i in range(2):
                breaker = CircuitBreaker(config=config)
                for _ in range(5):
                    breaker.record_failure()
                cm.save_circuit_breaker_state(breaker, name="search")

                # Simulate fresh restore
                breaker2 = CircuitBreaker(config=config)
                cm.load_circuit_breaker_state(breaker2, name="search")

            # Check restoration metrics
            metrics = cm.data.circuit_breaker_health.get('restoration_metrics', {})
            search_metrics = metrics.get('search', {})

            assert search_metrics['total_restorations'] == 2
            assert 'by_state' in search_metrics
            assert 'last_restoration' in search_metrics


class TestCircuitBreakerBaseStatePersistence:
    """Test base class state persistence."""

    @pytest.mark.fast
    def test_base_to_checkpoint_dict(self):
        """Base class should serialize all state fields."""
        from src.common.circuit_breaker_base import CircuitBreakerStateBase

        # Test the state dataclass directly
        state = CircuitBreakerStateBase()
        state.consecutive_failures = 3
        state.is_open = True
        state.opened_at = time.time()
        state.total_trips = 2
        state.total_paused_seconds = 30.0

        # Create a mock breaker that uses the state
        class MockBreaker:
            def __init__(self):
                self.state = state

            def to_checkpoint_dict(self):
                return {
                    'consecutive_failures': self.state.consecutive_failures,
                    'is_open': self.state.is_open,
                    'opened_at': self.state.opened_at,
                    'total_trips': self.state.total_trips,
                    'total_paused_seconds': self.state.total_paused_seconds,
                }

        breaker = MockBreaker()
        state_dict = breaker.to_checkpoint_dict()

        assert state_dict['consecutive_failures'] == 3
        assert state_dict['is_open'] is True
        assert state_dict['opened_at'] is not None
        assert state_dict['total_trips'] == 2
        assert state_dict['total_paused_seconds'] == 30.0


class TestCircuitBreakerFileStatePersistence:
    """Test file-based state persistence (US-144-004)."""

    @pytest.mark.fast
    def test_state_file_path_config(self):
        """CircuitBreakerConfig should have state_file_path field."""
        config = CircuitBreakerConfig()
        assert hasattr(config, 'state_file_path')
        config.state_file_path = "/tmp/test_cb_state.json"
        assert config.state_file_path == "/tmp/test_cb_state.json"

    @pytest.mark.fast
    def test_auto_save_config(self):
        """CircuitBreakerConfig should have auto_save_on_state_change field."""
        config = CircuitBreakerConfig()
        assert hasattr(config, 'auto_save_on_state_change')
        assert config.auto_save_on_state_change is False
        config.auto_save_on_state_change = True
        assert config.auto_save_on_state_change is True

    @pytest.mark.fast
    def test_load_state_from_file_on_init(self):
        """Should load state from file on initialization if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "cb_state.json"

            # Create state file with some data
            import json
            state_data = {
                'consecutive_failures': 3,
                'is_open': True,
                'opened_at': time.time() - 30,
                'total_trips': 2,
                'total_paused_seconds': 60.0,
                'saved_at': time.time(),
            }
            with open(state_file, 'w') as f:
                json.dump(state_data, f)

            # Create config with state file path
            config = CircuitBreakerConfig()
            config.state_file_path = str(state_file)

            # Create circuit breaker - should load state
            breaker = CircuitBreaker(config=config)

            # Verify state was loaded
            assert breaker.state.consecutive_failures == 3
            assert breaker.state.is_open is True
            assert breaker.state.total_trips == 2

    @pytest.mark.fast
    def test_save_state_on_trip(self):
        """Should save state to file when auto_save is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "cb_state.json"

            # Create config with state file path and auto-save enabled
            config = CircuitBreakerConfig()
            config.state_file_path = str(state_file)
            config.auto_save_on_state_change = True
            config.consecutive_failures_threshold = 2  # Trip after 2 failures

            # Create circuit breaker
            breaker = CircuitBreaker(config=config)

            # Record failures to trip the circuit
            breaker.record_failure()
            breaker.record_failure()  # This should trip

            # Verify state file was created and has correct data
            import json
            assert state_file.exists()

            with open(state_file, 'r') as f:
                saved_state = json.load(f)

            assert saved_state['is_open'] is True
            assert saved_state['consecutive_failures'] >= 2
            assert saved_state['total_trips'] == 1

    @pytest.mark.fast
    def test_save_state_on_recovery(self):
        """Should save state to file when circuit recovers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "cb_state.json"

            # Create config with state file path and auto-save enabled
            config = CircuitBreakerConfig()
            config.state_file_path = str(state_file)
            config.auto_save_on_state_change = True
            config.consecutive_failures_threshold = 5

            # Create circuit breaker and trip it
            breaker = CircuitBreaker(config=config)
            for _ in range(5):
                breaker.record_failure()  # Trip

            # Verify is_open was saved
            import json
            with open(state_file, 'r') as f:
                saved_before = json.load(f)
            assert saved_before['is_open'] is True
            assert saved_before['consecutive_failures'] == 5

            # Now simulate recovery by recording success
            # This should reset failure count and save state
            breaker.record_success()

            # Verify state file was updated (is_open=False, failures=0)
            with open(state_file, 'r') as f:
                saved_after = json.load(f)

            assert saved_after['is_open'] is False
            assert saved_after['consecutive_failures'] == 0

    @pytest.mark.fast
    def test_skip_load_when_no_file(self):
        """Should not fail when state file doesn't exist."""
        config = CircuitBreakerConfig()
        config.state_file_path = "/nonexistent/path/cb_state.json"

        # Should not raise an error
        breaker = CircuitBreaker(config=config)

        # Should have default state
        assert breaker.state.consecutive_failures == 0
        assert breaker.state.is_open is False

    @pytest.mark.fast
    def test_skip_save_when_disabled(self):
        """Should not save when auto_save is disabled (default)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "cb_state.json"

            # Create config without auto-save
            config = CircuitBreakerConfig()
            config.state_file_path = str(state_file)
            config.auto_save_on_state_change = False  # Default

            # Create and trip circuit breaker
            breaker = CircuitBreaker(config=config)
            breaker.record_failure()
            breaker.record_failure()
            breaker.record_failure()
            breaker.record_failure()
            breaker.record_failure()  # Trip

            # File should not exist (auto-save is disabled)
            assert not state_file.exists()

    @pytest.mark.fast
    def test_skip_save_when_no_path(self):
        """Should not save when state_file_path is not set."""
        config = CircuitBreakerConfig()
        config.state_file_path = ""
        config.auto_save_on_state_change = True

        # Create and trip circuit breaker
        breaker = CircuitBreaker(config=config)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()  # Trip

        # Should not error out (logged as debug)
        assert breaker.state.is_open is True
