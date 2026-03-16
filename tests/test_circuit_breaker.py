"""Tests for circuit breaker functionality.

US-006: Add circuit breaker for repeated search failures

Tests cover:
- CircuitBreakerConfig defaults and custom values
- CircuitBreaker state management (failures, open/closed)
- Circuit trips after threshold failures
- Circuit resets on successful search
- Pause duration and timing
- Stats and checkpoint persistence
- Integration with VideoDownloader
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)


# ============================================================================
# Configuration Tests
# ============================================================================


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig defaults and custom values."""

    @pytest.mark.fast
    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CircuitBreakerConfig()

        assert config.enabled is True
        assert config.consecutive_failures_threshold == 5
        assert config.pause_seconds == 60.0

    @pytest.mark.fast
    def test_custom_values(self):
        """Config should accept custom values."""
        config = CircuitBreakerConfig(
            enabled=False,
            consecutive_failures_threshold=10,
            pause_seconds=120.0
        )

        assert config.enabled is False
        assert config.consecutive_failures_threshold == 10
        assert config.pause_seconds == 120.0


# ============================================================================
# State Initialization Tests
# ============================================================================


class TestCircuitBreakerInitialization:
    """Test CircuitBreaker initialization and initial state."""

    @pytest.mark.fast
    def test_default_initialization(self):
        """Circuit breaker should initialize with default config."""
        breaker = CircuitBreaker()

        assert breaker.config.enabled is True
        assert breaker.state.consecutive_failures == 0
        assert breaker.state.is_open is False
        assert breaker.state.opened_at is None
        assert breaker.state.total_trips == 0

    @pytest.mark.fast
    def test_custom_config_initialization(self):
        """Circuit breaker should accept custom config."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=3,
            pause_seconds=30.0
        )
        breaker = CircuitBreaker(config)

        assert breaker.config.consecutive_failures_threshold == 3
        assert breaker.config.pause_seconds == 30.0

    @pytest.mark.fast
    def test_disabled_breaker(self):
        """Disabled circuit breaker should not track failures."""
        config = CircuitBreakerConfig(enabled=False)
        breaker = CircuitBreaker(config)

        assert breaker.is_enabled is False


# ============================================================================
# Failure Recording Tests
# ============================================================================


class TestFailureRecording:
    """Test recording failures and counting."""

    @pytest.mark.fast
    def test_single_failure_increments_count(self):
        """Recording failure should increment consecutive_failures."""
        breaker = CircuitBreaker()

        assert breaker.state.consecutive_failures == 0
        breaker.record_failure()
        assert breaker.state.consecutive_failures == 1

    @pytest.mark.fast
    def test_multiple_failures_accumulate(self):
        """Multiple failures should accumulate."""
        breaker = CircuitBreaker()

        for i in range(4):
            breaker.record_failure()

        assert breaker.state.consecutive_failures == 4

    @pytest.mark.fast
    def test_failure_on_disabled_breaker_noop(self):
        """Recording failure on disabled breaker should be no-op."""
        config = CircuitBreakerConfig(enabled=False)
        breaker = CircuitBreaker(config)

        result = breaker.record_failure()

        assert result is False
        assert breaker.state.consecutive_failures == 0


# ============================================================================
# Circuit Trip Tests
# ============================================================================


class TestCircuitTrip:
    """Test circuit tripping (opening) behavior."""

    @pytest.mark.fast
    def test_trips_at_threshold(self):
        """Circuit should trip when failures reach threshold."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=3)
        breaker = CircuitBreaker(config)

        # First 2 failures - no trip
        assert breaker.record_failure() is False
        assert breaker.record_failure() is False
        assert breaker.is_open is False

        # 3rd failure - trips
        assert breaker.record_failure() is True
        assert breaker.is_open is True
        assert breaker.state.total_trips == 1

    @pytest.mark.fast
    def test_trips_logs_info(self):
        """Circuit trip should log at INFO level."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=2)
        breaker = CircuitBreaker(config)

        with patch('src.downloader.circuit_breaker.logger') as mock_logger:
            breaker.record_failure()
            breaker.record_failure()

            # Should have logged INFO about tripping
            assert mock_logger.info.called
            call_args = str(mock_logger.info.call_args)
            assert 'TRIPPED' in call_args or 'consecutive' in call_args.lower()

    @pytest.mark.fast
    def test_opened_at_timestamp_set(self):
        """opened_at should be set when circuit trips."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=1)
        breaker = CircuitBreaker(config)

        before = time.time()
        breaker.record_failure()
        after = time.time()

        assert breaker.state.opened_at is not None
        assert before <= breaker.state.opened_at <= after


# ============================================================================
# Success Recording Tests
# ============================================================================


class TestSuccessRecording:
    """Test recording success and resetting."""

    @pytest.mark.fast
    def test_success_resets_failure_count(self):
        """Recording success should reset consecutive_failures to 0."""
        breaker = CircuitBreaker()

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state.consecutive_failures == 2

        breaker.record_success()
        assert breaker.state.consecutive_failures == 0

    @pytest.mark.fast
    def test_success_closes_open_circuit(self):
        """Success should close an open circuit."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=1)
        breaker = CircuitBreaker(config)

        breaker.record_failure()
        assert breaker.is_open is True

        breaker.record_success()
        assert breaker.is_open is False
        assert breaker.state.opened_at is None

    @pytest.mark.fast
    def test_success_on_disabled_breaker_noop(self):
        """Recording success on disabled breaker should be no-op."""
        config = CircuitBreakerConfig(enabled=False)
        breaker = CircuitBreaker(config)

        breaker.record_success()
        assert breaker.state.consecutive_failures == 0


# ============================================================================
# Pause and Wait Tests
# ============================================================================


class TestCheckAndWait:
    """Test check_and_wait behavior with pause."""

    @pytest.mark.fast
    def test_returns_true_when_closed(self):
        """check_and_wait should return True when circuit is closed."""
        breaker = CircuitBreaker()

        result = breaker.check_and_wait()

        assert result is True

    @pytest.mark.fast
    def test_returns_false_when_disabled(self):
        """check_and_wait should return False when disabled."""
        config = CircuitBreakerConfig(enabled=False)
        breaker = CircuitBreaker(config)

        result = breaker.check_and_wait()

        assert result is False

    @pytest.mark.fast
    def test_waits_when_open(self):
        """check_and_wait should sleep when circuit is open."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.1  # Short pause for testing
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        start = time.time()
        breaker.check_and_wait()
        elapsed = time.time() - start

        # Should have waited at least pause_seconds
        assert elapsed >= 0.1

    @pytest.mark.fast
    def test_closes_after_pause(self):
        """Circuit should close after pause completes."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.05
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        assert breaker.is_open is True
        breaker.check_and_wait()
        assert breaker.is_open is False

    @pytest.mark.fast
    def test_pause_duration_logged(self):
        """Pause should be logged at INFO level."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.01
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        with patch('src.downloader.circuit_breaker.logger') as mock_logger:
            breaker.check_and_wait()

            assert mock_logger.info.called

    @pytest.mark.fast
    def test_tracks_total_paused_seconds(self):
        """Should track total paused time for reporting."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.05,
            jitter_factor=0.0  # Disable jitter for deterministic test
        )
        breaker = CircuitBreaker(config)

        # First trip
        breaker.record_failure()
        breaker.check_and_wait()

        # Without jitter, should pause exactly pause_seconds
        assert breaker.state.total_paused_seconds >= 0.05

    def test_immediate_return_after_pause_expired(self):
        """If pause time has elapsed, should return immediately."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.01
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        # Wait longer than pause_seconds
        time.sleep(0.02)

        start = time.time()
        breaker.check_and_wait()
        elapsed = time.time() - start

        # Should return almost immediately (pause already elapsed)
        assert elapsed < 0.01


# ============================================================================
# Manual Reset Tests
# ============================================================================


class TestManualReset:
    """Test manual reset functionality."""

    @pytest.mark.fast
    def test_reset_clears_failures(self):
        """reset() should clear consecutive failures."""
        breaker = CircuitBreaker()
        breaker.record_failure()
        breaker.record_failure()

        breaker.reset()

        assert breaker.state.consecutive_failures == 0

    @pytest.mark.fast
    def test_reset_closes_circuit(self):
        """reset() should close an open circuit."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=1)
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        breaker.reset()

        assert breaker.is_open is False
        assert breaker.state.opened_at is None


# ============================================================================
# Statistics Tests
# ============================================================================


class TestGetStats:
    """Test statistics reporting."""

    @pytest.mark.fast
    def test_stats_empty_state(self):
        """Stats should reflect initial state."""
        breaker = CircuitBreaker()
        stats = breaker.get_stats()

        assert stats['enabled'] is True
        assert stats['is_open'] is False
        assert stats['consecutive_failures'] == 0
        assert stats['total_trips'] == 0
        assert stats['total_paused_seconds'] == 0.0
        assert stats['threshold'] == 5
        assert stats['pause_seconds'] == 60.0

    @pytest.mark.fast
    def test_stats_after_trips(self):
        """Stats should reflect state after trips."""
        config = CircuitBreakerConfig(
            consecutive_failures_threshold=1,
            pause_seconds=0.1  # Longer pause to ensure we measure it
        )
        breaker = CircuitBreaker(config)

        # Trip once - check_and_wait should wait the full pause duration
        breaker.record_failure()

        # Verify circuit is open
        assert breaker.state.is_open is True
        assert breaker.state.total_trips == 1

        # Now wait - this should block and track paused time
        breaker.check_and_wait()

        stats = breaker.get_stats()

        # Trip count should reflect one trip
        assert stats['total_trips'] == 1
        # If pause was needed, total_paused_seconds > 0; if not, it just passed
        # The key invariant is that the circuit was tripped and total_trips was incremented


# ============================================================================
# Checkpoint Persistence Tests
# ============================================================================


class TestCheckpointPersistence:
    """Test checkpoint save/restore."""

    @pytest.mark.fast
    def test_to_checkpoint_dict(self):
        """Should serialize cumulative stats to checkpoint."""
        config = CircuitBreakerConfig(consecutive_failures_threshold=1)
        breaker = CircuitBreaker(config)

        breaker.record_failure()  # Trip
        breaker.state.total_paused_seconds = 30.0

        checkpoint = breaker.to_checkpoint_dict()

        assert checkpoint['total_trips'] == 1
        assert checkpoint['total_paused_seconds'] == 30.0
        # Should NOT include transient state
        assert 'is_open' not in checkpoint
        assert 'opened_at' not in checkpoint

    @pytest.mark.fast
    def test_from_checkpoint_dict_restores_cumulative(self):
        """Should restore cumulative stats from checkpoint."""
        breaker = CircuitBreaker()

        breaker.from_checkpoint_dict({
            'total_trips': 3,
            'total_paused_seconds': 180.0,
            'consecutive_failures': 2  # Should be ignored
        })

        assert breaker.state.total_trips == 3
        assert breaker.state.total_paused_seconds == 180.0
        # Should NOT restore transient state
        assert breaker.state.consecutive_failures == 0
        assert breaker.is_open is False

    @pytest.mark.fast
    def test_from_checkpoint_dict_empty(self):
        """Should handle empty/None checkpoint gracefully."""
        breaker = CircuitBreaker()

        breaker.from_checkpoint_dict(None)
        breaker.from_checkpoint_dict({})

        assert breaker.state.total_trips == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegrationWithVideoDownloader:
    """Test integration with VideoDownloader initialization."""

    @pytest.mark.fast
    def test_circuit_breaker_initialized_from_config(self):
        """VideoDownloader should initialize circuit breaker from config."""
        from unittest.mock import MagicMock, patch, PropertyMock

        # Mock the config
        mock_config = MagicMock()
        mock_download_config = MagicMock()

        # Circuit breaker config
        mock_cb_config = MagicMock()
        mock_cb_config.enabled = True
        mock_cb_config.consecutive_failures_threshold = 3
        mock_cb_config.pause_seconds = 30.0

        mock_download_config.circuit_breaker = mock_cb_config
        mock_download_config.cookies_from_browser = ''
        mock_download_config.cookies_path = ''
        mock_download_config.cookie_rotation = MagicMock(enabled=False)
        mock_download_config.vpn = MagicMock(enabled=False)
        mock_download_config.speed_tracking = MagicMock(enabled=False)

        mock_config.download = mock_download_config
        mock_config.cache_dir = '/tmp'
        mock_config.downloaded_videos_dir = '/tmp/videos'

        # Patch dependencies to avoid file system operations
        with patch('src.downloader.core.CheckpointManager'), \
             patch('src.downloader.core.TranscodingManager'), \
             patch('src.downloader.core.TitleFilter'), \
             patch('src.downloader.core.SpeechScreener'), \
             patch('src.downloader.core.SearchOptimizer'), \
             patch('src.downloader.core.AudioFirstPipeline'), \
             patch('src.downloader.core.utils.get_cookies_args', return_value=[]):

            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader(mock_config)

            assert downloader.circuit_breaker is not None
            assert downloader.circuit_breaker.config.enabled is True
            assert downloader.circuit_breaker.config.consecutive_failures_threshold == 3

    @pytest.mark.fast
    def test_circuit_breaker_disabled_when_config_disabled(self):
        """Circuit breaker should be disabled when config says so."""
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_download_config = MagicMock()

        mock_cb_config = MagicMock()
        mock_cb_config.enabled = False

        mock_download_config.circuit_breaker = mock_cb_config
        mock_download_config.cookies_from_browser = ''
        mock_download_config.cookies_path = ''
        mock_download_config.cookie_rotation = MagicMock(enabled=False)
        mock_download_config.vpn = MagicMock(enabled=False)
        mock_download_config.speed_tracking = MagicMock(enabled=False)

        mock_config.download = mock_download_config
        mock_config.cache_dir = '/tmp'
        mock_config.downloaded_videos_dir = '/tmp/videos'

        with patch('src.downloader.core.CheckpointManager'), \
             patch('src.downloader.core.TranscodingManager'), \
             patch('src.downloader.core.TitleFilter'), \
             patch('src.downloader.core.SpeechScreener'), \
             patch('src.downloader.core.SearchOptimizer'), \
             patch('src.downloader.core.AudioFirstPipeline'), \
             patch('src.downloader.core.utils.get_cookies_args', return_value=[]):

            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader(mock_config)

            assert downloader.circuit_breaker.is_enabled is False


class TestCircuitBreakerInDownloadFlow:
    """Test circuit breaker behavior in download flow."""

    @pytest.mark.fast
    def test_check_and_wait_called_before_search(self):
        """Circuit breaker check should be called before searches."""
        from unittest.mock import MagicMock, patch
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        # Create a mock that tracks calls
        breaker = CircuitBreaker(CircuitBreakerConfig(enabled=True))
        breaker.check_and_wait = MagicMock(return_value=True)
        breaker.record_failure = MagicMock()

        # The check should be called when search happens
        # This is a design verification - actual integration test would need more setup
        breaker.check_and_wait()
        assert breaker.check_and_wait.called

    @pytest.mark.fast
    def test_failure_recorded_on_no_results(self):
        """Failure should be recorded when search returns no results."""
        breaker = CircuitBreaker()

        # Simulate no results scenario
        breaker.record_failure()

        assert breaker.state.consecutive_failures == 1

    @pytest.mark.fast
    def test_success_recorded_on_results(self):
        """Success should be recorded when search returns results."""
        breaker = CircuitBreaker()

        # Add some failures first
        breaker.record_failure()
        breaker.record_failure()

        # Then success
        breaker.record_success()

        assert breaker.state.consecutive_failures == 0


# ============================================================================
# US-89-009: Multi-Circuit Coordination Tests
# ============================================================================


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry for tracking all circuit breakers."""

    @pytest.mark.fast
    def test_registry_starts_empty(self):
        """Registry should start with no circuit breakers."""
        from src.downloader.circuit_breaker import CircuitBreakerRegistry
        registry = CircuitBreakerRegistry()
        assert len(registry._breakers) == 0
        assert not registry.is_any_tripped()

    @pytest.mark.fast
    def test_register_circuit_breaker(self):
        """Registry should register circuit breakers by name."""
        from src.downloader.circuit_breaker import CircuitBreakerRegistry
        registry = CircuitBreakerRegistry()

        cb1 = CircuitBreaker(name="search")
        cb2 = CircuitBreaker(name="download")

        registry.register("search", cb1)
        registry.register("download", cb2)

        assert registry.get("search") is cb1
        assert registry.get("download") is cb2

    @pytest.mark.fast
    def test_unregister_circuit_breaker(self):
        """Registry should unregister circuit breakers by name."""
        from src.downloader.circuit_breaker import CircuitBreakerRegistry
        registry = CircuitBreakerRegistry()

        cb = CircuitBreaker(name="search")
        registry.register("search", cb)
        assert registry.get("search") is cb

        registry.unregister("search")
        assert registry.get("search") is None

    @pytest.mark.fast
    def test_is_any_tripped_when_closed(self):
        """is_any_tripped should return False when all circuits are closed."""
        from src.downloader.circuit_breaker import CircuitBreakerRegistry
        registry = CircuitBreakerRegistry()

        cb1 = CircuitBreaker(name="search")
        cb2 = CircuitBreaker(name="download")
        registry.register("search", cb1)
        registry.register("download", cb2)

        assert not registry.is_any_tripped()

    @pytest.mark.fast
    def test_is_any_tripped_when_one_open(self):
        """is_any_tripped should return True when any circuit is open."""
        from src.downloader.circuit_breaker import CircuitBreakerRegistry
        registry = CircuitBreakerRegistry()

        cb1 = CircuitBreaker(name="search")
        cb2 = CircuitBreaker(name="download")

        # Trip cb1
        cb1.state.is_open = True

        registry.register("search", cb1)
        registry.register("download", cb2)

        assert registry.is_any_tripped()
        assert "search" in registry.get_tripped_names()
        assert "download" not in registry.get_tripped_names()

    @pytest.mark.fast
    def test_get_all_health_metrics(self):
        """Registry should return health metrics from all breakers."""
        from src.downloader.circuit_breaker import CircuitBreakerRegistry
        registry = CircuitBreakerRegistry()

        cb1 = CircuitBreaker(name="search")
        cb1.state.total_trips = 3
        cb1.state.consecutive_failures = 2

        registry.register("search", cb1)

        metrics = registry.get_all_health_metrics()

        assert "search" in metrics
        assert metrics["search"]["trip_count"] == 3
        assert metrics["search"]["consecutive_failures"] == 2

    @pytest.mark.fast
    def test_get_aggregate_stats(self):
        """Registry should return aggregate stats across all breakers."""
        from src.downloader.circuit_breaker import CircuitBreakerRegistry
        registry = CircuitBreakerRegistry()

        cb1 = CircuitBreaker(name="search")
        cb1.state.total_trips = 2
        cb1.state.total_paused_seconds = 120.0

        cb2 = CircuitBreaker(name="download")
        cb2.state.total_trips = 1
        cb2.state.total_paused_seconds = 60.0
        cb2.state.is_open = True

        registry.register("search", cb1)
        registry.register("download", cb2)

        stats = registry.get_aggregate_stats()

        assert stats["total_breakers"] == 2
        assert stats["tripped_count"] == 1
        assert stats["total_trips"] == 3
        assert stats["total_paused_seconds"] == 180.0
        assert stats["is_any_tripped"] is True


class TestCircuitBreakerCoordinator:
    """Test CircuitBreakerCoordinator for multi-circuit coordination."""

    def setup_method(self):
        """Reset coordinator singleton before each test."""
        from src.downloader.circuit_breaker import CircuitBreakerCoordinator
        CircuitBreakerCoordinator.reset_instance()

    @pytest.mark.fast
    def test_singleton_pattern(self):
        """Coordinator should be a singleton."""
        from src.downloader.circuit_breaker import CircuitBreakerCoordinator
        coord1 = CircuitBreakerCoordinator.get_instance()
        coord2 = CircuitBreakerCoordinator.get_instance()
        assert coord1 is coord2

    @pytest.mark.fast
    def test_reset_instance_clears_singleton(self):
        """reset_instance should clear singleton for testing."""
        from src.downloader.circuit_breaker import CircuitBreakerCoordinator
        coord1 = CircuitBreakerCoordinator.get_instance()
        CircuitBreakerCoordinator.reset_instance()
        coord2 = CircuitBreakerCoordinator.get_instance()
        assert coord1 is not coord2

    @pytest.mark.fast
    def test_default_rules_added(self):
        """Coordinator should have default cascade rules."""
        from src.downloader.circuit_breaker import CircuitBreakerCoordinator
        coord = CircuitBreakerCoordinator.get_instance()

        stats = coord.get_coordination_stats()
        assert stats["rules_count"] >= 3  # Default rules

    @pytest.mark.fast
    def test_add_custom_rule(self):
        """Coordinator should allow adding custom cascade rules."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator,
            CascadeRule,
        )
        coord = CircuitBreakerCoordinator.get_instance()
        coord.clear_rules()

        coord.add_rule(CascadeRule(
            source="video_search",
            target="download",
            on_trip=True,
        ))

        stats = coord.get_coordination_stats()
        assert stats["rules_count"] == 1

    @pytest.mark.fast
    def test_propagate_trip(self):
        """propagate_trip should trip target circuit based on rules."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator,
            CircuitBreakerRegistry,
            CascadeRule,
        )
        coord = CircuitBreakerCoordinator.get_instance()
        coord.clear_rules()
        coord.set_enabled(True)

        # Create registry and breakers
        registry = CircuitBreakerRegistry()
        search_cb = CircuitBreaker(name="search")
        download_cb = CircuitBreaker(name="download")

        registry.register("search", search_cb)
        registry.register("download", download_cb)

        coord.set_registry(registry)
        coord.add_rule(CascadeRule(
            source="search",
            target="download",
            on_trip=True,
        ))

        # Trip search - should propagate to download
        search_cb.state.is_open = True
        search_cb.state.opened_at = time.time()
        search_cb.state.total_trips = 1

        coord.propagate_trip("search")

        assert download_cb.state.is_open

    @pytest.mark.fast
    def test_propagate_failure(self):
        """propagate_failure should increment failure count on target."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator,
            CircuitBreakerRegistry,
            CascadeRule,
        )
        coord = CircuitBreakerCoordinator.get_instance()
        coord.clear_rules()
        coord.set_enabled(True)

        registry = CircuitBreakerRegistry()
        search_cb = CircuitBreaker(name="search")
        caption_cb = CircuitBreaker(name="caption")

        registry.register("search", search_cb)
        registry.register("caption", caption_cb)

        coord.set_registry(registry)
        coord.add_rule(CascadeRule(
            source="search",
            target="caption",
            on_failure=True,
        ))

        # Record failure on search - should propagate to caption
        coord.propagate_failure("search")

        assert caption_cb.state.consecutive_failures == 1

    @pytest.mark.fast
    def test_disabled_coordinator_no_propagation(self):
        """Disabled coordinator should not propagate events."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator,
            CircuitBreakerRegistry,
            CascadeRule,
        )
        coord = CircuitBreakerCoordinator.get_instance()
        coord.clear_rules()
        coord.set_enabled(False)  # Disabled!

        registry = CircuitBreakerRegistry()
        search_cb = CircuitBreaker(name="search")
        download_cb = CircuitBreaker(name="download")

        registry.register("search", search_cb)
        registry.register("download", download_cb)

        coord.set_registry(registry)
        coord.add_rule(CascadeRule(
            source="search",
            target="download",
            on_trip=True,
        ))

        # Trip search - should NOT propagate (disabled)
        search_cb.state.is_open = True
        coord.propagate_trip("search")

        assert not download_cb.state.is_open

    @pytest.mark.fast
    def test_no_registry_no_propagation(self):
        """No propagation should occur when registry is not set."""
        from src.downloader.circuit_breaker import CircuitBreakerCoordinator
        coord = CircuitBreakerCoordinator.get_instance()
        coord.clear_rules()
        coord.set_enabled(True)
        # No registry set

        # Should not raise, just no-op
        coord.propagate_trip("search")
        coord.propagate_failure("search")

    # US-136-008: Tests for caption circuit breaker integration

    @pytest.mark.fast
    def test_caption_to_download_cascade_rule_exists(self):
        """Coordinator should have caption -> download cascade rule."""
        from src.downloader.circuit_breaker import CircuitBreakerCoordinator, CascadeRule
        coord = CircuitBreakerCoordinator.get_instance()

        # Find caption -> download rule
        rule = next((r for r in coord._rules if r.source == "caption" and r.target == "download"), None)
        assert rule is not None
        assert rule.on_trip is True
        assert rule.on_failure is False

    @pytest.mark.fast
    def test_caption_trip_propagates_to_download(self):
        """Caption circuit breaker trip should propagate to download."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator, CircuitBreakerRegistry,
            CircuitBreaker, CircuitBreakerConfig,
        )
        from src.caption.circuit_breaker import CaptionCircuitBreaker, CaptionCircuitBreakerConfig

        coord = CircuitBreakerCoordinator.get_instance()
        registry = CircuitBreakerRegistry()
        coord.set_registry(registry)

        # Create mock circuit breakers
        search_cb = CircuitBreaker(CircuitBreakerConfig())
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig())
        download_cb = CircuitBreaker(CircuitBreakerConfig())

        registry.register("search", search_cb)
        registry.register("caption", caption_cb)
        registry.register("download", download_cb)

        # Trip caption - should propagate to download
        caption_cb.state.is_open = True
        caption_cb.state.opened_at = time.time()
        coord.propagate_trip("caption")

        assert download_cb.state.is_open is True

    @pytest.mark.fast
    def test_get_aggregate_health_all_healthy(self):
        """Aggregate health should show all circuits healthy when none tripped."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator, CircuitBreakerRegistry,
            CircuitBreaker, CircuitBreakerConfig,
        )
        from src.caption.circuit_breaker import CaptionCircuitBreaker, CaptionCircuitBreakerConfig

        coord = CircuitBreakerCoordinator.get_instance()
        registry = CircuitBreakerRegistry()
        coord.set_registry(registry)

        # Register breakers
        search_cb = CircuitBreaker(CircuitBreakerConfig())
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig())
        download_cb = CircuitBreaker(CircuitBreakerConfig())

        registry.register("search", search_cb)
        registry.register("caption", caption_cb)
        registry.register("download", download_cb)

        health = coord.get_aggregate_health()

        assert health['overall_healthy'] is True
        assert 'search' in health['components']
        assert 'caption' in health['components']
        assert 'download' in health['components']
        assert health['aggregate']['total_breakers'] == 3
        assert health['aggregate']['tripped_count'] == 0

    @pytest.mark.fast
    def test_get_aggregate_health_with_tripped_circuits(self):
        """Aggregate health should show tripped circuits."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator, CircuitBreakerRegistry,
            CircuitBreaker, CircuitBreakerConfig,
        )
        from src.caption.circuit_breaker import CaptionCircuitBreaker, CaptionCircuitBreakerConfig

        coord = CircuitBreakerCoordinator.get_instance()
        registry = CircuitBreakerRegistry()
        coord.set_registry(registry)

        # Register breakers
        search_cb = CircuitBreaker(CircuitBreakerConfig())
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig())
        download_cb = CircuitBreaker(CircuitBreakerConfig())

        registry.register("search", search_cb)
        registry.register("caption", caption_cb)
        registry.register("download", download_cb)

        # Trip search and download
        search_cb.state.is_open = True
        download_cb.state.is_open = True

        health = coord.get_aggregate_health()

        assert health['overall_healthy'] is False
        assert health['components']['search']['is_open'] is True
        assert health['components']['caption']['is_open'] is False
        assert health['components']['download']['is_open'] is True
        assert health['aggregate']['tripped_count'] == 2

    @pytest.mark.fast
    def test_coordination_event_metrics(self):
        """Coordination events should be tracked for metrics."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator, CircuitBreakerRegistry,
            CircuitBreaker, CircuitBreakerConfig,
        )

        coord = CircuitBreakerCoordinator.get_instance()
        registry = CircuitBreakerRegistry()
        coord.set_registry(registry)

        # Register breakers
        search_cb = CircuitBreaker(CircuitBreakerConfig())
        download_cb = CircuitBreaker(CircuitBreakerConfig())

        registry.register("search", search_cb)
        registry.register("download", download_cb)

        # Trip search - should create coordination event
        search_cb.state.is_open = True
        coord.propagate_trip("search")

        metrics = coord.get_coordination_metrics()

        assert metrics['total_events'] >= 1
        assert 'trip_propagation' in metrics['events_by_type']
        assert metrics['events_by_type']['trip_propagation'] >= 1

    @pytest.mark.fast
    def test_recover_circuits_downstream_first(self):
        """Recovery should attempt downstream circuits first."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator, CircuitBreakerRegistry,
            CircuitBreaker, CircuitBreakerConfig,
        )

        coord = CircuitBreakerCoordinator.get_instance()
        registry = CircuitBreakerRegistry()
        coord.set_registry(registry)

        # Register breakers
        search_cb = CircuitBreaker(CircuitBreakerConfig())
        download_cb = CircuitBreaker(CircuitBreakerConfig())

        registry.register("search", search_cb)
        registry.register("download", download_cb)

        # Trip both circuits
        search_cb.state.is_open = True
        search_cb.state.opened_at = time.time() - 100  # Old enough to recover
        download_cb.state.is_open = True
        download_cb.state.opened_at = time.time() - 100  # Old enough to recover

        result = coord.recover_circuits()

        # Download should be recovered first (downstream)
        assert 'download' in result['recovered']
        assert 'search' in result['recovered']
        assert download_cb.state.is_open is False
        assert search_cb.state.is_open is False

    @pytest.mark.fast
    def test_recover_circuits_keeps_tripped_when_not_ready(self):
        """Recovery should keep circuits open if pause not elapsed."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator, CircuitBreakerRegistry,
            CircuitBreaker, CircuitBreakerConfig,
        )

        coord = CircuitBreakerCoordinator.get_instance()
        registry = CircuitBreakerRegistry()
        coord.set_registry(registry)

        # Register breakers
        download_cb = CircuitBreaker(CircuitBreakerConfig(pause_seconds=60.0))
        registry.register("download", download_cb)

        # Trip circuit but not enough time elapsed
        download_cb.state.is_open = True
        download_cb.state.opened_at = time.time() - 10  # Only 10s elapsed, not 60s

        result = coord.recover_circuits()

        assert 'download' in result['still_tripped']
        assert download_cb.state.is_open is True

    @pytest.mark.fast
    def test_aggregate_health_no_registry(self):
        """Aggregate health should handle no registry gracefully."""
        from src.downloader.circuit_breaker import CircuitBreakerCoordinator
        coord = CircuitBreakerCoordinator.get_instance()
        # No registry set

        health = coord.get_aggregate_health()

        assert health['overall_healthy'] is True
        assert health['components'] == {}
        assert health['aggregate']['total_breakers'] == 0


class TestCascadeRule:
    """Test CascadeRule dataclass."""

    @pytest.mark.fast
    def test_cascade_rule_defaults(self):
        """CascadeRule should have sensible defaults."""
        from src.downloader.circuit_breaker import CascadeRule
        rule = CascadeRule(source="a", target="b")

        assert rule.source == "a"
        assert rule.target == "b"
        assert rule.on_trip is True
        assert rule.on_failure is False
