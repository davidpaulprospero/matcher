"""Integration tests for RateLimitBudget + CircuitBreaker + RetryQueue coordination.

Tests the interaction between rate limit budget management, circuit breaker tripping,
and retry queue processing under various rate-limiting scenarios.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.rate_limit_budget import RateLimitBudget
from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.downloader.retry_queue import RetryQueue, BatchRetryConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def budget():
    """Create a rate limit budget with sensible test defaults."""
    b = RateLimitBudget()
    b.max_rotations = 3
    b.max_backoff_time = 60.0
    b.max_vpn_switches = 2
    return b


@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker with low threshold for testing."""
    config = CircuitBreakerConfig(
        enabled=True,
        consecutive_failures_threshold=5,
        pause_seconds=0.1,  # Short pause for tests
        max_pause_seconds=1.0,
    )
    return CircuitBreaker(config)


@pytest.fixture
def retry_queue():
    """Create a retry queue with short delays for testing."""
    config = BatchRetryConfig(
        enabled=True,
        delay_seconds=0.01,  # Short delay for tests
        max_passes=2,
        respect_circuit_breaker=True,
        wait_for_cookie_cooldown=True,
        max_combined_wait_seconds=0.5,
    )
    return RetryQueue(config)


@pytest.fixture
def linked_components(budget, circuit_breaker, retry_queue):
    """Create and link all rate limit components."""
    circuit_breaker._budget = budget
    retry_queue.set_circuit_breaker(circuit_breaker)
    return {
        "budget": budget,
        "circuit_breaker": circuit_breaker,
        "retry_queue": retry_queue,
    }


# =============================================================================
# Test: 5 consecutive 429s trigger circuit breaker trip
# =============================================================================


class TestConsecutive429TriggerCircuitBreaker:
    """Test that 5 consecutive 429 errors trigger circuit breaker trip."""

    def test_circuit_trips_after_5_failures(self, circuit_breaker):
        """Circuit should trip after exactly 5 consecutive failures."""
        # First 4 failures should NOT trip
        for i in range(4):
            tripped = circuit_breaker.record_failure()
            assert not tripped, f"Should not trip after {i + 1} failures"
            assert not circuit_breaker.is_open

        # 5th failure should trip
        tripped = circuit_breaker.record_failure()
        assert tripped, "Should trip after 5 failures"
        assert circuit_breaker.is_open
        assert circuit_breaker.state.total_trips == 1

    def test_circuit_trip_blocks_searches(self, circuit_breaker):
        """Tripped circuit should block and delay subsequent searches."""
        # Trip the circuit
        for _ in range(5):
            circuit_breaker.record_failure()

        assert circuit_breaker.is_open

        # Record start time
        start = time.time()

        # check_and_wait should block
        result = circuit_breaker.check_and_wait()

        # Should have waited for pause duration
        elapsed = time.time() - start
        assert elapsed >= 0.05, "Should have waited for pause"
        assert result is True  # Circuit breaker was enabled and we waited

        # Circuit should now be closed (half-open)
        assert not circuit_breaker.is_open

    def test_success_resets_failure_counter(self, circuit_breaker):
        """Success should reset consecutive failure counter."""
        # Record 4 failures
        for _ in range(4):
            circuit_breaker.record_failure()

        assert circuit_breaker.state.consecutive_failures == 4

        # Success should reset
        circuit_breaker.record_success()
        assert circuit_breaker.state.consecutive_failures == 0

        # Now 4 more failures should not trip (need 5 in a row)
        for _ in range(4):
            circuit_breaker.record_failure()

        assert not circuit_breaker.is_open


# =============================================================================
# Test: Retry queue waits for circuit breaker recovery
# =============================================================================


class TestRetryQueueWaitsForCircuitBreaker:
    """Test that retry queue respects circuit breaker state."""

    def test_retry_queue_waits_when_cb_tripped(self, linked_components):
        """Retry queue should wait for circuit breaker before processing."""
        circuit_breaker = linked_components["circuit_breaker"]
        retry_queue = linked_components["retry_queue"]

        # Add item to retry queue
        retry_queue.add("video1", "keyword1", "short", "429 error")

        # Trip the circuit breaker
        for _ in range(5):
            circuit_breaker.record_failure()

        assert circuit_breaker.is_open

        # Start retry pass - should wait for circuit breaker
        start = time.time()
        pass_num = retry_queue.start_retry_pass()
        elapsed = time.time() - start

        # Should have started a pass
        assert pass_num == 1
        # Should have waited for circuit breaker (though it's very short in tests)
        # The delay_seconds (0.01) is also added

    def test_retry_queue_proceeds_when_cb_not_tripped(self, linked_components):
        """Retry queue should proceed immediately when CB is not tripped."""
        retry_queue = linked_components["retry_queue"]

        # Add item to retry queue
        retry_queue.add("video1", "keyword1", "short", "error")

        # Start retry pass without tripping CB
        pass_num = retry_queue.start_retry_pass()

        assert pass_num == 1
        assert retry_queue.current_pass == 1


# =============================================================================
# Test: Budget exhaustion flows correctly
# =============================================================================


class TestBudgetExhaustionFlow:
    """Test budget exhaustion cascade behavior."""

    def test_budget_exhaustion_detected(self, budget):
        """Budget should detect when all resources exhausted."""
        # Initially not exhausted
        assert not budget.is_exhausted()

        # Use all rotations
        for _ in range(3):
            budget.record_rotation()

        # Still not exhausted (backoff and VPN remain)
        assert not budget.is_exhausted()
        assert budget.get_recommended_escalation() == "backoff"

        # Use all backoff
        budget.record_backoff(60.0)

        # Still not exhausted (VPN remains)
        assert not budget.is_exhausted()
        assert budget.get_recommended_escalation() == "vpn"

        # Use all VPN switches
        for _ in range(2):
            budget.record_vpn_switch()

        # NOW exhausted
        assert budget.is_exhausted()
        assert budget.get_recommended_escalation() == "exhausted"

    def test_budget_nearly_exhausted_threshold(self, budget):
        """Budget should detect >80% usage as nearly exhausted."""
        # Use 81% of rotations (3 max, 3 used = 100%)
        # Need to use less - let's use 2 of 3 = 67%, not nearly exhausted
        budget.record_rotation()
        budget.record_rotation()
        assert not budget.is_nearly_exhausted()  # 67% < 80%

        # Use 3rd = 100% > 80%
        budget.record_rotation()
        assert budget.is_nearly_exhausted()

    def test_budget_advice_skip_to_vpn(self, budget):
        """Budget should advise skipping to VPN when rotations exhausted."""
        # Initially should continue
        assert budget.get_budget_advice() == "continue"

        # Use all rotations
        for _ in range(3):
            budget.record_rotation()

        # Should now advise skip to VPN
        assert budget.get_budget_advice() == "skip_to_vpn"

    def test_budget_advice_abort_keyword(self, budget):
        """Budget should advise aborting when all resources exhausted."""
        # Exhaust all resources
        for _ in range(3):
            budget.record_rotation()
        for _ in range(2):
            budget.record_vpn_switch()

        # Should advise abort (even without using backoff, because can_backoff
        # is checked first in get_recommended_escalation)
        # Actually, backoff is still available, so we need to exhaust that too
        budget.record_backoff(60.0)

        assert budget.get_budget_advice() == "abort_keyword"


# =============================================================================
# Test: Circuit breaker pause scaling with budget state
# =============================================================================


class TestCircuitBreakerPauseScaling:
    """Test that circuit breaker pause duration scales with budget state."""

    def test_pause_extended_when_budget_nearly_exhausted(
        self, budget, circuit_breaker
    ):
        """Pause should be 1.5x when budget is nearly exhausted."""
        circuit_breaker._budget = budget

        # Normal pause
        base_pause = circuit_breaker.config.pause_seconds
        assert circuit_breaker._get_effective_pause_seconds() == base_pause

        # Make budget nearly exhausted (>80% of rotations)
        for _ in range(3):  # 100% of max_rotations=3
            budget.record_rotation()

        # Pause should now be 1.5x
        expected = base_pause * 1.5
        assert circuit_breaker._get_effective_pause_seconds() == expected

    def test_pause_extended_when_budget_exhausted(self, budget, circuit_breaker):
        """Pause should be 2.5x when budget is fully exhausted."""
        circuit_breaker._budget = budget

        # Exhaust all resources
        for _ in range(3):
            budget.record_rotation()
        for _ in range(2):
            budget.record_vpn_switch()
        budget.record_backoff(60.0)

        assert budget.is_exhausted()

        # Pause should be 2.5x
        base_pause = circuit_breaker.config.pause_seconds
        expected = base_pause * 2.5
        assert circuit_breaker._get_effective_pause_seconds() == expected

    def test_pause_capped_at_max(self, budget):
        """Extended pause should be capped at max_pause_seconds."""
        config = CircuitBreakerConfig(
            pause_seconds=200.0,  # High base
            max_pause_seconds=300.0,  # But capped
        )
        cb = CircuitBreaker(config)
        cb._budget = budget

        # Exhaust budget (2.5x multiplier = 500s, but capped at 300)
        for _ in range(3):
            budget.record_rotation()
        for _ in range(2):
            budget.record_vpn_switch()
        budget.record_backoff(60.0)

        # Should be capped
        assert cb._get_effective_pause_seconds() == 300.0


# =============================================================================
# Test: Full integration scenario
# =============================================================================


class TestFullIntegrationScenario:
    """Test complete rate limiting scenario with all components."""

    def test_rate_limit_cascade_scenario(self, linked_components):
        """Simulate realistic rate limiting cascade."""
        budget = linked_components["budget"]
        circuit_breaker = linked_components["circuit_breaker"]
        retry_queue = linked_components["retry_queue"]

        # Simulate: downloading 10 videos, 5 fail with 429
        failed_videos = [f"video_{i}" for i in range(5)]

        # Record failures in circuit breaker and add to retry queue
        for video_id in failed_videos:
            # Record 429 in budget
            budget.record_failure()
            budget.record_backoff(2.0)  # 2 seconds per failure

            # Record in circuit breaker
            circuit_breaker.record_failure()

            # Add to retry queue
            retry_queue.add(video_id, "keyword", "short", "HTTP 429")

        # Circuit should be tripped after 5 failures
        assert circuit_breaker.is_open
        assert circuit_breaker.state.total_trips == 1

        # Retry queue should have 5 items
        assert len(retry_queue.items) == 5

        # Budget should show 5 failures and 10s backoff
        assert budget.failures == 5
        assert budget.backoff_time_spent == 10.0

        # Start retry pass - should wait for circuit breaker
        pass_num = retry_queue.start_retry_pass()
        assert pass_num == 1

        # Simulate: 3 succeed on retry, 2 still fail
        for i, video_id in enumerate(failed_videos):
            if i < 3:
                retry_queue.mark_success(video_id)
                circuit_breaker.record_success()
            else:
                retry_queue.mark_failed(video_id)
                circuit_breaker.record_failure()

        retry_queue.finish_retry_pass()

        # Should have 2 items remaining for next pass
        assert len(retry_queue.items) == 2

        # Second retry pass
        pass_num = retry_queue.start_retry_pass()
        assert pass_num == 2

        # All remaining fail
        for video_id in list(retry_queue.items.keys()):
            retry_queue.mark_failed(video_id)

        retry_queue.finish_retry_pass()

        # All should be permanently failed now (max_passes=2 reached)
        assert len(retry_queue.items) == 0
        assert len(retry_queue._failed_ids) == 2

    def test_budget_snapshot_in_retry_queue(self, linked_components):
        """Retry queue should store budget state for decisions."""
        budget = linked_components["budget"]
        retry_queue = linked_components["retry_queue"]

        # Use some budget
        budget.record_rotation()
        budget.record_failure()

        # Set budget state snapshot
        retry_queue.set_budget_state(budget.get_summary())

        # Verify snapshot stored
        snapshot = retry_queue.get_budget_state()
        assert snapshot is not None
        assert snapshot["rotations_used"] == 1
        assert snapshot["failures"] == 1


# =============================================================================
# Test: Retry queue stats tracking
# =============================================================================


class TestRetryQueueStatsTracking:
    """Test retry queue statistics and metrics."""

    def test_stats_track_circuit_breaker_wait(self, linked_components):
        """Stats should track time spent waiting for circuit breaker."""
        circuit_breaker = linked_components["circuit_breaker"]
        retry_queue = linked_components["retry_queue"]

        # Add item and trip circuit
        retry_queue.add("video1", "keyword", "short", "error")
        for _ in range(5):
            circuit_breaker.record_failure()

        # Start retry pass (will wait for CB)
        retry_queue.start_retry_pass()

        # Stats should show CB wait time
        stats = retry_queue.get_stats()
        # The wait time should be recorded (may be very small in tests)
        assert "circuit_breaker_wait_time" in stats

    def test_retry_metrics_calculated(self, retry_queue):
        """Retry metrics should be calculated correctly."""
        # Add items
        retry_queue.add("video1", "kw", "short", "err")
        retry_queue.add("video2", "kw", "short", "err")

        # Mark one success
        retry_queue.mark_success("video1")

        # get_retry_metrics returns analysis metrics
        metrics = retry_queue.get_retry_metrics()
        assert "success_rate" in metrics
        assert "retry_efficiency" in metrics
        assert "total_processed" in metrics

        # get_stats returns the full stats summary
        stats = retry_queue.get_stats()
        assert "pending" in stats
        assert "completed" in stats


# =============================================================================
# Test: Edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_circuit_breaker_disabled(self):
        """Disabled circuit breaker should pass through."""
        config = CircuitBreakerConfig(enabled=False)
        cb = CircuitBreaker(config)

        # Record failures - should not trip
        for _ in range(10):
            tripped = cb.record_failure()
            assert not tripped

        assert not cb.is_open
        assert cb.check_and_wait() is False  # Returns False when disabled

    def test_retry_queue_disabled(self):
        """Disabled retry queue should not accept items."""
        config = BatchRetryConfig(enabled=False)
        rq = RetryQueue(config)

        added = rq.add("video1", "kw", "short", "err")
        assert not added
        assert not rq.has_pending()

    def test_budget_unlimited_resources(self):
        """Unlimited resources (0) should always be available."""
        budget = RateLimitBudget()
        budget.max_rotations = 0  # Unlimited
        budget.max_backoff_time = 0  # Unlimited
        budget.max_vpn_switches = 0  # Unlimited

        # Use many resources
        for _ in range(100):
            budget.record_rotation()
            budget.record_vpn_switch()
            budget.record_backoff(100.0)

        # Should still have availability
        assert budget.can_rotate()
        assert budget.can_switch_vpn()
        assert budget.can_backoff(1000.0)
        assert not budget.is_exhausted()

    def test_circuit_breaker_reset(self, circuit_breaker):
        """Manual reset should clear all failure state."""
        # Trip the circuit
        for _ in range(5):
            circuit_breaker.record_failure()

        assert circuit_breaker.is_open

        # Reset
        circuit_breaker.reset()

        assert not circuit_breaker.is_open
        assert circuit_breaker.state.consecutive_failures == 0

    def test_retry_queue_duplicate_prevention(self, retry_queue):
        """Same video should not be added twice."""
        added1 = retry_queue.add("video1", "kw", "short", "err1")
        added2 = retry_queue.add("video1", "kw", "short", "err2")

        assert added1 is True
        assert added2 is False  # Duplicate
        assert len(retry_queue.items) == 1

    def test_budget_scaling_for_keywords(self, budget):
        """Budget should scale for large keyword counts."""
        original_rotations = budget.max_rotations
        original_backoff = budget.max_backoff_time

        # Scale for 15 keywords (ceil(15/5) = 3x multiplier)
        budget.scale_for_keywords(15)

        assert budget.max_rotations == original_rotations * 3
        assert budget.max_backoff_time == original_backoff * 3

    def test_budget_serialization(self, budget):
        """Budget should serialize/deserialize correctly."""
        budget.record_rotation()
        budget.record_failure()
        budget.record_backoff(5.0)

        # Serialize
        data = budget.to_dict()

        # Create new budget and deserialize
        new_budget = RateLimitBudget.from_dict(data)

        assert new_budget.rotations_used == 1
        assert new_budget.failures == 1
        assert new_budget.backoff_time_spent == 5.0


# =============================================================================
# Test: Checkpoint persistence
# =============================================================================


class TestCheckpointPersistence:
    """Test checkpoint save/restore for all components."""

    def test_circuit_breaker_checkpoint(self, circuit_breaker):
        """Circuit breaker state should persist through checkpoint."""
        # Record some state
        for _ in range(3):
            circuit_breaker.record_failure()

        circuit_breaker._trip()  # Force trip for stats

        # Serialize
        data = circuit_breaker.to_checkpoint_dict()

        # Create new CB and restore
        new_cb = CircuitBreaker()
        new_cb.from_checkpoint_dict(data)

        # Cumulative stats restored
        assert new_cb.state.total_trips == 1
        # Transient state NOT restored (fresh start)
        assert new_cb.state.consecutive_failures == 0
        assert not new_cb.is_open

    def test_retry_queue_checkpoint(self, retry_queue):
        """Retry queue state should persist through checkpoint."""
        # Add items and start a pass
        retry_queue.add("video1", "kw", "short", "err")
        retry_queue.add("video2", "kw", "short", "err")
        retry_queue.start_retry_pass()
        retry_queue.mark_success("video1")

        # Serialize
        data = retry_queue.to_checkpoint_dict()

        # Create new queue and restore
        new_rq = RetryQueue()
        new_rq.from_checkpoint_dict(data)

        # State restored
        assert new_rq.current_pass == 1
        assert len(new_rq.items) == 1  # video2 still pending
        assert "video1" in new_rq._completed_ids

    def test_budget_checkpoint(self, budget):
        """Budget state should persist through checkpoint."""
        budget.record_rotation()
        budget.record_vpn_switch()
        budget.record_success()

        # Serialize and restore
        data = budget.to_dict()
        new_budget = RateLimitBudget.from_dict(data)

        assert new_budget.rotations_used == 1
        assert new_budget.vpn_switches_used == 1
        assert new_budget.successes == 1
        assert new_budget.max_rotations == budget.max_rotations


# =============================================================================
# Test: GlobalRateLimitCoordinator integration with VideoDownloader (US-35-002)
# =============================================================================


class TestGlobalRateLimitCoordinatorIntegration:
    """Test GlobalRateLimitCoordinator integration with VideoDownloader."""

    @pytest.fixture
    def coordinator(self):
        """Create a fresh coordinator instance for testing."""
        from src.rate_limit.coordinator import GlobalRateLimitCoordinator, RateLimitConfig
        GlobalRateLimitCoordinator.reset_instance()
        config = RateLimitConfig(enabled=True, slots_per_second=10.0, burst_size=5)
        return GlobalRateLimitCoordinator(config)

    def test_coordinator_initialized_in_downloader(self):
        """VideoDownloader should initialize GlobalRateLimitCoordinator."""
        from unittest.mock import MagicMock, patch
        from src.rate_limit.coordinator import GlobalRateLimitCoordinator

        # Reset singleton
        GlobalRateLimitCoordinator.reset_instance()

        # Mock config to avoid file system access
        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.cache_dir = "/tmp/cache"
        mock_config.downloaded_videos_dir = "/tmp/videos"
        mock_config.download.cookies_from_browser = ""
        mock_config.download.cookie_rotation = None
        mock_config.download.impersonation = None
        mock_config.download.vpn = None
        mock_config.download.speed_tracking = None
        mock_config.download.circuit_breaker = None
        mock_config.download.batch_retry = None
        mock_config.download.rate_limit = None
        mock_config.download.rate_limit_budget = None
        mock_config.download.cookies_file = None
        mock_config.download.keyword_alternatives = None

        with patch('src.downloader.core.CheckpointManager'), \
             patch('src.downloader.core.TranscodingManager'), \
             patch('src.downloader.core.TitleFilter'), \
             patch('src.downloader.core.SpeechScreener'), \
             patch('src.downloader.core.SearchOptimizer'), \
             patch('src.downloader.core.AudioFirstPipeline'), \
             patch('src.downloader.core.CookieMethodFallback'), \
             patch.object(GlobalRateLimitCoordinator, 'reset_instance'):

            from src.downloader.core import VideoDownloader
            # Reset before creating downloader
            GlobalRateLimitCoordinator.reset_instance()
            downloader = VideoDownloader(config=mock_config)

            # Should have coordinator attribute
            assert hasattr(downloader, 'rate_limit_coordinator')
            assert downloader.rate_limit_coordinator is not None

    def test_acquire_release_slot_methods_exist(self, coordinator):
        """Downloader should have acquire/release slot methods."""
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.cache_dir = "/tmp/cache"
        mock_config.downloaded_videos_dir = "/tmp/videos"
        mock_config.download.cookies_from_browser = ""
        mock_config.download.cookie_rotation = None
        mock_config.download.impersonation = None
        mock_config.download.vpn = None
        mock_config.download.speed_tracking = None
        mock_config.download.circuit_breaker = None
        mock_config.download.batch_retry = None
        mock_config.download.rate_limit = None
        mock_config.download.rate_limit_budget = None
        mock_config.download.cookies_file = None
        mock_config.download.keyword_alternatives = None

        with patch('src.downloader.core.CheckpointManager'), \
             patch('src.downloader.core.TranscodingManager'), \
             patch('src.downloader.core.TitleFilter'), \
             patch('src.downloader.core.SpeechScreener'), \
             patch('src.downloader.core.SearchOptimizer'), \
             patch('src.downloader.core.AudioFirstPipeline'), \
             patch('src.downloader.core.CookieMethodFallback'):

            from src.downloader.core import VideoDownloader
            from src.rate_limit.coordinator import GlobalRateLimitCoordinator
            GlobalRateLimitCoordinator.reset_instance()
            downloader = VideoDownloader(config=mock_config)

            # Methods should exist
            assert hasattr(downloader, 'acquire_download_slot')
            assert hasattr(downloader, 'release_download_slot')
            assert callable(downloader.acquire_download_slot)
            assert callable(downloader.release_download_slot)

    def test_slot_acquisition_basic_flow(self, coordinator):
        """Test basic slot acquisition and release."""
        # Acquire slot
        acquired = coordinator.acquire_slot('download', timeout=1.0)
        assert acquired is True

        # Check active slots
        active = coordinator.get_active_slots('download')
        assert active == 1

        # Release slot
        coordinator.release_slot('download')
        active = coordinator.get_active_slots('download')
        assert active == 0

    def test_slot_acquisition_respects_rate_limit(self, coordinator):
        """Slot acquisition should respect rate limits."""
        # Exhaust burst capacity (5 slots)
        for i in range(5):
            acquired = coordinator.acquire_slot('download', timeout=0.1, block=False)
            assert acquired is True, f"Should acquire slot {i+1}"

        # Next acquire should fail (non-blocking) or timeout
        acquired = coordinator.acquire_slot('download', timeout=0.1, block=False)
        assert acquired is False, "Should not acquire beyond burst capacity"

    def test_coordinator_metrics_tracking(self, coordinator):
        """Coordinator should track acquisition metrics."""
        # Acquire and release
        coordinator.acquire_slot('download')
        coordinator.release_slot('download')

        metrics = coordinator.get_metrics('download')
        assert metrics['total_acquired'] >= 1
        assert metrics['total_released'] >= 1

    def test_coordinator_status(self, coordinator):
        """Coordinator should report status correctly."""
        status = coordinator.get_status()

        assert 'enabled' in status
        assert 'slots_per_second' in status
        assert 'burst_size' in status
        assert 'available_tokens' in status
        assert status['enabled'] is True
        assert status['slots_per_second'] == 10.0
        assert status['burst_size'] == 5

    def test_coordinator_disabled_passthrough(self):
        """Disabled coordinator should pass through immediately."""
        from src.rate_limit.coordinator import GlobalRateLimitCoordinator, RateLimitConfig
        GlobalRateLimitCoordinator.reset_instance()
        config = RateLimitConfig(enabled=False)
        coordinator = GlobalRateLimitCoordinator(config)

        # Should always succeed when disabled
        for _ in range(100):
            acquired = coordinator.acquire_slot('download', block=False)
            assert acquired is True

    def test_coordinator_singleton_pattern(self):
        """Coordinator should use singleton pattern."""
        from src.rate_limit.coordinator import GlobalRateLimitCoordinator
        GlobalRateLimitCoordinator.reset_instance()

        c1 = GlobalRateLimitCoordinator()
        c2 = GlobalRateLimitCoordinator()

        assert c1 is c2, "Should return same instance"


# =============================================================================
# Test: Rate limit backoff config extraction (US-66-006)
# =============================================================================


class TestRateLimitBackoffConfigExtraction:
    """Test that RateLimitState/RateLimitTracker accept configurable backoff params."""

    def test_rate_limit_state_custom_base_backoff(self):
        """RateLimitState uses custom base_backoff_seconds for delay calculation."""
        from src.caption.timeout import RateLimitState

        state = RateLimitState(base_backoff_seconds=10.0, max_backoff_seconds=600.0, max_history=5)
        delay = state.record_rate_limit("vid1")
        # First rate limit: base * 2^0 = 10.0
        assert delay == 10.0
        assert state.max_backoff_seconds == 600.0
        assert state.max_history == 5

    def test_rate_limit_state_custom_max_backoff(self):
        """RateLimitState respects custom max_backoff_seconds cap."""
        from src.caption.timeout import RateLimitState

        state = RateLimitState(base_backoff_seconds=100.0, max_backoff_seconds=150.0)
        # 1st: min(100*1, 150) = 100
        state.record_rate_limit("v1")
        # 2nd: min(100*2, 150) = 150
        delay = state.record_rate_limit("v2")
        assert delay == 150.0
        # 3rd: min(100*4, 150) = 150 (capped)
        delay = state.record_rate_limit("v3")
        assert delay == 150.0

    def test_rate_limit_state_custom_max_history(self):
        """RateLimitState trims history to custom max_history."""
        from src.caption.timeout import RateLimitState

        state = RateLimitState(max_history=3)
        for i in range(5):
            state.record_rate_limit(f"v{i}")
            state.record_success()  # reset consecutive for predictable delays
        assert len(state.rate_limit_history) == 3

    def test_tracker_passes_config_to_state(self):
        """RateLimitTracker passes custom params to its internal RateLimitState."""
        from src.caption.timeout import RateLimitTracker

        # Reset singleton to allow fresh init with custom params
        RateLimitTracker._instance = None
        try:
            tracker = RateLimitTracker(
                base_backoff_seconds=20.0,
                max_backoff_seconds=100.0,
                max_history=3,
            )
            assert tracker._state.base_backoff_seconds == 20.0
            assert tracker._state.max_backoff_seconds == 100.0
            assert tracker._state.max_history == 3
        finally:
            # Reset singleton so other tests are unaffected
            RateLimitTracker._instance = None

    def test_caption_first_config_has_rate_limit_fields(self):
        """CaptionFirstConfig exposes rate limit backoff fields with correct defaults."""
        from src.config.sections.download import CaptionFirstConfig

        cfg = CaptionFirstConfig()
        assert cfg.rate_limit_base_backoff_seconds == 5.0
        assert cfg.rate_limit_max_backoff_seconds == 300.0
        assert cfg.rate_limit_max_history == 10

    def test_caption_first_config_custom_rate_limit_values(self):
        """CaptionFirstConfig accepts custom rate limit backoff values."""
        from src.config.sections.download import CaptionFirstConfig

        cfg = CaptionFirstConfig(
            rate_limit_base_backoff_seconds=15.0,
            rate_limit_max_backoff_seconds=600.0,
            rate_limit_max_history=20,
        )
        assert cfg.rate_limit_base_backoff_seconds == 15.0
        assert cfg.rate_limit_max_backoff_seconds == 600.0
        assert cfg.rate_limit_max_history == 20

    def test_caption_first_config_dict_construction(self):
        """CaptionFirstConfig works when constructed from dict (YAML loading)."""
        from src.config.sections.download import CaptionFirstConfig

        data = {
            'rate_limit_base_backoff_seconds': 8.0,
            'rate_limit_max_backoff_seconds': 120.0,
            'rate_limit_max_history': 5,
        }
        cfg = CaptionFirstConfig(**data)
        assert cfg.rate_limit_base_backoff_seconds == 8.0
        assert cfg.rate_limit_max_backoff_seconds == 120.0
        assert cfg.rate_limit_max_history == 5
