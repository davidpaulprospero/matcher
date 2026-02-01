"""
Tests for GlobalRateLimitCoordinator.

Verifies:
- Singleton pattern
- Slot acquisition and release
- Thread safety across concurrent operations
- Token bucket rate limiting
- Metrics tracking
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from src.rate_limit.coordinator import (
    GlobalRateLimitCoordinator,
    RateLimitConfig,
    SlotMetrics,
    OperationType,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    GlobalRateLimitCoordinator.reset_instance()
    yield
    GlobalRateLimitCoordinator.reset_instance()


class TestSingletonPattern:
    """Test singleton behavior."""

    def test_singleton_returns_same_instance(self):
        """Multiple instantiations return same instance."""
        coord1 = GlobalRateLimitCoordinator()
        coord2 = GlobalRateLimitCoordinator()
        assert coord1 is coord2

    def test_reset_instance_creates_new(self):
        """reset_instance allows creating new instance."""
        coord1 = GlobalRateLimitCoordinator()
        GlobalRateLimitCoordinator.reset_instance()
        coord2 = GlobalRateLimitCoordinator()
        assert coord1 is not coord2

    def test_config_updates_on_existing_instance(self):
        """Config can be updated on existing singleton."""
        config1 = RateLimitConfig(slots_per_second=1.0)
        coord1 = GlobalRateLimitCoordinator(config1)

        config2 = RateLimitConfig(slots_per_second=5.0)
        coord2 = GlobalRateLimitCoordinator(config2)

        assert coord1 is coord2
        assert coord1._config.slots_per_second == 5.0


class TestSlotAcquisition:
    """Test acquire_slot and release_slot methods."""

    def test_acquire_slot_when_available(self):
        """Slot acquisition succeeds when tokens available."""
        coord = GlobalRateLimitCoordinator()
        result = coord.acquire_slot('caption')
        assert result is True
        assert coord.get_active_slots('caption') == 1

    def test_release_slot_decrements_active(self):
        """Releasing slot decrements active count."""
        coord = GlobalRateLimitCoordinator()
        coord.acquire_slot('caption')
        coord.release_slot('caption')
        assert coord.get_active_slots('caption') == 0

    def test_acquire_multiple_operations(self):
        """Different operation types tracked separately."""
        coord = GlobalRateLimitCoordinator()

        coord.acquire_slot('caption')
        coord.acquire_slot('download')
        coord.acquire_slot('api')

        assert coord.get_active_slots('caption') == 1
        assert coord.get_active_slots('download') == 1
        assert coord.get_active_slots('api') == 1
        assert coord.get_active_slots() == 3  # Total

    def test_acquire_non_blocking_fails_when_empty(self):
        """Non-blocking acquire fails when no tokens."""
        config = RateLimitConfig(slots_per_second=0.1, burst_size=1)
        coord = GlobalRateLimitCoordinator(config)

        # Exhaust tokens
        assert coord.acquire_slot('caption', block=False) is True
        # Next should fail
        assert coord.acquire_slot('caption', block=False) is False

    def test_acquire_timeout_returns_false(self):
        """Acquire with timeout returns False on timeout."""
        config = RateLimitConfig(slots_per_second=0.1, burst_size=1)
        coord = GlobalRateLimitCoordinator(config)

        # Exhaust tokens
        coord.acquire_slot('caption', block=False)

        # Should timeout quickly
        start = time.time()
        result = coord.acquire_slot('caption', timeout=0.2)
        elapsed = time.time() - start

        assert result is False
        assert elapsed >= 0.2
        assert elapsed < 0.5  # Shouldn't wait too long

    def test_disabled_coordinator_always_succeeds(self):
        """Disabled coordinator always grants slots."""
        config = RateLimitConfig(enabled=False, slots_per_second=0.001)
        coord = GlobalRateLimitCoordinator(config)

        # Should succeed immediately even with tiny rate
        for _ in range(100):
            assert coord.acquire_slot('caption', block=False) is True


class TestTokenBucket:
    """Test token bucket rate limiting algorithm."""

    def test_tokens_refill_over_time(self):
        """Tokens refill based on slots_per_second."""
        config = RateLimitConfig(slots_per_second=10.0, burst_size=5)
        coord = GlobalRateLimitCoordinator(config)

        # Exhaust all tokens
        for _ in range(5):
            coord.acquire_slot('test', block=False)

        assert coord.get_available_tokens() < 1.0

        # Wait for refill (0.2 seconds = 2 tokens at 10/s)
        time.sleep(0.2)

        tokens = coord.get_available_tokens()
        assert tokens >= 1.5  # Should have ~2 tokens

    def test_burst_size_caps_tokens(self):
        """Tokens cannot exceed burst_size."""
        config = RateLimitConfig(slots_per_second=100.0, burst_size=3)
        coord = GlobalRateLimitCoordinator(config)

        # Wait for "refill"
        time.sleep(0.1)

        tokens = coord.get_available_tokens()
        assert tokens <= 3.0  # Capped at burst_size

    def test_acquire_waits_for_token(self):
        """Blocking acquire waits for token refill."""
        config = RateLimitConfig(slots_per_second=10.0, burst_size=1)
        coord = GlobalRateLimitCoordinator(config)

        # Exhaust token
        coord.acquire_slot('test', block=False)

        # Should wait ~100ms for next token
        start = time.time()
        result = coord.acquire_slot('test', timeout=1.0, block=True)
        elapsed = time.time() - start

        assert result is True
        assert elapsed >= 0.05  # Waited for token
        assert elapsed < 0.3  # Didn't wait too long


class TestThreadSafety:
    """Test thread safety with concurrent operations."""

    def test_concurrent_acquisitions(self):
        """Multiple threads can safely acquire slots."""
        config = RateLimitConfig(slots_per_second=100.0, burst_size=50)
        coord = GlobalRateLimitCoordinator(config)

        results = []
        lock = threading.Lock()

        def acquire_and_release():
            if coord.acquire_slot('test', timeout=5.0):
                with lock:
                    results.append(True)
                time.sleep(0.01)  # Simulate work
                coord.release_slot('test')
            else:
                with lock:
                    results.append(False)

        threads = [threading.Thread(target=acquire_and_release) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Most should succeed with high rate limit
        success_count = sum(results)
        assert success_count >= 15  # At least 75% success

    def test_concurrent_operations_metrics_accurate(self):
        """Metrics remain accurate under concurrent access."""
        config = RateLimitConfig(slots_per_second=50.0, burst_size=20)
        coord = GlobalRateLimitCoordinator(config)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(30):
                op_type = ['caption', 'download', 'api'][i % 3]
                futures.append(executor.submit(
                    lambda op=op_type: (
                        coord.acquire_slot(op, timeout=2.0) and
                        (time.sleep(0.01), coord.release_slot(op), True)[-1]
                    )
                ))

            for f in as_completed(futures):
                f.result()

        # All slots should be released
        assert coord.get_active_slots() == 0

        # Metrics should be consistent
        metrics = coord.get_metrics()
        for op_type in ['caption', 'download', 'api']:
            if op_type in metrics:
                op_metrics = metrics[op_type]
                assert op_metrics['total_acquired'] >= op_metrics['total_released']

    def test_slot_limiting_across_threads(self):
        """Rate limiting actually limits concurrent operations."""
        config = RateLimitConfig(slots_per_second=2.0, burst_size=2)
        coord = GlobalRateLimitCoordinator(config)

        acquisition_times = []
        lock = threading.Lock()

        def timed_acquire():
            start = time.time()
            if coord.acquire_slot('test', timeout=5.0):
                with lock:
                    acquisition_times.append(time.time() - start)
                coord.release_slot('test')

        # Start 6 threads - should take ~2 seconds with 2/s rate
        threads = [threading.Thread(target=timed_acquire) for _ in range(6)]
        overall_start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        overall_elapsed = time.time() - overall_start

        # Should take at least 1.5 seconds (6 ops at 2/s = 3s theoretical,
        # but burst allows some to complete immediately)
        assert overall_elapsed >= 1.0
        assert len(acquisition_times) == 6


class TestMetrics:
    """Test metrics tracking."""

    def test_metrics_track_acquisitions(self):
        """Metrics track total acquisitions."""
        coord = GlobalRateLimitCoordinator()

        for _ in range(5):
            coord.acquire_slot('caption')
            coord.release_slot('caption')

        metrics = coord.get_metrics('caption')
        assert metrics['total_acquired'] == 5
        assert metrics['total_released'] == 5
        assert metrics['active_slots'] == 0

    def test_metrics_track_rejections(self):
        """Metrics track acquisition rejections."""
        config = RateLimitConfig(slots_per_second=0.1, burst_size=1)
        coord = GlobalRateLimitCoordinator(config)

        coord.acquire_slot('test', block=False)  # Succeeds
        coord.acquire_slot('test', block=False)  # Rejected
        coord.acquire_slot('test', block=False)  # Rejected

        metrics = coord.get_metrics('test')
        assert metrics['rejections'] == 2

    def test_metrics_track_waits(self):
        """Metrics track wait time."""
        config = RateLimitConfig(slots_per_second=10.0, burst_size=1)
        coord = GlobalRateLimitCoordinator(config)

        # First succeeds immediately
        coord.acquire_slot('test', block=False)

        # Second must wait
        coord.acquire_slot('test', timeout=1.0)

        metrics = coord.get_metrics('test')
        assert metrics['total_waits'] >= 1
        assert metrics['total_wait_time'] > 0

    def test_get_status(self):
        """get_status returns current state."""
        config = RateLimitConfig(slots_per_second=5.0, burst_size=10)
        coord = GlobalRateLimitCoordinator(config)

        coord.acquire_slot('caption')
        coord.acquire_slot('download')

        status = coord.get_status()
        assert status['enabled'] is True
        assert status['slots_per_second'] == 5.0
        assert status['burst_size'] == 10
        assert status['total_active'] == 2
        assert status['active_slots']['caption'] == 1
        assert status['active_slots']['download'] == 1


class TestConfiguration:
    """Test configuration management."""

    def test_default_config(self):
        """Default config has sensible values."""
        coord = GlobalRateLimitCoordinator()
        status = coord.get_status()

        assert status['enabled'] is True
        assert status['slots_per_second'] == 2.0
        assert status['burst_size'] == 5

    def test_custom_config(self):
        """Custom config is applied."""
        config = RateLimitConfig(
            enabled=True,
            slots_per_second=10.0,
            burst_size=20
        )
        coord = GlobalRateLimitCoordinator(config)
        status = coord.get_status()

        assert status['slots_per_second'] == 10.0
        assert status['burst_size'] == 20

    def test_set_enabled(self):
        """Can enable/disable rate limiting."""
        coord = GlobalRateLimitCoordinator()

        coord.set_enabled(False)
        assert coord.is_enabled() is False

        coord.set_enabled(True)
        assert coord.is_enabled() is True

    def test_update_config(self):
        """Can update config at runtime."""
        coord = GlobalRateLimitCoordinator()

        new_config = RateLimitConfig(slots_per_second=20.0, burst_size=30)
        coord.update_config(new_config)

        status = coord.get_status()
        assert status['slots_per_second'] == 20.0
        assert status['burst_size'] == 30


class TestSlotMetrics:
    """Test SlotMetrics dataclass."""

    def test_to_dict(self):
        """SlotMetrics serializes correctly."""
        metrics = SlotMetrics(
            total_acquired=10,
            total_released=8,
            total_waits=3,
            total_wait_time=1.234,
            rejections=2
        )

        result = metrics.to_dict()
        assert result['total_acquired'] == 10
        assert result['total_released'] == 8
        assert result['total_waits'] == 3
        assert result['total_wait_time'] == 1.234
        assert result['rejections'] == 2
        assert result['active_slots'] == 2  # 10 - 8


class TestOperationType:
    """Test OperationType enum."""

    def test_operation_types(self):
        """All operation types defined."""
        assert OperationType.CAPTION.value == 'caption'
        assert OperationType.DOWNLOAD.value == 'download'
        assert OperationType.API.value == 'api'


class TestRateLimitConfig:
    """Test RateLimitConfig dataclass."""

    def test_default_values(self):
        """Default config has correct values."""
        config = RateLimitConfig()
        assert config.enabled is True
        assert config.slots_per_second == 2.0
        assert config.burst_size == 5

    def test_get_limit_for_operation(self):
        """Per-operation limits work correctly."""
        config = RateLimitConfig(
            slots_per_second=2.0,
            per_operation_limits={'caption': 1.0, 'download': 0.5}
        )

        assert config.get_limit_for_operation('caption') == 1.0
        assert config.get_limit_for_operation('download') == 0.5
        assert config.get_limit_for_operation('api') == 2.0  # Default


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""

    def test_caption_download_api_isolation(self):
        """Different operations are tracked separately."""
        config = RateLimitConfig(slots_per_second=10.0, burst_size=10)
        coord = GlobalRateLimitCoordinator(config)

        # Simulate mixed operations
        coord.acquire_slot('caption')
        coord.acquire_slot('caption')
        coord.acquire_slot('download')
        coord.acquire_slot('api')

        assert coord.get_active_slots('caption') == 2
        assert coord.get_active_slots('download') == 1
        assert coord.get_active_slots('api') == 1

        coord.release_slot('caption')
        coord.release_slot('download')

        assert coord.get_active_slots('caption') == 1
        assert coord.get_active_slots('download') == 0

    def test_realistic_rate_limiting(self):
        """Simulates realistic rate-limited workflow."""
        config = RateLimitConfig(slots_per_second=5.0, burst_size=3)
        coord = GlobalRateLimitCoordinator(config)

        completed = 0
        start = time.time()

        # Simulate 10 caption fetches
        for _ in range(10):
            if coord.acquire_slot('caption', timeout=5.0):
                time.sleep(0.01)  # Simulate work
                coord.release_slot('caption')
                completed += 1

        elapsed = time.time() - start

        assert completed == 10
        # Should take ~1.5-2 seconds (10 ops, 3 burst, then 5/s)
        assert elapsed >= 1.0
        assert elapsed < 4.0


class TestCaptionStageIntegration:
    """Test integration with CaptionStage (US-34-002)."""

    def test_caption_operation_slot_acquisition_release(self):
        """Slot is acquired and released for caption operations."""
        config = RateLimitConfig(slots_per_second=10.0, burst_size=5)
        coord = GlobalRateLimitCoordinator(config)

        # Simulate caption fetch workflow
        assert coord.acquire_slot('caption', timeout=1.0) is True
        assert coord.get_active_slots('caption') == 1

        # Simulate work
        time.sleep(0.01)

        # Release after fetch
        coord.release_slot('caption')
        assert coord.get_active_slots('caption') == 0

        # Verify metrics
        metrics = coord.get_metrics('caption')
        assert metrics['total_acquired'] == 1
        assert metrics['total_released'] == 1

    def test_parallel_caption_fetches_rate_limited(self):
        """Parallel caption fetches are rate limited by coordinator."""
        config = RateLimitConfig(slots_per_second=5.0, burst_size=3)
        coord = GlobalRateLimitCoordinator(config)

        results = []
        lock = threading.Lock()

        def simulate_caption_fetch(video_id: str):
            """Simulate a caption fetch with slot acquisition."""
            if coord.acquire_slot('caption', timeout=5.0):
                try:
                    time.sleep(0.05)  # Simulate network latency
                    with lock:
                        results.append({'video_id': video_id, 'success': True})
                finally:
                    coord.release_slot('caption')
            else:
                with lock:
                    results.append({'video_id': video_id, 'success': False})

        # Simulate 10 parallel fetches
        start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(simulate_caption_fetch, f'vid_{i}')
                for i in range(10)
            ]
            for f in as_completed(futures):
                f.result()
        elapsed = time.time() - start

        # All should succeed
        assert len(results) == 10
        assert all(r['success'] for r in results)

        # Rate limiting should have added some delay
        # 10 fetches at 5/s with burst 3 should take ~1.5+ seconds
        assert elapsed >= 1.0

        # All slots should be released
        assert coord.get_active_slots('caption') == 0

    def test_graceful_degradation_when_coordinator_disabled(self):
        """Caption fetches work when coordinator is disabled."""
        config = RateLimitConfig(enabled=False)
        coord = GlobalRateLimitCoordinator(config)

        # Should always succeed immediately
        for _ in range(100):
            assert coord.acquire_slot('caption', block=False) is True

        # Active slots not tracked when disabled
        coord.release_slot('caption')  # Should not error

    def test_slot_timeout_returns_false(self):
        """Slot acquisition timeout allows graceful handling."""
        config = RateLimitConfig(slots_per_second=0.5, burst_size=1)
        coord = GlobalRateLimitCoordinator(config)

        # Exhaust the single slot
        coord.acquire_slot('caption', block=False)

        # Try to acquire with short timeout
        start = time.time()
        result = coord.acquire_slot('caption', timeout=0.3)
        elapsed = time.time() - start

        assert result is False
        assert elapsed >= 0.3  # Waited full timeout
        assert elapsed < 0.5  # Didn't wait too long

    def test_metrics_track_caption_operations(self):
        """Metrics properly track caption-specific operations."""
        config = RateLimitConfig(slots_per_second=10.0, burst_size=5)
        coord = GlobalRateLimitCoordinator(config)

        # Simulate multiple fetches
        for _ in range(5):
            coord.acquire_slot('caption')
            coord.release_slot('caption')

        # Also do some download ops to verify isolation
        for _ in range(3):
            coord.acquire_slot('download')
            coord.release_slot('download')

        # Verify caption metrics
        caption_metrics = coord.get_metrics('caption')
        assert caption_metrics['total_acquired'] == 5
        assert caption_metrics['total_released'] == 5

        # Verify download metrics are separate
        download_metrics = coord.get_metrics('download')
        assert download_metrics['total_acquired'] == 3
        assert download_metrics['total_released'] == 3

    def test_coordinator_status_reports_caption_slots(self):
        """Status correctly shows active caption slots."""
        coord = GlobalRateLimitCoordinator()

        coord.acquire_slot('caption')
        coord.acquire_slot('caption')
        coord.acquire_slot('download')

        status = coord.get_status()
        assert status['active_slots']['caption'] == 2
        assert status['active_slots']['download'] == 1
        assert status['total_active'] == 3
