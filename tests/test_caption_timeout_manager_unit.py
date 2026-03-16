"""Unit tests for caption_timeout_manager module.

Tests RateLimitState, RateLimitTracker, StalledOperationDetector, and
related components for caption rate limiting and timeout management.

These tests focus on thread safety, timeout escalation, per-video isolation,
and edge cases not covered by test_caption_timeout_escalation.py.
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

from src.caption_timeout_manager import (
    FormatTimeoutPolicy,
    ProgressiveTimeoutManager,
    RateLimitState,
    RateLimitTracker,
    StalledOperationDetector,
    TimeoutEscalationLevel,
    TimeoutEscalationPolicy,
)


# ============================================================================
# RateLimitState Tests
# ============================================================================


@pytest.mark.fast
class TestRateLimitStateBasics:
    """Test RateLimitState basic functionality."""

    def test_initial_state(self):
        """Fresh state has zero counters."""
        state = RateLimitState()
        assert state.last_rate_limit_time == 0.0
        assert state.consecutive_rate_limits == 0
        assert state.total_rate_limits == 0
        assert state.global_backoff_until == 0.0
        assert state.rate_limit_history == []

    def test_record_rate_limit_increments_counters(self):
        """Recording rate limit increments all counters."""
        state = RateLimitState()
        state.record_rate_limit("video1", jitter_factor=0.0)

        assert state.consecutive_rate_limits == 1
        assert state.total_rate_limits == 1
        assert state.last_rate_limit_time > 0
        assert len(state.rate_limit_history) == 1

    def test_consecutive_rate_limits_increment(self):
        """Consecutive rate limits increment the counter."""
        state = RateLimitState()

        state.record_rate_limit("v1", jitter_factor=0.0)
        assert state.consecutive_rate_limits == 1

        state.record_rate_limit("v2", jitter_factor=0.0)
        assert state.consecutive_rate_limits == 2

        state.record_rate_limit("v3", jitter_factor=0.0)
        assert state.consecutive_rate_limits == 3

    def test_record_success_resets_consecutive(self):
        """Success resets consecutive counter but keeps total."""
        state = RateLimitState()

        state.record_rate_limit("v1", jitter_factor=0.0)
        state.record_rate_limit("v2", jitter_factor=0.0)
        assert state.consecutive_rate_limits == 2
        assert state.total_rate_limits == 2

        state.record_success()

        assert state.consecutive_rate_limits == 0
        assert state.total_rate_limits == 2  # Total unchanged


@pytest.mark.fast
class TestRateLimitStateExponentialBackoff:
    """Test exponential backoff calculation."""

    def test_first_rate_limit_returns_base_delay(self):
        """First rate limit returns base delay."""
        state = RateLimitState(base_backoff_seconds=5.0)
        delay = state.record_rate_limit("v1", jitter_factor=0.0)
        assert delay == 5.0  # 5 * 2^0 = 5

    def test_exponential_backoff_growth(self):
        """Delay doubles with each consecutive rate limit."""
        state = RateLimitState(base_backoff_seconds=5.0, max_backoff_seconds=300.0)

        delay1 = state.record_rate_limit("v1", jitter_factor=0.0)
        delay2 = state.record_rate_limit("v2", jitter_factor=0.0)
        delay3 = state.record_rate_limit("v3", jitter_factor=0.0)
        delay4 = state.record_rate_limit("v4", jitter_factor=0.0)

        assert delay1 == 5.0    # 5 * 2^0 = 5
        assert delay2 == 10.0   # 5 * 2^1 = 10
        assert delay3 == 20.0   # 5 * 2^2 = 20
        assert delay4 == 40.0   # 5 * 2^3 = 40

    def test_backoff_capped_at_max(self):
        """Backoff is capped at max_backoff_seconds."""
        state = RateLimitState(base_backoff_seconds=100.0, max_backoff_seconds=200.0)

        # First: 100, Second: 200 (capped), Third: 200 (capped)
        delay1 = state.record_rate_limit("v1", jitter_factor=0.0)
        delay2 = state.record_rate_limit("v2", jitter_factor=0.0)
        delay3 = state.record_rate_limit("v3", jitter_factor=0.0)

        assert delay1 == 100.0
        assert delay2 == 200.0  # Capped
        assert delay3 == 200.0  # Still capped

    def test_jitter_applied_within_bounds(self):
        """Jitter randomizes delay within expected bounds."""
        state = RateLimitState(base_backoff_seconds=100.0, max_backoff_seconds=300.0)

        # With 20% jitter, delay should be in range [80, 120]
        delays = []
        for _ in range(50):
            state.consecutive_rate_limits = 0  # Reset for consistent base
            delay = state.record_rate_limit("v1", jitter_factor=0.2)
            delays.append(delay)

        # All delays should be within bounds
        for d in delays:
            assert 80.0 <= d <= 120.0, f"Delay {d} outside expected range"

        # With 50 samples, we should see some variation
        unique_delays = set(round(d, 2) for d in delays)
        assert len(unique_delays) > 1, "Expected jitter to create variation"

    def test_jitter_factor_clamped(self):
        """Jitter factor is clamped to [0.0, 1.0]."""
        state = RateLimitState(base_backoff_seconds=100.0, max_backoff_seconds=500.0)

        # Negative jitter should be treated as 0
        delay1 = state.record_rate_limit("v1", jitter_factor=-0.5)
        assert delay1 == 100.0  # No jitter applied

        state.consecutive_rate_limits = 0  # Reset

        # Jitter > 1.0 should be clamped to 1.0
        delays = []
        for _ in range(20):
            state.consecutive_rate_limits = 0
            delay = state.record_rate_limit("v1", jitter_factor=2.0)
            delays.append(delay)

        # With jitter clamped to 1.0, range is [0, 200] but still valid
        for d in delays:
            assert 0.0 <= d <= 200.0


@pytest.mark.fast
class TestRateLimitStatePauseLogic:
    """Test should_pause and get_recommended_delay."""

    def test_should_pause_during_backoff(self):
        """should_pause returns True during active backoff."""
        state = RateLimitState()
        state.record_rate_limit("v1", jitter_factor=0.0)

        assert state.should_pause() is True
        assert state.get_recommended_delay() > 0

    def test_should_not_pause_after_backoff_expires(self):
        """should_pause returns False after backoff expires."""
        state = RateLimitState(base_backoff_seconds=0.01)  # Very short
        state.record_rate_limit("v1", jitter_factor=0.0)

        time.sleep(0.02)  # Wait for backoff to expire

        assert state.should_pause() is False
        assert state.get_recommended_delay() == 0.0

    def test_fresh_state_does_not_pause(self):
        """Fresh state doesn't require pause."""
        state = RateLimitState()
        assert state.should_pause() is False
        assert state.get_recommended_delay() == 0.0


@pytest.mark.fast
class TestRateLimitStateHistory:
    """Test rate limit history tracking."""

    def test_history_records_events(self):
        """History records video IDs and timestamps."""
        state = RateLimitState()

        state.record_rate_limit("video1", jitter_factor=0.0)
        state.record_rate_limit("video2", jitter_factor=0.0)

        assert len(state.rate_limit_history) == 2
        assert state.rate_limit_history[0][1] == "video1"
        assert state.rate_limit_history[1][1] == "video2"

    def test_history_capped_at_max_history(self):
        """History is capped at max_history entries."""
        state = RateLimitState(max_history=5)

        for i in range(10):
            state.record_rate_limit(f"video{i}", jitter_factor=0.0)

        assert len(state.rate_limit_history) == 5
        # Should have the last 5 entries
        assert state.rate_limit_history[0][1] == "video5"
        assert state.rate_limit_history[-1][1] == "video9"

    def test_rate_limit_ratio_calculation(self):
        """Rate limit ratio calculated correctly."""
        state = RateLimitState()

        # Add 5 rate limits
        for i in range(5):
            state.record_rate_limit(f"v{i}", jitter_factor=0.0)

        # Ratio based on recent history
        ratio = state.get_rate_limit_ratio(window_seconds=60.0)
        assert ratio > 0.0

    def test_is_rate_limit_critical_by_consecutive(self):
        """Critical state triggered by 5+ consecutive rate limits."""
        state = RateLimitState()

        for i in range(4):
            state.record_rate_limit(f"v{i}", jitter_factor=0.0)
        assert state.is_rate_limit_critical() is False

        state.record_rate_limit("v5", jitter_factor=0.0)
        assert state.is_rate_limit_critical() is True


@pytest.mark.fast
class TestRateLimitStateSerialization:
    """Test RateLimitState serialization/deserialization."""

    def test_to_dict_roundtrip(self):
        """State survives to_dict/from_dict roundtrip."""
        state = RateLimitState()
        state.record_rate_limit("video1", jitter_factor=0.0)
        state.record_rate_limit("video2", jitter_factor=0.0)

        data = state.to_dict()
        restored = RateLimitState.from_dict(data)

        assert restored.consecutive_rate_limits == state.consecutive_rate_limits
        assert restored.total_rate_limits == state.total_rate_limits
        assert restored.last_rate_limit_time == state.last_rate_limit_time
        assert len(restored.rate_limit_history) == len(state.rate_limit_history)

    def test_from_dict_handles_missing_keys(self):
        """from_dict handles missing keys gracefully."""
        data = {}  # Empty dict
        state = RateLimitState.from_dict(data)

        assert state.consecutive_rate_limits == 0
        assert state.total_rate_limits == 0


# ============================================================================
# RateLimitTracker Tests (Singleton + Thread Safety)
# ============================================================================


@pytest.mark.fast
class TestRateLimitTrackerSingleton:
    """Test RateLimitTracker singleton behavior."""

    def test_singleton_returns_same_instance(self):
        """Multiple instantiations return same instance."""
        # Reset singleton for test
        RateLimitTracker._instance = None

        tracker1 = RateLimitTracker()
        tracker2 = RateLimitTracker()

        assert tracker1 is tracker2

    def test_singleton_shares_state(self):
        """Singleton instances share state."""
        RateLimitTracker._instance = None

        tracker1 = RateLimitTracker()
        tracker1.record_rate_limit("video1")

        tracker2 = RateLimitTracker()
        summary = tracker2.get_state_summary()

        assert summary["total_rate_limits"] >= 1


@pytest.mark.fast
class TestRateLimitTrackerThreadSafety:
    """Test RateLimitTracker thread safety."""

    def test_concurrent_rate_limit_recording(self):
        """Concurrent rate limit recordings don't corrupt state."""
        RateLimitTracker._instance = None
        tracker = RateLimitTracker()

        num_threads = 10
        records_per_thread = 50

        def record_limits():
            for i in range(records_per_thread):
                tracker.record_rate_limit(f"video_{threading.current_thread().name}_{i}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=record_limits, name=f"thread_{i}")
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        summary = tracker.get_state_summary()
        # Should have recorded all rate limits (consecutive resets on success though)
        assert summary["total_rate_limits"] == num_threads * records_per_thread

    def test_concurrent_success_and_rate_limit(self):
        """Concurrent success and rate_limit calls are safe."""
        RateLimitTracker._instance = None
        tracker = RateLimitTracker()

        num_iterations = 100
        errors = []

        def record_and_reset():
            try:
                for _ in range(num_iterations):
                    tracker.record_rate_limit("video1")
                    tracker.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_and_reset) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_wait_if_needed_respects_backoff(self):
        """wait_if_needed actually waits during backoff."""
        RateLimitTracker._instance = None
        tracker = RateLimitTracker()

        # Record rate limit to trigger backoff
        # Use internal state for very short backoff
        tracker._state.base_backoff_seconds = 0.05
        tracker.record_rate_limit("video1")

        start = time.time()
        waited = tracker.wait_if_needed()
        elapsed = time.time() - start

        # Should have waited approximately the backoff time
        assert waited > 0
        assert elapsed >= 0.04  # Allow some timing slack


# ============================================================================
# StalledOperationDetector Tests
# ============================================================================


@pytest.mark.fast
class TestStalledOperationDetectorBasics:
    """Test StalledOperationDetector basic functionality."""

    def test_initial_state(self):
        """Detector starts in non-running state."""
        detector = StalledOperationDetector(timeout_seconds=10.0)
        status = detector.get_status()

        assert status["is_running"] is False
        assert status["elapsed_seconds"] == 0.0

    def test_start_begins_monitoring(self):
        """start() begins monitoring."""
        detector = StalledOperationDetector(timeout_seconds=10.0)
        detector.start()

        status = detector.get_status()
        assert status["is_running"] is True
        assert status["elapsed_seconds"] >= 0.0

    def test_stop_ends_monitoring(self):
        """stop() ends monitoring."""
        detector = StalledOperationDetector(timeout_seconds=10.0)
        detector.start()
        detector.stop()

        status = detector.get_status()
        assert status["is_running"] is False

    def test_not_stalled_initially(self):
        """Not stalled immediately after start."""
        detector = StalledOperationDetector(timeout_seconds=0.1)
        detector.start()

        assert detector.is_stalled() is False

    def test_stalled_after_timeout(self):
        """Stalled if no progress for timeout_seconds."""
        detector = StalledOperationDetector(timeout_seconds=0.02)
        detector.start()

        time.sleep(0.03)  # Wait longer than timeout

        assert detector.is_stalled() is True

    def test_not_stalled_when_not_running(self):
        """Not stalled if not running (even after timeout)."""
        detector = StalledOperationDetector(timeout_seconds=0.01)
        # Never started

        time.sleep(0.02)

        assert detector.is_stalled() is False


@pytest.mark.fast
class TestStalledOperationDetectorProgress:
    """Test progress tracking in StalledOperationDetector."""

    def test_update_progress_resets_stall_timer(self):
        """update_progress resets the stall timer."""
        detector = StalledOperationDetector(timeout_seconds=0.05)
        detector.start()

        # Wait almost to timeout
        time.sleep(0.04)
        assert detector.is_stalled() is False

        # Update progress
        detector.update_progress()

        # Wait again - should not be stalled
        time.sleep(0.04)
        assert detector.is_stalled() is False

        # Now wait past timeout without progress
        time.sleep(0.06)
        assert detector.is_stalled() is True

    def test_elapsed_time_tracked(self):
        """elapsed() returns total time since start."""
        detector = StalledOperationDetector(timeout_seconds=10.0)
        detector.start()

        time.sleep(0.05)

        elapsed = detector.elapsed()
        assert elapsed >= 0.05

    def test_status_shows_since_last_progress(self):
        """Status shows time since last progress."""
        detector = StalledOperationDetector(timeout_seconds=1.0)
        detector.start()

        time.sleep(0.05)

        status = detector.get_status()
        assert status["since_last_progress"] >= 0.05


@pytest.mark.fast
class TestStalledOperationDetectorThreadSafety:
    """Test StalledOperationDetector thread safety."""

    def test_concurrent_progress_updates(self):
        """Concurrent progress updates are safe."""
        detector = StalledOperationDetector(timeout_seconds=1.0)
        detector.start()

        errors = []

        def update_progress():
            try:
                for _ in range(100):
                    detector.update_progress()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_progress) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert detector.is_stalled() is False

    def test_concurrent_start_stop(self):
        """Concurrent start/stop calls are safe."""
        detector = StalledOperationDetector(timeout_seconds=0.1)
        errors = []

        def start_stop_cycle():
            try:
                for _ in range(50):
                    detector.start()
                    detector.update_progress()
                    detector.stop()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=start_stop_cycle) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# ProgressiveTimeoutManager Additional Tests
# ============================================================================


@pytest.mark.fast
class TestProgressiveTimeoutManagerThreadSafety:
    """Test ProgressiveTimeoutManager thread safety for concurrent updates."""

    def test_concurrent_timeout_recording_isolated_videos(self):
        """Concurrent timeouts on different videos are isolated."""
        mgr = ProgressiveTimeoutManager()

        def record_for_video(video_id: str, count: int):
            for _ in range(count):
                mgr.record_timeout(video_id)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(record_for_video, f"video_{i}", 10)
                for i in range(10)
            ]
            for f in as_completed(futures):
                f.result()

        # Each video should have exactly 10 failures
        for i in range(10):
            assert mgr.get_failure_count(f"video_{i}") == 10

    def test_concurrent_success_and_timeout_same_video(self):
        """Concurrent success and timeout on same video is safe."""
        mgr = ProgressiveTimeoutManager()
        errors = []

        def timeout_loop():
            try:
                for _ in range(50):
                    mgr.record_timeout("shared_video")
            except Exception as e:
                errors.append(e)

        def success_loop():
            try:
                for _ in range(50):
                    mgr.record_success("shared_video")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=timeout_loop),
            threading.Thread(target=success_loop),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Final state is valid (either some count or 0)
        count = mgr.get_failure_count("shared_video")
        assert count >= 0


@pytest.mark.fast
class TestTimeoutEscalationMaxCap:
    """Test that max timeout cap is enforced correctly."""

    def test_max_timeout_enforced_at_all_levels(self):
        """Max timeout is enforced regardless of escalation level."""
        policy = TimeoutEscalationPolicy(
            base_timeout=50.0,
            max_timeout=60.0,
            escalation_multiplier=2.0
        )

        # Level 0: 50 (under max)
        assert policy.get_timeout(0) == 50.0

        # Level 1: 100 -> capped to 60
        assert policy.get_timeout(1) == 60.0

        # Level 2: 200 -> capped to 60
        assert policy.get_timeout(2) == 60.0

        # Level 10: huge -> capped to 60
        assert policy.get_timeout(10) == 60.0

    def test_progressive_manager_respects_policy_max(self):
        """ProgressiveTimeoutManager respects policy max timeout."""
        policy = TimeoutEscalationPolicy(
            base_timeout=30.0,
            max_timeout=50.0,
            escalation_multiplier=2.0
        )
        mgr = ProgressiveTimeoutManager(policy=policy)

        # Record many failures
        for _ in range(10):
            mgr.record_timeout("video1")

        # Timeout should be capped at 50
        timeout = mgr.get_timeout("video1")
        assert timeout <= 50.0


@pytest.mark.fast
class TestPerVideoTimeoutIsolation:
    """Test that per-video timeout tracking is properly isolated."""

    def test_different_videos_independent_escalation(self):
        """Different videos escalate independently."""
        policy = TimeoutEscalationPolicy(base_timeout=10.0, escalation_multiplier=2.0)
        mgr = ProgressiveTimeoutManager(policy=policy)

        # Video A: 2 failures
        mgr.record_timeout("video_a")
        mgr.record_timeout("video_a")

        # Video B: 0 failures
        # Video C: 1 failure
        mgr.record_timeout("video_c")

        assert mgr.get_timeout("video_a") == 40.0  # 10 * 2^2
        assert mgr.get_timeout("video_b") == 10.0  # Base
        assert mgr.get_timeout("video_c") == 20.0  # 10 * 2^1

    def test_success_resets_only_target_video(self):
        """Success on one video doesn't reset others."""
        mgr = ProgressiveTimeoutManager()

        mgr.record_timeout("video_a")
        mgr.record_timeout("video_a")
        mgr.record_timeout("video_b")
        mgr.record_timeout("video_b")

        mgr.record_success("video_a")

        assert mgr.get_failure_count("video_a") == 0
        assert mgr.get_failure_count("video_b") == 2  # Unchanged

    def test_reset_specific_video_leaves_others(self):
        """reset(video_id) only resets that video."""
        mgr = ProgressiveTimeoutManager()

        mgr.record_timeout("v1")
        mgr.record_timeout("v2")
        mgr.record_timeout("v3")

        mgr.reset("v2")

        assert mgr.get_failure_count("v1") == 1
        assert mgr.get_failure_count("v2") == 0
        assert mgr.get_failure_count("v3") == 1

    def test_reset_all_clears_everything(self):
        """reset() without args clears all videos."""
        mgr = ProgressiveTimeoutManager()

        for i in range(10):
            mgr.record_timeout(f"video_{i}")

        mgr.reset()

        for i in range(10):
            assert mgr.get_failure_count(f"video_{i}") == 0


# ============================================================================
# Additional Edge Cases
# ============================================================================


@pytest.mark.fast
class TestEdgeCasesExtended:
    """Extended edge case tests."""

    def test_rate_limit_empty_video_id(self):
        """Empty video ID is handled in rate limit tracking."""
        state = RateLimitState()
        delay = state.record_rate_limit("", jitter_factor=0.0)
        assert delay > 0
        assert state.rate_limit_history[0][1] == ""

    def test_stalled_detector_negative_timeout(self):
        """Negative timeout is effectively immediate stall."""
        detector = StalledOperationDetector(timeout_seconds=-1.0)
        detector.start()

        # Should be immediately stalled since timeout is negative
        assert detector.is_stalled() is True

    def test_format_policy_custom_timeouts(self):
        """Custom format timeouts work correctly."""
        policy = FormatTimeoutPolicy(
            timeouts={
                "custom1": 100.0,
                "custom2": 50.0,
            }
        )

        assert policy.get_timeout("custom1") == 100.0
        assert policy.get_timeout("custom2") == 50.0
        assert policy.get_timeout("unknown") == 30.0  # Default

    def test_rate_limit_tracker_is_critical(self):
        """is_critical() returns True at critical levels."""
        RateLimitTracker._instance = None
        tracker = RateLimitTracker()

        # Initially not critical
        assert tracker.is_critical() is False

        # Record 5 rate limits to trigger critical
        for i in range(5):
            tracker.record_rate_limit(f"v{i}")

        assert tracker.is_critical() is True

    def test_rate_limit_tracker_get_state_summary(self):
        """get_state_summary returns complete info."""
        RateLimitTracker._instance = None
        tracker = RateLimitTracker()

        tracker.record_rate_limit("test_video")

        summary = tracker.get_state_summary()

        assert "consecutive_rate_limits" in summary
        assert "total_rate_limits" in summary
        assert "should_pause" in summary
        assert "recommended_delay" in summary
        assert "rate_limit_ratio_60s" in summary
        assert "is_critical" in summary

        assert summary["consecutive_rate_limits"] >= 1
        assert summary["total_rate_limits"] >= 1
