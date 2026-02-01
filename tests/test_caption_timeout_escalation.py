"""Tests for caption timeout escalation state machine.

Tests the ProgressiveTimeoutManager and related timeout escalation
components in src/caption_timeout_manager.py.
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.caption_timeout_manager import (
    FormatTimeoutPolicy,
    ProgressiveTimeoutManager,
    TimeoutEscalationLevel,
    TimeoutEscalationPolicy,
)


class TestTimeoutEscalationLevels:
    """Test TimeoutEscalationLevel enum values."""

    def test_level_ordering(self):
        """Verify levels are ordered correctly."""
        levels = list(TimeoutEscalationLevel)
        assert levels[0] == TimeoutEscalationLevel.NORMAL
        assert levels[1] == TimeoutEscalationLevel.ELEVATED
        assert levels[2] == TimeoutEscalationLevel.HIGH
        assert levels[3] == TimeoutEscalationLevel.MAXIMUM

    def test_level_count(self):
        """There should be exactly 4 levels."""
        assert len(TimeoutEscalationLevel) == 4


class TestTimeoutEscalationPolicy:
    """Test TimeoutEscalationPolicy escalation logic."""

    def test_default_values(self):
        """Test default policy values."""
        policy = TimeoutEscalationPolicy()
        assert policy.base_timeout == 30.0
        assert policy.max_timeout == 120.0
        assert policy.escalation_multiplier == 1.5
        assert policy.reset_after_success is True

    def test_zero_failures_returns_base(self):
        """Zero failures returns base timeout."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0)
        assert policy.get_timeout(0) == 30.0

    def test_negative_failures_returns_base(self):
        """Negative failures returns base timeout."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0)
        assert policy.get_timeout(-1) == 30.0
        assert policy.get_timeout(-5) == 30.0

    def test_escalation_normal_to_elevated(self):
        """1 failure escalates to ELEVATED (1.5x base)."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0, escalation_multiplier=1.5)
        # 30 * 1.5 = 45
        assert policy.get_timeout(1) == 45.0
        assert policy.get_level(1) == TimeoutEscalationLevel.ELEVATED

    def test_escalation_elevated_to_high(self):
        """2 failures escalates to HIGH (2.25x base)."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0, escalation_multiplier=1.5)
        # 30 * 1.5^2 = 67.5
        assert policy.get_timeout(2) == 67.5
        assert policy.get_level(2) == TimeoutEscalationLevel.HIGH

    def test_escalation_high_to_maximum(self):
        """3+ failures escalates to MAXIMUM."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0, escalation_multiplier=1.5)
        # 30 * 1.5^3 = 101.25
        assert policy.get_timeout(3) == 101.25
        assert policy.get_level(3) == TimeoutEscalationLevel.MAXIMUM

    def test_escalation_capped_at_max_timeout(self):
        """Timeout is capped at max_timeout regardless of failures."""
        policy = TimeoutEscalationPolicy(
            base_timeout=30.0,
            max_timeout=120.0,
            escalation_multiplier=1.5
        )
        # 30 * 1.5^5 = 227.8, should cap at 120
        assert policy.get_timeout(5) == 120.0
        # Even more failures still capped
        assert policy.get_timeout(10) == 120.0

    def test_level_remains_maximum_after_3(self):
        """Level stays at MAXIMUM for 3+ failures."""
        policy = TimeoutEscalationPolicy()
        assert policy.get_level(3) == TimeoutEscalationLevel.MAXIMUM
        assert policy.get_level(4) == TimeoutEscalationLevel.MAXIMUM
        assert policy.get_level(100) == TimeoutEscalationLevel.MAXIMUM


class TestProgressiveTimeoutManager:
    """Test ProgressiveTimeoutManager state machine."""

    def test_initial_timeout(self):
        """Fresh video gets base timeout."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0)
        mgr = ProgressiveTimeoutManager(policy=policy)
        assert mgr.get_timeout("video123") == 30.0

    def test_escalation_on_timeout(self):
        """Timeout escalates after recording failure."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0, escalation_multiplier=1.5)
        mgr = ProgressiveTimeoutManager(policy=policy)

        # Initial
        assert mgr.get_timeout("video123") == 30.0

        # Record timeout -> ELEVATED
        mgr.record_timeout("video123")
        assert mgr.get_timeout("video123") == 45.0  # 30 * 1.5

        # Record timeout -> HIGH
        mgr.record_timeout("video123")
        assert mgr.get_timeout("video123") == 67.5  # 30 * 1.5^2

        # Record timeout -> MAXIMUM
        mgr.record_timeout("video123")
        assert mgr.get_timeout("video123") == 101.25  # 30 * 1.5^3 (capped at 3)

    def test_escalation_capped_at_maximum(self):
        """Escalation caps at MAXIMUM level (3 consecutive failures)."""
        policy = TimeoutEscalationPolicy(
            base_timeout=30.0,
            max_timeout=120.0,
            escalation_multiplier=1.5
        )
        mgr = ProgressiveTimeoutManager(policy=policy)

        # Record 5 timeouts
        for _ in range(5):
            mgr.record_timeout("video123")

        # Internal counter goes up but timeout calculation caps at 3
        assert mgr.get_failure_count("video123") == 5

        # But timeout is still calculated with cap
        # In get_timeout: total_escalation = min(consecutive + attempt, 3)
        # With 5 consecutive and attempt=0: min(5+0, 3) = 3
        # So: 30 * 1.5^3 = 101.25
        assert mgr.get_timeout("video123", attempt=0) == 101.25

    def test_reset_on_success(self):
        """Success resets escalation to NORMAL."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0, reset_after_success=True)
        mgr = ProgressiveTimeoutManager(policy=policy)

        # Escalate to MAXIMUM
        for _ in range(3):
            mgr.record_timeout("video123")

        assert mgr.get_failure_count("video123") == 3

        # Success resets
        mgr.record_success("video123")

        assert mgr.get_failure_count("video123") == 0
        assert mgr.get_timeout("video123") == 30.0  # Back to base

    def test_no_reset_when_disabled(self):
        """When reset_after_success=False, success doesn't reset."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0, reset_after_success=False)
        mgr = ProgressiveTimeoutManager(policy=policy)

        # Escalate
        mgr.record_timeout("video123")
        mgr.record_timeout("video123")

        assert mgr.get_failure_count("video123") == 2

        # Success should NOT reset
        mgr.record_success("video123")

        assert mgr.get_failure_count("video123") == 2  # Still 2

    def test_per_video_isolation(self):
        """Each video has independent escalation state."""
        mgr = ProgressiveTimeoutManager()

        # Video A gets 2 timeouts
        mgr.record_timeout("videoA")
        mgr.record_timeout("videoA")

        # Video B is fresh
        assert mgr.get_failure_count("videoA") == 2
        assert mgr.get_failure_count("videoB") == 0

        # Video B gets 1 timeout
        mgr.record_timeout("videoB")

        # They're independent
        assert mgr.get_failure_count("videoA") == 2
        assert mgr.get_failure_count("videoB") == 1

    def test_attempt_parameter_adds_to_escalation(self):
        """Attempt number adds to consecutive failures for timeout calc."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0, escalation_multiplier=1.5)
        mgr = ProgressiveTimeoutManager(policy=policy)

        # Fresh video, but attempt=1
        # total_escalation = min(0 + 1, 3) = 1
        assert mgr.get_timeout("video123", attempt=1) == 45.0  # 30 * 1.5

        # Fresh video, attempt=2
        # total_escalation = min(0 + 2, 3) = 2
        assert mgr.get_timeout("video123", attempt=2) == 67.5  # 30 * 1.5^2

    def test_reset_single_video(self):
        """Reset can target a single video."""
        mgr = ProgressiveTimeoutManager()

        mgr.record_timeout("videoA")
        mgr.record_timeout("videoB")

        assert mgr.get_failure_count("videoA") == 1
        assert mgr.get_failure_count("videoB") == 1

        mgr.reset("videoA")

        assert mgr.get_failure_count("videoA") == 0
        assert mgr.get_failure_count("videoB") == 1  # Unchanged

    def test_reset_all_videos(self):
        """Reset without argument clears all videos."""
        mgr = ProgressiveTimeoutManager()

        mgr.record_timeout("videoA")
        mgr.record_timeout("videoB")
        mgr.record_timeout("videoC")

        mgr.reset()

        assert mgr.get_failure_count("videoA") == 0
        assert mgr.get_failure_count("videoB") == 0
        assert mgr.get_failure_count("videoC") == 0


class TestFormatTimeoutPolicy:
    """Test per-format timeout policies."""

    def test_default_timeouts(self):
        """Test default format-specific timeouts."""
        policy = FormatTimeoutPolicy()

        assert policy.get_timeout("json3") == 45.0
        assert policy.get_timeout("srv3") == 30.0
        assert policy.get_timeout("vtt") == 25.0
        assert policy.get_timeout("srt") == 25.0

    def test_unknown_format_uses_default(self):
        """Unknown format gets default 30s timeout."""
        policy = FormatTimeoutPolicy()
        assert policy.get_timeout("unknown_format") == 30.0
        assert policy.get_timeout("xml") == 30.0

    def test_progressive_fallback_reduces_timeout(self):
        """When falling back to next format, timeout is reduced."""
        policy = FormatTimeoutPolicy(progressive_fallback=True, fallback_reduction=0.8)

        # First attempt (fallback_level=0): full timeout
        assert policy.get_timeout("json3", fallback_level=0) == 45.0

        # Second format (fallback_level=1): 80% of base
        assert policy.get_timeout("json3", fallback_level=1) == 36.0  # 45 * 0.8

        # Third format (fallback_level=2): 64% of base (use pytest.approx for float)
        assert policy.get_timeout("json3", fallback_level=2) == pytest.approx(28.8)  # 45 * 0.8^2

    def test_progressive_fallback_disabled(self):
        """When progressive_fallback=False, no reduction."""
        policy = FormatTimeoutPolicy(progressive_fallback=False)

        assert policy.get_timeout("json3", fallback_level=0) == 45.0
        assert policy.get_timeout("json3", fallback_level=1) == 45.0
        assert policy.get_timeout("json3", fallback_level=2) == 45.0

    def test_format_comparison(self):
        """json3 has highest timeout, vtt/srt lowest."""
        policy = FormatTimeoutPolicy()

        json3_timeout = policy.get_timeout("json3")
        srv3_timeout = policy.get_timeout("srv3")
        vtt_timeout = policy.get_timeout("vtt")

        assert json3_timeout > srv3_timeout > vtt_timeout


class TestRapidConsecutiveTimeouts:
    """Edge case: rapid consecutive timeouts."""

    def test_rapid_escalation_sequence(self):
        """Rapid timeouts escalate through all levels quickly."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0, escalation_multiplier=1.5)
        mgr = ProgressiveTimeoutManager(policy=policy)

        levels_seen = []
        timeouts_seen = []

        # Record 4 rapid timeouts
        for i in range(4):
            timeout = mgr.get_timeout("video123")
            level = policy.get_level(mgr.get_failure_count("video123"))

            timeouts_seen.append(timeout)
            levels_seen.append(level)

            mgr.record_timeout("video123")

        # Should have escalated through: NORMAL -> ELEVATED -> HIGH -> MAXIMUM
        assert levels_seen[0] == TimeoutEscalationLevel.NORMAL
        assert levels_seen[1] == TimeoutEscalationLevel.ELEVATED
        assert levels_seen[2] == TimeoutEscalationLevel.HIGH
        assert levels_seen[3] == TimeoutEscalationLevel.MAXIMUM

    def test_rapid_escalation_and_reset(self):
        """Rapid timeouts then success should reset to NORMAL."""
        mgr = ProgressiveTimeoutManager()

        # Rapidly escalate
        for _ in range(5):
            mgr.record_timeout("video123")

        assert mgr.get_failure_count("video123") == 5

        # Single success resets
        mgr.record_success("video123")

        assert mgr.get_failure_count("video123") == 0

    def test_rapid_alternating_success_failure(self):
        """Alternating success/failure prevents escalation."""
        mgr = ProgressiveTimeoutManager()

        for _ in range(10):
            mgr.record_timeout("video123")  # +1
            mgr.record_success("video123")  # reset to 0

        # Should never have escalated beyond initial
        assert mgr.get_failure_count("video123") == 0

    def test_rapid_concurrent_timeouts(self):
        """Thread safety: concurrent timeout recordings."""
        mgr = ProgressiveTimeoutManager()

        def record_timeouts(video_id: str, count: int):
            for _ in range(count):
                mgr.record_timeout(video_id)

        # Run 10 threads each recording 10 timeouts for same video
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(record_timeouts, "video123", 10)
                for _ in range(10)
            ]
            for f in futures:
                f.result()

        # Should have recorded exactly 100 timeouts
        assert mgr.get_failure_count("video123") == 100

    def test_burst_timeouts_multiple_videos(self):
        """Burst of timeouts across multiple videos."""
        mgr = ProgressiveTimeoutManager()

        # 3 rapid timeouts for 5 different videos
        for video_id in ["v1", "v2", "v3", "v4", "v5"]:
            for _ in range(3):
                mgr.record_timeout(video_id)

        # Each should be at level 3
        for video_id in ["v1", "v2", "v3", "v4", "v5"]:
            assert mgr.get_failure_count(video_id) == 3


class TestTimeoutEscalationStateTransitions:
    """Test full state machine transitions NORMAL -> ELEVATED -> HIGH -> MAXIMUM."""

    def test_full_escalation_cycle(self):
        """Complete escalation cycle: NORMAL -> ELEVATED -> HIGH -> MAXIMUM -> NORMAL."""
        policy = TimeoutEscalationPolicy(base_timeout=30.0)
        mgr = ProgressiveTimeoutManager(policy=policy)

        video_id = "test_video"

        # State: NORMAL (0 failures)
        assert policy.get_level(mgr.get_failure_count(video_id)) == TimeoutEscalationLevel.NORMAL

        # Transition: NORMAL -> ELEVATED
        mgr.record_timeout(video_id)
        assert policy.get_level(mgr.get_failure_count(video_id)) == TimeoutEscalationLevel.ELEVATED

        # Transition: ELEVATED -> HIGH
        mgr.record_timeout(video_id)
        assert policy.get_level(mgr.get_failure_count(video_id)) == TimeoutEscalationLevel.HIGH

        # Transition: HIGH -> MAXIMUM
        mgr.record_timeout(video_id)
        assert policy.get_level(mgr.get_failure_count(video_id)) == TimeoutEscalationLevel.MAXIMUM

        # State: MAXIMUM (stays at maximum)
        mgr.record_timeout(video_id)
        assert policy.get_level(mgr.get_failure_count(video_id)) == TimeoutEscalationLevel.MAXIMUM

        # Transition: MAXIMUM -> NORMAL (via success)
        mgr.record_success(video_id)
        assert policy.get_level(mgr.get_failure_count(video_id)) == TimeoutEscalationLevel.NORMAL

    def test_timeout_values_at_each_level(self):
        """Verify timeout values at each escalation level."""
        policy = TimeoutEscalationPolicy(
            base_timeout=30.0,
            max_timeout=120.0,
            escalation_multiplier=1.5
        )

        expected_timeouts = {
            TimeoutEscalationLevel.NORMAL: 30.0,      # base
            TimeoutEscalationLevel.ELEVATED: 45.0,   # 30 * 1.5
            TimeoutEscalationLevel.HIGH: 67.5,       # 30 * 1.5^2
            TimeoutEscalationLevel.MAXIMUM: 101.25,  # 30 * 1.5^3
        }

        for failures, (level, expected_timeout) in enumerate(expected_timeouts.items()):
            assert policy.get_level(failures) == level
            assert policy.get_timeout(failures) == expected_timeout


class TestEdgeCases:
    """Additional edge cases for timeout escalation."""

    def test_success_on_fresh_video_is_noop(self):
        """Recording success on never-failed video is safe."""
        mgr = ProgressiveTimeoutManager()

        # This should not raise
        mgr.record_success("never_failed_video")

        assert mgr.get_failure_count("never_failed_video") == 0

    def test_empty_video_id(self):
        """Empty video ID is handled."""
        mgr = ProgressiveTimeoutManager()

        mgr.record_timeout("")
        assert mgr.get_failure_count("") == 1

        mgr.record_success("")
        assert mgr.get_failure_count("") == 0

    def test_unicode_video_id(self):
        """Unicode video IDs are handled."""
        mgr = ProgressiveTimeoutManager()

        video_id = "视频_αβγ_🎬"
        mgr.record_timeout(video_id)
        assert mgr.get_failure_count(video_id) == 1

    def test_very_large_multiplier(self):
        """Large multiplier still respects max_timeout."""
        policy = TimeoutEscalationPolicy(
            base_timeout=10.0,
            max_timeout=50.0,
            escalation_multiplier=10.0  # Very aggressive
        )

        # 10 * 10 = 100, but capped at 50
        assert policy.get_timeout(1) == 50.0

    def test_very_small_multiplier(self):
        """Multiplier < 1 still works (decreasing timeouts)."""
        policy = TimeoutEscalationPolicy(
            base_timeout=100.0,
            max_timeout=200.0,
            escalation_multiplier=0.5  # Decreasing timeouts
        )

        # 100 * 0.5 = 50 (decreases with failures)
        assert policy.get_timeout(1) == 50.0
        assert policy.get_timeout(2) == 25.0
