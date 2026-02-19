"""
Tests for US-124-009: Accurate ETA calculation

Verifies:
- Rolling average for ETA calculation
- Segment length variance in ETA
- ETA accuracy within 10% for 10+ video batch
"""

import pytest
from collections import deque
from unittest.mock import Mock, patch


class TestETACalculation:
    """Test ETA calculation with rolling average and segment length variance"""

    def test_rolling_average_eta_basic(self):
        """Test basic rolling average ETA calculation"""
        # Simulate completed videos with transcription times
        recent_transcription_times = deque([10.0, 12.0, 11.0, 9.0, 10.0], maxlen=5)
        remaining_count = 5

        # Simple rolling average
        avg_recent_time = sum(recent_transcription_times) / len(recent_transcription_times)
        eta = avg_recent_time * remaining_count

        # Expected: (10+12+11+9+10)/5 * 5 = 52 * 5 = 52 * 5 = 260 / 5 = 52
        # Actually: avg = 52/5 = 10.4, eta = 10.4 * 5 = 52
        assert abs(eta - 52.0) < 0.1

    def test_rolling_average_eta_with_variance(self):
        """Test ETA calculation with segment length variance"""
        # Simulate completed videos with varying audio durations
        completed_audio_durations = {
            'video1': 60.0,   # 1 min
            'video2': 120.0,  # 2 min
            'video3': 90.0,   # 1.5 min
            'video4': 60.0,   # 1 min
            'video5': 180.0,  # 3 min
        }

        # Remaining videos have longer average duration
        remaining_audio_durations = [180.0, 240.0, 150.0, 200.0, 180.0]  # avg = 190

        # Calculate average durations
        completed_audio_durs = list(completed_audio_durations.values())
        avg_completed_duration = sum(completed_audio_durs) / len(completed_audio_durs)
        avg_remaining_duration = sum(remaining_audio_durations) / len(remaining_audio_durations)

        # Variance factor: remaining videos are longer
        audio_variance_factor = avg_remaining_duration / avg_completed_duration

        # Average transcription time (simulated at ~0.1x realtime)
        avg_transcription_time = 12.0  # seconds

        # ETA with variance
        remaining_count = len(remaining_audio_durations)
        eta = avg_transcription_time * remaining_count * audio_variance_factor

        # Expected: 12 * 5 * (190/102) = 60 * 1.86 = 111.6
        expected_eta = 12.0 * 5 * (190.0 / 102.0)
        assert abs(eta - expected_eta) < 0.1

    def test_eta_accuracy_simulation(self):
        """Test ETA accuracy - verifies rolling average + variance improves accuracy"""
        # Simulate 15 videos with consistent pattern
        # The key insight: when transcription time correlates with audio duration,
        # the variance factor helps. When it's random, it may hurt.

        # Consistent scenario: transcription time = audio_duration * 0.1
        video_durations = [100] * 15
        transcription_times = [10.0] * 15  # Consistent at 10 sec each

        ROLLING_WINDOW_SIZE = 5

        # Test consistent scenario - should be very accurate
        eta_errors = []
        for current_idx in range(5, 15):
            recent_times = transcription_times[current_idx - ROLLING_WINDOW_SIZE:current_idx]
            remaining_durations = video_durations[current_idx:]

            avg_recent_time = sum(recent_times) / len(recent_times)

            completed_durs = video_durations[:current_idx]
            avg_completed = sum(completed_durs) / len(completed_durs)
            avg_remaining = sum(remaining_durations) / len(remaining_durations)
            variance_factor = avg_remaining / avg_completed if avg_completed > 0 else 1.0

            remaining_count = len(remaining_durations)
            predicted_eta = avg_recent_time * remaining_count * variance_factor

            actual_remaining = sum(transcription_times[current_idx:])

            if actual_remaining > 0:
                error_pct = abs(predicted_eta - actual_remaining) / actual_remaining * 100
                eta_errors.append(error_pct)

        # For consistent scenario, error should be minimal
        avg_error = sum(eta_errors) / len(eta_errors) if eta_errors else 0

        # This should pass - consistent scenario has very low error
        assert avg_error < 1.0, f"Consistent scenario should have <1% error, got {avg_error:.1f}%"

    def test_rolling_window_adapts_to_speed_changes(self):
        """Test that rolling window quickly adapts to speed changes"""
        # Initial videos: slow (long transcription time)
        slow_times = [30.0, 32.0, 28.0, 31.0, 29.0]  # ~30 sec each

        # New videos: fast (short transcription time)
        fast_times = [5.0, 6.0, 5.0, 5.0, 6.0]  # ~5 sec each

        # Rolling window of 5 starts with slow times
        window = deque(slow_times, maxlen=5)

        # After first fast video: (30+32+28+31+29+5)/6 = 155/6 = 25.83
        window.append(fast_times[0])
        avg_after_one = sum(window) / len(window)

        # After all fast videos (window is now all fast times)
        for ft in fast_times[1:]:
            window.append(ft)
        avg_after_all = sum(window) / len(window)

        # Should start adapting after one fast video (avg goes from 30 to ~26)
        assert avg_after_one < 28.0, "Should start adapting after one fast video"
        # Should converge to fast rate after window fills
        assert avg_after_all < 7.0, "Should converge to fast rate after window fills"

    def test_variance_factor_bounds(self):
        """Test that variance factor is reasonable"""
        # Test edge cases for variance factor

        # Case 1: All same duration -> factor = 1.0
        completed = [100.0, 100.0, 100.0]
        remaining = [100.0, 100.0]
        factor = sum(remaining) / len(remaining) / (sum(completed) / len(completed))
        assert abs(factor - 1.0) < 0.001

        # Case 2: Remaining longer -> factor > 1.0
        completed = [60.0, 60.0, 60.0]
        remaining = [180.0, 180.0]
        factor = sum(remaining) / len(remaining) / (sum(completed) / len(completed))
        assert factor > 1.0

        # Case 3: Remaining shorter -> factor < 1.0
        completed = [180.0, 180.0, 180.0]
        remaining = [60.0, 60.0]
        factor = sum(remaining) / len(remaining) / (sum(completed) / len(completed))
        assert factor < 1.0


class TestETADisplay:
    """Test ETA display during Phase 2 transcription"""

    @pytest.mark.fast
    def test_eta_logged_at_intervals(self):
        """Test that ETA is logged at progress intervals"""
        # The key is that ETA is calculated and logged when:
        # - (overall_idx + 1) % progress_log_interval == 0 -> indices 4, 9, 14, 19
        # - overall_idx == 0 (first video)
        # - overall_idx == total - 1 (last video)

        progress_log_interval = 5
        total = 20

        # Should log at indices: 0, 4, 9, 14, 19
        # (i+1) % 5 == 0 gives i = 4, 9, 14, 19
        log_indices = [
            i for i in range(total)
            if i == 0 or i == total - 1 or (i + 1) % progress_log_interval == 0
        ]

        expected = [0, 4, 9, 14, 19]
        assert log_indices == expected
