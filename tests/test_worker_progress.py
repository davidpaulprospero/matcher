"""Tests for WorkerProgress and WorkerProgressTracker (US-008 Sprint 8).

Tests worker-level progress tracking with ETA calculation for parallel
caption fetching operations.
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from src.caption import (
    WorkerProgress,
    WorkerProgressTracker,
)


class TestWorkerProgress:
    """Tests for individual WorkerProgress tracking."""

    @pytest.mark.fast
    def test_init_default_values(self):
        """Test WorkerProgress initializes with correct defaults."""
        progress = WorkerProgress(worker_id=0)
        assert progress.worker_id == 0
        assert progress.current_video is None
        assert progress.start_time is None
        assert progress.videos_completed == 0
        assert progress.fetch_times == []
        assert progress.max_fetch_times == 10

    @pytest.mark.fast
    def test_start_video_sets_state(self):
        """Test start_video sets current_video and start_time."""
        progress = WorkerProgress(worker_id=1)
        progress.start_video("abc123")

        assert progress.current_video == "abc123"
        assert progress.start_time is not None
        assert progress.start_time > 0

    def test_complete_video_returns_elapsed(self):
        """Test complete_video returns elapsed time."""
        progress = WorkerProgress(worker_id=0)
        progress.start_video("video1")
        time.sleep(0.01)  # Small delay
        elapsed = progress.complete_video()

        assert elapsed > 0
        assert elapsed < 1.0  # Should be less than 1 second
        assert progress.current_video is None
        assert progress.start_time is None
        assert progress.videos_completed == 1

    def test_complete_video_records_fetch_time(self):
        """Test complete_video adds elapsed to fetch_times."""
        progress = WorkerProgress(worker_id=0)

        # Complete 3 videos
        for i in range(3):
            progress.start_video(f"video{i}")
            time.sleep(0.01)
            progress.complete_video()

        assert len(progress.fetch_times) == 3
        assert progress.videos_completed == 3
        for t in progress.fetch_times:
            assert t > 0

    @pytest.mark.fast
    def test_fetch_times_rolling_window(self):
        """Test fetch_times maintains rolling window."""
        progress = WorkerProgress(worker_id=0, max_fetch_times=3)

        # Complete more than max videos
        for i in range(5):
            progress.start_video(f"video{i}")
            progress.complete_video()

        assert len(progress.fetch_times) == 3
        assert progress.videos_completed == 5

    def test_get_elapsed_when_processing(self):
        """Test get_elapsed returns time since start."""
        progress = WorkerProgress(worker_id=0)
        progress.start_video("video1")
        time.sleep(0.01)

        elapsed = progress.get_elapsed()
        assert elapsed > 0
        assert elapsed < 1.0

    @pytest.mark.fast
    def test_get_elapsed_when_idle(self):
        """Test get_elapsed returns 0 when not processing."""
        progress = WorkerProgress(worker_id=0)
        assert progress.get_elapsed() == 0.0

    @pytest.mark.fast
    def test_is_stuck_true_when_exceeds_threshold(self):
        """Test is_stuck returns True when exceeding threshold."""
        progress = WorkerProgress(worker_id=0)
        progress.start_video("video1")
        progress.start_time = time.perf_counter() - 100  # Simulate 100s elapsed

        assert progress.is_stuck(threshold_seconds=60.0) is True

    @pytest.mark.fast
    def test_is_stuck_false_when_below_threshold(self):
        """Test is_stuck returns False when below threshold."""
        progress = WorkerProgress(worker_id=0)
        progress.start_video("video1")

        assert progress.is_stuck(threshold_seconds=60.0) is False

    @pytest.mark.fast
    def test_is_stuck_false_when_idle(self):
        """Test is_stuck returns False when not processing."""
        progress = WorkerProgress(worker_id=0)
        assert progress.is_stuck(threshold_seconds=60.0) is False

    @pytest.mark.fast
    def test_get_average_fetch_time(self):
        """Test get_average_fetch_time calculates correct average."""
        progress = WorkerProgress(worker_id=0)
        progress.fetch_times = [1.0, 2.0, 3.0]

        avg = progress.get_average_fetch_time()
        assert avg == 2.0

    @pytest.mark.fast
    def test_get_average_fetch_time_empty(self):
        """Test get_average_fetch_time returns 0 when no history."""
        progress = WorkerProgress(worker_id=0)
        assert progress.get_average_fetch_time() == 0.0

    @pytest.mark.fast
    def test_to_dict_serialization(self):
        """Test to_dict returns expected structure."""
        progress = WorkerProgress(worker_id=3)
        progress.current_video = "abc123"
        progress.start_time = time.perf_counter() - 10
        progress.videos_completed = 5
        progress.fetch_times = [1.0, 2.0, 3.0]

        data = progress.to_dict()

        assert data['worker_id'] == 3
        assert data['current_video'] == "abc123"
        assert data['elapsed_seconds'] >= 10
        assert data['videos_completed'] == 5
        assert data['avg_fetch_time'] == 2.0


class TestWorkerProgressTracker:
    """Tests for WorkerProgressTracker aggregate tracking."""

    @pytest.mark.fast
    def test_init_default_values(self):
        """Test WorkerProgressTracker initializes correctly."""
        tracker = WorkerProgressTracker(total_videos=100)

        assert tracker.total_videos == 100
        assert tracker.completed_videos == 0
        assert tracker.workers == {}
        assert tracker.all_fetch_times == []
        assert tracker.stuck_threshold_seconds == 60.0

    @pytest.mark.fast
    def test_initialize_workers(self):
        """Test initialize_workers creates correct number of workers."""
        tracker = WorkerProgressTracker(total_videos=50)
        tracker.initialize_workers(8)

        assert len(tracker.workers) == 8
        for i in range(8):
            assert i in tracker.workers
            assert tracker.workers[i].worker_id == i

    @pytest.mark.fast
    def test_worker_start_creates_if_missing(self):
        """Test worker_start creates worker if not exists."""
        tracker = WorkerProgressTracker(total_videos=10)
        tracker.worker_start(5, "video123")

        assert 5 in tracker.workers
        assert tracker.workers[5].current_video == "video123"

    def test_worker_complete_updates_counts(self):
        """Test worker_complete increments completed_videos."""
        tracker = WorkerProgressTracker(total_videos=10)
        tracker.initialize_workers(2)

        tracker.worker_start(0, "video1")
        time.sleep(0.01)
        tracker.worker_complete(0)

        assert tracker.completed_videos == 1
        assert len(tracker.all_fetch_times) == 1

    @pytest.mark.fast
    def test_get_active_workers(self):
        """Test get_active_workers returns workers with current_video."""
        tracker = WorkerProgressTracker(total_videos=10)
        tracker.initialize_workers(4)

        tracker.worker_start(0, "video1")
        tracker.worker_start(2, "video2")

        active = tracker.get_active_workers()
        assert len(active) == 2
        worker_ids = {w.worker_id for w in active}
        assert worker_ids == {0, 2}

    @pytest.mark.fast
    def test_get_slow_workers(self):
        """Test get_slow_workers identifies stuck workers."""
        tracker = WorkerProgressTracker(
            total_videos=10,
            stuck_threshold_seconds=30.0
        )
        tracker.initialize_workers(3)

        # Start all workers
        tracker.worker_start(0, "video1")
        tracker.worker_start(1, "video2")
        tracker.worker_start(2, "video3")

        # Simulate one worker being slow
        tracker.workers[1].start_time = time.perf_counter() - 100  # 100s ago

        slow = tracker.get_slow_workers()
        assert len(slow) == 1
        assert slow[0].worker_id == 1

    @pytest.mark.fast
    def test_calculate_eta_basic(self):
        """Test calculate_eta with known fetch times."""
        tracker = WorkerProgressTracker(total_videos=100)
        tracker.initialize_workers(4)

        # Set up known fetch times directly (simulating 10 completed videos at 2s each)
        tracker.all_fetch_times = [2.0] * 10
        tracker.completed_videos = 10

        # Start 4 workers so we have active workers
        for i in range(4):
            tracker.worker_start(i, f"video{i+10}")

        # With 4 active workers and 90 remaining videos at 2s each
        # ETA should be ~(90 * 2) / 4 = 45s
        eta = tracker.calculate_eta()

        # Allow 20% tolerance (as per acceptance criteria)
        expected = 45.0
        assert abs(eta - expected) / expected <= 0.20, f"ETA {eta} not within 20% of {expected}"

    @pytest.mark.fast
    def test_calculate_eta_no_history(self):
        """Test calculate_eta returns 0 with no history."""
        tracker = WorkerProgressTracker(total_videos=100)
        assert tracker.calculate_eta() == 0.0

    @pytest.mark.fast
    def test_calculate_eta_all_completed(self):
        """Test calculate_eta returns 0 when all completed."""
        tracker = WorkerProgressTracker(total_videos=10)
        tracker.completed_videos = 10
        tracker.all_fetch_times = [1.0, 2.0, 3.0]

        assert tracker.calculate_eta() == 0.0

    def test_get_worker_stats(self):
        """Test get_worker_stats returns complete statistics."""
        tracker = WorkerProgressTracker(
            total_videos=100,
            stuck_threshold_seconds=60.0
        )
        tracker.initialize_workers(8)

        # Start some workers
        tracker.worker_start(0, "video1")
        tracker.worker_start(1, "video2")
        tracker.worker_start(2, "video3")

        # Complete one
        time.sleep(0.01)
        tracker.worker_complete(2)

        stats = tracker.get_worker_stats()

        assert stats['active_count'] == 2
        assert stats['slow_count'] == 0
        assert stats['slow_workers'] == []
        assert stats['completed'] == 1
        assert stats['total'] == 100
        assert 'avg_fetch_time' in stats
        assert 'eta_seconds' in stats

    @pytest.mark.fast
    def test_format_progress_message(self):
        """Test format_progress_message returns readable string."""
        tracker = WorkerProgressTracker(
            total_videos=50,
            stuck_threshold_seconds=30.0
        )
        tracker.initialize_workers(4)

        tracker.worker_start(0, "video1")
        tracker.worker_start(1, "video2")

        msg = tracker.format_progress_message()
        assert "Workers: 2 active" in msg

    @pytest.mark.fast
    def test_format_progress_message_with_slow(self):
        """Test format_progress_message includes slow worker info."""
        tracker = WorkerProgressTracker(
            total_videos=50,
            stuck_threshold_seconds=30.0
        )
        tracker.initialize_workers(4)

        tracker.worker_start(0, "video1")
        tracker.worker_start(1, "slow_video")

        # Make worker 1 slow
        tracker.workers[1].start_time = time.perf_counter() - 100

        msg = tracker.format_progress_message()
        assert "Workers:" in msg
        assert "slow" in msg
        assert "slow_video" in msg


class TestWorkerProgressTrackerThreadSafety:
    """Tests for thread-safety of WorkerProgressTracker."""

    def test_concurrent_worker_operations(self):
        """Test concurrent worker_start/worker_complete calls."""
        tracker = WorkerProgressTracker(total_videos=100)
        tracker.initialize_workers(10)

        errors = []

        def worker_thread(worker_id, num_videos):
            try:
                for i in range(num_videos):
                    tracker.worker_start(worker_id, f"video_{worker_id}_{i}")
                    time.sleep(0.001)  # Tiny delay
                    tracker.worker_complete(worker_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker_thread, args=(i, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert tracker.completed_videos == 100


class TestETAAccuracy:
    """Tests verifying ETA calculation accuracy within 20%."""

    @pytest.mark.fast
    def test_eta_accuracy_with_uniform_times(self):
        """Test ETA accuracy within 20% for uniform fetch times."""
        tracker = WorkerProgressTracker(total_videos=100)
        tracker.initialize_workers(4)

        # Complete 10 videos at exactly 2.0s each
        tracker.all_fetch_times = [2.0] * 10
        tracker.completed_videos = 10

        # Simulate 4 active workers
        for i in range(4):
            tracker.worker_start(i, f"video{i}")

        # Expected: (100 - 10) * 2.0 / 4 = 45.0s
        eta = tracker.calculate_eta()
        expected = 45.0

        # Within 20%
        assert abs(eta - expected) / expected <= 0.20

    @pytest.mark.fast
    def test_eta_accuracy_with_variable_times(self):
        """Test ETA accuracy within 20% for variable fetch times."""
        tracker = WorkerProgressTracker(total_videos=100)
        tracker.initialize_workers(4)

        # Variable times averaging to 2.0s
        tracker.all_fetch_times = [1.0, 2.0, 3.0, 1.5, 2.5, 2.0, 1.8, 2.2, 1.9, 2.1]
        tracker.completed_videos = 10

        # Simulate 4 active workers
        for i in range(4):
            tracker.worker_start(i, f"video{i}")

        # Expected: (100 - 10) * avg(2.0) / 4 = 45.0s
        eta = tracker.calculate_eta()
        expected = 45.0

        # Within 20%
        tolerance = expected * 0.20
        assert abs(eta - expected) <= tolerance, f"ETA {eta} not within 20% of {expected}"

    @pytest.mark.fast
    def test_eta_accuracy_batch_50_videos(self):
        """Test ETA accuracy for batch of 50 videos (acceptance criteria)."""
        tracker = WorkerProgressTracker(total_videos=50)
        tracker.initialize_workers(8)

        # Simulate completing first 10 videos with variable times
        fetch_times = [
            1.5, 2.0, 1.8, 2.2, 1.9,
            2.1, 1.7, 2.3, 2.0, 1.8
        ]
        tracker.all_fetch_times = fetch_times
        tracker.completed_videos = 10

        # Simulate 8 active workers
        for i in range(8):
            tracker.worker_start(i, f"video{i}")

        # Calculate expected ETA
        avg_time = sum(fetch_times) / len(fetch_times)  # ~1.93s
        remaining = 50 - 10  # 40 videos
        expected = (remaining * avg_time) / 8  # ~9.65s

        eta = tracker.calculate_eta()

        # Within 20%
        tolerance = expected * 0.20
        assert abs(eta - expected) <= tolerance, \
            f"ETA {eta:.2f}s not within 20% of expected {expected:.2f}s"

    @pytest.mark.fast
    def test_eta_accounts_for_parallel_workers(self):
        """Test ETA correctly accounts for parallel workers."""
        tracker = WorkerProgressTracker(total_videos=100)
        tracker.all_fetch_times = [2.0] * 10
        tracker.completed_videos = 10

        # Test with 1 active worker
        tracker.workers = {0: WorkerProgress(worker_id=0)}
        tracker.worker_start(0, "video")
        eta_1_worker = tracker.calculate_eta()

        # Reset and test with 4 active workers
        tracker.workers = {i: WorkerProgress(worker_id=i) for i in range(4)}
        for i in range(4):
            tracker.worker_start(i, f"video{i}")
        eta_4_workers = tracker.calculate_eta()

        # 4 workers should give ~4x lower ETA
        ratio = eta_1_worker / eta_4_workers
        assert 3.5 <= ratio <= 4.5, f"Worker scaling ratio {ratio} not ~4x"
