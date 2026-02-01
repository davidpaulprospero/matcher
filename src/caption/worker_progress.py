"""
Worker progress tracking for parallel caption fetching.

Provides progress tracking for individual workers and aggregate ETA calculation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkerProgress:
    """Tracks progress and state of a single worker in parallel caption fetching (US-008 Sprint 8).

    Each worker in the ThreadPoolExecutor gets a WorkerProgress instance to track:
    - Which video it's currently processing
    - When it started processing that video
    - How many videos it has completed
    - Fetch time history for ETA calculation

    This enables:
    - Per-worker status in progress callbacks ("Worker 3: processing video abc123")
    - Stuck worker detection ("Worker 5 slow: video xyz >30s")
    - Accurate ETA calculation using rolling average of fetch times

    Attributes:
        worker_id: Unique identifier for this worker (0-indexed).
        current_video: Video ID currently being processed (None if idle).
        start_time: When current video processing started (time.perf_counter()).
        videos_completed: Number of videos this worker has completed.
        fetch_times: List of fetch times (seconds) for last N videos (rolling window).
        max_fetch_times: Maximum number of fetch times to keep for rolling average.
    """
    worker_id: int
    current_video: Optional[str] = None
    start_time: Optional[float] = None
    videos_completed: int = 0
    fetch_times: List[float] = field(default_factory=list)
    max_fetch_times: int = 10  # Rolling window for ETA calculation

    def start_video(self, video_id: str) -> None:
        """Mark start of processing a video.

        Args:
            video_id: YouTube video ID being processed.
        """
        self.current_video = video_id
        self.start_time = time.perf_counter()

    def complete_video(self) -> float:
        """Mark completion of current video and record fetch time.

        Returns:
            Elapsed time in seconds for this video.
        """
        elapsed = 0.0
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            # Add to rolling window
            self.fetch_times.append(elapsed)
            if len(self.fetch_times) > self.max_fetch_times:
                self.fetch_times.pop(0)

        self.videos_completed += 1
        self.current_video = None
        self.start_time = None
        return elapsed

    def get_elapsed(self) -> float:
        """Get elapsed time for current video (if processing).

        Returns:
            Elapsed seconds since start_video() was called, or 0.0 if idle.
        """
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    def is_stuck(self, threshold_seconds: float = 60.0) -> bool:
        """Check if worker is stuck on current video.

        Args:
            threshold_seconds: Time threshold for considering worker stuck.

        Returns:
            True if worker has been processing current video longer than threshold.
        """
        return self.get_elapsed() > threshold_seconds

    def get_average_fetch_time(self) -> float:
        """Get rolling average fetch time.

        Returns:
            Average fetch time in seconds, or 0.0 if no history.
        """
        if not self.fetch_times:
            return 0.0
        return sum(self.fetch_times) / len(self.fetch_times)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize worker state for progress callback.

        Returns:
            Dict with worker state suitable for progress updates.
        """
        return {
            'worker_id': self.worker_id,
            'current_video': self.current_video,
            'elapsed_seconds': self.get_elapsed(),
            'videos_completed': self.videos_completed,
            'avg_fetch_time': self.get_average_fetch_time(),
        }


@dataclass
class WorkerProgressTracker:
    """Tracks all workers and provides ETA calculation for batch operations (US-008 Sprint 8).

    Manages a pool of WorkerProgress instances and provides:
    - Aggregate statistics across all workers
    - ETA calculation using rolling average of last 10 fetch times
    - Slow worker detection with configurable threshold
    - Progress callback integration

    Thread Safety:
        All mutation methods are protected by a Lock for concurrent access
        from multiple worker threads.

    Attributes:
        total_videos: Total videos in the batch.
        completed_videos: Number of videos completed across all workers.
        workers: Dict mapping worker_id -> WorkerProgress.
        all_fetch_times: Rolling window of fetch times across all workers.
        stuck_threshold_seconds: Time threshold for considering a worker stuck.
    """
    total_videos: int = 0
    completed_videos: int = 0
    workers: Dict[int, WorkerProgress] = field(default_factory=dict)
    all_fetch_times: List[float] = field(default_factory=list)
    stuck_threshold_seconds: float = 60.0
    max_fetch_times: int = 10  # Rolling window size for ETA

    # Thread-safety lock (RLock for reentrant access from nested method calls)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def __post_init__(self):
        """Ensure workers dict is initialized."""
        if not self.workers:
            self.workers = {}

    def initialize_workers(self, num_workers: int) -> None:
        """Initialize worker progress instances.

        Args:
            num_workers: Number of workers in the pool.
        """
        with self._lock:
            self.workers = {
                i: WorkerProgress(worker_id=i, max_fetch_times=self.max_fetch_times)
                for i in range(num_workers)
            }

    def worker_start(self, worker_id: int, video_id: str) -> None:
        """Record that a worker started processing a video.

        Args:
            worker_id: Worker identifier.
            video_id: YouTube video ID being processed.
        """
        with self._lock:
            if worker_id not in self.workers:
                self.workers[worker_id] = WorkerProgress(
                    worker_id=worker_id, max_fetch_times=self.max_fetch_times
                )
            self.workers[worker_id].start_video(video_id)

    def worker_complete(self, worker_id: int) -> float:
        """Record that a worker completed processing.

        Args:
            worker_id: Worker identifier.

        Returns:
            Elapsed time for the completed video.
        """
        with self._lock:
            if worker_id not in self.workers:
                return 0.0

            elapsed = self.workers[worker_id].complete_video()
            self.completed_videos += 1

            # Add to global rolling window
            self.all_fetch_times.append(elapsed)
            if len(self.all_fetch_times) > self.max_fetch_times:
                self.all_fetch_times.pop(0)

            return elapsed

    def get_active_workers(self) -> List[WorkerProgress]:
        """Get list of workers currently processing videos.

        Returns:
            List of WorkerProgress instances with current_video set.
        """
        with self._lock:
            return [w for w in self.workers.values() if w.current_video is not None]

    def get_slow_workers(self) -> List[WorkerProgress]:
        """Get workers that are stuck (exceeding threshold).

        Returns:
            List of WorkerProgress instances exceeding stuck_threshold_seconds.
        """
        with self._lock:
            return [
                w for w in self.workers.values()
                if w.current_video is not None and w.is_stuck(self.stuck_threshold_seconds)
            ]

    def calculate_eta(self) -> float:
        """Calculate estimated time remaining using rolling average.

        Uses the last 10 fetch times across all workers to estimate
        average time per video, then multiplies by remaining videos.

        Returns:
            Estimated seconds remaining, or 0.0 if cannot calculate.
        """
        with self._lock:
            if not self.all_fetch_times:
                return 0.0

            remaining = self.total_videos - self.completed_videos
            if remaining <= 0:
                return 0.0

            avg_time = sum(self.all_fetch_times) / len(self.all_fetch_times)

            # Account for parallel workers
            num_active = max(1, len(self.get_active_workers()))
            eta = (remaining * avg_time) / num_active

            return eta

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics for progress callback.

        Returns:
            Dict with aggregate worker stats:
            - active_count: Number of workers currently processing
            - slow_workers: List of (worker_id, video_id, elapsed) for slow workers
            - avg_fetch_time: Rolling average fetch time
            - eta_seconds: Estimated time remaining
        """
        with self._lock:
            active = self.get_active_workers()
            slow = self.get_slow_workers()

            slow_info = [
                {
                    'worker_id': w.worker_id,
                    'video_id': w.current_video,
                    'elapsed_seconds': w.get_elapsed(),
                }
                for w in slow
            ]

            avg_time = 0.0
            if self.all_fetch_times:
                avg_time = sum(self.all_fetch_times) / len(self.all_fetch_times)

            return {
                'active_count': len(active),
                'slow_count': len(slow),
                'slow_workers': slow_info,
                'avg_fetch_time': avg_time,
                'eta_seconds': self.calculate_eta(),
                'completed': self.completed_videos,
                'total': self.total_videos,
            }

    def format_progress_message(self) -> str:
        """Format worker stats as a human-readable progress message.

        Returns:
            String like "Workers: 8 active, 1 slow (video abc >30s)"
        """
        with self._lock:
            stats = self.get_worker_stats()
            active = stats['active_count']
            slow = stats['slow_count']

            msg = f"Workers: {active} active"
            if slow > 0:
                slow_info = stats['slow_workers'][0]  # Show first slow worker
                msg += f", {slow} slow (video {slow_info['video_id']} >{slow_info['elapsed_seconds']:.0f}s)"

            return msg
