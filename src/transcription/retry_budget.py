"""
Batch retry budget management for transcription.

Provides TranscriptionRetryBudget to prevent infinite retry loops when
multiple videos fail transcription in a batch. Tracks total attempts,
failures, and cumulative backoff time across the batch.

Similar pattern to CaptionRetryBudget (src/caption/retry_budget.py) but
simpler since transcription retries are less complex (no format rotation,
VPN escalation, etc.).

Created for US-79-010.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionRetryBudget:
    """Tracks retry resources across a transcription batch (US-79-010).

    Prevents infinite retry loops by enforcing:
    - Maximum total attempts across all videos
    - Maximum cumulative backoff time

    When any limit is exceeded, is_exhausted() returns True, signaling
    transcribe_videos_parallel() to skip remaining retries.

    Thread Safety:
        All mutation methods are protected by a Lock for concurrent access.

    Attributes:
        attempts: Total transcription attempts across all videos.
        failures: Total failures across all videos.
        successes: Total successes across all videos.
        backoff_time_spent: Cumulative backoff delay (seconds).
        videos_skipped: Count of videos skipped due to budget exhaustion.
        skipped_video_ids: List of video IDs skipped due to budget exhaustion.
        max_attempts: Maximum allowed attempts (0 = unlimited).
        max_backoff_time: Maximum allowed backoff time in seconds (0 = unlimited).
    """

    # Usage counters
    attempts: int = 0
    failures: int = 0
    successes: int = 0
    backoff_time_spent: float = 0.0
    videos_skipped: int = 0
    skipped_video_ids: List[str] = field(default_factory=list)

    # Budget limits (set from config)
    max_attempts: int = 50
    max_backoff_time: float = 180.0  # 3 minutes total backoff budget

    # Thread-safety lock (RLock for reentrant calls from get_summary -> is_exhausted)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def record_attempt(self, video_id: str = "") -> None:
        """Record a transcription attempt.

        Args:
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.attempts += 1
            if self.attempts % 10 == 0:
                logger.debug(
                    f"TranscriptionRetryBudget: {self.attempts} attempts "
                    f"({self.failures} failures, {self.successes} successes)"
                )

    def record_failure(self, video_id: str = "") -> None:
        """Record a failed transcription attempt.

        Args:
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.failures += 1
            logger.debug(f"TranscriptionRetryBudget: failure for {video_id}")

    def record_success(self, video_id: str = "") -> None:
        """Record a successful transcription.

        Args:
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.successes += 1

    def record_backoff(self, delay: float) -> None:
        """Record backoff time spent waiting.

        Args:
            delay: Backoff delay in seconds.
        """
        with self._lock:
            self.backoff_time_spent += delay

    def record_skip(self, video_id: str) -> None:
        """Record a video skipped due to budget exhaustion.

        Args:
            video_id: ID of the skipped video.
        """
        with self._lock:
            self.videos_skipped += 1
            self.skipped_video_ids.append(video_id)
            logger.warning(
                f"TranscriptionRetryBudget: skipped {video_id} "
                f"(budget exhausted, {self.videos_skipped} total skipped)"
            )

    def is_exhausted(self) -> bool:
        """Check if the retry budget is exhausted.

        Returns:
            True if max_attempts or max_backoff_time exceeded.
        """
        with self._lock:
            if self.max_attempts > 0 and self.attempts >= self.max_attempts:
                return True
            if self.max_backoff_time > 0 and self.backoff_time_spent >= self.max_backoff_time:
                return True
            return False

    def exhaustion_reason(self) -> Optional[str]:
        """Return the reason for budget exhaustion, or None if not exhausted.

        Returns:
            String describing which limit was hit, or None.
        """
        with self._lock:
            if self.max_attempts > 0 and self.attempts >= self.max_attempts:
                return f"max_attempts reached ({self.attempts}/{self.max_attempts})"
            if self.max_backoff_time > 0 and self.backoff_time_spent >= self.max_backoff_time:
                return (
                    f"max_backoff_time reached "
                    f"({self.backoff_time_spent:.1f}s/{self.max_backoff_time:.1f}s)"
                )
            return None

    def get_summary(self) -> Dict:
        """Get budget summary for metrics/logging.

        Returns:
            Dict with budget usage summary.
        """
        with self._lock:
            return {
                'total_attempts': self.attempts,
                'failed_attempts': self.failures,
                'successful_attempts': self.successes,
                'backoff_time_spent': round(self.backoff_time_spent, 2),
                'videos_skipped': self.videos_skipped,
                'skipped_video_ids': list(self.skipped_video_ids),
                'max_attempts': self.max_attempts,
                'max_backoff_time': self.max_backoff_time,
                'is_exhausted': self.is_exhausted(),
                'exhaustion_reason': self.exhaustion_reason(),
            }
