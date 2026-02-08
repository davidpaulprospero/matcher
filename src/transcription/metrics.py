"""
Transcription metrics tracking module.

Provides TranscriptionMetrics dataclass for tracking batch transcription statistics.
Similar pattern to CaptionMetrics in src/caption/metrics.py.

Created for US-60-009: Add batch transcription progress metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionMetrics:
    """
    Tracks batch transcription statistics for pipeline reporting.

    Implements US-60-009: Add batch transcription progress metrics.

    Tracks:
    - Total videos in batch
    - Cache hits (skipped transcription)
    - Newly transcribed count
    - Total audio duration processed
    - Real-time speed ratio (audio duration / transcription time)
    - Per-video timing for slow transcription detection

    Usage:
        metrics = TranscriptionMetrics()
        metrics.record_cache_hit("video1.mp4")
        metrics.record_transcription("video2.mp4", audio_duration=120.5, transcription_time=45.2)

        # Get summary dict:
        summary = metrics.get_summary_dict()
        print(f"Speed ratio: {summary['avg_speed_ratio']:.1f}x realtime")

        # Full summary:
        print(metrics.summary())

    Attributes:
        total_videos: Total videos in the batch
        cached_hits: Number of cache hits (transcriptions loaded from cache)
        transcribed_count: Number of videos actually transcribed
        failed_count: Number of videos that failed transcription
        total_audio_duration_s: Total audio duration transcribed in seconds
        total_transcription_time_s: Total time spent transcribing in seconds
        video_durations: Dict mapping video_path -> audio duration in seconds
        video_transcription_times: Dict mapping video_path -> transcription time in seconds
        phase1_extraction_time_s: Time spent in Phase 1 (audio extraction)
        phase2_transcription_time_s: Time spent in Phase 2 (GPU transcription)
    """

    # Core counts
    total_videos: int = 0
    cached_hits: int = 0
    transcribed_count: int = 0
    failed_count: int = 0

    # Duration tracking
    total_audio_duration_s: float = 0.0
    total_transcription_time_s: float = 0.0

    # Per-video tracking
    video_durations: Dict[str, float] = field(default_factory=dict)
    video_transcription_times: Dict[str, float] = field(default_factory=dict)

    # Phase timing
    phase1_extraction_time_s: float = 0.0
    phase2_transcription_time_s: float = 0.0

    # Retry budget summary (US-79-010)
    budget_total_attempts: int = 0
    budget_failed_attempts: int = 0
    budget_exhausted_count: int = 0  # Number of videos skipped due to budget exhaustion
    budget_summary: Optional[Dict[str, Any]] = None

    def record_cache_hit(self, video_path: str) -> None:
        """Record a cache hit (transcription loaded from cache).

        Args:
            video_path: Path to the video file.
        """
        self.cached_hits += 1
        logger.debug(f"Transcription cache hit for {video_path}")

    def record_transcription(
        self,
        video_path: str,
        audio_duration_s: float,
        transcription_time_s: float
    ) -> None:
        """Record a completed transcription with timing.

        Args:
            video_path: Path to the video file.
            audio_duration_s: Duration of the audio in seconds.
            transcription_time_s: Time taken to transcribe in seconds.
        """
        self.transcribed_count += 1
        self.total_audio_duration_s += audio_duration_s
        self.total_transcription_time_s += transcription_time_s
        self.video_durations[video_path] = audio_duration_s
        self.video_transcription_times[video_path] = transcription_time_s

        speed_ratio = audio_duration_s / transcription_time_s if transcription_time_s > 0 else 0.0
        logger.debug(
            f"Transcription completed for {video_path}: "
            f"{audio_duration_s:.1f}s audio in {transcription_time_s:.1f}s "
            f"({speed_ratio:.1f}x realtime)"
        )

    def record_failure(self, video_path: str) -> None:
        """Record a failed transcription.

        Args:
            video_path: Path to the video file that failed.
        """
        self.failed_count += 1
        logger.debug(f"Transcription failed for {video_path}")

    def set_retry_budget_summary(self, summary: Dict[str, Any]) -> None:
        """Set retry budget summary from TranscriptionRetryBudget (US-79-010).

        Args:
            summary: Dict from TranscriptionRetryBudget.get_summary().
        """
        self.budget_total_attempts = summary.get('total_attempts', 0)
        self.budget_failed_attempts = summary.get('failed_attempts', 0)
        self.budget_exhausted_count = summary.get('videos_skipped', 0)
        self.budget_summary = summary
        logger.debug(
            f"Retry budget summary: {self.budget_total_attempts} attempts, "
            f"{self.budget_failed_attempts} failures, "
            f"{self.budget_exhausted_count} skipped"
        )

    def set_phase_times(
        self,
        phase1_time: float,
        phase2_time: float
    ) -> None:
        """Set the phase timing after batch completion.

        Args:
            phase1_time: Time spent in Phase 1 (audio extraction) in seconds.
            phase2_time: Time spent in Phase 2 (GPU transcription) in seconds.
        """
        self.phase1_extraction_time_s = phase1_time
        self.phase2_transcription_time_s = phase2_time

    @property
    def avg_speed_ratio(self) -> float:
        """Calculate average real-time speed ratio.

        Returns:
            Speed ratio (audio duration / transcription time).
            Values > 1.0 mean faster than real-time.
        """
        if self.total_transcription_time_s <= 0:
            return 0.0
        return self.total_audio_duration_s / self.total_transcription_time_s

    @property
    def total_processed(self) -> int:
        """Total videos processed (cached + transcribed + failed)."""
        return self.cached_hits + self.transcribed_count + self.failed_count

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage.

        Returns:
            Cache hit rate percentage (0-100), or 0.0 if no processed videos.
        """
        if self.total_processed == 0:
            return 0.0
        return round(100.0 * self.cached_hits / self.total_processed, 1)

    @property
    def success_rate(self) -> float:
        """Calculate transcription success rate as percentage.

        Returns:
            Success rate percentage (0-100), or 0.0 if no transcriptions attempted.
        """
        attempted = self.transcribed_count + self.failed_count
        if attempted == 0:
            return 100.0 if self.cached_hits > 0 else 0.0
        return round(100.0 * self.transcribed_count / attempted, 1)

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary statistics as a dictionary.

        Returns:
            Dict with summary statistics:
            - total_videos: Total videos in batch
            - cached_hits: Number of cache hits
            - transcribed_count: Number of videos transcribed
            - failed_count: Number of failed transcriptions
            - total_duration_s: Total audio duration in seconds
            - total_time_s: Total transcription time in seconds
            - avg_speed_ratio: Average speed ratio (audio/time)
            - cache_hit_rate: Cache hit rate percentage
            - success_rate: Success rate percentage
            - phase1_time_s: Phase 1 extraction time
            - phase2_time_s: Phase 2 transcription time
        """
        return {
            'total_videos': self.total_videos,
            'cached_hits': self.cached_hits,
            'transcribed_count': self.transcribed_count,
            'failed_count': self.failed_count,
            'total_duration_s': round(self.total_audio_duration_s, 1),
            'total_time_s': round(self.total_transcription_time_s, 1),
            'avg_speed_ratio': round(self.avg_speed_ratio, 2),
            'cache_hit_rate': self.cache_hit_rate,
            'success_rate': self.success_rate,
            'phase1_time_s': round(self.phase1_extraction_time_s, 1),
            'phase2_time_s': round(self.phase2_transcription_time_s, 1),
            'budget_total_attempts': self.budget_total_attempts,
            'budget_failed_attempts': self.budget_failed_attempts,
            'budget_exhausted_count': self.budget_exhausted_count,
        }

    def get_slowest_videos(self, n: int = 5) -> List[tuple]:
        """Get the n slowest video transcriptions by time.

        Args:
            n: Number of slowest videos to return (default: 5).

        Returns:
            List of (video_path, transcription_time_s) tuples sorted by time descending.
        """
        sorted_times = sorted(
            self.video_transcription_times.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_times[:n]

    def summary(self) -> str:
        """Generate human-readable summary for pipeline report.

        Returns:
            Multi-line string suitable for printing in the report.
        """
        lines = []

        # Basic stats
        basic_stats = (
            f"Transcription: {self.total_videos} videos, "
            f"{self.cached_hits} cached, {self.transcribed_count} transcribed"
        )
        if self.failed_count > 0:
            basic_stats += f", {self.failed_count} failed"
        lines.append(basic_stats)

        # Cache and success rates
        if self.total_processed > 0:
            lines.append(
                f"  Cache hit rate: {self.cache_hit_rate}%, "
                f"Success rate: {self.success_rate}%"
            )

        # Speed metrics
        if self.transcribed_count > 0:
            lines.append(
                f"  Audio processed: {self.total_audio_duration_s:.1f}s in "
                f"{self.total_transcription_time_s:.1f}s "
                f"({self.avg_speed_ratio:.1f}x realtime)"
            )

        # Phase timing
        if self.phase1_extraction_time_s > 0 or self.phase2_transcription_time_s > 0:
            lines.append(
                f"  Phase 1 (extraction): {self.phase1_extraction_time_s:.1f}s, "
                f"Phase 2 (transcription): {self.phase2_transcription_time_s:.1f}s"
            )

        # Slowest videos
        slowest = self.get_slowest_videos(3)
        if slowest:
            from pathlib import Path
            slowest_str = ", ".join(
                f"{Path(v).stem[:20]}={t:.1f}s"
                for v, t in slowest
            )
            lines.append(f"  Slowest: {slowest_str}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary for checkpoint/JSON.

        Returns:
            Dict representation of all metrics.
        """
        return {
            'total_videos': self.total_videos,
            'cached_hits': self.cached_hits,
            'transcribed_count': self.transcribed_count,
            'failed_count': self.failed_count,
            'total_audio_duration_s': self.total_audio_duration_s,
            'total_transcription_time_s': self.total_transcription_time_s,
            'video_durations': dict(self.video_durations),
            'video_transcription_times': dict(self.video_transcription_times),
            'phase1_extraction_time_s': self.phase1_extraction_time_s,
            'phase2_transcription_time_s': self.phase2_transcription_time_s,
            'budget_total_attempts': self.budget_total_attempts,
            'budget_failed_attempts': self.budget_failed_attempts,
            'budget_exhausted_count': self.budget_exhausted_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionMetrics':
        """Create TranscriptionMetrics from dictionary.

        Args:
            data: Dict from to_dict() or checkpoint.

        Returns:
            New TranscriptionMetrics instance.
        """
        if not data:
            return cls()

        return cls(
            total_videos=data.get('total_videos', 0),
            cached_hits=data.get('cached_hits', 0),
            transcribed_count=data.get('transcribed_count', 0),
            failed_count=data.get('failed_count', 0),
            total_audio_duration_s=data.get('total_audio_duration_s', 0.0),
            total_transcription_time_s=data.get('total_transcription_time_s', 0.0),
            video_durations=data.get('video_durations', {}),
            video_transcription_times=data.get('video_transcription_times', {}),
            phase1_extraction_time_s=data.get('phase1_extraction_time_s', 0.0),
            phase2_transcription_time_s=data.get('phase2_transcription_time_s', 0.0),
            budget_total_attempts=data.get('budget_total_attempts', 0),
            budget_failed_attempts=data.get('budget_failed_attempts', 0),
            budget_exhausted_count=data.get('budget_exhausted_count', 0),
        )
