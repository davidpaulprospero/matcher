"""
Transcription metrics tracking module.

Provides TranscriptionMetrics dataclass for tracking batch transcription statistics.
Similar pattern to CaptionMetrics in src/caption/metrics.py.

Created for US-60-009: Add batch transcription progress metrics.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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

    # GPU memory tracking (US-110-006)
    gpu_memory_peak_mb: float = 0.0  # Peak GPU memory used during transcription
    video_gpu_memory_usage: Dict[str, float] = field(default_factory=dict)  # video_path -> memory_mb

    # Quality metrics (US-110-007)
    avg_word_confidence: float = 0.0  # Average word-level confidence across all transcripts
    min_segment_confidence: float = 1.0  # Minimum segment confidence across all transcripts
    video_confidences: Dict[str, Dict[str, float]] = field(default_factory=dict)  # video -> {avg_word_conf, min_seg_conf}

    # Segment quality filtering (US-110-009)
    segment_rejection_count: int = 0  # Number of segments filtered due to low word count

    # Pipeline efficiency tracking (US-137-006)
    pipeline_avg_extraction_wait_s: float = 0.0  # Average time waiting for extraction
    pipeline_avg_transcription_s: float = 0.0  # Average transcription time
    pipeline_efficiency_ratio: float = 0.0  # Wait time / transcription time ratio

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

    def record_gpu_memory(self, video_path: str, memory_mb: float) -> None:
        """Record GPU memory usage for a transcribed video (US-110-006).

        Args:
            video_path: Path to the video file.
            memory_mb: GPU memory used in MB.
        """
        self.video_gpu_memory_usage[video_path] = memory_mb
        if memory_mb > self.gpu_memory_peak_mb:
            self.gpu_memory_peak_mb = memory_mb
        logger.debug(f"GPU memory for {Path(video_path).name}: {memory_mb:.1f} MB")

    def record_confidence(
        self,
        video_path: str,
        avg_word_confidence: float,
        min_segment_confidence: float
    ) -> None:
        """Record confidence metrics for a transcribed video (US-110-007).

        Args:
            video_path: Path to the video file.
            avg_word_confidence: Average word-level confidence (0-1).
            min_segment_confidence: Minimum segment confidence (0-1).
        """
        self.video_confidences[video_path] = {
            'avg_word_confidence': avg_word_confidence,
            'min_segment_confidence': min_segment_confidence
        }

        # Update aggregate metrics
        if self.transcribed_count > 0:
            # Recalculate weighted average across all videos
            total_weight = sum(
                v['avg_word_confidence'] * self.video_durations.get(k, 1.0)
                for k, v in self.video_confidences.items()
            )
            total_duration = sum(
                self.video_durations.get(k, 1.0)
                for k in self.video_confidences.keys()
            )
            if total_duration > 0:
                self.avg_word_confidence = total_weight / total_duration
            else:
                self.avg_word_confidence = avg_word_confidence
        else:
            self.avg_word_confidence = avg_word_confidence

        # Update minimum segment confidence
        if min_segment_confidence < self.min_segment_confidence:
            self.min_segment_confidence = min_segment_confidence

        # Log quality warning if confidence is below threshold (US-110-007)
        quality_threshold = 0.7
        if avg_word_confidence < quality_threshold:
            logger.warning(
                f"Low transcription quality for {Path(video_path).name}: "
                f"avg_word_confidence={avg_word_confidence:.2f} < {quality_threshold}"
            )

    def record_segment_rejections(self, rejected_count: int, video_path: str = "") -> None:
        """Record the number of segments rejected due to low word count (US-110-009).

        Args:
            rejected_count: Number of segments filtered out due to low word count.
            video_path: Optional path to the video for logging purposes.
        """
        self.segment_rejection_count += rejected_count
        if rejected_count > 0 and video_path:
            logger.warning(
                f"Filtered {rejected_count} low-quality segment(s) from {Path(video_path).name} "
                f"(fewer than min_segment_words)"
            )

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

    def set_pipeline_efficiency(
        self,
        avg_extraction_wait_s: float,
        avg_transcription_s: float,
        efficiency_ratio: float
    ) -> None:
        """Set pipeline efficiency metrics (US-137-006).

        Args:
            avg_extraction_wait_s: Average time waiting for audio extraction to complete.
            avg_transcription_s: Average time spent transcribing.
            efficiency_ratio: Ratio of wait time to transcription time.
        """
        self.pipeline_avg_extraction_wait_s = avg_extraction_wait_s
        self.pipeline_avg_transcription_s = avg_transcription_s
        self.pipeline_efficiency_ratio = efficiency_ratio

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
            - gpu_memory_peak_mb: Peak GPU memory used (US-110-006)
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
            'gpu_memory_peak_mb': round(self.gpu_memory_peak_mb, 1),
            'avg_word_confidence': round(self.avg_word_confidence, 3),
            'min_segment_confidence': round(self.min_segment_confidence, 3),
            'segment_rejection_count': self.segment_rejection_count,  # US-110-009
            # Pipeline efficiency (US-137-006)
            'pipeline_avg_extraction_wait_s': round(self.pipeline_avg_extraction_wait_s, 2),
            'pipeline_avg_transcription_s': round(self.pipeline_avg_transcription_s, 2),
            'pipeline_efficiency_ratio': round(self.pipeline_efficiency_ratio, 2),
        }

    def get_highest_memory_videos(self, n: int = 5) -> List[tuple]:
        """Get the n videos with highest GPU memory usage (US-110-006).

        Args:
            n: Number of highest memory videos to return (default: 5).

        Returns:
            List of (video_path, memory_mb) tuples sorted by memory descending.
        """
        sorted_mem = sorted(
            self.video_gpu_memory_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_mem[:n]

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
            slowest_str = ", ".join(
                f"{Path(v).stem[:20]}={t:.1f}s"
                for v, t in slowest
            )
            lines.append(f"  Slowest: {slowest_str}")

        # GPU memory (US-110-006)
        if self.gpu_memory_peak_mb > 0:
            lines.append(f"  GPU memory peak: {self.gpu_memory_peak_mb:.1f} MB")
            highest_mem = self.get_highest_memory_videos(3)
            if highest_mem:
                mem_str = ", ".join(
                    f"{Path(v).stem[:20]}={m:.0f}MB"
                    for v, m in highest_mem
                )
                lines.append(f"  Highest memory: {mem_str}")

        # Quality metrics (US-110-007)
        if self.transcribed_count > 0 and self.avg_word_confidence > 0:
            lines.append(f"  Avg word confidence: {self.avg_word_confidence:.2%}")
            if self.min_segment_confidence < 1.0:
                lines.append(f"  Min segment confidence: {self.min_segment_confidence:.2%}")
            # Quality warning
            if self.avg_word_confidence < 0.7:
                lines.append("  ⚠️ Low transcription quality detected!")

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
            'gpu_memory_peak_mb': self.gpu_memory_peak_mb,
            'video_gpu_memory_usage': dict(self.video_gpu_memory_usage),
            'avg_word_confidence': self.avg_word_confidence,
            'min_segment_confidence': self.min_segment_confidence,
            'video_confidences': dict(self.video_confidences),
            'segment_rejection_count': self.segment_rejection_count,  # US-110-009
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
            gpu_memory_peak_mb=data.get('gpu_memory_peak_mb', 0.0),
            video_gpu_memory_usage=data.get('video_gpu_memory_usage', {}),
            avg_word_confidence=data.get('avg_word_confidence', 0.0),
            min_segment_confidence=data.get('min_segment_confidence', 1.0),
            video_confidences=data.get('video_confidences', {}),
            segment_rejection_count=data.get('segment_rejection_count', 0),  # US-110-009
        )

    def export_json(self, output_path: Optional[Path] = None) -> Dict[str, Any] | str:
        """Export metrics to JSON format.

        Args:
            output_path: Optional file path to write JSON. If None, returns dict.

        Returns:
            If output_path provided, returns the path as string.
            Otherwise returns the JSON-serializable dict.
        """
        data = {
            "format": "transcription_metrics_json",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": self.get_summary_dict(),
            "details": self.to_dict(),
        }

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Exported transcription metrics JSON to {output_path}")
            return str(output_path)

        return data

    def export_prometheus(self, output_path: Optional[Path] = None) -> str:
        """Export metrics to Prometheus text format.

        Args:
            output_path: Optional file path to write Prometheus format.

        Returns:
            Prometheus-formatted string, or file path if output_path provided.
        """
        lines = [
            "# HELP transcription_total_videos Total videos in batch",
            "# TYPE transcription_total_videos gauge",
            f"transcription_total_videos {self.total_videos}",
            "",
            "# HELP transcription_cached_hits Number of cache hits",
            "# TYPE transcription_cached_hits counter",
            f"transcription_cached_hits {self.cached_hits}",
            "",
            "# HELP transcription_transcribed_count Number of transcribed videos",
            "# TYPE transcription_transcribed_count counter",
            f"transcription_transcribed_count {self.transcribed_count}",
            "",
            "# HELP transcription_failed_count Number of failed transcriptions",
            "# TYPE transcription_failed_count counter",
            f"transcription_failed_count {self.failed_count}",
            "",
            "# HELP transcription_total_duration_seconds Total audio duration processed",
            "# TYPE transcription_total_duration_seconds counter",
            f"transcription_total_duration_seconds {round(self.total_audio_duration_s, 3)}",
            "",
            "# HELP transcription_total_time_seconds Total transcription time",
            "# TYPE transcription_total_time_seconds counter",
            f"transcription_total_time_seconds {round(self.total_transcription_time_s, 3)}",
            "",
            "# HELP transcription_speed_ratio Real-time speed ratio",
            "# TYPE transcription_speed_ratio gauge",
            f"transcription_speed_ratio {round(self.avg_speed_ratio, 2)}",
            "",
            "# HELP transcription_cache_hit_rate_percent Cache hit rate percentage",
            "# TYPE transcription_cache_hit_rate_percent gauge",
            f"transcription_cache_hit_rate_percent {self.cache_hit_rate}",
            "",
            "# HELP transcription_success_rate_percent Transcription success rate",
            "# TYPE transcription_success_rate_percent gauge",
            f"transcription_success_rate_percent {self.success_rate}",
            "",
            "# HELP transcription_phase1_time_seconds Phase 1 (audio extraction) time",
            "# TYPE transcription_phase1_time_seconds counter",
            f"transcription_phase1_time_seconds {round(self.phase1_extraction_time_s, 3)}",
            "",
            "# HELP transcription_phase2_time_seconds Phase 2 (GPU transcription) time",
            "# TYPE transcription_phase2_time_seconds counter",
            f"transcription_phase2_time_seconds {round(self.phase2_transcription_time_s, 3)}",
            "",
            "# HELP transcription_gpu_memory_peak_mb Peak GPU memory used",
            "# TYPE transcription_gpu_memory_peak_mb gauge",
            f"transcription_gpu_memory_peak_mb {round(self.gpu_memory_peak_mb, 1)}",
            "",
            "# HELP transcription_avg_word_confidence Average word confidence",
            "# TYPE transcription_avg_word_confidence gauge",
            f"transcription_avg_word_confidence {round(self.avg_word_confidence, 3)}",
            "",
            "# HELP transcription_min_segment_confidence Minimum segment confidence",
            "# TYPE transcription_min_segment_confidence gauge",
            f"transcription_min_segment_confidence {round(self.min_segment_confidence, 3)}",
            "",
            "# HELP transcription_segment_rejections Number of rejected segments",
            "# TYPE transcription_segment_rejections counter",
            f"transcription_segment_rejections {self.segment_rejection_count}",
            "",
            # Budget metrics
            "# HELP transcription_budget_total_attempts Total retry budget attempts",
            "# TYPE transcription_budget_total_attempts counter",
            f"transcription_budget_total_attempts {self.budget_total_attempts}",
            "",
            "# HELP transcription_budget_failed_attempts Failed retry budget attempts",
            "# TYPE transcription_budget_failed_attempts counter",
            f"transcription_budget_failed_attempts {self.budget_failed_attempts}",
            "",
            "# HELP transcription_budget_exhausted_count Videos skipped due to budget",
            "# TYPE transcription_budget_exhausted_count counter",
            f"transcription_budget_exhausted_count {self.budget_exhausted_count}",
            "",
        ]

        prometheus_output = "\n".join(lines)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(prometheus_output)
            logger.debug(f"Exported transcription metrics Prometheus to {output_path}")
            return str(output_path)

        return prometheus_output

    def export_csv(self, output_path: Optional[Path] = None) -> str:
        """Export metrics to CSV format for time-series analysis.

        Creates multiple CSV files:
        - Main metrics CSV with summary statistics
        - Per-video CSV with individual video timings

        Args:
            output_path: Optional file path to write CSV.

        Returns:
            Path to main CSV file, or empty string if output_path not provided.
        """
        output_path = Path(output_path) if output_path else None
        base_path = output_path.stem if output_path else "transcription_metrics"
        output_dir = output_path.parent if output_path else Path(".")

        # Main metrics CSV
        summary = self.get_summary_dict()
        main_lines = ["metric,value"]
        for key, value in summary.items():
            main_lines.append(f"{key},{value}")

        if output_path:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                for line in main_lines:
                    writer.writerow(line.split(","))
            main_file = str(output_path)
        else:
            main_file = "\n".join(main_lines)

        # Per-video CSV (timings)
        if self.video_durations or self.video_transcription_times:
            video_rows = [
                ["video_path", "audio_duration_s", "transcription_time_s", "speed_ratio"]
            ]
            for video_path in self.video_durations:
                duration = self.video_durations.get(video_path, 0)
                trans_time = self.video_transcription_times.get(video_path, 0)
                speed = duration / trans_time if trans_time > 0 else 0
                video_rows.append([
                    video_path,
                    round(duration, 2),
                    round(trans_time, 2),
                    round(speed, 2)
                ])

            video_csv_path = output_dir / f"{base_path}_videos.csv" if output_path else None
            if video_csv_path:
                with open(video_csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(video_rows)
                logger.debug(f"Exported per-video metrics to {video_csv_path}")

        # Per-video GPU memory CSV
        if self.video_gpu_memory_usage:
            mem_rows = [["video_path", "gpu_memory_mb"]]
            for video_path, memory in self.video_gpu_memory_usage.items():
                mem_rows.append([video_path, round(memory, 1)])

            mem_csv_path = output_dir / f"{base_path}_gpu_memory.csv" if output_path else None
            if mem_csv_path:
                with open(mem_csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(mem_rows)
                logger.debug(f"Exported GPU memory metrics to {mem_csv_path}")

        if output_path:
            logger.debug(f"Exported transcription metrics CSV to {output_path}")
            return str(output_path)

        return main_file
